#!/usr/bin/env python3
"""Turn observed selected quantized operands into an explicit decode byte budget.

This is a streaming scenario, not a hardware DRAM counter: each distinct
quantized operand occurrence is read once per invocation; no cross-token cache
residency, compression, or skipped weight reads are assumed. Bandwidth is an
input assumption. Detailed timestamp percentages are deliberately not used.
"""
import argparse
import json
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('summary',type=Path)
p.add_argument('--reference-ms',type=float,required=True)
p.add_argument('--target',type=float,default=2)
p.add_argument('--bandwidth-gbs',type=float,nargs='+',default=[320,400])
p.add_argument('--other-ms',type=float,nargs='+',default=[0,2,4,6])
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
assert a.reference_ms>0 and a.target>0 and all(v>0 for v in a.bandwidth_gbs) and all(v>=0 for v in a.other_ms)
d=json.loads(a.summary.read_text())['decode']
assert d['samples']>0

def contracts(op):
    if op['kind']=='contract':yield op
    for child in op.get('members',[]):yield from contracts(child)
rows=[]
for group in d['groups']:
    count=group['count']/d['samples']
    for c in contracts(group['operation']):
        for side in ['a','b']:
            for operand in c[side]:
                if not operand['dtype'].startswith('Q('):continue
                size=operand.get('selected_storage_bytes')
                assert isinstance(size,int) and size>0, 'missing selected layout/extent; recapture metadata'
                rows.append(dict(m=c['m'],n=c['n'],k=c['k'],batch=c['batch'],
                                 dtype=operand['dtype'],layout=operand['selected_quant_layout'],
                                 invocations_per_token=count,bytes_per_invocation=size,
                                 bytes_per_token=count*size))
rows.sort(key=lambda x:-x['bytes_per_token'])
volume=sum(row['bytes_per_token'] for row in rows)
target_ms=a.reference_ms/a.target
result=dict(source=str(a.summary),assumption=__doc__,reference_ms=a.reference_ms,
            target_speedup=a.target,target_ms=target_ms,quantized_bytes_per_token=volume,operands=rows,
            streaming_scenarios=[dict(bandwidth_GBs=b,weight_only_ms=volume/b/1e6,
                                     remaining_ms_for_everything_else=target_ms-volume/b/1e6)
                                 for b in a.bandwidth_gbs],
            required_bandwidth=[dict(other_ms=t,required_weight_GBs=volume/(target_ms-t)/1e6 if t<target_ms else None)
                                for t in a.other_ms])
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(f'Observed quantized operand volume: {volume/1e9:.6f} GB/token')
print(f'{a.target:g}x target: {target_ms:.4f} ms/token from {a.reference_ms:.4f} ms')
for s in result['streaming_scenarios']:
    print(f"Assumed {s['bandwidth_GBs']:g} GB/s: weights alone {s['weight_only_ms']:.4f} ms; remaining budget {s['remaining_ms_for_everything_else']:.4f} ms")
for s in result['required_bandwidth']:
    print(f"Other work {s['other_ms']:g} ms: required weight bandwidth {s['required_weight_GBs']} GB/s")
print('Scenario only; no measured DRAM traffic, hardware peak, or achieved throughput claim.')
