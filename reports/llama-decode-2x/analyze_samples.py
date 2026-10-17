#!/usr/bin/env python3
"""Join rotating sparse GPU samples to exact selected operations.

Times remain local timestamp measurements, not clean wall latency. Estimates
weight a group's median sampled time by its actual mean invocation count;
unsampled groups are reported and never silently assigned zero cost.
"""
import argparse
import collections
import json
from pathlib import Path
import re
import statistics

SAMPLE = re.compile(r'TSAMPLE (\d+) (\S+) ([\d.]+) L(\d+) grid=(\[.*\])')

def analyze(path):
    groups = {}
    rows = 0
    phase = None
    live = []
    occurrences = collections.Counter()
    sampled = []
    metadata = {}
    with Path(path).open() as logfile:
        for line in logfile:
            if line.startswith('PHASE_BEGIN '):
                assert phase is None
                phase = line.split()[1]
                occurrences = collections.Counter()
                sampled = []
            elif line.startswith(('RESOLVE_META ', 'RESOLVE_META_REF ')):
                tag, payload = line.split(' ', 1)
                meta = json.loads(payload)
                if tag == 'RESOLVE_META_REF':
                    assert meta['metadata_id'] in metadata, 'missing metadata reference'
                    meta = metadata[meta['metadata_id']]
                elif 'metadata_id' in meta:
                    assert meta['metadata_id'] not in metadata, 'duplicate metadata identity'
                    metadata[meta['metadata_id']] = meta
                live = [l for l in meta['launches'] if all(l['grid'])]
                if phase is not None:
                    for l in live:
                        key = json.dumps(l['operation'], sort_keys=True, separators=(',', ':'))
                        occurrences[key] += 1
            elif m := SAMPLE.fullmatch(line.strip()):
                if phase is None:
                    continue
                ix, name, us, plan_ix, grid = m.groups()
                entry = live[int(ix)]
                assert entry['index'] == int(plan_ix) and entry['grid'] == json.loads(grid)
                key = json.dumps(entry['operation'], sort_keys=True, separators=(',', ':'))
                sampled.append((key, name, float(us)))
            elif line.startswith('PHASE_END '):
                row = json.loads(line[len('PHASE_END '):])
                assert row['key'] == phase
                if not row['warmup'] and row['phase'] == 'decode' and row['step'] > 1:
                    assert sum(occurrences.values()) == row['dispatches'], 'metadata coverage mismatch'
                    rows += 1
                    for key, count in occurrences.items():
                        g = groups.setdefault(key, dict(operation=json.loads(key), invocations=0, us=[], names=set()))
                        g['invocations'] += count
                    for key, name, us in sampled:
                        assert us > 0, 'device did not record a positive timing'
                        groups[key]['us'].append(us)
                        groups[key]['names'].add(name)
                phase = None
    assert phase is None and rows > 0
    result = []
    for g in groups.values():
        us = g.pop('us')
        median = statistics.median(us) if us else None
        g.update(names=sorted(g['names']), samples=len(us), median_us=median,
                 p25_us=sorted(us)[len(us)//4] if us else None,
                 p75_us=sorted(us)[3*len(us)//4] if us else None,
                 calls_per_token=g['invocations']/rows,
                 estimated_ms_per_token=median*g['invocations']/rows/1000 if us else None)
        result.append(g)
    result.sort(key=lambda g: -(g['estimated_ms_per_token'] or 0))
    return dict(steady_tokens=rows, sampled_dispatches=sum(g['samples'] for g in result),
                unsampled_groups=sum(not g['samples'] for g in result),
                estimated_sampled_groups_ms=sum(g['estimated_ms_per_token'] or 0 for g in result),groups=result)

def description(o):
    if o['kind'] == 'contract':
        return f"{o['family']} M{o['m']} N{o['n']} K{o['k']} B{o['batch']} {[a['dtype'] for a in o['b']]}"
    if o['kind'] in ('slab','group'):
        return o['kind']+' ['+', '.join(description(m) for m in o['members'])+']'
    return o['kind']+' '+str(o.get('shape',''))

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('log',type=Path);a=p.parse_args()
    result=analyze(a.log);a.log.with_suffix('.samples.json').write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k!='groups'})
    for g in result['groups'][:35]:
        print(f"{g['estimated_ms_per_token'] or 0:7.3f} ms  {g['calls_per_token']:6.2f} calls  {g['samples']:4d} samples  {g['median_us'] or 0:8.3f} us  {description(g['operation'])}")
