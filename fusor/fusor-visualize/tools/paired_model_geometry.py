#!/usr/bin/env python3
"""One resident WebGPU model, matching tokens from two bounded KV histories.

Alternate baseline/candidate order per token. Optional full-logit mode compares
both arms to the saved original reference and never reports a timing claim.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

from benchmark_guard import run_guarded
from benchmark_protocol import validate_decode_resource_scope
from check_model_logits import compare_vector, identity, phase_rows

ROOT = Path(__file__).resolve().parents[3]


def validate_pairs(rows, rounds, steps):
    expected = [(r, step) for r in range(rounds+1) for step in range(steps+1)]
    if [(r['round'], r['step']) for r in rows] != expected:
        raise ValueError('missing or reordered paired phases')
    for row in rows:
        step = row['step']
        warmup = row['round'] == 0 or step <= 1
        if (row['warmup'] != warmup or row['phase'] != ('prefill' if step == 0 else 'decode')
                or row['order'] != ([0, 1] if step % 2 == 0 else [1, 0])
                or row['prompt_tokens'] != 208):
            raise ValueError('paired phase identity differs')
        if not warmup and row['dispatches'][0] != row['dispatches'][1]:
            raise ValueError('steady decode dispatch counts differ')
        if step > 0 and not (row['matched'][0] > 0 and row['matched'][0] == row['matched'][1]):
            raise ValueError('candidate did not match the same decode projections')
        if any(not math.isfinite(row[k]) or row[k] <= 0 for k in ['baseline_ms', 'candidate_ms', 'speedup']):
            raise ValueError('invalid paired timing')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', type=Path)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--candidate-cell', required=True)
    p.add_argument('--candidate-poll-spin-ms', type=int, choices=range(65), metavar='0..64')
    p.add_argument('--candidate-poll-interval-us', type=int, choices=range(1001), metavar='0..1000')
    p.add_argument('--q4-variant', type=int, choices=[1,2,3])
    p.add_argument('--candidate-stream-rest', type=int, choices=range(32,4097), metavar='32..4096')
    p.add_argument('--rounds', type=int, choices=[1], default=1,
                   help='one warmup and one measured round; further fresh histories exceeded the memory guard')
    p.add_argument('--steps', type=int, default=32)
    p.add_argument('--target', type=float, default=1.01)
    p.add_argument('--capture-logits', action='store_true')
    a = p.parse_args()
    if not (a.rounds == 1 and 2 <= a.steps <= 32 and math.isfinite(a.target) and a.target > 0):
        p.error('rounds must be 1 and steps 2..32')
    config = json.loads(a.config.read_text())
    if any(k.startswith(('FUSOR_TIME_', 'PROFILE_REPLAY_', 'PROFILE_METADATA', 'FUSOR_GAPSTEP',
                         'PROFILE_LOGITS_', 'PROFILE_PAIRED_')) for k in config):
        p.error('configuration already contains benchmark instrumentation')
    if any(not k.startswith(('FUSOR_', 'PROFILE_')) or not isinstance(v, str) for k,v in config.items()):
        p.error('configuration must contain FUSOR_/PROFILE_ string switches')
    reference = ROOT/'reports/llama-decode-2x/baseline'
    expected = [('long',r,'prefill' if step==0 else 'decode',step)
                for r in range(a.rounds+1) for step in range(a.steps+1)]
    refs = {identity(row):(index,row) for index,row in enumerate(phase_rows(reference/'ab-reference.log'))}
    if a.capture_logits and not set(expected).issubset(refs):
        p.error('reference does not cover requested logit phases; use --rounds 1 --steps 8')
    out = a.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    binary = ROOT/'target/debug/examples/profile_kernel_share'
    command = [str(binary),str(a.model.resolve()),str(a.rounds),str(a.steps)]
    env = {k:v for k,v in os.environ.items() if not k.startswith(('FUSOR_','PROFILE_','LAB_'))}
    env.update(config)
    env.update(PROFILE_PAIRED_GEOMETRY=a.candidate_cell,PROFILE_PROMPT='long',PROFILE_PROMPT_REPEATS='8',
               FUSOR_AUTOTUNE_MIN_MACS=str(2**64-1),FUSOR_EXPLORE_EPS='0')
    if a.candidate_poll_spin_ms is not None:
        if a.candidate_cell != '32,2,4,4,32' or 'FUSOR_PROFILE_POLL_SPIN_MS' in config:
            p.error('polling comparison requires unchanged retained geometry and default baseline polling')
        env['PROFILE_PAIRED_POLL_SPIN_MS']=str(a.candidate_poll_spin_ms)
    if a.candidate_poll_interval_us is not None:
        if a.candidate_cell != '32,2,4,4,32' or 'FUSOR_PROFILE_POLL_INTERVAL_US' in config:
            p.error('interval comparison requires retained geometry and default baseline interval')
        env['PROFILE_PAIRED_POLL_INTERVAL_US']=str(a.candidate_poll_interval_us)
    if a.q4_variant is not None:
        if a.candidate_cell != '32,2,4,4,32' or 'FUSOR_PROFILE_Q4_VARIANT' in config:
            p.error('arithmetic comparison requires retained geometry and default baseline arithmetic')
        env['PROFILE_PAIRED_Q4_VARIANT']=str(a.q4_variant)
    if a.candidate_stream_rest is not None:
        if (a.candidate_cell != '32,2,4,4,32' or config.get('FUSOR_PROFILE_STREAM_SUBMIT') != '32'
                or 'FUSOR_PROFILE_STREAM_SUBMIT_REST' in config):
            p.error('submission comparison requires retained geometry and baseline stream size 32')
        env['PROFILE_PAIRED_STREAM_REST']=str(a.candidate_stream_rest)
    if a.capture_logits:
        env['PROFILE_LOGITS_DIR']=str(out/'logits')
    with binary.open('rb') as source:
        sha = hashlib.file_digest(source,'sha256').hexdigest()
    (out/'provenance.json').write_text(json.dumps(dict(command=command,binary_sha256=sha,
        environment={k:v for k,v in env.items() if k.startswith(('FUSOR_','PROFILE_'))},
        model_instances=1,kv_histories=2),indent=2)+'\n')
    with (out/'stdout.txt').open('w') as stdout,(out/'stderr.txt').open('w') as stderr:
        run_guarded(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,timeout=600,
                    model_bytes=a.model.stat().st_size,report_path=out/'resources.jsonl',
                    poll_seconds=.1,record_child_rss=True)
    rows = [json.loads(line.removeprefix('MODEL_GEOMETRY_PAIR ')) for line in (out/'stdout.txt').read_text().splitlines()
            if line.startswith('MODEL_GEOMETRY_PAIR ')]
    validate_pairs(rows,a.rounds,a.steps)
    expected_poll = None if a.candidate_poll_spin_ms is None else [2,a.candidate_poll_spin_ms]
    if any(row['poll_spin_ms'] != expected_poll for row in rows):
        raise ValueError('polling policy differs from requested comparison')
    expected_interval = None if a.candidate_poll_interval_us is None else [0,a.candidate_poll_interval_us]
    if any(row['poll_interval_us'] != expected_interval for row in rows):
        raise ValueError('poll interval differs from requested comparison')
    if any(row['q4_variant'] != (None if a.q4_variant is None else [0,a.q4_variant]) for row in rows):
        raise ValueError('Q4 variant differs from requested comparison')
    if any(row['stream_rest'] != (None if row['step']==0 or a.candidate_stream_rest is None
                                 else [32,a.candidate_stream_rest]) for row in rows):
        raise ValueError('submission policy differs from requested comparison')
    validate_decode_resource_scope([dict(r,dispatches=r['dispatches'][0]) for r in rows],env)
    if a.capture_logits:
        indices=[index for row in rows for index in row['logit_indices']]
        assert sorted(indices)==list(range(2*len(rows)))
        assert len(list((out/'logits').glob('*.f32')))==len(indices)
        checks=[]
        for row in rows:
            key=('long',row['round'],row['phase'],row['step'])
            ref_index,ref=refs[key]
            assert row['text']==ref['text']
            for arm,index in enumerate(row['logit_indices']):
                check=compare_vector(reference/'reference-logits'/f'{ref_index:05}.f32',out/'logits'/f'{index:05}.f32')
                check.update(arm=arm,phase=key)
                checks.append(check)
        result=dict(passed=all(c['passed'] for c in checks),values=sum(c['values'] for c in checks),
                    max_abs=max(c['max_abs'] for c in checks),max_rms=max(c['rms'] for c in checks),checks=checks,
                    note='Full-logit correctness only. Capture timings are not a speed measurement.')
    else:
        steady=[r for r in rows if not r['warmup'] and r['phase']=='decode']
        assert len(steady)==a.rounds*(a.steps-1)
        assert all(math.isfinite(r[k]) and r[k]>0 for r in steady for k in ['baseline_ms','candidate_ms','speedup'])
        directional=[statistics.median(r['speedup'] for r in steady if r['order'][0]==arm) for arm in [0,1]]
        result=dict(samples_per_arm=len(steady),baseline_ms=statistics.median(r['baseline_ms'] for r in steady),
                    candidate_ms=statistics.median(r['candidate_ms'] for r in steady),
                    paired_speedup=statistics.median(r['speedup'] for r in steady),directional_speedups=directional,
                    target=a.target,target_pass=all(v>=a.target for v in directional),text_matches=True,
                    note='One shared model with two short KV histories; no plan replay. Separate logits required.')
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
    if not result.get('passed',result.get('target_pass',False)):
        raise SystemExit('paired model comparison failed')


if __name__=='__main__':
    main()
