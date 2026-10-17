#!/usr/bin/env python3
"""Clean wall A/B: no timestamps or logit writes inside the timed decode call.

Accuracy must be established separately with run_ab.py. This checks generated
text but is not a substitute for comparing every logit. Both arms run the same
executable by default; --reference can reproduce a preserved historical build.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'fusor/fusor-visualize/tools'))
from benchmark_guard import run_guarded
from benchmark_protocol import validate_decode_resource_scope
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('model', type=Path)
p.add_argument('--output', required=True, type=Path)
p.add_argument('--baseline-config', type=Path, default=ROOT/'reports/llama-8x/candidate.json')
p.add_argument('--config', type=Path, default=ROOT/'reports/llama-decode-2x/packed-candidate.json')
p.add_argument('--binary', type=Path, default=ROOT/'target/debug/examples/profile_kernel_share')
p.add_argument('--reference', type=Path)
p.add_argument('--rounds', type=int, default=2)
p.add_argument('--steps', type=int, default=16)
p.add_argument('--prompt-repeats', type=int, default=8)
p.add_argument('--target', type=float, default=2)
a = p.parse_args()
assert 1 <= a.prompt_repeats <= 512
assert 1 <= a.rounds <= 10 and 2 <= a.steps <= 64 and a.target > 0
a.output.mkdir(parents=True, exist_ok=False)
base_env = {k:v for k,v in os.environ.items() if not k.startswith(('PROFILE_','FUSOR_','LAB_'))}
all_rows = {}
provenance = {}
# Sequential processes avoid GPU/CPU contention. Warmup round and first decode
# of each fresh cache are excluded consistently with the timestamp harness.
for label, binary, config in [('baseline',a.reference or a.binary,a.baseline_config),('candidate',a.binary,a.config)]:
    binary = binary.resolve()
    env = dict(base_env, **json.loads(config.read_text()))
    forbidden = ('FUSOR_TIME_', 'FUSOR_RESOLVE_', 'FUSOR_GAPSTEP', 'PROFILE_LOGITS_', 'PROFILE_METADATA', 'PROFILE_REPLAY_PLAN')
    assert not any(k.startswith(forbidden) for k in env), 'instrumented config cannot measure clean wall'
    env.update(PROFILE_PROMPT='long', PROFILE_PROMPT_REPEATS=str(a.prompt_repeats), FUSOR_EXPLORE_EPS='0')
    cmd = [str(binary),str(a.model.resolve()),str(a.rounds),str(a.steps)]
    provenance[label] = dict(command=cmd, binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                             environment={k:v for k,v in env.items() if k.startswith(('PROFILE_','FUSOR_'))})
    (a.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    with (a.output/f'{label}.jsonl').open('w') as out, (a.output/f'{label}.log').open('w') as err:
        run_guarded(cmd,env=env,cwd=ROOT,stdout=out,stderr=err,timeout=600,
                    model_bytes=a.model.stat().st_size,
                    report_path=a.output/f'{label}-resources.jsonl')
    all_rows[label] = [json.loads(line) for line in (a.output/f'{label}.jsonl').read_text().splitlines()]
    validate_decode_resource_scope(all_rows[label], env)
    assert all(row['mode']=='wall' for row in all_rows[label])
    assert len(set(row['prompt_tokens'] for row in all_rows[label])) == 1
    assert [(r['round'],r['phase'],r['step']) for r in all_rows[label]] == [
        (i,'prefill' if step == 0 else 'decode',step)
        for i in range(a.rounds+1) for step in range(a.steps+1)], 'missing or reordered phases'
    assert all(r['warmup'] == (r['round'] == 0) and math.isfinite(r['wall_ms']) and r['wall_ms'] > 0
               for r in all_rows[label]), 'invalid warmup identity or wall timing'

def summarize(rows):
    steady = [r for r in rows if not r['warmup'] and r['phase']=='decode' and r['step']>1]
    assert len(steady)==a.rounds*(a.steps-1)
    times = [r['wall_ms'] for r in steady]
    return dict(samples=len(times),median_ms=statistics.median(times),mean_ms=statistics.mean(times),
                min_ms=min(times),max_ms=max(times),tokens_per_second=1000/statistics.mean(times),
                dispatches=sorted(set(r['dispatches'] for r in steady)))
base,candidate = [summarize(all_rows[k]) for k in ('baseline','candidate')]
assert all_rows['baseline'][0]['prompt_tokens'] == all_rows['candidate'][0]['prompt_tokens']
b = {r['key']:r['text'] for r in all_rows['baseline']}
c = {r['key']:r['text'] for r in all_rows['candidate']}
result = dict(baseline=base,candidate=candidate,median_speedup=base['median_ms']/candidate['median_ms'],
              prompt_tokens=all_rows['baseline'][0]['prompt_tokens'],
              mean_speedup=base['mean_ms']/candidate['mean_ms'],text_matches=b==c,
              target=a.target, target_pass=base['median_ms']/candidate['median_ms']>=a.target,
              accuracy_note='Text check only; use separate full-logit A/B for numerical validation.')
(a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
assert result['text_matches'], 'generated text differs'
assert result['target_pass'], f"decode wall speedup {result['median_speedup']:.3f}x < {a.target}x"
