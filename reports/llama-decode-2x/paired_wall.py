#!/usr/bin/env python3
"""Clean WebGPU wall A/B in ABBA order, retaining all four process runs.

The original wall_ab.py validates phases, equal prompts, text, and executables.
This wrapper reverses the second pair and requires both orderings to meet the
target. Numerical correctness must be established separately on full logits.
Background GPU work can still perturb measurements; this is not isolation.
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def summarize(ab, ba, target):
    """Reorient the reversed pair before testing the same candidate twice."""
    for pair in [ab, ba]:
        assert pair['text_matches'], 'generated text differs'
        assert pair['prompt_tokens'] == ab['prompt_tokens'], 'prompt differs across orders'
        for arm in ['baseline', 'candidate']:
            assert pair[arm]['samples'] == ab['baseline']['samples'], 'sample count differs'
            assert pair[arm]['median_ms'] > 0
    ratios = [ab['baseline']['median_ms'] / ab['candidate']['median_ms'],
              ba['candidate']['median_ms'] / ba['baseline']['median_ms']]
    a_times = [ab['baseline']['median_ms'], ba['candidate']['median_ms']]
    b_times = [ab['candidate']['median_ms'], ba['baseline']['median_ms']]
    return dict(order='ABBA', prompt_tokens=ab['prompt_tokens'],
                samples_per_arm=2 * ab['baseline']['samples'],
                baseline_run_medians_ms=a_times, candidate_run_medians_ms=b_times,
                directional_speedups=ratios,
                median_run_speedup=statistics.median(a_times) / statistics.median(b_times),
                minimum_directional_speedup=min(ratios), target=target,
                target_pass=all(r >= target for r in ratios), text_matches=True,
                accuracy_note='Separate full-logit validation required; text equality alone is insufficient.',
                timing_note='No timestamps or logit writes; ABBA controls run order, not other GPU workloads.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--baseline-config', type=Path, default=ROOT / 'reports/llama-8x/candidate.json')
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=1)
    p.add_argument('--steps', type=int, default=16)
    p.add_argument('--prompt-repeats', type=int, default=8)
    p.add_argument('--target', type=float, default=2)
    a = p.parse_args()
    assert a.target > 0
    a.output.mkdir(parents=True, exist_ok=False)
    for name, base, candidate in [('AB', a.baseline_config, a.config),
                                  ('BA', a.config, a.baseline_config)]:
        cmd = [sys.executable, str(Path(__file__).with_name('wall_ab.py')), str(a.model.resolve()),
               '--output', str((a.output / name).resolve()), '--binary', str(a.binary.resolve()),
               '--baseline-config', str(base.resolve()), '--config', str(candidate.resolve()),
               '--rounds', str(a.rounds), '--steps', str(a.steps),
               '--prompt-repeats', str(a.prompt_repeats), '--target', '0.000001']
        with (a.output / f'{name}.txt').open('w') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    # Every process must have used the exact same executable and configurations.
    provenance = [json.loads((a.output / order / 'provenance.json').read_text()) for order in ['AB', 'BA']]
    assert len({arm['binary_sha256'] for pair in provenance for arm in pair.values()}) == 1
    assert provenance[0]['baseline']['environment'] == provenance[1]['candidate']['environment']
    assert provenance[0]['candidate']['environment'] == provenance[1]['baseline']['environment']
    phases = []
    for order in ['AB', 'BA']:
        for arm in ['baseline', 'candidate']:
            rows = [json.loads(line) for line in (a.output / order / f'{arm}.jsonl').read_text().splitlines()]
            phases.append([(r['round'], r['phase'], r['step'], r['prompt_tokens'], r['text']) for r in rows])
    assert all(rows == phases[0] for rows in phases), 'phase identity or generated text differs across orders'
    pairs = [json.loads((a.output / order / 'result.json').read_text()) for order in ['AB', 'BA']]
    result = summarize(*pairs, a.target)
    (a.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if not result['target_pass']:
        raise SystemExit(f"target unmet: directional speedups {result['directional_speedups']}")


if __name__ == '__main__':
    main()
