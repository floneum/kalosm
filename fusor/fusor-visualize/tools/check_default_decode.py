#!/usr/bin/env python3
"""ABBA wall timings of the ordinary Llama builder, with no candidate tuning flags.

Run sequentially under the memory guard. The baseline selects a general device
and takes an explicit configuration, so the historical control is reviewable.
Full-logit correctness is checked separately by check_model_logits.py.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

from benchmark_guard import run_guarded

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--binary', type=Path, default=ROOT/'target/debug/examples/default_decode')
    parser.add_argument('--baseline-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--target', type=float, default=2.)
    args = parser.parse_args()
    if not (2 <= args.steps <= 64 and 1 <= args.rounds <= 10):
        parser.error('steps must be 2..64 and rounds 1..10')
    model, binary, out = args.model.resolve(), args.binary.resolve(), args.output.resolve()
    config = json.loads(args.baseline_config.read_text())
    if any(k.startswith(('FUSOR_TIME_', 'PROFILE_REPLAY_', 'PROFILE_METADATA',
                         'PROFILE_LOGITS_', 'FUSOR_GAPSTEP', 'FUSOR_RESOLVE_')) for k in config):
        parser.error('baseline must not enable timing/metadata/logit instrumentation')
    clean = {k: v for k, v in os.environ.items()
             if not k.startswith(('FUSOR_', 'PROFILE_', 'LAB_', 'KALOSM_'))}
    out.mkdir(parents=True, exist_ok=False)
    with binary.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    medians = {'baseline': [], 'default': []}
    identity = None
    provenance = []
    for index, arm in enumerate(['baseline', 'default', 'default', 'baseline']):
        env = clean.copy()
        if arm == 'baseline':
            env.update(config)
            env['FUSOR_EXPLORE_EPS'] = '0'
        # This arm must exercise the public default, without any optimization flags.
        assert arm != 'default' or not any(k.startswith(('FUSOR_', 'PROFILE_', 'LAB_', 'KALOSM_')) for k in env)
        name = f'{index}-{arm}'
        command = [str(binary), str(model), str(args.steps), str(args.rounds),
                   'general' if arm == 'baseline' else 'default']
        provenance.append(dict(arm=arm, command=command, binary_sha256=sha,
                               environment={k: v for k, v in env.items() if k not in clean}))
        (out/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
        with (out/f'{name}.jsonl').open('w') as stdout, (out/f'{name}.log').open('w') as stderr:
            run_guarded(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                        timeout=900, model_bytes=model.stat().st_size,
                        report_path=out/f'{name}-resources.jsonl', poll_seconds=.1, record_child_rss=True)
        rows = [json.loads(line) for line in (out/f'{name}.jsonl').read_text().splitlines()]
        expected = [(r, s, 'prefill' if s == 0 else 'decode', r == 0)
                    for r in range(args.rounds+1) for s in range(args.steps+1)]
        assert [(r['round'], r['step'], r['phase'], r['warmup']) for r in rows] == expected
        assert all(r['prompt_tokens'] == 208 for r in rows)
        current = [(r['round'], r['step'], r['prompt_tokens'], r['text']) for r in rows]
        if identity is None:
            identity = current
        assert current == identity, 'workload or generated text differs'
        samples = [r['wall_ms'] for r in rows if r['round'] > 0 and r['step'] > 1]
        assert len(samples) == args.rounds*(args.steps-1) and all(math.isfinite(s) and s > 0 for s in samples)
        medians[arm].append(statistics.median(samples))
        print(name, medians[arm][-1], 'ms/token', flush=True)
    ratios = [a/b for a, b in zip(medians['baseline'], medians['default'])]
    result = dict(order='ABBA', prompt_tokens=208, samples_per_arm=2*args.rounds*(args.steps-1),
                  baseline_medians_ms=medians['baseline'], default_medians_ms=medians['default'],
                  directional_speedups=ratios,
                  median_speedup=statistics.median(medians['baseline'])/statistics.median(medians['default']),
                  text_matches=True, candidate_tuning_environment={},
                  target=args.target, target_pass=all(r >= args.target for r in ratios),
                  note='Ordinary builder, without kernel-visualize. Correctness requires the separate full-logit check.')
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    if not result['target_pass']:
        raise SystemExit('default decode speed target not met')


if __name__ == '__main__':
    main()
