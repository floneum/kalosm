#!/usr/bin/env python3
"""Alternating ABBA/BAAB schedule comparisons in one bounded WebGPU process.

Uses actual compiler lowering and one resident set of weights. Full vectors for
three changing inputs are compared by the executable; CPU references remain in
qmatmul_lab. This measures pure GEMV, never full-model decode throughput.
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--n', type=int, required=True)
    p.add_argument('--k', type=int, required=True)
    p.add_argument('--format', choices=['q4k', 'q6k'], default='q4k')
    p.add_argument('--baseline-cell', default='32,2,4,4,32')
    p.add_argument('--candidate-cell', required=True)
    p.add_argument('--copies', type=int, default=4)
    p.add_argument('--iterations', type=int, default=31)
    p.add_argument('--repeats', type=int, default=16)
    p.add_argument('--splits', type=int, choices=[1, 2, 4, 8], default=1)
    p.add_argument('--target', type=float, default=1.01)
    p.add_argument('--dependent', action='store_true', help='include a dependent activation guard after every GEMV')
    p.add_argument('--q4-variant', type=int, choices=[1,2,3], help='compare F32 lowering variants at unchanged geometry')
    a = p.parse_args()
    if not (1 <= a.n <= 131072 and 1024 <= a.k <= 32768 and a.k % 1024 == 0
            and 1 <= a.copies <= 32 and 3 <= a.iterations <= 101 and 2 <= a.repeats <= 32
            and math.isfinite(a.target) and a.target > 0):
        p.error('invalid shape or bounded repetition count')
    weight_bytes = a.n * a.k // 256 * (144 if a.format == 'q4k' else 212) * a.copies
    if weight_bytes > 512*1024**2:
        p.error('resident weights exceed 512 MiB')
    if a.q4_variant is not None and ((a.q4_variant != 3 and (a.format != 'q4k' or a.n < 16384)) or a.baseline_cell != a.candidate_cell):
        p.error('Q4 variant requires N >=16384, q4k and identical schedule cells')
    a.output.mkdir(parents=True, exist_ok=False)
    binary = ROOT / 'fusor/target/debug/examples/qmatmul_lab'
    env = {k:v for k,v in os.environ.items() if not k.startswith(('FUSOR_', 'PROFILE_', 'LAB_'))}
    switches = dict(
        FUSOR_PIN_CONTRACT_FAMILY='sgemv', FUSOR_PIN_SGEMV=a.baseline_cell,
        FUSOR_PROFILE_GEMV_PACKED_Q4='1', FUSOR_PROFILE_Q4_HALF_UNPACK='1',
        FUSOR_PROFILE_Q6_GROUP16='1', FUSOR_PROFILE_Q6_COALESCED='1',
        FUSOR_PROFILE_DISABLE_RULES='SPLIT_K', FUSOR_COMPILE_THREADS='1',
        FUSOR_PROFILE_WORKGROUP_THREADS='1024', FUSOR_PROFILE_SKIP_INITIALIZED_WORKGROUP_ZERO='1',
        FUSOR_PROFILE_NO_DEBUG_GROUPS='1', FUSOR_PROFILE_MAPPED_UPLOAD='1',
        FUSOR_PROFILE_MEMORY_BUDGET_BYTES=str(3*1024**3), FUSOR_PROFILE_SUBMIT_SIZE=str(2**31-1),
        FUSOR_AUTOTUNE_MIN_MACS=str(2**64-1), FUSOR_EXPLORE_EPS='0',
        FUSOR_PROFILE_PACKED_Q4_SPLITS=str(a.splits), FUSOR_PROFILE_Q6_SPLITS=str(a.splits),
        LAB_WEIGHT_COPIES=str(a.copies), LAB_BATCH_REPEAT=str(a.repeats),
        LAB_QLAYOUT='native' if a.format == 'q4k' else 'f32scales', LAB_FRACTIONAL_INPUT='1',
        LAB_COMPARE_CELL=a.candidate_cell)
    if a.dependent:
        switches['LAB_COMPARE_DEPENDENT'] = '1'
    if a.q4_variant is not None:
        switches['LAB_COMPARE_Q4_VARIANT'] = str(a.q4_variant)
    env.update(switches)
    command = [str(binary), '1', str(a.n), str(a.k), a.format, str(a.iterations)]
    with binary.open('rb') as source:
        sha = hashlib.file_digest(source, 'sha256').hexdigest()
    (a.output/'provenance.json').write_text(json.dumps(dict(command=command,
        binary_sha256=sha, environment=switches, resident_weight_bytes=weight_bytes), indent=2)+'\n')
    with (a.output/'stdout.txt').open('w') as stdout, (a.output/'stderr.txt').open('w') as stderr:
        run_guarded(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=300,
                    report_path=a.output/'resources.jsonl')
    lines = (a.output/'stdout.txt').read_text().splitlines()
    rows = [json.loads(line.removeprefix('GEOMETRY_PAIR ')) for line in lines if line.startswith('GEOMETRY_PAIR ')]
    assert len(rows) == a.iterations+3
    assert [r['iteration'] for r in rows] == list(range(len(rows)))
    assert all(r['checked_values'] >= a.n*a.copies*3 for r in rows[:3])
    assert all(r['dispatches_per_arm'] == a.copies for r in rows)
    assert all(r['q4_variant'] == (None if a.q4_variant is None else [0,a.q4_variant]) for r in rows)
    for r in rows:
        assert all(math.isfinite(r[k]) and r[k] > 0 for k in ['baseline_us', 'candidate_us', 'speedup'])
    directional = [statistics.median(r['speedup'] for r in rows[3:] if r['iteration'] % 2 == parity)
                   for parity in [0,1]]
    result = dict(samples=a.iterations, baseline_us=statistics.median(r['baseline_us'] for r in rows[3:]),
                  candidate_us=statistics.median(r['candidate_us'] for r in rows[3:]),
                  paired_speedup=statistics.median(r['speedup'] for r in rows[3:]),
                  directional_speedups=directional, target_pass=all(r >= a.target for r in directional),
                  max_abs_error=max(r['max_abs_error'] for r in rows[:3]),
                  checked_values=sum(r['checked_values'] for r in rows[:3]),
                  resident_weight_bytes=weight_bytes,
                  dependent_input_guard=a.dependent,
                  note='Pure GEMV microbenchmark. Full-model logits and clean wall comparison still required.')
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not result['target_pass']:
        raise SystemExit('geometry speed target unmet in one or both orders')


if __name__ == '__main__':
    main()
