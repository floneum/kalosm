#!/usr/bin/env python3
"""Run one default-path decode measurement under the shared resource guard.

Use the same profile_forward source and release configuration for both binaries.
Run each arm separately; compare_decode.py checks output and latency afterwards.
No model presets, shader tuning switches, automatic retries, or concurrent runs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from benchmark_guard import run_guarded
from compare_decode import read_log, summarize

ROOT = Path(__file__).resolve().parents[3]
PROMPT = ('Write a detailed numbered list of 100 practical Rust performance tips. '
          'Keep going until the list is complete.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tokens', type=int, default=64,
                   help='actual generated tokens per round; use a token-observing profile_forward build')
    p.add_argument('--warmup', type=int, default=8)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--prompt', default=PROMPT)
    p.add_argument('--timeout', type=int, default=900)
    a = p.parse_args()
    if not (1 <= a.tokens <= 512 and 1 <= a.warmup <= 512 and 2 <= a.rounds <= 10
            and 0 <= a.seed < 2**64 and 1 <= a.timeout <= 3600):
        p.error('invalid bounded measurement settings')
    binary, model = a.binary.resolve(strict=True), a.model.resolve(strict=True)
    if not binary.is_file() or not model.is_file():
        p.error('binary and model must be regular files')
    a.output.mkdir(parents=True, exist_ok=False)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('FUSOR_', 'PROFILE_', 'LAB_', 'KALOSM_'))}
    settings = dict(KALOSM_PROFILE_LLAMA_LOCAL_PATH=str(model),
                    KALOSM_PROFILE_LLAMA_CHAT='1',
                    KALOSM_PROFILE_LLAMA_WARMUP=str(a.warmup),
                    KALOSM_PROFILE_LLAMA_TOKENS=str(a.tokens),
                    KALOSM_PROFILE_LLAMA_REPEATS=str(a.rounds),
                    KALOSM_PROFILE_LLAMA_SEED=str(a.seed),
                    KALOSM_PROFILE_LLAMA_PROMPT=a.prompt)
    env.update(settings)
    with binary.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    stat = model.stat()
    (a.output / 'provenance.json').write_text(json.dumps(dict(
        binary=str(binary), binary_sha256=digest, model=str(model),
        model_bytes=stat.st_size, model_mtime_ns=stat.st_mtime_ns,
        settings=settings), indent=2) + '\n')
    with (a.output / 'decode.log').open('w') as log:
        result = run_guarded([str(binary)], cwd=ROOT, env=env,
                             stdout=log, stderr=subprocess.STDOUT, timeout=a.timeout,
                             model_bytes=stat.st_size,
                             report_path=a.output / 'resources.jsonl',
                             poll_seconds=.1, record_child_rss=True)
    result.check_returncode()
    _, records = read_log(a.output / 'decode.log')
    if len(records) != a.rounds or any(
        row['unit'] != 'tokens' or row['tokens'] != a.tokens or row['warmup'] != a.warmup
        for row in records
    ):
        raise ValueError('profiler must report the requested actual-token counts and boundaries')
    summary = dict(metric='actual-token decode latency', excluded_warmup_rounds=1,
                   **summarize(records[1:], first_round=1))
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print(a.output / 'decode.log')


if __name__ == '__main__':
    main()
