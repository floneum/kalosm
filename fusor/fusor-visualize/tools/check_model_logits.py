#!/usr/bin/env python3
"""Capture every model logit without GPU timestamp instrumentation, under the resource guard."""
import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import sys

from benchmark_guard import run_guarded

ROOT = Path(__file__).resolve().parents[3]


def phase_rows(path):
    return [json.loads(line.removeprefix("PHASE_END "))
            for line in path.read_text().splitlines() if line.startswith("PHASE_END ")]


def identity(row):
    return row["prompt"], row["round"], row["phase"], row["step"]


def compare_vector(left, right):
    vectors = []
    for path in [left, right]:
        values = array.array("f")
        values.frombytes(path.read_bytes())
        if sys.byteorder != "little":
            values.byteswap()
        if not values or not all(math.isfinite(v) for v in values):
            raise ValueError(f"empty or nonfinite logits: {path}")
        vectors.append(values)
    x, y = vectors
    if len(x) != len(y):
        raise ValueError("logit vector lengths differ")
    maximum, squared = 0., 0.
    for u, v in zip(x, y):
        error = abs(u-v)
        maximum = max(maximum, error)
        squared += error*error
    rms = math.sqrt(squared/len(x))
    argmax_equal = max(range(len(x)), key=x.__getitem__) == max(range(len(y)), key=y.__getitem__)
    return dict(values=len(x), max_abs=maximum, rms=rms, argmax_equal=argmax_equal,
                passed=maximum < .05 and rms < .005 and argmax_equal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=ROOT/"reports/llama-decode-2x/baseline")
    parser.add_argument("--binary", type=Path, default=ROOT/"target/debug/examples/profile_kernel_share")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    if not (1 <= args.rounds <= 10 and 2 <= args.steps <= 64):
        parser.error("rounds must be 1..10 and steps 2..64")
    reference = args.reference.resolve()
    ref_rows = phase_rows(reference/"ab-reference.log")
    by_identity = {identity(row): (index, row) for index, row in enumerate(ref_rows)}
    if len(by_identity) != len(ref_rows):
        raise ValueError("duplicate reference phases")
    expected = [("long", r, "prefill" if step == 0 else "decode", step)
                for r in range(args.rounds+1) for step in range(args.steps+1)]
    if not set(expected).issubset(by_identity):
        parser.error("reference does not cover requested phases")
    config = json.loads(args.config.read_text())
    if any(k.startswith(("FUSOR_TIME_", "PROFILE_REPLAY_", "PROFILE_METADATA", "FUSOR_GAPSTEP",
                         "FUSOR_RESOLVE_", "PROFILE_LOGITS_")) for k in config):
        parser.error("correctness capture must not contain timing/metadata instrumentation")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary = args.binary.resolve()
    command = [str(binary), str(args.model.resolve()), str(args.rounds), str(args.steps)]
    env = {k:v for k,v in os.environ.items() if not k.startswith(("FUSOR_", "PROFILE_", "LAB_"))}
    env.update(config)
    env.update(PROFILE_PROMPT="long", PROFILE_PROMPT_REPEATS="8", PROFILE_LOGITS_DIR=str(out/"logits"),
               FUSOR_EXPLORE_EPS="0")
    with binary.open("rb") as source:
        sha = hashlib.file_digest(source, "sha256").hexdigest()
    (out/"provenance.json").write_text(json.dumps(dict(command=command, binary_sha256=sha,
        reference=str(reference), environment={k:v for k,v in env.items() if k.startswith(("FUSOR_", "PROFILE_"))}), indent=2)+"\n")
    with (out/"model.jsonl").open("w") as stdout, (out/"model.log").open("w") as stderr:
        run_guarded(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                    model_bytes=args.model.stat().st_size, timeout=600, report_path=out/"resources.jsonl",
                    poll_seconds=.1, record_child_rss=True)
    rows = phase_rows(out/"model.log")
    if [identity(row) for row in rows] != expected:
        raise ValueError("missing or reordered model phases")
    if len(list((out/"logits").glob("*.f32"))) != len(rows):
        raise ValueError("missing or extra logit vectors")
    checks = []
    for index, row in enumerate(rows):
        ref_index, ref = by_identity[identity(row)]
        if row["prompt_tokens"] != ref["prompt_tokens"] or row["warmup"] != ref["warmup"]:
            raise ValueError("prompt or warmup identity differs")
        check = compare_vector(reference/"reference-logits"/f"{ref_index:05}.f32", out/"logits"/f"{index:05}.f32")
        check.update(phase=identity(row), text_equal=row["text"] == ref["text"])
        checks.append(check)
    result = dict(checks=checks, values=sum(c["values"] for c in checks),
                  max_abs=max(c["max_abs"] for c in checks), max_rms=max(c["rms"] for c in checks),
                  passed=all(c["passed"] and c["text_equal"] for c in checks),
                  note="Correctness only; no performance claim from logit-capture timings.")
    (out/"result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "checks"}, indent=2))
    if not result["passed"]:
        raise SystemExit("model numerical or generated-text regression")


if __name__ == "__main__":
    main()
