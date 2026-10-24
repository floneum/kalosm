#!/usr/bin/env python3
"""Compare saved model logits and phase identities without loading a model or using a GPU."""
import argparse
import array
import json
import math
from pathlib import Path
import sys



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
    parser.add_argument("reference", type=Path, help="directory of numbered .f32 vectors")
    parser.add_argument("candidate", type=Path, help="directory of numbered .f32 vectors")
    parser.add_argument("--reference-log", type=Path, required=True)
    parser.add_argument("--candidate-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ref_rows, rows = phase_rows(args.reference_log), phase_rows(args.candidate_log)
    if not rows or [identity(row) for row in rows] != [identity(row) for row in ref_rows]:
        raise ValueError("missing or reordered model phases")
    if len({identity(row) for row in rows}) != len(rows):
        raise ValueError("duplicate model phases")
    expected = {f"{i:05}.f32" for i in range(len(rows))}
    for directory in [args.reference, args.candidate]:
        if {p.name for p in directory.glob("*.f32")} != expected:
            raise ValueError(f"missing or extra logit vectors: {directory}")
    checks = []
    for index, (ref, row) in enumerate(zip(ref_rows, rows)):
        if row["prompt_tokens"] != ref["prompt_tokens"] or row["warmup"] != ref["warmup"]:
            raise ValueError("prompt or warmup identity differs")
        check = compare_vector(args.reference/f"{index:05}.f32", args.candidate/f"{index:05}.f32")
        check.update(phase=identity(row), text_equal=row["text"] == ref["text"])
        checks.append(check)
    result = dict(checks=checks, values=sum(c["values"] for c in checks),
                  max_abs=max(c["max_abs"] for c in checks), max_rms=max(c["rms"] for c in checks),
                  passed=all(c["passed"] and c["text_equal"] for c in checks),
                  note="Saved-output correctness comparison only; no performance claim.")
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "checks"}, indent=2))
    if not result["passed"]:
        raise SystemExit("model numerical or generated-text regression")


if __name__ == "__main__":
    main()
