#!/usr/bin/env python3
"""Bounded WebGPU GEMV A/B in ABBA process order; one resident arm at a time."""
import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

from benchmark_guard import run_guarded

ROOT = Path(__file__).resolve().parents[3]


def compare_outputs(left, right, n):
    a_files, b_files = sorted(left.glob("*.f32")), sorted(right.glob("*.f32"))
    if len(a_files) != 3 or [p.name for p in a_files] != [p.name for p in b_files]:
        raise ValueError("missing or mismatched output vectors")
    count, maximum, squared = 0, 0.0, 0.0
    for a, b in zip(a_files, b_files):
        x, y = array.array("f"), array.array("f")
        x.frombytes(a.read_bytes())
        y.frombytes(b.read_bytes())
        if sys.byteorder != "little":
            x.byteswap()
            y.byteswap()
        if len(x) != n or len(y) != n:
            raise ValueError("output length differs from N")
        for index, (u, v) in enumerate(zip(x, y)):
            error = abs(u-v)
            if not (math.isfinite(u) and math.isfinite(v)) or error > .001 + .0001*max(abs(u), abs(v)):
                raise ValueError(f"output mismatch {a.name} column {index}: {u} != {v}")
            maximum = max(maximum, error)
            squared += error*error
            count += 1
    return dict(values=count, max_abs=maximum, rms=math.sqrt(squared/count), passed=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--binary", type=Path, default=ROOT / "fusor/target/debug/examples/qmatmul_lab")
    p.add_argument("--baseline-binary", type=Path, help="baseline compiler build; defaults to --binary")
    p.add_argument("--candidate-binary", type=Path, help="candidate compiler build; defaults to --binary")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--format", choices=["q4k", "q6k"], default="q4k")
    p.add_argument("--weight-file", type=Path, help="native quantized blocks extracted from the model")
    p.add_argument("--copies", type=int, default=2)
    p.add_argument("--iterations", type=int, default=101)
    p.add_argument("--repeats", type=int, default=32)
    p.add_argument("--target", type=float, default=1.01)
    args = p.parse_args()
    if not (1 <= args.n <= 131072 and 1024 <= args.k <= 32768 and args.k % 1024 == 0
            and 1 <= args.copies <= 32 and 3 <= args.iterations <= 101
            and 2 <= args.repeats <= 32 and math.isfinite(args.target) and args.target > 0):
        p.error("invalid shape, samples or memory-bounded repetition count")
    weight_bytes = args.n * args.k // 256 * (144 if args.format == "q4k" else 212)
    if weight_bytes * args.copies > 512*1024**2:
        p.error("resident quantized weight copies exceed 512 MiB")
    weight_source = None
    if args.weight_file:
        source_path = args.weight_file.resolve()
        native_bytes = args.n * args.k // 256 * (144 if args.format == "q4k" else 210)
        if source_path.stat().st_size != native_bytes:
            p.error("weight file size differs from requested native quantized matrix")
        with source_path.open("rb") as source:
            weight_source = dict(path=str(source_path), sha256=hashlib.file_digest(source, "sha256").hexdigest())
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binaries = {}
    for arm, path in [("A", args.baseline_binary), ("B", args.candidate_binary)]:
        binary = (path or args.binary).resolve(strict=True)
        with binary.open("rb") as source:
            sha = hashlib.file_digest(source, "sha256").hexdigest()
        binaries[arm] = (binary, sha)
    clean = {k:v for k,v in os.environ.items() if not k.startswith(("FUSOR_", "PROFILE_", "LAB_"))}
    results, provenance = {}, {}
    for label, config in [("A0", args.baseline), ("B0", args.candidate),
                          ("B1", args.candidate), ("A1", args.baseline)]:
        binary, sha = binaries[label[0]]
        command = [str(binary), "1", str(args.n), str(args.k), args.format, str(args.iterations)]
        switches = json.loads(config.read_text())
        if not all(k.startswith("FUSOR_") and isinstance(v, str) for k,v in switches.items()):
            p.error("configuration must contain FUSOR_ string switches only")
        env = dict(clean, FUSOR_PIN_CONTRACT_FAMILY="sgemv", FUSOR_PIN_SGEMV="32,2,4,4,32",
                   FUSOR_PROFILE_DISABLE_RULES="SPLIT_K")
        env.update(switches)
        env.update(FUSOR_EXPLORE_EPS="0", FUSOR_AUTOTUNE_MIN_MACS=str(2**64-1), FUSOR_COMPILE_THREADS="1",
                   FUSOR_DUMP_CONTRACT="1",
                   FUSOR_PROFILE_MEMORY_BUDGET_BYTES=str(3*1024**3),
                   FUSOR_PROFILE_SUBMIT_SIZE=str(2**31-1), LAB_BATCH_REPEAT=str(args.repeats),
                   LAB_WEIGHT_COPIES=str(args.copies), LAB_PLAN="1", LAB_FRACTIONAL_INPUT="1",
                   LAB_QLAYOUT="native" if args.format == "q4k" else "f32scales",
                   LAB_OUTPUT_DIR=str(out / label))
        if weight_source:
            env["LAB_WEIGHT_FILE"] = weight_source["path"]
        provenance[label] = dict(command=command, binary_sha256=sha,
                                 weight_source=weight_source,
                                 environment={k:v for k,v in env.items() if k.startswith(("FUSOR_", "LAB_"))})
        (out / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
        with (out / f"{label}.txt").open("w") as stdout, (out / f"{label}.log").open("w") as stderr:
            run_guarded(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                        timeout=180, report_path=out / f"{label}-resources.jsonl")
        rows = [r for r in (out / f"{label}.txt").read_text().splitlines() if r.startswith("MATMUL ")]
        if len(rows) != 1:
            raise ValueError("expected one complete GEMV result")
        row = dict(field.split("=", 1) for field in rows[0].split()[1:])
        gpu_us = float(row["gpu_us"])
        if not math.isfinite(gpu_us) or gpu_us <= 0:
            raise ValueError("invalid GPU timing")
        log = (out / f"{label}.log").read_text()
        cells = {tuple(map(int, cell)) for cell in re.findall(
            r"theta=Sgemv\(SgemvParams \{ vector: (\d+), subgroups: (\d+), cols: (\d+), parts: (\d+), gap: (\d+) \}\)", log)}
        expected = tuple(map(int, env["FUSOR_PIN_SGEMV"].split(",")))
        kernels = sorted(set(re.findall(r"^LAB_COMMAND dispatch=(\S+)", log, re.M)))
        if cells != {expected} or not kernels or not all(k == "sgemv" or k.startswith("sgemv_") for k in kernels):
            raise ValueError(f"unexpected schedule/kernels: {cells}, {kernels}")
        results[label] = dict(gpu_us=gpu_us, row=row, kernels=kernels, cell=expected)
        print(label, rows[0], flush=True)
    checks = {label: compare_outputs(out / "A0", out / label, args.n) for label in ["B0", "B1", "A1"]}
    ratios = [results[a]["gpu_us"] / results[b]["gpu_us"] for a,b in [("A0", "B0"), ("A1", "B1")]]
    result = dict(order="ABBA", arms=results, directional_speedups=ratios, correctness=checks,
                  resident_weight_bytes=weight_bytes*args.copies, target=args.target,
                  target_pass=all(r >= args.target for r in ratios),
                  note="Sequential microbenchmark only; full-model wall and full-logit checks still required.")
    (out / "result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
    if not result["target_pass"]:
        raise SystemExit("microbenchmark speed target unmet in one or both orders")


if __name__ == "__main__":
    main()
