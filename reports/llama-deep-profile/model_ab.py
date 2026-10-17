#!/usr/bin/env python3
"""Sequential model A/B with whole-plan timing and full-logit capture."""
import array
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'fusor/fusor-visualize/tools'))
from benchmark_guard import run_guarded
OUT = Path(os.environ.get("PROFILE_REPORT_DIR", str(Path(__file__).resolve().parent)))
OUT.mkdir(parents=True, exist_ok=True)
model = sys.argv[1]
prompt = os.environ.get("PROFILE_AB_PROMPT", "long")
rounds = int(os.environ.get("PROFILE_AB_ROUNDS", "2"))
decode_steps = int(os.environ.get("PROFILE_AB_DECODE_STEPS", "4"))
assert prompt in ("short", "long") and 1 <= rounds <= 10 and 2 <= decode_steps <= 64
configs = [("reference", json.loads(os.environ.get("PROFILE_REFERENCE_ENV", "{}"))),
           ("batch128", {"PROFILE_PREFILL_WIDTH": "128", "FUSOR_PIN_COOP": "64,64,8,4",
                         **json.loads(os.environ.get("PROFILE_CANDIDATE_ENV", "{}"))})]
results = {}
provenance = {}
for name, config in configs:
    logits = OUT / f"{name}-logits"
    if logits.exists():
        raise SystemExit(f"refusing to mix previous logits: {logits}")
    # A candidate pin inherited from the shell must not contaminate the reference.
    # Per-arm overrides belong in PROFILE_REFERENCE_ENV / PROFILE_CANDIDATE_ENV.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FUSOR_", "PROFILE_"))}
    env.update(PROFILE_PROMPT=prompt, PROFILE_LOGITS_DIR=str(logits), FUSOR_EXPLORE_EPS="0",
               FUSOR_TIME_PLAN="1", FUSOR_GAPSTEP="1", FUSOR_RESOLVE_PROFILE="1")
    env.update(config)
    executable = os.environ.get("PROFILE_REFERENCE_BINARY" if name == "reference" else "PROFILE_CANDIDATE_BINARY",
                                str(ROOT / "target/debug/examples/profile_kernel_share"))
    cmd = [executable, model, str(rounds), str(decode_steps)]
    provenance[name] = {"command": cmd, "binary_sha256": hashlib.sha256(Path(executable).read_bytes()).hexdigest(),
                        "environment": {k: v for k, v in env.items() if k.startswith(("PROFILE_", "FUSOR_"))}}
    with (OUT / f"ab-{name}.log").open("w") as log, (OUT / f"ab-{name}.jsonl").open("w") as stdout:
        run_guarded(cmd, env=env, cwd=ROOT, stdout=stdout, stderr=log,
                    timeout=int(os.environ.get("PROFILE_TIMEOUT_SECONDS", "600")),
                    model_bytes=Path(model).stat().st_size,
                    report_path=OUT / f"{name}-resources.jsonl")
    rows = []
    gpu = 0
    dispatches = 0
    for line in (OUT / f"ab-{name}.log").read_text().splitlines():
        if line.startswith("PHASE_BEGIN "): gpu = 0; dispatches = 0
        elif line.startswith("TPLAN "):
            gpu += float(line.split()[1]) / 1000
            dispatches += int(line.split()[2].removeprefix("n="))
        elif line.startswith("PHASE_END "):
            r = json.loads(line[len("PHASE_END "):]); r["gpu_ms"] = gpu
            assert 0 < gpu <= r["wall_ms"] * 1.05 and dispatches == r["dispatches"], "incomplete/invalid GPU profile"
            rows.append(r)
    expected = [(r, "prefill" if s == 0 else "decode", s)
                for r in range(rounds + 1) for s in range(decode_steps + 1)]
    assert [(r["round"], r["phase"], r["step"]) for r in rows] == expected, "missing or reordered phases"
    assert all(r["prompt"] == prompt and r["warmup"] == (r["round"] == 0) for r in rows)
    assert len(list(logits.glob("*.f32"))) == len(rows), "missing or extra logit vectors"
    result = {}
    for phase in ("prefill", "decode"):
        selected = [r for r in rows if r["phase"] == phase and not r["warmup"] and (phase != "decode" or r["step"] > 1)]
        result[phase] = {k: statistics.median(r[k] for r in selected) for k in ("gpu_ms", "wall_ms", "dispatches")}
    first_decode = [r for r in rows if not r["warmup"] and r["phase"] == "decode" and r["step"] == 1]
    result["first_decode"] = {k: statistics.median(r[k] for r in first_decode)
                              for k in ("gpu_ms", "wall_ms", "dispatches")}
    result["session"] = {k: statistics.median(sum(r[k] for r in rows if r["round"] == round_id)
                                              for round_id in range(1, rounds + 1))
                          for k in ("gpu_ms", "wall_ms", "dispatches")}
    result["session"]["generated_tokens"] = decode_steps + 1
    result["text"] = [r["text"] for r in rows]
    results[name] = result
    print(name, json.dumps(result), flush=True)

left = sorted((OUT / "reference-logits").glob("*.f32"))
right = sorted((OUT / "batch128-logits").glob("*.f32"))
assert len(left) == len(right) and len(left) > 0
comparisons = []
for a, b in zip(left, right):
    x, y = array.array("f"), array.array("f")
    x.frombytes(a.read_bytes()); y.frombytes(b.read_bytes())
    if sys.byteorder != "little": x.byteswap(); y.byteswap()
    assert len(x) == len(y) and all(math.isfinite(v) for v in x) and all(math.isfinite(v) for v in y)
    errors = [abs(v-w) for v,w in zip(x,y)]
    row = {"file": a.name, "values": len(x), "max_abs": max(errors), "rms": math.sqrt(sum(e*e for e in errors)/len(errors)),
           "argmax_equal": max(range(len(x)), key=x.__getitem__) == max(range(len(y)), key=y.__getitem__)}
    comparisons.append(row)
results["logits"] = comparisons
results["configuration"] = provenance
results["generated_text_equal"] = results["reference"]["text"] == results["batch128"]["text"]
results["prefill_gpu_speedup"] = results["reference"]["prefill"]["gpu_ms"] / results["batch128"]["prefill"]["gpu_ms"]
results["prefill_wall_speedup"] = results["reference"]["prefill"]["wall_ms"] / results["batch128"]["prefill"]["wall_ms"]
results["session_gpu_speedup"] = results["reference"]["session"]["gpu_ms"] / results["batch128"]["session"]["gpu_ms"]
results["session_wall_speedup"] = results["reference"]["session"]["wall_ms"] / results["batch128"]["session"]["wall_ms"]
results["decode_gpu_ratio"] = results["batch128"]["decode"]["gpu_ms"] / results["reference"]["decode"]["gpu_ms"]
results["decode_gpu_speedup"] = 1 / results["decode_gpu_ratio"]
results["decode_wall_speedup"] = results["reference"]["decode"]["wall_ms"] / results["batch128"]["decode"]["wall_ms"]
(OUT / "model-ab.json").write_text(json.dumps(results, indent=2) + "\n")
print("GPU speedup", results["prefill_gpu_speedup"], "wall speedup", results["prefill_wall_speedup"], "text equal", results["generated_text_equal"], flush=True)
print("logits max_abs", max(r["max_abs"] for r in comparisons), "max_rms", max(r["rms"] for r in comparisons), flush=True)
print("session GPU speedup", results["session_gpu_speedup"], "session wall speedup", results["session_wall_speedup"],
      "decode GPU ratio", results["decode_gpu_ratio"], flush=True)
assert results["generated_text_equal"] and all(r["argmax_equal"] and r["max_abs"] < 0.05 and r["rms"] < 0.005 for r in comparisons), "model numerical regression"
assert results["decode_gpu_ratio"] <= float(os.environ.get("PROFILE_MAX_DECODE_GPU_RATIO", "1.10")), "steady decode GPU regression"
assert results["decode_gpu_speedup"] >= float(os.environ.get("PROFILE_MIN_DECODE_GPU_SPEEDUP", "0")), "steady decode GPU speedup target not met"
assert results["decode_wall_speedup"] >= float(os.environ.get("PROFILE_MIN_DECODE_WALL_SPEEDUP", "0")), "steady decode wall speedup target not met"
assert results["session_wall_speedup"] >= float(os.environ.get("PROFILE_MIN_SESSION_WALL_SPEEDUP", "0")), "session wall speedup target not met"
assert results["prefill_gpu_speedup"] >= float(os.environ.get("PROFILE_MIN_GPU_SPEEDUP", "2")), "prefill GPU speedup target not met"
