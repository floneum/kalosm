#!/usr/bin/env python3
"""Reproduce the isolated Llama A/B; preserve full logits and executable hashes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("model", type=Path)
parser.add_argument("--output", required=True, type=Path, help="new report directory")
parser.add_argument("--reference", required=True, type=Path, help="preserved original profiling executable")
parser.add_argument("--reference-config", type=Path, help="optional reference environment JSON for like-for-like comparisons")
parser.add_argument("--candidate", type=Path, default=ROOT / "target/debug/examples/profile_kernel_share")
parser.add_argument("--config", type=Path, default=HERE / "candidate.json")
parser.add_argument("--prompt", choices=("short", "long"), default="long")
parser.add_argument("--rounds", type=int, default=2)
parser.add_argument("--decode-steps", type=int, default=4)
parser.add_argument("--gpu-target", type=float, default=8)
parser.add_argument("--session-wall-target", type=float, default=0)
parser.add_argument("--max-decode-gpu-ratio", type=float, default=1.10)
parser.add_argument("--decode-gpu-target", type=float, default=0)
parser.add_argument("--decode-wall-target", type=float, default=0)
args = parser.parse_args()
config = json.loads(args.config.read_text())
assert isinstance(config, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in config.items())
reference_config = json.loads(args.reference_config.read_text()) if args.reference_config else {}
assert isinstance(reference_config, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in reference_config.items())
env = {k: v for k, v in os.environ.items() if not k.startswith(("FUSOR_", "PROFILE_"))}
env.update(PROFILE_REPORT_DIR=str(args.output.resolve()),
           PROFILE_REFERENCE_BINARY=str(args.reference.resolve()),
           PROFILE_CANDIDATE_BINARY=str(args.candidate.resolve()),
           PROFILE_CANDIDATE_ENV=json.dumps(config), PROFILE_REFERENCE_ENV=json.dumps(reference_config),
           PROFILE_AB_PROMPT=args.prompt, PROFILE_AB_ROUNDS=str(args.rounds),
           PROFILE_AB_DECODE_STEPS=str(args.decode_steps), PROFILE_MIN_GPU_SPEEDUP=str(args.gpu_target),
           PROFILE_MIN_SESSION_WALL_SPEEDUP=str(args.session_wall_target),
           PROFILE_MAX_DECODE_GPU_RATIO=str(args.max_decode_gpu_ratio))
env.update(PROFILE_MIN_DECODE_GPU_SPEEDUP=str(args.decode_gpu_target),
           PROFILE_MIN_DECODE_WALL_SPEEDUP=str(args.decode_wall_target))
raise SystemExit(subprocess.run([sys.executable, str(HERE.parent / "llama-deep-profile/model_ab.py"),
                                str(args.model.resolve())], cwd=ROOT, env=env).returncode)
