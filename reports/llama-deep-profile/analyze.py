#!/usr/bin/env python3
"""Join semantic resolve metadata and timestamps, preserving composite launches."""
import collections
import gzip
import json
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = gzip.decompress(path.read_bytes()).decode() if path.suffix == ".gz" else path.read_text()
span_re = re.compile(r"TSPAN (\d+) (\S+) ([\d.]+) L(\d+) grid=(\[.*\])")
phases = []
phase = None
live = []
body_index = 0

def contracts(op):
    if op["kind"] == "contract":
        yield op
    for member in op.get("members", []):
        yield from contracts(member)

for line in text.splitlines():
    if line.startswith("PHASE_BEGIN "):
        phase = {"key": line.split()[1], "spans": [], "seen": 0}
    elif line.startswith("RESOLVE_META "):
        assert body_index == len(live), "previous plan lacks dispatch identities"
        meta = json.loads(line[len("RESOLVE_META "):])
        live = [l for l in meta["launches"] if all(l["grid"])]
        body_index = 0
    elif line.startswith("BODY "):
        body = line.split()[1]
        grid = json.loads(line.split("grid=", 1)[1])
        assert live[body_index]["body"] == body and live[body_index]["grid"] == grid
        body_index += 1
        phase["seen"] += 1
    elif match := span_re.fullmatch(line):
        index, family, us, launch, grid = match.groups()
        entry = live[int(index)]
        assert entry["index"] == int(launch) and entry["grid"] == json.loads(grid)
        phase["spans"].append(dict(entry, us=float(us), emitted_family=family))
    elif line.startswith("PHASE_END "):
        row = json.loads(line[len("PHASE_END "):])
        assert phase["key"] == row["key"]
        assert len(phase["spans"]) == phase["seen"] == row["dispatches"]
        phases.append(dict(row, spans=phase["spans"]))
        phase = None
assert phase is None
assert body_index == len(live), "last plan lacks dispatch identities"
out = {}
for phase_name in ("prefill", "decode"):
    rows = [p for p in phases if not p["warmup"] and p["phase"] == phase_name and (phase_name != "decode" or p["step"] > 1)]
    groups = collections.defaultdict(lambda: {"us": 0, "count": 0, "macs": 0, "operand_bytes": 0, "bodies": set()})
    total = 0
    for phase in rows:
        for s in phase["spans"]:
            total += s["us"]
            matrices = list(contracts(s["operation"]))
            key = json.dumps(s["operation"], sort_keys=True, separators=(",", ":"))
            entry = groups[key]
            entry["operation"] = s["operation"]
            entry["us"] += s["us"]
            entry["count"] += 1
            entry["bodies"].add(s["body"])
            entry["macs"] += sum(c["macs"] or 0 for c in matrices)
            entry["operand_bytes"] += sum(o["logical_bytes_native_quant"] or 0 for c in matrices for side in ("a", "b") for o in c[side])
    result = []
    for entry in sorted(groups.values(), key=lambda e: -e["us"]):
        entry["bodies"] = sorted(entry["bodies"])
        entry["percent"] = entry["us"] / total * 100
        entry["ms_per_phase"] = entry["us"] / len(rows) / 1000
        entry["logical_GBs"] = entry["operand_bytes"] / entry["us"] / 1000
        entry["GFLOPs"] = entry["macs"] * 2 / entry["us"] / 1000
        result.append(entry)
    out[phase_name] = {"samples": len(rows), "total_ms_per_phase": total / len(rows) / 1000 if rows else None, "groups": result}
    print(phase_name, "samples", len(rows), "GPU ms", out[phase_name]["total_ms_per_phase"])
    for row in result[:12]:
        cs = list(contracts(row["operation"]))
        desc = [(c["family"],c["m"],c["n"],c["k"],c["batch"],[o["dtype"] for o in c["b"]],c["schedule"]) for c in cs]
        print(f"  {row['ms_per_phase']:.3f} ms {row['percent']:.2f}% {row['count']/len(rows):.1f} calls {row['GFLOPs']:.1f} GFLOP/s", desc or row["operation"])
base = path.with_suffix("") if path.suffix == ".gz" else path
base.with_suffix(".summary.json").write_text(json.dumps(out, indent=2) + "\n")
