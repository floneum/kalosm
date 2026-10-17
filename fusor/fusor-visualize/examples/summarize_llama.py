"""Render visualize_kernels' JSON inventory as a readable local report.

Usage: python3 summarize_llama.py /path/to/report.json
Writes report.txt beside the JSON. Requires only the Python standard library.
"""
import collections
import json
import pathlib
import sys


def summarize(report):
    rows = report["kernels"]
    lines = [
        "LLAMA VIRTUAL GPU KERNEL SWEEP",
        f"Model: {report['model_path']}",
        f"Variants: {report['variants']}; analyzed: {report['analyzed']}; trace bounds failures: {report.get('trace_bounds_errors', 0)}; unsupported: {report['unsupported']}",
        f"Configuration: {json.dumps(report['config'], sort_keys=True)}",
        f"Scope: {report['scope']}",
        f"Sampling: {report['sampling']}",
        "Counts describe explicit IR uses before backend CSE/vectorization, not measured device traffic.",
        "Observed variants include compiler candidates and are not weighted by actual dispatch frequency.",
        report.get("bounds_note", ""),
        "",
        "INFERENCE RUNS",
    ]
    for run in report["runs"]:
        lines.append(f"{run['name']}: prompt={run['prompt_tokens']} tokens; output={run['emitted_chunks']}/{run['requested_chunks']} text chunks; completed={run['completed']}; text={run['text']!r}")
    if report["inference_error"]:
        lines.append(f"INCOMPLETE INFERENCE: {report['inference_error']}")
    lines += ["", "COVERAGE BY KERNEL FAMILY", "family                 variants  analyzed  bounds_failed  unsupported"]
    families = collections.defaultdict(collections.Counter)
    errors = collections.Counter()
    for row in rows:
        families[row["name"]][row["status"]] += 1
        if row["status"] != "analyzed":
            error = row["error"]
            if "unknown" in error or "depends on loaded data" in error:
                error = "data-dependent address/count/predicate without concrete input values"
            elif "out-of-bounds" in error:
                error = "captured access exceeds declared resource extent"
            elif "budget" in error:
                error = "capture action budget exceeded"
            errors[error] += 1
    for name, counts in sorted(families.items()):
        lines.append(f"{name:24} {sum(counts.values()):7} {counts['analyzed']:9} {counts['trace_bounds_error']:14} {counts['unsupported']:12}")
    lines += ["", "COVERAGE GAPS"]
    for error, count in errors.most_common():
        lines.append(f"{count:5}  {error}")

    # Report one strongest example per static source/family, avoiding hundreds
    # of identical-looking rows for shapes and repeated compiler candidates.
    hotspots = {}
    for row in rows:
        for spot in row.get("hotspots", []):
            m = spot["metrics"]
            ratio = m["transactions"] / max(1, m["ideal_transactions"])
            score = (ratio, m["transactions"], m["max_bank_degree"] - m["ideal_bank_degree"])
            key = (row["name"], spot["source"])
            if key not in hotspots or score > hotspots[key][0]:
                hotspots[key] = (score, row, spot)
    lines += ["", "COALESCING / BANK HOTSPOTS (up to 15 distinct family/source pairs)"]
    for score, row, spot in sorted(hotspots.values(), key=lambda v: v[0], reverse=True)[:15]:
        m = spot["metrics"]
        lines.extend([
            f"{row['name']} [{row['id']}] {spot['source']}",
            f"  WG={spot['workgroup']} SG={spot['subgroup']} storage_sectors={m['transactions']} packed_lower_bound={m['ideal_transactions']} ratio={score[0]:.2f}",
            f"  bank_rounds={m['max_bank_degree']} ideal={m['ideal_bank_degree']} cross_subgroup_reuse={m['cross_subgroup_reuse_bytes']} B",
            f"  Replay: {row['trace']}",
        ])
        lines.extend(f"  {hint}" for hint in spot["hints"])

    lines += ["", "ALL KERNEL VARIANTS", "id  family  grid  block  observations  status  result"]
    for row in rows:
        if row["status"] == "analyzed":
            m = row["metrics"]
            result = f"sectors={m['transactions']}/{m['ideal_transactions']} cache_hit/miss={m['cache_hits']}/{m['cache_misses']} evictions={m['cache_evictions']} cross_subgroup_reuse={m['cross_subgroup_reuse_bytes']} B bank_conflict_steps={m['bank_conflict_steps']} trace={row['trace']}"
        else:
            result = row["error"]
        lines.append(f"{row['id']}  {row['name']}  {row['grid']}  {row['block']}  {row['observations']}  {row['status']}  {result}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    source = pathlib.Path(sys.argv[1])
    report = json.loads(source.read_text())
    destination = source.with_suffix(".txt")
    destination.write_text(summarize(report))
    print(destination)
