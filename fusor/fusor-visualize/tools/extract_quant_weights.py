#!/usr/bin/env python3
"""Copy named GGUF projections into native blocks for qmatmul_lab.

Multiple matrices concatenate output rows; their format and K must match.
The GGUF source is opened read-only. Output and provenance are create-new.
"""
import argparse
import hashlib
import json
import math
import mmap
from pathlib import Path

from gguf_directory import directory


def extract(model, names, output):
    metadata = output.with_suffix(output.suffix + ".json")
    assert not output.exists() and not metadata.exists(), "output already exists"
    assert names and len(set(names)) == len(names), "tensor names must be unique"
    with model.open("rb") as source, mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as data:
        base, tensors = directory(data)
        by_name = {t["name"]: t for t in tensors}
        selected = [by_name[name] for name in names]
        first = selected[0]
        assert len(first["shape"]) == 2 and first["tag"] in (12, 14)
        k = first["shape"][1]
        assert k > 0 and k % 256 == 0
        block_bytes = 144 if first["tag"] == 12 else 210
        spans = []
        for tensor in selected:
            assert len(tensor["shape"]) == 2 and tensor["shape"][0] > 0
            assert tensor["tag"] == first["tag"] and tensor["shape"][1] == k
            size = math.prod(tensor["shape"]) // 256 * block_bytes
            offset = base + tensor["offset"]
            assert offset + size <= len(data)
            spans.append(dict(**tensor, file_offset=offset, native_bytes=size))
        digest = hashlib.sha256()
        with output.open("xb") as target:
            for span in spans:
                start, end = span["file_offset"], span["file_offset"] + span["native_bytes"]
                for offset in range(start, end, 1024 * 1024):
                    chunk = data[offset:min(offset + 1024 * 1024, end)]
                    target.write(chunk)
                    digest.update(chunk)
        result = dict(model=str(model.resolve()), model_bytes=len(data), tensors=spans,
                      n=sum(t["shape"][0] for t in selected), k=k,
                      format="q4k" if first["tag"] == 12 else "q6k",
                      sha256=digest.hexdigest(), native_bytes=sum(s["native_bytes"] for s in spans))
        with metadata.open("x") as target:
            json.dump(result, target, indent=2)
            target.write("\n")
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("tensors", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(extract(args.model, args.tensors, args.output), indent=2))
