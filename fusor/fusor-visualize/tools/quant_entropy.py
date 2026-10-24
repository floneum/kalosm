#!/usr/bin/env python3
"""Inspect native GGUF Q4_K/Q6_K coefficient entropy without changing weights.

The byte estimate uses a separate zero-order coefficient histogram per tensor.
It is an entropy-coding scenario, not a bound on conditional compression or a GPU
speedup. It excludes offsets, padding, tables and decoding cost. Requires numpy.
"""
import argparse
import collections
import json
import math
import mmap
from pathlib import Path

import numpy as np


from gguf_directory import directory


def coefficients(blocks, tag):
    if tag == 12:
        payload = blocks[:, 16:144]
        return np.concatenate((payload & 15, payload >> 4), axis=1)
    assert tag == 14
    lo = blocks[:, :128].reshape(-1, 2, 2, 32)
    hi = blocks[:, 128:192].reshape(-1, 2, 32)
    return np.concatenate(((lo[:, :, 0] & 15) | ((hi & 3) << 4),
                           (lo[:, :, 1] & 15) | (((hi >> 2) & 3) << 4),
                           (lo[:, :, 0] >> 4) | (((hi >> 4) & 3) << 4),
                           (lo[:, :, 1] >> 4) | ((hi >> 6) << 4)), axis=2).reshape(-1, 256)


def entropy(hist):
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def analyze(path, samples):
    output = []
    with path.open('rb') as source, mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as data:
        base, records = directory(data)
        for tensor in records:
            tag = tensor['tag']
            if tag not in (12, 14) or tensor['name'] == 'token_embd.weight':
                continue
            size, payload, bits = (144, 128, 4) if tag == 12 else (210, 192, 6)
            count = math.prod(tensor['shape']) // 256
            assert count > 0 and math.prod(tensor['shape']) % 256 == 0
            offset = base + tensor['offset']
            assert offset + count * size <= len(data)
            view = np.ndarray((count, size), dtype=np.uint8, buffer=data, offset=offset)
            indices = np.linspace(0, count - 1, min(count, samples), dtype=np.int64)
            blocks = view[indices].copy()
            del view
            values = coefficients(blocks, tag)
            hist = np.bincount(values.ravel(), minlength=1 << bits)
            h = entropy(hist)
            output.append(dict(**tensor, sampled_blocks=len(indices), blocks=count,
                               native_bytes=count * size, coefficient_bits=bits,
                               entropy_bits=h, histogram=hist.tolist(),
                               ideal_entropy_bytes=count * ((size - payload) + 256 * h / 8)))
    families = collections.defaultdict(lambda: dict(native_bytes=0, ideal_entropy_bytes=0, tensors=0))
    for row in output:
        family = families['Q4_K' if row['tag'] == 12 else 'Q6_K']
        for key in ('native_bytes', 'ideal_entropy_bytes'):
            family[key] += row[key]
        family['tensors'] += 1
    native = sum(v['native_bytes'] for v in families.values())
    ideal = sum(v['ideal_entropy_bytes'] for v in families.values())
    return dict(model=str(path), samples_per_tensor=samples, assumption=__doc__,
                native_bytes=native, ideal_entropy_bytes=ideal,
                ideal_byte_ratio=ideal / native, families=dict(families), tensors=output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--samples', type=int, default=4096)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert 1 <= args.samples <= 65536
    result = analyze(args.model, args.samples)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('tensors', 'assumption')}, indent=2))
