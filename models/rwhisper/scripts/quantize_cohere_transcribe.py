#!/usr/bin/env python3

import argparse
import json
import struct
from pathlib import Path

import numpy as np


GGUF_VERSION = 3
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8
METADATA_TYPE_U32 = 4
METADATA_TYPE_STRING = 8
ALIGNMENT = 32


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def should_quantize(name: str, shape: tuple[int, ...]) -> bool:
    if len(shape) != 2:
        return False
    if shape[-1] % 32 != 0:
        return False
    if name.endswith(".pos_enc"):
        return False
    if name.endswith(".pos_bias_u") or name.endswith(".pos_bias_v"):
        return False
    return True


def load_safetensors_index(path: Path) -> tuple[int, list[tuple[str, str, tuple[int, ...], int, int]]]:
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))

    tensors = []
    for name, entry in sorted(header.items()):
        if name == "__metadata__" or name.endswith("num_batches_tracked"):
            continue
        dtype = entry["dtype"]
        shape = tuple(entry["shape"])
        start, end = entry["data_offsets"]
        tensors.append((name, dtype, shape, start, end))
    return 8 + header_len, tensors


def tensor_size_bytes(shape: tuple[int, ...], ggml_type: int) -> int:
    numel = 1
    for dim in shape:
        numel *= dim
    if ggml_type == GGML_TYPE_Q8_0:
        assert len(shape) == 2
        assert shape[-1] % 32 == 0
        return (numel // 32) * 34
    if ggml_type == GGML_TYPE_F16:
        return numel * 2
    raise ValueError(f"unsupported ggml type: {ggml_type}")


def raw_tensor_bytes(mm: np.memmap, data_base: int, start: int, end: int) -> bytes:
    return memoryview(mm[data_base + start : data_base + end]).tobytes()


def decode_tensor(mm: np.memmap, data_base: int, dtype: str, shape: tuple[int, ...], start: int, end: int) -> np.ndarray:
    raw = memoryview(mm[data_base + start : data_base + end])
    numel = int(np.prod(shape))

    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2", count=numel).reshape(shape)
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4", count=numel).reshape(shape)
    if dtype == "BF16":
        bf16 = np.frombuffer(raw, dtype="<u2", count=numel)
        fp32 = (bf16.astype(np.uint32) << 16).view(np.float32)
        return fp32.reshape(shape)

    raise ValueError(f"unsupported safetensors dtype: {dtype}")


def q8_0_bytes(array: np.ndarray) -> bytes:
    assert array.ndim == 2
    rows, cols = array.shape
    assert cols % 32 == 0

    blocks = np.asarray(array, dtype=np.float32).reshape(-1, 32)
    max_abs = np.max(np.abs(blocks), axis=1)
    scales = np.divide(max_abs, 127.0, out=np.zeros_like(max_abs), where=max_abs != 0).astype(np.float16)
    inv_scales = np.divide(127.0, max_abs, out=np.zeros_like(max_abs), where=max_abs != 0)
    quantized = np.clip(np.rint(blocks * inv_scales[:, None]), -127, 127).astype(np.int8)

    packed = np.empty(blocks.shape[0], dtype=np.dtype([("d", "<f2"), ("qs", "i1", (32,))]))
    packed["d"] = scales
    packed["qs"] = quantized
    return packed.tobytes()


def tensor_bytes(
    mm: np.memmap,
    data_base: int,
    name: str,
    dtype: str,
    shape: tuple[int, ...],
    start: int,
    end: int,
) -> tuple[int, bytes]:
    if should_quantize(name, shape):
        array = decode_tensor(mm, data_base, dtype, shape, start, end)
        return GGML_TYPE_Q8_0, q8_0_bytes(array)

    if dtype == "F16":
        return GGML_TYPE_F16, raw_tensor_bytes(mm, data_base, start, end)

    array = decode_tensor(mm, data_base, dtype, shape, start, end)
    return GGML_TYPE_F16, np.asarray(array, dtype=np.float16).tobytes()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Path to model.safetensors")
    parser.add_argument("--output", type=Path, required=True, help="Path to output model.gguf")
    args = parser.parse_args()

    metadata = {
        "general.architecture": ("string", "cohere_asr"),
        "general.alignment": ("u32", ALIGNMENT),
    }

    data_base, source_tensors = load_safetensors_index(args.input)
    tensors = []
    for name, _, shape, _, _ in source_tensors:
        ggml_type = GGML_TYPE_Q8_0 if should_quantize(name, shape) else GGML_TYPE_F16
        tensors.append((name, shape, ggml_type, tensor_size_bytes(shape, ggml_type)))

    header = bytearray()
    header.extend(b"GGUF")
    header.extend(struct.pack("<I", GGUF_VERSION))
    header.extend(struct.pack("<Q", len(tensors)))
    header.extend(struct.pack("<Q", len(metadata)))

    for key, (value_type, value) in metadata.items():
        header.extend(struct.pack("<Q", len(key)))
        header.extend(key.encode("utf-8"))
        if value_type == "string":
            header.extend(struct.pack("<I", METADATA_TYPE_STRING))
            value_bytes = value.encode("utf-8")
            header.extend(struct.pack("<Q", len(value_bytes)))
            header.extend(value_bytes)
        elif value_type == "u32":
            header.extend(struct.pack("<I", METADATA_TYPE_U32))
            header.extend(struct.pack("<I", int(value)))
        else:
            raise ValueError(f"unsupported metadata value type: {value_type}")

    offset = 0
    tensor_infos = []
    for name, shape, ggml_type, byte_len in tensors:
        offset = align_up(offset, ALIGNMENT)
        tensor_infos.append((name, shape, ggml_type, offset, byte_len))

        header.extend(struct.pack("<Q", len(name)))
        header.extend(name.encode("utf-8"))
        header.extend(struct.pack("<I", len(shape)))
        for dim in reversed(shape):
            header.extend(struct.pack("<Q", dim))
        header.extend(struct.pack("<I", ggml_type))
        header.extend(struct.pack("<Q", offset))

        offset += byte_len

    tensor_data_offset = align_up(len(header), ALIGNMENT)
    source_map = {name: (dtype, shape, start, end) for name, dtype, shape, start, end in source_tensors}
    mm = np.memmap(args.input, mode="r", dtype=np.uint8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        handle.write(header)
        if tensor_data_offset > len(header):
            handle.write(b"\0" * (tensor_data_offset - len(header)))

        cursor = 0
        for name, _, _, offset, _ in tensor_infos:
            if offset > cursor:
                handle.write(b"\0" * (offset - cursor))
            dtype, shape, start, end = source_map[name]
            _, raw = tensor_bytes(mm, data_base, name, dtype, shape, start, end)
            handle.write(raw)
            cursor = offset + len(raw)

    quantized = sum(1 for name, shape, _, _ in tensors if should_quantize(name, shape))
    print(f"wrote {args.output} with {len(tensors)} tensors ({quantized} q8_0)")


if __name__ == "__main__":
    main()
