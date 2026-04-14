#!/usr/bin/env python3

import argparse
import json
import struct
from pathlib import Path

import numpy as np


GGUF_VERSION = 3
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4K = 12
METADATA_TYPE_U32 = 4
METADATA_TYPE_STRING = 8
ALIGNMENT = 32
K_BLOCK_SIZE = 256
Q4K_BLOCK_BYTES = 2 + 2 + 12 + (K_BLOCK_SIZE // 2)
Q4_0_BLOCK_SIZE = 32
Q4_0_BLOCK_BYTES = 2 + (Q4_0_BLOCK_SIZE // 2)


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def quantization_type(name: str, shape: tuple[int, ...]) -> int:
    if len(shape) != 2:
        return GGML_TYPE_F16
    if shape[-1] % K_BLOCK_SIZE == 0:
        return GGML_TYPE_Q4K
    if shape[-1] % Q4_0_BLOCK_SIZE == 0:
        return GGML_TYPE_Q4_0
    return GGML_TYPE_F16


def gguf_name(name: str) -> str:
    if name == "proj_out.weight":
        return "model.decoder.proj_out.weight"
    return name


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
    if ggml_type == GGML_TYPE_Q4K:
        assert len(shape) == 2
        assert shape[-1] % K_BLOCK_SIZE == 0
        return (numel // K_BLOCK_SIZE) * Q4K_BLOCK_BYTES
    if ggml_type == GGML_TYPE_Q4_0:
        assert len(shape) == 2
        assert shape[-1] % Q4_0_BLOCK_SIZE == 0
        return (numel // Q4_0_BLOCK_SIZE) * Q4_0_BLOCK_BYTES
    if ggml_type == GGML_TYPE_F16:
        return numel * 2
    raise ValueError(f"unsupported ggml type: {ggml_type}")


def raw_tensor_bytes(mm: np.memmap, data_base: int, start: int, end: int) -> bytes:
    return memoryview(mm[data_base + start : data_base + end]).tobytes()


def decode_tensor(
    mm: np.memmap,
    data_base: int,
    dtype: str,
    shape: tuple[int, ...],
    start: int,
    end: int,
) -> np.ndarray:
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


def pack_q4k_scales(scales: np.ndarray, offsets: np.ndarray) -> bytes:
    first = np.zeros(4, dtype=np.uint8)
    middle = np.zeros(4, dtype=np.uint8)
    last = np.zeros(4, dtype=np.uint8)

    for i in range(4):
        first[i] = (int(scales[i]) & 0x3F) | (((int(scales[i + 4]) >> 4) & 0x03) << 6)
        middle[i] = (int(offsets[i]) & 0x3F) | (((int(offsets[i + 4]) >> 4) & 0x03) << 6)
        last[i] = (int(scales[i + 4]) & 0x0F) | ((int(offsets[i + 4]) & 0x0F) << 4)

    return bytes(first.tolist() + middle.tolist() + last.tolist())


def q4_0_bytes(array: np.ndarray) -> bytes:
    assert array.ndim == 2
    rows, cols = array.shape
    assert cols % Q4_0_BLOCK_SIZE == 0

    blocks = np.asarray(array, dtype=np.float32).reshape(-1, Q4_0_BLOCK_SIZE)

    # ggml Q4_0: d = max_signed / -8, where max_signed is the value with largest magnitude
    # (sign preserved). q = clamp(round(v / d) + 8, 0, 15).
    abs_blocks = np.abs(blocks)
    max_idx = np.argmax(abs_blocks, axis=1)
    max_signed = np.take_along_axis(blocks, max_idx[:, None], axis=1).squeeze(1)
    scales = (max_signed / -8.0).astype(np.float32)
    inv_scales = np.divide(1.0, scales, out=np.zeros_like(scales), where=scales != 0.0)
    quantized = np.clip(
        np.rint(blocks * inv_scales[:, None]) + 8.0, 0.0, 15.0
    ).astype(np.uint8)

    # Pack: low nibble at position i (i < 16), high nibble at position i+16.
    low = quantized[:, : Q4_0_BLOCK_SIZE // 2]
    high = quantized[:, Q4_0_BLOCK_SIZE // 2 :]
    packed_weights = (low | (high << 4)).astype(np.uint8)

    packed = np.empty(
        blocks.shape[0],
        dtype=np.dtype([("d", "<f2"), ("qs", "u1", (Q4_0_BLOCK_SIZE // 2,))]),
    )
    packed["d"] = scales.astype(np.float16)
    packed["qs"] = packed_weights
    return packed.tobytes()


def q4k_bytes(array: np.ndarray) -> bytes:
    assert array.ndim == 2
    rows, cols = array.shape
    assert cols % K_BLOCK_SIZE == 0

    blocks = np.asarray(array, dtype=np.float32).reshape(-1, K_BLOCK_SIZE)
    out = bytearray(blocks.shape[0] * Q4K_BLOCK_BYTES)
    cursor = 0

    for block in blocks:
        local_scales = np.zeros(8, dtype=np.float32)
        local_offsets = np.zeros(8, dtype=np.float32)
        quantized_groups = []

        groups = [block[i * 32 : (i + 1) * 32] for i in range(8)]
        for group in groups:
            group_min = float(group.min())
            group_max = float(group.max())
            offset = max(0.0, -group_min)
            scale = max(0.0, (group_max + offset) / 15.0)
            local_scales[len(quantized_groups)] = scale
            local_offsets[len(quantized_groups)] = offset
            quantized_groups.append(group)

        super_scale = float(local_scales.max() / 63.0) if local_scales.max() > 0 else 0.0
        super_min = float(local_offsets.max() / 63.0) if local_offsets.max() > 0 else 0.0

        scale_codes = np.zeros(8, dtype=np.uint8)
        offset_codes = np.zeros(8, dtype=np.uint8)
        packed_weights = np.zeros(K_BLOCK_SIZE // 2, dtype=np.uint8)

        if super_scale > 0:
            scale_codes = np.clip(np.rint(local_scales / super_scale), 0, 63).astype(np.uint8)
        if super_min > 0:
            offset_codes = np.clip(np.rint(local_offsets / super_min), 0, 63).astype(np.uint8)

        for pair in range(4):
            low_idx = pair * 2
            high_idx = low_idx + 1

            low_scale = float(scale_codes[low_idx]) * super_scale
            high_scale = float(scale_codes[high_idx]) * super_scale
            low_offset = float(offset_codes[low_idx]) * super_min
            high_offset = float(offset_codes[high_idx]) * super_min

            low_group = groups[low_idx]
            high_group = groups[high_idx]

            if low_scale > 0:
                low_q = np.clip(np.rint((low_group + low_offset) / low_scale), 0, 15).astype(np.uint8)
            else:
                low_q = np.zeros(32, dtype=np.uint8)
            if high_scale > 0:
                high_q = np.clip(np.rint((high_group + high_offset) / high_scale), 0, 15).astype(np.uint8)
            else:
                high_q = np.zeros(32, dtype=np.uint8)

            packed_weights[pair * 32 : (pair + 1) * 32] = low_q | (high_q << 4)

        out[cursor : cursor + 2] = np.asarray(super_scale, dtype="<f2").tobytes()
        cursor += 2
        out[cursor : cursor + 2] = np.asarray(super_min, dtype="<f2").tobytes()
        cursor += 2
        out[cursor : cursor + 12] = pack_q4k_scales(scale_codes, offset_codes)
        cursor += 12
        out[cursor : cursor + (K_BLOCK_SIZE // 2)] = packed_weights.tobytes()
        cursor += K_BLOCK_SIZE // 2

    return bytes(out)


def tensor_bytes(
    mm: np.memmap,
    data_base: int,
    name: str,
    dtype: str,
    shape: tuple[int, ...],
    start: int,
    end: int,
) -> tuple[int, bytes]:
    ggml_type = quantization_type(name, shape)
    if ggml_type == GGML_TYPE_Q4K:
        array = decode_tensor(mm, data_base, dtype, shape, start, end)
        return GGML_TYPE_Q4K, q4k_bytes(array)
    if ggml_type == GGML_TYPE_Q4_0:
        array = decode_tensor(mm, data_base, dtype, shape, start, end)
        return GGML_TYPE_Q4_0, q4_0_bytes(array)

    if dtype == "F16":
        return GGML_TYPE_F16, raw_tensor_bytes(mm, data_base, start, end)

    array = decode_tensor(mm, data_base, dtype, shape, start, end)
    return GGML_TYPE_F16, np.asarray(array, dtype=np.float16).tobytes()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Path to model.safetensors")
    parser.add_argument("--output", type=Path, required=True, help="Path to output model.gguf")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="Path to tokenizer.json (defaults to a tokenizer.json next to --input)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config.json (defaults to a config.json next to --input)",
    )
    args = parser.parse_args()

    tokenizer_path = args.tokenizer or args.input.with_name("tokenizer.json")
    config_path = args.config or args.input.with_name("config.json")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"missing tokenizer json: {tokenizer_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"missing config json: {config_path}")
    tokenizer_json = tokenizer_path.read_text(encoding="utf-8")
    config_json = config_path.read_text(encoding="utf-8")

    metadata = {
        "general.architecture": ("string", "moonshine_streaming_asr"),
        "general.alignment": ("u32", ALIGNMENT),
        "rwhisper.tokenizer.json": ("string", tokenizer_json),
        "rwhisper.config.json": ("string", config_json),
    }

    data_base, source_tensors = load_safetensors_index(args.input)
    tensors = []
    for source_name, _, shape, _, _ in source_tensors:
        name = gguf_name(source_name)
        ggml_type = quantization_type(source_name, shape)
        tensors.append((name, source_name, shape, ggml_type, tensor_size_bytes(shape, ggml_type)))

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
    for name, source_name, shape, ggml_type, byte_len in tensors:
        offset = align_up(offset, ALIGNMENT)
        tensor_infos.append((name, source_name, shape, ggml_type, offset, byte_len))

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
        for name, source_name, _, _, offset, _ in tensor_infos:
            if offset > cursor:
                handle.write(b"\0" * (offset - cursor))
            dtype, shape, start, end = source_map[source_name]
            _, raw = tensor_bytes(mm, data_base, source_name, dtype, shape, start, end)
            handle.write(raw)
            cursor = offset + len(raw)

    q4k_count = sum(
        1 for _, source_name, shape, _, _ in tensors if quantization_type(source_name, shape) == GGML_TYPE_Q4K
    )
    q4_0_count = sum(
        1 for _, source_name, shape, _, _ in tensors if quantization_type(source_name, shape) == GGML_TYPE_Q4_0
    )
    print(f"wrote {args.output} with {len(tensors)} tensors ({q4k_count} q4k, {q4_0_count} q4_0)")


if __name__ == "__main__":
    main()
