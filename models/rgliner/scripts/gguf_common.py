"""Shared GGUF-writing infrastructure for the GLiNER converters.

Both `convert_to_gguf.py` (bi-encoder) and `convert_relex_to_gguf.py` (RelEx)
emit GGUF files with the byte layout fusor expects. This module holds the
quantisation tables, GGUF/GGML constants, and the `GGUFWriter` they share; each
converter keeps its own model-specific weight mapping and `main()`.
"""

import struct
from typing import Any, Dict, List, Tuple

import numpy as np

__all__ = [
    "IN_PROCESS_QUANTS",
    "LLAMA_QUANT_TYPES",
    "QUANT_TYPES",
    "GGUF_MAGIC",
    "GGUF_VERSION",
    "GGUF_TYPE_UINT8",
    "GGUF_TYPE_INT8",
    "GGUF_TYPE_UINT16",
    "GGUF_TYPE_INT16",
    "GGUF_TYPE_UINT32",
    "GGUF_TYPE_INT32",
    "GGUF_TYPE_FLOAT32",
    "GGUF_TYPE_BOOL",
    "GGUF_TYPE_STRING",
    "GGUF_TYPE_ARRAY",
    "GGUF_TYPE_UINT64",
    "GGUF_TYPE_INT64",
    "GGUF_TYPE_FLOAT64",
    "GGML_TYPE_F32",
    "GGML_TYPE_F16",
    "GGML_TYPE_Q4_0",
    "GGML_TYPE_Q4_1",
    "GGML_TYPE_Q5_0",
    "GGML_TYPE_Q5_1",
    "GGML_TYPE_Q8_0",
    "GGML_TYPE_Q8_1",
    "GGML_TYPE_BF16",
    "_ggml_type_for",
    "_name_for_type",
    "_gguf_block_quant",
    "_U32",
    "GGUFWriter",
]


# Quantization targets. In-process ones are packed directly by `gguf.quants.quantize`;
# k-quants are produced by shelling out to `llama-quantize` post-hoc.
IN_PROCESS_QUANTS = {
    "f32", "f16", "bf16",
    "q4_0", "q4_1", "q5_0", "q5_1", "q8_0",
}
LLAMA_QUANT_TYPES = {
    "q2_k", "q3_k", "q3_k_s", "q3_k_m", "q3_k_l",
    "q4_k", "q4_k_s", "q4_k_m",
    "q5_k", "q5_k_s", "q5_k_m",
    "q6_k",
}
QUANT_TYPES = IN_PROCESS_QUANTS | LLAMA_QUANT_TYPES


# GGUF constants
GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 3

# GGUF data types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# GGML tensor types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_BF16 = 30


def _ggml_type_for(quant: str) -> int:
    return {
        "f32": GGML_TYPE_F32,
        "f16": GGML_TYPE_F16,
        "bf16": GGML_TYPE_BF16,
        "q4_0": GGML_TYPE_Q4_0,
        "q4_1": GGML_TYPE_Q4_1,
        "q5_0": GGML_TYPE_Q5_0,
        "q5_1": GGML_TYPE_Q5_1,
        "q8_0": GGML_TYPE_Q8_0,
    }[quant]


def _name_for_type(ggml_type: int) -> str:
    return {
        GGML_TYPE_F32: "F32",
        GGML_TYPE_F16: "F16",
        GGML_TYPE_BF16: "BF16",
        GGML_TYPE_Q4_0: "Q4_0",
        GGML_TYPE_Q4_1: "Q4_1",
        GGML_TYPE_Q5_0: "Q5_0",
        GGML_TYPE_Q5_1: "Q5_1",
        GGML_TYPE_Q8_0: "Q8_0",
    }.get(ggml_type, f"type={ggml_type}")


def _gguf_block_quant(data: np.ndarray, quant: str) -> np.ndarray:
    """Return the packed byte array for a block-quantised tensor.

    Uses `gguf.quants.quantize` (Python package from llama.cpp). The inner row
    dimension must be a multiple of 32 for q4_0/q5_0/q8_0, else we fall back to
    f16 for that tensor (caller decides).
    """
    import gguf  # Deferred import - only needed for block quants.
    qtype = {
        "q4_0": gguf.GGMLQuantizationType.Q4_0,
        "q4_1": gguf.GGMLQuantizationType.Q4_1,
        "q5_0": gguf.GGMLQuantizationType.Q5_0,
        "q5_1": gguf.GGMLQuantizationType.Q5_1,
        "q8_0": gguf.GGMLQuantizationType.Q8_0,
    }[quant]
    return gguf.quants.quantize(data, qtype)


class _U32(int):
    """Force a metadata int to be written as GGUF u32 (required by llama-quantize's arch loader)."""


class GGUFWriter:
    """Simple GGUF file writer."""

    def __init__(self, path: str):
        self.path = path
        self.metadata: Dict[str, Any] = {}
        # (name, data, ggml_type, logical_shape)
        self.tensors: List[Tuple[str, np.ndarray, int, Tuple[int, ...]]] = []

    def add_metadata(self, key: str, value: Any):
        """Add metadata key-value pair."""
        self.metadata[key] = value

    def add_tensor(
        self,
        name: str,
        data: np.ndarray,
        ggml_type: int = GGML_TYPE_F32,
        shape: Tuple[int, ...] = None,
    ):
        """Add a tensor. `shape` is the logical (un-packed) shape; defaults to data.shape."""
        if shape is None:
            shape = tuple(data.shape)
        self.tensors.append((name, data, ggml_type, tuple(shape)))

    def _write_string(self, f, s: str):
        """Write a GGUF string (length-prefixed UTF-8)."""
        encoded = s.encode('utf-8')
        f.write(struct.pack('<Q', len(encoded)))
        f.write(encoded)

    def _write_metadata_value(self, f, value: Any):
        """Write a metadata value with its type."""
        if isinstance(value, bool):
            f.write(struct.pack('<I', GGUF_TYPE_BOOL))
            f.write(struct.pack('<B', 1 if value else 0))
        elif isinstance(value, _U32):
            f.write(struct.pack('<I', GGUF_TYPE_UINT32))
            f.write(struct.pack('<I', int(value)))
        elif isinstance(value, int):
            if value < 0:
                f.write(struct.pack('<I', GGUF_TYPE_INT64))
                f.write(struct.pack('<q', value))
            else:
                f.write(struct.pack('<I', GGUF_TYPE_UINT64))
                f.write(struct.pack('<Q', value))
        elif isinstance(value, float):
            f.write(struct.pack('<I', GGUF_TYPE_FLOAT32))
            f.write(struct.pack('<f', value))
        elif isinstance(value, str):
            f.write(struct.pack('<I', GGUF_TYPE_STRING))
            self._write_string(f, value)
        elif isinstance(value, (list, tuple)):
            f.write(struct.pack('<I', GGUF_TYPE_ARRAY))
            if len(value) == 0:
                f.write(struct.pack('<I', GGUF_TYPE_UINT32))
                f.write(struct.pack('<Q', 0))
            elif isinstance(value[0], int):
                f.write(struct.pack('<I', GGUF_TYPE_INT64))
                f.write(struct.pack('<Q', len(value)))
                for v in value:
                    f.write(struct.pack('<q', v))
            elif isinstance(value[0], float):
                f.write(struct.pack('<I', GGUF_TYPE_FLOAT32))
                f.write(struct.pack('<Q', len(value)))
                for v in value:
                    f.write(struct.pack('<f', v))
            elif isinstance(value[0], str):
                f.write(struct.pack('<I', GGUF_TYPE_STRING))
                f.write(struct.pack('<Q', len(value)))
                for v in value:
                    self._write_string(f, v)
            else:
                raise ValueError(f"Unsupported array element type: {type(value[0])}")
        else:
            raise ValueError(f"Unsupported metadata type: {type(value)}")

    def write(self):
        """Write the GGUF file."""
        with open(self.path, 'wb') as f:
            # Header
            f.write(struct.pack('<I', GGUF_MAGIC))
            f.write(struct.pack('<I', GGUF_VERSION))
            f.write(struct.pack('<Q', len(self.tensors)))  # n_tensors
            f.write(struct.pack('<Q', len(self.metadata)))  # n_kv

            # Metadata
            for key, value in self.metadata.items():
                self._write_string(f, key)
                self._write_metadata_value(f, value)

            # Tensor infos (we'll write actual data after alignment)
            tensor_data_offset = 0
            tensor_infos = []

            for name, data, ggml_type, shape in self.tensors:
                # Ensure contiguous and correct dtype
                if ggml_type == GGML_TYPE_F32:
                    data = np.ascontiguousarray(data, dtype=np.float32)
                elif ggml_type == GGML_TYPE_F16:
                    data = np.ascontiguousarray(data, dtype=np.float16)
                elif ggml_type == GGML_TYPE_BF16:
                    # Convert to bfloat16 via float32
                    data = np.ascontiguousarray(data, dtype=np.float32)
                    data = data.view(np.uint32)
                    data = ((data >> 16) & 0xFFFF).astype(np.uint16)

                # Write tensor info
                # GGUF stores dimensions in reverse order (column-major)
                # Reader reverses them back, so we write reversed to get original order
                self._write_string(f, name)
                f.write(struct.pack('<I', len(shape)))
                for dim in reversed(shape):
                    f.write(struct.pack('<Q', dim))
                f.write(struct.pack('<I', ggml_type))
                f.write(struct.pack('<Q', tensor_data_offset))

                tensor_infos.append((data, tensor_data_offset))
                tensor_data_offset += data.nbytes
                # Align to 32 bytes
                padding = (32 - (tensor_data_offset % 32)) % 32
                tensor_data_offset += padding

            # Alignment padding before tensor data
            current_pos = f.tell()
            alignment = 32
            padding_needed = (alignment - (current_pos % alignment)) % alignment
            f.write(b'\x00' * padding_needed)

            # Tensor data
            for (data, _), (name, _, ggml_type, _shape) in zip(tensor_infos, self.tensors):
                f.write(data.tobytes())
                # Align to 32 bytes
                padding = (32 - (data.nbytes % 32)) % 32
                f.write(b'\x00' * padding)

        print(f"Wrote GGUF file: {self.path}")
