#!/usr/bin/env python3
"""
Convert GLiNER-RelEx PyTorch models to GGUF format.

Usage:
    python convert_relex_to_gguf.py --model knowledgator/gliner-relex-multi-v1.0 --output gliner-relex-multi-v1.0.gguf

This script converts the GLiNER-RelEx model including:
1. mDeBERTa-v3 encoder
2. Span representation layer
3. Relations representation layer (adjacency scoring)
4. Pair projector
5. Entity/relation label FFNs
"""

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

# In-process formats: we quantise per-tensor via `gguf.quants.quantize` so the
# file layout is byte-identical to what fusor expects.
IN_PROCESS_QUANTS = {
    "f32",
    "f16",
    "bf16",
    "q4_0",
    "q4_1",
    "q5_0",
    "q5_1",
    "q8_0",
}
# k-quants (Q4_K, Q5_K, Q6_K) need `llama-quantize` - the Python `gguf` package
# doesn't implement their `quantize_blocks` path. We write an f32 GGUF (with a
# temporary `general.architecture = "bert"` masquerade and the minimal bert.*
# metadata llama-quantize's loader validates) and then shell out to the binary.
LLAMA_QUANT_TYPES = {
    "q2_k",
    "q3_k",
    "q3_k_s",
    "q3_k_m",
    "q3_k_l",
    "q4_k",
    "q4_k_s",
    "q4_k_m",
    "q5_k",
    "q5_k_s",
    "q5_k_m",
    "q6_k",
}
QUANT_TYPES = IN_PROCESS_QUANTS | LLAMA_QUANT_TYPES

import numpy as np
import torch
from huggingface_hub import snapshot_download

# GGUF constants (same as convert_to_gguf.py)
GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3

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

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
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
    """Force a metadata int to be written as GGUF u32 (needed by llama-quantize for some keys)."""


class GGUFWriter:
    """Simple GGUF file writer."""

    def __init__(self, path: str):
        self.path = path
        self.metadata: Dict[str, Any] = {}
        self.tensors: List[Tuple[str, np.ndarray, int]] = []

    def add_metadata(self, key: str, value: Any):
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
        encoded = s.encode('utf-8')
        f.write(struct.pack('<Q', len(encoded)))
        f.write(encoded)

    def _write_metadata_value(self, f, value: Any):
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
            raise ValueError(f"Unsupported metadata type: {type(value)}")

    def write(self):
        with open(self.path, 'wb') as f:
            f.write(struct.pack('<I', GGUF_MAGIC))
            f.write(struct.pack('<I', GGUF_VERSION))
            f.write(struct.pack('<Q', len(self.tensors)))
            f.write(struct.pack('<Q', len(self.metadata)))

            for key, value in self.metadata.items():
                self._write_string(f, key)
                self._write_metadata_value(f, value)

            tensor_data_offset = 0
            tensor_infos = []

            for name, data, ggml_type, shape in self.tensors:
                if ggml_type == GGML_TYPE_F32:
                    data = np.ascontiguousarray(data, dtype=np.float32)
                elif ggml_type == GGML_TYPE_F16:
                    data = np.ascontiguousarray(data, dtype=np.float16)
                elif ggml_type == GGML_TYPE_BF16:
                    data = np.ascontiguousarray(data, dtype=np.float32)
                    data = data.view(np.uint32)
                    data = ((data >> 16) & 0xFFFF).astype(np.uint16)

                self._write_string(f, name)
                f.write(struct.pack('<I', len(shape)))
                for dim in reversed(shape):
                    f.write(struct.pack('<Q', dim))
                f.write(struct.pack('<I', ggml_type))
                f.write(struct.pack('<Q', tensor_data_offset))

                tensor_infos.append((data, tensor_data_offset))
                tensor_data_offset += data.nbytes
                padding = (32 - (tensor_data_offset % 32)) % 32
                tensor_data_offset += padding

            current_pos = f.tell()
            alignment = 32
            padding_needed = (alignment - (current_pos % alignment)) % alignment
            f.write(b'\x00' * padding_needed)

            for (data, _), (name, _, ggml_type, _shape) in zip(tensor_infos, self.tensors):
                f.write(data.tobytes())
                padding = (32 - (data.nbytes % 32)) % 32
                f.write(b'\x00' * padding)

        print(f"Wrote GGUF file: {self.path}")


def load_pytorch_model(model_id: str, cache_dir: str = None) -> Tuple[Dict[str, torch.Tensor], Dict, str, str]:
    """Load PyTorch model from HuggingFace.

    Returns:
        state_dict, config (parsed), gliner_config_json (raw text), tokenizer_json (raw text)
    """
    print(f"Downloading model: {model_id}")

    model_dir = snapshot_download(
        model_id,
        cache_dir=cache_dir,
        allow_patterns=["*.bin", "*.json", "*.safetensors"]
    )

    config_path = os.path.join(model_dir, "gliner_config.json")
    with open(config_path, 'r') as f:
        gliner_config_json = f.read()
    config = json.loads(gliner_config_json)

    tokenizer_path = os.path.join(model_dir, "tokenizer.json")
    with open(tokenizer_path, 'r') as f:
        tokenizer_json = f.read()

    weights_path = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.exists(weights_path):
        print(f"Loading weights from: {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
    else:
        from safetensors.torch import load_file
        weights_path = os.path.join(model_dir, "model.safetensors")
        print(f"Loading weights from: {weights_path}")
        state_dict = load_file(weights_path)

    return state_dict, config, gliner_config_json, tokenizer_json


def map_relex_weight_name(pytorch_name: str) -> str:
    """Map PyTorch weight names to GGUF conventions for GLiNER-RelEx."""
    name = pytorch_name

    # ===== Encoder output projection (large variants: 1024 -> 768) =====
    name = name.replace("token_rep_layer.projection.weight", "text.output_proj.weight")
    name = name.replace("token_rep_layer.projection.bias", "text.output_proj.bias")

    # ===== mDeBERTa Encoder (token_rep_layer.bert_layer.model.*) =====
    name = name.replace("token_rep_layer.bert_layer.model.embeddings.word_embeddings.weight", "text.token_embd.weight")
    name = name.replace("token_rep_layer.bert_layer.model.embeddings.LayerNorm.weight", "text.embd_norm.weight")
    name = name.replace("token_rep_layer.bert_layer.model.embeddings.LayerNorm.bias", "text.embd_norm.bias")

    # Relative position embeddings
    name = name.replace("token_rep_layer.bert_layer.model.encoder.rel_embeddings.weight", "text.rel_pos_embd.weight")
    name = name.replace("token_rep_layer.bert_layer.model.encoder.LayerNorm.weight", "text.output_norm.weight")
    name = name.replace("token_rep_layer.bert_layer.model.encoder.LayerNorm.bias", "text.output_norm.bias")

    # DeBERTa layers
    name = name.replace("token_rep_layer.bert_layer.model.encoder.layer.", "text.blk.")

    # DeBERTa attention
    name = name.replace(".attention.self.query_proj.", ".attention.query.")
    name = name.replace(".attention.self.key_proj.", ".attention.key.")
    name = name.replace(".attention.self.value_proj.", ".attention.value.")
    name = name.replace(".attention.self.pos_proj.", ".attention.pos_proj.")
    name = name.replace(".attention.self.pos_q_proj.", ".attention.pos_q_proj.")
    name = name.replace(".attention.output.dense.", ".attention.output.")
    name = name.replace(".attention.output.LayerNorm.", ".attention_norm.")

    # DeBERTa FFN
    name = name.replace(".intermediate.dense.", ".ffn.intermediate.")
    name = name.replace(".output.dense.", ".ffn.output.")
    name = name.replace(".output.LayerNorm.", ".output_norm.")

    # ===== Span Representation Layer =====
    name = name.replace("span_rep_layer.span_rep_layer.project_start.0.", "span.start_fc1.")
    name = name.replace("span_rep_layer.span_rep_layer.project_start.3.", "span.start_fc2.")
    name = name.replace("span_rep_layer.span_rep_layer.project_end.0.", "span.end_fc1.")
    name = name.replace("span_rep_layer.span_rep_layer.project_end.3.", "span.end_fc2.")
    name = name.replace("span_rep_layer.span_rep_layer.out_project.0.", "span.out_fc1.")
    name = name.replace("span_rep_layer.span_rep_layer.out_project.3.", "span.out_fc2.")

    # ===== BiLSTM (rnn.lstm.*) =====
    name = name.replace("rnn.lstm.", "rnn.")

    # ===== Scorer (scorer.*) =====
    # Keep scorer.* as-is: scorer.proj_token, scorer.proj_label, scorer.out_mlp

    # ===== Pair Representation Layer (pair_rep_layer.*) =====
    name = name.replace("pair_rep_layer.0.", "pair_proj.0.")
    name = name.replace("pair_rep_layer.3.", "pair_proj.3.")

    # ===== Prompt Representation Layer (prompt_rep_layer.*) =====
    # Keep prompt_rep_layer.* as-is

    return name


def _llama_quantize(f32_path: str, output_path: str, quant_type: str) -> None:
    """Run `llama-quantize` to convert an f32 GGUF to a k-quant type."""
    binary = shutil.which("llama-quantize")
    if binary is None:
        raise RuntimeError(
            "`llama-quantize` not found in PATH. Install llama.cpp to use k-quant types:\n"
            "    brew install llama.cpp        # macOS\n"
            "    or build from https://github.com/ggml-org/llama.cpp"
        )
    cmd = [binary]
    # Keep `scorer.out_mlp.3.weight` at F32 - fusor's quantised matmul produces
    # NaN on all-but-first rows when the output dim is tiny (shape is [3, 3072]).
    cmd += ["--tensor-type", "scorer.out_mlp.3.weight=f32"]
    cmd += [f32_path, output_path, quant_type.upper()]
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            f"llama-quantize failed (exit {result.returncode}) for {quant_type!r}. "
            f"See output above. The f32 GGUF was kept at {f32_path} for inspection."
        )


def convert_relex_to_gguf(
    model_id: str,
    output_path: str,
    quantize: str = "f32",
    cache_dir: str = None
):
    """Convert GLiNER-RelEx model to GGUF format."""

    state_dict, config, gliner_config_json, tokenizer_json = load_pytorch_model(model_id, cache_dir)

    print("\nModel weights:")
    for name, tensor in state_dict.items():
        print(f"  {name}: {tensor.shape} {tensor.dtype}")

    quantize = quantize.lower()
    if quantize not in QUANT_TYPES:
        raise ValueError(
            f"Unsupported quantization: {quantize!r}. Supported: {', '.join(sorted(QUANT_TYPES))}"
        )
    post_quant = None
    if quantize in LLAMA_QUANT_TYPES:
        post_quant = quantize
        quantize = "f32"
    default_ggml_type = _ggml_type_for(quantize)

    # For k-quant targets, write f32 first to a temp path then shell out.
    final_output_path = output_path
    if post_quant is not None:
        tmp = tempfile.NamedTemporaryFile(
            prefix=os.path.basename(output_path).rsplit(".", 1)[0] + ".f32.",
            suffix=".gguf",
            delete=False,
            dir=os.path.dirname(os.path.abspath(output_path)) or None,
        )
        output_path = tmp.name
        tmp.close()

    writer = GGUFWriter(output_path)

    # Masquerade as `bert` when we'll feed this to llama-quantize so its loader
    # accepts the custom arch. rgliner doesn't read general.architecture.
    arch = "bert" if post_quant is not None else "gliner-relex"
    writer.add_metadata("general.architecture", arch)
    writer.add_metadata("general.name", model_id.split("/")[-1])
    writer.add_metadata("general.quantization_version", 2)

    # GLiNER-RelEx specific metadata
    writer.add_metadata("gliner.max_width", config.get("max_width", 12))
    writer.add_metadata("gliner.span_mode", config.get("span_mode", "markerV0"))
    writer.add_metadata("gliner.subtoken_pooling", config.get("subtoken_pooling", "first"))

    # mDeBERTa config
    encoder_config = config.get("encoder_config", {})
    hidden_size = encoder_config.get("hidden_size", 768)
    num_heads = encoder_config.get("num_attention_heads", 12)
    num_layers = encoder_config.get("num_hidden_layers", 12)
    intermediate_size = encoder_config.get("intermediate_size", 3072)
    vocab_size = encoder_config.get("vocab_size", 250105)
    context_length = encoder_config.get("max_position_embeddings", 512)
    max_relative_positions = encoder_config.get("max_relative_positions", 512)

    # Handle -1 which means "use full context"
    if max_relative_positions <= 0:
        # Derive from rel_embeddings shape if available, otherwise use context_length // 2
        rel_emb_key = "token_rep_layer.bert_layer.model.encoder.rel_embeddings.weight"
        if rel_emb_key in state_dict:
            num_positions = state_dict[rel_emb_key].shape[0]
            max_relative_positions = num_positions // 2
            print(f"Derived max_relative_positions from rel_embeddings: {max_relative_positions}")
        else:
            max_relative_positions = context_length // 2
            print(f"Using default max_relative_positions: {max_relative_positions}")

    writer.add_metadata("gliner.attention.head_count", num_heads)
    writer.add_metadata("gliner.block_count", num_layers)
    writer.add_metadata("gliner.embedding_length", hidden_size)
    writer.add_metadata("gliner.feed_forward_length", intermediate_size)
    writer.add_metadata("gliner.context_length", context_length)
    writer.add_metadata("gliner.attention.max_relative_positions", max_relative_positions)
    writer.add_metadata("gliner.attention.layer_norm_epsilon", 1e-7)
    writer.add_metadata("gliner.vocab_size", vocab_size)

    # RelEx-specific metadata
    writer.add_metadata("gliner.relex.ent_token_id", 250102)  # <<ENT>>
    writer.add_metadata("gliner.relex.rel_token_id", 250104)  # <<REL>>

    # Embed tokenizer.json and gliner_config.json as string metadata so the
    # GGUF is self-contained (no separate files needed at inference time).
    writer.add_metadata("gliner.tokenizer_json", tokenizer_json)
    writer.add_metadata("gliner.config_json", gliner_config_json)

    # When we're masquerading as `bert` for llama-quantize, mirror the required
    # arch-scoped keys so its loader validates. llama.cpp's BERT loader expects
    # u32 (not u64) for these fields; rgliner never reads them.
    if post_quant is not None:
        writer.add_metadata("bert.context_length", _U32(context_length))
        writer.add_metadata("bert.embedding_length", _U32(hidden_size))
        writer.add_metadata("bert.feed_forward_length", _U32(intermediate_size))
        writer.add_metadata("bert.block_count", _U32(num_layers))
        writer.add_metadata("bert.attention.head_count", _U32(num_heads))
        writer.add_metadata("bert.attention.layer_norm_epsilon", 1e-7)

    # Convert tensors
    print(f"\nConverting {len(state_dict)} tensors to GGUF ({quantize.upper()})...")

    block_quant = quantize.startswith("q")
    # Allow debugging by overriding via env var:
    #   GLINER_QUANT_INCLUDE="text.blk.0.ffn" -> only quantise tensors containing this substring
    #   GLINER_QUANT_EXCLUDE="token_embd" -> never quantise tensors containing this substring
    quant_include = os.environ.get("GLINER_QUANT_INCLUDE")
    quant_exclude = os.environ.get("GLINER_QUANT_EXCLUDE")

    for pytorch_name, tensor in state_dict.items():
        gguf_name = map_relex_weight_name(pytorch_name)

        try:
            data = tensor.detach().float().cpu().numpy()
        except RuntimeError:
            import array
            t = tensor.detach().float().cpu().contiguous()
            data = np.frombuffer(
                array.array('f', t.flatten().tolist()),
                dtype=np.float32
            ).reshape(t.shape)

        logical_shape = tuple(data.shape)
        tensor_type = default_ggml_type

        should_quant = block_quant
        if should_quant and quant_include is not None and quant_include not in gguf_name:
            should_quant = False
        if should_quant and quant_exclude is not None and quant_exclude in gguf_name:
            should_quant = False

        if should_quant:
            # Row-wise block quantisation requires the inner dim to be a
            # multiple of 32. 1-D tensors (biases, norms) and non-conforming
            # tensors (odd vocab sizes etc.) stay at F32.
            #
            # Additional constraint: fusor's quantised matmul kernel produces
            # NaN for all but the first output row when the output dimension
            # is very small (the classifier head `scorer.out_mlp.3.weight` is
            # [3, 3072]). Keep these tiny-output tensors at F32 - they're
            # negligible bytes anyway.
            MIN_OUT_DIM_FOR_QUANT = 32
            if (
                data.ndim >= 2
                and data.shape[-1] % 32 == 0
                and data.shape[0] >= MIN_OUT_DIM_FOR_QUANT
            ):
                data = _gguf_block_quant(data, quantize)
                tensor_type = default_ggml_type
            else:
                data = np.ascontiguousarray(data, dtype=np.float32)
                tensor_type = GGML_TYPE_F32
        else:
            data = np.ascontiguousarray(data, dtype=np.float32)
            tensor_type = GGML_TYPE_F32

        print(f"  {pytorch_name} -> {gguf_name} {logical_shape}  [{_name_for_type(tensor_type)}]")
        writer.add_tensor(gguf_name, data, tensor_type, shape=logical_shape)

    writer.write()
    print(f"\nOutput: {output_path}")
    print(f"Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    if post_quant is not None:
        print(f"\nQuantizing to {post_quant.upper()} via llama-quantize...")
        try:
            _llama_quantize(output_path, final_output_path, post_quant)
        finally:
            try:
                os.remove(output_path)
            except OSError:
                pass
        print(f"\nFinal output: {final_output_path}")
        print(f"Final size: {os.path.getsize(final_output_path) / 1024 / 1024:.2f} MB")

    print("\nConversion complete!")


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


def main():
    parser = argparse.ArgumentParser(description="Convert GLiNER-RelEx PyTorch models to GGUF")
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g., knowledgator/gliner-relex-multi-v1.0)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output GGUF file path"
    )
    parser.add_argument(
        "--quantize", "-q",
        type=str,
        default="f32",
        choices=sorted(QUANT_TYPES),
        help=(
            "Quantization type (default: f32). "
            "All quantisation is done in-process via the `gguf` Python package. "
            "k-quants (q4_k, q5_k, q6_k) aren't in this list - they are not "
            "round-trip compatible with fusor's current quantised tensor reader."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory"
    )

    args = parser.parse_args()

    convert_relex_to_gguf(
        model_id=args.model,
        output_path=args.output,
        quantize=args.quantize,
        cache_dir=args.cache_dir
    )


if __name__ == "__main__":
    main()
