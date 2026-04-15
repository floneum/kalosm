#!/usr/bin/env python3
"""
Convert GLiNER PyTorch models to GGUF format.

Usage:
    python convert_to_gguf.py --model knowledgator/gliner-bi-edge-v2.0 --output gliner-bi-edge-v2.0.gguf

This script converts both:
1. The main text encoder (ModernBERT/Ettin) + span layer weights
2. The label encoder projection weights (sentence transformer is loaded separately)
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

import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download


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
    """Return the packed byte array for a block-quantised tensor."""
    import gguf
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


def load_pytorch_model(model_id: str, cache_dir: str = None) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """Load PyTorch model from HuggingFace."""
    print(f"Downloading model: {model_id}")

    # Download the model files
    model_dir = snapshot_download(
        model_id,
        cache_dir=cache_dir,
        allow_patterns=["*.bin", "*.json", "*.safetensors"]
    )

    # Load config
    config_path = os.path.join(model_dir, "gliner_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Load weights
    weights_path = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.exists(weights_path):
        print(f"Loading weights from: {weights_path}")
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
    else:
        # Try safetensors
        from safetensors.torch import load_file
        weights_path = os.path.join(model_dir, "model.safetensors")
        print(f"Loading weights from: {weights_path}")
        state_dict = load_file(weights_path)

    return state_dict, config


def map_weight_name(pytorch_name: str) -> str:
    """Map PyTorch weight names to GGUF conventions."""
    name = pytorch_name

    # ===== Encoder output projection =====
    # Some bi-encoder variants (small/base/large v2.0) project the text-encoder
    # hidden state down to the shared label-aligned dim (e.g. 512 -> 384).
    name = name.replace("token_rep_layer.projection.weight", "text.output_proj.weight")
    name = name.replace("token_rep_layer.projection.bias", "text.output_proj.bias")

    # ===== Text Encoder (ModernBERT/Ettin) =====
    # token_rep_layer.bert_layer.model.embeddings.tok_embeddings.weight -> text.token_embd.weight
    name = name.replace("token_rep_layer.bert_layer.model.embeddings.tok_embeddings.weight", "text.token_embd.weight")
    name = name.replace("token_rep_layer.bert_layer.model.embeddings.norm.weight", "text.embd_norm.weight")

    # token_rep_layer.bert_layer.model.layers.X -> text.blk.X
    name = name.replace("token_rep_layer.bert_layer.model.layers.", "text.blk.")
    name = name.replace("token_rep_layer.bert_layer.model.final_norm.weight", "text.output_norm.weight")

    # ModernBERT attention (fused Wqkv)
    name = name.replace(".attn.Wqkv.", ".attn_qkv.")
    name = name.replace(".attn.Wo.", ".attn_output.")

    # ModernBERT FFN (GeGLU with fused Wi)
    name = name.replace(".mlp.Wi.", ".ffn_gate_up.")  # Fused gate+up
    name = name.replace(".mlp.Wo.", ".ffn_down.")
    name = name.replace(".mlp_norm.", ".ffn_norm.")

    # ===== Label Encoder (BERT/MiniLM) =====
    name = name.replace("token_rep_layer.labels_encoder.model.", "label.")

    # BERT embeddings (rbert expects token_types, token_embd_norm)
    name = name.replace("label.embeddings.word_embeddings.", "label.token_embd.")
    name = name.replace("label.embeddings.position_embeddings.", "label.position_embd.")
    name = name.replace("label.embeddings.token_type_embeddings.", "label.token_types.")
    name = name.replace("label.embeddings.LayerNorm.", "label.token_embd_norm.")

    # BERT layers
    name = name.replace("label.encoder.layer.", "label.blk.")

    # BERT attention (rbert uses attn_output_norm, not attn_norm)
    name = name.replace(".attention.self.query.", ".attn_q.")
    name = name.replace(".attention.self.key.", ".attn_k.")
    name = name.replace(".attention.self.value.", ".attn_v.")
    name = name.replace(".attention.output.dense.", ".attn_output.")
    name = name.replace(".attention.output.LayerNorm.", ".attn_output_norm.")

    # BERT FFN (rbert uses layer_output_norm, not ffn_norm)
    name = name.replace(".intermediate.dense.", ".ffn_up.")
    name = name.replace(".output.dense.", ".ffn_down.")
    name = name.replace(".output.LayerNorm.", ".layer_output_norm.")

    # BERT pooler
    name = name.replace("label.pooler.dense.", "label.pooler.")

    # ===== BiLSTM =====
    # Keep rnn.lstm.* as-is for now
    name = name.replace("rnn.lstm.", "rnn.")

    # ===== Span Representation Layer =====
    name = name.replace("span_rep_layer.span_rep_layer.project_start.0.", "span.start_fc1.")
    name = name.replace("span_rep_layer.span_rep_layer.project_start.3.", "span.start_fc2.")
    name = name.replace("span_rep_layer.span_rep_layer.project_end.0.", "span.end_fc1.")
    name = name.replace("span_rep_layer.span_rep_layer.project_end.3.", "span.end_fc2.")
    name = name.replace("span_rep_layer.span_rep_layer.out_project.0.", "span.out_fc1.")
    name = name.replace("span_rep_layer.span_rep_layer.out_project.3.", "span.out_fc2.")

    # ===== Prompt/Label Projection =====
    name = name.replace("prompt_rep_layer.0.", "label_proj.0.")
    name = name.replace("prompt_rep_layer.3.", "label_proj.2.")

    return name


def _llama_quantize(f32_path: str, output_path: str, quant_type: str, keep_f32: List[str]) -> None:
    """Run `llama-quantize` to convert an f32 GGUF to a k-quant type.

    `keep_f32` is a list of tensor names that should remain at F32 (tiny
    classifier heads hit a fusor bug; see the gliner-relex project notes).
    """
    binary = shutil.which("llama-quantize")
    if binary is None:
        raise RuntimeError(
            "`llama-quantize` not found in PATH. Install llama.cpp to use k-quant types:\n"
            "    brew install llama.cpp        # macOS\n"
            "    or build from https://github.com/ggml-org/llama.cpp"
        )
    cmd = [binary]
    for name in keep_f32:
        cmd += ["--tensor-type", f"{name}=f32"]
    cmd += [f32_path, output_path, quant_type.upper()]
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            f"llama-quantize failed (exit {result.returncode}) for {quant_type!r}. "
            f"See output above. The f32 GGUF was kept at {f32_path} for inspection."
        )


def _quantize_tensor(data: np.ndarray, quant: str, default_ggml_type: int) -> Tuple[np.ndarray, int, Tuple[int, ...]]:
    """Quantise a single tensor, falling back to F32 for unsupported shapes.

    Returns (packed_data, ggml_type, logical_shape). Applies the same rules as
    the relex converter: only 2-D+ tensors with inner dim divisible by 32 and
    outer dim >= 32 are block-quantised. The lower outer-dim threshold avoids
    a fusor quantised-matmul bug on tiny classifier heads.
    """
    logical_shape = tuple(data.shape)
    MIN_OUT_DIM_FOR_QUANT = 32
    if (
        quant.startswith("q")
        and data.ndim >= 2
        and data.shape[-1] % 32 == 0
        and data.shape[0] >= MIN_OUT_DIM_FOR_QUANT
    ):
        return _gguf_block_quant(data, quant), default_ggml_type, logical_shape
    return np.ascontiguousarray(data, dtype=np.float32), GGML_TYPE_F32, logical_shape


def convert_gliner_to_gguf(
    model_id: str,
    output_path: str,
    quantize: str = "f32",
    cache_dir: str = None
):
    """Convert GLiNER model to GGUF format.

    Creates two GGUF files:
    - {output_path}: Main model (text encoder, span layer, projection)
    - {output_path_stem}-label-encoder.gguf: Label encoder (BERT/MiniLM)
    """

    # Load model
    state_dict, config = load_pytorch_model(model_id, cache_dir)

    # Print model structure
    print("\nModel weights:")
    for name, tensor in state_dict.items():
        print(f"  {name}: {tensor.shape} {tensor.dtype}")

    # Determine quantization type and post-processing
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

    # When post-quantising, write the raw f32 file to a temp path first.
    main_label_output = output_path.replace(".gguf", "-label-encoder.gguf")
    final_main = output_path
    final_label = main_label_output
    if post_quant is not None:
        main_tmp = tempfile.NamedTemporaryFile(
            prefix=os.path.basename(output_path).rsplit(".", 1)[0] + ".f32.",
            suffix=".gguf",
            delete=False,
            dir=os.path.dirname(os.path.abspath(output_path)) or None,
        )
        label_tmp = tempfile.NamedTemporaryFile(
            prefix=os.path.basename(main_label_output).rsplit(".", 1)[0] + ".f32.",
            suffix=".gguf",
            delete=False,
            dir=os.path.dirname(os.path.abspath(main_label_output)) or None,
        )
        output_path = main_tmp.name
        main_label_output = label_tmp.name
        main_tmp.close()
        label_tmp.close()

    # Separate label encoder weights from main model weights
    label_encoder_weights = {}
    main_model_weights = {}

    for pytorch_name, tensor in state_dict.items():
        if "token_rep_layer.labels_encoder" in pytorch_name:
            label_encoder_weights[pytorch_name] = tensor
        else:
            main_model_weights[pytorch_name] = tensor

    # ============ Main Model GGUF ============
    writer = GGUFWriter(output_path)

    # Add metadata. Masquerade as `bert` when we'll feed this to llama-quantize
    # so its loader accepts the custom architecture. rgliner reads `gliner.*`
    # keys, not `general.architecture`.
    main_arch = "bert" if post_quant is not None else "gliner"
    writer.add_metadata("general.architecture", main_arch)
    writer.add_metadata("general.name", model_id.split("/")[-1])
    writer.add_metadata("general.quantization_version", 2)

    # GLiNER-specific metadata
    writer.add_metadata("gliner.max_width", config.get("max_width", 12))
    writer.add_metadata("gliner.span_mode", config.get("span_mode", "markerV0"))
    writer.add_metadata("gliner.subtoken_pooling", config.get("subtoken_pooling", "first"))

    # Encoder config - use standard GGUF naming for model loading
    encoder_config = config.get("encoder_config", {})
    hidden_size = encoder_config.get("hidden_size", 384)
    num_heads = encoder_config.get("num_attention_heads", 6)
    num_layers = encoder_config.get("num_hidden_layers", 10)
    intermediate_size = encoder_config.get("intermediate_size", 576)
    vocab_size = encoder_config.get("vocab_size", 50368)
    context_length = encoder_config.get("max_position_embeddings", 8192)
    rope_theta = encoder_config.get("local_rope_theta", 160000.0)

    # Standard GGUF metadata (without architecture prefix - the loader adds it)
    writer.add_metadata("gliner.attention.head_count", num_heads)
    writer.add_metadata("gliner.attention.head_count_kv", num_heads)  # No GQA in this model
    writer.add_metadata("gliner.block_count", num_layers)
    writer.add_metadata("gliner.embedding_length", hidden_size)
    writer.add_metadata("gliner.feed_forward_length", intermediate_size)
    writer.add_metadata("gliner.context_length", context_length)
    writer.add_metadata("gliner.rope.freq_base", float(rope_theta))
    writer.add_metadata("gliner.attention.layer_norm_rms_epsilon", 1e-5)
    writer.add_metadata("gliner.vocab_size", vocab_size)

    # When feeding to llama-quantize, mirror arch keys with u32 scalars.
    if post_quant is not None:
        writer.add_metadata("bert.context_length", _U32(context_length))
        writer.add_metadata("bert.embedding_length", _U32(hidden_size))
        writer.add_metadata("bert.feed_forward_length", _U32(intermediate_size))
        writer.add_metadata("bert.block_count", _U32(num_layers))
        writer.add_metadata("bert.attention.head_count", _U32(num_heads))
        writer.add_metadata("bert.attention.layer_norm_epsilon", 1e-5)

    # Convert main model tensors
    print(f"\nConverting {len(main_model_weights)} main model tensors to GGUF ({quantize.upper()})...")

    for pytorch_name, tensor in main_model_weights.items():
        gguf_name = map_weight_name(pytorch_name)

        # Convert to numpy (workaround for numpy/torch incompatibility)
        try:
            data = tensor.detach().float().cpu().numpy()
        except RuntimeError:
            import array
            t = tensor.detach().float().cpu().contiguous()
            data = np.frombuffer(
                array.array('f', t.flatten().tolist()),
                dtype=np.float32
            ).reshape(t.shape)

        packed, tensor_type, logical_shape = _quantize_tensor(data, quantize, default_ggml_type)
        print(f"  {pytorch_name} -> {gguf_name} {logical_shape}  [{_name_for_type(tensor_type)}]")
        writer.add_tensor(gguf_name, packed, tensor_type, shape=logical_shape)

    writer.write()
    print(f"Main model output: {output_path}")
    print(f"Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    # ============ Label Encoder GGUF ============
    # Create separate file for label encoder (without prefix, for rbert compatibility)
    label_output_path = main_label_output
    label_writer = GGUFWriter(label_output_path)

    # Add BERT metadata
    labels_config = config.get("labels_encoder_config", {})
    label_writer.add_metadata("general.architecture", "bert")
    label_writer.add_metadata("general.name", model_id.split("/")[-1] + "-label-encoder")

    label_hidden = labels_config.get("hidden_size", 384)
    label_heads = labels_config.get("num_attention_heads", 12)
    label_layers = labels_config.get("num_hidden_layers", 6)
    label_intermediate = labels_config.get("intermediate_size", 1536)
    label_vocab = labels_config.get("vocab_size", 30522)
    label_max_pos = labels_config.get("max_position_embeddings", 512)

    # Use u32 for arch-scoped ints when we'll feed this through llama-quantize.
    int_ctor = _U32 if post_quant is not None else (lambda x: x)
    label_writer.add_metadata("bert.attention.head_count", int_ctor(label_heads))
    label_writer.add_metadata("bert.block_count", int_ctor(label_layers))
    label_writer.add_metadata("bert.embedding_length", int_ctor(label_hidden))
    label_writer.add_metadata("bert.feed_forward_length", int_ctor(label_intermediate))
    label_writer.add_metadata("bert.context_length", int_ctor(label_max_pos))
    label_writer.add_metadata("bert.attention.layer_norm_epsilon", 1e-12)
    label_writer.add_metadata("bert.vocab_size", int_ctor(label_vocab))

    # Convert label encoder tensors (remove prefix so rbert can load them)
    print(f"\nConverting {len(label_encoder_weights)} label encoder tensors to GGUF ({quantize.upper()})...")

    for pytorch_name, tensor in label_encoder_weights.items():
        # Map name but remove the "label." prefix for rbert compatibility
        gguf_name = map_weight_name(pytorch_name)
        if gguf_name.startswith("label."):
            gguf_name = gguf_name[6:]  # Remove "label." prefix

        try:
            data = tensor.detach().float().cpu().numpy()
        except RuntimeError:
            import array
            t = tensor.detach().float().cpu().contiguous()
            data = np.frombuffer(
                array.array('f', t.flatten().tolist()),
                dtype=np.float32
            ).reshape(t.shape)

        packed, tensor_type, logical_shape = _quantize_tensor(data, quantize, default_ggml_type)
        print(f"  {pytorch_name} -> {gguf_name} {logical_shape}  [{_name_for_type(tensor_type)}]")
        label_writer.add_tensor(gguf_name, packed, tensor_type, shape=logical_shape)

    label_writer.write()
    print(f"Label encoder output: {label_output_path}")
    print(f"Size: {os.path.getsize(label_output_path) / 1024 / 1024:.2f} MB")

    if post_quant is not None:
        print(f"\nQuantizing main model to {post_quant.upper()} via llama-quantize...")
        _llama_quantize(output_path, final_main, post_quant, keep_f32=[])
        try:
            os.remove(output_path)
        except OSError:
            pass
        print(f"Final main output: {final_main}")
        print(f"Final main size:   {os.path.getsize(final_main) / 1024 / 1024:.2f} MB")

        print(f"\nQuantizing label encoder to {post_quant.upper()} via llama-quantize...")
        _llama_quantize(label_output_path, final_label, post_quant, keep_f32=[])
        try:
            os.remove(label_output_path)
        except OSError:
            pass
        print(f"Final label output: {final_label}")
        print(f"Final label size:   {os.path.getsize(final_label) / 1024 / 1024:.2f} MB")

    print(f"\nConversion complete!")


def main():
    parser = argparse.ArgumentParser(description="Convert GLiNER PyTorch models to GGUF")
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="HuggingFace model ID (e.g., knowledgator/gliner-bi-edge-v2.0)"
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
            "Block quants (q4_0, q5_0, q8_0, ...) are packed in-process via the "
            "`gguf` Python package. K-quants (q4_k, q5_k, q6_k, ...) require "
            "`llama-quantize` in PATH and are applied as a post-processing step."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory"
    )

    args = parser.parse_args()

    convert_gliner_to_gguf(
        model_id=args.model,
        output_path=args.output,
        quantize=args.quantize,
        cache_dir=args.cache_dir
    )


if __name__ == "__main__":
    main()
