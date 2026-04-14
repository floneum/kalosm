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
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
GGML_TYPE_Q8_0 = 8
GGML_TYPE_BF16 = 30


class GGUFWriter:
    """Simple GGUF file writer."""

    def __init__(self, path: str):
        self.path = path
        self.metadata: Dict[str, Any] = {}
        self.tensors: List[Tuple[str, np.ndarray, int]] = []

    def add_metadata(self, key: str, value: Any):
        self.metadata[key] = value

    def add_tensor(self, name: str, data: np.ndarray, ggml_type: int = GGML_TYPE_F32):
        self.tensors.append((name, data, ggml_type))

    def _write_string(self, f, s: str):
        encoded = s.encode('utf-8')
        f.write(struct.pack('<Q', len(encoded)))
        f.write(encoded)

    def _write_metadata_value(self, f, value: Any):
        if isinstance(value, bool):
            f.write(struct.pack('<I', GGUF_TYPE_BOOL))
            f.write(struct.pack('<B', 1 if value else 0))
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

            for name, data, ggml_type in self.tensors:
                if ggml_type == GGML_TYPE_F32:
                    data = np.ascontiguousarray(data, dtype=np.float32)
                elif ggml_type == GGML_TYPE_F16:
                    data = np.ascontiguousarray(data, dtype=np.float16)
                elif ggml_type == GGML_TYPE_BF16:
                    data = np.ascontiguousarray(data, dtype=np.float32)
                    data = data.view(np.uint32)
                    data = ((data >> 16) & 0xFFFF).astype(np.uint16)

                self._write_string(f, name)
                f.write(struct.pack('<I', len(data.shape)))
                for dim in reversed(data.shape):
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

            for (data, _), (name, _, ggml_type) in zip(tensor_infos, self.tensors):
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

    if quantize == "f32":
        ggml_type = GGML_TYPE_F32
    elif quantize == "f16":
        ggml_type = GGML_TYPE_F16
    elif quantize == "bf16":
        ggml_type = GGML_TYPE_BF16
    else:
        raise ValueError(f"Unsupported quantization: {quantize}")

    writer = GGUFWriter(output_path)

    # Add metadata
    writer.add_metadata("general.architecture", "gliner-relex")
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

    # Convert tensors
    print(f"\nConverting {len(state_dict)} tensors to GGUF...")

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

        print(f"  {pytorch_name} -> {gguf_name} {data.shape}")
        writer.add_tensor(gguf_name, data, ggml_type)

    writer.write()
    print(f"\nOutput: {output_path}")
    print(f"Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    print("\nConversion complete!")


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
        choices=["f32", "f16", "bf16"],
        help="Quantization type (default: f32)"
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
