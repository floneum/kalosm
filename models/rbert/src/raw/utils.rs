//! Shared helpers used by the raw encoder implementations.

use fusor::{Result, Tensor, VarBuilder};

/// Reshape `[batch, seq_len, num_heads * head_dim]` into
/// `[batch, num_heads, seq_len, head_dim]` for multi-head attention.
pub(crate) fn split_heads(
    tensor: &Tensor<3, f32>,
    num_heads: usize,
    head_dim: usize,
) -> Tensor<4, f32> {
    let [batch, seq_len, _] = tensor.shape();
    tensor
        .reshape([batch, seq_len, num_heads, head_dim])
        .transpose(1, 2)
        .to_concrete()
}

/// Inverse of [`split_heads`]: collapse
/// `[batch, num_heads, seq_len, head_dim]` back to
/// `[batch, seq_len, num_heads * head_dim]`.
pub(crate) fn merge_heads(tensor: &Tensor<4, f32>) -> Tensor<3, f32> {
    let [batch, num_heads, seq_len, head_dim] = tensor.shape();
    tensor
        .transpose(1, 2)
        .to_concrete()
        .reshape([batch, seq_len, num_heads * head_dim])
        .to_concrete()
}

/// Large negative bias applied to masked-out positions so that, after
/// softmax, their probability collapses to ~0.
pub(crate) const MASK_NEG_VALUE: f32 = -10000.0;

/// Convert a `[batch, seq]` boolean attention mask (1 = attend, 0 = pad)
/// into an additive bias tensor of the same shape (0 for real tokens,
/// [`MASK_NEG_VALUE`] for padding).
pub(crate) fn attention_mask_to_bias(mask: &Tensor<2, u32>) -> Tensor<2, f32> {
    let mask_f32: Tensor<2, f32> = mask.cast();
    let zeros = mask_f32.zeros_like();
    let ones = (zeros + 1.0f32).to_concrete();
    ((ones - mask_f32) * MASK_NEG_VALUE).to_concrete()
}

/// Read a required `u32` GGUF metadata value. The `.`-prefix on keys is
/// interpreted by fusor as a suffix match, so callers pass architecture-agnostic
/// keys like `.attention.head_count`.
pub(crate) fn load_u32(vb: &VarBuilder, key: &str) -> Result<u32> {
    vb.get_metadata(key)
        .and_then(|v| v.to_u32().ok())
        .ok_or_else(|| fusor::Error::msg(format!("Missing required GGUF metadata: {key}")))
}

/// Read an optional `u32` GGUF metadata value, falling back to `default`.
pub(crate) fn load_u32_or(vb: &VarBuilder, key: &str, default: u32) -> u32 {
    vb.get_metadata(key)
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(default)
}

/// Read an optional `f32` GGUF metadata value, falling back to `default`.
pub(crate) fn load_f32_or(vb: &VarBuilder, key: &str, default: f32) -> f32 {
    vb.get_metadata(key)
        .and_then(|v| v.to_f32().ok())
        .unwrap_or(default)
}

/// Read an optional `bool` GGUF metadata value, falling back to `default`.
pub(crate) fn load_bool_or(vb: &VarBuilder, key: &str, default: bool) -> bool {
    vb.get_metadata(key)
        .and_then(|v| v.to_bool().ok())
        .unwrap_or(default)
}
