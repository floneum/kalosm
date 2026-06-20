//! ModernBERT configuration from GGUF metadata.

use fusor::{Result, VarBuilder};

use super::super::utils::{load_f32_or, load_u32, load_u32_or};

/// Configuration for ModernBERT loaded from GGUF metadata.
#[derive(Debug, Clone)]
pub struct ModernBertConfig {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of key-value heads (for GQA).
    pub num_kv_heads: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Dimension per attention head.
    pub head_dimension: usize,
    /// Maximum context length.
    pub context_length: usize,
    /// RoPE base frequency used by global-attention layers.
    pub rope_theta: f32,
    /// RoPE base frequency used by local (sliding-window) layers. Equal to
    /// `rope_theta` when the model uses a single frequency.
    pub local_rope_theta: f32,
    /// Every Nth layer (`layer_idx % N == 0`) uses full global attention; the
    /// rest use sliding-window local attention. `1` means every layer is global
    /// (the default when the GGUF predates this metadata), which reproduces the
    /// original all-global behaviour.
    pub global_attn_every_n_layers: usize,
    /// Sliding-window size for local-attention layers (ModernBERT `local_attention`,
    /// e.g. 128). A query attends to keys within `local_attention / 2` positions.
    /// `0` disables windowing entirely.
    pub local_attention: usize,
    /// LayerNorm epsilon.
    pub norm_eps: f32,
}

impl ModernBertConfig {
    /// Load configuration from GGUF metadata.
    pub fn from_gguf(vb: &VarBuilder) -> Result<Self> {
        let num_heads = load_u32(vb, ".attention.head_count")? as usize;
        let num_kv_heads = load_u32_or(vb, ".attention.head_count_kv", num_heads as u32) as usize;
        let num_layers = load_u32(vb, ".block_count")? as usize;
        let hidden_size = load_u32(vb, ".embedding_length")? as usize;

        if !hidden_size.is_multiple_of(num_heads) {
            return Err(fusor::Error::msg(format!(
                "hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )));
        }

        let context_length = load_u32_or(vb, ".context_length", 8192) as usize;
        let rope_theta = load_f32_or(vb, ".rope.freq_base", 10000.0);
        // Local layers may use a distinct RoPE base; absent (older GGUFs) it
        // falls back to the global base so behaviour is unchanged.
        let local_rope_theta = load_f32_or(vb, ".rope.local_freq_base", rope_theta);
        // Default 1 (every layer global) preserves the original behaviour for
        // GGUFs converted before this metadata existed.
        let global_attn_every_n_layers =
            load_u32_or(vb, ".attention.global_attn_every_n_layers", 1).max(1) as usize;
        let local_attention = load_u32_or(vb, ".attention.local_attention", 0) as usize;
        let norm_eps = load_f32_or(vb, ".attention.layer_norm_rms_epsilon", 1e-6);

        // Use attention.key_length for head dimension; fall back to
        // hidden_size / num_heads if not present.
        let head_dimension = load_u32_or(vb, ".attention.key_length", 0);
        let head_dimension = if head_dimension == 0 {
            hidden_size / num_heads
        } else {
            head_dimension as usize
        };

        Ok(Self {
            num_heads,
            num_kv_heads,
            num_layers,
            head_dimension,
            context_length,
            rope_theta,
            local_rope_theta,
            global_attn_every_n_layers,
            local_attention,
            norm_eps,
        })
    }
}
