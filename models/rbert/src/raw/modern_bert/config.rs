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
    /// Hidden size (embedding dimension).
    pub hidden_size: usize,
    /// Dimension per attention head.
    pub head_dimension: usize,
    /// Intermediate size for FFN.
    pub intermediate_size: usize,
    /// Maximum context length.
    pub context_length: usize,
    /// RoPE base frequency.
    pub rope_theta: f32,
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

        if hidden_size % num_heads != 0 {
            return Err(fusor::Error::msg(format!(
                "hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )));
        }

        let intermediate_size =
            load_u32_or(vb, ".feed_forward_length", (hidden_size * 4) as u32) as usize;
        let context_length = load_u32_or(vb, ".context_length", 8192) as usize;
        let rope_theta = load_f32_or(vb, ".rope.freq_base", 10000.0);
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
            hidden_size,
            head_dimension,
            intermediate_size,
            context_length,
            rope_theta,
            norm_eps,
        })
    }
}
