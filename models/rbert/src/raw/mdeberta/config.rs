//! mDeBERTa-v3 configuration from GGUF metadata.

use fusor::{Result, VarBuilder};

use super::super::utils::{load_bool_or, load_f32_or, load_u32, load_u32_or};

/// Configuration for mDeBERTa-v3 loaded from GGUF metadata.
#[derive(Debug, Clone)]
pub struct MDebertaConfig {
    /// Number of attention heads.
    pub num_heads: usize,
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
    /// Maximum relative position distance for attention.
    pub max_relative_positions: usize,
    /// LayerNorm epsilon.
    pub norm_eps: f32,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Position buckets for relative position encoding.
    pub position_buckets: usize,
    /// Whether to share attention weights across layers.
    pub share_att_key: bool,
}

impl MDebertaConfig {
    /// Load configuration from GGUF metadata.
    pub fn from_gguf(vb: &VarBuilder) -> Result<Self> {
        let num_heads = load_u32(vb, ".attention.head_count")? as usize;
        let num_layers = load_u32(vb, ".block_count")? as usize;
        let hidden_size = load_u32(vb, ".embedding_length")? as usize;

        if hidden_size % num_heads != 0 {
            return Err(fusor::Error::msg(format!(
                "hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )));
        }

        let head_dimension = load_u32_or(vb, ".attention.key_length", 0);
        let head_dimension = if head_dimension == 0 {
            hidden_size / num_heads
        } else {
            head_dimension as usize
        };

        let intermediate_size =
            load_u32_or(vb, ".feed_forward_length", (hidden_size * 4) as u32) as usize;
        let context_length = load_u32_or(vb, ".context_length", 512) as usize;
        let max_relative_positions =
            load_u32_or(vb, ".attention.max_relative_positions", 512) as usize;
        let norm_eps = load_f32_or(vb, ".attention.layer_norm_epsilon", 1e-7);
        let vocab_size = load_u32_or(vb, ".vocab_size", 250105) as usize;
        let position_buckets = load_u32_or(vb, ".attention.position_buckets", 256) as usize;
        let share_att_key = load_bool_or(vb, ".attention.share_att_key", true);

        Ok(Self {
            num_heads,
            num_layers,
            hidden_size,
            head_dimension,
            intermediate_size,
            context_length,
            max_relative_positions,
            norm_eps,
            vocab_size,
            position_buckets,
            share_att_key,
        })
    }

    /// Create a default config for mDeBERTa-v3-base.
    pub fn mdeberta_v3_base() -> Self {
        Self {
            num_heads: 12,
            num_layers: 12,
            hidden_size: 768,
            head_dimension: 64,
            intermediate_size: 3072,
            context_length: 512,
            max_relative_positions: 512,
            norm_eps: 1e-7,
            vocab_size: 250105,
            position_buckets: 256,
            share_att_key: true,
        }
    }
}
