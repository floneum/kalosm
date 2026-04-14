//! mDeBERTa-v3 configuration from GGUF metadata.

use fusor::{Result, VarBuilder};

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
        let num_heads = vb
            .get_metadata(".attention.head_count")
            .and_then(|v| v.to_u32().ok())
            .ok_or_else(|| {
                fusor::Error::msg("Missing required GGUF metadata: .attention.head_count")
            })? as usize;

        let num_layers = vb
            .get_metadata(".block_count")
            .and_then(|v| v.to_u32().ok())
            .ok_or_else(|| fusor::Error::msg("Missing required GGUF metadata: .block_count"))?
            as usize;

        let hidden_size = vb
            .get_metadata(".embedding_length")
            .and_then(|v| v.to_u32().ok())
            .ok_or_else(|| fusor::Error::msg("Missing required GGUF metadata: .embedding_length"))?
            as usize;

        if hidden_size % num_heads != 0 {
            return Err(fusor::Error::msg(format!(
                "hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )));
        }

        let head_dimension = vb
            .get_metadata(".attention.key_length")
            .and_then(|v| v.to_u32().ok())
            .map(|x| x as usize)
            .unwrap_or_else(|| hidden_size / num_heads);

        let intermediate_size = vb
            .get_metadata(".feed_forward_length")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or((hidden_size * 4) as u32) as usize;

        let context_length = vb
            .get_metadata(".context_length")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(512) as usize;

        let max_relative_positions = vb
            .get_metadata(".attention.max_relative_positions")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(512) as usize;

        let norm_eps = vb
            .get_metadata(".attention.layer_norm_epsilon")
            .and_then(|v| v.to_f32().ok())
            .unwrap_or(1e-7);

        let vocab_size = vb
            .get_metadata(".vocab_size")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(250105) as usize;

        let position_buckets = vb
            .get_metadata(".attention.position_buckets")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(256) as usize;

        let share_att_key = vb
            .get_metadata(".attention.share_att_key")
            .and_then(|v| v.to_bool().ok())
            .unwrap_or(true);

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
