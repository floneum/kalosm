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
    /// Dimension per attention head.
    pub head_dimension: usize,
    /// LayerNorm epsilon.
    pub norm_eps: f32,
    /// Whether the disentangled attention reuses the content Q/K projections for
    /// position embeddings (`share_att_key=true`; the only supported path).
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

        let norm_eps = load_f32_or(vb, ".attention.layer_norm_epsilon", 1e-7);
        let share_att_key = load_bool_or(vb, ".attention.share_att_key", true);

        Ok(Self {
            num_heads,
            num_layers,
            head_dimension,
            norm_eps,
            share_att_key,
        })
    }
}
