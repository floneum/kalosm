//! ModernBERT configuration from GGUF metadata.

use fusor::{Result, VarBuilder};

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
        let num_heads = vb
            .get_metadata(".attention.head_count")
            .and_then(|v| v.to_u32().ok())
            .ok_or_else(|| {
                fusor::Error::msg("Missing required GGUF metadata: .attention.head_count")
            })? as usize;

        let num_kv_heads = vb
            .get_metadata(".attention.head_count_kv")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(num_heads as u32) as usize;

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

        let intermediate_size = vb
            .get_metadata(".feed_forward_length")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or((hidden_size * 4) as u32) as usize;

        let context_length = vb
            .get_metadata(".context_length")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(8192) as usize;

        let rope_theta = vb
            .get_metadata(".rope.freq_base")
            .and_then(|v| v.to_f32().ok())
            .unwrap_or(10000.0);

        let norm_eps = vb
            .get_metadata(".attention.layer_norm_rms_epsilon")
            .and_then(|v| v.to_f32().ok())
            .unwrap_or(1e-6);

        // Use attention.key_length for head dimension
        // Fall back to hidden_size / num_heads if not present
        let head_dimension = vb
            .get_metadata(".attention.key_length")
            .and_then(|v| v.to_u32().ok())
            .map(|x| x as usize)
            .unwrap_or_else(|| hidden_size / num_heads);

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
