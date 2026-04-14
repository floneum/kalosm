//! ModernBERT transformer layer with pre-norm architecture.

use fusor::layers::LayerNorm;
use fusor::{Device, Result, RopeCache, Tensor, VarBuilder};

use super::attention::ModernBertAttention;
use super::feed_forward::GeGluFeedForward;

/// A single ModernBERT transformer layer with pre-norm architecture.
///
/// Note: The first layer (index 0) doesn't have its own attention_norm because
/// the embedding norm serves that purpose. This is handled by making attention_norm
/// optional and passing in the pre-normalized input for layer 0.
pub struct ModernBertLayer {
    /// Pre-attention RMSNorm (None for layer 0, which uses embedding norm)
    attention_norm: Option<LayerNorm<1, f32>>,
    attention: ModernBertAttention,
    ffn_norm: LayerNorm<1, f32>,
    feed_forward: GeGluFeedForward,
}

impl ModernBertLayer {
    pub fn load(
        device: &Device,
        vb: &mut VarBuilder,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        eps: f32,
        layer_idx: usize,
    ) -> Result<Self> {
        // Layer 0 doesn't have attn_norm - it uses embedding norm instead
        let attention_norm = if layer_idx > 0 {
            Some(LayerNorm::load(device, &mut vb.pp("attn_norm"), eps)?)
        } else {
            None
        };

        let attention =
            ModernBertAttention::load(device, vb, num_heads, num_kv_heads, head_dim, eps)?;
        let ffn_norm = LayerNorm::load(device, &mut vb.pp("ffn_norm"), eps)?;
        let feed_forward = GeGluFeedForward::load(device, vb)?;

        Ok(Self {
            attention_norm,
            attention,
            ffn_norm,
            feed_forward,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor<3, f32>,
        rope_cache: &RopeCache,
        attention_mask: Option<&Tensor<2, u32>>,
    ) -> Tensor<3, f32> {
        // Pre-norm + attention + residual
        // For layer 0, hidden_states is already normalized by embedding norm
        let residual = hidden_states;
        let hidden_states = if let Some(ref norm) = self.attention_norm {
            norm.forward(hidden_states)
        } else {
            hidden_states.clone()
        };
        let hidden_states = self
            .attention
            .forward(&hidden_states, rope_cache, attention_mask);
        let hidden_states = residual.add_(&hidden_states);

        // Pre-norm + FFN + residual
        let residual = &hidden_states;
        let ffn_input = self.ffn_norm.forward(&hidden_states);
        let ffn_output = self.feed_forward.forward(&ffn_input);
        residual.add_(&ffn_output)
    }
}
