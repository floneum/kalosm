//! mDeBERTa transformer layer.

use fusor::layers::LayerNorm;
use fusor::{Device, Result, Tensor, VarBuilder};

use super::attention::DisentangledSelfAttention;
use super::feed_forward::MDebertaFeedForward;

/// A single mDeBERTa transformer layer.
///
/// Architecture:
/// 1. Self-attention with disentangled attention
/// 2. Add & LayerNorm
/// 3. Feed-forward network
/// 4. Add & LayerNorm
pub struct MDebertaLayer {
    attention: DisentangledSelfAttention,
    attention_norm: LayerNorm<1, f32>,
    feed_forward: MDebertaFeedForward,
    output_norm: LayerNorm<1, f32>,
}

impl MDebertaLayer {
    pub fn load(
        device: &Device,
        vb: &mut VarBuilder,
        num_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<Self> {
        let attention = DisentangledSelfAttention::load(
            device,
            &mut vb.pp("attention"),
            num_heads,
            head_dim,
        )?;
        let attention_norm = LayerNorm::load(device, &mut vb.pp("attention_norm"), eps)?;
        let feed_forward = MDebertaFeedForward::load(device, &mut vb.pp("ffn"))?;
        let output_norm = LayerNorm::load(device, &mut vb.pp("output_norm"), eps)?;

        Ok(Self {
            attention,
            attention_norm,
            feed_forward,
            output_norm,
        })
    }

    /// Forward pass through the layer with proper position attention.
    ///
    /// # Arguments
    /// * `hidden_states` - Input [batch, seq_len, hidden_size]
    /// * `rel_pos_emb` - Relative position embedding table [2*max_pos, hidden_size]
    /// * `rel_pos_indices` - Relative position indices [seq_len, seq_len]
    /// * `attention_mask` - Optional attention mask [batch, seq_len]
    pub fn forward_with_rel(
        &self,
        hidden_states: &Tensor<3, f32>,
        rel_pos_emb: &Tensor<2, f32>,
        rel_pos_indices: &Tensor<2, u32>,
        attention_mask: Option<&Tensor<2, u32>>,
    ) -> Tensor<3, f32> {
        // Self-attention + residual + norm
        let attn_output = self.attention.forward_with_rel(hidden_states, rel_pos_emb, rel_pos_indices, attention_mask);
        let hidden_states = self.attention_norm.forward(&hidden_states.add_(&attn_output));

        // FFN + residual + norm
        let ffn_output = self.feed_forward.forward(&hidden_states);
        self.output_norm.forward(&hidden_states.add_(&ffn_output))
    }

    /// Legacy forward pass (for compatibility).
    pub fn forward(
        &self,
        hidden_states: &Tensor<3, f32>,
        rel_pos_emb: Option<&Tensor<3, f32>>,
        attention_mask: Option<&Tensor<2, u32>>,
    ) -> Tensor<3, f32> {
        // Self-attention + residual + norm
        let attn_output = self.attention.forward(hidden_states, rel_pos_emb, attention_mask);
        let hidden_states = self.attention_norm.forward(&hidden_states.add_(&attn_output));

        // FFN + residual + norm
        let ffn_output = self.feed_forward.forward(&hidden_states);
        self.output_norm.forward(&hidden_states.add_(&ffn_output))
    }
}
