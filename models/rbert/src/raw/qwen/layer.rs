use fusor2::layers::RmsNorm;
use fusor2::{Device, Result, Tensor, VarBuilder};

use super::attention::QwenSelfAttention;
use super::feed_forward::QwenFeedForward;
use super::model::QwenRope;

/// A single Qwen transformer layer with pre-norm architecture
pub struct QwenLayer {
    attention_norm: RmsNorm,
    attention: QwenSelfAttention,
    ffn_norm: RmsNorm,
    feed_forward: QwenFeedForward,
}

impl QwenLayer {
    pub fn load(
        device: &Device,
        vb: &VarBuilder,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<Self> {
        let graph = device.graph().handle();
        let attention_norm = RmsNorm::load(&vb.pp("attn_norm"), graph, eps)?;
        let attention =
            QwenSelfAttention::load(device, vb, num_heads, num_kv_heads, head_dim, eps)?;
        let ffn_norm = RmsNorm::load(&vb.pp("ffn_norm"), graph, eps)?;
        let feed_forward = QwenFeedForward::load(device, vb)?;

        Ok(Self {
            attention_norm,
            attention,
            ffn_norm,
            feed_forward,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor<3>,
        rope: &QwenRope,
        attention_mask: Option<&Tensor<2, u32>>,
    ) -> Tensor<3> {
        // Pre-norm + attention + residual
        let residual = hidden_states;
        let hidden_states = self.attention_norm.forward(hidden_states);
        let hidden_states = self.attention.forward(&hidden_states, rope, attention_mask);
        let hidden_states = residual.add(&hidden_states);

        // Pre-norm + FFN + residual
        let residual = &hidden_states;
        let ffn_input = self.ffn_norm.forward(&hidden_states);
        let ffn_output = self.feed_forward.forward(&ffn_input);
        residual.add(&ffn_output)
    }
}
