use fusor2::cache::MaskKind;
use fusor2::composite::attention::{attention, attention_masked};
use fusor2::device::Device;
use fusor2::layers::Linear;
use fusor2::tensor::Dyn as Tensor;
use fusor2::{Dim, Result, VarBuilder};

use super::{additive_key_mask, load_linear};

pub(crate) struct BertSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    num_attention_heads: usize,
    attention_head_size: usize,
    span: tracing::Span,
}

impl BertSelfAttention {
    pub(crate) fn load(device: &Device, vb: &VarBuilder, config: &super::Config) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let query = load_linear(&vb.pp("attn_q"), device)?;
        let value = load_linear(&vb.pp("attn_v"), device)?;
        let key = load_linear(&vb.pp("attn_k"), device)?;
        Ok(Self {
            query,
            key,
            value,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            span: tracing::span!(tracing::Level::TRACE, "self-attn"),
        })
    }

    /// `[B, L, H] -> [B, heads, L, head_dim]`.
    fn transpose_for_scores(&self, xs: &Tensor) -> Result<Tensor> {
        let shape = xs.shape();
        xs.reshape_dims(&[
            shape[0],
            shape[1],
            Dim::Const(self.num_attention_heads as u64),
            Dim::Const(self.attention_head_size as u64),
        ])?
        .transpose(1, 2)
    }

    pub(crate) fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let scale = 1.0 / (self.attention_head_size as f32).sqrt();
        let query_layer = self.transpose_for_scores(&self.query.forward(hidden_states)?)?;
        let key_layer = self.transpose_for_scores(&self.key.forward(hidden_states)?)?;
        let value_layer = self.transpose_for_scores(&self.value.forward(hidden_states)?)?;

        let context_layer = match attention_mask {
            Some(attention_mask) => {
                // `[B, Lk]` validity mask as an additive batch-key mask.
                let mask = additive_key_mask(attention_mask)?;
                attention_masked(
                    &query_layer,
                    &key_layer,
                    &value_layer,
                    MaskKind::BatchKeyMask,
                    Some(&mask),
                    Some(scale),
                )?
            }
            None => attention(
                &query_layer,
                &key_layer,
                &value_layer,
                MaskKind::None,
                Some(scale),
            )?,
        };
        // `[B, heads, L, head_dim] -> [B, L, H]`.
        context_layer.transpose(1, 2)?.flatten_last_n(1)
    }
}
