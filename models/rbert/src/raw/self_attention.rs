use fusor::cache::MaskKind;
use fusor::layers::Linear;
use fusor::{Device, Dim, Result, Tensor, VarBuilder};

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

    /// `[batch, seq, hidden] -> [batch, heads, seq, head_dim]`.
    fn transpose_for_scores(&self, xs: &Tensor<3>) -> Tensor<4> {
        let [batch, seq, _] = xs.extents();
        xs.reshape_dims([
            batch,
            seq,
            Dim::Const(self.num_attention_heads as u64),
            Dim::Const(self.attention_head_size as u64),
        ])
        .transpose(1, 2)
    }

    pub(crate) fn forward(
        &self,
        hidden_states: &Tensor<3>,
        attention_mask: Option<&Tensor<2, u32>>,
    ) -> Tensor<3> {
        let _enter = self.span.enter();
        let scale = 1.0 / (self.attention_head_size as f32).sqrt();
        let query_layer = self.transpose_for_scores(&self.query.forward(hidden_states));
        let key_layer = self.transpose_for_scores(&self.key.forward(hidden_states));
        let value_layer = self.transpose_for_scores(&self.value.forward(hidden_states));

        let context_layer = match attention_mask {
            Some(attention_mask) => {
                // `[batch, key]` validity mask as an additive batch-key mask.
                let mask = additive_key_mask(attention_mask);
                query_layer.attention_masked(
                    &key_layer,
                    &value_layer,
                    MaskKind::BatchKeyMask,
                    Some(&mask),
                    Some(scale),
                )
            }
            None => query_layer.attention(&key_layer, &value_layer, MaskKind::None, Some(scale)),
        };
        // `[batch, heads, seq, head_dim] -> [batch, seq, hidden]`.
        context_layer.transpose(1, 2).flatten_last_n(1)
    }
}
