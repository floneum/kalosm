use fusor::layers::Linear;
use fusor::{Device, VarBuilder};
use fusor::{Result, Tensor};

pub(crate) struct BertSelfAttention {
    query: Linear<f32>,
    key: Linear<f32>,
    value: Linear<f32>,
    num_attention_heads: usize,
    attention_head_size: usize,
    span: tracing::Span,
    span_softmax: tracing::Span,
}

impl BertSelfAttention {
    pub(crate) fn load(
        device: &Device,
        vb: &mut VarBuilder,
        config: &super::Config,
    ) -> Result<Self> {
        let attention_head_size = config.hidden_size / config.num_attention_heads;
        let query = Linear::load(device, &mut vb.pp("attn_q"))?;
        let value = Linear::load(device, &mut vb.pp("attn_v"))?;
        let key = Linear::load(device, &mut vb.pp("attn_k"))?;
        Ok(Self {
            query,
            key,
            value,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            span: tracing::span!(tracing::Level::TRACE, "self-attn"),
            span_softmax: tracing::span!(tracing::Level::TRACE, "softmax"),
        })
    }

    pub(crate) fn transpose_for_scores(&self, xs: &Tensor<3, f32>) -> Tensor<4, f32> {
        let shape = xs.shape();
        let new_x_shape = [
            shape[0],
            shape[1],
            self.num_attention_heads,
            self.attention_head_size,
        ];

        xs.reshape(new_x_shape).transpose(1, 2).to_concrete()
    }

    pub(crate) fn forward(
        &self,
        hidden_states: &Tensor<3, f32>,
        attention_mask: Option<&Tensor<2, u32>>,
    ) -> Tensor<3, f32> {
        let _enter = self.span.enter();
        let query_layer = self.query.forward(hidden_states);
        let key_layer = self.key.forward(hidden_states);
        let value_layer = self.value.forward(hidden_states);

        let query_layer = self.transpose_for_scores(&query_layer);
        let key_layer = self.transpose_for_scores(&key_layer);
        let value_layer = self.transpose_for_scores(&value_layer);

        let attention_scores = query_layer.mat_mul(&key_layer.t());
        let mut attention_scores =
            attention_scores.div_scalar((self.attention_head_size as f32).sqrt());

        // If there is an attention mask, filter the attention scores by that mask
        if let Some(attention_mask) = attention_mask {
            // The attention mask is a tensor of shape (bsize, seq_len)
            // the attention scores are a tensor of shape (bsize, _, seq_len, seq_len)
            // We expand the attention mask to (bsize, 1, 1, seq_len)
            let mask = attention_mask
                .unsqueeze::<3>(1)
                .unsqueeze::<4>(2)
                .to_concrete();
            let shape = attention_scores.shape();
            let mask: Tensor<4, f32> = mask.broadcast_as::<4>(shape).to_concrete().cast();
            // We use a value slightly larger that the true f32 min value to avoid NaN
            const FALSE_MIN: f32 = -3.4028235e34f32;
            let device = attention_scores.device();
            let on_false = Tensor::splat(&device, FALSE_MIN, shape);
            attention_scores = mask.where_cond(&attention_scores, &on_false);
        }

        let attention_probs = {
            let _enter_sm = self.span_softmax.enter();
            attention_scores.softmax_last_dim::<3>()
        };
        let context_layer = attention_probs.mat_mul(&value_layer);
        let context_layer = context_layer.transpose(1, 2).to_concrete();
        context_layer.flatten_last_n::<1, _>()
    }
}

// attention_probs before matmul: Tensor[dims 3, 12, 13, 13; f32]
// value_layer before matmul: Tensor[dims 3, 12, 13, 32; f32]
// context_layer after matmul: Tensor[dims 3, 12, 13, 32; f32]
// context_layer after change: Tensor[dims 3, 13, 384; f32]
