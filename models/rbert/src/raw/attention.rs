use fusor::{Device, VarBuilder};
use fusor::{Result, Tensor};

use super::{BertSelfAttention, BertSelfOutput};

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L392
pub(crate) struct BertAttention {
    self_attention: BertSelfAttention,
    self_output: BertSelfOutput,
    span: tracing::Span,
}

impl BertAttention {
    pub(crate) fn load(
        device: &Device,
        vb: &mut VarBuilder,
        config: &super::Config,
    ) -> Result<Self> {
        let self_attention = BertSelfAttention::load(device, vb, config)?;
        let self_output = BertSelfOutput::load(device, vb, config)?;
        Ok(Self {
            self_attention,
            self_output,
            span: tracing::span!(tracing::Level::TRACE, "attn"),
        })
    }

    pub(crate) fn forward(
        &self,
        hidden_states: &Tensor<3, f32>,
        attention_mask: Option<&Tensor<2, u32>>,
    ) -> Tensor<3, f32> {
        let _enter = self.span.enter();
        let self_outputs = self.self_attention.forward(hidden_states, attention_mask);
        self.self_output.forward(&self_outputs, hidden_states)
    }
}
