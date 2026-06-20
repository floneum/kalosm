use fusor::{Device, VarBuilder};
use fusor::{Result, Tensor};

use super::{BertAttention, BertIntermediate, BertOutput};

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L470
pub(crate) struct BertLayer {
    attention: BertAttention,
    intermediate: BertIntermediate,
    output: BertOutput,
    span: tracing::Span,
}

impl BertLayer {
    pub(crate) fn load(
        device: &Device,
        vb: &mut VarBuilder,
        config: &super::Config,
    ) -> Result<Self> {
        let attention = BertAttention::load(device, vb, config)?;
        let intermediate = BertIntermediate::load(device, vb, config)?;
        let output = BertOutput::load(device, vb, config)?;
        Ok(Self {
            attention,
            intermediate,
            output,
            span: tracing::span!(tracing::Level::TRACE, "layer"),
        })
    }

    pub(crate) fn forward(
        &self,
        hidden_states: &Tensor<3, f32>,
        attention_mask: Option<&Tensor<2, u32>>,
    ) -> Tensor<3, f32> {
        let _enter = self.span.enter();
        let attention_output = self.attention.forward(hidden_states, attention_mask);
        let intermediate_output = self.intermediate.forward(&attention_output);
        self.output.forward(&intermediate_output, &attention_output)
    }
}
