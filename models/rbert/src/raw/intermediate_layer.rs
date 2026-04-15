use fusor::layers::Linear;
use fusor::{Device, VarBuilder};
use fusor::{Result, Tensor};

use super::HiddenActLayer;

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L441
pub(crate) struct BertIntermediate {
    dense: Linear<f32>,
    intermediate_act: HiddenActLayer,
    span: tracing::Span,
}

impl BertIntermediate {
    pub(crate) fn load(
        device: &Device,
        vb: &mut VarBuilder,
        config: &super::Config,
    ) -> Result<Self> {
        let dense = Linear::load(device, &mut vb.pp("ffn_up"))?;
        Ok(Self {
            dense,
            intermediate_act: HiddenActLayer::new(config.hidden_act),
            span: tracing::span!(tracing::Level::TRACE, "inter"),
        })
    }
}

impl BertIntermediate {
    pub(crate) fn forward(&self, hidden_states: &Tensor<3, f32>) -> Tensor<3, f32> {
        let _enter = self.span.enter();
        let hidden_states = self.dense.forward(hidden_states);
        self.intermediate_act.forward(&hidden_states)
    }
}
