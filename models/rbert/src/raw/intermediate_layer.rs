use fusor2::device::Device;
use fusor2::layers::Linear;
use fusor2::tensor::Tensor;
use fusor2::{Result, VarBuilder};

use super::{load_linear, HiddenActLayer};

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L441
pub(crate) struct BertIntermediate {
    dense: Linear,
    intermediate_act: HiddenActLayer,
    span: tracing::Span,
}

impl BertIntermediate {
    pub(crate) fn load(device: &Device, vb: &VarBuilder, config: &super::Config) -> Result<Self> {
        let dense = load_linear(&vb.pp("ffn_up"), device)?;
        Ok(Self {
            dense,
            intermediate_act: HiddenActLayer::new(config.hidden_act),
            span: tracing::span!(tracing::Level::TRACE, "inter"),
        })
    }

    pub(crate) fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden_states = self.dense.forward(hidden_states)?;
        self.intermediate_act.forward(&hidden_states)
    }
}
