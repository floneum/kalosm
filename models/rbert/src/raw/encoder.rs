use fusor2::device::Device;
use fusor2::tensor::Tensor;
use fusor2::{Result, VarBuilder};

use super::BertLayer;

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L556
pub(crate) struct BertEncoder {
    layers: Vec<BertLayer>,
    span: tracing::Span,
}

impl BertEncoder {
    pub(crate) fn load(device: &Device, vb: &VarBuilder, config: &super::Config) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|index| BertLayer::load(device, &vb.pp(format!("blk.{index}")), config))
            .collect::<Result<Vec<_>>>()?;
        let span = tracing::span!(tracing::Level::TRACE, "encoder");
        Ok(BertEncoder { layers, span })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let mut hidden_states = hidden_states.clone();
        // Use a loop rather than a fold as it's easier to modify when adding debug/...
        for layer in self.layers.iter() {
            hidden_states = layer.forward(&hidden_states, attention_mask)?;
        }
        Ok(hidden_states)
    }
}
