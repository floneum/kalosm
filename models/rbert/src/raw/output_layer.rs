use fusor2::device::Device;
use fusor2::layers::{LayerNorm, Linear};
use fusor2::tensor::Dyn as Tensor;
use fusor2::{Result, VarBuilder};

use super::load_linear;

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L456
pub(crate) struct BertOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    span: tracing::Span,
}

impl BertOutput {
    pub(crate) fn load(device: &Device, vb: &VarBuilder, config: &super::Config) -> Result<Self> {
        let dense = load_linear(&vb.pp("ffn_down"), device)?;
        let layer_norm = LayerNorm::load(
            &vb.pp("layer_output_norm"),
            device.graph().handle(),
            config.layer_norm_eps as f32,
        )?;
        Ok(Self {
            dense,
            layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "out"),
        })
    }

    pub(crate) fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden_states = self.dense.forward(hidden_states)?;
        self.layer_norm.forward(&hidden_states.add(input_tensor)?)
    }
}
