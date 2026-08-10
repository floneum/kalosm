use fusor2::device::Device;
use fusor2::layers::{LayerNorm, Linear};
use fusor2::tensor::Dyn as Tensor;
use fusor2::{Result, VarBuilder};

use super::load_linear;

pub(crate) struct BertSelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
    span: tracing::Span,
}

impl BertSelfOutput {
    pub(crate) fn load(device: &Device, vb: &VarBuilder, config: &super::Config) -> Result<Self> {
        let dense = load_linear(&vb.pp("attn_output"), device)?;
        let layer_norm = LayerNorm::load(
            &vb.pp("attn_output_norm"),
            device.graph().handle(),
            config.layer_norm_eps as f32,
        )?;
        Ok(Self {
            dense,
            layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "self-out"),
        })
    }

    pub(crate) fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden_states = self.dense.forward(hidden_states)?;
        self.layer_norm.forward(&hidden_states.add(input_tensor)?)
    }
}
