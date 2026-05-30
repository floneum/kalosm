use fusor::layers::{LayerNorm, Linear};
use fusor::{Device, VarBuilder};
use fusor::{Result, Tensor};

pub(crate) struct BertSelfOutput {
    dense: Linear<f32>,
    layer_norm: LayerNorm<1, f32>,
    span: tracing::Span,
}

impl BertSelfOutput {
    pub(crate) fn load(
        device: &Device,
        vb: &mut VarBuilder,
        config: &super::Config,
    ) -> Result<Self> {
        let dense = Linear::load(device, &mut vb.pp("attn_output"))?;
        let layer_norm = LayerNorm::load(
            device,
            &mut vb.pp("attn_output_norm"),
            config.layer_norm_eps as _,
        )?;
        Ok(Self {
            dense,
            layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "self-out"),
        })
    }

    pub(crate) fn forward(
        &self,
        hidden_states: &Tensor<3, f32>,
        input_tensor: &Tensor<3, f32>,
    ) -> Tensor<3, f32> {
        let _enter = self.span.enter();
        let hidden_states = self.dense.forward(hidden_states);
        self.layer_norm.forward(&(&hidden_states + input_tensor))
    }
}
