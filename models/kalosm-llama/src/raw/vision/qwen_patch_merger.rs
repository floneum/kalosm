use fusor::{
    layers::{Linear, RmsNorm},
    CastTensor, CastTo, Device, FloatDataType, SimdElement, Tensor, VarBuilder,
};

pub(crate) struct Qwen2VLPatchMerger<F: FloatDataType + SimdElement> {
    hidden_size: usize,
    ln_q: RmsNorm<1, F>,
    mlp: [Linear<F>; 2],
}

impl<F: FloatDataType + SimdElement + Default> Qwen2VLPatchMerger<F>
where
    F: CastTo<f32> + CastTensor<f32>,
    f32: CastTo<F> + CastTensor<F>,
{
    pub(crate) fn new(
        in_channels: usize,
        _out_channels: usize,
        spatial_merge_size: usize,
        layer_norm_eps: f64,
        vb: &mut VarBuilder,
        device: &Device,
    ) -> fusor::Result<Self> {
        let hidden_size = in_channels * spatial_merge_size.pow(2);

        let ln_q_weight: Tensor<1, F> = vb.get("v.post_ln.weight", device)?.dequantize().cast();
        let ln_q = RmsNorm::new(ln_q_weight, None, layer_norm_eps as f32);

        let mlp_0_weight = vb.get("mm.0.weight", device)?;
        let mlp_0_bias: Tensor<1, F> = vb.get("mm.0.bias", device)?.dequantize().cast();
        let mlp_0 = Linear::new(mlp_0_weight, Some(mlp_0_bias));

        let mlp_2_weight = vb.get("mm.2.weight", device)?;
        let mlp_2_bias: Tensor<1, F> = vb.get("mm.2.bias", device)?.dequantize().cast();
        let mlp_2 = Linear::new(mlp_2_weight, Some(mlp_2_bias));

        Ok(Self {
            hidden_size,
            ln_q,
            mlp: [mlp_0, mlp_2],
        })
    }

    pub(crate) fn forward(&self, x: &Tensor<2, F>) -> Tensor<2, F> {
        let [seq_len, _] = x.shape();
        // Work in f32 for RmsNorm
        let x_3d = x.unsqueeze(0);
        let x_3d = self.ln_q.forward_generic(&x_3d);
        let x = x_3d.squeeze(0).to_concrete();
        let x = x.reshape([
            seq_len / (self.hidden_size / x.shape()[1]),
            self.hidden_size,
        ]);

        // Linear expects 3D input: (batch, seq, dim)
        // We have (seq_len, dim). Treat as (1, seq_len, dim).
        let x_3d = x.unsqueeze(0).to_concrete();

        let x_3d = self.mlp[0].forward_generic(&x_3d);
        // Work in f32 for gelu
        let x_3d_f32: Tensor<3, f32> = x_3d.cast();
        let x_3d: Tensor<3, F> = x_3d_f32.gelu().cast();
        let x_3d = self.mlp[1].forward_generic(&x_3d);

        // Squeeze back to 2D
        x_3d.squeeze(0).to_concrete()
    }
}
