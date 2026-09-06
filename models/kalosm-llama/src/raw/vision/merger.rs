//! The patch merger: every `merge × merge` block of vision tokens becomes
//! one language-model token through a norm and a two-layer MLP.

use fusor::layers::RmsNorm;
use fusor::{Device, Tensor};
use fusor_gguf::VarBuilder;

use crate::raw::dense_1d;
use crate::raw::weight::Weight;

pub(crate) struct PatchMerger {
    /// `hidden * merge²`: the width of one merged block.
    block_width: usize,
    ln_q: RmsNorm,
    fc1: Weight,
    fc1_bias: Tensor<1>,
    fc2: Weight,
    fc2_bias: Tensor<1>,
}

impl PatchMerger {
    pub(crate) fn load(
        vb: &VarBuilder,
        device: &Device,
        hidden_size: usize,
        merge: usize,
        eps: f32,
    ) -> fusor::Result<Self> {
        let graph = device.graph();
        Ok(Self {
            block_width: hidden_size * merge * merge,
            ln_q: RmsNorm::new(
                Some(dense_1d(device, &vb.get_raw("v.post_ln.weight")?)?),
                eps,
            ),
            fc1: Weight::from_raw(graph, &vb.get_raw("mm.0.weight")?)?,
            fc1_bias: dense_1d(device, &vb.get_raw("mm.0.bias")?)?,
            fc2: Weight::from_raw(graph, &vb.get_raw("mm.2.weight")?)?,
            fc2_bias: dense_1d(device, &vb.get_raw("mm.2.bias")?)?,
        })
    }

    /// `[patches, hidden]` in, `[patches / merge², llm_dim]` out.
    pub(crate) fn forward(&self, x: &Tensor<2>) -> Tensor<2> {
        let [seq_len, hidden] = x.shape();
        let x = self.ln_q.forward(x);
        let blocks = seq_len * hidden / self.block_width;
        let x = x.reshape([blocks, self.block_width]);
        let x: Tensor<2> = self.fc1.mat_mul(&x).add_(&self.fc1_bias);
        let x: Tensor<2> = self.fc2.mat_mul(&x.gelu_exact()).add_(&self.fc2_bias);
        x
    }
}
