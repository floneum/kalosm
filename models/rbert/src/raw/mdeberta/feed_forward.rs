//! mDeBERTa Feed Forward Network.

use fusor::layers::Linear;
use fusor::{Device, Result, Tensor, VarBuilder};

/// Standard GELU Feed Forward Network for mDeBERTa.
///
/// Formula: FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
pub struct MDebertaFeedForward {
    intermediate: Linear<f32>,
    output: Linear<f32>,
}

impl MDebertaFeedForward {
    pub fn load(device: &Device, vb: &mut VarBuilder) -> Result<Self> {
        let intermediate = Linear::load(device, &mut vb.pp("intermediate"))?;
        let output = Linear::load(device, &mut vb.pp("output"))?;

        Ok(Self {
            intermediate,
            output,
        })
    }

    pub fn forward(&self, x: &Tensor<3, f32>) -> Tensor<3, f32> {
        // Intermediate: x @ W1 + b1, then GELU
        let hidden = self.intermediate.forward(x).gelu();
        // Output: hidden @ W2 + b2
        self.output.forward(&hidden)
    }
}
