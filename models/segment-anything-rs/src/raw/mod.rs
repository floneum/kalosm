//! Raw SAM model implementation using fusor-ml.

use fusor::layers::Linear;
use fusor::{Device, Tensor, TensorBacking, VarBuilder};

#[allow(dead_code)]
pub mod image_encoder;
#[allow(dead_code)]
pub mod mask_decoder;
#[allow(dead_code)]
pub mod prompt_encoder;
#[allow(dead_code)]
pub mod sam;
#[allow(dead_code)]
pub mod tiny_vit;
#[allow(dead_code)]
pub mod transformer;

type Result<T> = fusor::Result<T>;

/// Activation function variants used in SAM.
#[derive(Debug, Clone, Copy)]
pub enum Activation {
    Gelu,
    Relu,
}

/// MLP block: Linear -> Activation -> Linear
pub struct MlpBlock {
    lin1: Linear<f32>,
    lin2: Linear<f32>,
    activation: Activation,
}

impl MlpBlock {
    pub fn load(
        device: &Device,
        vb: &mut VarBuilder,
        _embedding_dim: usize,
        _mlp_dim: usize,
        activation: Activation,
    ) -> Result<Self> {
        let lin1 = Linear::load(device, &mut vb.pp("lin1"))?;
        let lin2 = Linear::load(device, &mut vb.pp("lin2"))?;
        Ok(Self {
            lin1,
            lin2,
            activation,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor<3, f32, impl TensorBacking<3, Elem = f32>>,
    ) -> Tensor<3, f32> {
        let xs = self.lin1.forward(xs);
        let xs = match self.activation {
            Activation::Gelu => xs.gelu(),
            Activation::Relu => xs.relu(),
        };
        self.lin2.forward(&xs)
    }
}
