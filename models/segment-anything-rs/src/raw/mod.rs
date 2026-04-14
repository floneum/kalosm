//! Raw SAM model port to fusor.
//!
//! # GGUF tensor naming contract
//!
//! Weights are loaded from GGUF files produced by `convert_to_gguf.py`. The
//! Rust loader keys off these conventions; changing them on either side will
//! silently mis-load:
//!
//! - The architecture is detected by `convert_to_gguf.py` from the substring
//!   `"patch_embed.seq"` in the source state-dict (TinyViT / MobileSAM) vs.
//!   the standard ViT-B layout.
//! - TinyViT's fused `Conv2dBN` blocks expect a single tensor name with
//!   `.c.weight` (conv) and `.bn.weight` / `.bn.bias` (batch-norm) suffixes;
//!   the converter folds the BN stats into the conv kernel before writing.
//! - All other naming is the upstream Meta SegmentAnything PyTorch layout
//!   (`image_encoder.*`, `prompt_encoder.*`, `mask_decoder.*`).

use fusor::layers::Linear;
use fusor::{Device, Tensor, TensorBacking, VarBuilder};

pub mod image_encoder;
pub mod mask_decoder;
pub mod prompt_encoder;
pub mod sam;
pub mod tiny_vit;
pub mod transformer;

pub(crate) type Result<T> = fusor::Result<T>;

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
    /// Load an MLP block from `vb`. `expected_in` / `expected_hidden`, when
    /// provided, are checked against the actual loaded shapes so a mismatch
    /// fails at load time rather than producing wrong outputs.
    pub fn load(
        device: &Device,
        vb: &mut VarBuilder,
        expected_in: Option<usize>,
        expected_hidden: Option<usize>,
        activation: Activation,
    ) -> Result<Self> {
        let lin1 = Linear::load(device, &mut vb.pp("lin1"))?;
        let lin2 = Linear::load(device, &mut vb.pp("lin2"))?;
        if let Some(d_in) = expected_in {
            assert_eq!(
                lin1.in_features(),
                d_in,
                "MlpBlock lin1 in_features mismatch"
            );
            assert_eq!(
                lin2.out_features(),
                d_in,
                "MlpBlock lin2 out_features mismatch"
            );
        }
        if let Some(d_hidden) = expected_hidden {
            assert_eq!(
                lin1.out_features(),
                d_hidden,
                "MlpBlock lin1 out_features mismatch"
            );
            assert_eq!(
                lin2.in_features(),
                d_hidden,
                "MlpBlock lin2 in_features mismatch"
            );
        }
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
