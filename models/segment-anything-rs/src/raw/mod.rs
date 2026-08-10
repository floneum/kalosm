//! Raw SAM model port to fusor2.
//!
//! # GGUF tensor naming contract
//!
//! Weights are loaded from GGUF files using these tensor naming conventions;
//! changing them will silently mis-load:
//!
//! - TinyViT's fused `Conv2dBN` blocks expect a single tensor name with
//!   `.c.weight` (conv) and `.bn.weight` / `.bn.bias` (batch-norm) suffixes;
//!   the GGUF weights fold the BN stats into the conv kernel.
//! - All other naming is the upstream Meta SegmentAnything PyTorch layout
//!   (`image_encoder.*`, `prompt_encoder.*`, `mask_decoder.*`).

use fusor2::layers::{LayerNorm, Linear};
use fusor2::{Device, Dim, Dtype, Error, QMatrix, Tensor};
use fusor2_gguf::VarBuilder;

pub mod image_encoder;
pub mod mask_decoder;
pub mod prompt_encoder;
pub mod sam;
pub mod tiny_vit;
pub mod transformer;

/// Loading is the fallible boundary: it reads a file. Every `forward` below is
/// infallible, because a rank or extent mismatch inside one is a bug in the
/// port, not a condition a caller can act on.
pub(crate) type Result<T> = fusor2::Result<T>;

/// One GGUF tensor as a dense `F32` value of the rank the model expects.
///
/// `fusor2_gguf` reverses GGUF's fastest-varying-first extents at read, so
/// `raw.shape` is already row-major. `F16`/`BF16` entries are cast by
/// `Tensor::from_raw_bytes`; a block-quantized entry goes through `QMatrix`
/// and is dequantized on device.
pub(crate) fn load_dense<const R: usize>(
    vb: &VarBuilder,
    device: &Device,
    name: &str,
) -> Result<Tensor<R>> {
    let raw = vb.get_raw(name)?;
    if matches!(raw.fmt, Dtype::Q(_)) {
        let decoded = QMatrix::load(vb, device.graph(), name)?.to_tensor();
        return Tensor::<R>::try_from_dyn(decoded.into_dyn());
    }
    if raw.shape.len() != R {
        return Err(Error::Shape(format!(
            "{name} is rank {}, the model expects rank {R}",
            raw.shape.len()
        )));
    }
    let mut shape = [Dim::Const(0); R];
    for (slot, &d) in shape.iter_mut().zip(raw.shape.iter()) {
        *slot = Dim::Const(d);
    }
    Ok(Tensor::from_raw_bytes(device, raw.fmt, shape, &raw.bytes))
}

/// `Linear::load` with the bias auto-detected from the file, matching the
/// reference loader which accepted either spelling.
pub(crate) fn linear(vb: &VarBuilder, device: &Device) -> Result<Linear> {
    Linear::load(vb, device.graph().handle(), vb.contains_key("bias"))
}

/// LayerNorm over the **channel** axis of a `(B, C, H, W)` tensor - Meta's
/// `LayerNorm2d`. fusor2's `LayerNorm` normalizes the last axis, so this is a
/// permute to channels-last, the last-axis norm, and the permute back; all
/// three are views the compiler is free to fold.
pub(crate) fn channel_layer_norm(ln: &LayerNorm, x: &Tensor<4>) -> Tensor<4> {
    ln.forward(&x.permute([0, 2, 3, 1])).permute([0, 3, 1, 2])
}

/// Activation function variants used in SAM.
#[derive(Debug, Clone, Copy)]
pub enum Activation {
    Gelu,
    Relu,
}

/// MLP block: Linear -> Activation -> Linear
pub struct MlpBlock {
    lin1: Linear,
    lin2: Linear,
    activation: Activation,
}

impl MlpBlock {
    /// Load an MLP block from `vb`. `expected_in` / `expected_hidden`, when
    /// provided, are checked against the actual loaded shapes so a mismatch
    /// fails at load time rather than producing wrong outputs.
    pub fn load(
        device: &Device,
        vb: &VarBuilder,
        expected_in: Option<usize>,
        expected_hidden: Option<usize>,
        activation: Activation,
    ) -> Result<Self> {
        let lin1 = linear(&vb.pp("lin1"), device)?;
        let lin2 = linear(&vb.pp("lin2"), device)?;
        if let Some(d_in) = expected_in {
            assert_eq!(
                lin1.in_features().as_const(),
                Some(d_in as u64),
                "MlpBlock lin1 in_features mismatch"
            );
            assert_eq!(
                lin2.out_features().as_const(),
                Some(d_in as u64),
                "MlpBlock lin2 out_features mismatch"
            );
        }
        if let Some(d_hidden) = expected_hidden {
            assert_eq!(
                lin1.out_features().as_const(),
                Some(d_hidden as u64),
                "MlpBlock lin1 out_features mismatch"
            );
            assert_eq!(
                lin2.in_features().as_const(),
                Some(d_hidden as u64),
                "MlpBlock lin2 in_features mismatch"
            );
        }
        Ok(Self {
            lin1,
            lin2,
            activation,
        })
    }

    pub fn forward<const R: usize>(&self, xs: &Tensor<R>) -> Tensor<R> {
        let xs = self.lin1.forward(xs);
        let xs = match self.activation {
            Activation::Gelu => xs.gelu(),
            Activation::Relu => xs.relu(),
        };
        self.lin2.forward(&xs)
    }
}
