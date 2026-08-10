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

use fusor2::graph::Graph;
use fusor2::layers::{LayerNorm, Linear};
use fusor2::tensor::Dyn as Tensor;
use fusor2::{Dim, Dtype, QMatrix};
use fusor2_gguf::VarBuilder;

pub mod image_encoder;
pub mod mask_decoder;
pub mod prompt_encoder;
pub mod sam;
pub mod tiny_vit;
pub mod transformer;

pub(crate) type Result<T> = fusor2::Result<T>;

/// `&[usize]` -> the `&[Dim]` the runtime-rank API wants.
pub(crate) fn dims(v: &[usize]) -> Vec<Dim> {
    v.iter().map(|&d| Dim::Const(d as u64)).collect()
}

/// Constant extent of axis `i` as a `usize`. SAM shapes are all static.
pub(crate) fn udim(t: &Tensor, i: usize) -> usize {
    t.dim(i)
        .as_const()
        .expect("SAM tensor extents are constant") as usize
}

/// One GGUF tensor as a dense `F32` value in `graph`.
///
/// `fusor2_gguf` reverses GGUF's fastest-varying-first extents at read, so
/// `raw.shape` is already row-major. `F16`/`BF16` entries are cast; a
/// block-quantized entry goes through `QMatrix` and is dequantized on device.
pub(crate) fn load_dense(vb: &VarBuilder, graph: &Graph, name: &str) -> Result<Tensor> {
    let raw = vb.get_raw(name)?;
    if matches!(raw.fmt, Dtype::Q(_)) {
        return QMatrix::load(vb, graph, name)?.dequantize();
    }
    let shape: Vec<Dim> = raw.shape.iter().map(|&d| Dim::Const(d)).collect();
    let dense = Tensor::from_slice(graph.handle(), raw.fmt, &shape, &raw.bytes)?;
    match raw.fmt {
        Dtype::F32 => Ok(dense),
        _ => dense.cast(Dtype::F32),
    }
}

/// `Linear::load` with the bias auto-detected from the file, matching the
/// reference loader which accepted either spelling.
pub(crate) fn linear(vb: &VarBuilder, graph: &Graph) -> Result<Linear> {
    Linear::load(vb, graph.handle(), vb.contains_key("bias"))
}

/// LayerNorm over the **channel** axis of a `(B, C, H, W)` tensor - Meta's
/// `LayerNorm2d`. fusor2's `LayerNorm` normalizes the last axis, so this is a
/// permute to channels-last, the last-axis norm, and the permute back; all
/// three are views the compiler is free to fold.
pub(crate) fn channel_layer_norm(ln: &LayerNorm, x: &Tensor) -> Result<Tensor> {
    let nhwc = x.permute(&[0, 2, 3, 1])?;
    let normed = ln.forward(&nhwc)?;
    normed.permute(&[0, 3, 1, 2])
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
        graph: &Graph,
        vb: &VarBuilder,
        expected_in: Option<usize>,
        expected_hidden: Option<usize>,
        activation: Activation,
    ) -> Result<Self> {
        let lin1 = linear(&vb.pp("lin1"), graph)?;
        let lin2 = linear(&vb.pp("lin2"), graph)?;
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

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.lin1.forward(xs)?;
        let xs = match self.activation {
            Activation::Gelu => xs.gelu()?,
            Activation::Relu => xs.relu()?,
        };
        self.lin2.forward(&xs)
    }
}
