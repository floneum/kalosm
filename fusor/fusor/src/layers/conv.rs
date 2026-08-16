//! `ConvNd`: rank-generic convolution over the `Window` + `Contract` macro op.

use fusor_gguf::VarBuilder;
use smallvec::SmallVec;

use crate::device::ok;
use crate::tensor::typed::Element;
use crate::{Result, Tensor};

/// A convolution of any spatial rank.
///
/// `W` is the **weight's** rank — `[out_ch, in_ch / groups, ...kernel]`, so
/// `W - 2` is the number of spatial axes — and `T` its element type. Both
/// default so that a bare `ConvNd` is the 2-d f32 case.
pub struct ConvNd<const W: usize = 4, T: Element = f32> {
    weight: Tensor<W, T>,
    bias: Option<Tensor<1, T>>,
    /// Step of each spatial window.
    pub stride: SmallVec<[u32; 3]>,
    /// Zero padding on each spatial axis.
    pub padding: SmallVec<[u32; 3]>,
    dilation: SmallVec<[u32; 3]>,
    /// Number of channel groups.
    pub groups: u32,
}

impl<const W: usize, T: Element> ConvNd<W, T> {
    /// A convolution with unit stride, no padding, one group. The spatial
    /// rank comes from the weight, `[out_ch, in_ch / groups, ...kernel]`.
    ///
    /// Infallible; a weight of rank < 2 is refused by `forward`.
    pub fn new(weight: Tensor<W, T>, bias: Option<Tensor<1, T>>) -> Self {
        let spatial = W.saturating_sub(2);
        Self {
            weight,
            bias,
            stride: std::iter::repeat_n(1u32, spatial).collect(),
            padding: std::iter::repeat_n(0u32, spatial).collect(),
            dilation: std::iter::repeat_n(1u32, spatial).collect(),
            groups: 1,
        }
    }

    /// [`ConvNd::new`] with an explicit configuration.
    pub fn with_config(
        weight: Tensor<W, T>,
        bias: Option<Tensor<1, T>>,
        stride: &[u32],
        padding: &[u32],
        groups: u32,
    ) -> Self {
        let spatial = W.saturating_sub(2);
        Self {
            weight,
            bias,
            stride: stride.iter().copied().collect(),
            padding: padding.iter().copied().collect(),
            dilation: std::iter::repeat_n(1u32, spatial).collect(),
            groups: groups.max(1),
        }
    }

    /// `weight` is `[out_ch, in_ch / groups, ...kernel]` and `bias` is
    /// `[out_ch]`.
    pub fn load(vb: &VarBuilder, graph: &crate::graph::GraphRef, bias: bool) -> Result<Self> {
        let weight = crate::layers::load_dense(vb, graph, "weight")?;
        let weight = crate::layers::as_typed::<W, T>(
            weight,
            "a conv weight is [out_ch, in_ch / groups, ...kernel]",
        )?;
        let bias = if bias {
            let b = crate::layers::load_dense(vb, graph, "bias")?;
            let b = crate::layers::as_vector(b, "bias")?;
            Some(crate::layers::as_typed::<1, T>(b, "bias")?)
        } else {
            None
        };
        Ok(Self::new(weight, bias))
    }

    /// `[batch, in_ch, ...spatial] -> [batch, out_ch, ...out_spatial]`.
    /// Rank-preserving.
    ///
    /// One `Window` view plus one `Contract` over the channel label and every
    /// kernel label at once.
    #[track_caller]
    pub fn forward<const R: usize>(&self, x: &Tensor<R, T>) -> Tensor<R, T> {
        let spatial = match W.checked_sub(2) {
            Some(s) => s,
            None => ok(
                "ConvNd::forward",
                Err(crate::Error::Shape(format!(
                    "a conv weight is [out_ch, in_ch / groups, ...kernel]; got rank {W}"
                ))),
            ),
        };
        for (what, v) in [
            ("stride", &self.stride),
            ("padding", &self.padding),
            ("dilation", &self.dilation),
        ] {
            if v.len() != spatial {
                ok::<()>(
                    "ConvNd::forward",
                    Err(crate::Error::Shape(format!(
                        "a {spatial}-d convolution needs {spatial} {what} entries, got {}",
                        v.len()
                    ))),
                );
            }
        }
        Tensor::<R, T>::from_dyn(ok(
            "ConvNd::forward",
            crate::composite::conv::grouped_conv(
                x.as_dyn(),
                self.weight.as_dyn(),
                self.bias.as_ref().map(Tensor::as_dyn),
                &self.stride,
                &self.padding,
                &self.dilation,
                self.groups.max(1),
            ),
        ))
    }
}
