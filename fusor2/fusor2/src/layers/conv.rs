//! `ConvNd`: rank-generic convolution over the `Window` + `Contract` macro op.
//!
//! Owned by W13.

use fusor2_gguf::VarBuilder;
use smallvec::SmallVec;

use crate::device::ok;
use crate::tensor::typed::Element;
use crate::{Result, Tensor};

/// A convolution of any spatial rank.
///
/// `W` is the **weight's** rank — `[out_ch, in_ch / groups, ...kernel]`, so
/// `W - 2` is the number of spatial axes — and `T` its element type. Both
/// default so that a bare `ConvNd` is the 2-d f32 case, which is what every
/// vision model in this workspace loads. The reference has no `ConvNd` at all;
/// the parameterization follows the other layers here rather than inventing a
/// witness trait for the spatial rank.
pub struct ConvNd<const W: usize = 4, T: Element = f32> {
    pub weight: Tensor<W, T>,
    pub bias: Option<Tensor<1, T>>,
    pub stride: SmallVec<[u32; 3]>,
    pub padding: SmallVec<[u32; 3]>,
    pub dilation: SmallVec<[u32; 3]>,
    pub groups: u32,
}

impl<const W: usize, T: Element> ConvNd<W, T> {
    /// A convolution at the reference's `ConvNdConfig::default()`: unit
    /// stride, no padding, one group. The spatial rank comes from the weight,
    /// which is `[out_ch, in_ch / groups, ...kernel]`.
    ///
    /// Infallible. A weight of rank < 2 has no spatial axes at all; that is
    /// refused by `forward`, which has somewhere to put the diagnosis.
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
    /// One `Window` view plus one `Contract` that contracts the channel label
    /// and every kernel label at once — there is no im2col reshape, so the
    /// windowed operand never has to be materialized.
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

#[cfg(test)]
mod tests {
    use super::*;

    use crate::graph::Graph;
    use crate::layers::test_leaf as leaf;
    use crate::session::{Backend, Session};

    fn graph() -> Graph {
        Graph::new(&Session::new(Backend::cpu().expect("cpu device")).expect("session"))
    }

    #[test]
    fn the_default_configuration_is_unit_stride_no_padding_one_group() {
        let g = graph();
        let w: Tensor<3, f32> = leaf(&g, &[3, 2, 3]);
        let layer = ConvNd::new(w, None);
        assert_eq!(&layer.stride[..], &[1]);
        assert_eq!(&layer.padding[..], &[0]);
        assert_eq!(&layer.dilation[..], &[1]);
        assert_eq!(layer.groups, 1);
    }

    #[test]
    fn a_valid_1d_convolution_loses_k_minus_one_positions() {
        let g = graph();
        let x: Tensor<3, f32> = leaf(&g, &[1, 2, 6]);
        let w: Tensor<3, f32> = leaf(&g, &[3, 2, 3]);
        let b: Tensor<1, f32> = leaf(&g, &[3]);
        assert_eq!(ConvNd::new(w, Some(b)).forward(&x).shape(), [1, 3, 4]);
    }

    #[test]
    fn padding_puts_the_positions_back() {
        let g = graph();
        let x: Tensor<3, f32> = leaf(&g, &[1, 2, 6]);
        let w: Tensor<3, f32> = leaf(&g, &[3, 2, 3]);
        let y = ConvNd::with_config(w, None, &[1], &[1], 1).forward(&x);
        assert_eq!(y.shape(), [1, 3, 6]);
    }

    #[test]
    fn the_forward_is_exactly_the_conv_macro_op() {
        let g = graph();
        let x: Tensor<3, f32> = leaf(&g, &[1, 2, 6]);
        let w: Tensor<3, f32> = leaf(&g, &[3, 2, 3]);
        let b: Tensor<1, f32> = leaf(&g, &[3]);
        let by_layer = ConvNd::new(w.clone(), Some(b.clone())).forward(&x);
        let by_hand = crate::composite::conv::conv(
            x.as_dyn(),
            w.as_dyn(),
            Some(b.as_dyn()),
            &[1],
            &[0],
            &[1],
        )
        .unwrap();
        assert_eq!(by_layer.id(), by_hand.id());
    }

    #[test]
    fn a_grouped_layer_reaches_the_grouped_macro_op() {
        let g = graph();
        let x: Tensor<3, f32> = leaf(&g, &[1, 4, 6]);
        let w: Tensor<3, f32> = leaf(&g, &[4, 2, 3]);
        let by_layer = ConvNd::with_config(w.clone(), None, &[1], &[0], 2).forward(&x);
        let by_hand =
            crate::composite::conv::grouped_conv(x.as_dyn(), w.as_dyn(), None, &[1], &[0], &[1], 2)
                .unwrap();
        assert_eq!(by_layer.id(), by_hand.id());
    }

    /// A 2-d layer is the default parameterization, and it is what a vision
    /// model writes as a bare `ConvNd`.
    #[test]
    fn the_default_parameterization_is_the_two_dimensional_case() {
        let g = graph();
        let x: Tensor<4, f32> = leaf(&g, &[1, 2, 6, 6]);
        let w: Tensor<4, f32> = leaf(&g, &[3, 2, 3, 3]);
        let layer: ConvNd = ConvNd::new(w, None);
        assert_eq!(&layer.stride[..], &[1, 1]);
        assert_eq!(layer.forward(&x).shape(), [1, 3, 4, 4]);
    }
}
