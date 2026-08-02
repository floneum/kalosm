//! `ConvNd`: rank-generic convolution over the `Window` + `Contract` macro op.
//!
//! Owned by W13.

use fusor2_gguf::VarBuilder;
use smallvec::SmallVec;

use crate::tensor::Tensor;
use crate::{Error, Result};

pub struct ConvNd {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: SmallVec<[u32; 3]>,
    pub padding: SmallVec<[u32; 3]>,
    pub dilation: SmallVec<[u32; 3]>,
    pub groups: u32,
}

impl ConvNd {
    /// A convolution at the reference's `ConvNdConfig::default()`: unit
    /// stride, no padding, one group. The spatial rank comes from the weight,
    /// which is `[out_ch, in_ch / groups, ...kernel]`.
    ///
    /// Infallible, as the scaffold declares. A weight of rank < 2 has no
    /// spatial axes at all; that is refused by `forward`, which has an
    /// `Error` to put it in.
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        let spatial = weight.rank().saturating_sub(2);
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
        weight: Tensor,
        bias: Option<Tensor>,
        stride: &[u32],
        padding: &[u32],
        groups: u32,
    ) -> Self {
        let spatial = weight.rank().saturating_sub(2);
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
        let bias = if bias {
            let b = crate::layers::load_dense(vb, graph, "bias")?;
            Some(crate::layers::as_vector(b, "bias")?)
        } else {
            None
        };
        Ok(Self::new(weight, bias))
    }

    /// `[batch, in_ch, ...spatial] -> [batch, out_ch, ...out_spatial]`.
    ///
    /// One `Window` view plus one `Contract` that contracts the channel label
    /// and every kernel label at once — there is no im2col reshape, so the
    /// windowed operand never has to be materialized.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let spatial = self.weight.rank().checked_sub(2).ok_or_else(|| {
            Error::Shape(format!(
                "a conv weight is [out_ch, in_ch / groups, ...kernel]; got rank {}",
                self.weight.rank()
            ))
        })?;
        for (what, v) in [
            ("stride", &self.stride),
            ("padding", &self.padding),
            ("dilation", &self.dilation),
        ] {
            if v.len() != spatial {
                return Err(Error::Shape(format!(
                    "a {spatial}-d convolution needs {spatial} {what} entries, got {}",
                    v.len()
                )));
            }
        }
        crate::composite::conv::grouped_conv(
            x,
            &self.weight,
            self.bias.as_ref(),
            &self.stride,
            &self.padding,
            &self.dilation,
            self.groups.max(1),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fusor2_ir::dtype::Dtype;
    use fusor2_ir::shape::Dim;

    use crate::graph::Graph;
    use crate::session::{Device, Session};

    fn graph() -> Graph {
        Graph::new(&Session::new(Device::cpu().expect("cpu device")).expect("session"))
    }

    fn leaf(g: &Graph, shape: &[u64]) -> Tensor {
        let dims: Vec<Dim> = shape.iter().map(|d| Dim::Const(*d)).collect();
        g.leaf("t", &dims, Dtype::F32).unwrap()
    }

    #[test]
    fn the_default_configuration_is_unit_stride_no_padding_one_group() {
        let g = graph();
        let layer = ConvNd::new(leaf(&g, &[3, 2, 3]), None);
        assert_eq!(&layer.stride[..], &[1]);
        assert_eq!(&layer.padding[..], &[0]);
        assert_eq!(&layer.dilation[..], &[1]);
        assert_eq!(layer.groups, 1);
    }

    #[test]
    fn a_valid_1d_convolution_loses_k_minus_one_positions() {
        let g = graph();
        let x = leaf(&g, &[1, 2, 6]);
        let w = leaf(&g, &[3, 2, 3]);
        let b = leaf(&g, &[3]);
        let y = ConvNd::new(w, Some(b)).forward(&x).unwrap();
        assert_eq!(
            &y.shape()[..],
            &[Dim::Const(1), Dim::Const(3), Dim::Const(4)]
        );
    }

    #[test]
    fn padding_puts_the_positions_back() {
        let g = graph();
        let x = leaf(&g, &[1, 2, 6]);
        let w = leaf(&g, &[3, 2, 3]);
        let y = ConvNd::with_config(w, None, &[1], &[1], 1)
            .forward(&x)
            .unwrap();
        assert_eq!(
            &y.shape()[..],
            &[Dim::Const(1), Dim::Const(3), Dim::Const(6)]
        );
    }

    #[test]
    fn the_forward_is_exactly_the_conv_macro_op() {
        let g = graph();
        let x = leaf(&g, &[1, 2, 6]);
        let w = leaf(&g, &[3, 2, 3]);
        let b = leaf(&g, &[3]);
        let by_layer = ConvNd::new(w.clone(), Some(b.clone())).forward(&x).unwrap();
        let by_hand = crate::composite::conv::conv(&x, &w, Some(&b), &[1], &[0], &[1]).unwrap();
        assert_eq!(by_layer.id(), by_hand.id());
    }

    #[test]
    fn a_grouped_layer_reaches_the_grouped_macro_op() {
        let g = graph();
        let x = leaf(&g, &[1, 4, 6]);
        let w = leaf(&g, &[4, 2, 3]);
        let by_layer = ConvNd::with_config(w.clone(), None, &[1], &[0], 2)
            .forward(&x)
            .unwrap();
        let by_hand =
            crate::composite::conv::grouped_conv(&x, &w, None, &[1], &[0], &[1], 2).unwrap();
        assert_eq!(by_layer.id(), by_hand.id());
    }

    #[test]
    fn a_rank_one_weight_is_refused_rather_than_panicking() {
        let g = graph();
        let x = leaf(&g, &[1, 2, 6]);
        let w = leaf(&g, &[3]);
        assert!(ConvNd::new(w, None).forward(&x).is_err());
    }
}
