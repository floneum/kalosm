//! `LayerNorm` over the last axis and `LayerNormNd` over a trailing group.

use fusor2_gguf::VarBuilder;
use fusor2_ir::shape::Dim;

use crate::device::ok;
use crate::tensor::Dyn;
use crate::tensor::typed::Element;
use crate::{Error, Result, Tensor};

/// `(x - mean) / sqrt(var + eps) * weight + bias` over the last axis.
///
/// `N` is the **affine parameters'** rank and `T` their element type; both
/// default so that `LayerNorm` alone still names the rank-1 f32 case every
/// checkpoint ships. The activation's rank is [`LayerNorm::forward`]'s.
pub struct LayerNorm<const N: usize = 1, T: Element = f32> {
    pub weight: Tensor<N, T>,
    pub bias: Option<Tensor<N, T>>,
    pub eps: f32,
}

impl<const N: usize, T: Element> LayerNorm<N, T> {
    pub fn new(weight: Tensor<N, T>, bias: Option<Tensor<N, T>>, eps: f32) -> Self {
        Self { weight, bias, eps }
    }

    /// `weight` is required and `bias` is optional. Both are squeezed to
    /// rank 1 when the file writes them as `[1, n]`.
    pub fn load(vb: &VarBuilder, graph: &crate::graph::GraphRef, eps: f32) -> Result<Self> {
        let w = crate::layers::load_dense(vb, graph, "weight")?;
        let w = crate::layers::as_vector(w, "weight")?;
        let weight = crate::layers::as_typed::<N, T>(w, "a LayerNorm weight")?;
        let bias = match crate::layers::load_optional(vb, graph, "bias")? {
            Some(b) => {
                let b = crate::layers::as_vector(b, "bias")?;
                Some(crate::layers::as_typed::<N, T>(b, "a LayerNorm bias")?)
            }
            None => None,
        };
        Ok(Self { weight, bias, eps })
    }

    /// `(x - mean) / sqrt(var + eps) * weight + bias` over the last axis, with
    /// the **biased** variance — the divisor is the axis extent, not `n - 1`.
    /// Rank-preserving.
    #[track_caller]
    pub fn forward<const R: usize>(&self, x: &Tensor<R, T>) -> Tensor<R, T> {
        x.layer_norm(&self.weight, self.bias.as_ref(), self.eps, true)
    }
}

/// Normalizes over the trailing `axes` rather than just the last one.
///
/// The affine parameters may be shaped like the group — `[3, 4]` for
/// `axes == 2`.
pub struct LayerNormNd<const N: usize = 1, T: Element = f32> {
    pub inner: LayerNorm<N, T>,
    pub axes: u32,
}

impl<const N: usize, T: Element> LayerNormNd<N, T> {
    pub fn new(inner: LayerNorm<N, T>, axes: u32) -> Self {
        Self { inner, axes }
    }

    /// The statistic is taken over the **flattened tail**, not per trailing
    /// axis: `[a, b, c]` with `axes == 2` has one mean and one variance per
    /// `a`, over all `b * c` elements.
    ///
    /// So this is the last-axis path over a reshaped view, and the affine
    /// parameters flatten with it.
    #[track_caller]
    pub fn forward<const R: usize>(&self, x: &Tensor<R, T>) -> Tensor<R, T> {
        Tensor::<R, T>::from_dyn(ok(
            "LayerNormNd::forward",
            forward_nd_dyn(
                x.as_dyn(),
                self.inner.weight.as_dyn(),
                self.inner.bias.as_ref().map(Tensor::as_dyn),
                self.inner.eps,
                self.axes,
            ),
        ))
    }
}

/// The runtime-rank trailing-group forward. The typed `forward` wraps this
/// because the flattened view's rank is `R - (axes - 1)`, which no caller has
/// a reason to name.
pub(crate) fn forward_nd_dyn(
    x: &Dyn,
    weight: &Dyn,
    bias: Option<&Dyn>,
    eps: f32,
    axes: u32,
) -> Result<Dyn> {
    let axes = axes as usize;
    if axes == 0 {
        return Err(Error::Shape(
            "LayerNormNd needs at least one axis to normalize over".into(),
        ));
    }
    if axes > x.rank() {
        return Err(Error::Shape(format!(
            "LayerNormNd normalizes {axes} trailing axes of a rank-{} value",
            x.rank()
        )));
    }
    if axes == 1 {
        return x.layer_norm(weight, bias, eps, true);
    }

    let shape = x.shape();
    let flat = x.flatten_last_n(axes - 1)?;
    let tail = flat.dim(flat.rank() - 1);
    let weight = flatten_affine(weight, tail, "weight")?;
    let bias = match bias {
        Some(b) => Some(flatten_affine(b, tail, "bias")?),
        None => None,
    };
    let y = flat.layer_norm(&weight, bias.as_ref(), eps, true)?;
    y.reshape_dims(&shape)
}

/// An affine parameter shaped like the group being normalized, flattened to
/// the one axis the last-axis path scales.
fn flatten_affine(p: &Dyn, tail: Dim, what: &str) -> Result<Dyn> {
    let flat = match p.rank() {
        0 => {
            return Err(Error::Shape(format!(
                "the LayerNormNd {what} is rank 0; it must cover the normalized group"
            )));
        }
        1 => p.clone(),
        _ => p.flatten_all()?,
    };
    if !flat.dim(0).known_eq(tail) {
        return Err(Error::Shape(format!(
            "the LayerNormNd {what} covers {} elements, but the normalized group is {tail}",
            flat.dim(0)
        )));
    }
    Ok(flat)
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
    fn the_last_axis_form_preserves_the_shape() {
        let g = graph();
        let x: Tensor<2, f32> = leaf(&g, &[3, 6]);
        let w: Tensor<1, f32> = leaf(&g, &[6]);
        let b: Tensor<1, f32> = leaf(&g, &[6]);
        assert_eq!(LayerNorm::new(w, Some(b), 1e-5).forward(&x).shape(), [3, 6]);
    }

    #[test]
    fn the_nd_form_normalizes_the_flattened_tail_and_reshapes_back() {
        let g = graph();
        let x: Tensor<3, f32> = leaf(&g, &[2, 3, 4]);
        let w: Tensor<2, f32> = leaf(&g, &[3, 4]);
        let layer = LayerNormNd::new(LayerNorm::new(w, None, 1e-5), 2);
        assert_eq!(layer.forward(&x).shape(), [2, 3, 4]);
    }

    #[test]
    fn one_trailing_axis_is_exactly_the_last_axis_form() {
        let g = graph();
        let x: Tensor<3, f32> = leaf(&g, &[2, 3, 4]);
        let w: Tensor<1, f32> = leaf(&g, &[4]);
        let plain = LayerNorm::new(w.clone(), None, 1e-5).forward(&x);
        let nd = LayerNormNd::new(LayerNorm::new(w, None, 1e-5), 1).forward(&x);
        assert_eq!(plain.id(), nd.id());
    }

    /// The mean is removed and the variance is biased.
    #[test]
    fn the_forward_is_the_centered_macro_op() {
        let g = graph();
        let x: Tensor<2, f32> = leaf(&g, &[3, 6]);
        let w: Tensor<1, f32> = leaf(&g, &[6]);
        let b: Tensor<1, f32> = leaf(&g, &[6]);
        let by_layer = LayerNorm::new(w.clone(), Some(b.clone()), 1e-5).forward(&x);
        let by_hand = x.layer_norm(&w, Some(&b), 1e-5, true);
        assert_eq!(by_layer.id(), by_hand.id());
        // The uncentered spelling is a different value, not the same node.
        assert_ne!(by_layer.id(), x.layer_norm(&w, Some(&b), 1e-5, false).id());
    }

    /// The tail is normalized as one group, so the `[a, b, c]` form and the
    /// `[a, b * c]` form are the same node modulo the closing reshape.
    #[test]
    fn the_nd_form_is_the_flattened_last_axis_form() {
        let g = graph();
        let x: Tensor<3, f32> = leaf(&g, &[2, 3, 4]);
        let w: Tensor<2, f32> = leaf(&g, &[3, 4]);
        let by_layer = LayerNormNd::new(LayerNorm::new(w.clone(), None, 1e-5), 2).forward(&x);
        let by_hand = x
            .flatten_last_n::<2>(1)
            .layer_norm(&w.flatten_all(), None, 1e-5, true)
            .reshape_dims(x.extents());
        assert_eq!(by_layer.id(), by_hand.id());
    }

    #[test]
    #[should_panic(expected = "covers 4 elements")]
    fn an_affine_parameter_that_does_not_cover_the_group_is_refused() {
        let g = graph();
        let x: Tensor<3, f32> = leaf(&g, &[2, 3, 4]);
        let w: Tensor<1, f32> = leaf(&g, &[4]);
        let _ = LayerNormNd::new(LayerNorm::new(w, None, 1e-5), 2).forward(&x);
    }

    #[test]
    #[should_panic(expected = "trailing axes of a rank-2")]
    fn more_trailing_axes_than_rank_is_refused() {
        let g = graph();
        let x: Tensor<2, f32> = leaf(&g, &[3, 4]);
        let w: Tensor<1, f32> = leaf(&g, &[12]);
        let _ = LayerNormNd::new(LayerNorm::new(w, None, 1e-5), 3).forward(&x);
    }
}
