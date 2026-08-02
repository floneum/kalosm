//! `LayerNorm` over the last axis and `LayerNormNd` over a trailing group.
//!
//! Owned by W13.

use fusor2_gguf::VarBuilder;
use fusor2_ir::shape::Dim;

use crate::tensor::Tensor;
use crate::{Error, Result};

pub struct LayerNorm {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(weight: Tensor, bias: Option<Tensor>, eps: f32) -> Self {
        Self { weight, bias, eps }
    }

    /// `weight` is required and `bias` is optional, matching the reference's
    /// `load_vector_f32` / `load_optional_vector_f32` pair. Both are squeezed
    /// to rank 1 when the file writes them as `[1, n]`.
    pub fn load(vb: &VarBuilder, graph: &crate::graph::GraphRef, eps: f32) -> Result<Self> {
        let w = crate::layers::load_dense(vb, graph, "weight")?;
        let weight = crate::layers::as_vector(w, "weight")?;
        let bias = match crate::layers::load_optional(vb, graph, "bias")? {
            Some(b) => Some(crate::layers::as_vector(b, "bias")?),
            None => None,
        };
        Ok(Self { weight, bias, eps })
    }

    /// `(x - mean) / sqrt(var + eps) * weight + bias` over the last axis, with
    /// the **biased** variance the reference uses — the divisor is the axis
    /// extent, not `n - 1`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.layer_norm(&self.weight, self.bias.as_ref(), self.eps, true)
    }
}

/// Normalizes over the trailing `axes` rather than just the last one.
pub struct LayerNormNd {
    pub inner: LayerNorm,
    pub axes: u32,
}

impl LayerNormNd {
    pub fn new(inner: LayerNorm, axes: u32) -> Self {
        Self { inner, axes }
    }

    /// The statistic is taken over the **flattened tail**, not per trailing
    /// axis: `[a, b, c]` with `axes == 2` has one mean and one variance per
    /// `a`, over all `b * c` elements.
    ///
    /// So this is the last-axis path over a reshaped view, and the affine
    /// parameters flatten with it. No second normalization kernel exists and
    /// none is wanted — the reshape is a `Restride`, which costs nothing.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let axes = self.axes as usize;
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
            return self.inner.forward(x);
        }

        let shape = x.shape();
        let flat = x.flatten_last_n(axes - 1)?;
        let tail = flat.dim(flat.rank() - 1);
        let weight = flatten_affine(&self.inner.weight, tail, "weight")?;
        let bias = match &self.inner.bias {
            Some(b) => Some(flatten_affine(b, tail, "bias")?),
            None => None,
        };
        let y = flat.layer_norm(&weight, bias.as_ref(), self.inner.eps, true)?;
        y.reshape_dims(&shape)
    }
}

/// An affine parameter shaped like the group being normalized, flattened to
/// the one axis the last-axis path scales.
fn flatten_affine(p: &Tensor, tail: Dim, what: &str) -> Result<Tensor> {
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
    use fusor2_ir::dtype::Dtype;

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
    fn the_last_axis_form_preserves_the_shape() {
        let g = graph();
        let x = leaf(&g, &[3, 6]);
        let w = leaf(&g, &[6]);
        let b = leaf(&g, &[6]);
        let y = LayerNorm::new(w, Some(b), 1e-5).forward(&x).unwrap();
        assert_eq!(&y.shape()[..], &[Dim::Const(3), Dim::Const(6)]);
    }

    #[test]
    fn the_nd_form_normalizes_the_flattened_tail_and_reshapes_back() {
        let g = graph();
        let x = leaf(&g, &[2, 3, 4]);
        let w = leaf(&g, &[3, 4]);
        let layer = LayerNormNd::new(LayerNorm::new(w, None, 1e-5), 2);
        let y = layer.forward(&x).unwrap();
        assert_eq!(
            &y.shape()[..],
            &[Dim::Const(2), Dim::Const(3), Dim::Const(4)]
        );
    }

    #[test]
    fn one_trailing_axis_is_exactly_the_last_axis_form() {
        let g = graph();
        let x = leaf(&g, &[2, 3, 4]);
        let w = leaf(&g, &[4]);
        let plain = LayerNorm::new(w.clone(), None, 1e-5).forward(&x).unwrap();
        let nd = LayerNormNd::new(LayerNorm::new(w, None, 1e-5), 1)
            .forward(&x)
            .unwrap();
        assert_eq!(plain.id(), nd.id());
    }

    /// The mean is removed and the variance is biased, which is what the
    /// reference's `layer_norm(.., true)` means.
    #[test]
    fn the_forward_is_the_centered_macro_op() {
        let g = graph();
        let x = leaf(&g, &[3, 6]);
        let w = leaf(&g, &[6]);
        let b = leaf(&g, &[6]);
        let by_layer = LayerNorm::new(w.clone(), Some(b.clone()), 1e-5)
            .forward(&x)
            .unwrap();
        let by_hand = x.layer_norm(&w, Some(&b), 1e-5, true).unwrap();
        assert_eq!(by_layer.id(), by_hand.id());
        // The uncentered spelling is a different value, not the same node.
        assert_ne!(
            by_layer.id(),
            x.layer_norm(&w, Some(&b), 1e-5, false).unwrap().id()
        );
    }

    /// The tail is normalized as one group, so the `[a, b, c]` form and the
    /// `[a, b * c]` form are the same node modulo the closing reshape.
    #[test]
    fn the_nd_form_is_the_flattened_last_axis_form() {
        let g = graph();
        let x = leaf(&g, &[2, 3, 4]);
        let w = leaf(&g, &[3, 4]);
        let by_layer = LayerNormNd::new(LayerNorm::new(w.clone(), None, 1e-5), 2)
            .forward(&x)
            .unwrap();
        let by_hand = x
            .flatten_last_n(1)
            .unwrap()
            .layer_norm(&w.flatten_all().unwrap(), None, 1e-5, true)
            .unwrap()
            .reshape_dims(&x.shape())
            .unwrap();
        assert_eq!(by_layer.id(), by_hand.id());
    }

    #[test]
    fn an_affine_parameter_that_does_not_cover_the_group_is_refused() {
        let g = graph();
        let x = leaf(&g, &[2, 3, 4]);
        let w = leaf(&g, &[4]);
        let layer = LayerNormNd::new(LayerNorm::new(w, None, 1e-5), 2);
        assert!(layer.forward(&x).is_err());
    }

    #[test]
    fn more_trailing_axes_than_rank_is_refused() {
        let g = graph();
        let x = leaf(&g, &[3, 4]);
        let w = leaf(&g, &[12]);
        let layer = LayerNormNd::new(LayerNorm::new(w, None, 1e-5), 3);
        assert!(layer.forward(&x).is_err());
    }
}
