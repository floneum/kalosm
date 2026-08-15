//! `RmsNorm`. Its fused form is what `fold_split` + `map_into_fold` produce;
//! there is no fused kernel to select.

use fusor2_gguf::VarBuilder;

use crate::device::ok;
use crate::tensor::typed::Element;
use crate::{Error, Result, Tensor};

/// `x / sqrt(mean(x^2) + eps) * weight`.
///
/// `N` is the **weight's** rank and `T` its element type, matching the
/// reference's `RmsNorm<const N: usize, T: SimdElement>`. It defaults to a
/// rank-1 f32 scale, which is what every GGUF checkpoint ships, so `RmsNorm`
/// alone still names the common case. The activation's rank is
/// [`RmsNorm::forward`]'s own parameter.
pub struct RmsNorm<const N: usize = 1, T: Element = f32> {
    pub weight: Option<Tensor<N, T>>,
    pub eps: f32,
}

impl<const N: usize, T: Element> RmsNorm<N, T> {
    pub fn new(weight: Option<Tensor<N, T>>, eps: f32) -> Self {
        Self { weight, eps }
    }

    /// The GGUF `weight` entry. Required, as in the reference: an RMS norm
    /// whose scale silently defaulted to one is indistinguishable from a
    /// correctly loaded one until the outputs are compared.
    ///
    /// `RmsNorm::new(None, eps)` is the way to ask for the unweighted form.
    pub fn load(vb: &VarBuilder, graph: &crate::graph::GraphRef, eps: f32) -> Result<Self> {
        let w = crate::layers::load_dense(vb, graph, "weight")?;
        let w = crate::layers::as_vector(w, "weight")?;
        let weight = crate::layers::as_typed::<N, T>(w, "an RmsNorm weight")?;
        Ok(Self {
            weight: Some(weight),
            eps,
        })
    }

    /// `x / sqrt(mean(x^2) + eps) * weight` over the last axis.
    /// Rank-preserving.
    ///
    /// `eps` enters as a **uniform**, so two layers at one epsilon share a
    /// symbol and changing it recompiles nothing.
    #[track_caller]
    pub fn forward<const R: usize>(&self, x: &Tensor<R, T>) -> Tensor<R, T> {
        match &self.weight {
            Some(w) => x.rms_norm(w, self.eps),
            None => x.rms_norm_no_weight(self.eps),
        }
    }

    /// The transformer block boundary: `rms_norm(x + residual)`, as one macro
    /// op so the add is inside the normalization's launch rather than before
    /// it.
    #[track_caller]
    pub fn forward_residual<const R: usize>(
        &self,
        x: &Tensor<R, T>,
        residual: &Tensor<R, T>,
    ) -> Tensor<R, T> {
        let w = match &self.weight {
            Some(w) => w,
            None => ok(
                "RmsNorm::forward_residual",
                Err(Error::Shape(
                    "the fused residual RmsNorm needs a weight".into(),
                )),
            ),
        };
        x.rms_norm_residual(residual, w, None, self.eps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fusor2_ir::shape::Dim;

    use crate::graph::Graph;
    use crate::layers::test_leaf as leaf;
    use crate::session::{Backend, Session};

    fn graph() -> Graph {
        Graph::new(&Session::new(Backend::cpu().expect("cpu device")).expect("session"))
    }

    #[test]
    fn both_spellings_preserve_the_shape() {
        let g = graph();
        let x: Tensor<2, f32> = leaf(&g, &[3, 6]);
        let w: Tensor<1, f32> = leaf(&g, &[6]);
        for layer in [RmsNorm::new(Some(w), 1e-5), RmsNorm::new(None, 1e-5)] {
            assert_eq!(layer.forward(&x).shape(), [3, 6]);
        }
    }

    #[test]
    fn one_epsilon_is_one_uniform_across_two_layers() {
        let g = graph();
        let x: Tensor<2, f32> = leaf(&g, &[3, 6]);
        let w: Tensor<1, f32> = leaf(&g, &[6]);
        let a = RmsNorm::new(Some(w.clone()), 1e-5).forward(&x);
        let b = RmsNorm::new(Some(w), 1e-5).forward(&x);
        assert_eq!(a.id(), b.id(), "two layers at one eps hash-cons together");
    }

    #[test]
    fn the_forward_is_exactly_the_macro_op() {
        let g = graph();
        let x: Tensor<2, f32> = leaf(&g, &[3, 6]);
        let w: Tensor<1, f32> = leaf(&g, &[6]);
        let weighted = RmsNorm::new(Some(w.clone()), 1e-5).forward(&x);
        assert_eq!(weighted.id(), x.rms_norm(&w, 1e-5).id());
        let bare = RmsNorm::<1, f32>::new(None, 1e-5).forward(&x);
        assert_eq!(bare.id(), x.rms_norm_no_weight(1e-5).id());
    }

    #[test]
    #[should_panic(expected = "needs a weight")]
    fn a_weightless_residual_norm_is_refused_rather_than_defaulted() {
        let g = graph();
        let x: Tensor<3, f32> = leaf(&g, &[2, 3, 6]);
        let r: Tensor<3, f32> = leaf(&g, &[2, 3, 6]);
        let _ = RmsNorm::<1, f32>::new(None, 1e-5).forward_residual(&x, &r);
    }

    /// The element type is a parameter, and a non-f32 layer keeps its dtype
    /// through the forward.
    #[test]
    fn a_half_precision_layer_stays_half_precision() {
        let g = graph();
        let x: Tensor<2, half::f16> = leaf(&g, &[3, 6]);
        let w: Tensor<1, half::f16> = leaf(&g, &[6]);
        let y = RmsNorm::new(Some(w), 1e-5).forward(&x);
        assert_eq!(y.dtype(), fusor2_ir::dtype::Dtype::F16);
        let _ = Dim::Const(0);
    }
}
