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
    weight: Option<Tensor<N, T>>,
    eps: f32,
}

impl<const N: usize, T: Element> RmsNorm<N, T> {
    /// Create an RMS normalization with an optional learned scale.
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
