//! `RmsNorm`. Its fused form is what `fold_split` + `map_into_fold` produce;
//! there is no fused kernel to select.
//!
//! Owned by W13.

use fusor2_gguf::VarBuilder;

use crate::tensor::Tensor;
use crate::{Error, Result};

pub struct RmsNorm {
    pub weight: Option<Tensor>,
    pub eps: f32,
}

impl RmsNorm {
    pub fn new(weight: Option<Tensor>, eps: f32) -> Self {
        Self { weight, eps }
    }

    /// The GGUF `weight` entry. Required, as in the reference: an RMS norm
    /// whose scale silently defaulted to one is indistinguishable from a
    /// correctly loaded one until the outputs are compared.
    ///
    /// `RmsNorm::new(None, eps)` is the way to ask for the unweighted form.
    pub fn load(vb: &VarBuilder, graph: &crate::graph::GraphRef, eps: f32) -> Result<Self> {
        let w = crate::layers::load_dense(vb, graph, "weight")?;
        let weight = crate::layers::as_vector(w, "weight")?;
        Ok(Self {
            weight: Some(weight),
            eps,
        })
    }

    /// `x / sqrt(mean(x^2) + eps) * weight` over the last axis.
    ///
    /// `eps` enters as a **uniform**, so two layers at one epsilon share a
    /// symbol and changing it recompiles nothing.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match &self.weight {
            Some(w) => x.rms_norm(w, self.eps),
            None => x.rms_norm_no_weight(self.eps),
        }
    }

    /// The transformer block boundary: `rms_norm(x + residual)`, as one macro
    /// op so the add is inside the normalization's launch rather than before
    /// it.
    pub fn forward_residual(&self, x: &Tensor, residual: &Tensor) -> Result<Tensor> {
        let w = self
            .weight
            .as_ref()
            .ok_or_else(|| Error::Shape("the fused residual RmsNorm needs a weight".into()))?;
        x.rms_norm_residual_fused(residual, w, None, self.eps)
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
    fn both_spellings_preserve_the_shape() {
        let g = graph();
        let x = leaf(&g, &[3, 6]);
        let w = leaf(&g, &[6]);
        for layer in [RmsNorm::new(Some(w), 1e-5), RmsNorm::new(None, 1e-5)] {
            let y = layer.forward(&x).unwrap();
            assert_eq!(&y.shape()[..], &[Dim::Const(3), Dim::Const(6)]);
        }
    }

    #[test]
    fn one_epsilon_is_one_uniform_across_two_layers() {
        let g = graph();
        let x = leaf(&g, &[3, 6]);
        let w = leaf(&g, &[6]);
        let a = RmsNorm::new(Some(w.clone()), 1e-5).forward(&x).unwrap();
        let b = RmsNorm::new(Some(w), 1e-5).forward(&x).unwrap();
        assert_eq!(a.id(), b.id(), "two layers at one eps hash-cons together");
    }

    #[test]
    fn the_forward_is_exactly_the_macro_op() {
        let g = graph();
        let x = leaf(&g, &[3, 6]);
        let w = leaf(&g, &[6]);
        let weighted = RmsNorm::new(Some(w.clone()), 1e-5).forward(&x).unwrap();
        assert_eq!(weighted.id(), x.rms_norm(&w, 1e-5).unwrap().id());
        let bare = RmsNorm::new(None, 1e-5).forward(&x).unwrap();
        assert_eq!(bare.id(), x.rms_norm_no_weight(1e-5).unwrap().id());
    }

    #[test]
    fn a_weightless_residual_norm_is_refused_rather_than_defaulted() {
        let g = graph();
        let x = leaf(&g, &[2, 3, 6]);
        let r = leaf(&g, &[2, 3, 6]);
        assert!(RmsNorm::new(None, 1e-5).forward_residual(&x, &r).is_err());
    }
}
