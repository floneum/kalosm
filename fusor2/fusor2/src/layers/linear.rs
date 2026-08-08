//! `Linear`: `x @ Wt (+ b)`. The bias add is an epilogue the extractor fuses
//! or does not, on cost.

use fusor2_gguf::VarBuilder;
use fusor2_ir::shape::Dim;

use crate::tensor::Tensor;
use crate::{Error, Result};

pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    /// `weight` is `[out, in]` and `bias` is `[out]`, the GGUF layout.
    ///
    /// A missing `bias` entry is an error when `bias` is true: a model that
    /// declares a bias and does not ship it evaluates to a plausible but
    /// wrong function, and quietly dropping it is how that goes unnoticed.
    pub fn load(vb: &VarBuilder, graph: &crate::graph::GraphRef, bias: bool) -> Result<Self> {
        let weight = crate::layers::load_dense(vb, graph, "weight")?;
        let bias = if bias {
            let b = crate::layers::load_dense(vb, graph, "bias")?;
            Some(crate::layers::as_vector(b, "bias")?)
        } else {
            None
        };
        Ok(Self { weight, bias })
    }

    /// The extent the weight contracts over.
    pub fn in_features(&self) -> Dim {
        self.weight.dim(1)
    }

    pub fn out_features(&self) -> Dim {
        self.weight.dim(0)
    }

    /// `x @ weight^T (+ bias)`.
    ///
    /// Transposed-rhs is an `EinSpec`, not a second op, so this is one
    /// `L0::Contract` — and it is the transposed spelling specifically, so
    /// `d_weight` lands in the weight's own `[out, in]` layout rather than in
    /// a transposed view the optimizer would have to copy out of.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if self.weight.rank() != 2 {
            return Err(Error::Shape(format!(
                "a Linear weight is [out, in]; got rank {}",
                self.weight.rank()
            )));
        }
        if x.rank() < 2 {
            return Err(Error::Shape(format!(
                "Linear::forward needs rank >= 2, got rank {}",
                x.rank()
            )));
        }
        // `Contract` has no implicit batch broadcast, so a rank-N activation
        // meets the rank-2 weight through one stride-0 `Restride` over the
        // leading axes rather than through a reshape that would need x dense.
        let weight = if x.rank() == 2 {
            self.weight.clone()
        } else {
            let mut target: Vec<Dim> = x.shape()[..x.rank() - 2].to_vec();
            target.push(self.weight.dim(0));
            target.push(self.weight.dim(1));
            self.weight.broadcast_as(&target)?
        };
        let y = x.matmul_t(&weight)?;
        match &self.bias {
            Some(b) => y.add_(b),
            None => Ok(y),
        }
    }
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
    fn the_output_takes_the_weights_out_features() {
        let g = graph();
        let x = leaf(&g, &[3, 4]);
        let w = leaf(&g, &[5, 4]);
        let b = leaf(&g, &[5]);
        let y = Linear::new(w, Some(b)).forward(&x).unwrap();
        assert_eq!(&y.shape()[..], &[Dim::Const(3), Dim::Const(5)]);
    }

    #[test]
    fn a_batched_activation_meets_a_rank_two_weight() {
        let g = graph();
        let x = leaf(&g, &[2, 3, 4]);
        let w = leaf(&g, &[5, 4]);
        let y = Linear::new(w, None).forward(&x).unwrap();
        assert_eq!(
            &y.shape()[..],
            &[Dim::Const(2), Dim::Const(3), Dim::Const(5)]
        );
    }

    /// The layer owns no kernel: its forward hash-conses to exactly the
    /// composition its documentation names.
    #[test]
    fn the_forward_is_mat_mul_transposed_rhs_plus_a_broadcast_bias() {
        let g = graph();
        let x = leaf(&g, &[3, 4]);
        let w = leaf(&g, &[5, 4]);
        let b = leaf(&g, &[5]);
        let by_layer = Linear::new(w.clone(), Some(b.clone())).forward(&x).unwrap();
        let by_hand = x.mat_mul_transposed_rhs(&w).unwrap().add_(&b).unwrap();
        assert_eq!(by_layer.id(), by_hand.id());

        let no_bias = Linear::new(w.clone(), None).forward(&x).unwrap();
        assert_eq!(no_bias.id(), x.mat_mul_transposed_rhs(&w).unwrap().id());
    }

    #[test]
    fn a_disagreeing_inner_extent_is_refused() {
        let g = graph();
        let x = leaf(&g, &[3, 4]);
        let w = leaf(&g, &[5, 6]);
        assert!(Linear::new(w, None).forward(&x).is_err());
    }
}
