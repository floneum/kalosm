//! `Linear`: `x @ Wt (+ b)`.

use fusor2_gguf::VarBuilder;
use fusor2_ir::shape::Dim;

use crate::device::ok;
use crate::tensor::Dyn;
use crate::tensor::typed::Element;
use crate::{Error, Result, Tensor};

/// `x @ Wt (+ b)`.
///
/// The weight is `[out, in]`; the activation's rank is [`Linear::forward`]'s,
/// so one loaded layer serves a rank-2 and a rank-3 call in the same model.
pub struct Linear<T: Element = f32> {
    pub weight: Tensor<2, T>,
    pub bias: Option<Tensor<1, T>>,
}

impl<T: Element> Linear<T> {
    pub fn new(weight: Tensor<2, T>, bias: Option<Tensor<1, T>>) -> Self {
        Self { weight, bias }
    }

    /// `weight` is `[out, in]` and `bias` is `[out]`, the GGUF layout.
    ///
    /// A missing `bias` entry is an error when `bias` is true.
    pub fn load(vb: &VarBuilder, graph: &crate::graph::GraphRef, bias: bool) -> Result<Self> {
        let w = crate::layers::load_dense(vb, graph, "weight")?;
        let weight = crate::layers::as_typed::<2, T>(w, "a Linear weight is [out, in]")?;
        let bias = if bias {
            let b = crate::layers::load_dense(vb, graph, "bias")?;
            let b = crate::layers::as_vector(b, "bias")?;
            Some(crate::layers::as_typed::<1, T>(b, "bias")?)
        } else {
            None
        };
        Ok(Self { weight, bias })
    }

    /// The extent the weight contracts over.
    pub fn in_features(&self) -> Dim {
        self.weight.extent(1usize)
    }

    pub fn out_features(&self) -> Dim {
        self.weight.extent(0usize)
    }

    /// `x @ weight^T (+ bias)`. Rank-preserving.
    #[track_caller]
    pub fn forward<const R: usize>(&self, x: &Tensor<R, T>) -> Tensor<R, T> {
        Tensor::<R, T>::from_dyn(ok(
            "Linear::forward",
            forward_dyn(
                self.weight.as_dyn(),
                self.bias.as_ref().map(Tensor::as_dyn),
                x.as_dyn(),
            ),
        ))
    }
}

/// The runtime-rank forward.
pub(crate) fn forward_dyn(weight: &Dyn, bias: Option<&Dyn>, x: &Dyn) -> Result<Dyn> {
    if weight.rank() != 2 {
        return Err(Error::Shape(format!(
            "a Linear weight is [out, in]; got rank {}",
            weight.rank()
        )));
    }
    if x.rank() < 2 {
        return Err(Error::Shape(format!(
            "Linear::forward needs rank >= 2, got rank {}",
            x.rank()
        )));
    }
    // `Contract` has no implicit batch broadcast, so the weight is
    // broadcast over the leading axes.
    let weight = if x.rank() == 2 {
        weight.clone()
    } else {
        let mut target: Vec<Dim> = x.shape()[..x.rank() - 2].to_vec();
        target.push(weight.dim(0));
        target.push(weight.dim(1));
        weight.broadcast_as(&target)?
    };
    let y = x.matmul_t(&weight)?;
    match bias {
        Some(b) => y.add_(b),
        None => Ok(y),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fusor2_ir::dtype::Dtype;

    use crate::graph::Graph;
    use crate::session::{Backend, Session};

    fn graph() -> Graph {
        Graph::new(&Session::new(Backend::cpu().expect("cpu device")).expect("session"))
    }

    fn leaf(g: &Graph, shape: &[u64]) -> Dyn {
        let dims: Vec<Dim> = shape.iter().map(|d| Dim::Const(*d)).collect();
        g.leaf("t", &dims, Dtype::F32).unwrap()
    }

    #[test]
    fn the_output_takes_the_weights_out_features() {
        let g = graph();
        let x = leaf(&g, &[3, 4]);
        let w = leaf(&g, &[5, 4]);
        let b = leaf(&g, &[5]);
        let y = forward_dyn(&w, Some(&b), &x).unwrap();
        assert_eq!(&y.shape()[..], &[Dim::Const(3), Dim::Const(5)]);
    }

    #[test]
    fn a_batched_activation_meets_a_rank_two_weight() {
        let g = graph();
        let x = leaf(&g, &[2, 3, 4]);
        let w = leaf(&g, &[5, 4]);
        let y = forward_dyn(&w, None, &x).unwrap();
        assert_eq!(
            &y.shape()[..],
            &[Dim::Const(2), Dim::Const(3), Dim::Const(5)]
        );
    }

    /// The forward must hash-cons to exactly the composition it documents.
    #[test]
    fn the_forward_is_mat_mul_transposed_rhs_plus_a_broadcast_bias() {
        let g = graph();
        let x = leaf(&g, &[3, 4]);
        let w = leaf(&g, &[5, 4]);
        let b = leaf(&g, &[5]);
        let by_layer = forward_dyn(&w, Some(&b), &x).unwrap();
        let by_hand = x.matmul_t(&w).unwrap().add_(&b).unwrap();
        assert_eq!(by_layer.id(), by_hand.id());

        let no_bias = forward_dyn(&w, None, &x).unwrap();
        assert_eq!(no_bias.id(), x.matmul_t(&w).unwrap().id());
    }

    #[test]
    fn a_disagreeing_inner_extent_is_refused() {
        let g = graph();
        let x = leaf(&g, &[3, 4]);
        let w = leaf(&g, &[5, 6]);
        assert!(forward_dyn(&w, None, &x).is_err());
    }
}
