//! `Embedding`: a `Gather` whose adjoint is a `Scatter{Add}` with four
//! coexisting lowerings. No hand-written backward.

use fusor2_gguf::VarBuilder;
use fusor2_ir::shape::Dim;

use crate::tensor::typed::Element;
use crate::{Error, Result, Tensor};

/// A `[num_embeddings, embedding_dim]` lookup table.
pub struct Embedding<T: Element = f32> {
    pub table: Tensor<2, T>,
}

impl<T: Element> Embedding<T> {
    pub fn new(table: Tensor<2, T>) -> Self {
        Self { table }
    }

    /// The GGUF `weight` entry, `[num_embeddings, embedding_dim]`.
    pub fn load(vb: &VarBuilder, graph: &crate::graph::GraphRef) -> Result<Self> {
        let table = crate::layers::load_dense(vb, graph, "weight")?;
        let table = crate::layers::as_typed::<2, T>(
            table,
            "an embedding table is [num_embeddings, embedding_dim]",
        )?;
        Ok(Self { table })
    }

    pub fn num_embeddings(&self) -> Dim {
        self.table.extent(0usize)
    }

    pub fn embedding_dim(&self) -> Dim {
        self.table.extent(1usize)
    }

    /// `[..ids] -> [..ids, embedding_dim]`, so the output rank is `O = R + 1`.
    ///
    /// One `Logical::Gather` over the flattened index run, reshaped back.
    /// `Gather`'s declared adjoint is a `Scatter{Add}`, so a token appearing
    /// twice accumulates.
    #[track_caller]
    pub fn forward<const R: usize, const O: usize>(&self, ids: &Tensor<R, u32>) -> Tensor<O, T> {
        if R == 0 {
            crate::device::ok::<()>(
                "Embedding::forward",
                Err(Error::Shape(
                    "an embedding lookup needs at least a rank-1 index".into(),
                )),
            );
        }
        self.table.embedding(ids)
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
    fn the_index_rank_grows_by_the_embedding_axis() {
        let g = graph();
        let table: Tensor<2, f32> = leaf(&g, &[5, 3]);
        let ids: Tensor<2, u32> = leaf(&g, &[2, 2]);
        let y: Tensor<3, f32> = Embedding::new(table).forward(&ids);
        assert_eq!(y.shape(), [2, 2, 3]);
    }

    /// The layer must not mint a second node the adjoint would then have to
    /// know about.
    #[test]
    fn the_forward_is_exactly_the_gather() {
        let g = graph();
        let table: Tensor<2, f32> = leaf(&g, &[5, 3]);
        let ids: Tensor<2, u32> = leaf(&g, &[2, 2]);
        let by_layer: Tensor<3, f32> = Embedding::new(table.clone()).forward(&ids);
        assert_eq!(by_layer.id(), table.embedding::<2, 3>(&ids).id());
    }

    /// The table's rank is in the type, so a rank-3 one cannot reach the
    /// constructor.
    #[test]
    #[should_panic(expected = "value has rank 3")]
    fn a_rank_three_table_is_refused_by_the_type() {
        let g = graph();
        let _: Tensor<2, f32> = leaf(&g, &[2, 5, 3]);
    }

    #[test]
    fn a_half_precision_table_stays_half_precision() {
        let g = graph();
        let table: Tensor<2, half::f16> = leaf(&g, &[5, 3]);
        let ids: Tensor<1, u32> = leaf(&g, &[2]);
        let y: Tensor<2, half::f16> = Embedding::new(table).forward(&ids);
        assert_eq!(y.dtype(), fusor2_ir::dtype::Dtype::F16);
    }
}
