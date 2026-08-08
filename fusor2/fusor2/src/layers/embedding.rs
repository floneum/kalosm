//! `Embedding`: a `Gather` whose adjoint is a `Scatter{Add}` with four
//! coexisting lowerings.

use fusor2_gguf::VarBuilder;
use fusor2_ir::shape::Dim;

use crate::tensor::Tensor;
use crate::{Error, Result};

pub struct Embedding {
    pub table: Tensor,
}

impl Embedding {
    pub fn new(table: Tensor) -> Self {
        Self { table }
    }

    /// The GGUF `weight` entry, `[num_embeddings, embedding_dim]`.
    pub fn load(vb: &VarBuilder, graph: &crate::graph::GraphRef) -> Result<Self> {
        let table = crate::layers::load_dense(vb, graph, "weight")?;
        if table.rank() != 2 {
            return Err(Error::Shape(format!(
                "an embedding table is [num_embeddings, embedding_dim]; got rank {}",
                table.rank()
            )));
        }
        Ok(Self { table })
    }

    pub fn num_embeddings(&self) -> Dim {
        self.table.dim(0)
    }

    pub fn embedding_dim(&self) -> Dim {
        self.table.dim(1)
    }

    /// `[..ids] -> [..ids, embedding_dim]`.
    ///
    /// One `L0::Gather` over the flattened index run, reshaped back. `Gather`'s
    /// declared adjoint is a `Scatter{Add}`, so a token appearing twice
    /// accumulates.
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        if self.table.rank() != 2 {
            return Err(Error::Shape(format!(
                "an embedding table is [num_embeddings, embedding_dim]; got rank {}",
                self.table.rank()
            )));
        }
        if ids.rank() == 0 {
            return Err(Error::Shape(
                "an embedding lookup needs at least a rank-1 index".into(),
            ));
        }
        self.table.embedding(ids)
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

    fn leaf(g: &Graph, shape: &[u64], dtype: Dtype) -> Tensor {
        let dims: Vec<Dim> = shape.iter().map(|d| Dim::Const(*d)).collect();
        g.leaf("t", &dims, dtype).unwrap()
    }

    #[test]
    fn the_index_rank_grows_by_the_embedding_axis() {
        let g = graph();
        let table = leaf(&g, &[5, 3], Dtype::F32);
        let ids = leaf(&g, &[2, 2], Dtype::U32);
        let y = Embedding::new(table).forward(&ids).unwrap();
        assert_eq!(
            &y.shape()[..],
            &[Dim::Const(2), Dim::Const(2), Dim::Const(3)]
        );
    }

    /// The layer mints one `Gather` and nothing else.
    #[test]
    fn the_forward_is_exactly_the_gather() {
        let g = graph();
        let table = leaf(&g, &[5, 3], Dtype::F32);
        let ids = leaf(&g, &[2, 2], Dtype::U32);
        let by_layer = Embedding::new(table.clone()).forward(&ids).unwrap();
        assert_eq!(by_layer.id(), table.embedding(&ids).unwrap().id());
    }

    #[test]
    fn a_rank_three_table_is_refused_rather_than_flattened() {
        let g = graph();
        let table = leaf(&g, &[2, 5, 3], Dtype::F32);
        let ids = leaf(&g, &[4], Dtype::U32);
        assert!(Embedding::new(table).forward(&ids).is_err());
    }
}
