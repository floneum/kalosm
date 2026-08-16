//! `Embedding`: a `Gather` whose adjoint is a `Scatter{Add}` with four
//! coexisting lowerings. No hand-written backward.

use fusor2_gguf::VarBuilder;
use fusor2_ir::shape::Dim;

use crate::tensor::typed::Element;
use crate::{Error, Result, Tensor};

/// A `[num_embeddings, embedding_dim]` lookup table.
pub struct Embedding<T: Element = f32> {
    /// The `[num_embeddings, embedding_dim]` lookup table.
    pub table: Tensor<2, T>,
}

impl<T: Element> Embedding<T> {
    /// Wrap a lookup table.
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

    /// The number of lookup rows.
    pub fn num_embeddings(&self) -> Dim {
        self.table.extent(0usize)
    }

    /// The width of each lookup row.
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
