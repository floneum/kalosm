//! Trainable embedding layer implementation.

use crate::Tensor as RawTensor;
use crate::autograd::{Graph, Tensor};

/// Embedding layer for token/position embeddings.
///
/// Maps integer indices to dense vectors.
/// Embedding table shape: (num_embeddings, embedding_dim)
#[derive(Clone)]
pub struct Embedding {
    embeddings: Tensor<2>,
    num_embeddings: usize,
    embedding_dim: usize,
}

impl Embedding {
    /// Create a new embedding layer with the given embedding table.
    pub fn new_from_tensor(embeddings: Tensor<2>) -> Self {
        let shape = embeddings.shape();
        let num_embeddings = shape[0];
        let embedding_dim = shape[1];

        Self {
            embeddings,
            num_embeddings,
            embedding_dim,
        }
    }

    /// Import an inference embedding layer as a trainable layer whose
    /// embedding table is a gradient leaf on `graph`.
    pub fn from_inference(graph: &Graph, layer: &crate::layers::Embedding<f32>) -> Self {
        let table = match layer.dense_embeddings() {
            Some(dense) => dense.clone(),
            None => layer.embeddings_quantized().dequantize().into_concrete(),
        };
        Self::new_from_tensor(graph.leaf(table))
    }

    /// Looks up embeddings for an index tensor, appending the embedding dimension.
    pub fn forward<const N: usize, const M: usize>(&self, indices: &RawTensor<N, u32>) -> Tensor<M>
    where
        crate::gpu::Tensor<N, u32>: crate::gpu::NextRank<M, u32>,
    {
        assert_eq!(M, N + 1, "embedding output rank must be input rank + 1");
        let input_shape = indices.shape();
        let output_shape = std::array::from_fn(|axis| {
            if axis < N {
                input_shape[axis]
            } else {
                self.embedding_dim
            }
        });
        let indices = indices.flatten_all().to_concrete();
        self.embeddings
            .index_select(0, &indices)
            .reshape(output_shape)
    }

    /// Get the embedding table.
    pub fn embeddings(&self) -> &Tensor<2> {
        &self.embeddings
    }

    /// Get the number of embeddings.
    pub fn num_embeddings(&self) -> usize {
        self.num_embeddings
    }

    /// Get the embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}
