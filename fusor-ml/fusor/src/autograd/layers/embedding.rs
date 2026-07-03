//! Trainable embedding layer implementation.

use super::super::{Graph, RawTensor, Tensor};

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
            None => layer.embeddings_quantized().dequantize().to_concrete(),
        };
        Self::new_from_tensor(graph.leaf(table))
    }

    /// Forward pass: lookup embeddings for the given indices.
    ///
    /// Input: [batch, seq_len] with indices
    /// Output: [batch, seq_len, embedding_dim] with embeddings
    pub fn forward(&self, indices: &RawTensor<2, u32>) -> Tensor<3> {
        self.embeddings.embedding(indices)
    }

    /// Forward pass: lookup embeddings for flat indices.
    ///
    /// Input: [seq_len] with indices
    /// Output: [seq_len, embedding_dim] with embeddings
    pub fn forward_1d(&self, indices: &RawTensor<1, u32>) -> Tensor<2> {
        self.embeddings.index_select(0, indices)
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
