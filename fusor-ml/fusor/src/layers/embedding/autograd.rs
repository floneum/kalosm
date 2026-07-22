//! Trainable embedding layer implementation.

use crate::Tensor as RawTensor;
use crate::autograd::{AutogradElement, Graph, Tensor};

/// Embedding layer for token/position embeddings.
///
/// Maps integer indices to dense vectors.
/// Embedding table shape: (num_embeddings, embedding_dim)
#[derive(Clone)]
pub struct Embedding<T: AutogradElement = f32> {
    embeddings: Tensor<2, T>,
    num_embeddings: usize,
    embedding_dim: usize,
}

impl<T: AutogradElement> Embedding<T>
where
    crate::cpu::AddOp: crate::cpu::SimdBinaryOp<T>,
{
    /// Create a new embedding layer with the given embedding table.
    pub fn new_from_tensor(embeddings: Tensor<2, T>) -> Self {
        let shape = embeddings.shape();
        let num_embeddings = shape[0];
        let embedding_dim = shape[1];

        Self {
            embeddings,
            num_embeddings,
            embedding_dim,
        }
    }

    /// Get the embedding table.
    pub fn embeddings(&self) -> &Tensor<2, T> {
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

impl Embedding {
    /// Import an inference embedding layer as a trainable layer whose
    /// embedding table is a gradient leaf on `graph`.
    pub fn from_inference(graph: &Graph, layer: &crate::layers::Embedding<f32>) -> Self {
        let table = match layer.dense_embeddings() {
            Some(dense) => dense.clone(),
            None => layer.embeddings_quantized().dequantize().into_concrete(),
        };
        Self::new_from_tensor(graph.leaf(table))
    }
}

impl<T: AutogradElement> Embedding<T>
where
    crate::cpu::AddOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::SubOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::MulOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::DivOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::EqOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::NeOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::LtOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::LteOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::GtOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::GteOp: crate::cpu::SimdBinaryOp<T>,
    crate::cpu::SumOp: crate::cpu::SimdReduceOp<T>,
    crate::cpu::MaxOp: crate::cpu::SimdReduceOp<T>,
    crate::cpu::MinOp: crate::cpu::SimdReduceOp<T>,
    crate::cpu::NegOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AbsOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::SqrtOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::ExpOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::Exp2Op: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::LogOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::Log2Op: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::SinOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::CosOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::TanOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::TanhOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::SinhOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::CoshOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AsinOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AcosOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AtanOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AsinhOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AcoshOp: crate::cpu::SimdUnaryOp<T>,
    crate::cpu::AtanhOp: crate::cpu::SimdUnaryOp<T>,
    u32: crate::CastTo<T> + crate::CastTensor<T>,
{
    /// Looks up embeddings for an index tensor, appending the embedding dimension.
    pub fn forward<const N: usize, const M: usize>(&self, indices: &RawTensor<N, u32>) -> Tensor<M, T>
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
}
