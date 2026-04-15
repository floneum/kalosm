//! Embedding layer implementation.

use crate::{CastTensor, CastTo, DataType, Device, QMatrix, SimdElement, Tensor, VarBuilder};

/// Embedding layer for token/position embeddings.
///
/// Maps integer indices to dense vectors.
/// Embedding table shape: (num_embeddings, embedding_dim)
#[derive(Clone)]
pub struct Embedding<T: SimdElement> {
    embeddings_quantized: Option<QMatrix>,
    embeddings: Tensor<2, T>,
    num_embeddings: usize,
    embedding_dim: usize,
}

impl<T: DataType + SimdElement + Default> Embedding<T> {
    /// Create a new embedding layer with the given embedding table (no quantization).
    pub fn new_from_tensor(embeddings: Tensor<2, T>) -> Self {
        let shape = embeddings.shape();
        let num_embeddings = shape[0];
        let embedding_dim = shape[1];

        Self {
            embeddings_quantized: None,
            embeddings,
            num_embeddings,
            embedding_dim,
        }
    }

    /// Forward pass: lookup embeddings for the given indices.
    ///
    /// Input: indices tensor of rank N
    /// Output: embeddings tensor of rank M = N + 1
    ///
    /// Example:
    /// - Input: [batch, seq_len] with indices
    /// - Output: [batch, seq_len, embedding_dim] with embeddings
    pub fn forward<const N: usize, const M: usize, B>(
        &self,
        indices: &Tensor<N, u32, B>,
    ) -> Tensor<M, T>
    where
        B: fusor_cpu::TensorBacking<N, Elem = u32>,
        fusor_core::Tensor<N, u32>: fusor_core::NextRank<M, u32>,
    {
        // Calculate final output dimensions: input_dims + [embedding_dim]
        let input_shape = indices.shape();
        let final_dims: [usize; M] = std::array::from_fn(|i| {
            if i < N {
                input_shape[i]
            } else {
                self.embedding_dim
            }
        });

        match (indices, &self.embeddings) {
            (Tensor::Cpu(cpu_indices), Tensor::Cpu(cpu_embeddings)) => {
                // CPU path
                let indices_flat = cpu_indices.as_ref().flatten_all();
                let values = cpu_embeddings.as_ref().index_select(0, indices_flat);
                Tensor::Cpu(values.reshape(final_dims).to_concrete())
            }
            (Tensor::Gpu(gpu_indices), Tensor::Gpu(gpu_embeddings)) => {
                // GPU path
                let indices_flat = gpu_indices.flatten_all();
                let values = gpu_embeddings.index_select(0, &indices_flat);
                Tensor::Gpu(values.reshape(final_dims))
            }
            _ => panic!("Indices and embeddings must be on the same device"),
        }
    }

    /// Get the dequantized embedding table.
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

    /// Cast the Embedding layer to a different data type
    pub fn cast<U: DataType + SimdElement + Default>(self) -> Embedding<U>
    where
        T: CastTensor<U> + CastTo<U>,
    {
        Embedding {
            embeddings_quantized: self.embeddings_quantized,
            embeddings: self.embeddings.cast(),
            num_embeddings: self.num_embeddings,
            embedding_dim: self.embedding_dim,
        }
    }
}

// f32-specific implementations for loading from quantized data
impl Embedding<f32> {
    /// Create a new embedding layer with the given quantized embedding table.
    pub fn new(embeddings_quantized: QMatrix) -> Self {
        let embeddings: Tensor<2, f32> = embeddings_quantized.dequantize();
        let shape = embeddings.shape();
        let num_embeddings = shape[0];
        let embedding_dim = shape[1];

        Self {
            embeddings_quantized: Some(embeddings_quantized),
            embeddings,
            num_embeddings,
            embedding_dim,
        }
    }

    /// Load an embedding layer from a VarBuilder.
    ///
    /// Expects weight tensor with shape: (num_embeddings, embedding_dim)
    pub fn load(device: &Device, vb: &mut VarBuilder) -> crate::Result<Self> {
        let embeddings = vb.get("weight", device)?;
        Ok(Self::new(embeddings))
    }

    /// Load an embedding layer with explicit shape verification.
    pub fn load_with_shape(
        device: &Device,
        vb: &mut VarBuilder,
        num_embeddings: usize,
        embedding_dim: usize,
    ) -> crate::Result<Self> {
        let embeddings = vb.get("weight", device)?;
        let shape = embeddings.shape();
        assert_eq!(
            shape[0], num_embeddings,
            "Embedding num_embeddings mismatch: expected {}, got {}",
            num_embeddings, shape[0]
        );
        assert_eq!(
            shape[1], embedding_dim,
            "Embedding embedding_dim mismatch: expected {}, got {}",
            embedding_dim, shape[1]
        );
        Ok(Self::new(embeddings))
    }

    /// Get the quantized embedding table if available.
    pub fn embeddings_quantized(&self) -> &QMatrix {
        self.embeddings_quantized
            .as_ref()
            .expect("No quantized embeddings available")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_embedding_1d() {
        // Create embedding table: (3, 2) - 3 embeddings, each of dimension 2
        let emb_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let embeddings: Tensor<2, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([3, 2], &emb_data));

        let embedding_layer = Embedding::new_from_tensor(embeddings);

        // Input: indices [0, 2, 1]
        let indices_data = [0u32, 2, 1];
        let indices: Tensor<1, u32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([3], &indices_data));

        let result: Tensor<2, f32> = embedding_layer.forward(&indices);

        assert_eq!(result.shape(), [3, 2]);

        let output = result.as_slice().await.unwrap();

        // index 0 -> [1, 2]
        assert_eq!(output[[0, 0]], 1.0);
        assert_eq!(output[[0, 1]], 2.0);
        // index 2 -> [5, 6]
        assert_eq!(output[[1, 0]], 5.0);
        assert_eq!(output[[1, 1]], 6.0);
        // index 1 -> [3, 4]
        assert_eq!(output[[2, 0]], 3.0);
        assert_eq!(output[[2, 1]], 4.0);
    }

    #[tokio::test]
    async fn test_embedding_2d() {
        // Create embedding table: (3, 2) - 3 embeddings, each of dimension 2
        let emb_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let embeddings: Tensor<2, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([3, 2], &emb_data));

        let embedding_layer = Embedding::new_from_tensor(embeddings);

        // Input: 2D indices [[0, 1], [2, 0]]
        let indices_data = [0u32, 1, 2, 0];
        let indices: Tensor<2, u32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([2, 2], &indices_data));

        let result: Tensor<3, f32> = embedding_layer.forward(&indices);

        assert_eq!(result.shape(), [2, 2, 2]);

        let output = result.as_slice().await.unwrap();

        // batch 0, seq 0 -> index 0 -> [1, 2]
        assert_eq!(output[[0, 0, 0]], 1.0);
        assert_eq!(output[[0, 0, 1]], 2.0);
        // batch 0, seq 1 -> index 1 -> [3, 4]
        assert_eq!(output[[0, 1, 0]], 3.0);
        assert_eq!(output[[0, 1, 1]], 4.0);
        // batch 1, seq 0 -> index 2 -> [5, 6]
        assert_eq!(output[[1, 0, 0]], 5.0);
        assert_eq!(output[[1, 0, 1]], 6.0);
        // batch 1, seq 1 -> index 0 -> [1, 2]
        assert_eq!(output[[1, 1, 0]], 1.0);
        assert_eq!(output[[1, 1, 1]], 2.0);
    }

    #[tokio::test]
    async fn test_embedding_properties() {
        let emb_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let embeddings: Tensor<2, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([2, 3], &emb_data));

        let embedding_layer = Embedding::new_from_tensor(embeddings);

        assert_eq!(embedding_layer.num_embeddings(), 2);
        assert_eq!(embedding_layer.embedding_dim(), 3);
    }
}
