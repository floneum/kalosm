//! Embedding layer implementation.

use crate::{CastTensor, CastTo, DataType, Device, QMatrix, SimdElement, Tensor, VarBuilder};
use fusor_gguf::GgmlType;

/// Embedding layer for token/position embeddings.
///
/// Maps integer indices to dense vectors.
/// Embedding table shape: (num_embeddings, embedding_dim)
#[derive(Clone)]
pub struct Embedding<T: SimdElement> {
    embeddings_quantized: Option<QMatrix>,
    embeddings: Option<Tensor<2, T>>,
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
            embeddings: Some(embeddings),
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
        B: crate::cpu::TensorBacking<N, Elem = u32>,
        crate::gpu::Tensor<N, u32>: crate::gpu::NextRank<M, u32>,
        f32: CastTensor<T> + CastTo<T>,
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

        if self.embeddings.is_none()
            && let Some(quantized) = &self.embeddings_quantized
            && !matches!(quantized.ggml_type(), GgmlType::F16 | GgmlType::F32)
        {
            return match (indices, quantized) {
                (Tensor::Cpu(cpu_indices), quantized) if quantized.is_cpu() => {
                    let dense: Tensor<2, f32> = quantized.dequantize();
                    let dense: Tensor<2, T> = dense.cast();
                    let Tensor::Cpu(cpu_embeddings) = dense else {
                        unreachable!("CPU quantized embedding dequantized to a GPU tensor");
                    };
                    let indices_flat = cpu_indices.as_ref().flatten_all();
                    let values = cpu_embeddings.as_ref().index_select(0, indices_flat);
                    Tensor::Cpu(values.reshape(final_dims).to_concrete())
                }
                (Tensor::Gpu(gpu_indices), QMatrix::Gpu(gpu_embeddings)) => {
                    let indices_flat = gpu_indices.flatten_all();
                    let values = gpu_embeddings.index_select_rows(&indices_flat);
                    Tensor::Gpu(values.reshape(final_dims).cast())
                }
                _ => panic!("Indices and embeddings must be on the same device"),
            };
        }

        let embeddings = self
            .embeddings
            .as_ref()
            .expect("dense embeddings unavailable for this embedding table");

        match (indices, embeddings) {
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
        self.embeddings
            .as_ref()
            .expect("dense embeddings unavailable for this embedding table")
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
            embeddings: self.embeddings.map(|embeddings| embeddings.cast()),
            num_embeddings: self.num_embeddings,
            embedding_dim: self.embedding_dim,
        }
    }
}

// f32-specific implementations for loading from quantized data
impl Embedding<f32> {
    /// Create a new embedding layer with the given quantized embedding table.
    pub fn new(embeddings_quantized: QMatrix) -> Self {
        let shape = embeddings_quantized.shape();
        let num_embeddings = shape[0];
        let embedding_dim = shape[1];
        let embeddings = if embeddings_quantized.is_cpu()
            || matches!(
                embeddings_quantized.ggml_type(),
                GgmlType::F16 | GgmlType::F32
            ) {
            Some(embeddings_quantized.dequantize())
        } else {
            None
        };

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
