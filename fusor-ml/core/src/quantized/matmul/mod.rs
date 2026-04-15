use crate::{
    DataType, DataTypeEnum, Device, Tensor, TensorData,
    compute_graph::NodeIndex,
    mir::{inputs::MirValue, kernel::GenericKernel, operation::Operation},
};
use fusor_gguf::GgmlType;

use super::QMatrix;

mod sgemm;
mod sgemv;

pub use sgemm::{ChunkedSgemmConfig, GeneralSgemmConfig};

#[derive(Debug, Clone)]
pub(crate) struct QMatMulOperation {
    pub(crate) input_datatype: DataTypeEnum,
    pub(crate) input: NodeIndex,
    pub(crate) matrix: QMatrix,
    pub(crate) in_shape: Box<[usize]>,
    pub(crate) out_shape: Box<[usize]>,
    pub(crate) chunked_config: Option<ChunkedSgemmConfig>,
    pub(crate) general_config: Option<GeneralSgemmConfig>,
}

impl QMatMulOperation {
    pub(crate) fn new(
        input_datatype: DataTypeEnum,
        input_shape: &[usize],
        input: NodeIndex,
        matrix: QMatrix,
    ) -> Self {
        let last_dim = input_shape.len() - 1;
        let mut out_shape = input_shape.to_vec();
        out_shape[last_dim] = matrix.shape[0];
        assert_eq!(input_shape[last_dim], matrix.shape[1]);
        let out_shape = out_shape.into_boxed_slice();
        QMatMulOperation {
            input_datatype,
            input,
            matrix,
            in_shape: input_shape.into(),
            out_shape,
            chunked_config: None,
            general_config: None,
        }
    }

    fn elements_per_block(&self) -> u32 {
        self.matrix.datatype.block_size() as u32
    }

    fn sgemv(&self) -> bool {
        let m_dim_idx = self.in_shape.len() - 2;
        let m = self.in_shape[m_dim_idx];
        // Use SGEMV for tall and skinny matrices (small M, any K)
        // SGEMV is more efficient when M is small because:
        // - Each workgroup processes one M value independently
        // - Less workgroup synchronization overhead
        // - Better cache utilization for the K dimension
        // SGEMM becomes more efficient for larger M where it can use
        // tile-based processing with 16x16 workgroups
        m <= 32
    }

    fn m_size(&self) -> u32 {
        let m_dim_idx = self.in_shape.len() - 2;
        self.in_shape[m_dim_idx] as u32
    }

    fn n_size(&self) -> u32 {
        self.matrix.shape[0] as u32
    }
}

impl<const R: usize, T: DataType> Tensor<R, T> {
    pub fn q_mat_mul(&self, other: &QMatrix) -> Self {
        // For F16/F32 matrices, dequantize and use regular mat_mul
        // because they don't have block structure like quantized types
        if matches!(other.datatype(), GgmlType::F16 | GgmlType::F32) {
            let dequantized: Tensor<2, T> = other.dequantize();
            // Broadcast the 2D matrix to match input tensor rank by unsqueezing batch dimensions
            let mut broadcast_shape = Vec::from(self.shape().as_slice());
            broadcast_shape[R - 1] = other.shape()[0]; // Output dimension is first dim of weight

            // The weight matrix is [out_features, in_features], need to transpose for mat_mul
            // self: [..., M, K] @ weight.T: [K, N] -> [..., M, N]
            // Reshape weight to add batch dimensions: [1, 1, ..., K, N]
            let weight_t = dequantized.transpose(0, 1);

            // Create batch dimensions for the weight
            let weight_shape: [usize; R] = std::array::from_fn(|i| {
                if i < R - 2 {
                    1 // Broadcast batch dimensions
                } else if i == R - 2 {
                    other.shape()[1] // K dimension
                } else {
                    other.shape()[0] // N dimension
                }
            });
            let weight_broadcast: Tensor<R, T> = weight_t.reshape(weight_shape);

            return self.mat_mul(&weight_broadcast);
        }
        self.add_q_mat_mul(other)
    }
}

#[cfg(test)]
async fn setup_smol_lm_matrix(
    name: &str,
) -> (crate::Device, QMatrix, candle_core::quantized::QMatMul) {
    use kalosm_model_types::FileSource;
    let source = FileSource::HuggingFace {
        model_id: "unsloth/SmolLM2-135M-Instruct-GGUF".to_string(),
        revision: "main".to_string(),
        file: "SmolLM2-135M-Instruct-Q4_K_M.gguf".to_string(),
    };

    setup_smol_lm_matrix_with_source(name, source).await
}

#[cfg(test)]
async fn setup_smol_lm_matrix_with_source(
    name: &str,
    source: kalosm_model_types::FileSource,
) -> (crate::Device, QMatrix, candle_core::quantized::QMatMul) {
    use crate::Device;
    use fusor_gguf::GgufMetadata;
    use kalosm_common::Cache;

    let device = Device::test_instance();

    let cache = Cache::default();
    let bytes = cache.get_bytes(&source, |_| {}).await.unwrap();

    let mut reader = std::io::Cursor::new(&bytes);
    let metadata = GgufMetadata::read(&mut reader).unwrap();
    let mut reader = std::io::Cursor::new(&bytes);
    let candle_metadata = candle_core::quantized::gguf_file::Content::read(&mut reader).unwrap();
    let candle_q_matrix_metadata = candle_metadata.tensor_infos.get(name).unwrap();
    let candle_q_tensor = candle_q_matrix_metadata
        .read(
            &mut reader,
            candle_metadata.tensor_data_offset,
            &candle_core::Device::Cpu,
        )
        .unwrap();
    let candle_q_matrix = candle_core::quantized::QMatMul::from_qtensor(candle_q_tensor).unwrap();

    let q_matrix_metadata = metadata.tensor_infos.get(name).unwrap();

    let q_matrix = QMatrix::read(
        &device,
        q_matrix_metadata,
        &mut reader,
        metadata.tensor_data_offset,
    )
    .unwrap();

    (device, q_matrix, candle_q_matrix)
}

#[cfg(test)]
#[tokio::test]
async fn test_fuzz_q_mat_mul() {
    use crate::Tensor;
    use candle_core::Module;

    let (device, q_matrix, candle_q_matrix) = setup_smol_lm_matrix("blk.0.attn_q.weight").await;
    println!("q_matrix: {q_matrix:?}");

    for _ in 0..25 {
        let batch = (rand::random::<u32>() as usize % 4) + 1;
        let random_data: Vec<Vec<Vec<f32>>> = (0..batch)
            .map(|_| {
                (0..576)
                    .map(|_| (0..576).map(|_| rand::random()).collect())
                    .collect()
            })
            .collect();
        let tensor = Tensor::<3, f32>::new(&device, &random_data);
        println!("tensor: {tensor:?}");

        let result = tensor.q_mat_mul(&q_matrix);
        let fusor_shape = result.shape();
        let result = result.as_slice().await.unwrap();

        let candle_b = candle_core::Tensor::from_iter(
            random_data
                .iter()
                .flat_map(|x| x.iter().flat_map(|x| x.iter().copied())),
            &candle_core::Device::Cpu,
        )
        .unwrap()
        .reshape(&[batch, 576, 576])
        .unwrap();
        let candle_result = candle_q_matrix.forward(&candle_b).unwrap();
        assert_eq!(candle_result.shape().dims(), &[batch, 576, 576]);
        let candle_result = candle_result.to_vec3::<f32>().unwrap();

        assert_eq!(fusor_shape, &[batch, 576, 576]);

        for batch in 0..batch {
            for x in 0..576 {
                for y in 0..576 {
                    let expected = candle_result[batch][x][y];
                    let actual = result[[batch, x, y]];
                    if (expected - actual).abs() > 0.5 {
                        panic!("expected: {expected}, actual: {actual}");
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_fuzz_q_mat_mul_transposed() {
    use crate::Tensor;
    use candle_core::Module;

    let (device, q_matrix, candle_q_matrix) = setup_smol_lm_matrix("blk.0.attn_q.weight").await;
    println!("q_matrix: {q_matrix:?}");

    for _ in 0..25 {
        let batch = (rand::random::<u32>() as usize % 4) + 1;
        let random_data: Vec<Vec<Vec<f32>>> = (0..576)
            .map(|_| {
                (0..576)
                    .map(|_| (0..batch).map(|_| rand::random()).collect())
                    .collect()
            })
            .collect();
        let tensor = Tensor::<3, f32>::new(&device, &random_data);
        println!("tensor: {tensor:?}");

        let result = tensor.transpose(0, 2).q_mat_mul(&q_matrix);
        let fusor_shape = result.shape();
        let result = result.as_slice().await.unwrap();

        let candle_b = candle_core::Tensor::from_iter(
            random_data
                .iter()
                .flat_map(|x| x.iter().flat_map(|x| x.iter().copied())),
            &candle_core::Device::Cpu,
        )
        .unwrap()
        .reshape(&[576, 576, batch])
        .unwrap();
        let candle_result = candle_q_matrix
            .forward(&candle_b.transpose(0, 2).unwrap().contiguous().unwrap())
            .unwrap();
        assert_eq!(candle_result.shape().dims(), &[batch, 576, 576]);
        let candle_result = candle_result.to_vec3::<f32>().unwrap();

        assert_eq!(fusor_shape, &[batch, 576, 576]);

        for batch in 0..batch {
            for x in 0..576 {
                for y in 0..576 {
                    let expected = candle_result[batch][x][y];
                    let actual = result[[batch, x, y]];
                    if (expected - actual).abs() > 0.5 {
                        panic!("expected: {expected}, actual: {actual}");
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_fuzz_q_mat_mul_sgemv() {
    use crate::Tensor;
    use candle_core::Module;

    let (device, q_matrix, candle_q_matrix) = setup_smol_lm_matrix("token_embd.weight").await;

    for _ in 0..25 {
        let size = 576;
        let embed_dim = 49152;
        let batch = (rand::random::<u32>() as usize % 4) + 1;
        let random_data: Vec<Vec<Vec<f32>>> = (0..batch)
            .map(|_| {
                (0..1)
                    .map(|_| (0..size).map(|_| rand::random()).collect())
                    .collect()
            })
            .collect();
        let tensor = Tensor::<3, f32>::new(&device, &random_data);

        let result = tensor.q_mat_mul(&q_matrix);
        let fusor_shape = result.shape();
        let result = result.as_slice().await.unwrap();

        let candle_b = candle_core::Tensor::from_iter(
            random_data
                .iter()
                .flat_map(|x| x.iter().flat_map(|x| x.iter().copied())),
            &candle_core::Device::Cpu,
        )
        .unwrap()
        .reshape(&[batch, 1, size])
        .unwrap();
        let candle_result = candle_q_matrix.forward(&candle_b).unwrap();
        assert_eq!(candle_result.shape().dims(), &[batch, 1, embed_dim]);
        let candle_result = candle_result.to_vec3::<f32>().unwrap();

        assert_eq!(fusor_shape, &[batch, 1, embed_dim]);

        for batch in 0..batch {
            for x in 0..1 {
                for y in 0..embed_dim {
                    let expected = candle_result[batch][x][y];
                    let actual = result[[batch, x, y]];
                    if (expected - actual).abs() > 0.5 {
                        println!("Expected: {candle_result:?}");
                        println!("Actual: {result:?}");
                        panic!("expected: {expected}, actual: {actual}");
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_fuzz_q_mat_mul_gemv_transposed() {
    use crate::Tensor;
    use candle_core::Module;

    let (device, q_matrix, candle_q_matrix) = setup_smol_lm_matrix("blk.0.attn_q.weight").await;
    println!("q_matrix: {q_matrix:?}");

    for _ in 0..25 {
        let batch = (rand::random::<u32>() as usize % 4) + 1;
        let random_data: Vec<Vec<Vec<f32>>> = (0..576)
            .map(|_| {
                (0..1)
                    .map(|_| (0..batch).map(|_| rand::random()).collect())
                    .collect()
            })
            .collect();
        let tensor = Tensor::<3, f32>::new(&device, &random_data);
        println!("tensor: {tensor:?}");

        let result = tensor.transpose(0, 2).q_mat_mul(&q_matrix);
        let fusor_shape = result.shape();
        let result = result.as_slice().await.unwrap();

        let candle_b = candle_core::Tensor::from_iter(
            random_data
                .iter()
                .flat_map(|x| x.iter().flat_map(|x| x.iter().copied())),
            &candle_core::Device::Cpu,
        )
        .unwrap()
        .reshape(&[576, 1, batch])
        .unwrap();
        let candle_result = candle_q_matrix
            .forward(&candle_b.transpose(0, 2).unwrap().contiguous().unwrap())
            .unwrap();
        assert_eq!(candle_result.shape().dims(), &[batch, 1, 576]);
        let candle_result = candle_result.to_vec3::<f32>().unwrap();

        assert_eq!(fusor_shape, &[batch, 1, 576]);

        for batch in 0..batch {
            for x in 0..1 {
                for y in 0..576 {
                    let expected = candle_result[batch][x][y];
                    let actual = result[[batch, x, y]];
                    if (expected - actual).abs() > 0.5 {
                        panic!("expected: {expected}, actual: {actual}");
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_fuzz_q_mat_mul_q8_0() {
    use crate::Tensor;
    use candle_core::Module;

    let (device, q_matrix, candle_q_matrix) = setup_smol_lm_matrix("token_embd.weight").await;

    // Always test the edge cases
    let mut widths = vec![1, 256];
    // Then test a bunch of other random widths
    widths.extend((2..25).map(|_| rand::random_range(1..=64)));

    for width in widths {
        let batch = (rand::random::<u32>() as usize % 2) + 1;
        let random_data: Vec<Vec<Vec<f32>>> = (0..batch)
            .map(|_| {
                (0..width)
                    .map(|_| (0..576).map(|_| rand::random()).collect())
                    .collect()
            })
            .collect();
        let tensor = Tensor::<3, f32>::new(&device, &random_data);

        let result = tensor.q_mat_mul(&q_matrix);
        let fusor_shape = result.shape();
        let result = result.as_slice().await.unwrap();

        let candle_b = candle_core::Tensor::from_iter(
            random_data
                .iter()
                .flat_map(|x| x.iter().flat_map(|x| x.iter().copied())),
            &candle_core::Device::Cpu,
        )
        .unwrap()
        .reshape(&[batch, width, 576])
        .unwrap();
        let candle_result = candle_q_matrix.forward(&candle_b).unwrap();
        assert_eq!(candle_result.shape().dims(), &[batch, width, 49152]);
        let candle_result = candle_result.to_vec3::<f32>().unwrap();

        assert_eq!(fusor_shape, &[batch, width, 49152]);

        for batch in 0..batch {
            for x in 0..width {
                for y in 0..49152 {
                    let expected = candle_result[batch][x][y];
                    let actual = result[[batch, x, y]];
                    if (expected - actual).abs() > 0.5 {
                        println!("Expected: {candle_result:?}");
                        println!("Actual: {result:?}");
                        panic!("expected: {expected}, actual: {actual}");
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_fuzz_q_mat_mul_q5_0_gemv() {
    use crate::Tensor;
    use candle_core::Module;

    let (device, q_matrix, candle_q_matrix) = setup_smol_lm_matrix("blk.0.ffn_gate.weight").await;

    for _ in 0..25 {
        let width = 1;
        let height = 1536;
        let batch = (rand::random::<u32>() as usize % 4) + 1;
        let random_data: Vec<Vec<Vec<f32>>> = (0..batch)
            .map(|_| {
                (0..width)
                    .map(|_| (0..576).map(|_| rand::random()).collect())
                    .collect()
            })
            .collect();
        let tensor = Tensor::<3, f32>::new(&device, &random_data);

        let result = tensor.q_mat_mul(&q_matrix);
        let fusor_shape = result.shape();
        let result = result.as_slice().await.unwrap();

        let candle_b = candle_core::Tensor::from_iter(
            random_data
                .iter()
                .flat_map(|x| x.iter().flat_map(|x| x.iter().copied())),
            &candle_core::Device::Cpu,
        )
        .unwrap()
        .reshape(&[batch, width, 576])
        .unwrap();
        let candle_result = candle_q_matrix.forward(&candle_b).unwrap();
        assert_eq!(candle_result.shape().dims(), &[batch, width, height]);
        let candle_result = candle_result.to_vec3::<f32>().unwrap();

        assert_eq!(fusor_shape, &[batch, width, height]);

        for batch in 0..batch {
            for x in 0..width {
                for y in 0..height {
                    let expected = candle_result[batch][x][y];
                    let actual = result[[batch, x, y]];
                    if (expected - actual).abs() > 0.5 {
                        println!("Expected: {candle_result:?}");
                        println!("Actual: {result:?}");
                        panic!("expected: {expected}, actual: {actual}");
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_fuzz_q_mat_mul_q4_0_gemv() {
    use crate::Tensor;
    use candle_core::Module;
    use kalosm_model_types::FileSource;

    let source = FileSource::HuggingFace {
        model_id: "bartowski/SmolLM2-135M-Instruct-GGUF".to_string(),
        revision: "main".to_string(),
        file: "SmolLM2-135M-Instruct-Q4_0.gguf".to_string(),
    };
    let (device, q_matrix, candle_q_matrix) =
        setup_smol_lm_matrix_with_source("blk.0.ffn_gate.weight", source).await;

    for _ in 0..25 {
        let width = 1;
        let batch = (rand::random::<u32>() as usize % 4) + 1;
        let random_data: Vec<Vec<Vec<f32>>> = (0..batch)
            .map(|_| {
                (0..1)
                    .map(|_| (0..576).map(|_| rand::random()).collect())
                    .collect()
            })
            .collect();
        let tensor = Tensor::<3, f32>::new(&device, &random_data);

        let result = tensor.q_mat_mul(&q_matrix);
        let fusor_shape = result.shape();
        let result = result.as_slice().await.unwrap();

        let candle_b = candle_core::Tensor::from_iter(
            random_data
                .iter()
                .flat_map(|x| x.iter().flat_map(|x| x.iter().copied())),
            &candle_core::Device::Cpu,
        )
        .unwrap()
        .reshape(&[batch, width, 576])
        .unwrap();
        let candle_result = candle_q_matrix.forward(&candle_b).unwrap();
        assert_eq!(candle_result.shape().dims(), &[batch, width, 1536]);
        let candle_result = candle_result.to_vec3::<f32>().unwrap();

        assert_eq!(fusor_shape, &[batch, width, 1536]);

        for batch in 0..batch {
            for x in 0..width {
                for y in 0..1536 {
                    let expected = candle_result[batch][x][y];
                    let actual = result[[batch, x, y]];
                    if (expected - actual).abs() > 0.5 {
                        println!("Expected: {candle_result:?}");
                        println!("Actual: {result:?}");
                        panic!("expected: {expected}, actual: {actual}");
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_fuzz_q_mat_mul_q6k() {
    use crate::Tensor;
    use candle_core::Module;

    let (device, q_matrix, candle_q_matrix) = setup_smol_lm_matrix("blk.0.ffn_down.weight").await;

    // Always test the edge cases
    let mut widths = vec![1, 256];
    // Then test a bunch of other random widths
    widths.extend((2..25).map(|_| rand::random_range(1..=64)));

    for width in widths {
        let batch = (rand::random::<u32>() as usize % 4) + 1;
        let random_data: Vec<Vec<Vec<f32>>> = (0..batch)
            .map(|_| {
                (0..width)
                    .map(|_| (0..1536).map(|_| rand::random()).collect())
                    .collect()
            })
            .collect();
        let tensor = Tensor::<3, f32>::new(&device, &random_data);

        let result = tensor.q_mat_mul(&q_matrix);
        let fusor_shape = result.shape();
        let result = result.as_slice().await.unwrap();

        let candle_b = candle_core::Tensor::from_iter(
            random_data
                .iter()
                .flat_map(|x| x.iter().flat_map(|x| x.iter().copied())),
            &candle_core::Device::Cpu,
        )
        .unwrap()
        .reshape(&[batch, width, 1536])
        .unwrap();
        let candle_result = candle_q_matrix.forward(&candle_b).unwrap();
        assert_eq!(candle_result.shape().dims(), &[batch, width, 576]);
        let candle_result = candle_result.to_vec3::<f32>().unwrap();

        assert_eq!(fusor_shape, &[batch, width, 576]);

        for batch in 0..batch {
            for x in 0..width {
                for y in 0..576 {
                    let expected = candle_result[batch][x][y];
                    let actual = result[[batch, x, y]];
                    if (expected - actual).abs() > 0.5 {
                        println!("width: {width}");
                        println!("Expected: {candle_result:?}");
                        println!("Actual: {result:?}");
                        panic!("expected: {expected}, actual: {actual}");
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_fuzz_q_mat_mul_q4k() {
    use crate::Tensor;
    use candle_core::Module;

    let (device, q_matrix, candle_q_matrix) = setup_smol_lm_matrix("blk.3.ffn_down.weight").await;

    // Always test the edge cases
    let mut widths = vec![1, 256];
    // Then test a bunch of other random widths
    widths.extend((2..25).map(|_| rand::random_range(1..=64)));

    for width in widths {
        let batch = (rand::random::<u32>() as usize % 4) + 1;
        let random_data: Vec<Vec<Vec<f32>>> = (0..batch)
            .map(|_| {
                (0..width)
                    .map(|_| (0..1536).map(|_| rand::random()).collect())
                    .collect()
            })
            .collect();
        let tensor = Tensor::<3, f32>::new(&device, &random_data);

        let result = tensor.q_mat_mul(&q_matrix);
        let fusor_shape = result.shape();
        let result = result.as_slice().await.unwrap();

        let candle_b = candle_core::Tensor::from_iter(
            random_data
                .iter()
                .flat_map(|x| x.iter().flat_map(|x| x.iter().copied())),
            &candle_core::Device::Cpu,
        )
        .unwrap()
        .reshape(&[batch, width, 1536])
        .unwrap();
        let candle_result = candle_q_matrix.forward(&candle_b).unwrap();
        assert_eq!(candle_result.shape().dims(), &[batch, width, 576]);
        let candle_result = candle_result.to_vec3::<f32>().unwrap();

        assert_eq!(fusor_shape, &[batch, width, 576]);

        for batch in 0..batch {
            for x in 0..width {
                for y in 0..576 {
                    let expected = candle_result[batch][x][y];
                    let actual = result[[batch, x, y]];
                    if (expected - actual).abs() > 0.5 {
                        println!("width: {width}");
                        println!("Expected: {candle_result:?}");
                        println!("Actual: {result:?}");
                        panic!("expected: {expected}, actual: {actual}");
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_fuzz_q_mat_mul_q5k() {
    use crate::Tensor;
    use candle_core::Module;
    use fusor_gguf::GgufMetadata;
    use kalosm_common::Cache;

    // Phi-4 Q5_K_M model has Q5K matrices in attention weights
    let source = kalosm_model_types::FileSource::HuggingFace {
        model_id: "unsloth/Phi-4-mini-instruct-GGUF".to_string(),
        revision: "main".to_string(),
        file: "Phi-4-mini-instruct-Q5_K_M.gguf".to_string(),
    };

    // First find a Q5K tensor
    let cache = Cache::default();
    let bytes = cache.get_bytes(&source.clone(), |_| {}).await.unwrap();
    let mut reader = std::io::Cursor::new(&bytes);
    let metadata = GgufMetadata::read(&mut reader).unwrap();

    // Find a Q5K tensor
    let q5k_tensor = metadata
        .tensor_infos
        .iter()
        .find(|(_, info)| info.ty == fusor_gguf::GgmlType::Q5K)
        .map(|(name, _)| name.clone());

    let tensor_name = match q5k_tensor {
        Some(name) => {
            println!("Found Q5K tensor: {name}");
            name
        }
        None => {
            println!("No Q5K tensors found in model, skipping test");
            return;
        }
    };

    let (device, q_matrix, candle_q_matrix) =
        setup_smol_lm_matrix_with_source(&tensor_name, source).await;

    // Make sure it's actually Q5K
    assert_eq!(q_matrix.datatype, fusor_gguf::GgmlType::Q5K);

    // Test various widths
    let mut widths = vec![1, 256];
    widths.extend((2..10).map(|_| rand::random_range(1..=64)));

    let k_size = q_matrix.shape[1];
    let n_size = q_matrix.shape[0];

    for width in widths {
        let batch = (rand::random::<u32>() as usize % 2) + 1;
        let random_data: Vec<Vec<Vec<f32>>> = (0..batch)
            .map(|_| {
                (0..width)
                    .map(|_| (0..k_size).map(|_| rand::random()).collect())
                    .collect()
            })
            .collect();
        let tensor = Tensor::<3, f32>::new(&device, &random_data);

        let result = tensor.q_mat_mul(&q_matrix);
        let fusor_shape = result.shape();
        let result = result.as_slice().await.unwrap();

        let candle_b = candle_core::Tensor::from_iter(
            random_data
                .iter()
                .flat_map(|x| x.iter().flat_map(|x| x.iter().copied())),
            &candle_core::Device::Cpu,
        )
        .unwrap()
        .reshape(&[batch, width, k_size])
        .unwrap();
        let candle_result = candle_q_matrix.forward(&candle_b).unwrap();
        assert_eq!(candle_result.shape().dims(), &[batch, width, n_size]);
        let candle_result = candle_result.to_vec3::<f32>().unwrap();

        assert_eq!(fusor_shape, &[batch, width, n_size]);

        for batch in 0..batch {
            for x in 0..width {
                for y in 0..n_size {
                    let expected = candle_result[batch][x][y];
                    let actual = result[[batch, x, y]];
                    if (expected - actual).abs() > 1.0 {
                        println!("width: {width}");
                        println!("Expected: {candle_result:?}");
                        println!("Actual: {result:?}");
                        panic!("expected: {expected}, actual: {actual}");
                    }
                }
            }
        }
    }
}

impl Operation for QMatMulOperation {
    fn workgroup_shape_constraints(
        &self,
        device: &Device,
    ) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
        if self.sgemv() {
            sgemv::workgroup_shape_constraints(&self.matrix, device)
        } else {
            sgemm::workgroup_shape_constraints(&self.matrix, device)
        }
    }

    fn dispatch_size(
        &self,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        _: &[MirValue],
    ) -> [u32; 3] {
        let n = self.n_size();
        let m = self.m_size();
        // Calculate batch size for dimensions beyond the last two (M, K)
        let batch_size: u32 = self
            .in_shape
            .iter()
            .rev()
            .skip(2)
            .map(|x| *x as u32)
            .product();

        if self.sgemv() {
            sgemv::dispatch_size(&self.matrix, n, m, batch_size)
        } else {
            sgemm::dispatch_size(self, workgroup_shape, &self.matrix, n, m, batch_size)
        }
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        f(self.input);
    }

    fn inputs(&self, nodes: &crate::compute_graph::ComputeGraphInner) -> Vec<MirValue> {
        let input = nodes.get_result(self.input).unwrap();
        let q_matrix = self.matrix.clone();
        let device = input.device();
        let output_tensor = TensorData::new_for_shape(device, &self.out_shape, input.datatype());
        vec![input.into(), q_matrix.into(), output_tensor.into()]
    }

    // Related files/PRs in llama.cpp for reference:
    // https://github.com/ggml-org/llama.cpp/pull/2290
    // https://github.com/ggml-org/llama.cpp/blob/add2a3aa5a1571211aa5c7303b8e80c8d1824b91/ggml/src/ggml-metal/ggml-metal.metal#L4561
    // https://github.com/ggml-org/llama.cpp/blob/add2a3aa5a1571211aa5c7303b8e80c8d1824b91/ggml/src/ggml-metal/ggml-metal.metal#L5881
    // based on https://siboehm.com/articles/22/CUDA-MMM
    fn build_kernel(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        _: &[MirValue],
        generic_kernel: &mut GenericKernel,
    ) {
        let datatype = self.input_datatype;
        let rank = self.in_shape.len() as u32;
        let matrix_rank = self.matrix.shape.len() as u32;

        let input_a = generic_kernel.add_tensor_input(rank, false, datatype);
        let input_b = generic_kernel.add_q_matrix_input(matrix_rank, self.matrix.datatype);
        let output = generic_kernel.add_tensor_input(rank, true, datatype);

        // For batched operations, we need to get the correct dimension indices
        let k_size = input_a.shape_binding(rank - 1).to_string(); // Last dimension is K
        let m_size = input_a.shape_binding(rank - 2).to_string(); // Second-to-last dimension is M
        let n_size = input_b.shape_binding(0).to_string();

        // Check if this is a sgemv or sgemm operation
        let algo = if self.sgemv() {
            sgemv::sgemv
        } else {
            sgemm::sgemm
        };

        algo(
            self,
            generic_kernel,
            workgroup_shape,
            &input_a,
            &input_b,
            &output,
            &n_size,
            &m_size,
            &k_size,
            graph,
        );
    }

    fn output(&self, _: &crate::compute_graph::ComputeGraphInner, inputs: &[MirValue]) -> MirValue {
        let output_tensor = inputs[2].as_tensor().unwrap();
        output_tensor.clone().into()
    }

    fn name(&self) -> String {
        format!(
            "q_mat_mul_{}_{}_{}_{}",
            self.input_datatype,
            self.in_shape
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join("x"),
            self.matrix.datatype,
            self.matrix
                .shape
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join("x")
        )
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_fuzz_q_mat_mul_f16_sgemv() {
    use crate::Tensor;
    use half::f16;

    // Use a Q5_0 weight to test the q_n_sgemv path
    let (device, q_matrix, candle_q_matrix) = setup_smol_lm_matrix("blk.0.attn_q.weight").await;
    println!("q_matrix: {q_matrix:?}");

    if !device.f16_supported() {
        return;
    }

    // Use small M to trigger sgemv path
    let batch = 1;
    let seq_len = 12; // Small M to use sgemv
    let hidden = 576;

    // Use simple input data (all 0.5)
    let random_data: Vec<Vec<Vec<f16>>> = (0..batch)
        .map(|_| {
            (0..seq_len)
                .map(|_| (0..hidden).map(|_| f16::from_f32(0.5)).collect())
                .collect()
        })
        .collect();
    let tensor = Tensor::<3, f16>::new(&device, &random_data);
    println!("f16 tensor: {tensor:?}");

    // First check that the f16 tensor is valid
    let tensor_slice = tensor.as_slice().await.unwrap();
    let mut input_nan_count = 0;
    for b in 0..batch {
        for s in 0..seq_len {
            for h in 0..hidden {
                let val = tensor_slice[[b, s, h]];
                if val.is_nan() || val.is_infinite() {
                    input_nan_count += 1;
                }
            }
        }
    }
    println!("Input has {} NaN/inf values", input_nan_count);

    // Now do matmul
    let result: Tensor<3, f16> = tensor.q_mat_mul(&q_matrix);
    let fusor_shape = result.shape();
    let result = result.as_slice().await.unwrap();

    // Check for NaN/inf values
    let mut nan_count = 0;
    let mut inf_count = 0;
    for b in 0..batch {
        for s in 0..seq_len {
            for h in 0..hidden {
                let val = result[[b, s, h]];
                if val.is_nan() {
                    nan_count += 1;
                }
                if val.is_infinite() {
                    inf_count += 1;
                }
            }
        }
    }

    // Print some output values for debugging
    if nan_count > 0 {
        println!(
            "First few output values: {:?}",
            (0..10).map(|i| result[[0, 0, i]]).collect::<Vec<_>>()
        );
    }

    if nan_count > 0 || inf_count > 0 {
        panic!(
            "Result contains {} NaN and {} inf values out of {}",
            nan_count,
            inf_count,
            batch * seq_len * hidden
        );
    }

    // Compare with candle f32 reference
    let random_data_f32: Vec<Vec<Vec<f32>>> = random_data
        .iter()
        .map(|b| {
            b.iter()
                .map(|s| s.iter().map(|v| v.to_f32()).collect())
                .collect()
        })
        .collect();

    let candle_b = candle_core::Tensor::from_iter(
        random_data_f32
            .iter()
            .flat_map(|x| x.iter().flat_map(|x| x.iter().copied())),
        &candle_core::Device::Cpu,
    )
    .unwrap()
    .reshape(&[batch, seq_len, hidden])
    .unwrap();
    let candle_result = candle_core::Module::forward(&candle_q_matrix, &candle_b).unwrap();
    let candle_result = candle_result.to_vec3::<f32>().unwrap();

    for b in 0..batch {
        for s in 0..seq_len {
            for h in 0..hidden {
                let expected = candle_result[b][s][h];
                let actual = result[[b, s, h]].to_f32();
                // f16 has lower precision so we use a larger tolerance
                if (expected - actual).abs() > 2.0 {
                    panic!(
                        "Mismatch at [{}, {}, {}]: expected {}, got {}",
                        b, s, h, expected, actual
                    );
                }
            }
        }
    }

    println!("f16 sgemv test passed for shape {:?}", fusor_shape);
}
