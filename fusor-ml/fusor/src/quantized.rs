//! Unified CPU/GPU quantized tensor abstraction
//!
//! This module provides `QMatrix`, a runtime dispatch enum that holds either CPU
//! quantized tensors (`QuantizedTensor`) or GPU quantized matrices (`QMatrix`).

use crate::{Device, Tensor};
use fusor_core::QMatrix as GpuQMatrix;
use fusor_cpu::{
    ABox, AVec, BlockQ4_0, BlockQ4K, BlockQ5_0, BlockQ5K, BlockQ6K, BlockQ8_0, GgmlType, Layout,
    QuantizedTensor,
};
use half::f16;

/// CPU tensor with F32 data (not quantized).
///
/// This stores unquantized f32 data with a dynamic shape, matching the interface
/// of quantized tensors for uniform handling in `QMatrix`.
#[derive(Clone)]
pub struct CpuF32Tensor {
    /// The f32 data stored in aligned memory
    data: ABox<[f32]>,
    /// The shape of the tensor
    shape: Box<[usize]>,
}

impl CpuF32Tensor {
    /// Create a new CpuF32Tensor from data and shape.
    pub fn new(data: ABox<[f32]>, shape: Box<[usize]>) -> Self {
        Self { data, shape }
    }

    /// Returns the shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns a reference to the underlying data.
    pub fn data(&self) -> &ABox<[f32]> {
        &self.data
    }
}

/// CPU tensor with F16 data (not quantized, but stored as half precision).
///
/// This stores f16 data with a dynamic shape, matching the interface
/// of quantized tensors for uniform handling in `QMatrix`.
#[derive(Clone)]
pub struct CpuF16Tensor {
    /// The f16 data stored in aligned memory
    data: ABox<[f16]>,
    /// The shape of the tensor
    shape: Box<[usize]>,
}

impl CpuF16Tensor {
    /// Create a new CpuF16Tensor from data and shape.
    pub fn new(data: ABox<[f16]>, shape: Box<[usize]>) -> Self {
        Self { data, shape }
    }

    /// Returns the shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns a reference to the underlying data.
    pub fn data(&self) -> &ABox<[f16]> {
        &self.data
    }
}

/// Unified quantized tensor type that holds either CPU or GPU quantized data.
///
/// This enum enables writing generic code that works with both CPU and GPU
/// quantized tensors while preserving the benefits of each backend.
///
/// The CPU variants are parameterized by block type at compile time, while
/// the GPU variant uses runtime type information via `GgmlType`.
#[derive(Clone)]
pub enum QMatrix {
    /// CPU quantized tensor with Q4_0 quantization (4-bit, block size 32)
    CpuQ4_0(QuantizedTensor<BlockQ4_0>),
    /// CPU quantized tensor with Q5_0 quantization (5-bit, block size 32)
    CpuQ5_0(QuantizedTensor<BlockQ5_0>),
    /// CPU quantized tensor with Q8_0 quantization (8-bit, block size 32)
    CpuQ8_0(QuantizedTensor<BlockQ8_0>),
    /// CPU quantized tensor with Q4K quantization (4-bit, block size 256)
    CpuQ4K(QuantizedTensor<BlockQ4K>),
    /// CPU quantized tensor with Q5K quantization (5-bit, block size 256)
    CpuQ5K(QuantizedTensor<BlockQ5K>),
    /// CPU quantized tensor with Q6K quantization (6-bit, block size 256)
    CpuQ6K(QuantizedTensor<BlockQ6K>),
    /// CPU tensor with F32 data (not quantized)
    CpuF32(CpuF32Tensor),
    /// CPU tensor with F16 data (half precision)
    CpuF16(CpuF16Tensor),
    /// GPU quantized matrix (type-erased, uses runtime GgmlType)
    Gpu(GpuQMatrix),
}

impl QMatrix {
    /// Returns the quantization type (e.g., Q4_0, Q8_0, Q4K, etc.)
    pub fn ggml_type(&self) -> GgmlType {
        match self {
            QMatrix::CpuQ4_0(_) => GgmlType::Q4_0,
            QMatrix::CpuQ5_0(_) => GgmlType::Q5_0,
            QMatrix::CpuQ8_0(_) => GgmlType::Q8_0,
            QMatrix::CpuQ4K(_) => GgmlType::Q4K,
            QMatrix::CpuQ5K(_) => GgmlType::Q5K,
            QMatrix::CpuQ6K(_) => GgmlType::Q6K,
            QMatrix::CpuF32(_) => GgmlType::F32,
            QMatrix::CpuF16(_) => GgmlType::F16,
            QMatrix::Gpu(m) => m.datatype(),
        }
    }

    /// Returns true if this is the CPU variant.
    #[inline]
    pub fn is_cpu(&self) -> bool {
        !matches!(self, QMatrix::Gpu(_))
    }

    /// Returns true if this is the GPU variant.
    #[inline]
    pub fn is_gpu(&self) -> bool {
        matches!(self, QMatrix::Gpu(_))
    }

    /// Returns the shape of the quantized tensor.
    pub fn shape(&self) -> &[usize] {
        match self {
            QMatrix::CpuQ4_0(t) => t.element_shape(),
            QMatrix::CpuQ5_0(t) => t.element_shape(),
            QMatrix::CpuQ8_0(t) => t.element_shape(),
            QMatrix::CpuQ4K(t) => t.element_shape(),
            QMatrix::CpuQ5K(t) => t.element_shape(),
            QMatrix::CpuQ6K(t) => t.element_shape(),
            QMatrix::CpuF32(t) => t.shape(),
            QMatrix::CpuF16(t) => t.shape(),
            QMatrix::Gpu(m) => m.shape(),
        }
    }

    /// Returns the device this tensor is on.
    pub fn device(&self) -> Device {
        match self {
            QMatrix::CpuQ4_0(_)
            | QMatrix::CpuQ5_0(_)
            | QMatrix::CpuQ8_0(_)
            | QMatrix::CpuQ4K(_)
            | QMatrix::CpuQ5K(_)
            | QMatrix::CpuQ6K(_)
            | QMatrix::CpuF32(_)
            | QMatrix::CpuF16(_) => Device::Cpu,
            QMatrix::Gpu(m) => Device::Gpu(m.device().clone()),
        }
    }

    /// Create a quantized tensor from raw bytes.
    ///
    /// This dispatches to either CPU or GPU based on the device.
    ///
    /// # Arguments
    /// * `device` - The device to create the tensor on
    /// * `shape` - The logical shape in elements (not blocks)
    /// * `bytes` - Raw quantized bytes
    /// * `ty` - The quantization type
    ///
    /// # Panics
    /// Panics if the quantization type is not supported.
    pub fn from_raw_bytes(
        device: &Device,
        shape: impl Into<Box<[usize]>>,
        bytes: &[u8],
        ty: GgmlType,
    ) -> Result<Self, fusor_core::GgufReadError> {
        let shape = shape.into();
        match device {
            Device::Cpu => Ok(match ty {
                GgmlType::Q4_0 => QMatrix::CpuQ4_0(QuantizedTensor::from_raw_bytes(shape, bytes)),
                GgmlType::Q5_0 => QMatrix::CpuQ5_0(QuantizedTensor::from_raw_bytes(shape, bytes)),
                GgmlType::Q8_0 => QMatrix::CpuQ8_0(QuantizedTensor::from_raw_bytes(shape, bytes)),
                GgmlType::Q4K => QMatrix::CpuQ4K(QuantizedTensor::from_raw_bytes(shape, bytes)),
                GgmlType::Q5K => QMatrix::CpuQ5K(QuantizedTensor::from_raw_bytes(shape, bytes)),
                GgmlType::Q6K => QMatrix::CpuQ6K(QuantizedTensor::from_raw_bytes(shape, bytes)),
                GgmlType::F32 => {
                    // F32 is not quantized, load directly as f32 tensor
                    let f32_data: Vec<f32> = bytes
                        .chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                        .collect();
                    let mut data = AVec::<f32>::with_capacity(64, f32_data.len());
                    data.extend_from_slice(&f32_data);
                    QMatrix::CpuF32(CpuF32Tensor::new(data.into_boxed_slice(), shape))
                }
                GgmlType::F16 => {
                    // F16 is not quantized, load directly as f16 tensor
                    let f16_data: Vec<f16> = bytes
                        .chunks_exact(2)
                        .map(|chunk| f16::from_le_bytes(chunk.try_into().unwrap()))
                        .collect();
                    let mut data = AVec::<f16>::with_capacity(64, f16_data.len());
                    data.extend_from_slice(&f16_data);
                    QMatrix::CpuF16(CpuF16Tensor::new(data.into_boxed_slice(), shape))
                }
                _ => panic!("Unsupported quantization type for CPU: {:?}", ty),
            }),
            Device::Gpu(gpu_device) => {
                let q_matrix = GpuQMatrix::from_parts(gpu_device, bytes, shape, ty)?;
                Ok(QMatrix::Gpu(q_matrix))
            }
        }
    }

    /// Dequantize to an f32 tensor.
    ///
    /// This converts the quantized tensor to full-precision f32.
    ///
    /// # Panics
    /// Panics if the tensor's rank doesn't match R.
    pub fn dequantize<const R: usize>(&self) -> Tensor<R, f32> {
        match self {
            QMatrix::CpuQ4_0(t) => Tensor::Cpu(fusor_cpu::Tensor::new(t.dequantize::<R>())),
            QMatrix::CpuQ5_0(t) => Tensor::Cpu(fusor_cpu::Tensor::new(t.dequantize::<R>())),
            QMatrix::CpuQ8_0(t) => Tensor::Cpu(fusor_cpu::Tensor::new(t.dequantize::<R>())),
            QMatrix::CpuQ4K(t) => Tensor::Cpu(fusor_cpu::Tensor::new(t.dequantize::<R>())),
            QMatrix::CpuQ5K(t) => Tensor::Cpu(fusor_cpu::Tensor::new(t.dequantize::<R>())),
            QMatrix::CpuQ6K(t) => Tensor::Cpu(fusor_cpu::Tensor::new(t.dequantize::<R>())),
            QMatrix::CpuF32(t) => {
                let shape = t.shape();
                assert_eq!(
                    shape.len(),
                    R,
                    "CpuF32 rank {} doesn't match expected rank {}",
                    shape.len(),
                    R
                );
                let arr_shape: [usize; R] = shape.try_into().unwrap();
                let concrete = fusor_cpu::ConcreteTensor::from_parts(
                    Layout::contiguous(&arr_shape),
                    t.data().clone(),
                );
                Tensor::Cpu(fusor_cpu::Tensor::new(concrete))
            }
            QMatrix::CpuF16(t) => {
                let shape = t.shape();
                assert_eq!(
                    shape.len(),
                    R,
                    "CpuF16 rank {} doesn't match expected rank {}",
                    shape.len(),
                    R
                );
                let arr_shape: [usize; R] = shape.try_into().unwrap();
                // Convert f16 to f32
                let f32_data: Vec<f32> = t.data().iter().map(|v| v.to_f32()).collect();
                let mut data = AVec::<f32>::with_capacity(64, f32_data.len());
                data.extend_from_slice(&f32_data);
                let concrete = fusor_cpu::ConcreteTensor::from_parts(
                    Layout::contiguous(&arr_shape),
                    data.into_boxed_slice(),
                );
                Tensor::Cpu(fusor_cpu::Tensor::new(concrete))
            }
            QMatrix::Gpu(m) => Tensor::Gpu(m.dequantize()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qgpuor_cpu_creation_from_raw_bytes() {
        // Create a Q8_0 tensor on CPU using from_raw_bytes
        // Q8_0 has block size 32, so a [2, 64] tensor needs 4 blocks
        // Each Q8_0 block is 34 bytes (2 for f16 scale + 32 for i8 data)
        let shape = [2, 64];
        let num_blocks = 4;
        let block_size_bytes = 34;

        let mut raw_bytes = vec![0u8; num_blocks * block_size_bytes];
        for block_idx in 0..num_blocks {
            let offset = block_idx * block_size_bytes;
            // Set scale to 1.0 as f16
            let scale_f16 = half::f16::from_f32(1.0);
            raw_bytes[offset..offset + 2].copy_from_slice(&scale_f16.to_le_bytes());
            // Set data to sequential values
            for i in 0..32 {
                raw_bytes[offset + 2 + i] = ((block_idx * 32 + i) % 128) as u8;
            }
        }

        let qgpuor: QMatrix =
            QMatrix::from_raw_bytes(&Device::Cpu, shape, &raw_bytes, GgmlType::Q8_0).unwrap();

        assert!(qgpuor.is_cpu());
        assert!(!qgpuor.is_gpu());
        assert_eq!(qgpuor.ggml_type(), GgmlType::Q8_0);
        assert_eq!(qgpuor.shape(), &[2, 64]);
        assert!(matches!(qgpuor.device(), Device::Cpu));
    }

    #[test]
    fn test_qgpuor_dequantize_cpu() {
        // Create a simple Q8_0 tensor and verify dequantization
        // Q8_0: scale * data[i] = output
        let shape = [1, 32];
        let block_size_bytes = 34;

        let mut raw_bytes = vec![0u8; block_size_bytes];
        // Set scale to 0.5 as f16
        let scale_f16 = half::f16::from_f32(0.5);
        raw_bytes[0..2].copy_from_slice(&scale_f16.to_le_bytes());
        // Set data to 0, 1, 2, ..., 31 (as signed i8)
        for i in 0..32 {
            raw_bytes[2 + i] = i as u8;
        }

        let qgpuor: QMatrix =
            QMatrix::from_raw_bytes(&Device::Cpu, shape, &raw_bytes, GgmlType::Q8_0).unwrap();

        let dequantized = qgpuor.dequantize::<2>();
        assert!(dequantized.is_cpu());

        // Verify the dequantized values
        let cpu_result = dequantized.unwrap_cpu();
        for i in 0..32 {
            let expected = 0.5 * (i as f32);
            let actual = cpu_result.get([0, i]);
            assert!(
                (actual - expected).abs() < 1e-3,
                "Mismatch at index {}: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_cpu_qmatmul_simple() {
        // Test CPU qmatmul with a simple known input
        // Weight matrix [N, K] = [2, 32] (2 output features, 32 input features)
        // Input [M, K] = [1, 32] (1 sample, 32 features)
        // Output should be [M, N] = [1, 2]
        //
        // Weight row 0: all 1s -> dot product with input = sum(input)
        // Weight row 1: all 2s -> dot product with input = 2 * sum(input)

        let shape = [2, 32]; // [N, K] = [out_features, in_features]
        let block_size_bytes = 34; // Q8_0 block size

        // Create 2 blocks (one per row since K=32=BLOCK_SIZE)
        let mut raw_bytes = vec![0u8; 2 * block_size_bytes];

        // Block 0 (row 0): scale=1.0, data=all 1s
        let scale_f16 = half::f16::from_f32(1.0);
        raw_bytes[0..2].copy_from_slice(&scale_f16.to_le_bytes());
        for i in 0..32 {
            raw_bytes[2 + i] = 1i8 as u8; // all 1s
        }

        // Block 1 (row 1): scale=1.0, data=all 2s
        raw_bytes[block_size_bytes..block_size_bytes + 2].copy_from_slice(&scale_f16.to_le_bytes());
        for i in 0..32 {
            raw_bytes[block_size_bytes + 2 + i] = 2i8 as u8; // all 2s
        }

        let qmatrix: QMatrix =
            QMatrix::from_raw_bytes(&Device::Cpu, shape, &raw_bytes, GgmlType::Q8_0).unwrap();

        // Create input tensor [1, 32] with values 0.5 for each element
        let input_data: Vec<f32> = vec![0.5; 32];
        let input: Tensor<2, f32> = Tensor::from_slice(&Device::Cpu, [1, 32], &input_data);

        // Perform qmatmul
        let output = input.q_mat_mul(&qmatrix);

        // Expected:
        // output[0, 0] = dot(input, weight_row_0) = sum(0.5 * 1.0) for 32 elements = 16.0
        // output[0, 1] = dot(input, weight_row_1) = sum(0.5 * 2.0) for 32 elements = 32.0
        let cpu_result = output.unwrap_cpu();
        let result_0 = cpu_result.get([0, 0]);
        let result_1 = cpu_result.get([0, 1]);

        println!("result[0, 0] = {} (expected 16.0)", result_0);
        println!("result[0, 1] = {} (expected 32.0)", result_1);

        assert!(
            (result_0 - 16.0).abs() < 0.1,
            "result[0, 0] = {}, expected 16.0",
            result_0
        );
        assert!(
            (result_1 - 32.0).abs() < 0.1,
            "result[0, 1] = {}, expected 32.0",
            result_1
        );
    }

    #[test]
    fn test_cpu_f32_qmatmul() {
        // Test F32 (non-quantized) qmatmul
        // Weight matrix [N, K] = [2, 4] (2 output features, 4 input features)
        // Input [M, K] = [1, 4] (1 sample, 4 features)
        // Output should be [M, N] = [1, 2]

        let shape = [2, 4]; // [N, K] = [out_features, in_features]

        // Create F32 weight data:
        // Row 0: [1, 2, 3, 4]
        // Row 1: [5, 6, 7, 8]
        let weight_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let weight_bytes: Vec<u8> = weight_data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let qmatrix: QMatrix =
            QMatrix::from_raw_bytes(&Device::Cpu, shape, &weight_bytes, GgmlType::F32).unwrap();

        // Create input tensor [1, 4] with values [1, 1, 1, 1]
        let input_data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
        let input: Tensor<2, f32> = Tensor::from_slice(&Device::Cpu, [1, 4], &input_data);

        // Perform qmatmul
        let output = input.q_mat_mul(&qmatrix);

        // Expected:
        // output[0, 0] = dot([1,1,1,1], [1,2,3,4]) = 1+2+3+4 = 10
        // output[0, 1] = dot([1,1,1,1], [5,6,7,8]) = 5+6+7+8 = 26
        let cpu_result = output.unwrap_cpu();
        let result_0 = cpu_result.get([0, 0]);
        let result_1 = cpu_result.get([0, 1]);

        println!("F32 qmatmul result[0, 0] = {} (expected 10.0)", result_0);
        println!("F32 qmatmul result[0, 1] = {} (expected 26.0)", result_1);

        assert!(
            (result_0 - 10.0).abs() < 0.1,
            "result[0, 0] = {}, expected 10.0",
            result_0
        );
        assert!(
            (result_1 - 26.0).abs() < 0.1,
            "result[0, 1] = {}, expected 26.0",
            result_1
        );
    }
}
