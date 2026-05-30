//! Unified CPU/GPU quantized tensor abstraction
//!
//! This module provides `QMatrix`, a runtime dispatch enum that holds either CPU
//! quantized tensors (`QuantizedTensor`) or GPU quantized matrices (`QMatrix`).

use crate::cpu::{
    ABox, AVec, BlockQ4_0, BlockQ4K, BlockQ5_0, BlockQ5K, BlockQ6K, BlockQ8_0, GgmlType, Layout,
    QuantizedTensor,
};
use crate::gpu::QMatrix as GpuQMatrix;
use crate::{Device, Tensor};
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
    pub fn concat_rows(matrices: &[&Self]) -> Option<Self> {
        let gpu_matrices = matrices
            .iter()
            .map(|matrix| match *matrix {
                QMatrix::Gpu(matrix) => Some(matrix),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()?;

        GpuQMatrix::concat_rows(&gpu_matrices).map(QMatrix::Gpu)
    }

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
    ) -> Result<Self, crate::gpu::GgufReadError> {
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
            QMatrix::CpuQ4_0(t) => Tensor::Cpu(crate::cpu::TypedTensor::new(t.dequantize::<R>())),
            QMatrix::CpuQ5_0(t) => Tensor::Cpu(crate::cpu::TypedTensor::new(t.dequantize::<R>())),
            QMatrix::CpuQ8_0(t) => Tensor::Cpu(crate::cpu::TypedTensor::new(t.dequantize::<R>())),
            QMatrix::CpuQ4K(t) => Tensor::Cpu(crate::cpu::TypedTensor::new(t.dequantize::<R>())),
            QMatrix::CpuQ5K(t) => Tensor::Cpu(crate::cpu::TypedTensor::new(t.dequantize::<R>())),
            QMatrix::CpuQ6K(t) => Tensor::Cpu(crate::cpu::TypedTensor::new(t.dequantize::<R>())),
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
                let concrete = crate::cpu::ConcreteTensor::from_parts(
                    Layout::contiguous(&arr_shape),
                    t.data().clone(),
                );
                Tensor::Cpu(crate::cpu::TypedTensor::new(concrete))
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
                let concrete = crate::cpu::ConcreteTensor::from_parts(
                    Layout::contiguous(&arr_shape),
                    data.into_boxed_slice(),
                );
                Tensor::Cpu(crate::cpu::TypedTensor::new(concrete))
            }
            QMatrix::Gpu(m) => Tensor::Gpu(crate::GpuTensor::from_core(m.dequantize::<f32>())),
        }
    }
}
