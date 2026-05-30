//! Dynamic CPU tensor storage.
//!
//! This is a rank- and dtype-erased storage type for the CPU backend.  The
//! existing typed `ConcreteTensor<T, R>` remains the execution/fusion backing
//! for now; this type provides the storage shape needed to move rank and dtype
//! ownership up into the `fusor` facade without changing the SIMD expression
//! system in the same step.

use std::fmt::{Debug, Display};

use aligned_vec::{ABox, AVec};
use fusor_types::Layout;

use crate::{ConcreteTensor, SimdElement};

/// Runtime element type for CPU tensor storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CpuDType {
    F16,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

impl CpuDType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
        }
    }

    pub const fn element_size(self) -> usize {
        match self {
            Self::F16 => size_of::<half::f16>(),
            Self::F32 => size_of::<f32>(),
            Self::F64 => size_of::<f64>(),
            Self::I8 => size_of::<i8>(),
            Self::I16 => size_of::<i16>(),
            Self::I32 => size_of::<i32>(),
            Self::I64 => size_of::<i64>(),
            Self::U8 => size_of::<u8>(),
            Self::U16 => size_of::<u16>(),
            Self::U32 => size_of::<u32>(),
            Self::U64 => size_of::<u64>(),
        }
    }

    pub const fn of<T: CpuElement>() -> Self {
        T::DTYPE
    }
}

impl Display for CpuDType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// SIMD element types that can be stored in a dynamic CPU tensor.
pub trait CpuElement: SimdElement + 'static {
    const DTYPE: CpuDType;
}

macro_rules! impl_cpu_element {
    ($($ty:ty => $dtype:ident),* $(,)?) => {
        $(
            impl CpuElement for $ty {
                const DTYPE: CpuDType = CpuDType::$dtype;
            }
        )*
    };
}

impl_cpu_element! {
    half::f16 => F16,
    f32 => F32,
    f64 => F64,
    i8 => I8,
    i16 => I16,
    i32 => I32,
    i64 => I64,
    u8 => U8,
    u16 => U16,
    u32 => U32,
    u64 => U64,
}

/// Errors returned when dynamically typed CPU storage is viewed as a typed tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DynamicTensorError {
    DTypeMismatch {
        expected: CpuDType,
        actual: CpuDType,
    },
    RankMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidByteLength {
        dtype: CpuDType,
        byte_len: usize,
    },
    BufferTooSmall {
        required_bytes: usize,
        actual_bytes: usize,
    },
}

impl Display for DynamicTensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DTypeMismatch { expected, actual } => {
                write!(f, "expected CPU tensor dtype {expected}, got {actual}")
            }
            Self::RankMismatch { expected, actual } => {
                write!(f, "expected CPU tensor rank {expected}, got {actual}")
            }
            Self::InvalidByteLength { dtype, byte_len } => {
                write!(
                    f,
                    "byte length {byte_len} is not valid for CPU tensor dtype {dtype}"
                )
            }
            Self::BufferTooSmall {
                required_bytes,
                actual_bytes,
            } => {
                write!(
                    f,
                    "CPU tensor buffer has {actual_bytes} bytes, but layout requires {required_bytes}"
                )
            }
        }
    }
}

impl std::error::Error for DynamicTensorError {}

/// Rank- and dtype-erased CPU tensor storage.
#[derive(Clone)]
pub struct DynamicTensor {
    dtype: CpuDType,
    layout: Layout,
    bytes: ABox<[u8]>,
}

impl Debug for DynamicTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynamicTensor")
            .field("dtype", &self.dtype)
            .field("shape", &self.layout.shape())
            .field("strides", &self.layout.strides())
            .field("offset", &self.layout.offset())
            .field("byte_len", &self.bytes.len())
            .finish()
    }
}

impl DynamicTensor {
    /// Create dynamic storage from a typed slice and runtime shape.
    pub fn from_slice<T: CpuElement>(shape: impl AsRef<[usize]>, data: &[T]) -> Self {
        let layout = Layout::contiguous(shape.as_ref());
        assert_eq!(
            layout.num_elements(),
            data.len(),
            "Data length must match shape"
        );
        Self::from_typed_parts(layout, data)
    }

    /// Create dynamic storage from a typed concrete tensor.
    #[allow(dead_code)]
    pub(crate) fn from_concrete<T: CpuElement, const R: usize>(
        tensor: &ConcreteTensor<T, R>,
    ) -> Self {
        Self::from_typed_parts(tensor.layout().clone(), tensor.backing())
    }

    /// Create dynamic storage from an existing byte buffer.
    pub fn from_bytes(
        dtype: CpuDType,
        layout: Layout,
        bytes: &[u8],
    ) -> Result<Self, DynamicTensorError> {
        Self::validate_bytes(dtype, &layout, bytes.len())?;
        Ok(Self {
            dtype,
            layout,
            bytes: aligned_byte_copy(bytes),
        })
    }

    pub fn dtype(&self) -> CpuDType {
        self.dtype
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    pub fn rank(&self) -> usize {
        self.layout.rank()
    }

    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    pub fn num_elements(&self) -> usize {
        self.layout.num_elements()
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// View the backing bytes as typed storage.
    ///
    /// For non-contiguous layouts this returns the physical backing storage,
    /// not a logical contiguous view. Use `to_concrete` to recover the typed
    /// layout-aware tensor wrapper.
    pub fn as_typed_slice<T: CpuElement>(&self) -> Result<&[T], DynamicTensorError> {
        self.check_dtype::<T>()?;
        bytemuck::try_cast_slice(&self.bytes).map_err(|_| DynamicTensorError::InvalidByteLength {
            dtype: self.dtype,
            byte_len: self.bytes.len(),
        })
    }

    /// Convert dynamic storage back to the existing typed concrete tensor.
    #[allow(dead_code)]
    pub(crate) fn to_concrete<T: CpuElement, const R: usize>(
        &self,
    ) -> Result<ConcreteTensor<T, R>, DynamicTensorError> {
        self.check_rank::<R>()?;
        let data = self.as_typed_slice::<T>()?;
        let mut vec: AVec<T> = AVec::with_capacity(64, data.len());
        vec.extend_from_slice(data);
        Ok(ConcreteTensor::from_parts(
            self.layout.clone(),
            vec.into_boxed_slice(),
        ))
    }

    fn from_typed_parts<T: CpuElement>(layout: Layout, data: &[T]) -> Self {
        let bytes = bytemuck::cast_slice(data);
        Self {
            dtype: T::DTYPE,
            layout,
            bytes: aligned_byte_copy(bytes),
        }
    }

    fn validate_bytes(
        dtype: CpuDType,
        layout: &Layout,
        byte_len: usize,
    ) -> Result<(), DynamicTensorError> {
        let element_size = dtype.element_size();
        if !byte_len.is_multiple_of(element_size) {
            return Err(DynamicTensorError::InvalidByteLength { dtype, byte_len });
        }

        let required_bytes = required_element_capacity(layout) * element_size;
        if byte_len < required_bytes {
            return Err(DynamicTensorError::BufferTooSmall {
                required_bytes,
                actual_bytes: byte_len,
            });
        }

        Ok(())
    }

    fn check_dtype<T: CpuElement>(&self) -> Result<(), DynamicTensorError> {
        let expected = T::DTYPE;
        if self.dtype != expected {
            return Err(DynamicTensorError::DTypeMismatch {
                expected,
                actual: self.dtype,
            });
        }
        Ok(())
    }

    fn check_rank<const R: usize>(&self) -> Result<(), DynamicTensorError> {
        let actual = self.rank();
        if actual != R {
            return Err(DynamicTensorError::RankMismatch {
                expected: R,
                actual,
            });
        }
        Ok(())
    }
}

fn aligned_byte_copy(bytes: &[u8]) -> ABox<[u8]> {
    let mut vec = AVec::with_capacity(64, bytes.len());
    vec.extend_from_slice(bytes);
    vec.into_boxed_slice()
}

fn required_element_capacity(layout: &Layout) -> usize {
    if layout.num_elements() == 0 {
        return 0;
    }

    layout.offset()
        + layout
            .shape()
            .iter()
            .zip(layout.strides())
            .map(|(dim, stride)| (dim - 1) * stride)
            .sum::<usize>()
        + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_dynamic_tensor_from_typed_slice() {
        let tensor = DynamicTensor::from_slice([2, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);

        assert_eq!(tensor.dtype(), CpuDType::F32);
        assert_eq!(tensor.rank(), 2);
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.as_typed_slice::<f32>().unwrap()[4], 5.0);
    }

    #[test]
    fn round_trips_through_concrete_tensor() {
        let concrete = ConcreteTensor::from_slice([2, 2], &[1u32, 2, 3, 4]);
        let dynamic = DynamicTensor::from_concrete(&concrete);
        let round_trip = dynamic.to_concrete::<u32, 2>().unwrap();

        assert_eq!(round_trip.layout().shape(), &[2, 2]);
        assert_eq!(round_trip.get([1, 0]), 3);
    }

    #[test]
    fn rejects_wrong_dtype() {
        let tensor = DynamicTensor::from_slice([2], &[1.0f32, 2.0]);
        let err = tensor.as_typed_slice::<u32>().unwrap_err();

        assert_eq!(
            err,
            DynamicTensorError::DTypeMismatch {
                expected: CpuDType::U32,
                actual: CpuDType::F32,
            }
        );
    }

    #[test]
    fn rejects_wrong_rank() {
        let tensor = DynamicTensor::from_slice([2, 2], &[1u32, 2, 3, 4]);
        let err = match tensor.to_concrete::<u32, 1>() {
            Ok(_) => panic!("expected rank mismatch"),
            Err(err) => err,
        };

        assert_eq!(
            err,
            DynamicTensorError::RankMismatch {
                expected: 1,
                actual: 2,
            }
        );
    }

    #[test]
    fn validates_raw_byte_storage_size() {
        let layout = Layout::contiguous(&[2usize]);
        let err = DynamicTensor::from_bytes(CpuDType::F32, layout, &[0, 1, 2]).unwrap_err();

        assert_eq!(
            err,
            DynamicTensorError::InvalidByteLength {
                dtype: CpuDType::F32,
                byte_len: 3,
            }
        );
    }
}
