use std::fmt::Display;
use std::sync::Arc;

use crate::quantized::QMatrix;
use crate::{DataTypeEnum, TensorData, TensorLayoutInfo};

mod float;
mod integer;
mod qmatrix;
mod tensor;

pub(crate) use float::FloatInput;
use fusor_gguf::GgmlType;
pub(crate) use integer::IntegerInput;
pub(crate) use qmatrix::QMatrixInput;
pub(crate) use tensor::TensorInput;

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum KernelInputValue {
    QBuffer(Arc<wgpu::Buffer>),
    QInfo(Box<[usize]>),
    TensorBuffer(Arc<wgpu::Buffer>),
    TensorInfo(TensorLayoutInfo),
    Integer(u32),
    Float(f32),
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum MirValue {
    QMatrix(QMatrix),
    Tensor(TensorData),
    Integer(u32),
    Float(f32),
}

impl MirValue {
    pub(crate) fn as_tensor(&self) -> Option<&TensorData> {
        match self {
            MirValue::Tensor(tensor) => Some(tensor),
            _ => None,
        }
    }

    pub(crate) fn visit_input_values<F>(&self, mut f: F)
    where
        F: FnMut(KernelInputValue),
    {
        match self {
            MirValue::QMatrix(matrix) => {
                f(KernelInputValue::QBuffer(matrix.buffer().clone()));
                f(KernelInputValue::QInfo(matrix.shape().into()));
            }
            MirValue::Tensor(tensor) => {
                f(KernelInputValue::TensorBuffer(tensor.buffer().clone()));
                f(KernelInputValue::TensorInfo(tensor.info().clone()));
            }
            MirValue::Integer(integer) => {
                f(KernelInputValue::Integer(*integer));
            }
            MirValue::Float(float) => {
                f(KernelInputValue::Float(*float));
            }
        }
    }
}

impl From<QMatrix> for MirValue {
    fn from(value: QMatrix) -> Self {
        Self::QMatrix(value)
    }
}

impl From<TensorData> for MirValue {
    fn from(value: TensorData) -> Self {
        Self::Tensor(value)
    }
}

impl From<u32> for MirValue {
    fn from(value: u32) -> Self {
        Self::Integer(value)
    }
}

impl From<f32> for MirValue {
    fn from(value: f32) -> Self {
        Self::Float(value)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct KernelInput {
    pub(crate) ty: KernelInputType,
}

impl Display for KernelInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.ty {
            KernelInputType::QBuffer(matrix) => {
                let start_index = matrix.matrix_binding;
                let datatype = matrix.datatype;
                writeln!(
                    f,
                    "@group(0) @binding({start_index}) var<storage, read> i_{start_index}: array<{datatype}>;"
                )?;
            }
            KernelInputType::QInfo(matrix) => {
                let info_index = matrix.info_binding;
                writeln!(f, "struct Tensor{info_index}Info {{")?;
                if matrix.rank == 0 {
                    // WGSL structs must have at least one member; the field is
                    // unused for scalar tensors but keeps the type definition
                    // valid.
                    writeln!(f, "    _unused: u32,")?;
                } else {
                    for i in 0..matrix.rank {
                        writeln!(f, "    shape_{i}: u32,")?;
                    }
                }
                writeln!(f, "}};")?;

                writeln!(
                    f,
                    "@group(0) @binding({info_index}) var<uniform> i_{info_index}: Tensor{info_index}Info;"
                )?;
            }
            KernelInputType::TensorBuffer(tensor) => {
                let start_index = tensor.tensor_binding;
                let datatype = tensor.datatype;
                write!(f, "@group(0) @binding({start_index}) ")?;

                if tensor.mutable {
                    write!(f, "var<storage, read_write> ")?;
                } else {
                    write!(f, "var<storage, read> ")?;
                }

                writeln!(f, "i_{start_index}: array<{datatype}>;")?;
            }
            KernelInputType::TensorInfo(tensor) => {
                let info_index = tensor.info_binding;
                writeln!(f, "struct Tensor{info_index}Info {{")?;
                writeln!(f, "    offset: u32,")?;
                for i in 0..tensor.rank {
                    writeln!(f, "    stride_{i}: u32,")?;
                    writeln!(f, "    shape_{i}: u32,")?;
                }
                writeln!(f, "}};")?;

                writeln!(
                    f,
                    "@group(0) @binding({info_index}) var<uniform> i_{info_index}: Tensor{info_index}Info;"
                )?;
            }
            KernelInputType::Integer(integer) => {
                let index = integer.index;
                write!(
                    f,
                    "@group(0) @binding({index}) var<uniform> {integer}: u32;"
                )?
            }
            KernelInputType::Float(float) => {
                let index = float.index;
                write!(f, "@group(0) @binding({index}) var<uniform> {float}: f32;")?
            }
        }

        Ok(())
    }
}

#[derive(Clone, Debug)]
pub(crate) enum KernelInputType {
    QBuffer(QBufferInput),
    QInfo(QInfoInput),
    TensorBuffer(TensorBufferInput),
    TensorInfo(TensorInfoInput),
    Integer(IntegerInput),
    Float(FloatInput),
}

#[derive(Clone, Debug)]
pub(crate) struct TensorBufferInput {
    pub(crate) tensor_binding: u32,
    pub(crate) mutable: bool,
    pub(crate) datatype: DataTypeEnum,
}

#[derive(Clone, Debug)]
pub(crate) struct TensorInfoInput {
    pub(crate) info_binding: u32,
    pub(crate) rank: u32,
}

#[derive(Clone, Debug)]
pub(crate) struct QBufferInput {
    pub(crate) matrix_binding: u32,
    pub(crate) datatype: GgmlType,
}

#[derive(Clone, Debug)]
pub(crate) struct QInfoInput {
    pub(crate) info_binding: u32,
    pub(crate) rank: u32,
}
