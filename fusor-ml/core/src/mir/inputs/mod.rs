use crate::TensorData;
use crate::quantized::QMatrix;

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
}

impl From<QMatrix> for MirValue {
    fn from(value: QMatrix) -> Self {
        Self::QMatrix(value)
    }
}

impl From<&QMatrix> for MirValue {
    fn from(value: &QMatrix) -> Self {
        Self::QMatrix(value.clone())
    }
}

impl From<TensorData> for MirValue {
    fn from(value: TensorData) -> Self {
        Self::Tensor(value)
    }
}

impl From<&TensorData> for MirValue {
    fn from(value: &TensorData) -> Self {
        Self::Tensor(value.clone())
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
