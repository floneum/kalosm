use crate::TensorData;

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum MirValue {
    Tensor(TensorData),
    Integer(u32),
    Float(f32),
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
