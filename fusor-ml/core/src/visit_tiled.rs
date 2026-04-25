use crate::{DataTypeEnum, Layout, TensorData, mir::inputs::MirValue};

#[derive(Clone, Debug)]
pub(crate) enum MaybeQData {
    Tensor(TensorData),
}

impl MaybeQData {
    pub(crate) fn device(&self) -> &crate::Device {
        match self {
            MaybeQData::Tensor(tensor) => tensor.device(),
        }
    }

    pub(crate) fn layout(&self) -> Layout {
        match self {
            MaybeQData::Tensor(tensor) => tensor.layout().clone(),
        }
    }

    pub(crate) fn datatype(&self) -> DataTypeEnum {
        match self {
            MaybeQData::Tensor(tensor) => tensor.datatype(),
        }
    }

    pub(crate) fn owned(&self) -> bool {
        match self {
            MaybeQData::Tensor(tensor) => tensor.owned(),
        }
    }
}

impl From<TensorData> for MaybeQData {
    fn from(tensor: TensorData) -> Self {
        Self::Tensor(tensor)
    }
}

impl From<&TensorData> for MaybeQData {
    fn from(tensor: &TensorData) -> Self {
        Self::Tensor(tensor.clone())
    }
}

impl From<MaybeQData> for MirValue {
    fn from(val: MaybeQData) -> Self {
        match val {
            MaybeQData::Tensor(tensor) => MirValue::Tensor(tensor),
        }
    }
}

impl TryFrom<MirValue> for MaybeQData {
    type Error = ();

    fn try_from(value: MirValue) -> Result<Self, Self::Error> {
        match value {
            MirValue::Tensor(tensor) => Ok(MaybeQData::Tensor(tensor)),
            _ => Err(()),
        }
    }
}
