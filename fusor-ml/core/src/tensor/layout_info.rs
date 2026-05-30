use std::fmt::Display;

use crate::Layout;

use super::DataTypeEnum;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TensorLayoutInfo {
    pub(crate) layout: Layout,
    pub(crate) datatype: DataTypeEnum,
}

impl Display for TensorLayoutInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} {}", self.layout.shape(), self.datatype)
    }
}

impl TensorLayoutInfo {
    pub(crate) fn new(layout: Layout, datatype: DataTypeEnum) -> Self {
        Self { layout, datatype }
    }

    pub(crate) fn layout(&self) -> &Layout {
        &self.layout
    }

    pub(crate) fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    pub(crate) fn datatype(&self) -> DataTypeEnum {
        self.datatype
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TensorInfo {
    pub(crate) shape: Box<[usize]>,
    pub(crate) datatype: DataTypeEnum,
}

impl Display for TensorInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} {}", self.shape, self.datatype)
    }
}

impl TensorInfo {
    pub(crate) fn new(shape: Box<[usize]>, datatype: DataTypeEnum) -> Self {
        Self { shape, datatype }
    }

    pub(crate) fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub(crate) fn rank(&self) -> usize {
        self.shape.len()
    }

    pub(crate) fn datatype(&self) -> DataTypeEnum {
        self.datatype
    }
}
