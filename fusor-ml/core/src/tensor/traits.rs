use std::{
    fmt::{Debug, Display},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
};

use bytemuck::{AnyBitPattern, NoUninit};

pub trait DataType:
    Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Div<Output = Self>
    + DivAssign
    + PartialOrd
    + NoUninit
    + AnyBitPattern
    + Debug
    + Display
    + Send
    + Sync
    + 'static
{
    const DATA_TYPE: DataTypeEnum;

    fn zero() -> Self;
    fn one() -> Self;
}

pub trait FloatDataType: DataType {
    fn from_f32(value: f32) -> Self;

    fn is_finite(&self) -> bool;
}

impl DataType for f32 {
    const DATA_TYPE: DataTypeEnum = DataTypeEnum::F32;

    fn zero() -> Self {
        0.
    }

    fn one() -> Self {
        1.
    }
}

impl FloatDataType for f32 {
    fn from_f32(value: f32) -> Self {
        value
    }

    fn is_finite(&self) -> bool {
        f32::is_finite(*self)
    }
}

impl DataType for half::f16 {
    const DATA_TYPE: DataTypeEnum = DataTypeEnum::F16;

    fn zero() -> Self {
        half::f16::from_f32(0.)
    }

    fn one() -> Self {
        half::f16::from_f32(1.)
    }
}

impl FloatDataType for half::f16 {
    fn from_f32(value: f32) -> Self {
        half::f16::from_f32(value)
    }

    fn is_finite(&self) -> bool {
        half::f16::is_finite(*self)
    }
}

impl DataType for u32 {
    const DATA_TYPE: DataTypeEnum = DataTypeEnum::U32;

    fn zero() -> Self {
        0
    }

    fn one() -> Self {
        1
    }
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DataTypeEnum {
    F32,
    F16,
    U32,
}

impl DataTypeEnum {
    pub fn as_str(&self) -> &'static str {
        match self {
            DataTypeEnum::F32 => "f32",
            DataTypeEnum::F16 => "f16",
            DataTypeEnum::U32 => "u32",
        }
    }

    pub fn element_size(&self) -> usize {
        match self {
            DataTypeEnum::F32 => size_of::<f32>(),
            DataTypeEnum::F16 => size_of::<half::f16>(),
            DataTypeEnum::U32 => size_of::<u32>(),
        }
    }
}

impl Display for DataTypeEnum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
