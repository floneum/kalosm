use std::{
    any::Any,
    iter::Sum,
    ops::{Add, Div, Mul, Neg, Rem, Sub},
};

use crate::{
    Tensor,
    nary_wise::{NaryFunction, NaryOp, NaryScalar},
    tensor::{DataType, DataTypeEnum},
};

fn scalar_value<T: DataType>(value: &T) -> NaryScalar {
    let value = value as &dyn Any;
    if let Some(value) = value.downcast_ref::<f32>() {
        NaryScalar::F32(*value)
    } else if let Some(value) = value.downcast_ref::<half::f16>() {
        NaryScalar::F16(*value)
    } else if let Some(value) = value.downcast_ref::<u32>() {
        NaryScalar::U32(*value)
    } else {
        unreachable!("all fusor-core DataType implementations are covered")
    }
}

fn scalar_op<T: DataType>(input: &Tensor, name: &'static str, operation: NaryOp) -> Tensor {
    input.assert_datatype::<T>();
    input.unary_nary::<T>(NaryFunction::unary(
        Some(name.to_string()),
        operation,
        T::DATA_TYPE,
        T::DATA_TYPE,
    ))
}

fn unary_same(input: &Tensor, name: &'static str, operation: NaryOp) -> Tensor {
    input.unary_nary_dtype(NaryFunction::unary(
        Some(name.to_string()),
        operation,
        input.datatype(),
        input.datatype(),
    ))
}

impl<T: DataType> Add<T> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: T) -> Self::Output {
        scalar_op::<T>(&self, "add_const", NaryOp::AddConst(scalar_value(&rhs)))
    }
}

impl<T: DataType> Add<T> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: T) -> Self::Output {
        self.clone() + rhs
    }
}

impl Sum for Tensor {
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let first = iter.next().expect("Cannot sum over empty iterator");
        iter.fold(first, |acc, x| acc + x)
    }
}

impl<'a> Sum<&'a Tensor> for Tensor {
    fn sum<I: Iterator<Item = &'a Tensor>>(iter: I) -> Self {
        let mut iter = iter.cloned();
        let first = iter.next().expect("Cannot sum over empty iterator");
        iter.fold(first, |acc, x| acc + x)
    }
}

macro_rules! impl_lhs_scalar {
    ($trait:ident, $method:ident, $op:expr, $($t:ty),*) => {
        $(
            impl $trait<Tensor> for $t {
                type Output = Tensor;

                fn $method(self, rhs: Tensor) -> Self::Output {
                    scalar_op::<$t>(&rhs, stringify!($method), $op(scalar_value(&self)))
                }
            }
        )*
    };
}

impl_lhs_scalar!(Add, add, NaryOp::AddConst, f32, half::f16, u32);
impl_lhs_scalar!(Sub, sub, NaryOp::RSubConst, f32, half::f16, u32);
impl_lhs_scalar!(Mul, mul, NaryOp::MulConst, f32, half::f16, u32);
impl_lhs_scalar!(Div, div, NaryOp::RDivConst, f32, half::f16, u32);
impl_lhs_scalar!(Rem, rem, NaryOp::RRemConst, f32, half::f16, u32);

impl<T: DataType> Sub<T> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: T) -> Self::Output {
        scalar_op::<T>(
            &self,
            "subtract_const",
            NaryOp::SubConst(scalar_value(&rhs)),
        )
    }
}

impl<T: DataType> Sub<T> for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: T) -> Self::Output {
        self.clone() - rhs
    }
}

impl<T: DataType> Mul<T> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: T) -> Self::Output {
        scalar_op::<T>(
            &self,
            "multiply_const",
            NaryOp::MulConst(scalar_value(&rhs)),
        )
    }
}

impl<T: DataType> Mul<T> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: T) -> Self::Output {
        self.clone() * rhs
    }
}

impl<T: DataType> Div<T> for Tensor {
    type Output = Tensor;

    fn div(self, rhs: T) -> Self::Output {
        scalar_op::<T>(&self, "divide_const", NaryOp::DivConst(scalar_value(&rhs)))
    }
}

impl<T: DataType> Div<T> for &Tensor {
    type Output = Tensor;

    fn div(self, rhs: T) -> Self::Output {
        self.clone() / rhs
    }
}

impl Rem<u32> for Tensor {
    type Output = Tensor;

    fn rem(self, rhs: u32) -> Self::Output {
        scalar_op::<u32>(&self, "mod_const", NaryOp::RemConst(NaryScalar::U32(rhs)))
    }
}

impl Tensor {
    /// Check if each value in the tensor is equal to the given value. Returns 1 for true and 0 for false.
    pub fn eq<D: DataType, T: DataType>(&self, rhs: T) -> Tensor {
        self.unary_nary::<D>(NaryFunction::unary(
            Some("equal_const".to_string()),
            NaryOp::EqualConst(scalar_value(&rhs)),
            T::DATA_TYPE,
            D::DATA_TYPE,
        ))
    }

    /// Check if each value in the tensor is less than the given value. Returns 1 for true and 0 for false.
    pub fn lt<D: DataType, T: DataType>(&self, rhs: T) -> Tensor {
        self.unary_nary::<D>(NaryFunction::unary(
            Some("lt_const".to_string()),
            NaryOp::LessConst(scalar_value(&rhs)),
            T::DATA_TYPE,
            D::DATA_TYPE,
        ))
    }

    /// Check if each value in the tensor is less than or equal to the given value.
    pub fn lte<D: DataType, T: DataType>(&self, rhs: T) -> Tensor {
        self.unary_nary::<D>(NaryFunction::unary(
            Some("lte_const".to_string()),
            NaryOp::LessEqualConst(scalar_value(&rhs)),
            T::DATA_TYPE,
            D::DATA_TYPE,
        ))
    }

    /// Check if each value in the tensor is greater than the given value.
    pub fn mt<D: DataType, T: DataType>(&self, rhs: T) -> Tensor {
        self.unary_nary::<D>(NaryFunction::unary(
            Some("mt_const".to_string()),
            NaryOp::GreaterConst(scalar_value(&rhs)),
            T::DATA_TYPE,
            D::DATA_TYPE,
        ))
    }

    /// Check if each value in the tensor is greater than or equal to the given value.
    pub fn mte<D: DataType, T: DataType>(&self, rhs: T) -> Tensor {
        self.unary_nary::<D>(NaryFunction::unary(
            Some("mte_const".to_string()),
            NaryOp::GreaterEqualConst(scalar_value(&rhs)),
            T::DATA_TYPE,
            D::DATA_TYPE,
        ))
    }

    pub fn less_approximate_exp(&self) -> Self {
        if self.datatype() != DataTypeEnum::F32 {
            return self.exp();
        }
        unary_same(self, "less_approximate_exp", NaryOp::LessApproximateExp)
    }

    pub fn approximate_exp(&self) -> Self {
        if self.datatype() != DataTypeEnum::F32 {
            return self.exp();
        }
        unary_same(self, "approximate_exp", NaryOp::ApproximateExp)
    }

    pub fn exp(&self) -> Self {
        unary_same(self, "exp", NaryOp::Exp)
    }

    pub fn exp2(&self) -> Self {
        unary_same(self, "exp2", NaryOp::Exp2)
    }

    pub fn log(&self) -> Self {
        unary_same(self, "log", NaryOp::Log)
    }

    pub fn log2(&self) -> Self {
        unary_same(self, "log2", NaryOp::Log2)
    }

    pub fn pow_elementwise<D: DataType>(&self, exponent: D) -> Self {
        scalar_op::<D>(self, "pow", NaryOp::PowConst(scalar_value(&exponent)))
    }

    pub fn sqrt(&self) -> Self {
        unary_same(self, "sqrt", NaryOp::Sqrt)
    }

    pub fn sin(&self) -> Self {
        unary_same(self, "sin", NaryOp::Sin)
    }

    pub fn cos(&self) -> Self {
        unary_same(self, "cos", NaryOp::Cos)
    }

    pub fn tan(&self) -> Self {
        unary_same(self, "tan", NaryOp::Tan)
    }

    pub fn asin(&self) -> Self {
        unary_same(self, "asin", NaryOp::Asin)
    }

    pub fn acos(&self) -> Self {
        unary_same(self, "acos", NaryOp::Acos)
    }

    pub fn atan(&self) -> Self {
        unary_same(self, "atan", NaryOp::Atan)
    }

    pub fn sinh(&self) -> Self {
        unary_same(self, "sinh", NaryOp::Sinh)
    }

    pub fn cosh(&self) -> Self {
        unary_same(self, "cosh", NaryOp::Cosh)
    }

    pub fn tanh(&self) -> Self {
        unary_same(self, "tanh", NaryOp::Tanh)
    }

    /// Calculates tanh with (e^x - e^-x) / (e^x + e^-x)
    pub fn tanh_exact(&self) -> Self {
        unary_same(self, "tanh_exact", NaryOp::TanhExact)
    }

    pub fn asinh(&self) -> Self {
        unary_same(self, "asinh", NaryOp::Asinh)
    }

    pub fn acosh(&self) -> Self {
        unary_same(self, "acosh", NaryOp::Acosh)
    }

    pub fn atanh(&self) -> Self {
        unary_same(self, "atanh", NaryOp::Atanh)
    }

    pub fn abs(&self) -> Self {
        unary_same(self, "abs", NaryOp::Abs)
    }

    pub fn max_elementwise<D: DataType>(&self, element: D) -> Self {
        scalar_op::<D>(self, "max", NaryOp::MaxConst(scalar_value(&element)))
    }

    pub fn min_elementwise<D: DataType>(&self, element: D) -> Self {
        scalar_op::<D>(self, "min", NaryOp::MinConst(scalar_value(&element)))
    }

    pub fn cast<T2: DataType>(&self) -> Tensor {
        self.cast_to(T2::DATA_TYPE)
    }

    pub fn cast_to(&self, datatype: DataTypeEnum) -> Tensor {
        if self.datatype() == datatype {
            return self.clone();
        }
        self.unary_nary_dtype(NaryFunction::unary(
            Some("cast".to_string()),
            NaryOp::Cast,
            self.datatype(),
            datatype,
        ))
    }
}

impl Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Self {
        unary_same(&self, "neg", NaryOp::Neg)
    }
}

impl Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        -self.clone()
    }
}

pub trait CastTensor<T>: Sized {
    /// Casts the tensor to another type.
    fn cast(tensor: &Tensor) -> Tensor;
}

impl<T: DataType> CastTensor<T> for T {
    fn cast(tensor: &Tensor) -> Tensor {
        tensor.assert_datatype::<T>();
        tensor.clone()
    }
}

impl CastTensor<f32> for u32 {
    fn cast(tensor: &Tensor) -> Tensor {
        tensor.assert_datatype::<u32>();
        tensor.cast_to(DataTypeEnum::F32)
    }
}

impl CastTensor<half::f16> for u32 {
    fn cast(tensor: &Tensor) -> Tensor {
        tensor.assert_datatype::<u32>();
        tensor.cast_to(DataTypeEnum::F16)
    }
}

impl CastTensor<half::f16> for f32 {
    fn cast(tensor: &Tensor) -> Tensor {
        tensor.assert_datatype::<f32>();
        tensor.cast_to(DataTypeEnum::F16)
    }
}

impl CastTensor<f32> for half::f16 {
    fn cast(tensor: &Tensor) -> Tensor {
        tensor.assert_datatype::<half::f16>();
        tensor.cast_to(DataTypeEnum::F32)
    }
}
