use std::{
    iter::Sum,
    ops::{Add, Div, Mul, Neg, Rem, Sub},
};

use crate::{
    Tensor,
    nary_wise::NaryFunction,
    tensor::{DataType, DataTypeEnum},
};
use tensor_ir::{BinaryOp, UnaryOp};

fn unary_op<const R: usize, In: DataType, Out: DataType>(
    input: &Tensor<R, In>,
    name: Option<&str>,
    op: UnaryOp,
    _backward: impl Fn(Tensor<R, Out>, &Tensor<R, In>) -> Tensor<R, In> + Send + Sync + 'static,
) -> Tensor<R, Out> {
    input.unary_nary(NaryFunction::unary(
        name.map(|s| s.to_string()),
        op,
        In::WGSL_TYPE,
        Out::WGSL_TYPE,
    ))
}

fn unsupported_unary_op<const R: usize, In: DataType, Out: DataType>(
    input: &Tensor<R, In>,
    name: Option<&str>,
) -> Tensor<R, Out> {
    input.unary_nary(NaryFunction::unsupported_unary(
        name.map(|s| s.to_string()),
        In::WGSL_TYPE,
        Out::WGSL_TYPE,
    ))
}

fn binary_const_op<const R: usize, T: DataType>(
    input: &Tensor<R, T>,
    name: Option<&str>,
    op: BinaryOp,
    constant: impl ToString,
    input_first: bool,
) -> Tensor<R, T> {
    input.unary_nary(NaryFunction::binary_const(
        name.map(|s| s.to_string()),
        op,
        constant,
        input_first,
        T::WGSL_TYPE,
        T::WGSL_TYPE,
    ))
}

fn compare_const_op<const R: usize, In: DataType, Out: DataType>(
    input: &Tensor<R, In>,
    name: Option<&str>,
    op: BinaryOp,
    constant: impl ToString,
) -> Tensor<R, Out> {
    input.unary_nary(NaryFunction::compare_const(
        name.map(|s| s.to_string()),
        op,
        constant,
        In::WGSL_TYPE,
        Out::WGSL_TYPE,
    ))
}

impl<const R: usize, T: DataType> Add<T> for Tensor<R, T> {
    type Output = Tensor<R, T>;

    fn add(self, rhs: T) -> Self::Output {
        binary_const_op(&self, Some("add_const"), BinaryOp::Add, rhs, true)
    }
}

impl<const R: usize, T: DataType> Add<T> for &Tensor<R, T> {
    type Output = Tensor<R, T>;

    fn add(self, rhs: T) -> Self::Output {
        self.clone() + rhs
    }
}

impl<const R: usize, T: DataType> Sum for Tensor<R, T> {
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let first = iter.next().expect("Cannot sum over empty iterator");
        iter.fold(first, |acc, x| acc + x)
    }
}

impl<'a, const R: usize, T: DataType> Sum<&'a Tensor<R, T>> for Tensor<R, T> {
    fn sum<I: Iterator<Item = &'a Tensor<R, T>>>(iter: I) -> Self {
        let mut iter = iter.cloned();
        let first = iter.next().expect("Cannot sum over empty iterator");
        iter.fold(first, |acc, x| acc + x)
    }
}

macro_rules! impl_add {
    ($($t:ty),*) => {
        $(
            impl<const R: usize> Add<Tensor<R, $t>> for $t {
                type Output = Tensor<R, $t>;

                fn add(self, rhs: Tensor<R, $t>) -> Self::Output {
                    rhs + self
                }
            }
        )*

    };
}
impl_add!(f32, half::f16, u32);

impl<const R: usize, T: DataType> Sub<T> for Tensor<R, T> {
    type Output = Tensor<R, T>;

    fn sub(self, rhs: T) -> Self::Output {
        binary_const_op(&self, Some("subtract_const"), BinaryOp::Sub, rhs, true)
    }
}

macro_rules! impl_sub {
    ($($t:ty),*) => {
        $(
            impl<const R: usize> Sub<Tensor<R, $t>> for $t {
                type Output = Tensor<R, $t>;

                fn sub(self, rhs: Tensor<R, $t>) -> Self::Output {
                    binary_const_op(&rhs, Some("subtract_const"), BinaryOp::Sub, self, false)
                }
            }
        )*
    };
}
impl_sub!(f32, half::f16, u32);

impl<const R: usize, T: DataType> Mul<T> for Tensor<R, T> {
    type Output = Tensor<R, T>;

    fn mul(self, rhs: T) -> Self::Output {
        binary_const_op(&self, Some("multiply_const"), BinaryOp::Mul, rhs, true)
    }
}

impl<const R: usize, T: DataType> Mul<T> for &Tensor<R, T> {
    type Output = Tensor<R, T>;

    fn mul(self, rhs: T) -> Self::Output {
        self.clone() * rhs
    }
}

macro_rules! impl_mul {
    ($($t:ty),*) => {
        $(
            impl<const R: usize> Mul<Tensor<R, $t>> for $t {
                type Output = Tensor<R, $t>;

                fn mul(self, rhs: Tensor<R, $t>) -> Self::Output {
                    rhs * self
                }
            }
        )*
    };
}
impl_mul!(f32, half::f16, u32);

impl<const R: usize, T: DataType> Div<T> for Tensor<R, T> {
    type Output = Tensor<R, T>;

    fn div(self, rhs: T) -> Self::Output {
        binary_const_op(&self, Some("divide_const"), BinaryOp::Div, rhs, true)
    }
}

macro_rules! impl_div {
    ($($t:ty),*) => {
        $(
            impl<const R: usize> Div<Tensor<R, $t>> for $t {
                type Output = Tensor<R, $t>;

                fn div(self, rhs: Tensor<R, $t>) -> Self::Output {
                    binary_const_op(&rhs, Some("divide_const"), BinaryOp::Div, self, false)
                }
            }
        )*
    };
}
impl_div!(f32, half::f16, u32);

impl<const R: usize> Rem<u32> for Tensor<R, u32> {
    type Output = Tensor<R, u32>;

    fn rem(self, rhs: u32) -> Self::Output {
        binary_const_op(&self, Some("mod_const"), BinaryOp::Mod, rhs, true)
    }
}

macro_rules! impl_mod {
    ($($t:ty),*) => {
        $(
            impl<const R: usize> Rem<Tensor<R, $t>> for $t {
                type Output = Tensor<R, $t>;

                fn rem(self, rhs: Tensor<R, $t>) -> Self::Output {
                    binary_const_op(&rhs, Some("mod_const"), BinaryOp::Mod, self, false)
                }
            }
        )*
    };
}
impl_mod!(f32, half::f16, u32);

impl<const R: usize, T: DataType> Tensor<R, T> {
    /// Check if each value in the tensor is equal to the given value. Returns 1 for true and 0 for false.
    pub fn eq<D: DataType>(&self, rhs: T) -> Tensor<R, D> {
        compare_const_op(self, Some("equal_const"), BinaryOp::Eq, rhs)
    }
}

impl<const R: usize, T: DataType> Tensor<R, T> {
    /// Check if each value in the tensor is less than to the given value. Returns 1 for true and 0 for false.
    pub fn lt<D: DataType>(&self, rhs: T) -> Tensor<R, D> {
        compare_const_op(self, Some("lt_const"), BinaryOp::Lt, rhs)
    }

    /// Check if each value in the tensor is less than or equal to the given value. Returns 1 for true and 0 for false.
    pub fn lte<D: DataType>(&self, rhs: T) -> Tensor<R, D> {
        compare_const_op(self, Some("lte_const"), BinaryOp::Le, rhs)
    }

    /// Check if each value in the tensor is more than to the given value. Returns 1 for true and 0 for false.
    pub fn mt<D: DataType>(&self, rhs: T) -> Tensor<R, D> {
        compare_const_op(self, Some("mt_const"), BinaryOp::Gt, rhs)
    }

    /// Check if each value in the tensor is more than or equal to the given value. Returns 1 for true and 0 for false.
    pub fn mte<D: DataType>(&self, rhs: T) -> Tensor<R, D> {
        compare_const_op(self, Some("mte_const"), BinaryOp::Ge, rhs)
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn less_appoximate_exp(&self) -> Self {
        if D::WGSL_TYPE != DataTypeEnum::F32 {
            return self.exp();
        }
        unsupported_unary_op(self, Some("less_appoximate_exp"))
    }

    pub fn appoximate_exp(&self) -> Self {
        if D::WGSL_TYPE != DataTypeEnum::F32 {
            return self.exp();
        }
        unsupported_unary_op(self, Some("appoximate_exp"))
    }

    pub fn exp(&self) -> Self {
        unary_op(self, Some("exp"), UnaryOp::Exp, |grad, input| {
            grad * &input.exp()
        })
    }
}

impl<const R: usize, D: crate::FloatDataType> Tensor<R, D> {
    pub fn exp2(&self) -> Self {
        unsupported_unary_op(self, Some("exp2"))
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn log(&self) -> Self {
        unary_op(self, Some("log"), UnaryOp::Log, |grad, input| grad / input)
    }
}

impl<const R: usize, D: crate::FloatDataType> Tensor<R, D> {
    pub fn log2(&self) -> Self {
        unsupported_unary_op(self, Some("log2"))
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn pow_elementwise(&self, exponent: D) -> Self {
        let _ = exponent;
        unsupported_unary_op(self, Some("pow"))
    }
}

impl<const R: usize, D: crate::FloatDataType> Tensor<R, D> {
    pub fn sqrt(&self) -> Self {
        unary_op(self, Some("sqrt"), UnaryOp::Sqrt, |grad, input| {
            grad / &(input.sqrt() * D::from_f32(2.0))
        })
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn sin(&self) -> Self {
        unary_op(self, Some("sin"), UnaryOp::Sin, |grad, input| {
            grad * &input.cos()
        })
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn cos(&self) -> Self {
        unary_op(self, Some("cos"), UnaryOp::Cos, |grad, input| {
            -(grad * &input.sin())
        })
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn tan(&self) -> Self {
        unary_op(self, Some("tan"), UnaryOp::Tan, |grad, _input| grad)
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn asin(&self) -> Self {
        unary_op(self, Some("asin"), UnaryOp::Asin, |grad, _input| grad)
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn acos(&self) -> Self {
        unary_op(self, Some("acos"), UnaryOp::Acos, |grad, _input| grad)
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn atan(&self) -> Self {
        unary_op(self, Some("atan"), UnaryOp::Atan, |grad, _input| grad)
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn sinh(&self) -> Self {
        unary_op(self, Some("sinh"), UnaryOp::Sinh, |grad, _input| grad)
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn cosh(&self) -> Self {
        unary_op(self, Some("cosh"), UnaryOp::Cosh, |grad, _input| grad)
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn tanh(&self) -> Self {
        unary_op(self, Some("tanh"), UnaryOp::Tanh, |grad, input| {
            let output = input.tanh();
            let ones = Tensor::splat(input.device(), D::one(), *input.shape());
            let squared = &output * &output;
            grad * &(ones - squared)
        })
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    /// Calculates tanh with (e^x - e^-x) / (e^x + e^-x)
    pub fn tanh_exact(&self) -> Self {
        unary_op(self, Some("tanh_exact"), UnaryOp::Tanh, |grad, input| {
            let output = input.tanh();
            let ones = Tensor::splat(input.device(), D::one(), *input.shape());
            let squared = &output * &output;
            grad * &(ones - squared)
        })
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn asinh(&self) -> Self {
        unary_op(self, Some("asinh"), UnaryOp::Asinh, |grad, _input| grad)
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn acosh(&self) -> Self {
        unary_op(self, Some("acosh"), UnaryOp::Acosh, |grad, _input| grad)
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn atanh(&self) -> Self {
        unary_op(self, Some("atanh"), UnaryOp::Atanh, |grad, _input| grad)
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn abs(&self) -> Self {
        unary_op(self, Some("abs"), UnaryOp::Abs, |grad, _input| grad)
    }
}

impl<const R: usize, D: DataType> Neg for Tensor<R, D> {
    type Output = Tensor<R, D>;

    fn neg(self) -> Self {
        unary_op(&self, Some("neg"), UnaryOp::Neg, |grad, _input| -grad)
    }
}

impl<const R: usize, D: DataType> Neg for &Tensor<R, D> {
    type Output = Tensor<R, D>;

    fn neg(self) -> Self::Output {
        -self.clone()
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn max_elementwise(&self, element: D) -> Self {
        binary_const_op(self, Some("max"), BinaryOp::Max, element, true)
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn min_elementwise(&self, element: D) -> Self {
        binary_const_op(self, Some("min"), BinaryOp::Min, element, true)
    }
}

impl<const R: usize, T> Tensor<R, T> {
    pub fn cast<T2>(&self) -> Tensor<R, T2>
    where
        T: CastTensor<T2>,
    {
        T::cast(self)
    }
}

pub trait CastTensor<T>: Sized {
    /// Casts the tensor to another type
    fn cast<const R: usize>(tensor: &Tensor<R, Self>) -> Tensor<R, T>;
}

impl<T> CastTensor<T> for T {
    fn cast<const R: usize>(tensor: &Tensor<R, Self>) -> Tensor<R, Self> {
        tensor.clone()
    }
}

impl CastTensor<f32> for u32 {
    fn cast<const R: usize>(tensor: &Tensor<R, Self>) -> Tensor<R, f32> {
        tensor.unary_nary(NaryFunction::cast(DataTypeEnum::U32, DataTypeEnum::F32))
    }
}

impl CastTensor<half::f16> for u32 {
    fn cast<const R: usize>(tensor: &Tensor<R, Self>) -> Tensor<R, half::f16> {
        tensor.unary_nary(NaryFunction::cast(DataTypeEnum::U32, DataTypeEnum::F16))
    }
}

impl CastTensor<half::f16> for f32 {
    fn cast<const R: usize>(tensor: &Tensor<R, Self>) -> Tensor<R, half::f16> {
        tensor.unary_nary(NaryFunction::cast(DataTypeEnum::F32, DataTypeEnum::F16))
    }
}

impl CastTensor<f32> for half::f16 {
    fn cast<const R: usize>(tensor: &Tensor<R, Self>) -> Tensor<R, f32> {
        tensor.unary_nary(NaryFunction::cast(DataTypeEnum::F16, DataTypeEnum::F32))
    }
}
