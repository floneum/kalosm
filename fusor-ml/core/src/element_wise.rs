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

fn unary_op<const R: usize, In: DataType, Out: DataType>(
    input: &Tensor<R, In>,
    name: Option<&str>,
    operation: NaryOp,
    _backward: impl Fn(Tensor<R, Out>, &Tensor<R, In>) -> Tensor<R, In> + Send + Sync + 'static,
) -> Tensor<R, Out> {
    input.unary_nary(NaryFunction::unary(
        name.map(|s| s.to_string()),
        operation,
        In::DATA_TYPE,
        Out::DATA_TYPE,
    ))
}

fn greater_than_const_mask<const R: usize, D: DataType>(
    input: &Tensor<R, D>,
    value: &D,
) -> Tensor<R, D> {
    input.unary_nary(NaryFunction::unary(
        None,
        NaryOp::GreaterConst(scalar_value(value)),
        D::DATA_TYPE,
        D::DATA_TYPE,
    ))
}

fn less_than_const_mask<const R: usize, D: DataType>(
    input: &Tensor<R, D>,
    value: &D,
) -> Tensor<R, D> {
    input.unary_nary(NaryFunction::unary(
        None,
        NaryOp::LessConst(scalar_value(value)),
        D::DATA_TYPE,
        D::DATA_TYPE,
    ))
}

impl<const R: usize, T: DataType> Add<T> for Tensor<R, T> {
    type Output = Tensor<R, T>;

    fn add(self, rhs: T) -> Self::Output {
        unary_op(
            &self,
            Some("add_const"),
            NaryOp::AddConst(scalar_value(&rhs)),
            |grad, _input| grad,
        )
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
        unary_op(
            &self,
            Some("subtract_const"),
            NaryOp::SubConst(scalar_value(&rhs)),
            |grad, _input| grad,
        )
    }
}

macro_rules! impl_sub {
    ($($t:ty),*) => {
        $(
            impl<const R: usize> Sub<Tensor<R, $t>> for $t {
                type Output = Tensor<R, $t>;

                fn sub(self, rhs: Tensor<R, $t>) -> Self::Output {
                    unary_op(&rhs, Some("subtract_const"), NaryOp::RSubConst(scalar_value(&self)), |grad, _input| -grad)
                }
            }
        )*
    };
}
impl_sub!(f32, half::f16, u32);

impl<const R: usize, T: DataType> Mul<T> for Tensor<R, T> {
    type Output = Tensor<R, T>;

    fn mul(self, rhs: T) -> Self::Output {
        unary_op(
            &self,
            Some("multiply_const"),
            NaryOp::MulConst(scalar_value(&rhs)),
            move |grad, _input| grad * rhs,
        )
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
        unary_op(
            &self,
            Some("divide_const"),
            NaryOp::DivConst(scalar_value(&rhs)),
            move |grad, _input| grad / rhs,
        )
    }
}

macro_rules! impl_div {
    ($($t:ty),*) => {
        $(
            impl<const R: usize> Div<Tensor<R, $t>> for $t {
                type Output = Tensor<R, $t>;

                fn div(self, rhs: Tensor<R, $t>) -> Self::Output {
                    unary_op(&rhs, Some("divide_const"), NaryOp::RDivConst(scalar_value(&self)), move |grad, input| -((grad * self) / &(input * input)))
                }
            }
        )*
    };
}
impl_div!(f32, half::f16, u32);

impl<const R: usize> Rem<u32> for Tensor<R, u32> {
    type Output = Tensor<R, u32>;

    fn rem(self, rhs: u32) -> Self::Output {
        self.unary_nary(NaryFunction::unary(
            Some("mod_const".to_string()),
            NaryOp::RemConst(NaryScalar::U32(rhs)),
            u32::DATA_TYPE,
            u32::DATA_TYPE,
        ))
    }
}

macro_rules! impl_mod {
    ($($t:ty),*) => {
        $(
            impl<const R: usize> Rem<Tensor<R, $t>> for $t {
                type Output = Tensor<R, $t>;

                fn rem(self, rhs: Tensor<R, $t>) -> Self::Output {
                    rhs.unary_nary(NaryFunction::unary(Some("mod_const".to_string()), NaryOp::RRemConst(scalar_value(&self)), <$t>::DATA_TYPE, <$t>::DATA_TYPE))
                }
            }
        )*
    };
}
impl_mod!(f32, half::f16, u32);

impl<const R: usize, T: DataType> Tensor<R, T> {
    /// Check if each value in the tensor is equal to the given value. Returns 1 for true and 0 for false.
    pub fn eq<D: DataType>(&self, rhs: T) -> Tensor<R, D> {
        self.unary_nary(NaryFunction::unary(
            Some("equal_const".to_string()),
            NaryOp::EqualConst(scalar_value(&rhs)),
            T::DATA_TYPE,
            D::DATA_TYPE,
        ))
    }
}

impl<const R: usize, T: DataType> Tensor<R, T> {
    /// Check if each value in the tensor is less than to the given value. Returns 1 for true and 0 for false.
    pub fn lt<D: DataType>(&self, rhs: T) -> Tensor<R, D> {
        self.unary_nary(NaryFunction::unary(
            Some("lt_const".to_string()),
            NaryOp::LessConst(scalar_value(&rhs)),
            T::DATA_TYPE,
            D::DATA_TYPE,
        ))
    }

    /// Check if each value in the tensor is less than or equal to the given value. Returns 1 for true and 0 for false.
    pub fn lte<D: DataType>(&self, rhs: T) -> Tensor<R, D> {
        self.unary_nary(NaryFunction::unary(
            Some("lte_const".to_string()),
            NaryOp::LessEqualConst(scalar_value(&rhs)),
            T::DATA_TYPE,
            D::DATA_TYPE,
        ))
    }

    /// Check if each value in the tensor is more than to the given value. Returns 1 for true and 0 for false.
    pub fn mt<D: DataType>(&self, rhs: T) -> Tensor<R, D> {
        self.unary_nary(NaryFunction::unary(
            Some("mt_const".to_string()),
            NaryOp::GreaterConst(scalar_value(&rhs)),
            T::DATA_TYPE,
            D::DATA_TYPE,
        ))
    }

    /// Check if each value in the tensor is more than or equal to the given value. Returns 1 for true and 0 for false.
    pub fn mte<D: DataType>(&self, rhs: T) -> Tensor<R, D> {
        self.unary_nary(NaryFunction::unary(
            Some("mte_const".to_string()),
            NaryOp::GreaterEqualConst(scalar_value(&rhs)),
            T::DATA_TYPE,
            D::DATA_TYPE,
        ))
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn less_appoximate_exp(&self) -> Self {
        if D::DATA_TYPE != DataTypeEnum::F32 {
            return self.exp();
        }
        self.unary_nary(NaryFunction::unary(
            Some("less_appoximate_exp".to_string()),
            NaryOp::LessApproximateExp,
            D::DATA_TYPE,
            D::DATA_TYPE,
        ))
    }

    pub fn appoximate_exp(&self) -> Self {
        if D::DATA_TYPE != DataTypeEnum::F32 {
            return self.exp();
        }
        self.unary_nary(NaryFunction::unary(
            Some("appoximate_exp".to_string()),
            NaryOp::ApproximateExp,
            D::DATA_TYPE,
            D::DATA_TYPE,
        ))
    }

    pub fn exp(&self) -> Self {
        unary_op(self, Some("exp"), NaryOp::Exp, |grad, input| {
            grad * &input.exp()
        })
    }
}

impl<const R: usize, D: crate::FloatDataType> Tensor<R, D> {
    pub fn exp2(&self) -> Self {
        unary_op(self, Some("exp2"), NaryOp::Exp2, |grad, input| {
            (grad * &input.exp2()) * D::from_f32(std::f32::consts::LN_2)
        })
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn log(&self) -> Self {
        unary_op(self, Some("log"), NaryOp::Log, |grad, input| grad / input)
    }
}

impl<const R: usize, D: crate::FloatDataType> Tensor<R, D> {
    pub fn log2(&self) -> Self {
        unary_op(self, Some("log2"), NaryOp::Log2, |grad, input| {
            grad / &(input * D::from_f32(std::f32::consts::LN_2))
        })
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn pow_elementwise(&self, exponent: D) -> Self {
        unary_op(
            self,
            Some("pow"),
            NaryOp::PowConst(scalar_value(&exponent)),
            move |grad, input| (grad * exponent) * &input.pow_elementwise(exponent - D::one()),
        )
    }
}

impl<const R: usize, D: crate::FloatDataType> Tensor<R, D> {
    pub fn sqrt(&self) -> Self {
        unary_op(self, Some("sqrt"), NaryOp::Sqrt, |grad, input| {
            grad / &(input.sqrt() * D::from_f32(2.0))
        })
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn sin(&self) -> Self {
        unary_op(self, Some("sin"), NaryOp::Sin, |grad, input| {
            grad * &input.cos()
        })
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn cos(&self) -> Self {
        unary_op(self, Some("cos"), NaryOp::Cos, |grad, input| {
            -(grad * &input.sin())
        })
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn tan(&self) -> Self {
        self.unary_nary(NaryFunction::unary(
            Some("tan".to_string()),
            NaryOp::Tan,
            D::DATA_TYPE,
            D::DATA_TYPE,
        ))
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn asin(&self) -> Self {
        self.unary_nary(NaryFunction::unary(
            Some("asin".to_string()),
            NaryOp::Asin,
            D::DATA_TYPE,
            D::DATA_TYPE,
        ))
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn acos(&self) -> Self {
        self.unary_nary(NaryFunction::unary(
            Some("acos".to_string()),
            NaryOp::Acos,
            D::DATA_TYPE,
            D::DATA_TYPE,
        ))
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn atan(&self) -> Self {
        self.unary_nary(NaryFunction::unary(
            Some("atan".to_string()),
            NaryOp::Atan,
            D::DATA_TYPE,
            D::DATA_TYPE,
        ))
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn sinh(&self) -> Self {
        self.unary_nary(NaryFunction::unary(
            Some("sinh".to_string()),
            NaryOp::Sinh,
            D::DATA_TYPE,
            D::DATA_TYPE,
        ))
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn cosh(&self) -> Self {
        self.unary_nary(NaryFunction::unary(
            Some("cosh".to_string()),
            NaryOp::Cosh,
            D::DATA_TYPE,
            D::DATA_TYPE,
        ))
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn tanh(&self) -> Self {
        unary_op(self, Some("tanh"), NaryOp::Tanh, |grad, input| {
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
        unary_op(
            self,
            Some("tanh_exact"),
            NaryOp::TanhExact,
            |grad, input| {
                let output = input.tanh_exact();
                let ones = Tensor::splat(input.device(), D::one(), *input.shape());
                let squared = &output * &output;
                grad * &(ones - squared)
            },
        )
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn asinh(&self) -> Self {
        self.unary_nary(NaryFunction::unary(
            Some("asinh".to_string()),
            NaryOp::Asinh,
            D::DATA_TYPE,
            D::DATA_TYPE,
        ))
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn acosh(&self) -> Self {
        self.unary_nary(NaryFunction::unary(
            Some("acosh".to_string()),
            NaryOp::Acosh,
            D::DATA_TYPE,
            D::DATA_TYPE,
        ))
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn atanh(&self) -> Self {
        self.unary_nary(NaryFunction::unary(
            Some("atanh".to_string()),
            NaryOp::Atanh,
            D::DATA_TYPE,
            D::DATA_TYPE,
        ))
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn abs(&self) -> Self {
        self.unary_nary(NaryFunction::unary(
            Some("abs".to_string()),
            NaryOp::Abs,
            D::DATA_TYPE,
            D::DATA_TYPE,
        ))
    }
}

impl<const R: usize, D: DataType> Neg for Tensor<R, D> {
    type Output = Tensor<R, D>;

    fn neg(self) -> Self {
        unary_op(&self, Some("neg"), NaryOp::Neg, |grad, _input| -grad)
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
        unary_op(
            self,
            Some("max"),
            NaryOp::MaxConst(scalar_value(&element)),
            move |grad, input| grad * &greater_than_const_mask(input, &element),
        )
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn min_elementwise(&self, element: D) -> Self {
        unary_op(
            self,
            Some("min"),
            NaryOp::MinConst(scalar_value(&element)),
            move |grad, input| grad * &less_than_const_mask(input, &element),
        )
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
        tensor.unary_nary(NaryFunction::unary(
            Some("cast".to_string()),
            NaryOp::Cast,
            DataTypeEnum::U32,
            DataTypeEnum::F32,
        ))
    }
}

impl CastTensor<half::f16> for u32 {
    fn cast<const R: usize>(tensor: &Tensor<R, Self>) -> Tensor<R, half::f16> {
        tensor.unary_nary(NaryFunction::unary(
            Some("cast".to_string()),
            NaryOp::Cast,
            DataTypeEnum::U32,
            DataTypeEnum::F16,
        ))
    }
}

impl CastTensor<half::f16> for f32 {
    fn cast<const R: usize>(tensor: &Tensor<R, Self>) -> Tensor<R, half::f16> {
        unary_op(tensor, Some("cast"), NaryOp::Cast, |grad, _input| {
            grad.cast()
        })
    }
}

impl CastTensor<f32> for half::f16 {
    fn cast<const R: usize>(tensor: &Tensor<R, Self>) -> Tensor<R, f32> {
        unary_op(tensor, Some("cast"), NaryOp::Cast, |grad, _input| {
            grad.cast()
        })
    }
}
