use std::{
    fmt::Display,
    iter::Sum,
    ops::{Add, Div, Mul, Neg, Rem, Sub},
};

use crate::{
    Tensor,
    nary_wise::NaryFunction,
    tensor::{DataType, DataTypeEnum},
};

fn unary_op<const R: usize, In: DataType, Out: DataType>(
    input: &Tensor<R, In>,
    name: Option<&str>,
    operation: impl Display,
    _backward: impl Fn(Tensor<R, Out>, &Tensor<R, In>) -> Tensor<R, In> + Send + Sync + 'static,
) -> Tensor<R, Out> {
    input.unary_nary(NaryFunction::unary(
        name.map(|s| s.to_string()),
        operation.to_string(),
        In::WGSL_TYPE,
        Out::WGSL_TYPE,
    ))
}

fn greater_than_const_mask<const R: usize, D: DataType>(
    input: &Tensor<R, D>,
    value: &str,
) -> Tensor<R, D> {
    input.unary_nary(NaryFunction::unary(
        None,
        format!("let output = {}(input > {value});", D::WGSL_TYPE),
        D::WGSL_TYPE,
        D::WGSL_TYPE,
    ))
}

fn less_than_const_mask<const R: usize, D: DataType>(
    input: &Tensor<R, D>,
    value: &str,
) -> Tensor<R, D> {
    input.unary_nary(NaryFunction::unary(
        None,
        format!("let output = {}(input < {value});", D::WGSL_TYPE),
        D::WGSL_TYPE,
        D::WGSL_TYPE,
    ))
}

impl<const R: usize, T: DataType> Add<T> for Tensor<R, T> {
    type Output = Tensor<R, T>;

    fn add(self, rhs: T) -> Self::Output {
        unary_op(
            &self,
            Some("add_const"),
            format!("let output = input + {rhs};"),
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
            format!("let output = input - {rhs};"),
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
                    unary_op(&rhs, Some("subtract_const"), format!("let output = {self} - input;"), |grad, _input| -grad)
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
            format!("let output = input * {rhs};"),
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
            format!("let output = input / {rhs};"),
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
                    unary_op(&rhs, Some("divide_const"), format!("let output = {} / input;", self), move |grad, input| -((grad * self) / &(input * input)))
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
            format!("let output = input % {rhs};"),
            u32::WGSL_TYPE,
            u32::WGSL_TYPE,
        ))
    }
}

macro_rules! impl_mod {
    ($($t:ty),*) => {
        $(
            impl<const R: usize> Rem<Tensor<R, $t>> for $t {
                type Output = Tensor<R, $t>;

                fn rem(self, rhs: Tensor<R, $t>) -> Self::Output {
                    rhs.unary_nary(NaryFunction::unary(Some("mod_const".to_string()), format!("let output = {} % input;", self), <$t>::WGSL_TYPE, <$t>::WGSL_TYPE))
                }
            }
        )*
    };
}
impl_mod!(f32, half::f16, u32);

impl<const R: usize, T: DataType> Tensor<R, T> {
    /// Check if each value in the tensor is equal to the given value. Returns 1 for true and 0 for false.
    pub fn eq<D: DataType>(&self, rhs: T) -> Tensor<R, D> {
        let datatype = D::WGSL_TYPE;
        self.unary_nary(NaryFunction::unary(
            Some("equal_const".to_string()),
            format!("let output = {datatype}(input == {rhs});"),
            T::WGSL_TYPE,
            D::WGSL_TYPE,
        ))
    }
}

impl<const R: usize, T: DataType> Tensor<R, T> {
    /// Check if each value in the tensor is less than to the given value. Returns 1 for true and 0 for false.
    pub fn lt<D: DataType>(&self, rhs: T) -> Tensor<R, D> {
        let datatype = D::WGSL_TYPE;
        self.unary_nary(NaryFunction::unary(
            Some("lt_const".to_string()),
            format!("let output = {datatype}(input < {rhs});"),
            T::WGSL_TYPE,
            D::WGSL_TYPE,
        ))
    }

    /// Check if each value in the tensor is less than or equal to the given value. Returns 1 for true and 0 for false.
    pub fn lte<D: DataType>(&self, rhs: T) -> Tensor<R, D> {
        let datatype = D::WGSL_TYPE;
        self.unary_nary(NaryFunction::unary(
            Some("lte_const".to_string()),
            format!("let output = {datatype}(input <= {rhs});"),
            T::WGSL_TYPE,
            D::WGSL_TYPE,
        ))
    }

    /// Check if each value in the tensor is more than to the given value. Returns 1 for true and 0 for false.
    pub fn mt<D: DataType>(&self, rhs: T) -> Tensor<R, D> {
        let datatype = D::WGSL_TYPE;
        self.unary_nary(NaryFunction::unary(
            Some("mt_const".to_string()),
            format!("let output = {datatype}(input > {rhs});"),
            T::WGSL_TYPE,
            D::WGSL_TYPE,
        ))
    }

    /// Check if each value in the tensor is more than or equal to the given value. Returns 1 for true and 0 for false.
    pub fn mte<D: DataType>(&self, rhs: T) -> Tensor<R, D> {
        let datatype = D::WGSL_TYPE;
        self.unary_nary(NaryFunction::unary(
            Some("mte_const".to_string()),
            format!("let output = {datatype}(input >= {rhs});"),
            T::WGSL_TYPE,
            D::WGSL_TYPE,
        ))
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn less_appoximate_exp(&self) -> Self {
        if D::WGSL_TYPE != DataTypeEnum::F32 {
            return self.exp();
        }
        // https://specbranch.com/posts/fast-exp/
        self.unary_nary(NaryFunction::unary(Some("less_appoximate_exp".to_string()), "let first_order = i32(input * 12102203.0) + (127 << 23) - 345088;
                let correction_xi = (first_order & 0x7fffff) | (127 << 23);
                let correction_x = bitcast<f32>(correction_xi);
                let output = bitcast<f32>(first_order) * fma(fma(correction_x, 0.22670517861843109130859375, -0.671999752521514892578125), correction_x, 1.469318866729736328125);".to_string(), D::WGSL_TYPE, D::WGSL_TYPE))
    }

    pub fn appoximate_exp(&self) -> Self {
        if D::WGSL_TYPE != DataTypeEnum::F32 {
            return self.exp();
        }
        // https://specbranch.com/posts/fast-exp/
        self.unary_nary(NaryFunction::unary(
            Some("appoximate_exp".to_string()),
            "let output = bitcast<f32>(i32(input * 12102203.0) + (127 << 23) - 545948);"
                .to_string(),
            D::WGSL_TYPE,
            D::WGSL_TYPE,
        ))
    }

    pub fn exp(&self) -> Self {
        unary_op(
            self,
            Some("exp"),
            "let output = exp(input);",
            |grad, input| grad * &input.exp(),
        )
    }
}

impl<const R: usize, D: crate::FloatDataType> Tensor<R, D> {
    pub fn exp2(&self) -> Self {
        unary_op(
            self,
            Some("exp2"),
            "let output = exp2(input);",
            |grad, input| (grad * &input.exp2()) * D::from_f32(std::f32::consts::LN_2),
        )
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn log(&self) -> Self {
        unary_op(
            self,
            Some("log"),
            "let output = log(input);",
            |grad, input| grad / input,
        )
    }
}

impl<const R: usize, D: crate::FloatDataType> Tensor<R, D> {
    pub fn log2(&self) -> Self {
        unary_op(
            self,
            Some("log2"),
            "let output = log2(input);",
            |grad, input| grad / &(input * D::from_f32(std::f32::consts::LN_2)),
        )
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn pow_elementwise(&self, exponent: D) -> Self {
        unary_op(
            self,
            Some("pow"),
            format!("let output = pow(input, {exponent});"),
            move |grad, input| (grad * exponent) * &input.pow_elementwise(exponent - D::one()),
        )
    }
}

impl<const R: usize, D: crate::FloatDataType> Tensor<R, D> {
    pub fn sqrt(&self) -> Self {
        unary_op(
            self,
            Some("sqrt"),
            "let output = sqrt(input);",
            |grad, input| grad / &(input.sqrt() * D::from_f32(2.0)),
        )
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn sin(&self) -> Self {
        unary_op(
            self,
            Some("sin"),
            "let output = sin(input);",
            |grad, input| grad * &input.cos(),
        )
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn cos(&self) -> Self {
        unary_op(
            self,
            Some("cos"),
            "let output = cos(input);",
            |grad, input| -(grad * &input.sin()),
        )
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn tan(&self) -> Self {
        self.unary_nary(NaryFunction::unary(
            Some("tan".to_string()),
            "let output = tan(input);".to_string(),
            D::WGSL_TYPE,
            D::WGSL_TYPE,
        ))
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn asin(&self) -> Self {
        self.unary_nary(NaryFunction::unary(
            Some("asin".to_string()),
            "let output = asin(input);".to_string(),
            D::WGSL_TYPE,
            D::WGSL_TYPE,
        ))
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn acos(&self) -> Self {
        self.unary_nary(NaryFunction::unary(
            Some("acos".to_string()),
            "let output = acos(input);".to_string(),
            D::WGSL_TYPE,
            D::WGSL_TYPE,
        ))
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn atan(&self) -> Self {
        self.unary_nary(NaryFunction::unary(
            Some("atan".to_string()),
            "let output = atan(input);".to_string(),
            D::WGSL_TYPE,
            D::WGSL_TYPE,
        ))
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn sinh(&self) -> Self {
        self.unary_nary(NaryFunction::unary(
            Some("sinh".to_string()),
            "let output = sinh(input);".to_string(),
            D::WGSL_TYPE,
            D::WGSL_TYPE,
        ))
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn cosh(&self) -> Self {
        self.unary_nary(NaryFunction::unary(
            Some("cosh".to_string()),
            "let output = cosh(input);".to_string(),
            D::WGSL_TYPE,
            D::WGSL_TYPE,
        ))
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn tanh(&self) -> Self {
        unary_op(
            self,
            Some("tanh"),
            "let output = tanh(input);",
            |grad, input| {
                let output = input.tanh();
                let ones = Tensor::splat(input.device(), D::one(), *input.shape());
                let squared = &output * &output;
                grad * &(ones - squared)
            },
        )
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    /// Calculates tanh with (e^x - e^-x) / (e^x + e^-x)
    pub fn tanh_exact(&self) -> Self {
        unary_op(
            self,
            Some("tanh_exact"),
            "let output = (exp(input) - exp(-input)) / (exp(input) + exp(-input));",
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
            "let output = asinh(input);".to_string(),
            D::WGSL_TYPE,
            D::WGSL_TYPE,
        ))
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn acosh(&self) -> Self {
        self.unary_nary(NaryFunction::unary(
            Some("acosh".to_string()),
            "let output = acosh(input);".to_string(),
            D::WGSL_TYPE,
            D::WGSL_TYPE,
        ))
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn atanh(&self) -> Self {
        self.unary_nary(NaryFunction::unary(
            Some("atanh".to_string()),
            "let output = atanh(input);".to_string(),
            D::WGSL_TYPE,
            D::WGSL_TYPE,
        ))
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn abs(&self) -> Self {
        self.unary_nary(NaryFunction::unary(
            Some("abs".to_string()),
            "let output = abs(input);".to_string(),
            D::WGSL_TYPE,
            D::WGSL_TYPE,
        ))
    }
}

impl<const R: usize, D: DataType> Neg for Tensor<R, D> {
    type Output = Tensor<R, D>;

    fn neg(self) -> Self {
        unary_op(
            &self,
            Some("neg"),
            "let output = -input;",
            |grad, _input| -grad,
        )
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
        let element_str = element.to_string();
        unary_op(
            self,
            Some("max"),
            format!("let output = max(input, {element});"),
            move |grad, input| grad * &greater_than_const_mask(input, &element_str),
        )
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn min_elementwise(&self, element: D) -> Self {
        let element_str = element.to_string();
        unary_op(
            self,
            Some("min"),
            format!("let output = min(input, {element});"),
            move |grad, input| grad * &less_than_const_mask(input, &element_str),
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
            "let output = f32(input);".to_string(),
            DataTypeEnum::U32,
            DataTypeEnum::F32,
        ))
    }
}

impl CastTensor<half::f16> for u32 {
    fn cast<const R: usize>(tensor: &Tensor<R, Self>) -> Tensor<R, half::f16> {
        tensor.unary_nary(NaryFunction::unary(
            Some("cast".to_string()),
            "let output = f16(input);".to_string(),
            DataTypeEnum::U32,
            DataTypeEnum::F16,
        ))
    }
}

impl CastTensor<half::f16> for f32 {
    fn cast<const R: usize>(tensor: &Tensor<R, Self>) -> Tensor<R, half::f16> {
        unary_op(
            tensor,
            Some("cast"),
            "let output = f16(input);",
            |grad, _input| grad.cast(),
        )
    }
}

impl CastTensor<f32> for half::f16 {
    fn cast<const R: usize>(tensor: &Tensor<R, Self>) -> Tensor<R, f32> {
        unary_op(
            tensor,
            Some("cast"),
            "let output = f32(input);",
            |grad, _input| grad.cast(),
        )
    }
}

