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

#[cfg(test)]
use crate::Device;

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
        unary_op(&self, Some("add_const"), format!("let output = input + {rhs};"), |grad, _input| grad)
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

#[cfg(test)]
#[tokio::test]
async fn test_add_const() {
    let device = Device::test_instance();

    let data = [
        [[1., 2.], [1., 2.]],
        [[3., 4.], [3., 4.]],
        [[5., 6.], [5., 6.]],
    ];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor + 1.0;

    let output = tensor.as_slice().await.unwrap();
    let result = [
        [[2.0, 3.0], [2.0, 3.0]],
        [[4.0, 5.0], [4.0, 5.0]],
        [[6.0, 7.0], [6.0, 7.0]],
    ];
    let result = Tensor::new(&device, &result);
    let result = result.as_slice().await.unwrap();
    println!("{output:?}");
    println!("{result:?}");
    assert_eq!(output, result);

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor + 1.0;

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0, 0]], 2.);
    assert_eq!(output[[0, 1]], 3.);
    assert_eq!(output[[1, 0]], 4.);
    assert_eq!(output[[1, 1]], 5.);
    assert_eq!(output[[2, 0]], 6.);
    assert_eq!(output[[2, 1]], 7.);

    let data = [1., 2.];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor + 1.0;

    let output = tensor.as_slice().await.unwrap();
    assert_eq!(output[[0]], 2.);
    assert_eq!(output[[1]], 3.);
}

#[cfg(test)]
#[tokio::test]
async fn test_add_const_4_dim() {
    let device = Device::test_instance();

    let data = [
        [
            [[1., 2.], [1., 2.]],
            [[3., 4.], [3., 4.]],
            [[5., 6.], [5., 6.]],
        ],
        [
            [[6., 2.], [1., 2.]],
            [[3., 4.], [3., 4.]],
            [[5., 6.], [5., 6.]],
        ],
    ];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor + 1.0;

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    let result = [
        [
            [[2.0, 3.0], [2.0, 3.0]],
            [[4.0, 5.0], [4.0, 5.0]],
            [[6.0, 7.0], [6.0, 7.0]],
        ],
        [
            [[7.0, 3.0], [2.0, 3.0]],
            [[4.0, 5.0], [4.0, 5.0]],
            [[6.0, 7.0], [6.0, 7.0]],
        ],
    ];
    let result = Tensor::new(&device, &result);
    assert_eq!(output, result.as_slice().await.unwrap());
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

#[cfg(test)]
#[tokio::test]
async fn test_add_const_reversed() {
    let device = Device::test_instance();

    let data = [
        [[1., 2.], [1., 2.]],
        [[3., 4.], [3., 4.]],
        [[5., 6.], [5., 6.]],
    ];
    let tensor = Tensor::new(&device, &data);

    let tensor = 1.0 + tensor;

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    let result = [
        [[2.0, 3.0], [2.0, 3.0]],
        [[4.0, 5.0], [4.0, 5.0]],
        [[6.0, 7.0], [6.0, 7.0]],
    ];
    let result = Tensor::new(&device, &result);
    assert_eq!(output, result.as_slice().await.unwrap());

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = 1.0 + tensor;

    let output = tensor.as_slice().await.unwrap();
    assert_eq!(output[[0, 0]], 2.);
    assert_eq!(output[[0, 1]], 3.);
    assert_eq!(output[[1, 0]], 4.);
    assert_eq!(output[[1, 1]], 5.);
    assert_eq!(output[[2, 0]], 6.);
    assert_eq!(output[[2, 1]], 7.);

    let data = [1., 2.];
    let tensor = Tensor::new(&device, &data);

    let tensor = 1.0 + tensor;

    let output = tensor.as_slice().await.unwrap();
    assert_eq!(output[[0]], 2.);
    assert_eq!(output[[1]], 3.);
}

#[cfg(test)]
#[tokio::test]
async fn test_add_const_f16() {
    let device = Device::test_instance();
    if !device.f16_supported() {
        return;
    }

    let data = [
        [
            [half::f16::from_f32(1.), half::f16::from_f32(2.)],
            [half::f16::from_f32(1.), half::f16::from_f32(2.)],
        ],
        [
            [half::f16::from_f32(3.), half::f16::from_f32(4.)],
            [half::f16::from_f32(3.), half::f16::from_f32(4.)],
        ],
        [
            [half::f16::from_f32(5.), half::f16::from_f32(6.)],
            [half::f16::from_f32(5.), half::f16::from_f32(6.)],
        ],
    ];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor + half::f16::from_f32(1.0);

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    let result = [
        [
            [half::f16::from_f32(2.0), half::f16::from_f32(3.0)],
            [half::f16::from_f32(2.0), half::f16::from_f32(3.0)],
        ],
        [
            [half::f16::from_f32(4.0), half::f16::from_f32(5.0)],
            [half::f16::from_f32(4.0), half::f16::from_f32(5.0)],
        ],
        [
            [half::f16::from_f32(6.0), half::f16::from_f32(7.0)],
            [half::f16::from_f32(6.0), half::f16::from_f32(7.0)],
        ],
    ];
    let result = Tensor::new(&device, &result);
    assert_eq!(output, result.as_slice().await.unwrap());
}

#[cfg(test)]
#[tokio::test]
async fn test_add_const_sliced() {
    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let sliced = tensor.restride([
        crate::StrideSpec::dim(0, 3),
        crate::StrideSpec::dim(1, 1),
    ]);

    let sliced = sliced + 1.0;

    let output = sliced.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0, 0]], 2.);
    assert_eq!(output[[1, 0]], 4.);
    assert_eq!(output[[2, 0]], 6.);
}

#[cfg(test)]
#[tokio::test]
async fn test_merge_add_const() {
    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = (tensor + 1.0) * 2.0;

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0, 0]], 4.);
    assert_eq!(output[[0, 1]], 6.);
    assert_eq!(output[[1, 0]], 8.);
    assert_eq!(output[[1, 1]], 10.);
    assert_eq!(output[[2, 0]], 12.);
    assert_eq!(output[[2, 1]], 14.);
}

impl<const R: usize, T: DataType> Sub<T> for Tensor<R, T> {
    type Output = Tensor<R, T>;

    fn sub(self, rhs: T) -> Self::Output {
        unary_op(&self, Some("subtract_const"), format!("let output = input - {rhs};"), |grad, _input| grad)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_sub_const() {
    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor - 1.0;

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0, 0]], 0.);
    assert_eq!(output[[0, 1]], 1.);
    assert_eq!(output[[1, 0]], 2.);
    assert_eq!(output[[1, 1]], 3.);
    assert_eq!(output[[2, 0]], 4.);
    assert_eq!(output[[2, 1]], 5.);
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

#[cfg(test)]
#[tokio::test]
async fn test_sub_const_reversed() {
    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = 6.0 - tensor;

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0, 0]], 5.);
    assert_eq!(output[[0, 1]], 4.);
    assert_eq!(output[[1, 0]], 3.);
    assert_eq!(output[[1, 1]], 2.);
    assert_eq!(output[[2, 0]], 1.);
    assert_eq!(output[[2, 1]], 0.);
}

impl<const R: usize, T: DataType> Mul<T> for Tensor<R, T> {
    type Output = Tensor<R, T>;

    fn mul(self, rhs: T) -> Self::Output {
        unary_op(&self, Some("multiply_const"), format!("let output = input * {rhs};"), move |grad, _input| grad * rhs)
    }
}

impl<const R: usize, T: DataType> Mul<T> for &Tensor<R, T> {
    type Output = Tensor<R, T>;

    fn mul(self, rhs: T) -> Self::Output {
        self.clone() * rhs
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_mul_const() {
    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor * 2.0;

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0, 0]], 2.);
    assert_eq!(output[[0, 1]], 4.);
    assert_eq!(output[[1, 0]], 6.);
    assert_eq!(output[[1, 1]], 8.);
    assert_eq!(output[[2, 0]], 10.);
    assert_eq!(output[[2, 1]], 12.);
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

#[cfg(test)]
#[tokio::test]
async fn test_mul_const_reversed() {
    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = 2.0 * tensor;

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0, 0]], 2.);
    assert_eq!(output[[0, 1]], 4.);
    assert_eq!(output[[1, 0]], 6.);
    assert_eq!(output[[1, 1]], 8.);
    assert_eq!(output[[2, 0]], 10.);
    assert_eq!(output[[2, 1]], 12.);
}

impl<const R: usize, T: DataType> Div<T> for Tensor<R, T> {
    type Output = Tensor<R, T>;

    fn div(self, rhs: T) -> Self::Output {
        unary_op(&self, Some("divide_const"), format!("let output = input / {rhs};"), move |grad, _input| grad / rhs)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_div_const() {
    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor / 2.0;

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0, 0]], 0.5);
    assert_eq!(output[[0, 1]], 1.);
    assert_eq!(output[[1, 0]], 1.5);
    assert_eq!(output[[1, 1]], 2.);
    assert_eq!(output[[2, 0]], 2.5);
    assert_eq!(output[[2, 1]], 3.);
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

#[cfg(test)]
#[tokio::test]
async fn test_div_const_reversed() {
    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = 6.0 / tensor;

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0, 0]], 6.0 / data[0][0]);
    assert_eq!(output[[0, 1]], 6.0 / data[0][1]);
    assert_eq!(output[[1, 0]], 6.0 / data[1][0]);
    assert_eq!(output[[1, 1]], 6.0 / data[1][1]);
    assert_eq!(output[[2, 0]], 6.0 / data[2][0]);
    assert_eq!(output[[2, 1]], 6.0 / data[2][1]);
}

impl<const R: usize> Rem<u32> for Tensor<R, u32> {
    type Output = Tensor<R, u32>;

    fn rem(self, rhs: u32) -> Self::Output {
        self.unary_nary(NaryFunction::unary(Some("mod_const".to_string()), format!("let output = input % {rhs};"), u32::WGSL_TYPE, u32::WGSL_TYPE))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_mod_const() {
    let device = Device::test_instance();

    let data = [[1, 2], [3, 4], [5, 6]];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor % 2;

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0, 0]], 1);
    assert_eq!(output[[0, 1]], 0);
    assert_eq!(output[[1, 0]], 1);
    assert_eq!(output[[1, 1]], 0);
    assert_eq!(output[[2, 0]], 1);
    assert_eq!(output[[2, 1]], 0);
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

#[cfg(test)]
#[tokio::test]
async fn test_mod_const_reversed() {
    let device = Device::test_instance();

    let data = [[1, 2], [3, 4], [5, 6]];
    let tensor = Tensor::new(&device, &data);

    let tensor = 6 % tensor;

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0, 0]], 6 % data[0][0]);
    assert_eq!(output[[0, 1]], 6 % data[0][1]);
    assert_eq!(output[[1, 0]], 6 % data[1][0]);
    assert_eq!(output[[1, 1]], 6 % data[1][1]);
    assert_eq!(output[[2, 0]], 6 % data[2][0]);
    assert_eq!(output[[2, 1]], 6 % data[2][1]);
}

impl<const R: usize, T: DataType> Tensor<R, T> {
    /// Check if each value in the tensor is equal to the given value. Returns 1 for true and 0 for false.
    pub fn eq<D: DataType>(&self, rhs: T) -> Tensor<R, D> {
        let datatype = D::WGSL_TYPE;
        self.unary_nary(NaryFunction::unary(Some("equal_const".to_string()), format!("let output = {datatype}(input == {rhs});"), T::WGSL_TYPE, D::WGSL_TYPE))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_eq_const() {
    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor: Tensor<2, f32> = tensor.eq(1.0);

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0, 0]], 1.);
    assert_eq!(output[[0, 1]], 0.);
    assert_eq!(output[[1, 0]], 0.);
    assert_eq!(output[[1, 1]], 0.);
    assert_eq!(output[[2, 0]], 0.);
    assert_eq!(output[[2, 1]], 0.);
}

impl<const R: usize, T: DataType> Tensor<R, T> {
    /// Check if each value in the tensor is less than to the given value. Returns 1 for true and 0 for false.
    pub fn lt<D: DataType>(&self, rhs: T) -> Tensor<R, D> {
        let datatype = D::WGSL_TYPE;
        self.unary_nary(NaryFunction::unary(Some("lt_const".to_string()), format!("let output = {datatype}(input < {rhs});"), T::WGSL_TYPE, D::WGSL_TYPE))
    }

    /// Check if each value in the tensor is less than or equal to the given value. Returns 1 for true and 0 for false.
    pub fn lte<D: DataType>(&self, rhs: T) -> Tensor<R, D> {
        let datatype = D::WGSL_TYPE;
        self.unary_nary(NaryFunction::unary(Some("lte_const".to_string()), format!("let output = {datatype}(input <= {rhs});"), T::WGSL_TYPE, D::WGSL_TYPE))
    }

    /// Check if each value in the tensor is more than to the given value. Returns 1 for true and 0 for false.
    pub fn mt<D: DataType>(&self, rhs: T) -> Tensor<R, D> {
        let datatype = D::WGSL_TYPE;
        self.unary_nary(NaryFunction::unary(Some("mt_const".to_string()), format!("let output = {datatype}(input > {rhs});"), T::WGSL_TYPE, D::WGSL_TYPE))
    }

    /// Check if each value in the tensor is more than or equal to the given value. Returns 1 for true and 0 for false.
    pub fn mte<D: DataType>(&self, rhs: T) -> Tensor<R, D> {
        let datatype = D::WGSL_TYPE;
        self.unary_nary(NaryFunction::unary(Some("mte_const".to_string()), format!("let output = {datatype}(input >= {rhs});"), T::WGSL_TYPE, D::WGSL_TYPE))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_lt_const() {
    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor: Tensor<2, f32> = tensor.lt(2.0);

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0, 0]], 1.);
    assert_eq!(output[[0, 1]], 0.);
    assert_eq!(output[[1, 0]], 0.);
    assert_eq!(output[[1, 1]], 0.);
    assert_eq!(output[[2, 0]], 0.);
    assert_eq!(output[[2, 1]], 0.);
}

#[cfg(test)]
#[tokio::test]
async fn test_lte_const() {
    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor: Tensor<2, f32> = tensor.lte(2.0);

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0, 0]], 1.);
    assert_eq!(output[[0, 1]], 1.);
    assert_eq!(output[[1, 0]], 0.);
    assert_eq!(output[[1, 1]], 0.);
    assert_eq!(output[[2, 0]], 0.);
    assert_eq!(output[[2, 1]], 0.);
}

#[cfg(test)]
#[tokio::test]
async fn test_mt_const() {
    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor: Tensor<2, f32> = tensor.mt(2.0);

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0, 0]], 0.);
    assert_eq!(output[[0, 1]], 0.);
    assert_eq!(output[[1, 0]], 1.);
    assert_eq!(output[[1, 1]], 1.);
    assert_eq!(output[[2, 0]], 1.);
    assert_eq!(output[[2, 1]], 1.);
}

#[cfg(test)]
#[tokio::test]
async fn test_mte_const() {
    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor: Tensor<2, f32> = tensor.mte(2.0);

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0, 0]], 0.);
    assert_eq!(output[[0, 1]], 1.);
    assert_eq!(output[[1, 0]], 1.);
    assert_eq!(output[[1, 1]], 1.);
    assert_eq!(output[[2, 0]], 1.);
    assert_eq!(output[[2, 1]], 1.);
}

#[cfg(test)]
#[tokio::test]
async fn test_eq_const_cast() {
    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor: Tensor<2, u32> = tensor.eq(1.0);

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0, 0]], 1);
    assert_eq!(output[[0, 1]], 0);
    assert_eq!(output[[1, 0]], 0);
    assert_eq!(output[[1, 1]], 0);
    assert_eq!(output[[2, 0]], 0);
    assert_eq!(output[[2, 1]], 0);
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
        self.unary_nary(NaryFunction::unary(Some("appoximate_exp".to_string()), "let output = bitcast<f32>(i32(input * 12102203.0) + (127 << 23) - 545948);".to_string(), D::WGSL_TYPE, D::WGSL_TYPE))
    }

    pub fn exp(&self) -> Self {
        unary_op(self, Some("exp"), "let output = exp(input);", |grad, input| grad * &input.exp())
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_exp() {
    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.exp();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - data[0][0].exp()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].exp()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].exp()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].exp()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].exp()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].exp()).abs() < 0.001);
}

impl<const R: usize, D: crate::FloatDataType> Tensor<R, D> {
    pub fn exp2(&self) -> Self {
        unary_op(self, Some("exp2"), "let output = exp2(input);", |grad, input| (grad * &input.exp2()) * D::from_f32(0.6931471805599453))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_exp2() {
    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.exp2();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - data[0][0].exp2()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].exp2()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].exp2()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].exp2()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].exp2()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].exp2()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn log(&self) -> Self {
        unary_op(self, Some("log"), "let output = log(input);", |grad, input| grad / input)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_log() {
    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.log();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - data[0][0].ln()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].ln()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].ln()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].ln()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].ln()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].ln()).abs() < 0.001);
}

impl<const R: usize, D: crate::FloatDataType> Tensor<R, D> {
    pub fn log2(&self) -> Self {
        unary_op(self, Some("log2"), "let output = log2(input);", |grad, input| grad / &(input * D::from_f32(0.6931471805599453)))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_log2() {
    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.log2();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - data[0][0].log2()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].log2()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].log2()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].log2()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].log2()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].log2()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn pow_elementwise(&self, exponent: D) -> Self {
        unary_op(self, Some("pow"), format!("let output = pow(input, {exponent});"), move |grad, input| (grad * exponent) * &input.pow_elementwise(exponent - D::one()))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_pow() {
    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.pow_elementwise(2.0);

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - data[0][0].powi(2)).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].powi(2)).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].powi(2)).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].powi(2)).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].powi(2)).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].powi(2)).abs() < 0.001);
}

impl<const R: usize, D: crate::FloatDataType> Tensor<R, D> {
    pub fn sqrt(&self) -> Self {
        unary_op(self, Some("sqrt"), "let output = sqrt(input);", |grad, input| grad / &(input.sqrt() * D::from_f32(2.0)))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_sqrt() {
    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.sqrt();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - data[0][0].sqrt()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].sqrt()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].sqrt()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].sqrt()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].sqrt()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].sqrt()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn sin(&self) -> Self {
        unary_op(self, Some("sin"), "let output = sin(input);", |grad, input| grad * &input.cos())
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_sin() {
    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.sin();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - data[0][0].sin()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].sin()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].sin()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].sin()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].sin()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].sin()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn cos(&self) -> Self {
        unary_op(self, Some("cos"), "let output = cos(input);", |grad, input| -(grad * &input.sin()))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_cos() {
    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.cos();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - data[0][0].cos()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].cos()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].cos()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].cos()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].cos()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].cos()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn tan(&self) -> Self {
        self.unary_nary(NaryFunction::unary(Some("tan".to_string()), "let output = tan(input);".to_string(), D::WGSL_TYPE, D::WGSL_TYPE))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_tan() {
    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.tan();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - data[0][0].tan()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].tan()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].tan()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].tan()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].tan()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].tan()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn asin(&self) -> Self {
        self.unary_nary(NaryFunction::unary(Some("asin".to_string()), "let output = asin(input);".to_string(), D::WGSL_TYPE, D::WGSL_TYPE))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_asin() {
    let device = Device::test_instance();

    let data = [
        [1.0f32.sin(), 2.0f32.sin()],
        [3.0f32.sin(), 4.0f32.sin()],
        [5.0f32.sin(), 6.0f32.sin()],
    ];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.asin();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - data[0][0].asin()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].asin()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].asin()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].asin()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].asin()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].asin()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn acos(&self) -> Self {
        self.unary_nary(NaryFunction::unary(Some("acos".to_string()), "let output = acos(input);".to_string(), D::WGSL_TYPE, D::WGSL_TYPE))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_acos() {
    let device = Device::test_instance();

    let data = [
        [1.0f32.cos(), 2.0f32.cos()],
        [3.0f32.cos(), 4.0f32.cos()],
        [5.0f32.cos(), 6.0f32.cos()],
    ];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.acos();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - data[0][0].acos()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].acos()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].acos()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].acos()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].acos()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].acos()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn atan(&self) -> Self {
        self.unary_nary(NaryFunction::unary(Some("atan".to_string()), "let output = atan(input);".to_string(), D::WGSL_TYPE, D::WGSL_TYPE))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_atan() {
    let device = Device::test_instance();

    let data = [[1. / 1., 1. / 2.], [1. / 3., 1. / 4.], [1. / 5., 1. / 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.atan();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - data[0][0].atan()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].atan()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].atan()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].atan()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].atan()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].atan()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn sinh(&self) -> Self {
        self.unary_nary(NaryFunction::unary(Some("sinh".to_string()), "let output = sinh(input);".to_string(), D::WGSL_TYPE, D::WGSL_TYPE))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_sinh() {
    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.sinh();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - data[0][0].sinh()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].sinh()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].sinh()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].sinh()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].sinh()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].sinh()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn cosh(&self) -> Self {
        self.unary_nary(NaryFunction::unary(Some("cosh".to_string()), "let output = cosh(input);".to_string(), D::WGSL_TYPE, D::WGSL_TYPE))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_cosh() {
    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.cosh();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - data[0][0].cosh()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].cosh()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].cosh()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].cosh()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].cosh()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].cosh()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn tanh(&self) -> Self {
        unary_op(self, Some("tanh"), "let output = tanh(input);", |grad, input| {
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
        unary_op(self, Some("tanh_exact"), "let output = (exp(input) - exp(-input)) / (exp(input) + exp(-input));", |grad, input| {
            let output = input.tanh_exact();
            let ones = Tensor::splat(input.device(), D::one(), *input.shape());
            let squared = &output * &output;
            grad * &(ones - squared)
        })
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_tanh() {
    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.tanh();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - data[0][0].tanh()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].tanh()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].tanh()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].tanh()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].tanh()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].tanh()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn asinh(&self) -> Self {
        self.unary_nary(NaryFunction::unary(Some("asinh".to_string()), "let output = asinh(input);".to_string(), D::WGSL_TYPE, D::WGSL_TYPE))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_asinh() {
    let device = Device::test_instance();

    let data = [
        [1.0f32.sinh(), 2.0f32.sinh()],
        [3.0f32.sinh(), 4.0f32.sinh()],
        [5.0f32.sinh(), 6.0f32.sinh()],
    ];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.asinh();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - data[0][0].asinh()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].asinh()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].asinh()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].asinh()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].asinh()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].asinh()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn acosh(&self) -> Self {
        self.unary_nary(NaryFunction::unary(Some("acosh".to_string()), "let output = acosh(input);".to_string(), D::WGSL_TYPE, D::WGSL_TYPE))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_acosh() {
    let device = Device::test_instance();

    let data = [
        [1.0f32.cosh(), 2.0f32.cosh()],
        [3.0f32.cosh(), 4.0f32.cosh()],
        [5.0f32.cosh(), 6.0f32.cosh()],
    ];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.acosh();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - data[0][0].acosh()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].acosh()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].acosh()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].acosh()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].acosh()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].acosh()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn atanh(&self) -> Self {
        self.unary_nary(NaryFunction::unary(Some("atanh".to_string()), "let output = atanh(input);".to_string(), D::WGSL_TYPE, D::WGSL_TYPE))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_atanh() {
    let device = Device::test_instance();

    let data = [
        [1.0f32.tanh(), 2.0f32.tanh()],
        [3.0f32.tanh(), 4.0f32.tanh()],
        [5.0f32.tanh(), 6.0f32.tanh()],
    ];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.atanh();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - data[0][0].atanh()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].atanh()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].atanh()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].atanh()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].atanh()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].atanh()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn abs(&self) -> Self {
        self.unary_nary(NaryFunction::unary(Some("abs".to_string()), "let output = abs(input);".to_string(), D::WGSL_TYPE, D::WGSL_TYPE))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_abs() {
    let device = Device::test_instance();

    let data = [[1., -2.], [-3., 4.], [5., -6.]];

    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.abs();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - data[0][0].abs()).abs() < 0.001);
    assert!((output[[0, 1]] - data[0][1].abs()).abs() < 0.001);
    assert!((output[[1, 0]] - data[1][0].abs()).abs() < 0.001);
    assert!((output[[1, 1]] - data[1][1].abs()).abs() < 0.001);
    assert!((output[[2, 0]] - data[2][0].abs()).abs() < 0.001);
    assert!((output[[2, 1]] - data[2][1].abs()).abs() < 0.001);
}

impl<const R: usize, D: DataType> Neg for Tensor<R, D> {
    type Output = Tensor<R, D>;

    fn neg(self) -> Self {
        unary_op(&self, Some("neg"), "let output = -input;", |grad, _input| -grad)
    }
}

impl<const R: usize, D: DataType> Neg for &Tensor<R, D> {
    type Output = Tensor<R, D>;

    fn neg(self) -> Self::Output {
        -self.clone()
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_neg() {
    let device = Device::test_instance();

    let data = [[1., -2.], [-3., 4.], [5., -6.]];

    let tensor = Tensor::new(&device, &data);

    let tensor = -tensor;

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] + data[0][0]).abs() < 0.001);
    assert!((output[[0, 1]] + data[0][1]).abs() < 0.001);
    assert!((output[[1, 0]] + data[1][0]).abs() < 0.001);
    assert!((output[[1, 1]] + data[1][1]).abs() < 0.001);
    assert!((output[[2, 0]] + data[2][0]).abs() < 0.001);
    assert!((output[[2, 1]] + data[2][1]).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn max_elementwise(&self, element: D) -> Self {
        let element_str = element.to_string();
        unary_op(self, Some("max"), format!("let output = max(input, {element});"), move |grad, input| grad * &greater_than_const_mask(input, &element_str))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_max() {
    let device = Device::test_instance();

    let data = [[1., -2.], [-3., 4.], [5., -6.]];

    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.max_elementwise(0.0);

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - output[[0, 0]]).abs() < 0.001);
    assert!((output[[0, 1]] - 0.0).abs() < 0.001);
    assert!((output[[1, 0]] - 0.0).abs() < 0.001);
    assert!((output[[1, 1]] - output[[1, 1]]).abs() < 0.001);
    assert!((output[[2, 0]] - output[[2, 0]]).abs() < 0.001);
    assert!((output[[2, 1]] - 0.0).abs() < 0.001);
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn min_elementwise(&self, element: D) -> Self {
        let element_str = element.to_string();
        unary_op(self, Some("min"), format!("let output = min(input, {element});"), move |grad, input| grad * &less_than_const_mask(input, &element_str))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_min() {
    let device = Device::test_instance();

    let data = [[1., -2.], [-3., 4.], [5., -6.]];

    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.min_elementwise(0.0);

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0, 0]] - 0.0).abs() < 0.001);
    assert!((output[[0, 1]] - output[[0, 1]]).abs() < 0.001);
    assert!((output[[1, 0]] - output[[1, 0]]).abs() < 0.001);
    assert!((output[[1, 1]] - 0.0).abs() < 0.001);
    assert!((output[[2, 0]] - 0.0).abs() < 0.001);
    assert!((output[[2, 1]] - output[[2, 1]]).abs() < 0.001);
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
        tensor.unary_nary(NaryFunction::unary(Some("cast".to_string()), "let output = f32(input);".to_string(), DataTypeEnum::U32, DataTypeEnum::F32))
    }
}

impl CastTensor<half::f16> for u32 {
    fn cast<const R: usize>(tensor: &Tensor<R, Self>) -> Tensor<R, half::f16> {
        tensor.unary_nary(NaryFunction::unary(Some("cast".to_string()), "let output = f16(input);".to_string(), DataTypeEnum::U32, DataTypeEnum::F16))
    }
}

impl CastTensor<half::f16> for f32 {
    fn cast<const R: usize>(tensor: &Tensor<R, Self>) -> Tensor<R, half::f16> {
        unary_op(tensor, Some("cast"), "let output = f16(input);", |grad, _input| grad.cast())
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_f32_to_f16_cast() {
    let device = Device::test_instance();
    if !device.f16_supported() {
        return;
    }

    let data = [[1.0f32, 2.0f32], [3.0f32, 4.0f32], [5.0f32, 6.0f32]];
    let tensor = Tensor::new(&device, &data);

    let tensor: Tensor<2, half::f16> = tensor.cast();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0, 0]], half::f16::from_f32(data[0][0]));
    assert_eq!(output[[0, 1]], half::f16::from_f32(data[0][1]));
    assert_eq!(output[[1, 0]], half::f16::from_f32(data[1][0]));
    assert_eq!(output[[1, 1]], half::f16::from_f32(data[1][1]));
    assert_eq!(output[[2, 0]], half::f16::from_f32(data[2][0]));
    assert_eq!(output[[2, 1]], half::f16::from_f32(data[2][1]));
}

impl CastTensor<f32> for half::f16 {
    fn cast<const R: usize>(tensor: &Tensor<R, Self>) -> Tensor<R, f32> {
        unary_op(tensor, Some("cast"), "let output = f32(input);", |grad, _input| grad.cast())
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_f16_to_f32_cast() {
    let device = Device::test_instance();
    if !device.f16_supported() {
        return;
    }

    let data = [
        [half::f16::from_f32(1.0), half::f16::from_f32(2.0)],
        [half::f16::from_f32(3.0), half::f16::from_f32(4.0)],
        [half::f16::from_f32(5.0), half::f16::from_f32(6.0)],
    ];
    let tensor = Tensor::new(&device, &data);

    let tensor: Tensor<2, f32> = tensor.cast();

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0, 0]], data[0][0].to_f32());
    assert_eq!(output[[0, 1]], data[0][1].to_f32());
    assert_eq!(output[[1, 0]], data[1][0].to_f32());
    assert_eq!(output[[1, 1]], data[1][1].to_f32());
    assert_eq!(output[[2, 0]], data[2][0].to_f32());
    assert_eq!(output[[2, 1]], data[2][1].to_f32());
}

#[cfg(test)]
#[tokio::test]
async fn test_tanh_exact_large_values() {
    let device = Device::test_instance();

    let data = [[4., 5.], [6., 8.], [10., 15.]];
    let tensor = Tensor::new(&device, &data);

    let tensor = tensor.tanh_exact();

    let output = tensor.as_slice().await.unwrap();
    println!("tanh_exact output: {output:?}");
    println!(
        "Expected: {:?}",
        data.iter()
            .flat_map(|row| row.iter().map(|x| x.tanh()))
            .collect::<Vec<_>>()
    );

    for i in 0..3 {
        for j in 0..2 {
            let expected = data[i][j].tanh();
            let actual = output[[i, j]];
            assert!(
                (actual - expected).abs() < 0.01,
                "tanh_exact({}) = {}, expected {}",
                data[i][j],
                actual,
                expected
            );
        }
    }
}
