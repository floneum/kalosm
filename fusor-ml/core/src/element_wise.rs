use std::{
    fmt::Display,
    iter::Sum,
    ops::{Add, Div, Mul, Neg, Rem, Sub},
};

use crate::{
    Tensor,
    compute_graph::NodeIndex,
    mir::{function::Function, kernel::GenericKernel},
    tensor::{DataType, DataTypeEnum},
};

#[cfg(test)]
use crate::Device;

#[derive(Clone, Debug)]
pub(crate) struct ElementWiseFunctions {
    input_datatype: DataTypeEnum,
    pub(crate) functions: Vec<ElementWiseFunction>,
}

impl ElementWiseFunctions {
    pub fn new(functions: Vec<ElementWiseFunction>, input_datatype: DataTypeEnum) -> Self {
        Self {
            input_datatype,
            functions,
        }
    }

    pub fn empty(input_datatype: DataTypeEnum) -> Self {
        Self {
            input_datatype,
            functions: Vec::new(),
        }
    }

    pub fn add_functions(&self, kernel: &mut GenericKernel) -> Vec<Function> {
        let mut input_datatype = self.input_datatype;
        self.functions
            .iter()
            .rev()
            .map(|f| {
                let function = kernel.add_function(
                    f.datatype,
                    f.operation.clone(),
                    [("input".to_string(), input_datatype.to_string())],
                );
                input_datatype = f.datatype;
                function
            })
            .collect()
    }

    pub fn input_datatype(&self) -> DataTypeEnum {
        self.input_datatype
    }

    pub fn out_datatype(&self) -> DataTypeEnum {
        if let Some(first) = self.functions.first() {
            first.datatype
        } else {
            self.input_datatype
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ElementWiseOperation {
    pub(crate) value: NodeIndex,
    pub(crate) functions: ElementWiseFunctions,
    pub(crate) shape: Box<[usize]>,
}

impl ElementWiseOperation {
    pub fn new(
        input_datatype: DataTypeEnum,
        value: NodeIndex,
        functions: ElementWiseFunction,
        shape: impl Into<Box<[usize]>>,
    ) -> Self {
        Self {
            value,
            functions: ElementWiseFunctions {
                input_datatype,
                functions: vec![functions],
            },
            shape: shape.into(),
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub(crate) fn name(&self) -> String {
        self.functions
            .functions
            .iter()
            .map(|f| f.name())
            .collect::<Vec<_>>()
            .join("_")
    }
}

#[derive(Clone, Debug)]
pub struct ElementWiseFunction {
    pub(crate) name: Option<String>,
    pub(crate) operation: String,
    pub(crate) datatype: DataTypeEnum,
}

impl ElementWiseFunction {
    pub fn new(operation: impl Display, datatype: DataTypeEnum) -> Self {
        Self {
            name: None,
            operation: operation.to_string(),
            datatype,
        }
    }

    pub(crate) fn with_name(mut self, name: impl ToString) -> Self {
        self.name = Some(name.to_string());
        self
    }

    pub(crate) fn name(&self) -> &str {
        self.name.as_deref().unwrap_or("element_wise")
    }

    pub(crate) fn to_nary_function(
        &self,
        input_type: DataTypeEnum,
    ) -> crate::nary_wise::NaryFunction {
        crate::nary_wise::NaryFunction {
            name: self.name.clone(),
            operation: self.operation.clone(),
            input_names: vec!["input".to_string()],
            input_types: vec![input_type],
            output_type: self.datatype,
        }
    }
}

impl<const R: usize, T: DataType> Add<T> for Tensor<R, T> {
    type Output = Tensor<R, T>;

    fn add(self, rhs: T) -> Self::Output {
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new(format!("let output = input + {rhs};"), T::WGSL_TYPE)
                .with_name("add_const"),
            self.shape().as_slice(),
        ))
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
    let sliced = tensor.slice([0..3, 0..1]);

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
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new(format!("let output = input - {rhs};"), T::WGSL_TYPE)
                .with_name("subtract_const"),
            self.shape().as_slice(),
        ))
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
                    rhs.element_wise(ElementWiseOperation::new(
                        rhs.datatype(),
                        rhs.key(),
                        ElementWiseFunction::new(
                            format!("let output = {self} - input;"),
                            <$t>::WGSL_TYPE,
                        )
                        .with_name("subtract_const"),
                        rhs.shape().as_slice(),
                    ))
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
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new(format!("let output = input * {rhs};"), T::WGSL_TYPE)
                .with_name("multiply_const"),
            self.shape().as_slice(),
        ))
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
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new(format!("let output = input / {rhs};"), T::WGSL_TYPE)
                .with_name("divide_const"),
            self.shape().as_slice(),
        ))
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
                    rhs.element_wise(ElementWiseOperation::new(
                        rhs.datatype(),
                        rhs.key(),
                        ElementWiseFunction::new(
                            format!("let output = {} / input;", self),
                            <$t>::WGSL_TYPE,
                        )
                        .with_name("divide_const"),
                        rhs.shape().as_slice(),
                    ))
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
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new(format!("let output = input % {rhs};"), u32::WGSL_TYPE)
                .with_name("mod_const"),
            self.shape().as_slice(),
        ))
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
                    rhs.element_wise(ElementWiseOperation::new(
                        rhs.datatype(),
                        rhs.key(),
                        ElementWiseFunction::new(
                            format!("let output = {} % input;", self),
                            <$t>::WGSL_TYPE,
                        )
                        .with_name("mod_const"),
                        rhs.shape().as_slice(),
                    ))
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
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new(
                format!("let output = {datatype}(input == {rhs});"),
                D::WGSL_TYPE,
            )
            .with_name("equal_const"),
            self.shape().as_slice(),
        ))
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
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new(
                format!("let output = {datatype}(input < {rhs});"),
                D::WGSL_TYPE,
            )
            .with_name("lt_const"),
            self.shape().as_slice(),
        ))
    }

    /// Check if each value in the tensor is less than or equal to the given value. Returns 1 for true and 0 for false.
    pub fn lte<D: DataType>(&self, rhs: T) -> Tensor<R, D> {
        let datatype = D::WGSL_TYPE;
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new(
                format!("let output = {datatype}(input <= {rhs});"),
                D::WGSL_TYPE,
            )
            .with_name("lt_const"),
            self.shape().as_slice(),
        ))
    }

    /// Check if each value in the tensor is more than to the given value. Returns 1 for true and 0 for false.
    pub fn mt<D: DataType>(&self, rhs: T) -> Tensor<R, D> {
        let datatype = D::WGSL_TYPE;
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new(
                format!("let output = {datatype}(input > {rhs});"),
                D::WGSL_TYPE,
            )
            .with_name("lt_const"),
            self.shape().as_slice(),
        ))
    }

    /// Check if each value in the tensor is more than or equal to the given value. Returns 1 for true and 0 for false.
    pub fn mte<D: DataType>(&self, rhs: T) -> Tensor<R, D> {
        let datatype = D::WGSL_TYPE;
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new(
                format!("let output = {datatype}(input >= {rhs});"),
                D::WGSL_TYPE,
            )
            .with_name("lt_const"),
            self.shape().as_slice(),
        ))
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
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new(
                "let first_order = i32(input * 12102203.0) + (127 << 23) - 345088;
                let correction_xi = (first_order & 0x7fffff) | (127 << 23);
                let correction_x = bitcast<f32>(correction_xi);
                let output = bitcast<f32>(first_order) * fma(fma(correction_x, 0.22670517861843109130859375, -0.671999752521514892578125), correction_x, 1.469318866729736328125);",
                D::WGSL_TYPE,
            )
            .with_name("less_appoximate_exp"),
            self.shape().as_slice(),
        ))
    }

    pub fn appoximate_exp(&self) -> Self {
        if D::WGSL_TYPE != DataTypeEnum::F32 {
            return self.exp();
        }
        // https://specbranch.com/posts/fast-exp/
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new(
                "let output = bitcast<f32>(i32(input * 12102203.0) + (127 << 23) - 545948);",
                D::WGSL_TYPE,
            )
            .with_name("appoximate_exp"),
            self.shape().as_slice(),
        ))
    }

    pub fn exp(&self) -> Self {
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new("let output = exp(input);", D::WGSL_TYPE).with_name("exp"),
            self.shape().as_slice(),
        ))
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

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn exp2(&self) -> Self {
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new("let output = exp2(input);", D::WGSL_TYPE).with_name("exp2"),
            self.shape().as_slice(),
        ))
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
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new("let output = log(input);", D::WGSL_TYPE).with_name("log"),
            self.shape().as_slice(),
        ))
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

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn log2(&self) -> Self {
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new("let output = log2(input);", D::WGSL_TYPE).with_name("log2"),
            self.shape().as_slice(),
        ))
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
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new(
                format!("let output = pow(input, {exponent});"),
                D::WGSL_TYPE,
            )
            .with_name("pow"),
            self.shape().as_slice(),
        ))
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

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn sqrt(&self) -> Self {
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new("let output = sqrt(input);", D::WGSL_TYPE).with_name("sqrt"),
            self.shape().as_slice(),
        ))
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
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new("let output = sin(input);", D::WGSL_TYPE).with_name("sin"),
            self.shape().as_slice(),
        ))
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
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new("let output = cos(input);", D::WGSL_TYPE).with_name("cos"),
            self.shape().as_slice(),
        ))
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
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new("let output = tan(input);", D::WGSL_TYPE).with_name("tan"),
            self.shape().as_slice(),
        ))
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
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new("let output = asin(input);", D::WGSL_TYPE).with_name("asin"),
            self.shape().as_slice(),
        ))
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
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new("let output = acos(input);", D::WGSL_TYPE).with_name("acos"),
            self.shape().as_slice(),
        ))
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
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new("let output = atan(input);", D::WGSL_TYPE).with_name("atan"),
            self.shape().as_slice(),
        ))
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
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new("let output = sinh(input);", D::WGSL_TYPE).with_name("sinh"),
            self.shape().as_slice(),
        ))
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
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new("let output = cosh(input);", D::WGSL_TYPE).with_name("cosh"),
            self.shape().as_slice(),
        ))
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
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new("let output = tanh(input);", D::WGSL_TYPE).with_name("tanh"),
            self.shape().as_slice(),
        ))
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    /// Calculates tanh with (e^x - e^-x) / (e^x + e^-x)
    pub fn tanh_exact(&self) -> Self {
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new(
                "let output = (exp(input) - exp(-input)) / (exp(input) + exp(-input));",
                D::WGSL_TYPE,
            )
            .with_name("tanh_exact"),
            self.shape().as_slice(),
        ))
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
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new("let output = asinh(input);", D::WGSL_TYPE).with_name("asinh"),
            self.shape().as_slice(),
        ))
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
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new("let output = acosh(input);", D::WGSL_TYPE).with_name("acosh"),
            self.shape().as_slice(),
        ))
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
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new("let output = atanh(input);", D::WGSL_TYPE).with_name("atanh"),
            self.shape().as_slice(),
        ))
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
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new("let output = abs(input);", D::WGSL_TYPE).with_name("abs"),
            self.shape().as_slice(),
        ))
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
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new("let output = -input;", D::WGSL_TYPE).with_name("neg"),
            self.shape().as_slice(),
        ))
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
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new(format!("let output = max(input, {element});"), D::WGSL_TYPE)
                .with_name("max"),
            self.shape().as_slice(),
        ))
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
        self.element_wise(ElementWiseOperation::new(
            self.datatype(),
            self.key(),
            ElementWiseFunction::new(format!("let output = min(input, {element});"), D::WGSL_TYPE)
                .with_name("max"),
            self.shape().as_slice(),
        ))
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
        tensor.element_wise(ElementWiseOperation::new(
            tensor.datatype(),
            tensor.key(),
            ElementWiseFunction::new("let output = f32(input);", DataTypeEnum::F32)
                .with_name("cast"),
            tensor.shape().as_slice(),
        ))
    }
}

impl CastTensor<half::f16> for u32 {
    fn cast<const R: usize>(tensor: &Tensor<R, Self>) -> Tensor<R, half::f16> {
        tensor.element_wise(ElementWiseOperation::new(
            tensor.datatype(),
            tensor.key(),
            ElementWiseFunction::new("let output = f16(input);", DataTypeEnum::F16)
                .with_name("cast"),
            tensor.shape().as_slice(),
        ))
    }
}

impl CastTensor<half::f16> for f32 {
    fn cast<const R: usize>(tensor: &Tensor<R, Self>) -> Tensor<R, half::f16> {
        tensor.element_wise(ElementWiseOperation::new(
            tensor.datatype(),
            tensor.key(),
            ElementWiseFunction::new("let output = f16(input);", DataTypeEnum::F16)
                .with_name("cast"),
            tensor.shape().as_slice(),
        ))
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
        tensor.element_wise(ElementWiseOperation::new(
            tensor.datatype(),
            tensor.key(),
            ElementWiseFunction::new("let output = f32(input);", DataTypeEnum::F32)
                .with_name("cast"),
            tensor.shape().as_slice(),
        ))
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
