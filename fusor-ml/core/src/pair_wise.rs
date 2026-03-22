use std::ops::{Add, Div, Mul, Sub};

use crate::{
    MaxRank, Tensor,
    nary_wise::NaryFunction,
    tensor::DataType,
};

fn binary_op<const R: usize, T: DataType>(
    lhs: &Tensor<R, T>,
    rhs: &Tensor<R, T>,
    name: &str,
    operation: &str,
) -> Tensor<R, T> {
    lhs.binary_nary(
        rhs,
        NaryFunction::binary(
            Some(name.to_string()),
            operation.to_string(),
            T::WGSL_TYPE,
            T::WGSL_TYPE,
            T::WGSL_TYPE,
        ),
    )
}

/// Macro to implement pairwise operators (Add, Sub, Mul, Div) for Tensor.
///
/// Generates all four combinations of owned/reference implementations:
/// - `Tensor op Tensor` (owned + owned)
/// - `&Tensor op &Tensor` (ref + ref) - core implementation
/// - `Tensor op &Tensor` (owned + ref)
/// - `&Tensor op Tensor` (ref + owned)
///
/// Also generates a broadcast method `op_()` for tensors of different ranks.
macro_rules! impl_pairwise_op {
    ($trait:ident, $method:ident, $op_str:literal, $op_name:literal, $broadcast_method:ident, {$op:tt}) => {
        // Owned + Owned: delegates to ref + ref
        impl<const R: usize, T: DataType> $trait<Tensor<R, T>> for Tensor<R, T> {
            type Output = Tensor<R, T>;

            fn $method(self, rhs: Tensor<R, T>) -> Self::Output {
                (&self).$method(&rhs)
            }
        }

        // Ref + Ref: core implementation
        impl<const R: usize, T: DataType> $trait<&Tensor<R, T>> for &Tensor<R, T> {
            type Output = Tensor<R, T>;

            fn $method(self, rhs: &Tensor<R, T>) -> Self::Output {
                binary_op(
                    self,
                    rhs,
                    $op_name,
                    concat!("let output = a ", $op_str, " b;"),
                )
            }
        }

        // Owned + Ref: delegates to ref + ref
        impl<const R: usize, T: DataType> $trait<&Tensor<R, T>> for Tensor<R, T> {
            type Output = Tensor<R, T>;

            fn $method(self, rhs: &Tensor<R, T>) -> Self::Output {
                (&self).$method(rhs)
            }
        }

        // Ref + Owned: delegates to ref + ref
        impl<const R: usize, T: DataType> $trait<Tensor<R, T>> for &Tensor<R, T> {
            type Output = Tensor<R, T>;

            fn $method(self, rhs: Tensor<R, T>) -> Self::Output {
                self.$method(&rhs)
            }
        }

        // Broadcast method for tensors of different ranks
        impl<const R: usize, T: DataType> Tensor<R, T> {
            pub fn $broadcast_method<const R2: usize, const R3: usize>(
                &self,
                second: &Tensor<R2, T>,
            ) -> Tensor<R3, T>
            where
                (Tensor<R, T>, Tensor<R2, T>): MaxRank<R3, T>,
            {
                Self::broadcast_then_elementwise_op(self, second, |a, b| a $op b)
            }
        }
    };
}

impl_pairwise_op!(
    Add,
    add,
    "+",
    "add",
    add_,
    {+}
);

#[cfg(test)]
#[tokio::test]
async fn test_pair_wise_add() {
    use crate::Device;

    let device = Device::test_instance();

    let data_a = [[1., 2.], [3., 4.], [5., 6.]];
    let data_b = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);

    let tensor = &tensor_a + &tensor_b;
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");

    assert_eq!(as_slice[[0, 0]], 1. + 1.);
    assert_eq!(as_slice[[0, 1]], 2. + 2.);
    assert_eq!(as_slice[[1, 0]], 3. + 3.);
    assert_eq!(as_slice[[1, 1]], 4. + 4.);
    assert_eq!(as_slice[[2, 0]], 5. + 5.);
    assert_eq!(as_slice[[2, 1]], 6. + 6.);
}

#[cfg(test)]
#[tokio::test]
async fn test_pair_wise_add_f16() {
    use crate::Device;

    let device = Device::test_instance();
    if !device.f16_supported() {
        return;
    }

    let data_a = [
        [half::f16::from_f32(1.), half::f16::from_f32(2.)],
        [half::f16::from_f32(3.), half::f16::from_f32(4.)],
        [half::f16::from_f32(5.), half::f16::from_f32(6.)],
    ];
    let data_b = [
        [half::f16::from_f32(1.), half::f16::from_f32(2.)],
        [half::f16::from_f32(3.), half::f16::from_f32(4.)],
        [half::f16::from_f32(5.), half::f16::from_f32(6.)],
    ];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);

    let tensor = &tensor_a + &tensor_b;
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");

    assert_eq!(as_slice[[0, 0]], half::f16::from_f32(1. + 1.));
    assert_eq!(as_slice[[0, 1]], half::f16::from_f32(2. + 2.));
    assert_eq!(as_slice[[1, 0]], half::f16::from_f32(3. + 3.));
    assert_eq!(as_slice[[1, 1]], half::f16::from_f32(4. + 4.));
    assert_eq!(as_slice[[2, 0]], half::f16::from_f32(5. + 5.));
    assert_eq!(as_slice[[2, 1]], half::f16::from_f32(6. + 6.));
}

#[cfg(test)]
#[tokio::test]
async fn test_pair_wise_add_u32() {
    use crate::Device;

    let device = Device::test_instance();

    let data_a = [[1_u32, 2_u32], [3_u32, 4_u32], [5_u32, 6_u32]];
    let data_b = [[1_u32, 2_u32], [3_u32, 4_u32], [5_u32, 6_u32]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);

    let tensor = &tensor_a + &tensor_b;
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");

    assert_eq!(as_slice[[0, 0]], 1 + 1);
    assert_eq!(as_slice[[0, 1]], 2 + 2);
    assert_eq!(as_slice[[1, 0]], 3 + 3);
    assert_eq!(as_slice[[1, 1]], 4 + 4);
    assert_eq!(as_slice[[2, 0]], 5 + 5);
    assert_eq!(as_slice[[2, 1]], 6 + 6);
}

#[cfg(test)]
#[tokio::test]
async fn test_pair_wise_add_const_mul_const_add_fused() {
    use crate::Device;

    let device = Device::test_instance();

    let data_a = [[1., 2.], [3., 4.], [5., 6.]];
    let data_b = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);

    let tensor = &(tensor_a + 1.) + &(tensor_b * 2.);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");

    assert_eq!(as_slice[[0, 0]], (1. + 1.) + (1. * 2.));
    assert_eq!(as_slice[[0, 1]], (2. + 1.) + (2. * 2.));
    assert_eq!(as_slice[[1, 0]], (3. + 1.) + (3. * 2.));
    assert_eq!(as_slice[[1, 1]], (4. + 1.) + (4. * 2.));
    assert_eq!(as_slice[[2, 0]], (5. + 1.) + (5. * 2.));
    assert_eq!(as_slice[[2, 1]], (6. + 1.) + (6. * 2.));
}

#[cfg(test)]
#[tokio::test]
async fn test_pair_wise_add_sub_const_fused() {
    use crate::Device;

    let device = Device::test_instance();

    let data_a = [[1., 2.], [3., 4.], [5., 6.]];
    let data_b = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);

    let tensor = (&tensor_a + &tensor_b) - 1.;
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");

    assert_eq!(as_slice[[0, 0]], 1. + 1. - 1.);
    assert_eq!(as_slice[[0, 1]], 2. + 2. - 1.);
    assert_eq!(as_slice[[1, 0]], 3. + 3. - 1.);
    assert_eq!(as_slice[[1, 1]], 4. + 4. - 1.);
    assert_eq!(as_slice[[2, 0]], 5. + 5. - 1.);
    assert_eq!(as_slice[[2, 1]], 6. + 6. - 1.);
}

#[cfg(test)]
#[tokio::test]
async fn test_pair_wise_add_sparse() {
    use crate::Device;

    let device = Device::test_instance();

    let data_a = [[1., 2.], [3., 4.], [5., 6.]];
    let data_b = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_a = tensor_a.restride([
        crate::StrideSpec::dim(0, 3),
        crate::StrideSpec::dim(1, 1),
    ]);
    let tensor_b = Tensor::new(&device, &data_b);
    let tensor_b = tensor_b.restride([
        crate::StrideSpec::dim(0, 3),
        crate::StrideSpec::dim(1, 1),
    ]);

    let tensor = &tensor_a + &tensor_b;
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");

    assert_eq!(as_slice[[0, 0]], 1. + 1.);
    assert_eq!(as_slice[[1, 0]], 3. + 3.);
    assert_eq!(as_slice[[2, 0]], 5. + 5.);
}

impl_pairwise_op!(
    Sub,
    sub,
    "-",
    "sub",
    sub_,
    {-}
);

#[cfg(test)]
#[tokio::test]
async fn test_pair_wise_sub() {
    use crate::Device;

    let device = Device::test_instance();

    let data_a = [[1., 2.], [3., 4.], [5., 6.]];
    let data_b = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);

    let tensor = &tensor_a - &tensor_b;
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");

    assert_eq!(as_slice[[0, 0]], 1. - 1.);
    assert_eq!(as_slice[[0, 1]], 2. - 2.);
    assert_eq!(as_slice[[1, 0]], 3. - 3.);
    assert_eq!(as_slice[[1, 1]], 4. - 4.);
    assert_eq!(as_slice[[2, 0]], 5. - 5.);
    assert_eq!(as_slice[[2, 1]], 6. - 6.);
}

impl_pairwise_op!(
    Mul,
    mul,
    "*",
    "mul",
    mul_,
    {*}
);

#[cfg(test)]
#[tokio::test]
async fn test_pair_wise_mul() {
    use crate::Device;

    let device = Device::test_instance();

    let data_a = [[1., 2.], [3., 4.], [5., 6.]];
    let data_b = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);

    let tensor = &tensor_a * &tensor_b;
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");

    assert_eq!(as_slice[[0, 0]], 1. * 1.);
    assert_eq!(as_slice[[0, 1]], 2. * 2.);
    assert_eq!(as_slice[[1, 0]], 3. * 3.);
    assert_eq!(as_slice[[1, 1]], 4. * 4.);
    assert_eq!(as_slice[[2, 0]], 5. * 5.);
    assert_eq!(as_slice[[2, 1]], 6. * 6.);
}

impl_pairwise_op!(
    Div,
    div,
    "/",
    "div",
    div_,
    {/}
);

#[cfg(test)]
#[tokio::test]
async fn test_pair_wise_div() {
    use crate::Device;

    let device = Device::test_instance();

    let data_a = [[1., 4.], [3., 4.], [5., 6.]];
    let data_b = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);

    let tensor = &tensor_a / &tensor_b;
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");

    assert_eq!(as_slice[[0, 0]], 1. / 1.);
    assert_eq!(as_slice[[0, 1]], 4. / 2.);
    assert_eq!(as_slice[[1, 0]], 3. / 3.);
    assert_eq!(as_slice[[1, 1]], 4. / 4.);
    assert_eq!(as_slice[[2, 0]], 5. / 5.);
    assert_eq!(as_slice[[2, 1]], 6. / 6.);
}

/// Macro to implement method-based pairwise operations (like pow, min, max).
///
/// Unlike `impl_pairwise_op!` which implements std::ops traits, this macro generates
/// regular methods on Tensor for operations that don't have corresponding operators.
macro_rules! impl_pairwise_method {
    ($method:ident, $wgsl_op:literal, $op_name:literal, $broadcast_method:ident, |$a:ident, $b:ident| $expr:expr) => {
        impl<const R: usize, T: DataType> Tensor<R, T> {
            pub fn $method(&self, other: &Self) -> Self {
                binary_op(
                    self,
                    other,
                    $op_name,
                    concat!("let output = ", $wgsl_op, ";"),
                )
            }

            pub fn $broadcast_method<const R2: usize, const R3: usize>(
                &self,
                second: &Tensor<R2, T>,
            ) -> Tensor<R3, T>
            where
                (Tensor<R, T>, Tensor<R2, T>): MaxRank<R3, T>,
            {
                Self::broadcast_then_elementwise_op(self, second, |$a, $b| $expr)
            }
        }
    };
}

impl_pairwise_method!(pow, "pow(a, b)", "pow", pow_, |a, b| a.pow(&b));

#[cfg(test)]
#[tokio::test]
async fn test_pair_wise_pow() {
    use crate::Device;

    let device = Device::test_instance();

    let data_a = [[1., 2.], [3., 4.], [5., 6.]];
    let data_b = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);

    let tensor = &tensor_a.pow(&tensor_b);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");

    assert!((as_slice[[0, 0]] - 1_f32.powf(1.)) < 0.001);
    assert!((as_slice[[0, 1]] - 2_f32.powf(2.)) < 0.001);
    assert!((as_slice[[1, 0]] - 3_f32.powf(3.)) < 0.001);
    assert!((as_slice[[1, 1]] - 4_f32.powf(4.)) < 0.001);
    assert!((as_slice[[2, 0]] - 5_f32.powf(5.)) < 0.001);
    assert!((as_slice[[2, 1]] - 6_f32.powf(6.)) < 0.001);
}
