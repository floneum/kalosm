use std::ops::{Add, Div, Mul, Sub};

use crate::{MaxRank, Tensor, nary_wise::NaryFunction, tensor::DataType};
use tensor_ir::BinaryOp;

fn binary_op<const R: usize, T: DataType>(
    lhs: &Tensor<R, T>,
    rhs: &Tensor<R, T>,
    name: &str,
    op: BinaryOp,
) -> Tensor<R, T> {
    lhs.binary_nary(
        rhs,
        NaryFunction::binary(
            Some(name.to_string()),
            op,
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
    ($trait:ident, $method:ident, $ir_op:expr, $op_name:literal, $broadcast_method:ident, {$op:tt}) => {
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
                binary_op(self, rhs, $op_name, $ir_op)
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
    BinaryOp::Add,
    "add",
    add_,
    {+}
);

impl_pairwise_op!(
    Sub,
    sub,
    BinaryOp::Sub,
    "sub",
    sub_,
    {-}
);

impl_pairwise_op!(
    Mul,
    mul,
    BinaryOp::Mul,
    "mul",
    mul_,
    {*}
);

impl_pairwise_op!(
    Div,
    div,
    BinaryOp::Div,
    "div",
    div_,
    {/}
);

/// Macro to implement method-based pairwise operations (like pow, min, max).
///
/// Unlike `impl_pairwise_op!` which implements std::ops traits, this macro generates
/// regular methods on Tensor for operations that don't have corresponding operators.
macro_rules! impl_pairwise_method {
    ($method:ident, $op_name:literal, $ir_op:expr, $broadcast_method:ident, |$a:ident, $b:ident| $expr:expr) => {
        impl<const R: usize, T: DataType> Tensor<R, T> {
            pub fn $method(&self, other: &Self) -> Self {
                binary_op(self, other, $op_name, $ir_op)
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

impl_pairwise_method!(pow, "pow", BinaryOp::Pow, pow_, |a, b| a.pow(&b));
