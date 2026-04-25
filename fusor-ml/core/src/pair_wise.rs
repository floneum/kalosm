use std::ops::{Add, Div, Mul, Sub};

use crate::{MaxRank, Tensor, nary_wise::NaryFunction, tensor::DataType};

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

impl_pairwise_op!(
    Sub,
    sub,
    "-",
    "sub",
    sub_,
    {-}
);

impl_pairwise_op!(
    Mul,
    mul,
    "*",
    "mul",
    mul_,
    {*}
);

impl_pairwise_op!(
    Div,
    div,
    "/",
    "div",
    div_,
    {/}
);

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

impl<const R: usize, T: DataType> Tensor<R, T> {
    fn cmp_tensor(&self, other: &Self, op: &str, name: &str) -> Self {
        let datatype = T::WGSL_TYPE;
        binary_op(
            self,
            other,
            name,
            &format!("let output = {datatype}(a {op} b);"),
        )
    }

    /// Element-wise equality comparison between two tensors.
    /// Returns 1 (in T) where elements are equal, 0 otherwise.
    pub fn eq_tensor(&self, other: &Self) -> Self {
        self.cmp_tensor(other, "==", "eq_tensor")
    }

    /// Element-wise less-than comparison between two tensors.
    /// Returns 1 (in T) where self < other, 0 otherwise.
    pub fn lt_tensor(&self, other: &Self) -> Self {
        self.cmp_tensor(other, "<", "lt_tensor")
    }

    /// Element-wise less-than-or-equal comparison between two tensors.
    /// Returns 1 (in T) where self <= other, 0 otherwise.
    pub fn lte_tensor(&self, other: &Self) -> Self {
        self.cmp_tensor(other, "<=", "lte_tensor")
    }

    /// Element-wise greater-than comparison between two tensors.
    /// Returns 1 (in T) where self > other, 0 otherwise.
    pub fn gt_tensor(&self, other: &Self) -> Self {
        self.cmp_tensor(other, ">", "gt_tensor")
    }

    /// Element-wise greater-than-or-equal comparison between two tensors.
    /// Returns 1 (in T) where self >= other, 0 otherwise.
    pub fn gte_tensor(&self, other: &Self) -> Self {
        self.cmp_tensor(other, ">=", "gte_tensor")
    }
}
