use std::ops::{Add, Div, Mul, Sub};

use crate::{
    DataType, Tensor,
    nary_wise::{NaryFunction, NaryOp},
};

fn binary_op(lhs: &Tensor, rhs: &Tensor, name: &str, operation: NaryOp) -> Tensor {
    assert_eq!(lhs.datatype(), rhs.datatype());
    lhs.binary_nary(
        rhs,
        NaryFunction::binary(
            Some(name.to_string()),
            operation,
            lhs.datatype(),
            rhs.datatype(),
            lhs.datatype(),
        ),
    )
}

macro_rules! impl_pairwise_op {
    ($trait:ident, $method:ident, $nary_op:expr, $op_name:literal, $broadcast_method:ident, {$op:tt}) => {
        impl $trait<Tensor> for Tensor {
            type Output = Tensor;

            fn $method(self, rhs: Tensor) -> Self::Output {
                binary_op(&self, &rhs, $op_name, $nary_op)
            }
        }

        impl $trait<&Tensor> for &Tensor {
            type Output = Tensor;

            fn $method(self, rhs: &Tensor) -> Self::Output {
                binary_op(self, rhs, $op_name, $nary_op)
            }
        }

        impl $trait<&Tensor> for Tensor {
            type Output = Tensor;

            fn $method(self, rhs: &Tensor) -> Self::Output {
                (&self).$method(rhs)
            }
        }

        impl $trait<Tensor> for &Tensor {
            type Output = Tensor;

            fn $method(self, rhs: Tensor) -> Self::Output {
                self.$method(&rhs)
            }
        }

        impl Tensor {
            pub fn $broadcast_method(&self, second: &Tensor) -> Tensor {
                Tensor::broadcast_then_elementwise_op(self, second, |a, b| a $op b)
            }
        }
    };
}

impl_pairwise_op!(Add, add, NaryOp::Add, "add", add_, {+});
impl_pairwise_op!(Sub, sub, NaryOp::Sub, "sub", sub_, {-});
impl_pairwise_op!(Mul, mul, NaryOp::Mul, "mul", mul_, {*});
impl_pairwise_op!(Div, div, NaryOp::Div, "div", div_, {/});

macro_rules! impl_pairwise_method {
    ($method:ident, $nary_op:expr, $op_name:literal, $broadcast_method:ident, |$a:ident, $b:ident| $expr:expr) => {
        impl Tensor {
            pub fn $method(&self, other: &Self) -> Self {
                binary_op(self, other, $op_name, $nary_op)
            }

            pub fn $broadcast_method(&self, second: &Tensor) -> Tensor {
                Tensor::broadcast_then_elementwise_op(self, second, |$a, $b| $expr)
            }
        }
    };
}

impl_pairwise_method!(pow, NaryOp::Pow, "pow", pow_, |a, b| a.pow(&b));

/// Emit a tensor-tensor comparison method producing 1/0 in the output type
/// `D`, mirroring the scalar comparisons in `element_wise`.
macro_rules! impl_pairwise_cmp {
    ($(#[$meta:meta])* $method:ident, $nary_op:expr, $op_name:literal) => {
        impl Tensor {
            $(#[$meta])*
            pub fn $method<D: DataType>(&self, other: &Self) -> Tensor {
                assert_eq!(self.datatype(), other.datatype());
                self.binary_nary(
                    other,
                    NaryFunction::binary(
                        Some($op_name.to_string()),
                        $nary_op,
                        self.datatype(),
                        other.datatype(),
                        D::DATA_TYPE,
                    ),
                )
            }
        }
    };
}

impl_pairwise_cmp!(
    /// Element-wise `self == other` returning 1 for true and 0 for false.
    eq_tensor, NaryOp::Equal, "eq"
);
impl_pairwise_cmp!(
    /// Element-wise `self != other` returning 1 for true and 0 for false.
    ne_tensor, NaryOp::NotEqual, "ne"
);
impl_pairwise_cmp!(
    /// Element-wise `self < other` returning 1 for true and 0 for false.
    lt_tensor, NaryOp::Less, "lt"
);
impl_pairwise_cmp!(
    /// Element-wise `self <= other` returning 1 for true and 0 for false.
    lte_tensor, NaryOp::LessEqual, "lte"
);
impl_pairwise_cmp!(
    /// Element-wise `self > other` returning 1 for true and 0 for false.
    gt_tensor, NaryOp::Greater, "gt"
);
impl_pairwise_cmp!(
    /// Element-wise `self >= other` returning 1 for true and 0 for false.
    gte_tensor, NaryOp::GreaterEqual, "gte"
);
