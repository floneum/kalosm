use std::ops::{Add, Div, Mul, Sub};

use crate::{
    Tensor,
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
