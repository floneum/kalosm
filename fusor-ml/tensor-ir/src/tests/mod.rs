//! Integration tests for the tensor IR.
//!
//! Split by topic — each submodule is loadable independently. Shared imports
//! are duplicated rather than pulled from a common helpers module to keep each
//! file self-contained.

use crate::builders::IrBuilder;
use crate::stages::{ExprId, TensorExprBuilder};
use crate::types::{BinaryOp, Dim, ReduceOp, Shape, Strides};

pub(crate) fn build_binary_mul_add_contraction_ir(
    builder: &mut IrBuilder,
    lhs: egg::Id,
    rhs: egg::Id,
    rows: u32,
    cols: u32,
    depth: u32,
) -> egg::Id {
    let arg0 = builder.scalar_arg(0);
    let arg1 = builder.scalar_arg(1);
    let body = builder.bin_op(BinaryOp::Mul, arg0, arg1);
    builder.contraction(
        Shape(vec![Dim::Const(rows), Dim::Const(cols), Dim::Const(depth)]),
        &[
            (
                lhs,
                Strides(vec![Dim::Const(depth), Dim::Const(0), Dim::Const(1)]),
            ),
            (
                rhs,
                Strides(vec![Dim::Const(0), Dim::Const(1), Dim::Const(cols)]),
            ),
        ],
        body,
        &[(2, ReduceOp::Add)],
    )
}

pub(crate) fn build_binary_mul_add_contraction_expr(
    builder: &mut TensorExprBuilder,
    lhs: ExprId,
    rhs: ExprId,
    rows: u32,
    cols: u32,
    depth: u32,
) -> ExprId {
    let arg0 = builder.scalar_arg(0);
    let arg1 = builder.scalar_arg(1);
    let body = builder.scalar_binop(BinaryOp::Mul, [arg0, arg1]);
    builder.contraction(
        Shape(vec![Dim::Const(rows), Dim::Const(cols), Dim::Const(depth)]),
        &[
            (
                lhs,
                Strides(vec![Dim::Const(depth), Dim::Const(0), Dim::Const(1)]),
            ),
            (
                rhs,
                Strides(vec![Dim::Const(0), Dim::Const(1), Dim::Const(cols)]),
            ),
        ],
        body,
        &[(2, ReduceOp::Add)],
    )
}

pub(crate) fn build_centered_row_sum_ir(
    builder: &mut IrBuilder,
    x: egg::Id,
    rows: u32,
    cols: u32,
) -> egg::Id {
    let shape = Shape(vec![Dim::Const(rows), Dim::Const(cols)]);
    let row_sum = builder.reduce(x, 1, ReduceOp::Add);
    let row_sum_bcast = builder.restride(
        row_sum,
        shape.clone(),
        Strides(vec![Dim::Const(1), Dim::Const(0)]),
    );
    let arg0 = builder.scalar_arg(0);
    let arg1 = builder.scalar_arg(1);
    let sub = builder.bin_op(BinaryOp::Sub, arg0, arg1);
    let centered = builder.elementwise(shape, &[x, row_sum_bcast], sub);
    builder.reduce(centered, 1, ReduceOp::Add)
}

mod build_pipeline;
mod lowering;
