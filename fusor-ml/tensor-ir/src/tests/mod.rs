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

pub(crate) fn build_attention_ir(
    builder: &mut IrBuilder,
    q: egg::Id,
    k: egg::Id,
    v: egg::Id,
    seq: u32,
    d: u32,
) -> egg::Id {
    let qk_tile = Shape(vec![Dim::Const(seq), Dim::Const(seq), Dim::Const(d)]);
    let q_r = builder.restride(
        q,
        qk_tile.clone(),
        Strides(vec![Dim::Const(d), Dim::Const(0), Dim::Const(1)]),
    );
    let k_r = builder.restride(
        k,
        qk_tile.clone(),
        Strides(vec![Dim::Const(0), Dim::Const(d), Dim::Const(1)]),
    );
    let arg0 = builder.scalar_arg(0);
    let arg1 = builder.scalar_arg(1);
    let mul_body = builder.bin_op(BinaryOp::Mul, arg0, arg1);
    let qk_mul = builder.elementwise(qk_tile, &[q_r, k_r], mul_body);
    let scores = builder.reduce(qk_mul, 2, ReduceOp::Add);

    let scores_shape = Shape(vec![Dim::Const(seq), Dim::Const(seq)]);
    let probs = builder.softmax(scores, scores_shape, 1);

    let pv_tile = Shape(vec![Dim::Const(seq), Dim::Const(d), Dim::Const(seq)]);
    let p_r = builder.restride(
        probs,
        pv_tile.clone(),
        Strides(vec![Dim::Const(seq), Dim::Const(0), Dim::Const(1)]),
    );
    let v_r = builder.restride(
        v,
        pv_tile.clone(),
        Strides(vec![Dim::Const(0), Dim::Const(1), Dim::Const(d)]),
    );
    let arg0 = builder.scalar_arg(0);
    let arg1 = builder.scalar_arg(1);
    let mul_body = builder.bin_op(BinaryOp::Mul, arg0, arg1);
    let pv_mul = builder.elementwise(pv_tile, &[p_r, v_r], mul_body);
    builder.reduce(pv_mul, 2, ReduceOp::Add)
}

pub(crate) fn build_softmax_weighted_reduce_ir(
    builder: &mut IrBuilder,
    scores: egg::Id,
    values: egg::Id,
    rows: u32,
    outputs: u32,
    weights: u32,
) -> egg::Id {
    let scores_shape = Shape(vec![Dim::Const(rows), Dim::Const(weights)]);
    let probs = builder.softmax(scores, scores_shape, 1);

    let outer_shape = Shape(vec![
        Dim::Const(rows),
        Dim::Const(outputs),
        Dim::Const(weights),
    ]);
    let probs_r = builder.restride(
        probs,
        outer_shape.clone(),
        Strides(vec![Dim::Const(weights), Dim::Const(0), Dim::Const(1)]),
    );
    let values_r = builder.restride(
        values,
        outer_shape.clone(),
        Strides(vec![Dim::Const(0), Dim::Const(1), Dim::Const(outputs)]),
    );
    let arg0 = builder.scalar_arg(0);
    let arg1 = builder.scalar_arg(1);
    let mul_body = builder.bin_op(BinaryOp::Mul, arg0, arg1);
    let weighted = builder.elementwise(outer_shape, &[probs_r, values_r], mul_body);
    builder.reduce(weighted, 2, ReduceOp::Add)
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
mod dispatch;
mod extras;
mod lowering;
#[cfg(feature = "runtime")]
mod runtime_fuzz;
mod runtime_gpu;
