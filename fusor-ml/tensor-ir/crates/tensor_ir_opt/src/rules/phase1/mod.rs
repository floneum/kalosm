//! Phase 1: High-Level → Naive Dispatch.
//!
//! Only rules that produce *generic* dispatches live here. Shape choices
//! (tile size, register blocking, cooperative split, TG promotion, fusion)
//! are made in [`Phase::LateDispatch`] so the extractor's shape-aware cost
//! model can compare all variants side-by-side.
//!
//! - `recursive-to-dispatch`: analysis-driven whole-expression lowering for
//!   supported literal-shape tensor trees. Emits one composite Dispatch whose
//!   per-output body evaluates the full high-level expression tree
//!   recursively, introducing nested `Theta`s for internal reductions.
//! - `reduce-to-dispatch`: fallback Reduce → per-lane Dispatch for bodies
//!   that `recursive-to-dispatch` declines (e.g. non-literal shapes in a
//!   lowerable tree, or reductions whose input tree is too narrow for
//!   generic composite lowering).
//! - `ewise-fuse-inner`: fuses nested single-input Elementwise nodes so
//!   scalar algebra can see across the boundary.
//! - `exp-sub-split` / `factor-reduce-mul-bcast`: normalize softmax-style
//!   scalar math and factor axis-invariant broadcast terms out of reductions.
//! - `commutative-binop-swap`: exposes equivalent operand orders for
//!   commutative scalar ops so shape-specific operand order is not baked into
//!   later rewrites.
//! - `associative-binop-rotate-right`: exposes equivalent groupings for
//!   associative scalar ops.
//! - `arith-identity`: scalar identity folding (U32 only for now).

mod arithmetic_identity;
mod associativity;
mod commutativity;
mod ewise_fusion;
mod exp_algebra;
mod online_reduction;
mod recursive_dispatch_lowering;
mod reduce_lowering;

use egg::{EGraph, Id, Rewrite};

use crate::analysis::TensorAnalysis;
use crate::language::{DispatchNode, HighLevelNode, SimdNode, TensorIr, extract_list};
use crate::rules::RunnerConfig;
use crate::types::{BinaryOp, BufferRef, Dim, MemTier, ScalarValue, Shape};

/// Generate all Phase 1 rewrite rules.
///
/// `config` supplies the device profile and tile-config provider used by
/// fused-op lowering rules. Rules that don't need either are unaffected.
#[must_use]
pub fn rules(config: &RunnerConfig) -> Vec<Rewrite<TensorIr, TensorAnalysis>> {
    let mut rules = vec![
        reduce_lowering::build(config),
        recursive_dispatch_lowering::build(config),
        online_reduction::build(config),
        ewise_fusion::build(config),
        commutativity::build(config),
        associativity::build(config),
    ];
    rules.extend(exp_algebra::build_all(config));
    rules.extend(arithmetic_identity::build_all());
    rules
}

pub(super) fn decompose_flat_index(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    flat: Id,
    shape: &Shape,
) -> Vec<Id> {
    let rank = shape.rank();
    if rank == 0 {
        return vec![];
    }

    let mut indices = vec![Id::default(); rank];
    let mut remaining = flat;

    for i in (0..rank).rev() {
        let dim_value = dim_to_ir(egraph, &shape.0[i]);

        if i == 0 {
            indices[i] = remaining;
        } else {
            indices[i] = egraph.add(TensorIr::BinOp(BinaryOp::Mod, [remaining, dim_value]));
            remaining = egraph.add(TensorIr::BinOp(BinaryOp::Div, [remaining, dim_value]));
        }
    }

    indices
}

pub(super) fn dim_to_ir(egraph: &mut EGraph<TensorIr, TensorAnalysis>, dim: &Dim) -> Id {
    match dim {
        Dim::Const(value) => egraph.add(TensorIr::Const(ScalarValue::U32(*value))),
        Dim::Symbol(index) => egraph.add(TensorIr::ShapeParam(*index)),
        Dim::Add(lhs, rhs) => {
            let lhs = dim_to_ir(egraph, lhs);
            let rhs = dim_to_ir(egraph, rhs);
            egraph.add(TensorIr::BinOp(BinaryOp::Add, [lhs, rhs]))
        }
        Dim::Sub(lhs, rhs) => {
            let lhs = dim_to_ir(egraph, lhs);
            let rhs = dim_to_ir(egraph, rhs);
            egraph.add(TensorIr::BinOp(BinaryOp::Sub, [lhs, rhs]))
        }
        Dim::Mul(lhs, rhs) => {
            let lhs = dim_to_ir(egraph, lhs);
            let rhs = dim_to_ir(egraph, rhs);
            egraph.add(TensorIr::BinOp(BinaryOp::Mul, [lhs, rhs]))
        }
        Dim::Div(lhs, rhs) => {
            let lhs = dim_to_ir(egraph, lhs);
            let rhs = dim_to_ir(egraph, rhs);
            egraph.add(TensorIr::BinOp(BinaryOp::Div, [lhs, rhs]))
        }
        Dim::Mod(lhs, rhs) => {
            let lhs = dim_to_ir(egraph, lhs);
            let rhs = dim_to_ir(egraph, rhs);
            egraph.add(TensorIr::BinOp(BinaryOp::Mod, [lhs, rhs]))
        }
        Dim::CeilDiv(lhs, rhs) => {
            let lhs = dim_to_ir(egraph, lhs);
            let rhs = dim_to_ir(egraph, rhs);
            let one = egraph.add(TensorIr::Const(ScalarValue::U32(1)));
            let numerator = egraph.add(TensorIr::BinOp(BinaryOp::Add, [lhs, rhs]));
            let numerator = egraph.add(TensorIr::BinOp(BinaryOp::Sub, [numerator, one]));
            let quotient = egraph.add(TensorIr::BinOp(BinaryOp::Div, [numerator, rhs]));
            quotient
        }
        Dim::Min(lhs, rhs) => {
            let lhs = dim_to_ir(egraph, lhs);
            let rhs = dim_to_ir(egraph, rhs);
            egraph.add(TensorIr::BinOp(BinaryOp::Min, [lhs, rhs]))
        }
        Dim::Max(lhs, rhs) => {
            let lhs = dim_to_ir(egraph, lhs);
            let rhs = dim_to_ir(egraph, rhs);
            egraph.add(TensorIr::BinOp(BinaryOp::Max, [lhs, rhs]))
        }
    }
}

pub(super) fn compute_flat_addr(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    indices: &[Id],
    shape: &Shape,
) -> Id {
    if indices.is_empty() {
        return egraph.add(TensorIr::Const(ScalarValue::U32(0)));
    }
    if indices.len() == 1 {
        return indices[0];
    }

    let mut terms = Vec::with_capacity(indices.len());
    for (i, idx) in indices.iter().enumerate() {
        let stride = shape.0[i + 1..]
            .iter()
            .cloned()
            .fold(Dim::Const(1), Dim::mul);
        if stride == Dim::Const(1) {
            terms.push(*idx);
        } else {
            let stride_lit = dim_to_ir(egraph, &stride);
            terms.push(egraph.add(TensorIr::BinOp(BinaryOp::Mul, [*idx, stride_lit])));
        }
    }

    let mut result = terms[0];
    for term in &terms[1..] {
        result = egraph.add(TensorIr::BinOp(BinaryOp::Add, [result, *term]));
    }
    result
}

pub(super) fn lower_scalar_body_strided(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    body_id: Id,
    ewise_inputs: &[Id],
    indices: &[Id],
) -> Id {
    let node = egraph[body_id].iter().next().cloned();
    let Some(node) = node else {
        return body_id;
    };

    match node {
        TensorIr::HighLevel(HighLevelNode::Param(i)) => {
            let input_id = ewise_inputs[i as usize];
            let (strides, offset) = get_restride_layout(egraph, input_id);
            let mut addr = compute_strided_addr(egraph, indices, &strides);
            if offset != 0 {
                let offset = egraph.add(TensorIr::Const(crate::types::ScalarValue::U32(
                    u32::try_from(offset).expect("offset fits in u32"),
                )));
                addr = egraph.add(TensorIr::BinOp(crate::types::BinaryOp::Add, [addr, offset]));
            }
            let buf_ref = BufferRef::Input(i);
            let state = egraph.add(TensorIr::Dispatch(DispatchNode::Token));
            egraph.add(TensorIr::Simd(SimdNode::Load {
                tier: MemTier::Device(buf_ref),
                children: [addr, state],
            }))
        }
        TensorIr::HighLevel(HighLevelNode::Index(dim)) => {
            indices.get(dim as usize).copied().unwrap_or(body_id)
        }
        TensorIr::HighLevel(HighLevelNode::IndexedParam {
            index,
            children_list,
        }) => {
            let input_id = ewise_inputs[index as usize];
            let (strides, offset) = get_restride_layout(egraph, input_id);
            let indexed_children = extract_list(egraph, children_list);
            let indexed_indices = indexed_children
                .into_iter()
                .map(|child| lower_scalar_body_strided(egraph, child, ewise_inputs, indices))
                .collect::<Vec<_>>();
            let mut addr = compute_strided_addr(egraph, &indexed_indices, &strides);
            if offset != 0 {
                let offset = egraph.add(TensorIr::Const(crate::types::ScalarValue::U32(
                    u32::try_from(offset).expect("offset fits in u32"),
                )));
                addr = egraph.add(TensorIr::BinOp(crate::types::BinaryOp::Add, [addr, offset]));
            }
            let buf_ref = BufferRef::Input(index);
            let state = egraph.add(TensorIr::Dispatch(DispatchNode::Token));
            egraph.add(TensorIr::Simd(SimdNode::Load {
                tier: MemTier::Device(buf_ref),
                children: [addr, state],
            }))
        }
        TensorIr::Const(v) => egraph.add(TensorIr::Const(v)),
        TensorIr::ShapeParam(index) => egraph.add(TensorIr::ShapeParam(index)),
        TensorIr::BinOp(name, args) => {
            let lhs = lower_scalar_body_strided(egraph, args[0], ewise_inputs, indices);
            let rhs = lower_scalar_body_strided(egraph, args[1], ewise_inputs, indices);
            egraph.add(TensorIr::BinOp(name, [lhs, rhs]))
        }
        TensorIr::UnOp(name, arg) => {
            let arg = lower_scalar_body_strided(egraph, arg, ewise_inputs, indices);
            egraph.add(TensorIr::UnOp(name, arg))
        }
        TensorIr::TernOp(name, args) => {
            let a = lower_scalar_body_strided(egraph, args[0], ewise_inputs, indices);
            let b = lower_scalar_body_strided(egraph, args[1], ewise_inputs, indices);
            let c = lower_scalar_body_strided(egraph, args[2], ewise_inputs, indices);
            egraph.add(TensorIr::TernOp(name, [a, b, c]))
        }
        _ => body_id,
    }
}

pub(super) fn get_restride_layout(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    id: Id,
) -> (Vec<Dim>, i64) {
    for node in egraph[id].iter() {
        if let TensorIr::HighLevel(HighLevelNode::Restride {
            strides, offset, ..
        }) = node
        {
            return (strides.0.clone(), *offset);
        }
    }
    if let Some(shape) = &egraph[id].data.shape {
        return (crate::types::Strides::row_major_for_shape(shape).0, 0);
    }
    (vec![], 0)
}

pub(super) fn compute_strided_addr(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    indices: &[Id],
    strides: &[Dim],
) -> Id {
    let mut terms: Vec<Id> = Vec::new();
    for (idx, stride) in indices.iter().zip(strides.iter()) {
        if stride.as_const() == Some(0) {
            continue;
        }
        if stride.as_const() == Some(1) {
            terms.push(*idx);
        } else {
            let s = dim_to_ir(egraph, stride);
            let scaled = egraph.add(TensorIr::BinOp(BinaryOp::Mul, [*idx, s]));
            terms.push(scaled);
        }
    }
    if terms.is_empty() {
        return egraph.add(TensorIr::Const(ScalarValue::U32(0)));
    }
    let mut result = terms[0];
    for t in &terms[1..] {
        result = egraph.add(TensorIr::BinOp(BinaryOp::Add, [result, *t]));
    }
    result
}

pub(super) fn find_underlying_input(egraph: &EGraph<TensorIr, TensorAnalysis>, id: Id) -> Id {
    for node in egraph[id].iter() {
        if let TensorIr::HighLevel(HighLevelNode::Restride { expr, .. }) = node {
            return find_underlying_input(egraph, *expr);
        }
    }
    id
}
