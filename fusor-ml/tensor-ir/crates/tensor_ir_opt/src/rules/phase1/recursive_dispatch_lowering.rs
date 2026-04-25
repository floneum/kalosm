//! Generic whole-expression lowering to one composite Dispatch.
//!
//! This rule is intentionally broad: if analysis says a high-level tensor
//! e-class is a supported literal-shape expression tree, we emit a `Dispatch`
//! whose per-output body is the recursive point-evaluation of that tree.
//!
//! Supported tensor structure:
//! - `Input`
//! - `Restride`
//! - `Elementwise`
//! - `Reduce`
//!
//! Nested reductions become nested `Theta` loops, so a single Dispatch can
//! represent composite expressions like contractions feeding reductions or
//! deeper DAG-shaped compositions without introducing workload-specific
//! rewrite logic.

use std::collections::{HashMap, HashSet};

use egg::{EGraph, Id, Rewrite};
use tensor_ir_egraph::binding;

use crate::analysis::TensorAnalysis;
use crate::applier::SimpleEclassSearcher;
use crate::language::{DispatchNode, HighLevelNode, SimdNode, TensorIr, add_list, extract_list};
use crate::rules::RunnerConfig;
use crate::types::{
    BinaryOp, BinderKind, BufferRef, DType, Dim, IndexLevel, MemTier, ScalarValue, Shape, VarRef,
};

use super::{compute_flat_addr, compute_strided_addr, decompose_flat_index};

pub(super) fn build(config: &RunnerConfig) -> Rewrite<TensorIr, TensorAnalysis> {
    let simd_width = config.device.simd_width;
    Rewrite::new(
        "recursive-to-dispatch",
        SimpleEclassSearcher::new(move |egraph, eclass| {
            let data = &egraph[egraph.find(eclass)].data;
            data.composite_dispatch.lowerable
                && output_elements(egraph, eclass)
                    .is_some_and(|n| n >= simd_width && n % simd_width == 0)
        }),
        crate::applier::AdaptedApplier(RecursiveDispatchApplier { simd_width }),
    )
    .unwrap()
}

struct RecursiveDispatchApplier {
    simd_width: u32,
}

#[derive(Clone)]
pub(super) struct EvalContext {
    pub(super) input_slots: HashMap<Id, u32>,
}

pub(super) fn selected_high_level_node(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    id: Id,
) -> Option<HighLevelNode> {
    let canonical = egraph.find(id);
    egraph[canonical].iter().find_map(|node| {
        let TensorIr::HighLevel(high) = node else {
            return None;
        };
        match high {
            HighLevelNode::Input { .. } => Some(high.clone()),
            HighLevelNode::Restride { expr, .. } | HighLevelNode::Reduce { expr, .. } => egraph
                [egraph.find(*expr)]
            .data
            .composite_dispatch
            .lowerable
            .then_some(high.clone()),
            HighLevelNode::Elementwise { children_list, .. } => {
                let children = extract_list(egraph, *children_list);
                children[..children.len().saturating_sub(1)]
                    .iter()
                    .all(|child| {
                        egraph[egraph.find(*child)]
                            .data
                            .composite_dispatch
                            .lowerable
                    })
                    .then_some(high.clone())
            }
            HighLevelNode::Param(_)
            | HighLevelNode::Index(_)
            | HighLevelNode::IndexedParam { .. } => None,
        }
    })
}

fn find_lowerable_high_level(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    eclass: Id,
) -> Option<HighLevelNode> {
    selected_high_level_node(egraph, eclass).and_then(|node| match node {
        HighLevelNode::Restride { .. }
        | HighLevelNode::Elementwise { .. }
        | HighLevelNode::Reduce { .. } => Some(node),
        HighLevelNode::Input { .. }
        | HighLevelNode::Param(_)
        | HighLevelNode::Index(_)
        | HighLevelNode::IndexedParam { .. } => None,
    })
}

pub(super) fn shape_of(egraph: &EGraph<TensorIr, TensorAnalysis>, id: Id) -> Option<Shape> {
    egraph[egraph.find(id)].data.shape.clone()
}

fn output_elements(egraph: &EGraph<TensorIr, TensorAnalysis>, id: Id) -> Option<u32> {
    shape_of(egraph, id)?.static_numel().filter(|n| *n > 0)
}

pub(super) fn collect_underlying_inputs(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    expr: Id,
) -> Vec<Id> {
    let mut inputs = Vec::new();
    let mut seen_exprs = HashSet::new();
    let mut seen_inputs = HashSet::new();
    collect_inputs_rec(egraph, expr, &mut seen_exprs, &mut seen_inputs, &mut inputs);
    inputs
}

fn collect_inputs_rec(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    id: Id,
    seen_exprs: &mut HashSet<Id>,
    seen_inputs: &mut HashSet<Id>,
    inputs: &mut Vec<Id>,
) {
    let canonical = egraph.find(id);
    if !seen_exprs.insert(canonical) {
        return;
    }

    let Some(node) = selected_high_level_node(egraph, canonical) else {
        return;
    };

    match node {
        HighLevelNode::Input { .. } => {
            if seen_inputs.insert(canonical) {
                inputs.push(canonical);
            }
        }
        HighLevelNode::Restride { expr, .. } | HighLevelNode::Reduce { expr, .. } => {
            collect_inputs_rec(egraph, expr, seen_exprs, seen_inputs, inputs);
        }
        HighLevelNode::Elementwise { children_list, .. } => {
            let children = extract_list(egraph, children_list);
            for child in &children[..children.len().saturating_sub(1)] {
                collect_inputs_rec(egraph, *child, seen_exprs, seen_inputs, inputs);
            }
        }
        HighLevelNode::Param(_) | HighLevelNode::Index(_) | HighLevelNode::IndexedParam { .. } => {}
    }
}

/// Idempotency guard: has `recursive-to-dispatch` already produced a
/// matching flat Dispatch for this e-class? Checks the shape fields
/// (`num_inputs`, `workgroups`) plus that children are raw Inputs and
/// that the body has a single (value, addr) output pair (i.e. a scalar
/// dispatch, not a register-blocked variant).
fn has_flat_external_input_dispatch(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    eclass: Id,
    num_inputs: u32,
    workgroups: u32,
) -> bool {
    let canonical = egraph.find(eclass);
    egraph[canonical].iter().any(|node| {
        let TensorIr::Dispatch(DispatchNode::Dispatch {
            num_inputs: existing_inputs,
            workgroups: existing_workgroups,
            children_list,
        }) = node
        else {
            return false;
        };
        if *existing_inputs != num_inputs || *existing_workgroups != workgroups {
            return false;
        }

        let children = extract_list(egraph, *children_list);
        let body_len = children.len().saturating_sub(*existing_inputs as usize);
        if body_len != 2 {
            // Register-blocked or merged dispatch — not the single-output
            // flat form this rule would emit.
            return false;
        }
        children[..*existing_inputs as usize].iter().all(|input| {
            egraph[egraph.find(*input)]
                .iter()
                .any(|node| matches!(node, TensorIr::HighLevel(HighLevelNode::Input { .. })))
        })
    })
}

pub(super) fn build_reduce_input_indices(
    input_shape: &Shape,
    axis: u32,
    out_indices: &[Id],
    k_var: Id,
) -> Vec<Id> {
    let mut input_indices = Vec::with_capacity(input_shape.rank());
    let mut out_idx = 0;
    for dim_idx in 0..input_shape.rank() {
        if dim_idx == axis as usize {
            input_indices.push(k_var);
        } else {
            input_indices.push(out_indices[out_idx]);
            out_idx += 1;
        }
    }
    input_indices
}

pub(super) fn lower_scalar_with_values(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    body: Id,
    ewise_inputs: &[Id],
    indices: &[Id],
    ctx: &EvalContext,
) -> Option<Id> {
    let canonical = egraph.find(body);
    let node = egraph[canonical]
        .iter()
        .find(|node| {
            matches!(
                node,
                TensorIr::HighLevel(
                    HighLevelNode::Param(_)
                        | HighLevelNode::Index(_)
                        | HighLevelNode::IndexedParam { .. }
                ) | TensorIr::Const(_)
                    | TensorIr::BinOp(_, _)
                    | TensorIr::UnOp(_, _)
                    | TensorIr::TernOp(_, _)
            )
        })
        .cloned()?;

    match node {
        TensorIr::HighLevel(HighLevelNode::Param(i)) => {
            let source = *ewise_inputs.get(i as usize)?;
            lower_tensor_point(egraph, source, indices, ctx)
        }
        TensorIr::HighLevel(HighLevelNode::Index(dim)) => indices.get(dim as usize).copied(),
        TensorIr::HighLevel(HighLevelNode::IndexedParam {
            index,
            children_list,
        }) => {
            let source = *ewise_inputs.get(index as usize)?;
            let indexed_children = extract_list(egraph, children_list);
            let mut indexed_indices = Vec::with_capacity(indexed_children.len());
            for child in indexed_children {
                indexed_indices.push(lower_scalar_with_values(
                    egraph,
                    child,
                    ewise_inputs,
                    indices,
                    ctx,
                )?);
            }
            lower_tensor_point(egraph, source, &indexed_indices, ctx)
        }
        TensorIr::Const(v) => Some(egraph.add(TensorIr::Const(v))),
        TensorIr::BinOp(op, [lhs, rhs]) => {
            let lhs = lower_scalar_with_values(egraph, lhs, ewise_inputs, indices, ctx)?;
            let rhs = lower_scalar_with_values(egraph, rhs, ewise_inputs, indices, ctx)?;
            Some(egraph.add(TensorIr::BinOp(op, [lhs, rhs])))
        }
        TensorIr::UnOp(op, arg) => {
            let arg = lower_scalar_with_values(egraph, arg, ewise_inputs, indices, ctx)?;
            Some(egraph.add(TensorIr::UnOp(op, arg)))
        }
        TensorIr::TernOp(op, [a, b, c]) => {
            let a = lower_scalar_with_values(egraph, a, ewise_inputs, indices, ctx)?;
            let b = lower_scalar_with_values(egraph, b, ewise_inputs, indices, ctx)?;
            let c = lower_scalar_with_values(egraph, c, ewise_inputs, indices, ctx)?;
            Some(egraph.add(TensorIr::TernOp(op, [a, b, c])))
        }
        _ => None,
    }
}

pub(super) fn lower_tensor_point(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    expr: Id,
    indices: &[Id],
    ctx: &EvalContext,
) -> Option<Id> {
    let canonical = egraph.find(expr);
    let node = selected_high_level_node(egraph, canonical)?;

    match node {
        HighLevelNode::Input { .. } => {
            let slot = *ctx.input_slots.get(&canonical)?;
            let shape = shape_of(egraph, expr)?;
            let addr = compute_flat_addr(egraph, indices, &shape);
            let state = egraph.add(TensorIr::Dispatch(DispatchNode::Token));
            Some(egraph.add(TensorIr::Simd(SimdNode::Load {
                tier: MemTier::Device(BufferRef::Input(slot)),
                children: [addr, state],
            })))
        }
        HighLevelNode::Restride {
            strides,
            offset,
            expr: source,
            ..
        } => {
            let source_shape = shape_of(egraph, source)?;
            let mut flat = compute_strided_addr(egraph, indices, &strides.0);
            if offset != 0 {
                let offset = egraph.add(TensorIr::Const(ScalarValue::U32(
                    u32::try_from(offset).expect("offset fits in u32"),
                )));
                flat = egraph.add(TensorIr::BinOp(BinaryOp::Add, [flat, offset]));
            }
            let source_indices = decompose_flat_index(egraph, flat, &source_shape);
            lower_tensor_point(egraph, source, &source_indices, ctx)
        }
        HighLevelNode::Elementwise {
            num_inputs,
            children_list,
            ..
        } => {
            let children = extract_list(egraph, children_list);
            let inputs = &children[..num_inputs as usize];
            let body = *children.last()?;
            lower_scalar_with_values(egraph, body, inputs, indices, ctx)
        }
        HighLevelNode::Reduce {
            axis,
            op,
            expr: source,
        } => {
            let source_shape = shape_of(egraph, source)?;
            let reduce_dim = match &source_shape.0[axis as usize] {
                Dim::Lit(v) => *v,
                Dim::Sym(_) => return None,
            };

            let mut shifted_indices = Vec::with_capacity(indices.len());
            for id in indices {
                shifted_indices.push(binding::shift(egraph, *id, BinderKind::Theta, 0, 1));
            }
            let k_var = egraph.add(TensorIr::Simd(SimdNode::Var(VarRef::iter(0))));
            let source_indices =
                build_reduce_input_indices(&source_shape, axis, &shifted_indices, k_var);
            let inner = lower_tensor_point(egraph, source, &source_indices, ctx)?;

            let source_dtype = egraph[source].data.dtype.unwrap_or(DType::F32);
            let init = egraph.add(TensorIr::Const(op.identity(source_dtype)?));
            let count = egraph.add(TensorIr::Const(ScalarValue::U32(reduce_dim)));
            let acc = egraph.add(TensorIr::Simd(SimdNode::Var(VarRef::acc(0))));
            let update = egraph.add(TensorIr::BinOp(
                op.bin_op_for_dtype(source_dtype)?,
                [acc, inner],
            ));
            // The Theta has no role tag. Whether phase-2/4 rules can tile or
            // collapse this reduction is determined structurally by their own
            // soundness checks (scalar-init, associative BinOp body, no outer
            // iter binder reference).
            Some(egraph.add(TensorIr::Simd(SimdNode::Theta {
                children: [init, count, update],
            })))
        }
        HighLevelNode::Param(_) | HighLevelNode::Index(_) | HighLevelNode::IndexedParam { .. } => {
            None
        }
    }
}

impl crate::applier::TypedApplier for RecursiveDispatchApplier {
    fn apply(&self, egraph: &mut EGraph<TensorIr, TensorAnalysis>, eclass: Id) -> Vec<Id> {
        let data = &egraph[egraph.find(eclass)].data;
        if !data.composite_dispatch.lowerable || find_lowerable_high_level(egraph, eclass).is_none()
        {
            return vec![];
        }

        let output_shape = match shape_of(egraph, eclass) {
            Some(shape) => shape,
            None => return vec![],
        };
        let output_elements = match output_shape.static_numel() {
            Some(n) if n >= self.simd_width && n % self.simd_width == 0 => n,
            _ => return vec![],
        };

        let input_ids = collect_underlying_inputs(egraph, eclass);
        if input_ids.is_empty() {
            return vec![];
        }

        let workgroups = output_elements / self.simd_width;
        let num_inputs = u32::try_from(input_ids.len()).expect("input count fits in u32");
        if has_flat_external_input_dispatch(egraph, eclass, num_inputs, workgroups) {
            return vec![];
        }

        let input_slots: HashMap<Id, u32> = input_ids
            .iter()
            .enumerate()
            .map(|(slot, id)| {
                (
                    egraph.find(*id),
                    u32::try_from(slot).expect("input slot index fits in u32"),
                )
            })
            .collect();
        let ctx = EvalContext { input_slots };

        let wg = egraph.add(TensorIr::Simd(SimdNode::Var(VarRef::thread(
            IndexLevel::Workgroup,
        ))));
        let lane = egraph.add(TensorIr::Simd(SimdNode::Var(VarRef::thread(
            IndexLevel::Lane,
        ))));
        let sw = egraph.add(TensorIr::Const(ScalarValue::U32(self.simd_width)));
        let wg_offset = egraph.add(TensorIr::BinOp(BinaryOp::Mul, [wg, sw]));
        let out_flat = egraph.add(TensorIr::BinOp(BinaryOp::Add, [wg_offset, lane]));
        let out_indices = decompose_flat_index(egraph, out_flat, &output_shape);
        let value = match lower_tensor_point(egraph, eclass, &out_indices, &ctx) {
            Some(value) => value,
            None => return vec![],
        };

        let mut children = input_ids;
        children.push(value);
        children.push(out_flat);
        let children_list = add_list(egraph, &children);
        let dispatch = egraph.add(TensorIr::Dispatch(DispatchNode::Dispatch {
            workgroups,
            num_inputs,
            children_list,
        }));
        egraph.union(eclass, dispatch);
        vec![dispatch]
    }
}
