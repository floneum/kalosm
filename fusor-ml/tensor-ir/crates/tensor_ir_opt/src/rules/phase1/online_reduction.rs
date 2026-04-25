//! Online lowering for normalized weighted reductions.
//!
//! The rule consumes analysis facts rather than recognizing attention-shaped
//! IR.  If an input to an additive reduction is known to be normalized along
//! the reduction axis, and the reduced body multiplies that normalized weight
//! by another tensor, we can lower the whole reduction as a single online
//! accumulator.  Attention is just the case where the normalized weight is
//! `softmax(QK^T)` and the other tensor is `V`.

use std::collections::HashMap;

use egg::{EGraph, Id, Rewrite};
use ordered_float::OrderedFloat;

use crate::analysis::{NormalizedWeightInfo, TensorAnalysis};
use crate::applier::SimpleEclassSearcher;
use crate::language::{
    DispatchNode, HighLevelNode, SimdNode, TensorIr, add_list, try_add_value_addr_dispatch,
    try_extract_list,
};
use crate::rules::RunnerConfig;
use crate::types::{
    BinaryOp, Dim, IndexLevel, ReduceOp, ScalarValue, Shape, Strides, TernaryOp, UnaryOp, VarRef,
};

use super::recursive_dispatch_lowering::{
    EvalContext, build_reduce_input_indices, collect_underlying_inputs, lower_tensor_point,
    selected_high_level_node, shape_of,
};
use super::{compute_flat_addr, compute_strided_addr, decompose_flat_index};

pub(super) fn build(config: &RunnerConfig) -> Rewrite<TensorIr, TensorAnalysis> {
    let simd_width = config.device.simd_width;
    Rewrite::new(
        "online-normalized-weighted-reduce",
        SimpleEclassSearcher::new(move |egraph, eclass| {
            find_pattern(egraph, eclass, simd_width)
                .is_some_and(|pattern| !has_online_dispatch(egraph, eclass, &pattern))
        }),
        crate::applier::AdaptedApplier(OnlineReductionApplier { simd_width }),
    )
    .unwrap()
}

struct OnlineReductionApplier {
    simd_width: u32,
}

#[derive(Debug, Clone)]
struct Pattern {
    index_space: Shape,
    output_shape: Shape,
    reduce_axis: u32,
    simd_width: u32,
    workgroups: u32,
    normalized: NormalizedWeightInfo,
    weight_strides: Strides,
    value_source: Id,
    value_strides: Strides,
}

impl crate::applier::TypedApplier for OnlineReductionApplier {
    fn apply(&self, egraph: &mut EGraph<TensorIr, TensorAnalysis>, eclass: Id) -> Vec<Id> {
        let Some(pattern) = find_pattern(egraph, eclass, self.simd_width) else {
            return vec![];
        };
        let Some(dispatch) = build_online_dispatch(egraph, &pattern) else {
            return vec![];
        };
        egraph.union(eclass, dispatch);
        vec![dispatch]
    }
}

fn find_pattern(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    eclass: Id,
    simd_width: u32,
) -> Option<Pattern> {
    let canonical = egraph.find(eclass);
    for node in egraph[canonical].iter() {
        let TensorIr::HighLevel(HighLevelNode::Reduce {
            axis,
            op: ReduceOp::Add,
            expr,
        }) = node
        else {
            continue;
        };
        let output_shape = egraph[egraph.find(eclass)].data.shape.clone()?;
        let output_elements = output_shape.static_numel()?;
        if output_elements < simd_width || !output_elements.is_multiple_of(simd_width) {
            continue;
        }
        let Some(pattern) =
            match_weighted_elementwise(egraph, *expr, *axis, output_shape, simd_width)
        else {
            continue;
        };
        return Some(pattern);
    }
    None
}

fn match_weighted_elementwise(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    expr: Id,
    reduce_axis: u32,
    output_shape: Shape,
    simd_width: u32,
) -> Option<Pattern> {
    let canonical = egraph.find(expr);
    for node in egraph[canonical].iter() {
        let TensorIr::HighLevel(HighLevelNode::Elementwise {
            index_space,
            num_inputs: 2,
            children_list,
        }) = node
        else {
            continue;
        };
        let children = try_extract_list(egraph, *children_list)?;
        let [lhs, rhs, body] = children.as_slice() else {
            continue;
        };
        if !is_param_mul(egraph, *body) {
            continue;
        }
        for (weight, value) in [(*lhs, *rhs), (*rhs, *lhs)] {
            let Some((weight_source, weight_strides)) = source_and_strides(egraph, weight) else {
                continue;
            };
            let Some(normalized) = egraph[egraph.find(weight_source)].data.normalized_weight else {
                continue;
            };
            if normalized_exp_max(egraph, normalized).is_none() {
                continue;
            }
            if !reduce_axis_maps_to_normalized_axis(
                egraph,
                &weight_strides,
                weight_source,
                reduce_axis,
                normalized.axis,
            ) {
                continue;
            }
            let Some((value_source, value_strides)) = source_and_strides(egraph, value) else {
                continue;
            };
            let workgroups = output_shape.static_numel()? / simd_width;
            return Some(Pattern {
                index_space: index_space.clone(),
                output_shape: output_shape.clone(),
                reduce_axis,
                simd_width,
                workgroups,
                normalized,
                weight_strides,
                value_source: egraph.find(value_source),
                value_strides,
            });
        }
    }
    None
}

fn reduce_axis_maps_to_normalized_axis(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    weight_strides: &Strides,
    weight_source: Id,
    reduce_axis: u32,
    normalized_axis: u32,
) -> bool {
    let Some(weight_shape) = shape_of(egraph, weight_source) else {
        return false;
    };
    let Some(row_major) = Strides::row_major_for_shape(&weight_shape) else {
        return false;
    };
    let reduce_axis = reduce_axis as usize;
    let normalized_axis = normalized_axis as usize;
    reduce_axis < weight_strides.0.len()
        && normalized_axis < row_major.0.len()
        && weight_strides.0[reduce_axis] == row_major.0[normalized_axis]
}

fn source_and_strides(egraph: &EGraph<TensorIr, TensorAnalysis>, id: Id) -> Option<(Id, Strides)> {
    let canonical = egraph.find(id);
    for node in egraph[canonical].iter() {
        if let TensorIr::HighLevel(HighLevelNode::Restride { strides, expr, .. }) = node {
            return Some((egraph.find(*expr), strides.clone()));
        }
    }
    let shape = shape_of(egraph, canonical)?;
    Some((canonical, Strides::row_major_for_shape(&shape)?))
}

fn build_online_dispatch(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    pattern: &Pattern,
) -> Option<Id> {
    let exp_max = normalized_exp_max(egraph, pattern.normalized)?;
    let inputs = collect_inputs(egraph, pattern);
    let num_inputs = u32::try_from(inputs.len()).ok()?;
    let input_slots = inputs
        .iter()
        .enumerate()
        .map(|(slot, input)| (*input, slot as u32))
        .collect::<HashMap<_, _>>();
    let ctx = EvalContext { input_slots };

    let wg = var_thread(egraph, IndexLevel::Workgroup);
    let lane = var_thread(egraph, IndexLevel::Lane);
    let simd_width = u32_lit(egraph, pattern.simd_width);
    let wg_offset = bin(egraph, BinaryOp::Mul, wg, simd_width);
    let out_flat = bin(egraph, BinaryOp::Add, wg_offset, lane);
    let output_indices = decompose_flat_index(egraph, out_flat, &pattern.output_shape);

    let key = var_iter(egraph, 0);
    let outer_indices = insert_axis(&output_indices, pattern.reduce_axis, key);
    let score_shape = shape_of(egraph, exp_max.score)?;
    let score_indices = source_indices_from_strides(
        egraph,
        &outer_indices,
        &pattern.weight_strides,
        &score_shape,
    );
    let value_shape = shape_of(egraph, pattern.value_source)?;
    let value_indices =
        source_indices_from_strides(egraph, &outer_indices, &pattern.value_strides, &value_shape);

    let acc = var_acc(egraph, 0);
    let old_m = extract(egraph, acc, 0);
    let old_l = extract(egraph, acc, 1);
    let old_acc = extract(egraph, acc, 2);
    let score = lower_tensor_point_prefer_lane_reduce(
        egraph,
        exp_max.score,
        &score_indices,
        &ctx,
        pattern.simd_width,
    )?;
    let new_m = bin(egraph, BinaryOp::Max, old_m, score);
    let alpha_arg = bin(egraph, BinaryOp::Sub, old_m, new_m);
    let alpha = un(egraph, UnaryOp::Exp, alpha_arg);
    let beta_arg = bin(egraph, BinaryOp::Sub, score, new_m);
    let beta = un(egraph, UnaryOp::Exp, beta_arg);
    let new_l = fma(egraph, old_l, alpha, beta);

    let value = lower_tensor_point(egraph, pattern.value_source, &value_indices, &ctx)?;
    let weighted_value = bin(egraph, BinaryOp::Mul, beta, value);
    let new_acc = fma(egraph, old_acc, alpha, weighted_value);

    let init_m = f32_lit(egraph, -f32::MAX);
    let init_l = f32_lit(egraph, 0.0);
    let init_acc = f32_lit(egraph, 0.0);
    let init = pack(egraph, &[init_m, init_l, init_acc]);
    let update = pack(egraph, &[new_m, new_l, new_acc]);
    let count = index_dim(&pattern.index_space, pattern.reduce_axis)?;
    let count = u32_lit(egraph, count);
    let theta = egraph.add(TensorIr::Simd(SimdNode::Theta {
        children: [init, count, update],
    }));

    let numerator = extract(egraph, theta, 2);
    let denom = extract(egraph, theta, 1);
    let output_value = bin(egraph, BinaryOp::Div, numerator, denom);
    let output_addr = compute_flat_addr(egraph, &output_indices, &pattern.output_shape);

    let mut children = inputs;
    children.push(output_value);
    children.push(output_addr);
    try_add_value_addr_dispatch(egraph, pattern.workgroups, num_inputs, &children)
}

fn collect_inputs(egraph: &EGraph<TensorIr, TensorAnalysis>, pattern: &Pattern) -> Vec<Id> {
    let exp_max = normalized_exp_max(egraph, pattern.normalized)
        .expect("online lowering requires an exp-max normalized weight");
    let mut inputs = Vec::new();
    for root in [exp_max.score, pattern.value_source] {
        for input in collect_underlying_inputs(egraph, root) {
            let input = egraph.find(input);
            if !inputs
                .iter()
                .any(|existing| egraph.find(*existing) == input)
            {
                inputs.push(input);
            }
        }
    }
    inputs
}

fn source_indices_from_strides(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    outer_indices: &[Id],
    strides: &Strides,
    source_shape: &Shape,
) -> Vec<Id> {
    let flat = compute_strided_addr(egraph, outer_indices, &strides.0);
    decompose_flat_index(egraph, flat, source_shape)
}

fn normalized_exp_max(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    normalized: NormalizedWeightInfo,
) -> Option<crate::analysis::ExpMaxInfo> {
    normalized
        .exp_max
        .or_else(|| egraph[egraph.find(normalized.numerator)].data.exp_max)
}

fn has_online_dispatch(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    eclass: Id,
    pattern: &Pattern,
) -> bool {
    let num_inputs = collect_inputs(egraph, pattern).len() as u32;
    let canonical = egraph.find(eclass);
    egraph[canonical].iter().any(|node| {
        let TensorIr::Dispatch(DispatchNode::Dispatch {
            workgroups,
            num_inputs: node_inputs,
            children_list,
        }) = node
        else {
            return false;
        };
        if *workgroups != pattern.workgroups || *node_inputs != num_inputs {
            return false;
        }
        let Some(children) = try_extract_list(egraph, *children_list) else {
            return false;
        };
        let body_idx = *node_inputs as usize;
        if children.len() != body_idx + 2 {
            return false;
        }
        let value_data = &egraph[children[body_idx]].data;
        value_data.contains_pack && value_data.contains_reduce_simd
    })
}

fn lower_tensor_point_prefer_lane_reduce(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    expr: Id,
    indices: &[Id],
    ctx: &EvalContext,
    simd_width: u32,
) -> Option<Id> {
    if let Some(HighLevelNode::Reduce {
        axis,
        op,
        expr: source,
    }) = selected_high_level_node(egraph, expr)
    {
        let source_shape = shape_of(egraph, source)?;
        let Dim::Lit(reduce_dim) = source_shape.0.get(axis as usize)? else {
            return None;
        };
        if *reduce_dim == simd_width {
            let lane = var_thread(egraph, IndexLevel::Lane);
            let source_indices = build_reduce_input_indices(&source_shape, axis, indices, lane);
            let src = lower_tensor_point(egraph, source, &source_indices, ctx)?;
            return Some(egraph.add(TensorIr::Simd(SimdNode::ReduceSimd { op, src })));
        }
    }
    lower_tensor_point(egraph, expr, indices, ctx)
}

fn insert_axis(indices: &[Id], axis: u32, value: Id) -> Vec<Id> {
    let axis = axis as usize;
    let mut out = Vec::with_capacity(indices.len() + 1);
    for i in 0..=indices.len() {
        if i == axis {
            out.push(value);
        }
        if i < indices.len() {
            out.push(indices[i]);
        }
    }
    out
}

fn index_dim(shape: &Shape, axis: u32) -> Option<u32> {
    let Dim::Lit(value) = shape.0.get(axis as usize)? else {
        return None;
    };
    Some(*value)
}

fn is_param_mul(egraph: &EGraph<TensorIr, TensorAnalysis>, id: Id) -> bool {
    is_param_binop(egraph, id, BinaryOp::Mul, 0, 1)
        || is_param_binop(egraph, id, BinaryOp::Mul, 1, 0)
}

fn is_param_binop(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    id: Id,
    op: BinaryOp,
    lhs_param: u32,
    rhs_param: u32,
) -> bool {
    let canonical = egraph.find(id);
    egraph[canonical].iter().any(|node| {
        matches!(
            node,
            TensorIr::BinOp(node_op, [lhs, rhs])
                if *node_op == op
                    && is_param(egraph, *lhs, lhs_param)
                    && is_param(egraph, *rhs, rhs_param)
        )
    })
}

fn is_param(egraph: &EGraph<TensorIr, TensorAnalysis>, id: Id, param: u32) -> bool {
    let canonical = egraph.find(id);
    egraph[canonical]
        .iter()
        .any(|node| matches!(node, TensorIr::HighLevel(HighLevelNode::Param(p)) if *p == param))
}

fn f32_lit(egraph: &mut EGraph<TensorIr, TensorAnalysis>, value: f32) -> Id {
    egraph.add(TensorIr::Const(ScalarValue::F32(OrderedFloat(value))))
}

fn u32_lit(egraph: &mut EGraph<TensorIr, TensorAnalysis>, value: u32) -> Id {
    egraph.add(TensorIr::Const(ScalarValue::U32(value)))
}

fn var_thread(egraph: &mut EGraph<TensorIr, TensorAnalysis>, level: IndexLevel) -> Id {
    egraph.add(TensorIr::Simd(SimdNode::Var(VarRef::thread(level))))
}

fn var_iter(egraph: &mut EGraph<TensorIr, TensorAnalysis>, depth: u32) -> Id {
    egraph.add(TensorIr::Simd(SimdNode::Var(VarRef::iter(depth))))
}

fn var_acc(egraph: &mut EGraph<TensorIr, TensorAnalysis>, depth: u32) -> Id {
    egraph.add(TensorIr::Simd(SimdNode::Var(VarRef::acc(depth))))
}

fn bin(egraph: &mut EGraph<TensorIr, TensorAnalysis>, op: BinaryOp, lhs: Id, rhs: Id) -> Id {
    egraph.add(TensorIr::BinOp(op, [lhs, rhs]))
}

fn un(egraph: &mut EGraph<TensorIr, TensorAnalysis>, op: UnaryOp, arg: Id) -> Id {
    egraph.add(TensorIr::UnOp(op, arg))
}

fn fma(egraph: &mut EGraph<TensorIr, TensorAnalysis>, a: Id, b: Id, c: Id) -> Id {
    egraph.add(TensorIr::TernOp(TernaryOp::Fma, [a, b, c]))
}

fn pack(egraph: &mut EGraph<TensorIr, TensorAnalysis>, children: &[Id]) -> Id {
    let children_list = add_list(egraph, children);
    egraph.add(TensorIr::Dispatch(DispatchNode::Pack { children_list }))
}

fn extract(egraph: &mut EGraph<TensorIr, TensorAnalysis>, tuple: Id, index: u32) -> Id {
    egraph.add(TensorIr::Dispatch(DispatchNode::Extract { index, tuple }))
}
