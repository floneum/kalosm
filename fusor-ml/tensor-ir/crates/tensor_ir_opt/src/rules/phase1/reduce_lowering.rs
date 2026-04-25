use egg::{EGraph, Id, Rewrite};

use crate::analysis::TensorAnalysis;
use crate::applier::SimpleEclassSearcher;
use crate::language::{DispatchNode, HighLevelNode, SimdNode, TensorIr, add_list, extract_list};
use crate::rules::RunnerConfig;
use crate::types::{
    BinaryOp, BufferRef, DType, Dim, IndexLevel, LoweringOptions, MemTier, ReduceOp, ScalarValue,
    Shape, TernaryOp, VarRef,
};
use crate::unroll::unroll_fold_direct;

use super::{
    compute_flat_addr, decompose_flat_index, find_underlying_input, lower_scalar_body_strided,
};

const COOPERATIVE_REDUCE_UNROLL: u32 = 8;

const fn cooperative_reduce_unroll(lowering: &LoweringOptions) -> u32 {
    if lowering.unroll {
        COOPERATIVE_REDUCE_UNROLL
    } else {
        1
    }
}

pub(super) fn build(config: &RunnerConfig) -> Rewrite<TensorIr, TensorAnalysis> {
    Rewrite::new(
        "reduce-to-dispatch",
        SimpleEclassSearcher::new(|egraph, eclass| {
            let eclass = &egraph[eclass];
            if eclass
                .iter()
                .any(|n| matches!(n, TensorIr::Dispatch(DispatchNode::Dispatch { .. })))
            {
                return false;
            }
            eclass
                .iter()
                .any(|n| matches!(n, TensorIr::HighLevel(HighLevelNode::Reduce { .. })))
        }),
        crate::applier::AdaptedApplier(ReduceApplier {
            simd_width: config.device.simd_width,
            lowering: config.lowering,
        }),
    )
    .unwrap()
}

struct ReduceApplier {
    simd_width: u32,
    lowering: LoweringOptions,
}

struct ElementwiseReduceDispatchSpec<'a> {
    expr: Id,
    op: ReduceOp,
    init: Id,
    count: Id,
    out_flat: Id,
    input_indices: &'a [Id],
    workgroups: u32,
}

struct SimpleReduceDispatchSpec<'a> {
    expr: Id,
    op: ReduceOp,
    init: Id,
    count: Id,
    input_shape: &'a Shape,
    input_indices: &'a [Id],
    out_flat: Id,
    workgroups: u32,
}

struct CooperativeReduceDispatchSpec<'a> {
    expr: Id,
    op: ReduceOp,
    init: Id,
    input_shape: &'a Shape,
    output_shape: &'a Shape,
    axis: u32,
    reduce_dim: u32,
    output_elements: u32,
    wg: Id,
    lane: Id,
    k_var: Id,
    lowering: &'a LoweringOptions,
}

fn build_elementwise_reduce_dispatch(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    spec: &ElementwiseReduceDispatchSpec<'_>,
) -> Option<Id> {
    let ewise_node = egraph[spec.expr]
        .iter()
        .find(|n| matches!(n, TensorIr::HighLevel(HighLevelNode::Elementwise { .. })))
        .cloned();

    let Some(TensorIr::HighLevel(HighLevelNode::Elementwise {
        num_inputs: ewise_num_inputs,
        children_list: ewise_children,
        ..
    })) = ewise_node
    else {
        return None;
    };

    let ewise_children = extract_list(egraph, ewise_children);
    let ewise_inputs = &ewise_children[..ewise_num_inputs as usize];
    let ewise_body = *ewise_children
        .last()
        .expect("elementwise reduction body must exist");
    let lowered_body =
        lower_scalar_body_strided(egraph, ewise_body, ewise_inputs, spec.input_indices);
    let acc_var = egraph.add(TensorIr::Simd(SimdNode::Var(VarRef::acc(0))));
    let dtype = egraph[lowered_body].data.dtype.unwrap_or(DType::F32);
    let Some(op_name) = spec.op.bin_op_for_dtype(dtype) else {
        return None;
    };
    let update = egraph.add(TensorIr::BinOp(op_name, [acc_var, lowered_body]));
    let theta = egraph.add(TensorIr::Simd(SimdNode::Theta {
        children: [spec.init, spec.count, update],
    }));

    let mut dispatch_children: Vec<Id> = ewise_inputs
        .iter()
        .map(|id| find_underlying_input(egraph, *id))
        .collect();
    dispatch_children.push(theta);
    dispatch_children.push(spec.out_flat);

    let dispatch_children = add_list(egraph, &dispatch_children);
    Some(egraph.add(TensorIr::Dispatch(DispatchNode::Dispatch {
        workgroups: spec.workgroups,
        num_inputs: ewise_num_inputs,
        children_list: dispatch_children,
    })))
}

fn build_cooperative_reduce_dispatch(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    spec: &CooperativeReduceDispatchSpec<'_>,
    simd_width: u32,
) -> Option<Id> {
    let coop_workgroups = spec.output_elements;
    let coop_out_indices = decompose_flat_index(egraph, spec.wg, spec.output_shape);
    let unroll_factor = cooperative_reduce_unroll(spec.lowering);
    let coop_chunk_width = simd_width * unroll_factor;
    let coop_stride = egraph.add(TensorIr::Const(ScalarValue::U32(coop_chunk_width)));
    let k_base = egraph.add(TensorIr::BinOp(BinaryOp::Mul, [spec.k_var, coop_stride]));
    let k_total = egraph.add(TensorIr::Const(ScalarValue::U32(spec.reduce_dim)));
    let needs_tail_guards = !spec.reduce_dim.is_multiple_of(coop_chunk_width);
    let coop_count = egraph.add(TensorIr::Const(ScalarValue::U32(
        spec.reduce_dim.div_ceil(coop_chunk_width),
    )));

    // If the reduce's input is an Elementwise, walk its body via
    // `lower_scalar_body_strided` at each unrolled K step so stride-0 restride
    // terms collapse correctly (matmul's A/B broadcasts live in those strides).
    // Otherwise fall back to a single flat Input(0) load — the behaviour used
    // for a reduce over a raw tensor input.
    let ewise = egraph[spec.expr]
        .iter()
        .find(|n| matches!(n, TensorIr::HighLevel(HighLevelNode::Elementwise { .. })))
        .cloned();
    let ewise_ctx = if let Some(TensorIr::HighLevel(HighLevelNode::Elementwise {
        num_inputs,
        children_list,
        ..
    })) = ewise
    {
        let children = extract_list(egraph, children_list);
        let inputs: Vec<Id> = children[..num_inputs as usize].to_vec();
        let body = *children.last().expect("elementwise body");
        Some((inputs, body, num_inputs))
    } else {
        None
    };
    if ewise_ctx.is_some() && needs_tail_guards {
        return None;
    }

    let theta = unroll_fold_direct(
        egraph,
        spec.init,
        coop_count,
        unroll_factor,
        |egraph, unroll, acc| {
            let chunk_offset = egraph.add(TensorIr::Const(ScalarValue::U32(unroll * simd_width)));
            let lane_offset = egraph.add(TensorIr::BinOp(BinaryOp::Add, [spec.lane, chunk_offset]));
            let k_remapped = egraph.add(TensorIr::BinOp(BinaryOp::Add, [k_base, lane_offset]));
            let in_bounds = egraph.add(TensorIr::BinOp(BinaryOp::Lt, [k_remapped, k_total]));
            let safe_k = if needs_tail_guards {
                let zero = egraph.add(TensorIr::Const(ScalarValue::U32(0)));
                egraph.add(TensorIr::TernOp(
                    TernaryOp::Select,
                    [in_bounds, k_remapped, zero],
                ))
            } else {
                k_remapped
            };

            let mut coop_input_indices = Vec::with_capacity(spec.input_shape.rank());
            let mut out_idx = 0;
            for dim_idx in 0..spec.input_shape.rank() {
                if dim_idx == spec.axis as usize {
                    coop_input_indices.push(safe_k);
                } else if out_idx < coop_out_indices.len() {
                    coop_input_indices.push(coop_out_indices[out_idx]);
                    out_idx += 1;
                }
            }

            let load_val = match &ewise_ctx {
                Some((inputs, body, _)) => {
                    lower_scalar_body_strided(egraph, *body, inputs, &coop_input_indices)
                }
                None => {
                    let in_addr = compute_flat_addr(egraph, &coop_input_indices, spec.input_shape);
                    let state = egraph.add(TensorIr::Dispatch(DispatchNode::Token));
                    egraph.add(TensorIr::Simd(SimdNode::Load {
                        tier: MemTier::Device(BufferRef::Input(0)),
                        children: [in_addr, state],
                    }))
                }
            };
            let guarded_val = if needs_tail_guards {
                egraph.add(TensorIr::TernOp(
                    TernaryOp::Select,
                    [in_bounds, load_val, spec.init],
                ))
            } else {
                load_val
            };
            let dtype = egraph[guarded_val].data.dtype.unwrap_or(DType::F32);
            let op_name = spec
                .op
                .bin_op_for_dtype(dtype)
                .expect("reduce identity should guarantee supported update dtype");
            egraph.add(TensorIr::BinOp(op_name, [acc, guarded_val]))
        },
    );
    let reduced = egraph.add(TensorIr::Simd(SimdNode::ReduceSimd {
        op: spec.op,
        src: theta,
    }));

    let (dispatch_children, num_inputs) = match &ewise_ctx {
        Some((inputs, _, n)) => {
            let mut children: Vec<Id> = inputs
                .iter()
                .map(|id| find_underlying_input(egraph, *id))
                .collect();
            children.push(reduced);
            children.push(spec.wg);
            (add_list(egraph, &children), *n)
        }
        None => (add_list(egraph, &[spec.expr, reduced, spec.wg]), 1),
    };
    Some(egraph.add(TensorIr::Dispatch(DispatchNode::Dispatch {
        workgroups: coop_workgroups,
        num_inputs,
        children_list: dispatch_children,
    })))
}

fn build_simple_reduce_dispatch(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    spec: &SimpleReduceDispatchSpec<'_>,
) -> Id {
    let in_addr = compute_flat_addr(egraph, spec.input_indices, spec.input_shape);
    let in_buf = BufferRef::Input(0);
    let state = egraph.add(TensorIr::Dispatch(DispatchNode::Token));
    let load_val = egraph.add(TensorIr::Simd(SimdNode::Load {
        tier: MemTier::Device(in_buf),
        children: [in_addr, state],
    }));
    let acc_var = egraph.add(TensorIr::Simd(SimdNode::Var(VarRef::acc(0))));
    let dtype = egraph[load_val].data.dtype.unwrap_or(DType::F32);
    let op_name = spec
        .op
        .bin_op_for_dtype(dtype)
        .expect("reduce identity should guarantee supported update dtype");
    let update = egraph.add(TensorIr::BinOp(op_name, [acc_var, load_val]));
    let theta = egraph.add(TensorIr::Simd(SimdNode::Theta {
        children: [spec.init, spec.count, update],
    }));
    let dispatch_children = add_list(egraph, &[spec.expr, theta, spec.out_flat]);
    egraph.add(TensorIr::Dispatch(DispatchNode::Dispatch {
        workgroups: spec.workgroups,
        num_inputs: 1,
        children_list: dispatch_children,
    }))
}

fn build_reduce_input_indices(
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
        } else if out_idx < out_indices.len() {
            input_indices.push(out_indices[out_idx]);
            out_idx += 1;
        }
    }
    input_indices
}

impl crate::applier::TypedApplier for ReduceApplier {
    fn apply(&self, egraph: &mut EGraph<TensorIr, TensorAnalysis>, eclass: Id) -> Vec<Id> {
        let node = egraph[eclass]
            .iter()
            .find(|n| matches!(n, TensorIr::HighLevel(HighLevelNode::Reduce { .. })))
            .cloned();

        let Some(TensorIr::HighLevel(HighLevelNode::Reduce { axis, op, expr })) = node else {
            return vec![];
        };

        if egraph[eclass]
            .iter()
            .any(|n| matches!(n, TensorIr::Dispatch(DispatchNode::Dispatch { .. })))
        {
            return vec![];
        }

        let input_shape = match &egraph[expr].data.shape {
            Some(s) => s.clone(),
            None => return vec![],
        };

        let reduce_dim = match &input_shape.0[axis as usize] {
            Dim::Lit(v) => *v,
            Dim::Sym(_) => return vec![],
        };

        let output_shape = input_shape.remove_axis(axis as usize);
        let output_elements = match output_shape.static_numel() {
            Some(n) if n > 0 => n,
            _ => return vec![],
        };

        let workgroups = output_elements / self.simd_width;
        if workgroups == 0 {
            return vec![];
        }

        let input_dtype = egraph[expr].data.dtype.unwrap_or(DType::F32);
        let Some(identity) = op.identity(input_dtype) else {
            return vec![];
        };
        let init = egraph.add(TensorIr::Const(identity));
        let mut results = vec![];
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
        let k_var = egraph.add(TensorIr::Simd(SimdNode::Var(VarRef::iter(0))));
        let count = egraph.add(TensorIr::Const(ScalarValue::U32(reduce_dim)));
        let input_indices = build_reduce_input_indices(&input_shape, axis, &out_indices, k_var);

        if let Some(dispatch) = build_elementwise_reduce_dispatch(
            egraph,
            &ElementwiseReduceDispatchSpec {
                expr,
                op,
                init,
                count,
                out_flat,
                input_indices: &input_indices,
                workgroups,
            },
        ) {
            egraph.union(eclass, dispatch);
            results.push(dispatch);
        }

        if reduce_dim > self.simd_width {
            let dispatch = build_cooperative_reduce_dispatch(
                egraph,
                &CooperativeReduceDispatchSpec {
                    expr,
                    op,
                    init,
                    input_shape: &input_shape,
                    output_shape: &output_shape,
                    axis,
                    reduce_dim,
                    output_elements,
                    wg,
                    lane,
                    k_var,
                    lowering: &self.lowering,
                },
                self.simd_width,
            );
            if let Some(dispatch) = dispatch {
                egraph.union(eclass, dispatch);
                results.push(dispatch);
            }
        } else {
            let dispatch = build_simple_reduce_dispatch(
                egraph,
                &SimpleReduceDispatchSpec {
                    expr,
                    op,
                    init,
                    count,
                    input_shape: &input_shape,
                    input_indices: &input_indices,
                    out_flat,
                    workgroups,
                },
            );
            egraph.union(eclass, dispatch);
            results.push(dispatch);
        }
        results
    }
}
