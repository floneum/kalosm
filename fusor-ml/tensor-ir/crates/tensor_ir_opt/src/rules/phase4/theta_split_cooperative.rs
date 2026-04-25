use std::collections::{HashMap, HashSet};

use egg::{EGraph, Id, Language, Rewrite};
use tensor_ir_egraph::add_and_choose;

use crate::analysis::TensorAnalysis;
use crate::applier::SimpleEclassSearcher;
use crate::binding;
use crate::language::{
    DispatchNode, SimdNode, TensorIr, extract_list, try_add_value_addr_dispatch,
};
use crate::rules::RunnerConfig;
use crate::types::{
    BinaryOp, BinderKind, BufferRef, IndexLevel, MemTier, ReduceOp, ScalarValue, TernaryOp, VarRef,
    slots,
};

pub(super) fn build(config: &RunnerConfig) -> Rewrite<TensorIr, TensorAnalysis> {
    let simd_width = config.device.simd_width;
    Rewrite::new(
        "theta-split-cooperative",
        SimpleEclassSearcher::new(move |egraph, eclass| {
            let eclass_data = &egraph[eclass];
            eclass_data.iter().any(|node| {
                let TensorIr::Dispatch(DispatchNode::Dispatch {
                    workgroups,
                    num_inputs,
                    children_list: children,
                    ..
                }) = node
                else {
                    return false;
                };
                if *workgroups == 0 {
                    return false;
                }
                let children = extract_list(egraph, *children);
                let body_idx = *num_inputs as usize;
                if body_idx + 1 >= children.len() {
                    return false;
                }
                let body_id = children[body_idx];

                let has_matching_theta = egraph[body_id].iter().any(|n| {
                    let TensorIr::Simd(SimdNode::Theta {
                        children: [init, count, update],
                    }) = n
                    else {
                        return false;
                    };
                    // Cooperative split assumes a scalar-init, 2-arg-BinOp
                    // reduction body with no nested Theta. Pack inits
                    // (running reductions) have no `constant` and get
                    // rejected by the init check below.
                    if egraph[*update].data.contains_theta {
                        return false;
                    }
                    let Some(ScalarValue::U32(k)) = &egraph[*count].data.constant else {
                        return false;
                    };
                    if *k > simd_width {
                        return false;
                    }
                    // Phase-1 cooperative lowering covers large reductions;
                    // this split fills the small-K scalar-lowered gap.
                    if egraph[*init].data.constant.is_none() {
                        return false;
                    }
                    let has_reduce_op = egraph[*update].iter().any(|u| {
                        if let TensorIr::BinOp(name, args) = u {
                            args.len() == 2 && reduce_op_for_dtype(egraph, *name, *init).is_some()
                        } else {
                            false
                        }
                    });
                    has_reduce_op && egraph[*update].data.var_dep.contains(&VarRef::iter(0))
                });

                if !has_matching_theta {
                    return false;
                }

                let already_has_coop = eclass_data.iter().any(|n| {
                    let TensorIr::Dispatch(DispatchNode::Dispatch {
                        children_list: c,
                        num_inputs: ni,
                        ..
                    }) = n
                    else {
                        return false;
                    };
                    let c = extract_list(egraph, *c);
                    let bi = *ni as usize;
                    bi < c.len()
                        && egraph[c[bi]].iter().any(|inner| {
                            matches!(inner, TensorIr::Simd(SimdNode::ReduceSimd { .. }))
                        })
                });

                !already_has_coop
            })
        }),
        crate::applier::AdaptedApplier(ThetaSplitApplier { simd_width }),
    )
    .unwrap()
}

struct ThetaSplitApplier {
    simd_width: u32,
}

struct SplitThetaDispatchSpec<'a> {
    workgroups: u32,
    num_inputs: u32,
    children: &'a [Id],
    output_addr_id: Id,
    init: Id,
    k: u32,
    op_name: BinaryOp,
    value: Id,
    op: ReduceOp,
}

fn build_split_theta_dispatch(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    spec: &SplitThetaDispatchSpec<'_>,
    simd_width: u32,
) -> Option<Id> {
    if !is_flat_thread_output_addr(egraph, spec.output_addr_id, simd_width) {
        return None;
    }

    let mut chosen = HashMap::new();
    let k_var = add_and_choose(
        egraph,
        &mut chosen,
        TensorIr::Simd(SimdNode::Var(VarRef::iter(0))),
    );
    let new_workgroup = add_and_choose(
        egraph,
        &mut chosen,
        TensorIr::Simd(SimdNode::Var(VarRef::thread(IndexLevel::Workgroup))),
    );
    let new_lane = add_and_choose(
        egraph,
        &mut chosen,
        TensorIr::Simd(SimdNode::Var(VarRef::thread(IndexLevel::Lane))),
    );
    let sw_lit = add_and_choose(
        egraph,
        &mut chosen,
        TensorIr::Const(ScalarValue::U32(simd_width)),
    );
    let old_workgroup = add_and_choose(
        egraph,
        &mut chosen,
        TensorIr::BinOp(BinaryOp::Div, [new_workgroup, sw_lit]),
    );
    let old_lane = add_and_choose(
        egraph,
        &mut chosen,
        TensorIr::BinOp(BinaryOp::Mod, [new_workgroup, sw_lit]),
    );
    let k_scaled = add_and_choose(
        egraph,
        &mut chosen,
        TensorIr::BinOp(BinaryOp::Mul, [k_var, sw_lit]),
    );
    let k_remapped = add_and_choose(
        egraph,
        &mut chosen,
        TensorIr::BinOp(BinaryOp::Add, [k_scaled, new_lane]),
    );
    let k_total = add_and_choose(
        egraph,
        &mut chosen,
        TensorIr::Const(ScalarValue::U32(spec.k)),
    );
    let in_bounds = add_and_choose(
        egraph,
        &mut chosen,
        TensorIr::BinOp(BinaryOp::Lt, [k_remapped, k_total]),
    );
    let zero = add_and_choose(egraph, &mut chosen, TensorIr::Const(ScalarValue::U32(0)));
    let is_lane_aligned = spec.k.is_multiple_of(simd_width);
    let k_for_load = if is_lane_aligned {
        k_remapped
    } else {
        add_and_choose(
            egraph,
            &mut chosen,
            TensorIr::TernOp(TernaryOp::Select, [in_bounds, k_remapped, zero]),
        )
    };

    let value =
        remap_dispatch_output_coords(egraph, &mut chosen, spec.value, old_workgroup, old_lane);
    let remapped_value = binding::subst_in_egraph(
        egraph,
        &mut chosen,
        value,
        BinderKind::Theta,
        slots::THETA_ITER,
        0,
        k_for_load,
    );
    let guarded_value = if is_lane_aligned {
        remapped_value
    } else {
        add_and_choose(
            egraph,
            &mut chosen,
            TensorIr::TernOp(TernaryOp::Select, [in_bounds, remapped_value, spec.init]),
        )
    };
    let acc = add_and_choose(
        egraph,
        &mut chosen,
        TensorIr::Simd(SimdNode::Var(VarRef::acc(0))),
    );
    let new_update = add_and_choose(
        egraph,
        &mut chosen,
        TensorIr::BinOp(spec.op_name, [acc, guarded_value]),
    );
    let new_count_val = spec.k.div_ceil(simd_width);
    let new_count = add_and_choose(
        egraph,
        &mut chosen,
        TensorIr::Const(ScalarValue::U32(new_count_val)),
    );
    let new_theta = add_and_choose(
        egraph,
        &mut chosen,
        TensorIr::Simd(SimdNode::Theta {
            children: [spec.init, new_count, new_update],
        }),
    );
    let reduced = add_and_choose(
        egraph,
        &mut chosen,
        TensorIr::Simd(SimdNode::ReduceSimd {
            op: spec.op,
            src: new_theta,
        }),
    );

    let new_workgroups = spec.workgroups.checked_mul(simd_width)?;
    let mut new_children = spec.children[..spec.num_inputs as usize].to_vec();
    new_children.push(reduced);
    new_children.push(new_workgroup);

    if !candidate_device_loads_in_bounds(
        egraph,
        &chosen,
        reduced,
        &new_children[..spec.num_inputs as usize],
        new_workgroups,
        simd_width,
    ) {
        return None;
    }

    try_add_value_addr_dispatch(egraph, new_workgroups, spec.num_inputs, &new_children)
}

impl crate::applier::TypedApplier for ThetaSplitApplier {
    fn apply(&self, egraph: &mut EGraph<TensorIr, TensorAnalysis>, eclass: Id) -> Vec<Id> {
        let dispatch_node = egraph[eclass]
            .iter()
            .find(|n| matches!(n, TensorIr::Dispatch(DispatchNode::Dispatch { .. })))
            .cloned();

        let Some(TensorIr::Dispatch(DispatchNode::Dispatch {
            workgroups,
            num_inputs,
            children_list: children,
            ..
        })) = dispatch_node
        else {
            return vec![];
        };
        let children = extract_list(egraph, children);
        let body_idx = num_inputs as usize;
        let body_id = children[body_idx];
        let output_addr_id = children[body_idx + 1];

        let theta_node = egraph[body_id]
            .iter()
            .find(|n| {
                if let TensorIr::Simd(SimdNode::Theta {
                    children: [init, count, update],
                }) = n
                {
                    // Structural gate: scalar-init reductions with no
                    // nested Theta. Pack inits have no `constant`.
                    if egraph[*update].data.contains_theta {
                        return false;
                    }
                    matches!(&egraph[*count].data.constant, Some(ScalarValue::U32(_)))
                        && egraph[*init].data.constant.is_some()
                } else {
                    false
                }
            })
            .cloned();

        let Some(TensorIr::Simd(SimdNode::Theta {
            children: [init, count, update],
            ..
        })) = theta_node
        else {
            return vec![];
        };

        let k = match &egraph[count].data.constant {
            Some(ScalarValue::U32(v)) => *v,
            _ => return vec![],
        };
        if k > self.simd_width {
            return vec![];
        }

        let update_node = egraph[update]
            .iter()
            .find(|n| matches!(n, TensorIr::BinOp(_, args) if args.len() == 2))
            .cloned();

        let Some(TensorIr::BinOp(op_name, update_args)) = update_node else {
            return vec![];
        };

        let Some(op) = reduce_op_for_dtype(egraph, op_name, init) else {
            return vec![];
        };
        if update_args.len() != 2 {
            return vec![];
        }
        if !is_theta_acc(egraph, update_args[0]) {
            return vec![];
        }

        let Some(new_dispatch) = build_split_theta_dispatch(
            egraph,
            &SplitThetaDispatchSpec {
                workgroups,
                num_inputs,
                children: &children,
                output_addr_id,
                init,
                k,
                op_name,
                value: update_args[1],
                op,
            },
            self.simd_width,
        ) else {
            return vec![];
        };

        egraph.union(eclass, new_dispatch);
        vec![new_dispatch]
    }
}

fn remap_dispatch_output_coords(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    root: Id,
    old_workgroup: Id,
    old_lane: Id,
) -> Id {
    let remapped_workgroup = binding::subst_in_egraph(
        egraph,
        chosen,
        root,
        BinderKind::Dispatch,
        slots::DISPATCH_WORKGROUP,
        0,
        old_workgroup,
    );
    binding::subst_in_egraph(
        egraph,
        chosen,
        remapped_workgroup,
        BinderKind::Dispatch,
        slots::DISPATCH_LANE,
        0,
        old_lane,
    )
}

fn dispatch_var(egraph: &EGraph<TensorIr, TensorAnalysis>, id: Id, slot: u8) -> bool {
    egraph[egraph.find(id)].iter().any(|node| {
        matches!(
            node,
            TensorIr::Simd(SimdNode::Var(VarRef::Bound {
                kind: BinderKind::Dispatch,
                slot: node_slot,
                depth: 0,
            })) if *node_slot == slot
        )
    })
}

fn u32_const(egraph: &EGraph<TensorIr, TensorAnalysis>, id: Id) -> Option<u32> {
    egraph[egraph.find(id)].iter().find_map(|node| match node {
        TensorIr::Const(ScalarValue::U32(v)) => Some(*v),
        TensorIr::Const(ScalarValue::I32(v)) if *v >= 0 => Some((*v).cast_unsigned()),
        _ => None,
    })
}

fn is_flat_thread_output_addr(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    addr_id: Id,
    simd_width: u32,
) -> bool {
    let canonical = egraph.find(addr_id);
    for node in egraph[canonical].iter() {
        let TensorIr::BinOp(BinaryOp::Add, args) = node else {
            continue;
        };
        for (mul, lane) in [(args[0], args[1]), (args[1], args[0])] {
            if !dispatch_var(egraph, lane, slots::DISPATCH_LANE) {
                continue;
            }
            for mul_node in egraph[egraph.find(mul)].iter() {
                let TensorIr::BinOp(BinaryOp::Mul, mul_args) = mul_node else {
                    continue;
                };
                for (workgroup, width) in [(mul_args[0], mul_args[1]), (mul_args[1], mul_args[0])]
                {
                    if dispatch_var(egraph, workgroup, slots::DISPATCH_WORKGROUP)
                        && u32_const(egraph, width) == Some(simd_width)
                    {
                        return true;
                    }
                }
            }
        }
    }
    false
}

#[derive(Clone, Copy)]
struct AddrInterval {
    lo: u32,
    hi: u32,
}

impl AddrInterval {
    const fn point(value: u32) -> Self {
        Self {
            lo: value,
            hi: value,
        }
    }

    const fn range(lo: u32, hi: u32) -> Self {
        Self { lo, hi }
    }

    fn add(self, other: Self) -> Self {
        Self {
            lo: self.lo.saturating_add(other.lo),
            hi: self.hi.saturating_add(other.hi),
        }
    }

    fn sub(self, other: Self) -> Self {
        Self {
            lo: self.lo.saturating_sub(other.hi),
            hi: self.hi.saturating_sub(other.lo),
        }
    }

    fn mul(self, other: Self) -> Self {
        Self {
            lo: self.lo.saturating_mul(other.lo),
            hi: self.hi.saturating_mul(other.hi),
        }
    }

    fn div(self, other: Self) -> Self {
        let denom_hi = other.hi.max(1);
        let denom_lo = other.lo.max(1);
        Self {
            lo: self.lo / denom_hi,
            hi: self.hi / denom_lo,
        }
    }

    fn modu(self, other: Self) -> Self {
        let divisor_upper = if other.lo > 0 {
            other.hi.saturating_sub(1)
        } else {
            u32::MAX
        };
        Self {
            lo: 0,
            hi: self.hi.min(divisor_upper),
        }
    }
}

fn chosen_node(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen: &HashMap<Id, TensorIr>,
    id: Id,
) -> Option<TensorIr> {
    let canonical = egraph.find(id);
    chosen
        .get(&canonical)
        .cloned()
        .or_else(|| egraph[canonical].nodes.first().cloned())
}

fn selected_u32_const(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen: &HashMap<Id, TensorIr>,
    id: Id,
) -> Option<u32> {
    match chosen_node(egraph, chosen, id)? {
        TensorIr::Const(ScalarValue::U32(v)) => Some(v),
        TensorIr::Const(ScalarValue::I32(v)) if v >= 0 => Some(v.cast_unsigned()),
        _ => None,
    }
}

fn candidate_device_loads_in_bounds(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen: &HashMap<Id, TensorIr>,
    output_value: Id,
    input_ids: &[Id],
    workgroups: u32,
    simd_width: u32,
) -> bool {
    let input_lengths = input_ids
        .iter()
        .map(|input| {
            egraph[egraph.find(*input)]
                .data
                .shape
                .as_ref()?
                .static_numel()
        })
        .collect::<Option<Vec<_>>>();
    let Some(input_lengths) = input_lengths else {
        return false;
    };
    device_loads_in_bounds_rec(
        egraph,
        chosen,
        output_value,
        &input_lengths,
        workgroups,
        simd_width,
        &[],
        &mut HashSet::new(),
    )
}

#[allow(clippy::too_many_arguments)]
fn device_loads_in_bounds_rec(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen: &HashMap<Id, TensorIr>,
    id: Id,
    input_lengths: &[u32],
    workgroups: u32,
    simd_width: u32,
    theta_bounds: &[u32],
    visited: &mut HashSet<Id>,
) -> bool {
    let canonical = egraph.find(id);
    if !visited.insert(canonical) {
        return true;
    }
    let Some(node) = chosen_node(egraph, chosen, canonical) else {
        return false;
    };

    if let TensorIr::Simd(SimdNode::Load {
        tier: MemTier::Device(BufferRef::Input(slot)),
        children,
    }) = &node
    {
        let Some(input_len) = input_lengths.get(*slot as usize).copied() else {
            return false;
        };
        let Some(interval) = addr_interval(
            egraph,
            chosen,
            children[0],
            workgroups,
            simd_width,
            theta_bounds,
            &mut HashSet::new(),
        ) else {
            return false;
        };
        if interval.hi >= input_len {
            return false;
        }
    }

    if let TensorIr::Simd(SimdNode::Theta {
        children: [init, count, update],
    }) = &node
    {
        let Some(count) = selected_u32_const(egraph, chosen, *count) else {
            return false;
        };
        let mut inner_theta_bounds = Vec::with_capacity(theta_bounds.len() + 1);
        inner_theta_bounds.push(count.saturating_sub(1));
        inner_theta_bounds.extend_from_slice(theta_bounds);
        return device_loads_in_bounds_rec(
            egraph,
            chosen,
            *init,
            input_lengths,
            workgroups,
            simd_width,
            theta_bounds,
            visited,
        ) && device_loads_in_bounds_rec(
            egraph,
            chosen,
            *update,
            input_lengths,
            workgroups,
            simd_width,
            &inner_theta_bounds,
            visited,
        );
    }

    for child in node.children() {
        if !device_loads_in_bounds_rec(
            egraph,
            chosen,
            *child,
            input_lengths,
            workgroups,
            simd_width,
            theta_bounds,
            visited,
        ) {
            return false;
        }
    }
    true
}

fn same_eclass(egraph: &EGraph<TensorIr, TensorAnalysis>, lhs: Id, rhs: Id) -> bool {
    egraph.find(lhs) == egraph.find(rhs)
}

fn safe_select_interval(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen: &HashMap<Id, TensorIr>,
    args: [Id; 3],
) -> Option<AddrInterval> {
    let TensorIr::BinOp(BinaryOp::Lt, [lhs, rhs]) = chosen_node(egraph, chosen, args[0])? else {
        return None;
    };
    if !same_eclass(egraph, lhs, args[1]) || selected_u32_const(egraph, chosen, args[2]) != Some(0)
    {
        return None;
    }
    let upper = selected_u32_const(egraph, chosen, rhs)?.saturating_sub(1);
    Some(AddrInterval::range(0, upper))
}

#[allow(clippy::too_many_arguments)]
fn addr_interval(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen: &HashMap<Id, TensorIr>,
    id: Id,
    workgroups: u32,
    simd_width: u32,
    theta_bounds: &[u32],
    visiting: &mut HashSet<Id>,
) -> Option<AddrInterval> {
    let canonical = egraph.find(id);
    if !visiting.insert(canonical) {
        return None;
    }
    let node = chosen_node(egraph, chosen, canonical)?;
    let result = match node {
        TensorIr::Const(ScalarValue::U32(v)) => Some(AddrInterval::point(v)),
        TensorIr::Const(ScalarValue::I32(v)) if v >= 0 => {
            Some(AddrInterval::point(v.cast_unsigned()))
        }
        TensorIr::Simd(SimdNode::Var(VarRef::Bound {
            kind: BinderKind::Dispatch,
            slot: slots::DISPATCH_LANE,
            depth: 0,
        })) => Some(AddrInterval::range(0, simd_width.saturating_sub(1))),
        TensorIr::Simd(SimdNode::Var(VarRef::Bound {
            kind: BinderKind::Dispatch,
            slot: slots::DISPATCH_WORKGROUP,
            depth: 0,
        })) => Some(AddrInterval::range(0, workgroups.saturating_sub(1))),
        TensorIr::Simd(SimdNode::Var(VarRef::Bound {
            kind: BinderKind::Dispatch,
            slot: slots::DISPATCH_SIMDGROUP,
            depth: 0,
        })) => Some(AddrInterval::point(0)),
        TensorIr::Simd(SimdNode::Var(VarRef::Bound {
            kind: BinderKind::Theta,
            slot: slots::THETA_ITER,
            depth,
        })) => theta_bounds
            .get(depth as usize)
            .copied()
            .map(|bound| AddrInterval::range(0, bound)),
        TensorIr::BinOp(op, args) => {
            let lhs = addr_interval(
                egraph,
                chosen,
                args[0],
                workgroups,
                simd_width,
                theta_bounds,
                visiting,
            )?;
            let rhs = addr_interval(
                egraph,
                chosen,
                args[1],
                workgroups,
                simd_width,
                theta_bounds,
                visiting,
            )?;
            match op {
                BinaryOp::Add => Some(lhs.add(rhs)),
                BinaryOp::Sub => Some(lhs.sub(rhs)),
                BinaryOp::Mul => Some(lhs.mul(rhs)),
                BinaryOp::Div => Some(lhs.div(rhs)),
                BinaryOp::Mod => Some(lhs.modu(rhs)),
                _ => None,
            }
        }
        TensorIr::TernOp(TernaryOp::Select, args) => {
            if let Some(interval) = safe_select_interval(egraph, chosen, args) {
                Some(interval)
            } else {
                let accept = addr_interval(
                    egraph,
                    chosen,
                    args[1],
                    workgroups,
                    simd_width,
                    theta_bounds,
                    visiting,
                )?;
                let reject = addr_interval(
                    egraph,
                    chosen,
                    args[2],
                    workgroups,
                    simd_width,
                    theta_bounds,
                    visiting,
                )?;
                Some(AddrInterval::range(
                    accept.lo.min(reject.lo),
                    accept.hi.max(reject.hi),
                ))
            }
        }
        _ => None,
    };
    visiting.remove(&canonical);
    result
}

fn reduce_op_for_dtype(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    op_name: BinaryOp,
    init: Id,
) -> Option<ReduceOp> {
    let op = ReduceOp::from_bin_op(op_name)?;
    let dtype = egraph[init].data.dtype?;
    op.supports_dtype(dtype).then_some(op)
}

fn is_theta_acc(egraph: &EGraph<TensorIr, TensorAnalysis>, id: Id) -> bool {
    egraph[egraph.find(id)].iter().any(|node| {
        matches!(
            node,
            TensorIr::Simd(SimdNode::Var(VarRef::Bound {
                kind: BinderKind::Theta,
                slot: slots::THETA_ACC,
                depth: 0,
            }))
        )
    })
}
