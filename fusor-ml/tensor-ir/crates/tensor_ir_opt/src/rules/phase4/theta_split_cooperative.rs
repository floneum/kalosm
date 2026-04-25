use egg::{EGraph, Id, Rewrite};

use crate::analysis::TensorAnalysis;
use crate::applier::SimpleEclassSearcher;
use crate::binding;
use crate::language::{DispatchNode, SimdNode, TensorIr, add_list, extract_list};
use crate::rules::RunnerConfig;
use crate::types::{
    BinaryOp, BinderKind, IndexLevel, ReduceOp, ScalarValue, TernaryOp, VarRef, slots,
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
                    let Some(ScalarValue::U32(_)) = &egraph[*count].data.constant else {
                        return false;
                    };
                    // No `k > simd_width` gate — the rule unconditionally
                    // produces a cooperative alternative for any literal-K
                    // scalar-init reducing Theta. Whether the cooperative
                    // form wins is decided by the shape-aware cost model
                    // at extraction.
                    if egraph[*init].data.constant.is_none() {
                        return false;
                    }
                    let has_reduce_op = egraph[*update].iter().any(|u| {
                        if let TensorIr::BinOp(name, args) = u {
                            args.len() == 2
                                && matches!(
                                    name,
                                    BinaryOp::Add | BinaryOp::Mul | BinaryOp::Max | BinaryOp::Min
                                )
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
    update_args: [Id; 2],
    update: Id,
    op: ReduceOp,
}

fn build_split_theta_dispatch(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    spec: &SplitThetaDispatchSpec<'_>,
    simd_width: u32,
) -> Id {
    let k_var = egraph.add(TensorIr::Simd(SimdNode::Var(VarRef::iter(0))));
    let lane = egraph.add(TensorIr::Simd(SimdNode::Var(VarRef::thread(
        IndexLevel::Lane,
    ))));
    let sw_lit = egraph.add(TensorIr::Const(ScalarValue::U32(simd_width)));
    let k_scaled = egraph.add(TensorIr::BinOp(BinaryOp::Mul, [k_var, sw_lit]));
    let k_remapped = egraph.add(TensorIr::BinOp(BinaryOp::Add, [k_scaled, lane]));

    let remapped_update = binding::subst(
        egraph,
        spec.update,
        BinderKind::Theta,
        slots::THETA_ITER,
        0,
        k_remapped,
    );
    let remapped_value = binding::subst(
        egraph,
        spec.update_args[1],
        BinderKind::Theta,
        slots::THETA_ITER,
        0,
        k_remapped,
    );

    let k_total = egraph.add(TensorIr::Const(ScalarValue::U32(spec.k)));
    let in_bounds = egraph.add(TensorIr::BinOp(BinaryOp::Lt, [k_remapped, k_total]));
    let is_lane_aligned = spec.k.is_multiple_of(simd_width);
    let guarded_value = if is_lane_aligned {
        remapped_value
    } else {
        egraph.add(TensorIr::TernOp(
            TernaryOp::Select,
            [in_bounds, remapped_value, spec.init],
        ))
    };
    let new_update = if is_lane_aligned {
        remapped_update
    } else {
        egraph.add(TensorIr::BinOp(
            spec.op_name,
            [spec.update_args[0], guarded_value],
        ))
    };

    let new_count_val = spec.k.div_ceil(simd_width);
    let new_count = egraph.add(TensorIr::Const(ScalarValue::U32(new_count_val)));
    let new_theta = egraph.add(TensorIr::Simd(SimdNode::Theta {
        children: [spec.init, new_count, new_update],
    }));
    let reduced = egraph.add(TensorIr::Simd(SimdNode::ReduceSimd {
        op: spec.op,
        src: new_theta,
    }));
    let new_output_addr = simplify_output_addr(egraph, spec.output_addr_id, simd_width);

    let mut new_children = spec.children[..spec.num_inputs as usize].to_vec();
    new_children.push(reduced);
    new_children.push(new_output_addr);
    let new_children = add_list(egraph, &new_children);

    egraph.add(TensorIr::Dispatch(DispatchNode::Dispatch {
        workgroups: spec.workgroups * simd_width,
        num_inputs: spec.num_inputs,
        children_list: new_children,
    }))
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

        let update_node = egraph[update]
            .iter()
            .find(|n| matches!(n, TensorIr::BinOp(_, args) if args.len() == 2))
            .cloned();

        let Some(TensorIr::BinOp(op_name, update_args)) = update_node else {
            return vec![];
        };

        let op = match op_name {
            BinaryOp::Add => ReduceOp::Add,
            BinaryOp::Mul => ReduceOp::Mul,
            BinaryOp::Max => ReduceOp::Max,
            BinaryOp::Min => ReduceOp::Min,
            _ => return vec![],
        };
        if update_args.len() != 2 {
            return vec![];
        }

        let new_dispatch = build_split_theta_dispatch(
            egraph,
            &SplitThetaDispatchSpec {
                workgroups,
                num_inputs,
                children: &children,
                output_addr_id,
                init,
                k,
                op_name,
                update_args,
                update,
                op,
            },
            self.simd_width,
        );

        egraph.union(eclass, new_dispatch);
        vec![new_dispatch]
    }
}

fn simplify_output_addr(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    addr_id: Id,
    simd_width: u32,
) -> Id {
    let canonical = egraph.find(addr_id);
    for node in egraph[canonical].iter() {
        if let TensorIr::BinOp(name, args) = node {
            if !matches!(name, BinaryOp::Add) || args.len() != 2 {
                continue;
            }
            for (a, b) in [(args[0], args[1]), (args[1], args[0])] {
                let b_is_lane = egraph[b].iter().any(|n| {
                    matches!(
                        n,
                        TensorIr::Simd(SimdNode::Var(VarRef::Bound {
                            kind: BinderKind::Dispatch,
                            slot: slots::DISPATCH_LANE,
                            depth: 0,
                        }))
                    )
                });
                if !b_is_lane {
                    continue;
                }
                for mul_node in egraph[a].iter() {
                    if let TensorIr::BinOp(mul_name, mul_args) = mul_node {
                        if !matches!(mul_name, BinaryOp::Mul) || mul_args.len() != 2 {
                            continue;
                        }
                        for (x, s) in [(mul_args[0], mul_args[1]), (mul_args[1], mul_args[0])] {
                            let is_simd_width = matches!(
                                &egraph[s].data.constant,
                                Some(ScalarValue::U32(v)) if *v == simd_width
                            );
                            if is_simd_width {
                                return x;
                            }
                        }
                    }
                }
            }
        }
    }
    canonical
}
