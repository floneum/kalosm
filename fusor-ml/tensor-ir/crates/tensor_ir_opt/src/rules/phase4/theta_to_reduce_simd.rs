use std::collections::HashSet;

use egg::{EGraph, Id, Language, Rewrite};

use crate::analysis::TensorAnalysis;
use crate::applier::SimpleEclassSearcher;
use crate::binding;
use crate::language::{SimdNode, TensorIr};
use crate::rules::RunnerConfig;
use crate::types::{BinderKind, IndexLevel, ReduceOp, ScalarValue, VarRef, slots};

pub(super) fn build(config: &RunnerConfig) -> Rewrite<TensorIr, TensorAnalysis> {
    let simd_width = config.device.simd_width;
    Rewrite::new(
        "theta-to-reduce-simd",
        SimpleEclassSearcher::new(move |egraph, eclass| {
            egraph[eclass].iter().any(|node| {
                let TensorIr::Simd(SimdNode::Theta {
                    children: [init, count, update],
                }) = node
                else {
                    return false;
                };
                // Structural eligibility: scalar-init associative reduction
                // with count == simd_width, and a body that's a plain
                // 2-arg BinOp (no nested Theta). Pack inits (running
                // reductions) have no scalar `constant`, so they're
                // rejected by the `constant.is_none()` guard.
                let is_simd_width = matches!(
                    &egraph[*count].data.constant,
                    Some(ScalarValue::U32(v)) if *v == simd_width
                );
                if !is_simd_width || egraph[*init].data.constant.is_none() {
                    return false;
                }
                if egraph[*update].data.contains_theta {
                    return false;
                }
                if subtree_contains_dispatch_lane(egraph, *update) {
                    return false;
                }
                egraph[*update].iter().any(|n| {
                    matches!(
                        n,
                        TensorIr::BinOp(name, args)
                            if args.len() == 2 && reduce_op_for_dtype(egraph, *name, *init).is_some()
                    )
                })
            })
        }),
        crate::applier::AdaptedApplier(ThetaToReduceSimdApplier),
    )
    .unwrap()
}

fn subtree_contains_dispatch_lane(egraph: &EGraph<TensorIr, TensorAnalysis>, root: Id) -> bool {
    let mut stack = vec![root];
    let mut visited = HashSet::new();
    while let Some(id) = stack.pop() {
        let canonical = egraph.find(id);
        if !visited.insert(canonical) {
            continue;
        }
        for node in egraph[canonical].iter() {
            if matches!(
                node,
                TensorIr::Simd(SimdNode::Var(VarRef::Bound {
                    kind: BinderKind::Dispatch,
                    slot: slots::DISPATCH_LANE,
                    ..
                }))
            ) {
                return true;
            }
            stack.extend(node.children().iter().copied());
        }
    }
    false
}

struct ThetaToReduceSimdApplier;

impl crate::applier::TypedApplier for ThetaToReduceSimdApplier {
    fn apply(&self, egraph: &mut EGraph<TensorIr, TensorAnalysis>, eclass: Id) -> Vec<Id> {
        let node = egraph[eclass]
            .iter()
            .find(|n| matches!(n, TensorIr::Simd(SimdNode::Theta { .. })))
            .cloned();

        let Some(TensorIr::Simd(SimdNode::Theta {
            children: [init, _, update],
            ..
        })) = node
        else {
            return vec![];
        };

        if egraph[eclass]
            .iter()
            .any(|n| matches!(n, TensorIr::Simd(SimdNode::ReduceSimd { .. })))
        {
            return vec![];
        }

        let update_node = egraph[update].iter().next().cloned();
        let Some(TensorIr::BinOp(name, args)) = update_node else {
            return vec![];
        };

        let Some(op) = reduce_op_for_dtype(egraph, name, init) else {
            return vec![];
        };

        if args.len() != 2 {
            return vec![];
        }
        let value = args[1];
        let lane = egraph.add(TensorIr::Simd(SimdNode::Var(VarRef::thread(
            IndexLevel::Lane,
        ))));
        let value = binding::subst(egraph, value, BinderKind::Theta, slots::THETA_ITER, 0, lane);
        let value = binding::shift(egraph, value, BinderKind::Theta, 1, -1);
        let rsimd = egraph.add(TensorIr::Simd(SimdNode::ReduceSimd { op, src: value }));
        egraph.union(eclass, rsimd);
        vec![rsimd]
    }
}

fn reduce_op_for_dtype(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    op_name: crate::types::BinaryOp,
    init: Id,
) -> Option<ReduceOp> {
    let op = ReduceOp::from_bin_op(op_name)?;
    let dtype = egraph[init].data.dtype?;
    op.supports_dtype(dtype).then_some(op)
}
