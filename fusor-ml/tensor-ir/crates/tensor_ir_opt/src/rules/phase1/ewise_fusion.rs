//! Elementwise-into-elementwise fusion.
//!
//! Collapses `Elementwise(shape, [Elementwise(shape, inner_inputs,
//! body_a)], body_b)` into `Elementwise(shape, inner_inputs,
//! body_b[Param(0) := body_a])`. The inner ewise's single-output
//! result is inlined into the outer ewise's body by substituting for
//! the outer's `Param(0)`.
//!
//! Why: `exp_algebra::exp-sub-split` matches `Exp(Sub(_, _))` at the
//! scalar level, but our softmax builder splits Exp and Sub across two
//! adjacent elementwises — `Elementwise([x, bcast_max], Sub(P0, P1))`
//! fed into `Elementwise([shifted], Exp(P0))`. Without fusion, the
//! `Exp(Sub(...))` never appears syntactically, so exp-sub-split never
//! fires. After ewise-fusion, the body becomes `Exp(Sub(P0, P1))` over
//! inputs `[x, bcast_max]`, which exp-sub-split can rewrite, letting
//! `factor-reduce-mul-bcast` pull the max-term out of the downstream
//! sum.
//!
//! Scope today: only the single-input outer case (outer has exactly
//! one input, which is itself an Elementwise with matching shape).
//! Multi-input outer is a cleaner generalisation once we want it; for
//! the softmax/LSE chain the single-input form suffices.

use egg::{EGraph, Id, Rewrite};

use crate::analysis::TensorAnalysis;
use crate::applier::SimpleEclassSearcher;
use crate::language::{HighLevelNode, TensorIr, add_list, extract_list};
use crate::rules::RunnerConfig;

pub(super) fn build(_config: &RunnerConfig) -> Rewrite<TensorIr, TensorAnalysis> {
    Rewrite::new(
        "ewise-fuse-inner",
        SimpleEclassSearcher::new(|egraph, eclass| {
            egraph[eclass]
                .iter()
                .any(|node| try_match(egraph, node).is_some())
        }),
        crate::applier::AdaptedApplier(EwiseFuseApplier),
    )
    .unwrap()
}

struct OuterMatch {
    index_space: crate::types::Shape,
    inner_inputs: Vec<Id>,
    inner_body: Id,
    outer_body: Id,
    inner_num_inputs: u32,
}

fn try_match(egraph: &EGraph<TensorIr, TensorAnalysis>, node: &TensorIr) -> Option<OuterMatch> {
    let TensorIr::HighLevel(HighLevelNode::Elementwise {
        index_space,
        num_inputs,
        children_list,
    }) = node
    else {
        return None;
    };
    if *num_inputs != 1 {
        return None;
    }
    let outer_children = extract_list(egraph, *children_list);
    let outer_input = outer_children[0];
    let outer_body = *outer_children.last()?;

    egraph[outer_input].iter().find_map(|inner| {
        let TensorIr::HighLevel(HighLevelNode::Elementwise {
            index_space: inner_idx,
            num_inputs: inner_n,
            children_list: inner_children_list,
        }) = inner
        else {
            return None;
        };
        if inner_idx != index_space {
            return None;
        }
        let inner_children = extract_list(egraph, *inner_children_list);
        let inner_inputs: Vec<Id> = inner_children[..*inner_n as usize].to_vec();
        let inner_body = *inner_children.last()?;
        Some(OuterMatch {
            index_space: index_space.clone(),
            inner_inputs,
            inner_body,
            outer_body,
            inner_num_inputs: *inner_n,
        })
    })
}

/// Substitute `Param(0)` with `replacement` in `body_id`, returning the
/// new e-node id. Other `Param(i)` references are left alone (the
/// caller ensures outer has exactly 1 input, so other params shouldn't
/// appear).
fn subst_param_0(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    body_id: Id,
    replacement: Id,
    memo: &mut std::collections::HashMap<Id, Id>,
) -> Id {
    let canonical = egraph.find(body_id);
    if let Some(&cached) = memo.get(&canonical) {
        return cached;
    }
    let node = egraph[canonical].iter().next().cloned();
    let Some(node) = node else {
        return body_id;
    };
    let rebuilt = match node {
        TensorIr::HighLevel(HighLevelNode::Param(0)) => replacement,
        TensorIr::HighLevel(HighLevelNode::Param(_)) => body_id,
        TensorIr::Const(v) => egraph.add(TensorIr::Const(v)),
        TensorIr::BinOp(op, [a, b]) => {
            let a = subst_param_0(egraph, a, replacement, memo);
            let b = subst_param_0(egraph, b, replacement, memo);
            egraph.add(TensorIr::BinOp(op, [a, b]))
        }
        TensorIr::UnOp(op, a) => {
            let a = subst_param_0(egraph, a, replacement, memo);
            egraph.add(TensorIr::UnOp(op, a))
        }
        TensorIr::TernOp(op, [a, b, c]) => {
            let a = subst_param_0(egraph, a, replacement, memo);
            let b = subst_param_0(egraph, b, replacement, memo);
            let c = subst_param_0(egraph, c, replacement, memo);
            egraph.add(TensorIr::TernOp(op, [a, b, c]))
        }
        _ => body_id,
    };
    memo.insert(canonical, rebuilt);
    rebuilt
}

struct EwiseFuseApplier;

impl crate::applier::TypedApplier for EwiseFuseApplier {
    fn apply(&self, egraph: &mut EGraph<TensorIr, TensorAnalysis>, eclass: Id) -> Vec<Id> {
        let m = egraph[eclass]
            .iter()
            .find_map(|node| try_match(egraph, node));
        let Some(m) = m else {
            return vec![];
        };
        let mut memo = std::collections::HashMap::new();
        let fused_body = subst_param_0(egraph, m.outer_body, m.inner_body, &mut memo);

        let mut fused_children = m.inner_inputs.clone();
        fused_children.push(fused_body);
        let fused_children_list = add_list(egraph, &fused_children);
        let fused = egraph.add(TensorIr::HighLevel(HighLevelNode::Elementwise {
            index_space: m.index_space,
            num_inputs: m.inner_num_inputs,
            children_list: fused_children_list,
        }));

        egraph.union(eclass, fused);
        vec![fused]
    }
}
