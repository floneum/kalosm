//! General scalar commutativity rewrites.
//!
//! These rules make operand order an e-graph choice instead of something each
//! downstream rewrite has to rediscover with bespoke pattern branches.

use egg::{EGraph, Id, Rewrite};

use crate::analysis::TensorAnalysis;
use crate::applier::SimpleEclassSearcher;
use crate::language::TensorIr;
use crate::rules::RunnerConfig;

pub(super) fn build(_config: &RunnerConfig) -> Rewrite<TensorIr, TensorAnalysis> {
    Rewrite::new(
        "commutative-binop-swap",
        SimpleEclassSearcher::new(|egraph, eclass| {
            if !egraph[egraph.find(eclass)].data.has_commutative_binop {
                return false;
            }
            egraph[egraph.find(eclass)].iter().any(|node| {
                commutative_binop(node).is_some_and(|swapped| !has_node(egraph, eclass, &swapped))
            })
        }),
        crate::applier::AdaptedApplier(CommutativeSwapApplier),
    )
    .unwrap()
}

struct CommutativeSwapApplier;

impl crate::applier::TypedApplier for CommutativeSwapApplier {
    fn apply(&self, egraph: &mut EGraph<TensorIr, TensorAnalysis>, eclass: Id) -> Vec<Id> {
        let nodes: Vec<TensorIr> = egraph[egraph.find(eclass)].iter().cloned().collect();
        let mut out = Vec::new();
        for node in nodes {
            let Some(swapped) = commutative_binop(&node) else {
                continue;
            };
            if has_node(egraph, eclass, &swapped) {
                continue;
            }
            let id = egraph.add(swapped);
            egraph.union(eclass, id);
            out.push(id);
        }
        out
    }
}

fn commutative_binop(node: &TensorIr) -> Option<TensorIr> {
    let TensorIr::BinOp(op, [lhs, rhs]) = node else {
        return None;
    };
    if !op.is_commutative() || lhs == rhs {
        return None;
    }
    Some(TensorIr::BinOp(*op, [*rhs, *lhs]))
}

fn has_node(egraph: &EGraph<TensorIr, TensorAnalysis>, eclass: Id, needle: &TensorIr) -> bool {
    egraph[egraph.find(eclass)]
        .iter()
        .any(|node| node == needle)
}
