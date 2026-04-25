//! General scalar associativity rewrites.
//!
//! This intentionally only rotates one direction:
//! `(a op b) op c` -> `a op (b op c)`.
//! That gives downstream rules a more flexible grouping without the immediate
//! bidirectional churn of also materializing the inverse rotation.

use egg::{EGraph, Id, Rewrite};

use crate::analysis::TensorAnalysis;
use crate::applier::SimpleEclassSearcher;
use crate::language::TensorIr;
use crate::rules::RunnerConfig;
use crate::types::BinaryOp;

pub(super) fn build(_config: &RunnerConfig) -> Rewrite<TensorIr, TensorAnalysis> {
    Rewrite::new(
        "associative-binop-rotate-right",
        SimpleEclassSearcher::new(|egraph, eclass| {
            let eclass = egraph.find(eclass);
            if !egraph[eclass].data.has_associative_binop {
                return false;
            }
            find_right_rotation(egraph, eclass).is_some()
        }),
        crate::applier::AdaptedApplier(AssociativeRotateRightApplier),
    )
    .unwrap()
}

#[derive(Debug, Clone, Copy)]
struct RightRotation {
    op: BinaryOp,
    a: Id,
    b: Id,
    c: Id,
}

struct AssociativeRotateRightApplier;

impl crate::applier::TypedApplier for AssociativeRotateRightApplier {
    fn apply(&self, egraph: &mut EGraph<TensorIr, TensorAnalysis>, eclass: Id) -> Vec<Id> {
        let eclass = egraph.find(eclass);
        let Some(rotation) = find_right_rotation(egraph, eclass) else {
            return vec![];
        };

        let right = egraph.add(TensorIr::BinOp(rotation.op, [rotation.b, rotation.c]));
        let rotated_node = TensorIr::BinOp(rotation.op, [rotation.a, right]);
        if egraph
            .lookup(rotated_node.clone())
            .is_some_and(|id| egraph.find(id) == eclass)
        {
            return vec![];
        }

        let rotated = egraph.add(rotated_node);
        egraph.union(eclass, rotated);
        vec![rotated]
    }
}

fn find_right_rotation(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    eclass: Id,
) -> Option<RightRotation> {
    let eclass = egraph.find(eclass);
    for node in egraph[eclass].iter() {
        let TensorIr::BinOp(op, [lhs, c]) = node else {
            continue;
        };
        if !op.is_associative() {
            continue;
        }

        let lhs = egraph.find(*lhs);
        for lhs_node in egraph[lhs].iter() {
            let TensorIr::BinOp(lhs_op, [a, b]) = lhs_node else {
                continue;
            };
            if lhs_op != op {
                continue;
            }

            if right_rotation_exists(egraph, eclass, *op, *a, *b, *c) {
                continue;
            }

            return Some(RightRotation {
                op: *op,
                a: *a,
                b: *b,
                c: *c,
            });
        }
    }
    None
}

fn right_rotation_exists(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    eclass: Id,
    op: BinaryOp,
    a: Id,
    b: Id,
    c: Id,
) -> bool {
    let Some(right) = egraph.lookup(TensorIr::BinOp(op, [b, c])) else {
        return false;
    };
    egraph
        .lookup(TensorIr::BinOp(op, [a, right]))
        .is_some_and(|id| egraph.find(id) == egraph.find(eclass))
}
