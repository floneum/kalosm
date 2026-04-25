//! Exp/log algebraic equivalences.
//!
//! Scalar-expression rewrites that compose to normalize expressions
//! involving `exp`, `log`, and reductions into forms where independent
//! reductions can be spotted and structurally merged (see
//! `phase2::theta_merge`).
//!
//! Today the module implements two rewrites:
//!
//! - `exp-sub-split`: `exp(a - b) ≡ exp(a) * exp(-b)`.
//! - `factor-reduce-mul-bcast`: `Reduce(a, Add,
//!   Elementwise([x, bcast_a(c)], Mul(f(P0), g(P1))))
//!   ≡ Elementwise([Reduce(a, Add, Elementwise([x], f(P0))), c], Mul(P0, P1))`.
//!   Pulls an axis-invariant factor out of a sum so the remaining
//!   reduction becomes independent of the outer broadcast.
//!
//! Composed: `Σ_a exp(x - bcast(max_a(x)))` becomes `(Σ_a exp(x)) *
//! exp(-max)` after `exp-sub-split` + `factor-reduce-mul-bcast`, so
//! `phase2::theta-merge-reduction` can then fuse the decoupled `max`
//! and `sum` reductions into one `Theta { RunningReduction }`.
//!
//! Future rewrites slotted here:
//! - `exp-neg-log`: `exp(-log(x)) ≡ 1/x`.
//! - `log-exp-id`: `log(exp(x)) ≡ x`.
//! - Ewise-fusion: collapse adjacent single-input elementwises into one
//!   body so `exp-sub-split`'s output can match the Mul-body pattern
//!   `factor-reduce-mul-bcast` expects (today the softmax builder
//!   splits Exp and Sub across two ewises).

use std::collections::HashSet;

use egg::{EGraph, Id, Rewrite};

use crate::analysis::TensorAnalysis;
use crate::applier::SimpleEclassSearcher;
use crate::language::{HighLevelNode, TensorIr, add_list, extract_list};
use crate::rules::RunnerConfig;
use crate::types::{BinaryOp, ReduceOp, Shape, Strides, UnaryOp};

pub(super) fn build_all(_config: &RunnerConfig) -> Vec<Rewrite<TensorIr, TensorAnalysis>> {
    vec![
        Rewrite::new(
            "exp-sub-split",
            SimpleEclassSearcher::new(|egraph, eclass| {
                egraph[eclass]
                    .iter()
                    .any(|node| match_exp_of_sub(egraph, node).is_some())
            }),
            crate::applier::AdaptedApplier(ExpSubSplitApplier),
        )
        .unwrap(),
        Rewrite::new(
            "factor-reduce-mul-bcast",
            SimpleEclassSearcher::new(|egraph, eclass| {
                egraph[eclass]
                    .iter()
                    .any(|node| is_factorable_reduce(egraph, node))
            }),
            crate::applier::AdaptedApplier(FactorReduceMulBcastApplier),
        )
        .unwrap(),
    ]
}

/// Returns `(a, b)` if `node` is `UnOp(Exp, id)` where `id`'s e-class
/// contains a `BinOp(Sub, a, b)`.
fn match_exp_of_sub(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    node: &TensorIr,
) -> Option<(Id, Id)> {
    let TensorIr::UnOp(UnaryOp::Exp, inner) = node else {
        return None;
    };
    egraph[*inner].iter().find_map(|n| {
        if let TensorIr::BinOp(BinaryOp::Sub, [a, b]) = n {
            Some((*a, *b))
        } else {
            None
        }
    })
}

struct ExpSubSplitApplier;

impl crate::applier::TypedApplier for ExpSubSplitApplier {
    fn apply(&self, egraph: &mut EGraph<TensorIr, TensorAnalysis>, eclass: Id) -> Vec<Id> {
        let pair = egraph[eclass]
            .iter()
            .find_map(|node| match_exp_of_sub(egraph, node));
        let Some((a, b)) = pair else {
            return vec![];
        };

        let exp_a = egraph.add(TensorIr::UnOp(UnaryOp::Exp, a));
        let neg_b = egraph.add(TensorIr::UnOp(UnaryOp::Neg, b));
        let exp_neg_b = egraph.add(TensorIr::UnOp(UnaryOp::Exp, neg_b));
        let product = egraph.add(TensorIr::BinOp(BinaryOp::Mul, [exp_a, exp_neg_b]));

        egraph.union(eclass, product);
        vec![product]
    }
}

// ── factor-reduce-mul-bcast ─────────────────────────────────────────

/// Set of `Param(i)` indices referenced in the subtree of `id`. Walks
/// canonical e-nodes conservatively: we pick one representative e-node
/// per e-class (the first) and recurse. That's safe for this rule
/// because the applier below only fires when the body has a specific
/// shape — we don't need perfect precision, just a quick check that a
/// subtree references only the expected param.
fn subtree_params(egraph: &EGraph<TensorIr, TensorAnalysis>, id: Id) -> HashSet<u32> {
    fn walk(
        egraph: &EGraph<TensorIr, TensorAnalysis>,
        id: Id,
        out: &mut HashSet<u32>,
        seen: &mut HashSet<Id>,
    ) {
        let canonical = egraph.find(id);
        if !seen.insert(canonical) {
            return;
        }
        if let Some(node) = egraph[canonical].iter().next() {
            if let TensorIr::HighLevel(HighLevelNode::Param(i)) = node {
                out.insert(*i);
            }
            for child in egg::Language::children(node) {
                walk(egraph, *child, out, seen);
            }
        }
    }
    let mut out = HashSet::new();
    let mut seen = HashSet::new();
    walk(egraph, id, &mut out, &mut seen);
    out
}

/// Returns the input ids whose restride has stride 0 on `axis`. Inputs
/// without an explicit `Restride` return their row-major stride, which
/// is never 0.
fn bcast_input_indices(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    inputs: &[Id],
    axis: u32,
) -> Vec<usize> {
    inputs
        .iter()
        .enumerate()
        .filter(|(_, id)| is_bcast_on_axis(egraph, **id, axis))
        .map(|(i, _)| i)
        .collect()
}

fn is_bcast_on_axis(egraph: &EGraph<TensorIr, TensorAnalysis>, id: Id, axis: u32) -> bool {
    egraph[id].iter().any(|n| {
        let TensorIr::HighLevel(HighLevelNode::Restride { strides, .. }) = n else {
            return false;
        };
        let Strides(s) = strides;
        (axis as usize) < s.len() && s[axis as usize] == 0
    })
}

/// Walk a `Restride` chain to its underlying expression (one stride
/// layer past the broadcast). Returns `None` if the input isn't a
/// Restride.
fn restride_expr(egraph: &EGraph<TensorIr, TensorAnalysis>, id: Id) -> Option<Id> {
    egraph[id].iter().find_map(|n| {
        if let TensorIr::HighLevel(HighLevelNode::Restride { expr, .. }) = n {
            Some(*expr)
        } else {
            None
        }
    })
}

/// Shape of an expression, looked up through analysis metadata.
fn shape_of(egraph: &EGraph<TensorIr, TensorAnalysis>, id: Id) -> Option<Shape> {
    egraph[id].data.shape.clone()
}

struct FactorMatch {
    outer_axis: u32,
    outer_op: ReduceOp,
    index_space: Shape,
    ewise_inputs: Vec<Id>,
    ewise_body_lhs: Id,
    ewise_body_rhs: Id,
    /// Index into `ewise_inputs` — side whose subtree touches only
    /// broadcast params and whose input is a Restride.
    bcast_side: MulSide,
    /// Which ewise input is the broadcast. Expected to be a Restride
    /// whose underlying expression has the reduce-output shape.
    bcast_input_idx: usize,
    /// The remaining (non-broadcast) input(s) that stay inside the
    /// reduce. For the 2-input case, this is the single non-broadcast
    /// input.
    kept_input_idx: usize,
}

#[derive(Clone, Copy)]
enum MulSide {
    Lhs,
    Rhs,
}

/// Predicate form of `try_match_factorable` — used by the searcher so
/// we don't rebuild the full struct when just deciding whether to fire.
fn is_factorable_reduce(egraph: &EGraph<TensorIr, TensorAnalysis>, node: &TensorIr) -> bool {
    try_match_factorable(egraph, node).is_some()
}

fn try_match_factorable(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    node: &TensorIr,
) -> Option<FactorMatch> {
    let TensorIr::HighLevel(HighLevelNode::Reduce { axis, op, expr }) = node else {
        return None;
    };
    // Only Add (and Mul, symmetrically — but keep Add for now to avoid
    // changing the operator-algebra scope in one step).
    if *op != ReduceOp::Add {
        return None;
    }
    // The inner expr must be an Elementwise whose body is a Mul whose
    // two sides partition the ewise params into bcast-only vs
    // nonbcast-only subsets.
    egraph[*expr].iter().find_map(|inner| {
        let TensorIr::HighLevel(HighLevelNode::Elementwise {
            index_space,
            num_inputs,
            children_list,
        }) = inner
        else {
            return None;
        };
        if *num_inputs != 2 {
            return None;
        }
        let children = extract_list(egraph, *children_list);
        let inputs: Vec<Id> = children[..*num_inputs as usize].to_vec();
        let body_id = *children.last()?;

        let bcast_inputs = bcast_input_indices(egraph, &inputs, *axis);
        if bcast_inputs.len() != 1 {
            return None;
        }
        let bcast_idx = bcast_inputs[0];
        let kept_idx = 1 - bcast_idx;

        // Top-level body must be a Mul.
        let (lhs, rhs) = egraph[body_id].iter().find_map(|n| {
            if let TensorIr::BinOp(BinaryOp::Mul, [lhs, rhs]) = n {
                Some((*lhs, *rhs))
            } else {
                None
            }
        })?;
        let lhs_params = subtree_params(egraph, lhs);
        let rhs_params = subtree_params(egraph, rhs);
        let bcast_param = bcast_idx as u32;
        let kept_param = kept_idx as u32;

        // Determine which side is the bcast-only subtree. "bcast-only"
        // means: only references `Param(bcast_param)` (it may reference
        // nothing, which still qualifies — constants factor out).
        let side = if lhs_params.iter().all(|p| *p == bcast_param)
            && !rhs_params.contains(&bcast_param)
            && rhs_params
                .iter()
                .all(|p| *p == kept_param || p != &kept_param)
            && rhs_params.contains(&kept_param)
        {
            MulSide::Lhs
        } else if rhs_params.iter().all(|p| *p == bcast_param)
            && !lhs_params.contains(&bcast_param)
            && lhs_params.contains(&kept_param)
        {
            MulSide::Rhs
        } else {
            return None;
        };

        // The broadcast input must be a Restride so we can extract the
        // pre-broadcast underlying expression.
        restride_expr(egraph, inputs[bcast_idx])?;

        Some(FactorMatch {
            outer_axis: *axis,
            outer_op: *op,
            index_space: index_space.clone(),
            ewise_inputs: inputs,
            ewise_body_lhs: lhs,
            ewise_body_rhs: rhs,
            bcast_side: side,
            bcast_input_idx: bcast_idx,
            kept_input_idx: kept_idx,
        })
    })
}

struct FactorReduceMulBcastApplier;

impl crate::applier::TypedApplier for FactorReduceMulBcastApplier {
    fn apply(&self, egraph: &mut EGraph<TensorIr, TensorAnalysis>, eclass: Id) -> Vec<Id> {
        let m = egraph[eclass]
            .iter()
            .find_map(|node| try_match_factorable(egraph, node));
        let Some(m) = m else {
            return vec![];
        };

        let kept_subtree = match m.bcast_side {
            MulSide::Lhs => m.ewise_body_rhs,
            MulSide::Rhs => m.ewise_body_lhs,
        };
        let bcast_subtree = match m.bcast_side {
            MulSide::Lhs => m.ewise_body_lhs,
            MulSide::Rhs => m.ewise_body_rhs,
        };

        // Build inner_ewise = Elementwise(index_space, [kept_input], kept_subtree).
        let kept_input = m.ewise_inputs[m.kept_input_idx];
        let Some(inner_body) = relabel_params(
            egraph,
            kept_subtree,
            &[(
                u32::try_from(m.kept_input_idx).expect("param index fits in u32"),
                0,
            )],
        ) else {
            return vec![];
        };
        let inner_children_vec = vec![kept_input, inner_body];
        let inner_children_list = add_list(egraph, &inner_children_vec);
        let inner_ewise = egraph.add(TensorIr::HighLevel(HighLevelNode::Elementwise {
            index_space: m.index_space.clone(),
            num_inputs: 1,
            children_list: inner_children_list,
        }));

        // Inner reduce over the same axis/op.
        let inner_reduce = egraph.add(TensorIr::HighLevel(HighLevelNode::Reduce {
            axis: m.outer_axis,
            op: m.outer_op,
            expr: inner_ewise,
        }));

        // Outer wrapping: `Elementwise(output_shape, [inner_reduce,
        // bcast_underlying], Mul(Param(0), <bcast_subtree with Param(1)>))`.
        let bcast_input_id = m.ewise_inputs[m.bcast_input_idx];
        let Some(bcast_underlying) = restride_expr(egraph, bcast_input_id) else {
            return vec![];
        };
        let Some(output_shape) = shape_of(egraph, inner_reduce) else {
            return vec![];
        };
        let Some(outer_factor) = relabel_params(
            egraph,
            bcast_subtree,
            &[(
                u32::try_from(m.bcast_input_idx).expect("param index fits in u32"),
                1,
            )],
        ) else {
            return vec![];
        };
        // Build outer body = Mul(Param(0), bcast factor). The factor has
        // been relabeled to the new slot 1 regardless of its original
        // input position.
        let p0 = egraph.add(TensorIr::HighLevel(HighLevelNode::Param(0)));
        let outer_body = egraph.add(TensorIr::BinOp(BinaryOp::Mul, [p0, outer_factor]));
        let outer_children = vec![inner_reduce, bcast_underlying, outer_body];
        let outer_children_list = add_list(egraph, &outer_children);
        let factored = egraph.add(TensorIr::HighLevel(HighLevelNode::Elementwise {
            index_space: output_shape,
            num_inputs: 2,
            children_list: outer_children_list,
        }));

        egraph.union(eclass, factored);
        vec![factored]
    }
}

fn relabel_params(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    id: Id,
    mapping: &[(u32, u32)],
) -> Option<Id> {
    let canonical = egraph.find(id);
    let node = egraph[canonical]
        .iter()
        .find(|node| {
            matches!(
                node,
                TensorIr::HighLevel(HighLevelNode::Param(_))
                    | TensorIr::Const(_)
                    | TensorIr::BinOp(_, _)
                    | TensorIr::UnOp(_, _)
                    | TensorIr::TernOp(_, _)
            )
        })
        .cloned()?;

    match node {
        TensorIr::HighLevel(HighLevelNode::Param(i)) => {
            let (_, replacement) = mapping.iter().find(|(from, _)| *from == i)?;
            Some(egraph.add(TensorIr::HighLevel(HighLevelNode::Param(*replacement))))
        }
        TensorIr::Const(v) => Some(egraph.add(TensorIr::Const(v))),
        TensorIr::BinOp(op, [lhs, rhs]) => {
            let lhs = relabel_params(egraph, lhs, mapping)?;
            let rhs = relabel_params(egraph, rhs, mapping)?;
            Some(egraph.add(TensorIr::BinOp(op, [lhs, rhs])))
        }
        TensorIr::UnOp(op, arg) => {
            let arg = relabel_params(egraph, arg, mapping)?;
            Some(egraph.add(TensorIr::UnOp(op, arg)))
        }
        TensorIr::TernOp(op, [a, b, c]) => {
            let a = relabel_params(egraph, a, mapping)?;
            let b = relabel_params(egraph, b, mapping)?;
            let c = relabel_params(egraph, c, mapping)?;
            Some(egraph.add(TensorIr::TernOp(op, [a, b, c])))
        }
        _ => None,
    }
}
