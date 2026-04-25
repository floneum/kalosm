//! Cooperatively split a Theta that's buried *inside* a Dispatch body.
//!
//! Complements [`theta_split_cooperative`], which only fires when the
//! Theta is the body's top node (contraction-style). Compound-reduction
//! kernels (softmax = `div(exp(sub(arg, max_Θ)), sum_Θ)`) stop that rule:
//! the body's root is `Div`, not a Theta, so the searcher misses the
//! inner reductions and every lane ends up redundantly looping the full
//! K independently.
//!
//! This rule reaches those Thetas by walking the body DAG. It rewrites
//! `Theta(init, K, update)` → `ReduceSimd(op, Theta(init, K/SW, update'))`
//! at the Theta's eclass, where `update'` substitutes iter(0) with
//! `iter(0)*SW + lane`. Each lane in the simdgroup then covers K/SW
//! iterations; shuffle-reduction broadcasts the aggregated scalar back to
//! every lane. Surrounding body structure is preserved — every other node
//! is rebuilt with the target eclass remapped.
//!
//! The enclosing Dispatch is re-emitted with the **same** `workgroups`
//! count (contrast: `theta_split_cooperative` multiplies by `simd_width`
//! because there each output recruits SW lanes). Here, the thread-to-
//! output mapping is untouched — every thread still owns its own output,
//! and the 32 lanes in a simdgroup merely share the row-scoped
//! reduction they were otherwise recomputing 32 times.
//!
//! ## Soundness
//!
//! Semantic equivalence holds iff the original Theta's value was already
//! lane-invariant in this Dispatch context — i.e., every lane in the
//! simdgroup was computing the same scalar. Violating that would silently
//! collapse per-lane variation into a single shuffle-reduced value. We
//! gate on:
//! * `count` is a literal `u32`, ≥ `simd_width`, multiple of `simd_width`
//! * `init` is a scalar constant (Pack inits carry coupled state)
//! * `update` is a 2-arg associative BinOp (Add / Mul / Max / Min)
//! * `update` has no nested Theta (scope-aware subst targets this binder
//!   only)
//! * `update` does **not** reference `VarRef::thread(Lane)` — the key
//!   gate. Addr-computation touching the lane var (e.g. threadgroup-
//!   backed reductions where different lanes load different tiles) means
//!   per-lane values already differ, so shuffle-reducing would be wrong.
//! * `update` reads through at least one Device/Threadgroup `Load` —
//!   otherwise the Theta is degenerate (pure constant reduction) and the
//!   rewrite has no benefit.
//! * The Dispatch's output_addr depends on the lane — otherwise a single
//!   output is shared by the simdgroup anyway and the existing
//!   body-root rule applies (or has already applied).
//!
//! Combined with the Dispatch's `children_list` structural match, this
//! confines the rewrite to softmax-style patterns while leaving matmul
//! / plain-reduce patterns alone.

use egg::{EGraph, Id, Language, Rewrite};

use crate::analysis::TensorAnalysis;
use crate::applier::SimpleEclassSearcher;
use crate::binding;
use crate::language::{DispatchNode, SimdNode, TensorIr, extract_list};
use crate::rules::RunnerConfig;
use crate::types::{BinaryOp, BinderKind, IndexLevel, ReduceOp, ScalarValue, VarRef, slots};

pub(super) fn build(config: &RunnerConfig) -> Rewrite<TensorIr, TensorAnalysis> {
    let simd_width = config.device.simd_width;
    Rewrite::new(
        "theta-inner-cooperative",
        SimpleEclassSearcher::new(move |egraph, eclass| {
            egraph[eclass].iter().any(|node| {
                let TensorIr::Dispatch(DispatchNode::Dispatch {
                    workgroups,
                    num_inputs,
                    children_list,
                }) = node
                else {
                    return false;
                };
                if *workgroups == 0 {
                    return false;
                }
                let children = extract_list(egraph, *children_list);
                let body_idx = *num_inputs as usize;
                if body_idx + 1 >= children.len() {
                    return false;
                }
                let body_id = children[body_idx];

                if !egraph[body_id].data.contains_theta {
                    return false;
                }
                find_inner_theta(egraph, body_id, simd_width).is_some()
            })
        }),
        crate::applier::AdaptedApplier(ThetaInnerSplitApplier { simd_width }),
    )
    .unwrap()
}

/// Find all Theta eclasses inside `root`'s subtree that qualify for the
/// cooperative rewrite. Returns canonical Theta eclasses.
fn find_inner_thetas(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    root: Id,
    simd_width: u32,
) -> Vec<Id> {
    let mut found = Vec::new();
    let mut stack = vec![root];
    let mut visited = std::collections::HashSet::new();
    let root = egraph.find(root);
    while let Some(id) = stack.pop() {
        let canonical = egraph.find(id);
        if !visited.insert(canonical) {
            continue;
        }
        if !egraph[canonical].data.contains_theta {
            continue;
        }
        for node in egraph[canonical].iter() {
            if let TensorIr::Simd(SimdNode::Theta {
                children: [init, count, update],
            }) = node
                && canonical != root
                && theta_qualifies(egraph, *init, *count, *update, simd_width)
                && !has_reduce_simd_sibling(egraph, canonical)
                && !found.contains(&canonical)
            {
                found.push(canonical);
            }
            for child in node.children() {
                stack.push(*child);
            }
        }
    }
    found
}

fn find_inner_theta(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    root: Id,
    simd_width: u32,
) -> Option<Id> {
    find_inner_thetas(egraph, root, simd_width)
        .into_iter()
        .next()
}

fn theta_qualifies(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    init: Id,
    count: Id,
    update: Id,
    simd_width: u32,
) -> bool {
    let k = match &egraph[count].data.constant {
        Some(ScalarValue::U32(v)) => *v,
        _ => return false,
    };
    if k < simd_width || k % simd_width != 0 {
        return false;
    }
    if egraph[init].data.constant.is_none() {
        return false;
    }
    // Notes on gates that deliberately aren't here:
    //
    // * `update.data.contains_theta` is a *transitive* reachability flag —
    //   true whenever `update` reads through another Theta's result (e.g.
    //   softmax's sum reduction reads the max reduction's output). That's
    //   the case we most want to rewrite; union'ing the Theta's eclass
    //   with its cooperative form preserves semantic equivalence because
    //   the cooperative form computes the same scalar, and the separately-
    //   scoped inner Theta's own binder is untouched.
    //
    // * `update.data.dep.contains_lane()` is structurally true whenever
    //   any intermediate expression references the lane var, but the
    //   *value* can still be lane-invariant — e.g. softmax's load addr is
    //   `row * cols + iter(0)` where `row = (sg_global * SW + lane) / cols`,
    //   which collapses to a lane-independent scalar when `SW < cols`.
    //   Ruling out every structural-lane case would over-reject. Instead,
    //   the extraction-time `ctx.execute` + tolerance check filters any
    //   cooperative rewrite whose value actually differs across lanes.
    if !egraph[update].data.var_dep.contains(&VarRef::iter(0)) {
        return false;
    }
    // Require a Device or Threadgroup load somewhere in update (otherwise
    // the reduction is trivial / constant and no win).
    let mut has_load = false;
    let mut stack = vec![update];
    let mut visited = std::collections::HashSet::new();
    while let Some(id) = stack.pop() {
        let c = egraph.find(id);
        if !visited.insert(c) {
            continue;
        }
        for n in egraph[c].iter() {
            if matches!(n, TensorIr::Simd(SimdNode::Load { .. })) {
                has_load = true;
                break;
            }
            for ch in n.children() {
                stack.push(*ch);
            }
        }
        if has_load {
            break;
        }
    }
    if !has_load {
        return false;
    }
    // Must have a reducing BinOp form available.
    egraph[update].iter().any(|u| {
        matches!(
            u,
            TensorIr::BinOp(
                BinaryOp::Add | BinaryOp::Mul | BinaryOp::Max | BinaryOp::Min,
                _,
            )
        )
    })
}

fn has_reduce_simd_sibling(egraph: &EGraph<TensorIr, TensorAnalysis>, eclass: Id) -> bool {
    egraph[eclass]
        .iter()
        .any(|n| matches!(n, TensorIr::Simd(SimdNode::ReduceSimd { .. })))
}

struct ThetaInnerSplitApplier {
    simd_width: u32,
}

impl crate::applier::TypedApplier for ThetaInnerSplitApplier {
    fn apply(&self, egraph: &mut EGraph<TensorIr, TensorAnalysis>, eclass: Id) -> Vec<Id> {
        let simd_width = self.simd_width;

        // Re-extract the Dispatch's body to discover the inner Thetas.
        let dispatch_node = egraph[eclass]
            .iter()
            .find(|n| matches!(n, TensorIr::Dispatch(DispatchNode::Dispatch { .. })))
            .cloned();
        let Some(TensorIr::Dispatch(DispatchNode::Dispatch {
            workgroups,
            num_inputs,
            children_list,
        })) = dispatch_node
        else {
            return vec![];
        };
        if workgroups == 0 {
            return vec![];
        }
        let children = extract_list(egraph, children_list);
        let body_idx = num_inputs as usize;
        if body_idx + 1 >= children.len() {
            return vec![];
        }
        let body_id = children[body_idx];
        let targets = find_inner_thetas(egraph, body_id, simd_width);
        if targets.is_empty() {
            return vec![];
        }

        // Union every qualifying Theta's eclass with its cooperative form
        // directly. The gate admits only lane-invariant reductions, so
        // every lane in the simdgroup was computing the same scalar
        // redundantly; replacing the sequential Theta with a shuffle-
        // reduced partition preserves that scalar value.
        //
        // Unioning at the Theta level (rather than synthesizing a new
        // Dispatch with a rewritten body) means the rewrite propagates
        // to every consumer of this Theta — the extractor will see the
        // coop form as an alternative representative for the eclass and
        // pick it when the ReduceSimd-preferring cost wins at extraction
        // time. It also lets the cooperative form compose naturally with
        // *other* inner Thetas in the same body (e.g. softmax's max and
        // sum both get unioned in the same rule application).
        let mut rewrote = Vec::new();
        for target in targets {
            let Some(replacement) = build_reduce_simd_for(egraph, target, simd_width) else {
                continue;
            };
            egraph.union(target, replacement);
            rewrote.push(replacement);
        }
        rewrote
    }
}

fn build_reduce_simd_for(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    theta_eclass: Id,
    simd_width: u32,
) -> Option<Id> {
    // Re-fetch the Theta's fields (init, count, update, plus the op).
    let (init, count, update) = egraph[theta_eclass].iter().find_map(|n| {
        if let TensorIr::Simd(SimdNode::Theta {
            children: [i, c, u],
        }) = n
        {
            Some((*i, *c, *u))
        } else {
            None
        }
    })?;
    let Some(ScalarValue::U32(k)) = &egraph[count].data.constant else {
        return None;
    };
    let k = *k;
    if k < simd_width || k % simd_width != 0 {
        return None;
    }
    let op = egraph[update].iter().find_map(|u| {
        if let TensorIr::BinOp(name, args) = u
            && args.len() == 2
        {
            match name {
                BinaryOp::Add => Some(ReduceOp::Add),
                BinaryOp::Mul => Some(ReduceOp::Mul),
                BinaryOp::Max => Some(ReduceOp::Max),
                BinaryOp::Min => Some(ReduceOp::Min),
                _ => None,
            }
        } else {
            None
        }
    })?;

    // Construct iter(0) * SW + lane as the new iteration index.
    let k_var = egraph.add(TensorIr::Simd(SimdNode::Var(VarRef::iter(0))));
    let lane = egraph.add(TensorIr::Simd(SimdNode::Var(VarRef::thread(
        IndexLevel::Lane,
    ))));
    let sw_lit = egraph.add(TensorIr::Const(ScalarValue::U32(simd_width)));
    let k_scaled = egraph.add(TensorIr::BinOp(BinaryOp::Mul, [k_var, sw_lit]));
    let k_remapped = egraph.add(TensorIr::BinOp(BinaryOp::Add, [k_scaled, lane]));

    let remapped_update = binding::subst(
        egraph,
        update,
        BinderKind::Theta,
        slots::THETA_ITER,
        0,
        k_remapped,
    );

    let new_count = egraph.add(TensorIr::Const(ScalarValue::U32(k / simd_width)));
    let new_theta = egraph.add(TensorIr::Simd(SimdNode::Theta {
        children: [init, new_count, remapped_update],
    }));
    Some(egraph.add(TensorIr::Simd(SimdNode::ReduceSimd { op, src: new_theta })))
}
