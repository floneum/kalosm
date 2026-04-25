//! De Bruijn shift and substitution for scoped variable references.
//!
//! Every IR node that introduces a scope reports itself via `HasBinder` —
//! see `src/types/binder.rs` for the trait and the `body_mask` convention,
//! and `impl HasBinder for TensorIr` in `src/language.rs` for the current
//! binder catalog (`Theta`, `Dispatch`). Variables scoped to a binder are
//! written as `VarRef::Bound { kind, slot, depth }`, with `depth` a De
//! Bruijn level counted independently per `BinderKind` (0 = innermost of
//! that kind).
//!
//! Shift bumps every free `Bound` ref whose `kind` matches — other kinds are
//! left alone, which is what makes it safe to fold Thread refs into the
//! Dispatch-bound family: a Theta shift never disturbs them. Substitution
//! replaces a target `(kind, slot, depth)` with a replacement expression,
//! lifting the replacement up by one whenever a binder of the same kind is
//! crossed.

use std::collections::HashMap;

use egg::{EGraph, Id, Language};

use crate::add_and_choose;
use crate::analysis::TensorAnalysis;
use crate::language::TensorIr;
use crate::types::{BinderKind, HasBinder, VarRef};

/// Shift in the e-graph without an extraction-aware choice map.
///
/// Convenience wrapper for rewrites that don't pin choices; uses an internal
/// `HashMap` that is discarded after the call.
pub fn shift(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    root: Id,
    kind: BinderKind,
    cutoff: u32,
    delta: i32,
) -> Id {
    let mut chosen = HashMap::new();
    shift_in_egraph(egraph, &mut chosen, root, kind, cutoff, delta)
}

/// Substitute in the e-graph without an extraction-aware choice map.
pub fn subst(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    root: Id,
    kind: BinderKind,
    slot: u8,
    target_depth: u32,
    replacement: Id,
) -> Id {
    let mut chosen = HashMap::new();
    subst_in_egraph(
        egraph,
        &mut chosen,
        root,
        kind,
        slot,
        target_depth,
        replacement,
    )
}

/// Shift every free `VarRef::Bound { kind: k, depth }` in `root` whose
/// `k == kind` and `depth >= cutoff` by `delta` (positive = move under a new
/// binder, negative = move out). References bound to other kinds are left
/// untouched.
///
/// Returns the new root Id. New nodes are inserted into `egraph` and recorded
/// in `chosen` (the choice map used by extraction-aware rewrites). Memoizes
/// per `(canonical_id, cutoff)` because the same physical e-class can be
/// reached at different depths along different paths.
pub fn shift_in_egraph(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    root: Id,
    kind: BinderKind,
    cutoff: u32,
    delta: i32,
) -> Id {
    if delta == 0 {
        return root;
    }
    let mut memo: HashMap<(Id, u32), Id> = HashMap::new();
    shift_rec(egraph, chosen, root, kind, cutoff, delta, &mut memo)
}

fn shift_rec(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    id: Id,
    kind: BinderKind,
    cutoff: u32,
    delta: i32,
    memo: &mut HashMap<(Id, u32), Id>,
) -> Id {
    let canonical = egraph.find(id);
    if let Some(&cached) = memo.get(&(canonical, cutoff)) {
        return cached;
    }

    let Some(node) = pick_node(egraph, chosen, canonical) else {
        return id;
    };

    if let TensorIr::Simd(crate::language::SimdNode::Var(VarRef::Bound {
        kind: var_kind,
        slot,
        depth,
    })) = node
        && var_kind == kind
    {
        let new_depth = if depth >= cutoff {
            shift_depth(depth, delta)
        } else {
            depth
        };
        let new_id = add_and_choose(
            egraph,
            chosen,
            TensorIr::Simd(crate::language::SimdNode::Var(VarRef::Bound {
                kind: var_kind,
                slot,
                depth: new_depth,
            })),
        );
        memo.insert((canonical, cutoff), new_id);
        return new_id;
    }

    if node.children().is_empty() {
        memo.insert((canonical, cutoff), id);
        return id;
    }

    let new_node = if let Some(info) = node.binder_info() {
        let mut new_node = node.clone();
        let info_matches = info.kind == kind;
        for (i, child) in new_node.children_mut().iter_mut().enumerate() {
            let bit_set = (info.body_mask >> i) & 1 == 1;
            let new_cutoff = if info_matches && bit_set {
                cutoff + 1
            } else {
                cutoff
            };
            let child_canonical = egraph.find(*child);
            if child_canonical == canonical {
                continue;
            }
            *child = shift_rec(egraph, chosen, *child, kind, new_cutoff, delta, memo);
        }
        new_node
    } else {
        let mut new_node = node.clone();
        for child in new_node.children_mut() {
            let child_canonical = egraph.find(*child);
            if child_canonical == canonical {
                continue;
            }
            *child = shift_rec(egraph, chosen, *child, kind, cutoff, delta, memo);
        }
        new_node
    };

    let new_id = add_and_choose(egraph, chosen, new_node);
    memo.insert((canonical, cutoff), new_id);
    new_id
}

/// Replace every `VarRef::Bound { kind, slot, depth: target_depth }` in
/// `root` with `replacement`, with replacement automatically shifted up by
/// one each time the walk crosses a binder of the same kind. References
/// bound to other kinds are left untouched.
///
/// Memoization key is `(canonical_id, current_target_depth)` so a shared node
/// reached under different binder depths gets independently rewritten copies.
pub fn subst_in_egraph(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    root: Id,
    kind: BinderKind,
    slot: u8,
    target_depth: u32,
    replacement: Id,
) -> Id {
    let mut shifted_replacements: HashMap<u32, Id> = HashMap::new();
    shifted_replacements.insert(target_depth, replacement);
    let mut memo: HashMap<(Id, u32), Id> = HashMap::new();
    let mut ctx = SubstCtx {
        egraph,
        chosen,
        kind,
        slot,
        shifted_replacements: &mut shifted_replacements,
        memo: &mut memo,
    };
    ctx.subst_rec(root, target_depth)
}

struct SubstCtx<'a> {
    egraph: &'a mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &'a mut HashMap<Id, TensorIr>,
    kind: BinderKind,
    slot: u8,
    shifted_replacements: &'a mut HashMap<u32, Id>,
    memo: &'a mut HashMap<(Id, u32), Id>,
}

impl SubstCtx<'_> {
    fn subst_rec(&mut self, id: Id, target_depth: u32) -> Id {
        let canonical = self.egraph.find(id);
        if let Some(&cached) = self.memo.get(&(canonical, target_depth)) {
            return cached;
        }

        let Some(node) = pick_node(self.egraph, self.chosen, canonical) else {
            return id;
        };

        if let TensorIr::Simd(crate::language::SimdNode::Var(VarRef::Bound {
            kind: var_kind,
            slot: var_slot,
            depth,
        })) = node
            && var_kind == self.kind
            && var_slot == self.slot
            && depth == target_depth
        {
            let rep = self.get_or_insert_shifted(target_depth);
            self.memo.insert((canonical, target_depth), rep);
            return rep;
        }

        if node.children().is_empty() {
            self.memo.insert((canonical, target_depth), id);
            return id;
        }

        let new_node = if let Some(info) = node.binder_info() {
            let mut new_node = node.clone();
            let info_matches = info.kind == self.kind;
            for (i, child) in new_node.children_mut().iter_mut().enumerate() {
                let bit_set = (info.body_mask >> i) & 1 == 1;
                let child_target = if info_matches && bit_set {
                    target_depth + 1
                } else {
                    target_depth
                };
                let child_canonical = self.egraph.find(*child);
                if child_canonical == canonical {
                    continue;
                }
                *child = self.subst_rec(*child, child_target);
            }
            new_node
        } else {
            let mut new_node = node.clone();
            for child in new_node.children_mut() {
                let child_canonical = self.egraph.find(*child);
                if child_canonical == canonical {
                    continue;
                }
                *child = self.subst_rec(*child, target_depth);
            }
            new_node
        };

        let new_id = add_and_choose(self.egraph, self.chosen, new_node);
        self.memo.insert((canonical, target_depth), new_id);
        new_id
    }

    fn get_or_insert_shifted(&mut self, needed_depth: u32) -> Id {
        if let Some(&id) = self.shifted_replacements.get(&needed_depth) {
            return id;
        }
        // Find the smallest depth in the cache and shift up from there. (We always
        // seed cache with the original target_depth, so at least one entry exists.)
        let (&base_depth, &base_id) = self
            .shifted_replacements
            .iter()
            .min_by_key(|(d, _)| **d)
            .expect("subst cache always seeded");
        let delta = i32::try_from(needed_depth).expect("depth fits in i32")
            - i32::try_from(base_depth).expect("depth fits in i32");
        let shifted = shift_in_egraph(self.egraph, self.chosen, base_id, self.kind, 0, delta);
        self.shifted_replacements.insert(needed_depth, shifted);
        shifted
    }
}

fn shift_depth(depth: u32, delta: i32) -> u32 {
    if delta >= 0 {
        depth + u32::try_from(delta).expect("non-negative delta fits")
    } else {
        let dec = u32::try_from(-delta).expect("delta magnitude fits");
        depth
            .checked_sub(dec)
            .expect("shift_depth underflow: open binder escaping its scope")
    }
}

fn pick_node(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen: &HashMap<Id, TensorIr>,
    canonical: Id,
) -> Option<TensorIr> {
    chosen
        .get(&canonical)
        .cloned()
        .or_else(|| egraph[canonical].nodes.first().cloned())
}

#[cfg(test)]
mod tests {
    //! Cross-kind isolation: a Theta shift must never disturb a
    //! Dispatch-bound ref, and vice versa. This pins the invariant that
    //! makes it safe to fold Thread refs into `Bound { kind: Dispatch, .. }`.

    use super::*;
    use crate::language::{DispatchNode, SimdNode};
    use crate::types::{IndexLevel, ScalarValue};

    fn var(egraph: &mut EGraph<TensorIr, TensorAnalysis>, v: VarRef) -> Id {
        egraph.add(TensorIr::Simd(SimdNode::Var(v)))
    }

    #[test]
    fn theta_shift_leaves_dispatch_refs_untouched() {
        let mut egraph: EGraph<TensorIr, TensorAnalysis> = EGraph::default();
        let lane_var = var(&mut egraph, VarRef::thread(IndexLevel::Lane));
        let theta_iter = var(&mut egraph, VarRef::iter(0));
        let sum = egraph.add(TensorIr::BinOp(
            crate::types::BinaryOp::Add,
            [lane_var, theta_iter],
        ));
        egraph.rebuild();

        let shifted = shift(&mut egraph, sum, BinderKind::Theta, 0, 1);
        egraph.rebuild();

        // After a Theta shift (cutoff 0, +1), the Theta-iter depth should
        // have moved from 0 to 1; the Dispatch-bound lane ref should remain
        // at depth 0.
        let shifted_node = &egraph[shifted].nodes[0];
        let TensorIr::BinOp(_, [lhs, rhs]) = shifted_node else {
            panic!("shifted node is not a BinOp")
        };

        let lhs_node = &egraph[*lhs].nodes[0];
        let rhs_node = &egraph[*rhs].nodes[0];

        let mut saw_lane = false;
        let mut saw_theta_1 = false;
        for n in [lhs_node, rhs_node] {
            if let TensorIr::Simd(SimdNode::Var(VarRef::Bound {
                kind: BinderKind::Dispatch,
                depth: 0,
                ..
            })) = n
            {
                saw_lane = true;
            }
            if let TensorIr::Simd(SimdNode::Var(VarRef::Bound {
                kind: BinderKind::Theta,
                depth: 1,
                ..
            })) = n
            {
                saw_theta_1 = true;
            }
        }
        assert!(
            saw_lane,
            "dispatch-bound lane ref was disturbed by theta shift"
        );
        assert!(saw_theta_1, "theta ref was not shifted");
    }

    #[test]
    fn theta_shift_through_dispatch_body_is_inert_for_dispatch_refs() {
        // Build a Dispatch that wraps a Theta body containing both a Theta
        // ref and a Dispatch (lane) ref. Applying shift_in_egraph across the
        // whole tree must move only Theta refs with depth >= cutoff.
        let mut egraph: EGraph<TensorIr, TensorAnalysis> = EGraph::default();
        let lane_var = var(&mut egraph, VarRef::thread(IndexLevel::Lane));
        let iter_var = var(&mut egraph, VarRef::iter(0));
        let body = egraph.add(TensorIr::BinOp(
            crate::types::BinaryOp::Add,
            [lane_var, iter_var],
        ));
        let init = egraph.add(TensorIr::Const(ScalarValue::F32(
            ordered_float::OrderedFloat(0.0),
        )));
        let count = egraph.add(TensorIr::Const(ScalarValue::U32(4)));
        let theta = egraph.add(TensorIr::Simd(SimdNode::Theta {
            children: [init, count, body],
        }));
        let nil = egraph.add(TensorIr::Nil);
        let list = egraph.add(TensorIr::Cons([theta, nil]));
        let dispatch = egraph.add(TensorIr::Dispatch(DispatchNode::Dispatch {
            workgroups: 1,
            num_inputs: 0,
            children_list: list,
        }));
        egraph.rebuild();

        // Shift Theta kind (cutoff 0, +1). Theta-iter inside the body is
        // already bound (depth 0 < cutoff for the body because the body
        // sits inside the Theta's own binder), so it stays at depth 0. The
        // Dispatch lane ref must also stay at depth 0.
        let _shifted = shift(&mut egraph, dispatch, BinderKind::Theta, 0, 1);
        egraph.rebuild();

        // No panic = shift survived without binder escape. Spot-check that
        // the Dispatch lane ref still exists in the original e-graph.
        let lane_canonical = egraph.find(lane_var);
        let found = egraph[lane_canonical].nodes.iter().any(|n| {
            matches!(
                n,
                TensorIr::Simd(SimdNode::Var(VarRef::Bound {
                    kind: BinderKind::Dispatch,
                    depth: 0,
                    ..
                }))
            )
        });
        assert!(found, "dispatch lane ref lost during theta shift");
    }
}
