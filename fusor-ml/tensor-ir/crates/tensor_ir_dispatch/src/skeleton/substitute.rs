//! Substitute / select / range-estimation helpers.

use std::collections::{HashMap, HashSet};

use egg::{EGraph, Id, Language};

use crate::analysis::TensorAnalysis;
use crate::binding;
use crate::language::{SimdNode, TensorIr};
use crate::types::{
    BinaryOp, BinderKind, DeviceProfile, IndexLevel, MemTier, ScalarValue, VarRef,
    index_level_from_slot, slots,
};

use super::*;
use super::{TgBufferInfo, add_and_choose};

/// Compute the thread range (number of unique lane-derived values).
pub(super) fn compute_thread_range_from_dep(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen: &HashMap<Id, TensorIr>,
    expr: Id,
    device: &DeviceProfile,
) -> u32 {
    fn rec(
        egraph: &EGraph<TensorIr, TensorAnalysis>,
        chosen: &HashMap<Id, TensorIr>,
        expr: Id,
        simd_width: u32,
        memo: &mut HashMap<Id, u32>,
        visiting: &mut HashSet<Id>,
    ) -> u32 {
        let canonical = egraph.find(expr);
        if let Some(&cached) = memo.get(&canonical) {
            return cached;
        }
        if !visiting.insert(canonical) {
            return simd_width;
        }

        let candidates: Vec<TensorIr> = chosen.get(&canonical).cloned().map_or_else(
            || egraph[canonical].iter().cloned().collect(),
            |node| {
                let mut all: Vec<TensorIr> = egraph[canonical].iter().cloned().collect();
                if !all.iter().any(|candidate| candidate == &node) {
                    all.push(node);
                }
                all
            },
        );

        let mut best_range: Option<u32> = None;
        for node in &candidates {
            let candidate_range = match node {
                TensorIr::Simd(SimdNode::Var(VarRef::Bound {
                    kind: BinderKind::Dispatch,
                    slot: slots::DISPATCH_LANE,
                    depth: 0,
                })) => Some(simd_width),
                TensorIr::BinOp(name, args) => match name {
                    BinaryOp::Div => {
                        if let Some(ScalarValue::U32(c)) = &egraph[args[1]].data.constant {
                            if *c > 0 {
                                let numerator_range =
                                    rec(egraph, chosen, args[0], simd_width, memo, visiting);
                                Some(numerator_range.div_ceil(*c))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    }
                    BinaryOp::Mod => {
                        if let Some(ScalarValue::U32(c)) = &egraph[args[1]].data.constant {
                            Some((*c).min(rec(egraph, chosen, args[0], simd_width, memo, visiting)))
                        } else {
                            None
                        }
                    }
                    BinaryOp::Mul | BinaryOp::Sub | BinaryOp::Add => {
                        let left_dep = egraph[args[0]].data.dep;
                        let right_dep = egraph[args[1]].data.dep;
                        if left_dep.contains_lane() && !right_dep.contains_lane() {
                            Some(rec(egraph, chosen, args[0], simd_width, memo, visiting))
                        } else if right_dep.contains_lane() && !left_dep.contains_lane() {
                            Some(rec(egraph, chosen, args[1], simd_width, memo, visiting))
                        } else {
                            None
                        }
                    }
                    _ => None,
                },
                _ => None,
            };

            if let Some(range) = candidate_range.filter(|range| *range > 0) {
                best_range = Some(best_range.map_or(range, |best| best.min(range)));
            }
        }

        visiting.remove(&canonical);
        let result = best_range.unwrap_or(simd_width);
        memo.insert(canonical, result);
        result
    }

    let mut memo = HashMap::new();
    let mut visiting = HashSet::new();
    rec(
        egraph,
        chosen,
        expr,
        device.simd_width,
        &mut memo,
        &mut visiting,
    )
}

/// Substitute an Index level (e.g., Lane) with 0 in an expression.
/// Returns a new expression with the substitution applied.
pub(super) fn substitute_index_with_zero(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    expr: Id,
    level: IndexLevel,
) -> Id {
    let mut memo: HashMap<Id, Id> = HashMap::new();
    substitute_index_zero_rec(egraph, chosen, expr, level, &mut memo)
}

pub(super) fn substitute_index_zero_rec(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    id: Id,
    level: IndexLevel,
    memo: &mut HashMap<Id, Id>,
) -> Id {
    let canonical = egraph.find(id);
    if let Some(&cached) = memo.get(&canonical) {
        return cached;
    }

    let Some(node) = select_substitution_node(egraph, chosen, canonical) else {
        return id;
    };

    // Replace the target Index with 0
    if let TensorIr::Simd(SimdNode::Var(VarRef::Bound {
        kind: BinderKind::Dispatch,
        slot,
        depth: 0,
    })) = &node
        && index_level_from_slot(*slot) == Some(level)
    {
        let zero = add_and_choose(egraph, chosen, TensorIr::Const(ScalarValue::U32(0)));
        memo.insert(canonical, zero);
        return zero;
    }

    if node.children().is_empty() {
        memo.insert(canonical, id);
        return id;
    }

    let mut new_node = node;
    for child in new_node.children_mut() {
        let child_canonical = egraph.find(*child);
        if child_canonical == canonical {
            continue;
        }
        *child = substitute_index_zero_rec(egraph, chosen, *child, level, memo);
    }

    let new_id = add_and_choose(egraph, chosen, new_node);
    memo.insert(canonical, new_id);
    new_id
}

/// Substitute a named variable with `0u` inside an expression.
///
/// Used to make cooperative preload base addresses invariant with respect to
/// inner theta induction variables like `Bound { Iter, 0 }`.
pub(super) fn substitute_var_with_zero(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    expr: Id,
    var: VarRef,
) -> Id {
    let zero = add_and_choose(egraph, chosen, TensorIr::Const(ScalarValue::U32(0)));
    substitute_var_with_id(egraph, chosen, expr, var, zero)
}

pub(super) fn substitute_var_with_id(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    expr: Id,
    var: VarRef,
    replacement: Id,
) -> Id {
    match var {
        // Bound vars use the scope-aware De Bruijn substitution from binding.
        VarRef::Bound { kind, slot, depth } => {
            binding::subst_in_egraph(egraph, chosen, expr, kind, slot, depth, replacement)
        }
    }
}

pub(super) fn select_substitution_node(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen: &HashMap<Id, TensorIr>,
    canonical: Id,
) -> Option<TensorIr> {
    let preferred = chosen
        .get(&canonical)
        .cloned()
        .or_else(|| egraph[canonical].iter().next().cloned())?;

    let has_self_ref = !matches!(preferred, TensorIr::Simd(SimdNode::Theta { .. }))
        && preferred
            .children()
            .iter()
            .any(|child| egraph.find(*child) == canonical);
    if !has_self_ref {
        return Some(preferred);
    }

    egraph[canonical]
        .iter()
        .find(|node| {
            !matches!(node, TensorIr::Simd(SimdNode::Theta { .. }))
                && !node
                    .children()
                    .iter()
                    .any(|child| egraph.find(*child) == canonical)
        })
        .cloned()
        .or(Some(preferred))
}

pub(super) fn selected_subtree_is_k_only(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen: &HashMap<Id, TensorIr>,
    root: Id,
    k_var: VarRef,
    k_outer_var: VarRef,
) -> bool {
    let mut visited = HashSet::new();
    selected_subtree_is_k_only_rec(egraph, chosen, root, k_var, k_outer_var, &mut visited)
}

pub(super) fn selected_subtree_is_k_only_rec(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen: &HashMap<Id, TensorIr>,
    id: Id,
    k_var: VarRef,
    k_outer_var: VarRef,
    visited: &mut HashSet<Id>,
) -> bool {
    let canonical = egraph.find(id);
    if !visited.insert(canonical) {
        return true;
    }

    let Some(node) = select_substitution_node(egraph, chosen, canonical) else {
        return false;
    };

    match node {
        TensorIr::Simd(SimdNode::Var(name)) => name == k_var || name == k_outer_var,
        TensorIr::Const(_) => true,
        TensorIr::BinOp(BinaryOp::Div, args) if const_u32(egraph, args[1]) == Some(1) => {
            selected_subtree_is_k_only_rec(egraph, chosen, args[0], k_var, k_outer_var, visited)
        }
        TensorIr::BinOp(BinaryOp::Mod, args) if const_u32(egraph, args[1]) == Some(1) => true,
        TensorIr::BinOp(_, args) => args.iter().all(|child| {
            selected_subtree_is_k_only_rec(egraph, chosen, *child, k_var, k_outer_var, visited)
        }),
        _ => false,
    }
}

pub(super) fn selected_subtree_has_coherent_k_tile_stride(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    root: Id,
    theta_class: Id,
    tile_k: u32,
) -> bool {
    let mut visited = HashSet::new();
    selected_subtree_has_coherent_k_tile_stride_rec(
        egraph,
        chosen,
        root,
        theta_class,
        tile_k,
        &mut visited,
    )
}

pub(super) fn selected_subtree_has_coherent_k_tile_stride_rec(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    id: Id,
    theta_class: Id,
    tile_k: u32,
    visited: &mut HashSet<Id>,
) -> bool {
    let canonical = egraph.find(id);
    if canonical == egraph.find(theta_class) && !visited.is_empty() {
        return true;
    }
    if !visited.insert(canonical) {
        return true;
    }

    let Some(node) = select_substitution_node(egraph, chosen, canonical) else {
        return false;
    };

    if let TensorIr::Simd(SimdNode::Load {
        tier: MemTier::Device(_),
        children,
    }) = &node
    {
        let addr = children[0];
        let addr_canonical = egraph.find(addr);
        if egraph[addr_canonical]
            .data
            .var_dep
            .contains(&VarRef::iter(1))
            && !device_addr_has_coherent_k_tile_stride(egraph, chosen, addr, tile_k)
        {
            return false;
        }
    }

    for child in node.children() {
        if !selected_subtree_has_coherent_k_tile_stride_rec(
            egraph,
            chosen,
            *child,
            theta_class,
            tile_k,
            visited,
        ) {
            return false;
        }
    }

    true
}

fn device_addr_has_coherent_k_tile_stride(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    addr: Id,
    tile_k: u32,
) -> bool {
    fn rec(
        egraph: &EGraph<TensorIr, TensorAnalysis>,
        chosen: &mut HashMap<Id, TensorIr>,
        addr: Id,
        tile_k: u32,
        visited: &mut HashSet<Id>,
    ) -> bool {
        let canonical = egraph.find(addr);
        if !visited.insert(canonical) {
            return false;
        }

        let candidates: Vec<TensorIr> = chosen.get(&canonical).cloned().map_or_else(
            || egraph[canonical].iter().cloned().collect(),
            |node| vec![node],
        );
        for node in &candidates {
            if let Some(inner_addr) = divmod_recomposed_value(egraph, node)
                && egraph.find(inner_addr) != canonical
                && rec(egraph, chosen, inner_addr, tile_k, visited)
            {
                return true;
            }
        }

        let Some(base) = extract_additive_base(egraph, chosen, addr, VarRef::iter(1)) else {
            return false;
        };
        matches_k_outer_stride(egraph, chosen, base, tile_k)
    }

    rec(egraph, chosen, addr, tile_k, &mut HashSet::new())
}

fn divmod_recomposed_value(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    node: &TensorIr,
) -> Option<Id> {
    let TensorIr::BinOp(BinaryOp::Add, args) = node else {
        return None;
    };
    for (div_scaled, modulo) in [(args[0], args[1]), (args[1], args[0])] {
        let Some((value, divisor)) = scaled_division(egraph, div_scaled) else {
            continue;
        };
        if modulo_matches(egraph, modulo, value, divisor) {
            return Some(value);
        }
    }
    None
}

fn scaled_division(egraph: &EGraph<TensorIr, TensorAnalysis>, id: Id) -> Option<(Id, u32)> {
    for node in egraph[egraph.find(id)].iter() {
        let TensorIr::BinOp(BinaryOp::Mul, args) = node else {
            continue;
        };
        for (div_side, const_side) in [(args[0], args[1]), (args[1], args[0])] {
            let Some(divisor) = const_u32(egraph, const_side).filter(|v| *v > 0) else {
                continue;
            };
            let Some(value) = division_by(egraph, div_side, divisor) else {
                continue;
            };
            return Some((value, divisor));
        }
    }
    None
}

fn division_by(egraph: &EGraph<TensorIr, TensorAnalysis>, id: Id, divisor: u32) -> Option<Id> {
    for node in egraph[egraph.find(id)].iter() {
        let TensorIr::BinOp(BinaryOp::Div, args) = node else {
            continue;
        };
        if const_u32(egraph, args[1]) == Some(divisor) {
            return Some(args[0]);
        }
    }
    None
}

fn modulo_matches(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    id: Id,
    value: Id,
    divisor: u32,
) -> bool {
    egraph[egraph.find(id)].iter().any(|node| {
        let TensorIr::BinOp(BinaryOp::Mod, args) = node else {
            return false;
        };
        egraph.find(args[0]) == egraph.find(value) && const_u32(egraph, args[1]) == Some(divisor)
    })
}

fn const_u32(egraph: &EGraph<TensorIr, TensorAnalysis>, id: Id) -> Option<u32> {
    match &egraph[egraph.find(id)].data.constant {
        Some(ScalarValue::U32(v)) => Some(*v),
        _ => None,
    }
}

pub(super) fn matches_k_outer_stride(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen: &HashMap<Id, TensorIr>,
    expr: Id,
    tile_k: u32,
) -> bool {
    let canonical = egraph.find(expr);
    let candidates: Vec<TensorIr> = chosen.get(&canonical).cloned().map_or_else(
        || egraph[canonical].iter().cloned().collect(),
        |node| vec![node],
    );

    for node in &candidates {
        if let TensorIr::BinOp(name, args) = node
            && matches!(name, BinaryOp::Mul)
            && args.len() == 2
        {
            if let Some(ScalarValue::U32(v)) = &egraph[args[0]].data.constant
                && const_is_tile_stride_multiple(*v, tile_k)
            {
                return true;
            }
            if let Some(ScalarValue::U32(v)) = &egraph[args[1]].data.constant
                && const_is_tile_stride_multiple(*v, tile_k)
            {
                return true;
            }
        }
        if let TensorIr::Simd(SimdNode::Var(name)) = node
            && *name == VarRef::iter(1)
            && tile_k == 1
        {
            return true;
        }
    }

    false
}

const fn const_is_tile_stride_multiple(value: u32, tile_k: u32) -> bool {
    tile_k > 0 && value > 0 && value.is_multiple_of(tile_k)
}

/// Check if an e-graph subtree contains threadgroup loads.
pub(super) fn egraph_subtree_has_tg_loads(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    root: Id,
    theta_class: Id,
) -> bool {
    let mut visited = std::collections::HashSet::new();
    egraph_subtree_has_tg_loads_rec(egraph, root, theta_class, &mut visited)
}

pub(super) fn egraph_subtree_has_tg_loads_rec(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    id: Id,
    theta_class: Id,
    visited: &mut std::collections::HashSet<Id>,
) -> bool {
    let canonical = egraph.find(id);
    if canonical == egraph.find(theta_class) {
        return false;
    }
    if !visited.insert(canonical) {
        return false;
    }
    for node in egraph[canonical].iter() {
        if matches!(
            node,
            TensorIr::Simd(SimdNode::Load {
                tier: MemTier::Threadgroup(_),
                ..
            })
        ) {
            return true;
        }
        for child in node.children() {
            if egraph_subtree_has_tg_loads_rec(egraph, *child, theta_class, visited) {
                return true;
            }
        }
    }
    false
}

/// Extract the row stride from a tg address expression in the e-graph.
///
/// The tg address pattern is `add(mul(row_expr, stride_lit), col_expr)`.
/// Returns the stride literal, or None if the pattern doesn't match.
pub(super) fn extract_tg_addr_stride(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen: &HashMap<Id, TensorIr>,
    id: Id,
) -> Option<u32> {
    let canonical = egraph.find(id);
    let node = select_substitution_node(egraph, chosen, canonical)
        .unwrap_or_else(|| egraph[canonical].iter().next().unwrap().clone());
    if let TensorIr::BinOp(name, args) = &node
        && matches!(name, BinaryOp::Add)
        && args.len() == 2
    {
        // Check left child for mul(_, const)
        if let Some(stride) = extract_mul_constant_egraph(egraph, chosen, args[0]) {
            return Some(stride);
        }
        // Check right child for mul(_, const)
        if let Some(stride) = extract_mul_constant_egraph(egraph, chosen, args[1]) {
            return Some(stride);
        }
    }
    None
}

/// Extract the constant from a mul(expr, const) or mul(const, expr) pattern.
pub(super) fn extract_mul_constant_egraph(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen: &HashMap<Id, TensorIr>,
    id: Id,
) -> Option<u32> {
    let canonical = egraph.find(id);
    let node = select_substitution_node(egraph, chosen, canonical)
        .unwrap_or_else(|| egraph[canonical].iter().next().unwrap().clone());
    if let TensorIr::BinOp(name, args) = &node
        && matches!(name, BinaryOp::Mul)
        && args.len() == 2
    {
        if let Some(v) = egraph[egraph.find(args[1])].data.constant.as_ref()
            && let ScalarValue::U32(c) = v
        {
            return Some(*c);
        }
        if let Some(v) = egraph[egraph.find(args[0])].data.constant.as_ref()
            && let ScalarValue::U32(c) = v
        {
            return Some(*c);
        }
    }
    None
}

/// Closed interval `[lo, hi]` for bound analysis over u32 arithmetic.
#[derive(Clone, Copy, Debug)]
pub(super) struct AddrInterval {
    pub lo: u32,
    pub hi: u32,
}

impl AddrInterval {
    pub(super) const fn point(v: u32) -> Self {
        Self { lo: v, hi: v }
    }

    pub(super) const fn range(lo: u32, hi: u32) -> Self {
        Self { lo, hi }
    }

    /// Widest interval representable — the analysis uses this for subtrees
    /// it can't reason about (unsupported opcode, recursion cycle). Any
    /// bound comparison against this is trivially satisfied, so the
    /// caller's "reject on provable OOB" gate treats it as "don't know".
    pub(super) const fn unknown() -> Self {
        Self {
            lo: 0,
            hi: u32::MAX,
        }
    }

    pub(super) fn add(&self, other: &Self) -> Self {
        Self {
            lo: self.lo.saturating_add(other.lo),
            hi: self.hi.saturating_add(other.hi),
        }
    }

    /// `[a - b_hi, a_hi - b_lo]`, saturating at zero. This is the standard
    /// interval rule: the smallest result occurs at the smallest
    /// minuend minus the largest subtrahend. Note that when `a` and `b`
    /// are *correlated* (e.g. both derive from the same Var), interval
    /// arithmetic loses precision — for the phase-3 rewrite pattern
    /// `Sub(Add(X, Y), X)`, exact cancellation is handled by
    /// `simplify_sub_to_remainder`, which the `Sub` branch consults
    /// *before* falling back to interval subtraction.
    pub(super) fn sub(&self, other: &Self) -> Self {
        Self {
            lo: self.lo.saturating_sub(other.hi),
            hi: self.hi.saturating_sub(other.lo),
        }
    }

    pub(super) fn mul(&self, other: &Self) -> Self {
        Self {
            lo: self.lo.saturating_mul(other.lo),
            hi: self.hi.saturating_mul(other.hi),
        }
    }

    /// Integer division interval. Division by a range containing zero is
    /// undefined, so we widen to `[lo/hi, hi]` in that case; the caller's
    /// OOB gate is monotone in `hi`, so this stays sound.
    pub(super) fn div(&self, other: &Self) -> Self {
        let denom_hi = other.hi.max(1);
        let denom_lo = other.lo.max(1);
        Self {
            lo: self.lo / denom_hi,
            hi: self.hi / denom_lo,
        }
    }

    /// `Mod(a, b)` ranges from 0 up to `min(a.hi, b.hi - 1)`. If the
    /// divisor interval may include 0 we widen to `a.hi` — that matches
    /// runtime WGSL semantics where `a % 0` is implementation-defined but
    /// never exceeds `a`.
    pub(super) fn modu(&self, other: &Self) -> Self {
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

/// Match `Sub(Add(x, y), x)` / `Sub(Add(x, y), y)` and return the remaining
/// operand's id. Phase-3's tile-local address rewrite emits this pattern
/// everywhere (subtracting the grid base from `Add(grid_base, tile_local)`);
/// catching it syntactically gives an *exact* tight interval instead of
/// the looser `[a_lo - b_hi, a_hi - b_lo]` we'd get from treating the two
/// copies of `X` as independent intervals.
fn simplify_sub_to_remainder(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen: &HashMap<Id, TensorIr>,
    lhs: Id,
    rhs: Id,
) -> Option<Id> {
    let lhs_canonical = egraph.find(lhs);
    let rhs_canonical = egraph.find(rhs);
    if lhs_canonical == rhs_canonical {
        return None; // Sub(X, X) = 0, but the caller handles this via interval sub.
    }
    let lhs_candidates: Vec<TensorIr> = chosen.get(&lhs_canonical).cloned().map_or_else(
        || egraph[lhs_canonical].iter().cloned().collect(),
        |node| vec![node],
    );
    for node in lhs_candidates {
        if let TensorIr::BinOp(BinaryOp::Add, args) = node {
            let a = egraph.find(args[0]);
            let b = egraph.find(args[1]);
            if a == rhs_canonical {
                return Some(b);
            }
            if b == rhs_canonical {
                return Some(a);
            }
        }
    }
    None
}

/// Compute the max value of an address expression. Retained for external
/// callers that only care about the upper bound (e.g. auto-sizing a TG
/// buffer whose size is unknown up front).
pub(super) fn compute_max_addr_from_egraph(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen: &HashMap<Id, TensorIr>,
    id: Id,
    tile_k: u32,
    device: &DeviceProfile,
) -> u32 {
    compute_addr_interval_bounded(egraph, chosen, id, tile_k, device, 0, 0).hi
}

/// Bounded interval analysis for an address expression. `workgroup_bound`
/// and `simdgroup_bound` give the *intra-physical-workgroup* extent of the
/// corresponding thread vars after simdgroup promotion. Used to validate a
/// TG Load's tile-local address fits inside its buffer without running the
/// kernel.
pub(super) fn compute_addr_interval_bounded(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen: &HashMap<Id, TensorIr>,
    id: Id,
    tile_k: u32,
    device: &DeviceProfile,
    workgroup_bound: u32,
    simdgroup_bound: u32,
) -> AddrInterval {
    compute_addr_interval_rec(
        egraph,
        chosen,
        id,
        tile_k,
        device,
        workgroup_bound,
        simdgroup_bound,
        &mut HashSet::new(),
    )
}

#[allow(clippy::too_many_arguments)]
fn compute_addr_interval_rec(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen: &HashMap<Id, TensorIr>,
    id: Id,
    tile_k: u32,
    device: &DeviceProfile,
    workgroup_bound: u32,
    simdgroup_bound: u32,
    visiting: &mut HashSet<Id>,
) -> AddrInterval {
    let canonical = egraph.find(id);
    if !visiting.insert(canonical) {
        // Cycle — defer to the enclosing class's bound.
        return AddrInterval::unknown();
    }
    let candidates: Vec<TensorIr> = chosen.get(&canonical).cloned().map_or_else(
        || egraph[canonical].iter().cloned().collect(),
        |node| vec![node],
    );

    let mut result = AddrInterval::unknown();
    for node in &candidates {
        match node {
            TensorIr::Const(ScalarValue::U32(v)) => {
                result = AddrInterval::point(*v);
                break;
            }
            TensorIr::Simd(SimdNode::Var(VarRef::Bound {
                kind: BinderKind::Dispatch,
                slot: slots::DISPATCH_LANE,
                depth: 0,
            })) => {
                result = AddrInterval::range(0, device.simd_width - 1);
                break;
            }
            TensorIr::Simd(SimdNode::Var(VarRef::Bound {
                kind: BinderKind::Dispatch,
                slot: slots::DISPATCH_WORKGROUP,
                depth: 0,
            })) => {
                result = AddrInterval::range(0, workgroup_bound);
                break;
            }
            TensorIr::Simd(SimdNode::Var(VarRef::Bound {
                kind: BinderKind::Dispatch,
                slot: slots::DISPATCH_SIMDGROUP,
                depth: 0,
            })) => {
                result = AddrInterval::range(0, simdgroup_bound);
                break;
            }
            TensorIr::Simd(SimdNode::Var(_)) => {
                // Theta iterators (k_inner / k_outer / blocked-accumulator
                // indices) — bounded by the enclosing tile extent.
                result = AddrInterval::range(0, tile_k.saturating_sub(1));
                break;
            }
            TensorIr::BinOp(name, args) => {
                let fold = |lhs: Id,
                            rhs: Id,
                            visiting: &mut HashSet<Id>|
                 -> (AddrInterval, AddrInterval) {
                    let a = compute_addr_interval_rec(
                        egraph,
                        chosen,
                        lhs,
                        tile_k,
                        device,
                        workgroup_bound,
                        simdgroup_bound,
                        visiting,
                    );
                    let b = compute_addr_interval_rec(
                        egraph,
                        chosen,
                        rhs,
                        tile_k,
                        device,
                        workgroup_bound,
                        simdgroup_bound,
                        visiting,
                    );
                    (a, b)
                };
                match name {
                    BinaryOp::Add => {
                        let (a, b) = fold(args[0], args[1], visiting);
                        result = a.add(&b);
                        break;
                    }
                    BinaryOp::Mul => {
                        let (a, b) = fold(args[0], args[1], visiting);
                        result = a.mul(&b);
                        break;
                    }
                    BinaryOp::Sub => {
                        // Correlated-subtraction fast path: phase-3's
                        // rewrite emits `Sub(Add(X, Y), X) → Y`. Recognizing
                        // it syntactically recovers a tight interval that
                        // generic interval subtraction can't.
                        if let Some(simplified) =
                            simplify_sub_to_remainder(egraph, chosen, args[0], args[1])
                        {
                            result = compute_addr_interval_rec(
                                egraph,
                                chosen,
                                simplified,
                                tile_k,
                                device,
                                workgroup_bound,
                                simdgroup_bound,
                                visiting,
                            );
                        } else {
                            let (a, b) = fold(args[0], args[1], visiting);
                            result = a.sub(&b);
                        }
                        break;
                    }
                    BinaryOp::Div => {
                        let (a, b) = fold(args[0], args[1], visiting);
                        result = a.div(&b);
                        break;
                    }
                    BinaryOp::Mod => {
                        let (a, b) = fold(args[0], args[1], visiting);
                        result = a.modu(&b);
                        break;
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
    visiting.remove(&canonical);
    result
}

/// Recursively add a subtree from a `RecExpr` into the e-graph,
/// recording the extractor's chosen node for each canonical Id.
pub(super) fn add_recexpr_subtree(
    nodes: &[TensorIr],
    idx: usize,
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
) -> Id {
    let node = &nodes[idx];
    let mut new_node = node.clone();

    // Recursively add children first
    for child in new_node.children_mut() {
        let child_idx = usize::from(*child);
        *child = add_recexpr_subtree(nodes, child_idx, egraph, chosen);
    }

    let id = egraph.add(new_node.clone());
    let canonical = egraph.find(id);
    match chosen.get(&canonical) {
        Some(TensorIr::Simd(SimdNode::Load {
            tier: MemTier::Threadgroup(_),
            ..
        })) => {}
        Some(_) => {
            let prefer_new = matches!(
                new_node,
                TensorIr::Simd(SimdNode::Load {
                    tier: MemTier::Threadgroup(_),
                    ..
                })
            );
            if prefer_new {
                chosen.insert(canonical, new_node);
            }
        }
        None => {
            chosen.insert(canonical, new_node);
        }
    }
    id
}

/// Check if any node in the subtree is a threadgroup Load.
pub(super) fn subtree_has_tg_loads(nodes: &[TensorIr], idx: usize) -> bool {
    let node = &nodes[idx];
    if matches!(
        node,
        TensorIr::Simd(SimdNode::Load {
            tier: MemTier::Threadgroup(_),
            ..
        })
    ) {
        return true;
    }
    for child_id in node.children() {
        if subtree_has_tg_loads(nodes, usize::from(*child_id)) {
            return true;
        }
    }
    false
}

/// Get a u32 literal value from a `RecExpr` node.
pub(super) fn get_u32_lit(nodes: &[TensorIr], idx: usize) -> Option<u32> {
    if let TensorIr::Const(crate::types::ScalarValue::U32(v)) = &nodes[idx] {
        Some(*v)
    } else {
        None
    }
}

/// Collect `TgBufferInfo` from threadgroup loads in a `RecExpr` subtree.
///
/// Generically computes `region_size` from the tile-local address expression
/// by analyzing the `add(mul(row, stride), col)` pattern.
#[must_use]
pub fn collect_tg_buffer_info(
    nodes: &[TensorIr],
    idx: usize,
    inner_count: u32,
    device: &DeviceProfile,
) -> Vec<TgBufferInfo> {
    let mut results = Vec::new();
    let mut visited = HashSet::new();
    collect_tg_info_rec(nodes, idx, inner_count, device, &mut results, &mut visited);
    results
}

pub(super) fn collect_tg_info_rec(
    nodes: &[TensorIr],
    idx: usize,
    inner_count: u32,
    device: &DeviceProfile,
    results: &mut Vec<TgBufferInfo>,
    visited: &mut HashSet<usize>,
) {
    if !visited.insert(idx) {
        return;
    }
    let node = &nodes[idx];

    if let TensorIr::Simd(SimdNode::Load {
        tier: MemTier::Threadgroup(tg_name),
        children,
    }) = node
    {
        // Device-tier counterpart of a tg buffer is the same `BufferRef`.
        let dev_name = *tg_name;

        // Compute region_size from the tile-local address expression.
        let addr_idx = usize::from(children[0]);
        let size = compute_region_size_from_tile_addr(nodes, addr_idx, inner_count, device);

        if !results.iter().any(|b| b.tg_name == *tg_name) {
            results.push(TgBufferInfo {
                tg_name: *tg_name,
                device_name: dev_name,
                size,
                dtype_bytes: dtype_bytes_for_device_buffer_in_recexpr(nodes, dev_name),
                tile_cols: size,
                tile_rows: 1,
                device_row_base: None,
                device_col_base: None,
                device_row_stride: 0,
                sg_read_stride: 0,
            });
        }
    }

    for child_id in node.children() {
        collect_tg_info_rec(
            nodes,
            usize::from(*child_id),
            inner_count,
            device,
            results,
            visited,
        );
    }
}

/// Compute `region_size` from a tile-local address expression.
///
/// The tile-local address (built by Phase 3) has the pattern:
///   `add(mul(row_expr`, `stride_lit`), `col_expr`)
/// where `stride_lit` is the inner tile dimension. We find the stride constant
/// and estimate the row range to compute total region size.
pub(super) fn compute_region_size_from_tile_addr(
    nodes: &[TensorIr],
    addr_idx: usize,
    inner_count: u32,
    device: &DeviceProfile,
) -> u32 {
    if let TensorIr::BinOp(name, args) = &nodes[addr_idx]
        && matches!(name, BinaryOp::Add)
        && args.len() == 2
    {
        let left_idx = usize::from(args[0]);
        let right_idx = usize::from(args[1]);

        // Check if left is mul(_, const)
        if let Some(stride) = find_mul_constant(nodes, left_idx) {
            // Row range: estimate from the row expression
            let row_range = estimate_recexpr_var_range(nodes, left_idx, inner_count, device);
            return row_range * stride;
        }
        // Check right
        if let Some(stride) = find_mul_constant(nodes, right_idx) {
            let row_range = estimate_recexpr_var_range(nodes, right_idx, inner_count, device);
            return row_range * stride;
        }
    }

    // Fallback: use the largest constant in the address subtree
    let mut max_const: u32 = inner_count;
    find_max_constant_rec(nodes, addr_idx, &mut HashSet::new(), &mut max_const);
    max_const
}

/// Estimate the range (number of distinct values) of a `RecExpr` subtree.
///
/// `var_range` is used as the fallback for `Var` nodes. When called from
/// tiled promotion code, pass the inner Theta count (`tile_k`) so that the
/// loop counter variable _k gets the correct range. `device.simd_width` is
/// used as the range for the Lane index.
pub(super) fn estimate_recexpr_var_range(
    nodes: &[TensorIr],
    idx: usize,
    var_range: u32,
    device: &DeviceProfile,
) -> u32 {
    match &nodes[idx] {
        TensorIr::Simd(SimdNode::Var(VarRef::Bound {
            kind: BinderKind::Dispatch,
            slot: slots::DISPATCH_LANE,
            depth: 0,
        })) => {
            return device.simd_width;
        }
        TensorIr::Simd(SimdNode::Var(_)) => return var_range,
        TensorIr::BinOp(name, args) => match name {
            BinaryOp::Mul => {
                // For mul(expr, const), the range is in the non-const side
                let left_idx = usize::from(args[0]);
                let right_idx = usize::from(args[1]);
                if get_u32_lit(nodes, right_idx).is_some() {
                    return estimate_recexpr_var_range(nodes, left_idx, var_range, device);
                }
                if get_u32_lit(nodes, left_idx).is_some() {
                    return estimate_recexpr_var_range(nodes, right_idx, var_range, device);
                }
            }
            BinaryOp::Div => {
                if let Some(c) = get_u32_lit(nodes, usize::from(args[1]))
                    && c > 0
                {
                    let numerator_range =
                        estimate_recexpr_var_range(nodes, usize::from(args[0]), var_range, device);
                    return numerator_range / c;
                }
            }
            BinaryOp::Mod => {
                if let Some(c) = get_u32_lit(nodes, usize::from(args[1])) {
                    return c;
                }
            }
            BinaryOp::Add => {
                let left_idx = usize::from(args[0]);
                let right_idx = usize::from(args[1]);
                // For add(a, b), one side is typically a constant offset.
                // If one side is a constant, the range is the other side's range.
                if get_u32_lit(nodes, left_idx).is_some() {
                    return estimate_recexpr_var_range(nodes, right_idx, var_range, device);
                }
                if get_u32_lit(nodes, right_idx).is_some() {
                    return estimate_recexpr_var_range(nodes, left_idx, var_range, device);
                }
            }
            _ => {}
        },
        _ => {}
    }
    // Fallback
    var_range
}

/// Find a multiplication constant in an address expression.
pub(super) fn find_mul_constant(nodes: &[TensorIr], idx: usize) -> Option<u32> {
    if let TensorIr::BinOp(name, args) = &nodes[idx]
        && matches!(name, BinaryOp::Mul)
        && args.len() == 2
    {
        let left_idx = usize::from(args[0]);
        let right_idx = usize::from(args[1]);
        if let Some(v) = get_u32_lit(nodes, left_idx) {
            return Some(v);
        }
        if let Some(v) = get_u32_lit(nodes, right_idx) {
            return Some(v);
        }
    }
    None
}

/// Find the largest constant in a `RecExpr` subtree.
pub(super) fn find_max_constant_rec(
    nodes: &[TensorIr],
    idx: usize,
    visited: &mut HashSet<usize>,
    max_const: &mut u32,
) {
    if !visited.insert(idx) {
        return;
    }
    if let TensorIr::Const(crate::types::ScalarValue::U32(v)) = &nodes[idx]
        && *v > *max_const
    {
        *max_const = *v;
    }
    for child_id in nodes[idx].children() {
        find_max_constant_rec(nodes, usize::from(*child_id), visited, max_const);
    }
}
