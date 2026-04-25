use std::collections::{HashMap, HashSet};

use egg::{EGraph, Id, Language, Rewrite};

use crate::analysis::TensorAnalysis;
use crate::applier::SimpleEclassSearcher;
use crate::language::{SimdNode, TensorIr};
use crate::rules::RunnerConfig;
use crate::types::{BinaryOp, BinderKind, MemTier, ScalarValue, VarRef, slots};

pub(super) fn build(config: &RunnerConfig) -> Rewrite<TensorIr, TensorAnalysis> {
    Rewrite::new(
        "tiled-load-promote",
        SimpleEclassSearcher::new(|egraph, eclass| {
            egraph[eclass]
                .iter()
                .any(|node| theta_has_promotable_loads(egraph, eclass, node))
        }),
        crate::applier::AdaptedApplier(TiledLoadPromotionApplier {
            simd_width: config.device.simd_width,
        }),
    )
    .unwrap()
}

fn theta_has_promotable_loads(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    theta_class: Id,
    node: &TensorIr,
) -> bool {
    let TensorIr::Simd(SimdNode::Theta {
        children: [_, count, update],
    }) = node
    else {
        return false;
    };
    egraph[*count].data.constant.is_some()
        && subtree_has_promotable_device_loads(egraph, *update, theta_class)
}

fn subtree_has_promotable_device_loads(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    root: Id,
    theta_class: Id,
) -> bool {
    let mut visited = HashSet::new();
    let theta_class = egraph.find(theta_class);
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        let canonical = egraph.find(id);
        if canonical == theta_class || !visited.insert(canonical) {
            continue;
        }
        for node in egraph[canonical].iter() {
            if let TensorIr::Simd(SimdNode::Load {
                tier: MemTier::Device(_),
                children,
            }) = node
                && load_address_is_promotable(egraph, children[0])
            {
                return true;
            }
            for child in node.children() {
                stack.push(*child);
            }
        }
    }
    false
}

fn load_address_is_promotable(egraph: &EGraph<TensorIr, TensorAnalysis>, addr: Id) -> bool {
    egraph[egraph.find(addr)]
        .data
        .var_dep
        .contains(&VarRef::iter(1))
        || address_has_broadcast_tile_local_form(egraph, addr)
}

struct TiledLoadPromotionApplier {
    simd_width: u32,
}

impl crate::applier::TypedApplier for TiledLoadPromotionApplier {
    fn apply(&self, egraph: &mut EGraph<TensorIr, TensorAnalysis>, eclass: Id) -> Vec<Id> {
        let node = egraph[eclass]
            .iter()
            .find(|n| theta_has_promotable_loads(egraph, eclass, n))
            .cloned();

        let Some(TensorIr::Simd(SimdNode::Theta {
            children: [init, count, update],
        })) = node
        else {
            return vec![];
        };

        let tile_size = match &egraph[count].data.constant {
            Some(ScalarValue::U32(v)) => *v,
            _ => return vec![],
        };

        let mut memo = HashMap::new();
        let mut visiting = HashSet::new();
        let mut changed = false;
        let mut ctx = PromotionCtx {
            egraph,
            theta_class: eclass,
            tile_size,
            simd_width: self.simd_width,
            memo: &mut memo,
            visiting: &mut visiting,
            changed: &mut changed,
        };
        let promoted_update = ctx.promote_tiled_loads_in_subtree(update);

        if !changed || promoted_update == update {
            return vec![];
        }

        let promoted_theta = egraph.add(TensorIr::Simd(SimdNode::Theta {
            children: [init, count, promoted_update],
        }));
        egraph.union(eclass, promoted_theta);
        vec![promoted_theta]
    }
}

struct PromotionCtx<'a> {
    egraph: &'a mut EGraph<TensorIr, TensorAnalysis>,
    theta_class: Id,
    tile_size: u32,
    simd_width: u32,
    memo: &'a mut HashMap<Id, Id>,
    visiting: &'a mut HashSet<Id>,
    changed: &'a mut bool,
}

impl PromotionCtx<'_> {
    fn promote_tiled_loads_in_subtree(&mut self, root: Id) -> Id {
        let canonical = self.egraph.find(root);
        if let Some(&cached) = self.memo.get(&canonical) {
            return cached;
        }
        if !self.visiting.insert(canonical) {
            return root;
        }
        if canonical == self.egraph.find(self.theta_class) {
            self.visiting.remove(&canonical);
            return root;
        }

        let nodes: Vec<_> = self.egraph[canonical].iter().cloned().collect();
        let mut fallback = root;

        for node in nodes {
            if let TensorIr::Simd(SimdNode::Load {
                tier: MemTier::Device(buf),
                children,
            }) = &node
            {
                let addr = children[0];
                let state = children[1];
                let k_outer_var = VarRef::iter(1);
                let tile_local_addr = if self.egraph[addr].data.var_dep.contains(&k_outer_var) {
                    build_tile_local_address(self.egraph, addr, self.tile_size, self.simd_width)
                } else {
                    build_broadcast_tile_local_address(self.egraph, addr)
                };
                if let Some(tile_local_addr) = tile_local_addr {
                    let promoted = self.egraph.add(TensorIr::Simd(SimdNode::Load {
                        tier: MemTier::Device(*buf).to_threadgroup(),
                        children: [tile_local_addr, state],
                    }));
                    self.egraph.union(canonical, promoted);
                    self.memo.insert(canonical, promoted);
                    self.visiting.remove(&canonical);
                    *self.changed = true;
                    return promoted;
                }
            }

            if node.children().is_empty() {
                fallback = root;
                continue;
            }

            let mut rebuilt = node.clone();
            let mut node_changed = false;
            for child in rebuilt.children_mut() {
                let remapped = self.promote_tiled_loads_in_subtree(*child);
                node_changed |= remapped != *child;
                *child = remapped;
            }

            if node_changed {
                fallback = self.egraph.add(rebuilt);
                break;
            }
        }

        self.memo.insert(canonical, fallback);
        self.visiting.remove(&canonical);
        fallback
    }
}

fn build_tile_local_address(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    addr: Id,
    tile_size: u32,
    simd_width: u32,
) -> Option<Id> {
    let mut visited = HashSet::new();
    build_tile_local_address_rec(egraph, addr, tile_size, simd_width, &mut visited)
}

fn build_tile_local_address_rec(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    addr: Id,
    tile_size: u32,
    simd_width: u32,
    visited: &mut HashSet<Id>,
) -> Option<Id> {
    let canonical = egraph.find(addr);
    if !visited.insert(canonical) {
        return None;
    }
    let nodes: Vec<_> = egraph[canonical].iter().cloned().collect();
    for node in nodes {
        if let Some(inner_addr) = division_by_one_value(egraph, &node)
            && egraph.find(inner_addr) != canonical
            && let Some(local_addr) =
                build_tile_local_address_rec(egraph, inner_addr, tile_size, simd_width, visited)
        {
            return Some(local_addr);
        }
        if let Some(inner_addr) = divmod_recomposed_value(egraph, &node)
            && egraph.find(inner_addr) != canonical
            && let Some(local_addr) =
                build_tile_local_address_rec(egraph, inner_addr, tile_size, simd_width, visited)
        {
            return Some(local_addr);
        }
        if let TensorIr::BinOp(name, args) = node
            && matches!(name, BinaryOp::Add)
            && args.len() == 2
        {
            if let Some(local_addr) =
                rebuild_2d_tile_local_address(egraph, args[0], args[1], tile_size, simd_width)
            {
                return Some(local_addr);
            }
            for (lhs, rhs) in [(args[0], args[1]), (args[1], args[0])] {
                if let Some(local_rhs) = strip_outer_tile_term(egraph, rhs, tile_size) {
                    return Some(if lhs == rhs {
                        local_rhs
                    } else {
                        egraph.add(TensorIr::BinOp(BinaryOp::Add, [lhs, local_rhs]))
                    });
                }
                if let Some(local_lhs) = strip_outer_tile_term(egraph, lhs, tile_size) {
                    return Some(if lhs == rhs {
                        local_lhs
                    } else {
                        egraph.add(TensorIr::BinOp(BinaryOp::Add, [local_lhs, rhs]))
                    });
                }
            }
        }
    }
    None
}

fn address_has_broadcast_tile_local_form(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    addr: Id,
) -> bool {
    let mut visited = HashSet::new();
    address_has_broadcast_tile_local_form_rec(egraph, addr, &mut visited)
}

fn address_has_broadcast_tile_local_form_rec(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    addr: Id,
    visited: &mut HashSet<Id>,
) -> bool {
    let canonical = egraph.find(addr);
    if !visited.insert(canonical) {
        return false;
    }
    for node in egraph[canonical].iter() {
        if let Some(inner_addr) = division_by_one_value(egraph, node)
            && egraph.find(inner_addr) != canonical
            && address_has_broadcast_tile_local_form_rec(egraph, inner_addr, visited)
        {
            return true;
        }
        if let TensorIr::Simd(SimdNode::Var(var)) = node
            && *var == VarRef::iter(0)
        {
            return true;
        }
        if let TensorIr::BinOp(BinaryOp::Add | BinaryOp::Sub, args) = node {
            if address_is_zero(egraph, args[0])
                && address_has_broadcast_tile_local_form_rec(egraph, args[1], visited)
            {
                return true;
            }
            if matches!(node, TensorIr::BinOp(BinaryOp::Add, _))
                && address_is_zero(egraph, args[1])
                && address_has_broadcast_tile_local_form_rec(egraph, args[0], visited)
            {
                return true;
            }
        }
    }
    false
}

fn build_broadcast_tile_local_address(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    addr: Id,
) -> Option<Id> {
    let mut visited = HashSet::new();
    build_broadcast_tile_local_address_rec(egraph, addr, &mut visited)
}

fn build_broadcast_tile_local_address_rec(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    addr: Id,
    visited: &mut HashSet<Id>,
) -> Option<Id> {
    let canonical = egraph.find(addr);
    if !visited.insert(canonical) {
        return None;
    }
    let nodes: Vec<_> = egraph[canonical].iter().cloned().collect();
    for node in nodes {
        if let Some(inner_addr) = division_by_one_value(egraph, &node)
            && egraph.find(inner_addr) != canonical
            && let Some(local_addr) =
                build_broadcast_tile_local_address_rec(egraph, inner_addr, visited)
        {
            return Some(local_addr);
        }
        if let TensorIr::Simd(SimdNode::Var(var)) = node
            && var == VarRef::iter(0)
        {
            return Some(canonical);
        }
        if let TensorIr::BinOp(BinaryOp::Add | BinaryOp::Sub, args) = node {
            if address_is_zero(egraph, args[0])
                && let Some(local_addr) =
                    build_broadcast_tile_local_address_rec(egraph, args[1], visited)
            {
                return Some(local_addr);
            }
            if matches!(node, TensorIr::BinOp(BinaryOp::Add, _))
                && address_is_zero(egraph, args[1])
                && let Some(local_addr) =
                    build_broadcast_tile_local_address_rec(egraph, args[0], visited)
            {
                return Some(local_addr);
            }
        }
    }
    None
}

fn address_is_zero(egraph: &EGraph<TensorIr, TensorAnalysis>, id: Id) -> bool {
    fn rec(egraph: &EGraph<TensorIr, TensorAnalysis>, id: Id, visited: &mut HashSet<Id>) -> bool {
        let canonical = egraph.find(id);
        if !visited.insert(canonical) {
            return false;
        }
        egraph[canonical].iter().any(|node| {
            matches!(node, TensorIr::Const(ScalarValue::U32(0)))
                || modulo_by_one_value(egraph, node).is_some()
                || matches!(node, TensorIr::BinOp(BinaryOp::Add | BinaryOp::Sub, args) if rec(egraph, args[0], visited) && rec(egraph, args[1], visited))
        })
    }

    let mut visited = HashSet::new();
    rec(egraph, id, &mut visited)
}

fn division_by_one_value(egraph: &EGraph<TensorIr, TensorAnalysis>, node: &TensorIr) -> Option<Id> {
    let TensorIr::BinOp(BinaryOp::Div, args) = node else {
        return None;
    };
    (const_u32(egraph, args[1]) == Some(1)).then_some(args[0])
}

fn modulo_by_one_value(egraph: &EGraph<TensorIr, TensorAnalysis>, node: &TensorIr) -> Option<Id> {
    let TensorIr::BinOp(BinaryOp::Mod, args) = node else {
        return None;
    };
    (const_u32(egraph, args[1]) == Some(1)).then_some(args[0])
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

fn rebuild_2d_tile_local_address(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    lhs: Id,
    rhs: Id,
    tile_size: u32,
    simd_width: u32,
) -> Option<Id> {
    for (row_side, col_side) in [(lhs, rhs), (rhs, lhs)] {
        let row_expr = extract_mul_non_const_operand(egraph, row_side)?;
        let row_has_outer_k = expr_has_outer_k(egraph, row_expr);
        let col_has_outer_k = expr_has_outer_k(egraph, col_side);
        if row_has_outer_k == col_has_outer_k {
            continue;
        }

        let local_row = if row_has_outer_k {
            strip_outer_tile_term(egraph, row_expr, tile_size)?
        } else {
            strip_thread_axis_base(egraph, row_expr)?
        };
        let local_col = if col_has_outer_k {
            strip_outer_tile_term(egraph, col_side, tile_size)?
        } else {
            strip_thread_axis_base(egraph, col_side)?
        };

        let tile_cols = infer_local_axis_range(egraph, local_col, simd_width).unwrap_or(tile_size);
        let tile_size_lit = egraph.add(TensorIr::Const(ScalarValue::U32(tile_cols)));
        let local_row_stride =
            egraph.add(TensorIr::BinOp(BinaryOp::Mul, [local_row, tile_size_lit]));
        return Some(egraph.add(TensorIr::BinOp(
            BinaryOp::Add,
            [local_row_stride, local_col],
        )));
    }
    None
}

fn infer_local_axis_range(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    id: Id,
    simd_width: u32,
) -> Option<u32> {
    fn rec(
        egraph: &EGraph<TensorIr, TensorAnalysis>,
        id: Id,
        simd_width: u32,
        memo: &mut HashMap<Id, Option<u32>>,
        visiting: &mut HashSet<Id>,
    ) -> Option<u32> {
        let canonical = egraph.find(id);
        if let Some(&cached) = memo.get(&canonical) {
            return cached;
        }
        if !visiting.insert(canonical) {
            return None;
        }

        let mut best = None;
        for node in egraph[canonical].iter() {
            let candidate = match node {
                TensorIr::Simd(SimdNode::Var(VarRef::Bound {
                    kind: BinderKind::Dispatch,
                    slot: slots::DISPATCH_LANE,
                    depth: 0,
                })) => Some(simd_width),
                TensorIr::BinOp(BinaryOp::Mod, args) => {
                    if let Some(ScalarValue::U32(v)) = &egraph[args[1]].data.constant {
                        Some(*v)
                    } else {
                        None
                    }
                }
                TensorIr::BinOp(BinaryOp::Div, args) => {
                    if let Some(ScalarValue::U32(v)) = &egraph[args[1]].data.constant {
                        rec(egraph, args[0], simd_width, memo, visiting)
                            .map(|range| range.div_ceil(*v))
                    } else {
                        None
                    }
                }
                TensorIr::BinOp(BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul, args) => {
                    let left_dep = egraph[args[0]].data.dep;
                    let right_dep = egraph[args[1]].data.dep;
                    if left_dep.contains_lane() && !right_dep.contains_lane() {
                        rec(egraph, args[0], simd_width, memo, visiting)
                    } else if right_dep.contains_lane() && !left_dep.contains_lane() {
                        rec(egraph, args[1], simd_width, memo, visiting)
                    } else {
                        None
                    }
                }
                _ => None,
            };
            if let Some(range) = candidate.filter(|range| *range > 0) {
                best = Some(best.map_or(range, |current: u32| current.min(range)));
            }
        }

        visiting.remove(&canonical);
        memo.insert(canonical, best);
        best
    }

    let mut memo = HashMap::new();
    let mut visiting = HashSet::new();
    rec(egraph, id, simd_width, &mut memo, &mut visiting)
}

fn extract_mul_non_const_operand(egraph: &EGraph<TensorIr, TensorAnalysis>, id: Id) -> Option<Id> {
    let canonical = egraph.find(id);
    for node in egraph[canonical].iter() {
        if let TensorIr::BinOp(name, args) = node
            && matches!(name, BinaryOp::Mul)
            && args.len() == 2
        {
            if egraph[args[0]].data.constant.is_some() && egraph[args[1]].data.constant.is_none() {
                return Some(args[1]);
            }
            if egraph[args[1]].data.constant.is_some() && egraph[args[0]].data.constant.is_none() {
                return Some(args[0]);
            }
        }
    }
    None
}

fn expr_has_outer_k(egraph: &EGraph<TensorIr, TensorAnalysis>, id: Id) -> bool {
    egraph[egraph.find(id)]
        .data
        .var_dep
        .contains(&VarRef::iter(1))
}

fn strip_workgroup_base(egraph: &mut EGraph<TensorIr, TensorAnalysis>, id: Id) -> Option<Id> {
    let canonical = egraph.find(id);
    if !egraph[canonical].data.dep.contains_workgroup() {
        return Some(canonical);
    }
    let base = extract_workgroup_additive_base(egraph, canonical)?;
    Some(egraph.add(TensorIr::BinOp(BinaryOp::Sub, [canonical, base])))
}

fn strip_thread_axis_base(egraph: &mut EGraph<TensorIr, TensorAnalysis>, id: Id) -> Option<Id> {
    let canonical = egraph.find(id);
    let baseline = substitute_lane_with_zero(egraph, canonical);
    if baseline != canonical {
        return Some(egraph.add(TensorIr::BinOp(BinaryOp::Sub, [canonical, baseline])));
    }
    strip_workgroup_base(egraph, canonical)
}

fn substitute_lane_with_zero(egraph: &mut EGraph<TensorIr, TensorAnalysis>, id: Id) -> Id {
    fn rec(
        egraph: &mut EGraph<TensorIr, TensorAnalysis>,
        id: Id,
        memo: &mut HashMap<Id, Id>,
        visiting: &mut HashSet<Id>,
    ) -> Id {
        let canonical = egraph.find(id);
        if let Some(&cached) = memo.get(&canonical) {
            return cached;
        }
        if !visiting.insert(canonical) {
            return canonical;
        }

        let nodes: Vec<_> = egraph[canonical].iter().cloned().collect();
        let mut rebuilt = canonical;
        for node in nodes {
            match node {
                TensorIr::Simd(SimdNode::Var(VarRef::Bound {
                    kind: BinderKind::Dispatch,
                    slot: slots::DISPATCH_LANE,
                    depth: 0,
                })) => {
                    rebuilt = egraph.add(TensorIr::Const(ScalarValue::U32(0)));
                    break;
                }
                mut other => {
                    let original_children: Vec<Id> = other.children().to_vec();
                    for child in other.children_mut() {
                        *child = rec(egraph, *child, memo, visiting);
                    }
                    let child_changed = other
                        .children()
                        .iter()
                        .zip(original_children.iter())
                        .any(|(new, old)| new != old);
                    if child_changed {
                        rebuilt = egraph.add(other);
                        break;
                    }
                }
            }
        }

        visiting.remove(&canonical);
        memo.insert(canonical, rebuilt);
        rebuilt
    }

    let mut memo = HashMap::new();
    let mut visiting = HashSet::new();
    rec(egraph, id, &mut memo, &mut visiting)
}

fn extract_workgroup_additive_base(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    id: Id,
) -> Option<Id> {
    let canonical = egraph.find(id);
    for node in egraph[canonical].iter() {
        if let TensorIr::BinOp(name, args) = node
            && matches!(name, BinaryOp::Add)
            && args.len() == 2
        {
            let left_dep = egraph[args[0]].data.dep;
            let right_dep = egraph[args[1]].data.dep;
            let left_is_wg_only = left_dep.contains_workgroup()
                && !left_dep.contains_lane()
                && !left_dep.contains_simdgroup();
            let right_is_wg_only = right_dep.contains_workgroup()
                && !right_dep.contains_lane()
                && !right_dep.contains_simdgroup();
            if left_is_wg_only && !right_is_wg_only {
                return Some(args[0]);
            }
            if right_is_wg_only && !left_is_wg_only {
                return Some(args[1]);
            }
        }
    }
    None
}

fn strip_outer_tile_term(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    id: Id,
    tile_size: u32,
) -> Option<Id> {
    let canonical = egraph.find(id);
    if egraph[canonical].data.var_dep.contains(&VarRef::iter(0))
        && !egraph[canonical].data.var_dep.contains(&VarRef::iter(1))
    {
        return Some(canonical);
    }
    let nodes: Vec<_> = egraph[canonical].iter().cloned().collect();
    for node in nodes {
        if let TensorIr::BinOp(name, args) = node {
            if matches!(name, BinaryOp::Add) && args.len() == 2 {
                let left_has_outer = egraph[args[0]].data.var_dep.contains(&VarRef::iter(1));
                let right_has_outer = egraph[args[1]].data.var_dep.contains(&VarRef::iter(1));
                let left_has_inner_k = egraph[args[0]].data.var_dep.contains(&VarRef::iter(0));
                let right_has_inner_k = egraph[args[1]].data.var_dep.contains(&VarRef::iter(0));

                if left_has_outer && right_has_inner_k && !right_has_outer {
                    return Some(egraph.find(args[1]));
                }
                if right_has_outer && left_has_inner_k && !left_has_outer {
                    return Some(egraph.find(args[0]));
                }
            }
            if !matches!(name, BinaryOp::Mul) || args.len() != 2 {
                continue;
            }
            for (lhs, rhs) in [(args[0], args[1]), (args[1], args[0])] {
                let is_tile_size = matches!(
                    &egraph[rhs].data.constant,
                    Some(ScalarValue::U32(v)) if *v == tile_size
                );
                let has_outer_k = egraph[lhs].data.var_dep.contains(&VarRef::iter(1));
                if is_tile_size && has_outer_k {
                    return Some(egraph.add(TensorIr::Const(ScalarValue::U32(0))));
                }
            }
        }
    }
    None
}
