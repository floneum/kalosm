//! `verify_plan` — the hard conformance assert on the extraction winner.
//!
//! Six clauses, each an [`Error::Plan`], **never** a silent fallback:
//!
//! 1. every selected non-`Leaf` node is at `Level::Launch`;
//! 2. `theta` is a member of the node's `ScheduleDomain`, the geometry's own
//!    `legal` predicate holds, and the **exact** `ArenaPlanner` says the
//!    workgroup footprint fits;
//! 3. every operand's source class is selected, and its node is either in `M`
//!    or in the same launch;
//! 4. every `BufferPlan` layout has the rank its value needs and no
//!    undefined symbolic stride;
//! 5. no `Effect::InPlace` node is inlined;
//! 6. every root is in `M`, and every `Launch::Ext` node can actually run
//!    somewhere;
//! 7. every launch's bind group — its operands **plus the `Uniforms` block** —
//!    fits `max_storage_buffers_per_shader_stage`.

use crate::plan::UNKNOWN_SYM;
use crate::realize::{self, scalar_element, tiles_for};
use fusor2_ir::Result;
use fusor2_ir::device::Caps;
use fusor2_ir::egraph::{EGraph, Id};
use fusor2_ir::error::Error;
use fusor2_ir::extract::Plan;
use fusor2_ir::ir::Op;
use fusor2_ir::ir::launch::{Effect, Launch, SchedPoint, ScheduleDomain};
use fusor2_ir::ir::kernel::ArenaPlanner;
use fusor2_ir::ir::{OpDefId, OpDefRegistry};
use fusor2_ir::shape::Dim;
use rustc_hash::FxHashMap;

/// Clauses 1, 3, 4, 5 and the root half of 6 — everything derivable from the
/// graph and the plan alone.
pub(crate) fn verify_plan(graph: &EGraph, plan: &Plan) -> Result<()> {
    check_levels(graph, plan)?;
    check_operands(graph, plan)?;
    check_operand_spaces(graph, plan)?;
    check_buffers(graph, plan)?;
    check_effect_pinning(graph, plan)?;
    check_roots(graph, plan)?;
    Ok(())
}

/// Clause 8: a selected `Fold`'s aliased operands must be addressable by
/// the fold's own flat index map.
///
/// The fold lowerings read every operand by running the flat space index
/// through the operand's own layout map, unmasked. That map is exact when
/// the operand is
///
/// * a single element,
/// * stated over the space itself — full rank, each extent equal to the
///   space's (a stride-0 axis is a broadcast) or `1`, or
/// * a suffix of the space: each layout dim equal to the corresponding
///   trailing space dim (`weights[n]` under `[m, n]`).
///
/// Anything else is read at garbage addresses on every backend.
pub(crate) fn check_operand_spaces(graph: &EGraph, plan: &Plan) -> Result<()> {
    use fusor2_ir::ir::launch::AccessPlan;
    for id in selected(plan) {
        let Op::Launch(Launch::Fold { space, ops, .. }) = &graph.node(id).op else {
            continue;
        };
        for (i, o) in ops.iter().enumerate() {
            if o.access != AccessPlan::Alias {
                continue;
            }
            let shape = o.layout.shape();
            let single = shape.iter().all(|d| d.known_eq(Dim::Const(1)));
            if single {
                continue;
            }
            let full_rank = shape.len() == space.rank()
                && shape
                    .iter()
                    .zip(&space.dims)
                    .all(|(l, d)| l.known_eq(*d) || l.known_eq(Dim::Const(1)));
            let suffix = shape.len() < space.rank()
                && shape
                    .iter()
                    .zip(&space.dims[space.rank() - shape.len()..])
                    .all(|(l, d)| l.known_eq(*d) || l.known_eq(Dim::Const(1)));
            if !(full_rank || suffix) {
                return Err(Error::Plan(format!(
                    "selected {id}: fold operand {i} aliases a {:?} layout under the \
                     {:?} index space; the fold's flat index map cannot address it. \
                     The rule that minted this member states its operands over the \
                     wrong space — fix the rule, do not route around the member.",
                    shape, space.dims
                )));
            }
        }
    }
    Ok(())
}

/// All six clauses. The schedule clause needs the exact planner and the caps
/// it was admitted against; the extension clause needs the registry the
/// e-graph's semantics were built with.
pub(crate) fn verify_plan_with(
    graph: &EGraph,
    plan: &Plan,
    arena: &dyn ArenaPlanner,
    caps: &Caps,
    registry: Option<&OpDefRegistry>,
) -> Result<()> {
    verify_plan(graph, plan)?;
    check_schedules(graph, plan, arena, caps)?;
    check_bind_groups(plan, caps)?;
    check_extensions(graph, plan, registry)?;
    Ok(())
}

/// Clause 7: every launch's bind group fits
/// `max_storage_buffers_per_shader_stage`.
///
/// The uniform block counts: `plan::derive_bindings` reserves binding 0 for
/// `Uniforms` and does not list it, but it is emitted in the `storage`
/// address space, so the bound is `bindings.len() + 1`.
pub(crate) fn check_bind_groups(plan: &Plan, caps: &Caps) -> Result<()> {
    let limit = caps.limits.max_storage_buffers_per_shader_stage as usize;
    for (i, launch) in plan.launches.iter().enumerate() {
        let needed = launch.bindings.len() + 1;
        if needed > limit {
            return Err(Error::Plan(format!(
                "launch {i} (root {}) binds {} storage buffers — {} operands plus the \
                 Uniforms block — over the {limit}-buffer limit. A rule widened an \
                 operand list past what this device can bind.",
                launch.root,
                needed,
                launch.bindings.len()
            )));
        }
    }
    Ok(())
}

/// Clause 1: every selected non-leaf node is at Launch — nothing skipped a level.
pub(crate) fn check_levels(graph: &EGraph, plan: &Plan) -> Result<()> {
    for id in selected(plan) {
        // The same predicate the seed and the move generator select against.
        if !realize::is_runnable(graph, id) {
            return Err(Error::Plan(format!(
                "selected {id} is at {} but only Launch nodes are runnable",
                graph.level(id)
            )));
        }
    }
    Ok(())
}

/// Clause 2.
pub(crate) fn check_schedules(
    graph: &EGraph,
    plan: &Plan,
    arena: &dyn ArenaPlanner,
    caps: &Caps,
) -> Result<()> {
    let width = caps.subgroup_width();
    let max_lanes = caps.limits.max_compute_invocations_per_workgroup;
    let max_storage = caps.limits.max_compute_workgroup_storage_size;

    for id in selected(plan) {
        // Every lowering indexes the flattened iteration space in u32, so a
        // space past u32::MAX wraps.
        if let Op::Launch(l1) = &graph.node(id).op
            && let Some(iters) = l1.iter_space().iterations()
            && iters > u64::from(u32::MAX)
        {
            return Err(Error::Plan(format!(
                "{id} iterates {iters} elements, past u32 flat addressing"
            )));
        }
        let domain = match &graph.node(id).op {
            Op::Launch(l1) => l1.schedule(),
            _ => None,
        };
        let Some(domain) = domain else {
            continue;
        };
        let theta = match plan.extraction.theta.get(&id).copied() {
            Some(t) => t,
            None => {
                if matches!(domain, ScheduleDomain::Point) {
                    continue;
                }
                return Err(Error::Plan(format!(
                    "selected {id} carries a {}-point schedule domain but no theta",
                    domain.len()
                )));
            }
        };
        if !domain.iter().any(|p| p == theta) {
            return Err(Error::Plan(format!(
                "theta of {id} is not a member of its schedule domain"
            )));
        }
        match theta {
            SchedPoint::Coop { geom, .. } => {
                if !geom.legal(width, max_lanes) {
                    return Err(Error::Plan(format!(
                        "coop geometry of {id} is illegal at subgroup width {width}"
                    )));
                }
            }
            SchedPoint::Sgemm(p) => {
                let elem = graph.facts(id).dtype.byte_size().max(1) as u32;
                if !p.legal(elem, max_storage, max_lanes) {
                    return Err(Error::Plan(format!("sgemm geometry of {id} is illegal")));
                }
            }
            _ => {}
        }
        let lanes = realize::fold_footprint(graph, id).map(|(l, _)| l);
        let tiles = tiles_for(Some(theta), scalar_element(graph.facts(id).dtype), lanes, caps);
        let bytes = arena.workgroup_bytes(&tiles, caps)?;
        if bytes > max_storage {
            return Err(Error::Plan(format!(
                "{id} needs {bytes} workgroup bytes, over the {max_storage}-byte limit"
            )));
        }
    }
    Ok(())
}

/// Clause 3.
pub(crate) fn check_operands(graph: &EGraph, plan: &Plan) -> Result<()> {
    let launch_of = launch_index(plan);
    for (li, launch) in plan.launches.iter().enumerate() {
        for member in &launch.members {
            for child in graph.node(*member).children.iter() {
                let class = graph.class_of(*child);
                let Some(src) = plan.extraction.selected(class) else {
                    return Err(Error::Plan(format!(
                        "operand class {} of {member} is unselected",
                        class.0
                    )));
                };
                if plan.extraction.is_materialized(src) {
                    continue;
                }
                if realize::leaf_role(graph, src) != realize::LeafRole::NotLeaf {
                    continue;
                }
                if launch_of.get(&src).copied() == Some(li) {
                    continue;
                }
                return Err(Error::Plan(format!(
                    "operand {src} of {member} is neither materialized nor in launch {li}"
                )));
            }
        }
    }
    Ok(())
}

/// Clause 4.
pub(crate) fn check_buffers(graph: &EGraph, plan: &Plan) -> Result<()> {
    for b in &plan.buffers {
        let value_rank = graph.facts(b.value).rank();
        // A split-K scratch buffer carries one extra leading axis, one slice
        // per partial; every other buffer matches its value exactly.
        let extra = match plan.extraction.theta.get(&b.value) {
            Some(SchedPoint::Coop { splits, .. }) if *splits > 1 => 1,
            _ => 0,
        };
        if b.layout.rank() != value_rank + extra {
            return Err(Error::Plan(format!(
                "buffer for {} has rank {} but its value has rank {value_rank}",
                b.value,
                b.layout.rank()
            )));
        }
        for (axis, stride) in b.layout.strides().iter().enumerate() {
            if *stride == Dim::Sym(UNKNOWN_SYM) {
                // A `row_major_strides` placeholder is legal exactly when it
                // is derivable at dispatch: every following extent is a
                // constant or a bindable symbol.
                let derivable = b.layout.shape()[axis + 1..]
                    .iter()
                    .all(|d| !matches!(d, Dim::Sym(s) if *s == UNKNOWN_SYM));
                if !derivable {
                    return Err(Error::Plan(format!(
                        "buffer for {} has an underivable stride on axis {axis}",
                        b.value
                    )));
                }
            }
        }
    }
    Ok(())
}

/// Clause 5.
pub(crate) fn check_effect_pinning(graph: &EGraph, plan: &Plan) -> Result<()> {
    for id in selected(plan) {
        if let Effect::InPlace(role) = graph.semantics().effect(&graph.node(id).op)
            && !plan.extraction.is_materialized(id)
        {
            return Err(Error::Plan(format!(
                "in-place node {id} (buffer role {}) was inlined; its writes would apply once per consumer",
                role.0
            )));
        }
    }
    Ok(())
}

/// Clause 6, root half.
pub(crate) fn check_roots(graph: &EGraph, plan: &Plan) -> Result<()> {
    for root in graph.roots() {
        let class = graph.class_of(*root);
        let Some(sel) = plan.extraction.selected(class) else {
            return Err(Error::Plan(format!("root class {} is unselected", class.0)));
        };
        if realize::leaf_role(graph, sel) != realize::LeafRole::NotLeaf {
            continue;
        }
        if !plan.extraction.is_materialized(sel) {
            return Err(Error::Plan(format!(
                "root {sel} is not materialized; nothing would land in a buffer"
            )));
        }
    }
    Ok(())
}

/// Clause 6, extension half: an `Launch::Ext` whose `lower_per_target` is empty
/// cannot run on any target and must never be selected.
pub(crate) fn check_extensions(
    graph: &EGraph,
    plan: &Plan,
    registry: Option<&OpDefRegistry>,
) -> Result<()> {
    for id in selected(plan) {
        let Op::Launch(Launch::Ext { def, .. }) = &graph.node(id).op else {
            continue;
        };
        let Some(registry) = registry else {
            return Err(Error::Plan(format!(
                "extension node {id} selected but no OpDefRegistry was supplied to verify it"
            )));
        };
        let Some(entry) = registry.get(*def) else {
            return Err(Error::Plan(format!(
                "extension node {id} names unregistered {:?}",
                OpDefId(def.0)
            )));
        };
        if entry.lower_per_target.is_empty() {
            return Err(Error::Plan(format!(
                "extension `{}` selected at {id} lowers on no target",
                entry.name
            )));
        }
    }
    Ok(())
}

fn selected(plan: &Plan) -> Vec<Id> {
    let mut out: Vec<Id> = plan.extraction.sigma.values().copied().collect();
    out.sort_unstable();
    out.dedup();
    out
}

fn launch_index(plan: &Plan) -> FxHashMap<Id, usize> {
    let mut out = FxHashMap::default();
    for (i, launch) in plan.launches.iter().enumerate() {
        for m in &launch.members {
            out.insert(*m, i);
        }
    }
    out
}
