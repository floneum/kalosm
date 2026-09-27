//! `verify_plan` — the hard conformance assert on the extraction winner.
//!
//! Compiler invariants checked independently in tests:
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
//! 6. every root is in `M`;
//! 7. every launch's bind group — its operands **plus the `Uniforms` block** —
//!    fits `max_storage_buffers_per_shader_stage`.

use crate::nodes::{composite_members, domain_of, is_group, plan_values};
use crate::realize;
use fusor_ir::Result;
use fusor_ir::device::Caps;
use fusor_ir::egraph::{EGraph, Id};
use fusor_ir::error::Error;
use fusor_ir::extract::Plan;
use fusor_ir::ir::Op;
use fusor_ir::ir::kernel::ArenaPlanner;
use fusor_ir::ir::launch::{Effect, Launch, SchedPoint, ScheduleDomain};
use fusor_ir::shape::Dim;
use fusor_ir::shape::OPAQUE_SYM;
use rustc_hash::{FxHashMap, FxHashSet};

/// Every clause derivable from the graph and the plan alone.
pub(crate) fn verify_plan(graph: &EGraph, plan: &Plan) -> Result<()> {
    check_levels(graph, plan)?;
    check_operands(graph, plan)?;
    check_operand_spaces(graph, plan)?;
    check_buffers(graph, plan)?;
    check_launch_order(graph, plan)?;
    check_effect_pinning(graph, plan)?;
    check_roots(graph, plan)?;
    check_slabs(graph, plan)?;
    Ok(())
}

/// A selected slab is one materialized launch with all its members, each
/// middle member selected and materialized, the last's class selecting it.
pub(crate) fn check_slabs(graph: &EGraph, plan: &Plan) -> Result<()> {
    let launch_of = launch_index(plan);
    let mut realized: Vec<Id> = Vec::new();
    // A member slab's class selects the group it ends.
    let mut group_last: FxHashSet<Id> = FxHashSet::default();
    // A group's node count is checked on the group.
    let mut grouped: FxHashSet<Id> = FxHashSet::default();
    for id in plan.launches.iter().flat_map(|l| l.members.iter().copied()) {
        let Some(members) = composite_members(graph, id) else {
            continue;
        };
        realized.push(id);
        if is_group(graph, id) {
            group_last.extend(members.last().copied());
            grouped.extend(members.iter().copied());
        }
    }
    realized.sort_unstable();
    realized.dedup();
    for id in realized {
        let Some(members) = composite_members(graph, id) else {
            continue;
        };
        if group_last.contains(&id) {
            continue;
        }
        let expected = match &graph.node(id).op {
            Op::Launch(Launch::Group { .. }) => {
                members
                    .iter()
                    .map(|m| match &graph.node(*m).op {
                        Op::Launch(Launch::Slab { members: sm, .. }) => sm.len() + 1,
                        _ => 1,
                    })
                    .sum::<usize>()
                    + 1
            }
            _ => members.len() + 1,
        };
        if !plan.extraction.is_materialized(id) {
            return Err(Error::Plan(format!("slab {id} is inlined")));
        }
        let own = launch_of.get(&id).copied();
        let Some((last, middle)) = members.split_last() else {
            return Err(Error::Plan(format!("slab {id} has no members")));
        };
        if plan.extraction.selected(graph.class_of(*last)) != Some(id) {
            return Err(Error::Plan(format!(
                "slab {id}'s last member {last} is selected past the slab"
            )));
        }
        if plan.extraction.is_materialized(*last) {
            return Err(Error::Plan(format!(
                "slab {id}'s last member {last} is materialized beside the slab's own buffer"
            )));
        }
        if let Some(ix) = own
            && !grouped.contains(&id)
            && plan.launches[ix].members.len() != expected
        {
            let foreign: Vec<Id> = plan.launches[ix]
                .members
                .iter()
                .copied()
                .filter(|m| *m != id && !members.contains(m))
                .collect();
            return Err(Error::Plan(format!(
                "slab {id}'s launch has {} nodes for {} members: foreign {foreign:?}",
                plan.launches[ix].members.len(),
                members.len()
            )));
        }
        for m in members.iter() {
            if launch_of.get(m).copied() != own {
                let class = graph.class_of(*m);
                return Err(Error::Plan(format!(
                    "slab {id}'s member {m} is not in the slab's launch: member launch {:?}, \
                     slab launch {own:?}, member materialized {}, class {} selects {:?}, \
                     op {:?}",
                    launch_of.get(m),
                    plan.extraction.is_materialized(*m),
                    class.0,
                    plan.extraction.selected(class),
                    graph.node(*m).op.tag(),
                )));
            }
        }
        for m in middle {
            if plan.extraction.selected(graph.class_of(*m)) != Some(*m) {
                let class = graph.class_of(*m);
                return Err(Error::Plan(format!(
                    "slab {id} ({:?}) member {m} ({:?}) is not its class {}'s selection {:?} ({:?})",
                    graph.node(id).op.tag(),
                    graph.node(*m).op.tag(),
                    class.0.index(),
                    plan.extraction.selected(class),
                    plan.extraction
                        .selected(class)
                        .map(|s| graph.node(s).op.tag()),
                )));
            }
            if !plan.extraction.is_materialized(*m) {
                return Err(Error::Plan(format!("slab {id}'s member {m} is inlined")));
            }
        }
    }
    Ok(())
}

/// Clause 8: a selected `Fold`'s aliased operands are a single element, or
/// stated over its space or a suffix of it (extents equal or `1`), so the
/// flat index map addresses them.
pub(crate) fn check_operand_spaces(graph: &EGraph, plan: &Plan) -> Result<()> {
    use fusor_ir::ir::launch::AccessPlan;
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

/// Check the plan against the device and arena used to construct it.
pub(crate) fn verify_plan_with(
    graph: &EGraph,
    plan: &Plan,
    arena: &dyn ArenaPlanner,
    caps: &Caps,
) -> Result<()> {
    verify_plan(graph, plan)?;
    check_schedules(graph, plan, arena, caps)?;
    check_bind_groups(plan, caps)?;
    Ok(())
}

/// Clause 7: every launch's bindings plus the uniform block fit
/// `max_storage_buffers_per_shader_stage`.
pub(crate) fn check_bind_groups(plan: &Plan, caps: &Caps) -> Result<()> {
    let limit = caps.limits.max_storage_buffers_per_shader_stage as usize;
    for (i, launch) in plan.launches.iter().enumerate() {
        // Arena values share a binding: count distinct slots.
        let mut slots: Vec<u32> = launch.bindings.iter().map(|b| b.binding).collect();
        slots.sort_unstable();
        slots.dedup();
        let needed = slots.len() + 1;
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
        // Lowerings index the flat space in u32.
        if let Op::Launch(l1) = &graph.node(id).op
            && let Some(iters) = l1.iter_space().iterations()
            && iters > u64::from(u32::MAX)
        {
            let what = plan
                .launches
                .iter()
                .find(|d| d.root == id)
                .map(|d| crate::extract::launch_signature(graph, d))
                .unwrap_or_default();
            let siblings: Vec<String> = graph
                .members(graph.class_of(id))
                .iter()
                .map(|m| {
                    format!(
                        "{m:?} {} legal={} domain={:?}",
                        crate::debug::op_tag(&graph.node(*m).op),
                        realize::composite_bindings_fit(graph, *m, caps),
                        domain_of(graph, *m).map(|d| d.len())
                    )
                })
                .collect();
            return Err(Error::Plan(format!(
                "{id} iterates {iters} elements, past u32 flat addressing: {what}; class members: [{}]",
                siblings.join("; ")
            )));
        }
        let Some(domain) = domain_of(graph, id) else {
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
        let bytes = realize::tile_bytes(graph, id, Some(theta), caps, arena)?;
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
        if b.layout.rank() != value_rank {
            return Err(Error::Plan(format!(
                "buffer for {} has rank {} but its value has rank {value_rank}",
                b.value,
                b.layout.rank()
            )));
        }
        for (axis, stride) in b.layout.strides().iter().enumerate() {
            if *stride == Dim::Sym(OPAQUE_SYM) {
                // A placeholder is legal when every later extent binds.
                let derivable = b.layout.shape()[axis + 1..]
                    .iter()
                    .all(|d| !matches!(d, Dim::Sym(s) if *s == OPAQUE_SYM));
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

fn selected(plan: &Plan) -> Vec<Id> {
    // The executable DAG, not sigma's abandoned choices.
    let mut out: Vec<Id> = plan_values(plan).collect();
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

/// Every value a launch reads is written earlier or comes from outside.
pub(crate) fn check_launch_order(graph: &EGraph, plan: &Plan) -> Result<()> {
    let mut written: rustc_hash::FxHashSet<Id> = rustc_hash::FxHashSet::default();
    for (i, l) in plan.launches.iter().enumerate() {
        for b in &l.bindings {
            if matches!(b.kind, fusor_ir::extract::BindKind::Read)
                && realize::leaf_role(graph, b.value) == realize::LeafRole::NotLeaf
                && !written.contains(&b.value)
            {
                return Err(Error::Plan(format!(
                    "launch {i} (root {}) reads {} before any launch writes it: the launches \
                     have a dependency cycle",
                    l.root, b.value
                )));
            }
        }
        for b in &l.bindings {
            if !matches!(b.kind, fusor_ir::extract::BindKind::Read) {
                written.insert(b.value);
            }
        }
    }
    Ok(())
}
