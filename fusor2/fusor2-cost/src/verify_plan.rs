//! The hard conformance assert on the extraction winner.
//!
//! Seven clauses, each an [`Error::Plan`], never a silent fallback:
//!
//! 1. every selected non-`Leaf` node is at `Level::L1`;
//! 2. `theta` is a member of the node's `ScheduleDomain`, the geometry's own
//!    `legal` predicate holds, and the **exact** `ArenaPlanner` says the
//!    workgroup footprint fits;
//! 3. every operand's source class is selected, and its node is either in `M`
//!    or in the same launch;
//! 4. every `BufferPlan` layout has the rank its value needs and no
//!    undefined symbolic stride;
//! 5. no `Effect::InPlace` node is inlined;
//! 6. every root is in `M`, and every `L1::Ext` node can actually run
//!    somewhere;
//! 7. every launch's bind group — its operands plus the `Uniforms` block —
//!    fits `max_storage_buffers_per_shader_stage`.

use crate::plan::UNKNOWN_SYM;
use crate::realize::{self, tiles_for};
use fusor2_ir::Result;
use fusor2_ir::device::Caps;
use fusor2_ir::egraph::{EGraph, Id};
use fusor2_ir::error::Error;
use fusor2_ir::extract::Plan;
use fusor2_ir::ir::Op;
use fusor2_ir::ir::level1::{Effect, L1, SchedPoint, ScheduleDomain};
use fusor2_ir::ir::level2::ArenaPlanner;
use fusor2_ir::ir::{OpDefId, OpDefRegistry};
use fusor2_ir::shape::Dim;
use rustc_hash::FxHashMap;

/// Clauses 1, 3, 4, 5 and the root half of 6 — everything derivable from the
/// graph and the plan alone.
pub fn verify_plan(graph: &EGraph, plan: &Plan) -> Result<()> {
    check_levels(graph, plan)?;
    check_operands(graph, plan)?;
    check_buffers(graph, plan)?;
    check_effect_pinning(graph, plan)?;
    check_roots(graph, plan)?;
    Ok(())
}

/// All seven clauses. The schedule clause needs the exact planner and the caps
/// it was admitted against; the extension clause needs the registry the
/// e-graph's semantics were built with.
pub fn verify_plan_with(
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
/// The uniform block counts against the limit: `plan::derive_bindings`
/// reserves binding 0 for `Uniforms` and does not list it, but the block is
/// emitted in the `storage` address space. The bound is `bindings.len() + 1`.
pub fn check_bind_groups(plan: &Plan, caps: &Caps) -> Result<()> {
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

/// Clause 1: every selected non-leaf node is at L1 — nothing skipped a level.
pub fn check_levels(graph: &EGraph, plan: &Plan) -> Result<()> {
    for id in selected(plan) {
        // The same predicate the seed and the move generator select against, so
        // a violation here means no rule ever lowered the class.
        if !realize::is_runnable(graph, id) {
            return Err(Error::Plan(format!(
                "selected {id} is at {} but only L1 nodes are runnable",
                graph.level(id)
            )));
        }
    }
    Ok(())
}

/// Clause 2.
pub fn check_schedules(
    graph: &EGraph,
    plan: &Plan,
    arena: &dyn ArenaPlanner,
    caps: &Caps,
) -> Result<()> {
    let width = caps.subgroup_width();
    let max_lanes = caps.limits.max_compute_invocations_per_workgroup;
    let max_storage = caps.limits.max_compute_workgroup_storage_size;

    for id in selected(plan) {
        let domain = match &graph.node(id).op {
            Op::L1(l1) => l1.schedule(),
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
        let tiles = tiles_for(Some(theta), graph.facts(id).dtype.scalar_element(), lanes, caps);
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
pub fn check_operands(graph: &EGraph, plan: &Plan) -> Result<()> {
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
pub fn check_buffers(graph: &EGraph, plan: &Plan) -> Result<()> {
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
                return Err(Error::Plan(format!(
                    "buffer for {} has an underivable stride on axis {axis}",
                    b.value
                )));
            }
        }
    }
    Ok(())
}

/// Clause 5.
pub fn check_effect_pinning(graph: &EGraph, plan: &Plan) -> Result<()> {
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
pub fn check_roots(graph: &EGraph, plan: &Plan) -> Result<()> {
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

/// Clause 6, extension half: an `L1::Ext` whose `lower_per_target` is empty
/// cannot run on any target and must never be selected.
pub fn check_extensions(
    graph: &EGraph,
    plan: &Plan,
    registry: Option<&OpDefRegistry>,
) -> Result<()> {
    for id in selected(plan) {
        let Op::L1(L1::Ext { def, .. }) = &graph.node(id).op else {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::LocalSearch;
    use crate::realize::testkit::{
        N, TestCost, TestPlanner, buffer, chain_graph, kmap, kscatter, new_graph, test_caps,
    };
    use fusor2_ir::extract::{ExtractBudget, Extractor};
    use fusor2_ir::ir::level1::CoopGeom;
    use std::sync::Arc;

    fn search() -> LocalSearch {
        LocalSearch::new(Arc::new(TestPlanner), test_caps())
    }

    #[test]
    fn a_seeded_plan_verifies() {
        let (g, roots) = chain_graph(4);
        let cost = TestCost::default();
        let plan = search()
            .extract(&g, &roots, &cost, ExtractBudget::default())
            .unwrap();
        verify_plan(&g, &plan).unwrap();
    }

    #[test]
    fn verify_plan_rejects_inlined_inplace() {
        let mut g = new_graph();
        let shape = [N];
        let base = buffer(&mut g, 0, &shape);
        let idx = buffer(&mut g, 1, &shape);
        let upd = buffer(&mut g, 2, &shape);
        let sc = kscatter(&mut g, base, idx, upd, &shape);
        let a = kmap(&mut g, sc, &shape, 1);
        let b = kmap(&mut g, sc, &shape, 2);
        g.add_root(a);
        g.add_root(b);
        let roots = g.roots().to_vec();
        let cost = TestCost::default();
        let mut plan = search()
            .extract(&g, &roots, &cost, ExtractBudget::default())
            .unwrap();
        verify_plan(&g, &plan).unwrap();

        // Force the pin open, as only a broken extractor could.
        plan.extraction.m.set(sc.index(), false);
        assert!(matches!(verify_plan(&g, &plan), Err(Error::Plan(_)),));
    }

    #[test]
    fn verify_plan_rejects_illegal_geom() {
        let (g, roots) = chain_graph(2);
        let cost = TestCost::default();
        let mut plan = search()
            .extract(&g, &roots, &cost, ExtractBudget::default())
            .unwrap();
        let victim = *plan.extraction.sigma.values().max().unwrap();
        plan.extraction.theta.insert(
            victim,
            SchedPoint::Coop {
                // rg * cg * 32 lanes = 32,768, far past any workgroup, and
                // bm is not a multiple of COOP_DIM * rg either.
                geom: CoopGeom {
                    bm: 3,
                    bn: 3,
                    bk: 3,
                    n_passes: 1,
                    subgroups: 1024,
                    rg: 32,
                    cg: 32,
                },
                splits: 1,
                staging: 1,
            },
        );
        let arena = TestPlanner;
        assert!(matches!(
            verify_plan_with(&g, &plan, &arena, &test_caps(), None),
            Err(Error::Plan(_))
        ));
    }

    /// Clause 7 counts the `Uniforms` block: a launch with `limit` listed
    /// bindings needs `limit + 1` storage buffers and is rejected.
    #[test]
    fn verify_plan_rejects_a_bind_group_that_forgot_the_uniform_block() {
        use fusor2_ir::extract::{BindKind, BindingPlan};

        let (g, roots) = chain_graph(2);
        let cost = TestCost::default();
        let mut plan = search()
            .extract(&g, &roots, &cost, ExtractBudget::default())
            .unwrap();
        let caps = test_caps();
        let limit = caps.limits.max_storage_buffers_per_shader_stage as usize;
        verify_plan_with(&g, &plan, &TestPlanner, &caps, None).unwrap();

        let victim = &mut plan.launches[0];
        let value = victim.bindings[0].value;
        // One short of the limit still fits: `limit - 1` operands plus the
        // block is exactly `limit`.
        victim.bindings = (0..limit - 1)
            .map(|i| BindingPlan {
                binding: i as u32 + 1,
                value,
                kind: BindKind::Read,
            })
            .collect();
        verify_plan_with(&g, &plan, &TestPlanner, &caps, None).unwrap();

        // One more is `limit + 1` storage buffers, which the device refuses.
        plan.launches[0].bindings.push(BindingPlan {
            binding: limit as u32,
            value,
            kind: BindKind::Write,
        });
        let err = verify_plan_with(&g, &plan, &TestPlanner, &caps, None).unwrap_err();
        let Error::Plan(msg) = &err else {
            panic!("{err:?}")
        };
        assert!(msg.contains("Uniforms"), "{msg}");
    }
}
