//! The lowering rules that consult a schedule-domain generator: the
//! order-free contraction family rules plus `unfuse_coop_epilogue`, the
//! four `Scatter` lowerings, the two gather lowerings.
//!
//! All the families coexist in one chain and compete on cost.

pub mod contract;
pub mod gather;
pub mod scatter;

use fusor_ir::egraph::{Builder, Facts, Id, Rule, RuleTag};
use fusor_ir::ir::launch::{Launch, Operand, ScheduleDomain};
use fusor_ir::ir::{Level, Node, Op, OpTag};
use fusor_ir::rule;

use crate::domains::{DomainCtx, default_planner, fold_domain_for, map_domain};

rule!(
    TILE_FOLD,
    level = Level::Launch,
    head = OpTag::LaunchFold,
    tag = RuleTag::Additive,
    apply = tile_fold,
);

rule!(
    TILE_GATHER,
    level = Level::Launch,
    head = OpTag::LaunchGather,
    tag = RuleTag::Additive,
    apply = tile_gather,
);

rule!(
    TILE_SCATTER,
    level = Level::Launch,
    head = OpTag::LaunchScatter,
    tag = RuleTag::Additive,
    apply = tile_scatter,
);

/// Attach the complete legal reduction domain to a `Fold` that arrived
/// carrying [`ScheduleDomain::Point`].
///
/// The floor lowering (`fusor-ir`) cannot generate one: schedule domains are
/// filtered by the exact arena footprint, which lives here.
///
/// The domain is generated for this carrier's lane count, so a wide
/// accumulator is filtered by workgroup storage rather than admitted and
/// crashed at `verify_plan`. An empty domain means the rule does not apply,
/// never that the node is broken.
///
/// Promoted folds included: `space = free.. ++ vec.. ++ [reduced]` is a fold
/// like any other here; both backends lower it per promoted position.
pub fn tile_fold(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::Launch(l1) = &node.op else {
        return None;
    };
    let Launch::Fold {
        space,
        axis,
        vec_axes,
        carrier,
        acc,
        sched: ScheduleDomain::Point,
        ..
    } = l1
    else {
        return None;
    };
    // Neither backend lowers a promoted nest whose reduced axis is not last:
    // the address arithmetic reads one output row as
    // `vec_extent * axis_extent` consecutive elements. Pricing a schedule
    // point for it would make extraction prefer a plan that fails at
    // lowering.
    if !vec_axes.is_empty() && *axis as usize + 1 != space.rank() {
        return None;
    }
    let k = *space.dims.get(*axis as usize)?;
    // A symbolic `Vector` slot extent is allocatable on neither backend; the
    // rule declines rather than guessing a footprint.
    let lanes = carrier.lanes()?;
    let dom = fold_domain_for(
        k,
        lanes,
        acc.byte_size(),
        &DomainCtx::new(f.caps(), default_planner()),
    );
    if dom.strategies.is_empty() {
        return None;
    }

    let mut rebuilt = l1.clone();
    if let Launch::Fold { sched, .. } = &mut rebuilt {
        *sched = ScheduleDomain::Fold(dom);
    }
    let new = b.add_launch(rebuilt).ok()?;
    b.union(id, new).ok()?;
    Some(new)
}

/// The accesses of a node's operand list, as the map-domain generator reads
/// them: a per-lane gather has no vector load to widen into, so it forbids a
/// vectorized tiling. Legality, not preference.
fn accesses(ops: &[Operand]) -> Vec<fusor_ir::ir::launch::AccessPlan> {
    ops.iter().map(|o| o.access.clone()).collect()
}

/// Attach the elementwise tiling domain to a floor-lowered `Gather`,
/// without touching `mode`.
///
/// `gather::GATHER_*` mint a mode and a domain together; splitting them makes
/// both late decisions.
///
/// There is deliberately no `TILE_MAP` beside this: a `Map` domain minted as
/// an additive alternative measurably regresses extraction, and has to be
/// attached where the node is minted (`lower_floor.rs`) so it replaces
/// `ScheduleDomain::Point` instead of competing with it.
pub fn tile_gather(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::Launch(l1) = &node.op else {
        return None;
    };
    let Launch::Gather {
        space,
        ops,
        sched: ScheduleDomain::Point,
        ..
    } = l1
    else {
        return None;
    };
    let dom = map_domain(
        &space.dims,
        &accesses(ops),
        &DomainCtx::new(f.caps(), default_planner()),
    );
    if dom.tilings.len() <= 1 {
        return None;
    }
    let mut rebuilt = l1.clone();
    if let Launch::Gather { sched, .. } = &mut rebuilt {
        *sched = ScheduleDomain::Map(dom);
    }
    let new = b.add_launch(rebuilt).ok()?;
    b.union(id, new).ok()?;
    Some(new)
}

/// Attach the elementwise tiling domain to a floor-lowered `Scatter`,
/// without touching `mode`.
///
/// Same split as [`tile_gather`]: this only stops the floor's mode from
/// being the one alternative with no schedule.
pub fn tile_scatter(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::Launch(l1) = &node.op else {
        return None;
    };
    let Launch::Scatter {
        space,
        ops,
        sched: ScheduleDomain::Point,
        ..
    } = l1
    else {
        return None;
    };
    let dom = map_domain(
        &space.dims,
        &accesses(ops),
        &DomainCtx::new(f.caps(), default_planner()),
    );
    if dom.tilings.len() <= 1 {
        return None;
    }
    let mut rebuilt = l1.clone();
    if let Launch::Scatter { sched, .. } = &mut rebuilt {
        *sched = ScheduleDomain::Map(dom);
    }
    let new = b.add_launch(rebuilt).ok()?;
    b.union(id, new).ok()?;
    Some(new)
}

/// Every rule `fusor-tile` owns, in a fixed declaration order. Order carries
/// no semantics; it exists only so a run is reproducible.
pub static TILE_RULES: &[Rule] = &[
    TILE_FOLD,
    // `Map` is deliberately absent — see the note above `tile_gather`.
    TILE_GATHER,
    TILE_SCATTER,
    contract::LOWER_COOP,
    contract::LOWER_SGEMM,
    contract::LOWER_SGEMV,
    contract::LOWER_GENERIC,
    contract::UNFUSE_COOP_EPILOGUE,
    // scatter: two coexisting lowerings
    scatter::SCATTER_ATOMIC,
    scatter::SCATTER_SORT_SEGMENT,
    // gather: two coexisting lowerings
    gather::GATHER_ROW_PER_GROUP,
    gather::GATHER_QUANTIZED_ROWS,
];

/// The name `fusor-tile`'s rule table has always been exported under.
pub static SCHED_RULES: &[Rule] = TILE_RULES;
