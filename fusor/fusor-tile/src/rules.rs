//! The lowering rules that consult a schedule-domain generator: contraction
//! families, `unfuse_coop_epilogue`, and the scatter and gather lowerings.

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
    apply = tile_indexed,
);

rule!(
    TILE_SCATTER,
    level = Level::Launch,
    head = OpTag::LaunchScatter,
    tag = RuleTag::Additive,
    apply = tile_indexed,
);

/// Attach the complete legal reduction domain to a `Fold` carrying
/// [`ScheduleDomain::Point`]. Lives here because domains are filtered by the
/// exact arena footprint; an empty domain means the rule does not apply.
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
    // Neither backend lowers a promoted nest whose reduced axis is not last.
    if !vec_axes.is_empty() && *axis as usize + 1 != space.rank() {
        return None;
    }
    let k = *space.dims.get(*axis as usize)?;
    // A symbolic `Vector` slot extent is allocatable on neither backend.
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
        *sched = ScheduleDomain::Fold(dom.into());
    }
    adopt(b, id, rebuilt)
}

/// The accesses of a node's operand list: a per-lane gather forbids a
/// vectorized tiling.
fn accesses(ops: &[Operand]) -> Vec<fusor_ir::ir::launch::AccessPlan> {
    ops.iter().map(|o| o.access.clone()).collect()
}

/// Attach the elementwise tiling domain to a floor-lowered `Gather` or
/// `Scatter`, without touching `mode`. There is no `TILE_MAP`: a `Map` domain
/// as an additive alternative regresses extraction, so it is minted in place.
pub fn tile_indexed(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::Launch(
        l1 @ (Launch::Gather {
            space,
            ops,
            sched: ScheduleDomain::Point,
            ..
        }
        | Launch::Scatter {
            space,
            ops,
            sched: ScheduleDomain::Point,
            ..
        }),
    ) = &node.op
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
    if let Launch::Gather { sched, .. } | Launch::Scatter { sched, .. } = &mut rebuilt {
        *sched = ScheduleDomain::Map(dom.into());
    }
    adopt(b, id, rebuilt)
}

/// Add `op` as an alternative in `id`'s class.
pub(crate) fn adopt(b: &mut Builder<'_>, id: Id, op: Launch) -> Option<Id> {
    let new = b.add_launch(op).ok()?;
    b.union(id, new).ok()?;
    Some(new)
}

/// Every rule `fusor-tile` owns, in a fixed order for reproducibility.
pub static TILE_RULES: &[Rule] = &[
    TILE_FOLD,
    // `Map` is deliberately absent; see `tile_indexed`.
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
