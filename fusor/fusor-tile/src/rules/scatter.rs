//! R6 — the two `Scatter` lowerings, both coexisting.

use fusor_ir::egraph::{Builder, Facts, Id, RuleTag};
use fusor_ir::ir::launch::{IndexSpace, Launch, ScatterMode, ScheduleDomain};
use fusor_ir::ir::logical::{Logical, ScatterCombine};
use fusor_ir::ir::{Level, Node, Op, OpTag};
use fusor_ir::rule;

use crate::domains::{DomainCtx, default_planner, map_domain};
use crate::rules::contract::alias;

rule!(
    SCATTER_ATOMIC,
    level = Level::Logical,
    head = OpTag::Scatter,
    tag = RuleTag::StrictlyLowering,
    apply = scatter_atomic,
);

rule!(
    SCATTER_SORT_SEGMENT,
    level = Level::Logical,
    head = OpTag::Scatter,
    tag = RuleTag::StrictlyLowering,
    apply = scatter_sort_segment,
);

struct Parts {
    axis: u32,
    combine: ScatterCombine,
    base: Id,
    idx: Id,
    upd: Id,
}

fn parts(node: &Node) -> Option<Parts> {
    match &node.op {
        Op::Logical(Logical::Scatter {
            axis,
            combine,
            base,
            idx,
            upd,
            ..
        }) => Some(Parts {
            axis: *axis,
            combine: *combine,
            base: *base,
            idx: *idx,
            upd: *upd,
        }),
        _ => None,
    }
}

fn mint(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>, mode: ScatterMode) -> Option<Id> {
    let p = parts(node)?;
    let base = f.operand(0)?;
    let idx = f.operand(1)?;
    let upd = f.operand(2)?;
    let cx = DomainCtx::new(f.caps(), default_planner());
    let space = IndexSpace::new(upd.shape.iter().copied());
    let accesses = [
        alias(p.base, base).access,
        alias(p.idx, idx).access,
        alias(p.upd, upd).access,
    ];
    let op = Launch::Scatter {
        space,
        axis: p.axis,
        mode,
        combine: p.combine,
        ops: vec![alias(p.base, base), alias(p.idx, idx), alias(p.upd, upd)],
        sched: ScheduleDomain::Map(map_domain(&upd.shape, &accesses, &cx)),
    };
    let new = b.add_launch(op).ok()?;
    b.union(id, new).ok()?;
    Some(new)
}

/// One in-place write per update. `Add` needs `atomicAdd` on f32 in
/// storage; `Set` on caller-proved-unique indices is an ordinary store and
/// needs no capability. Carries `Effect::InPlace`, so extraction pins it in
/// the materialized set — without that, inlining it into two consumers
/// applies the atomics twice.
pub fn scatter_atomic(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let p = parts(node)?;
    let legal = match p.combine {
        ScatterCombine::Set => true,
        ScatterCombine::Add => f.caps().atomic_f32,
    };
    if !legal {
        return None;
    }
    mint(b, id, node, f, ScatterMode::Atomic)
}

/// Sort the updates by destination, then reduce each segment. Always
/// legal — it needs no device capability and no bound on the destination
/// extent.
pub fn scatter_sort_segment(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    parts(node)?;
    mint(b, id, node, f, ScatterMode::SortSegment)
}
