//! R6 — the two `Scatter` lowerings, both coexisting.

use fusor2_ir::egraph::{Builder, Facts, Id, RuleTag};
use fusor2_ir::ir::logical::{Logical, ScatterCombine};
use fusor2_ir::ir::launch::{IndexSpace, Launch, ScatterMode, ScheduleDomain};
use fusor2_ir::ir::{Level, Node, Op, OpTag};
use fusor2_ir::rule;

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

fn mint(
    b: &mut Builder<'_>,
    id: Id,
    node: &Node,
    f: &Facts<'_>,
    mode: ScatterMode,
) -> Option<Id> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domains::testing::apple_caps;
    use crate::rules::TILE_RULES;
    use crate::rules::testing::{Fixture, l1_of};
    use fusor2_ir::dtype::Dtype;

    fn modes(fx: &Fixture, id: Id) -> Vec<ScatterMode> {
        let mut out: Vec<ScatterMode> = fx
            .chain(id)
            .into_iter()
            .filter_map(|m| match l1_of(fx, m) {
                Some(Launch::Scatter { mode, .. }) => Some(mode),
                _ => None,
            })
            .collect();
        out.sort_by_key(|m| format!("{m:?}"));
        out
    }

    /// The trainer's embedding-gradient shape: 1024 bins, 24 f32 wide, on
    /// a device with f32 atomics and 32 KiB of threadgroup memory.
    fn trainer_scatter(rows: u64, caps: fusor2_ir::device::Caps) -> (Fixture, Id) {
        let mut fx = Fixture::new(caps);
        let base = fx.buffer(Dtype::F32, &[rows, 24]);
        let idx = fx.buffer(Dtype::U32, &[4096]);
        let upd = fx.buffer(Dtype::F32, &[4096, 24]);
        let s = fx.scatter(0, ScatterCombine::Add, base, idx, upd);
        fx.apply_all(TILE_RULES, s);
        (fx, s)
    }

    #[test]
    fn both_lowerings_on_capable_device() {
        let (fx, s) = trainer_scatter(1024, apple_caps());
        let modes = modes(&fx, s);
        assert_eq!(modes.len(), 2, "{modes:?}");
        for want in [ScatterMode::Atomic, ScatterMode::SortSegment] {
            assert!(modes.contains(&want), "{want:?} missing from {modes:?}");
        }
    }

    #[test]
    fn no_atomic_without_the_capability() {
        let mut caps = apple_caps();
        caps.atomic_f32 = false;
        let (fx, s) = trainer_scatter(1024, caps);
        let modes = modes(&fx, s);
        assert!(!modes.contains(&ScatterMode::Atomic), "{modes:?}");
        assert_eq!(modes.len(), 1);
    }

    #[test]
    fn set_combine_mints_two_lowerings() {
        let mut fx = Fixture::new(apple_caps());
        let base = fx.buffer(Dtype::F32, &[1024, 24]);
        let idx = fx.buffer(Dtype::U32, &[512]);
        let upd = fx.buffer(Dtype::F32, &[512, 24]);
        let s = fx.scatter(0, ScatterCombine::Set, base, idx, upd);
        fx.apply_all(TILE_RULES, s);
        let modes = modes(&fx, s);
        assert_eq!(modes.len(), 2, "{modes:?}");
        assert!(modes.contains(&ScatterMode::Atomic));
        assert!(modes.contains(&ScatterMode::SortSegment));
    }

    #[test]
    fn a_symbolic_destination_keeps_the_unbounded_lowerings() {
        use fusor2_ir::shape::{Dim, SymId};
        let mut fx = Fixture::new(apple_caps());
        let base = fx.buffer_dims(Dtype::F32, &[Dim::Sym(SymId(1)), Dim::Const(24)]);
        let idx = fx.buffer(Dtype::U32, &[512]);
        let upd = fx.buffer(Dtype::F32, &[512, 24]);
        let s = fx.scatter(0, ScatterCombine::Add, base, idx, upd);
        fx.apply_all(TILE_RULES, s);
        let modes = modes(&fx, s);
        assert!(modes.contains(&ScatterMode::Atomic));
        assert!(modes.contains(&ScatterMode::SortSegment));
    }
}
