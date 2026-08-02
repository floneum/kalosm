//! R6 — the four `Scatter` lowerings, all four coexisting.
//!
//! At the trainer's batch-128 / 768-unit / K=3 shape, `OneHotContract`
//! prices at 1.2 GB of traffic against `WgPrivateMerge`'s private
//! accumulator — so it survives only as the candidate the cost model
//! rejects, not as a candidate a rule vetoes. The trainer's host-side
//! three-level sorted gather-and-sum, its `ScatterShape` padding and its
//! ~0.9 ms/batch host cost all delete.
//!
//! Owned by W4.

use fusor2_ir::egraph::{Builder, Facts, Id, RuleTag};
use fusor2_ir::ir::level0::{L0, ScatterCombine};
use fusor2_ir::ir::level1::{IndexSpace, L1, ScatterMode, ScheduleDomain};
use fusor2_ir::ir::{Level, Node, Op, OpTag};
use fusor2_ir::rule;
use fusor2_ir::shape::Dim;

use crate::domains::{DomainCtx, default_planner, map_domain};
use crate::rules::contract::alias;

rule!(
    SCATTER_ATOMIC,
    level = Level::L0,
    head = OpTag::Scatter,
    tag = RuleTag::StrictlyLowering,
    apply = scatter_atomic,
);

rule!(
    SCATTER_SORT_SEGMENT,
    level = Level::L0,
    head = OpTag::Scatter,
    tag = RuleTag::StrictlyLowering,
    apply = scatter_sort_segment,
);

rule!(
    SCATTER_WG_PRIVATE_MERGE,
    level = Level::L0,
    head = OpTag::Scatter,
    tag = RuleTag::StrictlyLowering,
    apply = scatter_wg_private_merge,
);

rule!(
    SCATTER_ONE_HOT_CONTRACT,
    level = Level::L0,
    head = OpTag::Scatter,
    tag = RuleTag::StrictlyLowering,
    apply = scatter_one_hot_contract,
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
        Op::L0(L0::Scatter {
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

/// Extent of the scattered-into axis, when it is decidable.
fn rows(f: &Facts<'_>, axis: u32) -> Option<Dim> {
    f.operand(0)?.shape.get(axis as usize).copied()
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
    let op = L1::KScatter {
        space,
        axis: p.axis,
        mode,
        combine: p.combine,
        ops: vec![alias(p.base, base), alias(p.idx, idx), alias(p.upd, upd)],
        sched: ScheduleDomain::Map(map_domain(&upd.shape, &accesses, &cx)),
    };
    let new = b.add_l1(op).ok()?;
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

/// Accumulate into a workgroup-private histogram, then merge. Legal when
/// the whole destination axis fits threadgroup memory.
pub fn scatter_wg_private_merge(
    b: &mut Builder<'_>,
    id: Id,
    node: &Node,
    f: &Facts<'_>,
) -> Option<Id> {
    let p = parts(node)?;
    if p.combine != ScatterCombine::Add {
        return None;
    }
    let rows = rows(f, p.axis)?.as_const()?;
    let elem_bytes = f.operand(0)?.dtype.byte_size();
    let bytes = rows.checked_mul(elem_bytes)?;
    if bytes > u64::from(f.caps().limits.max_compute_workgroup_storage_size) {
        return None;
    }
    mint(b, id, node, f, ScatterMode::WgPrivateMerge)
}

/// A one-hot einsum: `one_hot(idx)^T @ upd`. Legal whenever the
/// destination extent is known, which is what lets the contraction be
/// shaped at all. Almost never selected — it exists so the cost model has
/// something to reject rather than a rule having something to veto.
pub fn scatter_one_hot_contract(
    b: &mut Builder<'_>,
    id: Id,
    node: &Node,
    f: &Facts<'_>,
) -> Option<Id> {
    let p = parts(node)?;
    if p.combine != ScatterCombine::Add {
        return None;
    }
    rows(f, p.axis)?.as_const()?;
    mint(b, id, node, f, ScatterMode::OneHotContract)
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
                Some(L1::KScatter { mode, .. }) => Some(mode),
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
    fn four_lowerings_on_capable_device() {
        let (fx, s) = trainer_scatter(1024, apple_caps());
        let modes = modes(&fx, s);
        assert_eq!(modes.len(), 4, "{modes:?}");
        for want in [
            ScatterMode::Atomic,
            ScatterMode::SortSegment,
            ScatterMode::WgPrivateMerge,
            ScatterMode::OneHotContract,
        ] {
            assert!(modes.contains(&want), "{want:?} missing from {modes:?}");
        }
    }

    #[test]
    fn wg_private_merge_declined_when_too_wide() {
        let (fx, s) = trainer_scatter(65536, apple_caps());
        let modes = modes(&fx, s);
        assert_eq!(modes.len(), 3, "{modes:?}");
        assert!(!modes.contains(&ScatterMode::WgPrivateMerge));
    }

    #[test]
    fn one_hot_survives() {
        // 1.2 GB of traffic at the trainer's shape, and still a candidate:
        // rejecting it is the cost model's job.
        let (fx, s) = trainer_scatter(1024, apple_caps());
        assert!(modes(&fx, s).contains(&ScatterMode::OneHotContract));
    }

    #[test]
    fn no_atomic_without_the_capability() {
        let mut caps = apple_caps();
        caps.atomic_f32 = false;
        let (fx, s) = trainer_scatter(1024, caps);
        let modes = modes(&fx, s);
        assert!(!modes.contains(&ScatterMode::Atomic), "{modes:?}");
        assert_eq!(modes.len(), 3);
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
        assert!(!modes.contains(&ScatterMode::WgPrivateMerge));
        assert!(!modes.contains(&ScatterMode::OneHotContract));
    }
}
