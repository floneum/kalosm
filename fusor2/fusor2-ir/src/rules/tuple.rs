//! TUPLE — two reduction nests over the same iteration space and the same
//! reduction axis are one nest over the concatenated carrier.
//!
//! ```text
//! < KFold{C1, a, ops1}, KFold{C2, a, ops2} >
//!   ==  slot views of  KFold{ C1 (x) C2, a, ops1 u ops2 }
//! ```
//!
//! `(x)` is [`Carrier::tuple`], which deduplicates slots, so joining `(m,l)`
//! with `(m,o)` yields three slots.
//!
//! Value-preserving: every slot folds in the order it folded alone, so the law
//! carries no `reassoc` guard and is legal under [`NumericContract::STRICT`].
//!
//! The rule is consumer-rooted: [`TUPLE`] at a `KMap` consumer,
//! [`TUPLE_SIBLING`] at a `KFold` consumer.
//!
//! Acyclicity guard: neither nest's operand closure may transitively reach the
//! other's result, checked through `Op::Union` chains as well as `children`.
//! TUPLE never discharges a carried dependence itself; that is RETARGET's job.
//!
//! [`fold_view`] does not look through a `post` epilogue.

use crate::carrier::{ArgRemap, Carrier, Tupled, map_args, probes_for, retype_args};
use crate::device::Caps;
use crate::dtype::{Dtype, NumericContract};
use crate::egraph::{Builder, Facts, Id, RuleTag, ViewSpine};
use crate::ir::level0::L0;
use crate::ir::level1::{IndexSpace, L1, Operand, ScheduleDomain};
use crate::ir::{Level, Node, Op, OpTag};
use crate::rule;
use crate::rules::alias_operand_of;
use crate::scalar::ScalarExpr;
use crate::shape::{BoundsProof, Dim, StrideSpec};
use rustc_hash::FxHashSet;
use smallvec::SmallVec;

rule!(
    TUPLE,
    level = Level::L1,
    head = OpTag::KMap,
    tag = RuleTag::Additive,
    apply = tuple_at_consumer,
);

rule!(
    TUPLE_SIBLING,
    level = Level::L1,
    head = OpTag::KFold,
    tag = RuleTag::Additive,
    apply = tuple_siblings,
);

/// The consumer rooting at a `KMap`.
pub fn tuple_at_consumer(b: &mut Builder<'_>, id: Id, n: &Node, f: &Facts<'_>) -> Option<Id> {
    tuple_at(b, id, n, f)
}

/// The consumer rooting at a `KFold` — a reducing nest that reads two reducing
/// nests.
pub fn tuple_siblings(b: &mut Builder<'_>, id: Id, n: &Node, f: &Facts<'_>) -> Option<Id> {
    tuple_at(b, id, n, f)
}

/// The private accumulator budget one invocation may hold, in bytes: 256 f32
/// registers per lane. The guard is
/// `carrier.lanes() * acc.bytes() <= private_acc_bytes`; a wider carrier is
/// unschedulable.
const fn private_acc_bytes(_caps: &Caps) -> u64 {
    1024
}

/// A reduction nest, normalized out of whichever spelling the operand named.
///
/// Equality in this e-graph is not congruent, so an `L0::Fold` and the
/// `L1::KFold` it lowered to are one class while a consumer's operand still
/// names whichever id the frontend built. Normalizing retypes the lift to the
/// operand dtype rather than replacing it, matching `lower_fold`, so the two
/// spellings produce one hash-consed joint node.
#[derive(Clone, Debug, PartialEq, Eq)]
struct FoldView {
    /// The id the operand named, which is the id the join unions against.
    id: Id,
    space: IndexSpace,
    axis: u32,
    vec_axes: SmallVec<[u32; 2]>,
    carrier: Carrier,
    acc: Dtype,
    post: SmallVec<[ScalarExpr; 4]>,
    ops: Vec<Operand>,
    sched: ScheduleDomain,
}

impl FoldView {
    /// The domain this nest's own expressions are written against.
    fn iter_space(&self) -> IndexSpace {
        IndexSpace::new(
            self.space
                .dims
                .iter()
                .enumerate()
                .filter(|(i, _)| !self.vec_axes.contains(&(*i as u32)))
                .map(|(_, d)| *d),
        )
    }

    /// The reduced axis's index in [`Self::iter_space`], the number both
    /// spellings of one reduction agree on. `vec_axes` is the contiguous block
    /// immediately before `axis`, so subtracting its length is the whole
    /// renumbering.
    fn reduced_iter_axis(&self) -> Option<u32> {
        self.axis.checked_sub(u32::try_from(self.vec_axes.len()).ok()?)
    }

    /// The output dims before the carrier axis: `space` minus the reduced axis
    /// and minus every accumulator-resident axis.
    fn base_dims(&self) -> SmallVec<[Dim; 6]> {
        self.space
            .dims
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != self.axis as usize && !self.vec_axes.contains(&(*i as u32)))
            .map(|(_, d)| *d)
            .collect()
    }
}

/// The same nest, whichever id spells it. Ignores `id` (the L0-versus-L1
/// spelling the acyclicity walk must not miss) and `sched` (a schedule domain
/// is not a value).
fn same_nest(a: &FoldView, b: &FoldView) -> bool {
    a.space == b.space
        && a.axis == b.axis
        && a.vec_axes == b.vec_axes
        && a.carrier == b.carrier
        && a.acc == b.acc
        && a.post == b.post
        && a.ops == b.ops
}

/// Read `id` as a reduction nest, in either spelling. Does not look through a
/// `post` epilogue.
fn fold_view(b: &Builder<'_>, id: Id) -> Option<FoldView> {
    let v = bare_fold_view(b, id)?;
    // The readback the join unions against is a strided view of the joint,
    // typed `acc` and shaped like the nest's output. A spelling whose facts
    // disagree is not redirectable.
    let f = b.facts_of(id);
    let mut want = v.base_dims();
    if let Some(d) = v.carrier.out_dim()? {
        want.push(d);
    }
    if f.dtype != v.acc
        || f.shape.len() != want.len()
        || !f.shape.iter().zip(want.iter()).all(|(a, c)| a.known_eq(*c))
    {
        return None;
    }
    Some(v)
}

/// The nest itself, in either spelling.
fn bare_fold_view(b: &Builder<'_>, id: Id) -> Option<FoldView> {
    match b.node(id).op.clone() {
        Op::L1(L1::KFold {
            space,
            axis,
            vec_axes,
            carrier,
            acc,
            post,
            ops,
            sched,
        }) => Some(FoldView {
            id,
            space,
            axis,
            vec_axes,
            carrier,
            acc,
            post,
            ops,
            sched,
        }),
        Op::L0(L0::Fold {
            carrier,
            axis,
            acc,
            ins,
        }) => {
            let src = *ins.first()?;
            let in_shape = b.facts_of(src).shape.clone();
            let dtype = b.facts_of(src).dtype;
            let width = carrier.width();
            let lift: SmallVec<[ScalarExpr; 4]> =
                carrier.lift.iter().map(|e| retype_args(e, dtype)).collect();
            Some(FoldView {
                id,
                space: IndexSpace::new(in_shape.iter().copied()),
                axis,
                vec_axes: SmallVec::new(),
                carrier: carrier.with_lift(lift),
                acc,
                post: (0..width).map(|i| ScalarExpr::arg(i as u32, acc)).collect(),
                ops: ins
                    .iter()
                    .map(|x| alias_operand_of(*x, &in_shape))
                    .collect(),
                sched: crate::rules::lower_floor::floor_sched(),
            })
        }
        _ => None,
    }
}

/// Which operand slots of the rewritten consumer read what.
struct Rewire {
    at: [(usize, Id); 2],
}

/// Each side's readback view of the minted joint nest.
struct Joint {
    lhs_read: Id,
    rhs_read: Id,
}

fn tuple_at(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
    let srcs: SmallVec<[Id; 4]> = match &node.op {
        Op::L1(L1::KMap { ops, .. }) | Op::L1(L1::KFold { ops, .. }) => {
            ops.iter().map(|o| o.src).collect()
        }
        _ => return None,
    };
    let rewire = join_pair(b, &srcs)?;
    let rebuilt = match node.op.clone() {
        Op::L1(L1::KMap {
            space,
            body,
            mut ops,
            sched,
        }) => {
            for (slot, src) in rewire.at {
                ops.get_mut(slot)?.src = src;
            }
            b.add_l1(L1::KMap {
                space,
                body,
                ops,
                sched,
            })
            .ok()?
        }
        Op::L1(L1::KFold {
            space,
            axis,
            vec_axes,
            carrier,
            acc,
            post,
            mut ops,
            sched,
        }) => {
            for (slot, src) in rewire.at {
                ops.get_mut(slot)?.src = src;
            }
            b.add_l1(L1::KFold {
                space,
                axis,
                vec_axes,
                carrier,
                acc,
                post,
                ops,
                sched,
            })
            .ok()?
        }
        _ => return None,
    };
    b.union(id, rebuilt).ok()
}

/// The first pair of operand slots reading joinable nests, in a deterministic
/// scan. One firing joins one pair; the rewritten consumer is a fresh node the
/// driver re-queues, so `F` nests cost `F-1` firings.
fn join_pair(b: &mut Builder<'_>, ops: &[Id]) -> Option<Rewire> {
    for i in 0..ops.len() {
        let si = b.trace_pure_views(ops[i]);
        let Some(vi) = fold_view(b, si.base) else {
            continue;
        };
        for (j, opj) in ops.iter().enumerate().skip(i + 1) {
            let sj = b.trace_pure_views(*opj);
            let Some(vj) = fold_view(b, sj.base) else {
                continue;
            };
            // Deterministic join order: the smaller id is the left carrier, so
            // operand slot order cannot change the slot order or the
            // `PlanHash`.
            let swapped = vj.id.0 < vi.id.0;
            let (lhs, rhs) = if swapped { (&vj, &vi) } else { (&vi, &vj) };
            let Some(joint) = join(b, lhs, rhs) else {
                continue;
            };
            let (li, ri) = if swapped {
                (joint.rhs_read, joint.lhs_read)
            } else {
                (joint.lhs_read, joint.rhs_read)
            };
            let a = rebuild_spine(b, &si, li)?;
            let c = rebuild_spine(b, &sj, ri)?;
            return Some(Rewire {
                at: [(i, a), (j, c)],
            });
        }
    }
    None
}

/// The law proper. Every legality check and derived value is computed before
/// the first `add`, so a declined join leaves no orphan nodes.
///
/// The sides are compared on `axis - vec_axes.len()`, the reduced axis's index
/// in [`FoldView::iter_space`], not on `axis`: `axis` indexes `space`, which a
/// promoted nest has widened with its carrier axes.
///
/// `vec_axes` equality is required only between two nests that are both
/// promoted — two different promotions of one space are two carrier geometries
/// with no common nest. Between a promoted nest and an unpromoted one the joint
/// takes the promoted side's `space`, `axis` and `vec_axes`, and [`widen_ops`]
/// restates the unpromoted side's operands with stride 0 at each carrier axis.
/// `verify_l1::check_vec_axes` refuses a `Scalar` slot whose lift reads an
/// operand that varies along a promoted axis, and stride 0 is what proves it
/// does not.
fn join(b: &mut Builder<'_>, f1: &FoldView, f2: &FoldView) -> Option<Joint> {
    if f1.id == f2.id || f1.acc != f2.acc {
        return None;
    }
    if f1.reduced_iter_axis()? != f2.reduced_iter_axis()? {
        return None;
    }
    // `covers` is a prefix test, so one direction is not enough.
    let (e1, e2) = (f1.iter_space(), f2.iter_space());
    if !(e1.covers(&e2) && e2.covers(&e1)) {
        return None;
    }
    if !f1
        .space
        .dims
        .get(f1.axis as usize)?
        .known_eq(*f2.space.dims.get(f2.axis as usize)?)
    {
        return None;
    }
    // Which side's carrier geometry the joint is minted in.
    let host = promotion_host(f1, f2)?;
    // One joint node carries one contract: adopting either side's would
    // rewrite the other's rounding.
    if b.facts_of(f1.id).numeric != b.facts_of(f2.id).numeric {
        return None;
    }
    // `sched` is unguarded and the joint takes neither side's: it is minted at
    // the floor `lower_fold` mints and the schedule rules expand it.

    // The joint reads both operand lists and is unioned into both classes, so
    // a realized DAG has a cycle exactly when some unified operand reaches
    // either result. A carried dependence is RETARGET's to discharge.
    let (ops, remap) = unify_ops(
        &widen_ops(f1, host)?,
        &widen_ops(f2, host)?,
    )?;
    let srcs: Vec<Id> = ops.iter().map(|o| o.src).collect();
    if reaches_either(b, &srcs, f1, f2) {
        return None;
    }

    let t: Tupled = f1.carrier.tuple(&f2.carrier, &remap);
    // Every `Vector` extent must be `Dim::Const`: a symbolic private-array
    // extent is allocatable on neither backend.
    let lanes = t.carrier.lanes()?;
    let bytes = lanes.checked_mul(f1.acc.byte_size())?;
    if bytes > private_acc_bytes(b.caps()) {
        return None;
    }
    // The obligation every carrier owes; a botched slot renumbering fails it.
    if !t.carrier.identity_closed(probes_for(f1.acc)) {
        return None;
    }
    // The rewritten nest's contract is the meet over the unified operand list,
    // which can be stricter than either side's.
    let joint_numeric = ops.iter().fold(NumericContract::RELAXED, |acc, o| {
        acc.meet(b.facts_of(o.src).numeric)
    });
    if f1.acc.accum_bits() < joint_numeric.min_accum_bits {
        return None;
    }
    let post = joint_post(f1, f2, &t)?;

    // Each side's slots must occupy one contiguous lane range of the joint
    // carrier axis, or its value is not a strided view of the joint.
    let lhs_range = lane_range(&t.carrier, &t.lhs)?;
    let rhs_range = lane_range(&t.carrier, &t.rhs)?;
    // The joint's free dims are the host's; the readbacks are views of the
    // joint, so they are spelled with the dims it was minted at.
    let base = host.base_dims();
    let joint_axis = t.carrier.out_dim()?;
    let l_out = f1.carrier.out_dim()?;
    let r_out = f2.carrier.out_dim()?;

    let joint = crate::rules::lower_floor::floor_fold(
        b,
        host.space.clone(),
        host.axis,
        host.vec_axes.clone(),
        t.carrier,
        f1.acc,
        post,
        ops,
    )?;
    let lhs_read = slot_view(b, joint, &base, joint_axis, l_out, lhs_range)?;
    let rhs_read = slot_view(b, joint, &base, joint_axis, r_out, rhs_range)?;
    // Both sides are redirected; redirecting one leaves extraction running two
    // nests.
    b.union(f1.id, lhs_read).ok()?;
    b.union(f2.id, rhs_read).ok()?;
    Some(Joint { lhs_read, rhs_read })
}

/// Which side's carrier geometry the joint is minted in, or `None` when there
/// is no single nest holding both.
///
/// Equal `vec_axes` takes the left side. When exactly one side is promoted, it
/// hosts. Two different promotions decline.
fn promotion_host<'v>(f1: &'v FoldView, f2: &'v FoldView) -> Option<&'v FoldView> {
    if f1.vec_axes == f2.vec_axes {
        // The promoted extents must agree, or the two carriers span different
        // numbers of positions.
        for &v in &f1.vec_axes {
            let (d1, d2) = (
                f1.space.dims.get(v as usize)?,
                f2.space.dims.get(v as usize)?,
            );
            if !d1.known_eq(*d2) {
                return None;
            }
        }
        return Some(f1);
    }
    match (f1.vec_axes.is_empty(), f2.vec_axes.is_empty()) {
        (false, true) => Some(f1),
        (true, false) => Some(f2),
        _ => None,
    }
}

/// One side's operands restated over the host's space.
///
/// The host's operands are already stated there and ride through untouched. An
/// unpromoted guest gets stride 0 at every carrier axis the host added, which
/// is what `verify_l1::check_vec_axes` demands of a `Scalar` slot's operands.
fn widen_ops(side: &FoldView, host: &FoldView) -> Option<Vec<Operand>> {
    if side.vec_axes == host.vec_axes {
        return Some(side.ops.clone());
    }
    side.ops
        .iter()
        .map(|o| {
            let groups = crate::rules::fusion::widen_groups(
                &crate::rules::fusion::operand_groups(o)?,
                &host.space,
                &host.vec_axes,
            )?;
            crate::rules::fusion::operand_from_groups(o, &groups, &host.space)
        })
        .collect()
}

/// The unified operand list plus how the right side's `Arg`s renumber onto it.
fn unify_ops(lhs: &[Operand], rhs: &[Operand]) -> Option<(Vec<Operand>, ArgRemap)> {
    let mut ops = lhs.to_vec();
    let mut map: SmallVec<[u32; 4]> = SmallVec::new();
    for o in rhs {
        match ops.iter().position(|p| same_read(p, o)) {
            Some(k) => map.push(u32::try_from(k).ok()?),
            None => {
                map.push(u32::try_from(ops.len()).ok()?);
                ops.push(o.clone());
            }
        }
    }
    Some((ops, ArgRemap { map }))
}

/// Two edges read the same elements. Deduplication is an assertion about
/// elements, not syntax, so the address maps must agree; `address_map` returns
/// `None` on a `Dim::Sym` extent or a `u32` overflow and both edges are kept.
fn same_read(a: &Operand, b: &Operand) -> bool {
    a == b && matches!((a.address_map(), b.address_map()), (Some(x), Some(y)) if x == y)
}

/// Whether either nest's result is transitively reachable from `from`.
fn reaches_either(b: &Builder<'_>, from: &[Id], f1: &FoldView, f2: &FoldView) -> bool {
    let floor = f1.id.0.min(f2.id.0);
    let mut seen: FxHashSet<Id> = FxHashSet::default();
    let mut stack: Vec<Id> = from.to_vec();
    while let Some(cur) = stack.pop() {
        // Every edge points at a strictly smaller id, `Op::Union`'s two
        // alternatives included, so nothing below the lower of the two nests
        // can reach either.
        if cur.0 < floor || !seen.insert(cur) {
            continue;
        }
        if cur == f1.id || cur == f2.id {
            return true;
        }
        // The L0 and L1 spellings of one nest are two ids in one class, and a
        // class member is not reachable from the id an operand named. Compare
        // the normalized nest, not the id.
        if matches!(b.node(cur).op.tag(), OpTag::Fold | OpTag::KFold)
            && let Some(v) = fold_view(b, cur)
            && (same_nest(&v, f1) || same_nest(&v, f2))
        {
            return true;
        }
        stack.extend(b.node(cur).children.iter().copied());
    }
    false
}

/// One post expression per joint slot. A deduplicated slot carries one post, so
/// the two sides must agree on it.
fn joint_post(f1: &FoldView, f2: &FoldView, t: &Tupled) -> Option<SmallVec<[ScalarExpr; 4]>> {
    let w = t.carrier.width();
    let ns = f1.carrier.width();
    let mut post: SmallVec<[ScalarExpr; 4]> = SmallVec::with_capacity(w);
    for k in 0..ns {
        post.push(f1.post.get(k)?.clone());
    }
    for k in ns..w {
        let j = t.rhs.iter().position(|&p| p as usize == k)?;
        post.push(renumber_slots(f2.post.get(j)?, &t.rhs)?);
    }
    for (j, &k) in t.rhs.iter().enumerate() {
        if (k as usize) < ns && renumber_slots(f2.post.get(j)?, &t.rhs)? != post[k as usize] {
            return None;
        }
    }
    Some(post)
}

/// Renumber an expression written over one side's slots onto the joint's.
fn renumber_slots(e: &ScalarExpr, map: &[u8]) -> Option<ScalarExpr> {
    let bad = std::cell::Cell::new(false);
    let out = map_args(e, &|i| match map.get(i as usize) {
        Some(&k) => u32::from(k),
        None => {
            bad.set(true);
            0
        }
    });
    (!bad.get()).then_some(out)
}

/// The lane range one side's slots occupy, or `None` when they are not one
/// contiguous run in order.
fn lane_range(c: &Carrier, slots: &[u8]) -> Option<(u64, u64)> {
    let mut it = slots.iter();
    let first = *it.next()?;
    let start = c.slot_offset(first as usize)?;
    let mut cur = start.checked_add(c.slots.get(first as usize)?.lanes()?)?;
    for &k in it {
        if c.slot_offset(k as usize)? != cur {
            return None;
        }
        cur = cur.checked_add(c.slots.get(k as usize)?.lanes()?)?;
    }
    Some((start, cur - start))
}

/// One side's readback: a `Restride` narrowing the joint carrier axis to that
/// side's lanes, plus — for a side that had no carrier axis of its own — the
/// unit-axis `Restride` that drops it.
fn slot_view(
    b: &mut Builder<'_>,
    joint: Id,
    base: &[Dim],
    joint_axis: Option<Dim>,
    side_axis: Option<Dim>,
    range: (u64, u64),
) -> Option<Id> {
    let (start, len) = range;
    let Some(joint_lanes) = joint_axis else {
        // The joint carrier is one scalar slot, so it appended no axis and the
        // side has to be that same slot.
        return (side_axis.is_none() && range == (0, 1)).then_some(joint);
    };
    if side_axis == joint_axis && start == 0 && joint_lanes.known_eq(Dim::Const(len)) {
        return Some(joint);
    }
    let r = u32::try_from(base.len()).ok()?;
    let mut specs: SmallVec<[StrideSpec; 6]> = (0..r)
        .map(|j| StrideSpec::dim(j, base[j as usize]))
        .collect();
    specs.push(StrideSpec::dim(r, Dim::Const(len)).with_offset(Dim::Const(start)));
    let narrowed = b
        .add_l0(L0::Restride {
            specs,
            bounds: BoundsProof::Static,
            x: joint,
        })
        .ok()?;
    match side_axis {
        Some(d) => d.known_eq(Dim::Const(len)).then_some(narrowed),
        None => {
            if len != 1 {
                return None;
            }
            let specs: SmallVec<[StrideSpec; 6]> = (0..r)
                .map(|j| StrideSpec::dim(j, base[j as usize]))
                .collect();
            b.add_l0(L0::Restride {
                specs,
                bounds: BoundsProof::Static,
                x: narrowed,
            })
            .ok()
        }
    }
}

/// Re-apply a chain of pure views over a rewritten base, innermost first.
/// Rebuilding the nodes rather than composing their specs keeps every relative
/// stride exactly as it was written.
fn rebuild_spine(b: &mut Builder<'_>, spine: &ViewSpine, base: Id) -> Option<Id> {
    let mut cur = base;
    for &v in &spine.views {
        let Op::L0(L0::Restride { specs, bounds, .. }) = b.node(v).op.clone() else {
            return None;
        };
        cur = b
            .add_l0(L0::Restride {
                specs,
                bounds,
                x: cur,
            })
            .ok()?;
    }
    Some(cur)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::carrier::SlotTy;
    use crate::dtype::{RoundMode, Splat};
    use crate::egraph::{EGraph, Rule, Saturate, SaturationBudget, SaturationReport};
    use crate::rules::CORE_RULES;
    use crate::rules::test_support as ts;
    use crate::saturate::CoreSaturate;
    use crate::scalar::{BinOp, ScalarKind, UnOp};
    use smallvec::smallvec;

    fn fire(g: &mut EGraph, id: Id, r: &Rule) -> Option<Id> {
        let caps = ts::caps();
        let node = g.node(id).clone();
        let facts = g.facts_view(id, &caps);
        let mut b = g.builder(&caps);
        (r.apply)(&mut b, id, &node, &facts)
    }

    /// Saturate with the real rule table.
    fn saturate(g: &mut EGraph) -> SaturationReport {
        let caps = ts::caps();
        CoreSaturate
            .saturate(g, &caps, CORE_RULES, SaturationBudget::default())
            .unwrap()
    }

    fn f32e(i: u32) -> ScalarExpr {
        ScalarExpr::arg(i, Dtype::F32)
    }

    /// Walk `id` down through pure views to the nest underneath, if any.
    fn base_fold(g: &EGraph, mut id: Id) -> Option<Id> {
        loop {
            match &g.node(id).op {
                Op::L0(L0::Restride { x, .. }) => id = *x,
                Op::L1(L1::KFold { .. }) | Op::L0(L0::Fold { .. }) => return Some(id),
                _ => return None,
            }
        }
    }

    /// The joined nest a consumer's class reads at operand slots `a` and `c`:
    /// the same multi-slot `KFold` under both. Every hit is checked against the
    /// carrier obligation and one `post` per slot.
    fn joined_under(
        g: &EGraph,
        consumer: Id,
        a: usize,
        c: usize,
    ) -> Option<(Id, Carrier, Vec<Operand>)> {
        for m in g.chain(consumer) {
            let ops = match &g.node(m).op {
                Op::L1(L1::KMap { ops, .. }) | Op::L1(L1::KFold { ops, .. }) => ops.clone(),
                _ => continue,
            };
            let (Some(fa), Some(fc)) = (
                ops.get(a).and_then(|o| base_fold(g, o.src)),
                ops.get(c).and_then(|o| base_fold(g, o.src)),
            ) else {
                continue;
            };
            if fa != fc {
                continue;
            }
            if let Op::L1(L1::KFold {
                carrier, ops, post, acc, ..
            }) = g.node(fa).op.clone()
                && carrier.width() > 1
            {
                crate::verify_l0::check_carrier(&carrier, acc)
                    .expect("the joined carrier fails the carrier obligation");
                assert_eq!(post.len(), carrier.width(), "one post per slot");
                return Some((fa, carrier, ops));
            }
        }
        None
    }

    /// Fold a carrier over rows of operand values the way a sequential inner
    /// loop does.
    fn run(c: &Carrier, rows: &[Vec<f32>]) -> Vec<f32> {
        rows.iter().fold(c.identity_f32(), |acc, r| {
            c.absorb(&acc, r)
                .expect("the host evaluator covers this carrier")
        })
    }

    const XS: [f32; 6] = [1.5, -3.0, 7.25, 0.5, -11.5, 2.0];
    const YS: [f32; 6] = [0.25, 2.0, -1.5, 4.0, 0.5, -0.75];

    // layer_norm / rms_norm backward: the composed adjoint emits `sum(dy)` and
    // `sum(dy * xhat)` over the same feature axis of the same operands, meeting
    // at the `dx` expression.

    /// `dy`, `xhat`, `sum dy`, `sum dy*xhat`, and the `dx` expression reading
    /// both sums broadcast back over the feature axis.
    fn norm_backward(g: &mut EGraph, rows: u64, feats: u64) -> (Id, Id, Id) {
        let shape = [Dim::Const(rows), Dim::Const(feats)];
        let dy = ts::buffer(g, Dtype::F32, &shape);
        let xhat = ts::buffer(g, Dtype::F32, &shape);
        let prod = ts::map(g, ScalarExpr::bin(BinOp::Mul, f32e(0), f32e(1)), &[dy, xhat]);
        let add = ts::binop_carrier(BinOp::Add, Dtype::F32);
        let s1 = ts::fold(g, add.clone(), 1, Dtype::F32, dy);
        let s2 = ts::fold(g, add, 1, Dtype::F32, prod);
        let bcast = [
            StrideSpec::dim(0, Dim::Const(rows)),
            StrideSpec::broadcast(Dim::Const(feats)),
        ];
        let b1 = ts::restride(g, &bcast, s1);
        let b2 = ts::restride(g, &bcast, s2);
        // dx = dy - (mean term) - xhat * (covariance term).
        let body = ScalarExpr::bin(
            BinOp::Sub,
            ScalarExpr::bin(BinOp::Sub, f32e(0), f32e(1)),
            ScalarExpr::bin(BinOp::Mul, f32e(2), f32e(3)),
        );
        let dx = ts::map(g, body, &[dy, b1, xhat, b2]);
        (s1, s2, dx)
    }

    #[test]
    fn tuple_fuses_a_normalization_backward_into_one_nest() {
        let mut g = ts::graph();
        let (s1, s2, dx) = norm_backward(&mut g, 4, 6);
        let report = saturate(&mut g);
        assert!(
            report.fired.iter().any(|(n, c)| *n == "TUPLE" && *c > 0),
            "TUPLE never fired: {:?}",
            report.fired
        );

        // The consumer's class reads ONE nest at both sum slots.
        let (joint, carrier, _) =
            joined_under(&g, dx, 1, 3).expect("dx reads one joined nest at slots 1 and 3");
        assert_eq!(carrier.width(), 2, "two sums, one two-slot carrier");
        assert!(carrier.slots.iter().all(|s| *s == SlotTy::Scalar));
        assert!(joint != s1 && joint != s2);

        // Both originals gained the joint as an alternative.
        for s in [s1, s2] {
            assert!(
                g.chain(s).iter().any(|&m| base_fold(&g, m) == Some(joint)),
                "the original nest {s} was not redirected onto the joint"
            );
        }
    }

    /// Each slot of the joined nest computes what its own fold computed alone.
    #[test]
    fn a_joined_normalization_backward_computes_both_sums() {
        let mut g = ts::graph();
        let dy_shape = [Dim::Const(1), Dim::Const(XS.len() as u64)];
        let dy = ts::buffer(&mut g, Dtype::F32, &dy_shape);
        let xhat = ts::buffer(&mut g, Dtype::F32, &dy_shape);
        let prod = ts::kmap(
            &mut g,
            &dy_shape,
            ScalarExpr::bin(BinOp::Mul, f32e(0), f32e(1)),
            vec![alias_operand_of(dy, &dy_shape), alias_operand_of(xhat, &dy_shape)],
        );
        let add = ts::binop_carrier(BinOp::Add, Dtype::F32);
        let s1 = ts::kfold(
            &mut g,
            &dy_shape,
            1,
            add.clone(),
            Dtype::F32,
            f32e(0),
            vec![alias_operand_of(dy, &dy_shape)],
        );
        let s2 = ts::kfold(
            &mut g,
            &dy_shape,
            1,
            add,
            Dtype::F32,
            f32e(0),
            vec![alias_operand_of(prod, &dy_shape)],
        );
        let out = [Dim::Const(1)];
        let dx = ts::kmap(
            &mut g,
            &out,
            ScalarExpr::bin(BinOp::Sub, f32e(0), f32e(1)),
            vec![alias_operand_of(s1, &out), alias_operand_of(s2, &out)],
        );
        assert!(fire(&mut g, dx, &TUPLE).is_some());
        let (_, carrier, ops) = joined_under(&g, dx, 0, 1).expect("one joined nest");
        assert_eq!(carrier.width(), 2);
        // Operand 0 is `dy`; operand 1 is the elementwise product nest.
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].src, dy);
        assert_eq!(ops[1].src, prod);

        let rows: Vec<Vec<f32>> = (0..XS.len()).map(|i| vec![XS[i], XS[i] * YS[i]]).collect();
        let got = run(&carrier, &rows);
        let want_sum: f32 = XS.iter().sum();
        let want_dot: f32 = XS.iter().zip(YS).map(|(x, y)| x * y).sum();
        assert!((got[0] - want_sum).abs() < 1e-5, "slot 0: {got:?}");
        assert!((got[1] - want_dot).abs() < 1e-5, "slot 1: {got:?}");
    }

    // Min and max in one pass: the law joins them because they fold one axis of
    // one operand, not because anything relates `Max` to `Min`.

    #[test]
    fn tuple_reads_its_input_once_for_min_and_max() {
        let mut g = ts::graph();
        let shape = [Dim::Const(4), Dim::Const(XS.len() as u64)];
        let x = ts::buffer(&mut g, Dtype::F32, &shape);
        let mx = ts::fold(
            &mut g,
            ts::binop_carrier(BinOp::Max, Dtype::F32),
            1,
            Dtype::F32,
            x,
        );
        let mn = ts::fold(
            &mut g,
            ts::binop_carrier(BinOp::Min, Dtype::F32),
            1,
            Dtype::F32,
            x,
        );
        let range = ts::map(
            &mut g,
            ScalarExpr::bin(BinOp::Sub, f32e(0), f32e(1)),
            &[mx, mn],
        );
        saturate(&mut g);

        let (_, carrier, ops) =
            joined_under(&g, range, 0, 1).expect("the range reads one joined nest");
        assert_eq!(carrier.width(), 2);
        // Read once: the two nests' operand lists unified onto a single edge.
        assert_eq!(ops.len(), 1, "the input is read through one edge");
        assert_eq!(ops[0].src, x);
        assert_eq!(carrier.identity[0], Splat::F32(f32::NEG_INFINITY));
        assert_eq!(carrier.identity[1], Splat::F32(f32::INFINITY));

        let rows: Vec<Vec<f32>> = XS.iter().map(|&v| vec![v]).collect();
        let got = run(&carrier, &rows);
        assert_eq!(got[0], XS.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        assert_eq!(got[1], XS.iter().copied().fold(f32::INFINITY, f32::min));
    }

    /// TUPLE carries no `reassoc` guard: every slot folds in the order it
    /// folded alone.
    #[test]
    fn tuple_fires_under_a_strict_numeric_contract() {
        let mut g = ts::graph();
        let shape = [Dim::Const(4), Dim::Const(6)];
        let raw = ts::buffer(&mut g, Dtype::F32, &shape);
        // A rounding body is the QAT fake-quant path: `reassoc: false`.
        let q = ts::map(
            &mut g,
            ScalarExpr::round(RoundMode::HalfAwayFromZero, f32e(0)),
            &[raw],
        );
        let mx = ts::fold(
            &mut g,
            ts::binop_carrier(BinOp::Max, Dtype::F32),
            1,
            Dtype::F32,
            q,
        );
        let mn = ts::fold(
            &mut g,
            ts::binop_carrier(BinOp::Min, Dtype::F32),
            1,
            Dtype::F32,
            q,
        );
        assert!(!g.facts(mx).numeric.reassoc, "the fixture is not STRICT");
        let range = ts::map(
            &mut g,
            ScalarExpr::bin(BinOp::Sub, f32e(0), f32e(1)),
            &[mx, mn],
        );
        saturate(&mut g);
        let (_, carrier, _) = joined_under(&g, range, 0, 1)
            .expect("TUPLE declined on a value that forbids reassociation");
        assert_eq!(carrier.width(), 2);
    }

    /// `attention_lse`'s chain, with the carried dependence present or
    /// discharged. With it present, `%l`'s operands contain `bcast(%m)` and
    /// unifying the operand lists would realize a cyclic DAG.
    fn lse_chain(g: &mut EGraph, feedback: bool) -> Id {
        let shape = [Dim::Const(4), Dim::Const(6)];
        let s = ts::buffer(g, Dtype::F32, &shape);
        let m = ts::fold(
            g,
            ts::binop_carrier(BinOp::Max, Dtype::F32),
            1,
            Dtype::F32,
            s,
        );
        let bcast = [
            StrideSpec::dim(0, Dim::Const(4)),
            StrideSpec::broadcast(Dim::Const(6)),
        ];
        // The reference the shifted sum subtracts: the running max itself (the
        // carried dependence) or an independently supplied buffer (what
        // RETARGET leaves behind once it has discharged the dependence).
        let reference = if feedback {
            ts::restride(g, &bcast, m)
        } else {
            ts::buffer(g, Dtype::F32, &shape)
        };
        let shifted = ts::map(
            g,
            ScalarExpr::un(UnOp::Exp, ScalarExpr::bin(BinOp::Sub, f32e(0), f32e(1))),
            &[s, reference],
        );
        let l = ts::fold(
            g,
            ts::binop_carrier(BinOp::Add, Dtype::F32),
            1,
            Dtype::F32,
            shifted,
        );
        ts::map(
            g,
            ScalarExpr::bin(BinOp::Add, f32e(0), ScalarExpr::un(UnOp::Log, f32e(1))),
            &[m, l],
        )
    }

    #[test]
    fn tuple_declines_a_carried_dependence() {
        let mut g = ts::graph();
        let lse = lse_chain(&mut g, true);
        let report = saturate(&mut g);
        assert!(
            joined_under(&g, lse, 0, 1).is_none(),
            "TUPLE joined two nests one of which reads the other"
        );
        assert!(
            !report.fired.iter().any(|(n, c)| *n == "TUPLE" && *c > 0),
            "TUPLE fired on a cyclic pair: {:?}",
            report.fired
        );
    }

    #[test]
    fn tuple_fires_once_the_dependence_is_discharged() {
        let mut g = ts::graph();
        let lse = lse_chain(&mut g, false);
        saturate(&mut g);
        let (_, carrier, _) = joined_under(&g, lse, 0, 1)
            .expect("with the reference supplied the pair is acyclic and joins");
        assert_eq!(carrier.width(), 2);
        assert_eq!(carrier.kind(), None, "a two-slot carrier has no binop kind");
    }

    /// A `(max, count)` carrier, hand-built. Two slots, the first structurally
    /// identical to a plain `Fold{Max}`'s.
    fn max_count() -> Carrier {
        Carrier {
            slots: smallvec![SlotTy::Scalar, SlotTy::Scalar],
            identity: smallvec![Splat::F32(f32::NEG_INFINITY), Splat::F32(0.0)],
            lift: smallvec![f32e(0), ScalarExpr::lit(Splat::F32(1.0))],
            merge: smallvec![
                ScalarExpr::bin(BinOp::Max, f32e(0), f32e(2)),
                ScalarExpr::bin(BinOp::Add, f32e(1), f32e(3)),
            ],
            associative: true,
            tie: None,
        }
    }

    /// Joining two carriers that share a slot yields one copy of it. Also
    /// covers re-applying a view spine over the readback: the count is read
    /// through a two-step narrow-then-drop chain.
    #[test]
    fn tuple_deduplicates_a_shared_slot() {
        let mut g = ts::graph();
        let shape = [Dim::Const(4), Dim::Const(XS.len() as u64)];
        let x = ts::buffer(&mut g, Dtype::F32, &shape);
        let out = [Dim::Const(4)];
        let mx = ts::kfold(
            &mut g,
            &shape,
            1,
            ts::binop_carrier(BinOp::Max, Dtype::F32),
            Dtype::F32,
            f32e(0),
            vec![alias_operand_of(x, &shape)],
        );
        let mc = g
            .add(Op::L1(L1::KFold {
                space: IndexSpace::new(shape.iter().copied()),
                axis: 1,
                vec_axes: smallvec![],
                carrier: max_count(),
                acc: Dtype::F32,
                post: smallvec![f32e(0), f32e(1)],
                ops: vec![alias_operand_of(x, &shape)],
                sched: ScheduleDomain::Point,
            }))
            .unwrap();
        let narrow = ts::restride(
            &mut g,
            &[
                StrideSpec::dim(0, Dim::Const(4)),
                StrideSpec::dim(1, Dim::Const(1)).with_offset(Dim::Const(1)),
            ],
            mc,
        );
        let count = ts::restride(&mut g, &[StrideSpec::dim(0, Dim::Const(4))], narrow);
        let mean = ts::kmap(
            &mut g,
            &out,
            ScalarExpr::bin(BinOp::Div, f32e(0), f32e(1)),
            vec![alias_operand_of(mx, &out), alias_operand_of(count, &out)],
        );
        assert!(fire(&mut g, mean, &TUPLE).is_some(), "TUPLE declined");

        let (_, carrier, ops) =
            joined_under(&g, mean, 0, 1).expect("one joined nest under both operands");
        // One slot plus two slots is three; the shared max collapsed to one.
        assert_eq!(1 + max_count().width(), 3);
        assert_eq!(
            carrier.width(),
            2,
            "the shared max slot was duplicated: {:?}",
            carrier.slots
        );
        assert_eq!(ops.len(), 1);
        assert_eq!(
            carrier.merge[0].kind(),
            &ScalarKind::Bin {
                op: BinOp::Max,
                a: f32e(0),
                b: f32e(2),
            }
        );

        let rows: Vec<Vec<f32>> = XS.iter().map(|&v| vec![v]).collect();
        let got = run(&carrier, &rows);
        assert_eq!(got[0], XS.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        assert_eq!(got[1], XS.len() as f32, "the count slot survived the join");
    }

    /// A strided `Alias` operand.
    fn strided(src: Id, shape: &[Dim], strides: &[Dim]) -> Operand {
        Operand {
            src,
            layout: crate::shape::Layout::from_parts(Dim::Const(0), shape, strides).unwrap(),
            access: crate::ir::level1::AccessPlan::Alias,
        }
    }

    /// A `[Scalar]` row statistic and the `[Vector(Dh)]` accumulator that reads
    /// it are one nest. The attention chain presents `l[q] = sum_k P` at
    /// `axis = 1` of `[Lq, Lk]` and `o[q,d] = sum_k P*V` at `axis = 2` of
    /// `[Lq, Dh, Lk]` with `Dh` promoted: they disagree on `axis` and
    /// `vec_axes` and on nothing else, the reduced axis's iteration index being
    /// 1 both ways. The row sum's `P` edge is legal on the joint only restated
    /// at stride 0 along `Dh`, which is also what makes it the same edge the
    /// output accumulator reads.
    #[test]
    fn tuple_joins_a_promoted_nest_with_an_unpromoted_one() {
        const LQ: u64 = 4;
        const DH: u64 = 2;
        const LK: u64 = 6;
        let mut g = ts::graph();
        let p_shape = [Dim::Const(LQ), Dim::Const(LK)];
        let v_shape = [Dim::Const(LK), Dim::Const(DH)];
        let p = ts::buffer(&mut g, Dtype::F32, &p_shape);
        let v = ts::buffer(&mut g, Dtype::F32, &v_shape);

        // `l[q] = sum_k P[q,k]`, unpromoted: space `[Lq, Lk]`, axis 1.
        let l = ts::kfold(
            &mut g,
            &p_shape,
            1,
            ts::binop_carrier(BinOp::Add, Dtype::F32),
            Dtype::F32,
            f32e(0),
            vec![alias_operand_of(p, &p_shape)],
        );
        // `o[q,d] = sum_k P[q,k] * V[k,d]`, promoted: space `[Lq, Dh, Lk]`,
        // axis 2, `Dh` accumulator-resident. Both operands are stated over the
        // full space; `P` does not vary along `Dh`.
        let o_space = [Dim::Const(LQ), Dim::Const(DH), Dim::Const(LK)];
        let o_carrier = ts::binop_carrier(BinOp::Add, Dtype::F32)
            .promote(Dim::Const(DH))
            .expect("a promotable carrier")
            .with_lift([ScalarExpr::bin(BinOp::Mul, f32e(0), f32e(1))]);
        let o = g
            .add(Op::L1(L1::KFold {
                space: IndexSpace::new(o_space.iter().copied()),
                axis: 2,
                vec_axes: smallvec![1],
                carrier: o_carrier,
                acc: Dtype::F32,
                post: smallvec![f32e(0)],
                ops: vec![
                    strided(p, &o_space, &[Dim::Const(LK), Dim::Const(0), Dim::Const(1)]),
                    strided(v, &o_space, &[Dim::Const(0), Dim::Const(1), Dim::Const(DH)]),
                ],
                sched: ScheduleDomain::Point,
            }))
            .unwrap();
        assert!(l.0 < o.0, "the fixture wants the unpromoted nest on the left");

        // The consumer both meet at: `o / l`, the softmax divide.
        let out = [Dim::Const(LQ), Dim::Const(DH)];
        let div = ts::kmap(
            &mut g,
            &out,
            ScalarExpr::bin(BinOp::Div, f32e(0), f32e(1)),
            vec![
                alias_operand_of(o, &out),
                strided(l, &out, &[Dim::Const(1), Dim::Const(0)]),
            ],
        );
        assert!(fire(&mut g, div, &TUPLE).is_some(), "TUPLE declined");

        let (joint, carrier, ops) =
            joined_under(&g, div, 0, 1).expect("one joined nest under both operands");
        assert_eq!(
            carrier.slots.as_slice(),
            &[SlotTy::Scalar, SlotTy::Vector(Dim::Const(DH))],
            "the joint is the mixed carrier, unpromoted side first"
        );

        // The joint is minted in the promoted domain, not the narrow one.
        let Op::L1(L1::KFold {
            space,
            axis,
            vec_axes,
            ..
        }) = g.node(joint).op.clone()
        else {
            unreachable!()
        };
        assert_eq!(space.dims.as_slice(), &o_space);
        assert_eq!(axis, 2);
        assert_eq!(vec_axes.as_slice(), &[1]);

        // `P` was widened at stride 0 along `Dh` — so it is the same edge the
        // output accumulator already read, and the joint reads it once.
        assert_eq!(ops.len(), 2, "the score matrix was read twice: {ops:?}");
        assert_eq!(
            ops[0].layout.strides(),
            &[Dim::Const(LK), Dim::Const(0), Dim::Const(1)],
            "the row sum's operand was not widened at stride 0 along the promoted axis"
        );

        // A `Vector` slot merges positionwise, so folding the carrier over one
        // promoted position is that position's accumulator; the scalar slot is
        // the whole row's sum. The rows are `[P, V]`, the order `unify_ops` put
        // them in.
        let rows: Vec<Vec<f32>> = (0..LK as usize).map(|k| vec![XS[k], YS[k]]).collect();
        let got = run(&carrier, &rows);
        let want_l: f32 = XS.iter().sum();
        let want_o: f32 = (0..LK as usize).map(|k| XS[k] * YS[k]).sum();
        assert!((got[0] - want_l).abs() < 1e-4, "row sum {got:?}");
        assert!((got[1] - want_o).abs() < 1e-4, "weighted sum {got:?}");
    }

    /// Two different promotions of one space have no common nest:
    /// `verify_l1::check_vec_axes` pins every `Vector` slot to the same
    /// promoted extent.
    #[test]
    fn tuple_declines_two_different_promotions() {
        let mut g = ts::graph();
        let space = [Dim::Const(4), Dim::Const(2), Dim::Const(6)];
        let x = ts::buffer(&mut g, Dtype::F32, &space);
        let ops = vec![alias_operand_of(x, &space)];
        let promoted = |g: &mut EGraph, vec_axes: SmallVec<[u32; 2]>, op: BinOp| {
            let carrier = ts::binop_carrier(op, Dtype::F32)
                .promote(Dim::Const(2))
                .unwrap();
            g.add(Op::L1(L1::KFold {
                space: IndexSpace::new(space.iter().copied()),
                axis: 2,
                vec_axes,
                carrier,
                acc: Dtype::F32,
                post: smallvec![f32e(0)],
                ops: ops.clone(),
                sched: ScheduleDomain::Point,
            }))
            .unwrap()
        };
        // Same space and same reduced axis, but one promotes axis 1 into the
        // carrier and the other promotes axis 0 — two geometries, not two
        // spellings of one.
        let a = promoted(&mut g, smallvec![1], BinOp::Add);
        let b = g
            .add(Op::L1(L1::KFold {
                space: IndexSpace::new([Dim::Const(2), Dim::Const(4), Dim::Const(6)]),
                axis: 2,
                vec_axes: smallvec![1],
                carrier: ts::binop_carrier(BinOp::Max, Dtype::F32)
                    .promote(Dim::Const(4))
                    .unwrap(),
                acc: Dtype::F32,
                post: smallvec![f32e(0)],
                ops: vec![strided(
                    x,
                    &[Dim::Const(2), Dim::Const(4), Dim::Const(6)],
                    &[Dim::Const(6), Dim::Const(12), Dim::Const(1)],
                )],
                sched: ScheduleDomain::Point,
            }))
            .unwrap();
        let out = [Dim::Const(4), Dim::Const(2)];
        let c = ts::kmap(
            &mut g,
            &out,
            ScalarExpr::bin(BinOp::Sub, f32e(0), f32e(1)),
            vec![
                alias_operand_of(a, &out),
                strided(b, &out, &[Dim::Const(1), Dim::Const(4)]),
            ],
        );
        assert!(fire(&mut g, c, &TUPLE).is_none(), "TUPLE joined two promotions");
    }

    /// Two nests carrying different schedule domains are still one join, and
    /// the joint takes neither side's: it is minted at the floor `lower_fold`
    /// mints.
    #[test]
    fn the_joint_takes_neither_sides_schedule_domain() {
        let mut g = ts::graph();
        let shape = [Dim::Const(4), Dim::Const(XS.len() as u64)];
        let x = ts::buffer(&mut g, Dtype::F32, &shape);
        let ops = vec![alias_operand_of(x, &shape)];
        let tiled = ScheduleDomain::Fold(crate::ir::level1::FoldDomain {
            strategies: smallvec![
                crate::ir::level1::FoldStrat::Subgroup,
                crate::ir::level1::FoldStrat::WgTree { lane_group: 64 },
            ],
        });
        // One nest at the floor, one carrying a reduction-strategy domain.
        let sum = ts::kfold(
            &mut g,
            &shape,
            1,
            ts::binop_carrier(BinOp::Add, Dtype::F32),
            Dtype::F32,
            f32e(0),
            ops.clone(),
        );
        let mx = g
            .add(Op::L1(L1::KFold {
                space: IndexSpace::new(shape.iter().copied()),
                axis: 1,
                vec_axes: smallvec![],
                carrier: ts::binop_carrier(BinOp::Max, Dtype::F32),
                acc: Dtype::F32,
                post: smallvec![f32e(0)],
                ops,
                sched: tiled.clone(),
            }))
            .unwrap();
        assert_ne!(
            g.node(sum).op,
            g.node(mx).op,
            "the fixture needs two different schedule domains"
        );
        let out = [Dim::Const(4)];
        let c = ts::kmap(
            &mut g,
            &out,
            ScalarExpr::bin(BinOp::Sub, f32e(0), f32e(1)),
            vec![alias_operand_of(sum, &out), alias_operand_of(mx, &out)],
        );
        assert!(fire(&mut g, c, &TUPLE).is_some(), "TUPLE declined");

        let (joint, carrier, _) = joined_under(&g, c, 0, 1).expect("one joined nest");
        assert_eq!(carrier.width(), 2);
        let Op::L1(L1::KFold { sched, .. }) = g.node(joint).op.clone() else {
            unreachable!()
        };
        assert_eq!(
            sched,
            ScheduleDomain::Point,
            "the joint inherited a side's schedule domain instead of the floor"
        );
        let rows: Vec<Vec<f32>> = XS.iter().map(|&v| vec![v]).collect();
        let got = run(&carrier, &rows);
        assert!((got[0] - XS.iter().sum::<f32>()).abs() < 1e-5, "{got:?}");
        assert_eq!(got[1], XS.iter().copied().fold(f32::NEG_INFINITY, f32::max));
    }

    /// What a side gets unioned with is a strided view of the joint, typed
    /// `acc` and shaped like that side's own output, on both spellings.
    #[test]
    fn a_slot_readback_carries_the_side_it_replaces_facts() {
        let mut g = ts::graph();
        let shape = [Dim::Const(4), Dim::Const(6)];
        let x = ts::buffer(&mut g, Dtype::F32, &shape);
        // `ts::fold` is the L0 spelling; `ts::kfold` the L1 one.
        let a = ts::fold(
            &mut g,
            ts::binop_carrier(BinOp::Add, Dtype::F32),
            1,
            Dtype::F32,
            x,
        );
        let b = ts::kfold(
            &mut g,
            &shape,
            1,
            ts::binop_carrier(BinOp::Max, Dtype::F32),
            Dtype::F32,
            f32e(0),
            vec![alias_operand_of(x, &shape)],
        );
        let out = [Dim::Const(4)];
        let c = ts::kmap(
            &mut g,
            &out,
            ScalarExpr::bin(BinOp::Sub, f32e(0), f32e(1)),
            vec![alias_operand_of(a, &out), alias_operand_of(b, &out)],
        );
        assert!(fire(&mut g, c, &TUPLE).is_some());
        let (joint, ..) = joined_under(&g, c, 0, 1).expect("one joined nest");
        assert_eq!(g.facts(joint).dtype, Dtype::F32);
        assert_eq!(&g.facts(joint).shape[..], &[Dim::Const(4), Dim::Const(2)]);
        // Every member either side gained agrees with that side's own facts.
        for side in [a, b] {
            let want = g.facts(side).clone();
            for m in g.chain(side) {
                assert_eq!(g.facts(m).dtype, want.dtype, "member {m} of {side}");
                assert_eq!(&g.facts(m).shape[..], &want.shape[..], "member {m}");
            }
        }
    }

    #[test]
    fn tuple_declines_a_different_reduction_axis() {
        let mut g = ts::graph();
        let shape = [Dim::Const(4), Dim::Const(4)];
        let x = ts::buffer(&mut g, Dtype::F32, &shape);
        let a = ts::fold(
            &mut g,
            ts::binop_carrier(BinOp::Add, Dtype::F32),
            0,
            Dtype::F32,
            x,
        );
        let b = ts::fold(
            &mut g,
            ts::binop_carrier(BinOp::Max, Dtype::F32),
            1,
            Dtype::F32,
            x,
        );
        let c = ts::map(
            &mut g,
            ScalarExpr::bin(BinOp::Sub, f32e(0), f32e(1)),
            &[a, b],
        );
        saturate(&mut g);
        assert!(joined_under(&g, c, 0, 1).is_none());
    }

    /// A `STRICT` nest may not join a `RELAXED` one: the joint node carries one
    /// contract, and adopting either side's rewrites the other's rounding.
    #[test]
    fn tuple_declines_a_disagreeing_numeric_contract() {
        let mut g = ts::graph();
        let shape = [Dim::Const(4), Dim::Const(6)];
        let raw = ts::buffer(&mut g, Dtype::F32, &shape);
        let relaxed = ts::buffer(&mut g, Dtype::F32, &shape);
        let strict = ts::map(
            &mut g,
            ScalarExpr::round(RoundMode::HalfAwayFromZero, f32e(0)),
            &[raw],
        );
        let a = ts::fold(
            &mut g,
            ts::binop_carrier(BinOp::Max, Dtype::F32),
            1,
            Dtype::F32,
            strict,
        );
        let b = ts::fold(
            &mut g,
            ts::binop_carrier(BinOp::Min, Dtype::F32),
            1,
            Dtype::F32,
            relaxed,
        );
        assert!(!g.facts(a).numeric.reassoc);
        assert!(g.facts(b).numeric.reassoc, "the fixture is not mixed");
        let c = ts::map(
            &mut g,
            ScalarExpr::bin(BinOp::Sub, f32e(0), f32e(1)),
            &[a, b],
        );
        saturate(&mut g);
        assert!(
            joined_under(&g, c, 0, 1).is_none(),
            "a STRICT nest was joined onto a RELAXED one"
        );
    }

    /// Slot order is a function of node id, never of which operand slot the
    /// consumer read first.
    #[test]
    fn slot_order_does_not_depend_on_operand_order() {
        let build = |reversed: bool| {
            let mut g = ts::graph();
            let shape = [Dim::Const(4), Dim::Const(6)];
            let x = ts::buffer(&mut g, Dtype::F32, &shape);
            let mx = ts::fold(
                &mut g,
                ts::binop_carrier(BinOp::Max, Dtype::F32),
                1,
                Dtype::F32,
                x,
            );
            let mn = ts::fold(
                &mut g,
                ts::binop_carrier(BinOp::Min, Dtype::F32),
                1,
                Dtype::F32,
                x,
            );
            let ins = if reversed { [mn, mx] } else { [mx, mn] };
            let c = ts::map(
                &mut g,
                ScalarExpr::bin(BinOp::Sub, f32e(0), f32e(1)),
                &ins,
            );
            saturate(&mut g);
            joined_under(&g, c, 0, 1).expect("one joined nest").1
        };
        let forward = build(false);
        let reversed = build(true);
        assert_eq!(forward, reversed);
        // The max nest has the smaller id, so it is slot 0 either way.
        assert_eq!(forward.identity[0], Splat::F32(f32::NEG_INFINITY));
    }

    #[test]
    fn tuple_declines_a_disagreeing_accumulator() {
        let mut g = ts::graph();
        let shape = [Dim::Const(4), Dim::Const(6)];
        let x = ts::buffer(&mut g, Dtype::F32, &shape);
        let a = ts::fold(
            &mut g,
            ts::binop_carrier(BinOp::Add, Dtype::F32),
            1,
            Dtype::F32,
            x,
        );
        let b = ts::fold(
            &mut g,
            ts::binop_carrier(BinOp::Max, Dtype::F16),
            1,
            Dtype::F16,
            x,
        );
        let c = ts::map(
            &mut g,
            ScalarExpr::bin(BinOp::Sub, f32e(0), f32e(1)),
            &[a, b],
        );
        saturate(&mut g);
        assert!(
            joined_under(&g, c, 0, 1).is_none(),
            "an f32 and an f16 accumulator were forced into one"
        );
    }

    /// A carrier too wide to hold in registers is unschedulable, so the
    /// footprint is a legality guard.
    #[test]
    fn tuple_declines_an_over_budget_carrier() {
        let lanes = private_acc_bytes(&ts::caps()) / Dtype::F32.byte_size();
        let wide = Carrier {
            slots: smallvec![SlotTy::Vector(Dim::Const(lanes))],
            identity: smallvec![Splat::F32(0.0)],
            lift: smallvec![f32e(0)],
            merge: smallvec![ScalarExpr::bin(BinOp::Add, f32e(0), f32e(1))],
            associative: true,
            tie: None,
        };
        let mut g = ts::graph();
        let shape = [Dim::Const(4), Dim::Const(6)];
        let x = ts::buffer(&mut g, Dtype::F32, &shape);
        let ops = vec![alias_operand_of(x, &shape)];
        let a = ts::kfold(
            &mut g,
            &shape,
            1,
            wide,
            Dtype::F32,
            f32e(0),
            ops.clone(),
        );
        let b = ts::kfold(
            &mut g,
            &shape,
            1,
            ts::binop_carrier(BinOp::Max, Dtype::F32),
            Dtype::F32,
            f32e(0),
            ops,
        );
        let out = [Dim::Const(4), Dim::Const(lanes)];
        let c = ts::kmap(
            &mut g,
            &out,
            ScalarExpr::bin(BinOp::Sub, f32e(0), f32e(1)),
            vec![alias_operand_of(a, &out), alias_operand_of(b, &out)],
        );
        assert!(
            fire(&mut g, c, &TUPLE).is_none(),
            "a carrier over the private accumulator budget was minted"
        );
    }

    /// The same law rooted at a reducing consumer: an outer nest reading two
    /// inner nests over one axis joins them as a `KMap` consumer does.
    #[test]
    fn tuple_sibling_roots_at_a_reducing_consumer() {
        let mut g = ts::graph();
        let shape = [Dim::Const(4), Dim::Const(XS.len() as u64)];
        let x = ts::buffer(&mut g, Dtype::F32, &shape);
        let y = ts::buffer(&mut g, Dtype::F32, &shape);
        let add = ts::binop_carrier(BinOp::Add, Dtype::F32);
        let a = ts::fold(&mut g, add.clone(), 1, Dtype::F32, x);
        let b = ts::fold(&mut g, add.clone(), 1, Dtype::F32, y);
        let out = [Dim::Const(4)];
        let outer = ts::kfold(
            &mut g,
            &out,
            0,
            add,
            Dtype::F32,
            f32e(0),
            vec![alias_operand_of(a, &out), alias_operand_of(b, &out)],
        );
        assert!(fire(&mut g, outer, &TUPLE_SIBLING).is_some());
        let (_, carrier, ops) =
            joined_under(&g, outer, 0, 1).expect("the outer nest reads one joined inner nest");
        assert_eq!(carrier.width(), 2);
        assert_eq!(ops.len(), 2, "two distinct inputs stay two edges");
        let rows: Vec<Vec<f32>> = (0..XS.len()).map(|i| vec![XS[i], YS[i]]).collect();
        let got = run(&carrier, &rows);
        assert!((got[0] - XS.iter().sum::<f32>()).abs() < 1e-5);
        assert!((got[1] - YS.iter().sum::<f32>()).abs() < 1e-5);
    }

    /// A third nest joins onto the node the rule just minted, so `F` nests cost
    /// `F-1` firings.
    #[test]
    fn a_third_nest_joins_onto_the_result() {
        let mut g = ts::graph();
        let shape = [Dim::Const(4), Dim::Const(6)];
        let add = ts::binop_carrier(BinOp::Add, Dtype::F32);
        let ins: Vec<Id> = (0..3)
            .map(|_| ts::buffer(&mut g, Dtype::F32, &shape))
            .collect();
        let folds: Vec<Id> = ins
            .iter()
            .map(|&x| ts::fold(&mut g, add.clone(), 1, Dtype::F32, x))
            .collect();
        let body = ScalarExpr::bin(
            BinOp::Add,
            ScalarExpr::bin(BinOp::Add, f32e(0), f32e(1)),
            f32e(2),
        );
        let sink = ts::map(&mut g, body, &folds);
        saturate(&mut g);
        let widest = g
            .chain(sink)
            .into_iter()
            .filter_map(|m| match &g.node(m).op {
                Op::L1(L1::KMap { ops, .. }) => Some(ops.clone()),
                _ => None,
            })
            .filter_map(|ops| {
                let f = base_fold(&g, ops.first()?.src)?;
                match g.node(f).op.clone() {
                    Op::L1(L1::KFold { carrier, .. }) => Some(carrier.width()),
                    _ => None,
                }
            })
            .max()
            .unwrap_or(0);
        assert_eq!(widest, 3, "three sums did not collapse into one nest");
    }
}
