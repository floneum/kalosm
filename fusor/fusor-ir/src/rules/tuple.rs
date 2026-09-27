//! TUPLE — two reduction nests over the same iteration space and reduction
//! axis are one nest over the concatenated carrier ([`Carrier::tuple`], which
//! dedups slots). Order-preserving per slot, so legal under
//! [`NumericContract::STRICT`]. Fires at a consumer reading both nests:
//! [`TUPLE`] at a `Map` or a `Fold`. Neither nest's
//! operand closure may reach the other's result.

use crate::carrier::{ArgRemap, Carrier, Tupled, map_args, retype_args};
use crate::dtype::NumericContract;
use crate::egraph::{Builder, Facts, Id, RuleTag};
use crate::ir::launch::{Launch, Operand};
use crate::ir::logical::Logical;
use crate::ir::{Level, Node, Op, OpTag};
use crate::rule;
use crate::rules::{FoldView, rebuild_spine};
use crate::scalar::ScalarExpr;
use crate::shape::{BoundsProof, Dim, StrideSpec};
use rustc_hash::FxHashSet;
use smallvec::SmallVec;

rule!(
    TUPLE,
    level = Level::Launch,
    heads = [OpTag::LaunchMap, OpTag::LaunchFold],
    tag = RuleTag::Additive,
    apply = tuple_at,
);

/// The same nest, whichever id spells it; ignores `id` (Logical vs Launch
/// spelling) and `sched` (not a value).
fn same_nest(a: &FoldView, b: &FoldView) -> bool {
    a.space == b.space
        && a.axis == b.axis
        && a.vec_axes == b.vec_axes
        && a.carrier == b.carrier
        && a.acc == b.acc
        && a.post == b.post
        && a.ops == b.ops
}

/// Read `id` as a reduction nest whose facts match its `acc` output, which
/// the joint's readback view must reproduce.
fn fold_view(b: &Builder<'_>, id: Id) -> Option<FoldView> {
    let v = bare_fold_view(b, id)?;
    let f = b.facts_of(id);
    let want = v.space.fold_shape(v.axis, &v.vec_axes, &v.carrier)?;
    if f.dtype != v.acc || f.shape != want {
        return None;
    }
    Some(v)
}

/// The nest in either spelling, a `Logical::Fold`'s lift retyped as
/// `lower_fold` does so both hash-cons to one joint.
fn bare_fold_view(b: &Builder<'_>, id: Id) -> Option<FoldView> {
    let mut v = crate::rules::fold_view(b, id)?;
    if let Op::Logical(_) = b.node(id).op {
        let dtype = b.facts_of(v.ops[0].src).dtype;
        v.carrier.lift = v
            .carrier
            .lift
            .iter()
            .map(|e| retype_args(e, dtype))
            .collect();
    }
    Some(v)
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

pub fn tuple_at(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
    let Op::Launch(op @ (Launch::Map { .. } | Launch::Fold { .. })) = &node.op else {
        return None;
    };
    let srcs: SmallVec<[Id; 4]> = op.operands().map(|o| o.src).collect();
    let rewire = join_pair(b, &srcs)?;
    let mut rebuilt = op.clone();
    for (slot, src) in rewire.at {
        rebuilt.operands_mut().nth(slot)?.src = src;
    }
    let rebuilt = b.add_launch(rebuilt).ok()?;
    b.union(id, rebuilt).ok()
}

/// Join the first pair of operand slots reading joinable nests; the rewritten
/// consumer is re-queued, so `F` nests cost `F-1` firings.
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
            // The smaller id is the left carrier, so slot order is stable.
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

/// The law proper; every check precedes the first `add`, so a declined join
/// leaves no orphans. Axes compare in [`FoldView::iter_space`], since
/// promotion shifts `axis`; a promoted host widens the other side's operands
/// with stride 0 at its carrier axes ([`widen_ops`]).
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
    let host = promotion_host(f1, f2)?;
    // One accumulator contract, or one side's rounding silently changes.
    if b.facts_of(f1.id).numeric != b.facts_of(f2.id).numeric {
        return None;
    }
    // The joint joins both classes: a cycle iff an operand reaches either.
    let (ops, remap) = unify_ops(&widen_ops(f1, host)?, &widen_ops(f2, host)?)?;
    let srcs: Vec<Id> = ops.iter().map(|o| o.src).collect();
    if reaches_either(b, &srcs, f1, f2) {
        return None;
    }

    let t: Tupled = f1.carrier.tuple(&f2.carrier, &remap);
    // Symbolic private-array extents are allocatable on neither backend.
    let lanes = t.carrier.lanes()?;
    let bytes = lanes.checked_mul(f1.acc.byte_size())?;
    if bytes > crate::rules::private_acc_bytes(b.caps(), false) {
        return None;
    }
    // The meet over the unified list can be stricter than either side's.
    let joint_numeric = ops.iter().fold(NumericContract::RELAXED, |acc, o| {
        acc.meet(b.facts_of(o.src).numeric)
    });
    if f1.acc.accum_bits() < joint_numeric.min_accum_bits {
        return None;
    }
    let post = joint_post(f1, f2, &t)?;

    // Each side must be one contiguous lane range to be a view of the joint.
    let lhs_range = lane_range(&t.carrier, &t.lhs)?;
    let rhs_range = lane_range(&t.carrier, &t.rhs)?;
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
    // Redirecting only one side leaves extraction running two nests.
    b.union(f1.id, lhs_read).ok()?;
    b.union(f2.id, rhs_read).ok()?;
    Some(Joint { lhs_read, rhs_read })
}

/// The side whose carrier geometry the joint takes: the left on equal
/// promotions (extents included), else the only promoted one.
fn promotion_host<'v>(f1: &'v FoldView, f2: &'v FoldView) -> Option<&'v FoldView> {
    if f1.vec_axes == f2.vec_axes {
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

/// One side's operands restated over the host's space: stride 0 at each
/// carrier axis, as `check_vec_axes` demands of a `Scalar` slot's operands.
fn widen_ops(side: &FoldView, host: &FoldView) -> Option<Vec<Operand>> {
    if side.vec_axes == host.vec_axes {
        return Some(side.ops.clone());
    }
    side.ops
        .iter()
        .map(|o| crate::rules::fusion::widen_operand(o, &side.space, &host.space, &host.vec_axes))
        .collect()
}

/// The unified operand list plus how the right side's `Arg`s renumber onto it.
fn unify_ops(lhs: &[Operand], rhs: &[Operand]) -> Option<(Vec<Operand>, ArgRemap)> {
    let mut ops = lhs.to_vec();
    let mut map: SmallVec<[u32; 4]> = SmallVec::new();
    for o in rhs {
        match ops.iter().position(|p| p == o) {
            Some(k) => map.push(u32::try_from(k).ok()?),
            None => {
                map.push(u32::try_from(ops.len()).ok()?);
                ops.push(o.clone());
            }
        }
    }
    Some((ops, ArgRemap { map }))
}

/// Whether either nest's result is transitively reachable from `from`.
fn reaches_either(b: &Builder<'_>, from: &[Id], f1: &FoldView, f2: &FoldView) -> bool {
    let floor = f1.id.0.min(f2.id.0);
    let mut seen: FxHashSet<Id> = FxHashSet::default();
    let mut stack: Vec<Id> = from.to_vec();
    while let Some(cur) = stack.pop() {
        // Edges point to smaller ids: nothing below the floor reaches either.
        if cur.0 < floor || !seen.insert(cur) {
            continue;
        }
        if cur == f1.id || cur == f2.id {
            return true;
        }
        // Compare the normalized nest: its other spelling has another id.
        if matches!(b.node(cur).op.tag(), OpTag::Fold | OpTag::LaunchFold)
            && let Some(v) = fold_view(b, cur)
            && (same_nest(&v, f1) || same_nest(&v, f2))
        {
            return true;
        }
        stack.extend(b.node(cur).children.iter().copied());
    }
    false
}

/// One post expression per joint slot; a deduplicated slot's posts must agree.
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

/// One side's readback: the joint carrier axis narrowed to its lanes, then
/// dropped if the side had no carrier axis.
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
        // A one-scalar joint appended no axis; the side is that slot.
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
        .add_logical(Logical::Restride {
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
            b.add_logical(Logical::Restride {
                specs,
                bounds: BoundsProof::Static,
                x: narrowed,
            })
            .ok()
        }
    }
}
