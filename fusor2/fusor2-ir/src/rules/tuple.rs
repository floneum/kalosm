//! TUPLE — two reduction nests over the same iteration space and the same
//! reduction axis are ONE nest over the concatenated carrier.
//!
//! ```text
//! < Fold{C1, a, ops1}, Fold{C2, a, ops2} >
//!   ==  slot views of  Fold{ C1 (x) C2, a, ops1 u ops2 }
//! ```
//!
//! `(x)` is [`Carrier::tuple`], whose slot deduplication happens inside the
//! constructor, so joining `(m,l)` with `(m,o)` yields three slots and not
//! four by construction.
//!
//! Exactly value-preserving: every slot folds in precisely the order it folded
//! alone, so this law needs no `reassoc` guard and is legal on an f16
//! accumulator and under [`NumericContract::STRICT`].
//!
//! Rooting is consumer-rooted: the rule fires at a node that already reads
//! both nests.
//!
//! * [`TUPLE`] roots at a `Map` consumer.
//! * [`TUPLE_SIBLING`] roots at a `Fold` consumer — a reducing nest that
//!   itself reads two reducing nests.
//!
//! Acyclicity: neither nest's operand closure may transitively reach the
//! other's result, checked through `Op::Union` chains as well as `children`,
//! because the acyclic id allocator does not see a cycle that runs through a
//! union. TUPLE never discharges a carried dependence itself; that is
//! RETARGET's job.

use crate::carrier::{ArgRemap, Carrier, Tupled, map_args, probes_for, retype_args};
use crate::device::Caps;
use crate::dtype::{Dtype, NumericContract};
use crate::egraph::{Builder, Facts, Id, RuleTag, ViewSpine};
use crate::ir::logical::Logical;
use crate::ir::launch::{IndexSpace, Launch, Operand, ScheduleDomain};
use crate::ir::{Level, Node, Op, OpTag};
use crate::rule;
use crate::rules::alias_operand_of;
use crate::scalar::ScalarExpr;
use crate::shape::{BoundsProof, Dim, StrideSpec};
use rustc_hash::FxHashSet;
use smallvec::SmallVec;

rule!(
    TUPLE,
    level = Level::Launch,
    head = OpTag::LaunchMap,
    tag = RuleTag::Additive,
    apply = tuple_at_consumer,
);

rule!(
    TUPLE_SIBLING,
    level = Level::Launch,
    head = OpTag::LaunchFold,
    tag = RuleTag::Additive,
    apply = tuple_siblings,
);

/// The consumer rooting at a `Map`.
pub fn tuple_at_consumer(b: &mut Builder<'_>, id: Id, n: &Node, f: &Facts<'_>) -> Option<Id> {
    tuple_at(b, id, n, f)
}

/// The consumer rooting at a `Fold` — a reducing nest that reads two reducing
/// nests.
pub fn tuple_siblings(b: &mut Builder<'_>, id: Id, n: &Node, f: &Facts<'_>) -> Option<Id> {
    tuple_at(b, id, n, f)
}

/// The private accumulator budget one invocation may hold, in bytes.
///
/// Placeholder until [`Caps`] carries a calibrated field: the conservative
/// constant every target can honour, 256 f32 registers per lane. A carrier
/// wider than the budget is unschedulable, not merely slower, so the rule
/// declines rather than minting a node no backend can lower.
const fn private_acc_bytes(_caps: &Caps) -> u64 {
    1024
}

/// A reduction nest, normalized out of whichever spelling the operand named.
///
/// Equality in this e-graph is not congruent, so a `Logical::Fold` and the
/// `Launch::Fold` it was lowered to are one class while a consumer's operand
/// still names whichever id the frontend built. This normalizes them the way
/// `lower_fold` does — retyping the lift to the operand dtype — so the two
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
    /// spellings of one reduction agree on. `vec_axes` is the contiguous
    /// block immediately before `axis` (`verify_launch::check_vec_axes`), so
    /// subtracting their count is the whole renumbering.
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

/// The same nest, whichever id spells it. Ignores `id`: that is the
/// Logical-versus-Launch spelling the acyclicity walk must not miss. Also
/// ignores `sched`: a schedule domain is not a value, so a tiled spelling of
/// a nest is that nest.
fn same_nest(a: &FoldView, b: &FoldView) -> bool {
    a.space == b.space
        && a.axis == b.axis
        && a.vec_axes == b.vec_axes
        && a.carrier == b.carrier
        && a.acc == b.acc
        && a.post == b.post
        && a.ops == b.ops
}

/// Read `id` as a reduction nest, in either spelling.
///
/// This does not look through a `post` epilogue.
fn fold_view(b: &Builder<'_>, id: Id) -> Option<FoldView> {
    let v = bare_fold_view(b, id)?;
    // The readback the join unions against is a strided view of the joint,
    // typed `acc` and shaped like the nest's output. A spelling whose facts
    // disagree is not a value this law may redirect.
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
        Op::Launch(Launch::Fold {
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
        Op::Logical(Logical::Fold {
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
        Op::Launch(Launch::Map { ops, .. }) | Op::Launch(Launch::Fold { ops, .. }) => {
            ops.iter().map(|o| o.src).collect()
        }
        _ => return None,
    };
    let rewire = join_pair(b, &srcs)?;
    let rebuilt = match node.op.clone() {
        Op::Launch(Launch::Map {
            space,
            body,
            mut ops,
            sched,
        }) => {
            for (slot, src) in rewire.at {
                ops.get_mut(slot)?.src = src;
            }
            b.add_launch(Launch::Map {
                space,
                body,
                ops,
                sched,
            })
            .ok()?
        }
        Op::Launch(Launch::Fold {
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
            b.add_launch(Launch::Fold {
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
/// scan. One firing joins one pair; the rewritten consumer is a fresh node
/// the driver re-queues, so `F` nests cost `F-1` firings.
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
            // Deterministic join order: the smaller id is the left carrier,
            // so operand order cannot change the slot order or the `PlanHash`.
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

/// The law proper. Every legality check and every derived value is computed
/// before the first `add`, so a declined join leaves no orphan nodes.
///
/// `axis` is the wrong number to compare: it indexes `space`, which a
/// promoted nest has widened with its carrier axes, so the same logical
/// reduction is `axis = 3` unpromoted and `axis = 4` with one carrier axis
/// ahead of it. The number both sides agree on is `axis - vec_axes.len()`,
/// the reduced axis's index in [`FoldView::iter_space`].
///
/// `vec_axes` equality is a real requirement only between two nests that are
/// both promoted: two different promotions are two different carrier
/// geometries. Between a promoted nest and an unpromoted one the joint takes
/// the promoted side's `space`, `axis` and `vec_axes`, and the unpromoted
/// side's operands are restated onto that wider space by [`widen_ops`] with
/// stride 0 at each carrier axis. Stride 0 is what makes the mixed carrier
/// legal: `check_vec_axes` refuses a `Scalar` slot whose lift reads an
/// operand that varies along a promoted axis, and a stride-0 widening
/// provably does not.
///
/// The cross-promotion clause is presently latent — the shipped rule table
/// never presents a promoted nest as one of a pair —
/// `tuple_joins_a_promoted_nest_with_an_unpromoted_one` is what exercises it.
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
    // Fusing two nests forces one accumulator contract: choosing the narrower
    // lowers `min_accum_bits` and choosing the wider silently rewrites the
    // other nest's rounding.
    if b.facts_of(f1.id).numeric != b.facts_of(f2.id).numeric {
        return None;
    }
    // No guard on `sched`: a schedule domain is not a value; the joint is
    // minted at the floor and the schedule rules expand it as any other nest.

    // Acyclicity: the joint reads both operand lists and is unioned into both
    // classes, so a realized DAG has a cycle exactly when some unified
    // operand reaches either result.
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
    // A botched slot renumbering fails this.
    if !t.carrier.identity_closed(probes_for(f1.acc)) {
        return None;
    }
    // The rewritten nest's contract is the meet over the unified operand
    // list, which can be stricter than either side's.
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
    // The readbacks are views of the joint, so they are spelled with the dims
    // the joint was minted at — the host's.
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

/// Which side's carrier geometry the joint is minted in, or `None` when there
/// is no single nest holding both.
///
/// Equal `vec_axes` (both unpromoted included) takes the left side. Exactly
/// one promoted side hosts. Two different promotions decline: the joint would
/// have to hold two carrier geometries at once.
fn promotion_host<'v>(f1: &'v FoldView, f2: &'v FoldView) -> Option<&'v FoldView> {
    if f1.vec_axes == f2.vec_axes {
        // The promoted extents must also agree, or the two carriers span
        // different numbers of positions.
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
/// The host's own ride through untouched. A guest that is not promoted gets
/// stride 0 at every carrier axis the host added, which is both true (the
/// guest never had the axis) and the condition
/// `verify_launch::check_vec_axes` demands of a `Scalar` slot's operands.
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
/// elements, not syntax: `address_map` returns `None` on a `Dim::Sym` extent
/// or a `u32` overflow, and the rule then keeps both edges rather than
/// guessing.
fn same_read(a: &Operand, b: &Operand) -> bool {
    a == b && matches!((a.address_map(), b.address_map()), (Some(x), Some(y)) if x == y)
}

/// Whether either nest's result is transitively reachable from `from`.
fn reaches_either(b: &Builder<'_>, from: &[Id], f1: &FoldView, f2: &FoldView) -> bool {
    let floor = f1.id.0.min(f2.id.0);
    let mut seen: FxHashSet<Id> = FxHashSet::default();
    let mut stack: Vec<Id> = from.to_vec();
    while let Some(cur) = stack.pop() {
        // Every edge points at a strictly smaller id, so nothing below the
        // lower of the two nests can reach either.
        if cur.0 < floor || !seen.insert(cur) {
            continue;
        }
        if cur == f1.id || cur == f2.id {
            return true;
        }
        // The Logical and Launch spellings of one nest are two ids in one
        // class; compare the normalized nest, not the id.
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

/// One post expression per joint slot.
///
/// A deduplicated slot carries one post, so the two sides have to agree on it:
/// they are two spellings of one value, and if their posts differ they are not.
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
        // The joint carrier is one scalar slot, so it appended no axis and
        // the side has to be that same slot.
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

/// Re-apply a chain of pure views over a rewritten base, innermost first.
/// Rebuilding the nodes rather than composing their specs keeps every relative
/// stride exactly as it was written.
fn rebuild_spine(b: &mut Builder<'_>, spine: &ViewSpine, base: Id) -> Option<Id> {
    let mut cur = base;
    for &v in &spine.views {
        let Op::Logical(Logical::Restride { specs, bounds, .. }) = b.node(v).op.clone() else {
            return None;
        };
        cur = b
            .add_logical(Logical::Restride {
                specs,
                bounds,
                x: cur,
            })
            .ok()?;
    }
    Some(cur)
}
