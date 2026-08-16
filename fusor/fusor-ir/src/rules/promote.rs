//! **PROMOTE** — a free axis of a reduction nest moves from the ITERATION
//! domain into the ACCUMULATOR's data space.
//!
//! A nest over free domain `F u {d}` and reduction axis `a` equals a nest over
//! free domain `F` whose every slot is replicated over `d`:
//! `SlotTy::Scalar -> SlotTy::Vector(D_d)`.
//!
//! ```text
//! Fold{space, axis a, vec_axes V, C}  ==  Fold{space, axis a, vec_axes V u {d}, C.promote(D_d)}
//! ```
//!
//! `d` is free, so the rewrite is unconditionally value-preserving and
//! carries no `reassoc` guard. Only the footprint changes, and footprint is a
//! legality guard against caps, not a cost term.
//!
//! The spelling is a rebinding, not a deletion: `space` is unchanged, only
//! the partition point between iterated and accumulated axes moves. Operand
//! address maps are stated against the full `space` and are untouched. The
//! node's own expressions — `carrier.lift`, `carrier.merge`, `post` — are
//! written against [`Launch::iter_space`], so every `IndexOf(j)` with `j > d`
//! renumbers down by one; the positionwise guard is the statement that there
//! is no `IndexOf(d)` to lose.
//!
//! Repeated promotion coalesces in the algebra: an existing `Vector(d0)`
//! becomes `Vector(d0 * extent)`, row-major over `vec_axes` in ascending
//! order, so `TM x TN` register tiling is two steps of this rewrite. The rule
//! mints the first step only; see the note beside [`promote`].
//!
//! The inverse — flattening a promoted slot back into a free axis — is minted
//! too, so promotion and no promotion stay live in one class and compete on
//! cost. Partial promotion needs no mode of its own: a strip-mine splits `D`
//! into `(D/DB, DB)` and this law promotes the inner factor.
//!
//! [`CORE_RULES`](crate::rules::CORE_RULES) registers `PROMOTE`; it does not
//! register [`PROMOTE_FLATTEN`].

use crate::carrier::{Carrier, SlotTy};
use crate::device::Caps;
use crate::egraph::{Builder, Facts, Id, RuleTag};
use crate::ir::launch::{IndexSpace, Launch};
use crate::ir::{Level, Node, Op, OpTag};
use crate::rule;
use crate::scalar::{ScalarExpr, ScalarKind};
use crate::shape::{Dim, Dims, StrideSpec};
use smallvec::SmallVec;

rule!(
    PROMOTE,
    level = Level::Launch,
    head = OpTag::LaunchFold,
    tag = RuleTag::Additive,
    apply = promote,
);

rule!(
    PROMOTE_FLATTEN,
    level = Level::Launch,
    head = OpTag::LaunchFold,
    tag = RuleTag::Additive,
    apply = promote_flatten,
);

/// Bytes one invocation may hold in private accumulator registers.
///
/// `Caps` has no portable register-budget fact, so this is a conservative
/// cross-backend policy: 256 B is 64 `f32` lanes, the widest accumulator the
/// shipped geometries ask for. Over budget the rule declines; the fallback is
/// strip-then-promote, reached on a later round.
pub fn private_acc_bytes(caps: &Caps) -> u64 {
    let _ = caps;
    256
}

/// One promotion's worth of node state. `space`, `ops` and `sched` never
/// change, which is the whole point of the rebinding spelling.
#[derive(Clone)]
struct Promoted {
    vec_axes: SmallVec<[u32; 2]>,
    carrier: Carrier,
    post: SmallVec<[ScalarExpr; 4]>,
}

/// Move the innermost free axis into the accumulator.
///
/// `d` is position `axis - 1`, and the promoted nest joins the unpromoted
/// one's class, so promotion and no promotion are both live and compete on
/// cost. Which other free axis to promote instead stays reachable through the
/// interchange the schedule domain carries.
pub fn promote(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::Launch(Launch::Fold {
        space,
        axis,
        vec_axes,
        carrier,
        acc,
        post,
        ops,
        sched,
    }) = &node.op
    else {
        return None;
    };
    // One axis per firing, and only out of a nest whose accumulator is still
    // wholly in the iteration domain. See the note below the function.
    if !vec_axes.is_empty() {
        return None;
    }
    let axis = *axis as usize;
    equal_slot_lanes(carrier)?;
    let want = f.own().shape.clone();

    let state = Promoted {
        vec_axes: vec_axes.clone(),
        carrier: carrier.clone(),
        post: post.clone(),
    };
    let next = promote_once(space, axis, *acc, &state, f)?;
    let got = fold_out_shape(space, axis, &next.vec_axes, &next.carrier)?;
    // The output shape is identical whenever the carrier appended nothing
    // before; a carrier that already had a slot axis absorbs the promoted
    // axis and a pure alias puts it back. Decide before minting.
    let view = recovery_view(carrier, &got, &want)?;
    let fold = b
        .add_launch(Launch::Fold {
            space: space.clone(),
            axis: axis as u32,
            vec_axes: next.vec_axes,
            carrier: next.carrier,
            acc: *acc,
            post: next.post,
            ops: ops.clone(),
            sched: sched.clone(),
        })
        .ok()?;
    let value = apply_view(b, fold, &view)?;

    // This law does not change the node's `ValueFacts` at all: a botched
    // renumbering or mis-ordered recovery view shows up here as a shape
    // mismatch instead of as a wrong number on a device.
    if !shapes_eq(&b.facts_of(value).shape, &want) {
        return None;
    }
    b.union(id, value).ok()
}

// One axis per firing: the first promotion of a single-slot carrier is
// shape-preserving — the axis leaves the free list and comes straight back as
// the carrier's lanes — but every promotion after it flattens two trailing
// axes into one and needs a recovery reshape, which crowds the shared
// saturation and extraction budgets on an alternative neither backend can
// lower today. `promote_declines_to_deepen_an_existing_promotion` pins the
// limit. Re-enabling the chain is deleting the `vec_axes.is_empty()` guard;
// the shape convention has to move first.
//
// A multi-slot carrier's FIRST promotion does need the alias — its slot axis
// was already there — and is minted, because there is one of those per nest
// rather than one per axis.

/// One step: promote the innermost remaining free axis, or `None`.
fn promote_once(
    space: &IndexSpace,
    axis: usize,
    acc: crate::dtype::Dtype,
    state: &Promoted,
    f: &Facts<'_>,
) -> Option<Promoted> {
    let d = promotable_axis(space, axis, &state.vec_axes)?;

    // 1. Positionwise in `d`. No expression on the node may read `IndexOf(d)`,
    //    and every `Arg` in a merge must be a slot reference. Checking `lift`
    //    and `post` too, not `merge` alone, is what stops a positional term
    //    being silently detached from its coordinate.
    //
    //    `d`'s iteration index *is* `d`: every already-promoted axis sits
    //    between `d` and `axis`, so removing them renumbers nothing at or
    //    below `d`.
    if !positionwise_in(&state.carrier, &state.post, d as u32) {
        return None;
    }

    // 2. A constant extent: a symbolic private array is allocatable on
    //    neither backend.
    let extent = *space.dims.get(d)?;
    let e = extent.as_const()?;
    // A unit axis promotes to a `Vector(1)` slot: the same one register under
    // a different name, so there is no alternative to mint.
    if e <= 1 {
        return None;
    }

    // 3. `acc` is wide enough for the value's contract. Read on `own()`, never
    //    `numeric(0)`: `own().numeric` is the meet over every operand and a
    //    multi-operand fold makes the operand-0 accessor blind to the rest.
    if acc.accum_bits() < f.own().numeric.min_accum_bits {
        return None;
    }

    // 4. Every slot must carry the same lane count, which is what
    //    `verify_launch`'s `lanes == positions * width` clause means and what
    //    makes slot readback a single strided view.
    equal_slot_lanes(&state.carrier)?;

    // 5. Footprint. The promoted accumulator lives in registers on both
    //    backends; over budget the rule declines rather than minting a plan
    //    `verify_plan` would have to reject.
    let promoted = state.carrier.promote(extent)?;
    let lanes = promoted.lanes()?;
    if lanes.checked_mul(acc.byte_size())? > private_acc_bytes(f.caps()) {
        return None;
    }

    // The rebinding. `space` is untouched and so are the operands; only the
    // partition point moves.
    let mut vec_axes: SmallVec<[u32; 2]> = SmallVec::with_capacity(state.vec_axes.len() + 1);
    vec_axes.push(d as u32);
    vec_axes.extend(state.vec_axes.iter().copied());

    let shift = |x: &ScalarExpr| shift_index_of(x, d as u32, -1);
    Some(Promoted {
        vec_axes,
        carrier: Carrier {
            lift: promoted.lift.iter().map(shift).collect(),
            merge: promoted.merge.iter().map(shift).collect(),
            ..promoted
        },
        post: state.post.iter().map(shift).collect(),
    })
}

/// The inverse: flatten the outermost promoted axis back into the iteration
/// domain, so full, partial and no promotion all stay live in one class.
///
/// The flattened axis's coordinate is one no expression referred to — it was
/// not in the iteration space — so the renumbering that reintroduces it is a
/// pure shift up.
pub fn promote_flatten(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::Launch(Launch::Fold {
        space,
        axis,
        vec_axes,
        carrier,
        acc,
        post,
        ops,
        sched,
    }) = &node.op
    else {
        return None;
    };
    let axis = *axis as usize;
    if axis + 1 != space.rank() || vec_axes.is_empty() {
        return None;
    }
    // The block is contiguous and ends at `axis - 1`; its *first* element is
    // the most recently promoted axis and the only one that can come back
    // without disturbing the row-major lane order of the rest.
    let d = *vec_axes.first()? as usize;
    if vec_axes
        .iter()
        .enumerate()
        .any(|(i, a)| *a as usize != d + i)
        || d + vec_axes.len() != axis
    {
        return None;
    }
    let e = space.dims.get(d)?.as_const()?;
    if e == 0 {
        return None;
    }
    let flattened = demote(carrier, e)?;

    // The flattened axis reappears in the free list exactly where the carrier
    // axis used to hold it, and only the `Vector(e) -> Scalar` case lands back
    // on the identical shape. A wider slot would need the trailing two axes
    // *merged*, which a `StrideSpec` vector cannot express (that is
    // `AccessPlan::Unflatten`'s job), so this declines rather than unioning two
    // different shapes into one class.
    let new_vec: SmallVec<[u32; 2]> = vec_axes.iter().skip(1).copied().collect();
    let want = f.own().shape.clone();
    let got = fold_out_shape(space, axis, &new_vec, &flattened)?;
    if !shapes_eq(&got, &want) {
        return None;
    }

    let shift = |x: &ScalarExpr| shift_index_of(x, d as u32, 1);
    let new_carrier = Carrier {
        lift: flattened.lift.iter().map(shift).collect(),
        merge: flattened.merge.iter().map(shift).collect(),
        ..flattened
    };
    let fold = b
        .add_launch(Launch::Fold {
            space: space.clone(),
            axis: axis as u32,
            vec_axes: new_vec,
            carrier: new_carrier,
            acc: *acc,
            post: post.iter().map(shift).collect(),
            ops: ops.clone(),
            sched: sched.clone(),
        })
        .ok()?;
    if !shapes_eq(&b.facts_of(fold).shape, &want) {
        return None;
    }
    b.union(id, fold).ok()
}

/// The innermost free axis of a well-formed reduction nest, or `None`.
///
/// `space` is `free.. ++ vec.. ++ [reduced]`. A node spelled any other way — a
/// reduced axis that is not last, a promoted block that is not the contiguous
/// run immediately before it — has no innermost free axis to move and declines
/// rather than guessing which axis the write map trails in.
fn promotable_axis(space: &IndexSpace, axis: usize, vec_axes: &[u32]) -> Option<usize> {
    if axis + 1 != space.rank() || axis < vec_axes.len() + 1 {
        return None;
    }
    let d = axis - vec_axes.len() - 1;
    vec_axes
        .iter()
        .enumerate()
        .all(|(i, a)| *a as usize == d + 1 + i)
        .then_some(d)
}

/// No expression on the node reads `IndexOf(axis)`, and every `Arg` a merge
/// reads is a slot of the accumulator — the condition that refuses to promote
/// an axis the accumulator depends on positionally.
fn positionwise_in(carrier: &Carrier, post: &[ScalarExpr], axis: u32) -> bool {
    if carrier.reads_index_of(axis) || post.iter().any(|e| reads_index_of(e, axis)) {
        return false;
    }
    let w = carrier.width() as u32;
    carrier
        .merge
        .iter()
        .all(|m| max_arg(m).is_none_or(|a| a < 2 * w))
}

/// Every slot carries the same lane count — `verify_launch`'s
/// `lanes == positions * width` clause, checked before minting rather than
/// after.
fn equal_slot_lanes(carrier: &Carrier) -> Option<u64> {
    let first = carrier.slots.first()?.lanes()?;
    carrier
        .slots
        .iter()
        .all(|s| s.lanes() == Some(first))
        .then_some(first)
}

/// The output shape of a `Fold`, spelled exactly as inference spells it: the
/// space minus the reduced axis and every promoted axis, then the carrier's
/// lane count appended.
fn fold_out_shape(
    space: &IndexSpace,
    axis: usize,
    vec_axes: &[u32],
    carrier: &Carrier,
) -> Option<Dims> {
    let mut shape: Dims = space
        .dims
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != axis && !vec_axes.contains(&(*i as u32)))
        .map(|(_, d)| *d)
        .collect();
    if let Some(d) = carrier.out_dim()? {
        shape.push(d);
    }
    Some(shape)
}

fn shapes_eq(a: &[Dim], b: &[Dim]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.known_eq(*y))
}

/// How the promoted fold's value is read back at the original shape: one
/// `(extent, relative multiplier)` per axis the trailing carrier axis splits
/// into, or empty when the shapes already agree.
type Recovery = SmallVec<[(u64, u32); 4]>;

/// Decide how to read the promoted node back at the pre-promotion shape, or
/// `None` when no strided view does it.
///
/// One axis of extent `e` moves onto a carrier of `w` slots, each of `q` lanes
/// (`q == 1` unless the carrier already carried a vector). The promoted node's
/// trailing carrier axis holds, at slot `s` and positions `(p, j)`, the lane
///
/// ```text
/// s*e*q + p*q + j
/// ```
///
/// and the wanted trailing axes are `e` followed by the original carrier axis
/// `w*q`, when the original carrier had one at all. That is a strided view in
/// exactly two cases:
///
/// * `w == 1` — the axis flattened into the one slot's own vector, so the view
///   is the contiguous reshape;
/// * `q == 1` — the slots were scalars, so the slot axis is the outermost
///   stride and the view transposes it past the promoted positions.
///
/// With several slots that are *already* vectors the lane index needs a
/// divmod of the wanted axis, which a `StrideSpec` vector cannot express, and
/// the rule declines.
fn recovery_view(base: &Carrier, got: &[Dim], want: &[Dim]) -> Option<Recovery> {
    if shapes_eq(got, want) {
        return Some(Recovery::new());
    }
    let q = equal_slot_lanes(base)?;
    let w = base.width();
    // Whether the original carrier appended an axis at all, read from the same
    // `out_dim` inference reads: a single `Scalar` slot appends nothing, a
    // `Vector(1)` slot appends a unit axis, and the two differ.
    let carried = usize::from(base.out_dim()?.is_some());
    let last = got.len().checked_sub(1)?;
    let slot_axis = q.checked_mul(w as u64)?;
    // The promoted extent, read off the two shapes rather than off the space:
    // it is what the flattened carrier axis holds beyond the original one.
    let e = got[last].as_const()? / slot_axis;
    if e.checked_mul(slot_axis)? != got[last].as_const()? {
        return None;
    }
    // `want` is the promoted node's shape with that axis put back, then the
    // original carrier axis if the original carrier had one.
    if want.len() != last + 1 + carried
        || !shapes_eq(&got[..last], &want[..last])
        || !want.get(last)?.known_eq(Dim::Const(e))
        || (carried == 1 && !want.last()?.known_eq(Dim::Const(slot_axis)))
    {
        return None;
    }

    let mut specs: Recovery = Recovery::new();
    specs.push((e, u32::try_from(q).ok()?));
    if carried == 1 {
        match (w, q) {
            (1, _) => specs.push((slot_axis, 1)),
            (_, 1) => specs.push((slot_axis, u32::try_from(e).ok()?)),
            _ => return None,
        }
    }
    Some(specs)
}

/// Mint the recovery view, if any: the pure alias that reads the promoted
/// node's flattened carrier axis back at the pre-promotion shape.
///
/// Minted at Launch, already lowered: the specs are handed to the same
/// [`composed_layout`](crate::rules::composed_layout) `LOWER_RESTRIDE` uses,
/// so the node is byte-for-byte the one that rule would have minted, without
/// spending saturation rounds on the `Restride -> Map -> operand plans`
/// cascade. Logical view algebra cannot compose with it; nothing needs to —
/// `FOLD_VIEWS_INTO_INDEX` and `SINK_EPILOGUE` both act at Launch.
fn apply_view(b: &mut Builder<'_>, fold: Id, view: &Recovery) -> Option<Id> {
    if view.is_empty() {
        return Some(fold);
    }
    let shape = b.facts_of(fold).shape.clone();
    let dtype = b.facts_of(fold).dtype;
    let last = shape.len().checked_sub(1)?;
    let covered = view.iter().try_fold(1u64, |a, (e, _)| a.checked_mul(*e))?;
    if covered != shape[last].as_const()? {
        return None;
    }

    let mut specs: SmallVec<[StrideSpec; 6]> = (0..last)
        .map(|j| StrideSpec::dim(j as u32, shape[j]))
        .collect();
    for (extent, mult) in view {
        specs.push(StrideSpec::dim_with(
            last as u32,
            Dim::Const(*extent),
            *mult,
        ));
    }
    let layout = crate::rules::composed_layout(&specs, &shape)?;
    let out: Dims = specs.iter().map(|s| s.size).collect();
    crate::rules::lower_floor::floor_alias_map(b, fold, layout, &out, dtype)
}

/// The inverse of [`Carrier::promote`] at one axis: `Vector(d0*e)` becomes
/// `Vector(d0)`, or `Scalar` when the whole slot was that axis.
fn demote(c: &Carrier, e: u64) -> Option<Carrier> {
    let slots: SmallVec<[SlotTy; 4]> = c
        .slots
        .iter()
        .map(|s| match s {
            SlotTy::Scalar => None,
            SlotTy::Vector(d) => {
                let n = d.as_const()?;
                if e == 0 || n % e != 0 {
                    return None;
                }
                Some(match n / e {
                    1 => SlotTy::Scalar,
                    r => SlotTy::Vector(Dim::Const(r)),
                })
            }
        })
        .collect::<Option<_>>()?;
    Some(Carrier { slots, ..c.clone() })
}

/// Renumber `IndexOf(j)` to `IndexOf(j + by)` for every `j > from` (shifting
/// down, `by < 0`) or `j >= from` (shifting up). Every other node rides
/// through untouched.
fn shift_index_of(e: &ScalarExpr, from: u32, by: i32) -> ScalarExpr {
    use ScalarKind as K;
    let rec = |x: &ScalarExpr| shift_index_of(x, from, by);
    match e.kind() {
        K::IndexOf(a) => {
            let moves = if by < 0 { *a > from } else { *a >= from };
            if moves {
                ScalarExpr::index_of(a.wrapping_add_signed(by))
            } else {
                e.clone()
            }
        }
        K::Un { op, x } => ScalarExpr::un(*op, rec(x)),
        K::Bin { op, a, b } => ScalarExpr::bin(*op, rec(a), rec(b)),
        K::Cmp { op, a, b } => ScalarExpr::cmp(*op, rec(a), rec(b)),
        K::Select { c, t, f } => ScalarExpr::select(rec(c), rec(t), rec(f)),
        K::Cast { to, x } => ScalarExpr::cast(*to, rec(x)),
        K::Bitcast { to, x } => ScalarExpr::bitcast(*to, rec(x)),
        K::Round { mode, x } => ScalarExpr::round(*mode, rec(x)),
        _ => e.clone(),
    }
}

fn reads_index_of(e: &ScalarExpr, axis: u32) -> bool {
    use ScalarKind as K;
    match e.kind() {
        K::IndexOf(a) => *a == axis,
        K::Un { x, .. } | K::Cast { x, .. } | K::Bitcast { x, .. } | K::Round { x, .. } => {
            reads_index_of(x, axis)
        }
        K::Bin { a, b, .. } | K::Cmp { a, b, .. } | K::Dot { a, b } => {
            reads_index_of(a, axis) || reads_index_of(b, axis)
        }
        K::Select { c, t, f } => {
            reads_index_of(c, axis) || reads_index_of(t, axis) || reads_index_of(f, axis)
        }
        K::Splat { x, .. } => reads_index_of(x, axis),
        _ => false,
    }
}

/// The largest `Arg` index an expression reads.
fn max_arg(e: &ScalarExpr) -> Option<u32> {
    use ScalarKind as K;
    match e.kind() {
        K::Arg(i) => Some(*i),
        K::Un { x, .. }
        | K::Cast { x, .. }
        | K::Bitcast { x, .. }
        | K::Round { x, .. }
        | K::Splat { x, .. } => max_arg(x),
        K::Bin { a, b, .. } | K::Cmp { a, b, .. } | K::Dot { a, b } => {
            max_arg(a).into_iter().chain(max_arg(b)).max()
        }
        K::Select { c, t, f } => max_arg(c)
            .into_iter()
            .chain(max_arg(t))
            .chain(max_arg(f))
            .max(),
        _ => None,
    }
}
