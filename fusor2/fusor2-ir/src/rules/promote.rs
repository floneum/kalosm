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
/// TODO: belongs in `DeviceFacts` as a calibrated field; neither `Caps` nor
/// `DeviceFacts` carries a register budget today. 256 B is 64 `f32` lanes,
/// the widest accumulator the shipped geometries ask for. Over budget the
/// rule declines; the fallback is strip-then-promote, reached on a later
/// round.
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
    if vec_axes.iter().enumerate().any(|(i, a)| *a as usize != d + i)
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
        specs.push(StrideSpec::dim_with(last as u32, Dim::Const(*extent), *mult));
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
    Some(Carrier {
        slots,
        ..c.clone()
    })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::carrier::SlotTy;
    use crate::dtype::{Dtype, Splat};
    use crate::ir::logical::Logical;
    use crate::egraph::{EGraph, Rule, SaturationBudget, Saturate};
    use crate::ir::logical::{EinSpec, Label};
    use crate::ir::launch::{AccessPlan, Operand, ScheduleDomain, SgemmParams};
    use crate::rules::test_support as ts;
    use crate::rules::{CORE_RULES, alias_operand_of, ident_expr};
    use crate::saturate::CoreSaturate;
    use crate::scalar::{BinOp, CmpOp, UnOp};
    use smallvec::smallvec;

    fn fire(g: &mut EGraph, id: Id, r: &Rule) -> Option<Id> {
        let caps = ts::caps();
        let node = g.node(id).clone();
        let facts = g.facts_view(id, &caps);
        let mut b = g.builder(&caps);
        (r.apply)(&mut b, id, &node, &facts)
    }

    fn dims(v: &[u64]) -> Vec<Dim> {
        v.iter().copied().map(Dim::Const).collect()
    }

    /// A `Fold` with an explicit `post` vector, which `ts::kfold` cannot
    /// spell because it takes one expression.
    #[allow(clippy::too_many_arguments)]
    fn kfold_n(
        g: &mut EGraph,
        space: &[Dim],
        axis: u32,
        vec_axes: &[u32],
        carrier: Carrier,
        acc: Dtype,
        post: &[ScalarExpr],
        ops: Vec<Operand>,
    ) -> Id {
        g.add(Op::Launch(Launch::Fold {
            space: IndexSpace::new(space.iter().copied()),
            axis,
            vec_axes: vec_axes.iter().copied().collect(),
            carrier,
            acc,
            post: post.iter().cloned().collect(),
            ops,
            sched: ScheduleDomain::Point,
        }))
        .unwrap()
    }

    /// The plain sum-of-products nest every dense contraction lowers to.
    fn dot_carrier() -> Carrier {
        Carrier::binop(BinOp::Add, Splat::F32(0.0), Dtype::F32).with_lift([ScalarExpr::bin(
            BinOp::Mul,
            ScalarExpr::arg(0, Dtype::F32),
            ScalarExpr::arg(1, Dtype::F32),
        )])
    }

    fn kfold_of(g: &EGraph, id: Id) -> Option<Launch> {
        match &g.node(id).op {
            Op::Launch(k @ Launch::Fold { .. }) => Some(k.clone()),
            _ => None,
        }
    }

    /// The node an identity-bodied alias `Map` reads — the shape this law
    /// mints its recovery view in.
    fn view_source(g: &EGraph, id: Id) -> Option<Id> {
        let Op::Launch(Launch::Map { body, ops, .. }) = &g.node(id).op else {
            return None;
        };
        let [op] = &ops[..] else { return None };
        (matches!(body.kind(), ScalarKind::Arg(0)) && op.access == AccessPlan::Alias)
            .then_some(op.src)
    }

    /// Every promoted `Fold` in `id`'s class, following a recovery view.
    fn promoted_members(g: &EGraph, id: Id) -> Vec<(Id, Launch)> {
        g.chain(id)
            .into_iter()
            .flat_map(|m| {
                let direct = kfold_of(g, m).map(|k| (m, k));
                let viewed = view_source(g, m).and_then(|x| kfold_of(g, x).map(|k| (x, k)));
                direct.into_iter().chain(viewed)
            })
            .filter(|(_, k)| matches!(k, Launch::Fold { vec_axes, .. } if !vec_axes.is_empty()))
            .collect()
    }

    /// The nest with `n` axes promoted, following a recovery view when the
    /// carrier's own slot axis made one necessary.
    fn promoted_at(g: &EGraph, id: Id, n: usize) -> Option<(Id, Launch)> {
        promoted_members(g, id)
            .into_iter()
            .find(|(_, k)| matches!(k, Launch::Fold { vec_axes, .. } if vec_axes.len() == n))
    }

    /// **Every member of the class carries the same `ValueFacts`.** A promoted
    /// nest whose own shape lost the trailing pair is not a member: what joined
    /// the class is the view over it, so extraction cannot select a value of a
    /// different shape.
    fn class_facts_agree(g: &EGraph, id: Id) {
        let want = g.facts(id).clone();
        for m in g.chain(id) {
            assert_eq!(g.facts(m), &want, "class member {m} disagrees on facts");
        }
    }

    fn lanes_of(k: &Launch) -> u64 {
        let Launch::Fold { carrier, .. } = k else {
            panic!("not a fold")
        };
        carrier.lanes().unwrap()
    }

    /// A planner that admits everything. Every nest this law mints carries
    /// `ScheduleDomain::Point`, so the verifier never asks it for a byte
    /// figure; it exists so this module can run the **real** `verify_launch` over
    /// the nodes the law mints rather than asserting they are well-formed.
    struct NullPlanner;

    impl crate::ir::kernel::ArenaPlanner for NullPlanner {
        fn arena_plan(
            &self,
            _ir: &crate::ir::kernel::KernelIr,
            _caps: &Caps,
        ) -> crate::error::Result<crate::ir::kernel::ArenaPlan> {
            Ok(crate::ir::kernel::ArenaPlan {
                mode: crate::ir::kernel::ArenaMode::Regions,
                total_bytes: 0,
                placements: Default::default(),
                barriers_inserted: Default::default(),
            })
        }
        fn workgroup_bytes(
            &self,
            _tiles: &crate::ir::kernel::Tiles,
            _caps: &Caps,
        ) -> crate::error::Result<u32> {
            Ok(0)
        }
        fn barrier_suggestions(
            &self,
            _ir: &crate::ir::kernel::KernelIr,
        ) -> Vec<crate::ir::kernel::BarrierSuggestion> {
            Vec::new()
        }
        fn verify_arena(
            &self,
            _ir: &crate::ir::kernel::KernelIr,
            _plan: &crate::ir::kernel::ArenaPlan,
        ) -> crate::error::Result<()> {
            Ok(())
        }
        fn verify_uniformity(
            &self,
            _ir: &crate::ir::kernel::KernelIr,
        ) -> crate::error::Result<()> {
            Ok(())
        }
    }

    /// Run the shipped `verify_launch` over one node of a graph.
    fn verify(g: &EGraph, id: Id) -> crate::error::Result<()> {
        let caps = ts::caps();
        let registry = crate::ir::OpDefRegistry::new();
        let node = g.node(id).clone();
        let operands: Vec<crate::facts::ValueFacts> = node
            .children
            .iter()
            .map(|c| g.facts(*c).clone())
            .collect();
        let result = g.facts(id).clone();
        let cx = crate::ir::VerifyCtx {
            node: &node,
            id,
            operands: &operands,
            result: &result,
            caps: &caps,
            registry: &registry,
        };
        crate::verify_launch::verify_launch(&cx, &NullPlanner)
    }

    /// This law's share of the shared saturation round budget, pinned as a
    /// delta: the round count with the law may not exceed the count without
    /// it by more than one, on the same graph.
    ///
    /// The graph is the chain a fused attention frontend emits (`q.k`, scale,
    /// `max`, `sub`, `exp`, `sum`, `div`, `p.v`), the densest arrangement of
    /// reduction nests available in this crate. Nothing in the law reads it.
    #[test]
    fn promote_does_not_overrun_the_shared_round_budget() {
        use crate::shape::BoundsProof;
        let (b, h, lq, lk, dh) = (2u64, 3, 8, 8, 16);
        let build = |g: &mut EGraph| {
            let q = ts::buffer(g, Dtype::F32, &dims(&[b, h, lq, dh]));
            let k = ts::buffer(g, Dtype::F32, &dims(&[b, h, lk, dh]));
            let v = ts::buffer(g, Dtype::F32, &dims(&[b, h, lk, dh]));
            let qk = EinSpec {
                a: smallvec![Label(0), Label(1), Label(2), Label(4)],
                b: smallvec![Label(0), Label(1), Label(3), Label(4)],
                out: smallvec![Label(0), Label(1), Label(2), Label(3)],
            };
            let s = ts::contract(g, qk, Dtype::F32, q, k);
            let f32a = |i| ScalarExpr::arg(i, Dtype::F32);
            let scaled = ts::map(
                g,
                ScalarExpr::bin(BinOp::Mul, f32a(0), ScalarExpr::lit(Splat::F32(0.25))),
                &[s],
            );
            let m = ts::fold(
                g,
                ts::binop_carrier(BinOp::Max, Dtype::F32),
                3,
                Dtype::F32,
                scaled,
            );
            let bcast = |g: &mut EGraph, x: Id| {
                g.add(Op::Logical(Logical::Restride {
                    specs: smallvec![
                        StrideSpec::dim(0, Dim::Const(b)),
                        StrideSpec::dim(1, Dim::Const(h)),
                        StrideSpec::dim(2, Dim::Const(lq)),
                        StrideSpec::broadcast(Dim::Const(lk)),
                    ],
                    bounds: BoundsProof::Static,
                    x,
                }))
                .unwrap()
            };
            let mb = bcast(g, m);
            let sub = ts::map(g, ScalarExpr::bin(BinOp::Sub, f32a(0), f32a(1)), &[scaled, mb]);
            let e = ts::map(g, ScalarExpr::un(UnOp::Exp, f32a(0)), &[sub]);
            let l = ts::fold(g, ts::binop_carrier(BinOp::Add, Dtype::F32), 3, Dtype::F32, e);
            let lb = bcast(g, l);
            let p = ts::map(g, ScalarExpr::bin(BinOp::Div, f32a(0), f32a(1)), &[e, lb]);
            let pv = EinSpec {
                a: smallvec![Label(0), Label(1), Label(2), Label(3)],
                b: smallvec![Label(0), Label(1), Label(3), Label(4)],
                out: smallvec![Label(0), Label(1), Label(2), Label(4)],
            };
            ts::contract(g, pv, Dtype::F32, p, v)
        };

        let mut rounds: Vec<(&str, u32)> = Vec::new();
        for (name, rules) in [
            ("with PROMOTE", CORE_RULES.to_vec()),
            (
                "no PROMOTE",
                CORE_RULES
                    .iter()
                    .filter(|r| r.name != "PROMOTE")
                    .cloned()
                    .collect::<Vec<_>>(),
            ),
        ] {
            let mut g = ts::graph();
            let out = build(&mut g);
            g.add_root(out);
            let r = crate::saturate::Driver::new()
                .saturate(
                    &mut g,
                    &ts::caps(),
                    &rules,
                    SaturationBudget::default(),
                )
                .unwrap();
            assert!(
                r.saturated && r.truncated.is_empty(),
                "{name}: {} rounds, {} nodes, not a fixpoint inside the shipped budget",
                r.rounds,
                r.final_nodes
            );
            if name == "with PROMOTE" {
                assert!(
                    r.fired.iter().any(|(n, c)| *n == "PROMOTE" && *c > 0),
                    "PROMOTE never fired, so the budget claim is vacuous: {:?}",
                    r.fired
                );
            }
            rounds.push((name, r.rounds));
        }
        let [(_, with), (_, without)] = rounds[..] else {
            unreachable!()
        };
        assert!(
            with <= without + 1,
            "PROMOTE costs {} extra rounds ({with} vs {without}) of a budget of {}",
            with - without,
            SaturationBudget::default().max_rounds
        );
    }

    #[test]
    fn promote_fires_on_a_saturated_contraction_nest() {
        let (b, h, lq, lk, dh) = (2u64, 3, 5, 7, 16);
        let mut g = ts::graph();
        let p = ts::buffer(&mut g, Dtype::F32, &dims(&[b, h, lq, lk]));
        let v = ts::buffer(&mut g, Dtype::F32, &dims(&[b, h, lk, dh]));
        let spec = EinSpec {
            a: smallvec![Label(0), Label(1), Label(2), Label(3)],
            b: smallvec![Label(0), Label(1), Label(3), Label(4)],
            out: smallvec![Label(0), Label(1), Label(2), Label(4)],
        };
        let out = ts::contract(&mut g, spec, Dtype::F32, p, v);
        let before = g.facts(out).clone();

        // The shipped table plus the inverse, which the table does not
        // register.
        assert_eq!(crate::rules::rule_id("PROMOTE").map(|_| ()), Some(()));
        let mut rules: Vec<Rule> = CORE_RULES.to_vec();
        rules.push(PROMOTE_FLATTEN);
        let report = CoreSaturate
            .saturate(&mut g, &ts::caps(), &rules, SaturationBudget::default())
            .unwrap();
        assert!(report.saturated, "{report:?}");
        assert!(
            report.fired.iter().any(|(n, c)| *n == "PROMOTE" && *c > 0),
            "PROMOTE did not fire: {:?}",
            report.fired
        );

        // One promoted nest: the innermost free axis of the nest the lowering
        // floor minted, which is `Dh`.
        let promoted = promoted_members(&g, out);
        assert_eq!(promoted.len(), 1, "expected exactly one promoted nest");
        class_facts_agree(&g, out);
        let (pid, k) = &promoted[0];
        let Launch::Fold {
            space,
            axis,
            vec_axes,
            carrier,
            ..
        } = k
        else {
            panic!()
        };
        // The rebinding: `space` is unchanged, only the partition point moved.
        assert_eq!(&space.dims[..], &dims(&[b, h, lq, dh, lk])[..]);
        assert_eq!(*axis, 4);
        assert_eq!(&vec_axes[..], &[3]);
        assert_eq!(carrier.slots[..], [SlotTy::Vector(Dim::Const(dh))]);
        assert_eq!(carrier.lanes(), Some(dh));
        // The iteration space is now the *score* space, which is what makes a
        // producer over that space absorbable with no renumbering.
        assert_eq!(&k.iter_space().dims[..], &dims(&[b, h, lq, lk])[..]);
        // And the node's facts are byte-for-byte the unpromoted node's.
        assert_eq!(g.facts(*pid), &before, "PROMOTE changed ValueFacts");
    }

    /// The invariant on its own, stated as the design states it: a law that
    /// does not change the node's `ValueFacts` at all.
    #[test]
    fn promotion_does_not_change_value_facts() {
        let mut g = ts::graph();
        let space = dims(&[4, 8, 16]);
        let x = ts::buffer(&mut g, Dtype::F32, &space);
        let fold = ts::kfold(
            &mut g,
            &space,
            2,
            ts::binop_carrier(BinOp::Add, Dtype::F32),
            Dtype::F32,
            ident_expr(Dtype::F32),
            vec![alias_operand_of(x, &space)],
        );
        let before = g.facts(fold).clone();
        assert!(fire(&mut g, fold, &PROMOTE).is_some());
        class_facts_agree(&g, fold);

        // The innermost free axis leaves the iteration domain and comes
        // straight back as the carrier's lanes, so the node's facts are the
        // ones it had — no view, nothing to renumber wrong that would not show
        // up here.
        let (pid, k) = promoted_at(&g, fold, 1).unwrap();
        assert_eq!(g.facts(pid), &before);
        assert_eq!(&k.iter_space().dims[..], &dims(&[4, 16])[..]);
        assert_eq!(lanes_of(&k), 8);
        assert!(g.chain(fold).contains(&pid), "the nest is not in the class");
    }

    /// **Three hand-enumerated register-tiling domains are one law.**
    ///
    /// `SgemmParams::legal` generates the `tn` values the reference's
    /// regression tree hand-enumerated. Promoting the innermost free axis of a
    /// plain sum-of-products nest whose extent is `tn` yields a carrier of
    /// exactly `tn` lanes — for every one of them. The two sets are computed
    /// independently and compared as sets.
    #[test]
    fn promote_derives_the_sgemm_tn_set() {
        let caps = ts::caps();
        let mut want: Vec<u64> = Vec::new();
        for bm in [32u32, 64, 128] {
            for bn in [32u32, 64, 128] {
                for bk in [8u32, 16] {
                    for tm in [1u32, 2, 4, 8] {
                        for tn in [1u32, 2, 4, 8] {
                            for double_buffer in [false, true] {
                                let p = SgemmParams {
                                    double_buffer,
                                    bm,
                                    bn,
                                    bk,
                                    tm,
                                    tn,
                                };
                                if p.legal(
                                    4,
                                    caps.limits.max_compute_workgroup_storage_size,
                                    caps.limits.max_compute_invocations_per_workgroup,
                                ) && !want.contains(&u64::from(tn))
                                {
                                    want.push(u64::from(tn));
                                }
                            }
                        }
                    }
                }
            }
        }
        want.sort_unstable();
        assert!(want.len() >= 3, "the SGEMM domain is empty: {want:?}");

        // `tn == 1` is the *unpromoted* nest — one output per lane in `n`, no
        // register tile — so it is the identity of this law rather than a
        // firing of it. Every other value is derived.
        let mut got: Vec<u64> = vec![1];
        for &tn in want.iter().filter(|&&t| t > 1) {
            // An SGEMM-shaped nest: no attention, no exp, no multi-slot
            // carrier. `[m, tn, k]` reducing `k`.
            let mut g = ts::graph();
            let space = dims(&[64, tn, 32]);
            let a = ts::buffer(&mut g, Dtype::F32, &space);
            let bb = ts::buffer(&mut g, Dtype::F32, &space);
            let fold = kfold_n(
                &mut g,
                &space,
                2,
                &[],
                dot_carrier(),
                Dtype::F32,
                &[ident_expr(Dtype::F32)],
                vec![alias_operand_of(a, &space), alias_operand_of(bb, &space)],
            );
            assert!(fire(&mut g, fold, &PROMOTE).is_some(), "tn = {tn}");
            let (_, k) = promoted_at(&g, fold, 1).unwrap();
            got.push(lanes_of(&k));
        }
        got.sort_unstable();
        assert_eq!(got, want, "PROMOTE does not derive the SgemmParams tn set");
    }

    /// **`TM x TN` is two steps of this rewrite, and the algebra composes.**
    ///
    /// The carrier is where the claim lives: promoting `TN` and then `TM`
    /// coalesces into ONE `Vector(TM*TN)` slot, row-major, with the lift and
    /// merge untouched — so a register tile is not a second mechanism beside
    /// the lane tile, it is this one applied twice. The nest that carries it is
    /// built here and put through the shipped `verify_launch`, which is what makes
    /// "the algebra composes" a statement about a runnable node rather than
    /// about a struct.
    ///
    /// The **rule** mints only the first step; `promote_declines_to_deepen_an_
    /// existing_promotion` pins that and says why.
    #[test]
    fn the_carrier_algebra_coalesces_to_tm_times_tn() {
        let (tm, tn) = (4u64, 8u64);
        let base = dot_carrier();
        let one = base.promote(Dim::Const(tn)).unwrap();
        let two = one.promote(Dim::Const(tm)).unwrap();
        assert_eq!(one.slots[..], [SlotTy::Vector(Dim::Const(tn))]);
        assert_eq!(two.slots[..], [SlotTy::Vector(Dim::Const(tm * tn))]);
        assert_eq!(two.lanes(), Some(tm * tn));
        // Promotion touches `slots` alone: the same arithmetic, replicated.
        assert_eq!((&base.lift, &base.merge), (&two.lift, &two.merge));

        // And the nest it belongs to is well-formed: `[m/TM, n/TN, TM, TN, k]`
        // reducing `k`, with both tile axes in the accumulator.
        let mut g = ts::graph();
        let space = dims(&[2, 3, tm, tn, 16]);
        let a = ts::buffer(&mut g, Dtype::F32, &space);
        let bb = ts::buffer(&mut g, Dtype::F32, &space);
        let tiled = kfold_n(
            &mut g,
            &space,
            4,
            &[2, 3],
            two,
            Dtype::F32,
            &[ident_expr(Dtype::F32)],
            vec![alias_operand_of(a, &space), alias_operand_of(bb, &space)],
        );
        verify(&g, tiled).expect("a TM x TN nest must verify");
        // Its shape is the tile flattened into the carrier axis — which is
        // exactly why the rule does not mint it into the unpromoted node's
        // class today.
        assert_eq!(&g.facts(tiled).shape[..], &dims(&[2, 3, tm * tn])[..]);
    }

    /// The rule promotes one free axis and declines to deepen an existing
    /// promotion; see the note beside `promote`. Delete this test with the
    /// `vec_axes.is_empty()` guard.
    #[test]
    fn promote_declines_to_deepen_an_existing_promotion() {
        let (tm, tn) = (4u64, 8u64);
        let mut g = ts::graph();
        let space = dims(&[2, 3, tm, tn, 16]);
        let a = ts::buffer(&mut g, Dtype::F32, &space);
        let bb = ts::buffer(&mut g, Dtype::F32, &space);
        let fold = kfold_n(
            &mut g,
            &space,
            4,
            &[],
            dot_carrier(),
            Dtype::F32,
            &[ident_expr(Dtype::F32)],
            vec![alias_operand_of(a, &space), alias_operand_of(bb, &space)],
        );
        let want = g.facts(fold).clone();

        // Step one lands, and lands shape-preserving.
        assert!(fire(&mut g, fold, &PROMOTE).is_some());
        class_facts_agree(&g, fold);
        let (first_id, first) = promoted_at(&g, fold, 1).unwrap();
        assert_eq!(lanes_of(&first), tn);
        assert_eq!(g.facts(first_id), &want);
        assert_eq!(&first.iter_space().dims[..], &dims(&[2, 3, tm, 16])[..]);

        // Step two does not, from either the original node or the promoted one.
        assert!(promoted_at(&g, fold, 2).is_none());
        assert!(
            fire(&mut g, first_id, &PROMOTE).is_none(),
            "the rule deepened a promotion; the recovery-view cost is back"
        );
    }

    /// The CPU lane tile: promoting a free axis by exactly the SIMD width `W`
    /// is `MapTiling::vector` — the same integer, reached by the same law that
    /// produced `TN`.
    #[test]
    fn promote_derives_the_cpu_lane_tile() {
        let caps = ts::caps();
        for &w in caps.simd_widths.iter() {
            let mut g = ts::graph();
            let space = dims(&[6, u64::from(w), 12]);
            let x = ts::buffer(&mut g, Dtype::F32, &space);
            // An elementwise chain feeding a reduction: `sum_k exp(x)`.
            let lift = ScalarExpr::un(UnOp::Exp, ScalarExpr::arg(0, Dtype::F32));
            let fold = kfold_n(
                &mut g,
                &space,
                2,
                &[],
                Carrier::binop(BinOp::Add, Splat::F32(0.0), Dtype::F32).with_lift([lift]),
                Dtype::F32,
                &[ident_expr(Dtype::F32)],
                vec![alias_operand_of(x, &space)],
            );
            assert!(fire(&mut g, fold, &PROMOTE).is_some(), "W = {w}");
            let (_, k) = promoted_at(&g, fold, 1).unwrap();
            assert_eq!(lanes_of(&k), u64::from(w));
        }
    }

    /// **Refuses to promote an axis the accumulator itself depends on.** One
    /// case per expression the guard covers; `merge` alone would let a
    /// positional term be silently detached from its coordinate.
    #[test]
    fn promote_declines_when_an_expression_reads_the_axis() {
        let space = dims(&[4, 8, 16]);
        let idx = ScalarExpr::cast(Dtype::F32, ScalarExpr::index_of(1));
        let sum = || Carrier::binop(BinOp::Add, Splat::F32(0.0), Dtype::F32);

        // (a) the merge reads it — an accumulator that is a function of the
        // axis is not replicable over it.
        let mut merge_reads = sum();
        merge_reads.merge[0] = ScalarExpr::bin(BinOp::Add, merge_reads.merge[0].clone(), idx.clone());
        // (b) the lift reads it: an ALiBi / rope / positional term.
        let lift_reads = sum().with_lift([ScalarExpr::bin(
            BinOp::Add,
            ScalarExpr::arg(0, Dtype::F32),
            idx.clone(),
        )]);

        for (name, carrier, post) in [
            ("merge", merge_reads, ident_expr(Dtype::F32)),
            ("lift", lift_reads, ident_expr(Dtype::F32)),
            (
                "post",
                sum(),
                ScalarExpr::bin(BinOp::Add, ScalarExpr::arg(0, Dtype::F32), idx.clone()),
            ),
        ] {
            let mut g = ts::graph();
            let x = ts::buffer(&mut g, Dtype::F32, &space);
            let fold = kfold_n(
                &mut g,
                &space,
                2,
                &[],
                carrier,
                Dtype::F32,
                &[post],
                vec![alias_operand_of(x, &space)],
            );
            assert!(
                fire(&mut g, fold, &PROMOTE).is_none(),
                "promoted an axis the {name} reads"
            );
        }
    }

    /// A symbolic extent declines: a private array of unknown width is
    /// allocatable on neither backend, so a too-permissive guard is a crash
    /// rather than a slow plan.
    #[test]
    fn promote_declines_on_a_symbolic_extent() {
        let mut g = ts::graph();
        let sym = Dim::Sym(g.fresh_sym());
        let space = [Dim::Const(4), sym, Dim::Const(16)];
        let x = ts::buffer(&mut g, Dtype::F32, &space);
        let fold = ts::kfold(
            &mut g,
            &space,
            2,
            ts::binop_carrier(BinOp::Add, Dtype::F32),
            Dtype::F32,
            ident_expr(Dtype::F32),
            vec![alias_operand_of(x, &space)],
        );
        assert!(fire(&mut g, fold, &PROMOTE).is_none());
    }

    /// Over the register budget the rule DECLINES — and a strip-mine of the
    /// same axis into `(D/DB, DB)` makes the inner factor promotable on the
    /// next round. Partial promotion needs no mode of its own; this is the
    /// continuous fallback.
    ///
    /// The strip is spelled by hand: `algebra::STRIP` splits the **reduced**
    /// axis, and splitting a *free* axis of the nest is a different rewrite
    /// that no rule mints today. That gap is why this is a fixture and not a
    /// two-rule saturation.
    #[test]
    fn promote_declines_over_budget_and_a_strip_rescues_it() {
        let caps = ts::caps();
        let budget = private_acc_bytes(&caps) / Dtype::F32.byte_size();
        let wide = budget * 2;

        let mut g = ts::graph();
        let space = dims(&[4, wide, 16]);
        let x = ts::buffer(&mut g, Dtype::F32, &space);
        let fold = ts::kfold(
            &mut g,
            &space,
            2,
            ts::binop_carrier(BinOp::Add, Dtype::F32),
            Dtype::F32,
            ident_expr(Dtype::F32),
            vec![alias_operand_of(x, &space)],
        );
        assert!(
            fire(&mut g, fold, &PROMOTE).is_none(),
            "{wide} lanes is over the {budget}-lane budget"
        );

        // The strip: the same nest with the wide axis split into (2, budget).
        let stripped_space = dims(&[4, 2, budget, 16]);
        let sx = ts::buffer(&mut g, Dtype::F32, &stripped_space);
        let stripped = ts::kfold(
            &mut g,
            &stripped_space,
            3,
            ts::binop_carrier(BinOp::Add, Dtype::F32),
            Dtype::F32,
            ident_expr(Dtype::F32),
            vec![alias_operand_of(sx, &stripped_space)],
        );
        assert!(fire(&mut g, stripped, &PROMOTE).is_some());
        let (_, k) = promoted_at(&g, stripped, 1).unwrap();
        assert_eq!(lanes_of(&k), budget);
        // And the chain stops there: promoting the next axis out would be
        // `4 * budget` lanes, four times over.
        assert!(promoted_at(&g, stripped, 2).is_none());
    }

    /// A reduced axis that is not last has no innermost free axis to move.
    #[test]
    fn promote_declines_when_the_reduced_axis_is_not_last() {
        let mut g = ts::graph();
        let space = dims(&[4, 8, 16]);
        let x = ts::buffer(&mut g, Dtype::F32, &space);
        for axis in [0u32, 1] {
            let fold = ts::kfold(
                &mut g,
                &space,
                axis,
                ts::binop_carrier(BinOp::Add, Dtype::F32),
                Dtype::F32,
                ident_expr(Dtype::F32),
                vec![alias_operand_of(x, &space)],
            );
            assert!(fire(&mut g, fold, &PROMOTE).is_none(), "axis {axis}");
        }
    }

    /// A narrow accumulator under the value's contract declines on
    /// `min_accum_bits` — the one numeric guard this law carries. There is
    /// deliberately **no** `reassoc` guard: see the STRICT case below.
    #[test]
    fn promote_declines_a_narrow_accumulator() {
        let mut g = ts::graph();
        let space = dims(&[4, 8, 16]);
        let x = ts::buffer(&mut g, Dtype::F16, &space);
        let fold = ts::kfold(
            &mut g,
            &space,
            2,
            ts::binop_carrier(BinOp::Add, Dtype::F16),
            Dtype::F16,
            ident_expr(Dtype::F16),
            vec![alias_operand_of(x, &space)],
        );
        assert_eq!(g.facts(fold).numeric.min_accum_bits, 32);
        assert!(fire(&mut g, fold, &PROMOTE).is_none());
    }

    /// **No `reassoc` permission is required, and requiring one would be a
    /// bug.** A free axis carries no dependence, so the rewrite is exact; it
    /// fires on a value whose contract is `STRICT` — the QAT fake-quant path
    /// where every inexact law declines.
    #[test]
    fn promote_fires_under_a_strict_numeric_contract() {
        let mut g = ts::graph();
        let space = dims(&[4, 8, 16]);
        let x = ts::buffer(&mut g, Dtype::F32, &space);
        // A rounding body is what gives a value `NumericContract::STRICT`.
        let quant = ts::map(
            &mut g,
            ScalarExpr::round(
                crate::dtype::RoundMode::HalfAwayFromZero,
                ScalarExpr::arg(0, Dtype::F32),
            ),
            &[x],
        );
        let fold = ts::kfold(
            &mut g,
            &space,
            2,
            ts::binop_carrier(BinOp::Add, Dtype::F32),
            Dtype::F32,
            ident_expr(Dtype::F32),
            vec![alias_operand_of(quant, &space)],
        );
        assert!(!g.facts(fold).numeric.reassoc, "the fixture is not STRICT");
        assert!(fire(&mut g, fold, &PROMOTE).is_some());
    }

    /// The inverse is minted, so full and no promotion both stay live in one
    /// class and compete on cost.
    #[test]
    fn promote_flatten_is_the_inverse() {
        let mut g = ts::graph();
        let space = dims(&[4, 8, 16]);
        let x = ts::buffer(&mut g, Dtype::F32, &space);
        let promoted = kfold_n(
            &mut g,
            &space,
            2,
            &[1],
            ts::binop_carrier(BinOp::Add, Dtype::F32)
                .promote(Dim::Const(8))
                .unwrap(),
            Dtype::F32,
            &[ident_expr(Dtype::F32)],
            vec![alias_operand_of(x, &space)],
        );
        assert!(fire(&mut g, promoted, &PROMOTE_FLATTEN).is_some());
        let flat = g
            .chain(promoted)
            .into_iter()
            .filter_map(|m| kfold_of(&g, m).map(|k| (m, k)))
            .find(|(_, k)| matches!(k, Launch::Fold { vec_axes, .. } if vec_axes.is_empty()))
            .expect("a flattened alternative");
        assert_eq!(g.facts(flat.0), g.facts(promoted));
        let Launch::Fold { carrier, .. } = &flat.1 else {
            panic!()
        };
        assert_eq!(carrier.slots[..], [SlotTy::Scalar]);
        // And promoting it again lands back on the same node: the two rules
        // are inverse, so the class does not grow without bound.
        assert!(fire(&mut g, flat.0, &PROMOTE).is_some());
        assert!(g.chain(promoted).contains(&promoted));
    }

    /// A host evaluator that also resolves `IndexOf`, so the tests do not
    /// borrow the carrier module's evaluator for the answer they are checking.
    fn evl(e: &ScalarExpr, args: &[f32], idx: &[u64]) -> f32 {
        use ScalarKind as K;
        match e.kind() {
            K::Arg(i) => args[*i as usize],
            K::IndexOf(a) => idx[*a as usize] as f32,
            K::Lit(l) => match l.0 {
                Splat::F32(v) => v,
                Splat::F16(b) => half::f16::from_bits(b).to_f32(),
                Splat::BF16(b) => half::bf16::from_bits(b).to_f32(),
                Splat::U32(v) => v as f32,
                Splat::I32(v) => v as f32,
            },
            K::Un { op, x } => {
                let v = evl(x, args, idx);
                match op {
                    UnOp::Exp => v.exp(),
                    UnOp::Neg => -v,
                    UnOp::Abs => v.abs(),
                    other => panic!("evaluator does not cover {other:?}"),
                }
            }
            K::Bin { op, a, b } => {
                let (x, y) = (evl(a, args, idx), evl(b, args, idx));
                match op {
                    BinOp::Add => x + y,
                    BinOp::Sub => x - y,
                    BinOp::Mul => x * y,
                    BinOp::Div => x / y,
                    BinOp::Max => x.max(y),
                    BinOp::Min => x.min(y),
                    other => panic!("evaluator does not cover {other:?}"),
                }
            }
            K::Cmp { op, a, b } => {
                let (x, y) = (evl(a, args, idx), evl(b, args, idx));
                let t = match op {
                    CmpOp::Ge => x >= y,
                    CmpOp::Gt => x > y,
                    CmpOp::Le => x <= y,
                    CmpOp::Lt => x < y,
                    CmpOp::Eq => x == y,
                    CmpOp::Ne => x != y,
                };
                if t { 1.0 } else { 0.0 }
            }
            K::Select { c, t, f } => {
                if evl(c, args, idx) != 0.0 {
                    evl(t, args, idx)
                } else {
                    evl(f, args, idx)
                }
            }
            K::Cast { x, .. } => evl(x, args, idx),
            other => panic!("evaluator does not cover {other:?}"),
        }
    }

    /// One accumulator's worth of state, rounded to `acc` after every merge.
    fn absorb_into(
        c: &Carrier,
        acc: &mut Vec<f32>,
        args: &[f32],
        idx: &[u64],
        round: fn(f32) -> f32,
    ) {
        let lift: Vec<f32> = c.lift.iter().map(|l| evl(l, args, idx)).collect();
        let both: Vec<f32> = acc.iter().copied().chain(lift).collect();
        *acc = c.merge.iter().map(|m| round(evl(m, &both, idx))).collect();
    }

    fn f16r(v: f32) -> f32 {
        half::f16::from_f32(v).to_f32()
    }
    fn f32r(v: f32) -> f32 {
        v
    }

    /// The **unpromoted** loop nest: the promoted-to-be axis is an ordinary
    /// iteration axis and the reduction is the inner loop, one accumulator
    /// alive at a time. `coord` supplies the *iteration* coordinates
    /// `IndexOf` resolves against — here `[.., p, k]`, because the axis is
    /// still iterated.
    fn run_iterated(
        c: &Carrier,
        (nf, ne, nk): (u64, u64, u64),
        x: &dyn Fn(u64, u64, u64) -> f32,
        coord: &dyn Fn(u64, u64, u64) -> Vec<u64>,
        round: fn(f32) -> f32,
    ) -> Vec<Vec<f32>> {
        let mut out = Vec::new();
        for fq in 0..nf {
            for p in 0..ne {
                let mut acc: Vec<f32> = c.identity_f32();
                for k in 0..nk {
                    absorb_into(c, &mut acc, &[x(fq, p, k)], &coord(fq, p, k), round);
                }
                out.push(acc);
            }
        }
        out
    }

    /// The **promoted** loop nest: the reduction is outermost and the promoted
    /// axis is the inner loop over `ne` live accumulators. `coord` no longer
    /// contains the promoted axis, which is exactly the renumbering the rule
    /// performs — executed, so a wrong shift shows up as a wrong number.
    fn run_promoted(
        c: &Carrier,
        (nf, ne, nk): (u64, u64, u64),
        x: &dyn Fn(u64, u64, u64) -> f32,
        coord: &dyn Fn(u64, u64, u64) -> Vec<u64>,
        round: fn(f32) -> f32,
    ) -> Vec<Vec<f32>> {
        let mut out = Vec::new();
        for fq in 0..nf {
            let mut accs: Vec<Vec<f32>> = (0..ne).map(|_| c.identity_f32()).collect();
            for k in 0..nk {
                for p in 0..ne {
                    absorb_into(
                        c,
                        &mut accs[p as usize],
                        &[x(fq, p, k)],
                        &coord(fq, p, k),
                        round,
                    );
                }
            }
            out.extend(accs);
        }
        out
    }

    /// **The law is exact.** Promotion changes the loop nest, not the
    /// arithmetic: each accumulator absorbs exactly the elements it absorbed
    /// before, in exactly the same order. So the two forms agree *bit for bit*
    /// even when every merge is rounded to f16, where float addition is
    /// emphatically not associative — and the negative control shows the test
    /// has teeth.
    #[test]
    fn promoted_and_unpromoted_agree_bit_for_bit_in_f16() {
        let shape = (3u64, 5u64, 64u64);
        // A few big terms among many small ones: at 2048 an f16 ulp is 2.0, so
        // every 0.75 added *after* a big term rounds away and every one added
        // before it survives. The order of the reduction is therefore visible
        // in the bits.
        let x = |f: u64, p: u64, k: u64| {
            if k.is_multiple_of(8) {
                2048.0 + (f as f32) * 2.0 + (p as f32) * 2.0
            } else {
                0.75
            }
        };
        let c = Carrier::binop(BinOp::Add, Splat::F32(0.0), Dtype::F32);
        let promoted = c.promote(Dim::Const(5)).unwrap();
        // Promotion touches `slots` alone; the arithmetic is the same terms.
        assert_eq!(c.lift, promoted.lift);
        assert_eq!(c.merge, promoted.merge);

        let flat = |_: u64, _: u64, _: u64| Vec::new();
        let a = run_iterated(&c, shape, &x, &flat, f16r);
        let b = run_promoted(&promoted, shape, &x, &flat, f16r);
        assert_eq!(a.len(), b.len());
        for (i, (l, r)) in a.iter().zip(&b).enumerate() {
            assert_eq!(
                l[0].to_bits(),
                r[0].to_bits(),
                "row {i}: {} vs {}",
                l[0],
                r[0]
            );
        }

        // Teeth: reversing the reduction order does move the f16 bits, so
        // "bit-identical" above is a statement and not a tautology.
        let reversed = run_iterated(&c, shape, &|f, p, k| x(f, p, 63 - k), &flat, f16r);
        assert!(
            reversed
                .iter()
                .zip(&a)
                .any(|(l, r)| l[0].to_bits() != r[0].to_bits()),
            "the f16 fixture is insensitive to order; the test proves nothing"
        );

        // The same at f32, against an independently computed expectation.
        let want: f32 = (0..64).map(|k| x(1, 2, k)).sum();
        let got = run_promoted(&promoted, shape, &x, &flat, f32r)[5 + 2][0];
        assert!((got - want).abs() <= 1e-3 * want.abs(), "{got} vs {want}");
    }

    /// **A `(value, index)` max-pool carrier promoted over a channel block.**
    ///
    /// Nobody aimed a rule at this. It is a two-slot carrier whose index slot's
    /// lift reads `IndexOf` of the *reduced* axis, so it exercises the
    /// renumbering that a single-slot sum cannot: promoting the channel axis
    /// moves the window coordinate from iteration index 1 to index 0, and a
    /// rule that forgot to renumber would have the index slot silently report
    /// the channel instead of the window position. The carrier's own lanes are
    /// scalars, so the promoted node is read back through a *transposed* view
    /// rather than a reshape.
    #[test]
    fn promote_fires_on_a_value_index_maxpool_carrier() {
        let (c, k) = (3u64, 4u64);
        let neg_inf = Splat::F32(f32::NEG_INFINITY);
        let arg = |i| ScalarExpr::arg(i, Dtype::F32);
        let carrier = Carrier {
            slots: smallvec![SlotTy::Scalar, SlotTy::Scalar],
            identity: smallvec![neg_inf, Splat::F32(0.0)],
            lift: smallvec![
                arg(0),
                ScalarExpr::cast(Dtype::F32, ScalarExpr::index_of(1)),
            ],
            merge: smallvec![
                ScalarExpr::bin(BinOp::Max, arg(0), arg(2)),
                ScalarExpr::select(ScalarExpr::cmp(CmpOp::Ge, arg(0), arg(2)), arg(1), arg(3)),
            ],
            associative: true,
            tie: None,
        };
        assert!(carrier.identity_closed(crate::carrier::probes_for(Dtype::F32)));

        let mut g = ts::graph();
        let space = dims(&[c, k]);
        let x = ts::buffer(&mut g, Dtype::F32, &space);
        let fold = kfold_n(
            &mut g,
            &space,
            1,
            &[],
            carrier.clone(),
            Dtype::F32,
            &[ident_expr(Dtype::F32), ident_expr(Dtype::F32)],
            vec![alias_operand_of(x, &space)],
        );
        let want = g.facts(fold).clone();
        assert_eq!(&want.shape[..], &dims(&[c, 2])[..]);

        assert!(fire(&mut g, fold, &PROMOTE).is_some());
        class_facts_agree(&g, fold);
        let (pid, promoted) = promoted_at(&g, fold, 1).unwrap();
        let Launch::Fold {
            carrier: pc,
            vec_axes,
            ..
        } = &promoted
        else {
            panic!()
        };
        assert_eq!(&vec_axes[..], &[0]);
        assert_eq!(
            pc.slots[..],
            [SlotTy::Vector(Dim::Const(c)), SlotTy::Vector(Dim::Const(c))]
        );
        // The renumbering: the window coordinate was iteration axis 1 and is
        // now axis 0, because the channel axis left the iteration domain.
        assert!(pc.reads_index_of(0));
        assert!(!pc.reads_index_of(1));
        assert_eq!(&promoted.iter_space().dims[..], &dims(&[k])[..]);
        // Slot-major lanes, read back at the original `[C, 2]` shape by a
        // transposed view.
        assert_eq!(&g.facts(pid).shape[..], &dims(&[2 * c])[..]);
        let view = g
            .chain(fold)
            .into_iter()
            .find(|&m| view_source(&g, m) == Some(pid))
            .expect("a recovery view");
        assert_eq!(g.facts(view), &want);

        // And it computes the right thing. The window index is deliberately
        // not monotone in the channel, so a mis-renumbered `IndexOf` produces
        // a different answer rather than the same one by luck.
        let data = |ch: u64, w: u64| ((ch * 7 + w * 3) % 5) as f32 + (w as f32) * 0.5;
        // The promoted node iterates `[k]` alone, so `IndexOf(0)` is the
        // window coordinate. Feeding the *unpromoted* coordinates here would
        // read the channel instead, which is the mistake the renumbering
        // exists to prevent.
        let got = run_promoted(
            pc,
            (1, c, k),
            &|_, p, w| data(p, w),
            &|_, _, w| vec![w],
            f32r,
        );
        for ch in 0..c {
            let (mut best, mut arg_best) = (f32::NEG_INFINITY, 0.0f32);
            for w in 0..k {
                if data(ch, w) > best {
                    best = data(ch, w);
                    arg_best = w as f32;
                }
            }
            assert_eq!(got[ch as usize][0], best, "channel {ch} value");
            assert_eq!(got[ch as usize][1], arg_best, "channel {ch} index");
        }
    }

    /// Everything this law mints goes through the shipped `verify_launch`: the
    /// promoted-axis clauses (contiguity, `lanes == positions * width`), the
    /// write map, the operand access predicates and the fold-axis rank check.
    #[test]
    fn the_minted_nests_pass_verify_l1() {
        // (a) the contraction nest, arrived at through the lowering floor.
        let (b, h, lq, lk, dh) = (2u64, 3, 5, 7, 16);
        let mut g = ts::graph();
        let p = ts::buffer(&mut g, Dtype::F32, &dims(&[b, h, lq, lk]));
        let v = ts::buffer(&mut g, Dtype::F32, &dims(&[b, h, lk, dh]));
        let spec = EinSpec {
            a: smallvec![Label(0), Label(1), Label(2), Label(3)],
            b: smallvec![Label(0), Label(1), Label(3), Label(4)],
            out: smallvec![Label(0), Label(1), Label(2), Label(4)],
        };
        let out = ts::contract(&mut g, spec, Dtype::F32, p, v);
        let rules: Vec<Rule> = CORE_RULES.to_vec();
        CoreSaturate
            .saturate(&mut g, &ts::caps(), &rules, SaturationBudget::default())
            .unwrap();
        for (pid, _) in promoted_members(&g, out) {
            verify(&g, pid).expect("the promoted contraction nest must verify");
        }

        // (b) the register tile.
        let mut g = ts::graph();
        let space = dims(&[2, 3, 4, 8, 16]);
        let a = ts::buffer(&mut g, Dtype::F32, &space);
        let bb = ts::buffer(&mut g, Dtype::F32, &space);
        let fold = kfold_n(
            &mut g,
            &space,
            4,
            &[],
            dot_carrier(),
            Dtype::F32,
            &[ident_expr(Dtype::F32)],
            vec![alias_operand_of(a, &space), alias_operand_of(bb, &space)],
        );
        fire(&mut g, fold, &PROMOTE).unwrap();
        let (first, _) = promoted_at(&g, fold, 1).unwrap();
        verify(&g, first).expect("TN must verify");

        // (c) a multi-slot carrier, whose promotion *does* mint a recovery
        //     view. The view is an ordinary aliasing nest and verifies as one:
        //     its `AccessPlan::Alias` predicate is the clause that would catch
        //     a stride vector the promoted lane order does not justify.
        let mut g = ts::graph();
        let space = dims(&[6, 5]);
        let x = ts::buffer(&mut g, Dtype::F32, &space);
        let sum = Carrier::binop(BinOp::Add, Splat::F32(0.0), Dtype::F32);
        let pair = sum.tuple(
            &Carrier::binop(BinOp::Max, Splat::F32(f32::NEG_INFINITY), Dtype::F32),
            &crate::carrier::ArgRemap::identity(1),
        );
        let fold = kfold_n(
            &mut g,
            &space,
            1,
            &[],
            pair.carrier,
            Dtype::F32,
            &[ident_expr(Dtype::F32), ident_expr(Dtype::F32)],
            vec![alias_operand_of(x, &space)],
        );
        fire(&mut g, fold, &PROMOTE).unwrap();
        let (pid, _) = promoted_at(&g, fold, 1).unwrap();
        verify(&g, pid).expect("the promoted two-slot nest must verify");
        let views: Vec<Id> = g
            .chain(fold)
            .into_iter()
            .filter(|&m| view_source(&g, m) == Some(pid))
            .collect();
        assert_eq!(views.len(), 1, "a two-slot promotion needs one view");
        verify(&g, views[0]).expect("the readback view must verify");
    }

    /// Pins `semantics::work`'s residual over-count on a promoted nest: the
    /// `e * lanes` term charges each output element `lanes` times, because a
    /// multi-slot carrier's output shape already counts its lanes.
    ///
    /// Delete this test with the row.
    #[test]
    fn work_prices_a_promoted_nest_lanes_times_too_high() {
        let d = 8u64;
        let mut g = ts::graph();
        let space = dims(&[4, d, 16]);
        let x = ts::buffer(&mut g, Dtype::F32, &space);
        let fold = ts::kfold(
            &mut g,
            &space,
            2,
            ts::binop_carrier(BinOp::Add, Dtype::F32),
            Dtype::F32,
            ident_expr(Dtype::F32),
            vec![alias_operand_of(x, &space)],
        );
        assert!(fire(&mut g, fold, &PROMOTE).is_some());
        let (pid, _) = promoted_at(&g, fold, 1).unwrap();

        let work_of = |id: Id| {
            let node = g.node(id).clone();
            let Op::Launch(op) = &node.op else { panic!() };
            let ins: Vec<crate::facts::ValueFacts> =
                node.children.iter().map(|c| g.facts(*c).clone()).collect();
            crate::semantics::work::work_l1(op, &ins, g.facts(id))
        };
        let (plain, promoted) = (work_of(fold), work_of(pid));

        // macs = ein * (lanes + lift) + e * (lanes + post). `ein` filters
        // `vec_axes`, so `ein * lanes` is 512 on both sides; the residual is
        // the `e * lanes` term, `e * (lanes - 1)` = 32 * 7 here, which no
        // longer scales the whole nest.
        assert_eq!(plain.macs, 512 + 32, "plain: ein*1 + e*1");
        assert_eq!(promoted.macs, 512 + 32 * d, "promoted: ein*lanes + e*lanes");
        assert!(
            promoted.macs < plain.macs * d,
            "the {d}x over-count is gone; only the e*lanes term remains"
        );
    }

    /// **A promoted carrier MAY read the reduced axis's coordinate.**
    ///
    /// Every `ScalarExpr` on a `Fold` is written against `iter_space()`, so
    /// the legal indices are `0..iter_rank` and a promoted axis is simply not
    /// nameable. After one promotion the reduced axis's iteration index
    /// equals the promoted axis's space index, so a clause phrased in space
    /// indices would reject exactly the nests whose lift reads the reduction
    /// coordinate — a max-pool's index slot, a causal
    /// `select(IndexOf(lk) <= ..)`. The promoted nest verifies; only an
    /// expression naming a coordinate outside the iteration domain is refused.
    #[test]
    fn a_promoted_carrier_may_read_the_reduced_axis() {
        let (c, k) = (3u64, 4u64);
        let arg = |i| ScalarExpr::arg(i, Dtype::F32);
        let carrier = Carrier {
            slots: smallvec![SlotTy::Scalar, SlotTy::Scalar],
            identity: smallvec![Splat::F32(f32::NEG_INFINITY), Splat::F32(0.0)],
            lift: smallvec![
                arg(0),
                ScalarExpr::cast(Dtype::F32, ScalarExpr::index_of(1)),
            ],
            merge: smallvec![
                ScalarExpr::bin(BinOp::Max, arg(0), arg(2)),
                ScalarExpr::select(ScalarExpr::cmp(CmpOp::Ge, arg(0), arg(2)), arg(1), arg(3)),
            ],
            associative: true,
            tie: None,
        };
        let mut g = ts::graph();
        let space = dims(&[c, k]);
        let x = ts::buffer(&mut g, Dtype::F32, &space);
        let fold = kfold_n(
            &mut g,
            &space,
            1,
            &[],
            carrier,
            Dtype::F32,
            &[ident_expr(Dtype::F32), ident_expr(Dtype::F32)],
            vec![alias_operand_of(x, &space)],
        );
        verify(&g, fold).expect("the unpromoted nest verifies");
        fire(&mut g, fold, &PROMOTE).unwrap();
        let (pid, _) = promoted_at(&g, fold, 1).unwrap();
        verify(&g, pid).expect(
            "a promoted nest whose lift reads the REDUCED axis's iteration \
             coordinate is legal; only a coordinate outside the iteration \
             domain is not",
        );
    }
}
