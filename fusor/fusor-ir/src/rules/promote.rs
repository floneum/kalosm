//! **PROMOTE**: a free axis `d` of a reduction nest moves from the iteration domain into
//! the accumulator (`Scalar -> Vector(D_d)`). Value-preserving since `d` is free; `space`
//! and operand maps are untouched, and the node's expressions renumber `IndexOf(j > d)`.

use crate::carrier::Carrier;
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

/// One promotion's worth of node state; `space` and `ops` stay fixed.
#[derive(Clone)]
struct Promoted {
    vec_axes: SmallVec<[u32; 2]>,
    carrier: Carrier,
    post: SmallVec<[ScalarExpr; 4]>,
}

/// Move the innermost free axis (`axis - 1`) into the accumulator; both forms stay live.
pub fn promote(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::Launch(
        op @ Launch::Fold {
            space,
            axis,
            vec_axes,
            carrier,
            acc,
            post,
            sched,
            ..
        },
    ) = &node.op
    else {
        return None;
    };
    // One axis per firing, only from a nest with nothing promoted yet: every later
    // promotion needs a recovery reshape neither backend lowers today.
    if !vec_axes.is_empty() {
        return None;
    }
    equal_slot_lanes(carrier)?;
    let want = f.own().shape.clone();

    let state = Promoted {
        vec_axes: vec_axes.clone(),
        carrier: carrier.clone(),
        post: post.clone(),
    };
    let next = promote_once(space, *axis as usize, *acc, &state, f)?;
    let got = space.fold_shape(*axis, &next.vec_axes, &next.carrier)?;
    // A carrier that already had a slot axis absorbs the promoted axis; a pure alias
    // puts it back. Decided before minting.
    let view = recovery_view(carrier, &got, &want)?;
    let new_sched = sched.with_fold_carrier(next.carrier.lanes()?, acc.byte_size(), f.caps())?;
    let mut fold = op.clone();
    if let Launch::Fold {
        vec_axes,
        carrier,
        post,
        sched,
        ..
    } = &mut fold
    {
        (*vec_axes, *carrier, *post, *sched) = (next.vec_axes, next.carrier, next.post, new_sched);
    }
    let fold = b.add_launch(fold).ok()?;
    let value = apply_view(b, fold, &view)?;

    // The facts are unchanged by this law; a botched renumbering shows as a shape
    // mismatch here.
    if b.facts_of(value).shape != want {
        return None;
    }
    b.union(id, value).ok()
}

/// One step: promote the innermost remaining free axis, or `None`.
fn promote_once(
    space: &IndexSpace,
    axis: usize,
    acc: crate::dtype::Dtype,
    state: &Promoted,
    f: &Facts<'_>,
) -> Option<Promoted> {
    let d = promotable_axis(space, axis, &state.vec_axes)?;

    // 1. Positionwise in `d`: no expression reads `IndexOf(d)` and every merge `Arg`
    //    is a slot reference. Promoted axes sit above `d`, so `d` is its own index.
    if !positionwise_in(&state.carrier, &state.post, d as u32) {
        return None;
    }

    // 2. A constant extent: a symbolic private array is unallocatable.
    let extent = *space.dims.get(d)?;
    let e = extent.as_const()?;
    // A unit axis would be the same register under another name.
    if e <= 1 {
        return None;
    }

    // 3. `acc` is wide enough; `own().numeric` is the meet over every operand.
    if acc.accum_bits() < f.own().numeric.min_accum_bits {
        return None;
    }

    // 4. Equal lane counts make slot readback a single strided view.
    equal_slot_lanes(&state.carrier)?;

    // 5. The promoted accumulator lives in registers; over budget declines.
    let promoted = state.carrier.promote(extent)?;
    let lanes = promoted.lanes()?;
    if lanes.checked_mul(acc.byte_size())? > crate::rules::private_acc_bytes(f.caps(), true) {
        return None;
    }

    // The rebinding: only the iterated/accumulated partition point moves.
    let mut vec_axes: SmallVec<[u32; 2]> = SmallVec::with_capacity(state.vec_axes.len() + 1);
    vec_axes.push(d as u32);
    vec_axes.extend(state.vec_axes.iter().copied());

    let shift = |x: &ScalarExpr| drop_index_axis(x, d as u32);
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

/// The innermost free axis of a nest spelled `free.. ++ vec.. ++ [reduced]`, or `None`.
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

/// Whether no expression reads `IndexOf(axis)` and every merge `Arg` is a slot.
fn positionwise_in(carrier: &Carrier, post: &[ScalarExpr], axis: u32) -> bool {
    if carrier.reads_index_of(axis) || post.iter().any(|e| e.reads_axis(axis)) {
        return false;
    }
    let w = carrier.width() as u32;
    carrier
        .merge
        .iter()
        .all(|m| max_arg(m).is_none_or(|a| a < 2 * w))
}

/// Every slot's common lane count (`verify_launch`'s `lanes == positions * width`).
fn equal_slot_lanes(carrier: &Carrier) -> Option<u64> {
    let first = carrier.slots.first()?.lanes()?;
    carrier
        .slots
        .iter()
        .all(|s| s.lanes() == Some(first))
        .then_some(first)
}

/// How the promoted value reads back at the original shape: `(extent, multiplier)` per
/// axis the trailing carrier axis splits into; empty when the shapes agree.
type Recovery = SmallVec<[(u64, u32); 4]>;

/// The read-back of the promoted node at the pre-promotion shape. Lane `s*e*q + p*q + j`
/// is a strided view only for one slot (`w == 1`, a reshape) or scalar slots (`q == 1`,
/// a transpose); several vector slots would need a divmod, so decline.
fn recovery_view(base: &Carrier, got: &[Dim], want: &[Dim]) -> Option<Recovery> {
    if got == want {
        return Some(Recovery::new());
    }
    let q = equal_slot_lanes(base)?;
    let w = base.width();
    // Whether the original carrier appended an axis (`Vector(1)` does, `Scalar` not).
    let carried = usize::from(base.out_dim()?.is_some());
    let last = got.len().checked_sub(1)?;
    let slot_axis = q.checked_mul(w as u64)?;
    // The promoted extent: what the flattened carrier axis holds beyond the original.
    let e = got[last].as_const()? / slot_axis;
    if e.checked_mul(slot_axis)? != got[last].as_const()? {
        return None;
    }
    // `want` is `got` with that axis put back, then the original carrier axis.
    if want.len() != last + 1 + carried
        || got[..last] != want[..last]
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

/// Mint the recovery view as the Launch alias `LOWER_RESTRIDE` would mint, skipping the
/// lowering cascade.
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

/// Renumber `IndexOf(j)` to `IndexOf(j - 1)` for every `j > from`.
fn drop_index_axis(e: &ScalarExpr, from: u32) -> ScalarExpr {
    e.rewrite(&mut |e| match e.kind() {
        ScalarKind::IndexOf(a) if *a > from => Some(ScalarExpr::index_of(a - 1)),
        ScalarKind::Dot { .. } | ScalarKind::Splat { .. } => Some(e.clone()),
        _ => None,
    })
}

/// The largest `Arg` index an expression reads.
fn max_arg(e: &ScalarExpr) -> Option<u32> {
    let mut max = None;
    e.walk(&mut |e| {
        if let ScalarKind::Arg(i) = e.kind() {
            max = max.max(Some(*i));
        }
    });
    max
}
