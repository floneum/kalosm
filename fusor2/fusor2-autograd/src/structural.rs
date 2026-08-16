//! The five structural adjoints, each read off the primal op's own attributes.
//!
//! [`window_adjoint`] is solved by two integers: from `(window, step)`,
//! `step >= window` proves the adjoint is an elementwise mask-and-broadcast;
//! overlapping windows give `Scatter{Add}`.

use crate::tape::{TapeExt, const_numel, const_row_major};
use fusor2_ir::autograd::{Grads, Tape, Val};
use fusor2_ir::dtype::Dtype;
use fusor2_ir::ir::Node;
use fusor2_ir::ir::Op;
use fusor2_ir::carrier::Carrier;
use fusor2_ir::ir::logical::{Logical, ScatterCombine, TiePolicy};
use fusor2_ir::ir::launch::WindowAdjoint;
use fusor2_ir::scalar::{BinOp, CmpOp, ScalarExpr};
use fusor2_ir::shape::{Dim, Dims, Layout, SlidingWindow, StrideSpec, reshape_specs};
use fusor2_ir::{Error, Result};
use smallvec::SmallVec;

/// Dispatch an [`fusor2_ir::autograd::AdjointKind::Structural`] row.
pub(crate) fn structural_adjoint(
    tape: &mut dyn Tape,
    node: &Node,
    grad: Val,
    ins: &[Val],
    out: Val,
) -> Result<Grads> {
    match &node.op {
        Op::Logical(Logical::Restride { .. }) => restride_adjoint(tape, node, grad, ins, out),
        Op::Logical(Logical::Window { .. }) => window_adjoint(tape, node, grad, ins, out),
        Op::Logical(Logical::Gather { .. }) => gather_adjoint(tape, node, grad, ins, out),
        Op::Logical(Logical::Scatter { .. }) => scatter_adjoint(tape, node, grad, ins, out),
        Op::Logical(Logical::Fold { .. }) => fold_adjoint(tape, node, grad, ins, out),
        other => Err(Error::Plan(format!(
            "no structural adjoint for {other:?}"
        ))),
    }
}
/// One input axis's contribution to the output index space.
#[derive(Clone, Debug, Default)]
struct RestrideRun {
    /// `(output position, multiplier, size)` of every non-broadcast spec
    /// referencing this input axis, in output order.
    parts: SmallVec<[(usize, u32, u64); 3]>,
}
/// Adjoint of `Restride`: sum over every stride-0 axis, then invert the
/// remaining injective index map.
///
/// Three stages, most-specific first:
/// 1. every `multiplier == 0` output axis becomes a `Fold{Add}` — this *is*
///    the broadcast backward;
/// 2. if the remaining specs form an invertible per-axis run set (a permute
///    and/or reshape), they invert into exactly one `Restride`;
/// 3. otherwise the map is non-injective or partial, and the adjoint is a
///    `Scatter{Add}` into a zero base with the index tensor computed by a
///    `Map` of `IndexOf(axis)` terms. Never a host loop.
pub(crate) fn restride_adjoint(
    tape: &mut dyn Tape,
    node: &Node,
    grad: Val,
    ins: &[Val],
    _out: Val,
) -> Result<Grads> {
    let Op::Logical(Logical::Restride { specs, .. }) = &node.op else {
        return Err(Error::Plan("restride_adjoint on a non-Restride node".into()));
    };
    let x = *ins
        .first()
        .ok_or_else(|| Error::Plan("Restride takes one operand".into()))?;
    let xshape = tape.shape_of(x);

    // Stage 1: fold every stride-0 axis away, outermost last so the axis
    // indices below it stay valid.
    let mut g = grad;
    for axis in (0..specs.len()).rev() {
        if specs[axis].is_broadcast() {
            g = tape.sum_axis(g, axis as u32)?;
        }
    }
    let kept: Vec<StrideSpec> = specs.iter().copied().filter(|s| !s.is_broadcast()).collect();

    // Stage 2: try to invert.
    if let Some(inverse) = invert_runs(&kept, &xshape) {
        let dx = tape.restride(&inverse, g)?;
        return Ok(smallvec::smallvec![Some(dx)]);
    }

    // Stage 2b: a reshape. A merge names only the group's innermost input
    // axis, so `invert_runs` declines it; a spec vector whose composed layout
    // is dense row-major over the whole input is the identity on flat
    // indices, and its adjoint is the reshape back.
    let view_shape: Dims = kept.iter().map(|s| s.size).collect();
    if is_dense_reshape(&kept, &xshape, &view_shape) {
        let inverse = reshape_specs(&view_shape, &xshape)?;
        let dx = tape.restride(&inverse, g)?;
        return Ok(smallvec::smallvec![Some(dx)]);
    }

    // Stage 3: the general index-tensor scatter.
    let idx_expr = restride_index_expr(&kept, &xshape)?;
    let dtype = tape.dtype_of(x);
    let dx = scatter_back(tape, g, idx_expr, &xshape, dtype)?;
    Ok(smallvec::smallvec![Some(dx)])
}

/// Whether `kept` reads every element of `xshape` exactly once in row-major
/// order — the composed layout is `Layout::contiguous(view_shape)`, so the
/// view's flat index *is* the source's flat index.
fn is_dense_reshape(kept: &[StrideSpec], xshape: &[Dim], view_shape: &[Dim]) -> bool {
    let (Some(have), Some(want)) = (const_numel(xshape), const_numel(view_shape)) else {
        return false;
    };
    have == want
        && composed_layout(kept, xshape).is_some_and(|l| l.is_contiguous())
}

/// The dense layout `kept` composes to over `xshape`, or `None` when a stride
/// is symbolic. Mirrors `fusor2_ir::rules::composed_layout`, which is private
/// to the rule set; the adjoint has to agree with it exactly.
fn composed_layout(kept: &[StrideSpec], xshape: &[Dim]) -> Option<Layout> {
    let strides = const_row_major(xshape)?;
    let mut shape: Vec<Dim> = Vec::with_capacity(kept.len());
    let mut out: Vec<Dim> = Vec::with_capacity(kept.len());
    let mut offset: u64 = 0;
    for s in kept {
        shape.push(s.size);
        if s.multiplier == 0 {
            out.push(Dim::Const(0));
            continue;
        }
        let base = *strides.get(s.input_dim as usize)?;
        out.push(Dim::Const(base.checked_mul(u64::from(s.multiplier))?));
        offset = offset.checked_add(s.offset.as_const()?.checked_mul(base)?)?;
    }
    Layout::from_parts(Dim::Const(offset), &shape, &out).ok()
}

/// Invert a broadcast-free spec list into a single `Restride`, or `None`
/// when the map is partial (a slice), offset, strided or interleaved.
fn invert_runs(kept: &[StrideSpec], xshape: &[Dim]) -> Option<SmallVec<[StrideSpec; 6]>> {
    let mut runs: Vec<RestrideRun> = vec![RestrideRun::default(); xshape.len()];
    for (pos, spec) in kept.iter().enumerate() {
        if spec.offset.as_const() != Some(0) {
            return None;
        }
        let size = spec.size.as_const()?;
        let d = spec.input_dim as usize;
        runs.get_mut(d)?.parts.push((pos, spec.multiplier, size));
    }

    let mut inverse: SmallVec<[StrideSpec; 6]> = SmallVec::with_capacity(xshape.len());
    for (d, run) in runs.iter().enumerate() {
        let extent = xshape[d];
        if run.parts.is_empty() {
            // An axis nothing reads must be degenerate; the inverse
            // re-inserts it as a size-1 axis.
            if extent.as_const() != Some(1) {
                return None;
            }
            inverse.push(StrideSpec::broadcast(Dim::Const(1)));
            continue;
        }
        // Positions must be consecutive and ascending, and multipliers must
        // form the row-major tiling of this axis.
        let first = run.parts[0].0;
        let mut expect_mult: u64 = run.parts.iter().map(|p| p.2).product();
        let mut covered: u64 = 1;
        for (k, (pos, mult, size)) in run.parts.iter().copied().enumerate() {
            if pos != first + k {
                return None;
            }
            expect_mult /= size;
            if mult as u64 != expect_mult {
                return None;
            }
            covered *= size;
        }
        if Some(covered) != extent.as_const() {
            return None;
        }
        let inner = run.parts.last()?.0 as u32;
        inverse.push(StrideSpec::dim_with(inner, extent, 1));
    }
    Some(inverse)
}

/// `sum_i IndexOf(i) * multiplier_i * rowmajor_stride(x)[input_dim_i]`
/// plus the constant offset, as a `u32` scalar body.
fn restride_index_expr(kept: &[StrideSpec], xshape: &[Dim]) -> Result<ScalarExpr> {
    let strides = const_row_major(xshape).ok_or_else(|| {
        Error::Shape("the general Restride adjoint needs decidable source extents".into())
    })?;
    let mut terms: Vec<ScalarExpr> = Vec::new();
    let mut constant: u64 = 0;
    for (pos, spec) in kept.iter().enumerate() {
        let stride = *strides
            .get(spec.input_dim as usize)
            .ok_or_else(|| Error::Shape("restride input_dim out of range".into()))?;
        let offset = spec
            .offset
            .as_const()
            .ok_or_else(|| Error::Shape("symbolic restride offset".into()))?;
        constant += stride * offset;
        let step = stride * spec.multiplier as u64;
        if step != 0 {
            terms.push(u32_mul(ScalarExpr::index_of(pos as u32), step));
        }
    }
    Ok(u32_sum(terms, constant))
}

/// Adjoint of `Window`.
///
/// [`WindowAdjoint::of`] reads `(window, step)` off each spec: when every
/// spec has `step == window` and tiles its axis exactly, the adjoint is a
/// pure view — permute the trailing window axis next to its position axis and
/// merge the two. Otherwise the windows overlap or leave a tail, and the
/// adjoint is the overlap-add `Scatter{Add}`.
pub(crate) fn window_adjoint(
    tape: &mut dyn Tape,
    node: &Node,
    grad: Val,
    ins: &[Val],
    _out: Val,
) -> Result<Grads> {
    let Op::Logical(Logical::Window { specs, .. }) = &node.op else {
        return Err(Error::Plan("window_adjoint on a non-Window node".into()));
    };
    let x = *ins
        .first()
        .ok_or_else(|| Error::Plan("Window takes one operand".into()))?;
    let xshape = tape.shape_of(x);
    let rank = xshape.len();

    if let Some(dx) = window_view_adjoint(tape, specs, &xshape, grad)? {
        return Ok(smallvec::smallvec![Some(dx)]);
    }

    // Overlap-add. `IndexOf` over the position axis and the trailing window
    // axis reconstruct the source coordinate; duplicates accumulate.
    let strides = const_row_major(&xshape).ok_or_else(|| {
        Error::Shape("the overlapping Window adjoint needs decidable source extents".into())
    })?;
    let mut windowed: Vec<Option<(usize, u32)>> = vec![None; rank];
    for (i, w) in specs.iter().enumerate() {
        windowed[w.axis as usize] = Some((rank + i, w.step));
    }
    let mut terms: Vec<ScalarExpr> = Vec::new();
    for (d, stride) in strides.iter().copied().enumerate() {
        match windowed[d] {
            Some((wax, step)) => {
                terms.push(u32_mul(ScalarExpr::index_of(d as u32), stride * step as u64));
                terms.push(u32_mul(ScalarExpr::index_of(wax as u32), stride));
            }
            None => terms.push(u32_mul(ScalarExpr::index_of(d as u32), stride)),
        }
    }
    let idx_expr = u32_sum(terms, 0);
    let dtype = tape.dtype_of(x);
    let dx = scatter_back(tape, grad, idx_expr, &xshape, dtype)?;
    Ok(smallvec::smallvec![Some(dx)])
}

/// The `is_mask` branch: a chain of pure views, or `None` when the geometry
/// does not tile exactly.
fn window_view_adjoint(
    tape: &mut dyn Tape,
    specs: &[SlidingWindow],
    xshape: &[Dim],
    grad: Val,
) -> Result<Option<Val>> {
    let rank = xshape.len();
    for w in specs {
        let adj = WindowAdjoint::of(*w);
        if !adj.is_mask || w.step != w.window {
            return Ok(None);
        }
        let Some(extent) = xshape.get(w.axis as usize).and_then(|d| d.as_const()) else {
            return Ok(None);
        };
        if extent % w.window as u64 != 0 {
            // A tail the windows never reach would need a zero-fill, which
            // is a scatter. Fall through to the general path.
            return Ok(None);
        }
    }

    let mut cur = grad;
    for i in (0..specs.len()).rev() {
        let w = specs[i];
        let a = w.axis as usize;
        let shape = tape.shape_of(cur);
        let last = shape.len() - 1;
        debug_assert_eq!(last, rank + i);

        // Move the trailing window axis next to its position axis.
        let mut perm: Vec<u32> = Vec::with_capacity(shape.len());
        perm.extend((0..=a).map(|j| j as u32));
        perm.push(last as u32);
        perm.extend((a + 1..last).map(|j| j as u32));
        let permuted = tape.permute(cur, &perm)?;

        // Merge the (position, window) pair back into the source axis.
        let pshape = tape.shape_of(permuted);
        let merged = Dim::Const(w.window as u64 * dim_const(pshape[a])?);
        let mut merge: SmallVec<[StrideSpec; 6]> = SmallVec::with_capacity(pshape.len() - 1);
        for j in 0..a {
            merge.push(StrideSpec::dim(j as u32, pshape[j]));
        }
        merge.push(StrideSpec::dim_with(a as u32 + 1, merged, 1));
        for j in a + 2..pshape.len() {
            merge.push(StrideSpec::dim(j as u32, pshape[j]));
        }
        cur = tape.restride(&merge, permuted)?;
    }
    Ok(Some(cur))
}

/// Adjoint of `Gather` is `Scatter{Add}`; the index operand gets no gradient.
/// Duplicates accumulate, so an embedding table receiving one token twice
/// gets the summed gradient.
pub(crate) fn gather_adjoint(
    tape: &mut dyn Tape,
    node: &Node,
    grad: Val,
    ins: &[Val],
    _out: Val,
) -> Result<Grads> {
    let Op::Logical(Logical::Gather { axis, .. }) = &node.op else {
        return Err(Error::Plan("gather_adjoint on a non-Gather node".into()));
    };
    let (x, idx) = match ins {
        [x, idx] => (*x, *idx),
        other => {
            return Err(Error::Plan(format!(
                "Gather takes two operands, got {}",
                other.len()
            )));
        }
    };
    let base = tape.zeros_like(x)?;
    let dx = tape.scatter_add(*axis, base, idx, grad)?;
    Ok(smallvec::smallvec![Some(dx), None])
}

/// Adjoint of `Scatter`: `Gather` for the update operand, masked
/// pass-through for the base.
///
/// This covers `cat`, `stack`, `pad_axis`, `repeat`, `resize` and
/// `slice_assign`, all of which are `Scatter{Set}` into a const leaf, so
/// each input receives the slice of the gradient covering its range.
pub(crate) fn scatter_adjoint(
    tape: &mut dyn Tape,
    node: &Node,
    grad: Val,
    ins: &[Val],
    _out: Val,
) -> Result<Grads> {
    let Op::Logical(Logical::Scatter {
        axis,
        combine,
        unique,
        ..
    }) = &node.op
    else {
        return Err(Error::Plan("scatter_adjoint on a non-Scatter node".into()));
    };
    let (_base, idx, upd) = match ins {
        [b, i, u] => (*b, *i, *u),
        other => {
            return Err(Error::Plan(format!(
                "Scatter takes three operands, got {}",
                other.len()
            )));
        }
    };
    let d_upd = tape.gather(*axis, grad, idx)?;
    let d_base = match combine {
        // The written region no longer depends on `base`.
        ScatterCombine::Set => {
            let holes = tape.zeros_like(upd)?;
            tape.scatter_set(*axis, grad, idx, holes, *unique)?
        }
        // `base` passes straight through an accumulating scatter.
        ScatterCombine::Add => grad,
    };
    Ok(smallvec::smallvec![Some(d_base), None, Some(d_upd)])
}

/// Adjoint of `Fold`, read off `combine`: `Add` broadcasts, `Mul` is the
/// zero-aware product rule, `Max`/`Min` use the op's declared `TiePolicy`
/// rather than an implicit even split.
///
/// `mean` is `Fold{Add}` followed by `mul_scalar(1/n)`, so its adjoint is
/// this divided by the axis size with no separate rule.
pub(crate) fn fold_adjoint(
    tape: &mut dyn Tape,
    node: &Node,
    grad: Val,
    ins: &[Val],
    out: Val,
) -> Result<Grads> {
    let Op::Logical(Logical::Fold {
        carrier, axis, acc, ..
    }) = &node.op
    else {
        return Err(Error::Plan("fold_adjoint on a non-Fold node".into()));
    };
    let x = *ins
        .first()
        .ok_or_else(|| Error::Plan("Fold takes one operand".into()))?;
    let axis = *axis;
    let xshape = tape.shape_of(x);
    let extent = *xshape
        .get(axis as usize)
        .ok_or_else(|| Error::Shape(format!("fold axis {axis} out of range")))?;
    let dtype = tape.dtype_of(x);

    // The adjoint reads the carrier's own merge, not a variant name. A
    // carrier that is not one of Add/Mul/Max/Min has no analytic adjoint
    // here and says so.
    if ins.len() != 1 {
        return Err(Error::Numeric(
            "a multi-operand fold's adjoint is the composition of its lift's              adjoint with this one; autograd runs before fusion mints one"
                .into(),
        ));
    }
    if carrier.lift[0].kind() != &fusor2_ir::scalar::ScalarKind::Arg(0) {
        return Err(Error::Numeric(
            "a fold whose lift computes is minted by fusion, after autograd".into(),
        ));
    }
    let tie = carrier.tie.unwrap_or(TiePolicy::SplitEvenly);
    let dx = match carrier.kind() {
        Some(BinOp::Add) => tape.broadcast_axis(grad, axis, extent)?,
        Some(BinOp::Mul) => product_adjoint(tape, x, grad, axis, extent, *acc, dtype)?,
        Some(BinOp::Max) | Some(BinOp::Min) => {
            extremum_adjoint(tape, x, out, grad, axis, extent, tie, dtype)?
        }
        _ => {
            return Err(Error::Numeric(format!(
                "no adjoint for a {}-slot carrier; multi-slot carriers are minted \
                 by the fold laws after autograd",
                carrier.width()
            )));
        }
    };
    // A fold's output is at `acc`, so an f16 `max` hands this adjoint an f32
    // output and an f32 incoming gradient. The adjoint of a value must carry
    // that value's own dtype, so narrow back on the way out.
    let dx = cast_to(tape, dx, dtype)?;
    Ok(smallvec::smallvec![Some(dx)])
}

/// `v` at `dtype`, or `v` unchanged when it is already there. The identity
/// case emits no node.
fn cast_to(tape: &mut dyn Tape, v: Val, dtype: Dtype) -> Result<Val> {
    if tape.dtype_of(v) == dtype {
        return Ok(v);
    }
    let body = ScalarExpr::cast(dtype, tape.arg_like(v, 0));
    tape.map(body, &[v])
}

/// `Arg(slot)` declared at the operand's real dtype `from`, cast into `to`
/// when the two differ. Emits the bare `Arg` when they agree.
fn arg_at(slot: u32, from: Dtype, to: Dtype) -> ScalarExpr {
    let a = ScalarExpr::arg(slot, from);
    if from == to { a } else { ScalarExpr::cast(to, a) }
}

/// The zero-aware product rule. A row with no zero gets `g*p/x`; a row with
/// exactly one zero gives that slot `g * prod(others)` and every other slot
/// zero; a row with two or more zeros gets exactly zero everywhere.
///
/// Like [`extremum_adjoint`], the running product and the incoming gradient
/// arrive at the accumulator width, so the body is built there and `x` is
/// widened into it. `fold_adjoint` narrows the result back.
fn product_adjoint(
    tape: &mut dyn Tape,
    x: Val,
    grad: Val,
    axis: u32,
    extent: Dim,
    acc: Dtype,
    dtype: Dtype,
) -> Result<Val> {
    let a0 = ScalarExpr::arg(0, dtype);
    let zero_x = crate::tape::lit(0.0, dtype)?;
    let one_x = crate::tape::lit(1.0, dtype)?;
    // The zero test is on `x` itself, at `x`'s width: a value is zero in f16
    // exactly when its widening is zero, so testing before or after the cast
    // gives the same mask.
    let is_zero_x = ScalarExpr::cmp(CmpOp::Eq, a0.clone(), zero_x.clone());

    let safe = tape.map(
        ScalarExpr::select(is_zero_x.clone(), one_x.clone(), a0.clone()),
        &[x],
    )?;
    let zero_mask = tape.map(is_zero_x.clone(), &[x])?;
    let p = tape.fold_binop(BinOp::Mul, axis, acc, safe)?;
    let zero_count = tape.sum_axis(zero_mask, axis)?;

    let bg = tape.broadcast_axis(grad, axis, extent)?;
    let bp = tape.broadcast_axis(p, axis, extent)?;
    let bzc = tape.broadcast_axis(zero_count, axis, extent)?;

    // Everything below is at the width the product and the gradient live at.
    let w = tape.dtype_of(bp);
    let bg = cast_to(tape, bg, w)?;
    let a0w = arg_at(0, dtype, w);
    let zero = crate::tape::lit(0.0, w)?;
    let one = crate::tape::lit(1.0, w)?;
    let is_zero = ScalarExpr::cmp(CmpOp::Eq, a0w.clone(), zero.clone());

    let a1 = ScalarExpr::arg(1, tape.dtype_of(bg));
    let a2 = ScalarExpr::arg(2, w);
    let a3 = ScalarExpr::arg(3, tape.dtype_of(bzc));
    let sf = ScalarExpr::select(is_zero.clone(), one, a0w);
    let gp = ScalarExpr::bin(BinOp::Mul, a1, a2);
    let no_zero = ScalarExpr::bin(BinOp::Div, gp.clone(), sf);
    let one_zero = ScalarExpr::select(is_zero, gp, zero.clone());
    let zc_zero = crate::tape::lit(0.0, tape.dtype_of(bzc))?;
    let zc_one = crate::tape::lit(1.0, tape.dtype_of(bzc))?;
    let body = ScalarExpr::select(
        ScalarExpr::cmp(CmpOp::Eq, a3.clone(), zc_zero),
        no_zero,
        ScalarExpr::select(
            ScalarExpr::cmp(CmpOp::Eq, a3, zc_one),
            one_zero,
            zero,
        ),
    );
    tape.map(body, &[x, bg, bp, bzc])
}

/// Max/Min adjoint. The tie policy is read off the op, never assumed.
///
/// Everything is built at `out`'s dtype (the accumulator width): widening the
/// operand into it is exact, and the extremum is one of those widened operand
/// values, so the equality test that finds the argmax stays exact.
/// `fold_adjoint` narrows the result back. When the widths already agree no
/// cast is emitted.
#[allow(clippy::too_many_arguments)]
fn extremum_adjoint(
    tape: &mut dyn Tape,
    x: Val,
    out: Val,
    grad: Val,
    axis: u32,
    extent: Dim,
    tie: TiePolicy,
    dtype: Dtype,
) -> Result<Val> {
    let bout = tape.broadcast_axis(out, axis, extent)?;
    let bg = tape.broadcast_axis(grad, axis, extent)?;
    let acc = tape.dtype_of(bout);
    let bg = cast_to(tape, bg, acc)?;
    // Arg 0 is `x` at its own width, read into the accumulator; arg 1 is the
    // broadcast fold output, which already is the accumulator.
    let a0 = arg_at(0, dtype, acc);
    let a1 = ScalarExpr::arg(1, acc);
    let zero = crate::tape::lit(0.0, acc)?;

    match tie {
        TiePolicy::SplitEvenly => {
            let mask = tape.map(ScalarExpr::cmp(CmpOp::Eq, a0.clone(), a1.clone()), &[x, bout])?;
            let count = tape.sum_axis(mask, axis)?;
            let bcount = tape.broadcast_axis(count, axis, extent)?;
            let m0 = tape.arg_like(mask, 0);
            let g1 = tape.arg_like(bg, 1);
            let c2 = tape.arg_like(bcount, 2);
            let body = ScalarExpr::bin(BinOp::Div, ScalarExpr::bin(BinOp::Mul, m0, g1), c2);
            tape.map(body, &[mask, bg, bcount])
        }
        TiePolicy::FirstWins => {
            // A masked index fold picks the lowest matching position; every
            // other slot then compares unequal and receives zero.
            let big = crate::tape::lit(sentinel(acc, extent), acc)?;
            let idx = ScalarExpr::cast(acc, ScalarExpr::index_of(axis));
            let pos = tape.map(
                ScalarExpr::select(
                    ScalarExpr::cmp(CmpOp::Eq, a0.clone(), a1.clone()),
                    idx.clone(),
                    big,
                ),
                &[x, bout],
            )?;
            // `FirstWins` on the index fold itself: its own adjoint, if
            // anyone ever takes a second derivative, must not split a tie
            // between two positions that are equal by construction.
            let facc = tape.dtype_of(pos).compute_dtype();
            let ident = Carrier::binop_identity(BinOp::Min, facc).ok_or_else(|| {
                Error::Dtype(format!("Min has no identity in {facc:?}"))
            })?;
            let first = tape.fold(
                Carrier::binop(BinOp::Min, ident, facc).with_tie(TiePolicy::FirstWins),
                axis,
                facc,
                pos,
            )?;
            let bfirst = tape.broadcast_axis(first, axis, extent)?;
            let f0 = tape.arg_like(bfirst, 0);
            let g1 = tape.arg_like(bg, 1);
            let idx_at = ScalarExpr::cast(tape.dtype_of(bfirst), ScalarExpr::index_of(axis));
            let zero_at = if tape.dtype_of(bg) == acc {
                zero
            } else {
                crate::tape::lit(0.0, tape.dtype_of(bg))?
            };
            let body = ScalarExpr::select(ScalarExpr::cmp(CmpOp::Eq, idx_at, f0), g1, zero_at);
            tape.map(body, &[bfirst, bg])
        }
    }
}

/// A value strictly larger than any legal index along `extent`.
fn sentinel(dtype: Dtype, extent: Dim) -> f32 {
    let bound = extent.as_const().map_or(60_000.0, |e| e as f32 + 1.0);
    match dtype {
        Dtype::F16 => bound.min(60_000.0),
        _ => bound,
    }
}

/// `Scatter{Add}` a gradient back through an arbitrary index map: flatten,
/// scatter into a zero base, reshape. The index tensor is a `Map` of
/// `IndexOf` terms — never a host loop.
fn scatter_back(
    tape: &mut dyn Tape,
    grad: Val,
    idx_expr: ScalarExpr,
    xshape: &[Dim],
    dtype: Dtype,
) -> Result<Val> {
    let numel = const_numel(xshape)
        .ok_or_else(|| Error::Shape("scatter adjoint needs decidable source extents".into()))?;
    let idx = tape.map(idx_expr, &[grad])?;
    let flat_idx = tape.flatten(idx)?;
    let flat_grad = tape.flatten(grad)?;
    let base = tape.zeros_shaped(dtype, &[Dim::Const(numel)])?;
    let scattered = tape.scatter_add(0, base, flat_idx, flat_grad)?;
    let shape: Dims = xshape.iter().copied().collect();
    tape.reshape(scattered, &shape)
}

fn u32_mul(e: ScalarExpr, k: u64) -> ScalarExpr {
    if k == 1 {
        return e;
    }
    ScalarExpr::bin(
        BinOp::Mul,
        e,
        ScalarExpr::lit(fusor2_ir::dtype::Splat::U32(k as u32)),
    )
}

fn u32_sum(terms: Vec<ScalarExpr>, constant: u64) -> ScalarExpr {
    let mut acc: Option<ScalarExpr> = None;
    for t in terms {
        acc = Some(match acc {
            Some(a) => ScalarExpr::bin(BinOp::Add, a, t),
            None => t,
        });
    }
    let base = acc.unwrap_or_else(|| ScalarExpr::lit(fusor2_ir::dtype::Splat::U32(0)));
    if constant == 0 {
        base
    } else {
        ScalarExpr::bin(
            BinOp::Add,
            base,
            ScalarExpr::lit(fusor2_ir::dtype::Splat::U32(constant as u32)),
        )
    }
}

fn dim_const(d: Dim) -> Result<u64> {
    d.as_const()
        .ok_or_else(|| Error::Shape("expected a decidable extent".into()))
}
