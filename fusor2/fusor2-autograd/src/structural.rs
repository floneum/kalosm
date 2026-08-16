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
pub fn structural_adjoint(
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
pub fn restride_adjoint(
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
pub fn window_adjoint(
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
pub fn gather_adjoint(
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
pub fn scatter_adjoint(
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
pub fn fold_adjoint(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::GraphTape;
    use crate::tape::testing::graph;
    use fusor2_ir::egraph::{EGraph, Id};
    use fusor2_ir::carrier::{ArgRemap, Carrier};
    use fusor2_ir::ir::logical::{BufferId, LeafKind};

    fn param(g: &mut EGraph, shape: &[u64]) -> Val {
        let n = g.len() as u32;
        g.add(Op::Logical(Logical::Leaf(LeafKind::Param {
            name: BufferId(n),
            dtype: Dtype::F32,
            shape: shape.iter().map(|d| Dim::Const(*d)).collect(),
        })))
        .unwrap()
    }

    /// Every node reachable from `root`, for structural assertions.
    fn reachable(g: &EGraph, root: Id) -> Vec<Id> {
        let mut seen = vec![false; g.len()];
        let mut stack = vec![root];
        let mut out = Vec::new();
        while let Some(id) = stack.pop() {
            if seen[id.index()] {
                continue;
            }
            seen[id.index()] = true;
            out.push(id);
            for c in g.node(id).children.iter() {
                stack.push(*c);
            }
        }
        out
    }

    fn count<F: Fn(&Logical) -> bool>(g: &EGraph, root: Id, f: F) -> usize {
        reachable(g, root)
            .into_iter()
            .filter(|id| matches!(&g.node(*id).op, Op::Logical(op) if f(op)))
            .count()
    }

    #[test]
    fn a_broadcast_restride_adjoint_is_exactly_one_fold_add() {
        let mut g = graph();
        let x = param(&mut g, &[1, 4]);
        let specs: SmallVec<[StrideSpec; 6]> =
            smallvec::smallvec![StrideSpec::broadcast(Dim::Const(3)), StrideSpec::dim(1, Dim::Const(4))];
        let y = g
            .add(Op::Logical(Logical::Restride {
                specs,
                bounds: fusor2_ir::shape::BoundsProof::Static,
                x,
            }))
            .unwrap();
        let node = g.node(y).clone();
        let grad = param(&mut g, &[3, 4]);
        let mut t = GraphTape::new(&mut g);
        let dx = restride_adjoint(&mut t, &node, grad, &[x], y).unwrap()[0].unwrap();
        assert_eq!(
            t.facts(dx).shape,
            Dims::from_slice(&[Dim::Const(1), Dim::Const(4)])
        );
        let g = t.graph();
        assert_eq!(
            count(g, dx, |op| matches!(
                op,
                Logical::Fold { carrier, .. } if carrier.kind() == Some(BinOp::Add)
            )),
            1,
            "sum-reduce over the one stride-0 axis, and nothing else"
        );
        assert_eq!(
            count(g, dx, |op| matches!(op, Logical::Scatter { .. })),
            0,
            "the broadcast backward is never a scatter"
        );
    }

    #[test]
    fn a_permutation_restride_inverts_into_one_restride() {
        let mut g = graph();
        let x = param(&mut g, &[2, 3, 4]);
        // transpose(0, 2)
        let specs: SmallVec<[StrideSpec; 6]> = smallvec::smallvec![
            StrideSpec::dim(2, Dim::Const(4)),
            StrideSpec::dim(1, Dim::Const(3)),
            StrideSpec::dim(0, Dim::Const(2)),
        ];
        let y = g
            .add(Op::Logical(Logical::Restride {
                specs,
                bounds: fusor2_ir::shape::BoundsProof::Static,
                x,
            }))
            .unwrap();
        let node = g.node(y).clone();
        let grad = param(&mut g, &[4, 3, 2]);
        let mut t = GraphTape::new(&mut g);
        let dx = restride_adjoint(&mut t, &node, grad, &[x], y).unwrap()[0].unwrap();
        assert_eq!(
            t.facts(dx).shape,
            Dims::from_slice(&[Dim::Const(2), Dim::Const(3), Dim::Const(4)])
        );
        assert!(matches!(t.node(dx).op, Op::Logical(Logical::Restride { .. })));
    }

    /// `flatten_all` names only the innermost input axis, so `invert_runs`
    /// declines it; the reshape stage must catch it instead of falling
    /// through to the index-tensor scatter.
    #[test]
    fn a_merging_reshape_inverts_into_one_restride() {
        for (xshape, view) in [
            (vec![2u64, 3], vec![6u64]),
            (vec![2, 3, 4], vec![6, 4]),
            (vec![2, 3, 4], vec![24]),
            (vec![4, 6], vec![24]),
        ] {
            let mut g = graph();
            let x = param(&mut g, &xshape);
            let xdims: Vec<Dim> = xshape.iter().map(|d| Dim::Const(*d)).collect();
            let vdims: Vec<Dim> = view.iter().map(|d| Dim::Const(*d)).collect();
            let specs = reshape_specs(&xdims, &vdims).unwrap();
            let y = g
                .add(Op::Logical(Logical::Restride {
                    specs,
                    bounds: fusor2_ir::shape::BoundsProof::Static,
                    x,
                }))
                .unwrap();
            let node = g.node(y).clone();
            let grad = param(&mut g, &view);
            let mut t = GraphTape::new(&mut g);
            let dx = restride_adjoint(&mut t, &node, grad, &[x], y).unwrap()[0].unwrap();
            assert_eq!(
                t.facts(dx).shape,
                Dims::from_slice(&xdims),
                "{xshape:?} -> {view:?}"
            );
            assert!(
                matches!(t.node(dx).op, Op::Logical(Logical::Restride { .. })),
                "{xshape:?} -> {view:?} should invert into one Restride"
            );
            let gr = t.graph();
            assert_eq!(
                count(gr, dx, |op| matches!(op, Logical::Scatter { .. })),
                0,
                "{xshape:?} -> {view:?}: a bijective reshape is never a scatter"
            );
        }
    }

    #[test]
    fn a_sliced_restride_falls_back_to_scatter_add() {
        let mut g = graph();
        let x = param(&mut g, &[8]);
        let specs: SmallVec<[StrideSpec; 6]> =
            smallvec::smallvec![StrideSpec::dim(0, Dim::Const(3)).with_offset(Dim::Const(2))];
        let y = g
            .add(Op::Logical(Logical::Restride {
                specs,
                bounds: fusor2_ir::shape::BoundsProof::Static,
                x,
            }))
            .unwrap();
        let node = g.node(y).clone();
        let grad = param(&mut g, &[3]);
        let mut t = GraphTape::new(&mut g);
        let dx = restride_adjoint(&mut t, &node, grad, &[x], y).unwrap()[0].unwrap();
        assert_eq!(t.facts(dx).shape, Dims::from_slice(&[Dim::Const(8)]));
        let g = t.graph();
        assert_eq!(
            count(g, dx, |op| matches!(
                op,
                Logical::Scatter {
                    combine: ScatterCombine::Add,
                    ..
                }
            )),
            1
        );
    }

    fn window(g: &mut EGraph, x: Val, specs: &[SlidingWindow]) -> Val {
        g.add(Op::Logical(Logical::Window {
            specs: specs.iter().copied().collect(),
            x,
        }))
        .unwrap()
    }

    /// Trainer constraint 4: a non-overlapping pool's adjoint is a view.
    #[test]
    fn a_non_overlapping_window_adjoint_contains_no_scatter() {
        let mut g = graph();
        let x = param(&mut g, &[1, 24, 768]);
        let spec = SlidingWindow::new(2, 4, 4);
        let y = window(&mut g, x, &[spec]);
        assert_eq!(
            g.facts(y).shape,
            Dims::from_slice(&[Dim::Const(1), Dim::Const(24), Dim::Const(192), Dim::Const(4)])
        );
        let node = g.node(y).clone();
        let grad = param(&mut g, &[1, 24, 192, 4]);
        let mut t = GraphTape::new(&mut g);
        let dx = window_adjoint(&mut t, &node, grad, &[x], y).unwrap()[0].unwrap();
        assert_eq!(
            t.facts(dx).shape,
            Dims::from_slice(&[Dim::Const(1), Dim::Const(24), Dim::Const(768)])
        );
        let g = t.graph();
        assert_eq!(
            count(g, dx, |op| matches!(op, Logical::Scatter { .. })),
            0,
            "pool_max with stride == window must produce no scatter"
        );
    }

    #[test]
    fn an_overlapping_window_adjoint_is_an_overlap_add_scatter() {
        let mut g = graph();
        let x = param(&mut g, &[6]);
        let spec = SlidingWindow::new(0, 3, 1);
        let y = window(&mut g, x, &[spec]);
        assert_eq!(
            g.facts(y).shape,
            Dims::from_slice(&[Dim::Const(4), Dim::Const(3)])
        );
        let node = g.node(y).clone();
        let grad = param(&mut g, &[4, 3]);
        let mut t = GraphTape::new(&mut g);
        let dx = window_adjoint(&mut t, &node, grad, &[x], y).unwrap()[0].unwrap();
        assert_eq!(t.facts(dx).shape, Dims::from_slice(&[Dim::Const(6)]));
        let g = t.graph();
        assert_eq!(
            count(g, dx, |op| matches!(
                op,
                Logical::Scatter {
                    combine: ScatterCombine::Add,
                    unique: false,
                    ..
                }
            )),
            1
        );
    }

    #[test]
    fn multi_axis_non_overlapping_windows_still_avoid_a_scatter() {
        let mut g = graph();
        let x = param(&mut g, &[2, 4, 6]);
        let specs = [SlidingWindow::new(1, 2, 2), SlidingWindow::new(2, 3, 3)];
        let y = window(&mut g, x, &specs);
        let node = g.node(y).clone();
        let gshape = g.facts(y).shape.clone();
        let grad = g
            .add(Op::Logical(Logical::Leaf(LeafKind::Param {
                name: BufferId(99),
                dtype: Dtype::F32,
                shape: gshape,
            })))
            .unwrap();
        let mut t = GraphTape::new(&mut g);
        let dx = window_adjoint(&mut t, &node, grad, &[x], y).unwrap()[0].unwrap();
        assert_eq!(
            t.facts(dx).shape,
            Dims::from_slice(&[Dim::Const(2), Dim::Const(4), Dim::Const(6)])
        );
        let g = t.graph();
        assert_eq!(count(g, dx, |op| matches!(op, Logical::Scatter { .. })), 0);
    }

    /// Trainer constraint 3: no one-hot matmul anywhere.
    #[test]
    fn gather_adjoint_is_a_scatter_add_with_no_contraction() {
        let mut g = graph();
        let table = param(&mut g, &[1024, 24]);
        let idx = g
            .add(Op::Logical(Logical::Leaf(LeafKind::Buffer {
                name: BufferId(7),
                dtype: Dtype::U32,
                shape: smallvec::smallvec![Dim::Const(6)],
            })))
            .unwrap();
        let y = g.add(Op::Logical(Logical::Gather { axis: 0, x: table, idx })).unwrap();
        let node = g.node(y).clone();
        let grad = param(&mut g, &[6, 24]);
        let mut t = GraphTape::new(&mut g);
        let grads = gather_adjoint(&mut t, &node, grad, &[table, idx], y).unwrap();
        assert!(grads[1].is_none(), "indices are not differentiable");
        let dx = grads[0].unwrap();
        assert_eq!(
            t.facts(dx).shape,
            Dims::from_slice(&[Dim::Const(1024), Dim::Const(24)])
        );
        let g = t.graph();
        assert_eq!(
            count(g, dx, |op| matches!(
                op,
                Logical::Scatter {
                    combine: ScatterCombine::Add,
                    unique: false,
                    ..
                }
            )),
            1
        );
        assert_eq!(
            count(g, dx, |op| matches!(op, Logical::Contract { .. })),
            0,
            "the one-hot matmul is structurally deleted"
        );
    }

    #[test]
    fn scatter_set_zeroes_the_written_region_for_the_base() {
        let mut g = graph();
        let base = param(&mut g, &[8, 3]);
        let upd = param(&mut g, &[2, 3]);
        let idx = g
            .add(Op::Logical(Logical::Leaf(LeafKind::Buffer {
                name: BufferId(5),
                dtype: Dtype::U32,
                shape: smallvec::smallvec![Dim::Const(2)],
            })))
            .unwrap();
        let y = g
            .add(Op::Logical(Logical::Scatter {
                axis: 0,
                combine: ScatterCombine::Set,
                base,
                idx,
                upd,
                unique: true,
            }))
            .unwrap();
        let node = g.node(y).clone();
        let grad = param(&mut g, &[8, 3]);
        let mut t = GraphTape::new(&mut g);
        let grads = scatter_adjoint(&mut t, &node, grad, &[base, idx, upd], y).unwrap();
        let d_base = grads[0].unwrap();
        let d_upd = grads[2].unwrap();
        assert!(grads[1].is_none());
        assert!(matches!(
            t.node(d_base).op,
            Op::Logical(Logical::Scatter {
                combine: ScatterCombine::Set,
                ..
            })
        ));
        assert_eq!(
            t.facts(d_upd).shape,
            Dims::from_slice(&[Dim::Const(2), Dim::Const(3)])
        );
    }

    #[test]
    fn scatter_add_passes_the_base_gradient_through_unchanged() {
        let mut g = graph();
        let base = param(&mut g, &[8]);
        let upd = param(&mut g, &[2]);
        let idx = g
            .add(Op::Logical(Logical::Leaf(LeafKind::Buffer {
                name: BufferId(5),
                dtype: Dtype::U32,
                shape: smallvec::smallvec![Dim::Const(2)],
            })))
            .unwrap();
        let y = g
            .add(Op::Logical(Logical::Scatter {
                axis: 0,
                combine: ScatterCombine::Add,
                base,
                idx,
                upd,
                unique: false,
            }))
            .unwrap();
        let node = g.node(y).clone();
        let grad = param(&mut g, &[8]);
        let mut t = GraphTape::new(&mut g);
        let grads = scatter_adjoint(&mut t, &node, grad, &[base, idx, upd], y).unwrap();
        assert_eq!(grads[0], Some(grad));
    }

    fn carrier(op: BinOp) -> Carrier {
        Carrier::binop(op, Carrier::binop_identity(op, Dtype::F32).unwrap(), Dtype::F32)
    }

    fn fold(g: &mut EGraph, x: Val, op: BinOp, axis: u32) -> Val {
        fold_tied(g, x, carrier(op), axis)
    }

    fn fold_tied(g: &mut EGraph, x: Val, carrier: Carrier, axis: u32) -> Val {
        g.add(Op::Logical(Logical::Fold {
            carrier,
            axis,
            acc: Dtype::F32,
            ins: smallvec::smallvec![x],
        }))
        .unwrap()
    }

    #[test]
    fn sum_broadcasts_its_gradient_back_over_the_axis() {
        let mut g = graph();
        let x = param(&mut g, &[3, 5]);
        let y = fold(&mut g, x, BinOp::Add, 1);
        let node = g.node(y).clone();
        let grad = param(&mut g, &[3]);
        let mut t = GraphTape::new(&mut g);
        let dx = fold_adjoint(&mut t, &node, grad, &[x], y).unwrap()[0].unwrap();
        assert_eq!(
            t.facts(dx).shape,
            Dims::from_slice(&[Dim::Const(3), Dim::Const(5)])
        );
        match &t.node(dx).op {
            Op::Logical(Logical::Restride { specs, .. }) => assert!(specs[1].is_broadcast()),
            other => panic!("expected a broadcast Restride, got {other:?}"),
        }
    }

    #[test]
    fn product_uses_the_zero_aware_rule() {
        let mut g = graph();
        let x = param(&mut g, &[2, 3]);
        let y = fold(&mut g, x, BinOp::Mul, 1);
        let node = g.node(y).clone();
        let grad = param(&mut g, &[2]);
        let mut t = GraphTape::new(&mut g);
        let dx = fold_adjoint(&mut t, &node, grad, &[x], y).unwrap()[0].unwrap();
        let g = t.graph();
        // The adjoint recomputes the product over zero-substituted values
        // rather than reusing the primal's, so `g*p/x` never divides by zero.
        assert_eq!(
            count(g, dx, |op| matches!(
                op,
                Logical::Fold { carrier, .. } if carrier.kind() == Some(BinOp::Mul)
            )),
            1,
            "the zero-substituted product"
        );
        assert_eq!(
            count(g, dx, |op| matches!(
                op,
                Logical::Fold { carrier, .. } if carrier.kind() == Some(BinOp::Add)
            )),
            1,
            "the zero count"
        );
    }

    #[test]
    fn extrema_read_the_declared_tie_policy() {
        for (tie, extra_folds) in [(TiePolicy::SplitEvenly, 1), (TiePolicy::FirstWins, 1)] {
            let mut g = graph();
            let x = param(&mut g, &[2, 3]);
            let y = fold_tied(&mut g, x, carrier(BinOp::Max).with_tie(tie), 1);
            let node = g.node(y).clone();
            let grad = param(&mut g, &[2]);
            let mut t = GraphTape::new(&mut g);
            let dx = fold_adjoint(&mut t, &node, grad, &[x], y).unwrap()[0].unwrap();
            let g = t.graph();
            let folds = count(g, dx, |op| matches!(op, Logical::Fold { .. }));
            assert_eq!(
                folds,
                1 + extra_folds,
                "{tie:?}: the primal extremum plus its tie-resolution fold"
            );
        }
    }

    /// A multi-slot carrier has no analytic adjoint here: those carriers are
    /// minted by the fold laws after autograd, so this must be an `Err`,
    /// never a silently wrong slot-0 gradient.
    #[test]
    fn multi_slot_carrier_folds_are_refused() {
        let mut g = graph();
        let x = param(&mut g, &[2, 3]);
        let pair = carrier(BinOp::Max)
            .tuple(&carrier(BinOp::Add), &ArgRemap::identity(1))
            .carrier;
        let y = fold_tied(&mut g, x, pair, 1);
        let node = g.node(y).clone();
        let grad = param(&mut g, &[2]);
        let mut t = GraphTape::new(&mut g);
        assert!(matches!(
            fold_adjoint(&mut t, &node, grad, &[x], y),
            Err(Error::Numeric(_))
        ));

        // So is a fold whose lift computes — fusion mints those, also after
        // autograd.
        let fused = carrier(BinOp::Add).with_lift([ScalarExpr::un(
            fusor2_ir::scalar::UnOp::Exp,
            ScalarExpr::arg(0, Dtype::F32),
        )]);
        let z = fold_tied(&mut g, x, fused, 1);
        let node = g.node(z).clone();
        let mut t = GraphTape::new(&mut g);
        assert!(matches!(
            fold_adjoint(&mut t, &node, grad, &[x], z),
            Err(Error::Numeric(_))
        ));
    }
}

#[cfg(test)]
mod numeric {
    //! Every structural adjoint against a central difference of the forward
    //! it claims to differentiate, plus the exact values the parity bullets
    //! name where a finite difference cannot see them (ties, zeros).

    use super::*;
    use crate::backward::backward_into;
    use crate::tape::TapeExt;
    use crate::tape::testing::{Env, caps, check_gradients, eval, graph};
    use crate::tape::GraphTape;
    use fusor2_ir::scalar::UnOp;
    use fusor2_ir::egraph::{EGraph, Id};
    use fusor2_ir::carrier::Carrier;
    use fusor2_ir::ir::logical::{BufferId, LeafKind};
    use rustc_hash::FxHashMap;

    fn param(g: &mut EGraph, shape: &[u64]) -> Id {
        let n = g.len() as u32;
        g.add(Op::Logical(Logical::Leaf(LeafKind::Param {
            name: BufferId(n),
            dtype: Dtype::F32,
            shape: shape.iter().map(|d| Dim::Const(*d)).collect(),
        })))
        .unwrap()
    }

    fn u32_buffer(g: &mut EGraph, len: u64) -> Id {
        let n = g.len() as u32;
        g.add(Op::Logical(Logical::Leaf(LeafKind::Buffer {
            name: BufferId(n),
            dtype: Dtype::U32,
            shape: smallvec::smallvec![Dim::Const(len)],
        })))
        .unwrap()
    }

    fn ones(g: &mut EGraph, shape: &[u64]) -> Id {
        g.add(Op::Logical(Logical::Leaf(LeafKind::Const {
            value: fusor2_ir::dtype::Splat::F32(1.0),
            shape: shape.iter().map(|d| Dim::Const(*d)).collect(),
        })))
        .unwrap()
    }

    fn env(pairs: &[(Id, Vec<f32>)]) -> Env {
        let mut e: FxHashMap<Id, Vec<f32>> = FxHashMap::default();
        for (k, v) in pairs {
            e.insert(*k, v.clone());
        }
        e
    }

    fn fold_node(g: &mut EGraph, x: Id, op: BinOp, axis: u32) -> Id {
        fold_node_tied(g, x, op, None, axis)
    }

    fn fold_node_tied(
        g: &mut EGraph,
        x: Id,
        op: BinOp,
        tie: Option<TiePolicy>,
        axis: u32,
    ) -> Id {
        fold_node_acc(g, x, op, tie, axis, Dtype::F32)
    }

    /// A fold whose accumulator may be wider than its operand — the shape the
    /// runtime always builds for a narrow float, since `accum_dtype` floors
    /// every f16/bf16 fold at f32.
    fn fold_node_acc(
        g: &mut EGraph,
        x: Id,
        op: BinOp,
        tie: Option<TiePolicy>,
        axis: u32,
        acc: Dtype,
    ) -> Id {
        let mut carrier = Carrier::binop(op, Carrier::binop_identity(op, acc).unwrap(), acc);
        if let Some(t) = tie {
            carrier.tie = Some(t);
        }
        g.add(Op::Logical(Logical::Fold {
            carrier,
            axis,
            acc,
            ins: smallvec::smallvec![x],
        }))
        .unwrap()
    }

    fn param_dtype(g: &mut EGraph, shape: &[u64], dtype: Dtype) -> Id {
        let n = g.len() as u32;
        g.add(Op::Logical(Logical::Leaf(LeafKind::Param {
            name: BufferId(n),
            dtype,
            shape: shape.iter().map(|d| Dim::Const(*d)).collect(),
        })))
        .unwrap()
    }

    fn splat_ones(g: &mut EGraph, shape: &[u64], dtype: Dtype) -> Id {
        g.add(Op::Logical(Logical::Leaf(LeafKind::Const {
            value: crate::tape::splat_of(dtype, 1.0).unwrap(),
            shape: shape.iter().map(|d| Dim::Const(*d)).collect(),
        })))
        .unwrap()
    }

    #[test]
    fn sum_and_mean_gradients_match_finite_differences() {
        let mut g = graph();
        let x = param(&mut g, &[2, 4]);
        let s = fold_node(&mut g, x, BinOp::Add, 1);
        let m = {
            let mut t = GraphTape::new(&mut g);
            t.mul_scalar(s, 0.25).unwrap()
        };
        let seed = ones(&mut g, &[2]);
        let grads = backward_into(&mut g, &caps(), m, seed, &[x]).unwrap();
        let e = env(&[(x, vec![0.3, -1.2, 2.0, 0.7, 1.1, -0.4, 0.9, 1.5])]);
        check_gradients(&g, m, &[x], &grads, &e, 2e-3);
        // mean over 4 elements: every slot gets exactly 1/4.
        let analytic = eval(&g, grads[0].unwrap(), &e);
        assert!(analytic.iter().all(|v| (v - 0.25).abs() < 1e-6));
    }

    #[test]
    fn a_broadcast_backward_sums_to_the_broadcast_factor() {
        let mut g = graph();
        let x = param(&mut g, &[1, 4]);
        let b = {
            let mut t = GraphTape::new(&mut g);
            t.broadcast_to(x, &[Dim::Const(3), Dim::Const(4)]).unwrap()
        };
        let seed = ones(&mut g, &[3, 4]);
        let grads = backward_into(&mut g, &caps(), b, seed, &[x]).unwrap();
        let e = env(&[(x, vec![1.0, 2.0, 3.0, 4.0])]);
        let analytic = eval(&g, grads[0].unwrap(), &e);
        assert_eq!(analytic, vec![3.0; 4], "a unit seed sums to 3*g");
        check_gradients(&g, b, &[x], &grads, &e, 2e-3);
    }

    #[test]
    fn max_with_split_evenly_divides_a_three_way_tie() {
        let mut g = graph();
        let x = param(&mut g, &[1, 4]);
        let y = fold_node_tied(&mut g, x, BinOp::Max, Some(TiePolicy::SplitEvenly), 1);
        let seed = ones(&mut g, &[1]);
        let grads = backward_into(&mut g, &caps(), y, seed, &[x]).unwrap();
        let e = env(&[(x, vec![5.0, 5.0, 1.0, 5.0])]);
        let analytic = eval(&g, grads[0].unwrap(), &e);
        let third = 1.0 / 3.0;
        for (got, want) in analytic.iter().zip([third, third, 0.0, third]) {
            assert!((got - want).abs() < 1e-6, "{analytic:?}");
        }
    }

    #[test]
    fn max_with_first_wins_gives_the_lowest_index_everything() {
        let mut g = graph();
        let x = param(&mut g, &[1, 4]);
        let y = fold_node_tied(&mut g, x, BinOp::Max, Some(TiePolicy::FirstWins), 1);
        let seed = ones(&mut g, &[1]);
        let grads = backward_into(&mut g, &caps(), y, seed, &[x]).unwrap();
        let e = env(&[(x, vec![5.0, 5.0, 1.0, 5.0])]);
        let analytic = eval(&g, grads[0].unwrap(), &e);
        assert_eq!(analytic, vec![1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn min_reads_its_own_tie_policy() {
        let mut g = graph();
        let x = param(&mut g, &[1, 3]);
        let y = fold_node_tied(&mut g, x, BinOp::Min, Some(TiePolicy::SplitEvenly), 1);
        let seed = ones(&mut g, &[1]);
        let grads = backward_into(&mut g, &caps(), y, seed, &[x]).unwrap();
        let e = env(&[(x, vec![-2.0, 7.0, -2.0])]);
        let analytic = eval(&g, grads[0].unwrap(), &e);
        assert_eq!(analytic, vec![0.5, 0.0, 0.5]);
    }

    /// Every narrow float accumulates at `compute_dtype`, so a `max`/`min`
    /// fold's output comes back one width wider than the operand. The adjoint
    /// must read each operand at the dtype that operand actually has.
    #[test]
    fn the_extremum_adjoint_reads_a_wider_accumulator_at_its_own_width() {
        for dtype in [Dtype::F16, Dtype::BF16] {
            for tie in [TiePolicy::SplitEvenly, TiePolicy::FirstWins] {
                for op in [BinOp::Max, BinOp::Min] {
                    let mut g = graph();
                    let x = param_dtype(&mut g, &[1, 4], dtype);
                    let acc = dtype.compute_dtype();
                    assert_ne!(acc, dtype, "{dtype:?} must widen, or this proves nothing");
                    let y = fold_node_acc(&mut g, x, op, Some(tie), 1, acc);
                    let seed = splat_ones(&mut g, &[1], acc);
                    let grads = backward_into(&mut g, &caps(), y, seed, &[x])
                        .unwrap_or_else(|e| panic!("{dtype:?}/{tie:?}/{op:?} backward: {e}"));
                    let dx = grads[0].expect("x must receive a gradient");
                    // The gradient of an f16 value is an f16 value.
                    assert_eq!(g.facts(dx).dtype, dtype, "{dtype:?}/{tie:?}/{op:?}");
                    assert_eq!(g.facts(dx).shape, g.facts(x).shape);
                }
            }
        }
    }

    /// The narrowing is a property of `fold_adjoint`, not of one branch: every
    /// carrier it knows how to differentiate hands back a gradient at the
    /// operand's own width, `Add` and `Mul` included.
    #[test]
    fn every_fold_carrier_returns_a_gradient_at_the_operands_width() {
        for dtype in [Dtype::F16, Dtype::BF16, Dtype::F32] {
            for op in [BinOp::Add, BinOp::Mul, BinOp::Max, BinOp::Min] {
                let mut g = graph();
                let x = param_dtype(&mut g, &[1, 3], dtype);
                let acc = dtype.compute_dtype();
                let y = fold_node_acc(&mut g, x, op, None, 1, acc);
                let seed = splat_ones(&mut g, &[1], acc);
                let grads = backward_into(&mut g, &caps(), y, seed, &[x])
                    .unwrap_or_else(|e| panic!("{dtype:?}/{op:?} backward: {e}"));
                let dx = grads[0].expect("x must receive a gradient");
                assert_eq!(g.facts(dx).dtype, dtype, "{dtype:?}/{op:?}");
            }
        }
    }

    /// Returns the first `Bin`/`Cmp`/`Select` in `e` whose operands or arms
    /// differ in width. `check_arg_dtypes` cannot see this: it only checks
    /// that each `Arg(i)` leaf is declared at operand `i`'s dtype, so a
    /// mixed-width node is silently constructible.
    fn width_mismatch(e: &ScalarExpr) -> Option<String> {
        use fusor2_ir::scalar::ScalarKind as K;
        let pair = |what: &str, a: &ScalarExpr, b: &ScalarExpr| {
            (a.dtype() != b.dtype())
                .then(|| format!("{what} mixes {:?} and {:?}", a.dtype(), b.dtype()))
        };
        let here = match e.kind() {
            K::Bin { op, a, b } => pair(&format!("{op:?}"), a, b),
            K::Cmp { op, a, b } => pair(&format!("{op:?}"), a, b),
            K::Select { t, f, .. } => pair("select arms", t, f),
            _ => None,
        };
        if here.is_some() {
            return here;
        }
        let kids: Vec<&ScalarExpr> = match e.kind() {
            K::Un { x, .. } => vec![x],
            K::Bin { a, b, .. } | K::Cmp { a, b, .. } => vec![a, b],
            K::Select { c, t, f } => vec![c, t, f],
            K::Cast { x, .. } | K::Bitcast { x, .. } | K::Round { x, .. } => vec![x],
            _ => vec![],
        };
        kids.into_iter().find_map(width_mismatch)
    }

    /// No adjoint this file builds may hand an emitter a mixed-width
    /// expression.
    #[test]
    fn no_fold_adjoint_builds_a_mixed_width_expression() {
        for dtype in [Dtype::F16, Dtype::BF16, Dtype::F32] {
            for op in [BinOp::Add, BinOp::Mul, BinOp::Max, BinOp::Min] {
                for tie in [None, Some(TiePolicy::SplitEvenly), Some(TiePolicy::FirstWins)] {
                    let mut g = graph();
                    let x = param_dtype(&mut g, &[1, 3], dtype);
                    let acc = dtype.compute_dtype();
                    let y = fold_node_acc(&mut g, x, op, tie, 1, acc);
                    let seed = splat_ones(&mut g, &[1], acc);
                    let grads = backward_into(&mut g, &caps(), y, seed, &[x]).unwrap();
                    let dx = grads[0].unwrap();
                    // Every node up to `dx` — the adjoint chain is the tail of
                    // the graph, so scanning all of it also covers the forward.
                    for i in 0..=dx.0 {
                        if let Op::Logical(Logical::Map { expr, .. }) = &g.node(Id(i)).op
                            && let Some(bad) = width_mismatch(expr)
                        {
                            panic!("{dtype:?}/{op:?}/{tie:?} built a Map body where {bad}");
                        }
                    }
                }
            }
        }
    }

    /// The f16 product rule by value, since a finite difference cannot see the
    /// zero cases: the widened accumulator must not change which branch fires.
    #[test]
    fn a_narrow_product_rule_still_handles_zeros() {
        for (row, want) in [
            (vec![2.0, 3.0, 4.0], vec![12.0, 8.0, 6.0]),
            (vec![2.0, 0.0, 4.0], vec![0.0, 8.0, 0.0]),
            (vec![0.0, 0.0, 4.0], vec![0.0, 0.0, 0.0]),
        ] {
            let mut g = graph();
            let x = param_dtype(&mut g, &[1, 3], Dtype::F16);
            let y = fold_node_acc(&mut g, x, BinOp::Mul, None, 1, Dtype::F32);
            let seed = splat_ones(&mut g, &[1], Dtype::F32);
            let grads = backward_into(&mut g, &caps(), y, seed, &[x]).unwrap();
            let e = env(&[(x, row.clone())]);
            let analytic = eval(&g, grads[0].unwrap(), &e);
            assert_eq!(analytic, want, "row {row:?}");
        }
    }

    /// The same widening on the operand side of the fold rather than the
    /// accumulator side: `sum_axis` inside the `SplitEvenly` branch folds the
    /// mask, and the mask is at the mask's dtype, not the tensor's.
    #[test]
    fn a_narrow_max_gradient_still_splits_a_tie_evenly() {
        let mut g = graph();
        let x = param_dtype(&mut g, &[1, 4], Dtype::F16);
        let y = fold_node_acc(&mut g, x, BinOp::Max, Some(TiePolicy::SplitEvenly), 1, Dtype::F32);
        let seed = splat_ones(&mut g, &[1], Dtype::F32);
        let grads = backward_into(&mut g, &caps(), y, seed, &[x]).unwrap();
        let e = env(&[(x, vec![5.0, 5.0, 1.0, 5.0])]);
        let analytic = eval(&g, grads[0].unwrap(), &e);
        let third = 1.0 / 3.0;
        for (got, want) in analytic.iter().zip([third, third, 0.0, third]) {
            assert!((got - want).abs() < 1e-3, "{analytic:?}");
        }
    }

    /// `FirstWins` on a narrow float, by value: the lowest matching index
    /// takes the whole gradient and every other slot takes zero.
    #[test]
    fn a_narrow_first_wins_max_gives_the_lowest_index_everything() {
        let mut g = graph();
        let x = param_dtype(&mut g, &[1, 4], Dtype::F16);
        let y = fold_node_acc(&mut g, x, BinOp::Max, Some(TiePolicy::FirstWins), 1, Dtype::F32);
        let seed = splat_ones(&mut g, &[1], Dtype::F32);
        let grads = backward_into(&mut g, &caps(), y, seed, &[x]).unwrap();
        let e = env(&[(x, vec![5.0, 5.0, 1.0, 5.0])]);
        let analytic = eval(&g, grads[0].unwrap(), &e);
        assert_eq!(analytic, vec![1.0, 0.0, 0.0, 0.0]);
    }

    /// The narrowing is a consequence of the *fold's* widening, never a blanket
    /// cast: an adjoint whose operands genuinely disagree in dtype for any
    /// other reason must still be rejected by `check_arg_dtypes`.
    #[test]
    fn a_dtype_disagreement_that_is_not_the_accumulator_is_still_an_error() {
        let mut g = graph();
        let x = param_dtype(&mut g, &[1, 4], Dtype::F16);
        let bad = ScalarExpr::bin(
            BinOp::Mul,
            ScalarExpr::arg(0, Dtype::F32),
            crate::tape::lit(2.0, Dtype::F32).unwrap(),
        );
        let mut t = GraphTape::new(&mut g);
        let err = t.map(bad, &[x]).expect_err("F32 arg over an F16 operand");
        assert!(
            format!("{err}").contains("Arg(0)"),
            "the arity/dtype check must still fire: {err}"
        );
    }

    #[test]
    fn the_product_rule_handles_zero_one_and_two_zeros() {
        for (row, want) in [
            (vec![2.0, 3.0, 4.0], vec![12.0, 8.0, 6.0]),
            (vec![2.0, 0.0, 4.0], vec![0.0, 8.0, 0.0]),
            (vec![0.0, 0.0, 4.0], vec![0.0, 0.0, 0.0]),
        ] {
            let mut g = graph();
            let x = param(&mut g, &[1, 3]);
            let y = fold_node(&mut g, x, BinOp::Mul, 1);
            let seed = ones(&mut g, &[1]);
            let grads = backward_into(&mut g, &caps(), y, seed, &[x]).unwrap();
            let e = env(&[(x, row.clone())]);
            let analytic = eval(&g, grads[0].unwrap(), &e);
            for (got, exp) in analytic.iter().zip(&want) {
                assert!(
                    (got - exp).abs() < 1e-5,
                    "product over {row:?}: {analytic:?} vs {want:?}"
                );
            }
        }
    }

    /// Trainer constraint 3, numerically: the same bin twice gets the sum.
    #[test]
    fn a_duplicated_gather_index_accumulates_both_gradients() {
        let mut g = graph();
        let table = param(&mut g, &[4, 2]);
        let idx = u32_buffer(&mut g, 3);
        let rows = g.add(Op::Logical(Logical::Gather { axis: 0, x: table, idx })).unwrap();
        // Weight each gathered row differently so the two hits on bin 1 are
        // distinguishable in the sum.
        let w = param(&mut g, &[3, 2]);
        let y = {
            let mut t = GraphTape::new(&mut g);
            t.binary(BinOp::Mul, rows, w).unwrap()
        };
        let seed = ones(&mut g, &[3, 2]);
        let grads = backward_into(&mut g, &caps(), y, seed, &[table]).unwrap();
        let e = env(&[
            (table, vec![0.0; 8]),
            (idx, vec![1.0, 3.0, 1.0]),
            (w, vec![1.0, 2.0, 10.0, 20.0, 100.0, 200.0]),
        ]);
        let analytic = eval(&g, grads[0].unwrap(), &e);
        // bin 1 is hit by rows 0 and 2: 1 + 100 and 2 + 200.
        assert_eq!(analytic, vec![0.0, 0.0, 101.0, 202.0, 0.0, 0.0, 10.0, 20.0]);
    }

    /// Trainer constraint 4, numerically: a non-overlapping pool routes each
    /// gradient to its argmax slot, through a chain with no scatter.
    #[test]
    fn a_non_overlapping_pool_routes_the_gradient_to_the_argmax() {
        let mut g = graph();
        let x = param(&mut g, &[1, 6]);
        let win = g
            .add(Op::Logical(Logical::Window {
                specs: smallvec::smallvec![SlidingWindow::new(1, 3, 3)],
                x,
            }))
            .unwrap();
        // [1, 2, 3] -> max over the trailing window axis
        let pooled = fold_node_tied(&mut g, win, BinOp::Max, Some(TiePolicy::FirstWins), 2);
        let seed = ones(&mut g, &[1, 2]);
        let grads = backward_into(&mut g, &caps(), pooled, seed, &[x]).unwrap();
        let e = env(&[(x, vec![1.0, 9.0, 2.0, 4.0, 3.0, 8.0])]);
        let analytic = eval(&g, grads[0].unwrap(), &e);
        assert_eq!(analytic, vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn an_overlapping_window_matches_an_overlap_add_reference() {
        let mut g = graph();
        let x = param(&mut g, &[5]);
        let win = g
            .add(Op::Logical(Logical::Window {
                specs: smallvec::smallvec![SlidingWindow::new(0, 3, 1)],
                x,
            }))
            .unwrap();
        let seed = ones(&mut g, &[3, 3]);
        let grads = backward_into(&mut g, &caps(), win, seed, &[x]).unwrap();
        let e = env(&[(x, vec![0.0; 5])]);
        let analytic = eval(&g, grads[0].unwrap(), &e);
        // Element k appears in min(k, 2, 4-k) + 1 windows.
        assert_eq!(analytic, vec![1.0, 2.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn slice_assign_splits_the_gradient_between_base_and_value() {
        let mut g = graph();
        let base = param(&mut g, &[5]);
        let upd = param(&mut g, &[2]);
        let idx = u32_buffer(&mut g, 2);
        let y = g
            .add(Op::Logical(Logical::Scatter {
                axis: 0,
                combine: ScatterCombine::Set,
                base,
                idx,
                upd,
                unique: true,
            }))
            .unwrap();
        let seed = ones(&mut g, &[5]);
        let grads = backward_into(&mut g, &caps(), y, seed, &[base, upd]).unwrap();
        let e = env(&[
            (base, vec![0.0; 5]),
            (upd, vec![0.0; 2]),
            (idx, vec![1.0, 3.0]),
        ]);
        let d_base = eval(&g, grads[0].unwrap(), &e);
        let d_upd = eval(&g, grads[1].unwrap(), &e);
        assert_eq!(d_base, vec![1.0, 0.0, 1.0, 0.0, 1.0], "the region is zeroed");
        assert_eq!(d_upd, vec![1.0, 1.0], "the value gets its slice");
    }

    #[test]
    fn a_transpose_then_reshape_chain_matches_finite_differences() {
        let mut g = graph();
        let x = param(&mut g, &[2, 3]);
        let y = {
            let mut t = GraphTape::new(&mut g);
            let p = t.permute(x, &[1, 0]).unwrap();
            let r = t.reshape(p, &[Dim::Const(6)]).unwrap();
            t.unary(UnOp::Exp, r).unwrap()
        };
        let seed = ones(&mut g, &[6]);
        let grads = backward_into(&mut g, &caps(), y, seed, &[x]).unwrap();
        let e = env(&[(x, vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.6])]);
        check_gradients(&g, y, &[x], &grads, &e, 2e-3);
    }

    #[test]
    fn a_sliced_view_backward_matches_finite_differences() {
        let mut g = graph();
        let x = param(&mut g, &[6]);
        let y = {
            let mut t = GraphTape::new(&mut g);
            let sliced = t
                .restride(
                    &[StrideSpec::dim(0, Dim::Const(3)).with_offset(Dim::Const(2))],
                    x,
                )
                .unwrap();
            t.unary(UnOp::Sin, sliced).unwrap()
        };
        let seed = ones(&mut g, &[3]);
        let grads = backward_into(&mut g, &caps(), y, seed, &[x]).unwrap();
        let e = env(&[(x, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6])]);
        check_gradients(&g, y, &[x], &grads, &e, 2e-3);
        let analytic = eval(&g, grads[0].unwrap(), &e);
        assert_eq!(&analytic[..2], &[0.0, 0.0], "the unread prefix gets zero");
    }

    #[test]
    fn a_scatter_add_backward_matches_finite_differences() {
        let mut g = graph();
        let base = param(&mut g, &[4]);
        let upd = param(&mut g, &[3]);
        let idx = u32_buffer(&mut g, 3);
        let y = g
            .add(Op::Logical(Logical::Scatter {
                axis: 0,
                combine: ScatterCombine::Add,
                base,
                idx,
                upd,
                unique: false,
            }))
            .unwrap();
        let sq = {
            let mut t = GraphTape::new(&mut g);
            t.binary(BinOp::Mul, y, y).unwrap()
        };
        let seed = ones(&mut g, &[4]);
        let grads = backward_into(&mut g, &caps(), sq, seed, &[base, upd]).unwrap();
        let e = env(&[
            (base, vec![0.5, -1.0, 2.0, 0.25]),
            (upd, vec![1.0, 2.0, 3.0]),
            (idx, vec![0.0, 2.0, 0.0]),
        ]);
        check_gradients(&g, sq, &[base, upd], &grads, &e, 3e-3);
    }

    #[test]
    fn an_f32_f16_f32_cast_round_trip_keeps_the_master_in_f32() {
        let mut g = graph();
        let x = param(&mut g, &[4]);
        let y = {
            let mut t = GraphTape::new(&mut g);
            let half = t.cast(Dtype::F16, x).unwrap();
            let scaled = t.mul_scalar(half, 2.0).unwrap();
            t.cast(Dtype::F32, scaled).unwrap()
        };
        let seed = ones(&mut g, &[4]);
        let grads = backward_into(&mut g, &caps(), y, seed, &[x]).unwrap();
        let dx = grads[0].unwrap();
        assert_eq!(g.facts(dx).dtype, Dtype::F32, "the gradient lands in f32");
        let e = env(&[(x, vec![0.5, 1.5, -2.25, 3.0])]);
        let analytic = eval(&g, dx, &e);
        for v in analytic {
            assert!((v - 2.0).abs() < 1e-2, "within f16 rounding of 2.0");
        }
    }

    #[test]
    fn a_where_cond_backward_is_elementwise_exact() {
        let mut g = graph();
        let c = param(&mut g, &[4]);
        let t_val = param(&mut g, &[4]);
        let f_val = param(&mut g, &[4]);
        let y = {
            let mut t = GraphTape::new(&mut g);
            t.select(c, t_val, f_val).unwrap()
        };
        let seed = ones(&mut g, &[4]);
        let grads = backward_into(&mut g, &caps(), y, seed, &[c, t_val, f_val]).unwrap();
        let e = env(&[
            (c, vec![1.0, 0.0, 1.0, 0.0]),
            (t_val, vec![0.0; 4]),
            (f_val, vec![0.0; 4]),
        ]);
        assert_eq!(eval(&g, grads[0].unwrap(), &e), vec![0.0; 4]);
        assert_eq!(eval(&g, grads[1].unwrap(), &e), vec![1.0, 0.0, 1.0, 0.0]);
        assert_eq!(eval(&g, grads[2].unwrap(), &e), vec![0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn a_clamp_backward_is_the_two_sided_mask() {
        let mut g = graph();
        let x = param(&mut g, &[4]);
        let y = {
            let mut t = GraphTape::new(&mut g);
            let lo = crate::tape::lit(-1.0, Dtype::F32).unwrap();
            let hi = crate::tape::lit(2.0, Dtype::F32).unwrap();
            let body = ScalarExpr::bin(
                BinOp::Min,
                ScalarExpr::bin(BinOp::Max, ScalarExpr::arg(0, Dtype::F32), lo),
                hi,
            );
            t.map(body, &[x]).unwrap()
        };
        let seed = ones(&mut g, &[4]);
        let grads = backward_into(&mut g, &caps(), y, seed, &[x]).unwrap();
        let e = env(&[(x, vec![-3.0, 0.0, 1.5, 9.0])]);
        assert_eq!(eval(&g, grads[0].unwrap(), &e), vec![0.0, 1.0, 1.0, 0.0]);
    }
}
