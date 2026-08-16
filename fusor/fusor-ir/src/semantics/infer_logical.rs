//! Total shape/dtype/numeric/persistence inference for the ten Logical nodes.
//! Never panics; every failure is an [`crate::Error`].

use crate::carrier::Carrier;
use crate::contract_spec;
use crate::dtype::{Dtype, NumericContract, Persistence};
use crate::error::{Error, Result};
use crate::facts::ValueFacts;
use crate::ir::logical::{LeafKind, Logical};
use crate::scalar::{ScalarExpr, ScalarKind};
use crate::shape::{Dim, Dims, Layout, StrideSpec, SymId};
use smallvec::SmallVec;

/// Infer the result facts of a Logical node from its operands' facts.
///
/// `numeric` is the meet of the operands' contracts, never wider: the
/// monotonicity that makes `fold_split` sound is established here.
/// `persistence` is `Persistent` only for a `Param`/`Quantized` leaf and for
/// pure views over one.
pub fn infer_logical(op: &Logical, ins: &[ValueFacts]) -> Result<ValueFacts> {
    match op {
        Logical::Leaf(kind) => infer_leaf(kind),
        Logical::Map { expr, ins: _, outs } => infer_map(expr, ins, *outs),
        Logical::Fold {
            carrier, axis, acc, ..
        } => infer_fold(carrier, *axis, *acc, ins),
        Logical::Contract {
            spec, acc, outs, ..
        } => {
            let (a, b) = two(ins, "Contract")?;
            let e = contract_spec::extents(spec, &a.shape, &b.shape)?;
            let shape = contract_spec::out_shape(spec, &e)?;
            Ok(ValueFacts {
                dtype: *acc,
                shape,
                numeric: a.numeric.meet(b.numeric),
                persistence: Persistence::Step,
                outs: *outs,
            })
        }
        Logical::Restride { specs, .. } => {
            let x = one(ins, "Restride")?;
            check_restride_specs(specs, x.rank())?;
            Ok(ValueFacts {
                dtype: x.dtype,
                shape: specs.iter().map(|s| s.size).collect(),
                numeric: x.numeric,
                persistence: x.persistence,
                outs: 1,
            })
        }
        Logical::Window { specs, .. } => {
            let x = one(ins, "Window")?;
            let (shape, _) = window_shape(specs, &x.shape)?;
            Ok(ValueFacts {
                dtype: x.dtype,
                shape,
                numeric: x.numeric,
                persistence: x.persistence,
                outs: 1,
            })
        }
        Logical::Gather { axis, .. } => {
            let (x, idx) = two(ins, "Gather")?;
            if !matches!(idx.dtype, Dtype::U32 | Dtype::I32) {
                return Err(Error::Dtype(format!(
                    "Gather indices must be U32 or I32, not {:?}",
                    idx.dtype
                )));
            }
            if idx.rank() != 1 {
                return Err(Error::Shape(format!(
                    "Gather indices must be rank 1, not rank {}",
                    idx.rank()
                )));
            }
            let axis = *axis as usize;
            if axis >= x.rank() {
                return Err(Error::Shape(format!(
                    "Gather axis {axis} out of range for rank {}",
                    x.rank()
                )));
            }
            let mut shape = x.shape.clone();
            shape[axis] = idx.shape[0];
            Ok(ValueFacts {
                dtype: x.dtype,
                shape,
                numeric: x.numeric,
                persistence: Persistence::Step,
                outs: 1,
            })
        }
        Logical::Scatter { axis, .. } => {
            let (base, idx, upd) = three(ins, "Scatter")?;
            if !matches!(idx.dtype, Dtype::U32 | Dtype::I32) {
                return Err(Error::Dtype(format!(
                    "Scatter indices must be U32 or I32, not {:?}",
                    idx.dtype
                )));
            }
            if idx.rank() != 1 {
                return Err(Error::Shape(format!(
                    "Scatter indices must be rank 1, not rank {}",
                    idx.rank()
                )));
            }
            let axis = *axis as usize;
            if axis >= base.rank() {
                return Err(Error::Shape(format!(
                    "Scatter axis {axis} out of range for rank {}",
                    base.rank()
                )));
            }
            if upd.rank() != base.rank() {
                return Err(Error::Shape(format!(
                    "Scatter update rank {} does not match base rank {}",
                    upd.rank(),
                    base.rank()
                )));
            }
            if !upd.shape[axis].known_eq(idx.shape[0]) {
                return Err(Error::Shape(format!(
                    "Scatter update axis {axis} is {} but there are {} indices",
                    upd.shape[axis], idx.shape[0]
                )));
            }
            for (i, (u, b)) in upd.shape.iter().zip(&base.shape).enumerate() {
                if i != axis && !u.known_eq(*b) {
                    return Err(Error::Shape(format!(
                        "Scatter update axis {i} is {u} but the base is {b}"
                    )));
                }
            }
            // The output *is* the base: a scatter writes through it.
            Ok(base.clone())
        }
        Logical::Dequant { fmt, .. } => {
            let x = one(ins, "Dequant")?;
            if x.dtype != Dtype::Q(*fmt) {
                return Err(Error::Dtype(format!(
                    "Dequant of {:?} applied to a {:?} value",
                    fmt, x.dtype
                )));
            }
            Ok(ValueFacts {
                dtype: Dtype::F32,
                shape: x.shape.clone(),
                numeric: x.numeric,
                persistence: x.persistence,
                outs: 1,
            })
        }
        Logical::Project { slot, .. } => {
            let x = one(ins, "Project")?;
            if *slot >= x.outs {
                return Err(Error::Shape(format!(
                    "Project slot {slot} out of range: the producer has {} results",
                    x.outs
                )));
            }
            Ok(ValueFacts {
                dtype: x.dtype,
                shape: x.shape.clone(),
                numeric: x.numeric,
                persistence: x.persistence,
                outs: 1,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Per-node rules
// ---------------------------------------------------------------------------

fn infer_leaf(kind: &LeafKind) -> Result<ValueFacts> {
    Ok(match kind {
        LeafKind::Buffer { dtype, shape, .. } => ValueFacts {
            dtype: *dtype,
            shape: shape.clone(),
            numeric: NumericContract::RELAXED,
            persistence: Persistence::Step,
            outs: 1,
        },
        LeafKind::Param { dtype, shape, .. } => ValueFacts {
            dtype: *dtype,
            shape: shape.clone(),
            numeric: NumericContract::RELAXED,
            persistence: Persistence::Persistent,
            outs: 1,
        },
        LeafKind::Const { value, shape } => ValueFacts {
            dtype: value.dtype(),
            shape: shape.clone(),
            numeric: NumericContract::RELAXED,
            persistence: Persistence::Step,
            outs: 1,
        },
        // A runtime scalar read from the uniform block: rank 0, never baked
        // into a kernel key.
        LeafKind::Uniform { dtype, .. } => ValueFacts {
            dtype: *dtype,
            shape: Dims::new(),
            numeric: NumericContract::RELAXED,
            persistence: Persistence::Step,
            outs: 1,
        },
        LeafKind::Quantized { fmt, shape, .. } => ValueFacts {
            dtype: Dtype::Q(*fmt),
            shape: shape.iter().copied().collect(),
            numeric: NumericContract::RELAXED,
            persistence: Persistence::Persistent,
            outs: 1,
        },
    })
}

fn infer_map(expr: &ScalarExpr, ins: &[ValueFacts], outs: u8) -> Result<ValueFacts> {
    // **No implicit broadcasting**: every operand carries the output shape.
    // The frontend emits `Restride { multiplier: 0 }` instead.
    if let Some(first) = ins.first() {
        for other in &ins[1..] {
            let same = other.rank() == first.rank()
                && other
                    .shape
                    .iter()
                    .zip(&first.shape)
                    .all(|(a, b)| a.known_eq(*b));
            if !same {
                return Err(Error::Shape(format!(
                    "Map operands must have identical shape; the frontend emits \
                     Restride{{multiplier:0}} ({:?} vs {:?})",
                    first.shape, other.shape
                )));
            }
        }
    } else if !expr_is_closed(expr) {
        return Err(Error::Shape(
            "a Map with no operands must be closed over Lit/Uniform".into(),
        ));
    }

    check_arg_dtypes(expr, ins)?;

    let numeric = ins
        .iter()
        .map(|f| f.numeric)
        .reduce(NumericContract::meet)
        .unwrap_or(NumericContract::RELAXED);

    Ok(ValueFacts {
        dtype: expr.dtype(),
        shape: ins.first().map(|f| f.shape.clone()).unwrap_or_default(),
        numeric,
        persistence: Persistence::Step,
        outs,
    })
}

/// A fold's result: the operand shape minus the reduced axis, with the
/// carrier's lane count appended when it is more than one. That appended axis
/// is how a multi-slot accumulator is read back — slot `i` is an ordinary
/// `Restride` of it, so no new node kind appears.
///
/// Every operand must have the same shape: the lift reads them all at one
/// coordinate, exactly as a `Map` body does.
fn infer_fold(carrier: &Carrier, axis: u32, acc: Dtype, ins: &[ValueFacts]) -> Result<ValueFacts> {
    let x = ins
        .first()
        .ok_or_else(|| Error::Shape("Fold takes at least one operand".into()))?;
    let axis = axis as usize;
    if axis >= x.rank() {
        return Err(Error::Shape(format!(
            "Fold axis {axis} out of range for rank {}",
            x.rank()
        )));
    }
    for (i, f) in ins.iter().enumerate().skip(1) {
        if f.shape.len() != x.shape.len()
            || !f.shape.iter().zip(&x.shape).all(|(a, b)| a.known_eq(*b))
        {
            return Err(Error::Shape(format!(
                "Fold operand {i} has shape {:?}, expected {:?}",
                f.shape, x.shape
            )));
        }
    }
    crate::verify_l0::check_carrier(carrier, acc)?;

    let mut shape: Dims = x.shape.clone();
    shape.remove(axis);
    if let Some(d) = carrier
        .out_dim()
        .ok_or_else(|| Error::Shape("a multi-slot carrier needs a constant Vector extent".into()))?
    {
        shape.push(d);
    }
    Ok(ValueFacts {
        dtype: acc,
        shape,
        numeric: ins
            .iter()
            .map(|f| f.numeric)
            .reduce(NumericContract::meet)
            .unwrap_or(NumericContract::RELAXED),
        persistence: Persistence::Step,
        outs: 1,
    })
}

// ---------------------------------------------------------------------------
// Restride
// ---------------------------------------------------------------------------

/// A spec references its `input_dim` when it is not a pure stride-0 axis at
/// offset 0. The reference's `Layout::restride` reads `strides[input_dim]`
/// for the offset term regardless of `multiplier`, so a broadcast spec with
/// a nonzero offset still names an input dim.
pub fn spec_reads_input_dim(s: &StrideSpec) -> bool {
    s.multiplier != 0 || !s.offset.known_eq(Dim::Const(0))
}

fn check_restride_specs(specs: &[StrideSpec], in_rank: usize) -> Result<()> {
    for (i, s) in specs.iter().enumerate() {
        if spec_reads_input_dim(s) && s.input_dim as usize >= in_rank {
            return Err(Error::Shape(format!(
                "Restride spec {i} names input dim {} of a rank-{in_rank} value",
                s.input_dim
            )));
        }
    }
    Ok(())
}

/// The reference's `types/src/layout.rs::Layout::restride`, lifted to
/// [`Dim`]: `out_shape[i] = spec.size`,
/// `out_stride[i] = if multiplier == 0 { 0 } else { in_stride[input_dim] *
/// multiplier }`, and the offset gains `sum(offset * in_stride[input_dim])`.
/// Composition is **relative to the current strides**, which is what makes a
/// view survive an upstream layout rewrite.
///
/// A product or sum that is not decidable over `Const` dims becomes the
/// symbolic stride placeholder `Dim::Sym(SymId(u32::MAX))`, the same
/// convention `Layout::row_major_strides` already uses.
pub fn restride_layout(input: &Layout, specs: &[StrideSpec]) -> Result<Layout> {
    check_restride_specs(specs, input.rank())?;
    let in_strides = input.strides();

    let shape: Dims = specs.iter().map(|s| s.size).collect();
    let strides: SmallVec<[Dim; 6]> = specs
        .iter()
        .map(|s| {
            if s.multiplier == 0 {
                Dim::Const(0)
            } else {
                dim_mul(in_strides[s.input_dim as usize], s.multiplier as u64)
            }
        })
        .collect();

    let mut offset = input.offset();
    for s in specs {
        if s.offset.known_eq(Dim::Const(0)) {
            continue;
        }
        let stride = in_strides[s.input_dim as usize];
        offset = dim_add(offset, dim_mul_dim(s.offset, stride));
    }
    Layout::from_parts(offset, &shape, &strides)
}

/// The placeholder a non-decidable stride/offset carries.
const OPAQUE: Dim = Dim::Sym(SymId(u32::MAX));

fn dim_mul(a: Dim, b: u64) -> Dim {
    match a {
        Dim::Const(v) => v.checked_mul(b).map_or(OPAQUE, Dim::Const),
        Dim::Sym(_) if b == 1 => a,
        Dim::Sym(_) => OPAQUE,
    }
}

fn dim_mul_dim(a: Dim, b: Dim) -> Dim {
    match (a, b) {
        (Dim::Const(x), Dim::Const(y)) => x.checked_mul(y).map_or(OPAQUE, Dim::Const),
        (Dim::Const(0), _) | (_, Dim::Const(0)) => Dim::Const(0),
        (Dim::Const(1), other) | (other, Dim::Const(1)) => other,
        _ => OPAQUE,
    }
}

fn dim_add(a: Dim, b: Dim) -> Dim {
    match (a, b) {
        (Dim::Const(x), Dim::Const(y)) => x.checked_add(y).map_or(OPAQUE, Dim::Const),
        (Dim::Const(0), other) | (other, Dim::Const(0)) => other,
        _ => OPAQUE,
    }
}

// ---------------------------------------------------------------------------
// Window
// ---------------------------------------------------------------------------

/// `types/src/layout.rs::Layout::sliding_window`, lifted to [`Dim`].
///
/// Returns the output shape plus `true` when any windowed axis was symbolic.
/// A symbolic axis does **not** mint a fresh extent: the output dim stays the
/// input `Sym` (refined at dispatch) and the node carries a
/// `BoundsProof::RuntimeMask` obligation, which is what keeps a symbolic
/// sequence length from forcing a recompile.
pub fn window_shape(
    specs: &[crate::shape::SlidingWindow],
    in_shape: &[Dim],
) -> Result<(Dims, bool)> {
    let mut sorted: SmallVec<[crate::shape::SlidingWindow; 3]> = specs.iter().copied().collect();
    sorted.sort_by_key(|w| w.axis);
    for pair in sorted.windows(2) {
        if pair[0].axis == pair[1].axis {
            return Err(Error::Shape(format!(
                "Window axes must be unique; axis {} appears twice",
                pair[0].axis
            )));
        }
    }
    for w in &sorted {
        if w.axis as usize >= in_shape.len() {
            return Err(Error::Shape(format!(
                "Window axis {} out of range for rank {}",
                w.axis,
                in_shape.len()
            )));
        }
        if w.window == 0 || w.step == 0 {
            return Err(Error::Shape(
                "Window size and step must both be nonzero".into(),
            ));
        }
    }

    let mut runtime_mask = false;
    let mut shape: Dims = in_shape.iter().copied().collect();
    for w in &sorted {
        let axis = w.axis as usize;
        shape[axis] = match in_shape[axis] {
            Dim::Const(d) => {
                if d < w.window as u64 {
                    return Err(Error::Shape(format!(
                        "Window of {} does not fit axis {axis} of extent {d}",
                        w.window
                    )));
                }
                Dim::Const((d - w.window as u64) / w.step as u64 + 1)
            }
            sym => {
                runtime_mask = true;
                sym
            }
        };
    }
    for w in &sorted {
        shape.push(Dim::Const(w.window as u64));
    }
    Ok((shape, runtime_mask))
}

// ---------------------------------------------------------------------------
// Scalar-expression helpers
// ---------------------------------------------------------------------------

/// Every `Arg(i)` in `expr` names an operand whose dtype matches the leaf's.
fn check_arg_dtypes(expr: &ScalarExpr, ins: &[ValueFacts]) -> Result<()> {
    let mut err = None;
    walk_expr(expr, &mut |e| {
        if err.is_some() {
            return;
        }
        if let ScalarKind::Arg(i) = e.kind() {
            match ins.get(*i as usize) {
                None => {
                    err = Some(Error::Shape(format!(
                        "Map body reads Arg({i}) but only {} operands were supplied",
                        ins.len()
                    )));
                }
                Some(f) if f.dtype != e.dtype() => {
                    err = Some(Error::Dtype(format!(
                        "Map body reads Arg({i}) as {:?} but the operand is {:?}",
                        e.dtype(),
                        f.dtype
                    )));
                }
                Some(_) => {}
            }
        }
    });
    match err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// True when `expr` reads nothing outside `Lit`/`Uniform` — the only case in
/// which a zero-operand `Map` is meaningful.
fn expr_is_closed(expr: &ScalarExpr) -> bool {
    let mut closed = true;
    walk_expr(expr, &mut |e| {
        if matches!(e.kind(), ScalarKind::Arg(_) | ScalarKind::IndexOf(_)) {
            closed = false;
        }
    });
    closed
}

/// Pre-order walk over a hash-consed scalar tree. Shared subtrees are
/// revisited; callers that must count once memoize on
/// [`ScalarExpr::structural_hash`].
pub(crate) fn walk_expr(e: &ScalarExpr, f: &mut impl FnMut(&ScalarExpr)) {
    f(e);
    match e.kind() {
        ScalarKind::Arg(_)
        | ScalarKind::Lit(_)
        | ScalarKind::Uniform(_)
        | ScalarKind::IndexOf(_) => {}
        ScalarKind::Un { x, .. }
        | ScalarKind::Cast { x, .. }
        | ScalarKind::Bitcast { x, .. }
        | ScalarKind::Round { x, .. }
        | ScalarKind::Splat { x, .. } => walk_expr(x, f),
        ScalarKind::Bin { a, b, .. } | ScalarKind::Cmp { a, b, .. } | ScalarKind::Dot { a, b } => {
            walk_expr(a, f);
            walk_expr(b, f);
        }
        ScalarKind::Select { c, t, f: e_f } => {
            walk_expr(c, f);
            walk_expr(t, f);
            walk_expr(e_f, f);
        }
    }
}

// ---------------------------------------------------------------------------
// Arity helpers — every access is length-checked, so inference is total.
// ---------------------------------------------------------------------------

fn one<'a>(ins: &'a [ValueFacts], what: &str) -> Result<&'a ValueFacts> {
    ins.first()
        .ok_or_else(|| Error::Shape(format!("{what} needs 1 operand, got {}", ins.len())))
}

fn two<'a>(ins: &'a [ValueFacts], what: &str) -> Result<(&'a ValueFacts, &'a ValueFacts)> {
    if ins.len() < 2 {
        return Err(Error::Shape(format!(
            "{what} needs 2 operands, got {}",
            ins.len()
        )));
    }
    Ok((&ins[0], &ins[1]))
}

fn three<'a>(
    ins: &'a [ValueFacts],
    what: &str,
) -> Result<(&'a ValueFacts, &'a ValueFacts, &'a ValueFacts)> {
    if ins.len() < 3 {
        return Err(Error::Shape(format!(
            "{what} needs 3 operands, got {}",
            ins.len()
        )));
    }
    Ok((&ins[0], &ins[1], &ins[2]))
}
