//! Total shape/dtype/numeric/persistence inference for the ten Logical nodes.
//! Never panics; every failure is an [`crate::Error`].

use crate::carrier::Carrier;
use crate::contract_spec;
use crate::dtype::{Dtype, Persistence};
use crate::error::{Error, Result};
use crate::facts::ValueFacts;
use crate::ir::logical::{LeafKind, Logical};
use crate::scalar::{ScalarExpr, ScalarKind};
use crate::shape::{Dim, Dims, Layout, StrideSpec};
use smallvec::SmallVec;

/// Infer the result facts of a Logical node from its operands' facts.
/// `numeric` is the meet of the operands' contracts, never wider, which is
/// what makes `fold_split` sound.
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
                numeric: a.numeric.meet(b.numeric),
                outs: *outs,
                ..ValueFacts::step(*acc, shape, &[])
            })
        }
        Logical::Restride { specs, .. } => {
            let x = one(ins, "Restride")?;
            check_restride_specs(specs, x.rank())?;
            Ok(x.view(specs.iter().map(|s| s.size).collect()))
        }
        Logical::Window { specs, .. } => {
            let x = one(ins, "Window")?;
            Ok(x.view(window_shape(specs, &x.shape)?.0))
        }
        Logical::Gather { axis, .. } => {
            let (x, idx) = two(ins, "Gather")?;
            let axis = check_indices("Gather", idx, *axis, x.rank())?;
            let mut shape = x.shape.clone();
            shape[axis] = idx.shape[0];
            Ok(ValueFacts {
                persistence: Persistence::Step,
                ..x.view(shape)
            })
        }
        Logical::Scatter { axis, .. } => {
            let (base, idx, upd) = three(ins, "Scatter")?;
            let axis = check_indices("Scatter", idx, *axis, base.rank())?;
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
                ..x.view(x.shape.clone())
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
            Ok(x.view(x.shape.clone()))
        }
    }
}

/// An index operand is a rank-1 `U32`/`I32` vector and `axis` names an axis
/// of the indexed value; returns `axis`.
fn check_indices(what: &str, idx: &ValueFacts, axis: u32, rank: usize) -> Result<usize> {
    if !matches!(idx.dtype, Dtype::U32 | Dtype::I32) {
        return Err(Error::Dtype(format!(
            "{what} indices must be U32 or I32, not {:?}",
            idx.dtype
        )));
    }
    if idx.rank() != 1 {
        return Err(Error::Shape(format!(
            "{what} indices must be rank 1, not rank {}",
            idx.rank()
        )));
    }
    let axis = axis as usize;
    if axis >= rank {
        return Err(Error::Shape(format!(
            "{what} axis {axis} out of range for rank {rank}"
        )));
    }
    Ok(axis)
}

fn infer_leaf(kind: &LeafKind) -> Result<ValueFacts> {
    let persistent = |f: ValueFacts| ValueFacts {
        persistence: Persistence::Persistent,
        ..f
    };
    Ok(match kind {
        LeafKind::Buffer { dtype, shape, .. } => ValueFacts::new(*dtype, shape.iter().copied()),
        LeafKind::Param { dtype, shape, .. } => {
            persistent(ValueFacts::new(*dtype, shape.iter().copied()))
        }
        LeafKind::Const { value, shape } => ValueFacts::new(value.dtype(), shape.iter().copied()),
        // A runtime scalar read from the uniform block: rank 0, never baked
        // into a kernel key.
        LeafKind::Uniform { dtype, .. } => ValueFacts::new(*dtype, []),
        LeafKind::Quantized { fmt, shape, .. } => {
            persistent(ValueFacts::new(Dtype::Q(*fmt), shape.iter().copied()))
        }
    })
}

fn infer_map(expr: &ScalarExpr, ins: &[ValueFacts], outs: u8) -> Result<ValueFacts> {
    // No implicit broadcasting: every operand carries the output shape.
    if let Some(first) = ins.first() {
        for other in &ins[1..] {
            if other.shape != first.shape {
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

    let shape = ins.first().map(|f| f.shape.clone()).unwrap_or_default();
    Ok(ValueFacts {
        outs,
        ..ValueFacts::step(expr.dtype(), shape, ins)
    })
}

/// A fold's result: the operand shape minus the reduced axis, plus the
/// carrier's lane axis when wider than one (slots are read back by
/// `Restride`). Every operand has the same shape, as for a `Map`.
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
        if f.shape != x.shape {
            return Err(Error::Shape(format!(
                "Fold operand {i} has shape {:?}, expected {:?}",
                f.shape, x.shape
            )));
        }
    }
    let mut unbound = None;
    for lift in &carrier.lift {
        lift.walk(&mut |expr| {
            if let ScalarKind::Arg(index) = expr.kind()
                && *index as usize >= ins.len()
            {
                unbound = Some(*index);
            }
        });
    }
    if let Some(index) = unbound {
        return Err(Error::Shape(format!(
            "Fold lift reads Arg({index}) but only {} operands were supplied",
            ins.len()
        )));
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
    Ok(ValueFacts::step(acc, shape, ins))
}

/// A spec references its `input_dim` unless it is a stride-0 axis at offset
/// 0: the offset term reads `strides[input_dim]` whatever the multiplier.
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

/// `Layout::restride` lifted to [`Dim`]. Composition is relative to the
/// current strides, so a view survives an upstream layout rewrite; symbolic
/// terms become derived symbols evaluated at dispatch.
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
                in_strides[s.input_dim as usize] * Dim::Const(s.multiplier as u64)
            }
        })
        .collect();

    let mut offset = input.offset();
    for s in specs {
        if s.offset.known_eq(Dim::Const(0)) {
            continue;
        }
        let stride = in_strides[s.input_dim as usize];
        offset = offset + s.offset * stride;
    }
    Layout::from_parts(offset, &shape, &strides)
}

/// `Layout::sliding_window` lifted to [`Dim`], plus `true` when a windowed
/// axis is symbolic: it keeps the input `Sym` under a runtime mask rather
/// than minting a fresh extent, so it never forces a recompile.
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

/// Every `Arg(i)` in `expr` names an operand whose dtype matches the leaf's.
fn check_arg_dtypes(expr: &ScalarExpr, ins: &[ValueFacts]) -> Result<()> {
    let mut err = None;
    expr.walk(&mut |e| {
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

/// True when `expr` reads nothing outside `Lit`/`Uniform`.
fn expr_is_closed(expr: &ScalarExpr) -> bool {
    let mut closed = true;
    expr.walk(&mut |e| {
        if matches!(e.kind(), ScalarKind::Arg(_) | ScalarKind::IndexOf(_)) {
            closed = false;
        }
    });
    closed
}

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
