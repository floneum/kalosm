//! Total shape/dtype/numeric/persistence inference for the ten L0 nodes.
//! Never panics; every failure is an [`crate::Error`].

use crate::carrier::Carrier;
use crate::contract_spec;
use crate::dtype::{Dtype, NumericContract, Persistence};
use crate::error::{Error, Result};
use crate::facts::ValueFacts;
use crate::ir::level0::{L0, LeafKind};
use crate::scalar::{ScalarExpr, ScalarKind};
use crate::shape::{Dim, Dims, Layout, StrideSpec, SymId};
use smallvec::SmallVec;

/// Infer the result facts of an L0 node from its operands' facts.
///
/// `numeric` is the meet of the operands' contracts, never wider: the
/// monotonicity that makes `fold_split` sound is established here.
/// `persistence` is `Persistent` only for a `Param`/`Quantized` leaf and for
/// pure views over one.
pub fn infer_l0(op: &L0, ins: &[ValueFacts]) -> Result<ValueFacts> {
    match op {
        L0::Leaf(kind) => infer_leaf(kind),
        L0::Map { expr, ins: _, outs } => infer_map(expr, ins, *outs),
        L0::Fold {
            carrier, axis, acc, ..
        } => infer_fold(carrier, *axis, *acc, ins),
        L0::Contract {
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
        L0::Restride { specs, .. } => {
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
        L0::Window { specs, .. } => {
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
        L0::Gather { axis, .. } => {
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
        L0::Scatter { axis, .. } => {
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
        L0::Dequant { fmt, .. } => {
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
        L0::Project { slot, .. } => {
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
/// carrier's lane count appended when it is more than one. Slot `i` of a
/// multi-slot accumulator is read back as a `Restride` of that axis.
///
/// Every operand must have the same shape; the lift reads them all at one
/// coordinate.
fn infer_fold(
    carrier: &Carrier,
    axis: u32,
    acc: Dtype,
    ins: &[ValueFacts],
) -> Result<ValueFacts> {
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
    if let Some(d) = carrier.out_dim().ok_or_else(|| {
        Error::Shape("a multi-slot carrier needs a constant Vector extent".into())
    })? {
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

/// A spec references its `input_dim` when it is not a pure stride-0 axis at
/// offset 0. `Layout::restride` reads `strides[input_dim]` for the offset
/// term regardless of `multiplier`, so a broadcast spec with a nonzero
/// offset still names an input dim.
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

/// Restride a layout over [`Dim`]: `out_shape[i] = spec.size`,
/// `out_stride[i] = if multiplier == 0 { 0 } else { in_stride[input_dim] *
/// multiplier }`, and the offset gains `sum(offset * in_stride[input_dim])`.
/// Composition is relative to the current strides.
///
/// A product or sum that is not decidable over `Const` dims becomes the
/// symbolic stride placeholder `Dim::Sym(SymId(u32::MAX))`, the same
/// convention `Layout::row_major_strides` uses.
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

/// The sliding-window output shape over [`Dim`].
///
/// Returns the output shape plus `true` when any windowed axis was symbolic.
/// A symbolic axis mints no fresh extent: the output dim stays the input
/// `Sym`, refined at dispatch, and the node carries a
/// `BoundsProof::RuntimeMask` obligation.
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
    e.for_each_child(|c| walk_expr(c, f));
}

/// Length-checked operand access, so inference is total.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::Splat;
    use crate::egraph::Id;
    use crate::carrier::{ArgRemap, SlotTy};
    use crate::ir::level0::{EinSpec, Label, ScatterCombine, TiePolicy};

    fn binop(op: BinOp) -> Carrier {
        Carrier::binop(op, Carrier::binop_identity(op, Dtype::F32).unwrap(), Dtype::F32)
    }
    use crate::scalar::BinOp;
    use crate::shape::{BoundsProof, SlidingWindow, broadcast_specs};
    use smallvec::smallvec;

    fn f32s(shape: &[u64]) -> ValueFacts {
        ValueFacts::new(Dtype::F32, shape.iter().map(|&d| Dim::Const(d)))
    }
    fn u32s(shape: &[u64]) -> ValueFacts {
        ValueFacts::new(Dtype::U32, shape.iter().map(|&d| Dim::Const(d)))
    }
    fn dims(v: &[u64]) -> Dims {
        v.iter().map(|&d| Dim::Const(d)).collect()
    }

    #[test]
    fn broadcast_specs_are_right_aligned() {
        let src = dims(&[3, 1, 5]);
        let dst = dims(&[2, 3, 4, 5]);
        let specs = broadcast_specs(&src, &dst).unwrap();
        assert_eq!(
            &specs[..],
            &[
                StrideSpec::broadcast(Dim::Const(2)),
                StrideSpec::dim(0, Dim::Const(3)),
                StrideSpec::broadcast(Dim::Const(4)),
                StrideSpec::dim(2, Dim::Const(5)),
            ]
        );
        // An unconsumed source dim is an error.
        assert!(broadcast_specs(&dims(&[7, 3]), &dims(&[2, 3])).is_err());
    }

    #[test]
    fn map_requires_identical_operand_shapes() {
        let expr = ScalarExpr::bin(
            BinOp::Add,
            ScalarExpr::arg(0, Dtype::F32),
            ScalarExpr::arg(1, Dtype::F32),
        );
        let op = L0::Map {
            expr,
            ins: smallvec![Id(0), Id(1)],
            outs: 1,
        };
        assert!(matches!(
            infer_l0(&op, &[f32s(&[4, 8]), f32s(&[8])]),
            Err(Error::Shape(_))
        ));
        let ok = infer_l0(&op, &[f32s(&[4, 8]), f32s(&[4, 8])]).unwrap();
        assert_eq!(&ok.shape[..], &dims(&[4, 8])[..]);
        assert_eq!(ok.dtype, Dtype::F32);
    }

    #[test]
    fn map_checks_arg_dtypes_and_closedness() {
        let expr = ScalarExpr::arg(0, Dtype::F32);
        let op = L0::Map {
            expr,
            ins: smallvec![Id(0)],
            outs: 1,
        };
        assert!(matches!(infer_l0(&op, &[u32s(&[4])]), Err(Error::Dtype(_))));

        // Zero operands: legal only for a closed expression.
        let closed = L0::Map {
            expr: ScalarExpr::lit(Splat::F32(1.0)),
            ins: smallvec![],
            outs: 1,
        };
        let facts = infer_l0(&closed, &[]).unwrap();
        assert_eq!(facts.rank(), 0);

        let open = L0::Map {
            expr: ScalarExpr::arg(0, Dtype::F32),
            ins: smallvec![],
            outs: 1,
        };
        assert!(infer_l0(&open, &[]).is_err());
    }

    #[test]
    fn map_numeric_is_the_meet() {
        let mut strict = f32s(&[4]);
        strict.numeric = NumericContract::STRICT;
        let op = L0::Map {
            expr: ScalarExpr::bin(
                BinOp::Add,
                ScalarExpr::arg(0, Dtype::F32),
                ScalarExpr::arg(1, Dtype::F32),
            ),
            ins: smallvec![Id(0), Id(1)],
            outs: 1,
        };
        let facts = infer_l0(&op, &[strict, f32s(&[4])]).unwrap();
        assert!(!facts.numeric.reassoc);
        assert!(!facts.numeric.contract);
    }

    #[test]
    fn contract_infers_the_output_shape() {
        let spec = EinSpec {
            a: smallvec![Label(b'b'), Label(b'i'), Label(b'k')],
            b: smallvec![Label(b'b'), Label(b'j'), Label(b'k')],
            out: smallvec![Label(b'b'), Label(b'i'), Label(b'j')],
        };
        let op = L0::Contract {
            spec,
            acc: Dtype::F32,
            a: Id(0),
            b: Id(1),
            outs: 1,
        };
        let facts = infer_l0(&op, &[f32s(&[2, 3, 4]), f32s(&[2, 5, 4])]).unwrap();
        assert_eq!(&facts.shape[..], &dims(&[2, 3, 5])[..]);
    }

    #[test]
    fn restride_composes_relative_to_current_strides() {
        let input = Layout::contiguous(&dims(&[2, 3, 4]));
        assert_eq!(input.strides(), &dims(&[12, 4, 1])[..]);
        let specs = [
            StrideSpec::dim(1, Dim::Const(3)),
            StrideSpec::dim_with(2, Dim::Const(2), 2),
        ];
        let out = restride_layout(&input, &specs).unwrap();
        assert_eq!(out.shape(), &dims(&[3, 2])[..]);
        assert_eq!(out.strides(), &dims(&[4, 2])[..]);

        let op = L0::Restride {
            specs: specs.iter().copied().collect(),
            bounds: BoundsProof::Static,
            x: Id(0),
        };
        let facts = infer_l0(&op, &[f32s(&[2, 3, 4])]).unwrap();
        assert_eq!(&facts.shape[..], &dims(&[3, 2])[..]);
    }

    #[test]
    fn restride_rejects_an_out_of_range_input_dim() {
        let op = L0::Restride {
            specs: smallvec![StrideSpec::dim(9, Dim::Const(3))],
            bounds: BoundsProof::Static,
            x: Id(0),
        };
        assert!(matches!(
            infer_l0(&op, &[f32s(&[2, 3])]),
            Err(Error::Shape(_))
        ));
        // A pure broadcast spec names no input dim, so it is fine.
        let bcast = L0::Restride {
            specs: smallvec![StrideSpec::broadcast(Dim::Const(3))],
            bounds: BoundsProof::Static,
            x: Id(0),
        };
        assert!(infer_l0(&bcast, &[f32s(&[])]).is_ok());
    }

    #[test]
    fn window_shapes() {
        let x = f32s(&[1, 24, 768]);
        let non_overlapping = SlidingWindow::new(2, 4, 4);
        assert!(non_overlapping.is_non_overlapping());
        let op = L0::Window {
            specs: smallvec![non_overlapping],
            x: Id(0),
        };
        let facts = infer_l0(&op, std::slice::from_ref(&x)).unwrap();
        assert_eq!(&facts.shape[..], &dims(&[1, 24, 192, 4])[..]);

        let overlapping = SlidingWindow::new(2, 4, 2);
        assert!(!overlapping.is_non_overlapping());
        let op = L0::Window {
            specs: smallvec![overlapping],
            x: Id(0),
        };
        let facts = infer_l0(&op, &[x]).unwrap();
        assert_eq!(&facts.shape[..], &dims(&[1, 24, 383, 4])[..]);
    }

    #[test]
    fn window_over_a_symbolic_axis_keeps_the_sym() {
        let x = ValueFacts::new(Dtype::F32, [Dim::Const(4), Dim::Sym(SymId(1))]);
        let (shape, runtime) = window_shape(&[SlidingWindow::new(1, 4, 4)], &x.shape).unwrap();
        assert!(runtime);
        assert_eq!(
            &shape[..],
            &[Dim::Const(4), Dim::Sym(SymId(1)), Dim::Const(4)]
        );
    }

    #[test]
    fn window_rejects_duplicate_axes_and_overlong_windows() {
        let x = f32s(&[8]);
        assert!(
            window_shape(
                &[SlidingWindow::new(0, 2, 1), SlidingWindow::new(0, 3, 1)],
                &x.shape
            )
            .is_err()
        );
        assert!(window_shape(&[SlidingWindow::new(0, 9, 1)], &x.shape).is_err());
        assert!(window_shape(&[SlidingWindow::new(3, 2, 1)], &x.shape).is_err());
    }

    /// A carrier's lanes become a trailing axis of the output shape: three
    /// scalar slots and one `Vector(3)` slot append the same axis.
    #[test]
    fn fold_carrier_appends_lanes() {
        let three = binop(BinOp::Add)
            .tuple(&binop(BinOp::Max), &ArgRemap::identity(1))
            .carrier
            .tuple(&binop(BinOp::Mul), &ArgRemap::identity(1))
            .carrier;
        assert_eq!(three.width(), 3);
        let op = L0::Fold {
            carrier: three,
            axis: 1,
            acc: Dtype::F32,
            ins: smallvec![Id(0)],
        };
        let facts = infer_l0(&op, &[f32s(&[4, 8, 16])]).unwrap();
        assert_eq!(&facts.shape[..], &dims(&[4, 16, 3])[..]);
    }

    #[test]
    fn fold_scalar_and_vector_carriers() {
        let scalar = L0::Fold {
            carrier: binop(BinOp::Add),
            axis: 0,
            acc: Dtype::F32,
            ins: smallvec![Id(0)],
        };
        assert_eq!(
            &infer_l0(&scalar, &[f32s(&[6, 7])]).unwrap().shape[..],
            &dims(&[7])[..]
        );

        let vector = L0::Fold {
            carrier: binop(BinOp::Add).promote(Dim::Const(64)).unwrap(),
            axis: 0,
            acc: Dtype::F32,
            ins: smallvec![Id(0)],
        };
        assert_eq!(
            &infer_l0(&vector, &[f32s(&[6, 7])]).unwrap().shape[..],
            &dims(&[7, 64])[..]
        );

        // A symbolic slot extent is refused, not guessed: a private array of
        // unknown width is allocatable on neither backend.
        let symbolic = L0::Fold {
            carrier: Carrier {
                slots: smallvec![SlotTy::Vector(Dim::Sym(SymId(9)))],
                ..binop(BinOp::Add)
            },
            axis: 0,
            acc: Dtype::F32,
            ins: smallvec![Id(0)],
        };
        assert!(infer_l0(&symbolic, &[f32s(&[6, 7])]).is_err());

        let out_of_range = L0::Fold {
            carrier: binop(BinOp::Add),
            axis: 4,
            acc: Dtype::F32,
            ins: smallvec![Id(0)],
        };
        assert!(infer_l0(&out_of_range, &[f32s(&[6, 7])]).is_err());
    }

    /// Every operand is read at one coordinate, exactly as a `Map` body is, so
    /// a fold whose operands disagree in shape is a typed error.
    #[test]
    fn a_multi_operand_fold_requires_agreeing_shapes() {
        let op = L0::Fold {
            carrier: binop(BinOp::Add).with_lift([ScalarExpr::bin(
                BinOp::Mul,
                ScalarExpr::arg(0, Dtype::F32),
                ScalarExpr::arg(1, Dtype::F32),
            )]),
            axis: 1,
            acc: Dtype::F32,
            ins: smallvec![Id(0), Id(1)],
        };
        assert_eq!(
            &infer_l0(&op, &[f32s(&[4, 8]), f32s(&[4, 8])]).unwrap().shape[..],
            &dims(&[4])[..]
        );
        assert!(infer_l0(&op, &[f32s(&[4, 8]), f32s(&[4, 9])]).is_err());
    }

    #[test]
    fn gather_replaces_the_axis_extent() {
        let op = L0::Gather {
            axis: 0,
            x: Id(0),
            idx: Id(1),
        };
        let facts = infer_l0(&op, &[f32s(&[1024, 24]), u32s(&[300])]).unwrap();
        assert_eq!(&facts.shape[..], &dims(&[300, 24])[..]);

        assert!(infer_l0(&op, &[f32s(&[1024, 24]), f32s(&[300])]).is_err());
        assert!(infer_l0(&op, &[f32s(&[1024, 24]), u32s(&[3, 4])]).is_err());
    }

    #[test]
    fn scatter_returns_the_base_facts() {
        let op = L0::Scatter {
            axis: 0,
            combine: ScatterCombine::Add,
            base: Id(0),
            idx: Id(1),
            upd: Id(2),
            unique: false,
        };
        let base = f32s(&[1024, 24]);
        let facts = infer_l0(&op, &[base.clone(), u32s(&[300]), f32s(&[300, 24])]).unwrap();
        assert_eq!(facts, base);

        // Update extent must match the index count.
        assert!(infer_l0(&op, &[base.clone(), u32s(&[300]), f32s(&[299, 24])]).is_err());
        // Non-scattered axes must agree with the base.
        assert!(infer_l0(&op, &[base, u32s(&[300]), f32s(&[300, 25])]).is_err());
    }

    #[test]
    fn dequant_and_project() {
        let fmt = crate::dtype::QFmt::Q4K;
        let q = ValueFacts {
            dtype: Dtype::Q(fmt),
            shape: dims(&[64, 256]),
            numeric: NumericContract::RELAXED,
            persistence: Persistence::Persistent,
            outs: 1,
        };
        let op = L0::Dequant {
            fmt,
            layout: crate::dtype::QLayout::Native,
            x: Id(0),
        };
        let facts = infer_l0(&op, std::slice::from_ref(&q)).unwrap();
        assert_eq!(facts.dtype, Dtype::F32);
        assert_eq!(facts.persistence, Persistence::Persistent);
        // The format on the node must match the operand's.
        let mismatched = L0::Dequant {
            fmt: crate::dtype::QFmt::Q6K,
            layout: crate::dtype::QLayout::Native,
            x: Id(0),
        };
        assert!(infer_l0(&mismatched, &[q]).is_err());

        let mut pair = f32s(&[4]);
        pair.outs = 2;
        assert!(infer_l0(&L0::Project { slot: 1, x: Id(0) }, std::slice::from_ref(&pair)).is_ok());
        assert!(infer_l0(&L0::Project { slot: 2, x: Id(0) }, &[pair]).is_err());
    }

    #[test]
    fn leaf_persistence() {
        let param = infer_l0(
            &L0::Leaf(LeafKind::Param {
                name: crate::ir::level0::BufferId(0),
                dtype: Dtype::F32,
                shape: dims(&[8]),
            }),
            &[],
        )
        .unwrap();
        assert_eq!(param.persistence, Persistence::Persistent);

        let buffer = infer_l0(
            &L0::Leaf(LeafKind::Buffer {
                name: crate::ir::level0::BufferId(1),
                dtype: Dtype::F32,
                shape: dims(&[8]),
            }),
            &[],
        )
        .unwrap();
        assert_eq!(buffer.persistence, Persistence::Step);

        let uniform = infer_l0(
            &L0::Leaf(LeafKind::Uniform {
                sym: SymId(2),
                dtype: Dtype::F32,
            }),
            &[],
        )
        .unwrap();
        assert_eq!(uniform.rank(), 0);
    }

    #[test]
    fn every_node_is_total_under_a_missing_operand() {
        // A malformed operand list must be a typed error, never a panic.
        let ops = [
            L0::Fold {
                carrier: binop(BinOp::Add),
                axis: 0,
                acc: Dtype::F32,
                ins: smallvec![Id(0)],
            },
            L0::Contract {
                spec: EinSpec {
                    a: smallvec![Label(0)],
                    b: smallvec![Label(0)],
                    out: smallvec![],
                },
                acc: Dtype::F32,
                a: Id(0),
                b: Id(1),
                outs: 1,
            },
            L0::Gather {
                axis: 0,
                x: Id(0),
                idx: Id(1),
            },
            L0::Scatter {
                axis: 0,
                combine: ScatterCombine::Set,
                base: Id(0),
                idx: Id(1),
                upd: Id(2),
                unique: true,
            },
            L0::Project { slot: 0, x: Id(0) },
            L0::Window {
                specs: smallvec![SlidingWindow::new(0, 2, 1)],
                x: Id(0),
            },
            L0::Restride {
                specs: smallvec![StrideSpec::dim(0, Dim::Const(1))],
                bounds: BoundsProof::Static,
                x: Id(0),
            },
            L0::Dequant {
                fmt: crate::dtype::QFmt::Q4_0,
                layout: crate::dtype::QLayout::Native,
                x: Id(0),
            },
            L0::Fold {
                carrier: binop(BinOp::Min).with_tie(TiePolicy::SplitEvenly),
                axis: 0,
                acc: Dtype::F32,
                ins: smallvec![Id(0)],
            },
        ];
        for op in &ops {
            assert!(infer_l0(op, &[]).is_err(), "{op:?} should reject 0 operands");
        }
    }
}
