//! `verify_l0` — the eight Logical invariants.
//!
//! 1. Inference is total.
//! 2. **No implicit broadcasting**: all `Map` operands share the output shape.
//! 3. `Fold`: `axis < rank`; the carrier's slot vectors agree; every identity
//!    is a value of `acc`; every `Vector` slot extent is constant; and
//!    `merge(identity, identity) == identity`.
//! 4. `Contract`: every label appears in >= 2 of {a, b, out}; contracted
//!    extents agree; `acc.bits >= numeric.min_accum_bits`.
//! 5. `Restride` composes relative to current strides; `Const` dims are
//!    checked statically, `Sym` dims record a runtime mask obligation. There
//!    is no third case and no user `assume`.
//! 6. `Scatter{Set}` with possibly-duplicate indices is rejected unless the
//!    node carries `unique: true`. `Scatter{Add}` is always legal and
//!    duplicates accumulate (normative).
//! 7. `Dequant`: `shape[-1] % fmt.block_elements == 0`.
//! 8. Every op's `work` varies with shape.

use crate::carrier::{Carrier, probes_for};
use crate::contract_spec;
use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::facts::{ValueFacts, Work};
use crate::ir::logical::{Logical, ScatterCombine};
use crate::ir::{Level, Op, VerifyCtx};
use crate::semantics::infer_logical::infer_logical;
use crate::semantics::work::work_of;
use crate::shape::{BoundsProof, Dim, Dims};

/// Clause 3, shared with inference and with `verify_launch`: is this carrier a
/// well-formed accumulator in `acc`?
///
/// * the four slot vectors agree in length and are non-empty;
/// * every identity is a value of `acc` (never a quantized dtype — a
///   quantized value is not an accumulator);
/// * every `Vector` slot has a constant extent, because a symbolic private
///   array is allocatable on neither backend;
/// * the carrier obligation: `merge(identity, identity) == identity`. A
///   rescale spelled without `Carrier::safe_delta` computes
///   `0 * exp((-inf) - (-inf)) = NaN`, and every workgroup-tree and subgroup
///   schedule merges padded identity lanes, so the NaN reaches real output.
pub fn check_carrier(c: &Carrier, acc: Dtype) -> Result<()> {
    let w = c.slots.len();
    if w == 0 || c.identity.len() != w || c.lift.len() != w || c.merge.len() != w {
        return Err(Error::Legality(format!(
            "carrier has {w} slots but {} identities, {} lifts and {} merges",
            c.identity.len(),
            c.lift.len(),
            c.merge.len()
        )));
    }
    if acc.is_quantized() {
        return Err(Error::Dtype(format!(
            "a carrier has no identity in quantized dtype {acc:?}"
        )));
    }
    for (i, s) in c.identity.iter().enumerate() {
        if s.dtype() != acc {
            return Err(Error::Dtype(format!(
                "slot {i}'s identity is {:?} but the accumulator is {acc:?}",
                s.dtype()
            )));
        }
    }
    for (i, s) in c.slots.iter().enumerate() {
        if s.lanes().is_none() {
            return Err(Error::Shape(format!(
                "slot {i} has a symbolic Vector extent; a private accumulator of
                 unknown width is allocatable on neither backend"
            )));
        }
    }
    if !c.identity_closed(probes_for(acc)) {
        return Err(Error::Numeric(format!(
            "carrier is not identity-closed: merge(identity, identity) != identity \
             (identity {:?})",
            c.identity
        )));
    }
    Ok(())
}

/// Verify one Logical node. A failure means a rule or the frontend built
/// something illegal; it is never recoverable.
pub fn verify_l0(cx: &VerifyCtx<'_>) -> Result<()> {
    let Op::Logical(op) = &cx.node.op else {
        return Err(Error::verify(
            Level::Logical,
            cx.id,
            "verify_l0 applied to a node that is not Logical",
        ));
    };

    // 1. Inference is total and agrees with the recorded facts.
    let inferred = infer_logical(op, cx.operands).map_err(|e| fail(cx, format!("{e}")))?;
    if inferred != *cx.result {
        return Err(fail(
            cx,
            format!(
                "inference disagrees with the recorded facts: {inferred:?} vs {:?}",
                cx.result
            ),
        ));
    }

    // 2.
    check_map_shapes(cx)?;

    match op {
        // 3.
        Logical::Fold {
            carrier, axis, acc, ..
        } => {
            let rank = cx.operands.first().map_or(0, ValueFacts::rank);
            if *axis as usize >= rank {
                return Err(fail(
                    cx,
                    format!("fold axis {axis} out of range for rank {rank}"),
                ));
            }
            check_carrier(carrier, *acc).map_err(|e| fail(cx, format!("{e}")))?;
        }

        // 4.
        Logical::Contract { spec, acc, .. } => {
            contract_spec::partition(spec).map_err(|e| fail(cx, format!("{e}")))?;
            if let (Some(a), Some(b)) = (cx.operands.first(), cx.operands.get(1)) {
                contract_spec::extents(spec, &a.shape, &b.shape)
                    .map_err(|e| fail(cx, format!("{e}")))?;
            }
            contract_spec::check_adjoint_specs(spec).map_err(|e| fail(cx, format!("{e}")))?;
            if acc.accum_bits() < cx.result.numeric.min_accum_bits {
                return Err(fail(
                    cx,
                    format!(
                        "accumulator {acc:?} has {} bits but the value requires {}",
                        acc.accum_bits(),
                        cx.result.numeric.min_accum_bits
                    ),
                ));
            }
        }

        // 5.
        Logical::Restride { bounds, .. } => {
            let proved = check_restride_bounds(cx)?;
            if *bounds != proved {
                return Err(fail(
                    cx,
                    format!("node claims {bounds:?} but its specs prove {proved:?}"),
                ));
            }
        }

        // 6.
        Logical::Scatter {
            combine: ScatterCombine::Set,
            unique: false,
            ..
        } => {
            return Err(fail(
                cx,
                "Scatter{Set} with possibly-duplicate indices; declare unique: true or use Add",
            ));
        }

        // 7.
        Logical::Dequant { fmt, .. } => {
            let last = cx
                .operands
                .first()
                .and_then(|f| f.shape.last().copied())
                .ok_or_else(|| fail(cx, "Dequant of a rank-0 value"))?;
            match last {
                Dim::Const(v) => {
                    let block = fmt.block_elements() as u64;
                    if block == 0 || v % block != 0 {
                        return Err(fail(
                            cx,
                            format!(
                                "Dequant inner extent {v} is not a multiple of {fmt:?}'s \
                                 {block}-element block"
                            ),
                        ));
                    }
                }
                // A symbolic inner extent is admitted here: the divisibility
                // obligation rides on the producing `Restride`'s
                // `BoundsProof::RuntimeMask`, which clause 5 checks when that
                // node is verified, and codegen discharges it as a mask.
                Dim::Sym(_) => {}
            }
        }

        _ => {}
    }

    // 8.
    check_work_varies(cx, op)?;

    Ok(())
}

/// Invariant 2, split out because the frontend calls it directly before
/// emitting the stride-0 `Restride` that replaces implicit broadcasting.
pub fn check_map_shapes(cx: &VerifyCtx<'_>) -> Result<()> {
    let Op::Logical(Logical::Map { .. }) = &cx.node.op else {
        return Ok(());
    };
    let Some(first) = cx.operands.first() else {
        return Ok(());
    };
    for (i, other) in cx.operands.iter().enumerate().skip(1) {
        let same = other.rank() == first.rank()
            && other
                .shape
                .iter()
                .zip(&first.shape)
                .all(|(a, b)| a.known_eq(*b));
        if !same {
            return Err(fail(
                cx,
                format!(
                    "Map operand {i} has shape {:?} but operand 0 has {:?}; the frontend emits \
                     Restride{{multiplier:0}} rather than broadcasting implicitly",
                    other.shape, first.shape
                ),
            ));
        }
    }
    Ok(())
}

/// Invariant 5. Returns the `BoundsProof` the node must carry.
///
/// A spec is statically decidable when its `size`, its `offset` and the
/// input dim it references are all `Const`; then the last element it
/// addresses, `offset + (size - 1) * multiplier`, must be inside that dim.
/// Anything else is a runtime mask obligation — there is no third case and
/// no user `assume`.
pub fn check_restride_bounds(cx: &VerifyCtx<'_>) -> Result<BoundsProof> {
    let Op::Logical(Logical::Restride { specs, .. }) = &cx.node.op else {
        return Ok(BoundsProof::Static);
    };
    let in_shape: &Dims = match cx.operands.first() {
        Some(f) => &f.shape,
        None => return Ok(BoundsProof::RuntimeMask),
    };

    let mut all_static = true;
    for (i, s) in specs.iter().enumerate() {
        // A pure stride-0 axis at offset 0 addresses element 0 of nothing.
        if s.multiplier == 0 && s.offset.known_eq(Dim::Const(0)) {
            continue;
        }
        let in_dim = in_shape.get(s.input_dim as usize).copied();
        match (s.size, s.offset, in_dim) {
            (Dim::Const(size), Dim::Const(offset), Some(Dim::Const(extent))) => {
                let last = offset.saturating_add(size.saturating_sub(1) * s.multiplier as u64);
                if size > 0 && last >= extent {
                    return Err(fail(
                        cx,
                        format!(
                            "Restride spec {i} addresses element {last} of an input dim of \
                             extent {extent}"
                        ),
                    ));
                }
            }
            _ => all_static = false,
        }
    }
    Ok(if all_static {
        BoundsProof::Static
    } else {
        BoundsProof::RuntimeMask
    })
}

/// Invariant 8: `work` must vary with shape. Evaluate it at `cx`'s shapes and
/// again with every `Const` dim doubled.
///
/// Two exemptions: `Leaf` and `Project` are constant-work, and a node whose
/// work is zero at both bindings (an identity `Map`, a `Restride` over an
/// empty value) genuinely performs no arithmetic. The tripwire targets a
/// nonzero constant.
fn check_work_varies(cx: &VerifyCtx<'_>, op: &Logical) -> Result<()> {
    if matches!(op, Logical::Leaf(_) | Logical::Project { .. }) {
        return Ok(());
    }
    // Skip when there is no `Const` dim to double: a fully symbolic binding
    // is priced at 1 everywhere by construction.
    let has_const = cx
        .operands
        .iter()
        .chain(std::iter::once(cx.result))
        .flat_map(|f| f.shape.iter())
        .any(|d| d.as_const().is_some());
    if !has_const {
        return Ok(());
    }

    let node_op = &cx.node.op;
    let small = work_of(node_op, cx.operands, cx.result);
    let doubled_ins: Vec<ValueFacts> = cx.operands.iter().map(doubled).collect();
    let doubled_out = doubled(cx.result);
    let large = work_of(node_op, &doubled_ins, &doubled_out);

    if small == large && small != Work::default() {
        return Err(fail(cx, "work() does not vary with shape"));
    }
    Ok(())
}

fn doubled(f: &ValueFacts) -> ValueFacts {
    let mut out = f.clone();
    for d in out.shape.iter_mut() {
        if let Dim::Const(v) = *d {
            *d = Dim::Const(v.saturating_mul(2));
        }
    }
    out
}

fn fail(cx: &VerifyCtx<'_>, msg: impl Into<String>) -> Error {
    Error::verify(Level::Logical, cx.id, msg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{Caps, DeviceKind, Limits};
    use crate::dtype::{Dtype, NumericContract, QFmt, QLayout, Splat};
    use crate::egraph::Id;
    use crate::carrier::{ArgRemap, SlotTy};
    use crate::ir::logical::{EinSpec, Label, LeafKind};
    use crate::scalar::BinOp;
    use crate::ir::{Node, OpDef, OpDefRegistry, OpTag};
    use crate::scalar::ScalarExpr;
    use crate::shape::{StrideSpec, SymId};
    use smallvec::smallvec;

    fn caps() -> Caps {
        Caps {
            kind: DeviceKind::Cpu,
            name: "test".into(),
            limits: Limits::default(),
            subgroups: None,
            f16: false,
            bf16: false,
            coop: Default::default(),
            atomic_f32: false,
            workgroup_alias: false,
            mixed_precision_coop_store: false,
            pipeline_cache: false,
            timestamp_query: false,
            simd_widths: Default::default(),
            threads: 1,
        }
    }

    fn f32s(shape: &[u64]) -> ValueFacts {
        ValueFacts::new(Dtype::F32, shape.iter().map(|&d| Dim::Const(d)))
    }
    fn u32s(shape: &[u64]) -> ValueFacts {
        ValueFacts::new(Dtype::U32, shape.iter().map(|&d| Dim::Const(d)))
    }

    /// Build a node, infer its facts, and run the verifier.
    fn check(op: Logical, operands: &[ValueFacts]) -> Result<()> {
        let result = infer_logical(&op, operands)?;
        check_with_result(op, operands, &result)
    }

    fn check_with_result(op: Logical, operands: &[ValueFacts], result: &ValueFacts) -> Result<()> {
        let caps = caps();
        let registry = OpDefRegistry::new();
        let node = Node {
            children: crate::semantics::children::children_logical(&op),
            op: Op::Logical(op),
            level: Level::Logical,
        };
        let cx = VerifyCtx {
            node: &node,
            id: Id(3),
            operands,
            result,
            caps: &caps,
            registry: &registry,
        };
        verify_l0(&cx)
    }

    #[test]
    fn contract_rejects_a_label_only_in_out_and_a_narrow_accumulator() {
        let bad_spec = EinSpec {
            a: smallvec![Label(b'm'), Label(b'k')],
            b: smallvec![Label(b'n'), Label(b'k')],
            out: smallvec![Label(b'm'), Label(b'n'), Label(b'z')],
        };
        let op = Logical::Contract {
            spec: bad_spec,
            acc: Dtype::F32,
            a: Id(0),
            b: Id(1),
            outs: 1,
        };
        assert!(check(op, &[f32s(&[3, 4]), f32s(&[5, 4])]).is_err());

        // f16 accumulation under min_accum_bits: 32.
        let spec = EinSpec {
            a: smallvec![Label(b'm'), Label(b'k')],
            b: smallvec![Label(b'n'), Label(b'k')],
            out: smallvec![Label(b'm'), Label(b'n')],
        };
        let op = Logical::Contract {
            spec,
            acc: Dtype::F16,
            a: Id(0),
            b: Id(1),
            outs: 1,
        };
        // The value's contract is RELAXED, whose min_accum_bits is 32.
        assert_eq!(NumericContract::RELAXED.min_accum_bits, 32);
        let err = check(op, &[f32s(&[3, 4]), f32s(&[5, 4])]).unwrap_err();
        assert!(format!("{err}").contains("accumulator"));
    }

    #[test]
    fn a_wide_enough_accumulator_passes() {
        let spec = EinSpec {
            a: smallvec![Label(b'm'), Label(b'k')],
            b: smallvec![Label(b'n'), Label(b'k')],
            out: smallvec![Label(b'm'), Label(b'n')],
        };
        let op = Logical::Contract {
            spec,
            acc: Dtype::F32,
            a: Id(0),
            b: Id(1),
            outs: 1,
        };
        check(op, &[f32s(&[3, 4]), f32s(&[5, 4])]).unwrap();
    }

    #[test]
    fn scatter_set_needs_unique_indices() {
        let make = |combine, unique| Logical::Scatter {
            axis: 0,
            combine,
            base: Id(0),
            idx: Id(1),
            upd: Id(2),
            unique,
        };
        let ins = [f32s(&[16, 3]), u32s(&[4]), f32s(&[4, 3])];

        assert!(check(make(ScatterCombine::Set, false), &ins).is_err());
        check(make(ScatterCombine::Set, true), &ins).unwrap();
        check(make(ScatterCombine::Add, false), &ins).unwrap();
    }

    #[test]
    fn dequant_block_divisibility() {
        let q = |fmt: QFmt, last: u64| ValueFacts {
            dtype: Dtype::Q(fmt),
            shape: smallvec![Dim::Const(8), Dim::Const(last)],
            numeric: NumericContract::RELAXED,
            persistence: crate::dtype::Persistence::Persistent,
            outs: 1,
        };
        let op = |fmt| Logical::Dequant {
            fmt,
            layout: QLayout::Native,
            x: Id(0),
        };
        assert!(check(op(QFmt::Q4K), &[q(QFmt::Q4K, 255)]).is_err());
        check(op(QFmt::Q4K), &[q(QFmt::Q4K, 256)]).unwrap();
        check(op(QFmt::Q4_0), &[q(QFmt::Q4_0, 32)]).unwrap();
        assert!(check(op(QFmt::Q4_0), &[q(QFmt::Q4_0, 33)]).is_err());
    }

    #[test]
    fn restride_bounds_are_static_or_masked_and_never_both() {
        // In range and fully constant: the node must claim Static.
        let ok = Logical::Restride {
            specs: smallvec![StrideSpec::dim(0, Dim::Const(3)).with_offset(Dim::Const(1))],
            bounds: BoundsProof::Static,
            x: Id(0),
        };
        check(ok, &[f32s(&[4])]).unwrap();

        // Out of range: offset 2 + (3-1)*1 = 4 >= 4.
        let oob = Logical::Restride {
            specs: smallvec![StrideSpec::dim(0, Dim::Const(3)).with_offset(Dim::Const(2))],
            bounds: BoundsProof::Static,
            x: Id(0),
        };
        assert!(check(oob, &[f32s(&[4])]).is_err());

        // Claiming Static under a Sym is an error; RuntimeMask is required.
        let sym = ValueFacts::new(Dtype::F32, [Dim::Sym(SymId(1))]);
        let lying = Logical::Restride {
            specs: smallvec![StrideSpec::dim(0, Dim::Sym(SymId(1)))],
            bounds: BoundsProof::Static,
            x: Id(0),
        };
        assert!(check(lying, std::slice::from_ref(&sym)).is_err());

        let honest = Logical::Restride {
            specs: smallvec![StrideSpec::dim(0, Dim::Sym(SymId(1)))],
            bounds: BoundsProof::RuntimeMask,
            x: Id(0),
        };
        check(honest, &[sym]).unwrap();
    }

    #[test]
    fn map_shape_identity_and_fold_carrier() {
        let expr = ScalarExpr::bin(
            crate::scalar::BinOp::Add,
            ScalarExpr::arg(0, Dtype::F32),
            ScalarExpr::arg(1, Dtype::F32),
        );
        let op = Logical::Map {
            expr,
            ins: smallvec![Id(0), Id(1)],
            outs: 1,
        };
        assert!(check(op, &[f32s(&[4, 8]), f32s(&[8])]).is_err());

        let sum = Carrier::binop(BinOp::Add, Splat::F32(0.0), Dtype::F32);
        let good = Logical::Fold {
            carrier: sum.clone(),
            axis: 0,
            acc: Dtype::F32,
            ins: smallvec![Id(0)],
        };
        check(good, &[f32s(&[8, 4])]).unwrap();

        // The slot vectors must agree in length: a carrier with three slots
        // and two merges is not an accumulator at all.
        let ragged = Logical::Fold {
            carrier: Carrier {
                slots: smallvec![SlotTy::Scalar, SlotTy::Scalar],
                ..sum.clone()
            },
            axis: 0,
            acc: Dtype::F32,
            ins: smallvec![Id(0)],
        };
        assert!(check(ragged, &[f32s(&[8, 4])]).is_err());

        // The identity must be a value of the accumulator dtype.
        let wrong_dtype = Logical::Fold {
            carrier: sum.clone(),
            axis: 0,
            acc: Dtype::U32,
            ins: smallvec![Id(0)],
        };
        assert!(check(wrong_dtype, &[f32s(&[8, 4])]).is_err());

        // A quantized value is never an accumulator.
        let quantized = Logical::Fold {
            carrier: sum.clone(),
            axis: 0,
            acc: Dtype::Q(crate::dtype::QFmt::Q4K),
            ins: smallvec![Id(0)],
        };
        assert!(check(quantized, &[f32s(&[8, 4])]).is_err());
    }

    /// A rescale spelled without `Carrier::safe_delta` computes
    /// `0 * exp((-inf) - (-inf)) = NaN` when two padded identity lanes merge;
    /// `verify_l0` refuses the node rather than letting the NaN reach real
    /// output.
    #[test]
    fn a_carrier_that_is_not_identity_closed_is_rejected() {
        let d = Dtype::F32;
        let (m_a, l_a) = (ScalarExpr::arg(0, d), ScalarExpr::arg(1, d));
        let (m_b, l_b) = (ScalarExpr::arg(2, d), ScalarExpr::arg(3, d));
        let m = ScalarExpr::bin(BinOp::Max, m_a.clone(), m_b.clone());
        let raw = |ms: ScalarExpr, ls: ScalarExpr| {
            ScalarExpr::bin(
                BinOp::Mul,
                ls,
                ScalarExpr::un(
                    crate::scalar::UnOp::Exp,
                    ScalarExpr::bin(BinOp::Sub, ms, m.clone()),
                ),
            )
        };
        let unguarded = Carrier {
            slots: smallvec![SlotTy::Scalar, SlotTy::Scalar],
            identity: smallvec![Splat::F32(f32::NEG_INFINITY), Splat::F32(0.0)],
            lift: smallvec![ScalarExpr::arg(0, d), ScalarExpr::lit(Splat::F32(1.0))],
            merge: smallvec![
                m.clone(),
                ScalarExpr::bin(BinOp::Add, raw(m_a, l_a), raw(m_b, l_b))
            ],
            associative: true,
            tie: None,
        };
        assert!(
            check(
                Logical::Fold {
                    carrier: unguarded.clone(),
                    axis: 0,
                    acc: d,
                    ins: smallvec![Id(0)],
                },
                &[f32s(&[8, 4])]
            )
            .is_err()
        );

        // The guarded spelling of the same algebra passes.
        let guarded = Carrier {
            merge: smallvec![
                unguarded.merge[0].clone(),
                ScalarExpr::bin(
                    BinOp::Add,
                    ScalarExpr::bin(
                        BinOp::Mul,
                        ScalarExpr::arg(1, d),
                        ScalarExpr::un(
                            crate::scalar::UnOp::Exp,
                            Carrier::safe_delta(
                                ScalarExpr::arg(0, d),
                                unguarded.merge[0].clone(),
                                Splat::F32(0.0)
                            )
                        )
                    ),
                    ScalarExpr::bin(
                        BinOp::Mul,
                        ScalarExpr::arg(3, d),
                        ScalarExpr::un(
                            crate::scalar::UnOp::Exp,
                            Carrier::safe_delta(
                                ScalarExpr::arg(2, d),
                                unguarded.merge[0].clone(),
                                Splat::F32(0.0)
                            )
                        )
                    ),
                )
            ],
            ..unguarded
        };
        check(
            Logical::Fold {
                carrier: guarded,
                axis: 0,
                acc: d,
                ins: smallvec![Id(0)],
            },
            &[f32s(&[8, 4])],
        )
        .unwrap();
        let _ = ArgRemap::identity(1);
    }

    #[test]
    fn constant_work_tripwire() {
        // The shape of the check, applied directly: a work row that reports
        // the same nonzero figure at both bindings is rejected.
        fn constant_work(_: &[ValueFacts], _: &ValueFacts) -> Work {
            Work {
                macs: 1,
                ..Work::default()
            }
        }
        fn real_work(_: &[ValueFacts], out: &ValueFacts) -> Work {
            Work {
                macs: out.elements().unwrap_or(1),
                ..Work::default()
            }
        }
        let small = f32s(&[4, 4]);
        let large = f32s(&[8, 8]);
        assert!(!crate::semantics::work::work_is_shape_sensitive(
            constant_work,
            (&[], &small),
            (&[], &large)
        ));
        assert!(crate::semantics::work::work_is_shape_sensitive(
            real_work,
            (&[], &small),
            (&[], &large)
        ));

        // And an `OpDef` carrying that row is what the registry would hold.
        let def = OpDef {
            name: "constant_work_op",
            tag: OpTag::Ext,
            verify: |_| Ok(()),
            infer: |ins| {
                ins.first()
                    .cloned()
                    .ok_or_else(|| Error::Shape("no operand".into()))
            },
            work: constant_work,
            adjoint: None,
            lower_per_target: &[],
            effect: crate::ir::launch::Effect::Pure,
        };
        assert!(!crate::semantics::work::work_is_shape_sensitive(
            def.work,
            (&[], &small),
            (&[], &large)
        ));
    }

    #[test]
    fn a_real_l0_node_passes_the_work_tripwire() {
        let expr = ScalarExpr::un(crate::scalar::UnOp::Exp, ScalarExpr::arg(0, Dtype::F32));
        let op = Logical::Map {
            expr,
            ins: smallvec![Id(0)],
            outs: 1,
        };
        check(op, &[f32s(&[4, 8])]).unwrap();
    }

    #[test]
    fn an_identity_map_is_exempt_because_its_work_is_zero() {
        let op = Logical::Map {
            expr: ScalarExpr::arg(0, Dtype::F32),
            ins: smallvec![Id(0)],
            outs: 1,
        };
        check(op, &[f32s(&[4, 8])]).unwrap();
    }

    #[test]
    fn recorded_facts_must_match_inference() {
        let op = Logical::Leaf(LeafKind::Const {
            value: Splat::F32(0.0),
            shape: smallvec![Dim::Const(4)],
        });
        // A deliberately wrong result: rank 2 where inference says rank 1.
        assert!(check_with_result(op, &[], &f32s(&[4, 4])).is_err());
    }
}
