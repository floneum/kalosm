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
pub(crate) fn check_carrier(c: &Carrier, acc: Dtype) -> Result<()> {
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
pub(crate) fn check_map_shapes(cx: &VerifyCtx<'_>) -> Result<()> {
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
pub(crate) fn check_restride_bounds(cx: &VerifyCtx<'_>) -> Result<BoundsProof> {
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
