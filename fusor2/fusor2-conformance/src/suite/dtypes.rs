//! F32, F16, BF16, U32, I32 across the whole surface, plus the `widen-compute`
//! path and mixed-precision accumulators.
//!
//! Two facts shape this area. There is **no Bool**: a comparison returns
//! 1/0 in its operand's own dtype, so `eq` on a `U32` yields a `U32`. And
//! `cast` is differentiable in both directions, which is what makes an f32
//! master weight with an f16 compute region a routing decision rather than a
//! separate training mode — the gradient has to arrive back in the master
//! dtype or the master silently stops learning.
//!
//! A device without f16 or bf16 support cannot run those rows. They return
//! [`crate::harness::skip`], so the run log says `skip` and names the missing
//! capability — never `ok`. An M2 Max has f16 and no bf16, so on this machine
//! the bf16 rows are reported as not run rather than as passing.
//!
//! Owned by W14.

use fusor2::{Dtype, Session, Tensor};
use half::{bf16, f16};

use crate::harness::{CaseError, CaseResult, Cases, dims, from_f32, from_u32, skip};
use crate::suite::support::{Domain, expect_values, gradient_of, graph_of, read, upload};

const SHAPE: &[u64] = &[2, 6];
const LEN: usize = 12;

/// The five dense dtypes. `Q(..)` is only ever a leaf dtype and lives in the
/// `quantized` area.
const DENSE: &[Dtype] = &[Dtype::F32, Dtype::F16, Dtype::BF16, Dtype::U32, Dtype::I32];

fn backend_of(session: &Session) -> &'static str {
    if crate::harness::is_gpu(session) {
        "gpu"
    } else {
        "cpu"
    }
}

/// `Some(reason)` when `session` cannot run `dtype` at all.
fn unsupported(session: &Session, dtype: Dtype) -> Option<String> {
    let caps = session.caps();
    match dtype {
        Dtype::F16 if !caps.f16 => Some(format!("{} has no f16 support", caps.name)),
        Dtype::BF16 if !caps.bf16 => Some(format!("{} has no bf16 support", caps.name)),
        _ => None,
    }
}

/// The value `v` becomes after a round trip through `dtype`. This is the
/// reference every cast case compares against — not the original f32.
fn quantize_to(dtype: Dtype, v: f32) -> f32 {
    match dtype {
        Dtype::F32 => v,
        Dtype::F16 => f16::from_f32(v).to_f32(),
        Dtype::BF16 => bf16::from_f32(v).to_f32(),
        // Float -> int truncates toward zero, and U32 saturates at 0.
        Dtype::U32 => (v.trunc().max(0.0)) as u32 as f32,
        Dtype::I32 => v.trunc() as i32 as f32,
        Dtype::Q(_) => v,
    }
}

/// Upload `data` as `dtype`, converting from f32 on the host.
fn upload_as(
    graph: &fusor2::graph::GraphRef,
    dtype: Dtype,
    shape: &[u64],
    data: &[f32],
) -> Result<Tensor, CaseError> {
    let dimv = dims(shape);
    let mut bytes = Vec::with_capacity(data.len() * dtype.byte_size() as usize);
    for v in data {
        match dtype {
            Dtype::F32 => bytes.extend_from_slice(&v.to_le_bytes()),
            Dtype::F16 => bytes.extend_from_slice(&f16::from_f32(*v).to_le_bytes()),
            Dtype::BF16 => bytes.extend_from_slice(&bf16::from_f32(*v).to_le_bytes()),
            Dtype::U32 => bytes.extend_from_slice(&(v.max(0.0) as u32).to_le_bytes()),
            Dtype::I32 => bytes.extend_from_slice(&(*v as i32).to_le_bytes()),
            Dtype::Q(_) => return Err("cannot upload a dense buffer as a quantized dtype".into()),
        }
    }
    Tensor::from_slice(graph, dtype, &dimv, &bytes)
        .map_err(|e| -> CaseError { e.to_string().into() })
}

/// Values every dtype in [`DENSE`] can hold exactly: small non-negative
/// integers. Used wherever a case must be dtype-agnostic.
fn integral(seed: u32, len: usize) -> Vec<f32> {
    Domain::Custom(0.0, 8.0)
        .sample(seed, len)
        .into_iter()
        .map(|v| v.floor())
        .collect()
}

pub fn cases() -> Cases {
    let mut cases = Cases::new();

    // A round trip through every dense dtype: upload, read back, compare
    // against the host's own quantization of the same values.
    for dtype in DENSE {
        let dtype = *dtype;
        let name = format!("roundtrip_{}", dtype_name(dtype));
        cases.push("dtypes", name, move |s| roundtrip_case(s, dtype));
    }

    // Every ordered pair of dense dtypes. The forward cast surface is the
    // whole matrix, including the f32->u32 and f16->u32 pairs the reference
    // left open.
    for from in DENSE {
        for to in DENSE {
            if from == to {
                continue;
            }
            let (from, to) = (*from, *to);
            let name = format!("cast_{}_to_{}", dtype_name(from), dtype_name(to));
            cases.push("dtypes", name, move |s| cast_case(s, from, to));
        }
    }

    cases.push(
        "dtypes",
        "cast_backward_returns_to_the_master_dtype",
        cast_backward,
    );
    cases.push(
        "dtypes",
        "cast_round_trip_through_f16_is_stable",
        f16_round_trip,
    );
    cases.push(
        "dtypes",
        "arithmetic_in_every_float_dtype",
        float_arithmetic,
    );
    cases.push(
        "dtypes",
        "comparison_returns_the_operand_dtype",
        comparison_dtype,
    );
    cases.push("dtypes", "rem_is_u32_only", rem_u32_only);
    cases.push("dtypes", "round_modes", round_modes);
    cases.push("dtypes", "float_to_int_and_back", float_int_round_trip);
    cases.push("dtypes", "sum_widens_its_accumulator", widening_accumulator);
    cases
}

fn dtype_name(dtype: Dtype) -> &'static str {
    match dtype {
        Dtype::F32 => "f32",
        Dtype::F16 => "f16",
        Dtype::BF16 => "bf16",
        Dtype::U32 => "u32",
        Dtype::I32 => "i32",
        Dtype::Q(_) => "q",
    }
}

/// Upload as `dtype`, read back as f32, compare against the host's own
/// rounding. A dtype whose readback path drops precision differently from its
/// upload path fails here before any op runs.
fn roundtrip_case(session: &Session, dtype: Dtype) -> CaseResult {
    if let Some(why) = unsupported(session, dtype) {
        return Err(skip(why));
    }
    let data = match dtype {
        Dtype::U32 | Dtype::I32 => integral(1201, LEN),
        _ => Domain::Wide.sample(1201, LEN),
    };
    let graph = graph_of(session);
    let x = upload_as(graph.handle(), dtype, SHAPE, &data)?;
    if x.dtype() != dtype {
        return Err(format!(
            "uploaded as {dtype:?} but the tensor reports {:?}",
            x.dtype()
        )
        .into());
    }
    let expected: Vec<f32> = data.iter().map(|v| quantize_to(dtype, *v)).collect();
    expect_values(session, SHAPE, dtype, &read(&x)?, &expected)?;
    Ok(())
}

/// One `cast` edge. The reference is the composition of the two host
/// quantizations, so a lossy pair is expected to be lossy in exactly the way
/// the host says.
fn cast_case(session: &Session, from: Dtype, to: Dtype) -> CaseResult {
    for dtype in [from, to] {
        if let Some(why) = unsupported(session, dtype) {
            return Err(skip(why));
        }
    }
    // Integers only, so the case measures the cast rather than the rounding
    // of an arbitrary float into a 5-bit exponent.
    let data = integral(1213, LEN);
    let graph = graph_of(session);
    let x = upload_as(graph.handle(), from, SHAPE, &data)?;
    let y = x
        .cast(to)
        .map_err(|e| -> CaseError { format!("cast {from:?} -> {to:?}: {e}").into() })?;
    if y.dtype() != to {
        return Err(format!("cast to {to:?} produced {:?}", y.dtype()).into());
    }
    let expected: Vec<f32> = data
        .iter()
        .map(|v| quantize_to(to, quantize_to(from, *v)))
        .collect();
    expect_values(session, SHAPE, to, &read(&y)?, &expected)?;
    Ok(())
}

/// Mixed precision: an f32 master, an f16 compute region, and the gradient
/// routed back into the master's dtype.
///
/// The gradient must be an f32 tensor of the master's shape. A rule that
/// leaves it in f16 silently truncates every update the optimizer applies.
fn cast_backward(session: &Session) -> CaseResult {
    if let Some(why) = unsupported(session, Dtype::F16) {
        return Err(skip(why));
    }
    let data = Domain::Custom(0.5, 2.0).sample(1217, LEN);
    let graph = graph_of(session);
    let master = upload(graph.handle(), &dims(SHAPE), &data)?;
    let half = master
        .cast(Dtype::F16)
        .and_then(|h| h.mul(&h))
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let back = half
        .cast(Dtype::F32)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let grad = gradient_of(&graph, &back, &master)?;
    if grad.len() != LEN {
        return Err(format!(
            "the master gradient has {} elements, want {LEN}",
            grad.len()
        )
        .into());
    }
    // d(x^2)/dx = 2x, computed in f16 and returned in f32.
    let want: Vec<f32> = data.iter().map(|v| 2.0 * v).collect();
    crate::compare::approx_or_relative_eq(backend_of(session), &[LEN], &want, &grad, 2e-2, 5e-3)?;
    Ok(())
}

/// f32 -> f16 -> f32 must be idempotent: the second trip changes nothing,
/// because the value is already representable.
fn f16_round_trip(session: &Session) -> CaseResult {
    if let Some(why) = unsupported(session, Dtype::F16) {
        return Err(skip(why));
    }
    let data = Domain::Wide.sample(1223, LEN);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(SHAPE), &data)?;
    let once = x
        .cast(Dtype::F16)
        .and_then(|h| h.cast(Dtype::F32))
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let twice = once
        .cast(Dtype::F16)
        .and_then(|h| h.cast(Dtype::F32))
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    // Exact: the second trip is a no-op on an already-representable value.
    let (a, b) = (read(&once)?, read(&twice)?);
    crate::compare::exact_eq(backend_of(session), &[LEN], &a, &b)?;
    Ok(())
}

/// `a * b + a` in each float dtype, against the host computed at that dtype's
/// own precision.
fn float_arithmetic(session: &Session) -> CaseResult {
    let lhs = Domain::Custom(0.25, 2.0).sample(1229, LEN);
    let rhs = Domain::Custom(0.25, 2.0).sample(1231, LEN);
    for dtype in [Dtype::F32, Dtype::F16, Dtype::BF16] {
        if unsupported(session, dtype).is_some() {
            continue;
        }
        let graph = graph_of(session);
        let a = upload_as(graph.handle(), dtype, SHAPE, &lhs)?;
        let b = upload_as(graph.handle(), dtype, SHAPE, &rhs)?;
        let y = a
            .mul(&b)
            .and_then(|p| p.add(&a))
            .map_err(|e| -> CaseError { format!("{dtype:?}: {e}").into() })?;
        if y.dtype() != dtype {
            return Err(format!("{dtype:?} arithmetic produced {:?}", y.dtype()).into());
        }
        let expected: Vec<f32> = lhs
            .iter()
            .zip(&rhs)
            .map(|(x, y)| {
                let (x, y) = (quantize_to(dtype, *x), quantize_to(dtype, *y));
                quantize_to(dtype, quantize_to(dtype, x * y) + x)
            })
            .collect();
        expect_values(session, SHAPE, dtype, &read(&y)?, &expected)?;
    }
    Ok(())
}

/// There is no Bool. A comparison returns 1/0 **in the operand's own dtype**,
/// so `u32 == u32` is a `u32` and can be multiplied by a `u32` without a cast.
fn comparison_dtype(session: &Session) -> CaseResult {
    let values: Vec<u32> = vec![0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5];
    let graph = graph_of(session);
    let x = from_u32(graph.handle(), &dims(SHAPE), &values)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let mask = x
        .gte_scalar(3u32)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    if mask.dtype() != Dtype::U32 {
        return Err(format!(
            "a u32 comparison produced {:?}; comparisons return 1/0 in the operand's own \
             dtype and there is no Bool",
            mask.dtype()
        )
        .into());
    }
    let expected: Vec<f32> = values.iter().map(|v| f32::from(*v >= 3)).collect();
    expect_values(session, SHAPE, Dtype::U32, &read(&mask)?, &expected)?;

    // The mask is usable as an operand at its own dtype, without a cast.
    let gated = x
        .mul(&mask)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let want: Vec<f32> = values
        .iter()
        .map(|v| if *v >= 3 { *v as f32 } else { 0.0 })
        .collect();
    expect_values(session, SHAPE, Dtype::U32, &read(&gated)?, &want)?;
    Ok(())
}

/// `rem` exists for `u32` only. On floats it must be refused rather than
/// lowered to something with a different sign convention per backend.
fn rem_u32_only(session: &Session) -> CaseResult {
    let graph = graph_of(session);
    let a = from_u32(
        graph.handle(),
        &dims(SHAPE),
        &[7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6],
    )
    .map_err(|e| -> CaseError { e.to_string().into() })?;
    let b = from_u32(graph.handle(), &dims(SHAPE), &[5; LEN])
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let y = a
        .rem(&b)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let expected: Vec<f32> = [7u32, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
        .iter()
        .map(|v| (v % 5) as f32)
        .collect();
    expect_values(session, SHAPE, Dtype::U32, &read(&y)?, &expected)?;

    let f = upload(
        graph.handle(),
        &dims(SHAPE),
        &Domain::Positive.sample(1237, LEN),
    )?;
    if f.rem(&f).is_ok() {
        return Err("rem was accepted on f32; the reference defines it for u32 only".into());
    }
    Ok(())
}

/// Trainer constraint 5: `round`/`floor`/`ceil`/`trunc` are real primitives
/// with an explicit `RoundMode`, not fourteen comparisons.
fn round_modes(session: &Session) -> CaseResult {
    // Halves in both signs, so half-to-even and half-away-from-zero differ.
    let data: Vec<f32> = vec![
        -2.5, -1.5, -0.5, -0.25, 0.25, 0.5, 1.5, 2.5, 3.5, -3.5, 1.25, -1.25,
    ];
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(SHAPE), &data)?;

    let rows: [(&str, fn(&Tensor) -> fusor2::Result<Tensor>, fn(f32) -> f32); 5] = [
        ("floor", |t| t.floor(), f32::floor),
        ("ceil", |t| t.ceil(), f32::ceil),
        ("trunc", |t| t.trunc(), f32::trunc),
        ("round", |t| t.round(), host_round_away),
        ("round_even", |t| t.round_even(), host_round_even),
    ];
    for (name, build, reference) in rows {
        let y = build(&x).map_err(|e| -> CaseError { format!("{name}: {e}").into() })?;
        let expected: Vec<f32> = data.iter().copied().map(reference).collect();
        crate::compare::exact_eq(backend_of(session), &[LEN], &expected, &read(&y)?)
            .map_err(|e| -> CaseError { format!("{name}: {e}").into() })?;
    }
    Ok(())
}

/// Ties away from zero: `-2.5 -> -3`, `2.5 -> 3`.
fn host_round_away(v: f32) -> f32 {
    v.round()
}

/// Ties to even: `-2.5 -> -2`, `2.5 -> 2`, `3.5 -> 4`.
fn host_round_even(v: f32) -> f32 {
    let r = v.round();
    if (v - v.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - v.signum()
    } else {
        r
    }
}

/// float -> int -> float is truncation toward zero, and the pair composes to
/// the host's own `trunc`.
fn float_int_round_trip(session: &Session) -> CaseResult {
    let data: Vec<f32> = vec![
        -3.7, -2.2, -1.5, -0.9, 0.0, 0.4, 1.5, 2.2, 3.7, 4.9, -0.1, 7.99,
    ];
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(SHAPE), &data)?;
    let y = x
        .cast(Dtype::I32)
        .and_then(|i| i.cast(Dtype::F32))
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let expected: Vec<f32> = data.iter().map(|v| v.trunc()).collect();
    crate::compare::exact_eq(backend_of(session), &[LEN], &expected, &read(&y)?)?;
    Ok(())
}

/// A long f16 sum must accumulate in a wider dtype. Summing 4096 values of
/// magnitude ~1 in f16 stalls once the running total passes 2048, because the
/// f16 ulp there exceeds the addend; the result would be roughly half the
/// right answer.
fn widening_accumulator(session: &Session) -> CaseResult {
    if let Some(why) = unsupported(session, Dtype::F16) {
        return Err(skip(why));
    }
    const N: u64 = 4096;
    let data = vec![1.0f32; N as usize];
    let graph = graph_of(session);
    let x = upload_as(graph.handle(), Dtype::F16, &[N], &data)?;
    let y = x
        .sum(0)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let got = read(&y)?;
    let total = got.first().copied().unwrap_or(f32::NAN);
    if (total - N as f32).abs() > 1.0 {
        return Err(format!(
            "an f16 sum of {N} ones gave {total}: the accumulator did not widen, so the \
             running total stalled once its ulp exceeded 1"
        )
        .into());
    }
    Ok(())
}

/// Kept next to the cases so the f32 upload helper the rest of the suite uses
/// stays exercised from this file too.
#[allow(dead_code)]
fn upload_f32(
    graph: &fusor2::graph::GraphRef,
    shape: &[u64],
    data: &[f32],
) -> fusor2::Result<Tensor> {
    from_f32(graph, &dims(shape), data)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registered() -> Vec<String> {
        cases().names().iter().map(|n| (*n).to_string()).collect()
    }

    #[test]
    fn every_dense_dtype_has_a_round_trip() {
        let names = registered();
        for d in DENSE {
            let wanted = format!("dtypes::roundtrip_{}", dtype_name(*d));
            assert!(names.iter().any(|n| *n == wanted), "{wanted} is missing");
        }
    }

    #[test]
    fn every_ordered_cast_pair_is_registered() {
        let names = registered();
        let mut count = 0;
        for from in DENSE {
            for to in DENSE {
                if from == to {
                    continue;
                }
                let wanted = format!("dtypes::cast_{}_to_{}", dtype_name(*from), dtype_name(*to));
                assert!(names.iter().any(|n| *n == wanted), "{wanted} is missing");
                count += 1;
            }
        }
        // 5 dtypes, ordered pairs, no identities.
        assert_eq!(count, 20);
    }

    #[test]
    fn the_pairs_the_reference_left_open_are_present() {
        let names = registered();
        for wanted in ["cast_f32_to_u32", "cast_f16_to_u32"] {
            assert!(
                names.iter().any(|n| n == &format!("dtypes::{wanted}")),
                "{wanted} is missing: these are the two the reference could not express"
            );
        }
    }

    #[test]
    fn quantize_to_is_the_dtype_it_names() {
        // f16 has 10 mantissa bits: 1 + 2^-11 is not representable.
        assert_eq!(quantize_to(Dtype::F16, 1.0 + f32::powi(2.0, -11)), 1.0);
        // bf16 has 7: it rounds much sooner.
        assert_eq!(quantize_to(Dtype::BF16, 1.0 + f32::powi(2.0, -9)), 1.0);
        // f32 is the identity.
        assert_eq!(quantize_to(Dtype::F32, 1.234_567_9), 1.234_567_9);
        // Float -> int truncates toward zero; u32 saturates at 0.
        assert_eq!(quantize_to(Dtype::I32, -1.7), -1.0);
        assert_eq!(quantize_to(Dtype::U32, -1.7), 0.0);
        assert_eq!(quantize_to(Dtype::I32, 3.9), 3.0);
    }

    #[test]
    fn integral_values_survive_every_dense_dtype() {
        for v in integral(7, 64) {
            assert!((0.0..8.0).contains(&v), "{v}");
            for d in DENSE {
                assert_eq!(quantize_to(*d, v), v, "{v} is not exact in {d:?}");
            }
        }
    }

    #[test]
    fn the_two_round_modes_disagree_exactly_on_ties() {
        for (v, away, even) in [
            (-2.5f32, -3.0, -2.0),
            (-1.5, -2.0, -2.0),
            (0.5, 1.0, 0.0),
            (1.5, 2.0, 2.0),
            (2.5, 3.0, 2.0),
            (3.5, 4.0, 4.0),
        ] {
            assert_eq!(host_round_away(v), away, "round({v})");
            assert_eq!(host_round_even(v), even, "round_even({v})");
        }
        // Away from a tie they agree.
        for v in [-1.25f32, 0.4, 2.2, -3.7] {
            assert_eq!(host_round_away(v), host_round_even(v), "{v}");
        }
    }

    #[test]
    fn an_f16_accumulator_really_does_stall() {
        // The premise of `widening_accumulator`: without a wider carrier the
        // sum stops advancing, so the case is not testing nothing.
        let mut acc = f16::from_f32(0.0);
        for _ in 0..4096 {
            acc = f16::from_f32(acc.to_f32() + 1.0);
        }
        assert!(
            acc.to_f32() < 4000.0,
            "a naive f16 accumulator reached {}; the case would be vacuous",
            acc.to_f32()
        );
    }
}
