//! F32, F16, BF16, U32, I32 across the whole surface, plus the `widen-compute`
//! path and mixed-precision accumulators.
//!
//! Two facts shape this area. There is **no Bool**: a comparison returns
//! 1/0 in its operand's own dtype, so `eq` on a `U32` yields a `U32`. And
//! `cast` is differentiable in both directions: the gradient has to arrive
//! back in the master dtype or the master silently stops learning.
//!
//! A device without f16 or bf16 support cannot run those rows. They return
//! [`crate::harness::skip`], so the run log says `skip` and names the missing
//! capability — never `ok`.

use fusor2::{Dtype, Session, };
use fusor2::tensor::Dyn as Tensor;
use half::{bf16, f16};

use crate::harness::{
    CaseError, CaseResult, Cases, FuzzDim, dims, fill_indices, fill_range, from_u32, fuzz_case,
    skip,
};
use crate::suite::support::{Domain, expect_values, gradient_of, graph_of, read, upload};

/// The rank-2 shape every table-driven case here runs at. The tables are over
/// dtypes, not shapes, so one shared spec fuzzes them all.
const SPEC: &[FuzzDim] = &[FuzzDim::Range(1, 6), FuzzDim::Range(1, 32)];

/// The widening-sum length: long enough that an f16 accumulator provably
/// stalls (its ulp passes 1 at 2048), forward-only so it can go long.
const ACCUM_SPEC: &[FuzzDim] = &[FuzzDim::Range(4096, 16384)];

fn len_of(shape: &[u64]) -> usize {
    shape.iter().product::<u64>() as usize
}

/// `fuzz_case` keys its per-run seed off the case name for the life of the
/// process, so a formatted per-dtype name is leaked once at registration.
fn leak(name: String) -> &'static str {
    Box::leak(name.into_boxed_str())
}

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
        let name = leak(format!("roundtrip_{}", dtype_name(dtype)));
        cases.push_case(fuzz_case("dtypes", name, SPEC, move |s, shape, seed| {
            roundtrip_case(s, dtype, shape, seed)
        }));
    }

    // Every ordered pair of dense dtypes.
    for from in DENSE {
        for to in DENSE {
            if from == to {
                continue;
            }
            let (from, to) = (*from, *to);
            let name = leak(format!("cast_{}_to_{}", dtype_name(from), dtype_name(to)));
            cases.push_case(fuzz_case("dtypes", name, SPEC, move |s, shape, seed| {
                cast_case(s, from, to, shape, seed)
            }));
        }
    }

    cases.push_case(fuzz_case(
        "dtypes",
        "cast_backward_returns_to_the_master_dtype",
        SPEC,
        cast_backward,
    ));
    cases.push_case(fuzz_case(
        "dtypes",
        "cast_round_trip_through_f16_is_stable",
        SPEC,
        f16_round_trip,
    ));
    cases.push_case(fuzz_case(
        "dtypes",
        "arithmetic_in_every_float_dtype",
        SPEC,
        float_arithmetic,
    ));
    cases.push_case(fuzz_case(
        "dtypes",
        "comparison_returns_the_operand_dtype",
        SPEC,
        comparison_dtype,
    ));
    cases.push_case(fuzz_case("dtypes", "rem_is_u32_only", SPEC, rem_u32_only));
    cases.push_case(fuzz_case("dtypes", "round_modes", SPEC, round_modes));
    cases.push_case(fuzz_case(
        "dtypes",
        "float_to_int_and_back",
        SPEC,
        float_int_round_trip,
    ));
    cases.push_case(fuzz_case(
        "dtypes",
        "sum_widens_its_accumulator",
        ACCUM_SPEC,
        widening_accumulator,
    ));
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
fn roundtrip_case(session: &Session, dtype: Dtype, shape: &[u64], seed: u32) -> CaseResult {
    if let Some(why) = unsupported(session, dtype) {
        return Err(skip(why));
    }
    let len = len_of(shape);
    let data = match dtype {
        Dtype::U32 | Dtype::I32 => integral(seed, len),
        _ => Domain::Wide.sample(seed, len),
    };
    let graph = graph_of(session);
    let x = upload_as(graph.handle(), dtype, shape, &data)?;
    if x.dtype() != dtype {
        return Err(format!(
            "uploaded as {dtype:?} but the tensor reports {:?}",
            x.dtype()
        )
        .into());
    }
    let expected: Vec<f32> = data.iter().map(|v| quantize_to(dtype, *v)).collect();
    expect_values(session, shape, dtype, &read(&x)?, &expected)?;
    Ok(())
}

/// One `cast` edge. The reference is the composition of the two host
/// quantizations, so a lossy pair is expected to be lossy in exactly the way
/// the host says.
fn cast_case(session: &Session, from: Dtype, to: Dtype, shape: &[u64], seed: u32) -> CaseResult {
    for dtype in [from, to] {
        if let Some(why) = unsupported(session, dtype) {
            return Err(skip(why));
        }
    }
    // Integers only, so the case measures the cast rather than the rounding
    // of an arbitrary float into a 5-bit exponent.
    let data = integral(seed, len_of(shape));
    let graph = graph_of(session);
    let x = upload_as(graph.handle(), from, shape, &data)?;
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
    expect_values(session, shape, to, &read(&y)?, &expected)?;
    Ok(())
}

/// Mixed precision: an f32 master, an f16 compute region, and the gradient
/// routed back into the master's dtype.
///
/// The gradient must be an f32 tensor of the master's shape. A rule that
/// leaves it in f16 silently truncates every update the optimizer applies.
fn cast_backward(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    if let Some(why) = unsupported(session, Dtype::F16) {
        return Err(skip(why));
    }
    let len = len_of(shape);
    let data = Domain::Custom(0.5, 2.0).sample(seed, len);
    let graph = graph_of(session);
    let master = upload(graph.handle(), &dims(shape), &data)?;
    let half = master
        .cast(Dtype::F16)
        .and_then(|h| h.mul(&h))
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let back = half
        .cast(Dtype::F32)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let grad = gradient_of(&graph, &back, &master)?;
    if grad.len() != len {
        return Err(format!(
            "the master gradient has {} elements, want {len}",
            grad.len()
        )
        .into());
    }
    // d(x^2)/dx = 2x, computed in f16 and returned in f32.
    let want: Vec<f32> = data.iter().map(|v| 2.0 * v).collect();
    crate::compare::approx_or_relative_eq(backend_of(session), &[len], &want, &grad, 2e-2, 5e-3)?;
    Ok(())
}

/// f32 -> f16 -> f32 must be idempotent: the second trip changes nothing,
/// because the value is already representable.
fn f16_round_trip(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    if let Some(why) = unsupported(session, Dtype::F16) {
        return Err(skip(why));
    }
    let len = len_of(shape);
    let data = Domain::Wide.sample(seed, len);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(shape), &data)?;
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
    crate::compare::exact_eq(backend_of(session), &[len], &a, &b)?;
    Ok(())
}

/// `a * b + a` in each float dtype, against the host computed at that dtype's
/// own precision.
fn float_arithmetic(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let len = len_of(shape);
    let lhs = Domain::Custom(0.25, 2.0).sample(seed, len);
    let rhs = Domain::Custom(0.25, 2.0).sample(seed ^ 0x9e37_79b9, len);
    for dtype in [Dtype::F32, Dtype::F16, Dtype::BF16] {
        if unsupported(session, dtype).is_some() {
            continue;
        }
        let graph = graph_of(session);
        let a = upload_as(graph.handle(), dtype, shape, &lhs)?;
        let b = upload_as(graph.handle(), dtype, shape, &rhs)?;
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
        expect_values(session, shape, dtype, &read(&y)?, &expected)?;
    }
    Ok(())
}

/// There is no Bool. A comparison returns 1/0 **in the operand's own dtype**,
/// so `u32 == u32` is a `u32` and can be multiplied by a `u32` without a cast.
fn comparison_dtype(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    // Values in [0, 8) straddle the >= 3 threshold, so both mask values occur.
    let values = fill_indices(seed, len_of(shape), 8);
    let graph = graph_of(session);
    let x = from_u32(graph.handle(), &dims(shape), &values)
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
    expect_values(session, shape, Dtype::U32, &read(&mask)?, &expected)?;

    // The mask is usable as an operand at its own dtype, without a cast.
    let gated = x
        .mul(&mask)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let want: Vec<f32> = values
        .iter()
        .map(|v| if *v >= 3 { *v as f32 } else { 0.0 })
        .collect();
    expect_values(session, shape, Dtype::U32, &read(&gated)?, &want)?;
    Ok(())
}

/// `rem` exists for `u32` only. On floats it must be refused rather than
/// lowered to something with a different sign convention per backend.
fn rem_u32_only(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let len = len_of(shape);
    // The divisor must be nonzero; the dividends must cross it in both
    // directions so the remainder is not the identity.
    let divisor = crate::harness::Rng::new(seed ^ 0x5eed).range(1, 9) as u32;
    let values = fill_indices(seed, len, 64);
    let graph = graph_of(session);
    let a = from_u32(graph.handle(), &dims(shape), &values)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let b = from_u32(graph.handle(), &dims(shape), &vec![divisor; len])
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let y = a
        .rem(&b)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let expected: Vec<f32> = values.iter().map(|v| (v % divisor) as f32).collect();
    expect_values(session, shape, Dtype::U32, &read(&y)?, &expected)?;

    let f = upload(
        graph.handle(),
        &dims(shape),
        &Domain::Positive.sample(seed ^ 0x9e37_79b9, len),
    )?;
    if f.rem(&f).is_ok() {
        return Err("rem was accepted on f32; the reference defines it for u32 only".into());
    }
    Ok(())
}

/// Trainer constraint 5: `round`/`floor`/`ceil`/`trunc` are real primitives
/// with an explicit `RoundMode`, not fourteen comparisons.
fn round_modes(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let len = len_of(shape);
    // The quarter grid over [-4, 4]: exact in f32, lands halves in both signs,
    // which is where half-to-even and half-away-from-zero differ.
    let data: Vec<f32> = fill_indices(seed, len, 33)
        .into_iter()
        .map(|i| (i as f32 - 16.0) / 4.0)
        .collect();
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(shape), &data)?;

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
        crate::compare::exact_eq(backend_of(session), &[len], &expected, &read(&y)?)
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
fn float_int_round_trip(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let len = len_of(shape);
    // Both signs, so truncation toward zero differs from floor.
    let data = fill_range(seed, len, -8.0, 8.0);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(shape), &data)?;
    let y = x
        .cast(Dtype::I32)
        .and_then(|i| i.cast(Dtype::F32))
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let expected: Vec<f32> = data.iter().map(|v| v.trunc()).collect();
    crate::compare::exact_eq(backend_of(session), &[len], &expected, &read(&y)?)?;
    Ok(())
}

/// A long f16 sum must accumulate in a wider dtype. Summing 4096 values of
/// magnitude ~1 in f16 stalls once the running total passes 2048, because the
/// f16 ulp there exceeds the addend; the result would be roughly half the
/// right answer.
fn widening_accumulator(session: &Session, shape: &[u64], _seed: u32) -> CaseResult {
    if let Some(why) = unsupported(session, Dtype::F16) {
        return Err(skip(why));
    }
    let n = shape[0];
    let data = vec![1.0f32; n as usize];
    let graph = graph_of(session);
    let x = upload_as(graph.handle(), Dtype::F16, &[n], &data)?;
    let y = x
        .sum(0)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let got = read(&y)?;
    let total = got.first().copied().unwrap_or(f32::NAN);
    if (total - n as f32).abs() > 1.0 {
        return Err(format!(
            "an f16 sum of {n} ones gave {total}: the accumulator did not widen, so the \
             running total stalled once its ulp exceeded 1"
        )
        .into());
    }
    Ok(())
}
