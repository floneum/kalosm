//! Multi-slot folds, on real hardware, on both backends.
//!
//! One accumulator per carrier slot, one `merge` per slot, and the answer
//! compared against the equivalent two-pass algorithm. Every slot is checked.
//!
//! The two carriers come from `fusor2_ir::carrier::oracle`, the same
//! definitions that crate's unit tests run on the host evaluator.

use fusor2::{Dtype, Session, Tensor};
use fusor2_ir::carrier::{Carrier, oracle};
use fusor2_ir::scalar::UnOp;

use crate::harness::{CaseError, CaseResult, Cases, dims};
use crate::suite::support::{Domain, expect_values, graph_of, read, upload};

/// `[rows, axis]` for the one-pass body: `AXIS` fits inside a lane group.
const ROWS: u64 = 3;
const AXIS: u64 = 5;
/// Longer than one pass of the lane group on either backend (256 lanes), so
/// the per-lane strided loop runs three passes and the tree merges real
/// partial accumulators rather than one element each.
const AXIS_LONG: u64 = 600;

pub fn cases() -> Cases {
    let mut cases = Cases::new();
    for (name, axis) in [
        ("shift_stabilized_sum", AXIS),
        ("shift_stabilized_sum_long", AXIS_LONG),
    ] {
        cases.push("multi_slot", name, move |s| shift_stabilized_case(s, axis));
    }
    for (name, axis) in [("welford", AXIS), ("welford_long", AXIS_LONG)] {
        cases.push("multi_slot", name, move |s| welford_case(s, axis));
    }
    // A lane group that is not a multiple of the extent merges padded identity
    // lanes, where an unguarded rescale computes `0 * exp((-inf) - (-inf))`.
    cases.push(
        "multi_slot",
        "identity_lanes_do_not_poison_the_merge",
        identity_lane_case,
    );
    cases
}

fn run_fold(
    session: &Session,
    carrier: Carrier,
    rows: u64,
    axis: u64,
    data: &[f32],
) -> Result<Vec<f32>, CaseError> {
    let graph = graph_of(session);
    let dimv = dims(&[rows, axis]);
    let x = upload(graph.handle(), &dimv, data)?;
    let y = fold(&x, carrier)?;
    read(&y)
}

fn fold(x: &Tensor, carrier: Carrier) -> Result<Tensor, CaseError> {
    x.fold_carrier(carrier, 1)
        .map_err(|e| -> CaseError { e.to_string().into() })
}

/// The `(running max, sum of exp(element - running max))` carrier against a
/// plain two-pass max-then-sum.
///
/// Slot 0 is the row max and slot 1 the shifted sum, so `slot0 + ln(slot1)` is
/// log-sum-exp.
fn shift_stabilized_case(session: &Session, axis: u64) -> CaseResult {
    let len = (ROWS * axis) as usize;
    let data = Domain::Wide.sample(211, len);
    let carrier = oracle::shift_stabilized_sum(UnOp::Exp, Dtype::F32);
    let actual = run_fold(session, carrier, ROWS, axis, &data)?;

    // Two passes: the max, then the sum of the shifted exponentials.
    let mut expected = Vec::with_capacity((ROWS * 2) as usize);
    for row in data.chunks(axis as usize) {
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let l: f32 = row.iter().map(|v| (v - m).exp()).sum();
        expected.push(m);
        expected.push(l);
    }
    expect_values(session, &[ROWS, 2], Dtype::F32, &actual, &expected)?;

    // The log-sum-exp stays finite where the naive `sum(exp(x))` overflows.
    for (i, chunk) in actual.chunks(2).enumerate() {
        let lse = chunk[0] + chunk[1].ln();
        let row = &data[i * axis as usize..(i + 1) * axis as usize];
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let want = m + row.iter().map(|v| (v - m).exp()).sum::<f32>().ln();
        if (lse - want).abs() > 1e-4 * want.abs().max(1.0) {
            return Err(format!("row {i} log-sum-exp: got {lse}, want {want}").into());
        }
    }
    Ok(())
}

/// The `(n, mean, m2)` carrier against a two-pass variance.
///
/// Every slot is checked, not just the one the answer is read from.
fn welford_case(session: &Session, axis: u64) -> CaseResult {
    let len = (ROWS * axis) as usize;
    let data = Domain::Wide.sample(307, len);
    let carrier = oracle::welford(Dtype::F32);
    let actual = run_fold(session, carrier, ROWS, axis, &data)?;

    let mut expected = Vec::with_capacity((ROWS * 3) as usize);
    for row in data.chunks(axis as usize) {
        let n = row.len() as f32;
        let mean = row.iter().sum::<f32>() / n;
        let m2: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum();
        expected.extend([n, mean, m2]);
    }
    expect_values(session, &[ROWS, 3], Dtype::F32, &actual, &expected)?;

    // `m2 / n` is the biased variance the two-pass form computes.
    for (i, chunk) in actual.chunks(3).enumerate() {
        let row = &data[i * axis as usize..(i + 1) * axis as usize];
        let mean = row.iter().sum::<f32>() / row.len() as f32;
        let want = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / row.len() as f32;
        let got = chunk[2] / chunk[0];
        if (got - want).abs() > 1e-4 * want.abs().max(1.0) {
            return Err(format!("row {i} variance: got {got}, want {want}").into());
        }
    }
    Ok(())
}

/// A reduced extent of 1 against a whole lane group: every lane but one merges
/// the identity against the identity.
///
/// `merge(identity, identity) == identity` must hold on a real launch, where
/// the rescale `l_a * exp(m_a - m)` meets `0 * exp((-inf) - (-inf))`.
fn identity_lane_case(session: &Session) -> CaseResult {
    let data = vec![2.5f32, -1.0, 7.25];
    let carrier = oracle::shift_stabilized_sum(UnOp::Exp, Dtype::F32);
    let actual = run_fold(session, carrier, 3, 1, &data)?;
    let expected: Vec<f32> = data.iter().flat_map(|v| [*v, 1.0]).collect();
    for (i, (g, e)) in actual.iter().zip(&expected).enumerate() {
        if !g.is_finite() {
            return Err(format!(
                "lane {i} came back {g}: merging the identity against itself must give \
                 the identity, and an unguarded rescale gives 0 * exp(NaN)"
            )
            .into());
        }
        if (g - e).abs() > 1e-5 {
            return Err(format!("lane {i}: got {g}, want {e}").into());
        }
    }
    Ok(())
}
