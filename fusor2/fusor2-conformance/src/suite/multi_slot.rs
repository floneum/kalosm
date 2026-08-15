//! Multi-slot folds, on real hardware, on both backends.
//!
//! One accumulator per carrier slot, one `merge` per slot, and the answer
//! compared against the two-pass algorithm the carrier replaces. This is the
//! coverage that was missing, and it is why the `accs[0]` bug survived: the L2
//! reduction resolved one `TileReduceOp` for a whole fold and updated only slot
//! 0, so `Fold{(max, sum)}` computed `max(x)` and silently discarded the sum
//! while every test in the suite passed.
//!
//! The two carriers are the demoted oracles from `fusor2_ir::carrier::oracle` —
//! the same definitions that crate's unit tests run on the host evaluator, so
//! there is one spelling of each algorithm and two independent executions of it.

use fusor2::{Dtype, Session};
use fusor2::tensor::Dyn as Tensor;
use fusor2_ir::carrier::{Carrier, oracle};
use fusor2_ir::scalar::UnOp;

use crate::harness::{CaseError, CaseResult, Cases, FuzzDim, dims, fuzz_case};
use crate::suite::support::{Domain, expect_values, graph_of, read, upload};

/// `[rows, axis]`, axis at most one pass of the lane group: the one-pass body
/// and the strided loop are separate code paths and `LONG_SPEC` covers the
/// other.
const SHORT_SPEC: &[FuzzDim] = &[FuzzDim::Range(1, 6), FuzzDim::Range(1, 64)];
/// Longer than one pass of the lane group on **either** backend (256 lanes), so
/// the per-lane strided loop runs multiple passes and the tree merges real
/// partial accumulators rather than one element each. The distinction is
/// load-bearing: the loop absorbs elements with `merge(acc, lift(x))` while the
/// tree merges two accumulators, and a carrier that got only one of the two
/// right would pass at the short extent.
const LONG_SPEC: &[FuzzDim] = &[FuzzDim::Range(1, 6), FuzzDim::Range(300, 2000)];

pub fn cases() -> Cases {
    let mut cases = Cases::new();
    for (name, spec) in [
        ("shift_stabilized_sum", SHORT_SPEC),
        ("shift_stabilized_sum_long", LONG_SPEC),
    ] {
        cases.push_case(fuzz_case("multi_slot", name, spec, shift_stabilized_case));
    }
    for (name, spec) in [("welford", SHORT_SPEC), ("welford_long", LONG_SPEC)] {
        cases.push_case(fuzz_case("multi_slot", name, spec, welford_case));
    }
    // The obligation every carrier owes, exercised on a real launch: a lane
    // group that is not a multiple of the extent merges padded identity lanes,
    // and an unguarded rescale computes `0 * exp((-inf) - (-inf)) = NaN` there.
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
/// log-sum-exp: the value the two-pass form computes in two traversals.
fn shift_stabilized_case(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (rows, axis) = (shape[0], shape[1]);
    let data = Domain::Wide.sample(seed, (rows * axis) as usize);
    let carrier = oracle::shift_stabilized_sum(UnOp::Exp, Dtype::F32);
    let actual = run_fold(session, carrier, rows, axis, &data)?;

    // Two passes: the max, then the sum of the shifted exponentials.
    let mut expected = Vec::with_capacity((rows * 2) as usize);
    for row in data.chunks(axis as usize) {
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let l: f32 = row.iter().map(|v| (v - m).exp()).sum();
        expected.push(m);
        expected.push(l);
    }
    expect_values(session, &[rows, 2], Dtype::F32, &actual, &expected)?;

    // And the value the carrier exists for: a log-sum-exp that stays finite
    // where the naive `sum(exp(x))` overflows.
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
/// Every slot is checked, not just the one the answer is read from: a merge that
/// updates only slot 0 would still produce a plausible `n`.
fn welford_case(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (rows, axis) = (shape[0], shape[1]);
    let data = Domain::Wide.sample(seed, (rows * axis) as usize);
    let carrier = oracle::welford(Dtype::F32);
    let actual = run_fold(session, carrier, rows, axis, &data)?;

    let mut expected = Vec::with_capacity((rows * 3) as usize);
    for row in data.chunks(axis as usize) {
        let n = row.len() as f32;
        let mean = row.iter().sum::<f32>() / n;
        let m2: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum();
        expected.extend([n, mean, m2]);
    }
    expect_values(session, &[rows, 3], Dtype::F32, &actual, &expected)?;

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
/// `merge(identity, identity) == identity` is checked at
/// `Builder::intern_carrier`; this is the same obligation on a real launch,
/// where the rescale `l_a * exp(m_a - m)` meets `0 * exp((-inf) - (-inf))` and
/// an unguarded spelling returns NaN — and merging `(-inf, NaN)` against a real
/// partial propagates it into the answer.
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
