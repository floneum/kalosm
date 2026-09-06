//! The 12 reductions, plus the two adjoints whose rule is not "broadcast the
//! gradient": `max`/`min` split evenly among ties, and `product` is
//! zero-aware in three branches.

use fusor::tensor::Dyn as Tensor;
use fusor::{Dtype, Session};

use crate::compare::{assert_gradient_matches_finite_difference, finite_difference_gradient};
use crate::harness::{CaseError, CaseResult, Cases, FuzzDim, Rng, dims, fuzz_case};
use crate::suite::support::{
    Domain, expect_values, gradient_of, graph_of, loss_of, read, read_probe_loss, read_scalar,
    upload,
};

/// `[rows, axis]`. Every table case runs a finite-difference backward, which
/// perturbs every element, so the ceiling stays small.
const SPEC: &[FuzzDim] = &[FuzzDim::Range(1, 4), FuzzDim::Range(1, 8)];

type Reduce = fn(&Tensor) -> fusor::Result<Tensor>;
type HostReduce = fn(&[f32]) -> f32;

/// `(name, keepdim, op, per-row host reference, input domain)`.
#[rustfmt::skip]
fn table() -> Vec<(&'static str, bool, Reduce, HostReduce, Domain)> {
    vec![
        ("sum_axis",         false, |x| x.sum(1),          |r| r.iter().sum(),        Domain::Wide),
        ("sum_keepdim",      true,  |x| x.sum_keepdim(1),  |r| r.iter().sum(),        Domain::Wide),
        ("mean",             false, |x| x.mean(1),         host_mean,                 Domain::Wide),
        ("max",              false, |x| x.max(1),          host_max,                  Domain::Wide),
        ("min",              false, |x| x.min(1),          host_min,                  Domain::Wide),
        ("product",          false, |x| x.product(1),      host_product,              Domain::Positive),
        ("product_keepdim",  true,  |x| x.product(1)?.unsqueeze(1), host_product,      Domain::Positive),
        ("var",              false, |x| x.var(1),          host_var,                  Domain::Wide),
        ("var_keepdim",      true,  |x| x.var(1)?.unsqueeze(1), host_var,              Domain::Wide),
        ("log_sum_exp",      false, host_lse_op,           host_lse,                  Domain::Wide),
        ("squared_sum",      false, |x| x.sqr()?.sum(1),   |r| r.iter().map(|v| v * v).sum(), Domain::Wide),
    ]
}

fn host_mean(row: &[f32]) -> f32 {
    row.iter().sum::<f32>() / row.len() as f32
}
fn host_max(row: &[f32]) -> f32 {
    row.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}
fn host_min(row: &[f32]) -> f32 {
    row.iter().copied().fold(f32::INFINITY, f32::min)
}
fn host_product(row: &[f32]) -> f32 {
    row.iter().product()
}
/// The biased variance, `mean((x - mean)^2)`, which is what the reference
/// computes on the autograd path.
fn host_var(row: &[f32]) -> f32 {
    let mean = host_mean(row);
    row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / row.len() as f32
}
fn host_lse(row: &[f32]) -> f32 {
    let max = host_max(row);
    max + row.iter().map(|v| (v - max).exp()).sum::<f32>().ln()
}
/// `log_sum_exp` as a composition, with the max shift that keeps it stable.
fn host_lse_op(x: &Tensor) -> fusor::Result<Tensor> {
    let max = x.max(1)?.unsqueeze(1)?;
    let shifted = x.sub_(&max)?;
    shifted.exp()?.sum(1)?.log()?.add(&max.squeeze(1)?)
}

pub fn cases() -> Cases {
    let mut cases = Cases::new();

    for (name, keepdim, op, reference, domain) in table() {
        cases.push_case(fuzz_case(
            "reductions",
            name,
            SPEC,
            async move |session: &Session, shape: &[u64], seed: u32| {
                reduction_case(session, shape, seed, keepdim, op, reference, domain).await
            },
        ));
    }

    // A rank-4 reduction over an interior axis: the axis-removal bookkeeping
    // is where a rank-generic `Fold` goes wrong.
    cases.push_case(fuzz_case(
        "reductions",
        "sum_high_rank",
        HIGH_RANK_SPEC,
        sum_high_rank,
    ));

    // The two adjoints whose rule is not "broadcast the gradient". The tie
    // cases are hand-authored tables; the zero-aware case plants its zeros.
    cases.push("reductions", "max_ties_split_evenly", max_ties_split_evenly);
    cases.push("reductions", "min_ties_split_evenly", min_ties_split_evenly);
    cases.push_case(fuzz_case(
        "reductions",
        "product_zero_aware",
        ZERO_AWARE_SPEC,
        product_zero_aware,
    ));

    // `fold_split` is only sound where `NumericContract::reassoc` allows it.
    cases.push_case(fuzz_case(
        "reductions",
        "fold_split_agrees_when_reassoc",
        FOLD_SPLIT_SPEC,
        fold_split_agrees,
    ));

    cases.extend(generality::cases());
    cases
}

/// The generality half: programs the fold laws were not designed for, each
/// carrying an independent host oracle.
pub mod generality {
    use fusor::{Dtype, Session};
    use fusor_ir::carrier::{ArgRemap, Carrier};
    use fusor_ir::scalar::BinOp;

    use crate::harness::{CaseError, CaseResult, Cases, FuzzDim, Rng, dims, fuzz_case};
    use crate::suite::support::{Domain, expect_shaped, graph_of, read, upload};

    pub fn cases() -> Cases {
        let mut cases = Cases::new();
        // ABSORB, second clause: a reduction over a reduction whose inner
        // result is never a buffer. No attention, no softmax, no split.
        cases.push_case(fuzz_case(
            "reductions",
            "kmeans_assignment_min_of_sums",
            KMEANS_SPEC,
            kmeans_assignment,
        ));
        // ABSORB under NumericContract::STRICT: the QAT chain every inexact
        // law must decline on, reduced by a plain sum.
        cases.push_case(fuzz_case(
            "reductions",
            "qat_fake_quant_chain_is_exact",
            QAT_SPEC,
            qat_fake_quant_chain,
        ));
        // HOIST, three rows, all EXACT in float: (*c) into an extremum,
        // (+c) into an extremum, and Neg swapping Max for Min.
        cases.push_case(fuzz_case(
            "reductions",
            "sampling_temperature_hoists_out_of_argmax",
            TEMPERATURE_SPEC,
            temperature,
        ));
        cases.push_case(fuzz_case(
            "reductions",
            "max_of_shifted_is_shifted_max",
            HOIST_SPEC,
            shifted_max,
        ));
        cases.push_case(fuzz_case(
            "reductions",
            "min_of_negated_is_negated_max",
            HOIST_SPEC,
            negated_min,
        ));
        // TUPLE: two folds over one axis, read once. Dynamic-range
        // quantization calibration — no shared algebra between the two.
        cases.push_case(fuzz_case(
            "reductions",
            "min_and_max_in_one_pass",
            TUPLE_SPEC,
            min_and_max_one_pass,
        ));
        // TUPLE / RETARGET's rotation row: a single-bin DFT is two
        // projections of one windowed signal over one axis.
        cases.push_case(fuzz_case(
            "reductions",
            "goertzel_single_bin_dft",
            GOERTZEL_SPEC,
            goertzel,
        ));
        // RETARGET: the trainer's own loss. A weighted log-sum-exp, stable
        // where the naive form overflows, with no mention of softmax.
        cases.push_case(fuzz_case(
            "reductions",
            "weighted_log_sum_exp_distillation_loss",
            DISTILLATION_SPEC,
            distillation,
        ));
        // STRIP's elide clause: an additive-identity mask over a ragged
        // batch. Nobody would write a MaskKind variant for this.
        cases.push_case(fuzz_case(
            "reductions",
            "ragged_batch_padding_is_identity",
            RAGGED_SPEC,
            ragged_padding,
        ));
        cases.push_case(fuzz_case(
            "reductions",
            "long_sum_agrees_with_f64",
            LONG_SUM_SPEC,
            long_sum_agrees_with_f64,
        ));
        cases
    }

    fn err(e: impl std::fmt::Display) -> CaseError {
        e.to_string().into()
    }

    /// `[n_points, n_centroids, dim]`. Forward only.
    const KMEANS_SPEC: &[FuzzDim] = &[
        FuzzDim::Range(2, 8),
        FuzzDim::Range(2, 8),
        FuzzDim::Range(2, 8),
    ];

    /// Nearest-centroid assignment: `min over M of sum over D of (a-b)^2`.
    /// The `[N, M]` distance matrix is an intermediate of a reduction that
    /// covers it and is never materialized.
    async fn kmeans_assignment(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
        let (n, m, d) = (shape[0], shape[1], shape[2]);
        let points = Domain::Wide.sample(seed, (n * d) as usize);
        let centroids = Domain::Wide.sample(seed ^ 0x9e37_79b9, (m * d) as usize);

        let graph = graph_of(session);
        let a = upload(graph.handle(), &dims(&[n, 1, d]), &points)?;
        let b = upload(graph.handle(), &dims(&[1, m, d]), &centroids)?;
        let dist = a
            .sub_(&b)
            .and_then(|v| v.sqr())
            .and_then(|v| v.sum(2))
            .map_err(err)?;
        let nearest = dist.min(1).map_err(err)?;
        let actual = read(&nearest).await?;

        let mut expected = Vec::with_capacity(n as usize);
        for pt in 0..n as usize {
            let mut best = f32::INFINITY;
            for ct in 0..m as usize {
                let mut acc = 0.0f32;
                for k in 0..d as usize {
                    let delta = points[pt * d as usize + k] - centroids[ct * d as usize + k];
                    acc += delta * delta;
                }
                best = best.min(acc);
            }
            expected.push(best);
        }
        expect_shaped(session, &[n], &actual, &expected).await?;

        Ok(())
    }

    /// `[rows, cols]`. Forward only, bit-exact elementwise.
    const QAT_SPEC: &[FuzzDim] = &[FuzzDim::Range(2, 6), FuzzDim::Range(16, 128)];

    /// A QAT fake-quant chain reduced by a plain `Fold{Add}`.
    ///
    /// `round(clamp(x/s, -lim, lim), HalfAwayFromZero) * s` carries
    /// `NumericContract::STRICT`, so every inexact law must decline on it.
    /// The elementwise values are asserted bit-identically against the host
    /// formula; the sum is compared to tolerance, because a reduction's order
    /// is a schedule decision.
    async fn qat_fake_quant_chain(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
        let (rows, cols) = (shape[0], shape[1]);
        const LEVELS: u32 = 127;
        let data = Domain::Custom(-3.0, 3.0).sample(seed, (rows * cols) as usize);
        let scale = 0.031_25f32; // a power of two: the division is exact.

        let graph = graph_of(session);
        let x = upload(graph.handle(), &dims(&[rows, cols]), &data)?;
        let s = upload(graph.handle(), &dims(&[1, 1]), &[scale])?;
        let q = x.fake_quant(LEVELS, &s).map_err(err)?;
        let total = q.sum(1).map_err(err)?;

        let quantized = read(&q).await?;
        let lim = LEVELS as f32;
        let host: Vec<f32> = data
            .iter()
            .map(|v| {
                let r = (v / scale).abs().round().copysign(v / scale);
                r.max(-lim).min(lim) * scale
            })
            .collect();
        // Bit equality, with `-0.0 == 0.0`: the sign of a zero is not a
        // numeric difference and no export reads it.
        let same = |a: f32, b: f32| a.to_bits() == b.to_bits() || (a == 0.0 && b == 0.0);
        for (i, (got, want)) in quantized.iter().zip(&host).enumerate() {
            if !same(*got, *want) {
                return Err(format!(
                    "fake_quant element {i}: got {got}, want {want}. This value carries \
                     NumericContract::STRICT — an inexact rewrite fired where reassoc is \
                     false, and the MSQ1 export is no longer byte-identical."
                )
                .into());
            }
        }

        let actual = read(&total).await?;
        let expected: Vec<f32> = host.chunks(cols as usize).map(|r| r.iter().sum()).collect();
        expect_shaped(session, &[rows], &actual, &expected).await?;

        Ok(())
    }

    /// `[rows, vocab]`. Forward only, bit-exact.
    const TEMPERATURE_SPEC: &[FuzzDim] = &[FuzzDim::Range(1, 4), FuzzDim::Range(8, 128)];

    /// Sampling temperature: `argmax(logits / T)`. The `(*c)` row hoists `1/T`
    /// out of the reduction; float division is monotone, so the argmax is
    /// preserved.
    async fn temperature(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
        let (rows, vocab) = (shape[0], shape[1]);
        const T: f32 = 0.7;
        let logits = Domain::Custom(-8.0, 8.0).sample(seed, (rows * vocab) as usize);

        let graph = graph_of(session);
        let x = upload(graph.handle(), &dims(&[rows, vocab]), &logits)?;
        let scaled = x.div_scalar(T).map_err(err)?;
        let hot = scaled.max(1).map_err(err)?;
        let cold = x.max(1).map_err(err)?;
        let picked = scaled.arg_max(1).map_err(err)?;

        let hot = read(&hot).await?;
        let cold = read(&cold).await?;
        let picked = read(&picked).await?;
        for (r, ((h, c), p)) in hot.iter().zip(&cold).zip(&picked).enumerate() {
            let row = &logits[r * vocab as usize..(r + 1) * vocab as usize];
            let want = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if c.to_bits() != want.to_bits() {
                return Err(format!("row {r}: max {c} != host max {want}").into());
            }
            // One ulp, not bit equality: the GPU backend compiles `/` under
            // fast math, where a divide may land one ulp from the host's.
            // The undivided max above stays bit-exact.
            let want_hot = want / T;
            if h.to_bits().abs_diff(want_hot.to_bits()) > 1 {
                return Err(format!(
                    "row {r}: max(x/T) = {h} but max(x)/T = {want_hot}. Division is \
                     monotone, so past one ulp of fast-math slack the reduction \
                     changed the value, not the rounding."
                )
                .into());
            }
            let want_arg = row
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |best, (i, v)| {
                    if *v > best.1 { (i, *v) } else { best }
                })
                .0;
            if (*p - want_arg as f32).abs() > 0.5 {
                return Err(format!("row {r}: argmax(x/T) = {p}, want {want_arg}").into());
            }
        }

        Ok(())
    }

    /// `[rows, cols]`. Forward only, bit-exact.
    const HOIST_SPEC: &[FuzzDim] = &[FuzzDim::Range(1, 6), FuzzDim::Range(4, 64)];

    /// `max(x + bias) == max(x) + bias` for a bias invariant along the
    /// reduced axis. Exact in float.
    async fn shifted_max(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
        let (rows, cols) = (shape[0], shape[1]);
        let data = Domain::Wide.sample(seed, (rows * cols) as usize);
        let bias: Vec<f32> = (0..rows).map(|r| 0.25 * (r as f32) - 0.5).collect();

        let graph = graph_of(session);
        let x = upload(graph.handle(), &dims(&[rows, cols]), &data)?;
        let b = upload(graph.handle(), &dims(&[rows, 1]), &bias)?;
        let shifted = x.add_(&b).and_then(|y| y.max(1)).map_err(err)?;
        let plain = x.max(1).map_err(err)?;

        let shifted = read(&shifted).await?;
        let plain = read(&plain).await?;
        for (r, (s, p)) in shifted.iter().zip(&plain).enumerate() {
            let want = p + bias[r];
            if s.to_bits() != want.to_bits() {
                return Err(format!(
                    "row {r}: max(x + b) = {s}, max(x) + b = {want}. The (+c) : Max -> Max \
                     row claims bit exactness; a mismatch means it does not hold."
                )
                .into());
            }
        }

        Ok(())
    }

    /// `min(-x) == -max(x)`, exactly. The `Neg` row is total on every dtype.
    async fn negated_min(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
        let (rows, cols) = (shape[0], shape[1]);
        let data = Domain::Wide.sample(seed, (rows * cols) as usize);

        let graph = graph_of(session);
        let x = upload(graph.handle(), &dims(&[rows, cols]), &data)?;
        let lhs = x.mul_scalar(-1.0).and_then(|v| v.min(1)).map_err(err)?;
        let rhs = x.max(1).and_then(|m| m.mul_scalar(-1.0)).map_err(err)?;
        let lhs = read(&lhs).await?;
        let rhs = read(&rhs).await?;
        for (r, (a, b)) in lhs.iter().zip(&rhs).enumerate() {
            if a.to_bits() != b.to_bits() {
                return Err(format!("row {r}: min(-x) = {a}, -max(x) = {b}").into());
            }
        }
        Ok(())
    }

    /// `[rows, cols]`. The axis stays longer than one lane pass (256 lanes),
    /// so the tree merges real partial accumulators.
    const TUPLE_SPEC: &[FuzzDim] = &[FuzzDim::Range(1, 4), FuzzDim::Range(300, 2000)];

    /// Dynamic-range quantization calibration: the min and the max of one
    /// tensor, in one traversal. The joint fold is compared slot by slot
    /// against two separate reductions of the same data.
    async fn min_and_max_one_pass(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
        let (rows, cols) = (shape[0], shape[1]);
        let data = Domain::Wide.sample(seed, (rows * cols) as usize);

        let max = Carrier::binop(
            BinOp::Max,
            Carrier::binop_identity(BinOp::Max, Dtype::F32)
                .ok_or_else(|| err("no Max identity"))?,
            Dtype::F32,
        );
        let min = Carrier::binop(
            BinOp::Min,
            Carrier::binop_identity(BinOp::Min, Dtype::F32)
                .ok_or_else(|| err("no Min identity"))?,
            Dtype::F32,
        );
        let joined = max.tuple(&min, &ArgRemap::identity(1));
        if joined.carrier.width() != 2 {
            return Err(format!(
                "tupling Max with Min gave {} slots: dedup collapsed two slots that are \
                 not the same statistic",
                joined.carrier.width()
            )
            .into());
        }

        let graph = graph_of(session);
        let x = upload(graph.handle(), &dims(&[rows, cols]), &data)?;
        let both = x.fold_carrier(joined.carrier, 1).map_err(err)?;
        let actual = read(&both).await?;

        let mut expected = Vec::with_capacity((rows * 2) as usize);
        for row in data.chunks(cols as usize) {
            expected.push(row.iter().copied().fold(f32::NEG_INFINITY, f32::max));
            expected.push(row.iter().copied().fold(f32::INFINITY, f32::min));
        }
        expect_shaped(session, &[rows, 2], &actual, &expected).await
    }

    /// `[rows, len]`. `len > 2 * BIN` and never a divisor of `2 * BIN` or
    /// `3 * BIN ± BIN`, so neither the fundamental nor the third harmonic
    /// aliases onto the probed bin.
    const GOERTZEL_SPEC: &[FuzzDim] = &[FuzzDim::Range(1, 4), FuzzDim::Range(24, 256)];

    /// A single-bin DFT: the real and imaginary projections of one windowed
    /// signal, over one axis, in one pass.
    async fn goertzel(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
        let (rows, len) = (shape[0], shape[1]);
        let _ = seed; // the signal is the case: a sine at the probed bin.
        const BIN: usize = 5;
        let signal: Vec<f32> = (0..rows * len)
            .map(|i| {
                let n = (i % len) as f32;
                let phase = std::f32::consts::TAU * BIN as f32 * n / len as f32;
                phase.sin() + 0.25 * (3.0 * phase).cos()
            })
            .collect();
        let cos_w: Vec<f32> = (0..len)
            .map(|n| (std::f32::consts::TAU * BIN as f32 * n as f32 / len as f32).cos())
            .collect();
        let sin_w: Vec<f32> = (0..len)
            .map(|n| (std::f32::consts::TAU * BIN as f32 * n as f32 / len as f32).sin())
            .collect();

        let graph = graph_of(session);
        let x = upload(graph.handle(), &dims(&[rows, len]), &signal)?;
        let c = upload(graph.handle(), &dims(&[1, len]), &cos_w)?;
        let s = upload(graph.handle(), &dims(&[1, len]), &sin_w)?;
        let re = x.mul_(&c).and_then(|p| p.sum(1)).map_err(err)?;
        let im = x.mul_(&s).and_then(|p| p.sum(1)).map_err(err)?;
        let re = read(&re).await?;
        let im = read(&im).await?;

        for r in 0..rows as usize {
            let row = &signal[r * len as usize..(r + 1) * len as usize];
            let (mut want_re, mut want_im) = (0.0f64, 0.0f64);
            for (n, v) in row.iter().enumerate() {
                let phase = std::f64::consts::TAU * BIN as f64 * n as f64 / len as f64;
                want_re += *v as f64 * phase.cos();
                want_im += *v as f64 * phase.sin();
            }
            let tol = 1e-2 * (want_re.abs().max(want_im.abs()).max(1.0)) as f32;
            if (re[r] - want_re as f32).abs() > tol || (im[r] - want_im as f32).abs() > tol {
                return Err(format!(
                    "bin {BIN} of row {r}: got ({}, {}), want ({want_re}, {want_im})",
                    re[r], im[r]
                )
                .into());
            }
            // The signal carries a unit sine at this bin, so the imaginary
            // projection dominates: a rotation that lost its phase would not.
            if want_im.abs() < 4.0 * want_re.abs() {
                return Err("the DFT reference lost the bin it was built at".into());
            }
        }
        Ok(())
    }

    /// `[rows, classes]`. Forward only.
    const DISTILLATION_SPEC: &[FuzzDim] = &[FuzzDim::Range(1, 4), FuzzDim::Range(8, 96)];

    /// Soft-label distillation loss: `-sum_c p_c * (x_c - lse(x))`.
    /// The logits sit at ~900, where a naive `sum(exp(x))` is `inf` and the
    /// loss is `NaN`, so the finiteness check is a real assert.
    async fn distillation(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
        let (rows, classes) = (shape[0], shape[1]);
        let mut logits = Domain::Custom(-4.0, 4.0).sample(seed, (rows * classes) as usize);
        for v in logits.iter_mut() {
            *v += 900.0;
        }
        let weights: Vec<f32> = {
            let raw = Domain::Positive.sample(seed ^ 0x9e37_79b9, (rows * classes) as usize);
            let mut w = raw.clone();
            for row in w.chunks_mut(classes as usize) {
                let total: f32 = row.iter().sum();
                for v in row.iter_mut() {
                    *v /= total;
                }
            }
            w
        };

        let graph = graph_of(session);
        let x = upload(graph.handle(), &dims(&[rows, classes]), &logits)?;
        let p = upload(graph.handle(), &dims(&[rows, classes]), &weights)?;
        let m = x.max_keepdim(1).map_err(err)?;
        let lse = x
            .sub_(&m)
            .and_then(|z| z.exp())
            .and_then(|z| z.sum_keepdim(1))
            .and_then(|z| z.log())
            .and_then(|z| z.add(&m))
            .map_err(err)?;
        let loss = x
            .sub_(&lse)
            .and_then(|z| z.mul(&p))
            .and_then(|z| z.sum(1))
            .and_then(|z| z.mul_scalar(-1.0))
            .map_err(err)?;
        let actual = read(&loss).await?;

        let mut expected = Vec::with_capacity(rows as usize);
        for r in 0..rows as usize {
            let row = &logits[r * classes as usize..(r + 1) * classes as usize];
            let w = &weights[r * classes as usize..(r + 1) * classes as usize];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let sum: f64 = row.iter().map(|v| ((v - max) as f64).exp()).sum();
            let lse = max as f64 + sum.ln();
            let l: f64 = row
                .iter()
                .zip(w)
                .map(|(v, w)| *w as f64 * (*v as f64 - lse))
                .sum();
            expected.push(-l as f32);
        }
        for (r, (got, want)) in actual.iter().zip(&expected).enumerate() {
            if !got.is_finite() {
                return Err(format!(
                    "row {r} came back {got}: the logits sit at ~900, where the unstable \
                     spelling overflows to inf and the loss is NaN"
                )
                .into());
            }
            if (got - want).abs() > 1e-2 * want.abs().max(1.0) {
                return Err(format!("row {r}: loss {got}, want {want}").into());
            }
        }
        Ok(())
    }

    /// `[rows, cols]`. Valid lengths are sampled per row in `[1, cols]`, so
    /// full rows, near-empty rows and everything between all occur.
    const RAGGED_SPEC: &[FuzzDim] = &[FuzzDim::Range(2, 6), FuzzDim::Range(8, 256)];

    /// A ragged batch: `sum(select(position < valid_len, x, 0))`. Zero is the
    /// `Add` identity, so the padded tail contributes nothing by the monoid
    /// law rather than by a padding flag on a node.
    async fn ragged_padding(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
        let (rows, cols) = (shape[0], shape[1]);
        let data = Domain::Wide.sample(seed, (rows * cols) as usize);
        let mut rng = Rng::new(seed ^ 0x5eed);
        let valid: Vec<u64> = (0..rows).map(|_| rng.range(1, cols)).collect();
        let position: Vec<f32> = (0..rows * cols).map(|i| (i % cols) as f32).collect();
        let limit: Vec<f32> = (0..rows * cols)
            .map(|i| valid[(i / cols) as usize] as f32)
            .collect();
        let zeros = vec![0.0f32; (rows * cols) as usize];

        let graph = graph_of(session);
        let shape = dims(&[rows, cols]);
        let x = upload(graph.handle(), &shape, &data)?;
        let pos = upload(graph.handle(), &shape, &position)?;
        let lim = upload(graph.handle(), &shape, &limit)?;
        let zero = upload(graph.handle(), &shape, &zeros)?;
        let keep = pos.lt_tensor(&lim).map_err(err)?;
        let masked = keep.where_cond(&x, &zero).map_err(err)?;
        let total = masked.sum(1).map_err(err)?;
        let actual = read(&total).await?;

        let expected: Vec<f32> = (0..rows as usize)
            .map(|r| {
                data[r * cols as usize..r * cols as usize + valid[r] as usize]
                    .iter()
                    .sum()
            })
            .collect();
        expect_shaped(session, &[rows], &actual, &expected).await?;

        Ok(())
    }

    /// `[rows, cols]`. Forward only, so the axis ranges over multi-workgroup
    /// lengths; the flash-aligned multiples keep every sampled length past
    /// the split threshold's neighborhood on both sides.
    const LONG_SUM_SPEC: &[FuzzDim] = &[FuzzDim::Range(1, 6), FuzzDim::Mult(512, 1024, 8192)];

    /// A plain long sum against an f64 host reference: thousands of elements
    /// is exactly where the reduction *order* matters, so whether the compiler
    /// runs it split or unsplit, the answer must sit within reassociation
    /// tolerance of the true sum.
    async fn long_sum_agrees_with_f64(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
        let (rows, cols) = (shape[0], shape[1]);
        let data = Domain::Wide.sample(seed, (rows * cols) as usize);

        let graph = graph_of(session);
        let x = upload(graph.handle(), &dims(&[rows, cols]), &data)?;
        let total = x.sum(1).map_err(err)?;
        let actual = read(&total).await?;
        // f64 accumulation on the host: at these lengths the f32 order matters.
        let expected: Vec<f32> = data
            .chunks(cols as usize)
            .map(|r| r.iter().map(|v| *v as f64).sum::<f64>() as f32)
            .collect();
        expect_shaped(session, &[rows], &actual, &expected).await?;

        Ok(())
    }
}

async fn reduction_case(
    session: &Session,
    shape: &[u64],
    seed: u32,
    keepdim: bool,
    op: Reduce,
    reference: HostReduce,
    domain: Domain,
) -> CaseResult {
    let (rows, axis) = (shape[0], shape[1]);
    let data = domain.sample(seed, (rows * axis) as usize);
    let dimv = dims(shape);
    let out_shape: Vec<u64> = if keepdim { vec![rows, 1] } else { vec![rows] };

    let graph = graph_of(session);
    let x = upload(graph.handle(), &dimv, &data)?;
    let y = op(&x).map_err(|e| -> CaseError { e.to_string().into() })?;

    let actual = read(&y).await?;
    let expected: Vec<f32> = data.chunks(axis as usize).map(reference).collect();
    expect_values(session, &out_shape, Dtype::F32, &actual, &expected).await?;

    let analytic = gradient_of(&graph, &y, &x).await?;
    let probe_graph = graph_of(session);
    let probe_x = upload(probe_graph.handle(), &dimv, &data)?;
    let probe_y = op(&probe_x).map_err(|e| -> CaseError { e.to_string().into() })?;
    let probe_loss = loss_of(&probe_y)?;
    let numeric = finite_difference_gradient(&[rows as usize, axis as usize], &data, |p| {
        read_probe_loss(&probe_x, &probe_loss, p)
    })
    .await?;
    assert_gradient_matches_finite_difference(&analytic, &numeric)?;
    Ok(())
}

/// `[b, c, h, w]`. The backward here is analytic-only (`sum`'s adjoint is a
/// broadcast of ones), so the extents can exceed the finite-difference budget.
const HIGH_RANK_SPEC: &[FuzzDim] = &[
    FuzzDim::Range(1, 4),
    FuzzDim::Range(1, 4),
    FuzzDim::Range(1, 5),
    FuzzDim::Range(1, 6),
];

async fn sum_high_rank(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (b_n, c_n, h_n, w_n) = (
        shape[0] as usize,
        shape[1] as usize,
        shape[2] as usize,
        shape[3] as usize,
    );
    let data = Domain::Wide.sample(seed, b_n * c_n * h_n * w_n);
    let dimv = dims(shape);

    let graph = graph_of(session);
    let x = upload(graph.handle(), &dimv, &data)?;
    // Axis 2 of 4: interior, so neither the innermost nor the outermost
    // special case covers it.
    let y = x
        .sum(2)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let actual = read(&y).await?;
    let mut expected = vec![0.0f32; b_n * c_n * w_n];
    for b in 0..b_n {
        for c in 0..c_n {
            for h in 0..h_n {
                for w in 0..w_n {
                    expected[(b * c_n + c) * w_n + w] += data[((b * c_n + c) * h_n + h) * w_n + w];
                }
            }
        }
    }
    expect_values(
        session,
        &[shape[0], shape[1], shape[3]],
        Dtype::F32,
        &actual,
        &expected,
    )
    .await?;

    // `sum`'s adjoint broadcasts, so every element gets exactly 1.
    let grad = gradient_of(&graph, &y, &x).await?;
    if let Some((i, v)) = grad
        .iter()
        .enumerate()
        .find(|(_, v)| (**v - 1.0).abs() > 1e-5)
    {
        return Err(format!("sum_high_rank gradient {i} is {v}, not 1").into());
    }
    Ok(())
}

/// Ties split evenly under `TiePolicy::SplitEvenly`: the gradient divides by
/// the tie count.
async fn extrema_tie_case(session: &Session, is_max: bool) -> CaseResult {
    // Row 0 has a three-way tie at the extremum; row 1 and row 2 have a unique
    // extremum, so the case covers both branches at once.
    let peak = if is_max { 1.0 } else { -1.0 };
    let filler = if is_max { 0.0 } else { 0.5 };
    let data: Vec<f32> = vec![
        peak, filler, peak, filler, peak, // three-way tie
        filler, peak, filler, filler, filler, // unique
        filler, filler, filler, peak, filler, // unique
    ];
    let dimv = dims(&[3, 5]);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dimv, &data)?;
    let y = if is_max { x.max(1) } else { x.min(1) }
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let grad = gradient_of(&graph, &y, &x).await?;
    let expected: Vec<f32> = vec![
        1.0 / 3.0,
        0.0,
        1.0 / 3.0,
        0.0,
        1.0 / 3.0, //
        0.0,
        1.0,
        0.0,
        0.0,
        0.0, //
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ];
    for (i, (g, e)) in grad.iter().zip(&expected).enumerate() {
        if (g - e).abs() > 1e-5 {
            return Err(format!(
                "{} tie gradient {i}: got {g}, want {e}. TiePolicy::SplitEvenly divides \
                 the incoming gradient by the tie count; FirstWins would give 1 to the \
                 first extremum only.",
                if is_max { "max" } else { "min" }
            )
            .into());
        }
    }
    Ok(())
}

async fn max_ties_split_evenly(session: &Session) -> CaseResult {
    extrema_tie_case(session, true).await
}

async fn min_ties_split_evenly(session: &Session) -> CaseResult {
    extrema_tie_case(session, false).await
}

/// `[rows, axis]`. Rows >= 3 so `row % 3` zeros covers all three branches of
/// the zero-aware rule every run; axis >= 2 admits the two-zero row.
const ZERO_AWARE_SPEC: &[FuzzDim] = &[FuzzDim::Range(3, 5), FuzzDim::Range(2, 8)];

/// `product`'s three-branch zero-aware rule: no zeros in the row, exactly one
/// zero, and two or more zeros (which give a zero gradient everywhere in that
/// row). The zeros are planted at sampled positions in otherwise-nonzero data.
async fn product_zero_aware(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (rows, axis) = (shape[0], shape[1]);
    let mut data = Domain::Positive.sample(seed, (rows * axis) as usize);
    let mut rng = Rng::new(seed ^ 0x5eed);
    for row in 0..rows as usize {
        let zeros = row % 3;
        if zeros >= 1 {
            let p1 = rng.range(0, axis - 1);
            data[row * axis as usize + p1 as usize] = 0.0;
            if zeros >= 2 {
                let p2 = (p1 + 1 + rng.range(0, axis - 2)) % axis;
                data[row * axis as usize + p2 as usize] = 0.0;
            }
        }
    }
    let dimv = dims(shape);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dimv, &data)?;
    let y = x
        .product(1)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let grad = gradient_of(&graph, &y, &x).await?;

    let mut expected = vec![0.0f32; data.len()];
    for row in 0..rows as usize {
        let values = &data[row * axis as usize..(row + 1) * axis as usize];
        let zeros = values.iter().filter(|v| **v == 0.0).count();
        let nonzero_product: f32 = values.iter().filter(|v| **v != 0.0).product();
        for (col, v) in values.iter().enumerate() {
            expected[row * axis as usize + col] = match zeros {
                // d(prod)/dx_i = prod / x_i, exactly.
                0 => nonzero_product / v,
                // Only the zero entry has a nonzero derivative, and it is the
                // product of the others.
                1 if *v == 0.0 => nonzero_product,
                // Two or more zeros: every partial derivative still contains a
                // zero factor.
                _ => 0.0,
            };
        }
    }
    for (i, (g, e)) in grad.iter().zip(&expected).enumerate() {
        if (g - e).abs() > 1e-4 * e.abs().max(1.0) {
            return Err(format!(
                "product gradient {i} (row {}): got {g}, want {e}",
                i / axis as usize
            )
            .into());
        }
    }
    Ok(())
}

/// The axis of the split-agreement sum: `fold_split` needs `dim >= 4096`
/// before it fires, so every sampled length sits at or past the threshold.
const FOLD_SPLIT_SPEC: &[FuzzDim] = &[FuzzDim::Range(4096, 16384)];

/// A long-axis sum whose split and unsplit forms must agree to tolerance.
///
/// The two forms are not bit-identical — float `Add` is not associative,
/// which is the whole reason the rule carries a `reassoc` guard — so they are
/// compared relatively rather than exactly.
async fn fold_split_agrees(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let long = shape[0];
    let data = Domain::Wide.sample(seed, long as usize);
    let dimv = dims(&[long]);

    let graph = graph_of(session);
    let x = upload(graph.handle(), &dimv, &data)?;
    let y = x
        .sum_all()
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let actual = read_scalar(&y).await?;

    // f64-accumulated reference: a split fold and an unsplit one must both
    // land near the true sum.
    let mut sum = 0.0f64;
    for v in &data {
        sum += *v as f64;
    }
    let expected = sum as f32;
    let scale = expected.abs().max(1.0);
    if (actual - expected).abs() > 1e-3 * scale {
        return Err(format!(
            "a {long}-element sum came out {actual}, reference {expected}. If \
             `fold_split` fired on a value whose NumericContract forbids reassociation, \
             the split and unsplit forms are not value-equal."
        )
        .into());
    }
    Ok(())
}
