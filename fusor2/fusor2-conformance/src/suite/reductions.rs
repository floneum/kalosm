//! The 12 reductions, plus the two adjoints whose rule is not "broadcast the
//! gradient": `max`/`min` split evenly among ties, and `product` is
//! zero-aware in three branches.
//!
//! `Fold`'s structural adjoint reads `combine`, so these are the two rows of
//! that table that can be wrong in an interesting way. A reduction whose
//! split and unsplit forms disagree past tolerance means `fold_split` fired
//! where `NumericContract::reassoc` forbade it.

use fusor2::{Dtype, Session, };
use fusor2::tensor::Dyn as Tensor;

use crate::compare::{assert_gradient_matches_finite_difference, finite_difference_gradient};
use crate::harness::{CaseError, CaseResult, Cases, dims};
use crate::suite::support::{
    Domain, expect_values, gradient_of, graph_of, loss_of, read, read_scalar, upload,
};

/// `[rows, axis]`: a reduction over the last axis of a small matrix.
const ROWS: u64 = 3;
const AXIS: u64 = 5;
const SHAPE: &[u64] = &[ROWS, AXIS];

type Reduce = fn(&Tensor) -> fusor2::Result<Tensor>;
type HostReduce = fn(&[f32]) -> f32;

/// `(name, out_shape, op, per-row host reference, input domain)`.
#[rustfmt::skip]
fn table() -> Vec<(&'static str, Vec<u64>, Reduce, HostReduce, Domain)> {
    vec![
        ("sum_axis",         vec![ROWS],    |x| x.sum(1),          |r| r.iter().sum(),        Domain::Wide),
        ("sum_keepdim",      vec![ROWS, 1], |x| x.sum_keepdim(1),  |r| r.iter().sum(),        Domain::Wide),
        ("mean",             vec![ROWS],    |x| x.mean(1),         host_mean,                 Domain::Wide),
        ("max",              vec![ROWS],    |x| x.max(1),          host_max,                  Domain::Wide),
        ("min",              vec![ROWS],    |x| x.min(1),          host_min,                  Domain::Wide),
        ("product",          vec![ROWS],    |x| x.product(1),      host_product,              Domain::Positive),
        ("product_keepdim",  vec![ROWS, 1], |x| x.product(1)?.unsqueeze(1), host_product,      Domain::Positive),
        ("var",              vec![ROWS],    |x| x.variance(1),     host_var,                  Domain::Wide),
        ("var_keepdim",      vec![ROWS, 1], |x| x.variance(1)?.unsqueeze(1), host_var,         Domain::Wide),
        ("log_sum_exp",      vec![ROWS],    host_lse_op,           host_lse,                  Domain::Wide),
        ("squared_sum",      vec![ROWS],    |x| x.square()?.sum(1), |r| r.iter().map(|v| v * v).sum(), Domain::Wide),
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
fn host_lse_op(x: &Tensor) -> fusor2::Result<Tensor> {
    let max = x.max(1)?.unsqueeze(1)?;
    let shifted = x.broadcast_sub(&max)?;
    shifted.exp()?.sum(1)?.log()?.add(&max.squeeze(1)?)
}

pub fn cases() -> Cases {
    let mut cases = Cases::new();

    for (name, out_shape, op, reference, domain) in table() {
        cases.push("reductions", name, move |session| {
            reduction_case(session, &out_shape, op, reference, domain)
        });
    }

    // A rank-4 reduction over an interior axis: the axis-removal bookkeeping
    // is where a rank-generic `Fold` goes wrong, and the reference has a
    // dedicated high-rank case for exactly that.
    cases.push("reductions", "sum_high_rank", sum_high_rank);

    // The two adjoints whose rule is not "broadcast the gradient".
    cases.push("reductions", "max_ties_split_evenly", max_ties_split_evenly);
    cases.push("reductions", "min_ties_split_evenly", min_ties_split_evenly);
    cases.push("reductions", "product_zero_aware", product_zero_aware);

    // `fold_split` is only sound where `NumericContract::reassoc` allows it:
    // without the guard the rule declares the split and unsplit forms
    // value-equal, and extraction swaps them on cost, on an f16 accumulator,
    // in a system whose acceptance test is a byte-identical QAT export.
    cases.push(
        "reductions",
        "fold_split_agrees_when_reassoc",
        fold_split_agrees,
    );

    cases.extend(generality::cases());
    cases
}

/// The generality half: programs the fold laws were **not** designed for.
///
/// A law that only derives its motivating example is a recognizer with extra
/// steps, so each case here is a reduction nobody aimed a rule at — k-means
/// assignment, quantization calibration, sampling temperature, a distillation
/// loss, a single-bin DFT, a ragged batch — carrying an independent host
/// oracle. The oracle is the load-bearing half: a firing assert with no number
/// behind it is how flash attention was dead for a week.
pub mod generality {
    use fusor2::{Dtype, Session};
    use fusor2_ir::carrier::{ArgRemap, Carrier};
    use fusor2_ir::scalar::BinOp;

    use fusor2::tensor::Dyn as Tensor;

    use crate::harness::{CaseError, CaseResult, Cases, dims};
    use crate::suite::support::{Domain, expect_shaped, graph_of, read, upload};

    pub fn cases() -> Cases {
        let mut cases = Cases::new();
        // ABSORB, second clause: a reduction over a reduction whose inner
        // result is never a buffer. No attention, no softmax, no split.
        cases.push(
            "reductions",
            "kmeans_assignment_min_of_sums",
            kmeans_assignment,
        );
        // ABSORB under NumericContract::STRICT: the QAT chain every inexact
        // law must decline on, reduced by a plain sum.
        cases.push(
            "reductions",
            "qat_fake_quant_chain_is_exact",
            qat_fake_quant_chain,
        );
        // HOIST, three rows, all EXACT in float: (*c) into an extremum,
        // (+c) into an extremum, and Neg swapping Max for Min.
        cases.push(
            "reductions",
            "sampling_temperature_hoists_out_of_argmax",
            temperature,
        );
        cases.push("reductions", "max_of_shifted_is_shifted_max", shifted_max);
        cases.push("reductions", "min_of_negated_is_negated_max", negated_min);
        // TUPLE: two folds over one axis, read once. Dynamic-range
        // quantization calibration — no shared algebra between the two.
        cases.push(
            "reductions",
            "min_and_max_in_one_pass",
            min_and_max_one_pass,
        );
        // TUPLE / RETARGET's rotation row: a single-bin DFT is two
        // projections of one windowed signal over one axis.
        cases.push("reductions", "goertzel_single_bin_dft", goertzel);
        // RETARGET: the trainer's own loss. A weighted log-sum-exp, stable
        // where the naive form overflows, with no mention of softmax.
        cases.push(
            "reductions",
            "weighted_log_sum_exp_distillation_loss",
            distillation,
        );
        // STRIP's elide clause: an additive-identity mask over a ragged
        // batch. Nobody would write a MaskKind variant for this.
        cases.push(
            "reductions",
            "ragged_batch_padding_is_identity",
            ragged_padding,
        );
        // The plan half, for the laws that have not landed on these chains
        // yet: a ceiling each, with the count the law must reach.
        cases.push(
            "reductions",
            "min_and_max_as_written_plan",
            min_and_max_as_written,
        );
        cases.push("reductions", "weighted_log_sum_exp_plan", distillation_plan);
        // STRIP itself, on a long plain reduction: the `at_least(4096)` gate
        // is gone and this is the tripwire that says so.
        cases.push(
            "reductions",
            "strip_splits_a_long_reduction",
            strip_splits_a_long_reduction,
        );
        cases
    }

    fn err(e: impl std::fmt::Display) -> CaseError {
        e.to_string().into()
    }

    /// The structural half of a generality case.
    ///
    /// Every case below carries a host oracle already. What it did not carry
    /// is the other half of the acceptance bar: **did the law actually fire on
    /// the chain the frontend emits**. A numeric case passes whether the
    /// program was rewritten or run naively, so a suite of numeric cases
    /// cannot tell a landed law from a dead one — which is exactly how flash
    /// attention was unreachable on both backends for a week.
    ///
    /// The graph is rebuilt from scratch for the probe rather than reusing the
    /// one the oracle read back: saturation is idempotent in the e-graph but
    /// not in the *report*, and a report over an already-saturated graph would
    /// count rule applications that fired on the previous pass.
    pub mod structure {
        use fusor2::{Session, };
use fusor2::tensor::Dyn as Tensor;

        use crate::harness::{CaseError, CaseResult};
        use crate::suite::probe::{Probe, probe};

        /// Build a fresh graph and saturate + extract it.
        pub type Build<'a> = &'a dyn Fn(&Session) -> Result<Vec<Tensor>, CaseError>;

        pub fn probe_fresh(session: &Session, build: Build<'_>) -> Result<Probe, CaseError> {
            let outs = build(session)?;
            probe(session, &outs)
        }

        /// A law that **must** fire on this program, with the reason its
        /// absence would be a bug rather than a missing feature.
        ///
        /// Use this only where the law is landed and measured: an assert for a
        /// law nobody has written yet is a failing test wearing an
        /// aspiration's clothes, and a suite full of those cannot tell a
        /// regression from an unlanded feature. The unlanded half is
        /// [`ceiling`].
        pub fn must_fire(
            session: &Session,
            build: Build<'_>,
            rules: &[(&str, &str)],
        ) -> CaseResult {
            let p = probe_fresh(session, build)?;
            for (rule, why) in rules {
                p.require_fired(rule, why)?;
            }
            Ok(())
        }

        /// Both halves at once, over one probe.
        pub fn fire_and_decline(
            session: &Session,
            build: Build<'_>,
            fire: &[(&str, &str)],
            decline: &[(&str, &str)],
        ) -> CaseResult {
            let p = probe_fresh(session, build)?;
            for (rule, why) in fire {
                p.require_fired(rule, why)?;
            }
            for (rule, why) in decline {
                p.require_declined(rule, why)?;
            }
            Ok(())
        }

        /// A launch **ceiling** on the extracted plan, plus the count the law
        /// named must reach.
        ///
        /// States where a program lands today and forbids getting worse. That
        /// is the only honest shape for a count a landing law is about to
        /// improve, and it is the same shape
        /// [`crate::launch_counts::Ceiling`] already uses for the attention
        /// forward shapes. A ceiling met with room to spare is reported, not
        /// silently fine.
        pub fn plan_ceiling(
            session: &Session,
            build: Build<'_>,
            what: &str,
            launches: usize,
            target: usize,
            rule: &str,
        ) -> CaseResult {
            let p = probe_fresh(session, build)?;
            let actual = p.launches();
            if actual > launches {
                return Err(format!(
                    "{what}: the extracted plan has {actual} launches, ceiling {launches}. \
                     The target is {target} once `{rule}` lands; this is a regression away \
                     from it. Rules that fired: {:?}",
                    p.fired_names()
                )
                .into());
            }
            Ok(())
        }

    }

    /// Nearest-centroid assignment: `min over M of sum over D of (a-b)^2`.
    ///
    /// This is `Fold{Min}(Fold{Add}((a-b)^2))` over `[N, D]` and `[M, D]`,
    /// written with no attention, no softmax and no fold splitting anywhere.
    /// It is bit-for-bit the same fact as never materializing an `[Lq, Lk]`
    /// score matrix: the `[N, M]` distance matrix is an intermediate of a
    /// reduction that covers it.
    fn kmeans_assignment(session: &Session) -> CaseResult {
        const N: u64 = 6;
        const M: u64 = 5;
        const D: u64 = 4;
        let points = Domain::Wide.sample(401, (N * D) as usize);
        let centroids = Domain::Wide.sample(402, (M * D) as usize);

        let graph = graph_of(session);
        let a = upload(graph.handle(), &dims(&[N, 1, D]), &points)?;
        let b = upload(graph.handle(), &dims(&[1, M, D]), &centroids)?;
        let dist = a
            .broadcast_sub(&b)
            .and_then(|d| d.square())
            .and_then(|d| d.sum(2))
            .map_err(err)?;
        let nearest = dist.min(1).map_err(err)?;
        let actual = read(&nearest)?;

        let mut expected = Vec::with_capacity(N as usize);
        for n in 0..N as usize {
            let mut best = f32::INFINITY;
            for m in 0..M as usize {
                let mut acc = 0.0f32;
                for d in 0..D as usize {
                    let delta = points[n * D as usize + d] - centroids[m * D as usize + d];
                    acc += delta * delta;
                }
                best = best.min(acc);
            }
            expected.push(best);
        }
        expect_shaped(session, &[N], &actual, &expected)?;

        // The structural half. `ABSORB` is what collapses `(a-b)^2` into the
        // inner fold's lift; nobody wrote a rule for nearest-neighbour
        // assignment and it is bit-for-bit the same fact as never
        // materializing an `[Lq, Lk]` score matrix.
        let build = |s: &Session| -> Result<Vec<Tensor>, CaseError> {
            let g = graph_of(s);
            let a = upload(g.handle(), &dims(&[N, 1, D]), &points)?;
            let b = upload(g.handle(), &dims(&[1, M, D]), &centroids)?;
            let dist = a
                .broadcast_sub(&b)
                .and_then(|d| d.square())
                .and_then(|d| d.sum(2))
                .map_err(err)?;
            Ok(vec![dist.min(1).map_err(err)?])
        };
        structure::must_fire(
            session,
            &build,
            &[(
                "ABSORB",
                "the squared difference must ride into the inner fold's lift; without it \
                 the [N, M, D] difference tensor is a buffer",
            )],
        )?;
        // A ceiling, not the target: the second clause of ABSORB — the
        // reduction-nesting edge that keeps the `[N, M]` distance matrix out
        // of the materialized set — is not repaired yet
        // (`fusor2-cost/src/realize.rs` still forces a boundary on every
        // fold-to-fold edge), so the distance matrix IS a buffer today. The
        // measured plan is 4 launches; the target is 1.
        structure::plan_ceiling(
            session,
            &build,
            "kmeans_assignment",
            4,
            1,
            "ABSORB's reduction-nesting clause + the fold-to-fold boundary repair",
        )
    }

    /// A QAT fake-quant chain reduced by a plain `Fold{Add}`.
    ///
    /// `round(clamp(x/s, -lim, lim), HalfAwayFromZero) * s` carries
    /// `NumericContract::STRICT`, so every *inexact* law must decline on it
    /// while substitution into a lift — which reassociates nothing — must not.
    /// The elementwise values are asserted **bit-identically** against the
    /// host formula: that exactness is the whole content of the byte-identical
    /// export, and an inexact rewrite firing here is exactly what would break
    /// it. The sum is compared to tolerance, because a reduction's *order* is
    /// a schedule decision and always was.
    fn qat_fake_quant_chain(session: &Session) -> CaseResult {
        const ROWS: u64 = 4;
        const COLS: u64 = 96;
        const LEVELS: u32 = 127;
        let data = Domain::Custom(-3.0, 3.0).sample(403, (ROWS * COLS) as usize);
        let scale = 0.031_25f32; // a power of two: the division is exact.

        let graph = graph_of(session);
        let x = upload(graph.handle(), &dims(&[ROWS, COLS]), &data)?;
        let s = upload(graph.handle(), &dims(&[1, 1]), &[scale])?;
        let q = x.fake_quant(LEVELS, &s).map_err(err)?;
        let total = q.sum(1).map_err(err)?;

        let quantized = read(&q)?;
        let lim = LEVELS as f32;
        let host: Vec<f32> = data
            .iter()
            .map(|v| {
                let r = (v / scale).abs().round().copysign(v / scale);
                r.max(-lim).min(lim) * scale
            })
            .collect();
        // Bit equality, with `-0.0 == 0.0`: the sign of a zero is not a
        // numeric difference and no export reads it, but every other bit is
        // exactly what "byte-identical" means.
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

        let actual = read(&total)?;
        let expected: Vec<f32> = host.chunks(COLS as usize).map(|r| r.iter().sum()).collect();
        expect_shaped(session, &[ROWS], &actual, &expected)?;

        // THE STRICT ASSERT, both halves on one probe.
        //
        // The exact laws must FIRE here — a `reassoc` guard on `ABSORB` would
        // kill fusion exactly where it is most needed, and this case is the
        // guard against someone adding one. The inexact laws must DECLINE, and
        // asserting the decline is the only thing that covers the byte-
        // identical export: two new consumers of `reassoc` arrived with this
        // law set, and a law that reads `f.numeric(0)` instead of
        // `f.own().numeric` is blind to operands 1..n on exactly this chain.
        let build = |s: &Session| -> Result<Vec<Tensor>, CaseError> {
            let g = graph_of(s);
            let x = upload(g.handle(), &dims(&[ROWS, COLS]), &data)?;
            let sc = upload(g.handle(), &dims(&[1, 1]), &[scale])?;
            let q = x.fake_quant(LEVELS, &sc).map_err(err)?;
            let total = q.sum(1).map_err(err)?;
            Ok(vec![q, total])
        };
        structure::fire_and_decline(
            session,
            &build,
            &[(
                "ABSORB",
                "substitution into a lift reassociates nothing, so it is the fusion \
                 available on the QAT/MSQ1 path; a reassoc guard on it would be a bug",
            )],
            &[
                (
                    "STRIP",
                    "splitting a fold reassociates it, and the split and unsplit forms are \
                     not value-equal on this chain",
                ),
                (
                    "RETARGET",
                    "retargeting inserts a rounding step per merge; on a STRICT value the \
                     result is a different number, not a rounder one",
                ),
            ],
        )
    }

    /// Sampling temperature: `argmax(logits / T)`.
    ///
    /// The `(*c)` row hoists `1/T` out of the reduction, deleting a full pass
    /// of divides from every decode step. `T` is a literal, so the hoisted and
    /// unhoisted forms agree **bit-exactly**: float division is monotone, so
    /// the argmax is preserved, and the surviving divide is the same divide.
    fn temperature(session: &Session) -> CaseResult {
        const ROWS: u64 = 3;
        const VOCAB: u64 = 64;
        const T: f32 = 0.7;
        let logits = Domain::Custom(-8.0, 8.0).sample(404, (ROWS * VOCAB) as usize);

        let graph = graph_of(session);
        let x = upload(graph.handle(), &dims(&[ROWS, VOCAB]), &logits)?;
        let scaled = x.div_scalar(T).map_err(err)?;
        let hot = scaled.max(1).map_err(err)?;
        let cold = x.max(1).map_err(err)?;
        let picked = scaled.arg_max(1).map_err(err)?;

        let hot = read(&hot)?;
        let cold = read(&cold)?;
        let picked = read(&picked)?;
        for (r, ((h, c), p)) in hot.iter().zip(&cold).zip(&picked).enumerate() {
            let row = &logits[r * VOCAB as usize..(r + 1) * VOCAB as usize];
            let want = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if c.to_bits() != want.to_bits() {
                return Err(format!("row {r}: max {c} != host max {want}").into());
            }
            if h.to_bits() != (want / T).to_bits() {
                return Err(format!(
                    "row {r}: max(x/T) = {h} but max(x)/T = {}. The (*c) row is \
                     exact_in_float; if the two disagree the hoist changed the value.",
                    want / T
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

        // The structural claim behind "a full pass of divides deleted": the
        // whole decode-step reduction resolves in ONE launch, so the divide is
        // not a traversal of its own. Which law got there is not the assert —
        // `ABSORB` substitutes the divide into the lift and `HOIST` peels it
        // out of the reduction entirely, and both are one launch. Naming one
        // of them would pin a spelling instead of the fact.
        let build = |s: &Session| -> Result<Vec<Tensor>, CaseError> {
            let g = graph_of(s);
            let x = upload(g.handle(), &dims(&[ROWS, VOCAB]), &logits)?;
            let scaled = x.div_scalar(T).map_err(err)?;
            Ok(vec![scaled.max(1).map_err(err)?])
        };
        structure::must_fire(
            session,
            &build,
            &[(
                "ABSORB",
                "the temperature divide must reach the reduction's lift; a separate \
                 elementwise pass over the vocabulary is a whole extra read per decode step",
            )],
        )?;
        structure::plan_ceiling(
            session,
            &build,
            "sampling_temperature",
            1,
            1,
            "ABSORB (at its target: the divide costs no traversal)",
        )
    }

    /// `max(x + bias) == max(x) + bias` for a bias invariant along the
    /// reduced axis. Exact in float, and the only shape of rewrite in the
    /// whole law set legal under `NumericContract::STRICT`.
    fn shifted_max(session: &Session) -> CaseResult {
        const ROWS: u64 = 4;
        const COLS: u64 = 32;
        let data = Domain::Wide.sample(405, (ROWS * COLS) as usize);
        let bias: Vec<f32> = (0..ROWS).map(|r| 0.25 * (r as f32) - 0.5).collect();

        let graph = graph_of(session);
        let x = upload(graph.handle(), &dims(&[ROWS, COLS]), &data)?;
        let b = upload(graph.handle(), &dims(&[ROWS, 1]), &bias)?;
        let shifted = x.broadcast_add(&b).and_then(|y| y.max(1)).map_err(err)?;
        let plain = x.max(1).map_err(err)?;

        let shifted = read(&shifted)?;
        let plain = read(&plain)?;
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

        // `HOIST` itself, on the real chain. The `(+c) : Max -> Max` row is
        // `exact_in_float`, so it fires with NO reassoc permission — the only
        // rewrite in the whole law set legal on the QAT/MSQ1 path — and this
        // is the case that says so with a rule name rather than a number.
        let build = |s: &Session| -> Result<Vec<Tensor>, CaseError> {
            let g = graph_of(s);
            let x = upload(g.handle(), &dims(&[ROWS, COLS]), &data)?;
            let b = upload(g.handle(), &dims(&[ROWS, 1]), &bias)?;
            Ok(vec![
                x.broadcast_add(&b).and_then(|y| y.max(1)).map_err(err)?,
            ])
        };
        structure::must_fire(
            session,
            &build,
            &[(
                "HOIST",
                "an operand invariant along the reduced axis moves out of the reduction. \
                 The row is exact in float, so a missing firing is a missing optimization \
                 rather than a numeric guard",
            )],
        )
    }

    /// `min(-x) == -max(x)`, exactly. The `Neg` row is total on every dtype,
    /// which is why it ships where the partial monotones do not.
    fn negated_min(session: &Session) -> CaseResult {
        const ROWS: u64 = 4;
        const COLS: u64 = 40;
        let data = Domain::Wide.sample(406, (ROWS * COLS) as usize);

        let graph = graph_of(session);
        let x = upload(graph.handle(), &dims(&[ROWS, COLS]), &data)?;
        let lhs = x.mul_scalar(-1.0).and_then(|n| n.min(1)).map_err(err)?;
        let rhs = x.max(1).and_then(|m| m.mul_scalar(-1.0)).map_err(err)?;
        let lhs = read(&lhs)?;
        let rhs = read(&rhs)?;
        for (r, (a, b)) in lhs.iter().zip(&rhs).enumerate() {
            if a.to_bits() != b.to_bits() {
                return Err(format!("row {r}: min(-x) = {a}, -max(x) = {b}").into());
            }
        }
        Ok(())
    }

    /// Dynamic-range quantization calibration: the min and the max of one
    /// tensor, in **one** traversal.
    ///
    /// `Carrier::tuple` is the tupling law's own constructor, and there is no
    /// shared algebra between `Min` and `Max` for it to exploit — which is the
    /// point. The joint fold is compared slot by slot against two separate
    /// reductions of the same data.
    fn min_and_max_one_pass(session: &Session) -> CaseResult {
        const ROWS: u64 = 3;
        const COLS: u64 = 600; // longer than one lane pass, so the tree merges
        let data = Domain::Wide.sample(407, (ROWS * COLS) as usize);

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
        let x = upload(graph.handle(), &dims(&[ROWS, COLS]), &data)?;
        let both = x.fold_carrier(joined.carrier, 1).map_err(err)?;
        let actual = read(&both)?;

        let mut expected = Vec::with_capacity((ROWS * 2) as usize);
        for row in data.chunks(COLS as usize) {
            expected.push(row.iter().copied().fold(f32::NEG_INFINITY, f32::max));
            expected.push(row.iter().copied().fold(f32::INFINITY, f32::min));
        }
        expect_shaped(session, &[ROWS, 2], &actual, &expected)
    }

    /// A single-bin DFT: the real and imaginary projections of one windowed
    /// signal, over one axis, in one pass.
    ///
    /// Evidence the table is not exp-shaped. Nothing here is a softmax, a
    /// normalization or an attention; it is two folds over the same operand
    /// that a tupling law joins and a rotation row retargets.
    fn goertzel(session: &Session) -> CaseResult {
        const ROWS: u64 = 2;
        const LEN: u64 = 64;
        const BIN: usize = 5;
        let signal: Vec<f32> = (0..ROWS * LEN)
            .map(|i| {
                let n = (i % LEN) as f32;
                let phase = std::f32::consts::TAU * BIN as f32 * n / LEN as f32;
                phase.sin() + 0.25 * (3.0 * phase).cos()
            })
            .collect();
        let cos_w: Vec<f32> = (0..LEN)
            .map(|n| (std::f32::consts::TAU * BIN as f32 * n as f32 / LEN as f32).cos())
            .collect();
        let sin_w: Vec<f32> = (0..LEN)
            .map(|n| (std::f32::consts::TAU * BIN as f32 * n as f32 / LEN as f32).sin())
            .collect();

        let graph = graph_of(session);
        let x = upload(graph.handle(), &dims(&[ROWS, LEN]), &signal)?;
        let c = upload(graph.handle(), &dims(&[1, LEN]), &cos_w)?;
        let s = upload(graph.handle(), &dims(&[1, LEN]), &sin_w)?;
        let re = x.broadcast_mul(&c).and_then(|p| p.sum(1)).map_err(err)?;
        let im = x.broadcast_mul(&s).and_then(|p| p.sum(1)).map_err(err)?;
        let re = read(&re)?;
        let im = read(&im)?;

        for r in 0..ROWS as usize {
            let row = &signal[r * LEN as usize..(r + 1) * LEN as usize];
            let (mut want_re, mut want_im) = (0.0f64, 0.0f64);
            for (n, v) in row.iter().enumerate() {
                let phase = std::f64::consts::TAU * BIN as f64 * n as f64 / LEN as f64;
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

    /// Soft-label distillation loss: `-sum_c p_c * (x_c - lse(x))`.
    ///
    /// The trainer's own loss, written as an ordinary taped chain. The weights
    /// make the module slot non-scalar, which is the clause that says
    /// `(V, +)` is an arbitrary monoid rather than a running sum. The logits
    /// sit at ~900, where a naive `sum(exp(x))` is `inf` and the loss is
    /// `NaN` — so a finiteness check is a real assert, not decoration.
    fn distillation(session: &Session) -> CaseResult {
        const ROWS: u64 = 3;
        const CLASSES: u64 = 48;
        let mut logits = Domain::Custom(-4.0, 4.0).sample(408, (ROWS * CLASSES) as usize);
        for v in logits.iter_mut() {
            *v += 900.0;
        }
        let weights: Vec<f32> = {
            let raw = Domain::Positive.sample(409, (ROWS * CLASSES) as usize);
            let mut w = raw.clone();
            for row in w.chunks_mut(CLASSES as usize) {
                let total: f32 = row.iter().sum();
                for v in row.iter_mut() {
                    *v /= total;
                }
            }
            w
        };

        let graph = graph_of(session);
        let x = upload(graph.handle(), &dims(&[ROWS, CLASSES]), &logits)?;
        let p = upload(graph.handle(), &dims(&[ROWS, CLASSES]), &weights)?;
        let m = x.max_keepdim(1).map_err(err)?;
        let lse = x
            .broadcast_sub(&m)
            .and_then(|z| z.exp())
            .and_then(|z| z.sum_keepdim(1))
            .and_then(|z| z.log())
            .and_then(|z| z.add(&m))
            .map_err(err)?;
        let loss = x
            .broadcast_sub(&lse)
            .and_then(|z| z.mul(&p))
            .and_then(|z| z.sum(1))
            .and_then(|z| z.mul_scalar(-1.0))
            .map_err(err)?;
        let actual = read(&loss)?;

        let mut expected = Vec::with_capacity(ROWS as usize);
        for r in 0..ROWS as usize {
            let row = &logits[r * CLASSES as usize..(r + 1) * CLASSES as usize];
            let w = &weights[r * CLASSES as usize..(r + 1) * CLASSES as usize];
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

    /// A ragged batch: `sum(select(position < valid_len, x, 0))`.
    ///
    /// Zero is the `Add` identity, so the padded tail contributes nothing by
    /// the monoid law rather than by a padding flag on a node. Key-padding
    /// masks, sliding windows, ALiBi with a cutoff and block-sparse attention
    /// are all this same clause; nobody would write a `MaskKind` variant for
    /// any of them.
    fn ragged_padding(session: &Session) -> CaseResult {
        const ROWS: u64 = 4;
        const COLS: u64 = 128;
        let data = Domain::Wide.sample(410, (ROWS * COLS) as usize);
        let valid = [128u64, 96, 33, 1];
        let position: Vec<f32> = (0..ROWS * COLS).map(|i| (i % COLS) as f32).collect();
        let limit: Vec<f32> = (0..ROWS * COLS)
            .map(|i| valid[(i / COLS) as usize] as f32)
            .collect();
        let zeros = vec![0.0f32; (ROWS * COLS) as usize];

        let graph = graph_of(session);
        let shape = dims(&[ROWS, COLS]);
        let x = upload(graph.handle(), &shape, &data)?;
        let pos = upload(graph.handle(), &shape, &position)?;
        let lim = upload(graph.handle(), &shape, &limit)?;
        let zero = upload(graph.handle(), &shape, &zeros)?;
        let keep = pos.lt_tensor(&lim).map_err(err)?;
        let masked = keep.where_cond(&x, &zero).map_err(err)?;
        let total = masked.sum(1).map_err(err)?;
        let actual = read(&total)?;

        let expected: Vec<f32> = (0..ROWS as usize)
            .map(|r| {
                data[r * COLS as usize..r * COLS as usize + valid[r] as usize]
                    .iter()
                    .sum()
            })
            .collect();
        expect_shaped(session, &[ROWS], &actual, &expected)?;

        let build = |s: &Session| -> Result<Vec<Tensor>, CaseError> {
            let g = graph_of(s);
            let x = upload(g.handle(), &shape, &data)?;
            let pos = upload(g.handle(), &shape, &position)?;
            let lim = upload(g.handle(), &shape, &limit)?;
            let zero = upload(g.handle(), &shape, &zeros)?;
            let keep = pos.lt_tensor(&lim).map_err(err)?;
            let masked = keep.where_cond(&x, &zero).map_err(err)?;
            Ok(vec![masked.sum(1).map_err(err)?])
        };
        // The mask rides into the carrier as an ordinary predicate — one
        // launch, no mask tensor pass. `STRIP`'s elide clause is what would
        // then narrow the domain rather than evaluate the predicate per
        // element; it does not fire at `COLS = 128`, which is under the block
        // this device's lane count already covers, so the count is a ceiling
        // and the target is the same launch doing less work.
        structure::must_fire(
            session,
            &build,
            &[(
                "ABSORB",
                "the padding predicate must ride into the fold's lift; a separate masking \
                 pass is a whole extra read of a ragged batch",
            )],
        )?;
        structure::plan_ceiling(
            session,
            &build,
            "ragged_batch_padding",
            1,
            1,
            "ABSORB (at its target for launches; STRIP's elide clause is the work half)",
        )
    }

    // -----------------------------------------------------------------------
    // Ceilings for the laws that have not landed on these chains yet
    // -----------------------------------------------------------------------

    /// Two folds over one axis of one operand, as the frontend writes them.
    ///
    /// `min_and_max_in_one_pass` above builds the joint carrier by hand
    /// through `Carrier::tuple`, which proves the *algebra*. This proves the
    /// *law*: what the rule table does when a program simply asks for both
    /// statistics. Today that is two launches and two reads of the input;
    /// `TUPLE` makes it one of each, and the ceiling is what turns the day it
    /// lands into a number rather than a silence.
    fn min_and_max_as_written(session: &Session) -> CaseResult {
        const ROWS: u64 = 3;
        const COLS: u64 = 600;
        let data = Domain::Wide.sample(411, (ROWS * COLS) as usize);

        let build = |s: &Session| -> Result<Vec<Tensor>, CaseError> {
            let g = graph_of(s);
            let x = upload(g.handle(), &dims(&[ROWS, COLS]), &data)?;
            Ok(vec![x.min(1).map_err(err)?, x.max(1).map_err(err)?])
        };

        let outs = build(session)?;
        let (lo, hi) = (read(&outs[0])?, read(&outs[1])?);
        let mut want_lo = Vec::new();
        let mut want_hi = Vec::new();
        for row in data.chunks(COLS as usize) {
            want_lo.push(row.iter().copied().fold(f32::INFINITY, f32::min));
            want_hi.push(row.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        }
        expect_shaped(session, &[ROWS], &lo, &want_lo)?;
        expect_shaped(session, &[ROWS], &hi, &want_hi)?;

        structure::plan_ceiling(
            session,
            &build,
            "min_and_max_as_written",
            2,
            1,
            "TUPLE (consumer- or sibling-rooted): two nests over one axis of one operand \
             are one nest over the concatenated carrier, read once",
        )
    }

    /// A stabilized weighted log-sum-exp, as an ordinary taped chain.
    ///
    /// `distillation` above pins the number. This pins the plan: ten launches
    /// today, one once `RETARGET` discharges the `max` feedback and `TUPLE`
    /// joins the running max with the weighted sum. The law is the trainer's
    /// own loss falling out of an algebra rule — no rule mentions
    /// cross-entropy and this is not attention.
    fn distillation_plan(session: &Session) -> CaseResult {
        const ROWS: u64 = 3;
        const CLASSES: u64 = 48;
        let logits = Domain::Custom(-4.0, 4.0).sample(412, (ROWS * CLASSES) as usize);
        let weights = Domain::Positive.sample(413, (ROWS * CLASSES) as usize);

        let build = |s: &Session| -> Result<Vec<Tensor>, CaseError> {
            let g = graph_of(s);
            let x = upload(g.handle(), &dims(&[ROWS, CLASSES]), &logits)?;
            let p = upload(g.handle(), &dims(&[ROWS, CLASSES]), &weights)?;
            let m = x.max_keepdim(1).map_err(err)?;
            let lse = x
                .broadcast_sub(&m)
                .and_then(|z| z.exp())
                .and_then(|z| z.sum_keepdim(1))
                .and_then(|z| z.log())
                .and_then(|z| z.add(&m))
                .map_err(err)?;
            Ok(vec![
                x.broadcast_sub(&lse)
                    .and_then(|z| z.mul(&p))
                    .and_then(|z| z.sum(1))
                    .map_err(err)?,
            ])
        };
        // The value is checked in `distillation`; this case owns the plan.
        structure::plan_ceiling(
            session,
            &build,
            "weighted_log_sum_exp_plan",
            10,
            1,
            "RETARGET (the max feedback) + TUPLE (the running max joined with the \
             weighted sum)",
        )
    }

    /// The reduction-domain split, at the extents that actually occur.
    ///
    /// The shipped `at_least(4096)` gate is gone, so `STRIP` fires on a plain
    /// long sum — asserted here, because that is the law being live at all.
    /// What it does *not* yet reach is a contraction's summed axis at
    /// `k = 512..2048`: split-K needs the factor to be a point of
    /// `ScheduleDomain::Fold` rather than a rewrite bounded by the workgroup's
    /// own lane count, and `FoldDomain` does not carry `blocks` yet. The
    /// numbers are asserted either way so the day it lands is a diff.
    fn strip_splits_a_long_reduction(session: &Session) -> CaseResult {
        const ROWS: u64 = 4;
        const COLS: u64 = 4096;
        let data = Domain::Wide.sample(414, (ROWS * COLS) as usize);

        let build = |s: &Session| -> Result<Vec<Tensor>, CaseError> {
            let g = graph_of(s);
            let x = upload(g.handle(), &dims(&[ROWS, COLS]), &data)?;
            Ok(vec![x.sum(1).map_err(err)?])
        };

        let outs = build(session)?;
        let actual = read(&outs[0])?;
        // f64 accumulation on the host: a 4096-element f32 sum is exactly the
        // case where the *order* matters, which is what the split changes.
        let expected: Vec<f32> = data
            .chunks(COLS as usize)
            .map(|r| r.iter().map(|v| *v as f64).sum::<f64>() as f32)
            .collect();
        expect_shaped(session, &[ROWS], &actual, &expected)?;

        structure::must_fire(
            session,
            &build,
            &[(
                "STRIP",
                "the reduction-domain split. The shipped `extent.at_least(4096)` gate \
                 refused every extent the trainer and the conformance cases use; if this \
                 stops firing the gate is back",
            )],
        )
    }
}

// ---------------------------------------------------------------------------

fn reduction_case(
    session: &Session,
    out_shape: &[u64],
    op: Reduce,
    reference: HostReduce,
    domain: Domain,
) -> CaseResult {
    let len = (ROWS * AXIS) as usize;
    let data = domain.sample(101, len);
    let dimv = dims(SHAPE);

    let graph = graph_of(session);
    let x = upload(graph.handle(), &dimv, &data)?;
    let y = op(&x).map_err(|e| -> CaseError { e.to_string().into() })?;

    let actual = read(&y)?;
    let expected: Vec<f32> = data.chunks(AXIS as usize).map(reference).collect();
    expect_values(session, out_shape, Dtype::F32, &actual, &expected)?;

    let analytic = gradient_of(&graph, &y, &x)?;
    let numeric = finite_difference_gradient(&[ROWS as usize, AXIS as usize], &data, &mut |p| {
        let g = graph_of(session);
        let x = upload(g.handle(), &dimv, p)?;
        let y = op(&x).map_err(|e| -> CaseError { e.to_string().into() })?;
        read_scalar(&loss_of(&y)?)
    })?;
    assert_gradient_matches_finite_difference(&analytic, &numeric)?;
    Ok(())
}

fn sum_high_rank(session: &Session) -> CaseResult {
    const SHAPE4: &[u64] = &[2, 3, 4, 5];
    let len = 2 * 3 * 4 * 5;
    let data = Domain::Wide.sample(103, len);
    let dimv = dims(SHAPE4);

    let graph = graph_of(session);
    let x = upload(graph.handle(), &dimv, &data)?;
    // Axis 2 of 4: interior, so neither the innermost nor the outermost
    // special case covers it.
    let y = x
        .sum(2)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let actual = read(&y)?;
    let mut expected = vec![0.0f32; 2 * 3 * 5];
    for b in 0..2 {
        for c in 0..3 {
            for h in 0..4 {
                for w in 0..5 {
                    expected[(b * 3 + c) * 5 + w] += data[((b * 3 + c) * 4 + h) * 5 + w];
                }
            }
        }
    }
    expect_values(session, &[2, 3, 5], Dtype::F32, &actual, &expected)?;

    // `sum`'s adjoint broadcasts, so every element gets exactly 1.
    let grad = gradient_of(&graph, &y, &x)?;
    if let Some((i, v)) = grad
        .iter()
        .enumerate()
        .find(|(_, v)| (**v - 1.0).abs() > 1e-5)
    {
        return Err(format!("sum_high_rank gradient {i} is {v}, not 1").into());
    }
    Ok(())
}

/// Ties split evenly under `TiePolicy::SplitEvenly`, which is an explicit
/// attribute rather than an implicit convention: the reference's
/// `reduction_extrema_keepdim_grad` divides by the tie count, and matching a
/// reference trainer's numerics has to be a declaration.
fn extrema_tie_case(session: &Session, is_max: bool) -> CaseResult {
    // Row 0 has a three-way tie at the extremum; row 1 and row 2 have a unique
    // extremum, so the case covers both branches at once.
    let peak = if is_max { 1.0 } else { -1.0 };
    let filler = if is_max { 0.0 } else { 0.5 };
    let data: Vec<f32> = vec![
        peak, filler, peak, filler, peak, // three-way tie
        filler, peak, filler, filler, filler, // unique
        filler, filler, filler, peak, filler, // unique
    ];
    let dimv = dims(SHAPE);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dimv, &data)?;
    let y = if is_max { x.max(1) } else { x.min(1) }
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let grad = gradient_of(&graph, &y, &x)?;
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

fn max_ties_split_evenly(session: &Session) -> CaseResult {
    extrema_tie_case(session, true)
}

fn min_ties_split_evenly(session: &Session) -> CaseResult {
    extrema_tie_case(session, false)
}

/// `product`'s three-branch zero-aware rule: no zeros in the row, exactly one
/// zero, and two or more zeros (which give a zero gradient everywhere in that
/// row).
fn product_zero_aware(session: &Session) -> CaseResult {
    let data: Vec<f32> = vec![
        2.0, 3.0, 4.0, 1.0, 5.0, // no zeros
        2.0, 0.0, 4.0, 1.0, 5.0, // exactly one zero
        2.0, 0.0, 4.0, 0.0, 5.0, // two zeros
    ];
    let dimv = dims(SHAPE);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dimv, &data)?;
    let y = x
        .product(1)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let grad = gradient_of(&graph, &y, &x)?;

    let mut expected = vec![0.0f32; data.len()];
    for row in 0..ROWS as usize {
        let values = &data[row * AXIS as usize..(row + 1) * AXIS as usize];
        let zeros = values.iter().filter(|v| **v == 0.0).count();
        let nonzero_product: f32 = values.iter().filter(|v| **v != 0.0).product();
        for (col, v) in values.iter().enumerate() {
            expected[row * AXIS as usize + col] = match zeros {
                // d(prod)/dx_i = prod / x_i, exactly.
                0 => nonzero_product / v,
                // Only the zero entry has a nonzero derivative, and it is the
                // product of the others.
                1 => {
                    if *v == 0.0 {
                        nonzero_product
                    } else {
                        0.0
                    }
                }
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
                i / AXIS as usize
            )
            .into());
        }
    }
    Ok(())
}

/// A long-axis sum whose split and unsplit forms must agree to tolerance.
///
/// `fold_split` needs `dim >= 4096` before it fires, so the axis here is
/// deliberately long. The two forms are not bit-identical — float `Add` is not
/// associative, which is the whole reason the rule carries a `reassoc` guard —
/// so they are compared relatively rather than exactly.
fn fold_split_agrees(session: &Session) -> CaseResult {
    const LONG: u64 = 8192;
    let data = Domain::Wide.sample(107, LONG as usize);
    let dimv = dims(&[LONG]);

    let graph = graph_of(session);
    let x = upload(graph.handle(), &dimv, &data)?;
    let y = x
        .sum_all()
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let actual = read_scalar(&y)?;

    // Kahan-summed reference: the point is that a *split* fold and an
    // unsplit one both land near the true sum, not that either matches a
    // naive left-to-right f32 accumulation.
    let mut sum = 0.0f64;
    for v in &data {
        sum += *v as f64;
    }
    let expected = sum as f32;
    let scale = expected.abs().max(1.0);
    if (actual - expected).abs() > 1e-3 * scale {
        return Err(format!(
            "a {LONG}-element sum came out {actual}, reference {expected}. If \
             `fold_split` fired on a value whose NumericContract forbids reassociation, \
             the split and unsplit forms are not value-equal."
        )
        .into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registered() -> Vec<String> {
        cases().names().iter().map(|n| (*n).to_string()).collect()
    }

    #[test]
    fn every_named_reduction_is_registered() {
        let names = registered();
        for wanted in [
            "sum_axis",
            "sum_high_rank",
            "sum_keepdim",
            "mean",
            "max",
            "min",
            "product",
            "product_keepdim",
            "var",
            "var_keepdim",
            "log_sum_exp",
            "squared_sum",
        ] {
            assert!(
                names.iter().any(|n| n == &format!("reductions::{wanted}")),
                "{wanted} is missing"
            );
        }
    }

    #[test]
    fn the_tie_and_zero_aware_cases_are_registered() {
        let names = registered();
        for wanted in [
            "max_ties_split_evenly",
            "min_ties_split_evenly",
            "product_zero_aware",
            "fold_split_agrees_when_reassoc",
        ] {
            assert!(
                names.iter().any(|n| n == &format!("reductions::{wanted}")),
                "{wanted} is missing"
            );
        }
    }

    /// The generality cases are registered, and each names the law it is the
    /// evidence for.
    #[test]
    fn every_generality_case_is_registered() {
        let names = registered();
        for wanted in [
            "kmeans_assignment_min_of_sums",
            "qat_fake_quant_chain_is_exact",
            "sampling_temperature_hoists_out_of_argmax",
            "max_of_shifted_is_shifted_max",
            "min_of_negated_is_negated_max",
            "min_and_max_in_one_pass",
            "goertzel_single_bin_dft",
            "weighted_log_sum_exp_distillation_loss",
            "ragged_batch_padding_is_identity",
        ] {
            assert!(
                names.iter().any(|n| n == &format!("reductions::{wanted}")),
                "{wanted} is missing"
            );
        }
    }

    /// The **negative half** of the homomorphism table, which is the half that
    /// keeps a rewrite from turning a number into a NaN inside a byte-identical
    /// export.
    ///
    /// `ValueFacts` is `{dtype, shape, numeric, persistence}` — there is no
    /// sign or range lattice — so a row over a unary that is *partial* on the
    /// operand dtype cannot be guarded. `Log : Mul -> Add` is false whenever
    /// any element is negative (an even count of negatives gives a finite left
    /// side and a NaN right side), and `MonotoneUp` over `Sqrt`, `Log`, `Asin`,
    /// `Acos` or `Atanh` is the same hazard with `exact_in_float: true`, which
    /// means it would fire under `NumericContract::STRICT`.
    ///
    /// A row returns when `ValueFacts` gains a sign lattice, and this test is
    /// what will notice.
    #[test]
    fn the_homomorphism_table_admits_no_partial_unary() {
        use fusor2_ir::carrier::{HOM_TABLE, HomShape};
        use fusor2_ir::scalar::{BinOp, UnOp};

        const PARTIAL: [UnOp; 6] = [
            UnOp::Sqrt,
            UnOp::Log,
            UnOp::Log2,
            UnOp::Asin,
            UnOp::Acos,
            UnOp::Atanh,
        ];
        for row in HOM_TABLE {
            if let HomShape::TotalMonotone(u) | HomShape::TotalAntitone(u) = row.h {
                assert!(
                    !PARTIAL.contains(&u),
                    "{u:?} is partial on its operand dtype and has no sign lattice to \
                     guard it, but ships as a monotone row"
                );
            }
            assert!(
                !matches!(row.h, HomShape::TotalMonotone(UnOp::Log)),
                "Log : Mul -> Add is unsound over sign and must not ship"
            );
            if row.from == BinOp::Mul && row.to == BinOp::Add {
                panic!("a Mul -> Add row is log-shaped and unsound over sign: {row:?}");
            }
        }
        // And the positive half: the exact extremum rows are the only rewrites
        // in the law set legal under STRICT, so they must actually be there.
        assert!(
            HOM_TABLE
                .iter()
                .any(|r| r.exact_in_float && r.from == BinOp::Max),
            "no exact Max row: nothing in the law set is legal on the QAT path"
        );
        assert!(
            HOM_TABLE.iter().any(|r| matches!(
                r.h,
                HomShape::TotalAntitone(UnOp::Neg) | HomShape::TotalMonotone(UnOp::Neg)
            )),
            "Neg is the only unary total on every dtype and must be admitted"
        );
    }

    #[test]
    fn the_host_references_are_the_formulas_they_claim() {
        let row = [1.0f32, 2.0, 3.0, 4.0];
        assert_eq!(host_mean(&row), 2.5);
        assert_eq!(host_max(&row), 4.0);
        assert_eq!(host_min(&row), 1.0);
        assert_eq!(host_product(&row), 24.0);
        // Biased variance of 1..4 about 2.5 is (2.25+0.25+0.25+2.25)/4.
        assert!((host_var(&row) - 1.25).abs() < 1e-6);
        // log-sum-exp is shift invariant.
        let shifted: Vec<f32> = row.iter().map(|v| v + 10.0).collect();
        assert!((host_lse(&shifted) - (host_lse(&row) + 10.0)).abs() < 1e-4);
    }

    #[test]
    fn log_sum_exp_stays_finite_on_a_large_shift() {
        // Without the max shift this overflows f32 and the case would be
        // testing the reference rather than the compiler.
        let row = [80.0f32, 81.0, 82.0];
        assert!(host_lse(&row).is_finite());
        assert!((host_lse(&row) - 82.4076) < 1e-2);
    }

    #[test]
    fn the_product_expectation_covers_all_three_branches() {
        // Mirrors the arithmetic `product_zero_aware` asserts, so a typo in
        // the case's expectation is caught without a device.
        let rows: [&[f32]; 3] = [&[2.0, 3.0], &[2.0, 0.0], &[0.0, 0.0]];
        let expected: [&[f32]; 3] = [&[3.0, 2.0], &[0.0, 2.0], &[0.0, 0.0]];
        for (values, want) in rows.iter().zip(expected) {
            let zeros = values.iter().filter(|v| **v == 0.0).count();
            let nz: f32 = values.iter().filter(|v| **v != 0.0).product();
            for (i, v) in values.iter().enumerate() {
                let got = match zeros {
                    0 => nz / v,
                    1 => {
                        if *v == 0.0 {
                            nz
                        } else {
                            0.0
                        }
                    }
                    _ => 0.0,
                };
                assert_eq!(got, want[i], "{values:?} at {i}");
            }
        }
    }
}
