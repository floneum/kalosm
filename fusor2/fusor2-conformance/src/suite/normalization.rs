//! softmax x4, rms_norm x3, layer_norm x2, plus the shift-stabilized and
//! stable-variance carriers.
//!
//! Every normalization here is a row program over the last axis: a max fold, a
//! map, a sum fold and a divide.
//!
//! The counts dropped when the `*_fused` aliases and `softmax_slow*` were
//! deleted: each was a one-line delegation into the same e-class, so its case
//! only ever re-tested the twin beside it. That the macro node and its `defn`
//! agree is the `FUSOR2_VERIFY_MEMBERS` sweep's job, and it sweeps every
//! class, not this one.
//!
//! Owned by W14.

use fusor2::{Dtype, Session, };
use fusor2::tensor::Dyn as Tensor;

use crate::compare::{assert_gradient_matches_finite_difference, finite_difference_gradient};
use crate::harness::{CaseError, CaseResult, Cases, dims};
use crate::suite::support::{
    Domain, expect_values, gradient_of, graph_of, loss_of, read, read_scalar, upload,
};

/// `[rows, width]`. Small enough that finite differences stay cheap, wide
/// enough that a row fold is not a single lane.
const ROWS: usize = 3;
const WIDTH: usize = 5;
const SHAPE: &[u64] = &[ROWS as u64, WIDTH as u64];
const LEN: usize = ROWS * WIDTH;

/// The eps every norm case uses. Large enough to matter at these magnitudes,
/// so a case that silently drops it fails rather than passing by luck.
const EPS: f32 = 1e-3;

type Build = fn(&Tensor) -> fusor2::Result<Tensor>;
/// A host reference over one row, producing that row's output.
type RowRef = fn(&[f32]) -> Vec<f32>;

// ---------------------------------------------------------------------------
// Host references
// ---------------------------------------------------------------------------

fn host_softmax(row: &[f32]) -> Vec<f32> {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = row.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = e.iter().sum();
    e.into_iter().map(|v| v / sum).collect()
}

fn host_log_softmax(row: &[f32]) -> Vec<f32> {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = row.iter().map(|v| (v - max).exp()).sum();
    let lse = max + sum.ln();
    row.iter().map(|v| v - lse).collect()
}

/// `x / sqrt(mean(x^2) + eps)`, the weightless rms_norm.
fn host_rms(row: &[f32]) -> Vec<f32> {
    let ms: f32 = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
    let inv = 1.0 / (ms + EPS).sqrt();
    row.iter().map(|v| v * inv).collect()
}

/// `(x - mean) / sqrt(var + eps)`, the bias-free, weight-free layer_norm.
fn host_layer(row: &[f32], remove_mean: bool) -> Vec<f32> {
    let mean = if remove_mean {
        row.iter().sum::<f32>() / row.len() as f32
    } else {
        0.0
    };
    let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / row.len() as f32;
    let inv = 1.0 / (var + EPS).sqrt();
    row.iter().map(|v| (v - mean) * inv).collect()
}

fn host_layer_centered(row: &[f32]) -> Vec<f32> {
    host_layer(row, true)
}

fn host_layer_uncentered(row: &[f32]) -> Vec<f32> {
    host_layer(row, false)
}

/// The whole-tensor reference: apply `row` to each row of `[ROWS, WIDTH]`.
fn by_row(data: &[f32], row: RowRef) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len());
    for r in data.chunks(WIDTH) {
        out.extend(row(r));
    }
    out
}

/// The weight/bias affine the fused spellings apply after normalizing.
fn affine(normalized: &[f32], weight: &[f32], bias: Option<&[f32]>) -> Vec<f32> {
    normalized
        .iter()
        .enumerate()
        .map(|(i, v)| v * weight[i % WIDTH] + bias.map_or(0.0, |b| b[i % WIDTH]))
        .collect()
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

/// The single-input row programs: forward against a host reference, backward
/// against central differences.
#[rustfmt::skip]
fn plain_rows() -> Vec<(&'static str, Build, RowRef)> {
    vec![
        ("softmax_axis_last",       |x| x.softmax(1),              host_softmax),
        ("softmax_last_dim",        |x| x.softmax_last_dim(),      host_softmax),
        ("log_softmax",             |x| x.log_softmax(1),          host_log_softmax),
        ("rms_norm_no_weight",      |x| x.rms_norm_no_weight(EPS), host_rms),
        ("layer_norm_centered",     |x| layer_norm_bare(x, true),  host_layer_centered),
        ("layer_norm_uncentered",   |x| layer_norm_bare(x, false), host_layer_uncentered),
    ]
}

/// `layer_norm` with an all-ones weight and no bias, so the host reference is
/// the bare statistic. The weight is a constant leaf, not a parameter: this
/// row checks the normalization, `layer_norm_fused` below checks the affine.
fn layer_norm_bare(x: &Tensor, remove_mean: bool) -> fusor2::Result<Tensor> {
    let ones = Tensor::ones(x.graph(), Dtype::F32, &dims(&[WIDTH as u64]))?;
    x.layer_norm(&ones, None, EPS, remove_mean)
}

pub fn cases() -> Cases {
    let mut cases = Cases::new();

    for (name, build, reference) in plain_rows() {
        cases.push("normalization", name, move |session| {
            row_case(session, name, build, reference)
        });
    }

    // The weighted spellings. Each is checked against `normalized * w (+ b)`
    // with a *non-constant* weight, so a lowering that drops the affine is a
    // value failure rather than a no-op.
    cases.push("normalization", "rms_norm", |s| {
        weighted_case(s, "rms_norm", host_rms, false, |x, w, _| x.rms_norm(w, EPS))
    });
    cases.push("normalization", "rms_norm_with_bias", |s| {
        weighted_case(s, "rms_norm_with_bias", host_rms, true, |x, w, b| {
            x.rms_norm_with_bias(w, b.expect("bias"), EPS)
        })
    });
    cases.push("normalization", "layer_norm_fused", |s| {
        weighted_case(
            s,
            "layer_norm_fused",
            host_layer_centered,
            true,
            |x, w, b| x.layer_norm(w, b, EPS, true),
        )
    });
    cases.push("normalization", "layer_norm_no_bias", |s| {
        weighted_case(
            s,
            "layer_norm_no_bias",
            host_layer_centered,
            false,
            |x, w, _| x.layer_norm(w, None, EPS, true),
        )
    });

    cases.push("normalization", "rms_norm_residual", residual_case);
    cases.push("normalization", "variance_last", variance_case);
    cases.push("normalization", "softmax_rows_sum_to_one", rows_sum_to_one);
    cases.push(
        "normalization",
        "softmax_is_shift_invariant",
        shift_invariance,
    );
    cases.push(
        "normalization",
        "softmax_backward_is_the_analytic_jacobian",
        softmax_backward,
    );
    cases.push(
        "normalization",
        "welford_agrees_with_the_two_pass_variance",
        welford_carrier,
    );
    cases.extend(structural::cases());
    cases
}

/// The structural half: which law actually fired on the chain the frontend
/// emits, and what the extracted plan did with it.
///
/// Every case above passes whether the program was rewritten or run naively —
/// a numeric oracle cannot tell a landed law from a dead one. These read the
/// saturation report and the extracted plan instead, on the *same* frontend
/// calls, so a law that silently stops matching reports a rule name and a
/// count rather than nothing at all.
mod structural {
    use fusor2::{Session, };
use fusor2::tensor::Dyn as Tensor;

    use crate::harness::{CaseError, CaseResult, Cases, dims};
    use crate::suite::probe::probe;
    use crate::suite::reductions::generality::structure;
    use crate::suite::support::{Domain, graph_of, upload};

    /// Wide enough that the row fold is a real reduction rather than a single
    /// lane, and small enough to stay cheap on both backends.
    const ROWS: u64 = 8;
    const WIDTH: u64 = 64;

    pub fn cases() -> Cases {
        let mut cases = Cases::new();
        cases.push(
            "normalization",
            "softmax_retargets_its_own_max",
            softmax_retargets,
        );
        cases.push(
            "normalization",
            "variance_absorbs_its_own_chain",
            variance_absorbs,
        );
        cases.push(
            "normalization",
            "layer_norm_backward_plan",
            layer_norm_backward_plan,
        );
        cases.push(
            "normalization",
            "rms_norm_backward_plan",
            rms_norm_backward_plan,
        );
        // The saturation tripwire, on a composed backward rather than on
        // attention: this is the one case that says whether the budget is big
        // enough for an ordinary program.
        cases.push(
            "normalization",
            "composed_backward_saturates",
            composed_backward_saturates,
        );
        cases
    }

    fn err(e: impl std::fmt::Display) -> CaseError {
        e.to_string().into()
    }

    /// `RETARGET` on the softmax the frontend writes.
    ///
    /// The law never invents a reference: it fires only where the source
    /// program already computed one, and `softmax_last_dim`'s `defn` computes
    /// a row max and reinjects it by a broadcast along the reduced axis. That
    /// is the structural condition, stated on address maps, and it is the same
    /// condition that makes the law fire on a CRF forward recursion or an MoE
    /// router — none of which is a softmax.
    ///
    /// The assert is on the **rule name**: a two-pass softmax computes the
    /// same numbers as a one-pass one, so no numeric case can distinguish
    /// them. `softmax_rows_sum_to_one` and `softmax_is_shift_invariant` above
    /// carry the values.
    fn softmax_retargets(session: &Session) -> CaseResult {
        let data = Domain::Wide.sample(801, (ROWS * WIDTH) as usize);
        let build = |s: &Session| -> Result<Vec<Tensor>, CaseError> {
            let g = graph_of(s);
            let x = upload(g.handle(), &dims(&[ROWS, WIDTH]), &data)?;
            Ok(vec![x.softmax_last_dim().map_err(err)?])
        };
        structure::must_fire(
            session,
            &build,
            &[
                (
                    "RETARGET",
                    "the running max is a reduction-carried dependence on another reduction \
                     over the same axis; discharging it is what makes softmax single-pass, \
                     and it is the same row that makes a KV-cache decode step single-pass \
                     at a symbolic length",
                ),
                (
                    "ABSORB",
                    "the exp and the subtract must reach the sum's lift; without it the \
                     shifted logits are a buffer",
                ),
            ],
        )
    }

    /// `ABSORB` on `variance`, which is `mean((x - mean)^2)`: two reductions
    /// over one axis with an elementwise chain between them.
    ///
    /// This is the shape `RETARGET`'s raw-moment row turns into Welford. The
    /// firing assert here is on `ABSORB`, which is landed; the plan count is a
    /// ceiling, because the one-pass form needs the raw-moment row and that
    /// row is not in `RETARGET_TABLE` yet.
    fn variance_absorbs(session: &Session) -> CaseResult {
        let data = Domain::Wide.sample(802, (ROWS * WIDTH) as usize);
        let build = |s: &Session| -> Result<Vec<Tensor>, CaseError> {
            let g = graph_of(s);
            let x = upload(g.handle(), &dims(&[ROWS, WIDTH]), &data)?;
            Ok(vec![x.variance(1).map_err(err)?])
        };
        structure::must_fire(
            session,
            &build,
            &[(
                "ABSORB",
                "the centring subtract and the square must ride into the second fold's \
                 lift; without it a two-pass variance is a three-launch program",
            )],
        )?;
        structure::plan_ceiling(
            session,
            &build,
            "variance_last",
            6,
            1,
            "RETARGET's raw-moment row at rho = the running mean (Welford, derived)",
        )
    }

    /// The composed layer-norm backward emits `sum(dy)` and `sum(dy * xhat)`
    /// over the same feature axis of the same operands.
    ///
    /// No rule mentions layer_norm, normalization or backward. `TUPLE` joins
    /// the two into one 2-slot fold and one launch — the fused
    /// layer-norm-backward kernel every framework hand-writes, derived by the
    /// same law as Welford. Until it does, this is a ceiling and the count is
    /// the diff the day it lands.
    fn layer_norm_backward_plan(session: &Session) -> CaseResult {
        backward_plan(session, "layer_norm_backward", true, 34)
    }

    /// The same law on `rms_norm`, whose backward emits one sum instead of
    /// two — so `TUPLE` has less to do and `ABSORB` more. Having both says
    /// which law the count belongs to.
    fn rms_norm_backward_plan(session: &Session) -> CaseResult {
        backward_plan(session, "rms_norm_backward", false, 28)
    }

    fn backward_plan(
        session: &Session,
        what: &'static str,
        centered: bool,
        launches: usize,
    ) -> CaseResult {
        let data = Domain::Wide.sample(803, (ROWS * WIDTH) as usize);
        let weight = Domain::Positive.sample(804, WIDTH as usize);

        let build = |s: &Session| -> Result<Vec<Tensor>, CaseError> {
            let g = graph_of(s);
            let x = upload(g.handle(), &dims(&[ROWS, WIDTH]), &data)?;
            let w = upload(g.handle(), &dims(&[WIDTH]), &weight)?;
            let y = if centered {
                x.layer_norm(&w, None, 1e-5, true)
            } else {
                x.rms_norm(&w, 1e-5)
            }
            .map_err(err)?;
            let loss = y.sum_all().map_err(err)?;
            let grads = g
                .backward_with(&loss, &[x.clone(), w.clone()])
                .map_err(err)?;
            let dx = grads
                .get(&x)
                .ok_or_else(|| -> CaseError { "no gradient reached x".into() })?;
            let dw = grads
                .get(&w)
                .ok_or_else(|| -> CaseError { "no gradient reached w".into() })?;
            Ok(vec![dx, dw])
        };

        // `composed_backward_saturates` owns the saturation claim; this case
        // owns the count. A count read off a truncated saturation is still a
        // regression guard, because the driver is deterministic.
        let p = structure::probe_fresh(session, &build)?;
        p.require_fired(
            "ABSORB",
            "the adjoint's elementwise chain must reach the feature-axis folds; without \
             it every term of the backward is its own launch",
        )?;
        structure::plan_ceiling(
            session,
            &build,
            what,
            launches,
            2,
            "TUPLE (the two feature-axis sums are one 2-slot fold) + ABSORB",
        )
    }

    /// The saturation budget, measured on a composed backward.
    ///
    /// `attention_rope::attention_defn_saturates` makes the same claim about
    /// attention, where it is easy to read as "the flash derivation is not
    /// finished". This says the wider thing: an eight-row layer-norm backward
    /// — five lines of frontend, no attention anywhere — also runs out of
    /// `MAX_ROUNDS`. The two failures have one cause and one fix, and the fix
    /// is not in attention.
    ///
    /// The value is checked first, so a graph that stopped computing cannot
    /// pass this by saturating trivially.
    fn composed_backward_saturates(session: &Session) -> CaseResult {
        let data = Domain::Wide.sample(806, (ROWS * WIDTH) as usize);
        // An all-ones weight, so `sum(y)` is `sum(xhat)` and the row is
        // centred: the sum is identically zero and so is its gradient.
        let weight = vec![1.0f32; WIDTH as usize];

        let build = |s: &Session| -> Result<Vec<Tensor>, CaseError> {
            let g = graph_of(s);
            let x = upload(g.handle(), &dims(&[ROWS, WIDTH]), &data)?;
            let w = upload(g.handle(), &dims(&[WIDTH]), &weight)?;
            let y = x.layer_norm(&w, None, 1e-5, true).map_err(err)?;
            let loss = y.sum_all().map_err(err)?;
            let grads = g.backward_with(&loss, &[x.clone()]).map_err(err)?;
            let dx = grads
                .get(&x)
                .ok_or_else(|| -> CaseError { "no gradient reached x".into() })?;
            Ok(vec![dx])
        };

        // `sum(layer_norm(x))` with a unit weight is `sum((x - mean)/sd)`,
        // which is identically zero for every input, so every entry of
        // `d(sum y)/dx` is zero to rounding. An independent number, and one a
        // broken adjoint cannot produce by accident.
        let outs = build(session)?;
        let dx = crate::suite::support::read(&outs[0])?;
        let scale = data.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1.0);
        if let Some((i, v)) = dx.iter().enumerate().find(|(_, v)| v.abs() > 2e-3 * scale) {
            return Err(format!(
                "d(sum(layer_norm(x)))/dx[{i}] = {v}, and the sum of a centred, scaled row                  does not move when the row shifts, so every entry must be zero"
            )
            .into());
        }

        let p = probe(session, &outs)?;
        p.require_saturated("layer_norm backward (five frontend lines, no attention anywhere)")
    }
}

/// Forward against the host row reference, then backward against central
/// differences.
fn row_case(session: &Session, name: &'static str, build: Build, reference: RowRef) -> CaseResult {
    let data = Domain::Wide.sample(401, LEN);
    let dimv = dims(SHAPE);

    let graph = graph_of(session);
    let x = upload(graph.handle(), &dimv, &data)?;
    let y = build(&x).map_err(|e| -> CaseError { format!("{name}: {e}").into() })?;

    let actual = read(&y)?;
    let expected = by_row(&data, reference);
    expect_values(session, SHAPE, Dtype::F32, &actual, &expected)?;

    let analytic = gradient_of(&graph, &y, &x)?;
    let numeric = finite_difference_gradient(&[ROWS, WIDTH], &data, &mut |probe| {
        let g = graph_of(session);
        let x = upload(g.handle(), &dimv, probe)?;
        let y = build(&x).map_err(|e| -> CaseError { e.to_string().into() })?;
        read_scalar(&loss_of(&y)?)
    })?;
    assert_gradient_matches_finite_difference(&analytic, &numeric)?;
    Ok(())
}

/// A norm with a learned weight and optional bias. All three gradients are
/// checked: dropping `d_weight` is the classic way a fused epilogue rule goes
/// wrong while the forward stays correct.
fn weighted_case(
    session: &Session,
    name: &'static str,
    normalize: RowRef,
    with_bias: bool,
    build: fn(&Tensor, &Tensor, Option<&Tensor>) -> fusor2::Result<Tensor>,
) -> CaseResult {
    let data = Domain::Wide.sample(409, LEN);
    // Weights away from 1 and biases away from 0, so an unapplied affine
    // cannot pass.
    let weight = Domain::Custom(0.5, 1.5).sample(419, WIDTH);
    let bias = Domain::Custom(-0.4, 0.4).sample(421, WIDTH);
    let dimv = dims(SHAPE);
    let wdim = dims(&[WIDTH as u64]);

    let graph = graph_of(session);
    let x = upload(graph.handle(), &dimv, &data)?;
    let w = upload(graph.handle(), &wdim, &weight)?;
    let b = with_bias
        .then(|| upload(graph.handle(), &wdim, &bias))
        .transpose()?;
    let y =
        build(&x, &w, b.as_ref()).map_err(|e| -> CaseError { format!("{name}: {e}").into() })?;

    let normalized = by_row(&data, normalize);
    let expected = affine(&normalized, &weight, with_bias.then_some(&bias[..]));
    expect_values(session, SHAPE, Dtype::F32, &read(&y)?, &expected)?;

    let d_x = gradient_of(&graph, &y, &x)?;
    let numeric = finite_difference_gradient(&[ROWS, WIDTH], &data, &mut |probe| {
        let g = graph_of(session);
        let x = upload(g.handle(), &dimv, probe)?;
        let w = upload(g.handle(), &wdim, &weight)?;
        let b = with_bias
            .then(|| upload(g.handle(), &wdim, &bias))
            .transpose()?;
        let y = build(&x, &w, b.as_ref()).map_err(|e| -> CaseError { e.to_string().into() })?;
        read_scalar(&loss_of(&y)?)
    })?;
    assert_gradient_matches_finite_difference(&d_x, &numeric)?;

    // d_weight[j] = sum over rows of normalized[r, j] — the stride-0 axis's
    // adjoint is a sum, and it is over the *rows*, not the columns.
    let d_w = gradient_of(&graph, &y, &w)?;
    let want_w: Vec<f32> = (0..WIDTH)
        .map(|j| (0..ROWS).map(|r| normalized[r * WIDTH + j]).sum())
        .collect();
    let backend = if crate::harness::is_gpu(session) {
        "gpu"
    } else {
        "cpu"
    };
    crate::compare::approx_or_relative_eq(backend, &[WIDTH], &want_w, &d_w, 1e-3, 1e-3)?;

    if let Some(b) = &b {
        // Every bias element is broadcast over ROWS rows, so its gradient is
        // exactly the row count under an all-ones seed.
        let d_b = gradient_of(&graph, &y, b)?;
        let want_b = vec![ROWS as f32; WIDTH];
        crate::compare::approx_or_relative_eq(backend, &[WIDTH], &want_b, &d_b, 1e-4, 1e-4)?;
    }
    Ok(())
}

/// The transformer block boundary: `rms_norm(x + residual) * w`. The residual
/// add must be inside the statistic, not applied to the normalized value.
fn residual_case(session: &Session) -> CaseResult {
    let data = Domain::Wide.sample(431, LEN);
    let residual = Domain::Wide.sample(433, LEN);
    let weight = Domain::Custom(0.5, 1.5).sample(439, WIDTH);
    let dimv = dims(SHAPE);
    let wdim = dims(&[WIDTH as u64]);

    let graph = graph_of(session);
    let x = upload(graph.handle(), &dimv, &data)?;
    let r = upload(graph.handle(), &dimv, &residual)?;
    let w = upload(graph.handle(), &wdim, &weight)?;
    let y = x
        .rms_norm_residual(&r, &w, None, EPS)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let summed: Vec<f32> = data.iter().zip(&residual).map(|(a, b)| a + b).collect();
    let expected = affine(&by_row(&summed, host_rms), &weight, None);
    expect_values(session, SHAPE, Dtype::F32, &read(&y)?, &expected)?;

    // Both inputs enter the same sum, so their gradients must be identical —
    // a rule that normalizes before adding gives the residual a different one.
    let d_x = gradient_of(&graph, &y, &x)?;
    let d_r = gradient_of(&graph, &y, &r)?;
    let backend = if crate::harness::is_gpu(session) {
        "gpu"
    } else {
        "cpu"
    };
    crate::compare::approx_or_relative_eq(backend, &[LEN], &d_x, &d_r, 1e-4, 1e-3)?;
    Ok(())
}

/// `variance_last` as the statistic, against the two-pass host formula.
fn variance_case(session: &Session) -> CaseResult {
    let data = Domain::Wide.sample(443, LEN);
    let dimv = dims(SHAPE);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dimv, &data)?;
    let y = x
        .variance_last()
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let expected: Vec<f32> = data
        .chunks(WIDTH)
        .map(|row| {
            let m = row.iter().sum::<f32>() / WIDTH as f32;
            row.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / WIDTH as f32
        })
        .collect();
    expect_values(session, &[ROWS as u64], Dtype::F32, &read(&y)?, &expected)?;

    let analytic = gradient_of(&graph, &y, &x)?;
    let numeric = finite_difference_gradient(&[ROWS, WIDTH], &data, &mut |probe| {
        let g = graph_of(session);
        let x = upload(g.handle(), &dimv, probe)?;
        let y = x
            .variance_last()
            .map_err(|e| -> CaseError { e.to_string().into() })?;
        read_scalar(&loss_of(&y)?)
    })?;
    assert_gradient_matches_finite_difference(&analytic, &numeric)?;
    Ok(())
}

/// Every softmax row sums to exactly 1 within tolerance. Cheap, but it is the
/// invariant an online-softmax carrier with a mis-rescaled running sum breaks
/// while still looking plausible element by element.
fn rows_sum_to_one(session: &Session) -> CaseResult {
    let data = Domain::Custom(-4.0, 4.0).sample(449, LEN);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(SHAPE), &data)?;
    let p = x
        .softmax_last_dim()
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let got = read(&p)?;
    for (r, row) in got.chunks(WIDTH).enumerate() {
        let sum: f32 = row.iter().sum();
        if (sum - 1.0).abs() > 1e-4 {
            return Err(format!("softmax row {r} sums to {sum}, not 1").into());
        }
        if let Some(v) = row.iter().find(|v| **v < 0.0) {
            return Err(format!("softmax row {r} has a negative probability {v}").into());
        }
    }
    Ok(())
}

/// softmax(x + c) == softmax(x). The max fold is the only thing that makes
/// this true, so a lowering that drops it fails here before it overflows.
fn shift_invariance(session: &Session) -> CaseResult {
    let data = Domain::Custom(-2.0, 2.0).sample(457, LEN);
    let shifted: Vec<f32> = data.iter().map(|v| v + 60.0).collect();
    let graph = graph_of(session);
    let a = upload(graph.handle(), &dims(SHAPE), &data)?;
    let b = upload(graph.handle(), &dims(SHAPE), &shifted)?;
    let pa = a
        .softmax_last_dim()
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let pb = b
        .softmax_last_dim()
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let (va, vb) = (read(&pa)?, read(&pb)?);
    if vb.iter().any(|v| !v.is_finite()) {
        return Err("softmax overflowed on a +60 shift: the max fold was elided".into());
    }
    expect_values(session, SHAPE, Dtype::F32, &vb, &va)?;
    Ok(())
}

/// `dS = P * (dP - rowsum(dP * P))`.
///
/// Seeded with a non-uniform upstream gradient: under `sum_all` the softmax
/// adjoint is identically zero, so an all-ones seed cannot tell a correct
/// Jacobian from a missing one.
fn softmax_backward(session: &Session) -> CaseResult {
    let data = Domain::Wide.sample(463, LEN);
    // A fixed, non-uniform upstream weight, applied as `sum(w * softmax(x))`.
    let weights = Domain::Custom(0.25, 2.0).sample(467, LEN);
    let dimv = dims(SHAPE);

    let build = |x: &Tensor, w: &Tensor| -> fusor2::Result<Tensor> { x.softmax_last_dim()?.mul(w) };

    let graph = graph_of(session);
    let x = upload(graph.handle(), &dimv, &data)?;
    let w = upload(graph.handle(), &dimv, &weights)?;
    let y = build(&x, &w).map_err(|e| -> CaseError { e.to_string().into() })?;
    let analytic = gradient_of(&graph, &y, &x)?;

    // Host Jacobian-vector product, row by row.
    let mut expected = vec![0.0f32; LEN];
    for r in 0..ROWS {
        let p = host_softmax(&data[r * WIDTH..(r + 1) * WIDTH]);
        let dp = &weights[r * WIDTH..(r + 1) * WIDTH];
        let dot: f32 = p.iter().zip(dp).map(|(a, b)| a * b).sum();
        for j in 0..WIDTH {
            expected[r * WIDTH + j] = p[j] * (dp[j] - dot);
        }
    }
    let backend = if crate::harness::is_gpu(session) {
        "gpu"
    } else {
        "cpu"
    };
    crate::compare::approx_or_relative_eq(
        backend,
        &[ROWS, WIDTH],
        &expected,
        &analytic,
        1e-4,
        1e-3,
    )?;
    Ok(())
}

/// `mean((x - mean)^2)` computed by `variance_last` must agree with the
/// `mean(x^2) - mean(x)^2` spelling to f32 tolerance on well-conditioned data,
/// and the composed form is the one autograd differentiates.
fn welford_carrier(session: &Session) -> CaseResult {
    let data = Domain::Custom(10.0, 11.0).sample(479, LEN);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(SHAPE), &data)?;

    let welford = x
        .variance_last()
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let naive = x
        .square()
        .and_then(|s| s.mean(1))
        .and_then(|ms| {
            let m = x.mean(1)?;
            ms.sub(&m.square()?)
        })
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let (a, b) = (read(&welford)?, read(&naive)?);
    // The naive form loses precision at mean ~10.5, so the bar is relative to
    // the variance itself, not absolute.
    let backend = if crate::harness::is_gpu(session) {
        "gpu"
    } else {
        "cpu"
    };
    crate::compare::approx_or_relative_eq(backend, &[ROWS], &a, &b, 1e-3, 1e-2)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registered() -> Vec<String> {
        cases().names().iter().map(|n| (*n).to_string()).collect()
    }

    #[test]
    fn all_four_softmax_spellings_are_registered() {
        let names = registered();
        for wanted in [
            "softmax_axis_last",
            "softmax_last_dim",
        ] {
            assert!(
                names
                    .iter()
                    .any(|n| n == &format!("normalization::{wanted}")),
                "{wanted} is missing"
            );
        }
    }

    #[test]
    fn all_rms_norm_and_layer_norm_spellings_are_registered() {
        let names = registered();
        for wanted in [
            "rms_norm",
            "rms_norm_no_weight",
            "rms_norm_with_bias",
            "rms_norm_residual",
            "layer_norm_centered",
            "layer_norm_uncentered",
            "layer_norm_fused",
        ] {
            assert!(
                names
                    .iter()
                    .any(|n| n == &format!("normalization::{wanted}")),
                "{wanted} is missing"
            );
        }
    }

    #[test]
    fn the_carriers_and_the_analytic_backward_are_registered() {
        let names = registered();
        for wanted in [
            "softmax_backward_is_the_analytic_jacobian",
            "welford_agrees_with_the_two_pass_variance",
            "softmax_rows_sum_to_one",
        ] {
            assert!(
                names
                    .iter()
                    .any(|n| n == &format!("normalization::{wanted}")),
                "{wanted} is missing"
            );
        }
    }

    #[test]
    fn the_host_softmax_is_a_distribution() {
        let p = host_softmax(&[1.0, 2.0, 3.0]);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        // Monotone in the input.
        assert!(p[0] < p[1] && p[1] < p[2]);
        // Shift invariance, which is what the max fold buys.
        let q = host_softmax(&[101.0, 102.0, 103.0]);
        for (a, b) in p.iter().zip(&q) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn the_host_log_softmax_is_the_log_of_the_host_softmax() {
        let row = [0.5f32, -1.0, 2.0, 0.0];
        for (l, p) in host_log_softmax(&row).iter().zip(host_softmax(&row)) {
            assert!((l - p.ln()).abs() < 1e-5, "{l} vs ln({p})");
        }
    }

    #[test]
    fn the_host_norms_are_the_formulas_they_claim() {
        // rms_norm of a constant row is that constant over its own magnitude.
        let row = [2.0f32, 2.0, 2.0, 2.0];
        let want = 2.0 / (4.0f32 + EPS).sqrt();
        for v in host_rms(&row) {
            assert!((v - want).abs() < 1e-6, "{v} vs {want}");
        }
        // A centered layer_norm has zero mean and unit variance up to eps.
        let row = [1.0f32, 2.0, 3.0, 6.0];
        let out = host_layer_centered(&row);
        assert!(out.iter().sum::<f32>().abs() < 1e-5);
        let var = out.iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert!((var - 1.0).abs() < 1e-2, "{var}");
        // Uncentered keeps the mean in the statistic.
        assert_ne!(host_layer_uncentered(&row), out);
    }

    #[test]
    fn by_row_keeps_rows_independent() {
        let data: Vec<f32> = (0..LEN).map(|i| i as f32).collect();
        let out = by_row(&data, host_softmax);
        assert_eq!(out.len(), LEN);
        for row in out.chunks(WIDTH) {
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn the_affine_is_applied_per_column() {
        let normalized = vec![1.0f32; LEN];
        let weight: Vec<f32> = (0..WIDTH).map(|j| j as f32).collect();
        let bias = vec![0.5f32; WIDTH];
        let out = affine(&normalized, &weight, Some(&bias));
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, (i % WIDTH) as f32 + 0.5);
        }
    }
}
