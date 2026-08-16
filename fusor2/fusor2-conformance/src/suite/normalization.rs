//! softmax x4, rms_norm x3, layer_norm x2, plus the shift-stabilized and
//! stable-variance carriers.
//!
//! Every normalization here is a row program over the last axis: a max fold, a
//! map, a sum fold and a divide.

use fusor2::tensor::Dyn as Tensor;
use fusor2::{Dtype, Session};

use crate::compare::{assert_gradient_matches_finite_difference, finite_difference_gradient};
use crate::harness::{CaseError, CaseResult, Cases, FuzzDim, dims, fuzz_case};
use crate::suite::support::{
    Domain, expect_values, gradient_of, graph_of, loss_of, read, read_probe_loss, upload,
};

/// `[rows, width]` for the finite-difference-backed cases. The ceiling stays
/// modest because every input element is perturbed; width starts at 2 so a
/// row fold is never a single lane.
const FD_SPEC: &[FuzzDim] = &[FuzzDim::Range(1, 6), FuzzDim::Range(2, 32)];

/// Forward-only invariants can afford multi-workgroup rows.
const FWD_SPEC: &[FuzzDim] = &[FuzzDim::Range(1, 64), FuzzDim::Range(2, 256)];

/// One backward, no FD, so width can grow past a workgroup.
const BWD_SPEC: &[FuzzDim] = &[FuzzDim::Range(1, 8), FuzzDim::Range(2, 128)];

/// The precision carrier wants a long fold axis, not many rows.
const WELFORD_SPEC: &[FuzzDim] = &[FuzzDim::Range(1, 4), FuzzDim::Range(64, 512)];

/// The eps every norm case uses. Large enough to matter at these magnitudes,
/// so a case that silently drops it fails rather than passing by luck.
const EPS: f32 = 1e-3;

type Build = fn(&Tensor, u64) -> fusor2::Result<Tensor>;
/// A host reference over one row, producing that row's output.
type RowRef = fn(&[f32]) -> Vec<f32>;

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

/// The whole-tensor reference: apply `row` to each `width`-wide row.
fn by_row(data: &[f32], width: usize, row: RowRef) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len());
    for r in data.chunks(width) {
        out.extend(row(r));
    }
    out
}

/// The weight/bias affine the fused spellings apply after normalizing.
fn affine(normalized: &[f32], weight: &[f32], bias: Option<&[f32]>) -> Vec<f32> {
    let width = weight.len();
    normalized
        .iter()
        .enumerate()
        .map(|(i, v)| v * weight[i % width] + bias.map_or(0.0, |b| b[i % width]))
        .collect()
}

/// The single-input row programs: forward against a host reference, backward
/// against central differences.
#[rustfmt::skip]
fn plain_rows() -> Vec<(&'static str, Build, RowRef)> {
    vec![
        ("softmax_axis_last",       |x, _| x.softmax(1),              host_softmax),
        ("softmax_last_dim",        |x, _| x.softmax_last_dim(),      host_softmax),
        ("log_softmax",             |x, _| x.log_softmax(1),          host_log_softmax),
        ("rms_norm_no_weight",      |x, _| x.rms_norm_no_weight(EPS), host_rms),
        ("layer_norm_centered",     |x, w| layer_norm_bare(x, w, true),  host_layer_centered),
        ("layer_norm_uncentered",   |x, w| layer_norm_bare(x, w, false), host_layer_uncentered),
    ]
}

/// `layer_norm` with an all-ones weight and no bias, so the host reference is
/// the bare statistic. The weight is a constant leaf, not a parameter: this
/// row checks the normalization, `layer_norm_fused` below checks the affine.
fn layer_norm_bare(x: &Tensor, width: u64, remove_mean: bool) -> fusor2::Result<Tensor> {
    let ones = Tensor::ones(x.graph(), Dtype::F32, &dims(&[width]))?;
    x.layer_norm(&ones, None, EPS, remove_mean)
}

pub fn cases() -> Cases {
    let mut cases = Cases::new();

    for (name, build, reference) in plain_rows() {
        cases.push_case(fuzz_case(
            "normalization",
            name,
            FD_SPEC,
            move |s, shape, seed| row_case(s, shape, seed, name, build, reference),
        ));
    }

    // The weighted spellings. Each is checked against `normalized * w (+ b)`
    // with a *non-constant* weight, so a lowering that drops the affine is a
    // value failure rather than a no-op.
    cases.push_case(fuzz_case(
        "normalization",
        "rms_norm",
        FD_SPEC,
        |s, shape, seed| {
            weighted_case(s, shape, seed, "rms_norm", host_rms, false, |x, w, _| {
                x.rms_norm(w, EPS)
            })
        },
    ));
    cases.push_case(fuzz_case(
        "normalization",
        "rms_norm_with_bias",
        FD_SPEC,
        |s, shape, seed| {
            weighted_case(
                s,
                shape,
                seed,
                "rms_norm_with_bias",
                host_rms,
                true,
                |x, w, b| x.rms_norm_with_bias(w, b.expect("bias"), EPS),
            )
        },
    ));
    cases.push_case(fuzz_case(
        "normalization",
        "layer_norm_fused",
        FD_SPEC,
        |s, shape, seed| {
            weighted_case(
                s,
                shape,
                seed,
                "layer_norm_fused",
                host_layer_centered,
                true,
                |x, w, b| x.layer_norm(w, b, EPS, true),
            )
        },
    ));
    cases.push_case(fuzz_case(
        "normalization",
        "layer_norm_no_bias",
        FD_SPEC,
        |s, shape, seed| {
            weighted_case(
                s,
                shape,
                seed,
                "layer_norm_no_bias",
                host_layer_centered,
                false,
                |x, w, _| x.layer_norm(w, None, EPS, true),
            )
        },
    ));

    cases.push_case(fuzz_case(
        "normalization",
        "rms_norm_residual",
        BWD_SPEC,
        residual_case,
    ));
    cases.push_case(fuzz_case(
        "normalization",
        "variance_last",
        FD_SPEC,
        variance_case,
    ));
    cases.push_case(fuzz_case(
        "normalization",
        "softmax_rows_sum_to_one",
        FWD_SPEC,
        rows_sum_to_one,
    ));
    cases.push_case(fuzz_case(
        "normalization",
        "softmax_is_shift_invariant",
        FWD_SPEC,
        shift_invariance,
    ));
    cases.push_case(fuzz_case(
        "normalization",
        "softmax_backward_is_the_analytic_jacobian",
        BWD_SPEC,
        softmax_backward,
    ));
    cases.push_case(fuzz_case(
        "normalization",
        "welford_agrees_with_the_two_pass_variance",
        WELFORD_SPEC,
        welford_carrier,
    ));
    cases.push_case(fuzz_case(
        "normalization",
        "layer_norm_sum_gradient_is_zero",
        BWD_SPEC,
        layer_norm_sum_gradient_is_zero,
    ));
    cases
}

/// `sum(layer_norm(x))` with a unit weight is `sum((x - mean)/sd)`, which is
/// identically zero for every input — so every entry of `d(sum y)/dx` must be
/// zero to rounding. An independent number a broken adjoint cannot produce by
/// accident.
fn layer_norm_sum_gradient_is_zero(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (rows, width) = (shape[0], shape[1]);
    let data = Domain::Wide.sample(seed, (rows * width) as usize);
    let weight = vec![1.0f32; width as usize];

    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(&[rows, width]), &data)?;
    let w = upload(graph.handle(), &dims(&[width]), &weight)?;
    let y = x
        .layer_norm(&w, None, 1e-5, true)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let loss = y
        .sum_all()
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let grads = graph
        .backward_with(&loss, std::slice::from_ref(&x))
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let dx = grads
        .get(&x)
        .ok_or_else(|| -> CaseError { "no gradient reached x".into() })?;
    let dx = read(&dx)?;
    let scale = data.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1.0);
    if let Some((i, v)) = dx.iter().enumerate().find(|(_, v)| v.abs() > 2e-3 * scale) {
        return Err(format!(
            "d(sum(layer_norm(x)))/dx[{i}] = {v}, and the sum of a centred, scaled row \
             does not move when the row shifts, so every entry must be zero"
        )
        .into());
    }
    Ok(())
}

/// Forward against the host row reference, then backward against central
/// differences.
fn row_case(
    session: &Session,
    shape: &[u64],
    seed: u32,
    name: &'static str,
    build: Build,
    reference: RowRef,
) -> CaseResult {
    let (rows, width) = (shape[0] as usize, shape[1] as usize);
    let data = Domain::Wide.sample(seed, rows * width);
    let dimv = dims(shape);

    let graph = graph_of(session);
    let x = upload(graph.handle(), &dimv, &data)?;
    let y = build(&x, width as u64).map_err(|e| -> CaseError { format!("{name}: {e}").into() })?;

    let actual = read(&y)?;
    let expected = by_row(&data, width, reference);
    expect_values(session, shape, Dtype::F32, &actual, &expected)?;

    let analytic = gradient_of(&graph, &y, &x)?;
    let probe_graph = graph_of(session);
    let probe_x = upload(probe_graph.handle(), &dimv, &data)?;
    let probe_y =
        build(&probe_x, width as u64).map_err(|e| -> CaseError { e.to_string().into() })?;
    let probe_loss = loss_of(&probe_y)?;
    let numeric = finite_difference_gradient(&[rows, width], &data, &mut |probe| {
        read_probe_loss(&probe_x, &probe_loss, probe)
    })?;
    assert_gradient_matches_finite_difference(&analytic, &numeric)?;
    Ok(())
}

/// A norm with a learned weight and optional bias. All three gradients are
/// checked: dropping `d_weight` is the classic way a fused epilogue rule goes
/// wrong while the forward stays correct.
fn weighted_case(
    session: &Session,
    shape: &[u64],
    seed: u32,
    name: &'static str,
    normalize: RowRef,
    with_bias: bool,
    build: fn(&Tensor, &Tensor, Option<&Tensor>) -> fusor2::Result<Tensor>,
) -> CaseResult {
    let (rows, width) = (shape[0] as usize, shape[1] as usize);
    let data = Domain::Wide.sample(seed, rows * width);
    // Weights away from 1 and biases away from 0, so an unapplied affine
    // cannot pass.
    let weight = Domain::Custom(0.5, 1.5).sample(seed ^ 0x9e37_79b9, width);
    let bias = Domain::Custom(-0.4, 0.4).sample(seed.wrapping_add(1), width);
    let dimv = dims(shape);
    let wdim = dims(&[width as u64]);

    let graph = graph_of(session);
    let x = upload(graph.handle(), &dimv, &data)?;
    let w = upload(graph.handle(), &wdim, &weight)?;
    let b = with_bias
        .then(|| upload(graph.handle(), &wdim, &bias))
        .transpose()?;
    let y =
        build(&x, &w, b.as_ref()).map_err(|e| -> CaseError { format!("{name}: {e}").into() })?;

    let normalized = by_row(&data, width, normalize);
    let expected = affine(&normalized, &weight, with_bias.then_some(&bias[..]));
    expect_values(session, shape, Dtype::F32, &read(&y)?, &expected)?;

    let d_x = gradient_of(&graph, &y, &x)?;
    let probe_graph = graph_of(session);
    let probe_x = upload(probe_graph.handle(), &dimv, &data)?;
    let probe_w = upload(probe_graph.handle(), &wdim, &weight)?;
    let probe_b = with_bias
        .then(|| upload(probe_graph.handle(), &wdim, &bias))
        .transpose()?;
    let probe_y = build(&probe_x, &probe_w, probe_b.as_ref())
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let probe_loss = loss_of(&probe_y)?;
    let numeric = finite_difference_gradient(&[rows, width], &data, &mut |probe| {
        read_probe_loss(&probe_x, &probe_loss, probe)
    })?;
    assert_gradient_matches_finite_difference(&d_x, &numeric)?;

    // d_weight[j] = sum over rows of normalized[r, j] — the stride-0 axis's
    // adjoint is a sum, and it is over the *rows*, not the columns.
    let d_w = gradient_of(&graph, &y, &w)?;
    let want_w: Vec<f32> = (0..width)
        .map(|j| (0..rows).map(|r| normalized[r * width + j]).sum())
        .collect();
    let backend = if crate::harness::is_gpu(session) {
        "gpu"
    } else {
        "cpu"
    };
    crate::compare::approx_or_relative_eq(backend, &[width], &want_w, &d_w, 1e-3, 1e-3)?;

    if let Some(b) = &b {
        // Every bias element is broadcast over the rows, so its gradient is
        // exactly the row count under an all-ones seed.
        let d_b = gradient_of(&graph, &y, b)?;
        let want_b = vec![rows as f32; width];
        crate::compare::approx_or_relative_eq(backend, &[width], &want_b, &d_b, 1e-4, 1e-4)?;
    }
    Ok(())
}

/// The transformer block boundary: `rms_norm(x + residual) * w`. The residual
/// add must be inside the statistic, not applied to the normalized value.
fn residual_case(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (rows, width) = (shape[0] as usize, shape[1] as usize);
    let len = rows * width;
    let data = Domain::Wide.sample(seed, len);
    let residual = Domain::Wide.sample(seed ^ 0x9e37_79b9, len);
    let weight = Domain::Custom(0.5, 1.5).sample(seed.wrapping_add(1), width);
    let dimv = dims(shape);
    let wdim = dims(&[width as u64]);

    let graph = graph_of(session);
    let x = upload(graph.handle(), &dimv, &data)?;
    let r = upload(graph.handle(), &dimv, &residual)?;
    let w = upload(graph.handle(), &wdim, &weight)?;
    let y = x
        .rms_norm_residual(&r, &w, None, EPS)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let summed: Vec<f32> = data.iter().zip(&residual).map(|(a, b)| a + b).collect();
    let expected = affine(&by_row(&summed, width, host_rms), &weight, None);
    expect_values(session, shape, Dtype::F32, &read(&y)?, &expected)?;

    // Both inputs enter the same sum, so their gradients must be identical —
    // a rule that normalizes before adding gives the residual a different one.
    let d_x = gradient_of(&graph, &y, &x)?;
    let d_r = gradient_of(&graph, &y, &r)?;
    let backend = if crate::harness::is_gpu(session) {
        "gpu"
    } else {
        "cpu"
    };
    crate::compare::approx_or_relative_eq(backend, &[len], &d_x, &d_r, 1e-4, 1e-3)?;
    Ok(())
}

/// `variance_last` as the statistic, against the two-pass host formula.
fn variance_case(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (rows, width) = (shape[0] as usize, shape[1] as usize);
    let data = Domain::Wide.sample(seed, rows * width);
    let dimv = dims(shape);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dimv, &data)?;
    let y = x
        .variance_last()
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let expected: Vec<f32> = data
        .chunks(width)
        .map(|row| {
            let m = row.iter().sum::<f32>() / width as f32;
            row.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / width as f32
        })
        .collect();
    expect_values(session, &[rows as u64], Dtype::F32, &read(&y)?, &expected)?;

    let analytic = gradient_of(&graph, &y, &x)?;
    let probe_graph = graph_of(session);
    let probe_x = upload(probe_graph.handle(), &dimv, &data)?;
    let probe_y = probe_x
        .variance_last()
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let probe_loss = loss_of(&probe_y)?;
    let numeric = finite_difference_gradient(&[rows, width], &data, &mut |probe| {
        read_probe_loss(&probe_x, &probe_loss, probe)
    })?;
    assert_gradient_matches_finite_difference(&analytic, &numeric)?;
    Ok(())
}

/// Every softmax row sums to exactly 1 within tolerance. Cheap, but it is the
/// invariant an online-softmax carrier with a mis-rescaled running sum breaks
/// while still looking plausible element by element.
fn rows_sum_to_one(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (rows, width) = (shape[0] as usize, shape[1] as usize);
    let data = Domain::Custom(-4.0, 4.0).sample(seed, rows * width);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(shape), &data)?;
    let p = x
        .softmax_last_dim()
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let got = read(&p)?;
    for (r, row) in got.chunks(width).enumerate() {
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
fn shift_invariance(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let len = (shape[0] * shape[1]) as usize;
    let data = Domain::Custom(-2.0, 2.0).sample(seed, len);
    let shifted: Vec<f32> = data.iter().map(|v| v + 60.0).collect();
    let graph = graph_of(session);
    let a = upload(graph.handle(), &dims(shape), &data)?;
    let b = upload(graph.handle(), &dims(shape), &shifted)?;
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
    expect_values(session, shape, Dtype::F32, &vb, &va)?;
    Ok(())
}

/// `dS = P * (dP - rowsum(dP * P))`.
///
/// Seeded with a non-uniform upstream gradient: under `sum_all` the softmax
/// adjoint is identically zero, so an all-ones seed cannot tell a correct
/// Jacobian from a missing one.
fn softmax_backward(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (rows, width) = (shape[0] as usize, shape[1] as usize);
    let len = rows * width;
    let data = Domain::Wide.sample(seed, len);
    // A non-uniform upstream weight, applied as `sum(w * softmax(x))`.
    let weights = Domain::Custom(0.25, 2.0).sample(seed ^ 0x9e37_79b9, len);
    let dimv = dims(shape);

    let build = |x: &Tensor, w: &Tensor| -> fusor2::Result<Tensor> { x.softmax_last_dim()?.mul(w) };

    let graph = graph_of(session);
    let x = upload(graph.handle(), &dimv, &data)?;
    let w = upload(graph.handle(), &dimv, &weights)?;
    let y = build(&x, &w).map_err(|e| -> CaseError { e.to_string().into() })?;
    let analytic = gradient_of(&graph, &y, &x)?;

    // Host Jacobian-vector product, row by row.
    let mut expected = vec![0.0f32; len];
    for r in 0..rows {
        let p = host_softmax(&data[r * width..(r + 1) * width]);
        let dp = &weights[r * width..(r + 1) * width];
        let dot: f32 = p.iter().zip(dp).map(|(a, b)| a * b).sum();
        for j in 0..width {
            expected[r * width + j] = p[j] * (dp[j] - dot);
        }
    }
    let backend = if crate::harness::is_gpu(session) {
        "gpu"
    } else {
        "cpu"
    };
    crate::compare::approx_or_relative_eq(
        backend,
        &[rows, width],
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
fn welford_carrier(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (rows, width) = (shape[0] as usize, shape[1] as usize);
    let data = Domain::Custom(10.0, 11.0).sample(seed, rows * width);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(shape), &data)?;

    let welford = x
        .variance_last()
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let naive = x
        .sqr()
        .and_then(|s| s.mean(1))
        .and_then(|ms| {
            let m = x.mean(1)?;
            ms.sub(&m.sqr()?)
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
    crate::compare::approx_or_relative_eq(backend, &[rows], &a, &b, 1e-3, 1e-2)?;
    Ok(())
}
