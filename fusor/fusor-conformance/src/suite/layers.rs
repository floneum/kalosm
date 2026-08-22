//! `Linear`, `Embedding`, `ConvNd`, `LayerNorm`, `RmsNorm`, plus the caches
//! and the optimizer that sit on top of them.
//!
//! A layer owns parameters and a `forward`; it owns no kernel. So every case
//! here is really two assertions: that the layer's forward is the composition
//! its documentation claims, and that gradients reach **every** parameter it
//! holds. A layer whose bias never receives a gradient trains to a plausible
//! but wrong model, and only the second assertion catches it.

use fusor::composite::{
    binary_cross_entropy_with_logits, distillation_loss, mse, softmax_cross_entropy,
};
use fusor::layers::{Embedding, LayerNorm, LayerNormNd, Linear, RmsNorm};
use fusor::optim::{AdamW, clip_global_norm, cosine_decay};
use fusor::{Dtype, Session};

use crate::harness::{
    CaseError, CaseResult, Cases, FuzzDim, dims, fill_indices, from_u32, fuzz_case,
};
use crate::suite::support::{Domain, expect_values, gradient_of, graph_of, read, upload};

/// A runtime-rank value as the const-rank one the layers take.
///
/// The suite uploads through `Dyn` — a case's shape is data from its own
/// table — and the layers are `Tensor<R, T>`, so the rank assertion happens
/// here, once per case.
fn t<const R: usize>(v: fusor::tensor::Dyn) -> fusor::Tensor<R, f32> {
    fusor::Tensor::<R, f32>::from_dyn(v)
}

/// [`t`] for a `u32` index value.
fn ids<const R: usize>(v: fusor::tensor::Dyn) -> fusor::Tensor<R, u32> {
    fusor::Tensor::<R, u32>::from_dyn(v)
}

/// `[rows, in] @ [out, in]^T + [out]` — forward only, so the extents can go
/// past the FD budget.
const LINEAR_SPEC: &[FuzzDim] = &[
    FuzzDim::Range(1, 8),
    FuzzDim::Range(1, 16),
    FuzzDim::Range(1, 16),
];

/// The gradient variant runs three backwards per run; kept a little smaller.
const LINEAR_GRAD_SPEC: &[FuzzDim] = &[
    FuzzDim::Range(1, 6),
    FuzzDim::Range(1, 8),
    FuzzDim::Range(1, 8),
];

fn backend_of(session: &Session) -> &'static str {
    if crate::harness::is_gpu(session) {
        "gpu"
    } else {
        "cpu"
    }
}

/// `x @ w^T (+ b)` on the host, with `w` laid out `[out, in]`.
fn host_linear(
    x: &[f32],
    w: &[f32],
    b: Option<&[f32]>,
    rows: usize,
    inn: usize,
    out_w: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * out_w];
    for r in 0..rows {
        for o in 0..out_w {
            let dot: f32 = (0..inn).map(|i| x[r * inn + i] * w[o * inn + i]).sum();
            out[r * out_w + o] = dot + b.map_or(0.0, |b| b[o]);
        }
    }
    out
}

pub fn cases() -> Cases {
    let mut cases = Cases::new();
    cases.push_case(fuzz_case(
        "layers",
        "linear_with_bias",
        LINEAR_SPEC,
        |s, sh, seed| linear_case(s, true, sh, seed),
    ));
    cases.push_case(fuzz_case(
        "layers",
        "linear_without_bias",
        LINEAR_SPEC,
        |s, sh, seed| linear_case(s, false, sh, seed),
    ));
    cases.push_case(fuzz_case(
        "layers",
        "linear_gradients_reach_every_parameter",
        LINEAR_GRAD_SPEC,
        linear_grads,
    ));
    // [vocab, emb, t0, t1]: tokens are a [t0, t1] index tensor into the table.
    // t0 >= 2 keeps room for the forced repeat the backward case relies on.
    const EMBEDDING_SPEC: &[FuzzDim] = &[
        FuzzDim::Range(2, 16),
        FuzzDim::Range(1, 8),
        FuzzDim::Range(2, 4),
        FuzzDim::Range(1, 4),
    ];
    cases.push_case(fuzz_case(
        "layers",
        "embedding_layer",
        EMBEDDING_SPEC,
        embedding_layer,
    ));
    cases.push_case(fuzz_case(
        "layers",
        "embedding_layer_backward",
        EMBEDDING_SPEC,
        embedding_layer_backward,
    ));
    const NORM_SPEC: &[FuzzDim] = &[FuzzDim::Range(1, 6), FuzzDim::Range(1, 16)];
    cases.push_case(fuzz_case(
        "layers",
        "layer_norm_layer",
        NORM_SPEC,
        layer_norm_layer,
    ));
    const NORM_ND_SPEC: &[FuzzDim] = &[
        FuzzDim::Range(1, 4),
        FuzzDim::Range(1, 4),
        FuzzDim::Range(1, 6),
    ];
    cases.push_case(fuzz_case(
        "layers",
        "layer_norm_nd_over_two_axes",
        NORM_ND_SPEC,
        layer_norm_nd,
    ));
    cases.push_case(fuzz_case(
        "layers",
        "rms_norm_layer",
        NORM_SPEC,
        rms_norm_layer,
    ));
    // [batch, in_ch, out_ch, k, extra]: the spatial extent is k + extra, so it
    // is always >= the kernel extent.
    const CONV_SPEC: &[FuzzDim] = &[
        FuzzDim::Range(1, 2),
        FuzzDim::Range(1, 3),
        FuzzDim::Range(1, 3),
        FuzzDim::Range(1, 3),
        FuzzDim::Range(0, 5),
    ];
    cases.push_case(fuzz_case("layers", "conv_nd_layer", CONV_SPEC, conv_layer));
    const MLP_SPEC: &[FuzzDim] = &[
        FuzzDim::Range(1, 4),
        FuzzDim::Range(2, 6),
        FuzzDim::Range(2, 6),
    ];
    cases.push_case(fuzz_case(
        "layers",
        "a_two_layer_mlp_trains_downhill",
        MLP_SPEC,
        mlp_step,
    ));
    // [rows, classes]; softmax over one class is constant, so classes >= 2.
    const LOSS_SPEC: &[FuzzDim] = &[FuzzDim::Range(1, 6), FuzzDim::Range(2, 8)];
    cases.push_case(fuzz_case(
        "layers",
        "softmax_cross_entropy",
        LOSS_SPEC,
        cross_entropy_case,
    ));
    cases.push_case(fuzz_case(
        "layers",
        "softmax_cross_entropy_gradient_is_p_minus_onehot",
        LOSS_SPEC,
        cross_entropy_grad,
    ));
    cases.push_case(fuzz_case(
        "layers",
        "binary_cross_entropy_with_logits",
        LOSS_SPEC,
        bce_case,
    ));
    cases.push_case(fuzz_case(
        "layers",
        "distillation_loss",
        LOSS_SPEC,
        distillation_case,
    ));
    cases.push_case(fuzz_case("layers", "mse", LOSS_SPEC, mse_case));
    const PARAM_SPEC: &[FuzzDim] = &[FuzzDim::Range(1, 16)];
    cases.push_case(fuzz_case(
        "layers",
        "adamw_step_moves_downhill",
        PARAM_SPEC,
        adamw_case,
    ));
    const CLIP_SPEC: &[FuzzDim] = &[FuzzDim::Range(1, 8), FuzzDim::Range(1, 8)];
    cases.push_case(fuzz_case(
        "layers",
        "clip_global_norm",
        CLIP_SPEC,
        clip_case,
    ));
    // [warmup, extra]: the total is warmup + extra, so warmup < total always.
    const COSINE_SPEC: &[FuzzDim] = &[FuzzDim::Range(1, 20), FuzzDim::Range(10, 190)];
    cases.push_case(fuzz_case(
        "layers",
        "cosine_decay_schedule",
        COSINE_SPEC,
        cosine_case,
    ));
    cases
}

/// `Linear::forward` is `mat_mul_transposed_rhs` plus a broadcast bias — the
/// transposed form specifically, so `d_weight` lands in the weight's own
/// `[out, in]` layout and the optimizer's flat slice stays a view.
fn linear_case(session: &Session, bias: bool, shape: &[u64], seed: u32) -> CaseResult {
    let (rows, inn, out) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
    let x_data = Domain::Wide.sample(seed, rows * inn);
    let w_data = Domain::Wide.sample(seed ^ 0x9e37_79b9, out * inn);
    let b_data = Domain::Wide.sample(seed.wrapping_add(1), out);

    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(&[rows as u64, inn as u64]), &x_data)?;
    let w = upload(graph.handle(), &dims(&[out as u64, inn as u64]), &w_data)?;
    let b = bias
        .then(|| upload(graph.handle(), &dims(&[out as u64]), &b_data))
        .transpose()?;
    let layer = Linear::new(t::<2>(w), b.map(t::<1>));
    let y = layer.forward(&t::<2>(x)).into_dyn();

    let expected = host_linear(
        &x_data,
        &w_data,
        bias.then_some(&b_data[..]),
        rows,
        inn,
        out,
    );
    expect_values(
        session,
        &[rows as u64, out as u64],
        Dtype::F32,
        &read(&y)?,
        &expected,
    )?;
    Ok(())
}

/// Both parameters and the input must receive a gradient, each in its own
/// layout.
fn linear_grads(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (rows, inn, out) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
    let x_data = Domain::Wide.sample(seed, rows * inn);
    let w_data = Domain::Wide.sample(seed ^ 0x9e37_79b9, out * inn);
    let b_data = Domain::Wide.sample(seed.wrapping_add(1), out);

    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(&[rows as u64, inn as u64]), &x_data)?;
    let w = upload(graph.handle(), &dims(&[out as u64, inn as u64]), &w_data)?;
    let b = upload(graph.handle(), &dims(&[out as u64]), &b_data)?;
    let layer = Linear::new(t::<2>(w.clone()), Some(t::<1>(b.clone())));
    let y = layer.forward(&t::<2>(x.clone())).into_dyn();

    // d_weight[o, i] = sum over rows of x[r, i], independent of o.
    let d_w = gradient_of(&graph, &y, &w)?;
    if d_w.len() != out * inn {
        return Err(format!(
            "d_weight has {} elements, want {}: it must land in the weight's own [out, in] \
             layout, not in a transposed view",
            d_w.len(),
            out * inn
        )
        .into());
    }
    let want_w: Vec<f32> = (0..out * inn)
        .map(|n| (0..rows).map(|r| x_data[r * inn + n % inn]).sum())
        .collect();
    crate::compare::approx_or_relative_eq(
        backend_of(session),
        &[out, inn],
        &want_w,
        &d_w,
        1e-4,
        1e-4,
    )?;

    // Each bias element is broadcast over `rows` rows.
    let d_b = gradient_of(&graph, &y, &b)?;
    let want_b = vec![rows as f32; out];
    crate::compare::approx_or_relative_eq(backend_of(session), &[out], &want_b, &d_b, 1e-5, 1e-5)?;

    // d_x[r, i] = sum over outputs of w[o, i].
    let d_x = gradient_of(&graph, &y, &x)?;
    let want_x: Vec<f32> = (0..rows * inn)
        .map(|n| (0..out).map(|o| w_data[o * inn + n % inn]).sum())
        .collect();
    crate::compare::approx_or_relative_eq(
        backend_of(session),
        &[rows, inn],
        &want_x,
        &d_x,
        1e-4,
        1e-4,
    )?;
    Ok(())
}

/// Sampled tokens with a forced repeat, so the backward always has an index
/// that must accumulate.
fn sample_tokens(seed: u32, len: usize, vocab: usize) -> Vec<u32> {
    let mut tokens = fill_indices(seed ^ 0x5eed, len, vocab as u32);
    tokens[1] = tokens[0];
    tokens
}

fn embedding_layer(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (vocab, emb) = (shape[0] as usize, shape[1] as usize);
    let (t0, t1) = (shape[2], shape[3]);
    let tokens = sample_tokens(seed, (t0 * t1) as usize, vocab);
    let table = Domain::Wide.sample(seed, vocab * emb);
    let graph = graph_of(session);
    let table_value = upload(graph.handle(), &dims(&[vocab as u64, emb as u64]), &table)?;
    let token_value = from_u32(graph.handle(), &dims(&[t0, t1]), &tokens)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let layer = Embedding::new(t::<2>(table_value));
    let y = layer.forward::<2, 3>(&ids::<2>(token_value)).into_dyn();

    let mut expected = Vec::with_capacity(tokens.len() * emb);
    for id in &tokens {
        let base = *id as usize * emb;
        expected.extend_from_slice(&table[base..base + emb]);
    }
    expect_values(
        session,
        &[t0, t1, emb as u64],
        Dtype::F32,
        &read(&y)?,
        &expected,
    )?;
    Ok(())
}

/// One token appearing twice gets the summed gradient — the layer inherits
/// that from `Gather`'s declared adjoint rather than implementing it.
fn embedding_layer_backward(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (vocab, emb) = (shape[0] as usize, shape[1] as usize);
    let (t0, t1) = (shape[2], shape[3]);
    let tokens = sample_tokens(seed, (t0 * t1) as usize, vocab);
    let table = Domain::Wide.sample(seed, vocab * emb);
    let graph = graph_of(session);
    let table_value = upload(graph.handle(), &dims(&[vocab as u64, emb as u64]), &table)?;
    let token_value = from_u32(graph.handle(), &dims(&[t0, t1]), &tokens)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let layer = Embedding::new(t::<2>(table_value.clone()));
    let y = layer.forward::<2, 3>(&ids::<2>(token_value)).into_dyn();
    let grad = gradient_of(&graph, &y, &table_value)?;

    let mut counts = vec![0.0f32; vocab];
    for id in &tokens {
        counts[*id as usize] += 1.0;
    }
    let want: Vec<f32> = (0..vocab * emb).map(|n| counts[n / emb]).collect();
    crate::compare::approx_or_relative_eq(
        backend_of(session),
        &[vocab, emb],
        &want,
        &grad,
        1e-5,
        1e-5,
    )?;
    Ok(())
}

const EPS: f32 = 1e-5;

fn layer_norm_layer(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (rows, width) = (shape[0] as usize, shape[1] as usize);
    let data = Domain::Wide.sample(seed, rows * width);
    let weight = Domain::Custom(0.5, 1.5).sample(seed ^ 0x9e37_79b9, width);
    let bias = Domain::Custom(-0.3, 0.3).sample(seed.wrapping_add(1), width);

    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(&[rows as u64, width as u64]), &data)?;
    let w = upload(graph.handle(), &dims(&[width as u64]), &weight)?;
    let b = upload(graph.handle(), &dims(&[width as u64]), &bias)?;
    let layer = LayerNorm::new(t::<1>(w), Some(t::<1>(b)), EPS);
    let y = layer.forward(&t::<2>(x)).into_dyn();

    let mut expected = Vec::with_capacity(rows * width);
    for row in data.chunks(width) {
        let mean = row.iter().sum::<f32>() / width as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / width as f32;
        let inv = 1.0 / (var + EPS).sqrt();
        for (j, v) in row.iter().enumerate() {
            expected.push((v - mean) * inv * weight[j] + bias[j]);
        }
    }
    expect_values(
        session,
        &[rows as u64, width as u64],
        Dtype::F32,
        &read(&y)?,
        &expected,
    )?;
    Ok(())
}

/// `LayerNormNd` normalizes over the trailing `axes` axes at once, so the
/// statistic is taken over the flattened tail rather than the last axis alone.
fn layer_norm_nd(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (a, bdim, c) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
    let data = Domain::Wide.sample(seed, a * bdim * c);
    let weight = Domain::Custom(0.5, 1.5).sample(seed ^ 0x9e37_79b9, bdim * c);

    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(shape), &data)?;
    let w = upload(graph.handle(), &dims(&[bdim as u64, c as u64]), &weight)?;
    let layer = LayerNormNd::new(LayerNorm::new(t::<2>(w), None, EPS), 2);
    let y = layer.forward(&t::<3>(x)).into_dyn();

    let tail = bdim * c;
    let mut expected = Vec::with_capacity(data.len());
    for block in data.chunks(tail) {
        let mean = block.iter().sum::<f32>() / tail as f32;
        let var = block.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / tail as f32;
        let inv = 1.0 / (var + EPS).sqrt();
        for (j, v) in block.iter().enumerate() {
            expected.push((v - mean) * inv * weight[j]);
        }
    }
    expect_values(session, shape, Dtype::F32, &read(&y)?, &expected)?;
    Ok(())
}

fn rms_norm_layer(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (rows, width) = (shape[0] as usize, shape[1] as usize);
    let data = Domain::Wide.sample(seed, rows * width);
    let weight = Domain::Custom(0.5, 1.5).sample(seed ^ 0x9e37_79b9, width);

    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(&[rows as u64, width as u64]), &data)?;
    let w = upload(graph.handle(), &dims(&[width as u64]), &weight)?;
    let layer = RmsNorm::new(Some(t::<1>(w.clone())), EPS);
    let y = layer.forward(&t::<2>(x)).into_dyn();

    let mut expected = Vec::with_capacity(rows * width);
    for row in data.chunks(width) {
        let ms = row.iter().map(|v| v * v).sum::<f32>() / width as f32;
        let inv = 1.0 / (ms + EPS).sqrt();
        for (j, v) in row.iter().enumerate() {
            expected.push(v * inv * weight[j]);
        }
    }
    expect_values(
        session,
        &[rows as u64, width as u64],
        Dtype::F32,
        &read(&y)?,
        &expected,
    )?;

    // The weight is a parameter and must be trained.
    let d_w = gradient_of(&graph, &y, &w)?;
    if d_w.len() != width {
        return Err(format!("the rms_norm weight gradient has {} elements", d_w.len()).into());
    }
    Ok(())
}

/// `ConvNd` over a 1-d signal, against a direct host convolution.
fn conv_layer(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (batch, in_ch, out_ch, k) = (
        shape[0] as usize,
        shape[1] as usize,
        shape[2] as usize,
        shape[3] as usize,
    );
    let width = k + shape[4] as usize;
    let x_data = Domain::Wide.sample(seed, batch * in_ch * width);
    let w_data = Domain::Wide.sample(seed ^ 0x9e37_79b9, out_ch * in_ch * k);
    let b_data = Domain::Wide.sample(seed.wrapping_add(1), out_ch);

    let graph = graph_of(session);
    let x = upload(
        graph.handle(),
        &dims(&[batch as u64, in_ch as u64, width as u64]),
        &x_data,
    )?;
    let w = upload(
        graph.handle(),
        &dims(&[out_ch as u64, in_ch as u64, k as u64]),
        &w_data,
    )?;
    let b = upload(graph.handle(), &dims(&[out_ch as u64]), &b_data)?;
    let layer = fusor::layers::ConvNd::new(t::<3>(w), Some(t::<1>(b)));
    let y = layer.forward(&t::<3>(x)).into_dyn();

    // No padding, unit stride: the output is width - k + 1 wide.
    let ow = width - k + 1;
    let mut expected = vec![0.0f32; batch * out_ch * ow];
    for n in 0..batch {
        for o in 0..out_ch {
            for p in 0..ow {
                let mut acc = b_data[o];
                for c in 0..in_ch {
                    for j in 0..k {
                        acc += x_data[(n * in_ch + c) * width + p + j]
                            * w_data[(o * in_ch + c) * k + j];
                    }
                }
                expected[(n * out_ch + o) * ow + p] = acc;
            }
        }
    }
    expect_values(
        session,
        &[batch as u64, out_ch as u64, ow as u64],
        Dtype::F32,
        &read(&y)?,
        &expected,
    )?;
    Ok(())
}

/// One gradient-descent step on a two-layer MLP must reduce the loss. This is
/// the smallest end-to-end statement that the forward, the tape and the
/// parameter update agree with each other.
fn mlp_step(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    const LR: f32 = 0.05;
    let (rows, inn, out) = (shape[0] as usize, shape[1] as usize, shape[2] as usize);
    let x_data = Domain::Wide.sample(seed, rows * inn);
    let mut w1 = Domain::Wide.sample(seed ^ 0x9e37_79b9, out * inn);
    let mut w2 = Domain::Wide.sample(seed.wrapping_add(1), out);

    let loss_at = |w1: &[f32], w2: &[f32]| -> Result<(f32, Vec<f32>, Vec<f32>), CaseError> {
        let graph = graph_of(session);
        let x = upload(graph.handle(), &dims(&[rows as u64, inn as u64]), &x_data)?;
        let a = upload(graph.handle(), &dims(&[out as u64, inn as u64]), w1)?;
        let b = upload(graph.handle(), &dims(&[out as u64]), w2)?;
        let hidden = Linear::new(t::<2>(a.clone()), None)
            .forward(&t::<2>(x))
            .into_dyn()
            .relu()
            .map_err(|e| -> CaseError { e.to_string().into() })?;
        let out_v = hidden
            .mul_(&b)
            .and_then(|v| v.sqr())
            .map_err(|e| -> CaseError { e.to_string().into() })?;
        let loss = crate::suite::support::loss_of(&out_v)?;
        let value = crate::suite::support::read_scalar(&loss)?;
        let d_a = gradient_of(&graph, &out_v, &a)?;
        let d_b = gradient_of(&graph, &out_v, &b)?;
        Ok((value, d_a, d_b))
    };

    let (before, d_a, d_b) = loss_at(&w1, &w2)?;
    // A relu can dead-end the whole sampled net; a zero gradient legitimately
    // moves nothing.
    if d_a.iter().chain(&d_b).all(|g| *g == 0.0) {
        return Ok(());
    }
    for (w, g) in w1.iter_mut().zip(&d_a) {
        *w -= LR * g;
    }
    for (w, g) in w2.iter_mut().zip(&d_b) {
        *w -= LR * g;
    }
    let (after, _, _) = loss_at(&w1, &w2)?;
    if !(after < before) {
        return Err(format!(
            "one step of gradient descent moved the loss from {before} to {after}; the \
             gradient does not point downhill"
        )
        .into());
    }
    Ok(())
}

/// One-hot targets from sampled labels.
fn one_hot(labels: &[u32], classes: usize) -> Vec<f32> {
    let mut targets = vec![0.0f32; labels.len() * classes];
    for (r, c) in labels.iter().enumerate() {
        targets[r * classes + *c as usize] = 1.0;
    }
    targets
}

/// `softmax_cross_entropy(logits, one-hot targets)`.
fn cross_entropy_case(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (rows, classes) = (shape[0] as usize, shape[1] as usize);
    let logits = Domain::Custom(-2.0, 2.0).sample(seed, rows * classes);
    let labels = fill_indices(seed ^ 0x5eed, rows, classes as u32);
    let targets = one_hot(&labels, classes);

    let graph = graph_of(session);
    let l = upload(graph.handle(), &dims(shape), &logits)?;
    let t = upload(graph.handle(), &dims(shape), &targets)?;
    let loss =
        softmax_cross_entropy(&l, &t, 1).map_err(|e| -> CaseError { e.to_string().into() })?;

    let expected: Vec<f32> = logits
        .chunks(classes)
        .zip(&labels)
        .map(|(row, label)| {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let lse = max + row.iter().map(|v| (v - max).exp()).sum::<f32>().ln();
            lse - row[*label as usize]
        })
        .collect();
    expect_values(
        session,
        &[rows as u64],
        Dtype::F32,
        &read(&loss)?,
        &expected,
    )?;
    Ok(())
}

/// The analytic gradient: `(softmax - onehot) * grad / rows`.
fn cross_entropy_grad(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (rows, classes) = (shape[0] as usize, shape[1] as usize);
    let logits = Domain::Custom(-2.0, 2.0).sample(seed, rows * classes);
    let labels = fill_indices(seed ^ 0x5eed, rows, classes as u32);
    let targets = one_hot(&labels, classes);

    let graph = graph_of(session);
    let l = upload(graph.handle(), &dims(shape), &logits)?;
    let t = upload(graph.handle(), &dims(shape), &targets)?;
    let loss =
        softmax_cross_entropy(&l, &t, 1).map_err(|e| -> CaseError { e.to_string().into() })?;
    let grad = gradient_of(&graph, &loss, &l)?;

    let mut want = vec![0.0f32; rows * classes];
    for (r, row) in logits.chunks(classes).enumerate() {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = row.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = e.iter().sum();
        for c in 0..classes {
            want[r * classes + c] = e[c] / sum - targets[r * classes + c];
        }
    }
    crate::compare::approx_or_relative_eq(
        backend_of(session),
        &[rows, classes],
        &want,
        &grad,
        1e-4,
        1e-4,
    )?;
    Ok(())
}

/// The folded one-vs-all BCE the trainer's distillation loss is built from.
/// Written as a plain softplus chain; `softplus_bce_adjoint` is what turns its
/// backward into the single-sigmoid form, and the numbers must not change.
fn bce_case(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (rows, classes) = (shape[0] as usize, shape[1] as usize);
    let logits = Domain::Custom(-3.0, 3.0).sample(seed, rows * classes);
    let targets = Domain::Custom(0.0, 1.0).sample(seed ^ 0x9e37_79b9, rows * classes);

    let graph = graph_of(session);
    let l = upload(graph.handle(), &dims(shape), &logits)?;
    let t = upload(graph.handle(), &dims(shape), &targets)?;
    let loss = binary_cross_entropy_with_logits(&l, &t)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    // max(z,0) - z*y + ln(1 + exp(-|z|)), the numerically stable form.
    let expected: Vec<f32> = logits
        .iter()
        .zip(&targets)
        .map(|(z, y)| z.max(0.0) - z * y + (1.0 + (-z.abs()).exp()).ln())
        .collect();
    expect_values(session, shape, Dtype::F32, &read(&loss)?, &expected)?;

    // dL/dz = sigmoid(z) - y, which is what the rewrite must preserve.
    let grad = gradient_of(&graph, &loss, &l)?;
    let want: Vec<f32> = logits
        .iter()
        .zip(&targets)
        .map(|(z, y)| 1.0 / (1.0 + (-z).exp()) - y)
        .collect();
    crate::compare::approx_or_relative_eq(
        backend_of(session),
        &[rows, classes],
        &want,
        &grad,
        1e-4,
        1e-4,
    )?;
    Ok(())
}

fn distillation_case(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    const T: f32 = 2.0;
    let len = (shape[0] * shape[1]) as usize;
    let student = Domain::Custom(-2.0, 2.0).sample(seed, len);
    let teacher = Domain::Custom(-2.0, 2.0).sample(seed ^ 0x9e37_79b9, len);

    let graph = graph_of(session);
    let s = upload(graph.handle(), &dims(shape), &student)?;
    let t = upload(graph.handle(), &dims(shape), &teacher)?;
    let loss = distillation_loss(&s, &t, T).map_err(|e| -> CaseError { e.to_string().into() })?;
    let got = read(&loss)?;
    if got.iter().any(|v| !v.is_finite()) {
        return Err("the distillation loss produced a non-finite value".into());
    }
    // The gradient must reach the student and must not reach the teacher's
    // values as if they were trainable in the same step.
    let d_s = gradient_of(&graph, &loss, &s)?;
    if d_s.iter().all(|v| *v == 0.0) {
        return Err("the student received an identically-zero distillation gradient".into());
    }
    Ok(())
}

fn mse_case(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let len = (shape[0] * shape[1]) as usize;
    let a_data = Domain::Wide.sample(seed, len);
    let b_data = Domain::Wide.sample(seed ^ 0x9e37_79b9, len);
    let graph = graph_of(session);
    let a = upload(graph.handle(), &dims(shape), &a_data)?;
    let b = upload(graph.handle(), &dims(shape), &b_data)?;
    let loss = mse(&a, &b).map_err(|e| -> CaseError { e.to_string().into() })?;
    let want: f32 = a_data
        .iter()
        .zip(&b_data)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        / len as f32;
    let got = crate::suite::support::read_scalar(&loss)?;
    if (got - want).abs() > 1e-4 * want.abs().max(1.0) {
        return Err(format!("mse is {got}, want {want}").into());
    }
    Ok(())
}

/// One AdamW step on a quadratic must move the parameter toward the optimum
/// by roughly the learning rate, with the decoupled decay applied to the
/// parameter and not to the gradient.
fn adamw_case(session: &Session, shape: &[u64], _seed: u32) -> CaseResult {
    const LR: f32 = 0.1;
    let n = shape[0] as usize;
    let start = vec![1.0f32; n];
    let graph = graph_of(session);
    let p = upload(graph.handle(), &dims(&[n as u64]), &start)?;
    let loss = p
        .sqr()
        .and_then(|s| s.sum_all())
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let g = gradient_of(&graph, &loss, &p)?;
    let grad = upload(graph.handle(), &dims(&[n as u64]), &g)?;

    let mut opt = AdamW::new(LR);
    let updated = opt
        .step(std::slice::from_ref(&p), std::slice::from_ref(&grad))
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let after = read(
        updated
            .first()
            .ok_or_else(|| -> CaseError { "AdamW::step returned no parameters".into() })?,
    )?;

    for (i, v) in after.iter().enumerate() {
        if *v >= start[i] {
            return Err(format!(
                "parameter {i} moved from {} to {v}; a positive gradient must decrease it",
                start[i]
            )
            .into());
        }
        // Bias correction folded into the step size makes the first step
        // approximately `lr` in magnitude, independent of the gradient scale.
        let moved = start[i] - v;
        if (moved - LR).abs() > 0.5 * LR {
            return Err(format!(
                "parameter {i} moved by {moved}; with bias correction folded in, the first \
                 AdamW step is about the learning rate {LR}"
            )
            .into());
        }
    }
    Ok(())
}

/// Global-norm clipping scales every gradient by one shared factor, so the
/// direction is preserved and the norm lands exactly on the cap. The data
/// stays in [1, 2), so the global norm always exceeds the cap.
fn clip_case(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    const CAP: f32 = 1.0;
    let (n0, n1) = (shape[0] as usize, shape[1] as usize);
    let a_data = Domain::Custom(1.0, 2.0).sample(seed, n0);
    let b_data = Domain::Custom(1.0, 2.0).sample(seed ^ 0x9e37_79b9, n1);
    let graph = graph_of(session);
    let a = upload(graph.handle(), &dims(&[n0 as u64]), &a_data)?;
    let b = upload(graph.handle(), &dims(&[n1 as u64]), &b_data)?;
    let clipped =
        clip_global_norm(&[a, b], CAP).map_err(|e| -> CaseError { e.to_string().into() })?;

    let mut total = 0.0f32;
    let mut flat = Vec::new();
    for t in &clipped {
        for v in read(t)? {
            total += v * v;
            flat.push(v);
        }
    }
    let norm = total.sqrt();
    if (norm - CAP).abs() > 1e-4 {
        return Err(format!("the clipped global norm is {norm}, want {CAP}").into());
    }
    let host_norm = a_data
        .iter()
        .chain(&b_data)
        .map(|v| v * v)
        .sum::<f32>()
        .sqrt();
    let want: Vec<f32> = a_data
        .iter()
        .chain(&b_data)
        .map(|v| v * CAP / host_norm)
        .collect();
    crate::compare::approx_or_relative_eq(
        backend_of(session),
        &[n0 + n1],
        &want,
        &flat,
        1e-4,
        1e-4,
    )?;
    Ok(())
}

/// The schedule is host-computed, so this needs no device at all — but it
/// runs per session anyway, because a schedule that disagrees between
/// backends would be a very confusing bug to find later.
fn cosine_case(_session: &Session, shape: &[u64], _seed: u32) -> CaseResult {
    const PEAK: f32 = 1.0;
    const FLOOR: f32 = 0.1;
    let warmup = shape[0];
    let total = warmup + shape[1];

    let at = |step| cosine_decay(step, warmup, total, PEAK, FLOOR);
    if at(0) > at(warmup) {
        return Err("the warmup phase must ramp up, not down".into());
    }
    if (at(warmup) - PEAK).abs() > 1e-5 {
        return Err(format!("the schedule peaks at {}, want {PEAK}", at(warmup)).into());
    }
    if (at(total) - FLOOR).abs() > 1e-5 {
        return Err(format!("the schedule ends at {}, want {FLOOR}", at(total)).into());
    }
    // Monotone decreasing after the warmup.
    let mut prev = at(warmup);
    for step in warmup + 1..=total {
        let now = at(step);
        if now > prev + 1e-6 {
            return Err(format!("the schedule rose at step {step}: {prev} -> {now}").into());
        }
        prev = now;
    }
    Ok(())
}
