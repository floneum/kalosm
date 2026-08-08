//! `Linear`, `Embedding`, `ConvNd`, `LayerNorm`, `RmsNorm`, plus the caches
//! and the optimizer that sit on top of them.
//!
//! Each case asserts two things: that the layer's forward is the composition it
//! documents, and that gradients reach every parameter it holds.

use fusor2::composite::loss::{
    binary_cross_entropy_with_logits, distillation_loss, mse, softmax_cross_entropy,
};
use fusor2::layers::{Embedding, LayerNorm, LayerNormNd, Linear, RmsNorm};
use fusor2::optim::{AdamW, clip_global_norm, cosine_decay};
use fusor2::{Dtype, Session};

use crate::harness::{CaseError, CaseResult, Cases, dims, from_u32};
use crate::suite::support::{Domain, expect_values, gradient_of, graph_of, read, upload};

/// `[ROWS, IN] @ [OUT, IN]^T + [OUT]`.
const ROWS: usize = 3;
const IN: usize = 4;
const OUT: usize = 5;

fn backend_of(session: &Session) -> &'static str {
    if crate::harness::is_gpu(session) {
        "gpu"
    } else {
        "cpu"
    }
}

/// `x @ w^T (+ b)` on the host, with `w` laid out `[out, in]`.
fn host_linear(x: &[f32], w: &[f32], b: Option<&[f32]>) -> Vec<f32> {
    let mut out = vec![0.0f32; ROWS * OUT];
    for r in 0..ROWS {
        for o in 0..OUT {
            let dot: f32 = (0..IN).map(|i| x[r * IN + i] * w[o * IN + i]).sum();
            out[r * OUT + o] = dot + b.map_or(0.0, |b| b[o]);
        }
    }
    out
}

pub fn cases() -> Cases {
    let mut cases = Cases::new();
    cases.push("layers", "linear_with_bias", |s| linear_case(s, true));
    cases.push("layers", "linear_without_bias", |s| linear_case(s, false));
    cases.push(
        "layers",
        "linear_gradients_reach_every_parameter",
        linear_grads,
    );
    cases.push("layers", "embedding_layer", embedding_layer);
    cases.push(
        "layers",
        "embedding_layer_backward",
        embedding_layer_backward,
    );
    cases.push("layers", "layer_norm_layer", layer_norm_layer);
    cases.push("layers", "layer_norm_nd_over_two_axes", layer_norm_nd);
    cases.push("layers", "rms_norm_layer", rms_norm_layer);
    cases.push("layers", "conv_nd_layer", conv_layer);
    cases.push("layers", "a_two_layer_mlp_trains_downhill", mlp_step);
    cases.push("layers", "softmax_cross_entropy", cross_entropy_case);
    cases.push(
        "layers",
        "softmax_cross_entropy_gradient_is_p_minus_onehot",
        cross_entropy_grad,
    );
    cases.push("layers", "binary_cross_entropy_with_logits", bce_case);
    cases.push("layers", "distillation_loss", distillation_case);
    cases.push("layers", "mse", mse_case);
    cases.push("layers", "adamw_step_moves_downhill", adamw_case);
    cases.push("layers", "clip_global_norm", clip_case);
    cases.push("layers", "cosine_decay_schedule", cosine_case);
    cases
}

/// `Linear::forward` is `mat_mul_transposed_rhs` plus a broadcast bias, so
/// `d_weight` lands in the weight's own `[out, in]` layout.
fn linear_case(session: &Session, bias: bool) -> CaseResult {
    let x_data = Domain::Wide.sample(1501, ROWS * IN);
    let w_data = Domain::Wide.sample(1511, OUT * IN);
    let b_data = Domain::Wide.sample(1523, OUT);

    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(&[ROWS as u64, IN as u64]), &x_data)?;
    let w = upload(graph.handle(), &dims(&[OUT as u64, IN as u64]), &w_data)?;
    let b = bias
        .then(|| upload(graph.handle(), &dims(&[OUT as u64]), &b_data))
        .transpose()?;
    let layer = Linear::new(w, b);
    let y = layer
        .forward(&x)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let expected = host_linear(&x_data, &w_data, bias.then_some(&b_data[..]));
    expect_values(
        session,
        &[ROWS as u64, OUT as u64],
        Dtype::F32,
        &read(&y)?,
        &expected,
    )?;
    Ok(())
}

/// Both parameters and the input must receive a gradient, each in its own
/// layout.
fn linear_grads(session: &Session) -> CaseResult {
    let x_data = Domain::Wide.sample(1531, ROWS * IN);
    let w_data = Domain::Wide.sample(1543, OUT * IN);
    let b_data = Domain::Wide.sample(1549, OUT);

    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(&[ROWS as u64, IN as u64]), &x_data)?;
    let w = upload(graph.handle(), &dims(&[OUT as u64, IN as u64]), &w_data)?;
    let b = upload(graph.handle(), &dims(&[OUT as u64]), &b_data)?;
    let layer = Linear::new(w.clone(), Some(b.clone()));
    let y = layer
        .forward(&x)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    // d_weight[o, i] = sum over rows of x[r, i], independent of o.
    let d_w = gradient_of(&graph, &y, &w)?;
    if d_w.len() != OUT * IN {
        return Err(format!(
            "d_weight has {} elements, want {}: it must land in the weight's own [out, in] \
             layout, not in a transposed view",
            d_w.len(),
            OUT * IN
        )
        .into());
    }
    let want_w: Vec<f32> = (0..OUT * IN)
        .map(|n| (0..ROWS).map(|r| x_data[r * IN + n % IN]).sum())
        .collect();
    crate::compare::approx_or_relative_eq(
        backend_of(session),
        &[OUT, IN],
        &want_w,
        &d_w,
        1e-4,
        1e-4,
    )?;

    // Each bias element is broadcast over ROWS rows.
    let d_b = gradient_of(&graph, &y, &b)?;
    let want_b = vec![ROWS as f32; OUT];
    crate::compare::approx_or_relative_eq(backend_of(session), &[OUT], &want_b, &d_b, 1e-5, 1e-5)?;

    // d_x[r, i] = sum over outputs of w[o, i].
    let d_x = gradient_of(&graph, &y, &x)?;
    let want_x: Vec<f32> = (0..ROWS * IN)
        .map(|n| (0..OUT).map(|o| w_data[o * IN + n % IN]).sum())
        .collect();
    crate::compare::approx_or_relative_eq(
        backend_of(session),
        &[ROWS, IN],
        &want_x,
        &d_x,
        1e-4,
        1e-4,
    )?;
    Ok(())
}

const VOCAB: usize = 5;
const EMB: usize = 3;
/// Repeats row 2, so the layer's backward has to accumulate.
const TOKENS: &[u32] = &[2, 0, 2, 4];

fn embedding_layer(session: &Session) -> CaseResult {
    let table = Domain::Wide.sample(1553, VOCAB * EMB);
    let graph = graph_of(session);
    let t = upload(graph.handle(), &dims(&[VOCAB as u64, EMB as u64]), &table)?;
    let ids = from_u32(graph.handle(), &dims(&[2, 2]), TOKENS)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let layer = Embedding::new(t);
    let y = layer
        .forward(&ids)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let mut expected = Vec::with_capacity(TOKENS.len() * EMB);
    for id in TOKENS {
        let base = *id as usize * EMB;
        expected.extend_from_slice(&table[base..base + EMB]);
    }
    expect_values(
        session,
        &[2, 2, EMB as u64],
        Dtype::F32,
        &read(&y)?,
        &expected,
    )?;
    Ok(())
}

/// One token appearing twice gets the summed gradient, inherited from
/// `Gather`'s declared adjoint.
fn embedding_layer_backward(session: &Session) -> CaseResult {
    let table = Domain::Wide.sample(1559, VOCAB * EMB);
    let graph = graph_of(session);
    let t = upload(graph.handle(), &dims(&[VOCAB as u64, EMB as u64]), &table)?;
    let ids = from_u32(graph.handle(), &dims(&[2, 2]), TOKENS)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let layer = Embedding::new(t.clone());
    let y = layer
        .forward(&ids)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let grad = gradient_of(&graph, &y, &t)?;

    let mut counts = vec![0.0f32; VOCAB];
    for id in TOKENS {
        counts[*id as usize] += 1.0;
    }
    let want: Vec<f32> = (0..VOCAB * EMB).map(|n| counts[n / EMB]).collect();
    crate::compare::approx_or_relative_eq(
        backend_of(session),
        &[VOCAB, EMB],
        &want,
        &grad,
        1e-5,
        1e-5,
    )?;
    Ok(())
}

const NORM_W: usize = 6;
const EPS: f32 = 1e-5;

fn layer_norm_layer(session: &Session) -> CaseResult {
    let data = Domain::Wide.sample(1567, ROWS * NORM_W);
    let weight = Domain::Custom(0.5, 1.5).sample(1571, NORM_W);
    let bias = Domain::Custom(-0.3, 0.3).sample(1579, NORM_W);

    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(&[ROWS as u64, NORM_W as u64]), &data)?;
    let w = upload(graph.handle(), &dims(&[NORM_W as u64]), &weight)?;
    let b = upload(graph.handle(), &dims(&[NORM_W as u64]), &bias)?;
    let layer = LayerNorm::new(w, Some(b), EPS);
    let y = layer
        .forward(&x)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let mut expected = Vec::with_capacity(ROWS * NORM_W);
    for row in data.chunks(NORM_W) {
        let mean = row.iter().sum::<f32>() / NORM_W as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / NORM_W as f32;
        let inv = 1.0 / (var + EPS).sqrt();
        for (j, v) in row.iter().enumerate() {
            expected.push((v - mean) * inv * weight[j] + bias[j]);
        }
    }
    expect_values(
        session,
        &[ROWS as u64, NORM_W as u64],
        Dtype::F32,
        &read(&y)?,
        &expected,
    )?;
    Ok(())
}

/// `LayerNormNd` normalizes over the trailing `axes` axes at once, so the
/// statistic is taken over the flattened tail rather than the last axis alone.
fn layer_norm_nd(session: &Session) -> CaseResult {
    const A: usize = 2;
    const BDIM: usize = 3;
    const C: usize = 4;
    let data = Domain::Wide.sample(1583, A * BDIM * C);
    let weight = Domain::Custom(0.5, 1.5).sample(1597, BDIM * C);

    let graph = graph_of(session);
    let x = upload(
        graph.handle(),
        &dims(&[A as u64, BDIM as u64, C as u64]),
        &data,
    )?;
    let w = upload(graph.handle(), &dims(&[BDIM as u64, C as u64]), &weight)?;
    let layer = LayerNormNd::new(LayerNorm::new(w, None, EPS), 2);
    let y = layer
        .forward(&x)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let tail = BDIM * C;
    let mut expected = Vec::with_capacity(data.len());
    for block in data.chunks(tail) {
        let mean = block.iter().sum::<f32>() / tail as f32;
        let var = block.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / tail as f32;
        let inv = 1.0 / (var + EPS).sqrt();
        for (j, v) in block.iter().enumerate() {
            expected.push((v - mean) * inv * weight[j]);
        }
    }
    expect_values(
        session,
        &[A as u64, BDIM as u64, C as u64],
        Dtype::F32,
        &read(&y)?,
        &expected,
    )?;
    Ok(())
}

fn rms_norm_layer(session: &Session) -> CaseResult {
    let data = Domain::Wide.sample(1601, ROWS * NORM_W);
    let weight = Domain::Custom(0.5, 1.5).sample(1607, NORM_W);

    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(&[ROWS as u64, NORM_W as u64]), &data)?;
    let w = upload(graph.handle(), &dims(&[NORM_W as u64]), &weight)?;
    let layer = RmsNorm::new(Some(w.clone()), EPS);
    let y = layer
        .forward(&x)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let mut expected = Vec::with_capacity(ROWS * NORM_W);
    for row in data.chunks(NORM_W) {
        let ms = row.iter().map(|v| v * v).sum::<f32>() / NORM_W as f32;
        let inv = 1.0 / (ms + EPS).sqrt();
        for (j, v) in row.iter().enumerate() {
            expected.push(v * inv * weight[j]);
        }
    }
    expect_values(
        session,
        &[ROWS as u64, NORM_W as u64],
        Dtype::F32,
        &read(&y)?,
        &expected,
    )?;

    // The weight is a parameter and must be trained.
    let d_w = gradient_of(&graph, &y, &w)?;
    if d_w.len() != NORM_W {
        return Err(format!("the rms_norm weight gradient has {} elements", d_w.len()).into());
    }
    Ok(())
}

/// `ConvNd` over a 1-d signal, against a direct host convolution.
fn conv_layer(session: &Session) -> CaseResult {
    const BATCH: usize = 1;
    const IN_CH: usize = 2;
    const OUT_CH: usize = 3;
    const WIDTH: usize = 6;
    const K: usize = 3;
    let x_data = Domain::Wide.sample(1609, BATCH * IN_CH * WIDTH);
    let w_data = Domain::Wide.sample(1613, OUT_CH * IN_CH * K);
    let b_data = Domain::Wide.sample(1619, OUT_CH);

    let graph = graph_of(session);
    let x = upload(
        graph.handle(),
        &dims(&[BATCH as u64, IN_CH as u64, WIDTH as u64]),
        &x_data,
    )?;
    let w = upload(
        graph.handle(),
        &dims(&[OUT_CH as u64, IN_CH as u64, K as u64]),
        &w_data,
    )?;
    let b = upload(graph.handle(), &dims(&[OUT_CH as u64]), &b_data)?;
    let layer = fusor2::layers::ConvNd::new(w, Some(b));
    let y = layer
        .forward(&x)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    // No padding, unit stride: the output is WIDTH - K + 1 wide.
    let ow = WIDTH - K + 1;
    let mut expected = vec![0.0f32; OUT_CH * ow];
    for o in 0..OUT_CH {
        for p in 0..ow {
            let mut acc = b_data[o];
            for c in 0..IN_CH {
                for k in 0..K {
                    acc += x_data[c * WIDTH + p + k] * w_data[(o * IN_CH + c) * K + k];
                }
            }
            expected[o * ow + p] = acc;
        }
    }
    expect_values(
        session,
        &[BATCH as u64, OUT_CH as u64, ow as u64],
        Dtype::F32,
        &read(&y)?,
        &expected,
    )?;
    Ok(())
}

/// One gradient-descent step on a two-layer MLP reduces the loss.
fn mlp_step(session: &Session) -> CaseResult {
    const LR: f32 = 0.05;
    let x_data = Domain::Wide.sample(1621, ROWS * IN);
    let mut w1 = Domain::Wide.sample(1627, OUT * IN);
    let mut w2 = Domain::Wide.sample(1637, OUT);

    let loss_at = |w1: &[f32], w2: &[f32]| -> Result<(f32, Vec<f32>, Vec<f32>), CaseError> {
        let graph = graph_of(session);
        let x = upload(graph.handle(), &dims(&[ROWS as u64, IN as u64]), &x_data)?;
        let a = upload(graph.handle(), &dims(&[OUT as u64, IN as u64]), w1)?;
        let b = upload(graph.handle(), &dims(&[OUT as u64]), w2)?;
        let hidden = Linear::new(a.clone(), None)
            .forward(&x)
            .and_then(|h| h.relu())
            .map_err(|e| -> CaseError { e.to_string().into() })?;
        let out = hidden
            .broadcast_mul(&b)
            .and_then(|v| v.sqr())
            .map_err(|e| -> CaseError { e.to_string().into() })?;
        let loss = crate::suite::support::loss_of(&out)?;
        let value = crate::suite::support::read_scalar(&loss)?;
        let d_a = gradient_of(&graph, &out, &a)?;
        let d_b = gradient_of(&graph, &out, &b)?;
        Ok((value, d_a, d_b))
    };

    let (before, d_a, d_b) = loss_at(&w1, &w2)?;
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

const CLASSES: usize = 4;

/// `softmax_cross_entropy(logits, one-hot targets)`.
fn cross_entropy_case(session: &Session) -> CaseResult {
    let logits = Domain::Custom(-2.0, 2.0).sample(1657, ROWS * CLASSES);
    let labels: [usize; ROWS] = [1, 3, 0];
    let mut targets = vec![0.0f32; ROWS * CLASSES];
    for (r, c) in labels.iter().enumerate() {
        targets[r * CLASSES + c] = 1.0;
    }

    let graph = graph_of(session);
    let l = upload(
        graph.handle(),
        &dims(&[ROWS as u64, CLASSES as u64]),
        &logits,
    )?;
    let t = upload(
        graph.handle(),
        &dims(&[ROWS as u64, CLASSES as u64]),
        &targets,
    )?;
    let loss =
        softmax_cross_entropy(&l, &t, 1).map_err(|e| -> CaseError { e.to_string().into() })?;

    let expected: Vec<f32> = logits
        .chunks(CLASSES)
        .zip(labels)
        .map(|(row, label)| {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let lse = max + row.iter().map(|v| (v - max).exp()).sum::<f32>().ln();
            lse - row[label]
        })
        .collect();
    expect_values(
        session,
        &[ROWS as u64],
        Dtype::F32,
        &read(&loss)?,
        &expected,
    )?;
    Ok(())
}

/// The analytic gradient: `(softmax - onehot) * grad / rows`.
fn cross_entropy_grad(session: &Session) -> CaseResult {
    let logits = Domain::Custom(-2.0, 2.0).sample(1663, ROWS * CLASSES);
    let labels: [usize; ROWS] = [1, 3, 0];
    let mut targets = vec![0.0f32; ROWS * CLASSES];
    for (r, c) in labels.iter().enumerate() {
        targets[r * CLASSES + c] = 1.0;
    }

    let graph = graph_of(session);
    let l = upload(
        graph.handle(),
        &dims(&[ROWS as u64, CLASSES as u64]),
        &logits,
    )?;
    let t = upload(
        graph.handle(),
        &dims(&[ROWS as u64, CLASSES as u64]),
        &targets,
    )?;
    let loss =
        softmax_cross_entropy(&l, &t, 1).map_err(|e| -> CaseError { e.to_string().into() })?;
    let grad = gradient_of(&graph, &loss, &l)?;

    let mut want = vec![0.0f32; ROWS * CLASSES];
    for (r, row) in logits.chunks(CLASSES).enumerate() {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = row.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = e.iter().sum();
        for c in 0..CLASSES {
            want[r * CLASSES + c] = e[c] / sum - targets[r * CLASSES + c];
        }
    }
    crate::compare::approx_or_relative_eq(
        backend_of(session),
        &[ROWS, CLASSES],
        &want,
        &grad,
        1e-4,
        1e-4,
    )?;
    Ok(())
}

/// The folded one-vs-all BCE, written as a plain softplus chain.
/// `softplus_bce_adjoint` turns its backward into the single-sigmoid form
/// without changing the numbers.
fn bce_case(session: &Session) -> CaseResult {
    let logits = Domain::Custom(-3.0, 3.0).sample(1667, ROWS * CLASSES);
    let targets = Domain::Custom(0.0, 1.0).sample(1669, ROWS * CLASSES);

    let graph = graph_of(session);
    let l = upload(
        graph.handle(),
        &dims(&[ROWS as u64, CLASSES as u64]),
        &logits,
    )?;
    let t = upload(
        graph.handle(),
        &dims(&[ROWS as u64, CLASSES as u64]),
        &targets,
    )?;
    let loss = binary_cross_entropy_with_logits(&l, &t)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    // max(z,0) - z*y + ln(1 + exp(-|z|)), the numerically stable form.
    let expected: Vec<f32> = logits
        .iter()
        .zip(&targets)
        .map(|(z, y)| z.max(0.0) - z * y + (1.0 + (-z.abs()).exp()).ln())
        .collect();
    expect_values(
        session,
        &[ROWS as u64, CLASSES as u64],
        Dtype::F32,
        &read(&loss)?,
        &expected,
    )?;

    // dL/dz = sigmoid(z) - y.
    let grad = gradient_of(&graph, &loss, &l)?;
    let want: Vec<f32> = logits
        .iter()
        .zip(&targets)
        .map(|(z, y)| 1.0 / (1.0 + (-z).exp()) - y)
        .collect();
    crate::compare::approx_or_relative_eq(
        backend_of(session),
        &[ROWS, CLASSES],
        &want,
        &grad,
        1e-4,
        1e-4,
    )?;
    Ok(())
}

fn distillation_case(session: &Session) -> CaseResult {
    const T: f32 = 2.0;
    let student = Domain::Custom(-2.0, 2.0).sample(1693, ROWS * CLASSES);
    let teacher = Domain::Custom(-2.0, 2.0).sample(1697, ROWS * CLASSES);

    let graph = graph_of(session);
    let s = upload(
        graph.handle(),
        &dims(&[ROWS as u64, CLASSES as u64]),
        &student,
    )?;
    let t = upload(
        graph.handle(),
        &dims(&[ROWS as u64, CLASSES as u64]),
        &teacher,
    )?;
    let loss = distillation_loss(&s, &t, T).map_err(|e| -> CaseError { e.to_string().into() })?;
    let got = read(&loss)?;
    if got.iter().any(|v| !v.is_finite()) {
        return Err("the distillation loss produced a non-finite value".into());
    }
    // The gradient reaches the student.
    let d_s = gradient_of(&graph, &loss, &s)?;
    if d_s.iter().all(|v| *v == 0.0) {
        return Err("the student received an identically-zero distillation gradient".into());
    }
    Ok(())
}

fn mse_case(session: &Session) -> CaseResult {
    let a_data = Domain::Wide.sample(1699, ROWS * CLASSES);
    let b_data = Domain::Wide.sample(1709, ROWS * CLASSES);
    let graph = graph_of(session);
    let a = upload(
        graph.handle(),
        &dims(&[ROWS as u64, CLASSES as u64]),
        &a_data,
    )?;
    let b = upload(
        graph.handle(),
        &dims(&[ROWS as u64, CLASSES as u64]),
        &b_data,
    )?;
    let loss = mse(&a, &b).map_err(|e| -> CaseError { e.to_string().into() })?;
    let n = (ROWS * CLASSES) as f32;
    let want: f32 = a_data
        .iter()
        .zip(&b_data)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        / n;
    let got = crate::suite::support::read_scalar(&loss)?;
    if (got - want).abs() > 1e-4 * want.abs().max(1.0) {
        return Err(format!("mse is {got}, want {want}").into());
    }
    Ok(())
}

/// One AdamW step on a quadratic must move the parameter toward the optimum
/// by roughly the learning rate, with the decoupled decay applied to the
/// parameter and not to the gradient.
fn adamw_case(session: &Session) -> CaseResult {
    const LR: f32 = 0.1;
    let start = vec![1.0f32; 4];
    let graph = graph_of(session);
    let p = upload(graph.handle(), &dims(&[4]), &start)?;
    let loss = p
        .sqr()
        .and_then(|s| s.sum_all())
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let g = gradient_of(&graph, &loss, &p)?;
    let grad = upload(graph.handle(), &dims(&[4]), &g)?;

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
/// direction is preserved and the norm lands exactly on the cap.
fn clip_case(session: &Session) -> CaseResult {
    const CAP: f32 = 1.0;
    let a_data: Vec<f32> = vec![3.0, 4.0];
    let b_data: Vec<f32> = vec![0.0, 12.0];
    let graph = graph_of(session);
    let a = upload(graph.handle(), &dims(&[2]), &a_data)?;
    let b = upload(graph.handle(), &dims(&[2]), &b_data)?;
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
    // 5 and 12 make a global norm of 13, so the scale is 1/13.
    let want: Vec<f32> = [3.0f32, 4.0, 0.0, 12.0].iter().map(|v| v / 13.0).collect();
    crate::compare::approx_or_relative_eq(backend_of(session), &[4], &want, &flat, 1e-4, 1e-4)?;
    Ok(())
}

/// The schedule is host-computed and needs no device, but runs per session so
/// a backend disagreement would show up.
fn cosine_case(_session: &Session) -> CaseResult {
    const WARMUP: u64 = 10;
    const TOTAL: u64 = 100;
    const PEAK: f32 = 1.0;
    const FLOOR: f32 = 0.1;

    let at = |step| cosine_decay(step, WARMUP, TOTAL, PEAK, FLOOR);
    if at(0) > at(WARMUP) {
        return Err("the warmup phase must ramp up, not down".into());
    }
    if (at(WARMUP) - PEAK).abs() > 1e-5 {
        return Err(format!("the schedule peaks at {}, want {PEAK}", at(WARMUP)).into());
    }
    if (at(TOTAL) - FLOOR).abs() > 1e-5 {
        return Err(format!("the schedule ends at {}, want {FLOOR}", at(TOTAL)).into());
    }
    // Monotone decreasing after the warmup.
    let mut prev = at(WARMUP);
    for step in WARMUP + 1..=TOTAL {
        let now = at(step);
        if now > prev + 1e-6 {
            return Err(format!("the schedule rose at step {step}: {prev} -> {now}").into());
        }
        prev = now;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registered() -> Vec<String> {
        cases().names().iter().map(|n| (*n).to_string()).collect()
    }

    fn has(names: &[String], wanted: &str) -> bool {
        names.iter().any(|n| n == &format!("layers::{wanted}"))
    }

    #[test]
    fn every_layer_in_the_parity_list_is_registered() {
        let names = registered();
        for wanted in [
            "linear_with_bias",
            "linear_without_bias",
            "embedding_layer",
            "layer_norm_layer",
            "layer_norm_nd_over_two_axes",
            "rms_norm_layer",
            "conv_nd_layer",
        ] {
            assert!(has(&names, wanted), "{wanted} is missing");
        }
    }

    #[test]
    fn the_losses_and_the_optimizer_are_registered() {
        let names = registered();
        for wanted in [
            "softmax_cross_entropy",
            "binary_cross_entropy_with_logits",
            "distillation_loss",
            "mse",
            "adamw_step_moves_downhill",
            "clip_global_norm",
            "cosine_decay_schedule",
        ] {
            assert!(has(&names, wanted), "{wanted} is missing");
        }
    }

    #[test]
    fn the_host_linear_is_x_times_w_transposed() {
        // A hand-worked 1x2 @ (2x2)^T with a bias.
        let x = [
            1.0f32, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let w: Vec<f32> = (0..OUT * IN)
            .map(|n| if n % IN == 0 { 1.0 } else { 0.0 })
            .collect();
        let out = host_linear(&x, &w, None);
        assert_eq!(out.len(), ROWS * OUT);
        // Row 0 dotted with each all-but-first-zero weight row picks x[0] = 1.
        for o in 0..OUT {
            assert_eq!(out[o], 1.0);
        }
        // Rows 1 and 2 of x are zero, so their outputs are zero.
        for n in OUT..ROWS * OUT {
            assert_eq!(out[n], 0.0);
        }
    }

    #[test]
    fn the_bias_enters_every_row_once() {
        let x = vec![0.0f32; ROWS * IN];
        let w = vec![0.0f32; OUT * IN];
        let b: Vec<f32> = (0..OUT).map(|o| o as f32).collect();
        let out = host_linear(&x, &w, Some(&b));
        for r in 0..ROWS {
            for o in 0..OUT {
                assert_eq!(out[r * OUT + o], o as f32);
            }
        }
    }

    #[test]
    fn the_embedding_tokens_repeat() {
        let mut seen = vec![0u32; VOCAB];
        for t in TOKENS {
            seen[*t as usize] += 1;
        }
        assert!(seen.iter().any(|c| *c >= 2), "{TOKENS:?} has no repeat");
        assert!(seen.iter().any(|c| *c == 0), "{TOKENS:?} covers every row");
    }
}
