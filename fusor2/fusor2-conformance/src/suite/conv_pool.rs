//! `conv`, `grouped_conv`, the three pools and `upsample`.
//!
//! `pool_max_non_overlapping_adjoint_is_mask` pins that `step >= window`
//! proves the adjoint is an elementwise mask-and-broadcast, i.e. the adjoint
//! graph contains no `Scatter` node.

use fusor2::composite::{
    PoolSize, conv, grouped_conv as grouped_conv_op, pool_avg, pool_max, pool_min,
    upsample_nearest,
};
use fusor2::{Dim, Dtype, Session};

use crate::compare::{assert_gradient_matches_finite_difference, finite_difference_gradient};
use crate::harness::{CaseError, CaseResult, Cases, FuzzDim, dims, fuzz_case};
use crate::suite::support::{
    Domain, expect_values, gradient_of, graph_of, loss_of, read, read_scalar, upload,
};

// Spatial extents start at 3 so they never fall under the kernel extent
// (kernels sample from [1, 3]). conv1d runs a finite-difference gradient over
// the weight, so its channel counts stay small.
const CONV1D_SPEC: &[FuzzDim] = &[
    FuzzDim::Range(1, 2),
    FuzzDim::Range(1, 3),
    FuzzDim::Range(3, 8),
    FuzzDim::Range(1, 3),
    FuzzDim::Range(1, 3),
];
const CONV2D_SPEC: &[FuzzDim] = &[
    FuzzDim::Range(1, 2),
    FuzzDim::Range(1, 3),
    FuzzDim::Range(3, 10),
    FuzzDim::Range(3, 10),
    FuzzDim::Range(1, 3),
    FuzzDim::Range(1, 3),
];
// Groups must divide both channel counts: per-group counts are sampled and
// multiplied.
const GROUPED_CONV_SPEC: &[FuzzDim] = &[
    FuzzDim::Range(1, 2),
    FuzzDim::Range(1, 3),
    FuzzDim::Range(1, 3),
    FuzzDim::Range(1, 3),
    FuzzDim::Range(3, 8),
    FuzzDim::Range(1, 3),
];
// The length is `window * positions`, so the non-overlapping pool always
// tiles it exactly.
const POOL_SPEC: &[FuzzDim] = &[
    FuzzDim::Range(1, 4),
    FuzzDim::Range(1, 4),
    FuzzDim::Range(1, 4),
];
const UPSAMPLE_SPEC: &[FuzzDim] = &[
    FuzzDim::Range(1, 3),
    FuzzDim::Range(1, 4),
    FuzzDim::Range(1, 4),
    FuzzDim::Range(1, 3),
];

pub fn cases() -> Cases {
    let mut cases = Cases::new();
    cases.push_case(fuzz_case("conv_pool", "conv1d", CONV1D_SPEC, conv1d));
    cases.push_case(fuzz_case(
        "conv_pool",
        "conv2d_strided",
        CONV2D_SPEC,
        conv2d_strided,
    ));
    cases.push_case(fuzz_case(
        "conv_pool",
        "grouped_conv",
        GROUPED_CONV_SPEC,
        grouped_conv,
    ));
    cases.push_case(fuzz_case("conv_pool", "pool", POOL_SPEC, |s, sh, seed| {
        pool_case(s, Pool::Avg, sh, seed)
    }));
    cases.push_case(fuzz_case(
        "conv_pool",
        "pool_max",
        POOL_SPEC,
        |s, sh, seed| pool_case(s, Pool::Max, sh, seed),
    ));
    cases.push_case(fuzz_case(
        "conv_pool",
        "pool_min",
        POOL_SPEC,
        |s, sh, seed| pool_case(s, Pool::Min, sh, seed),
    ));
    cases.push_case(fuzz_case(
        "conv_pool",
        "upsample_nearest2d",
        UPSAMPLE_SPEC,
        upsample_nearest2d,
    ));
    cases.push(
        "conv_pool",
        "pool_max_non_overlapping_adjoint_is_mask",
        non_overlapping_adjoint_is_mask,
    );
    cases
}

/// `[batch, in_ch, len] * [out_ch, in_ch, k]` with `padding` and unit stride.
#[allow(clippy::too_many_arguments)]
fn host_conv1d(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    batch: usize,
    in_ch: usize,
    len: usize,
    out_ch: usize,
    kernel: usize,
    padding: usize,
    stride: usize,
) -> (usize, Vec<f32>) {
    let out_len = (len + 2 * padding - kernel) / stride + 1;
    let mut out = vec![0.0f32; batch * out_ch * out_len];
    for b in 0..batch {
        for oc in 0..out_ch {
            for o in 0..out_len {
                let mut acc = bias.get(oc).copied().unwrap_or(0.0);
                for ic in 0..in_ch {
                    for t in 0..kernel {
                        let pos = (o * stride + t) as isize - padding as isize;
                        if pos < 0 || pos >= len as isize {
                            continue;
                        }
                        acc += x[(b * in_ch + ic) * len + pos as usize]
                            * w[(oc * in_ch + ic) * kernel + t];
                    }
                }
                out[(b * out_ch + oc) * out_len + o] = acc;
            }
        }
    }
    (out_len, out)
}

fn conv1d(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let [batch, in_ch, len, out_ch, k] = [
        shape[0] as usize,
        shape[1] as usize,
        shape[2] as usize,
        shape[3] as usize,
        shape[4] as usize,
    ];
    let x_data = Domain::Wide.sample(seed, batch * in_ch * len);
    let w_data = Domain::Wide.sample(seed ^ 0x9e37_79b9, out_ch * in_ch * k);
    let b_data = Domain::Wide.sample(seed.wrapping_add(1), out_ch);

    let graph = graph_of(session);
    let x = upload(
        graph.handle(),
        &dims(&[batch as u64, in_ch as u64, len as u64]),
        &x_data,
    )?;
    let w = upload(
        graph.handle(),
        &dims(&[out_ch as u64, in_ch as u64, k as u64]),
        &w_data,
    )?;
    let b = upload(graph.handle(), &dims(&[out_ch as u64]), &b_data)?;

    let y = conv(&x, &w, Some(&b), &[1], &[k as u32 / 2], &[1])
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let (out_len, expected) = host_conv1d(
        &x_data, &w_data, &b_data, batch, in_ch, len, out_ch, k, k / 2, 1,
    );
    expect_values(
        session,
        &[batch as u64, out_ch as u64, out_len as u64],
        Dtype::F32,
        &read(&y)?,
        &expected,
    )?;

    // The bias gradient is one per output position per batch.
    let d_bias = gradient_of(&graph, &y, &b)?;
    let want = (batch * out_len) as f32;
    for (i, v) in d_bias.iter().enumerate() {
        if (v - want).abs() > 1e-3 * want {
            return Err(format!("conv1d bias gradient {i} is {v}, want {want}").into());
        }
    }

    let d_w = gradient_of(&graph, &y, &w)?;
    let numeric = finite_difference_gradient(&[out_ch * in_ch * k], &w_data, &mut |probe| {
        let g = graph_of(session);
        let x = upload(
            g.handle(),
            &dims(&[batch as u64, in_ch as u64, len as u64]),
            &x_data,
        )?;
        let w = upload(
            g.handle(),
            &dims(&[out_ch as u64, in_ch as u64, k as u64]),
            probe,
        )?;
        let b = upload(g.handle(), &dims(&[out_ch as u64]), &b_data)?;
        let y = conv(&x, &w, Some(&b), &[1], &[k as u32 / 2], &[1])
            .map_err(|e| -> CaseError { e.to_string().into() })?;
        read_scalar(&loss_of(&y)?)
    })?;
    assert_gradient_matches_finite_difference(&d_w, &numeric)?;
    Ok(())
}

fn conv2d_strided(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let [batch, in_ch, h, w_ext, out_ch, k] = [
        shape[0] as usize,
        shape[1] as usize,
        shape[2] as usize,
        shape[3] as usize,
        shape[4] as usize,
        shape[5] as usize,
    ];
    let x_data = Domain::Wide.sample(seed, batch * in_ch * h * w_ext);
    let w_data = Domain::Wide.sample(seed ^ 0x9e37_79b9, out_ch * in_ch * k * k);

    let graph = graph_of(session);
    let x = upload(
        graph.handle(),
        &dims(&[batch as u64, in_ch as u64, h as u64, w_ext as u64]),
        &x_data,
    )?;
    let w = upload(
        graph.handle(),
        &dims(&[out_ch as u64, in_ch as u64, k as u64, k as u64]),
        &w_data,
    )?;
    let y = conv(&x, &w, None, &[2, 2], &[0, 0], &[1, 1])
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let out_h = (h - k) / 2 + 1;
    let out_w = (w_ext - k) / 2 + 1;
    let mut expected = vec![0.0f32; batch * out_ch * out_h * out_w];
    for b in 0..batch {
        for oc in 0..out_ch {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut acc = 0.0f32;
                    for ic in 0..in_ch {
                        for kh in 0..k {
                            for kw in 0..k {
                                let ih = oh * 2 + kh;
                                let iw = ow * 2 + kw;
                                acc += x_data[((b * in_ch + ic) * h + ih) * w_ext + iw]
                                    * w_data[((oc * in_ch + ic) * k + kh) * k + kw];
                            }
                        }
                    }
                    expected[((b * out_ch + oc) * out_h + oh) * out_w + ow] = acc;
                }
            }
        }
    }
    expect_values(
        session,
        &[batch as u64, out_ch as u64, out_h as u64, out_w as u64],
        Dtype::F32,
        &read(&y)?,
        &expected,
    )?;
    Ok(())
}

/// PyTorch grouped layout: `weight` is `[out_ch, in_ch / groups, ...kernel]`.
/// `shape` is `[batch, groups, per_group_in, per_group_out, len, k]` — the
/// per-group channel counts are sampled and multiplied so groups always
/// divides both.
fn grouped_conv(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let [batch, groups, per_group_in, per_group_out, len, k] = [
        shape[0] as usize,
        shape[1] as usize,
        shape[2] as usize,
        shape[3] as usize,
        shape[4] as usize,
        shape[5] as usize,
    ];
    let in_ch = groups * per_group_in;
    let out_ch = groups * per_group_out;

    let x_data = Domain::Wide.sample(seed, batch * in_ch * len);
    let w_data = Domain::Wide.sample(seed ^ 0x9e37_79b9, out_ch * per_group_in * k);

    let graph = graph_of(session);
    let x = upload(
        graph.handle(),
        &dims(&[batch as u64, in_ch as u64, len as u64]),
        &x_data,
    )?;
    let w = upload(
        graph.handle(),
        &dims(&[out_ch as u64, per_group_in as u64, k as u64]),
        &w_data,
    )?;
    let y =
        grouped_conv_op(&x, &w, None, &[1], &[1], &[1], groups as u32)
            .map_err(|e| -> CaseError { e.to_string().into() })?;

    let out_len = len + 2 - k + 1;
    let mut expected = vec![0.0f32; batch * out_ch * out_len];
    for b in 0..batch {
        for oc in 0..out_ch {
            let group = oc / per_group_out;
            for o in 0..out_len {
                let mut acc = 0.0f32;
                for ic in 0..per_group_in {
                    for t in 0..k {
                        let pos = (o + t) as isize - 1;
                        if pos < 0 || pos >= len as isize {
                            continue;
                        }
                        let channel = group * per_group_in + ic;
                        acc += x_data[(b * in_ch + channel) * len + pos as usize]
                            * w_data[(oc * per_group_in + ic) * k + t];
                    }
                }
                expected[(b * out_ch + oc) * out_len + o] = acc;
            }
        }
    }
    expect_values(
        session,
        &[batch as u64, out_ch as u64, out_len as u64],
        Dtype::F32,
        &read(&y)?,
        &expected,
    )?;
    Ok(())
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum Pool {
    Max,
    Min,
    Avg,
}

/// A non-overlapping pool over the last axis of `[1, ch, window * positions]`.
fn pool_case(session: &Session, kind: Pool, shape: &[u64], seed: u32) -> CaseResult {
    let [ch, window, positions] = [shape[0] as usize, shape[1] as usize, shape[2] as usize];
    let len = window * positions;
    let data = Domain::Wide.sample(seed, ch * len);

    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(&[1, ch as u64, len as u64]), &data)?;
    let y = match kind {
        Pool::Max => pool_max(&x, &[PoolSize::new(window, window)]),
        Pool::Min => pool_min(&x, &[PoolSize::new(window, window)]),
        Pool::Avg => pool_avg(&x, &[PoolSize::new(window, window)]),
    }
    .map_err(|e| -> CaseError { e.to_string().into() })?;

    let mut expected = Vec::with_capacity(ch * positions);
    for c in 0..ch {
        for p in 0..positions {
            let win = &data[c * len + p * window..c * len + (p + 1) * window];
            expected.push(match kind {
                Pool::Max => win.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                Pool::Min => win.iter().copied().fold(f32::INFINITY, f32::min),
                Pool::Avg => win.iter().sum::<f32>() / window as f32,
            });
        }
    }
    expect_values(
        session,
        &[1, ch as u64, positions as u64],
        Dtype::F32,
        &read(&y)?,
        &expected,
    )?;
    Ok(())
}

/// `step >= window` proves the adjoint is an elementwise mask: each input
/// element receives either the whole gradient of its window (if it is the
/// extremum) or nothing, and **no element receives a contribution from two
/// windows**. That last clause is the one a scatter would be needed for, and
/// it is what this case falsifies.
fn non_overlapping_adjoint_is_mask(session: &Session) -> CaseResult {
    const LEN: usize = 8;
    const WINDOW: usize = 4;
    // Distinct values, so every window has a unique maximum and the tie rule
    // does not enter.
    let data: Vec<f32> = (0..LEN).map(|i| (i as f32 * 0.37).sin()).collect();

    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(&[1, 1, LEN as u64]), &data)?;
    let y = pool_max(&x, &[PoolSize::new(WINDOW, WINDOW)])
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let grad = gradient_of(&graph, &y, &x)?;
    if grad.len() != LEN {
        return Err(format!(
            "the pool adjoint produced {} values, want {LEN}",
            grad.len()
        )
        .into());
    }
    for p in 0..LEN / WINDOW {
        let window = &data[p * WINDOW..(p + 1) * WINDOW];
        let arg = window
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap();
        for w in 0..WINDOW {
            let want = f32::from(w == arg);
            let got = grad[p * WINDOW + w];
            if (got - want).abs() > 1e-5 {
                return Err(format!(
                    "pool adjoint at {}: got {got}, want {want}. With step >= window the \
                     adjoint is an elementwise mask-and-broadcast; anything else means \
                     the verifier failed to prove injectivity and fell back to a scatter.",
                    p * WINDOW + w
                )
                .into());
            }
        }
    }
    // Every gradient is 0 or 1: no element accumulated from two windows.
    if let Some((i, v)) = grad
        .iter()
        .enumerate()
        .find(|(_, v)| **v != 0.0 && (**v - 1.0).abs() > 1e-5)
    {
        return Err(format!(
            "pool adjoint at {i} is {v}: a value that is neither 0 nor 1 means two \
             windows accumulated into one element, which only happens under a Scatter"
        )
        .into());
    }
    Ok(())
}

fn upsample_nearest2d(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let [c, h, w, scale] = [shape[0], shape[1], shape[2], shape[3]];
    let data = Domain::Wide.sample(seed, (c * h * w) as usize);

    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(&[1, c, h, w]), &data)?;
    let y = upsample_nearest(
        &x,
        &[
            Dim::Const(1),
            Dim::Const(c),
            Dim::Const(h * scale),
            Dim::Const(w * scale),
        ],
    )
    .map_err(|e| -> CaseError { e.to_string().into() })?;

    let mut expected = Vec::new();
    for ci in 0..c as usize {
        for hi in 0..(h * scale) as usize {
            for wi in 0..(w * scale) as usize {
                expected.push(
                    data[(ci * h as usize + hi / scale as usize) * w as usize
                        + wi / scale as usize],
                );
            }
        }
    }
    expect_values(
        session,
        &[1, c, h * scale, w * scale],
        Dtype::F32,
        &read(&y)?,
        &expected,
    )?;

    // Each source element feeds `scale^2` outputs, so its gradient is that.
    let grad = gradient_of(&graph, &y, &x)?;
    let want = (scale * scale) as f32;
    if let Some((i, v)) = grad
        .iter()
        .enumerate()
        .find(|(_, v)| (**v - want).abs() > 1e-4)
    {
        return Err(format!("upsample gradient {i} is {v}, want {want}").into());
    }
    Ok(())
}
