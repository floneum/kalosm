//! `conv`, `grouped_conv`, the three pools and `upsample`.
//!
//! `pool_max_non_overlapping_adjoint_is_mask` asserts a structural property:
//! `step >= window` makes the `Window` adjoint an elementwise
//! mask-and-broadcast, so the adjoint graph contains no `Scatter` node.

use fusor2::composite::pool::PoolSize;
use fusor2::{Dim, Dtype, Session};

use crate::compare::{assert_gradient_matches_finite_difference, finite_difference_gradient};
use crate::harness::{CaseError, CaseResult, Cases, dims};
use crate::suite::support::{
    Domain, expect_values, gradient_of, graph_of, loss_of, read, read_scalar, upload,
};

pub fn cases() -> Cases {
    let mut cases = Cases::new();
    cases.push("conv_pool", "conv1d", conv1d);
    cases.push("conv_pool", "conv2d_strided", conv2d_strided);
    cases.push("conv_pool", "grouped_conv", grouped_conv);
    cases.push("conv_pool", "pool", |s| pool_case(s, Pool::Avg));
    cases.push("conv_pool", "pool_max", |s| pool_case(s, Pool::Max));
    cases.push("conv_pool", "pool_min", |s| pool_case(s, Pool::Min));
    cases.push("conv_pool", "upsample_nearest2d", upsample_nearest2d);
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

fn conv1d(session: &Session) -> CaseResult {
    const BATCH: usize = 2;
    const IN_CH: usize = 3;
    const LEN: usize = 8;
    const OUT_CH: usize = 4;
    const K: usize = 3;
    let x_data = Domain::Wide.sample(401, BATCH * IN_CH * LEN);
    let w_data = Domain::Wide.sample(409, OUT_CH * IN_CH * K);
    let b_data = Domain::Wide.sample(419, OUT_CH);

    let graph = graph_of(session);
    let x = upload(
        graph.handle(),
        &dims(&[BATCH as u64, IN_CH as u64, LEN as u64]),
        &x_data,
    )?;
    let w = upload(
        graph.handle(),
        &dims(&[OUT_CH as u64, IN_CH as u64, K as u64]),
        &w_data,
    )?;
    let b = upload(graph.handle(), &dims(&[OUT_CH as u64]), &b_data)?;

    let y = fusor2::composite::conv::conv(&x, &w, Some(&b), &[1], &[K as u32 / 2], &[1])
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let (out_len, expected) = host_conv1d(
        &x_data,
        &w_data,
        &b_data,
        BATCH,
        IN_CH,
        LEN,
        OUT_CH,
        K,
        K / 2,
        1,
    );
    expect_values(
        session,
        &[BATCH as u64, OUT_CH as u64, out_len as u64],
        Dtype::F32,
        &read(&y)?,
        &expected,
    )?;

    // The bias gradient is one per output position per batch.
    let d_bias = gradient_of(&graph, &y, &b)?;
    let want = (BATCH * out_len) as f32;
    for (i, v) in d_bias.iter().enumerate() {
        if (v - want).abs() > 1e-3 * want {
            return Err(format!("conv1d bias gradient {i} is {v}, want {want}").into());
        }
    }

    let d_w = gradient_of(&graph, &y, &w)?;
    let numeric = finite_difference_gradient(&[OUT_CH * IN_CH * K], &w_data, &mut |probe| {
        let g = graph_of(session);
        let x = upload(
            g.handle(),
            &dims(&[BATCH as u64, IN_CH as u64, LEN as u64]),
            &x_data,
        )?;
        let w = upload(
            g.handle(),
            &dims(&[OUT_CH as u64, IN_CH as u64, K as u64]),
            probe,
        )?;
        let b = upload(g.handle(), &dims(&[OUT_CH as u64]), &b_data)?;
        let y = fusor2::composite::conv::conv(&x, &w, Some(&b), &[1], &[K as u32 / 2], &[1])
            .map_err(|e| -> CaseError { e.to_string().into() })?;
        read_scalar(&loss_of(&y)?)
    })?;
    assert_gradient_matches_finite_difference(&d_w, &numeric)?;
    Ok(())
}

fn conv2d_strided(session: &Session) -> CaseResult {
    const BATCH: u64 = 1;
    const IN_CH: u64 = 2;
    const H: u64 = 6;
    const W: u64 = 6;
    const OUT_CH: u64 = 3;
    const K: u64 = 3;
    let x_data = Domain::Wide.sample(421, (BATCH * IN_CH * H * W) as usize);
    let w_data = Domain::Wide.sample(431, (OUT_CH * IN_CH * K * K) as usize);

    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(&[BATCH, IN_CH, H, W]), &x_data)?;
    let w = upload(graph.handle(), &dims(&[OUT_CH, IN_CH, K, K]), &w_data)?;
    let y = fusor2::composite::conv::conv(&x, &w, None, &[2, 2], &[0, 0], &[1, 1])
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let out_h = ((H - K) / 2 + 1) as usize;
    let out_w = ((W - K) / 2 + 1) as usize;
    let mut expected = vec![0.0f32; (BATCH * OUT_CH) as usize * out_h * out_w];
    for oc in 0..OUT_CH as usize {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut acc = 0.0f32;
                for ic in 0..IN_CH as usize {
                    for kh in 0..K as usize {
                        for kw in 0..K as usize {
                            let ih = oh * 2 + kh;
                            let iw = ow * 2 + kw;
                            acc += x_data[(ic * H as usize + ih) * W as usize + iw]
                                * w_data[((oc * IN_CH as usize + ic) * K as usize + kh)
                                    * K as usize
                                    + kw];
                        }
                    }
                }
                expected[(oc * out_h + oh) * out_w + ow] = acc;
            }
        }
    }
    expect_values(
        session,
        &[BATCH, OUT_CH, out_h as u64, out_w as u64],
        Dtype::F32,
        &read(&y)?,
        &expected,
    )?;
    Ok(())
}

/// PyTorch grouped layout: `weight` is `[out_ch, in_ch / groups, ...kernel]`.
fn grouped_conv(session: &Session) -> CaseResult {
    const BATCH: u64 = 1;
    const IN_CH: u64 = 4;
    const LEN: u64 = 8;
    const OUT_CH: u64 = 4;
    const K: u64 = 3;
    const GROUPS: u32 = 2;
    let per_group_in = IN_CH / GROUPS as u64;

    let x_data = Domain::Wide.sample(433, (BATCH * IN_CH * LEN) as usize);
    let w_data = Domain::Wide.sample(439, (OUT_CH * per_group_in * K) as usize);

    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(&[BATCH, IN_CH, LEN]), &x_data)?;
    let w = upload(graph.handle(), &dims(&[OUT_CH, per_group_in, K]), &w_data)?;
    let y = fusor2::composite::conv::grouped_conv(&x, &w, None, &[1], &[1], &[1], GROUPS)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let out_len = (LEN as usize + 2 - K as usize) + 1;
    let per_group_out = (OUT_CH / GROUPS as u64) as usize;
    let mut expected = vec![0.0f32; OUT_CH as usize * out_len];
    for oc in 0..OUT_CH as usize {
        let group = oc / per_group_out;
        for o in 0..out_len {
            let mut acc = 0.0f32;
            for ic in 0..per_group_in as usize {
                for t in 0..K as usize {
                    let pos = (o + t) as isize - 1;
                    if pos < 0 || pos >= LEN as isize {
                        continue;
                    }
                    let channel = group * per_group_in as usize + ic;
                    acc += x_data[channel * LEN as usize + pos as usize]
                        * w_data[(oc * per_group_in as usize + ic) * K as usize + t];
                }
            }
            expected[oc * out_len + o] = acc;
        }
    }
    expect_values(
        session,
        &[BATCH, OUT_CH, out_len as u64],
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

/// A non-overlapping pool over the last axis of `[1, 2, 8]`.
fn pool_case(session: &Session, kind: Pool) -> CaseResult {
    const CH: usize = 2;
    const LEN: usize = 8;
    const WINDOW: usize = 4;
    let data = Domain::Wide.sample(443, CH * LEN);

    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(&[1, CH as u64, LEN as u64]), &data)?;
    let y = match kind {
        Pool::Max => {
            fusor2::composite::pool::pool_max(&x, &[PoolSize::new(WINDOW as u32, WINDOW as u32)])
        }
        Pool::Min => {
            fusor2::composite::pool::pool_min(&x, &[PoolSize::new(WINDOW as u32, WINDOW as u32)])
        }
        Pool::Avg => {
            fusor2::composite::pool::pool_avg(&x, &[PoolSize::new(WINDOW as u32, WINDOW as u32)])
        }
    }
    .map_err(|e| -> CaseError { e.to_string().into() })?;

    let positions = LEN / WINDOW;
    let mut expected = Vec::with_capacity(CH * positions);
    for c in 0..CH {
        for p in 0..positions {
            let window = &data[c * LEN + p * WINDOW..c * LEN + (p + 1) * WINDOW];
            expected.push(match kind {
                Pool::Max => window.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                Pool::Min => window.iter().copied().fold(f32::INFINITY, f32::min),
                Pool::Avg => window.iter().sum::<f32>() / WINDOW as f32,
            });
        }
    }
    expect_values(
        session,
        &[1, CH as u64, positions as u64],
        Dtype::F32,
        &read(&y)?,
        &expected,
    )?;
    Ok(())
}

/// With `step >= window` the adjoint is an elementwise mask: each input
/// element receives either the whole gradient of its window (when it is the
/// extremum) or nothing, and no element receives a contribution from two
/// windows.
fn non_overlapping_adjoint_is_mask(session: &Session) -> CaseResult {
    const LEN: usize = 8;
    const WINDOW: usize = 4;
    // Distinct values, so every window has a unique maximum and the tie rule
    // does not enter.
    let data: Vec<f32> = (0..LEN).map(|i| (i as f32 * 0.37).sin()).collect();

    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(&[1, 1, LEN as u64]), &data)?;
    let y = fusor2::composite::pool::pool_max(&x, &[PoolSize::new(WINDOW as u32, WINDOW as u32)])
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

fn upsample_nearest2d(session: &Session) -> CaseResult {
    const C: u64 = 2;
    const H: u64 = 2;
    const W: u64 = 3;
    const SCALE: u64 = 2;
    let data = Domain::Wide.sample(449, (C * H * W) as usize);

    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(&[1, C, H, W]), &data)?;
    let y = fusor2::composite::upsample::upsample_nearest(
        &x,
        &[
            Dim::Const(1),
            Dim::Const(C),
            Dim::Const(H * SCALE),
            Dim::Const(W * SCALE),
        ],
    )
    .map_err(|e| -> CaseError { e.to_string().into() })?;

    let mut expected = Vec::new();
    for c in 0..C as usize {
        for h in 0..(H * SCALE) as usize {
            for w in 0..(W * SCALE) as usize {
                expected.push(
                    data[(c * H as usize + h / SCALE as usize) * W as usize + w / SCALE as usize],
                );
            }
        }
    }
    expect_values(
        session,
        &[1, C, H * SCALE, W * SCALE],
        Dtype::F32,
        &read(&y)?,
        &expected,
    )?;

    // Each source element feeds `SCALE^2` outputs, so its gradient is that.
    let grad = gradient_of(&graph, &y, &x)?;
    let want = (SCALE * SCALE) as f32;
    if let Some((i, v)) = grad
        .iter()
        .enumerate()
        .find(|(_, v)| (**v - want).abs() > 1e-4)
    {
        return Err(format!("upsample gradient {i} is {v}, want {want}").into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_named_conv_and_pool_case_is_registered() {
        let names: Vec<String> = cases().names().iter().map(|n| (*n).to_string()).collect();
        for wanted in [
            "conv1d",
            "conv2d_strided",
            "grouped_conv",
            "pool",
            "pool_max",
            "pool_min",
            "upsample_nearest2d",
            "pool_max_non_overlapping_adjoint_is_mask",
        ] {
            assert!(
                names.iter().any(|n| n == &format!("conv_pool::{wanted}")),
                "{wanted} is missing"
            );
        }
        assert_eq!(names.len(), 8);
    }

    #[test]
    fn the_host_conv1d_reference_reproduces_an_identity_kernel() {
        // A single 1x1x1 kernel of weight 1 with no padding is the identity.
        let x = [1.0f32, 2.0, 3.0];
        let (len, out) = host_conv1d(&x, &[1.0], &[0.0], 1, 1, 3, 1, 1, 0, 1);
        assert_eq!(len, 3);
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn the_host_conv1d_reference_pads_with_zeros() {
        // A width-3 averaging kernel over [1, 2, 3] with padding 1 sees a
        // virtual zero on each side.
        let x = [1.0f32, 2.0, 3.0];
        let w = [1.0f32, 1.0, 1.0];
        let (len, out) = host_conv1d(&x, &w, &[0.0], 1, 1, 3, 1, 3, 1, 1);
        assert_eq!(len, 3);
        assert_eq!(out, vec![3.0, 6.0, 5.0]);
    }

    #[test]
    fn the_host_conv1d_reference_strides() {
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let (len, out) = host_conv1d(&x, &[1.0], &[0.0], 1, 1, 4, 1, 1, 0, 2);
        assert_eq!(len, 2);
        assert_eq!(out, vec![1.0, 3.0]);
    }

    #[test]
    fn the_bias_enters_every_output_position_once() {
        let x = [0.0f32; 4];
        let (_, out) = host_conv1d(&x, &[1.0], &[7.0], 1, 1, 4, 1, 1, 0, 1);
        assert_eq!(out, vec![7.0; 4]);
    }
}
