//! Fused map-reduce gates: composed reduce clusters that recognition does
//! NOT claim must still collapse to a single kernel. The resolver inlines the
//! elementwise producer into the reduce, and contraction-shaped clusters —
//! whatever their dim order or broadcast structure — lower through the tiled
//! (workgroup-cached) path. A fusion miss materializes the full index space,
//! which these tests turn into a kernel-count failure.

use fusor_core::{Device, QMatrix, Tensor};
use fusor_gguf::GgmlType;

/// `a [M, K]` and `b [N, K]` broadcast into the `[M, N, K]` index space.
fn broadcast_factors(
    device: &Device,
    m: usize,
    n: usize,
    k: usize,
    a_values: &[f32],
    b_values: &[f32],
) -> (Tensor, Tensor) {
    let a = Tensor::from_slice(device, [m, k], a_values);
    let b = Tensor::from_slice(device, [n, k], b_values);
    let a3 = a.reshape([m, 1, k]).broadcast_as([m, n, k]);
    let b3 = b.reshape([1, n, k]).broadcast_as([m, n, k]);
    (a3, b3)
}

fn pattern(len: usize, scale: f32) -> Vec<f32> {
    (0..len).map(|i| ((i as f32) * scale).sin()).collect()
}

#[test]
fn broadcast_composed_contraction_fuses_to_single_kernel() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };

        // Edge shapes on every tiled dim: M not a multiple of the 32-row
        // tile, K not a multiple of the 8-deep tile.
        let (m, n, k) = (72usize, 64usize, 20usize);
        let a_values = pattern(m * k, 0.13);
        let b_values = pattern(n * k, 0.07);
        let (a3, b3) = broadcast_factors(&device, m, n, k, &a_values, &b_values);

        let out = (&a3 * &b3).sum(2);
        assert_eq!(
            out.count_kernels_to_resolve(),
            1,
            "broadcast-composed contraction must fuse into one map-reduce kernel"
        );

        let slice = out.as_slice::<2, f32>().await.unwrap();
        for row in [0usize, 31, 32, m - 1] {
            for col in [0usize, 33, n - 1] {
                let expected: f32 = (0..k)
                    .map(|kk| a_values[row * k + kk] * b_values[col * k + kk])
                    .sum();
                let actual = slice[[row, col]];
                assert!(
                    (actual - expected).abs() < 1e-3,
                    "[{row}, {col}]: got {actual}, expected {expected}"
                );
            }
        }
    });
}

#[test]
fn small_composed_contraction_fuses_through_serial_path() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };

        // Below the tiled gates (m < 32): the fused reduce still collapses
        // to one kernel through the serial per-output path.
        let (m, n, k) = (8usize, 6usize, 16usize);
        let a_values = pattern(m * k, 0.21);
        let b_values = pattern(n * k, 0.17);
        let (a3, b3) = broadcast_factors(&device, m, n, k, &a_values, &b_values);

        let out = (&a3 * &b3).sum(2);
        assert_eq!(out.count_kernels_to_resolve(), 1, "small fused contraction");

        let slice = out.as_slice::<2, f32>().await.unwrap();
        for row in 0..m {
            for col in 0..n {
                let expected: f32 = (0..k)
                    .map(|kk| a_values[row * k + kk] * b_values[col * k + kk])
                    .sum();
                let actual = slice[[row, col]];
                assert!(
                    (actual - expected).abs() < 1e-4,
                    "[{row}, {col}]: got {actual}, expected {expected}"
                );
            }
        }
    });
}

#[test]
fn max_contraction_fuses_with_masked_tiles() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };

        // A non-sum reduction over the same contraction cluster: the tiled
        // path cannot rely on zero fills collapsing out-of-bounds slots, so
        // this exercises the masked accumulate with edge tiles on M and K.
        let (m, n, k) = (72usize, 64usize, 20usize);
        let a_values = pattern(m * k, 0.31);
        let b_values = pattern(n * k, 0.23);
        let (a3, b3) = broadcast_factors(&device, m, n, k, &a_values, &b_values);

        let out = (&a3 * &b3).max(2);
        assert_eq!(out.count_kernels_to_resolve(), 1, "fused max contraction");

        let slice = out.as_slice::<2, f32>().await.unwrap();
        for row in [0usize, 39, m - 1] {
            for col in [0usize, 40, n - 1] {
                let expected = (0..k)
                    .map(|kk| a_values[row * k + kk] * b_values[col * k + kk])
                    .fold(f32::NEG_INFINITY, f32::max);
                let actual = slice[[row, col]];
                assert!(
                    (actual - expected).abs() < 1e-4,
                    "[{row}, {col}]: got {actual}, expected {expected}"
                );
            }
        }
    });
}

#[test]
fn quantized_weighted_reduce_fuses_to_single_kernel() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };

        // `sum_k w_q8[n, k] * x[k]`: the dequantize node feeds the fused
        // reduce directly as a block-quantized input — no dense
        // materialization kernel, one fused dispatch decoding per element.
        const Q8_BLOCK: usize = 32;
        let (n, k) = (64usize, 64usize);
        let scale = half::f16::from_f32(0.02);
        let mut bytes = Vec::new();
        let mut dequantized = Vec::with_capacity(n * k);
        for block in 0..(n * k / Q8_BLOCK) {
            bytes.extend_from_slice(&scale.to_le_bytes());
            for i in 0..Q8_BLOCK {
                let q = (((block * 7 + i * 5) % 64) as i32 - 32) as i8;
                bytes.push(q as u8);
                dequantized.push(scale.to_f32() * q as f32);
            }
        }
        let w = QMatrix::from_parts(
            &device,
            &bytes,
            vec![n, k].into_boxed_slice(),
            GgmlType::Q8_0,
        )
        .unwrap();
        let x_values = pattern(k, 0.19);
        let x = Tensor::from_slice(&device, [k], &x_values);

        let wd = w.dequantize::<f32>();
        let xb = x.reshape([1, k]).broadcast_as([n, k]);
        let out = (&wd * &xb).sum(1);
        assert_eq!(out.count_kernels_to_resolve(), 1, "fused quantized reduce");

        let slice = out.as_slice::<1, f32>().await.unwrap();
        for row in [0usize, 17, n - 1] {
            let expected: f32 = (0..k)
                .map(|kk| dequantized[row * k + kk] * x_values[kk])
                .sum();
            let actual = slice[[row]];
            assert!(
                (actual - expected).abs() < 1e-3,
                "[{row}]: got {actual}, expected {expected}"
            );
        }
    });
}

#[test]
fn weighted_sum_fuses_to_single_kernel() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };

        // `sum_k w[k] * x[m, k]`: the k-dependent inputs share no (row, col)
        // pair. The automatic dispatch policy decides whether to use the 1D
        // register tiling; both legal lowerings are reference-correct.
        let (m, k) = (512usize, 64usize);
        let x_values = pattern(m * k, 0.13);
        let w_values = pattern(k, 0.07);
        let x = Tensor::from_slice(&device, [m, k], &x_values);
        let w = Tensor::from_slice(&device, [k], &w_values);
        let w2 = w.reshape([1, k]).broadcast_as([m, k]);

        let out = (&x * &w2).sum(1);
        assert_eq!(out.count_kernels_to_resolve(), 1, "fused weighted sum");

        let slice = out.as_slice::<1, f32>().await.unwrap();
        for row in [0usize, 100, 255, m - 1] {
            let expected: f32 = (0..k).map(|kk| x_values[row * k + kk] * w_values[kk]).sum();
            let actual = slice[[row]];
            assert!(
                (actual - expected).abs() < 1e-4,
                "[{row}]: got {actual}, expected {expected}"
            );
        }
    });
}

#[test]
fn broadcast_table_elementwise_reuses_invariant_loads() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };

        // The table is 8 MiB — at the cache-residency threshold — so the
        // elementwise lowering tiles the batch dim and hoists the table
        // loads out of each thread's run of outputs.
        let (b, s, h) = (4usize, 1024usize, 2048usize);
        let x_values = pattern(b * s * h, 0.13);
        let t_values = pattern(s * h, 0.07);
        let x = Tensor::from_slice(&device, [b, s, h], &x_values);
        let t = Tensor::from_slice(&device, [s, h], &t_values);
        let t3 = t.reshape([1, s, h]).broadcast_as([b, s, h]);

        let out = &x * &t3;
        assert_eq!(out.count_kernels_to_resolve(), 1, "broadcast table apply");

        let slice = out.as_slice::<3, f32>().await.unwrap();
        for batch in 0..b {
            for (row, col) in [(0usize, 0usize), (511, 1023), (s - 1, h - 1)] {
                let expected = x_values[batch * s * h + row * h + col] * t_values[row * h + col];
                let actual = slice[[batch, row, col]];
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "[{batch}, {row}, {col}]: got {actual}, expected {expected}"
                );
            }
        }
    });
}

#[test]
fn contraction_with_k_independent_factor_fuses() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };

        // `c [M, N]` joins the product but never varies along K: it loads
        // once per output slot instead of being staged per k-tile.
        let (m, n, k) = (64usize, 64usize, 32usize);
        let a_values = pattern(m * k, 0.19);
        let b_values = pattern(n * k, 0.29);
        let c_values = pattern(m * n, 0.11);
        let (a3, b3) = broadcast_factors(&device, m, n, k, &a_values, &b_values);
        let c = Tensor::from_slice(&device, [m, n], &c_values);
        let c3 = c.reshape([m, n, 1]).broadcast_as([m, n, k]);

        let out = (&(&a3 * &b3) * &c3).sum(2);
        assert_eq!(
            out.count_kernels_to_resolve(),
            1,
            "contraction with k-independent factor"
        );

        let slice = out.as_slice::<2, f32>().await.unwrap();
        for row in [0usize, 17, m - 1] {
            for col in [0usize, 25, n - 1] {
                let expected: f32 = (0..k)
                    .map(|kk| {
                        a_values[row * k + kk] * b_values[col * k + kk] * c_values[row * n + col]
                    })
                    .sum();
                let actual = slice[[row, col]];
                assert!(
                    (actual - expected).abs() < 1e-3,
                    "[{row}, {col}]: got {actual}, expected {expected}"
                );
            }
        }
    });
}

#[test]
fn reshaped_elementwise_producer_folds_into_reduce() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };

        // The broadcast-gradient pattern: an elementwise product reshaped
        // flat and then summed. The reshape view folds into the reduce and
        // the producer inlines through the composed index expressions, so
        // the whole chain is one dispatch.
        let (b, s, h) = (4usize, 8usize, 16usize);
        let x_values = pattern(b * s * h, 0.13);
        let y_values = pattern(b * s * h, 0.29);
        let x = Tensor::from_slice(&device, [b, s, h], &x_values);
        let y = Tensor::from_slice(&device, [b, s, h], &y_values);

        let out = (&x * &y).reshape([b * s, h]).sum(0);
        assert_eq!(
            out.count_kernels_to_resolve(),
            1,
            "reshape + sum over an exclusive producer must fuse"
        );

        let slice = out.as_slice::<1, f32>().await.unwrap();
        for col in 0..h {
            let expected: f32 = (0..b * s)
                .map(|row| x_values[row * h + col] * y_values[row * h + col])
                .sum();
            let actual = slice[[col]];
            assert!(
                (actual - expected).abs() < 1e-3,
                "[{col}]: got {actual}, expected {expected}"
            );
        }
    });
}

#[test]
fn unary_chain_across_keepdim_view_fuses_into_reduce() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };

        // `sum_keepdim(x, 1) / k` applies the division at the keepdim'd
        // `[m, 1]` shape, behind the unsqueeze view. The chain rewrites into
        // a reduce of its own (indices substituted into the row dims, the
        // axis appended) instead of a separate scalar dispatch.
        let (m, k) = (64usize, 48usize);
        let x_values = pattern(m * k, 0.23);
        let x = Tensor::from_slice(&device, [m, k], &x_values);

        let mean = &x.sum_keepdim(1) / (k as f32);
        assert_eq!(
            mean.count_kernels_to_resolve(),
            1,
            "post-keepdim unary chain must fold into the reduce"
        );

        let slice = mean.as_slice::<2, f32>().await.unwrap();
        for row in [0usize, 17, m - 1] {
            let expected: f32 = (0..k).map(|col| x_values[row * k + col]).sum::<f32>() / k as f32;
            let actual = slice[[row, 0]];
            assert!(
                (actual - expected).abs() < 1e-4,
                "[{row}]: got {actual}, expected {expected}"
            );
        }
    });
}

#[test]
fn unit_axis_reduce_collapses_into_consumer() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };

        // Reducing a size-1 axis is the identity: the node collapses to an
        // elementwise gather and inlines into its (binary) consumer, which a
        // reduce post chain could never absorb.
        let (m, k) = (32usize, 24usize);
        let x_values = pattern(m * k, 0.11);
        let y_values = pattern(m * k, 0.31);
        let x = Tensor::from_slice(&device, [1, m, k], &x_values);
        let y = Tensor::from_slice(&device, [m, k], &y_values);

        let out = &x.sum(0) + &y;
        assert_eq!(
            out.count_kernels_to_resolve(),
            1,
            "size-1-axis reduce must collapse into the consumer"
        );

        let slice = out.as_slice::<2, f32>().await.unwrap();
        for (row, col) in [(0usize, 0usize), (13, 7), (m - 1, k - 1)] {
            let expected = x_values[row * k + col] + y_values[row * k + col];
            let actual = slice[[row, col]];
            assert!(
                (actual - expected).abs() < 1e-5,
                "[{row}, {col}]: got {actual}, expected {expected}"
            );
        }
    });
}
