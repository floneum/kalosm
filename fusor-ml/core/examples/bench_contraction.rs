//! A/B kernel timing for the dense matmul routes and composed contractions.
//! Run on two builds and compare medians; wall-clock per resolve with a
//! device sync, warmup excluded.

use fusor_core::{Device, QMatrix, StrideSpec, Tensor};
use fusor_gguf::GgmlType;
use std::time::Instant;

fn values(len: usize, scale: f32) -> Vec<f32> {
    (0..len).map(|i| ((i as f32) * scale).sin()).collect()
}

fn time_case(name: &str, warmup: usize, iters: usize, mut run: impl FnMut()) {
    for _ in 0..warmup {
        run();
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        run();
        samples.push(start.elapsed().as_secs_f64() * 1e3);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = samples[samples.len() / 2];
    let min = samples[0];
    println!("{name}: median {median:.3} ms, min {min:.3} ms over {iters} iters");
}

fn main() {
    pollster::block_on(async {
        let device = Device::new().await.expect("gpu device");

        // Coop-ineligible dense matmul (1000 % 64 != 0): the workgroup-tiled
        // route.
        {
            let a = Tensor::from_slice(&device, [1000, 1000], &values(1_000_000, 0.13));
            let b = Tensor::from_slice(&device, [1000, 1000], &values(1_000_000, 0.07));
            time_case("matmul_1000_tiled", 3, 15, || {
                let out = a.mat_mul(&b);
                out.materialize_sync();
            });
        }

        // Coop-eligible square (512 divisible by 64): should be unchanged.
        {
            let a = Tensor::from_slice(&device, [512, 512], &values(512 * 512, 0.13));
            let b = Tensor::from_slice(&device, [512, 512], &values(512 * 512, 0.07));
            time_case("matmul_512_coop", 3, 30, || {
                let out = a.mat_mul(&b);
                out.materialize_sync();
            });
        }

        // Low tile utilization (65/96/65): the register-tile route.
        {
            let a = Tensor::from_slice(&device, [16, 65, 96], &values(16 * 65 * 96, 0.13));
            let b = Tensor::from_slice(&device, [16, 96, 65], &values(16 * 96 * 65, 0.07));
            time_case("matmul_batched_65_register", 3, 30, || {
                let out = a.mat_mul(&b);
                out.materialize_sync();
            });
        }

        // Weighted sum: `sum_k w[k] * x[m, k]` — k-dependent inputs share no
        // (row, col) pair, but `w` misses the row dim.
        {
            let (m, k) = (4096usize, 4096usize);
            let x = Tensor::from_slice(&device, [m, k], &values(m * k, 0.13));
            let w = Tensor::from_slice(&device, [k], &values(k, 0.07));
            time_case("weighted_sum_4096", 3, 30, || {
                let w2 = w.restride([StrideSpec::dim_with(0, m, 0), StrideSpec::dim(0, k)]);
                let out = (&x * &w2).sum(1);
                out.materialize_sync();
            });
        }

        // Broadcast scale: `x[m, n] * w[n]` — `w` is invariant along rows.
        {
            let (m, n) = (4096usize, 4096usize);
            let x = Tensor::from_slice(&device, [m, n], &values(m * n, 0.13));
            let w = Tensor::from_slice(&device, [n], &values(n, 0.07));
            time_case("broadcast_scale_4096", 3, 30, || {
                let w2 = w.restride([StrideSpec::dim_with(0, m, 0), StrideSpec::dim(0, n)]);
                let out = &x * &w2;
                out.materialize_sync();
            });
        }

        // Broadcast table apply: `x[b, s, h] * t[s, h]` with a table too
        // large for cache — the invariant table loads hoist out of each
        // thread's output run.
        {
            let (b, s, h) = (4usize, 2048usize, 2048usize);
            let x = Tensor::from_slice(&device, [b, s, h], &values(b * s * h, 0.13));
            let t = Tensor::from_slice(&device, [s, h], &values(s * h, 0.07));
            time_case("broadcast_table_4x2048x2048", 3, 30, || {
                let t3 = t.restride([
                    StrideSpec::dim_with(0, b, 0),
                    StrideSpec::dim(0, s),
                    StrideSpec::dim(1, h),
                ]);
                let out = &x * &t3;
                out.materialize_sync();
            });
        }

        // Dequantize a Q8_0 matrix to dense f32.
        {
            const Q8_BLOCK: usize = 32;
            let (n, k) = (4096usize, 4096usize);
            let scale = half::f16::from_f32(0.02);
            let mut bytes = Vec::with_capacity(n * k / Q8_BLOCK * 34);
            for block in 0..(n * k / Q8_BLOCK) {
                bytes.extend_from_slice(&scale.to_le_bytes());
                for i in 0..Q8_BLOCK {
                    bytes.push((((block * 7 + i * 5) % 64) as i32 - 32) as i8 as u8);
                }
            }
            let w = QMatrix::from_parts(
                &device,
                &bytes,
                vec![n, k].into_boxed_slice(),
                GgmlType::Q8_0,
            )
            .unwrap();
            time_case("dequantize_q8_4096", 3, 30, || {
                let out = w.dequantize::<f32>();
                out.materialize_sync();
            });
        }

        // Dense gemv: [m, k] x [k, 1].
        {
            let (m, k) = (4096usize, 4096usize);
            let a = Tensor::from_slice(&device, [m, k], &values(m * k, 0.13));
            let b = Tensor::from_slice(&device, [k, 1], &values(k, 0.07));
            time_case("gemv_4096", 3, 50, || {
                let out = a.mat_mul(&b);
                out.materialize_sync();
            });
        }

        // Norm shapes: decode-like single row and prefill-like many rows.
        {
            let hidden = 4096usize;
            let x1 = Tensor::from_slice(&device, [1, hidden], &values(hidden, 0.13));
            let xs = Tensor::from_slice(&device, [512, hidden], &values(512 * hidden, 0.13));
            let w = Tensor::from_slice(&device, [hidden], &values(hidden, 0.07));
            time_case("rms_norm_1x4096", 3, 50, || {
                let out = x1.rms_norm_fused(&w, None, 1e-5);
                out.materialize_sync();
            });
            time_case("rms_norm_512x4096", 3, 30, || {
                let out = xs.rms_norm_fused(&w, None, 1e-5);
                out.materialize_sync();
            });
            time_case("softmax_512x4096", 3, 30, || {
                let out = xs.softmax(1);
                out.materialize_sync();
            });
            let small = Tensor::from_slice(&device, [32, 128], &values(32 * 128, 0.17));
            time_case("softmax_32x128", 3, 50, || {
                let out = small.softmax(1);
                out.materialize_sync();
            });
        }

        // Decode-shape attention (q_len = 1): the per-token hot path. kv
        // lengths cover the single-dispatch decode buckets and (at 2048) the
        // split two-dispatch route.
        {
            let (batch, heads, head_dim) = (1usize, 32usize, 128usize);
            let scale = 1.0 / (head_dim as f32).sqrt();
            let q = Tensor::from_slice(
                &device,
                [batch, heads, 1, head_dim],
                &values(batch * heads * head_dim, 0.13),
            );
            for kv in [512usize, 1024, 2048, 4096, 8192] {
                let k = Tensor::from_slice(
                    &device,
                    [batch, heads, kv, head_dim],
                    &values(batch * heads * kv * head_dim, 0.07),
                );
                let v = Tensor::from_slice(
                    &device,
                    [batch, heads, kv, head_dim],
                    &values(batch * heads * kv * head_dim, 0.11),
                );
                time_case(&format!("attn_decode_kv{kv}"), 3, 50, || {
                    let out = q.flash_attention(&k, &v, scale, None);
                    out.materialize_sync();
                });
            }
            // Grouped-query variant: 32 query heads over 8 KV heads.
            let kv_heads = 8usize;
            let kv = 1024usize;
            let k = Tensor::from_slice(
                &device,
                [batch, kv_heads, kv, head_dim],
                &values(batch * kv_heads * kv * head_dim, 0.07),
            );
            let v = Tensor::from_slice(
                &device,
                [batch, kv_heads, kv, head_dim],
                &values(batch * kv_heads * kv * head_dim, 0.11),
            );
            time_case("attn_decode_gqa_kv1024", 3, 50, || {
                let out = q.flash_attention(&k, &v, scale, None);
                out.materialize_sync();
            });
        }

        // Prefill-shape attention: causal self-attention at q == kv (the
        // streaming-tiled regime) and a mid-length query block (the plain
        // streaming regime).
        {
            let (batch, heads, head_dim) = (1usize, 32usize, 128usize);
            let scale = 1.0 / (head_dim as f32).sqrt();
            let seq = 512usize;
            let q = Tensor::from_slice(
                &device,
                [batch, heads, seq, head_dim],
                &values(batch * heads * seq * head_dim, 0.13),
            );
            let k = Tensor::from_slice(
                &device,
                [batch, heads, seq, head_dim],
                &values(batch * heads * seq * head_dim, 0.07),
            );
            let v = Tensor::from_slice(
                &device,
                [batch, heads, seq, head_dim],
                &values(batch * heads * seq * head_dim, 0.11),
            );
            time_case("attn_prefill_512_causal", 3, 20, || {
                let out = q.flash_attention_causal(&k, &v, scale);
                out.materialize_sync();
            });

            let q64 = Tensor::from_slice(
                &device,
                [batch, heads, 64, head_dim],
                &values(batch * heads * 64 * head_dim, 0.13),
            );
            let kv = 1024usize;
            let k = Tensor::from_slice(
                &device,
                [batch, heads, kv, head_dim],
                &values(batch * heads * kv * head_dim, 0.07),
            );
            let v = Tensor::from_slice(
                &device,
                [batch, heads, kv, head_dim],
                &values(batch * heads * kv * head_dim, 0.11),
            );
            time_case("attn_prefill_q64_kv1024", 3, 30, || {
                let out = q64.flash_attention(&k, &v, scale, None);
                out.materialize_sync();
            });
        }

        // Composed broadcast contraction: not recognized as a matmul.
        {
            let (m, n, k) = (256usize, 256usize, 256usize);
            let a = Tensor::from_slice(&device, [m, k], &values(m * k, 0.13));
            let b = Tensor::from_slice(&device, [n, k], &values(n * k, 0.07));
            time_case("broadcast_contraction_256", 3, 30, || {
                let a3 = a.restride([
                    StrideSpec::dim(0, m),
                    StrideSpec::dim_with(0, n, 0),
                    StrideSpec::dim(1, k),
                ]);
                let b3 = b.restride([
                    StrideSpec::dim_with(0, m, 0),
                    StrideSpec::dim(0, n),
                    StrideSpec::dim(1, k),
                ]);
                let out = (&a3 * &b3).sum(2);
                out.materialize_sync();
            });
        }
    });
}
