//! Correctness gates for the generic attention row program across its
//! lowering regimes: single-tile decode, the online multi-tile streaming
//! loop (KV beyond one workgroup bucket), causal prefill (axis bound),
//! additive masks, GQA head mapping, and f16 IO.

use fusor_core::{Device, Tensor};

fn values(len: usize, scale: f32) -> Vec<f32> {
    (0..len).map(|i| ((i as f32) * scale).sin()).collect()
}

struct AttentionCase {
    batch: usize,
    heads: usize,
    kv_heads: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    causal: bool,
    masked: bool,
}

/// CPU reference for `softmax(q·kᵀ·scale [+ mask])·v` with GQA expansion.
#[allow(clippy::too_many_arguments)]
fn cpu_attention(
    case: &AttentionCase,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    mask: Option<&[f32]>,
    scale: f32,
) -> Vec<f32> {
    let AttentionCase {
        batch,
        heads,
        kv_heads,
        q_len,
        kv_len,
        head_dim,
        causal,
        ..
    } = *case;
    let groups = heads / kv_heads;
    let mut out = vec![0.0f32; batch * heads * q_len * head_dim];
    for b in 0..batch {
        for h in 0..heads {
            let kv_h = h / groups;
            for qi in 0..q_len {
                let q_base = ((b * heads + h) * q_len + qi) * head_dim;
                let mut scores = vec![f32::NEG_INFINITY; kv_len];
                for (pos, score) in scores.iter_mut().enumerate() {
                    if causal && pos > qi {
                        continue;
                    }
                    let k_base = ((b * kv_heads + kv_h) * kv_len + pos) * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_base + d] * k[k_base + d];
                    }
                    let mut value = dot * scale;
                    if let Some(mask) = mask {
                        value += mask[qi * kv_len + pos];
                    }
                    *score = value;
                }
                let max = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let weights: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
                let total: f32 = weights.iter().sum();
                for d in 0..head_dim {
                    let mut acc = 0.0f32;
                    for (pos, weight) in weights.iter().enumerate() {
                        let v_base = ((b * kv_heads + kv_h) * kv_len + pos) * head_dim;
                        acc += weight / total * v[v_base + d];
                    }
                    out[q_base + d] = acc;
                }
            }
        }
    }
    out
}

fn check_attention(case: AttentionCase, tolerance: f32) {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };
        let AttentionCase {
            batch,
            heads,
            kv_heads,
            q_len,
            kv_len,
            head_dim,
            causal,
            masked,
        } = case;
        let q_data = values(batch * heads * q_len * head_dim, 0.13);
        let k_data = values(batch * kv_heads * kv_len * head_dim, 0.07);
        let v_data = values(batch * kv_heads * kv_len * head_dim, 0.11);
        let mask_data = masked.then(|| {
            (0..q_len * kv_len)
                .map(|i| if i % 7 == 0 { -1.5 } else { 0.25 })
                .collect::<Vec<f32>>()
        });
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q = Tensor::from_slice(&device, [batch, heads, q_len, head_dim], &q_data);
        let k = Tensor::from_slice(&device, [batch, kv_heads, kv_len, head_dim], &k_data);
        let v = Tensor::from_slice(&device, [batch, kv_heads, kv_len, head_dim], &v_data);
        let mask_tensor = mask_data
            .as_ref()
            .map(|data| Tensor::from_slice(&device, [q_len, kv_len], data.as_slice()));

        let out = if causal {
            q.attention_causal(&k, &v, scale)
        } else {
            q.attention(&k, &v, scale, mask_tensor.as_ref())
        };
        assert_eq!(
            out.count_kernels_to_resolve(),
            1,
            "attention should lower as one row-program kernel"
        );
        let actual = out.as_slice::<4, f32>().await.unwrap();

        let expected = cpu_attention(
            &case,
            &q_data,
            &k_data,
            &v_data,
            mask_data.as_deref(),
            scale,
        );
        for b in 0..batch {
            for h in [0, heads - 1] {
                for qi in [0, q_len / 2, q_len - 1] {
                    for d in [0, head_dim / 2, head_dim - 1] {
                        let want = expected[((b * heads + h) * q_len + qi) * head_dim + d];
                        let got = actual[[b, h, qi, d]];
                        assert!(
                            (got - want).abs() < tolerance,
                            "b={b} h={h} q={qi} d={d}: got {got}, expected {want}"
                        );
                    }
                }
            }
        }
    });
}

#[test]
fn attention_decode_single_tile() {
    check_attention(
        AttentionCase {
            batch: 1,
            heads: 8,
            kv_heads: 2,
            q_len: 1,
            kv_len: 100,
            head_dim: 64,
            causal: false,
            masked: false,
        },
        1e-4,
    );
}

#[test]
fn attention_decode_streams_long_kv() {
    // KV beyond the largest workgroup bucket: the online tile loop with
    // rescaling, including a ragged final tile.
    check_attention(
        AttentionCase {
            batch: 1,
            heads: 4,
            kv_heads: 4,
            q_len: 1,
            kv_len: 3000,
            head_dim: 64,
            causal: false,
            masked: false,
        },
        1e-4,
    );
}

#[test]
fn attention_causal_prefill() {
    check_attention(
        AttentionCase {
            batch: 1,
            heads: 4,
            kv_heads: 2,
            q_len: 96,
            kv_len: 96,
            head_dim: 32,
            causal: true,
            masked: false,
        },
        1e-4,
    );
}

#[test]
fn attention_causal_prefill_streams_long_kv() {
    // Causal with q == kv beyond one bucket: the per-row axis bound must
    // stop each query row at its own position while later rows stream on.
    check_attention(
        AttentionCase {
            batch: 1,
            heads: 2,
            kv_heads: 2,
            q_len: 1536,
            kv_len: 1536,
            head_dim: 32,
            causal: true,
            masked: false,
        },
        1e-4,
    );
}

#[test]
fn attention_odd_sized_causal_prefill() {
    // Odd extents that don't align with any tile or bucket boundary.
    check_attention(
        AttentionCase {
            batch: 1,
            heads: 4,
            kv_heads: 2,
            q_len: 100,
            kv_len: 100,
            head_dim: 32,
            causal: true,
            masked: false,
        },
        1e-4,
    );
}

#[test]
fn attention_masked_prefill() {
    check_attention(
        AttentionCase {
            batch: 1,
            heads: 4,
            kv_heads: 4,
            q_len: 48,
            kv_len: 80,
            head_dim: 32,
            causal: false,
            masked: true,
        },
        1e-4,
    );
}

#[test]
fn attention_f16_io() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };
        if !device.f16_supported() {
            return;
        }
        let (heads, kv_len, head_dim) = (4usize, 64usize, 32usize);
        let q_data = values(heads * head_dim, 0.13);
        let k_data = values(heads * kv_len * head_dim, 0.07);
        let v_data = values(heads * kv_len * head_dim, 0.11);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let to_f16 = |data: &[f32]| {
            data.iter()
                .map(|&x| half::f16::from_f32(x))
                .collect::<Vec<_>>()
        };
        let q = Tensor::from_slice(&device, [1, heads, 1, head_dim], &to_f16(&q_data));
        let k = Tensor::from_slice(&device, [1, heads, kv_len, head_dim], &to_f16(&k_data));
        let v = Tensor::from_slice(&device, [1, heads, kv_len, head_dim], &to_f16(&v_data));

        let out = q.attention(&k, &v, scale, None);
        let actual = out.as_slice::<4, half::f16>().await.unwrap();

        let case = AttentionCase {
            batch: 1,
            heads,
            kv_heads: heads,
            q_len: 1,
            kv_len,
            head_dim,
            causal: false,
            masked: false,
        };
        let expected = cpu_attention(&case, &q_data, &k_data, &v_data, None, scale);
        for h in [0, heads - 1] {
            for d in [0, head_dim - 1] {
                let want = expected[h * head_dim + d];
                let got = actual[[0, h, 0, d]].to_f32();
                assert!(
                    (got - want).abs() < 1e-2,
                    "h={h} d={d}: got {got}, expected {want}"
                );
            }
        }
    });
}

#[test]
fn attention_many_rows_streams_tiles() {
    // Prefill-shaped rows with the axis spanning several online tiles.
    check_attention(
        AttentionCase {
            batch: 1,
            heads: 2,
            kv_heads: 2,
            q_len: 256,
            kv_len: 1536,
            head_dim: 32,
            causal: false,
            masked: false,
        },
        1e-4,
    );
}
