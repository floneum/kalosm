use fusor::{Device, MaskKind, Tensor};
use fusor_conformance::{approx_eq, available_devices, f16_capable_devices};
use half::f16;

#[derive(Clone, Copy)]
struct FlashCase {
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    q_seq_len: usize,
    kv_seq_len: usize,
    head_dim: usize,
}

fn attention_data(len: usize, offset: f32) -> Vec<f32> {
    (0..len)
        .map(|i| (((i % 17) as f32) - 8.0) * 0.12 + offset)
        .collect()
}

fn qk_mask_data(q_seq_len: usize, kv_seq_len: usize) -> Vec<f32> {
    let mut data = vec![0.0; q_seq_len * kv_seq_len];
    for q in 0..q_seq_len {
        let allowed = (q + 1).min(kv_seq_len);
        for k in allowed..kv_seq_len {
            data[q * kv_seq_len + k] = f32::NEG_INFINITY;
        }
    }
    data
}

fn batch_key_mask_data(batch: usize, kv_seq_len: usize) -> Vec<f32> {
    let mut data = vec![0.0; batch * kv_seq_len];
    for b in 0..batch {
        let masked = (b + 1).min(kv_seq_len.saturating_sub(1));
        for k in (kv_seq_len - masked)..kv_seq_len {
            data[b * kv_seq_len + k] = f32::NEG_INFINITY;
        }
    }
    data
}

async fn assert_flash_attention_case_f16(
    case: FlashCase,
    mask: Option<(Vec<f32>, MaskKind, [usize; 2])>,
    tol: f16,
) {
    let q_data: Vec<f16> = attention_data(
        case.batch * case.num_heads * case.q_seq_len * case.head_dim,
        0.1,
    )
    .into_iter()
    .map(f16::from_f32)
    .collect();
    let k_data: Vec<f16> = attention_data(
        case.batch * case.num_kv_heads * case.kv_seq_len * case.head_dim,
        -0.1,
    )
    .into_iter()
    .map(f16::from_f32)
    .collect();
    let v_data: Vec<f16> = attention_data(
        case.batch * case.num_kv_heads * case.kv_seq_len * case.head_dim,
        0.05,
    )
    .into_iter()
    .map(f16::from_f32)
    .collect();
    let scale = 1.0 / (case.head_dim as f32).sqrt();

    let q_cpu: Tensor<4, f16> = Tensor::from_slice(
        &Device::Cpu,
        [case.batch, case.num_heads, case.q_seq_len, case.head_dim],
        &q_data,
    );
    let k_cpu: Tensor<4, f16> = Tensor::from_slice(
        &Device::Cpu,
        [
            case.batch,
            case.num_kv_heads,
            case.kv_seq_len,
            case.head_dim,
        ],
        &k_data,
    );
    let v_cpu: Tensor<4, f16> = Tensor::from_slice(
        &Device::Cpu,
        [
            case.batch,
            case.num_kv_heads,
            case.kv_seq_len,
            case.head_dim,
        ],
        &v_data,
    );
    let expected = if let Some((mask_data, kind, shape)) = mask.as_ref() {
        let mask_f16: Vec<f16> = mask_data.iter().copied().map(f16::from_f32).collect();
        let mask_cpu: Tensor<2, f16> = Tensor::from_slice(&Device::Cpu, *shape, &mask_f16);
        q_cpu
            .flash_attention(&k_cpu, &v_cpu, scale, Some((&mask_cpu, *kind)))
            .to_concrete()
    } else {
        q_cpu
            .flash_attention(&k_cpu, &v_cpu, scale, None)
            .to_concrete()
    };

    for device in f16_capable_devices().await {
        let q: Tensor<4, f16> = Tensor::from_slice(
            &device,
            [case.batch, case.num_heads, case.q_seq_len, case.head_dim],
            &q_data,
        );
        let k: Tensor<4, f16> = Tensor::from_slice(
            &device,
            [
                case.batch,
                case.num_kv_heads,
                case.kv_seq_len,
                case.head_dim,
            ],
            &k_data,
        );
        let v: Tensor<4, f16> = Tensor::from_slice(
            &device,
            [
                case.batch,
                case.num_kv_heads,
                case.kv_seq_len,
                case.head_dim,
            ],
            &v_data,
        );
        let actual = if let Some((mask_data, kind, shape)) = mask.as_ref() {
            let mask_f16: Vec<f16> = mask_data.iter().copied().map(f16::from_f32).collect();
            let device_mask: Tensor<2, f16> = Tensor::from_slice(&device, *shape, &mask_f16);
            q.flash_attention(&k, &v, scale, Some((&device_mask, *kind)))
                .to_concrete()
        } else {
            q.flash_attention(&k, &v, scale, None).to_concrete()
        };
        approx_eq(&actual, &expected, tol).await.unwrap();
    }
}

#[tokio::test]
async fn flash_attention_f16_matches_cpu_reference_on_varied_shapes() {
    for case in [
        FlashCase {
            batch: 1,
            num_heads: 1,
            num_kv_heads: 1,
            q_seq_len: 2,
            kv_seq_len: 2,
            head_dim: 2,
        },
        FlashCase {
            batch: 2,
            num_heads: 2,
            num_kv_heads: 2,
            q_seq_len: 4,
            kv_seq_len: 5,
            head_dim: 3,
        },
        FlashCase {
            batch: 1,
            num_heads: 2,
            num_kv_heads: 1,
            q_seq_len: 1,
            kv_seq_len: 9,
            head_dim: 128,
        },
    ] {
        assert_flash_attention_case_f16(case, None, f16::from_f32(5e-3)).await;
    }
}

#[tokio::test]
async fn flash_attention_f16_with_qk_mask_matches_cpu_reference() {
    for case in [
        FlashCase {
            batch: 1,
            num_heads: 1,
            num_kv_heads: 1,
            q_seq_len: 2,
            kv_seq_len: 2,
            head_dim: 2,
        },
        FlashCase {
            batch: 1,
            num_heads: 2,
            num_kv_heads: 2,
            q_seq_len: 5,
            kv_seq_len: 5,
            head_dim: 4,
        },
    ] {
        let shape = [case.q_seq_len, case.kv_seq_len];
        assert_flash_attention_case_f16(
            case,
            Some((
                qk_mask_data(case.q_seq_len, case.kv_seq_len),
                MaskKind::QKMask,
                shape,
            )),
            f16::from_f32(5e-3),
        )
        .await;
    }
}

async fn assert_flash_attention_case(
    case: FlashCase,
    mask: Option<(Vec<f32>, MaskKind, [usize; 2])>,
    tol: f32,
) {
    let q_data = attention_data(
        case.batch * case.num_heads * case.q_seq_len * case.head_dim,
        0.1,
    );
    let k_data = attention_data(
        case.batch * case.num_kv_heads * case.kv_seq_len * case.head_dim,
        -0.15,
    );
    let v_data = attention_data(
        case.batch * case.num_kv_heads * case.kv_seq_len * case.head_dim,
        0.35,
    );
    let scale = 1.0 / (case.head_dim as f32).sqrt();

    let q_cpu = Tensor::from_slice(
        &Device::Cpu,
        [case.batch, case.num_heads, case.q_seq_len, case.head_dim],
        &q_data,
    );
    let k_cpu = Tensor::from_slice(
        &Device::Cpu,
        [
            case.batch,
            case.num_kv_heads,
            case.kv_seq_len,
            case.head_dim,
        ],
        &k_data,
    );
    let v_cpu = Tensor::from_slice(
        &Device::Cpu,
        [
            case.batch,
            case.num_kv_heads,
            case.kv_seq_len,
            case.head_dim,
        ],
        &v_data,
    );
    let expected = if let Some((mask_data, kind, shape)) = mask.as_ref() {
        let mask_cpu = Tensor::from_slice(&Device::Cpu, *shape, mask_data);
        q_cpu
            .flash_attention(&k_cpu, &v_cpu, scale, Some((&mask_cpu, *kind)))
            .to_concrete()
    } else {
        q_cpu
            .flash_attention(&k_cpu, &v_cpu, scale, None)
            .to_concrete()
    };

    for device in available_devices().await {
        let q = Tensor::from_slice(
            &device,
            [case.batch, case.num_heads, case.q_seq_len, case.head_dim],
            &q_data,
        );
        let k = Tensor::from_slice(
            &device,
            [
                case.batch,
                case.num_kv_heads,
                case.kv_seq_len,
                case.head_dim,
            ],
            &k_data,
        );
        let v = Tensor::from_slice(
            &device,
            [
                case.batch,
                case.num_kv_heads,
                case.kv_seq_len,
                case.head_dim,
            ],
            &v_data,
        );
        let actual = if let Some((mask_data, kind, shape)) = mask.as_ref() {
            let device_mask = Tensor::from_slice(&device, *shape, mask_data);
            q.flash_attention(&k, &v, scale, Some((&device_mask, *kind)))
                .to_concrete()
        } else {
            q.flash_attention(&k, &v, scale, None).to_concrete()
        };
        approx_eq(&actual, &expected, tol).await.unwrap();
    }
}

/// Exercises the **tiled** branch of the Metal decode-small kernel
/// (`flash_decode_small_block` in `fusor-ml/tile-ir-kernels/src/kernels/flash.rs`).
/// On Metal `choose_decode_block` always returns `Small` (BLOCK=128), so
/// `tiled = kv_seq_len > 128`. The rest of the conformance suite never
/// crosses that boundary with `head_dim = 128`, which is why the Qwen-3B
/// decode bug (`flash_attention` output entirely NaN for one head at
/// `kv_seq_len ≈ 569`) slipped through.
///
/// Each shape is run multiple times because the failure is non-deterministic
/// (looks like a workgroup-memory race in the tiled kernel).
#[tokio::test]
async fn flash_attention_decode_tiled_matches_cpu_reference() {
    // (num_heads, num_kv_heads, kv_seq_len)
    // Shapes specifically chosen to stress the tiled flash_decode_small_block
    // path. head_dim=128 to force the decode-small kernel; kv_seq_len chosen
    // to span tile counts (2..=5 tiles) and to land just over the BLOCK=128
    // boundary.
    let shapes = [
        (16, 2, 129), // just past one full tile
        (16, 2, 192), // mid-second tile
        (16, 2, 256), // exactly two full tiles
        (16, 2, 257), // just past two full tiles (start of third)
        (16, 2, 384), // exactly three full tiles
        (16, 2, 511), // last lane of fourth tile inactive
        (16, 2, 512), // exactly four full tiles
        (16, 2, 569), // five tiles, matches Qwen-3B decode failure
        (32, 8, 200), // larger GQA group, kv_seq_len in second tile
    ];

    for (num_heads, num_kv_heads, kv_seq_len) in shapes {
        let case = FlashCase {
            batch: 1,
            num_heads,
            num_kv_heads,
            q_seq_len: 1,
            kv_seq_len,
            head_dim: 128,
        };
        // Run each shape several times to catch non-deterministic kernel races.
        for trial in 0..4 {
            assert_flash_attention_case(case, None, 1e-3).await;
            let _ = trial;
        }
    }
}

/// Same as the tiled test above, but builds Q with non-canonical strides
/// (reshape + transpose) — matching how the real attention path produces Q
/// in `models/kalosm-llama/src/raw/attention_layer.rs`. The kernel reads Q
/// via `index_n(meta.q_offset, meta.q_strides, ...)`, so different strides
/// hit different memory addresses and exercise different control flow paths
/// inside `flash_decode_small_block`.
#[tokio::test]
async fn flash_attention_decode_tiled_with_transposed_q_matches_cpu_reference() {
    let shapes = [(16, 2, 129), (16, 2, 257), (16, 2, 384), (16, 2, 569)];

    for (num_heads, num_kv_heads, kv_seq_len) in shapes {
        let head_dim = 128;
        let batch = 1;
        let q_seq_len = 1;

        // Build Q starting in [batch, q_seq_len, num_heads, head_dim]
        // contiguous layout, then transpose(1, 2) to land at the kernel's
        // expected [batch, num_heads, q_seq_len, head_dim] shape with
        // non-canonical strides ([H*D, D, H*D, 1] when q_seq_len=1).
        let q_data = attention_data(batch * num_heads * q_seq_len * head_dim, 0.1);
        let k_data = attention_data(batch * num_kv_heads * kv_seq_len * head_dim, -0.15);
        let v_data = attention_data(batch * num_kv_heads * kv_seq_len * head_dim, 0.35);
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Reference computed on CPU with canonical-stride Q.
        let q_cpu: Tensor<4, f32> = Tensor::from_slice(
            &Device::Cpu,
            [batch, num_heads, q_seq_len, head_dim],
            &q_data,
        );
        let k_cpu: Tensor<4, f32> = Tensor::from_slice(
            &Device::Cpu,
            [batch, num_kv_heads, kv_seq_len, head_dim],
            &k_data,
        );
        let v_cpu: Tensor<4, f32> = Tensor::from_slice(
            &Device::Cpu,
            [batch, num_kv_heads, kv_seq_len, head_dim],
            &v_data,
        );
        let expected = q_cpu
            .flash_attention(&k_cpu, &v_cpu, scale, None)
            .to_concrete();

        for device in available_devices().await {
            // Lay Q out as [batch, q_seq_len, num_heads, head_dim] (the way
            // it comes out of the QKV matmul before reshape+transpose) and
            // then transpose(1, 2) to get [batch, num_heads, q_seq_len, head_dim]
            // with non-canonical strides — the same layout the model uses.
            let q_pre: Tensor<4, f32> =
                Tensor::from_slice(&device, [batch, q_seq_len, num_heads, head_dim], &q_data);
            let q = q_pre.transpose(1, 2).to_concrete();
            let k: Tensor<4, f32> = Tensor::from_slice(
                &device,
                [batch, num_kv_heads, kv_seq_len, head_dim],
                &k_data,
            );
            let v: Tensor<4, f32> = Tensor::from_slice(
                &device,
                [batch, num_kv_heads, kv_seq_len, head_dim],
                &v_data,
            );
            // Several trials to catch races.
            for _ in 0..4 {
                let actual = q.flash_attention(&k, &v, scale, None).to_concrete();
                approx_eq(&actual, &expected, 1e-3).await.unwrap();
            }
        }
    }
}

#[tokio::test]
async fn flash_attention_subgroup_fallback_preserves_gpu_backend() {
    let q_shape = [1, 1, 2, 4];
    let kv_shape = [1, 1, 3, 4];
    let q_data = attention_data(q_shape.iter().product(), 0.1);
    let k_data = attention_data(kv_shape.iter().product(), -0.15);
    let v_data = attention_data(kv_shape.iter().product(), 0.35);
    let scale = 1.0 / (q_shape[3] as f32).sqrt();

    for device in available_devices().await {
        let Some(gpu) = device.as_gpu() else {
            continue;
        };
        if gpu.subgroups_supported() {
            continue;
        }

        let q = Tensor::from_slice(&device, q_shape, &q_data);
        let k = Tensor::from_slice(&device, kv_shape, &k_data);
        let v = Tensor::from_slice(&device, kv_shape, &v_data);
        let output = q.flash_attention(&k, &v, scale, None);
        assert!(
            output.is_gpu(),
            "subgroup fallback should preserve the GPU backend"
        );
    }
}

#[tokio::test]
async fn flash_attention_matches_cpu_reference_on_varied_shapes() {
    for case in [
        FlashCase {
            batch: 1,
            num_heads: 1,
            num_kv_heads: 1,
            q_seq_len: 2,
            kv_seq_len: 2,
            head_dim: 2,
        },
        FlashCase {
            batch: 2,
            num_heads: 2,
            num_kv_heads: 2,
            q_seq_len: 4,
            kv_seq_len: 5,
            head_dim: 3,
        },
        FlashCase {
            batch: 1,
            num_heads: 3,
            num_kv_heads: 3,
            q_seq_len: 5,
            kv_seq_len: 3,
            head_dim: 4,
        },
        FlashCase {
            batch: 1,
            num_heads: 2,
            num_kv_heads: 1,
            q_seq_len: 1,
            kv_seq_len: 9,
            head_dim: 128,
        },
    ] {
        assert_flash_attention_case(case, None, 1e-4).await;
    }
}

#[tokio::test]
async fn flash_attention_with_qk_mask_matches_cpu_reference_on_varied_shapes() {
    for case in [
        FlashCase {
            batch: 1,
            num_heads: 1,
            num_kv_heads: 1,
            q_seq_len: 2,
            kv_seq_len: 2,
            head_dim: 2,
        },
        FlashCase {
            batch: 2,
            num_heads: 3,
            num_kv_heads: 3,
            q_seq_len: 4,
            kv_seq_len: 6,
            head_dim: 3,
        },
        FlashCase {
            batch: 1,
            num_heads: 2,
            num_kv_heads: 2,
            q_seq_len: 5,
            kv_seq_len: 5,
            head_dim: 4,
        },
    ] {
        let shape = [case.q_seq_len, case.kv_seq_len];
        assert_flash_attention_case(
            case,
            Some((
                qk_mask_data(case.q_seq_len, case.kv_seq_len),
                MaskKind::QKMask,
                shape,
            )),
            1e-4,
        )
        .await;
    }
}

#[tokio::test]
async fn flash_attention_gqa_matches_cpu_reference_on_varied_shapes() {
    for case in [
        FlashCase {
            batch: 1,
            num_heads: 4,
            num_kv_heads: 2,
            q_seq_len: 2,
            kv_seq_len: 2,
            head_dim: 2,
        },
        FlashCase {
            batch: 2,
            num_heads: 6,
            num_kv_heads: 2,
            q_seq_len: 4,
            kv_seq_len: 5,
            head_dim: 3,
        },
        FlashCase {
            batch: 1,
            num_heads: 8,
            num_kv_heads: 4,
            q_seq_len: 3,
            kv_seq_len: 6,
            head_dim: 4,
        },
    ] {
        assert_flash_attention_case(case, None, 1e-4).await;
    }
}

#[tokio::test]
async fn flash_attention_with_kv_cache_matches_cpu_reference_on_varied_shapes() {
    // KV-cache regression: short Q sequence with a longer K/V sequence — the
    // typical autoregressive decode shape after appending to a KvCache.
    // Replaces the deleted `core/src/composite/flash_attention.rs::test_flash_attention_kv_cache_fuzz`.
    for case in [
        FlashCase {
            batch: 1,
            num_heads: 1,
            num_kv_heads: 1,
            q_seq_len: 1,
            kv_seq_len: 5,
            head_dim: 4,
        },
        FlashCase {
            batch: 2,
            num_heads: 4,
            num_kv_heads: 4,
            q_seq_len: 1,
            kv_seq_len: 16,
            head_dim: 8,
        },
        FlashCase {
            batch: 1,
            num_heads: 32,
            num_kv_heads: 8,
            q_seq_len: 1,
            kv_seq_len: 10,
            head_dim: 128,
        },
        FlashCase {
            batch: 2,
            num_heads: 8,
            num_kv_heads: 8,
            q_seq_len: 2,
            kv_seq_len: 17,
            head_dim: 16,
        },
        FlashCase {
            batch: 1,
            num_heads: 6,
            num_kv_heads: 2,
            q_seq_len: 3,
            kv_seq_len: 32,
            head_dim: 12,
        },
    ] {
        // No-op mask exercises the KV-cache path through the QKMask code with
        // an all-zero mask, mirroring the pre-migration test.
        let mask = vec![0.0f32; case.q_seq_len * case.kv_seq_len];
        let shape = [case.q_seq_len, case.kv_seq_len];
        assert_flash_attention_case(case, Some((mask, MaskKind::QKMask, shape)), 1e-3).await;
    }
}

#[tokio::test]
async fn flash_attention_with_batch_key_mask_matches_cpu_reference_on_varied_shapes() {
    for case in [
        FlashCase {
            batch: 2,
            num_heads: 1,
            num_kv_heads: 1,
            q_seq_len: 3,
            kv_seq_len: 3,
            head_dim: 2,
        },
        FlashCase {
            batch: 3,
            num_heads: 2,
            num_kv_heads: 2,
            q_seq_len: 4,
            kv_seq_len: 5,
            head_dim: 3,
        },
        FlashCase {
            batch: 2,
            num_heads: 4,
            num_kv_heads: 4,
            q_seq_len: 2,
            kv_seq_len: 6,
            head_dim: 4,
        },
    ] {
        let shape = [case.batch, case.kv_seq_len];
        assert_flash_attention_case(
            case,
            Some((
                batch_key_mask_data(case.batch, case.kv_seq_len),
                MaskKind::BatchKeyMask,
                shape,
            )),
            1e-4,
        )
        .await;
    }
}

/// Exercises the tiled (Q-batched) streaming flash attention kernel. The
/// selector switches to that variant when `q_seq_len >= 64` and `head_dim` is
/// a multiple of 8. Shapes are chosen to span:
/// - exact Q-block alignment (q_seq_len = 64 = 8*8),
/// - non-aligned tail (q_seq_len = 72, last block is partially valid),
/// - non-aligned head_dim and odd kv_seq_len.
#[tokio::test]
async fn flash_attention_tiled_matches_cpu_reference_on_varied_shapes() {
    for case in [
        FlashCase {
            batch: 1,
            num_heads: 2,
            num_kv_heads: 2,
            q_seq_len: 64,
            kv_seq_len: 64,
            head_dim: 8,
        },
        FlashCase {
            batch: 1,
            num_heads: 2,
            num_kv_heads: 1,
            q_seq_len: 72,
            kv_seq_len: 80,
            head_dim: 16,
        },
        FlashCase {
            batch: 2,
            num_heads: 4,
            num_kv_heads: 2,
            q_seq_len: 65,
            kv_seq_len: 35,
            head_dim: 8,
        },
        FlashCase {
            batch: 1,
            num_heads: 4,
            num_kv_heads: 4,
            q_seq_len: 128,
            kv_seq_len: 128,
            head_dim: 24,
        },
    ] {
        assert_flash_attention_case(case, None, 1e-3).await;
    }
}

/// Same as above but with an additive QK mask, exercising the masked path
/// through the tiled kernel.
#[tokio::test]
async fn flash_attention_tiled_with_mask_matches_cpu_reference() {
    for case in [
        FlashCase {
            batch: 1,
            num_heads: 2,
            num_kv_heads: 2,
            q_seq_len: 64,
            kv_seq_len: 32,
            head_dim: 8,
        },
        FlashCase {
            batch: 2,
            num_heads: 2,
            num_kv_heads: 2,
            q_seq_len: 96,
            kv_seq_len: 64,
            head_dim: 16,
        },
    ] {
        let shape = [case.q_seq_len, case.kv_seq_len];
        assert_flash_attention_case(
            case,
            Some((
                qk_mask_data(case.q_seq_len, case.kv_seq_len),
                MaskKind::QKMask,
                shape,
            )),
            1e-3,
        )
        .await;
    }
}
