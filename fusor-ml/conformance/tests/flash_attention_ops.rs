use fusor::{Device, MaskKind, Tensor};
use fusor_conformance::{approx_eq, available_devices};

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

#[tokio::test]
async fn flash_attention_batched_matches_per_item() {
    let Ok(device) = Device::gpu().await else {
        return;
    };

    for case in [
        FlashCase {
            batch: 4,
            num_heads: 2,
            num_kv_heads: 2,
            q_seq_len: 5,
            kv_seq_len: 7,
            head_dim: 8,
        },
        FlashCase {
            batch: 8,
            num_heads: 8,
            num_kv_heads: 8,
            q_seq_len: 6,
            kv_seq_len: 256,
            head_dim: 32,
        },
        FlashCase {
            batch: 8,
            num_heads: 8,
            num_kv_heads: 8,
            q_seq_len: 256,
            kv_seq_len: 6,
            head_dim: 32,
        },
    ] {
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

        let batched = q.flash_attention(&k, &v, scale, None).to_concrete();
        let mut items = Vec::with_capacity(case.batch);
        for batch in 0..case.batch {
            let q_i = q.narrow(0, batch, 1).to_concrete();
            let k_i = k.narrow(0, batch, 1).to_concrete();
            let v_i = v.narrow(0, batch, 1).to_concrete();
            items.push(q_i.flash_attention(&k_i, &v_i, scale, None).to_concrete());
        }
        let per_item = Tensor::cat(items, 0).to_concrete();

        approx_eq(&batched, &per_item, 1e-4).await.unwrap();
    }
}
