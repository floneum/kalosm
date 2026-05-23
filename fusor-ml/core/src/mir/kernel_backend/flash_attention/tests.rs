use super::*;
use crate::{
    Device, Tensor,
    kernel_selection::{CooperativeMatrixCaps, assert_selector_generates},
};

const TEST_HEAD_DIM: usize = DECODE_HEAD_DIM as usize;

fn caps(max_compute_invocations_per_workgroup: u32) -> KernelDeviceCaps {
    KernelDeviceCaps {
        cooperative_matrix: CooperativeMatrixCaps::default(),
        max_compute_invocations_per_workgroup,
        ..KernelDeviceCaps::test_caps()
    }
}

fn tensor_meta4() -> TensorMeta {
    TensorMeta {
        datatype: DataTypeEnum::F32,
        tile: tile_ir_kernels::TensorMeta::new(vec![65_536, 8_192, 128, 1], 0),
    }
}

fn decode_dims(kv_seq_len: u32) -> FlashAttentionDims {
    FlashAttentionDims {
        batch: 1,
        num_heads: 32,
        num_kv_heads: 8,
        q_seq_len: 1,
        kv_seq_len,
        head_dim: DECODE_HEAD_DIM,
    }
}

fn decode_shape(kv_seq_len: usize) -> KernelShape<6> {
    KernelShape::new([1, 32, 8, 1, kv_seq_len, DECODE_HEAD_DIM as usize])
}

#[test]
fn decode_block_choice_uses_smallest_covering_supported_block() {
    assert_eq!(
        choose_decode_block(64, caps(DECODE_LARGE_BLOCK)),
        Some(DecodeBlock::Small)
    );
    assert_eq!(
        choose_decode_block(200, caps(DECODE_LARGE_BLOCK)),
        Some(DecodeBlock::Medium)
    );
    assert_eq!(
        choose_decode_block(600, caps(DECODE_LARGE_BLOCK)),
        Some(DecodeBlock::Large)
    );
    assert_eq!(
        choose_decode_block(600, caps(DECODE_MEDIUM_BLOCK)),
        Some(DecodeBlock::Medium)
    );
    assert_eq!(
        choose_decode_block(DECODE_LARGE_BLOCK + 1, caps(DECODE_LARGE_BLOCK)),
        Some(DecodeBlock::Large)
    );
    assert_eq!(
        choose_decode_block(DECODE_SMALL_BLOCK, caps(DECODE_SMALL_BLOCK - 1)),
        None
    );
}

#[test]
fn decode_small_meta_buckets_dynamic_kv_len() {
    let meta = build_flash_decode_small_meta(
        decode_dims(DECODE_SMALL_BLOCK + 1),
        1.0,
        caps(DECODE_LARGE_BLOCK),
        FlashDecodeSmallTensors {
            q: tensor_meta4(),
            k: tensor_meta4(),
            v: tensor_meta4(),
            mask: None,
            output: tensor_meta4(),
        },
    )
    .unwrap();

    assert_eq!(meta.active_kv_len, DECODE_SMALL_BLOCK + 1);
    assert_eq!(meta.decode_block, DECODE_MEDIUM_BLOCK);
    assert_eq!(meta.dims.kv_seq_len, DECODE_MEDIUM_BLOCK);
    assert!(!meta.tiled);
}

#[test]
fn decode_small_meta_tiles_with_largest_supported_block() {
    let meta = build_flash_decode_small_meta(
        decode_dims(DECODE_MEDIUM_BLOCK + 1),
        1.0,
        caps(DECODE_MEDIUM_BLOCK),
        FlashDecodeSmallTensors {
            q: tensor_meta4(),
            k: tensor_meta4(),
            v: tensor_meta4(),
            mask: None,
            output: tensor_meta4(),
        },
    );

    let meta = meta.unwrap();
    assert_eq!(meta.active_kv_len, DECODE_MEDIUM_BLOCK + 1);
    assert_eq!(meta.decode_block, DECODE_MEDIUM_BLOCK);
    assert_eq!(meta.dims.kv_seq_len, DECODE_MEDIUM_BLOCK);
    assert!(meta.tiled);
}

#[test]
fn decode_small_meta_requires_minimum_workgroup_limit() {
    let meta = build_flash_decode_small_meta(
        decode_dims(DECODE_SMALL_BLOCK),
        1.0,
        caps(DECODE_SMALL_BLOCK - 1),
        FlashDecodeSmallTensors {
            q: tensor_meta4(),
            k: tensor_meta4(),
            v: tensor_meta4(),
            mask: None,
            output: tensor_meta4(),
        },
    );

    assert!(meta.is_none());
}

#[test]
fn flash_attention_selector_selects_decode_block_buckets() {
    let selector = flash_attention_selector();
    let decode_ctx = FlashAttentionSelectionCtx { has_mask: false };
    let masked_ctx = FlashAttentionSelectionCtx { has_mask: true };

    assert_eq!(
        selector.select(decode_shape(64), &decode_ctx, caps(DECODE_LARGE_BLOCK)),
        Some(FlashAttentionSelectedVariant::DecodeSmall(
            DecodeBlock::Small
        ))
    );
    assert_eq!(
        selector.select(decode_shape(200), &decode_ctx, caps(DECODE_LARGE_BLOCK)),
        Some(FlashAttentionSelectedVariant::DecodeSmall(
            DecodeBlock::Medium
        ))
    );
    assert_eq!(
        selector.select(decode_shape(600), &decode_ctx, caps(DECODE_LARGE_BLOCK)),
        Some(FlashAttentionSelectedVariant::DecodeSmall(
            DecodeBlock::Large
        ))
    );
    assert_eq!(
        selector.select(decode_shape(600), &decode_ctx, caps(DECODE_MEDIUM_BLOCK)),
        Some(FlashAttentionSelectedVariant::DecodeSmall(
            DecodeBlock::Medium
        ))
    );
    assert_eq!(
        selector.select(
            decode_shape(DECODE_LARGE_BLOCK as usize + 1),
            &decode_ctx,
            caps(DECODE_LARGE_BLOCK)
        ),
        Some(FlashAttentionSelectedVariant::DecodeSmall(
            DecodeBlock::Large
        ))
    );
    assert_eq!(
        selector.select(decode_shape(200), &decode_ctx, caps(DECODE_SMALL_BLOCK - 1)),
        Some(FlashAttentionSelectedVariant::Streaming)
    );
    assert_eq!(
        selector.select(decode_shape(200), &masked_ctx, caps(DECODE_LARGE_BLOCK)),
        Some(FlashAttentionSelectedVariant::Streaming)
    );
}

#[test]
fn flash_attention_selector_generates_each_variant() {
    let selector = flash_attention_selector();
    let decode_ctx = FlashAttentionSelectionCtx { has_mask: false };
    let streaming_ctx = FlashAttentionSelectionCtx { has_mask: true };
    let cases = [
        (
            FlashAttentionSelectedVariant::DecodeSmall(DecodeBlock::Small),
            decode_ctx,
            caps(DECODE_SMALL_BLOCK),
        ),
        (
            FlashAttentionSelectedVariant::DecodeSmall(DecodeBlock::Medium),
            decode_ctx,
            caps(DECODE_MEDIUM_BLOCK),
        ),
        (
            FlashAttentionSelectedVariant::DecodeSmall(DecodeBlock::Large),
            decode_ctx,
            caps(DECODE_LARGE_BLOCK),
        ),
        (
            FlashAttentionSelectedVariant::Streaming,
            streaming_ctx,
            caps(DECODE_LARGE_BLOCK),
        ),
    ];
    assert_selector_generates(&selector, cases);
}

type AttentionFixture = Vec<Vec<Vec<Vec<f32>>>>;

fn attention_fixture(
    heads: usize,
    tokens: usize,
    f: impl Fn(usize, usize, usize) -> f32,
) -> AttentionFixture {
    vec![
        (0..heads)
            .map(|head| {
                (0..tokens)
                    .map(|token| (0..TEST_HEAD_DIM).map(|dim| f(head, token, dim)).collect())
                    .collect()
            })
            .collect(),
    ]
}

fn decode_q() -> AttentionFixture {
    attention_fixture(1, 1, |_, _, dim| ((dim % 17) as f32 - 8.0) * 0.0075)
}

fn decode_k(kv_len: usize) -> AttentionFixture {
    attention_fixture(1, kv_len, |_, token, dim| {
        let value = ((token * 13 + dim * 7) % 31) as f32 - 15.0;
        value * 0.004
    })
}

fn decode_v(kv_len: usize) -> AttentionFixture {
    attention_fixture(1, kv_len, |_, token, dim| {
        let value = ((token * 5 + dim * 11) % 37) as f32 - 18.0;
        0.25 + value * 0.01
    })
}

fn decode_q_gqa(num_heads: usize) -> AttentionFixture {
    attention_fixture(num_heads, 1, |head, _, dim| {
        let value = ((head * 19 + dim * 7) % 43) as f32 - 21.0;
        value * 0.003
    })
}

fn decode_k_gqa(num_kv_heads: usize, kv_len: usize) -> AttentionFixture {
    attention_fixture(num_kv_heads, kv_len, |kv_head, token, dim| {
        let value = ((kv_head * 23 + token * 13 + dim * 5) % 47) as f32 - 23.0;
        value * 0.0025
    })
}

fn decode_v_gqa(num_kv_heads: usize, kv_len: usize) -> AttentionFixture {
    attention_fixture(num_kv_heads, kv_len, |kv_head, token, dim| {
        let value = ((kv_head * 29 + token * 3 + dim * 11) % 53) as f32 - 26.0;
        0.05 + value * 0.004
    })
}

fn cpu_decode_reference(q: &[f32], k: &[Vec<f32>], v: &[Vec<f32>], scale: f32) -> Vec<f32> {
    let scores = k
        .iter()
        .map(|key| {
            q.iter()
                .zip(key)
                .map(|(q, k)| (*q as f64) * (*k as f64))
                .sum::<f64>()
                * scale as f64
        })
        .collect::<Vec<_>>();
    let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let denom = scores
        .iter()
        .map(|score| (score - max_score).exp())
        .sum::<f64>();
    let mut output = vec![0.0; TEST_HEAD_DIM];
    for (token, score) in scores.iter().copied().enumerate() {
        let prob = (score - max_score).exp() / denom;
        for (dim, output) in output.iter_mut().enumerate() {
            *output += prob * v[token][dim] as f64;
        }
    }
    output.into_iter().map(|value| value as f32).collect()
}

fn decode_max_error(
    num_heads: usize,
    groups: usize,
    q_data: &AttentionFixture,
    k_data: &AttentionFixture,
    v_data: &AttentionFixture,
    scale: f32,
    actual: impl Fn(usize, usize) -> f32,
) -> (f32, usize, usize, f32, f32) {
    let mut max_error = 0.0f32;
    let mut max_head = 0usize;
    let mut max_dim = 0usize;
    let mut max_actual = 0.0f32;
    let mut max_expected = 0.0f32;
    for (head, q_head) in q_data[0].iter().enumerate().take(num_heads) {
        let kv_head = head / groups;
        let expected =
            cpu_decode_reference(&q_head[0], &k_data[0][kv_head], &v_data[0][kv_head], scale);
        for (dim, expected) in expected.into_iter().enumerate() {
            let actual = actual(head, dim);
            let error = (actual - expected).abs();
            if error > max_error {
                max_error = error;
                max_head = head;
                max_dim = dim;
                max_actual = actual;
                max_expected = expected;
            }
        }
    }
    (max_error, max_head, max_dim, max_actual, max_expected)
}

#[tokio::test]
async fn tiled_decode_attention_matches_cpu_reference() {
    let Ok(device) = Device::new().await else {
        return;
    };

    let kv_len = DECODE_LARGE_BLOCK as usize + 1;
    let q_data = decode_q();
    let k_data = decode_k(kv_len);
    let v_data = decode_v(kv_len);
    let scale = 1.0 / f32::sqrt(TEST_HEAD_DIM as f32);

    let q = Tensor::new(&device, &q_data);
    let k = Tensor::new(&device, &k_data);
    let v = Tensor::new(&device, &v_data);
    let output = q.try_flash_attention_direct(&k, &v, scale, None).unwrap();
    let output = output.as_slice().await.unwrap();
    let (max_error, _, max_dim, max_actual, max_expected) =
        decode_max_error(1, 1, &q_data, &k_data, &v_data, scale, |_, dim| {
            output[[0, 0, 0, dim]]
        });
    assert!(
        max_error < 2.0e-4,
        "dim {max_dim}: actual={max_actual} expected={max_expected} error={max_error}"
    );
}

#[tokio::test]
async fn tiled_decode_attention_gqa_matches_cpu_reference() {
    let Ok(device) = Device::new().await else {
        return;
    };

    let num_heads = 32;
    let num_kv_heads = 8;
    let groups = num_heads / num_kv_heads;
    let kv_len = DECODE_LARGE_BLOCK as usize + 1;
    let q_data = decode_q_gqa(num_heads);
    let k_data = decode_k_gqa(num_kv_heads, kv_len);
    let v_data = decode_v_gqa(num_kv_heads, kv_len);
    let scale = 1.0 / f32::sqrt(TEST_HEAD_DIM as f32);

    let q = Tensor::new(&device, &q_data);
    let k = Tensor::new(&device, &k_data);
    let v = Tensor::new(&device, &v_data);
    let output = q.try_flash_attention_direct(&k, &v, scale, None).unwrap();
    let output = output.as_slice().await.unwrap();

    let (max_error, max_head, max_dim, max_actual, max_expected) = decode_max_error(
        num_heads,
        groups,
        &q_data,
        &k_data,
        &v_data,
        scale,
        |head, dim| output[[0, head, 0, dim]],
    );
    assert!(
        max_error < 3.0e-4,
        "head {max_head} dim {max_dim}: actual={max_actual} expected={max_expected} error={max_error}"
    );
}

/// Regression test for the non-tiled 512/1024-thread decode blocks.
/// Before the fix, the per-thread score loop folded its 128 q*k
/// accumulations into a single deeply-nested Naga expression, which
/// miscompiled on Metal once the kernel's `workgroup_size` exceeded 128;
/// the kernel produced all-zero output. The fix emits the dot product as a
/// shader loop with a function-scope accumulator.
#[tokio::test]
async fn decode_gqa_non_tiled_large_blocks_match_cpu_reference() {
    let Ok(device) = Device::new().await else {
        return;
    };

    let num_heads = 32;
    let num_kv_heads = 8;
    let groups = num_heads / num_kv_heads;
    let caps = KernelDeviceCaps::from_device(&device);

    // On devices that support the larger workgroups, 200 uses the 512
    // block and 600 uses the 1024 block.
    for (kv_len, expected_block) in [(200usize, DecodeBlock::Medium), (600, DecodeBlock::Large)] {
        if choose_decode_block(kv_len as u32, caps) != Some(expected_block) {
            continue;
        }
        let q_data = decode_q_gqa(num_heads);
        let k_data = decode_k_gqa(num_kv_heads, kv_len);
        let v_data = decode_v_gqa(num_kv_heads, kv_len);
        let scale = 1.0 / f32::sqrt(TEST_HEAD_DIM as f32);

        let q = Tensor::new(&device, &q_data);
        let k = Tensor::new(&device, &k_data);
        let v = Tensor::new(&device, &v_data);
        let output = q.try_flash_attention_direct(&k, &v, scale, None).unwrap();
        let output = output.as_slice().await.unwrap();

        let (max_error, max_head, max_dim, max_actual, max_expected) = decode_max_error(
            num_heads,
            groups,
            &q_data,
            &k_data,
            &v_data,
            scale,
            |head, dim| output[[0, head, 0, dim]],
        );
        assert!(
            max_error < 5.0e-4,
            "kv_len={kv_len} head={max_head} dim={max_dim}: actual={max_actual} expected={max_expected} error={max_error}"
        );
    }
}

#[tokio::test]
async fn streaming_gqa_regression_shape_builds_direct_kernel() {
    let Ok(device) = Device::new().await else {
        return;
    };

    let q_data = vec![
        (0..32)
            .map(|head| {
                (0..48)
                    .map(|token| {
                        (0..TEST_HEAD_DIM)
                            .map(|dim| {
                                let value = ((head * 17 + token * 11 + dim * 5) % 41) as f32 - 20.0;
                                value * 0.002
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
    ];
    let k_data = decode_k_gqa(8, 48);
    let v_data = decode_v_gqa(8, 48);
    let scale = 1.0 / f32::sqrt(TEST_HEAD_DIM as f32);

    let q = Tensor::new(&device, &q_data);
    let k = Tensor::new(&device, &k_data);
    let v = Tensor::new(&device, &v_data);
    let output = q.try_flash_attention_direct(&k, &v, scale, None).unwrap();
    output.materialize().await;
}
