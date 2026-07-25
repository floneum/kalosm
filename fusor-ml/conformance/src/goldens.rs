//! Machine-pinned output-hash goldens.
//!
//! Each test replays one deterministic GPU trace and compares an FNV-1a hash
//! of the exact output bytes against a golden captured on the baseline
//! machine (`goldens/`). The traces are bit-reproducible for a fixed
//! device/driver, so any hash drift means a numeric change: refactors that
//! claim "no numeric change" must keep these green, and intentional numeric
//! changes must re-capture the goldens from the failure output.

use fusor::autograd::Graph;
use fusor::{Device, GgmlType, MaskKind, Tensor};

use crate::common::quantized::{q4k_raw_bytes, q6k_raw_bytes, qmatrix_from_raw_bytes};

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn fill(seed: u32, len: usize) -> Vec<f32> {
    let mut state = seed as u64 | 1;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

async fn tensor_hash<const R: usize>(tensor: &Tensor<R, f32>) -> u64 {
    let len = tensor.shape().iter().product();
    let flat: Tensor<1, f32> = tensor.reshape([len]).to_concrete();
    let values = flat.as_slice().await.unwrap().as_slice().to_vec();
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in &values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    fnv1a(&bytes)
}

fn assert_matches_golden(name: &str, golden: &str, actual: &str) {
    assert!(
        golden.trim() == actual.trim(),
        "{name} golden mismatch; measured values:\n{actual}"
    );
}

/// One grouped-query attention config with a causal additive mask, forward
/// and backward: 8 query heads over 2 key/value heads takes the composite
/// replay backward, so the hashes pin both the fused forward kernel and the
/// recomputed probability chain.
#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn attention_gqa_causal_fwd_bwd_matches_golden() {
    let _gpu_guard = crate::suite::registry::gpu_test_guard();
    let Ok(device) = Device::gpu().await else {
        return;
    };
    const BATCH: usize = 2;
    const Q_HEADS: usize = 8;
    const KV_HEADS: usize = 2;
    const SEQ: usize = 32;
    const HEAD_DIM: usize = 64;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();

    let mask_values: Vec<f32> = (0..SEQ * SEQ)
        .map(|i| if i % SEQ <= i / SEQ { 0.0 } else { -1e9 })
        .collect();
    let mask: Tensor<2, f32> = Tensor::from_slice(&device, [SEQ, SEQ], &mask_values);

    let graph = Graph::new();
    let q = graph.leaf(Tensor::from_slice(
        &device,
        [BATCH, Q_HEADS, SEQ, HEAD_DIM],
        &fill(11, BATCH * Q_HEADS * SEQ * HEAD_DIM),
    ));
    let k = graph.leaf(Tensor::from_slice(
        &device,
        [BATCH, KV_HEADS, SEQ, HEAD_DIM],
        &fill(23, BATCH * KV_HEADS * SEQ * HEAD_DIM),
    ));
    let v = graph.leaf(Tensor::from_slice(
        &device,
        [BATCH, KV_HEADS, SEQ, HEAD_DIM],
        &fill(37, BATCH * KV_HEADS * SEQ * HEAD_DIM),
    ));

    let out = q.attention(&k, &v, scale, Some((&mask, MaskKind::Causal)));
    let gradients = out
        .reshape([BATCH * Q_HEADS * SEQ * HEAD_DIM])
        .sum()
        .backward()
        .unwrap();
    let dq = gradients.get(&q).expect("missing q gradient");
    let dk = gradients.get(&k).expect("missing k gradient");
    let dv = gradients.get(&v).expect("missing v gradient");

    let actual = [
        format!("out {:#018x}", tensor_hash(out.raw()).await),
        format!("dq {:#018x}", tensor_hash(&dq).await),
        format!("dk {:#018x}", tensor_hash(&dk).await),
        format!("dv {:#018x}", tensor_hash(&dv).await),
    ]
    .join("\n");
    assert_matches_golden(
        "attention_gqa_causal",
        include_str!("../goldens/attention_gqa_causal.txt"),
        &actual,
    );
}

/// A bidirectional recurrent trace in the bilstm style: per-timestep narrows
/// off one input, a matmul + recurrent matmul + tanh per step, and the step
/// outputs reassembled with `cat` (per direction, then across directions).
#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn bilstm_split_op_cat_trace_matches_golden() {
    let _gpu_guard = crate::suite::registry::gpu_test_guard();
    let Ok(device) = Device::gpu().await else {
        return;
    };
    const STEPS: usize = 8;
    const BATCH: usize = 4;
    const FEATURES: usize = 32;
    const HIDDEN: usize = 16;

    let x: Tensor<3, f32> = Tensor::from_slice(
        &device,
        [STEPS, BATCH, FEATURES],
        &fill(5, STEPS * BATCH * FEATURES),
    );
    let direction = |input_seed: u32, recurrent_seed: u32, reverse: bool| {
        let input_weight: Tensor<2, f32> = Tensor::from_slice(
            &device,
            [FEATURES, HIDDEN],
            &fill(input_seed, FEATURES * HIDDEN),
        );
        let recurrent_weight: Tensor<2, f32> = Tensor::from_slice(
            &device,
            [HIDDEN, HIDDEN],
            &fill(recurrent_seed, HIDDEN * HIDDEN),
        );
        let mut hidden: Tensor<2, f32> = Tensor::zeros(&device, [BATCH, HIDDEN]);
        let mut outputs = vec![Tensor::zeros(&device, [1, BATCH, HIDDEN]); STEPS];
        for step in 0..STEPS {
            let step = if reverse { STEPS - 1 - step } else { step };
            let input = x
                .narrow(0, step, 1)
                .reshape([BATCH, FEATURES])
                .to_concrete();
            hidden = (input.mat_mul(&input_weight) + hidden.mat_mul(&recurrent_weight))
                .tanh()
                .to_concrete();
            outputs[step] = hidden.reshape([1, BATCH, HIDDEN]).to_concrete();
        }
        Tensor::cat(outputs, 0)
    };
    let forward = direction(41, 43, false);
    let backward = direction(47, 53, true);
    let out = Tensor::cat(vec![forward, backward], 2);

    let actual = format!("out {:#018x}", tensor_hash(&out).await);
    assert_matches_golden(
        "bilstm_trace",
        include_str!("../goldens/bilstm_trace.txt"),
        &actual,
    );
}

/// Every Q4K/Q6K decode (M=1 qgemv) shape exercised by the ggml qgemv
/// lowering suite (`tile-ir-kernels/tests/lowering.rs`), run end-to-end with
/// deterministic weights: the main 4096x8192 shape, the tail-column and mid
/// variants, and the Q6K shape.
#[allow(clippy::await_holding_lock)]
#[tokio::test]
async fn qgemv_decode_ggml_shapes_match_golden() {
    let _gpu_guard = crate::suite::registry::gpu_test_guard();
    let Ok(device) = Device::gpu().await else {
        return;
    };
    let cases: [(GgmlType, &str, usize, usize); 4] = [
        (GgmlType::Q4K, "q4k", 4096, 8192),
        (GgmlType::Q4K, "q4k", 4096, 8193),
        (GgmlType::Q4K, "q4k", 4096, 5120),
        (GgmlType::Q6K, "q6k", 4096, 8192),
    ];
    let mut actual = Vec::new();
    for (ty, label, rows, cols) in cases {
        let weight_shape = [cols, rows];
        let raw_bytes = match ty {
            GgmlType::Q4K => q4k_raw_bytes(weight_shape),
            GgmlType::Q6K => q6k_raw_bytes(weight_shape),
            _ => unreachable!(),
        };
        let weights = qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, ty);
        let input: Tensor<2, f32> = Tensor::from_slice(&device, [1, rows], &fill(3, rows));
        let out = input.q_mat_mul(&weights).to_concrete();
        actual.push(format!(
            "{label}_{rows}x{cols} {:#018x}",
            tensor_hash(&out).await
        ));
    }
    assert_matches_golden(
        "qgemv_decode_ggml",
        include_str!("../goldens/qgemv_decode_ggml.txt"),
        &actual.join("\n"),
    );
}
