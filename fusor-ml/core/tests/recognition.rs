//! Recognition fidelity gates: composed contraction clusters must resolve to
//! the same kernel counts as the specialized operations they replace. A
//! recognition miss falls back to the generic elementwise + reduce path,
//! which is correct but slow — these tests turn that silent regression into
//! a failure.

use fusor_core::{Device, QMatrix, Tensor};
use fusor_gguf::GgmlType;

fn f32_weight(device: &Device, n: usize, k: usize) -> QMatrix {
    let bytes: Vec<u8> = (0..n * k)
        .map(|i| 0.1 + (i as f32) * 0.05)
        .flat_map(f32::to_le_bytes)
        .collect();
    QMatrix::from_parts(device, &bytes, vec![n, k].into_boxed_slice(), GgmlType::F32).unwrap()
}

const Q8_BLOCK: usize = 32;

/// Patterned Q8_0 blocks: one f16 scale + 32 i8 weights per block, blocks
/// row-major over `[n, k]`.
fn q8_0_weight(device: &Device, n: usize, k: usize) -> (QMatrix, Vec<f32>) {
    let block_count = n * k / Q8_BLOCK;
    let scale = half::f16::from_f32(0.01);
    let mut bytes = Vec::new();
    let mut dequantized = Vec::with_capacity(n * k);
    for block in 0..block_count {
        bytes.extend_from_slice(&scale.to_le_bytes());
        for i in 0..Q8_BLOCK {
            let q = (((block * 5 + i * 3) % 64) as i32 - 32) as i8;
            bytes.push(q as u8);
            dequantized.push(scale.to_f32() * q as f32);
        }
    }
    let matrix = QMatrix::from_parts(
        device,
        &bytes,
        vec![n, k].into_boxed_slice(),
        GgmlType::Q8_0,
    )
    .unwrap();
    (matrix, dequantized)
}

#[test]
fn composed_dense_matmul_resolves_to_single_kernel() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };

        let a = Tensor::new::<f32, 2, _>(&device, &[[1.0f32, 2.0], [3.0, 4.0]]);
        let b = Tensor::new::<f32, 2, _>(&device, &[[5.0f32, 6.0], [7.0, 8.0]]);
        let out = a.mat_mul(&b);
        assert_eq!(out.count_kernels_to_resolve(), 1, "dense matmul");
        let slice = out.as_slice::<2, f32>().await.unwrap();
        assert_eq!(slice[[0, 0]], 19.0);
        assert_eq!(slice[[0, 1]], 22.0);
        assert_eq!(slice[[1, 0]], 43.0);
        assert_eq!(slice[[1, 1]], 50.0);

        let a = Tensor::new::<f32, 3, _>(
            &device,
            &[[[1.0f32, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        );
        let b = Tensor::new::<f32, 3, _>(
            &device,
            &[[[1.0f32, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]],
        );
        let out = a.mat_mul(&b);
        assert_eq!(out.count_kernels_to_resolve(), 1, "batched matmul");
        let slice = out.as_slice::<3, f32>().await.unwrap();
        assert_eq!(slice[[0, 0, 0]], 1.0);
        assert_eq!(slice[[1, 0, 0]], 10.0);
        assert_eq!(slice[[1, 1, 1]], 16.0);
    });
}

#[test]
fn composed_qmatmul_resolves_to_single_kernel() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };

        const K: usize = 8;
        let w = f32_weight(&device, 4, K);
        let x = Tensor::new::<f32, 2, _>(&device, &[[1.0f32, 0.5, -1.0, 2.0, 0.0, 1.0, -0.5, 3.0]]);
        let out = x.q_mat_mul(&w);
        assert_eq!(out.count_kernels_to_resolve(), 1, "bare quantized matmul");
    });
}

#[test]
fn composed_q8_qmatmul_with_epilogue_resolves_to_single_kernel() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };

        const K: usize = 32;
        const N: usize = 4;
        let (w, dequantized) = q8_0_weight(&device, N, K);
        let x_values: Vec<f32> = (0..K).map(|i| ((i as f32) * 0.37).sin()).collect();
        let bias_values = [0.5f32, -1.0, 2.0, -0.25];

        let x = Tensor::from_slice(&device, [1, K], &x_values);
        let bias = Tensor::from_slice(&device, [1, N], &bias_values);
        let out = x.q_mat_mul(&w) + &bias;
        assert_eq!(
            out.count_kernels_to_resolve(),
            1,
            "q8_0 matmul + bias epilogue"
        );

        let slice = out.as_slice::<2, f32>().await.unwrap();
        for col in 0..N {
            let expected: f32 = (0..K)
                .map(|k| x_values[k] * dequantized[col * K + k])
                .sum::<f32>()
                + bias_values[col];
            assert!(
                (slice[[0, col]] - expected).abs() < 1e-3,
                "column {col}: got {}, expected {expected}",
                slice[[0, col]]
            );
        }
    });
}

#[test]
fn f32_weight_epilogue_takes_correct_fallback() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };

        // F32-native weights lower through the dense matmul kernel, which has
        // no epilogue slots: the epilogue must run as its own kernel rather
        // than being dropped (regression: the dense shortcut used to discard
        // fused epilogues).
        const K: usize = 8;
        const N: usize = 4;
        let w = f32_weight(&device, N, K);
        let x_values = [2.0f32, 1.5, -1.0, 2.0, 1.0, 1.0, -0.5, 3.0];
        let bias_values = [0.5f32, -1.0, 2.0, -0.25];
        let x = Tensor::from_slice(&device, [1, K], &x_values);
        let bias = Tensor::from_slice(&device, [1, N], &bias_values);
        let out = x.q_mat_mul(&w) + &bias;

        let slice = out.as_slice::<2, f32>().await.unwrap();
        for col in 0..N {
            let expected: f32 = (0..K)
                .map(|k| x_values[k] * (0.1 + ((col * K + k) as f32) * 0.05))
                .sum::<f32>()
                + bias_values[col];
            assert!(
                (slice[[0, col]] - expected).abs() < 1e-3,
                "column {col}: got {}, expected {expected}",
                slice[[0, col]]
            );
        }
    });
}

#[test]
fn composed_softmax_resolves_to_fused_kernel() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };
        let values: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.1).sin()).collect();
        let x = Tensor::from_slice(&device, [2, 128], &values);
        let out = x.softmax(1);
        assert_eq!(out.count_kernels_to_resolve(), 1, "single-pass softmax");

        let slice = out.as_slice::<2, f32>().await.unwrap();
        for row in 0..2 {
            let max = (0..128)
                .map(|col| values[row * 128 + col])
                .fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = (0..128)
                .map(|col| (values[row * 128 + col] - max).exp())
                .sum();
            for col in [0, 63, 127] {
                let expected = (values[row * 128 + col] - max).exp() / sum;
                assert!(
                    (slice[[row, col]] - expected).abs() < 1e-5,
                    "row {row} col {col}: got {}, expected {expected}",
                    slice[[row, col]]
                );
            }
        }
    });
}

#[test]
fn composed_rms_norm_with_bias_resolves_to_fused_kernel() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };
        let x_values = [1.0f32, 2.0, 3.0, 4.0];
        let x = Tensor::from_slice(&device, [1, 4], &x_values);
        let weight = Tensor::from_slice(&device, [4], &[0.5f32, 1.0, 1.5, 2.0]);
        let bias = Tensor::from_slice(&device, [4], &[0.1f32, -0.2, 0.3, -0.4]);
        let out = x.rms_norm_fused(&weight, Some(&bias), 1e-5);
        assert_eq!(out.count_kernels_to_resolve(), 1, "rms norm with bias");

        let slice = out.as_slice::<2, f32>().await.unwrap();
        let mean_square = (1.0 + 4.0 + 9.0 + 16.0) / 4.0;
        let rms = f32::sqrt(mean_square + 1e-5);
        let weights = [0.5f32, 1.0, 1.5, 2.0];
        let biases = [0.1f32, -0.2, 0.3, -0.4];
        for col in 0..4 {
            let expected = x_values[col] / rms * weights[col] + biases[col];
            assert!(
                (slice[[0, col]] - expected).abs() < 1e-5,
                "col {col}: got {}, expected {expected}",
                slice[[0, col]]
            );
        }
    });
}

#[test]
fn composed_attention_resolves_to_flash_kernel() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };

        // Decode shape: q [1, heads, 1, d] against a kv history, GQA 4:1.
        let (heads, kv_heads, kv_len, d) = (8usize, 2usize, 64usize, 16usize);
        let q_data: Vec<f32> = (0..heads * d).map(|i| ((i as f32) * 0.13).sin()).collect();
        let k_data: Vec<f32> = (0..kv_heads * kv_len * d)
            .map(|i| ((i as f32) * 0.07).cos())
            .collect();
        let v_data: Vec<f32> = (0..kv_heads * kv_len * d)
            .map(|i| ((i as f32) * 0.11).sin())
            .collect();
        let q = Tensor::from_slice(&device, [1, heads, 1, d], &q_data);
        let k = Tensor::from_slice(&device, [1, kv_heads, kv_len, d], &k_data);
        let v = Tensor::from_slice(&device, [1, kv_heads, kv_len, d], &v_data);
        let scale = 1.0 / (d as f32).sqrt();

        let out = q.flash_attention(&k, &v, scale, None);
        assert_eq!(
            out.count_kernels_to_resolve(),
            1,
            "gqa decode attention should recognize as one fused flash kernel"
        );
        let slice = out.as_slice::<4, f32>().await.unwrap();

        // Host reference.
        for head in [0usize, 5] {
            let kv_head = head / (heads / kv_heads);
            let scores: Vec<f32> = (0..kv_len)
                .map(|pos| {
                    (0..d)
                        .map(|i| q_data[head * d + i] * k_data[kv_head * kv_len * d + pos * d + i])
                        .sum::<f32>()
                        * scale
                })
                .collect();
            let max = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let weights: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let total: f32 = weights.iter().sum();
            for dim in [0usize, d - 1] {
                let expected: f32 = (0..kv_len)
                    .map(|pos| weights[pos] / total * v_data[kv_head * kv_len * d + pos * d + dim])
                    .sum();
                let actual = slice[[0, head, 0, dim]];
                assert!(
                    (actual - expected).abs() < 1e-4,
                    "head {head} dim {dim}: got {actual}, expected {expected}"
                );
            }
        }
    });
}
