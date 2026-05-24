mod common;

use common::quantized::{
    concrete_to_rows, q_mat_mul_input_fuzz, q4k_raw_bytes, qmatrix_from_raw_bytes,
};
use common::{matmul2, transpose2};
use fusor::{BlockQ4K, Device, GgmlType, GgufBlock, QuantizedTensor, Tensor, ToVec2};
use fusor_conformance::approx_compare;
use rand::distr::Uniform;
use std::mem::size_of;

#[tokio::test]
async fn q4k_paired_natural_form_matches_cpu_reference() {
    for kind in [PairedKind::SwiGLU, PairedKind::GeGLU, PairedKind::ReGLU] {
        for rows in [1, 4] {
            paired_matches_cpu_for_rows(rows, kind).await;
        }
    }
}

#[tokio::test]
async fn q4k_paired_llama_shape_one_hot_matches_cpu_reference() {
    use fusor::D;

    let Ok(device) = Device::new().await else {
        return;
    };
    let weight_shape = [14336usize, 4096usize];
    let pair_len = weight_shape[0] / 2;
    let input_rows = 48usize;
    let selected_k = 777usize;
    let blocks_per_row = weight_shape[1] / BlockQ4K::BLOCK_SIZE;
    let selected_block_in_row = selected_k / BlockQ4K::BLOCK_SIZE;
    let selected_offset = selected_k % BlockQ4K::BLOCK_SIZE;
    let raw_bytes = q4k_raw_bytes(weight_shape);
    let weights = qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, GgmlType::Q4K);
    let selected_weight = |row: usize| {
        let block_index = row * blocks_per_row + selected_block_in_row;
        let offset = block_index * size_of::<BlockQ4K>();
        assert!(offset + size_of::<BlockQ4K>() <= raw_bytes.len());
        let block =
            unsafe { std::ptr::read_unaligned(raw_bytes.as_ptr().add(offset).cast::<BlockQ4K>()) };
        block.dequantize().as_ref()[selected_offset]
    };
    let mut input_data = vec![0.0f32; input_rows * weight_shape[1]];
    for row in 0..input_rows {
        input_data[row * weight_shape[1] + selected_k] = 0.125 + row as f32 * 0.01;
    }
    let input: Tensor<2, f32> =
        Tensor::from_slice(&device, [input_rows, weight_shape[1]], &input_data);

    let projected = input.q_mat_mul(&weights);
    let gate = projected.narrow(D::Minus1, 0, pair_len).to_concrete();
    let up = projected
        .narrow(D::Minus1, pair_len, pair_len)
        .to_concrete();
    let actual = (gate.silu() * up).to_concrete().as_slice().await.unwrap();

    assert_eq!(actual.shape(), &[input_rows, pair_len]);
    for row in 0..input_rows {
        let input_value = input_data[row * weight_shape[1] + selected_k];
        for col in [0usize, 1, 63, 64, 511, 1024, 4095, pair_len - 1] {
            let gate = input_value * selected_weight(col);
            let up = input_value * selected_weight(col + pair_len);
            let expected = (gate / (1.0 + (-gate).exp())) * up;
            let actual = actual[[row, col]];
            let tolerance = 2.0f32.max(expected.abs() * 1.0e-4);
            assert!(
                (actual - expected).abs() <= tolerance,
                "row={row} col={col} actual={actual} expected={expected} tolerance={tolerance}"
            );
        }
    }
}

#[tokio::test]
async fn q4k_paired_llama_shape_dense_sampled_columns_match_cpu_reference() {
    use fusor::D;

    let Ok(device) = Device::new().await else {
        return;
    };
    let weight_shape = [14336usize, 4096usize];
    let pair_len = weight_shape[0] / 2;
    let input_rows = 48usize;
    let blocks_per_row = weight_shape[1] / BlockQ4K::BLOCK_SIZE;
    let raw_bytes = q4k_raw_bytes(weight_shape);
    let weights = qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, GgmlType::Q4K);
    let block_at = |matrix_row: usize, block_col: usize| {
        let block_index = matrix_row * blocks_per_row + block_col;
        let offset = block_index * size_of::<BlockQ4K>();
        assert!(offset + size_of::<BlockQ4K>() <= raw_bytes.len());
        unsafe { std::ptr::read_unaligned(raw_bytes.as_ptr().add(offset).cast::<BlockQ4K>()) }
    };
    let mut input_data = vec![0.0f32; input_rows * weight_shape[1]];
    for (index, value) in input_data.iter_mut().enumerate() {
        let bucket = (index.wrapping_mul(37).wrapping_add(11)) % 101;
        *value = (bucket as f32 - 50.0) * 0.0025;
    }
    let input: Tensor<2, f32> =
        Tensor::from_slice(&device, [input_rows, weight_shape[1]], &input_data);

    let projected = input.q_mat_mul(&weights);
    let gate = projected.narrow(D::Minus1, 0, pair_len).to_concrete();
    let up = projected
        .narrow(D::Minus1, pair_len, pair_len)
        .to_concrete();
    let actual = (gate.silu() * up).to_concrete().as_slice().await.unwrap();

    assert_eq!(actual.shape(), &[input_rows, pair_len]);
    for row in [0usize, 1, 7, 17, 31, 47] {
        let input_row = &input_data[row * weight_shape[1]..(row + 1) * weight_shape[1]];
        for col in [0usize, 1, 63, 64, 511, 1024, 4095, pair_len - 1] {
            let dot = |matrix_row: usize| {
                (0..blocks_per_row)
                    .map(|block_col| {
                        let block = block_at(matrix_row, block_col);
                        let weights = block.dequantize();
                        weights
                            .as_ref()
                            .iter()
                            .enumerate()
                            .map(|(offset, weight)| {
                                input_row[block_col * BlockQ4K::BLOCK_SIZE + offset] * *weight
                            })
                            .sum::<f32>()
                    })
                    .sum::<f32>()
            };
            let gate = dot(col);
            let up = dot(col + pair_len);
            let expected = (gate / (1.0 + (-gate).exp())) * up;
            let actual = actual[[row, col]];
            let tolerance = 2.0f32.max(expected.abs() * 1.0e-4);
            assert!(
                (actual - expected).abs() <= tolerance,
                "row={row} col={col} actual={actual} expected={expected} tolerance={tolerance}"
            );
        }
    }
}

#[derive(Clone, Copy)]
enum PairedKind {
    SwiGLU,
    GeGLU,
    ReGLU,
}

impl PairedKind {
    fn cpu_activation(self, x: f32) -> f32 {
        match self {
            PairedKind::SwiGLU => x / (1.0 + (-x).exp()),
            PairedKind::GeGLU => {
                // tanh approximation matching the kernel-side helper
                0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh())
            }
            PairedKind::ReGLU => {
                if x > 0.0 {
                    x
                } else {
                    0.0
                }
            }
        }
    }
}

async fn paired_matches_cpu_for_rows(input_row_count: usize, kind: PairedKind) {
    use fusor::D;
    let ty = GgmlType::Q4K;
    let weight_shape = [4, 512];
    let raw_bytes = q4k_raw_bytes(weight_shape);
    let weights = QuantizedTensor::<BlockQ4K>::from_raw_bytes(weight_shape, &raw_bytes);
    let expected_weights = concrete_to_rows(&weights.dequantize::<2>(), weight_shape);

    // Author the natural unfused source — the resolver's paired-fusion rule
    // rewrites this into a single paired-mode QMatMul kernel. Correctness verifies
    // both the rewrite and the kernel's per-output epilogue evaluation.
    fusor_conformance::assert(move |input: Tensor<2, f32>| {
        let raw_bytes = raw_bytes.clone();
        async move {
            let weights = qmatrix_from_raw_bytes(&input.device(), weight_shape, &raw_bytes, ty);
            let projected = input.q_mat_mul(&weights);
            let gate = projected.narrow(D::Minus1, 0, 2).to_concrete();
            let up = projected.narrow(D::Minus1, 2, 2).to_concrete();
            let activated = match kind {
                PairedKind::SwiGLU => gate.silu(),
                PairedKind::GeGLU => gate.gelu(),
                PairedKind::ReGLU => gate.relu(),
            };
            (activated * up).to_concrete()
        }
    })
    .arg(q_mat_mul_input_fuzz(
        input_row_count,
        [2, weight_shape[1]],
        0x5A17_5516_6C75,
        Uniform::new(-0.25, 0.25).unwrap(),
    ))
    .equal_to(move |input: Tensor<2, f32>| {
        let expected_weights = expected_weights.clone();
        async move {
            let device = input.device();
            let input_values = input.as_slice().await.unwrap().to_vec2();
            let projected = matmul2(&input_values, &transpose2(&expected_weights));
            let expected = projected
                .iter()
                .map(|row| {
                    let gate0 = row[0];
                    let gate1 = row[1];
                    vec![
                        kind.cpu_activation(gate0) * row[2],
                        kind.cpu_activation(gate1) * row[3],
                    ]
                })
                .collect::<Vec<_>>();
            Tensor::new(&device, &expected)
        }
    })
    .compare_with(approx_compare::<2, f32>(2.0))
    .runs(3)
    .await
    .unwrap();
}

/// The fuser must collapse the natural `q_mat_mul → narrow → silu → mul(narrow)`
/// pattern into a single dispatch. Without the rule the source would emit
/// multiple kernels (matmul + nary).
#[tokio::test]
async fn q4k_paired_pattern_resolves_to_single_kernel() {
    use fusor::D;
    let Ok(device) = fusor::Device::new().await else {
        return; // No GPU available in this environment.
    };
    let weight_shape = [4, 512];
    let raw_bytes = q4k_raw_bytes(weight_shape);
    let input_data = vec![vec![0.1f32; weight_shape[1]]; 1];
    let Some(gpu_device) = device.as_gpu() else {
        panic!("expected GPU device");
    };
    if !gpu_device.subgroups_supported() {
        return;
    }

    let weights = qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, GgmlType::Q4K);
    let input: Tensor<2, f32> = Tensor::new(&device, &input_data);
    let projected = input.q_mat_mul(&weights);
    let gate = projected.narrow(D::Minus1, 0, 2).to_concrete();
    let up = projected.narrow(D::Minus1, 2, 2).to_concrete();
    let output = (gate.silu() * up).to_concrete();
    let Tensor::Gpu(natural_gpu) = output else {
        panic!("expected GPU tensor");
    };
    let natural_kernels = gpu_device.resolve_batch(&[natural_gpu.key()]);

    assert_eq!(
        natural_kernels, 1,
        "expected fuser to collapse paired pattern to 1 dispatch, got {natural_kernels}"
    );
}

/// Authoring the natural source (`q_mat_mul → narrow → silu → mul(narrow)`)
/// should produce results identical to the CPU reference, because the
/// compute-graph fuser rewrites the pattern to a `paired-mode QMatMul` kernel.
#[tokio::test]
async fn q4k_paired_pattern_auto_fuses_to_paired_kernel() {
    use fusor::D;
    let ty = GgmlType::Q4K;
    let weight_shape = [4, 512];
    let raw_bytes = q4k_raw_bytes(weight_shape);
    let weights = QuantizedTensor::<BlockQ4K>::from_raw_bytes(weight_shape, &raw_bytes);
    let expected_weights = concrete_to_rows(&weights.dequantize::<2>(), weight_shape);

    fusor_conformance::assert(move |input: Tensor<2, f32>| {
        let raw_bytes = raw_bytes.clone();
        async move {
            let weights = qmatrix_from_raw_bytes(&input.device(), weight_shape, &raw_bytes, ty);
            // Natural unfused authoring — the resolver's `try_fuse_paired_qmatmul`
            // rule rewrites this subgraph to a single paired-fused kernel.
            let projected = input.q_mat_mul(&weights);
            let gate = projected.narrow(D::Minus1, 0, 2).to_concrete();
            let up = projected.narrow(D::Minus1, 2, 2).to_concrete();
            (gate.silu() * up).to_concrete()
        }
    })
    .arg(q_mat_mul_input_fuzz(
        1,
        [2, weight_shape[1]],
        0x5A17_5516_6C76,
        Uniform::new(-0.25, 0.25).unwrap(),
    ))
    .equal_to(move |input: Tensor<2, f32>| {
        let expected_weights = expected_weights.clone();
        async move {
            let device = input.device();
            let input_values = input.as_slice().await.unwrap().to_vec2();
            let projected = matmul2(&input_values, &transpose2(&expected_weights));
            let expected = projected
                .iter()
                .map(|row| {
                    let gate0 = row[0];
                    let gate1 = row[1];
                    let silu = |x: f32| x / (1.0 + (-x).exp());
                    vec![silu(gate0) * row[2], silu(gate1) * row[3]]
                })
                .collect::<Vec<_>>();
            Tensor::new(&device, &expected)
        }
    })
    .compare_with(approx_compare::<2, f32>(2.0))
    .runs(3)
    .await
    .unwrap();
}

/// Biased FFN pattern: `silu(gate + gate_bias) * (up + up_bias)`.
/// The fuser detects this as a paired-with-extras pattern (2 matmul views +
/// 2 broadcast bias vectors) and rewrites it to a single `paired-mode QMatMul`
/// kernel that loads the biases per output column at epilogue time.
#[tokio::test]
async fn q4k_paired_with_bias_resolves_to_single_kernel() {
    use fusor::D;
    let Ok(device) = fusor::Device::new().await else {
        return;
    };
    let Some(gpu_device) = device.as_gpu() else {
        panic!("expected GPU device");
    };
    if !gpu_device.subgroups_supported() {
        return;
    }
    let weight_shape = [4, 512];
    let raw_bytes = q4k_raw_bytes(weight_shape);
    let input_data = vec![vec![0.1f32; weight_shape[1]]; 1];
    let weights = qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, GgmlType::Q4K);
    let input: Tensor<2, f32> = Tensor::new(&device, &input_data);
    let gate_bias: Tensor<1, f32> = Tensor::new(&device, &vec![0.05f32, -0.05]);
    let up_bias: Tensor<1, f32> = Tensor::new(&device, &vec![0.02f32, 0.03]);

    let projected = input.q_mat_mul(&weights);
    let gate = projected.narrow(D::Minus1, 0, 2).to_concrete();
    let up = projected.narrow(D::Minus1, 2, 2).to_concrete();
    let gate_biased = gate.add_(&gate_bias);
    let up_biased = up.add_(&up_bias);
    let output = (gate_biased.silu() * up_biased).to_concrete();

    let Tensor::Gpu(gpu_out) = output else {
        panic!("expected GPU tensor");
    };
    let kernels = gpu_device.resolve_batch(&[gpu_out.key()]);
    assert_eq!(
        kernels, 1,
        "expected fuser to collapse biased silu(gate+gb)*up_biased to 1 dispatch, got {kernels}"
    );
}
