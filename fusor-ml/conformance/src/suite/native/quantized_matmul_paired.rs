//! Paired quantized matmul conformance cases.

use crate::common::quantized::{
    concrete_to_rows, q_mat_mul_input_fuzz, q4k_raw_bytes, qmatrix_from_raw_bytes,
};
use crate::common::{matmul2, transpose2};
use fusor::{BlockQ4K, Device, GgmlType, GgufBlock, QuantizedTensor, Tensor, ToVec};
use fusor_conformance::{
    AssertionCase, AssertionCases, approx_compare, approx_or_relative_compare, available_devices,
    cases_from_rows,
};
use rand::distr::Uniform;
use std::mem::size_of;

async fn gpu_devices() -> Vec<Device> {
    available_devices()
        .await
        .into_iter()
        .filter(|device| device.as_gpu().is_some())
        .collect()
}

pub fn q4k_concat_split_gated_natural_form_matches_cpu_reference() -> AssertionCases {
    let mut assertions = AssertionCases::new();
    for kind in [GatedKind::SwiGLU, GatedKind::GeGLU, GatedKind::ReGLU] {
        for rows in [1, 4] {
            assertions.push(gated_matches_cpu_for_rows(rows, kind));
        }
    }
    assertions
}

pub fn q4k_dynamic_paired_helper_swiglu_matches_cpu_reference_for_decode_row() -> AssertionCase {
    let weight_shape = [1024, 1024];
    let pair_len = weight_shape[0] / 2;
    let input_shape = [1usize, 2usize, weight_shape[1]];
    let raw_bytes = q4k_raw_bytes(weight_shape);
    let expected_weights = concrete_to_rows(
        &QuantizedTensor::<BlockQ4K>::from_raw_bytes(weight_shape, &raw_bytes).dequantize::<2>(),
        weight_shape,
    );
    let input_data = (0..input_shape.iter().product::<usize>())
        .map(|i| {
            let bucket = (i.wrapping_mul(37).wrapping_add(11)) % 101;
            (bucket as f32 - 50.0) * 0.0018
        })
        .collect::<Vec<_>>();
    let input_rows = input_data
        .chunks(weight_shape[1])
        .map(|row| row.to_vec())
        .collect::<Vec<_>>();
    let projected = matmul2(&input_rows, &transpose2(&expected_weights));
    let expected = vec![
        projected
            .iter()
            .map(|row| {
                (0..pair_len)
                    .map(|col| {
                        let gate = row[col];
                        let up = row[col + pair_len];
                        (gate / (1.0 + (-gate).exp())) * up
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
    ];

    fusor_conformance::assert(move |device: Device| {
        let raw_bytes = raw_bytes.clone();
        let input_data = input_data.clone();
        async move {
            use fusor::D;
            let weights = qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, GgmlType::Q4K);
            let input: Tensor<3, f32> = Tensor::from_slice(&device, input_shape, &input_data);
            let projected = input.q_mat_mul(&weights);
            let gate = projected.narrow(D::Minus1, 0, pair_len).to_concrete();
            let up = projected
                .narrow(D::Minus1, pair_len, pair_len)
                .to_concrete();
            (gate.silu() * up).to_concrete()
        }
    })
    .arg(|device: &Device| device.clone())
    .equal_to(move |device: Device| {
        let expected = expected.clone();
        async move { Tensor::new(&device, &expected) }
    })
    .compare_with(approx_compare::<3, f32>(5.0))
    .baseline_on_test_device()
    .devices_async(gpu_devices())
    .runs(1)
    .into_case(
        "quantized_matmul_paired::q4k_dynamic_paired_helper_swiglu_matches_cpu_reference_for_decode_row",
    )
}

/// Both Llama-shaped (`[14336, 4096]`, 48 input rows) paired q4k matmul cases:
/// the one-hot input (`selected_k = 777`) form and the dense-sampled-columns
/// form. Each remains an independently named GPU-only sub-case so the distinct
/// behaviors (sparse single-column dispatch vs. full-width accumulation) are
/// preserved.
pub fn q4k_concat_split_llama_shape_match_cpu_reference() -> AssertionCases {
    cases_from_rows([
        llama_shape_one_hot_case(),
        llama_shape_dense_sampled_columns_case(),
    ])
}

fn llama_shape_one_hot_case() -> AssertionCase {
    use fusor::D;
    let weight_shape = [14336usize, 4096usize];
    let pair_len = weight_shape[0] / 2;
    let input_rows = 48usize;
    let selected_k = 777usize;
    let blocks_per_row = weight_shape[1] / BlockQ4K::BLOCK_SIZE;
    let selected_block_in_row = selected_k / BlockQ4K::BLOCK_SIZE;
    let selected_offset = selected_k % BlockQ4K::BLOCK_SIZE;
    let raw_bytes = q4k_raw_bytes(weight_shape);
    let mut input_data = vec![0.0f32; input_rows * weight_shape[1]];
    for row in 0..input_rows {
        input_data[row * weight_shape[1] + selected_k] = 0.125 + row as f32 * 0.01;
    }
    let sample_cols = vec![0usize, 1, 63, 64, 511, 1024, 4095, pair_len - 1];
    let sample_count = input_rows * sample_cols.len();
    let expected_input_data = input_data.clone();
    let expected_raw_bytes = raw_bytes.clone();
    let expected_sample_cols = sample_cols.clone();

    fusor_conformance::assert(move |device: Device| {
        let raw_bytes = raw_bytes.clone();
        let input_data = input_data.clone();
        let sample_cols = sample_cols.clone();
        async move {
            let weights = qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, GgmlType::Q4K);
            let input: Tensor<2, f32> =
                Tensor::from_slice(&device, [input_rows, weight_shape[1]], &input_data);
            let projected = input.q_mat_mul(&weights);
            let gate = projected.narrow(D::Minus1, 0, pair_len).to_concrete();
            let up = projected
                .narrow(D::Minus1, pair_len, pair_len)
                .to_concrete();
            let actual = (gate.silu() * up).to_concrete().as_slice().await.unwrap();
            let mut samples = Vec::with_capacity(sample_count);
            for row in 0..input_rows {
                for &col in &sample_cols {
                    samples.push(actual[[row, col]]);
                }
            }
            Tensor::from_slice(&device, [sample_count], &samples)
        }
    })
    .arg(|device: &Device| device.clone())
    .equal_to(move |device: Device| {
        let input_data = expected_input_data.clone();
        let raw_bytes = expected_raw_bytes.clone();
        let sample_cols = expected_sample_cols.clone();
        async move {
            let selected_weight = |row: usize| {
                let block_index = row * blocks_per_row + selected_block_in_row;
                let offset = block_index * size_of::<BlockQ4K>();
                assert!(offset + size_of::<BlockQ4K>() <= raw_bytes.len());
                let block = unsafe {
                    std::ptr::read_unaligned(raw_bytes.as_ptr().add(offset).cast::<BlockQ4K>())
                };
                block.dequantize().as_ref()[selected_offset]
            };
            let expected = (0..input_rows)
                .flat_map(|row| {
                    let input_value = input_data[row * weight_shape[1] + selected_k];
                    sample_cols.iter().map(move |&col| {
                        let gate = input_value * selected_weight(col);
                        let up = input_value * selected_weight(col + pair_len);
                        (gate / (1.0 + (-gate).exp())) * up
                    })
                })
                .collect::<Vec<_>>();
            Tensor::from_slice(&device, [sample_count], &expected)
        }
    })
    .compare_with(approx_or_relative_compare::<1>(2.0, 1e-4))
    .baseline_on_test_device()
    .devices_async(gpu_devices())
    .runs(1)
    .into_case("quantized_matmul_paired::q4k_concat_split_llama_shape_match_cpu_reference::one_hot")
}

fn llama_shape_dense_sampled_columns_case() -> AssertionCase {
    use fusor::D;

    let weight_shape = [14336usize, 4096usize];
    let pair_len = weight_shape[0] / 2;
    let input_rows = 48usize;
    let blocks_per_row = weight_shape[1] / BlockQ4K::BLOCK_SIZE;
    let raw_bytes = q4k_raw_bytes(weight_shape);
    let mut input_data = vec![0.0f32; input_rows * weight_shape[1]];
    for (index, value) in input_data.iter_mut().enumerate() {
        let bucket = (index.wrapping_mul(37).wrapping_add(11)) % 101;
        *value = (bucket as f32 - 50.0) * 0.0025;
    }
    let sample_rows = vec![0usize, 1, 7, 17, 31, 47];
    let sample_cols = vec![0usize, 1, 63, 64, 511, 1024, 4095, pair_len - 1];
    let sample_count = sample_rows.len() * sample_cols.len();
    let expected_input_data = input_data.clone();
    let expected_raw_bytes = raw_bytes.clone();
    let expected_sample_rows = sample_rows.clone();
    let expected_sample_cols = sample_cols.clone();

    fusor_conformance::assert(move |device: Device| {
        let raw_bytes = raw_bytes.clone();
        let input_data = input_data.clone();
        let sample_rows = sample_rows.clone();
        let sample_cols = sample_cols.clone();
        async move {
            let weights = qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, GgmlType::Q4K);
            let input: Tensor<2, f32> =
                Tensor::from_slice(&device, [input_rows, weight_shape[1]], &input_data);
            let projected = input.q_mat_mul(&weights);
            let gate = projected.narrow(D::Minus1, 0, pair_len).to_concrete();
            let up = projected
                .narrow(D::Minus1, pair_len, pair_len)
                .to_concrete();
            let actual = (gate.silu() * up).to_concrete().as_slice().await.unwrap();
            let mut samples = Vec::with_capacity(sample_count);
            for &row in &sample_rows {
                for &col in &sample_cols {
                    samples.push(actual[[row, col]]);
                }
            }
            Tensor::from_slice(&device, [sample_count], &samples)
        }
    })
    .arg(|device: &Device| device.clone())
    .equal_to(move |device: Device| {
        let input_data = expected_input_data.clone();
        let raw_bytes = expected_raw_bytes.clone();
        let sample_rows = expected_sample_rows.clone();
        let sample_cols = expected_sample_cols.clone();
        async move {
            let block_at = |matrix_row: usize, block_col: usize| {
                let block_index = matrix_row * blocks_per_row + block_col;
                let offset = block_index * size_of::<BlockQ4K>();
                assert!(offset + size_of::<BlockQ4K>() <= raw_bytes.len());
                unsafe {
                    std::ptr::read_unaligned(raw_bytes.as_ptr().add(offset).cast::<BlockQ4K>())
                }
            };
            let expected = sample_rows
                .iter()
                .flat_map(|&row| {
                    let input_row = &input_data[row * weight_shape[1]..(row + 1) * weight_shape[1]];
                    sample_cols.iter().map(move |&col| {
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
                                            input_row
                                                [block_col * BlockQ4K::BLOCK_SIZE + offset]
                                                * *weight
                                        })
                                        .sum::<f32>()
                                })
                                .sum::<f32>()
                        };
                        let gate = dot(col);
                        let up = dot(col + pair_len);
                        (gate / (1.0 + (-gate).exp())) * up
                    })
                })
                .collect::<Vec<_>>();
            Tensor::from_slice(&device, [sample_count], &expected)
        }
    })
    // This case multiplies SiLU(gate) by up after two independent 4096-term
    // q4k reductions. Subgroup and no-subgroup paths use different reduction
    // trees, so the final product needs a little more relative slack than a
    // single qmatmul output.
    .compare_with(approx_or_relative_compare::<1>(2.0, 2e-4))
    .baseline_on_test_device()
    .devices_async(gpu_devices())
    .runs(1)
    .into_case(
        "quantized_matmul_paired::q4k_concat_split_llama_shape_match_cpu_reference::dense_sampled_columns",
    )
}

#[derive(Clone, Copy)]
enum GatedKind {
    SwiGLU,
    GeGLU,
    ReGLU,
}

impl GatedKind {
    fn name(self) -> &'static str {
        match self {
            GatedKind::SwiGLU => "swiglu",
            GatedKind::GeGLU => "geglu",
            GatedKind::ReGLU => "reglu",
        }
    }

    fn cpu_activation(self, x: f32) -> f32 {
        match self {
            GatedKind::SwiGLU => x / (1.0 + (-x).exp()),
            GatedKind::GeGLU => {
                // tanh approximation matching the kernel-side helper
                0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh())
            }
            GatedKind::ReGLU => {
                if x > 0.0 {
                    x
                } else {
                    0.0
                }
            }
        }
    }
}

fn gated_matches_cpu_for_rows(input_row_count: usize, kind: GatedKind) -> AssertionCase {
    use fusor::D;
    let ty = GgmlType::Q4K;
    let weight_shape = [4, 512];
    let pair_len = weight_shape[0] / 2;
    let raw_bytes = q4k_raw_bytes(weight_shape);
    let weights = QuantizedTensor::<BlockQ4K>::from_raw_bytes(weight_shape, &raw_bytes);
    let expected_weights = concrete_to_rows(&weights.dequantize::<2>(), weight_shape);

    // Author the natural graph source. Correctness verifies the generic qmatmul
    // plus dynamic nary path across gated FFN expressions.
    fusor_conformance::assert(move |input: Tensor<2, f32>| {
        let raw_bytes = raw_bytes.clone();
        async move {
            let weights = qmatrix_from_raw_bytes(&input.device(), weight_shape, &raw_bytes, ty);
            let projected = input.q_mat_mul(&weights);
            let gate = projected.narrow(D::Minus1, 0, pair_len).to_concrete();
            let up = projected
                .narrow(D::Minus1, pair_len, pair_len)
                .to_concrete();
            let activated = match kind {
                GatedKind::SwiGLU => gate.silu(),
                GatedKind::GeGLU => gate.gelu(),
                GatedKind::ReGLU => gate.relu(),
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
            let input_values = input.as_slice().await.unwrap().to_vec();
            let projected = matmul2(&input_values, &transpose2(&expected_weights));
            let expected = projected
                .iter()
                .map(|row| {
                    (0..pair_len)
                        .map(|col| kind.cpu_activation(row[col]) * row[col + pair_len])
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            Tensor::new(&device, &expected)
        }
    })
    .compare_with(approx_compare::<2, f32>(2.0))
    .baseline_on_test_device()
    .devices_async(gpu_devices())
    .runs(3)
    .into_case(format!(
        "quantized_matmul_paired::q4k_concat_split_gated_natural_form_matches_cpu_reference::{}_rows{}",
        kind.name(),
        input_row_count
    ))
}
