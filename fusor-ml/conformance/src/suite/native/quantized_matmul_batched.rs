//! Batched quantized matmul conformance cases.

use crate::common::quantized::{
    deterministic_input, q4k_raw_bytes, q8_0_raw_bytes, qmatrix_from_raw_bytes,
};
use crate::common::{matmul2, transpose2};
use fusor::{BlockQ4K, Device, GgmlType, GgufBlock, Tensor, ToVec2};
use fusor_conformance::{AssertionCase, AssertionCases, approx_compare, available_devices};
use std::mem::size_of;

async fn gpu_devices() -> Vec<Device> {
    available_devices()
        .await
        .into_iter()
        .filter(|device| device.as_gpu().is_some())
        .collect()
}

/// Build a single contiguous `[batch, input_rows, K]` batched q_mat_mul case
/// against the host reference (dequantize -> transpose -> per-batch matmul2).
fn assert_q_mat_mul_3d_contiguous(input_rows: usize, batch: usize) -> AssertionCase {
    let weight_shape = [2usize, 64];
    let raw_bytes = q8_0_raw_bytes(weight_shape);
    let shape = [batch, input_rows, weight_shape[1]];
    let data = deterministic_input(&shape, 901 + batch as u32);

    fusor_conformance::assert({
        let raw_bytes = raw_bytes.clone();
        let data = data.clone();
        move |device: Device| {
            let raw_bytes = raw_bytes.clone();
            let data = data.clone();
            async move {
                let weights =
                    qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, GgmlType::Q8_0);
                let input: Tensor<3, f32> = Tensor::from_slice(&device, shape, &data);
                input.q_mat_mul(&weights).to_concrete()
            }
        }
    })
    .arg(|device: &Device| device.clone())
    .equal_to({
        let raw_bytes = raw_bytes.clone();
        let data = data.clone();
        move |device: Device| {
            let raw_bytes = raw_bytes.clone();
            let data = data.clone();
            async move {
                let cpu_weights =
                    qmatrix_from_raw_bytes(&Device::Cpu, weight_shape, &raw_bytes, GgmlType::Q8_0);
                let dequantized_rows = cpu_weights
                    .dequantize::<2>()
                    .as_slice()
                    .await
                    .unwrap()
                    .to_vec2();
                let weights_t = transpose2(&dequantized_rows);
                let mut expected_rows = Vec::with_capacity(batch);
                for b in 0..batch {
                    let slice: Vec<Vec<f32>> = (0..input_rows)
                        .map(|m| {
                            let start = ((b * input_rows) + m) * weight_shape[1];
                            data[start..start + weight_shape[1]].to_vec()
                        })
                        .collect();
                    expected_rows.push(matmul2(&slice, &weights_t));
                }
                Tensor::new(&device, &expected_rows)
            }
        }
    })
    .compare_with(approx_compare::<3, f32>(5e-2))
    .runs(1)
    .into_case(format!(
        "quantized_matmul_batched::q_mat_mul_batched_layouts_match_host_reference::contiguous_rows{input_rows}_batch{batch}"
    ))
}

/// Build a single `[K, input_rows, batch]` -> `transpose(0, 2)` -> `[batch, input_rows, K]`
/// batched q_mat_mul case against the host reference, exercising the non-contiguous
/// transpose layout (matches the deleted `test_fuzz_q_mat_mul_transposed` topology).
fn assert_q_mat_mul_3d_transposed(input_rows: usize, batch: usize) -> AssertionCase {
    let weight_shape = [2usize, 64];
    let raw_bytes = q8_0_raw_bytes(weight_shape);
    let shape = [weight_shape[1], input_rows, batch];
    let data = deterministic_input(&shape, 1100 + batch as u32);

    fusor_conformance::assert({
        let raw_bytes = raw_bytes.clone();
        let data = data.clone();
        move |device: Device| {
            let raw_bytes = raw_bytes.clone();
            let data = data.clone();
            async move {
                let weights =
                    qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, GgmlType::Q8_0);
                let input: Tensor<3, f32> = Tensor::from_slice(&device, shape, &data);
                input.transpose(0, 2).q_mat_mul(&weights).to_concrete()
            }
        }
    })
    .arg(|device: &Device| device.clone())
    .equal_to({
        let raw_bytes = raw_bytes.clone();
        let data = data.clone();
        move |device: Device| {
            let raw_bytes = raw_bytes.clone();
            let data = data.clone();
            async move {
                let cpu_weights =
                    qmatrix_from_raw_bytes(&Device::Cpu, weight_shape, &raw_bytes, GgmlType::Q8_0);
                let dequantized_rows = cpu_weights
                    .dequantize::<2>()
                    .as_slice()
                    .await
                    .unwrap()
                    .to_vec2();
                let weights_t = transpose2(&dequantized_rows);
                let mut expected_rows = Vec::with_capacity(batch);
                for b in 0..batch {
                    let slice: Vec<Vec<f32>> = (0..input_rows)
                        .map(|m| {
                            (0..weight_shape[1])
                                .map(|n| {
                                    let idx = (n * input_rows + m) * batch + b;
                                    data[idx]
                                })
                                .collect()
                        })
                        .collect();
                    expected_rows.push(matmul2(&slice, &weights_t));
                }
                Tensor::new(&device, &expected_rows)
            }
        }
    })
    .compare_with(approx_compare::<3, f32>(5e-2))
    .runs(1)
    .into_case(format!(
        "quantized_matmul_batched::q_mat_mul_batched_layouts_match_host_reference::transposed_rows{input_rows}_batch{batch}"
    ))
}

/// Batched 3D q_mat_mul matches the host reference across both the contiguous
/// `[batch, rows, K]` layout (batch{1,2,3} x rows{1,3}) and the non-contiguous
/// `transpose(0, 2)` layout (rows/batch tuples [64,2,2] and [64,1,3]). All cases
/// use a Q8_0 `[2, 64]` weight, `approx_compare::<3>(5e-2)`, and run on all devices.
pub fn q_mat_mul_batched_layouts_match_host_reference() -> AssertionCases {
    let mut assertions = AssertionCases::new();
    for input_rows in [1usize, 3] {
        for batch in [1usize, 2, 3] {
            assertions.push(assert_q_mat_mul_3d_contiguous(input_rows, batch));
        }
    }
    for (input_rows, batch) in [(2usize, 2usize), (1, 3)] {
        assertions.push(assert_q_mat_mul_3d_transposed(input_rows, batch));
    }
    assertions
}

pub fn q_mat_mul_consumes_transpose_reshape_copy_matches_cpu_reference() -> AssertionCase {
    let weight_shape = [4usize, 4096usize];
    let raw_bytes = q8_0_raw_bytes(weight_shape);
    let input_shape = [1usize, 32usize, 2usize, 128usize];
    let data = deterministic_input(&input_shape, 1401);

    fusor_conformance::assert(move |device: Device| {
        let raw_bytes = raw_bytes.clone();
        let data = data.clone();
        async move {
            let weights = qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, GgmlType::Q8_0);
            let input: Tensor<4, f32> = Tensor::from_slice(&device, input_shape, &data);
            let produced = input + 0.25;
            let transposed = produced.transpose(1, 2);
            let reshaped = transposed.reshape([1, 2, 32 * 128]);
            reshaped.q_mat_mul(&weights).to_concrete()
        }
    })
    .arg(|device: &Device| device.clone())
    .compare_with(approx_compare::<3, f32>(5e-2))
    .runs(1)
    .into_case(
        "quantized_matmul_batched::q_mat_mul_consumes_transpose_reshape_copy_matches_cpu_reference",
    )
}

pub fn q4k_llama_decode_transpose_reshape_qmatmul_matches_one_hot_reference() -> AssertionCases {
    [
        (
            [5120usize, 4096usize],
            &[0usize, 1, 63, 64, 511, 1024, 4095, 5119][..],
        ),
        (
            [14336usize, 4096usize],
            &[0usize, 1, 63, 64, 511, 1024, 4095, 8191, 14335][..],
        ),
    ]
    .into_iter()
    .map(|(weight_shape, sample_cols)| {
        assert_q4k_llama_decode_transpose_reshape_shape(weight_shape, sample_cols)
    })
    .collect::<Vec<_>>()
    .into()
}

fn assert_q4k_llama_decode_transpose_reshape_shape(
    weight_shape: [usize; 2],
    sample_cols: &[usize],
) -> AssertionCase {
    let [output_cols, hidden] = weight_shape;
    let input_shape = [1usize, 32usize, 48usize, 128usize];
    assert_eq!(hidden, input_shape[1] * input_shape[3]);
    let selected_k = 777usize;
    let selected_head = selected_k / input_shape[3];
    let selected_dim = selected_k % input_shape[3];
    let selected_block_in_row = selected_k / BlockQ4K::BLOCK_SIZE;
    let selected_offset = selected_k % BlockQ4K::BLOCK_SIZE;
    let blocks_per_row = hidden / BlockQ4K::BLOCK_SIZE;
    let raw_bytes = q4k_raw_bytes(weight_shape);
    let sample_cols = sample_cols.to_vec();
    let sample_rows = vec![0usize, 1, 7, 17, 31, 47];
    let sample_count = sample_rows.len() * sample_cols.len();

    let mut input_data = vec![-0.25f32; input_shape.iter().product()];
    let mut row_values = Vec::with_capacity(input_shape[2]);
    for row in 0..input_shape[2] {
        let row_value = 0.125 + row as f32 * 0.01;
        row_values.push(row_value);
        let index = ((selected_head * input_shape[2] + row) * input_shape[3]) + selected_dim;
        input_data[index] = row_value - 0.25;
    }
    let expected_raw_bytes = raw_bytes.clone();
    let expected_row_values = row_values.clone();
    let expected_sample_rows = sample_rows.clone();
    let expected_sample_cols = sample_cols.clone();

    fusor_conformance::assert(move |device: Device| {
        let raw_bytes = raw_bytes.clone();
        let input_data = input_data.clone();
        let sample_rows = sample_rows.clone();
        let sample_cols = sample_cols.clone();
        async move {
            let weights = qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, GgmlType::Q4K);
            let input: Tensor<4, f32> = Tensor::from_slice(&device, input_shape, &input_data);
            let actual = (input + 0.25)
                .transpose(1, 2)
                .reshape([1, input_shape[2], hidden])
                .q_mat_mul(&weights)
                .as_slice()
                .await
                .unwrap();
            assert_eq!(actual.shape(), &[1, input_shape[2], output_cols]);
            let mut samples = Vec::with_capacity(sample_count);
            for &row in &sample_rows {
                for &col in &sample_cols {
                    samples.push(actual[[0, row, col]]);
                }
            }
            Tensor::from_slice(&device, [sample_count], &samples)
        }
    })
    .arg(|device: &Device| device.clone())
    .equal_to(move |device: Device| {
        let raw_bytes = expected_raw_bytes.clone();
        let row_values = expected_row_values.clone();
        let sample_rows = expected_sample_rows.clone();
        let sample_cols = expected_sample_cols.clone();
        async move {
            let mut expected = Vec::with_capacity(sample_count);
            for row in sample_rows {
                for &col in &sample_cols {
                    let block_index = col * blocks_per_row + selected_block_in_row;
                    let offset = block_index * size_of::<BlockQ4K>();
                    assert!(offset + size_of::<BlockQ4K>() <= raw_bytes.len());
                    let block = unsafe {
                        std::ptr::read_unaligned(raw_bytes.as_ptr().add(offset).cast::<BlockQ4K>())
                    };
                    expected.push(row_values[row] * block.dequantize().as_ref()[selected_offset]);
                }
            }
            Tensor::from_slice(&device, [sample_count], &expected)
        }
    })
    .compare_with(approx_compare::<1, f32>(1e-2))
    .baseline_on_test_device()
    .devices_async(gpu_devices())
    .runs(1)
    .into_case(format!(
        "quantized_matmul_batched::q4k_llama_decode_transpose_reshape_qmatmul_matches_one_hot_reference::{weight_shape:?}"
    ))
}

pub fn q_mat_mul_batched_matches_unbatched_property() -> AssertionCase {
    // Batched 3D q_mat_mul produces the same per-batch slice as 2D q_mat_mul
    // applied independently. Replaces
    // `cpu/src/quantized.rs::test_batched_q_mat_mul_matches_unbatched`.
    let weight_shape = [2usize, 64];
    let raw_bytes = q8_0_raw_bytes(weight_shape);
    let batch = 3;
    let input_rows = 2;
    let shape = [batch, input_rows, weight_shape[1]];
    let data = deterministic_input(&shape, 1300);

    fusor_conformance::assert({
        let raw_bytes = raw_bytes.clone();
        let data = data.clone();
        move |device: Device| {
            let raw_bytes = raw_bytes.clone();
            let data = data.clone();
            async move {
                let weights =
                    qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, GgmlType::Q8_0);
                let batched: Tensor<3, f32> = Tensor::from_slice(&device, shape, &data);
                batched.q_mat_mul(&weights).to_concrete()
            }
        }
    })
    .arg(|device: &Device| device.clone())
    .equal_to(move |device: Device| {
        let raw_bytes = raw_bytes.clone();
        let data = data.clone();
        async move {
            let weights = qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, GgmlType::Q8_0);
            let mut expected = Vec::with_capacity(batch);
            for b in 0..batch {
                let slice_data: Vec<f32> = data
                    [b * input_rows * weight_shape[1]..(b + 1) * input_rows * weight_shape[1]]
                    .to_vec();
                let unbatched: Tensor<2, f32> =
                    Tensor::from_slice(&device, [input_rows, weight_shape[1]], &slice_data);
                let result = unbatched.q_mat_mul(&weights).to_concrete();
                expected.push(result.as_slice().await.unwrap().to_vec2());
            }
            Tensor::new(&device, &expected)
        }
    })
    .compare_with(approx_compare::<3, f32>(1e-4))
    .runs(1)
    .into_case("quantized_matmul_batched::q_mat_mul_batched_matches_unbatched_property")
}
