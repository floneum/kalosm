mod common;

use common::quantized::{
    deterministic_input, q4k_raw_bytes, q8_0_raw_bytes, qmatrix_from_raw_bytes,
};
use common::{matmul2, transpose2};
use fusor::{Device, Tensor, ToVec2};
use fusor_conformance::available_devices;
use fusor_cpu::{BlockQ4K, GgmlType, GgufBlock};
use std::mem::size_of;

async fn assert_q_mat_mul_3d_batch(input_rows: usize) {
    use fusor::Device;

    let weight_shape = [2usize, 64];
    let raw_bytes = q8_0_raw_bytes(weight_shape);
    let cpu_weights =
        qmatrix_from_raw_bytes(&Device::Cpu, weight_shape, &raw_bytes, GgmlType::Q8_0);
    let dequantized_rows = cpu_weights
        .dequantize::<2>()
        .as_slice()
        .await
        .unwrap()
        .to_vec2();
    let weights_t = transpose2(&dequantized_rows);

    for batch in [1usize, 2, 3] {
        let shape = [batch, input_rows, weight_shape[1]];
        let data = deterministic_input(&shape, 901 + batch as u32);

        let cpu_input: Tensor<3, f32> = Tensor::from_slice(&Device::Cpu, shape, &data);
        let cpu_result = cpu_input.q_mat_mul(&cpu_weights).to_concrete();

        // Reference: batched matmul against the dequantized weights.
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
        let expected = Tensor::new(&Device::Cpu, &expected_rows);
        fusor_conformance::approx_eq(&cpu_result, &expected, 5e-2)
            .await
            .unwrap();

        for device in available_devices().await {
            let weights = qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, GgmlType::Q8_0);
            let input: Tensor<3, f32> = Tensor::from_slice(&device, shape, &data);
            let actual = input.q_mat_mul(&weights).to_concrete();
            fusor_conformance::approx_eq(&actual, &cpu_result, 5e-2)
                .await
                .unwrap();
        }
    }
}

#[tokio::test]
async fn q_mat_mul_batched_3d_matches_host_reference() {
    assert_q_mat_mul_3d_batch(1).await;
    assert_q_mat_mul_3d_batch(3).await;
}

#[tokio::test]
async fn q_mat_mul_transposed_input_matches_host_reference() {
    use fusor::Device;

    // Build [N, M, B] and transpose(0, 2) -> [B, M, N], matching the deleted
    // `test_fuzz_q_mat_mul_transposed` topology.
    let weight_shape = [2usize, 64];
    let raw_bytes = q8_0_raw_bytes(weight_shape);
    let cpu_weights =
        qmatrix_from_raw_bytes(&Device::Cpu, weight_shape, &raw_bytes, GgmlType::Q8_0);
    let dequantized_rows = cpu_weights
        .dequantize::<2>()
        .as_slice()
        .await
        .unwrap()
        .to_vec2();
    let weights_t = transpose2(&dequantized_rows);

    for &(input_rows, batch) in &[(2usize, 2usize), (1, 3)] {
        let shape = [weight_shape[1], input_rows, batch];
        let data = deterministic_input(&shape, 1100 + batch as u32);

        let cpu_input: Tensor<3, f32> = Tensor::from_slice(&Device::Cpu, shape, &data);
        let cpu_result = cpu_input
            .transpose(0, 2)
            .q_mat_mul(&cpu_weights)
            .to_concrete();

        // Build expected via the transposed input layout.
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
        let expected = Tensor::new(&Device::Cpu, &expected_rows);
        fusor_conformance::approx_eq(&cpu_result, &expected, 5e-2)
            .await
            .unwrap();

        for device in available_devices().await {
            let weights = qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, GgmlType::Q8_0);
            let input: Tensor<3, f32> = Tensor::from_slice(&device, shape, &data);
            let actual = input.transpose(0, 2).q_mat_mul(&weights).to_concrete();
            fusor_conformance::approx_eq(&actual, &cpu_result, 5e-2)
                .await
                .unwrap();
        }
    }
}

#[tokio::test]
async fn q_mat_mul_consumes_transpose_reshape_copy_matches_cpu_reference() {
    let weight_shape = [4usize, 4096usize];
    let raw_bytes = q8_0_raw_bytes(weight_shape);
    let input_shape = [1usize, 32usize, 2usize, 128usize];
    let data = deterministic_input(&input_shape, 1401);

    let cpu_weights =
        qmatrix_from_raw_bytes(&Device::Cpu, weight_shape, &raw_bytes, GgmlType::Q8_0);
    let cpu_input: Tensor<4, f32> = Tensor::from_slice(&Device::Cpu, input_shape, &data);
    let produced = cpu_input + 0.25;
    let transposed = produced.transpose(1, 2);
    let reshaped = transposed.reshape([1, 2, 32 * 128]);
    let cpu_result = reshaped.q_mat_mul(&cpu_weights).to_concrete();

    for device in available_devices().await {
        let weights = qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, GgmlType::Q8_0);
        let input: Tensor<4, f32> = Tensor::from_slice(&device, input_shape, &data);
        let produced = input + 0.25;
        let transposed = produced.transpose(1, 2);
        let reshaped = transposed.reshape([1, 2, 32 * 128]);
        let actual = reshaped.q_mat_mul(&weights).to_concrete();
        fusor_conformance::approx_eq(&actual, &cpu_result, 5e-2)
            .await
            .unwrap();
    }
}

#[tokio::test]
async fn q4k_llama_decode_transpose_reshape_qmatmul_matches_one_hot_reference() {
    let Some(device) = available_devices()
        .await
        .into_iter()
        .find(|device| device.as_gpu().is_some())
    else {
        return;
    };
    let Some(gpu_device) = device.as_gpu() else {
        return;
    };
    if !gpu_device.subgroups_supported() {
        return;
    }

    for (weight_shape, sample_cols) in [
        (
            [5120usize, 4096usize],
            &[0usize, 1, 63, 64, 511, 1024, 4095, 5119][..],
        ),
        (
            [14336usize, 4096usize],
            &[0usize, 1, 63, 64, 511, 1024, 4095, 8191, 14335][..],
        ),
    ] {
        assert_q4k_llama_decode_transpose_reshape_shape(&device, weight_shape, sample_cols).await;
    }
}

async fn assert_q4k_llama_decode_transpose_reshape_shape(
    device: &Device,
    weight_shape: [usize; 2],
    sample_cols: &[usize],
) {
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
    let weights = qmatrix_from_raw_bytes(device, weight_shape, &raw_bytes, GgmlType::Q4K);

    let mut input_data = vec![-0.25f32; input_shape.iter().product()];
    let mut row_values = Vec::with_capacity(input_shape[2]);
    for row in 0..input_shape[2] {
        let row_value = 0.125 + row as f32 * 0.01;
        row_values.push(row_value);
        let index = ((selected_head * input_shape[2] + row) * input_shape[3]) + selected_dim;
        input_data[index] = row_value - 0.25;
    }

    let input: Tensor<4, f32> = Tensor::from_slice(device, input_shape, &input_data);
    let actual = (input + 0.25)
        .transpose(1, 2)
        .reshape([1, input_shape[2], hidden])
        .q_mat_mul(&weights)
        .as_slice()
        .await
        .unwrap();

    assert_eq!(actual.shape(), &[1, input_shape[2], output_cols]);
    for row in [0usize, 1, 7, 17, 31, 47] {
        for &col in sample_cols {
            let block_index = col * blocks_per_row + selected_block_in_row;
            let offset = block_index * size_of::<BlockQ4K>();
            assert!(offset + size_of::<BlockQ4K>() <= raw_bytes.len());
            let block = unsafe {
                std::ptr::read_unaligned(raw_bytes.as_ptr().add(offset).cast::<BlockQ4K>())
            };
            let expected = row_values[row] * block.dequantize().as_ref()[selected_offset];
            let actual = actual[[0, row, col]];
            let tolerance = 1e-2_f32.max(expected.abs() * 1.0e-4);
            assert!(
                (actual - expected).abs() <= tolerance,
                "shape={weight_shape:?} row={row} col={col} actual={actual} expected={expected} tolerance={tolerance}"
            );
        }
    }
}

#[tokio::test]
async fn q_mat_mul_batched_matches_unbatched_property() {
    // Batched 3D q_mat_mul produces the same per-batch slice as 2D q_mat_mul
    // applied independently. Replaces
    // `cpu/src/quantized.rs::test_batched_q_mat_mul_matches_unbatched`.
    let weight_shape = [2usize, 64];
    let raw_bytes = q8_0_raw_bytes(weight_shape);
    let batch = 3;
    let input_rows = 2;
    let shape = [batch, input_rows, weight_shape[1]];
    let data = deterministic_input(&shape, 1300);

    for device in available_devices().await {
        let weights = qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, GgmlType::Q8_0);
        let batched: Tensor<3, f32> = Tensor::from_slice(&device, shape, &data);
        let batched_result = batched.q_mat_mul(&weights).to_concrete();

        for b in 0..batch {
            let slice_data: Vec<f32> = data
                [b * input_rows * weight_shape[1]..(b + 1) * input_rows * weight_shape[1]]
                .to_vec();
            let unbatched: Tensor<2, f32> =
                Tensor::from_slice(&device, [input_rows, weight_shape[1]], &slice_data);
            let unbatched_result = unbatched.q_mat_mul(&weights).to_concrete();

            // Pull batched slice as 2D for comparison.
            let batched_slice = batched_result
                .clone()
                .slice([b..b + 1, 0..input_rows, 0..weight_shape[0]])
                .reshape([input_rows, weight_shape[0]])
                .to_concrete();
            fusor_conformance::approx_eq(&batched_slice, &unbatched_result, 1e-4)
                .await
                .unwrap();
        }
    }
}
