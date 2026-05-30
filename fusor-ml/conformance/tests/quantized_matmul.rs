mod common;

use common::quantized::{
    QMatMulFuzz, QUANTIZED_FIXTURE_CASES, QuantizedFixture,
    assert_dequantize_matches_host_reference, assert_q_mat_mul_matches_host_reference, block_count,
    f16_weight_bytes, f32_weight_bytes, f32_weight_rows, q8_0_fixture, qmatrix_from_raw_bytes,
};
use fusor::{BlockQ5_0, Device, GgmlType, Tensor};
use fusor_conformance::{approx_compare, available_devices};
use rand::distr::Uniform;
use std::mem::size_of;

#[tokio::test]
async fn quantized_dequantize_matches_cpu_reference() {
    for &(fixture, _, _) in QUANTIZED_FIXTURE_CASES
        .iter()
        .filter(|&&(_, _, dequantize)| dequantize)
    {
        let QuantizedFixture {
            ty,
            weight_shape,
            raw_bytes,
            dequantized,
            dequantize_tol,
            ..
        } = fixture();
        assert_dequantize_matches_host_reference(
            ty,
            weight_shape,
            raw_bytes,
            dequantized,
            dequantize_tol,
        )
        .await;
    }
}

#[tokio::test]
async fn quantized_q_mat_mul_matches_cpu_reference() {
    for &(fixture, seed, _) in QUANTIZED_FIXTURE_CASES {
        let fixture = fixture();
        assert_q_mat_mul_matches_host_reference(
            &fixture,
            QMatMulFuzz {
                seed,
                distribution: Uniform::new(-0.25, 0.25).unwrap(),
            },
        )
        .await;
    }
}

#[tokio::test]
async fn q8_0_dequantize_then_add_matches_cpu_reference() {
    let QuantizedFixture {
        ty,
        weight_shape,
        raw_bytes,
        dequantized,
        ..
    } = q8_0_fixture();
    let expected = dequantized
        .iter()
        .map(|row| row.iter().map(|value| value + 1.25).collect::<Vec<_>>())
        .collect::<Vec<_>>();

    fusor_conformance::assert(move |device: Device| {
        let raw_bytes = raw_bytes.clone();
        async move {
            (qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, ty).dequantize::<2>() + 1.25)
                .to_concrete()
        }
    })
    .arg(|device: &Device| device.clone())
    .equal_to(move |device: Device| {
        let expected = expected.clone();
        async move { Tensor::new(&device, &expected) }
    })
    .compare_with(approx_compare::<2, f32>(1e-5))
    .await
    .unwrap();
}

#[tokio::test]
async fn q5_0_q_mat_mul_single_row_splits_large_qgemv_dispatch() {
    const Q5_0_QGEMV_COLS_PER_WORKGROUP: usize = 8;
    const QMATMUL_MAX_WORKGROUPS_PER_DIMENSION: usize = 1_024;

    let mut exercised_subgroup_gpu = false;
    for device in available_devices().await {
        let Some(gpu) = device.as_gpu() else {
            continue;
        };
        if !gpu.subgroups_supported() {
            continue;
        }
        exercised_subgroup_gpu = true;

        let max_workgroups = (gpu.limits().max_compute_workgroups_per_dimension as usize)
            .min(QMATMUL_MAX_WORKGROUPS_PER_DIMENSION);
        let output_cols = max_workgroups * Q5_0_QGEMV_COLS_PER_WORKGROUP + 1;
        let weight_shape = [output_cols, BlockQ5_0::BLOCK_SIZE];
        let raw_bytes =
            vec![0u8; block_count(weight_shape, BlockQ5_0::BLOCK_SIZE) * size_of::<BlockQ5_0>()];
        let input_values = vec![0.25f32; weight_shape[1]];
        let weights = qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, GgmlType::Q5_0);
        let input: Tensor<2, f32> =
            Tensor::from_slice(&device, [1, weight_shape[1]], &input_values);

        let result = input.q_mat_mul(&weights).as_slice().await.unwrap();

        assert_eq!(result.shape(), &[1, output_cols]);
        assert!(
            result.as_slice().iter().all(|value| *value == 0.0),
            "zero Q5_0 weights should produce zero qgemv output"
        );
    }

    if !exercised_subgroup_gpu {
        return;
    }
}

#[tokio::test]
async fn f32_q_matrix_q_mat_mul_matches_host_reference() {
    let fixture = QuantizedFixture {
        ty: GgmlType::F32,
        weight_shape: [2, 4],
        raw_bytes: f32_weight_bytes(),
        input_row_count: 2,
        dequantized: f32_weight_rows(),
        dequantize_tol: 1e-6,
        q_mat_mul_tol: 1e-6,
    };
    assert_q_mat_mul_matches_host_reference(
        &fixture,
        QMatMulFuzz {
            seed: 820,
            distribution: Uniform::new(-0.5, 0.5).unwrap(),
        },
    )
    .await;
}

#[tokio::test]
async fn f16_q_matrix_q_mat_mul_matches_host_reference() {
    let fixture = QuantizedFixture {
        ty: GgmlType::F16,
        weight_shape: [2, 4],
        raw_bytes: f16_weight_bytes(),
        input_row_count: 2,
        dequantized: f32_weight_rows(),
        dequantize_tol: 1e-3,
        q_mat_mul_tol: 1e-3,
    };
    assert_q_mat_mul_matches_host_reference(
        &fixture,
        QMatMulFuzz {
            seed: 821,
            distribution: Uniform::new(-0.5, 0.5).unwrap(),
        },
    )
    .await;
}
