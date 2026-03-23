mod common;

use common::{
    assert_approx_devices, layer_norm_last_dim_3d, rms_norm_last_dim_3d, softmax_last_dim_2d,
};
use fusor::Tensor;

fn softmax_input() -> Vec<Vec<f32>> {
    vec![
        vec![-3.0, -1.0, 0.5, 2.0],
        vec![1.0, -2.0, 3.0, -4.0],
        vec![0.25, -0.75, 1.25, -1.5],
    ]
}

fn norm_input() -> Vec<Vec<Vec<f32>>> {
    common::reshape3(
        &(-8..16).map(|value| value as f32 / 2.0).collect::<Vec<_>>(),
        [2, 3, 4],
    )
}

#[tokio::test]
async fn softmax_and_normalization_match_reference_paths() {
    let softmax = softmax_input();
    let norm = norm_input();
    let weight = [1.0, 1.5, 0.5, 2.0];
    let bias = [0.1, -0.2, 0.3, -0.4];

    assert_approx_devices(
        |device| Tensor::new(device, &softmax).softmax::<1>(1),
        |device| Tensor::new(device, &softmax_last_dim_2d(&softmax)),
        1e-5,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &softmax).softmax_last_dim::<1>(),
        |device| Tensor::new(device, &softmax).softmax::<1>(1),
        1e-6,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &softmax).softmax_slow::<1>(1),
        |device| Tensor::new(device, &softmax).softmax::<1>(1),
        1e-6,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &softmax).softmax_slow_last_dim::<1>(),
        |device| Tensor::new(device, &softmax).softmax_last_dim::<1>(),
        1e-6,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &softmax).softmax_last_dim_fused::<1>(),
        |device| Tensor::new(device, &softmax).softmax_last_dim::<1>(),
        1e-5,
    )
    .await;

    assert_approx_devices(
        |device| {
            let x: Tensor<3, f32> = Tensor::new(device, &norm);
            let weight: Tensor<3, f32> = Tensor::from_slice(device, [1, 1, 4], &weight)
                .broadcast_as([2, 3, 4])
                .to_concrete();
            x.rms_norm::<2, _>(&weight, 1e-5)
        },
        |device| Tensor::new(device, &rms_norm_last_dim_3d(&norm, &weight, 1e-5)),
        1e-5,
    )
    .await;

    assert_approx_devices(
        |device| {
            let x: Tensor<3, f32> = Tensor::new(device, &norm);
            let weight: Tensor<3, f32> = Tensor::from_slice(device, [1, 1, 4], &weight)
                .broadcast_as([2, 3, 4])
                .to_concrete();
            let bias: Tensor<3, f32> = Tensor::from_slice(device, [1, 1, 4], &bias)
                .broadcast_as([2, 3, 4])
                .to_concrete();
            x.layer_norm::<2, _, _>(&weight, Some(&bias), 1e-5, true)
        },
        |device| Tensor::new(device, &layer_norm_last_dim_3d(&norm, &weight, &bias, 1e-5)),
        1e-5,
    )
    .await;

    assert_approx_devices(
        |device| {
            let x: Tensor<3, f32> = Tensor::new(device, &norm);
            let weight = Tensor::from_slice(device, [4], &weight);
            let bias = Tensor::from_slice(device, [4], &bias);
            x.rms_norm_fused::<1, 2>(&weight, Some(&bias), 1e-5)
        },
        |device| {
            let expected = rms_norm_last_dim_3d(&norm, &weight, 1e-5)
                .into_iter()
                .map(|matrix| {
                    matrix
                        .into_iter()
                        .map(|row| {
                            row.into_iter()
                                .zip(bias)
                                .map(|(value, bias)| value + bias)
                                .collect()
                        })
                        .collect()
                })
                .collect::<Vec<Vec<Vec<f32>>>>();
            Tensor::new(device, &expected)
        },
        1e-5,
    )
    .await;

    assert_approx_devices(
        |device| {
            let x: Tensor<3, f32> = Tensor::new(device, &norm);
            let weight = Tensor::from_slice(device, [4], &weight);
            x.rms_norm_fused_no_bias::<1, 2>(&weight, 1e-5)
        },
        |device| Tensor::new(device, &rms_norm_last_dim_3d(&norm, &weight, 1e-5)),
        1e-5,
    )
    .await;
}
