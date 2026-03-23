mod common;

use common::{assert_approx_devices, conv1d_ncw, matmul2, pool1d_ncw, reshape3};
use fusor::Tensor;

fn matmul_lhs() -> Vec<Vec<f32>> {
    vec![
        vec![-2.0, -1.0, 0.0, 1.0],
        vec![2.0, 3.0, 4.0, 5.0],
        vec![6.0, 7.0, 8.0, 9.0],
    ]
}

fn matmul_rhs() -> Vec<Vec<f32>> {
    vec![
        vec![-1.0, 0.5],
        vec![1.0, -0.5],
        vec![2.0, 1.5],
        vec![3.0, -1.5],
    ]
}

fn conv_input() -> Vec<Vec<Vec<f32>>> {
    vec![vec![vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]]]
}

fn pool_input() -> Vec<Vec<Vec<f32>>> {
    reshape3(
        &(0..16).map(|value| value as f32 - 4.0).collect::<Vec<_>>(),
        [1, 2, 8],
    )
}

#[tokio::test]
async fn matmul_conv_and_pool_match_host_reference() {
    let lhs = matmul_lhs();
    let rhs = matmul_rhs();
    let conv = conv_input();
    let pool = pool_input();

    assert_approx_devices(
        |device| {
            let lhs: Tensor<2, f32> = Tensor::new(device, &lhs);
            let rhs: Tensor<2, f32> = Tensor::new(device, &rhs);
            lhs.matmul(&rhs)
        },
        |device| Tensor::new(device, &matmul2(&lhs, &rhs)),
        1e-5,
    )
    .await;

    assert_approx_devices(
        |device| {
            let lhs: Tensor<2, f32> = Tensor::new(device, &lhs);
            let rhs: Tensor<2, f32> = Tensor::new(device, &rhs);
            lhs.mat_mul(&rhs)
        },
        |device| {
            let lhs: Tensor<2, f32> = Tensor::new(device, &lhs);
            let rhs: Tensor<2, f32> = Tensor::new(device, &rhs);
            lhs.matmul(&rhs)
        },
        1e-6,
    )
    .await;

    assert_approx_devices(
        |device| {
            let input: Tensor<3, f32> = Tensor::new(device, &conv);
            let weight = Tensor::from_slice(device, [1, 1, 3], &[0.25, -0.5, 1.0]);
            let bias = Tensor::from_slice(device, [1], &[0.1]);
            input.conv(&weight, Some(&bias), [1], [2])
        },
        |device| {
            let expected = conv1d_ncw(&conv, &[vec![vec![0.25, -0.5, 1.0]]], Some(&[0.1]), 1, 2);
            Tensor::new(device, &expected)
        },
        1e-5,
    )
    .await;

    assert_approx_devices(
        |device| -> Tensor<3, f32> { Tensor::new(device, &pool).pool([(2, 2)], Tensor::max) },
        |device| Tensor::new(device, &pool).pool_max([(2, 2)]),
        1e-6,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &pool).pool_max([(2, 2)]),
        |device| {
            Tensor::new(
                device,
                &pool1d_ncw(&pool, 2, 2, f32::max, f32::NEG_INFINITY),
            )
        },
        1e-6,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &pool).pool_min([(2, 2)]),
        |device| Tensor::new(device, &pool1d_ncw(&pool, 2, 2, f32::min, f32::INFINITY)),
        1e-6,
    )
    .await;
}
