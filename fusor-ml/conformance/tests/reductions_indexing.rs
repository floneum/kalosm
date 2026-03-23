mod common;

use common::{
    assert_approx_devices, assert_approx_tensors, index_select2, keepdim2, mean_axis2,
    reduce_axis2, slice_assign2, var_axis2,
};
use fusor::{Tensor, arange};
use fusor_conformance::{available_devices, sequential_tensor};
use half::f16;

fn matrix_3x4() -> Vec<Vec<f32>> {
    vec![
        vec![0.0, 1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0, 7.0],
        vec![8.0, 9.0, 10.0, 11.0],
    ]
}

#[tokio::test]
async fn reductions_match_host_reference() {
    let matrix = matrix_3x4();

    assert_approx_devices(
        |device| sequential_tensor::<2, f32>(device, [3, 4]).sum::<1>(1),
        |device| {
            Tensor::new(
                device,
                &reduce_axis2(&matrix, 1, 0.0, |acc, value| acc + value),
            )
        },
        1e-6,
    )
    .await;

    assert_approx_devices(
        |device| sequential_tensor::<2, f32>(device, [3, 4]).max::<1>(0),
        |device| {
            Tensor::new(
                device,
                &reduce_axis2(&matrix, 0, f32::NEG_INFINITY, f32::max),
            )
        },
        1e-6,
    )
    .await;

    assert_approx_devices(
        |device| sequential_tensor::<2, f32>(device, [3, 4]).min::<1>(0),
        |device| Tensor::new(device, &reduce_axis2(&matrix, 0, f32::INFINITY, f32::min)),
        1e-6,
    )
    .await;

    assert_approx_devices(
        |device| sequential_tensor::<2, f32>(device, [3, 4]).product::<1>(1),
        |device| {
            Tensor::new(
                device,
                &reduce_axis2(&matrix, 1, 1.0, |acc, value| acc * value),
            )
        },
        1e-6,
    )
    .await;

    assert_approx_devices(
        |device| sequential_tensor::<2, f32>(device, [3, 4]).sum_keepdim::<1>(1),
        |device| {
            Tensor::new(
                device,
                &keepdim2(&reduce_axis2(&matrix, 1, 0.0, |acc, value| acc + value), 1),
            )
        },
        1e-6,
    )
    .await;

    assert_approx_devices(
        |device| sequential_tensor::<2, f32>(device, [3, 4]).max_keepdim::<1>(0),
        |device| {
            Tensor::new(
                device,
                &keepdim2(&reduce_axis2(&matrix, 0, f32::NEG_INFINITY, f32::max), 0),
            )
        },
        1e-6,
    )
    .await;

    assert_approx_devices(
        |device| sequential_tensor::<2, f32>(device, [3, 4]).min_keepdim::<1>(0),
        |device| {
            Tensor::new(
                device,
                &keepdim2(&reduce_axis2(&matrix, 0, f32::INFINITY, f32::min), 0),
            )
        },
        1e-6,
    )
    .await;

    assert_approx_devices(
        |device| sequential_tensor::<2, f32>(device, [3, 4]).product_keepdim::<1>(1),
        |device| {
            Tensor::new(
                device,
                &keepdim2(&reduce_axis2(&matrix, 1, 1.0, |acc, value| acc * value), 1),
            )
        },
        1e-6,
    )
    .await;

    assert_approx_devices(
        |device| sequential_tensor::<2, f32>(device, [3, 4]).mean::<1>(1),
        |device| Tensor::new(device, &mean_axis2(&matrix, 1)),
        1e-6,
    )
    .await;

    assert_approx_devices(
        |device| sequential_tensor::<2, f32>(device, [3, 4]).mean_keepdim::<1>(1),
        |device| Tensor::new(device, &keepdim2(&mean_axis2(&matrix, 1), 1)),
        1e-6,
    )
    .await;

    assert_approx_devices(
        |device| sequential_tensor::<2, f32>(device, [3, 4]).var::<1>(1),
        |device| Tensor::new(device, &var_axis2(&matrix, 1)),
        1e-5,
    )
    .await;

    assert_approx_devices(
        |device| sequential_tensor::<2, f32>(device, [3, 4]).var_keepdim::<1>(1),
        |device| Tensor::new(device, &keepdim2(&var_axis2(&matrix, 1), 1)),
        1e-5,
    )
    .await;
}

#[tokio::test]
async fn indexing_cast_and_rank_specific_indexing_match_reference() {
    let matrix = vec![
        vec![0.0, 1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0, 7.0],
        vec![8.0, 9.0, 10.0, 11.0],
        vec![12.0, 13.0, 14.0, 15.0],
    ];

    assert_approx_devices(
        |device| {
            let x: Tensor<2, f32> = Tensor::new(device, &matrix);
            let indices: Tensor<1, u32> = Tensor::from_slice(device, [3], &[3u32, 1, 0]);
            x.index_select(1, &indices)
        },
        |device| Tensor::new(device, &index_select2(&matrix, 1, &[3, 1, 0])),
        1e-6,
    )
    .await;

    assert_approx_devices(
        |device| {
            let x: Tensor<2, f32> = Tensor::new(device, &matrix);
            let patch: Tensor<2, f32> =
                Tensor::from_slice(device, [2, 2], &[100.0, 101.0, 102.0, 103.0]);
            x.slice_assign([1..3, 1..3], &patch)
        },
        |device| {
            let patch = vec![vec![100.0, 101.0], vec![102.0, 103.0]];
            Tensor::new(device, &slice_assign2(&matrix, 1..3, 1..3, &patch))
        },
        1e-6,
    )
    .await;

    assert_approx_devices(
        |device| {
            Tensor::from_slice(device, [2, 3], &[0.1, 1.9, 2.4, 3.8, 4.2, 5.9])
                .cast::<f16>()
                .cast::<f32>()
                .to_concrete()
        },
        |device| {
            let expected = vec![
                vec![
                    f16::from_f32(0.1).to_f32(),
                    f16::from_f32(1.9).to_f32(),
                    f16::from_f32(2.4).to_f32(),
                ],
                vec![
                    f16::from_f32(3.8).to_f32(),
                    f16::from_f32(4.2).to_f32(),
                    f16::from_f32(5.9).to_f32(),
                ],
            ];
            Tensor::new(device, &expected)
        },
        1e-6,
    )
    .await;

    for device in available_devices().await {
        let matrix: Tensor<2, f32> = sequential_tensor(&device, [3, 4]);
        let indexed_2d = matrix.i((1, 1..4));
        let sliced_2d = matrix.slice([1..2, 1..4]).squeeze::<1>(0).to_concrete();
        assert_approx_tensors(indexed_2d, sliced_2d, 1e-6).await;

        let tensor3: Tensor<3, f32> = sequential_tensor(&device, [2, 3, 4]);
        let indexed_3d = tensor3.i((0, 1..3, 1..4));
        let sliced_3d = tensor3
            .slice([0..1, 1..3, 1..4])
            .squeeze::<2>(0)
            .to_concrete();
        assert_approx_tensors(indexed_3d, sliced_3d, 1e-6).await;

        let tensor4: Tensor<4, f32> = arange(&device, 0.0f32, 48.0)
            .reshape([2, 2, 3, 4])
            .to_concrete();
        let indexed_4d = tensor4.i((1, 0..2, 1..3, 1..4));
        let sliced_4d = tensor4
            .slice([1..2, 0..2, 1..3, 1..4])
            .squeeze::<3>(0)
            .to_concrete();
        assert_approx_tensors(indexed_4d, sliced_4d, 1e-6).await;
    }
}
