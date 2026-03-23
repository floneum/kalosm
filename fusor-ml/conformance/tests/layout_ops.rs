mod common;

use common::{
    assert_approx_devices, assert_approx_tensors, assert_exact_devices, broadcast_1d_to_2d,
    flatten2, flatten3, permute3, repeat2, reshape2, reshape3, resize2, slice2,
    sliding_window_1d_ncw, transpose2,
};
use fusor::{Tensor, cat, stack};
use fusor_conformance::{available_devices, sequential_tensor};
use fusor_types::{SlidingWindow, StrideSpec};

fn matrix_2x3() -> Vec<Vec<f32>> {
    vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]]
}

fn matrix_2x4() -> Vec<Vec<f32>> {
    vec![vec![0.0, 1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0, 7.0]]
}

fn tensor_2x3x4() -> Vec<Vec<Vec<f32>>> {
    reshape3(
        &(0..24).map(|value| value as f32).collect::<Vec<_>>(),
        [2, 3, 4],
    )
}

fn tensor_2x1x3x1() -> Vec<Vec<Vec<Vec<f32>>>> {
    vec![
        vec![vec![vec![0.0], vec![1.0], vec![2.0]]],
        vec![vec![vec![3.0], vec![4.0], vec![5.0]]],
    ]
}

fn tensor_1x2x5() -> Vec<Vec<Vec<f32>>> {
    reshape3(
        &(0..10).map(|value| value as f32).collect::<Vec<_>>(),
        [1, 2, 5],
    )
}

#[tokio::test]
async fn shape_and_layout_ops_match_host_reference() {
    let matrix = matrix_2x3();
    let sliced_matrix = matrix_2x4();
    let tensor3 = tensor_2x3x4();
    let tensor_with_singletons = tensor_2x1x3x1();
    let vector = vec![0.0, 1.0, 2.0];
    let sliding = tensor_1x2x5();
    let restride_specs = [StrideSpec::dim(1, 3), StrideSpec::dim(0, 2)];

    assert_approx_devices(
        |device| {
            sequential_tensor::<2, f32>(device, [2, 3])
                .reshape([3, 2])
                .to_concrete()
        },
        |device| {
            let expected = reshape2(&flatten2(&matrix), [3, 2]);
            Tensor::new(device, &expected)
        },
        0.0,
    )
    .await;

    assert_approx_devices(
        |device| {
            sequential_tensor::<2, f32>(device, [2, 3])
                .transpose(0, 1)
                .to_concrete()
        },
        |device| Tensor::new(device, &transpose2(&matrix)),
        0.0,
    )
    .await;

    assert_approx_devices(
        |device| {
            sequential_tensor::<2, f32>(device, [2, 3])
                .restride(restride_specs)
                .to_concrete()
        },
        |device| Tensor::new(device, &transpose2(&matrix)),
        0.0,
    )
    .await;

    assert_approx_devices(
        |device| {
            sequential_tensor::<2, f32>(device, [2, 3])
                .t()
                .to_concrete()
        },
        |device| {
            sequential_tensor::<2, f32>(device, [2, 3])
                .transpose(0, 1)
                .to_concrete()
        },
        0.0,
    )
    .await;

    assert_approx_devices(
        |device| {
            sequential_tensor::<3, f32>(device, [2, 3, 4])
                .permute([1, 2, 0])
                .to_concrete()
        },
        |device| Tensor::new(device, &permute3(&tensor3, [1, 2, 0])),
        0.0,
    )
    .await;

    assert_approx_devices(
        |device| {
            sequential_tensor::<2, f32>(device, [2, 4])
                .slice([0..2, 1..3])
                .to_concrete()
        },
        |device| Tensor::new(device, &slice2(&sliced_matrix, 0..2, 1..3)),
        0.0,
    )
    .await;

    assert_approx_devices(
        |device| {
            sequential_tensor::<2, f32>(device, [2, 4])
                .narrow(1, 1, 2)
                .to_concrete()
        },
        |device| Tensor::new(device, &slice2(&sliced_matrix, 0..2, 1..3)),
        0.0,
    )
    .await;

    assert_approx_devices(
        |device| {
            sequential_tensor::<1, f32>(device, [3])
                .broadcast_as([2, 3])
                .to_concrete()
        },
        |device| Tensor::new(device, &broadcast_1d_to_2d(&vector, 2)),
        0.0,
    )
    .await;

    assert_approx_devices(
        |device| {
            sequential_tensor::<1, f32>(device, [3])
                .expand([2, 3])
                .to_concrete()
        },
        |device| {
            sequential_tensor::<1, f32>(device, [3])
                .broadcast_as([2, 3])
                .to_concrete()
        },
        0.0,
    )
    .await;

    assert_approx_devices(
        |device| {
            sequential_tensor::<2, f32>(device, [2, 3])
                .flatten_all()
                .to_concrete()
        },
        |device| Tensor::new(device, &flatten2(&matrix)),
        0.0,
    )
    .await;

    assert_approx_devices(
        |device| sequential_tensor::<3, f32>(device, [2, 3, 4]).flatten_first_n::<1, 2>(),
        |device| Tensor::new(device, &reshape2(&flatten3(&tensor3), [6, 4])),
        0.0,
    )
    .await;

    assert_approx_devices(
        |device| sequential_tensor::<3, f32>(device, [2, 3, 4]).flatten_last_n::<1, 2>(),
        |device| Tensor::new(device, &reshape2(&flatten3(&tensor3), [2, 12])),
        0.0,
    )
    .await;

    assert_approx_devices(
        |device| sequential_tensor::<2, f32>(device, [2, 2]).repeat([2, 3]),
        |device| {
            let base = vec![vec![0.0, 1.0], vec![2.0, 3.0]];
            Tensor::new(device, &repeat2(&base, [2, 3]))
        },
        0.0,
    )
    .await;

    assert_approx_devices(
        |device| {
            sequential_tensor::<2, f32>(device, [2, 3])
                .unsqueeze::<3>(1)
                .to_concrete()
        },
        |device| {
            let expected: Vec<Vec<Vec<f32>>> =
                matrix.iter().cloned().map(|row| vec![row]).collect();
            Tensor::new(device, &expected)
        },
        0.0,
    )
    .await;

    assert_approx_devices(
        |device| {
            sequential_tensor::<3, f32>(device, [2, 1, 3])
                .squeeze::<2>(1)
                .to_concrete()
        },
        |device| Tensor::new(device, &matrix),
        0.0,
    )
    .await;

    assert_approx_devices(
        |device| {
            sequential_tensor::<2, f32>(device, [2, 3])
                .unsqueeze_dims::<2, 4>([1, 3])
                .to_concrete()
        },
        |device| {
            sequential_tensor::<2, f32>(device, [2, 3])
                .unsqueeze::<3>(1)
                .unsqueeze::<4>(3)
                .to_concrete()
        },
        0.0,
    )
    .await;

    assert_approx_devices(
        |device| {
            sequential_tensor::<4, f32>(device, [2, 1, 3, 1])
                .squeeze_dims::<2, 2>([1, 3])
                .to_concrete()
        },
        |device| {
            sequential_tensor::<4, f32>(device, [2, 1, 3, 1])
                .squeeze::<3>(3)
                .squeeze::<2>(1)
                .to_concrete()
        },
        0.0,
    )
    .await;

    assert_approx_devices(
        |device| {
            sequential_tensor::<3, f32>(device, [1, 2, 5])
                .sliding_window_view::<1, 4>([SlidingWindow::new(2, 3, 2)])
                .to_concrete()
        },
        |device| Tensor::new(device, &sliding_window_1d_ncw(&sliding, 3, 2)),
        0.0,
    )
    .await;

    assert_approx_devices(
        |device| {
            sequential_tensor::<2, f32>(device, [2, 3])
                .to_concrete()
                .resize([3, 4])
        },
        |device| Tensor::new(device, &resize2(&matrix, [3, 4])),
        0.0,
    )
    .await;

    assert_exact_devices(
        |device| {
            let tensor: Tensor<4, f32> = Tensor::new(device, &tensor_with_singletons);
            tensor.shape();
            tensor
        },
        |device| Tensor::new(device, &tensor_with_singletons),
    )
    .await;
}

#[tokio::test]
async fn cat_stack_and_chunk_match_expected_views() {
    for device in available_devices().await {
        let left: Tensor<2, f32> = sequential_tensor(&device, [2, 3]);
        let right = left.add_scalar(10.0);

        let free_cat = cat([left.clone().to_concrete(), right.clone().to_concrete()], 0);
        let assoc_cat = Tensor::cat([left.clone().to_concrete(), right.clone().to_concrete()], 0);
        assert_approx_tensors(free_cat.clone(), assoc_cat, 1e-6).await;
        let expected_cat: Tensor<2, f32> = Tensor::from_slice(
            &device,
            [4, 3],
            &[
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            ],
        );
        assert_approx_tensors(free_cat, expected_cat, 1e-6).await;

        let free_stack: Tensor<3, f32> =
            stack([left.clone().to_concrete(), right.clone().to_concrete()], 0);
        let assoc_stack: Tensor<3, f32> =
            Tensor::stack([left.clone().to_concrete(), right.clone().to_concrete()], 0);
        assert_approx_tensors(free_stack.clone(), assoc_stack, 1e-6).await;
        let expected_stack: Tensor<3, f32> = Tensor::from_slice(
            &device,
            [2, 2, 3],
            &[
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            ],
        );
        assert_approx_tensors(free_stack, expected_stack, 1e-6).await;

        let input: Tensor<2, f32> = sequential_tensor(&device, [2, 5]);
        let chunks = input.chunk(2, 1);
        assert_eq!(chunks.len(), 2);
        assert_approx_tensors(
            chunks[0].to_concrete(),
            input.slice([0..2, 0..3]).to_concrete(),
            1e-6,
        )
        .await;
        assert_approx_tensors(
            chunks[1].to_concrete(),
            input.slice([0..2, 3..5]).to_concrete(),
            1e-6,
        )
        .await;
    }
}
