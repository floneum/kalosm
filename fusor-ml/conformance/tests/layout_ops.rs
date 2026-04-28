mod common;

use common::{
    assert_approx_tensors, broadcast_1d_to_2d, flatten2, flatten3, permute3, repeat2, reshape2,
    resize2, slice2, sliding_window_1d_ncw, transpose2,
};
use fusor::{Device, Tensor};
use fusor_conformance::{FuzzGenerator, GenerateFromDevice, approx_compare, available_devices};
use fusor_types::SlidingWindow;
use rand::distr::Uniform;

#[tokio::test]
async fn shape_and_layout_ops_match_host_reference() {
    let gen_2x3 = FuzzGenerator::<2, f32>::new([2, 3])
        .with_seed(500)
        .with_distribution(Uniform::new(-5.0, 5.0).unwrap());
    let gen_2x4 = FuzzGenerator::<2, f32>::new([2, 4])
        .with_seed(501)
        .with_distribution(Uniform::new(-5.0, 5.0).unwrap());
    let gen_2x3x4 = FuzzGenerator::<3, f32>::new([2, 3, 4])
        .with_seed(502)
        .with_distribution(Uniform::new(-5.0, 5.0).unwrap());
    let gen_1d_3 = FuzzGenerator::<1, f32>::new([3])
        .with_seed(503)
        .with_distribution(Uniform::new(-5.0, 5.0).unwrap());
    let gen_2x1x3 = FuzzGenerator::<3, f32>::new([2, 1, 3])
        .with_seed(504)
        .with_distribution(Uniform::new(-5.0, 5.0).unwrap());
    let gen_2x2 = FuzzGenerator::<2, f32>::new([2, 2])
        .with_seed(505)
        .with_distribution(Uniform::new(-5.0, 5.0).unwrap());

    // reshape
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.reshape([3, 2]).to_concrete())
        .arg(gen_2x3.clone())
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            Tensor::new(&device, &reshape2(&flatten2(&v), [3, 2]))
        })
        .compare_with(approx_compare::<2, f32>(0.0))
        .runs(3)
        .await
        .unwrap();

    // transpose
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.transpose(0, 1).to_concrete())
        .arg(gen_2x3.clone())
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            Tensor::new(&device, &transpose2(&v))
        })
        .compare_with(approx_compare::<2, f32>(0.0))
        .runs(3)
        .await
        .unwrap();

    // permute 3D
    fusor_conformance::assert(async |x: Tensor<3, f32>| x.permute([1, 2, 0]).to_concrete())
        .arg(gen_2x3x4.clone())
        .equal_to_resolved_with_device(async |v: Vec<Vec<Vec<f32>>>, device: Device| {
            Tensor::new(&device, &permute3(&v, [1, 2, 0]))
        })
        .compare_with(approx_compare::<3, f32>(0.0))
        .runs(3)
        .await
        .unwrap();

    // slice
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.slice([0..2, 1..3]).to_concrete())
        .arg(gen_2x4.clone())
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            Tensor::new(&device, &slice2(&v, 0..2, 1..3))
        })
        .compare_with(approx_compare::<2, f32>(0.0))
        .runs(3)
        .await
        .unwrap();

    // broadcast_as
    fusor_conformance::assert(async |x: Tensor<1, f32>| x.broadcast_as([2, 3]).to_concrete())
        .arg(gen_1d_3.clone())
        .equal_to_resolved_with_device(async |v: Vec<f32>, device: Device| {
            Tensor::new(&device, &broadcast_1d_to_2d(&v, 2))
        })
        .compare_with(approx_compare::<2, f32>(0.0))
        .runs(3)
        .await
        .unwrap();

    // flatten_all
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.flatten_all().to_concrete())
        .arg(gen_2x3.clone())
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            Tensor::new(&device, &flatten2(&v))
        })
        .compare_with(approx_compare::<1, f32>(0.0))
        .runs(3)
        .await
        .unwrap();

    // flatten_first_n
    fusor_conformance::assert(async |x: Tensor<3, f32>| x.flatten_first_n::<1, 2>())
        .arg(gen_2x3x4.clone())
        .equal_to_resolved_with_device(async |v: Vec<Vec<Vec<f32>>>, device: Device| {
            Tensor::new(&device, &reshape2(&flatten3(&v), [6, 4]))
        })
        .compare_with(approx_compare::<2, f32>(0.0))
        .runs(3)
        .await
        .unwrap();

    // flatten_last_n
    fusor_conformance::assert(async |x: Tensor<3, f32>| x.flatten_last_n::<1, 2>())
        .arg(gen_2x3x4)
        .equal_to_resolved_with_device(async |v: Vec<Vec<Vec<f32>>>, device: Device| {
            Tensor::new(&device, &reshape2(&flatten3(&v), [2, 12]))
        })
        .compare_with(approx_compare::<2, f32>(0.0))
        .runs(3)
        .await
        .unwrap();

    // repeat
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.repeat([2, 3]))
        .arg(gen_2x2.clone())
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            Tensor::new(&device, &repeat2(&v, [2, 3]))
        })
        .compare_with(approx_compare::<2, f32>(0.0))
        .runs(3)
        .await
        .unwrap();

    fusor_conformance::assert(async |x: Tensor<2, f32>| x.repeat([0, 3]))
        .arg(gen_2x2)
        .equal_to_resolved_with_device(async |_v: Vec<Vec<f32>>, device: Device| {
            Tensor::<2, f32>::zeros(&device, [0, 6])
        })
        .compare_with(approx_compare::<2, f32>(0.0))
        .runs(1)
        .await
        .unwrap();

    // unsqueeze
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.unsqueeze::<3>(1).to_concrete())
        .arg(gen_2x3.clone())
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            let expected: Vec<Vec<Vec<f32>>> = v.into_iter().map(|row| vec![row]).collect();
            Tensor::new(&device, &expected)
        })
        .compare_with(approx_compare::<3, f32>(0.0))
        .runs(3)
        .await
        .unwrap();

    // squeeze
    fusor_conformance::assert(async |x: Tensor<3, f32>| x.squeeze::<2>(1).to_concrete())
        .arg(gen_2x1x3)
        .equal_to_resolved_with_device(async |v: Vec<Vec<Vec<f32>>>, device: Device| {
            let squeezed: Vec<Vec<f32>> = v
                .into_iter()
                .map(|m| m.into_iter().next().unwrap())
                .collect();
            Tensor::new(&device, &squeezed)
        })
        .compare_with(approx_compare::<2, f32>(0.0))
        .runs(3)
        .await
        .unwrap();

    let gen_2x1x3x1 = FuzzGenerator::<4, f32>::new([2, 1, 3, 1])
        .with_seed(506)
        .with_distribution(Uniform::new(-5.0, 5.0).unwrap());
    let _ = gen_2x1x3x1;

    // sliding_window_view
    let gen_1x2x5 = FuzzGenerator::<3, f32>::new([1, 2, 5])
        .with_seed(507)
        .with_distribution(Uniform::new(-5.0, 5.0).unwrap());
    fusor_conformance::assert(async |x: Tensor<3, f32>| {
        x.sliding_window_view::<1, 4>([SlidingWindow::new(2, 3, 2)])
            .to_concrete()
    })
    .arg(gen_1x2x5)
    .equal_to_resolved_with_device(async |v: Vec<Vec<Vec<f32>>>, device: Device| {
        Tensor::new(&device, &sliding_window_1d_ncw(&v, 3, 2))
    })
    .compare_with(approx_compare::<4, f32>(0.0))
    .runs(3)
    .await
    .unwrap();

    // resize
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.to_concrete().resize([3, 4]))
        .arg(gen_2x3)
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            Tensor::new(&device, &resize2(&v, [3, 4]))
        })
        .compare_with(approx_compare::<2, f32>(0.0))
        .runs(3)
        .await
        .unwrap();
}

#[tokio::test]
async fn tensor_i_op_matches_expected_views() {
    // 2D row select via PyTorch-style `i((row, ..))`
    for device in available_devices().await {
        let matrix: Tensor<2, f32> = Tensor::new(&device, &[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        let row1 = matrix.i((1, ..));
        let expected = Tensor::new(&device, &[3.0f32, 4.0]);
        assert_approx_tensors(row1, expected, 0.0).await;

        // 2D column select via `i((.., col))`
        let col0 = matrix.i((.., 0));
        let expected = Tensor::new(&device, &[1.0f32, 3.0, 5.0]);
        assert_approx_tensors(col0, expected, 0.0).await;
    }

    // 3D index along middle dim
    for device in available_devices().await {
        let cube: Tensor<3, f32> = Tensor::new(
            &device,
            &[[[1.0f32, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        );
        let mid0 = cube.i((.., 0, ..));
        let expected = Tensor::new(&device, &[[1.0f32, 2.0], [5.0, 6.0]]);
        assert_approx_tensors(mid0, expected, 0.0).await;
    }

    // 4D outer select
    for device in available_devices().await {
        let tesseract: Tensor<4, f32> = Tensor::new(
            &device,
            &[[[[1.0f32, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]],
        );
        let outer = tesseract.i((0, .., .., ..));
        let expected = Tensor::new(
            &device,
            &[[[1.0f32, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        );
        assert_approx_tensors(outer, expected, 0.0).await;
    }
}

#[tokio::test]
async fn broadcast_as_non_contiguous_input_matches_expected_view() {
    for device in available_devices().await {
        // Build a 2x3 tensor, slice the middle column out, and broadcast along
        // a new last axis. The slice gives the broadcast a non-contiguous source.
        let source: Tensor<2, f32> = Tensor::new(&device, &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let sliced = source.slice([0..2, 1..3]).to_concrete();
        let broadcast = sliced
            .unsqueeze::<3>(2)
            .broadcast_as([2, 2, 4])
            .to_concrete();

        let expected_rows: Vec<Vec<Vec<f32>>> = vec![
            vec![vec![2.0; 4], vec![3.0; 4]],
            vec![vec![5.0; 4], vec![6.0; 4]],
        ];
        let expected = Tensor::new(&device, &expected_rows);
        assert_approx_tensors(broadcast, expected, 0.0).await;
    }
}

#[tokio::test]
async fn sliding_window_then_transpose_then_reshape_matches_expected() {
    use fusor_types::SlidingWindow;
    // Conv1d-style layout regression that combines sliding_window_view, transpose,
    // and reshape. Replaces the deleted `cpu/tests/index.rs::test_sliding_window_transpose_reshape`.
    let input_data: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let expected_rows = [
        [0.0f32, 1.0, 2.0, 5.0, 6.0, 7.0],
        [1.0, 2.0, 3.0, 6.0, 7.0, 8.0],
        [2.0, 3.0, 4.0, 7.0, 8.0, 9.0],
    ];

    for device in available_devices().await {
        let input: Tensor<3, f32> = Tensor::from_slice(&device, [1, 2, 5], &input_data);
        let windows = input.sliding_window_view::<1, 4>([SlidingWindow::new(2, 3, 1)]);
        let transposed = windows.transpose(1, 2);
        let reshaped: Tensor<2, f32> = transposed.reshape([3, 6]).to_concrete();

        let expected = Tensor::new(&device, &expected_rows);
        assert_approx_tensors(reshaped, expected, 0.0).await;
    }
}

#[tokio::test]
async fn transpose_then_reshape_preserves_logical_order() {
    let input_data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let mut expected = Vec::new();
    for b in 0..2 {
        for col in 0..4 {
            for row in 0..3 {
                expected.push(input_data[(b * 3 + row) * 4 + col]);
            }
        }
    }

    for device in available_devices().await {
        let input: Tensor<3, f32> = Tensor::from_slice(&device, [2, 3, 4], &input_data);
        let output = input
            .transpose(1, 2)
            .reshape([2, 12])
            .to_concrete()
            .reshape([24])
            .to_concrete();
        let expected = Tensor::from_slice(&device, [24], &expected);
        assert_approx_tensors(output, expected, 0.0).await;
    }
}

#[tokio::test]
async fn broadcast_then_reshape_preserves_repeated_logical_values() {
    let input_data: Vec<f32> = (0..24).map(|i| i as f32 * 0.25 - 3.0).collect();
    let repeats = 5;
    let mut expected = Vec::with_capacity(repeats * input_data.len());
    for _ in 0..repeats {
        expected.extend(input_data.iter().copied());
    }

    for device in available_devices().await {
        let input: Tensor<4, f32> = Tensor::from_slice(&device, [1, 2, 3, 4], &input_data);
        let repeated = input
            .reshape([1, 1, 2, 3, 4])
            .broadcast_as([1, repeats, 2, 3, 4])
            .to_concrete()
            .reshape([repeats, 2, 3, 4])
            .to_concrete();
        let expected = Tensor::from_slice(&device, [repeats, 2, 3, 4], &expected);
        assert_approx_tensors(repeated, expected, 0.0).await;
    }
}

#[tokio::test]
async fn sliding_window_with_cat_padding_matches_expected() {
    use fusor_types::SlidingWindow;
    // Conv1d-style padding regression: pad an input with `cat`, then sliding-window.
    // Replaces the deleted `cpu/tests/index.rs::test_sliding_window_with_cat_padding`.
    let input_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let expected_rows = [
        [0.0f32, 1.0, 2.0, 0.0, 4.0, 5.0],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [2.0, 3.0, 0.0, 5.0, 6.0, 0.0],
    ];

    for device in available_devices().await {
        let input: Tensor<3, f32> = Tensor::from_slice(&device, [1, 2, 3], &input_data);
        let pad_left: Tensor<3, f32> = Tensor::<3, f32>::zeros(&device, [1, 2, 1]);
        let pad_right: Tensor<3, f32> = Tensor::<3, f32>::zeros(&device, [1, 2, 1]);
        let padded = Tensor::<3, f32>::cat([pad_left, input, pad_right], 2);

        let windows = padded.sliding_window_view::<1, 4>([SlidingWindow::new(2, 3, 1)]);
        let transposed = windows.transpose(1, 2);
        let reshaped: Tensor<2, _> = transposed.reshape([3, 6]).to_concrete();

        let expected = Tensor::new(&device, &expected_rows);
        assert_approx_tensors(reshaped, expected, 0.0).await;
    }
}

#[tokio::test]
async fn cat_stack_and_chunk_match_expected_views() {
    // chunk: verify chunk pieces match slices
    for device in available_devices().await {
        let gen_2x5 = FuzzGenerator::<2, f32>::new([2, 5])
            .with_seed(511)
            .with_distribution(Uniform::new(-5.0, 5.0).unwrap());
        let input = gen_2x5.clone().generate(&device, 0);
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
