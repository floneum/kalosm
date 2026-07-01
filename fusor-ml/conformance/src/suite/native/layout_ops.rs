//! Layout op conformance cases.

use crate::common::{
    broadcast_1d_to_2d, flatten2, flatten3, permute3, repeat2, reshape2, resize2, slice2,
    sliding_window_1d_ncw, transpose2,
};
use fusor::{Device, Tensor};
use fusor_conformance::{
    AssertionCase, AssertionCases, FuzzGenerator, approx_compare, exact_value_compare,
};
use fusor_types::SlidingWindow;
use rand::distr::Uniform;

pub fn shape_and_layout_ops_match_host_reference() -> AssertionCases {
    let mut assertions = AssertionCases::new();

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
    assertions.push(
        fusor_conformance::assert(async |x: Tensor<2, f32>| x.reshape([3, 2]).to_concrete())
            .arg(gen_2x3.clone())
            .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
                Tensor::new(&device, &reshape2(&flatten2(&v), [3, 2]))
            })
            .compare_with(approx_compare::<2, f32>(0.0))
            .runs(3)
            .into_case("layout_ops::shape_and_layout_ops_match_host_reference::reshape"),
    );

    // transpose
    assertions.push(
        fusor_conformance::assert(async |x: Tensor<2, f32>| x.transpose(0, 1).to_concrete())
            .arg(gen_2x3.clone())
            .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
                Tensor::new(&device, &transpose2(&v))
            })
            .compare_with(approx_compare::<2, f32>(0.0))
            .runs(3)
            .into_case("layout_ops::shape_and_layout_ops_match_host_reference::transpose"),
    );

    // permute 3D
    assertions.push(
        fusor_conformance::assert(async |x: Tensor<3, f32>| x.permute([1, 2, 0]).to_concrete())
            .arg(gen_2x3x4.clone())
            .equal_to_resolved_with_device(async |v: Vec<Vec<Vec<f32>>>, device: Device| {
                Tensor::new(&device, &permute3(&v, [1, 2, 0]))
            })
            .compare_with(approx_compare::<3, f32>(0.0))
            .runs(3)
            .into_case("layout_ops::shape_and_layout_ops_match_host_reference::permute3"),
    );

    // slice
    assertions.push(
        fusor_conformance::assert(async |x: Tensor<2, f32>| x.slice([0..2, 1..3]).to_concrete())
            .arg(gen_2x4.clone())
            .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
                Tensor::new(&device, &slice2(&v, 0..2, 1..3))
            })
            .compare_with(approx_compare::<2, f32>(0.0))
            .runs(3)
            .into_case("layout_ops::shape_and_layout_ops_match_host_reference::slice"),
    );

    // broadcast_as
    assertions.push(
        fusor_conformance::assert(async |x: Tensor<1, f32>| x.broadcast_as([2, 3]).to_concrete())
            .arg(gen_1d_3.clone())
            .equal_to_resolved_with_device(async |v: Vec<f32>, device: Device| {
                Tensor::new(&device, &broadcast_1d_to_2d(&v, 2))
            })
            .compare_with(approx_compare::<2, f32>(0.0))
            .runs(3)
            .into_case("layout_ops::shape_and_layout_ops_match_host_reference::broadcast_as"),
    );

    // flatten_all
    assertions.push(
        fusor_conformance::assert(async |x: Tensor<2, f32>| x.flatten_all().to_concrete())
            .arg(gen_2x3.clone())
            .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
                Tensor::new(&device, &flatten2(&v))
            })
            .compare_with(approx_compare::<1, f32>(0.0))
            .runs(3)
            .into_case("layout_ops::shape_and_layout_ops_match_host_reference::flatten_all"),
    );

    // flatten_first_n
    assertions.push(
        fusor_conformance::assert(async |x: Tensor<3, f32>| x.flatten_first_n::<1, 2>())
            .arg(gen_2x3x4.clone())
            .equal_to_resolved_with_device(async |v: Vec<Vec<Vec<f32>>>, device: Device| {
                Tensor::new(&device, &reshape2(&flatten3(&v), [6, 4]))
            })
            .compare_with(approx_compare::<2, f32>(0.0))
            .runs(3)
            .into_case("layout_ops::shape_and_layout_ops_match_host_reference::flatten_first_n"),
    );

    // flatten_last_n
    assertions.push(
        fusor_conformance::assert(async |x: Tensor<3, f32>| x.flatten_last_n::<1, 2>())
            .arg(gen_2x3x4)
            .equal_to_resolved_with_device(async |v: Vec<Vec<Vec<f32>>>, device: Device| {
                Tensor::new(&device, &reshape2(&flatten3(&v), [2, 12]))
            })
            .compare_with(approx_compare::<2, f32>(0.0))
            .runs(3)
            .into_case("layout_ops::shape_and_layout_ops_match_host_reference::flatten_last_n"),
    );

    // repeat
    assertions.push(
        fusor_conformance::assert(async |x: Tensor<2, f32>| x.repeat([2, 3]))
            .arg(gen_2x2.clone())
            .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
                Tensor::new(&device, &repeat2(&v, [2, 3]))
            })
            .compare_with(approx_compare::<2, f32>(0.0))
            .runs(3)
            .into_case("layout_ops::shape_and_layout_ops_match_host_reference::repeat"),
    );

    assertions.push(
        fusor_conformance::assert(async |x: Tensor<2, f32>| x.repeat([0, 3]))
            .arg(gen_2x2)
            .equal_to_resolved_with_device(async |_v: Vec<Vec<f32>>, device: Device| {
                Tensor::<2, f32>::zeros(&device, [0, 6])
            })
            .compare_with(approx_compare::<2, f32>(0.0))
            .runs(1)
            .into_case("layout_ops::shape_and_layout_ops_match_host_reference::repeat_empty"),
    );

    // unsqueeze
    assertions.push(
        fusor_conformance::assert(async |x: Tensor<2, f32>| x.unsqueeze::<3>(1).to_concrete())
            .arg(gen_2x3.clone())
            .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
                let expected: Vec<Vec<Vec<f32>>> = v.into_iter().map(|row| vec![row]).collect();
                Tensor::new(&device, &expected)
            })
            .compare_with(approx_compare::<3, f32>(0.0))
            .runs(3)
            .into_case("layout_ops::shape_and_layout_ops_match_host_reference::unsqueeze"),
    );

    // squeeze
    assertions.push(
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
            .into_case("layout_ops::shape_and_layout_ops_match_host_reference::squeeze"),
    );

    let gen_2x1x3x1 = FuzzGenerator::<4, f32>::new([2, 1, 3, 1])
        .with_seed(506)
        .with_distribution(Uniform::new(-5.0, 5.0).unwrap());
    let _ = gen_2x1x3x1;

    // sliding_window_view
    let gen_1x2x5 = FuzzGenerator::<3, f32>::new([1, 2, 5])
        .with_seed(507)
        .with_distribution(Uniform::new(-5.0, 5.0).unwrap());
    assertions.push(
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
        .into_case("layout_ops::shape_and_layout_ops_match_host_reference::sliding_window_view"),
    );

    // resize
    assertions.push(
        fusor_conformance::assert(async |x: Tensor<2, f32>| x.to_concrete().resize([3, 4]))
            .arg(gen_2x3)
            .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
                Tensor::new(&device, &resize2(&v, [3, 4]))
            })
            .compare_with(approx_compare::<2, f32>(0.0))
            .runs(3)
            .into_case("layout_ops::shape_and_layout_ops_match_host_reference::resize"),
    );

    assertions
}

pub fn tensor_i_op_matches_expected_views() -> AssertionCases {
    let mut assertions = AssertionCases::new();

    // 2D row select via PyTorch-style `i((row, ..))`
    assertions.push(
        fusor_conformance::assert(async |matrix: Tensor<2, f32>| matrix.i((1, ..)).to_concrete())
            .arg(|device: &Device| Tensor::new(device, &[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]]))
            .equal_to(async |matrix: Tensor<2, f32>| Tensor::new(&matrix.device(), &[3.0f32, 4.0]))
            .compare_with(approx_compare::<1, f32>(0.0))
            .runs(1)
            .into_case("layout_ops::tensor_i_op_matches_expected_views::row"),
    );

    // 2D column select via `i((.., col))`
    assertions.push(
        fusor_conformance::assert(async |matrix: Tensor<2, f32>| matrix.i((.., 0)).to_concrete())
            .arg(|device: &Device| Tensor::new(device, &[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]]))
            .equal_to(async |matrix: Tensor<2, f32>| {
                Tensor::new(&matrix.device(), &[1.0f32, 3.0, 5.0])
            })
            .compare_with(approx_compare::<1, f32>(0.0))
            .runs(1)
            .into_case("layout_ops::tensor_i_op_matches_expected_views::column"),
    );

    // 3D index along middle dim
    assertions.push(
        fusor_conformance::assert(async |cube: Tensor<3, f32>| cube.i((.., 0, ..)).to_concrete())
            .arg(|device: &Device| {
                Tensor::new(
                    device,
                    &[[[1.0f32, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                )
            })
            .equal_to(async |cube: Tensor<3, f32>| {
                Tensor::new(&cube.device(), &[[1.0f32, 2.0], [5.0, 6.0]])
            })
            .compare_with(approx_compare::<2, f32>(0.0))
            .runs(1)
            .into_case("layout_ops::tensor_i_op_matches_expected_views::middle_dim"),
    );

    // 4D outer select
    assertions.push(
        fusor_conformance::assert(async |tesseract: Tensor<4, f32>| {
            tesseract.i((0, .., .., ..)).to_concrete()
        })
        .arg(|device: &Device| {
            Tensor::new(
                device,
                &[[[[1.0f32, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]],
            )
        })
        .equal_to(async |tesseract: Tensor<4, f32>| {
            Tensor::new(
                &tesseract.device(),
                &[[[1.0f32, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            )
        })
        .compare_with(approx_compare::<3, f32>(0.0))
        .runs(1)
        .into_case("layout_ops::tensor_i_op_matches_expected_views::outer"),
    );

    assertions
}

pub fn broadcast_as_non_contiguous_input_matches_expected_view() -> AssertionCase {
    // Build a 2x3 tensor, slice the middle column out, and broadcast along
    // a new last axis. The slice gives the broadcast a non-contiguous source.
    fusor_conformance::assert(async |source: Tensor<2, f32>| {
        source
            .slice([0..2, 1..3])
            .to_concrete()
            .unsqueeze::<3>(2)
            .broadcast_as([2, 2, 4])
            .to_concrete()
    })
    .arg(|device: &Device| Tensor::new(device, &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    .equal_to(async |source: Tensor<2, f32>| {
        let expected_rows: Vec<Vec<Vec<f32>>> = vec![
            vec![vec![2.0; 4], vec![3.0; 4]],
            vec![vec![5.0; 4], vec![6.0; 4]],
        ];
        Tensor::new(&source.device(), &expected_rows)
    })
    .compare_with(approx_compare::<3, f32>(0.0))
    .runs(1)
    .into_case("layout_ops::broadcast_as_non_contiguous_input_matches_expected_view")
}

pub fn sliding_window_then_transpose_then_reshape_matches_expected() -> AssertionCase {
    use fusor_types::SlidingWindow;
    // Conv1d-style layout regression that combines sliding_window_view, transpose,
    // and reshape. Replaces the deleted `cpu/tests/index.rs::test_sliding_window_transpose_reshape`.
    let input_data: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let expected_rows = [
        [0.0f32, 1.0, 2.0, 5.0, 6.0, 7.0],
        [1.0, 2.0, 3.0, 6.0, 7.0, 8.0],
        [2.0, 3.0, 4.0, 7.0, 8.0, 9.0],
    ];

    fusor_conformance::assert(async |input: Tensor<3, f32>| {
        let windows = input.sliding_window_view::<1, 4>([SlidingWindow::new(2, 3, 1)]);
        let transposed = windows.transpose(1, 2);
        transposed.reshape([3, 6]).to_concrete()
    })
    .arg(move |device: &Device| Tensor::from_slice(device, [1, 2, 5], &input_data))
    .equal_to(
        move |input: Tensor<3, f32>| async move { Tensor::new(&input.device(), &expected_rows) },
    )
    .compare_with(approx_compare::<2, f32>(0.0))
    .runs(1)
    .into_case("layout_ops::sliding_window_then_transpose_then_reshape_matches_expected")
}

pub fn transpose_reshape_consumed_by_elementwise_matches_expected() -> AssertionCase {
    let shape = [1usize, 32, 2, 128];
    let data = (0..shape.iter().product::<usize>())
        .map(|i| i as f32 * 0.001)
        .collect::<Vec<_>>();
    let mut expected = Vec::with_capacity(data.len());
    for batch in 0..shape[0] {
        for seq in 0..shape[2] {
            for head in 0..shape[1] {
                for dim in 0..shape[3] {
                    let index = (((batch * shape[1] + head) * shape[2] + seq) * shape[3]) + dim;
                    expected.push(data[index] + 0.25);
                }
            }
        }
    }

    fusor_conformance::assert(async |input: Tensor<4, f32>| {
        let produced = input + 0.25;
        let transposed = produced.transpose(1, 2);
        let reshaped = transposed.reshape([1, 2, 32 * 128]);
        (reshaped + 0.0).to_concrete()
    })
    .arg(move |device: &Device| Tensor::from_slice(device, shape, &data))
    .equal_to(move |input: Tensor<4, f32>| {
        let expected = expected.clone();
        async move { Tensor::from_slice(&input.device(), [1, 2, 32 * 128], &expected) }
    })
    .compare_with(approx_compare::<3, f32>(0.0))
    .runs(1)
    .into_case("layout_ops::transpose_reshape_consumed_by_elementwise_matches_expected")
}

pub fn sliding_window_with_cat_padding_matches_expected() -> AssertionCase {
    use fusor_types::SlidingWindow;
    // Conv1d-style padding regression: pad an input with `cat`, then sliding-window.
    // Replaces the deleted `cpu/tests/index.rs::test_sliding_window_with_cat_padding`.
    let input_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let expected_rows = [
        [0.0f32, 1.0, 2.0, 0.0, 4.0, 5.0],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [2.0, 3.0, 0.0, 5.0, 6.0, 0.0],
    ];

    fusor_conformance::assert(async |input: Tensor<3, f32>| {
        let device = input.device();
        let pad_left: Tensor<3, f32> = Tensor::<3, f32>::zeros(&device, [1, 2, 1]);
        let pad_right: Tensor<3, f32> = Tensor::<3, f32>::zeros(&device, [1, 2, 1]);
        let padded = Tensor::<3, f32>::cat([pad_left, input, pad_right], 2);

        let windows = padded.sliding_window_view::<1, 4>([SlidingWindow::new(2, 3, 1)]);
        let transposed = windows.transpose(1, 2);
        transposed.reshape([3, 6]).to_concrete()
    })
    .arg(move |device: &Device| Tensor::from_slice(device, [1, 2, 3], &input_data))
    .equal_to(
        move |input: Tensor<3, f32>| async move { Tensor::new(&input.device(), &expected_rows) },
    )
    .compare_with(approx_compare::<2, f32>(0.0))
    .runs(1)
    .into_case("layout_ops::sliding_window_with_cat_padding_matches_expected")
}

pub fn restride_and_restride_layout_match_expected_views() -> AssertionCases {
    use fusor_types::{Layout, StrideSpec};
    let gen_4x6 = FuzzGenerator::<2, f32>::new([4, 6])
        .with_seed(520)
        .with_distribution(Uniform::new(-5.0, 5.0).unwrap());
    let mut assertions = AssertionCases::new();

    // restride with a stride multiplier — pick every other column.
    // (`permute`, `slice`, and `narrow` already cover the cases that don't use
    // `dim_with`; this exercises the stride-multiplier path that no other op
    // exposes.)
    assertions.push(
        fusor_conformance::assert(async |x: Tensor<2, f32>| {
            let [rows, _] = x.shape();
            x.restride([StrideSpec::dim(0, rows), StrideSpec::dim_with(1, 3, 2)])
                .to_concrete()
        })
        .arg(gen_4x6.clone())
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            let stepped: Vec<Vec<f32>> = v
                .iter()
                .map(|row| (0..3).map(|i| row[i * 2]).collect())
                .collect();
            Tensor::new(&device, &stepped)
        })
        .compare_with(approx_compare::<2, f32>(0.0))
        .runs(3)
        .into_case("layout_ops::restride_and_restride_layout_match_expected_views::restride"),
    );

    // restride_layout takes a raw absolute Layout, so by contract its input
    // must already be contiguous (the op asserts this). FuzzGenerator yields
    // non-contig views on alternating runs, so build the input from a Device
    // closure that returns a fresh contiguous tensor.
    fn input_data() -> Vec<f32> {
        (0..24).map(|i| ((i as f32) - 12.0) * 0.31).collect()
    }
    assertions.push(
        fusor_conformance::assert(async |x: Tensor<2, f32>| {
            let new_layout = Layout::from_parts(0, Box::from([4usize, 3]), Box::from([6usize, 2]));
            x.restride_layout::<2>(new_layout).to_concrete()
        })
        .arg(|device: &Device| Tensor::from_slice(device, [4, 6], &input_data()))
        .equal_to(async |_x: Tensor<2, f32>| {
            let device = _x.device();
            let data = input_data();
            let stepped: Vec<Vec<f32>> = (0..4)
                .map(|r| (0..3).map(|c| data[r * 6 + c * 2]).collect())
                .collect();
            Tensor::new(&device, &stepped)
        })
        .compare_with(approx_compare::<2, f32>(0.0))
        .runs(1)
        .into_case(
            "layout_ops::restride_and_restride_layout_match_expected_views::restride_layout",
        ),
    );

    assertions
}

pub fn cat_stack_and_chunk_match_expected_views() -> AssertionCases {
    // chunk: verify chunk pieces match slices
    let gen_2x5 = FuzzGenerator::<2, f32>::new([2, 5])
        .with_seed(511)
        .with_distribution(Uniform::new(-5.0, 5.0).unwrap());
    let mut assertions = AssertionCases::new();

    assertions.push(
        fusor_conformance::assert(async |input: Tensor<2, f32>| input.chunk(2, 1).len())
            .arg(gen_2x5.clone())
            .equal_to(async |_input: Tensor<2, f32>| 2usize)
            .compare_with(exact_value_compare())
            .runs(1)
            .into_case("layout_ops::cat_stack_and_chunk_match_expected_views::chunk_len"),
    );

    assertions.push(
        fusor_conformance::assert(async |input: Tensor<2, f32>| input.chunk(2, 1)[0].to_concrete())
            .arg(gen_2x5.clone())
            .equal_to(async |input: Tensor<2, f32>| input.slice([0..2, 0..3]).to_concrete())
            .compare_with(approx_compare::<2, f32>(1e-6))
            .runs(1)
            .into_case("layout_ops::cat_stack_and_chunk_match_expected_views::chunk0"),
    );

    assertions.push(
        fusor_conformance::assert(async |input: Tensor<2, f32>| input.chunk(2, 1)[1].to_concrete())
            .arg(gen_2x5)
            .equal_to(async |input: Tensor<2, f32>| input.slice([0..2, 3..5]).to_concrete())
            .compare_with(approx_compare::<2, f32>(1e-6))
            .runs(1)
            .into_case("layout_ops::cat_stack_and_chunk_match_expected_views::chunk1"),
    );

    assertions
}
