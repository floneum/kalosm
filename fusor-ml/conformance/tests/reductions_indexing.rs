mod common;

use common::{index_select1, index_select2, keepdim2, mean_axis2, reduce_axis2, var_axis2};
use fusor::{Device, Tensor, arange};
use fusor_conformance::{FuzzGenerator, approx_compare, available_devices, sequential_tensor};
use half::f16;
use rand::distr::Uniform;

#[tokio::test]
async fn reductions_match_host_reference() {
    const SHAPE: [usize; 2] = [3, 4];
    let fuzz = FuzzGenerator::<2, f32>::new(SHAPE)
        .with_seed(200)
        .with_distribution(Uniform::new(-5.0, 5.0).unwrap());

    // sum along axis 1
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.sum::<1>(1))
        .arg(fuzz.clone())
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            Tensor::from_slice(&device, [SHAPE[0]], &reduce_axis2(&v, 1, 0.0, |a, b| a + b))
        })
        .compare_with(approx_compare::<1, f32>(1e-4))
        .runs(3)
        .await
        .unwrap();

    // max along axis 0
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.max::<1>(0))
        .arg(fuzz.clone())
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            Tensor::from_slice(
                &device,
                [SHAPE[1]],
                &reduce_axis2(&v, 0, f32::NEG_INFINITY, f32::max),
            )
        })
        .compare_with(approx_compare::<1, f32>(1e-5))
        .runs(3)
        .await
        .unwrap();

    // min along axis 0
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.min::<1>(0))
        .arg(fuzz.clone())
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            Tensor::from_slice(
                &device,
                [SHAPE[1]],
                &reduce_axis2(&v, 0, f32::INFINITY, f32::min),
            )
        })
        .compare_with(approx_compare::<1, f32>(1e-5))
        .runs(3)
        .await
        .unwrap();

    // product along axis 1 (bounded to avoid overflow)
    let fuzz_small = FuzzGenerator::<2, f32>::new(SHAPE)
        .with_seed(201)
        .with_distribution(Uniform::new(0.5, 2.0).unwrap());
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.product::<1>(1))
        .arg(fuzz_small)
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            Tensor::from_slice(&device, [SHAPE[0]], &reduce_axis2(&v, 1, 1.0, |a, b| a * b))
        })
        .compare_with(approx_compare::<1, f32>(1e-3))
        .runs(3)
        .await
        .unwrap();

    // sum_keepdim along axis 1
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.sum_keepdim::<1>(1))
        .arg(fuzz.clone())
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            Tensor::new(
                &device,
                &keepdim2(&reduce_axis2(&v, 1, 0.0, |a, b| a + b), 1),
            )
        })
        .compare_with(approx_compare::<2, f32>(1e-4))
        .runs(3)
        .await
        .unwrap();

    // max_keepdim along axis 0
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.max_keepdim::<1>(0))
        .arg(fuzz.clone())
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            Tensor::new(
                &device,
                &keepdim2(&reduce_axis2(&v, 0, f32::NEG_INFINITY, f32::max), 0),
            )
        })
        .compare_with(approx_compare::<2, f32>(1e-5))
        .runs(3)
        .await
        .unwrap();

    // min_keepdim along axis 0
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.min_keepdim::<1>(0))
        .arg(fuzz.clone())
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            Tensor::new(
                &device,
                &keepdim2(&reduce_axis2(&v, 0, f32::INFINITY, f32::min), 0),
            )
        })
        .compare_with(approx_compare::<2, f32>(1e-5))
        .runs(3)
        .await
        .unwrap();

    // product_keepdim along axis 1
    let fuzz_small2 = FuzzGenerator::<2, f32>::new(SHAPE)
        .with_seed(202)
        .with_distribution(Uniform::new(0.5, 2.0).unwrap());
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.product_keepdim::<1>(1))
        .arg(fuzz_small2)
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            Tensor::new(
                &device,
                &keepdim2(&reduce_axis2(&v, 1, 1.0, |a, b| a * b), 1),
            )
        })
        .compare_with(approx_compare::<2, f32>(1e-3))
        .runs(3)
        .await
        .unwrap();

    // mean along axis 1
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.mean::<1>(1))
        .arg(fuzz.clone())
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            Tensor::from_slice(&device, [SHAPE[0]], &mean_axis2(&v, 1))
        })
        .compare_with(approx_compare::<1, f32>(1e-4))
        .runs(3)
        .await
        .unwrap();

    // mean_keepdim along axis 1
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.mean_keepdim::<1>(1))
        .arg(fuzz.clone())
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            Tensor::new(&device, &keepdim2(&mean_axis2(&v, 1), 1))
        })
        .compare_with(approx_compare::<2, f32>(1e-4))
        .runs(3)
        .await
        .unwrap();

    // var along axis 1
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.var::<1>(1))
        .arg(fuzz.clone())
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            Tensor::from_slice(&device, [SHAPE[0]], &var_axis2(&v, 1))
        })
        .compare_with(approx_compare::<1, f32>(1e-3))
        .runs(3)
        .await
        .unwrap();

    // var_keepdim along axis 1
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.var_keepdim::<1>(1))
        .arg(fuzz)
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            Tensor::new(&device, &keepdim2(&var_axis2(&v, 1), 1))
        })
        .compare_with(approx_compare::<2, f32>(1e-3))
        .runs(3)
        .await
        .unwrap();
}

#[tokio::test]
async fn indexing_cast_and_rank_specific_indexing_match_reference() {
    // index_select, slice_assign, and cast use fuzzed data
    const SHAPE: [usize; 2] = [4, 4];
    let fuzz = FuzzGenerator::<2, f32>::new(SHAPE)
        .with_seed(210)
        .with_distribution(Uniform::new(-5.0, 5.0).unwrap());

    static IDX_DIM1: &[u32] = &[3, 1, 0];

    // index_select dim=1
    fusor_conformance::assert(async |x: Tensor<2, f32>| {
        let indices = Tensor::from_slice(&x.device(), [IDX_DIM1.len()], IDX_DIM1);
        x.index_select(1, &indices)
    })
    .arg(fuzz.clone())
    .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
        Tensor::new(&device, &index_select2(&v, 1, IDX_DIM1))
    })
    .compare_with(approx_compare::<2, f32>(1e-6))
    .runs(3)
    .await
    .unwrap();

    // slice_assign
    let fuzz_patch = FuzzGenerator::<2, f32>::new([2, 2])
        .with_seed(211)
        .with_distribution(Uniform::new(90.0, 110.0).unwrap());
    fusor_conformance::assert(async |x: Tensor<2, f32>, patch: Tensor<2, f32>| {
        x.slice_assign([1..3, 1..3], &patch)
    })
    .arg(fuzz.clone())
    .arg(fuzz_patch)
    .equal_to_resolved_with_device(
        async |v: Vec<Vec<f32>>, patch: Vec<Vec<f32>>, device: Device| {
            let out = common::slice_assign2(&v, 1..3, 1..3, &patch);
            Tensor::new(&device, &out)
        },
    )
    .compare_with(approx_compare::<2, f32>(1e-6))
    .runs(3)
    .await
    .unwrap();

    // cast f32 -> f16 -> f32
    fusor_conformance::assert(async |x: Tensor<2, f32>| {
        x.cast::<f16>().cast::<f32>().to_concrete()
    })
    .arg(fuzz)
    .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
        let out: Vec<Vec<f32>> = v
            .iter()
            .map(|row| row.iter().map(|&x| f16::from_f32(x).to_f32()).collect())
            .collect();
        Tensor::new(&device, &out)
    })
    .compare_with(approx_compare::<2, f32>(1e-6))
    .runs(3)
    .await
    .unwrap();

    // .i() rank-specific indexing: compare against slice+squeeze
    for device in available_devices().await {
        let matrix: Tensor<2, f32> = sequential_tensor(&device, [3, 4]);
        let indexed_2d = matrix.i((1, 1..4));
        let sliced_2d = matrix.slice([1..2, 1..4]).squeeze::<1>(0).to_concrete();
        common::assert_approx_tensors(indexed_2d, sliced_2d, 1e-6).await;

        let tensor3: Tensor<3, f32> = sequential_tensor(&device, [2, 3, 4]);
        let indexed_3d = tensor3.i((0, 1..3, 1..4));
        let sliced_3d = tensor3
            .slice([0..1, 1..3, 1..4])
            .squeeze::<2>(0)
            .to_concrete();
        common::assert_approx_tensors(indexed_3d, sliced_3d, 1e-6).await;

        let tensor4: Tensor<4, f32> = arange(&device, 0.0f32, 48.0)
            .reshape([2, 2, 3, 4])
            .to_concrete();
        let indexed_4d = tensor4.i((1, 0..2, 1..3, 1..4));
        let sliced_4d = tensor4
            .slice([1..2, 0..2, 1..3, 1..4])
            .squeeze::<3>(0)
            .to_concrete();
        common::assert_approx_tensors(indexed_4d, sliced_4d, 1e-6).await;
    }
}

#[tokio::test]
async fn full_tensor_reductions_fuzzed() {
    // 2D reductions with fuzzed data + non-contiguous layouts
    const SHAPE: [usize; 2] = [8, 16];
    let fuzz = FuzzGenerator::<2, f32>::new(SHAPE)
        .with_seed(42)
        .with_distribution(Uniform::new(-5.0, 5.0).unwrap());

    // sum along axis 1
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.sum::<1>(1))
        .arg(fuzz.clone())
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            let out: Vec<f32> = v.iter().map(|row| row.iter().sum()).collect();
            Tensor::from_slice(&device, [SHAPE[0]], &out)
        })
        .compare_with(approx_compare::<1, f32>(1e-3))
        .runs(3)
        .await
        .unwrap();

    // sum along axis 0
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.sum::<1>(0))
        .arg(fuzz.clone())
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            let out = reduce_axis2(&v, 0, 0.0, |a, b| a + b);
            Tensor::from_slice(&device, [SHAPE[1]], &out)
        })
        .compare_with(approx_compare::<1, f32>(1e-3))
        .runs(3)
        .await
        .unwrap();

    // max along axis 1
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.max::<1>(1))
        .arg(fuzz.clone())
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            let out = reduce_axis2(&v, 1, f32::NEG_INFINITY, f32::max);
            Tensor::from_slice(&device, [SHAPE[0]], &out)
        })
        .compare_with(approx_compare::<1, f32>(1e-5))
        .runs(3)
        .await
        .unwrap();

    // min along axis 0
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.min::<1>(0))
        .arg(fuzz.clone())
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            let out = reduce_axis2(&v, 0, f32::INFINITY, f32::min);
            Tensor::from_slice(&device, [SHAPE[1]], &out)
        })
        .compare_with(approx_compare::<1, f32>(1e-5))
        .runs(3)
        .await
        .unwrap();

    // product along axis 1 (small range to avoid overflow)
    let fuzz_small = FuzzGenerator::<2, f32>::new([4, 6])
        .with_seed(43)
        .with_distribution(Uniform::new(0.5, 2.0).unwrap());
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.product::<1>(1))
        .arg(fuzz_small)
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            let out = reduce_axis2(&v, 1, 1.0, |a, b| a * b);
            Tensor::from_slice(&device, [4], &out)
        })
        .compare_with(approx_compare::<1, f32>(1e-2))
        .runs(3)
        .await
        .unwrap();

    // mean along axis 1
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.mean::<1>(1))
        .arg(fuzz.clone())
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            let out = mean_axis2(&v, 1);
            Tensor::from_slice(&device, [SHAPE[0]], &out)
        })
        .compare_with(approx_compare::<1, f32>(1e-4))
        .runs(3)
        .await
        .unwrap();

    // var along axis 1
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.var::<1>(1))
        .arg(fuzz)
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            let out = var_axis2(&v, 1);
            Tensor::from_slice(&device, [SHAPE[0]], &out)
        })
        .compare_with(approx_compare::<1, f32>(1e-3))
        .runs(3)
        .await
        .unwrap();
}

#[tokio::test]
async fn full_tensor_sum_large_fuzzed() {
    // Large 2D sum to test accumulation precision with non-contiguous layouts
    const SHAPE: [usize; 2] = [100, 100];
    let fuzz = FuzzGenerator::<2, f32>::new(SHAPE)
        .with_seed(99)
        .with_distribution(Uniform::new(0.0, 10.0).unwrap());

    fusor_conformance::assert(async |x: Tensor<2, f32>| x.sum::<1>(1))
        .arg(fuzz)
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            let out: Vec<f32> = v.iter().map(|row| row.iter().sum()).collect();
            Tensor::from_slice(&device, [SHAPE[0]], &out)
        })
        .compare_with(approx_compare::<1, f32>(1e-1))
        .runs(3)
        .await
        .unwrap();
}

#[tokio::test]
async fn index_select_fuzzed() {
    static INDICES_1D: &[u32] = &[31, 15, 0, 7, 23, 3, 28, 10];
    static INDICES_2D_DIM0: &[u32] = &[7, 3, 0, 5, 1];
    static INDICES_2D_DIM1: &[u32] = &[5, 2, 0, 4];
    static DUP_INDICES: &[u32] = &[0, 0, 2, 2, 1, 1];

    // 1D index_select with fuzzed data, dim=0
    const SHAPE_1D: [usize; 1] = [32];
    let gen_1d = FuzzGenerator::<1, f32>::new(SHAPE_1D).with_seed(50);

    fusor_conformance::assert(async |x: Tensor<1, f32>| {
        let indices = Tensor::from_slice(&x.device(), [INDICES_1D.len()], INDICES_1D);
        x.index_select(0, &indices)
    })
    .arg(gen_1d)
    .equal_to_resolved_with_device(async |v: Vec<f32>, device: Device| {
        let out = index_select1(&v, INDICES_1D);
        Tensor::from_slice(&device, [INDICES_1D.len()], &out)
    })
    .compare_with(approx_compare::<1, f32>(1e-6))
    .runs(3)
    .await
    .unwrap();

    // 2D index_select dim=0 with fuzzed data
    const SHAPE_2D: [usize; 2] = [8, 6];
    let gen_2d = FuzzGenerator::<2, f32>::new(SHAPE_2D).with_seed(51);

    fusor_conformance::assert(async |x: Tensor<2, f32>| {
        let indices = Tensor::from_slice(&x.device(), [INDICES_2D_DIM0.len()], INDICES_2D_DIM0);
        x.index_select(0, &indices)
    })
    .arg(gen_2d.clone())
    .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
        let out = index_select2(&v, 0, INDICES_2D_DIM0);
        Tensor::new(&device, &out)
    })
    .compare_with(approx_compare::<2, f32>(1e-6))
    .runs(3)
    .await
    .unwrap();

    // 2D index_select dim=1 with fuzzed data
    fusor_conformance::assert(async |x: Tensor<2, f32>| {
        let indices = Tensor::from_slice(&x.device(), [INDICES_2D_DIM1.len()], INDICES_2D_DIM1);
        x.index_select(1, &indices)
    })
    .arg(gen_2d)
    .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
        let out = index_select2(&v, 1, INDICES_2D_DIM1);
        Tensor::new(&device, &out)
    })
    .compare_with(approx_compare::<2, f32>(1e-6))
    .runs(3)
    .await
    .unwrap();

    // Duplicate indices with fuzzed data
    let gen_3x4 = FuzzGenerator::<2, f32>::new([3, 4]).with_seed(52);
    fusor_conformance::assert(async |x: Tensor<2, f32>| {
        let indices = Tensor::from_slice(&x.device(), [DUP_INDICES.len()], DUP_INDICES);
        x.index_select(0, &indices)
    })
    .arg(gen_3x4)
    .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
        let out = index_select2(&v, 0, DUP_INDICES);
        Tensor::new(&device, &out)
    })
    .compare_with(approx_compare::<2, f32>(1e-6))
    .runs(3)
    .await
    .unwrap();
}

#[tokio::test]
async fn index_select_single_rank_and_large_regressions() {
    for device in available_devices().await {
        let single_input: Tensor<2, f32> =
            Tensor::from_slice(&device, [3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let single_indices = Tensor::from_slice(&device, [1], &[1u32]);
        let single_actual = single_input.index_select(0, &single_indices);
        let single_expected: Tensor<2, f32> = Tensor::from_slice(&device, [1, 2], &[3.0, 4.0]);
        common::assert_approx_tensors(single_actual, single_expected, 0.0).await;

        let input_3d: Tensor<3, f32> = Tensor::from_slice(
            &device,
            [2, 2, 2],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        );
        let indices_3d = Tensor::from_slice(&device, [2], &[1u32, 0]);
        let actual_3d = input_3d.index_select(0, &indices_3d);
        let expected_3d: Tensor<3, f32> = Tensor::new(
            &device,
            &[[[5.0, 6.0], [7.0, 8.0]], [[1.0, 2.0], [3.0, 4.0]]],
        );
        common::assert_approx_tensors(actual_3d, expected_3d, 0.0).await;

        const SIZE: usize = 100;
        let large_input: Tensor<2, f32> = arange(&device, 0.0f32, (SIZE * SIZE) as f32)
            .reshape([SIZE, SIZE])
            .to_concrete();
        let reverse_indices: Vec<u32> = (0..SIZE).rev().map(|i| i as u32).collect();
        let large_indices = Tensor::from_slice(&device, [SIZE], &reverse_indices);
        let large_actual = large_input.index_select(0, &large_indices);
        let large_expected_rows: Vec<Vec<f32>> = (0..SIZE)
            .map(|row| {
                let source_row = SIZE - 1 - row;
                (0..SIZE)
                    .map(|col| (source_row * SIZE + col) as f32)
                    .collect()
            })
            .collect();
        let large_expected = Tensor::new(&device, &large_expected_rows);
        common::assert_approx_tensors(large_actual, large_expected, 0.0).await;
    }
}
