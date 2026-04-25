mod common;

use common::{index_select1, index_select2, keepdim2, mean_axis2, reduce_axis2, var_axis2};
use fusor::{Device, Tensor, arange};
use fusor_conformance::{FuzzGenerator, approx_compare, f16_capable_devices, relative_compare};
use half::f16;
use rand::distr::Uniform;

#[tokio::test]
async fn reductions_match_host_reference() {
    const SHAPE: [usize; 2] = [45, 45];
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
        // Product of 45 values in [0.5, 2.0] grows to ~1e5 in some seeds;
        // an absolute 1e-3 tolerance becomes meaningless. 0.01% relative
        // catches accuracy regressions while accommodating GPU-vs-host
        // accumulation order.
        .compare_with(relative_compare::<1>(1e-4))
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
        .compare_with(relative_compare::<2>(1e-4))
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
    const SHAPE: [usize; 2] = [45, 45];
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
    .devices(f16_capable_devices().await)
    .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
        let out: Vec<Vec<f32>> = v
            .iter()
            .map(|row| row.iter().map(|&x| f16::from_f32(x).to_f32()).collect())
            .collect();
        Tensor::new(&device, &out)
    })
    // WARP/DX12 can round f32 -> f16 one half-precision ULP differently
    // from the CPU reference in this value range.
    .compare_with(approx_compare::<2, f32>(5e-3))
    .runs(3)
    .await
    .unwrap();
}

#[tokio::test]
async fn full_tensor_reductions_fuzzed() {
    // 2D reductions with fuzzed data + non-contiguous layouts
    const SHAPE: [usize; 2] = [45, 45];
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
    const SHAPE_1D: [usize; 1] = [2048];
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
    const SHAPE_2D: [usize; 2] = [45, 45];
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
    let single_gen = FuzzGenerator::<2, f32>::new([3, 2])
        .with_seed(53)
        .with_distribution(Uniform::new(-5.0, 5.0).unwrap());
    fusor_conformance::assert(async |x: Tensor<2, f32>| {
        let indices = Tensor::from_slice(&x.device(), [1], &[1u32]);
        x.index_select(0, &indices)
    })
    .arg(single_gen)
    .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
        Tensor::new(&device, &[v[1].clone()])
    })
    .compare_with(approx_compare::<2, f32>(0.0))
    .runs(3)
    .await
    .unwrap();

    let gen_3d = FuzzGenerator::<3, f32>::new([2, 2, 2])
        .with_seed(54)
        .with_distribution(Uniform::new(-5.0, 5.0).unwrap());
    fusor_conformance::assert(async |x: Tensor<3, f32>| {
        let indices = Tensor::from_slice(&x.device(), [2], &[1u32, 0]);
        x.index_select(0, &indices)
    })
    .arg(gen_3d)
    .equal_to_resolved_with_device(async |v: Vec<Vec<Vec<f32>>>, device: Device| {
        Tensor::new(&device, &[v[1].clone(), v[0].clone()])
    })
    .compare_with(approx_compare::<3, f32>(0.0))
    .runs(3)
    .await
    .unwrap();

    const SIZE: usize = 100;
    fusor_conformance::assert(async |device: Device| {
        let input: Tensor<2, f32> = arange(&device, 0.0f32, (SIZE * SIZE) as f32)
            .reshape([SIZE, SIZE])
            .to_concrete();
        let reverse_indices: Vec<u32> = (0..SIZE).rev().map(|i| i as u32).collect();
        let indices = Tensor::from_slice(&device, [SIZE], &reverse_indices);
        input.index_select(0, &indices)
    })
    .arg(|device: &Device| device.clone())
    .equal_to(async |device: Device| {
        let rows: Vec<Vec<f32>> = (0..SIZE)
            .map(|row| {
                let source_row = SIZE - 1 - row;
                (0..SIZE)
                    .map(|col| (source_row * SIZE + col) as f32)
                    .collect()
            })
            .collect();
        Tensor::new(&device, &rows)
    })
    .compare_with(approx_compare::<2, f32>(0.0))
    .await
    .unwrap();
}
