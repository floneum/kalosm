mod common;

use common::{
    assert_approx_tensors, broadcast_1d_to_2d, flatten2, flatten3, permute3, repeat2, reshape2,
    resize2, slice2, sliding_window_1d_ncw, transpose2,
};
use fusor::{Device, Tensor, cat, stack};
use fusor_conformance::{FuzzGenerator, GenerateFromDevice, approx_compare, available_devices};
use fusor_types::{SlidingWindow, StrideSpec};
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

    // restride = transpose
    fusor_conformance::assert(async |x: Tensor<2, f32>| {
        x.restride([StrideSpec::dim(1, 3), StrideSpec::dim(0, 2)])
            .to_concrete()
    })
    .arg(gen_2x3.clone())
    .equal_to(async |x: Tensor<2, f32>| x.transpose(0, 1).to_concrete())
    .compare_with(approx_compare::<2, f32>(0.0))
    .runs(3)
    .await
    .unwrap();

    // .t() = transpose(0, 1)
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.t().to_concrete())
        .arg(gen_2x3.clone())
        .equal_to(async |x: Tensor<2, f32>| x.transpose(0, 1).to_concrete())
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

    // narrow = slice
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.narrow(1, 1, 2).to_concrete())
        .arg(gen_2x4.clone())
        .equal_to(async |x: Tensor<2, f32>| x.slice([0..2, 1..3]).to_concrete())
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

    // expand = broadcast_as
    fusor_conformance::assert(async |x: Tensor<1, f32>| x.expand([2, 3]).to_concrete())
        .arg(gen_1d_3.clone())
        .equal_to(async |x: Tensor<1, f32>| x.broadcast_as([2, 3]).to_concrete())
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
            let squeezed: Vec<Vec<f32>> = v.into_iter().map(|m| m.into_iter().next().unwrap()).collect();
            Tensor::new(&device, &squeezed)
        })
        .compare_with(approx_compare::<2, f32>(0.0))
        .runs(3)
        .await
        .unwrap();

    // unsqueeze_dims = chained unsqueeze
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.unsqueeze_dims::<2, 4>([1, 3]).to_concrete())
        .arg(gen_2x3.clone())
        .equal_to(async |x: Tensor<2, f32>| x.unsqueeze::<3>(1).unsqueeze::<4>(3).to_concrete())
        .compare_with(approx_compare::<4, f32>(0.0))
        .runs(3)
        .await
        .unwrap();

    // squeeze_dims = chained squeeze
    let gen_2x1x3x1 = FuzzGenerator::<4, f32>::new([2, 1, 3, 1])
        .with_seed(506)
        .with_distribution(Uniform::new(-5.0, 5.0).unwrap());
    fusor_conformance::assert(async |x: Tensor<4, f32>| x.squeeze_dims::<2, 2>([1, 3]).to_concrete())
        .arg(gen_2x1x3x1.clone())
        .equal_to(async |x: Tensor<4, f32>| x.squeeze::<3>(3).squeeze::<2>(1).to_concrete())
        .compare_with(approx_compare::<2, f32>(0.0))
        .runs(3)
        .await
        .unwrap();

    // shape preserves correctness
    fusor_conformance::assert(async |x: Tensor<4, f32>| {
        x.shape();
        x
    })
    .arg(gen_2x1x3x1)
    .equal_to(async |x: Tensor<4, f32>| x)
    .runs(3)
    .await
    .unwrap();

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
async fn cat_stack_and_chunk_match_expected_views() {
    let fuzz_2x3 = FuzzGenerator::<2, f32>::new([2, 3])
        .with_seed(510)
        .with_distribution(Uniform::new(-5.0, 5.0).unwrap());

    // cat (free fn vs assoc fn)
    fusor_conformance::assert(
        async |left: Tensor<2, f32>, right: Tensor<2, f32>| {
            cat([left.to_concrete(), right.to_concrete()], 0)
        },
    )
    .arg(fuzz_2x3.clone())
    .arg(fuzz_2x3.clone())
    .equal_to(async |left: Tensor<2, f32>, right: Tensor<2, f32>| {
        Tensor::cat([left.to_concrete(), right.to_concrete()], 0)
    })
    .compare_with(approx_compare::<2, f32>(1e-6))
    .runs(3)
    .await
    .unwrap();

    // stack (free fn vs assoc fn)
    fusor_conformance::assert(
        async |left: Tensor<2, f32>, right: Tensor<2, f32>| -> Tensor<3, f32> {
            stack([left.to_concrete(), right.to_concrete()], 0)
        },
    )
    .arg(fuzz_2x3.clone())
    .arg(fuzz_2x3.clone())
    .equal_to(async |left: Tensor<2, f32>, right: Tensor<2, f32>| -> Tensor<3, f32> {
        Tensor::stack([left.to_concrete(), right.to_concrete()], 0)
    })
    .compare_with(approx_compare::<3, f32>(1e-6))
    .runs(3)
    .await
    .unwrap();

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
