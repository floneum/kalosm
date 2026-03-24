mod common;

use common::{conv1d_ncw, matmul2, pool1d_ncw};
use fusor::{Device, Tensor};
use fusor_conformance::{FuzzGenerator, approx_compare};
use rand::distr::Uniform;

#[tokio::test]
async fn matmul_match_host_reference() {
    const M: usize = 3;
    const K: usize = 4;
    const N: usize = 2;

    let gen_lhs = FuzzGenerator::<2, f32>::new([M, K])
        .with_seed(300)
        .with_distribution(Uniform::new(-3.0, 3.0).unwrap());
    let gen_rhs = FuzzGenerator::<2, f32>::new([K, N])
        .with_seed(301)
        .with_distribution(Uniform::new(-3.0, 3.0).unwrap());

    // matmul vs host reference
    fusor_conformance::assert(async |a: Tensor<2, f32>, b: Tensor<2, f32>| a.matmul(&b))
        .arg(gen_lhs.clone())
        .arg(gen_rhs.clone())
        .equal_to_resolved_with_device(
            async |a: Vec<Vec<f32>>, b: Vec<Vec<f32>>, device: Device| {
                Tensor::new(&device, &matmul2(&a, &b))
            },
        )
        .compare_with(approx_compare::<2, f32>(1e-4))
        .runs(3)
        .await
        .unwrap();

    // mat_mul vs matmul (two API paths)
    fusor_conformance::assert(async |a: Tensor<2, f32>, b: Tensor<2, f32>| a.mat_mul(&b))
        .arg(gen_lhs)
        .arg(gen_rhs)
        .equal_to(async |a: Tensor<2, f32>, b: Tensor<2, f32>| a.matmul(&b))
        .compare_with(approx_compare::<2, f32>(1e-6))
        .runs(3)
        .await
        .unwrap();
}

#[tokio::test]
async fn matmul_small_fixed_regression() {
    const LHS: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    const RHS: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    fusor_conformance::assert(async |a: Tensor<2, f32>, b: Tensor<2, f32>| a.matmul(&b))
        .arg(move |device: &Device| Tensor::from_slice(device, [2, 3], &LHS))
        .arg(move |device: &Device| Tensor::from_slice(device, [3, 2], &RHS))
        .equal_to_resolved_with_device(
            async |_a: Vec<Vec<f32>>, _b: Vec<Vec<f32>>, device: Device| {
                Tensor::new(&device, &[[22.0f32, 28.0], [49.0, 64.0]])
            },
        )
        .compare_with(approx_compare::<2, f32>(1e-6))
        .runs(1)
        .await
        .unwrap();
}

#[tokio::test]
async fn conv_and_pool_match_host_reference() {
    // Conv1D with fuzzed input
    const CONV_SHAPE: [usize; 3] = [1, 1, 7];
    let gen_conv = FuzzGenerator::<3, f32>::new(CONV_SHAPE)
        .with_seed(310)
        .with_distribution(Uniform::new(-3.0, 3.0).unwrap());

    static CONV_WEIGHT: &[f32] = &[0.25, -0.5, 1.0];
    static CONV_BIAS: &[f32] = &[0.1];

    fusor_conformance::assert(async |input: Tensor<3, f32>| {
        let weight = Tensor::from_slice(&input.device(), [1, 1, 3], CONV_WEIGHT);
        let bias = Tensor::from_slice(&input.device(), [1], CONV_BIAS);
        input.conv(&weight, Some(&bias), [1], [2])
    })
    .arg(gen_conv)
    .equal_to_resolved_with_device(async |v: Vec<Vec<Vec<f32>>>, device: Device| {
        let expected = conv1d_ncw(&v, &[vec![vec![0.25, -0.5, 1.0]]], Some(&[0.1]), 1, 2);
        Tensor::new(&device, &expected)
    })
    .compare_with(approx_compare::<3, f32>(1e-4))
    .runs(3)
    .await
    .unwrap();

    // Pool: pool_max with fuzzed input
    const POOL_SHAPE: [usize; 3] = [1, 2, 8];
    let gen_pool = FuzzGenerator::<3, f32>::new(POOL_SHAPE)
        .with_seed(320)
        .with_distribution(Uniform::new(-4.0, 12.0).unwrap());

    // pool(Tensor::max) vs pool_max (two API paths)
    fusor_conformance::assert(async |x: Tensor<3, f32>| -> Tensor<3, f32> {
        x.pool([(2, 2)], Tensor::max)
    })
    .arg(gen_pool.clone())
    .equal_to(async |x: Tensor<3, f32>| x.pool_max([(2, 2)]))
    .compare_with(approx_compare::<3, f32>(1e-6))
    .runs(3)
    .await
    .unwrap();

    // pool_max vs host reference
    fusor_conformance::assert(async |x: Tensor<3, f32>| x.pool_max([(2, 2)]))
        .arg(gen_pool.clone())
        .equal_to_resolved_with_device(async |v: Vec<Vec<Vec<f32>>>, device: Device| {
            Tensor::new(&device, &pool1d_ncw(&v, 2, 2, f32::max, f32::NEG_INFINITY))
        })
        .compare_with(approx_compare::<3, f32>(1e-6))
        .runs(3)
        .await
        .unwrap();

    // pool_min vs host reference
    fusor_conformance::assert(async |x: Tensor<3, f32>| x.pool_min([(2, 2)]))
        .arg(gen_pool)
        .equal_to_resolved_with_device(async |v: Vec<Vec<Vec<f32>>>, device: Device| {
            Tensor::new(&device, &pool1d_ncw(&v, 2, 2, f32::min, f32::INFINITY))
        })
        .compare_with(approx_compare::<3, f32>(1e-6))
        .runs(3)
        .await
        .unwrap();
}

#[tokio::test]
async fn matmul_identity_matrix() {
    const M: usize = 2;
    const N: usize = 3;
    let fuzz = FuzzGenerator::<2, f32>::new([M, N])
        .with_seed(330)
        .with_distribution(Uniform::new(-5.0, 5.0).unwrap());

    // A @ I = A
    fusor_conformance::assert(async |a: Tensor<2, f32>| {
        let device = a.device();
        let identity = Tensor::from_slice(
            &device,
            [N, N],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        );
        a.matmul(&identity)
    })
    .arg(fuzz)
    .equal_to(async |a: Tensor<2, f32>| a)
    .compare_with(approx_compare::<2, f32>(1e-5))
    .runs(3)
    .await
    .unwrap();
}

#[tokio::test]
async fn matmul_large_fuzzed() {
    const M: usize = 32;
    const K: usize = 48;
    const N: usize = 32;

    let gen_lhs = FuzzGenerator::<2, f32>::new([M, K])
        .with_seed(100)
        .with_distribution(Uniform::new(-2.0, 2.0).unwrap());
    let gen_rhs = FuzzGenerator::<2, f32>::new([K, N])
        .with_seed(101)
        .with_distribution(Uniform::new(-2.0, 2.0).unwrap());

    fusor_conformance::assert(async |a: Tensor<2, f32>, b: Tensor<2, f32>| a.matmul(&b))
        .arg(gen_lhs)
        .arg(gen_rhs)
        .equal_to_resolved_with_device(
            async |a: Vec<Vec<f32>>, b: Vec<Vec<f32>>, device: Device| {
                let expected = matmul2(&a, &b);
                Tensor::new(&device, &expected)
            },
        )
        .compare_with(approx_compare::<2, f32>(1e-2))
        .runs(3)
        .await
        .unwrap();
}
