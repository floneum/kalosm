mod common;

use common::{layer_norm_last_dim_3d, rms_norm_last_dim_3d, softmax_last_dim_2d};
use fusor::{Device, Tensor};
use fusor_conformance::{FuzzGenerator, approx_compare};
use rand::distr::Uniform;

#[tokio::test]
async fn softmax_and_normalization_match_reference_paths() {
    // Softmax with fuzzed input
    const SOFTMAX_SHAPE: [usize; 2] = [3, 4];
    let gen_softmax = FuzzGenerator::<2, f32>::new(SOFTMAX_SHAPE)
        .with_seed(400)
        .with_distribution(Uniform::new(-4.0, 4.0).unwrap());

    // softmax vs host reference
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.softmax::<1>(1))
        .arg(gen_softmax.clone())
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            Tensor::new(&device, &softmax_last_dim_2d(&v))
        })
        .compare_with(approx_compare::<2, f32>(1e-5))
        .runs(3)
        .await
        .unwrap();

    // softmax_last_dim vs softmax
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.softmax_last_dim::<1>())
        .arg(gen_softmax.clone())
        .equal_to(async |x: Tensor<2, f32>| x.softmax::<1>(1))
        .compare_with(approx_compare::<2, f32>(1e-6))
        .runs(3)
        .await
        .unwrap();

    // softmax_slow vs softmax
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.softmax_slow::<1>(1))
        .arg(gen_softmax.clone())
        .equal_to(async |x: Tensor<2, f32>| x.softmax::<1>(1))
        .compare_with(approx_compare::<2, f32>(1e-6))
        .runs(3)
        .await
        .unwrap();

    // softmax_slow_last_dim vs softmax_last_dim
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.softmax_slow_last_dim::<1>())
        .arg(gen_softmax.clone())
        .equal_to(async |x: Tensor<2, f32>| x.softmax_last_dim::<1>())
        .compare_with(approx_compare::<2, f32>(1e-6))
        .runs(3)
        .await
        .unwrap();

    // softmax_last_dim_fused vs softmax_last_dim
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.softmax_last_dim_fused::<1>())
        .arg(gen_softmax)
        .equal_to(async |x: Tensor<2, f32>| x.softmax_last_dim::<1>())
        .compare_with(approx_compare::<2, f32>(1e-5))
        .runs(3)
        .await
        .unwrap();

    // RMS norm with fuzzed input
    const NORM_SHAPE: [usize; 3] = [2, 3, 4];
    let gen_norm = FuzzGenerator::<3, f32>::new(NORM_SHAPE)
        .with_seed(410)
        .with_distribution(Uniform::new(-4.0, 4.0).unwrap());

    static WEIGHT: [f32; 4] = [1.0, 1.5, 0.5, 2.0];
    static BIAS: [f32; 4] = [0.1, -0.2, 0.3, -0.4];

    // rms_norm vs host reference
    fusor_conformance::assert(async |x: Tensor<3, f32>| {
        let device = x.device();
        let weight: Tensor<3, f32> = Tensor::from_slice(&device, [1, 1, 4], &WEIGHT)
            .broadcast_as([2, 3, 4])
            .to_concrete();
        x.rms_norm::<2, _>(&weight, 1e-5)
    })
    .arg(gen_norm.clone())
    .equal_to_resolved_with_device(async |v: Vec<Vec<Vec<f32>>>, device: Device| {
        Tensor::new(&device, &rms_norm_last_dim_3d(&v, &WEIGHT, 1e-5))
    })
    .compare_with(approx_compare::<3, f32>(1e-4))
    .runs(3)
    .await
    .unwrap();

    // layer_norm vs host reference
    fusor_conformance::assert(async |x: Tensor<3, f32>| {
        let device = x.device();
        let weight: Tensor<3, f32> = Tensor::from_slice(&device, [1, 1, 4], &WEIGHT)
            .broadcast_as([2, 3, 4])
            .to_concrete();
        let bias: Tensor<3, f32> = Tensor::from_slice(&device, [1, 1, 4], &BIAS)
            .broadcast_as([2, 3, 4])
            .to_concrete();
        x.layer_norm::<2, _, _>(&weight, Some(&bias), 1e-5, true)
    })
    .arg(gen_norm.clone())
    .equal_to_resolved_with_device(async |v: Vec<Vec<Vec<f32>>>, device: Device| {
        Tensor::new(&device, &layer_norm_last_dim_3d(&v, &WEIGHT, &BIAS, 1e-5))
    })
    .compare_with(approx_compare::<3, f32>(1e-4))
    .runs(3)
    .await
    .unwrap();

    // rms_norm_fused (with bias) vs rms_norm + bias
    fusor_conformance::assert(async |x: Tensor<3, f32>| {
        let device = x.device();
        let weight = Tensor::from_slice(&device, [4], &WEIGHT);
        let bias = Tensor::from_slice(&device, [4], &BIAS);
        x.rms_norm_fused::<1, 2>(&weight, Some(&bias), 1e-5)
    })
    .arg(gen_norm.clone())
    .equal_to_resolved_with_device(async |v: Vec<Vec<Vec<f32>>>, device: Device| {
        let rms = rms_norm_last_dim_3d(&v, &WEIGHT, 1e-5);
        let out: Vec<Vec<Vec<f32>>> = rms
            .into_iter()
            .map(|matrix| {
                matrix
                    .into_iter()
                    .map(|row| {
                        row.into_iter()
                            .zip(BIAS)
                            .map(|(v, b)| v + b)
                            .collect()
                    })
                    .collect()
            })
            .collect();
        Tensor::new(&device, &out)
    })
    .compare_with(approx_compare::<3, f32>(1e-4))
    .runs(3)
    .await
    .unwrap();

    // rms_norm_fused_no_bias vs rms_norm reference
    fusor_conformance::assert(async |x: Tensor<3, f32>| {
        let device = x.device();
        let weight = Tensor::from_slice(&device, [4], &WEIGHT);
        x.rms_norm_fused_no_bias::<1, 2>(&weight, 1e-5)
    })
    .arg(gen_norm)
    .equal_to_resolved_with_device(async |v: Vec<Vec<Vec<f32>>>, device: Device| {
        Tensor::new(&device, &rms_norm_last_dim_3d(&v, &WEIGHT, 1e-5))
    })
    .compare_with(approx_compare::<3, f32>(1e-4))
    .runs(3)
    .await
    .unwrap();
}
