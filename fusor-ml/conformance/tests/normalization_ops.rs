mod common;

use common::{layer_norm_last_dim_3d, rms_norm_last_dim_3d, softmax_last_dim_2d};
use fusor::{Device, Tensor};
use fusor_conformance::{FuzzGenerator, approx_compare, available_devices};
use rand::distr::Uniform;

fn norm_weight(feature_count: usize) -> Vec<f32> {
    (0..feature_count)
        .map(|i| 1.0 + ((i % 5) as f32) * 0.25)
        .collect()
}

fn norm_bias(feature_count: usize) -> Vec<f32> {
    (0..feature_count)
        .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
        .collect()
}

fn softmax_last_dim_4d(shape: [usize; 4], input: &[f32]) -> Vec<f32> {
    let outer = shape[0] * shape[1] * shape[2];
    let cols = shape[3];
    let mut output = vec![0.0; input.len()];
    for row in 0..outer {
        let start = row * cols;
        let row_values = &input[start..start + cols];
        let max = row_values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for col in 0..cols {
            let value = (row_values[col] - max).exp();
            output[start + col] = value;
            sum += value;
        }
        for col in 0..cols {
            output[start + col] /= sum;
        }
    }
    output
}

#[tokio::test]
async fn softmax_last_dim_rank4_matches_host_reference() {
    for shape in [[2, 4, 1, 16], [2, 1, 3, 3]] {
        let data = (0..shape.iter().product::<usize>())
            .map(|i| (((i % 29) as f32) - 14.0) * 0.11)
            .collect::<Vec<_>>();
        let expected_data = softmax_last_dim_4d(shape, &data);
        for device in available_devices().await {
            let input = Tensor::from_slice(&device, shape, &data);
            let actual = input.softmax::<3>(3).to_concrete();
            let expected = Tensor::from_slice(&device, shape, &expected_data);
            fusor_conformance::approx_eq(&actual, &expected, 1e-5)
                .await
                .unwrap();
        }
    }
}

#[tokio::test]
async fn softmax_and_normalization_match_reference_paths() {
    // Softmax with fuzzed input
    let gen_softmax = FuzzGenerator::<2, f32>::new([16..=45, 16..=45])
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

    // RMS norm with fuzzed input
    let gen_norm = FuzzGenerator::<3, f32>::new([2..=3, 16..=17, 255..=257])
        .with_seed(410)
        .with_distribution(Uniform::new(-4.0, 4.0).unwrap());

    // rms_norm vs host reference
    fusor_conformance::assert(async |x: Tensor<3, f32>| {
        let device = x.device();
        let feature_count = x.shape()[2];
        let weight_data = norm_weight(feature_count);
        let weight: Tensor<3, f32> =
            Tensor::from_slice(&device, [1, 1, feature_count], &weight_data)
                .broadcast_as(x.shape())
                .to_concrete();
        x.rms_norm::<2, _>(&weight, 1e-5)
    })
    .arg(gen_norm.clone())
    .equal_to_resolved_with_device(async |v: Vec<Vec<Vec<f32>>>, device: Device| {
        let feature_count = v[0][0].len();
        let weight_data = norm_weight(feature_count);
        Tensor::new(&device, &rms_norm_last_dim_3d(&v, &weight_data, 1e-5))
    })
    .compare_with(approx_compare::<3, f32>(1e-4))
    .runs(3)
    .await
    .unwrap();

    // layer_norm vs host reference
    fusor_conformance::assert(async |x: Tensor<3, f32>| {
        let device = x.device();
        let feature_count = x.shape()[2];
        let weight_data = norm_weight(feature_count);
        let bias_data = norm_bias(feature_count);
        let weight: Tensor<3, f32> =
            Tensor::from_slice(&device, [1, 1, feature_count], &weight_data)
                .broadcast_as(x.shape())
                .to_concrete();
        let bias: Tensor<3, f32> = Tensor::from_slice(&device, [1, 1, feature_count], &bias_data)
            .broadcast_as(x.shape())
            .to_concrete();
        x.layer_norm::<2, _, _>(&weight, Some(&bias), 1e-5, true)
    })
    .arg(gen_norm.clone())
    .equal_to_resolved_with_device(async |v: Vec<Vec<Vec<f32>>>, device: Device| {
        let feature_count = v[0][0].len();
        let weight_data = norm_weight(feature_count);
        let bias_data = norm_bias(feature_count);
        Tensor::new(
            &device,
            &layer_norm_last_dim_3d(&v, &weight_data, &bias_data, 1e-5),
        )
    })
    .compare_with(approx_compare::<3, f32>(1e-4))
    .runs(3)
    .await
    .unwrap();

    // rms_norm_fused (with bias) vs rms_norm + bias
    fusor_conformance::assert(async |x: Tensor<3, f32>| {
        let device = x.device();
        let feature_count = x.shape()[2];
        let weight_data = norm_weight(feature_count);
        let bias_data = norm_bias(feature_count);
        let weight = Tensor::from_slice(&device, [feature_count], &weight_data);
        let bias = Tensor::from_slice(&device, [feature_count], &bias_data);
        x.rms_norm_fused::<1, 2>(&weight, Some(&bias), 1e-5)
    })
    .arg(gen_norm.clone())
    .equal_to_resolved_with_device(async |v: Vec<Vec<Vec<f32>>>, device: Device| {
        let feature_count = v[0][0].len();
        let weight_data = norm_weight(feature_count);
        let bias_data = norm_bias(feature_count);
        let rms = rms_norm_last_dim_3d(&v, &weight_data, 1e-5);
        let out: Vec<Vec<Vec<f32>>> = rms
            .into_iter()
            .map(|matrix| {
                matrix
                    .into_iter()
                    .map(|row| {
                        row.into_iter()
                            .zip(bias_data.iter().copied())
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
        let feature_count = x.shape()[2];
        let weight_data = norm_weight(feature_count);
        let weight = Tensor::from_slice(&device, [feature_count], &weight_data);
        x.rms_norm_fused_no_bias::<1, 2>(&weight, 1e-5)
    })
    .arg(gen_norm)
    .equal_to_resolved_with_device(async |v: Vec<Vec<Vec<f32>>>, device: Device| {
        let feature_count = v[0][0].len();
        let weight_data = norm_weight(feature_count);
        Tensor::new(&device, &rms_norm_last_dim_3d(&v, &weight_data, 1e-5))
    })
    .compare_with(approx_compare::<3, f32>(1e-4))
    .runs(3)
    .await
    .unwrap();
}
