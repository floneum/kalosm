//! Conformance for ranks above 3 and shapes containing zero-sized dimensions.
//! Existing conformance stops at rank 3, but reductions, softmax, and rms_norm
//! are rank-generic and need direct coverage. Empty-tensor cases cover
//! elementwise ops and axis reductions when one or more dims is zero.

use fusor::Tensor;
use fusor_conformance::{approx_eq, approx_or_relative_eq, available_devices, exact_eq};

fn deterministic_data(total: usize, seed: u32) -> Vec<f32> {
    (0..total)
        .map(|i| (((i + seed as usize) % 19) as f32 - 9.0) * 0.17)
        .collect()
}

fn rank4_strides(shape: [usize; 4]) -> [usize; 4] {
    [
        shape[1] * shape[2] * shape[3],
        shape[2] * shape[3],
        shape[3],
        1,
    ]
}

fn idx4(shape: [usize; 4], i0: usize, i1: usize, i2: usize, i3: usize) -> usize {
    let s = rank4_strides(shape);
    i0 * s[0] + i1 * s[1] + i2 * s[2] + i3 * s[3]
}

fn sum_axis_4d(input: &[f32], shape: [usize; 4], axis: usize) -> (Vec<f32>, [usize; 3]) {
    let mut out_shape = [0usize; 3];
    let mut out_dim = 0;
    for (d, size) in shape.iter().enumerate() {
        if d != axis {
            out_shape[out_dim] = *size;
            out_dim += 1;
        }
    }
    let out_total: usize = out_shape.iter().product();
    let mut out = vec![0.0f32; out_total];
    let dims = [0usize, 1, 2, 3];
    for i0 in 0..shape[0] {
        for i1 in 0..shape[1] {
            for i2 in 0..shape[2] {
                for i3 in 0..shape[3] {
                    let idx_in = [i0, i1, i2, i3];
                    let v = input[idx4(shape, i0, i1, i2, i3)];
                    let mut out_idx = 0usize;
                    let mut stride = 1usize;
                    for d in (0..4).rev() {
                        if d == axis {
                            continue;
                        }
                        out_idx += idx_in[dims[d]] * stride;
                        stride *= shape[d];
                    }
                    out[out_idx] += v;
                }
            }
        }
    }
    (out, out_shape)
}

fn softmax_last_dim_4d(input: &[f32], shape: [usize; 4]) -> Vec<f32> {
    let last = shape[3];
    let outer: usize = shape[0] * shape[1] * shape[2];
    let mut out = vec![0.0f32; outer * last];
    for o in 0..outer {
        let base = o * last;
        let mut max = f32::NEG_INFINITY;
        for j in 0..last {
            if input[base + j] > max {
                max = input[base + j];
            }
        }
        let mut sum = 0.0f32;
        for j in 0..last {
            let v = (input[base + j] - max).exp();
            out[base + j] = v;
            sum += v;
        }
        for j in 0..last {
            out[base + j] /= sum;
        }
    }
    out
}

fn rms_norm_fused_4d(input: &[f32], shape: [usize; 4], weight: &[f32], eps: f32) -> Vec<f32> {
    let last = shape[3];
    let outer = shape[0] * shape[1] * shape[2];
    let mut out = vec![0.0f32; outer * last];
    for o in 0..outer {
        let base = o * last;
        let mean_sq: f32 = (0..last)
            .map(|j| input[base + j] * input[base + j])
            .sum::<f32>()
            / last as f32;
        let denom = (mean_sq + eps).sqrt();
        for j in 0..last {
            out[base + j] = (input[base + j] / denom) * weight[j];
        }
    }
    out
}

#[tokio::test]
async fn rank4_sum_per_axis_matches_reference() {
    const SHAPE: [usize; 4] = [2, 3, 4, 5];
    let data = deterministic_data(SHAPE.iter().product(), 600);

    for axis in 0..4 {
        let (expected_flat, out_shape) = sum_axis_4d(&data, SHAPE, axis);
        for device in available_devices().await {
            let input: Tensor<4, f32> = Tensor::from_slice(&device, SHAPE, &data);
            let actual: Tensor<3, f32> = match axis {
                0 => input.sum::<3>(0),
                1 => input.sum::<3>(1),
                2 => input.sum::<3>(2),
                _ => input.sum::<3>(3),
            };
            let actual = actual.to_concrete();
            let expected: Tensor<3, f32> = Tensor::from_slice(&device, out_shape, &expected_flat);
            approx_eq(&actual, &expected, 1e-4).await.unwrap();
        }
    }
}

#[tokio::test]
async fn rank4_mean_axis0_matches_reference() {
    const SHAPE: [usize; 4] = [3, 2, 4, 5];
    let data = deterministic_data(SHAPE.iter().product(), 601);
    let (sum_flat, out_shape) = sum_axis_4d(&data, SHAPE, 0);
    let divisor = SHAPE[0] as f32;
    let expected_flat: Vec<f32> = sum_flat.iter().map(|v| v / divisor).collect();

    for device in available_devices().await {
        let input: Tensor<4, f32> = Tensor::from_slice(&device, SHAPE, &data);
        let actual = input.mean::<3>(0).to_concrete();
        let expected: Tensor<3, f32> = Tensor::from_slice(&device, out_shape, &expected_flat);
        approx_eq(&actual, &expected, 1e-4).await.unwrap();
    }
}

#[tokio::test]
async fn rank4_softmax_last_dim_matches_reference() {
    const SHAPE: [usize; 4] = [2, 2, 3, 8];
    let data = deterministic_data(SHAPE.iter().product(), 602);
    let expected_flat = softmax_last_dim_4d(&data, SHAPE);

    for device in available_devices().await {
        let input: Tensor<4, f32> = Tensor::from_slice(&device, SHAPE, &data);
        let actual = input.softmax_last_dim::<3>().to_concrete();
        let expected: Tensor<4, f32> = Tensor::from_slice(&device, SHAPE, &expected_flat);
        approx_eq(&actual, &expected, 1e-5).await.unwrap();
    }
}

#[tokio::test]
async fn rank4_rms_norm_fused_matches_reference() {
    const SHAPE: [usize; 4] = [2, 2, 3, 16];
    let data = deterministic_data(SHAPE.iter().product(), 603);
    let weight: Vec<f32> = (0..SHAPE[3]).map(|i| 1.0 + (i % 5) as f32 * 0.25).collect();
    let expected_flat = rms_norm_fused_4d(&data, SHAPE, &weight, 1e-5);

    for device in available_devices().await {
        let input: Tensor<4, f32> = Tensor::from_slice(&device, SHAPE, &data);
        let w: Tensor<1, f32> = Tensor::from_slice(&device, [SHAPE[3]], &weight);
        let actual = input.rms_norm_fused::<1, 3>(&w, None, 1e-5).to_concrete();
        let expected: Tensor<4, f32> = Tensor::from_slice(&device, SHAPE, &expected_flat);
        approx_or_relative_eq(&actual, &expected, 1e-4, 1e-4)
            .await
            .unwrap();
    }
}

#[tokio::test]
async fn empty_tensor_elementwise_add_returns_empty() {
    // 0-sized leading dim — elementwise op must be well-defined on empty inputs.
    for device in available_devices().await {
        let a: Tensor<2, f32> = Tensor::zeros(&device, [0, 6]);
        let b: Tensor<2, f32> = Tensor::zeros(&device, [0, 6]);
        let out = a.add_::<2, 2, _>(&b).to_concrete();
        assert_eq!(out.shape(), [0, 6]);
        let expected: Tensor<2, f32> = Tensor::zeros(&device, [0, 6]);
        exact_eq(&out, &expected).await.unwrap();
    }
}

#[tokio::test]
async fn empty_tensor_sum_along_zero_axis_returns_identity() {
    // Reducing over a 0-sized axis: sum-identity is 0, so each output element
    // must be exactly 0 on both backends.
    for device in available_devices().await {
        let input: Tensor<2, f32> = Tensor::zeros(&device, [0, 4]);
        let out = input.sum::<1>(0).to_concrete();
        assert_eq!(out.shape(), [4]);
        let expected: Tensor<1, f32> = Tensor::zeros(&device, [4]);
        exact_eq(&out, &expected).await.unwrap();
    }
}

#[tokio::test]
async fn rank4_max_min_match_reference() {
    const SHAPE: [usize; 4] = [2, 3, 2, 4];
    let data = deterministic_data(SHAPE.iter().product(), 604);
    // Reduce along last axis; compute reference via flat indexing.
    let outer = SHAPE[0] * SHAPE[1] * SHAPE[2];
    let last = SHAPE[3];
    let out_shape = [SHAPE[0], SHAPE[1], SHAPE[2]];
    let mut max_ref = vec![f32::NEG_INFINITY; outer];
    let mut min_ref = vec![f32::INFINITY; outer];
    for o in 0..outer {
        for j in 0..last {
            let v = data[o * last + j];
            if v > max_ref[o] {
                max_ref[o] = v;
            }
            if v < min_ref[o] {
                min_ref[o] = v;
            }
        }
    }

    for device in available_devices().await {
        let input: Tensor<4, f32> = Tensor::from_slice(&device, SHAPE, &data);
        let max_actual = input.max::<3>(3).to_concrete();
        let max_expected: Tensor<3, f32> = Tensor::from_slice(&device, out_shape, &max_ref);
        approx_eq(&max_actual, &max_expected, 1e-6).await.unwrap();

        let min_actual = input.min::<3>(3).to_concrete();
        let min_expected: Tensor<3, f32> = Tensor::from_slice(&device, out_shape, &min_ref);
        approx_eq(&min_actual, &min_expected, 1e-6).await.unwrap();
    }
}
