//! Rank/empty-tensor conformance cases.

use fusor::{Device, Tensor};
use fusor_conformance::{
    AssertionCase, AssertionCases, approx_compare, approx_or_relative_compare, exact_compare,
};

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

pub fn rank4_sum_per_axis_matches_reference() -> AssertionCases {
    const SHAPE: [usize; 4] = [2, 3, 4, 5];
    let data = deterministic_data(SHAPE.iter().product(), 600);
    let mut assertions = AssertionCases::new();

    for axis in 0..4 {
        let (expected_flat, out_shape) = sum_axis_4d(&data, SHAPE, axis);
        let input_data = data.clone();
        assertions.push(
            fusor_conformance::assert(move |input: Tensor<4, f32>| async move {
                let actual: Tensor<3, f32> = match axis {
                    0 => input.sum::<3>(0),
                    1 => input.sum::<3>(1),
                    2 => input.sum::<3>(2),
                    _ => input.sum::<3>(3),
                };
                actual.to_concrete()
            })
            .arg(move |device: &Device| Tensor::from_slice(device, SHAPE, &input_data))
            .equal_to(move |input: Tensor<4, f32>| {
                let expected_flat = expected_flat.clone();
                async move { Tensor::from_slice(&input.device(), out_shape, &expected_flat) }
            })
            .compare_with(approx_compare::<3, f32>(1e-4))
            .runs(1)
            .into_case(format!(
                "rank_and_empty::rank4_sum_per_axis_matches_reference::axis{axis}"
            )),
        );
    }
    assertions
}

pub fn rank4_mean_axis0_matches_reference() -> AssertionCase {
    const SHAPE: [usize; 4] = [3, 2, 4, 5];
    let data = deterministic_data(SHAPE.iter().product(), 601);
    let (sum_flat, out_shape) = sum_axis_4d(&data, SHAPE, 0);
    let divisor = SHAPE[0] as f32;
    let expected_flat: Vec<f32> = sum_flat.iter().map(|v| v / divisor).collect();

    fusor_conformance::assert(async |input: Tensor<4, f32>| input.mean::<3>(0).to_concrete())
        .arg(move |device: &Device| Tensor::from_slice(device, SHAPE, &data))
        .equal_to(move |input: Tensor<4, f32>| {
            let expected_flat = expected_flat.clone();
            async move { Tensor::from_slice(&input.device(), out_shape, &expected_flat) }
        })
        .compare_with(approx_compare::<3, f32>(1e-4))
        .runs(1)
        .into_case("rank_and_empty::rank4_mean_axis0_matches_reference")
}

pub fn rank4_softmax_last_dim_matches_reference() -> AssertionCase {
    const SHAPE: [usize; 4] = [2, 2, 3, 8];
    let data = deterministic_data(SHAPE.iter().product(), 602);
    let expected_flat = softmax_last_dim_4d(&data, SHAPE);

    fusor_conformance::assert(async |input: Tensor<4, f32>| {
        input.softmax_last_dim::<3>().to_concrete()
    })
    .arg(move |device: &Device| Tensor::from_slice(device, SHAPE, &data))
    .equal_to(move |input: Tensor<4, f32>| {
        let expected_flat = expected_flat.clone();
        async move { Tensor::from_slice(&input.device(), SHAPE, &expected_flat) }
    })
    .compare_with(approx_compare::<4, f32>(1e-5))
    .runs(1)
    .into_case("rank_and_empty::rank4_softmax_last_dim_matches_reference")
}

pub fn rank4_rms_norm_fused_matches_reference() -> AssertionCase {
    const SHAPE: [usize; 4] = [2, 2, 3, 16];
    let data = deterministic_data(SHAPE.iter().product(), 603);
    let weight: Vec<f32> = (0..SHAPE[3]).map(|i| 1.0 + (i % 5) as f32 * 0.25).collect();
    let expected_flat = rms_norm_fused_4d(&data, SHAPE, &weight, 1e-5);

    fusor_conformance::assert(move |input: Tensor<4, f32>| {
        let weight = weight.clone();
        async move {
            let w: Tensor<1, f32> = Tensor::from_slice(&input.device(), [SHAPE[3]], &weight);
            input.rms_norm_fused::<1, 3>(&w, None, 1e-5).to_concrete()
        }
    })
    .arg(move |device: &Device| Tensor::from_slice(device, SHAPE, &data))
    .equal_to(move |input: Tensor<4, f32>| {
        let expected_flat = expected_flat.clone();
        async move { Tensor::from_slice(&input.device(), SHAPE, &expected_flat) }
    })
    .compare_with(approx_or_relative_compare::<4>(1e-4, 1e-4))
    .runs(1)
    .into_case("rank_and_empty::rank4_rms_norm_fused_matches_reference")
}

pub fn empty_tensor_elementwise_add_returns_empty() -> AssertionCase {
    // 0-sized leading dim — elementwise op must be well-defined on empty inputs.
    fusor_conformance::assert(async |device: Device| {
        let a: Tensor<2, f32> = Tensor::zeros(&device, [0, 6]);
        let b: Tensor<2, f32> = Tensor::zeros(&device, [0, 6]);
        a.add_::<2, 2, _>(&b).to_concrete()
    })
    .arg(|device: &Device| device.clone())
    .equal_to(async |device: Device| Tensor::<2, f32>::zeros(&device, [0, 6]))
    .compare_with(exact_compare::<2, f32>())
    .runs(1)
    .into_case("rank_and_empty::empty_tensor_elementwise_add_returns_empty")
}

pub fn empty_tensor_sum_along_zero_axis_returns_identity() -> AssertionCase {
    // Reducing over a 0-sized axis: sum-identity is 0, so each output element
    // must be exactly 0 on both backends.
    fusor_conformance::assert(async |input: Tensor<2, f32>| input.sum::<1>(0).to_concrete())
        .arg(|device: &Device| Tensor::<2, f32>::zeros(device, [0, 4]))
        .equal_to(async |input: Tensor<2, f32>| Tensor::<1, f32>::zeros(&input.device(), [4]))
        .compare_with(exact_compare::<1, f32>())
        .runs(1)
        .into_case("rank_and_empty::empty_tensor_sum_along_zero_axis_returns_identity")
}

pub fn rank4_max_min_match_reference() -> AssertionCases {
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

    let mut assertions = AssertionCases::new();

    assertions.push(
        fusor_conformance::assert(async |input: Tensor<4, f32>| input.max::<3>(3).to_concrete())
            .arg({
                let data = data.clone();
                move |device: &Device| Tensor::from_slice(device, SHAPE, &data)
            })
            .equal_to(move |input: Tensor<4, f32>| {
                let max_ref = max_ref.clone();
                async move { Tensor::from_slice(&input.device(), out_shape, &max_ref) }
            })
            .compare_with(approx_compare::<3, f32>(1e-6))
            .runs(1)
            .into_case("rank_and_empty::rank4_max_min_match_reference::max"),
    );

    assertions.push(
        fusor_conformance::assert(async |input: Tensor<4, f32>| input.min::<3>(3).to_concrete())
            .arg(move |device: &Device| Tensor::from_slice(device, SHAPE, &data))
            .equal_to(move |input: Tensor<4, f32>| {
                let min_ref = min_ref.clone();
                async move { Tensor::from_slice(&input.device(), out_shape, &min_ref) }
            })
            .compare_with(approx_compare::<3, f32>(1e-6))
            .runs(1)
            .into_case("rank_and_empty::rank4_max_min_match_reference::min"),
    );

    assertions
}
