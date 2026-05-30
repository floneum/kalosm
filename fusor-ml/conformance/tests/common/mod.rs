#![allow(dead_code)]

pub mod quantized;

use fusor::{DataType, Device, SimdElement, Tensor};
use fusor_conformance::{approx_compare, available_devices, exact_compare};

pub async fn assert_approx_tensors<const R: usize>(
    actual: Tensor<R, f32>,
    expected: Tensor<R, f32>,
    tol: f32,
) {
    fusor_conformance::assert(async |actual: Tensor<R, f32>, _expected: Tensor<R, f32>| actual)
        .arg({
            let actual = actual.clone();
            move |_: &Device| actual.clone()
        })
        .arg({
            let expected = expected.clone();
            move |_: &Device| expected.clone()
        })
        .equal_to(async |_actual: Tensor<R, f32>, expected: Tensor<R, f32>| expected)
        .compare_with(approx_compare::<R, f32>(tol))
        .devices([Device::Cpu])
        .await
        .unwrap();
}

pub async fn assert_exact_tensors<const R: usize, T>(actual: Tensor<R, T>, expected: Tensor<R, T>)
where
    T: DataType + SimdElement + PartialEq + 'static,
{
    fusor_conformance::assert(async |actual: Tensor<R, T>, _expected: Tensor<R, T>| actual)
        .arg({
            let actual = actual.clone();
            move |_: &Device| actual.clone()
        })
        .arg({
            let expected = expected.clone();
            move |_: &Device| expected.clone()
        })
        .equal_to(async |_actual: Tensor<R, T>, expected: Tensor<R, T>| expected)
        .compare_with(exact_compare::<R, T>())
        .devices([Device::Cpu])
        .await
        .unwrap();
}

pub async fn assert_approx_devices<const R: usize>(
    actual: impl Fn(&Device) -> Tensor<R, f32>,
    expected: impl Fn(&Device) -> Tensor<R, f32>,
    tol: f32,
) {
    for device in available_devices().await {
        assert_approx_tensors(actual(&device), expected(&device), tol).await;
    }
}

pub async fn assert_exact_devices<const R: usize, T>(
    actual: impl Fn(&Device) -> Tensor<R, T>,
    expected: impl Fn(&Device) -> Tensor<R, T>,
) where
    T: DataType + SimdElement + PartialEq + 'static,
{
    for device in available_devices().await {
        assert_exact_tensors(actual(&device), expected(&device)).await;
    }
}

pub async fn assert_approx_cpu<const R: usize>(
    actual: impl Fn(&Device) -> Tensor<R, f32>,
    expected: impl Fn(&Device) -> Tensor<R, f32>,
    tol: f32,
) {
    assert_approx_tensors(actual(&Device::Cpu), expected(&Device::Cpu), tol).await;
}

pub fn flatten2(input: &[Vec<f32>]) -> Vec<f32> {
    input.iter().flat_map(|row| row.iter().copied()).collect()
}

pub fn flatten3(input: &[Vec<Vec<f32>>]) -> Vec<f32> {
    input
        .iter()
        .flat_map(|matrix| matrix.iter().flat_map(|row| row.iter().copied()))
        .collect()
}

pub fn reshape2(flat: &[f32], shape: [usize; 2]) -> Vec<Vec<f32>> {
    flat.chunks(shape[1])
        .take(shape[0])
        .map(|row| row.to_vec())
        .collect()
}

pub fn reshape3(flat: &[f32], shape: [usize; 3]) -> Vec<Vec<Vec<f32>>> {
    let stride = shape[1] * shape[2];
    flat.chunks(stride)
        .take(shape[0])
        .map(|matrix| reshape2(matrix, [shape[1], shape[2]]))
        .collect()
}

pub fn reshape4(flat: &[f32], shape: [usize; 4]) -> Vec<Vec<Vec<Vec<f32>>>> {
    let stride = shape[1] * shape[2] * shape[3];
    flat.chunks(stride)
        .take(shape[0])
        .map(|tensor3| reshape3(tensor3, [shape[1], shape[2], shape[3]]))
        .collect()
}

pub fn transpose2(input: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let rows = input.len();
    let cols = input[0].len();
    let mut out = vec![vec![0.0; rows]; cols];
    for (row, values) in input.iter().enumerate() {
        for (col, value) in values.iter().enumerate() {
            out[col][row] = *value;
        }
    }
    out
}

pub fn permute3(input: &[Vec<Vec<f32>>], axes: [usize; 3]) -> Vec<Vec<Vec<f32>>> {
    let shape = [input.len(), input[0].len(), input[0][0].len()];
    let out_shape = [shape[axes[0]], shape[axes[1]], shape[axes[2]]];
    let mut out = vec![vec![vec![0.0; out_shape[2]]; out_shape[1]]; out_shape[0]];
    for (i, plane) in out.iter_mut().enumerate() {
        for (j, row) in plane.iter_mut().enumerate() {
            for (k, slot) in row.iter_mut().enumerate() {
                let mut src = [0usize; 3];
                src[axes[0]] = i;
                src[axes[1]] = j;
                src[axes[2]] = k;
                *slot = input[src[0]][src[1]][src[2]];
            }
        }
    }
    out
}

pub fn slice2(
    input: &[Vec<f32>],
    rows: std::ops::Range<usize>,
    cols: std::ops::Range<usize>,
) -> Vec<Vec<f32>> {
    input[rows]
        .iter()
        .map(|row| row[cols.clone()].to_vec())
        .collect()
}

pub fn broadcast_1d_to_2d(input: &[f32], rows: usize) -> Vec<Vec<f32>> {
    (0..rows).map(|_| input.to_vec()).collect()
}

pub fn repeat2(input: &[Vec<f32>], repeats: [usize; 2]) -> Vec<Vec<f32>> {
    let rows = (0..repeats[0])
        .flat_map(|_| input.iter().cloned())
        .collect::<Vec<_>>();

    rows.into_iter()
        .map(|row| {
            (0..repeats[1])
                .flat_map(|_| row.iter().copied())
                .collect::<Vec<_>>()
        })
        .collect()
}

pub fn resize2(input: &[Vec<f32>], new_shape: [usize; 2]) -> Vec<Vec<f32>> {
    let mut out = vec![vec![0.0; new_shape[1]]; new_shape[0]];
    let rows = input.len().min(new_shape[0]);
    let cols = input[0].len().min(new_shape[1]);
    for row in 0..rows {
        for col in 0..cols {
            out[row][col] = input[row][col];
        }
    }
    out
}

pub fn sliding_window_1d_ncw(
    input: &[Vec<Vec<f32>>],
    size: usize,
    stride: usize,
) -> Vec<Vec<Vec<Vec<f32>>>> {
    let batch = input.len();
    let channels = input[0].len();
    let out_len = (input[0][0].len() - size) / stride + 1;
    let mut out = vec![vec![vec![vec![0.0; size]; out_len]; channels]; batch];
    for (b, batch_out) in out.iter_mut().enumerate() {
        for (c, channel_out) in batch_out.iter_mut().enumerate() {
            for (out_idx, window) in channel_out.iter_mut().enumerate() {
                let start = out_idx * stride;
                for (i, slot) in window.iter_mut().enumerate() {
                    *slot = input[b][c][start + i];
                }
            }
        }
    }
    out
}

pub fn unary_map2(input: &[Vec<f32>], f: impl Fn(f32) -> f32) -> Vec<Vec<f32>> {
    input
        .iter()
        .map(|row| row.iter().copied().map(&f).collect())
        .collect()
}

pub fn binary_map2(
    lhs: &[Vec<f32>],
    rhs: &[Vec<f32>],
    f: impl Fn(f32, f32) -> f32,
) -> Vec<Vec<f32>> {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(lhs_row, rhs_row)| {
            lhs_row
                .iter()
                .copied()
                .zip(rhs_row.iter().copied())
                .map(|(l, r)| f(l, r))
                .collect()
        })
        .collect()
}

pub fn broadcast_binary_2d_1d(
    lhs: &[Vec<f32>],
    rhs: &[f32],
    f: impl Fn(f32, f32) -> f32,
) -> Vec<Vec<f32>> {
    lhs.iter()
        .map(|row| {
            row.iter()
                .copied()
                .zip(rhs.iter().copied())
                .map(|(l, r)| f(l, r))
                .collect()
        })
        .collect()
}

pub fn compare_scalar_map2(
    input: &[Vec<f32>],
    scalar: f32,
    f: impl Fn(f32, f32) -> bool,
) -> Vec<Vec<f32>> {
    unary_map2(input, |value| if f(value, scalar) { 1.0 } else { 0.0 })
}

pub fn compare_tensor_map2(
    lhs: &[Vec<f32>],
    rhs: &[Vec<f32>],
    f: impl Fn(f32, f32) -> bool,
) -> Vec<Vec<f32>> {
    binary_map2(lhs, rhs, |l, r| if f(l, r) { 1.0 } else { 0.0 })
}

pub fn where_cond2(
    cond: &[Vec<f32>],
    on_true: &[Vec<f32>],
    on_false: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    cond.iter()
        .zip(on_true.iter())
        .zip(on_false.iter())
        .map(|((cond_row, true_row), false_row)| {
            cond_row
                .iter()
                .copied()
                .zip(true_row.iter().copied())
                .zip(false_row.iter().copied())
                .map(
                    |((cond, when_true), when_false)| {
                        if cond != 0.0 { when_true } else { when_false }
                    },
                )
                .collect()
        })
        .collect()
}

pub fn reduce_axis2(
    input: &[Vec<f32>],
    axis: usize,
    init: f32,
    f: impl Fn(f32, f32) -> f32 + Copy,
) -> Vec<f32> {
    match axis {
        0 => {
            let cols = input[0].len();
            (0..cols)
                .map(|col| input.iter().fold(init, |acc, row| f(acc, row[col])))
                .collect()
        }
        1 => input
            .iter()
            .map(|row| row.iter().copied().fold(init, f))
            .collect(),
        _ => panic!("unsupported axis"),
    }
}

pub fn keepdim2(values: &[f32], axis: usize) -> Vec<Vec<f32>> {
    match axis {
        0 => vec![values.to_vec()],
        1 => values.iter().copied().map(|value| vec![value]).collect(),
        _ => panic!("unsupported axis"),
    }
}

pub fn mean_axis2(input: &[Vec<f32>], axis: usize) -> Vec<f32> {
    let divisor = if axis == 0 {
        input.len()
    } else {
        input[0].len()
    } as f32;
    reduce_axis2(input, axis, 0.0, |acc, value| acc + value)
        .into_iter()
        .map(|value| value / divisor)
        .collect()
}

pub fn var_axis2(input: &[Vec<f32>], axis: usize) -> Vec<f32> {
    let mean = mean_axis2(input, axis);
    match axis {
        0 => {
            let cols = input[0].len();
            (0..cols)
                .map(|col| {
                    input
                        .iter()
                        .map(|row| {
                            let diff = row[col] - mean[col];
                            diff * diff
                        })
                        .sum::<f32>()
                        / input.len() as f32
                })
                .collect()
        }
        1 => input
            .iter()
            .zip(mean.iter())
            .map(|(row, mean)| {
                row.iter()
                    .map(|value| {
                        let diff = *value - *mean;
                        diff * diff
                    })
                    .sum::<f32>()
                    / row.len() as f32
            })
            .collect(),
        _ => panic!("unsupported axis"),
    }
}

pub fn softmax_last_dim_2d(input: &[Vec<f32>]) -> Vec<Vec<f32>> {
    input
        .iter()
        .map(|row| {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|value| (*value - max).exp()).collect();
            let sum = exps.iter().sum::<f32>();
            exps.into_iter().map(|value| value / sum).collect()
        })
        .collect()
}

pub fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

pub fn gelu(value: f32) -> f32 {
    0.5 * value
        * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (value + 0.044_715 * value.powi(3))).tanh())
}

pub fn rms_norm_last_dim_3d(
    input: &[Vec<Vec<f32>>],
    weight: &[f32],
    eps: f32,
) -> Vec<Vec<Vec<f32>>> {
    input
        .iter()
        .map(|matrix| {
            matrix
                .iter()
                .map(|row| {
                    let mean_sq =
                        row.iter().map(|value| value * value).sum::<f32>() / row.len() as f32;
                    let denom = (mean_sq + eps).sqrt();
                    row.iter()
                        .copied()
                        .zip(weight.iter().copied())
                        .map(|(value, weight)| (value / denom) * weight)
                        .collect()
                })
                .collect()
        })
        .collect()
}

pub fn layer_norm_last_dim_3d(
    input: &[Vec<Vec<f32>>],
    weight: &[f32],
    bias: &[f32],
    eps: f32,
) -> Vec<Vec<Vec<f32>>> {
    input
        .iter()
        .map(|matrix| {
            matrix
                .iter()
                .map(|row| {
                    let mean = row.iter().sum::<f32>() / row.len() as f32;
                    let variance = row
                        .iter()
                        .map(|value| {
                            let diff = *value - mean;
                            diff * diff
                        })
                        .sum::<f32>()
                        / row.len() as f32;
                    let denom = (variance + eps).sqrt();
                    row.iter()
                        .copied()
                        .zip(weight.iter().copied())
                        .zip(bias.iter().copied())
                        .map(|((value, weight), bias)| ((value - mean) / denom) * weight + bias)
                        .collect()
                })
                .collect()
        })
        .collect()
}

pub fn matmul2(lhs: &[Vec<f32>], rhs: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let rows = lhs.len();
    let inner = lhs[0].len();
    let cols = rhs[0].len();
    let mut out = vec![vec![0.0; cols]; rows];
    for row in 0..rows {
        for col in 0..cols {
            let mut acc = 0.0;
            for idx in 0..inner {
                acc += lhs[row][idx] * rhs[idx][col];
            }
            out[row][col] = acc;
        }
    }
    out
}

pub fn conv1d_ncw(
    input: &[Vec<Vec<f32>>],
    weight: &[Vec<Vec<f32>>],
    bias: Option<&[f32]>,
    padding: usize,
    stride: usize,
) -> Vec<Vec<Vec<f32>>> {
    let batch = input.len();
    let out_channels = weight.len();
    let in_channels = input[0].len();
    let kernel = weight[0][0].len();
    let padded_len = input[0][0].len() + 2 * padding;
    let out_len = (padded_len - kernel) / stride + 1;
    let mut out = vec![vec![vec![0.0; out_len]; out_channels]; batch];
    for (b, batch_out) in out.iter_mut().enumerate() {
        for (oc, channel_out) in batch_out.iter_mut().enumerate() {
            for (out_idx, slot) in channel_out.iter_mut().enumerate() {
                let start = out_idx * stride;
                let mut acc = bias.map_or(0.0, |bias| bias[oc]);
                for ic in 0..in_channels {
                    for (k, w) in weight[oc][ic].iter().enumerate() {
                        let src = start + k;
                        let value = if src < padding || src >= padding + input[b][ic].len() {
                            0.0
                        } else {
                            input[b][ic][src - padding]
                        };
                        acc += value * w;
                    }
                }
                *slot = acc;
            }
        }
    }
    out
}

pub fn pool1d_ncw(
    input: &[Vec<Vec<f32>>],
    size: usize,
    stride: usize,
    reduce: impl Fn(f32, f32) -> f32 + Copy,
    init: f32,
) -> Vec<Vec<Vec<f32>>> {
    let batch = input.len();
    let channels = input[0].len();
    let out_len = (input[0][0].len() - size) / stride + 1;
    let mut out = vec![vec![vec![0.0; out_len]; channels]; batch];
    for (b, batch_out) in out.iter_mut().enumerate() {
        for (c, channel_out) in batch_out.iter_mut().enumerate() {
            for (out_idx, slot) in channel_out.iter_mut().enumerate() {
                let start = out_idx * stride;
                *slot = input[b][c][start..start + size]
                    .iter()
                    .copied()
                    .fold(init, reduce);
            }
        }
    }
    out
}

pub fn slice_assign2(
    input: &[Vec<f32>],
    row_range: std::ops::Range<usize>,
    col_range: std::ops::Range<usize>,
    patch: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let mut out = input.to_vec();
    for (dst_row, patch_row) in row_range.zip(patch.iter()) {
        for (dst_col, value) in col_range.clone().zip(patch_row.iter()) {
            out[dst_row][dst_col] = *value;
        }
    }
    out
}

pub fn index_select2(input: &[Vec<f32>], dimension: usize, indices: &[u32]) -> Vec<Vec<f32>> {
    match dimension {
        0 => indices
            .iter()
            .map(|index| input[*index as usize].clone())
            .collect(),
        1 => input
            .iter()
            .map(|row| indices.iter().map(|index| row[*index as usize]).collect())
            .collect(),
        _ => panic!("unsupported dimension"),
    }
}

pub fn rope_normal_4d(
    input: &[Vec<Vec<Vec<f32>>>],
    cos: &[Vec<f32>],
    sin: &[Vec<f32>],
) -> Vec<Vec<Vec<Vec<f32>>>> {
    let mut out = input.to_vec();
    for batch in 0..input.len() {
        for head in 0..input[batch].len() {
            for position in 0..input[batch][head].len() {
                let row = &input[batch][head][position];
                let half = row.len() / 2;
                for idx in 0..half {
                    let x0 = row[idx];
                    let x1 = row[idx + half];
                    let c = cos[position][idx];
                    let s = sin[position][idx];
                    out[batch][head][position][idx] = x0 * c - x1 * s;
                    out[batch][head][position][idx + half] = x1 * c + x0 * s;
                }
            }
        }
    }
    out
}

pub fn index_select1(input: &[f32], indices: &[u32]) -> Vec<f32> {
    indices.iter().map(|index| input[*index as usize]).collect()
}

pub fn where_cond1(cond: &[f32], on_true: &[f32], on_false: &[f32]) -> Vec<f32> {
    cond.iter()
        .copied()
        .zip(on_true.iter().copied())
        .zip(on_false.iter().copied())
        .map(|((c, t), f)| if c != 0.0 { t } else { f })
        .collect()
}

pub fn rope_interleaved_4d(
    input: &[Vec<Vec<Vec<f32>>>],
    cos: &[Vec<f32>],
    sin: &[Vec<f32>],
) -> Vec<Vec<Vec<Vec<f32>>>> {
    let mut out = input.to_vec();
    for batch in 0..input.len() {
        for head in 0..input[batch].len() {
            for position in 0..input[batch][head].len() {
                let row = &input[batch][head][position];
                for pair in 0..(row.len() / 2) {
                    let x0 = row[pair * 2];
                    let x1 = row[pair * 2 + 1];
                    let c = cos[position][pair];
                    let s = sin[position][pair];
                    out[batch][head][position][pair * 2] = x0 * c - x1 * s;
                    out[batch][head][position][pair * 2 + 1] = x0 * s + x1 * c;
                }
            }
        }
    }
    out
}
