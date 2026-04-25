//! Convolution operations that work on both CPU and GPU backends.

use crate::{ConcreteTensor, FloatOps, MatmulImpl, SimdElement, Tensor};
use fusor_core::{DataType, FloatDataType};
use fusor_types::SlidingWindow;

impl<const R: usize, D> Tensor<R, D>
where
    D: SimdElement + DataType + FloatDataType + FloatOps + Default,
{
    /// Pad a specific axis with zeros on both sides (symmetric).
    ///
    /// Equivalent to `pad_with_zeros(axis, padding, padding)`.
    pub fn pad_axis(&self, axis: usize, padding: usize) -> Self {
        self.pad_with_zeros(axis, padding, padding)
    }

    /// Pad a specific axis with zeros on left and right sides separately.
    pub fn pad_with_zeros(&self, axis: usize, left: usize, right: usize) -> Self {
        if left == 0 && right == 0 {
            return self.clone();
        }

        let shape = self.shape();
        let mut parts: Vec<Self> = Vec::new();

        if left > 0 {
            let mut pad_shape = shape;
            pad_shape[axis] = left;
            parts.push(Self::zeros(&self.device(), pad_shape));
        }
        parts.push(self.clone());
        if right > 0 {
            let mut pad_shape = shape;
            pad_shape[axis] = right;
            parts.push(Self::zeros(&self.device(), pad_shape));
        }

        super::cat(parts, axis)
    }
}

impl<const R: usize, D> Tensor<R, D>
where
    D: SimdElement
        + DataType
        + FloatDataType
        + FloatOps
        + Default
        + MatmulImpl
        + std::ops::Mul<Output = D>
        + std::ops::Add<Output = D>,
{
    fn bias_broadcast_shape(out_channels: usize) -> [usize; R] {
        std::array::from_fn(|axis| if axis == 1 { out_channels } else { 1 })
    }

    fn window_permutation<const R2: usize, const DIFF: usize>() -> [usize; R2] {
        std::array::from_fn(|index| {
            if index == 0 {
                0
            } else if index <= DIFF {
                index + 1
            } else if index == DIFF + 1 {
                1
            } else {
                index
            }
        })
    }

    fn output_permutation<const DIFF: usize>() -> [usize; R] {
        std::array::from_fn(|index| {
            if index == 0 {
                0
            } else if index == 1 {
                DIFF + 1
            } else {
                index - 1
            }
        })
    }

    /// Unified convolution method that handles different tensor formats:
    /// - Multi-channel convolution (R = 2 + DIFF): (batch, channels, ...spatial) format
    ///
    /// For 1D conv: R=3, DIFF=1 gives (batch, in_channels, length) -> (batch, out_channels, out_length)
    pub fn conv<const WEIGHT_RANK: usize, const DIFF: usize, const R2: usize>(
        &self,
        weight: &Tensor<WEIGHT_RANK, D, ConcreteTensor<D, WEIGHT_RANK>>,
        bias: Option<&Tensor<1, D, ConcreteTensor<D, 1>>>,
        padding: [usize; DIFF],
        strides: [usize; DIFF],
    ) -> Self
    where
        ConcreteTensor<D, R>: fusor_cpu::LargerRank<R2, DIFF, D>,
        fusor_core::Tensor<R, D>: fusor_core::LargerRank<DIFF, R2, D>,
        crate::MulOp: fusor_cpu::SimdBinaryOp<D>,
        crate::AddOp: fusor_cpu::SimdBinaryOp<D>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<D>,
    {
        // Extract dimensions
        let input_shape = self.shape();
        let weight_shape = weight.shape();
        let spatial_start = R - DIFF;

        // Multi-channel convolution: (batch, channels, ...spatial)
        assert_eq!(
            R,
            2 + DIFF,
            "Conv expects (batch, channels, ...spatial) format where R = 2 + DIFF"
        );
        let batch_axis = 0;
        let in_channels_axis = 1;

        let batch = input_shape[batch_axis];
        let in_channels = input_shape[in_channels_axis];
        let out_channels = weight_shape[0];

        // Weight shape is (out_channels, in_channels, ...kernel_dims)
        assert_eq!(
            weight_shape[1], in_channels,
            "Weight in_channels must match input in_channels"
        );

        // Step 1: Apply padding to the spatial dimensions (last DIFF dimensions)
        let padded = if padding.iter().any(|&p| p > 0) {
            let mut result = self.clone();
            for (i, padding) in padding.iter().copied().enumerate() {
                let axis = R - DIFF + i;
                if padding > 0 {
                    result = result.pad_axis(axis, padding);
                }
            }
            result
        } else {
            self.clone()
        };

        // Calculate output spatial dimensions
        let mut out_spatial_size = 1;
        for i in 0..DIFF {
            let padded_len = input_shape[spatial_start + i] + 2 * padding[i];
            let kernel_len = weight_shape[spatial_start + i];
            let out_len = (padded_len - kernel_len) / strides[i] + 1;
            out_spatial_size *= out_len;
        }

        // Step 2: Create sliding windows over the spatial dimensions
        let windows: [SlidingWindow; DIFF] = std::array::from_fn(|i| {
            let axis = R - DIFF + i;
            let kernel_size = weight_shape[spatial_start + i];
            SlidingWindow::new(axis, kernel_size, strides[i])
        });
        let windows_tensor: Tensor<R2, D, _> = padded.sliding_window_view(windows);

        // Step 3: Prepare for matmul by reshaping and transposing
        let kernel_size: usize = weight_shape[spatial_start..].iter().product();

        // Move the output spatial dimensions in front of channels so each output location
        // becomes one matmul row after flattening.
        let windows_transposed = windows_tensor.permute(Self::window_permutation::<R2, DIFF>());

        // Flatten to (batch * out_spatial_size, in_channels * kernel_size)
        let windows_flat: Tensor<2, D, _> =
            windows_transposed.reshape([batch * out_spatial_size, in_channels * kernel_size]);

        // Step 4: Reshape weight for matmul
        let weight_reshaped: Tensor<2, D, _> =
            weight.reshape([out_channels, in_channels * kernel_size]);
        // Transpose for matmul: (in_channels * kernel_size, out_channels)
        let weight_t = weight_reshaped.t();

        // Step 5: Matrix multiplication
        let output = windows_flat.mat_mul(&weight_t);

        // Step 6: Reshape and permute back to (batch, out_channels, ...out_spatial...)
        let output_reshaped: Tensor<R, D, _> = output.reshape(std::array::from_fn(|axis| {
            if axis == 0 {
                batch
            } else if axis <= DIFF {
                let spatial_axis = spatial_start + axis - 1;
                let padded_len = input_shape[spatial_axis] + 2 * padding[axis - 1];
                let kernel_len = weight_shape[spatial_axis];
                (padded_len - kernel_len) / strides[axis - 1] + 1
            } else {
                out_channels
            }
        }));
        let output_transposed = output_reshaped.permute(Self::output_permutation::<DIFF>());

        // Reshape to (batch, out_channels, ...out_spatial_dims...)
        let mut output_shape = input_shape;
        output_shape[in_channels_axis] = out_channels;
        for i in 0..DIFF {
            let padded_len = input_shape[spatial_start + i] + 2 * padding[i];
            let kernel_len = weight_shape[spatial_start + i];
            output_shape[spatial_start + i] = (padded_len - kernel_len) / strides[i] + 1;
        }
        let output_final = output_transposed.reshape(output_shape);

        // Step 7: Add bias if present
        if let Some(bias) = bias {
            // Bias shape: (out_channels,)
            // Broadcast along the channel axis, leaving batch/spatial dims singleton.
            let bias_reshaped = bias.reshape(Self::bias_broadcast_shape(out_channels));
            let bias_broadcast: Tensor<R, D, _> = bias_reshaped.broadcast_as(output_shape);
            output_final.add_(&bias_broadcast)
        } else {
            output_final.to_concrete()
        }
    }

    /// Grouped convolution lowered to a single sliding_window_view + batched matmul.
    /// Weight is in PyTorch grouped layout: `(out_channels, in_channels / groups, ...kernel)`.
    pub fn grouped_conv<const WEIGHT_RANK: usize, const DIFF: usize, const R2: usize>(
        &self,
        weight: &Tensor<WEIGHT_RANK, D, ConcreteTensor<D, WEIGHT_RANK>>,
        bias: Option<&Tensor<1, D, ConcreteTensor<D, 1>>>,
        padding: [usize; DIFF],
        strides: [usize; DIFF],
        groups: usize,
    ) -> Self
    where
        ConcreteTensor<D, R>: fusor_cpu::LargerRank<R2, DIFF, D>,
        fusor_core::Tensor<R, D>: fusor_core::LargerRank<DIFF, R2, D>,
        crate::MulOp: fusor_cpu::SimdBinaryOp<D>,
        crate::AddOp: fusor_cpu::SimdBinaryOp<D>,
        fusor_cpu::SumOp: fusor_cpu::SimdReduceOp<D>,
    {
        let input_shape = self.shape();
        let weight_shape = weight.shape();
        let spatial_start = R - DIFF;

        assert_eq!(R, 2 + DIFF);
        let batch = input_shape[0];
        let in_channels = input_shape[1];
        let out_channels = weight_shape[0];
        assert_eq!(in_channels % groups, 0);
        assert_eq!(out_channels % groups, 0);
        let in_ch_per_group = in_channels / groups;
        let out_ch_per_group = out_channels / groups;
        assert_eq!(weight_shape[1], in_ch_per_group);

        let padded = if padding.iter().any(|&p| p > 0) {
            let mut result = self.clone();
            for (i, padding) in padding.iter().copied().enumerate() {
                let axis = R - DIFF + i;
                if padding > 0 {
                    result = result.pad_axis(axis, padding);
                }
            }
            result
        } else {
            self.clone()
        };

        let mut out_spatial_size = 1;
        for i in 0..DIFF {
            let padded_len = input_shape[spatial_start + i] + 2 * padding[i];
            let kernel_len = weight_shape[spatial_start + i];
            let out_len = (padded_len - kernel_len) / strides[i] + 1;
            out_spatial_size *= out_len;
        }

        let windows: [SlidingWindow; DIFF] = std::array::from_fn(|i| {
            let axis = R - DIFF + i;
            let kernel_size = weight_shape[spatial_start + i];
            SlidingWindow::new(axis, kernel_size, strides[i])
        });
        let windows_tensor: Tensor<R2, D, _> = padded.sliding_window_view(windows);

        let kernel_size: usize = weight_shape[spatial_start..].iter().product();

        // Permute and flatten exactly like the groups=1 path. Materialize
        // before the rank-3 split so the channel-dim split is over actual
        // contiguous memory rather than a permuted strided view.
        let windows_transposed = windows_tensor.permute(Self::window_permutation::<R2, DIFF>());
        let windows_flat: Tensor<2, D, _> = windows_transposed
            .reshape([batch * out_spatial_size, in_channels * kernel_size])
            .to_concrete();

        // Split inner dim into (groups, in_ch_per_group * kernel_size).
        let windows_3d: Tensor<3, D, _> = windows_flat
            .reshape([
                batch * out_spatial_size,
                groups,
                in_ch_per_group * kernel_size,
            ])
            .to_concrete();
        let windows_grouped = windows_3d.transpose(0, 1).to_concrete();
        // (groups, batch * out_spatial, in_ch_per_group * kernel_size)

        // Weight: (out_channels, ipg, ...kernel) -> (groups, opg, ipg * kernel_size)
        let weight_grouped: Tensor<3, D, _> = weight
            .reshape([groups, out_ch_per_group, in_ch_per_group * kernel_size])
            .to_concrete();
        let weight_grouped_t = weight_grouped.transpose(1, 2).to_concrete();

        let output_grouped = windows_grouped.mat_mul(&weight_grouped_t).to_concrete();
        // (groups, batch * out_spatial, out_ch_per_group)

        let output_t = output_grouped.transpose(0, 1).to_concrete();
        let output: Tensor<2, D, _> = output_t
            .reshape([batch * out_spatial_size, out_channels])
            .to_concrete();

        let output_reshaped: Tensor<R, D, _> = output.reshape(std::array::from_fn(|axis| {
            if axis == 0 {
                batch
            } else if axis <= DIFF {
                let spatial_axis = spatial_start + axis - 1;
                let padded_len = input_shape[spatial_axis] + 2 * padding[axis - 1];
                let kernel_len = weight_shape[spatial_axis];
                (padded_len - kernel_len) / strides[axis - 1] + 1
            } else {
                out_channels
            }
        }));
        let output_transposed = output_reshaped.permute(Self::output_permutation::<DIFF>());

        let mut output_shape = input_shape;
        output_shape[1] = out_channels;
        for i in 0..DIFF {
            let padded_len = input_shape[spatial_start + i] + 2 * padding[i];
            let kernel_len = weight_shape[spatial_start + i];
            output_shape[spatial_start + i] = (padded_len - kernel_len) / strides[i] + 1;
        }
        let output_final = output_transposed.reshape(output_shape);

        if let Some(bias) = bias {
            let bias_reshaped = bias.reshape(Self::bias_broadcast_shape(out_channels));
            let bias_broadcast: Tensor<R, D, _> = bias_reshaped.broadcast_as(output_shape);
            output_final.add_(&bias_broadcast)
        } else {
            output_final.to_concrete()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pad_with_zeros_asymmetric_cpu() {
        // Asymmetric padding: 2 zeros on the left, 1 on the right, axis=1.
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input: Tensor<2, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([2, 3], &data));
        let padded = input.pad_with_zeros(1, 2, 1);

        assert_eq!(padded.shape(), [2, 6]);
        let result = padded.as_slice().await.unwrap();
        let expected = [
            [0.0, 0.0, 1.0, 2.0, 3.0, 0.0],
            [0.0, 0.0, 4.0, 5.0, 6.0, 0.0],
        ];
        for r in 0..2 {
            for c in 0..6 {
                assert_eq!(result[[r, c]], expected[r][c], "mismatch at [{r},{c}]");
            }
        }
    }

    #[tokio::test]
    async fn test_pad_with_zeros_left_only_cpu() {
        let data: Vec<f32> = vec![7.0, 8.0, 9.0];
        let input: Tensor<1, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([3], &data));
        let padded = input.pad_with_zeros(0, 3, 0);

        assert_eq!(padded.shape(), [6]);
        let result = padded.as_slice().await.unwrap();
        for (i, &expect) in [0.0, 0.0, 0.0, 7.0, 8.0, 9.0].iter().enumerate() {
            assert_eq!(result[[i]], expect, "left-only pad mismatch at {i}");
        }
    }

    #[tokio::test]
    async fn test_pad_with_zeros_right_only_cpu() {
        let data: Vec<f32> = vec![1.0, 2.0];
        let input: Tensor<1, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([2], &data));
        let padded = input.pad_with_zeros(0, 0, 4);

        assert_eq!(padded.shape(), [6]);
        let result = padded.as_slice().await.unwrap();
        for (i, &expect) in [1.0, 2.0, 0.0, 0.0, 0.0, 0.0].iter().enumerate() {
            assert_eq!(result[[i]], expect, "right-only pad mismatch at {i}");
        }
    }

    #[tokio::test]
    async fn test_pad_with_zeros_zero_returns_self_cpu() {
        let data: Vec<f32> = vec![5.0, 6.0, 7.0];
        let input: Tensor<1, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([3], &data));
        let padded = input.pad_with_zeros(0, 0, 0);
        assert_eq!(padded.shape(), [3]);
        let result = padded.as_slice().await.unwrap();
        for (i, &expect) in [5.0, 6.0, 7.0].iter().enumerate() {
            assert_eq!(result[[i]], expect, "zero-pad should be identity at {i}");
        }
    }

    #[tokio::test]
    async fn test_conv_1d_cpu() {
        // Input: (batch=1, in_channels=1, length=5)
        let input_data = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let input: Tensor<3, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 5], &input_data));

        // Weight: (out_channels=1, in_channels=1, kernel_size=3)
        let weight_data = [0.2f32, 0.5, 0.3];
        let weight: Tensor<3, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 3], &weight_data));

        let bias_val = 0.1f32;
        let bias: Tensor<1, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([1], &[bias_val]));

        // Perform convolution with stride 1 and no padding
        let output = input.conv(&weight, Some(&bias), [0], [1]);

        // Expected values for the 1D convolution
        let input_flat = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let weight_flat = [0.2f32, 0.5, 0.3];
        let expected: Vec<f32> = input_flat
            .windows(weight_flat.len())
            .map(|window| {
                window
                    .iter()
                    .zip(weight_flat.iter())
                    .map(|(x, w)| x * w)
                    .sum::<f32>()
                    + bias_val
            })
            .collect();

        let output_data = output.as_slice().await.unwrap();
        assert_eq!(output_data.shape(), &[1, 1, expected.len()]);
        for i in 0..expected.len() {
            let val = output_data[[0, 0, i]];
            let expected_val = expected[i];
            assert!(
                (val - expected_val).abs() < 1e-5,
                "Mismatch at index {}: got {}, expected {}",
                i,
                val,
                expected_val
            );
        }
    }

    #[tokio::test]
    async fn test_conv_1d_strided_cpu() {
        // Input: (batch=1, in_channels=1, length=5)
        let input_data = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let input: Tensor<3, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 5], &input_data));

        // Weight: (out_channels=1, in_channels=1, kernel_size=3)
        let weight_data = [0.2f32, 0.5, 0.3];
        let weight: Tensor<3, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 3], &weight_data));

        let bias_val = 0.1f32;
        let bias: Tensor<1, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([1], &[bias_val]));
        let stride = 2;

        let output = input.conv(&weight, Some(&bias), [0], [stride]);

        let input_flat = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let weight_flat = [0.2f32, 0.5, 0.3];
        let expected: Vec<f32> = input_flat
            .windows(weight_flat.len())
            .step_by(stride)
            .map(|window| {
                window
                    .iter()
                    .zip(weight_flat.iter())
                    .map(|(x, w)| x * w)
                    .sum::<f32>()
                    + bias_val
            })
            .collect();

        let output_data = output.as_slice().await.unwrap();
        assert_eq!(output_data.shape(), &[1, 1, expected.len()]);
        for i in 0..expected.len() {
            let val = output_data[[0, 0, i]];
            let expected_val = expected[i];
            assert!(
                (val - expected_val).abs() < 1e-5,
                "Mismatch at index {}: got {}, expected {}",
                i,
                val,
                expected_val
            );
        }
    }

    #[tokio::test]
    async fn test_conv_1d_with_padding_cpu() {
        // Input: (1, 1, 3)
        let input_data = [1.0f32, 2.0, 3.0];
        let input: Tensor<3, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 3], &input_data));

        // Weight: (1, 1, 3)
        let weight_data = [1.0f32, 1.0, 1.0];
        let weight: Tensor<3, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 3], &weight_data));

        let output = input.conv(&weight, None, [1], [1]);

        // With padding=1, input becomes [0, 1, 2, 3, 0]
        // Output shape should be (1, 1, 3)
        assert_eq!(output.shape(), [1, 1, 3]);

        let result = output.as_slice().await.unwrap();

        // Manual calculation:
        // output[0] = 0*1 + 1*1 + 2*1 = 3
        // output[1] = 1*1 + 2*1 + 3*1 = 6
        // output[2] = 2*1 + 3*1 + 0*1 = 5

        assert!((result[[0, 0, 0]] - 3.0).abs() < 1e-5);
        assert!((result[[0, 0, 1]] - 6.0).abs() < 1e-5);
        assert!((result[[0, 0, 2]] - 5.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_conv_2d_simple_cpu() {
        // Input: (batch=1, in_channels=1, height=4, width=4)
        let input_data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let input: Tensor<4, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 4, 4], &input_data));

        // Weight: (out_channels=1, in_channels=1, kH=3, kW=3) - all ones
        let weight_data = vec![1.0f32; 9];
        let weight: Tensor<4, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 3, 3], &weight_data));

        let output = input.conv(&weight, None, [0, 0], [1, 1]);
        assert_eq!(output.shape(), [1, 1, 2, 2]);

        let result = output.as_slice().await.unwrap();

        // Input is:
        //  0  1  2  3
        //  4  5  6  7
        //  8  9 10 11
        // 12 13 14 15
        //
        // With 3x3 kernel of all 1s:
        // [0,0]: 0+1+2+4+5+6+8+9+10 = 45
        // [0,1]: 1+2+3+5+6+7+9+10+11 = 54
        // [1,0]: 4+5+6+8+9+10+12+13+14 = 81
        // [1,1]: 5+6+7+9+10+11+13+14+15 = 90

        assert!(
            (result[[0, 0, 0, 0]] - 45.0).abs() < 1e-4,
            "got {} expected 45",
            result[[0, 0, 0, 0]]
        );
        assert!(
            (result[[0, 0, 0, 1]] - 54.0).abs() < 1e-4,
            "got {} expected 54",
            result[[0, 0, 0, 1]]
        );
        assert!(
            (result[[0, 0, 1, 0]] - 81.0).abs() < 1e-4,
            "got {} expected 81",
            result[[0, 0, 1, 0]]
        );
        assert!(
            (result[[0, 0, 1, 1]] - 90.0).abs() < 1e-4,
            "got {} expected 90",
            result[[0, 0, 1, 1]]
        );
    }

    #[tokio::test]
    async fn test_conv_2d_multi_channel_cpu() {
        // Input: (batch=1, in_channels=2, height=3, width=3)
        // Channel 0: [[1,2,3],[4,5,6],[7,8,9]]
        // Channel 1: [[10,20,30],[40,50,60],[70,80,90]]
        let input_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, // ch0
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, // ch1
        ];
        let input: Tensor<4, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 2, 3, 3], &input_data));

        // Weight: (out_channels=1, in_channels=2, kH=2, kW=2)
        // For ch0: [[1, 0], [0, 0]]
        // For ch1: [[0, 0], [0, 1]]
        let weight_data: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, // ch0 kernel
            0.0, 0.0, 0.0, 1.0, // ch1 kernel
        ];
        let weight: Tensor<4, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 2, 2, 2], &weight_data));

        let output = input.conv(&weight, None, [0, 0], [1, 1]);
        assert_eq!(output.shape(), [1, 1, 2, 2]);

        let result = output.as_slice().await.unwrap();

        // Each output = ch0[top_left] * 1 + ch1[bottom_right] * 1
        // [0,0]: ch0[0,0]*1 + ch1[1,1]*1 = 1 + 50 = 51
        // [0,1]: ch0[0,1]*1 + ch1[1,2]*1 = 2 + 60 = 62
        // [1,0]: ch0[1,0]*1 + ch1[2,1]*1 = 4 + 80 = 84
        // [1,1]: ch0[1,1]*1 + ch1[2,2]*1 = 5 + 90 = 95

        assert!(
            (result[[0, 0, 0, 0]] - 51.0).abs() < 1e-4,
            "got {} expected 51",
            result[[0, 0, 0, 0]]
        );
        assert!(
            (result[[0, 0, 0, 1]] - 62.0).abs() < 1e-4,
            "got {} expected 62",
            result[[0, 0, 0, 1]]
        );
        assert!(
            (result[[0, 0, 1, 0]] - 84.0).abs() < 1e-4,
            "got {} expected 84",
            result[[0, 0, 1, 0]]
        );
        assert!(
            (result[[0, 0, 1, 1]] - 95.0).abs() < 1e-4,
            "got {} expected 95",
            result[[0, 0, 1, 1]]
        );
    }

    #[tokio::test]
    async fn test_conv_2d_strided_cpu() {
        // Input: (batch=1, in_channels=1, height=4, width=4)
        let input_data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let input: Tensor<4, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 4, 4], &input_data));

        // Weight: (out_channels=1, in_channels=1, kH=2, kW=2) - all ones
        let weight_data = vec![1.0f32; 4];
        let weight: Tensor<4, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 2, 2], &weight_data));

        // Stride 2: output should be (1, 1, 2, 2) since (4-2)/2+1 = 2
        let output = input.conv(&weight, None, [0, 0], [2, 2]);
        assert_eq!(output.shape(), [1, 1, 2, 2]);

        let result = output.as_slice().await.unwrap();

        // Input is:
        //  0  1  2  3
        //  4  5  6  7
        //  8  9 10 11
        // 12 13 14 15
        //
        // With 2x2 kernel of all 1s, stride 2:
        // [0,0]: 0+1+4+5 = 10 (window at (0,0))
        // [0,1]: 2+3+6+7 = 18 (window at (0,2))
        // [1,0]: 8+9+12+13 = 42 (window at (2,0))
        // [1,1]: 10+11+14+15 = 50 (window at (2,2))

        assert!(
            (result[[0, 0, 0, 0]] - 10.0).abs() < 1e-4,
            "got {} expected 10",
            result[[0, 0, 0, 0]]
        );
        assert!(
            (result[[0, 0, 0, 1]] - 18.0).abs() < 1e-4,
            "got {} expected 18",
            result[[0, 0, 0, 1]]
        );
        assert!(
            (result[[0, 0, 1, 0]] - 42.0).abs() < 1e-4,
            "got {} expected 42",
            result[[0, 0, 1, 0]]
        );
        assert!(
            (result[[0, 0, 1, 1]] - 50.0).abs() < 1e-4,
            "got {} expected 50",
            result[[0, 0, 1, 1]]
        );
    }

    #[tokio::test]
    async fn test_conv_2d_strided_multi_channel_cpu() {
        // Input: (batch=1, in_channels=2, height=4, width=4)
        let mut input_data: Vec<f32> = (0..16).map(|i| i as f32).collect(); // ch0
        input_data.extend((0..16).map(|i| (i as f32) * 10.0)); // ch1
        let input: Tensor<4, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 2, 4, 4], &input_data));

        // Weight: (out_channels=1, in_channels=2, kH=2, kW=2)
        // ch0 kernel: [[1,0],[0,0]], ch1 kernel: [[0,0],[0,1]]
        let weight_data: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, // ch0 kernel
            0.0, 0.0, 0.0, 1.0, // ch1 kernel
        ];
        let weight: Tensor<4, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 2, 2, 2], &weight_data));

        // Stride 2
        let output = input.conv(&weight, None, [0, 0], [2, 2]);
        assert_eq!(output.shape(), [1, 1, 2, 2]);

        let result = output.as_slice().await.unwrap();

        // Ch0: [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]
        // Ch1: [[0,10,20,30],[40,50,60,70],[80,90,100,110],[120,130,140,150]]
        //
        // With stride=2:
        // [0,0]: ch0[0,0]*1 + ch1[1,1]*1 = 0 + 50 = 50  (window at rows 0:2, cols 0:2)
        // [0,1]: ch0[0,2]*1 + ch1[1,3]*1 = 2 + 70 = 72  (window at rows 0:2, cols 2:4)
        // [1,0]: ch0[2,0]*1 + ch1[3,1]*1 = 8 + 130 = 138 (window at rows 2:4, cols 0:2)
        // [1,1]: ch0[2,2]*1 + ch1[3,3]*1 = 10 + 150 = 160 (window at rows 2:4, cols 2:4)

        assert!(
            (result[[0, 0, 0, 0]] - 50.0).abs() < 1e-4,
            "got {} expected 50",
            result[[0, 0, 0, 0]]
        );
        assert!(
            (result[[0, 0, 0, 1]] - 72.0).abs() < 1e-4,
            "got {} expected 72",
            result[[0, 0, 0, 1]]
        );
        assert!(
            (result[[0, 0, 1, 0]] - 138.0).abs() < 1e-4,
            "got {} expected 138",
            result[[0, 0, 1, 0]]
        );
        assert!(
            (result[[0, 0, 1, 1]] - 160.0).abs() < 1e-4,
            "got {} expected 160",
            result[[0, 0, 1, 1]]
        );
    }

    #[tokio::test]
    async fn test_conv_2d_depthwise_cpu() {
        use crate::layers::{ConvNd, ConvNdConfig};

        // Input: (batch=1, channels=2, height=4, width=4)
        // Channel 0: 0..15, Channel 1: 100..115
        let mut input_data = Vec::new();
        for i in 0..16 {
            input_data.push(i as f32);
        }
        for i in 0..16 {
            input_data.push(100.0 + i as f32);
        }
        let input: Tensor<4, f32, ConcreteTensor<f32, 4>> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 2, 4, 4], &input_data));

        // Depthwise conv: weight (2, 1, 3, 3)
        // Channel 0 kernel: all ones -> sum 3x3 window
        // Channel 1 kernel: center-only (identity-like for 3x3)
        let mut weight_data = vec![1.0f32; 9]; // ch0: all ones
        weight_data.extend_from_slice(&[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]); // ch1: center only
        let weight: Tensor<4, f32, ConcreteTensor<f32, 4>> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([2, 1, 3, 3], &weight_data));

        let config = ConvNdConfig::<2> {
            padding: [0, 0],
            stride: [1, 1],
            groups: 2,
        };
        let conv = ConvNd::<2, 4, _>::new(weight, None, config);
        let output = conv.forward(&input);
        assert_eq!(output.shape(), [1, 2, 2, 2]);

        let result = output.as_slice().await.unwrap();

        // Channel 0 with all-ones kernel (same as test_conv_2d_simple):
        // [0,0]: 0+1+2+4+5+6+8+9+10 = 45
        // [0,1]: 1+2+3+5+6+7+9+10+11 = 54
        // [1,0]: 4+5+6+8+9+10+12+13+14 = 81
        // [1,1]: 5+6+7+9+10+11+13+14+15 = 90
        assert!(
            (result[[0, 0, 0, 0]] - 45.0).abs() < 1e-4,
            "ch0[0,0] got {} expected 45",
            result[[0, 0, 0, 0]]
        );
        assert!(
            (result[[0, 0, 0, 1]] - 54.0).abs() < 1e-4,
            "ch0[0,1] got {} expected 54",
            result[[0, 0, 0, 1]]
        );
        assert!(
            (result[[0, 0, 1, 0]] - 81.0).abs() < 1e-4,
            "ch0[1,0] got {} expected 81",
            result[[0, 0, 1, 0]]
        );
        assert!(
            (result[[0, 0, 1, 1]] - 90.0).abs() < 1e-4,
            "ch0[1,1] got {} expected 90",
            result[[0, 0, 1, 1]]
        );

        // Channel 1 with center-only kernel: picks out center element of each 3x3 window
        // Input ch1: [[100,101,102,103],[104,105,106,107],[108,109,110,111],[112,113,114,115]]
        // [0,0]: center of (0:3,0:3) = 105
        // [0,1]: center of (0:3,1:4) = 106
        // [1,0]: center of (1:4,0:3) = 109
        // [1,1]: center of (1:4,1:4) = 110
        assert!(
            (result[[0, 1, 0, 0]] - 105.0).abs() < 1e-4,
            "ch1[0,0] got {} expected 105",
            result[[0, 1, 0, 0]]
        );
        assert!(
            (result[[0, 1, 0, 1]] - 106.0).abs() < 1e-4,
            "ch1[0,1] got {} expected 106",
            result[[0, 1, 0, 1]]
        );
        assert!(
            (result[[0, 1, 1, 0]] - 109.0).abs() < 1e-4,
            "ch1[1,0] got {} expected 109",
            result[[0, 1, 1, 0]]
        );
        assert!(
            (result[[0, 1, 1, 1]] - 110.0).abs() < 1e-4,
            "ch1[1,1] got {} expected 110",
            result[[0, 1, 1, 1]]
        );
    }

    #[tokio::test]
    async fn test_conv_2d_depthwise_with_padding_cpu() {
        use crate::layers::{ConvNd, ConvNdConfig};

        // Input: (batch=1, channels=2, height=3, width=3)
        let input_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, // ch1
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0,
        ];
        let input: Tensor<4, f32, ConcreteTensor<f32, 4>> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 2, 3, 3], &input_data));

        // Weight (2, 1, 3, 3): all ones for both channels
        let weight_data = vec![1.0f32; 18]; // 2 * 9
        let weight: Tensor<4, f32, ConcreteTensor<f32, 4>> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([2, 1, 3, 3], &weight_data));

        let config = ConvNdConfig::<2> {
            padding: [1, 1],
            stride: [1, 1],
            groups: 2,
        };
        let conv = ConvNd::<2, 4, _>::new(weight, None, config);
        let output = conv.forward(&input);
        assert_eq!(output.shape(), [1, 2, 3, 3]);

        let result = output.as_slice().await.unwrap();

        // Channel 0 with padding=1 and all-ones 3x3:
        // padded: [[0,0,0,0,0],[0,1,2,3,0],[0,4,5,6,0],[0,7,8,9,0],[0,0,0,0,0]]
        // [0,0]: 0+0+0+0+1+2+0+4+5 = 12
        // [1,1]: 1+2+3+4+5+6+7+8+9 = 45 (center, no padding effect)
        assert!(
            (result[[0, 0, 0, 0]] - 12.0).abs() < 1e-4,
            "ch0[0,0] got {} expected 12",
            result[[0, 0, 0, 0]]
        );
        assert!(
            (result[[0, 0, 1, 1]] - 45.0).abs() < 1e-4,
            "ch0[1,1] got {} expected 45",
            result[[0, 0, 1, 1]]
        );

        // Channel 1: same but 10x values
        // [0,0]: 0+0+0+0+10+20+0+40+50 = 120
        // [1,1]: 10+20+30+40+50+60+70+80+90 = 450
        assert!(
            (result[[0, 1, 0, 0]] - 120.0).abs() < 1e-4,
            "ch1[0,0] got {} expected 120",
            result[[0, 1, 0, 0]]
        );
        assert!(
            (result[[0, 1, 1, 1]] - 450.0).abs() < 1e-4,
            "ch1[1,1] got {} expected 450",
            result[[0, 1, 1, 1]]
        );
    }

    #[tokio::test]
    async fn test_conv_1d_multi_channel_cpu() {
        // Input: (1, 2, 4) - 2 input channels
        // Channel 0: [1, 2, 3, 4], Channel 1: [5, 6, 7, 8]
        let input_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let input: Tensor<3, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 2, 4], &input_data));

        // Weight: (3, 2, 2) - 3 output channels, 2 input channels, kernel size 2
        // out_ch 0: [[1, 0], [0, 1]]
        // out_ch 1: [[0.5, 0.5], [0.5, 0.5]]
        // out_ch 2: [[1, 1], [1, 1]]
        let weight_data = [
            1.0f32, 0.0, 0.0, 1.0, // out_channel 0
            0.5, 0.5, 0.5, 0.5, // out_channel 1
            1.0, 1.0, 1.0, 1.0, // out_channel 2
        ];
        let weight: Tensor<3, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([3, 2, 2], &weight_data));

        let output = input.conv(&weight, None, [0], [1]);

        // Output shape should be (1, 3, 3)
        assert_eq!(output.shape(), [1, 3, 3]);

        let result = output.as_slice().await.unwrap();

        // For position 0: in_ch0 window [1,2], in_ch1 window [5,6]
        //   out_ch 0 weights [[1,0], [0,1]]: 1*1 + 2*0 + 5*0 + 6*1 = 7
        //   out_ch 1 weights [[0.5,0.5], [0.5,0.5]]: 1*0.5 + 2*0.5 + 5*0.5 + 6*0.5 = 7
        //   out_ch 2 weights [[1,1], [1,1]]: 1*1 + 2*1 + 5*1 + 6*1 = 14

        // Out channel 0
        assert!((result[[0, 0, 0]] - 7.0).abs() < 1e-5);
        assert!((result[[0, 0, 1]] - 9.0).abs() < 1e-5);
        assert!((result[[0, 0, 2]] - 11.0).abs() < 1e-5);

        // Out channel 1
        assert!((result[[0, 1, 0]] - 7.0).abs() < 1e-5);
        assert!((result[[0, 1, 1]] - 9.0).abs() < 1e-5);
        assert!((result[[0, 1, 2]] - 11.0).abs() < 1e-5);

        // Out channel 2
        assert!((result[[0, 2, 0]] - 14.0).abs() < 1e-5);
        assert!((result[[0, 2, 1]] - 18.0).abs() < 1e-5);
        assert!((result[[0, 2, 2]] - 22.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_conv_2d_bias_channel_dim() {
        // Regression test: bias must be added along the channel dim (axis 1),
        // not the last spatial dim. Use out_channels=3 with spatial dims 2x2
        // so that out_channels != any spatial dim.
        let input_data: Vec<f32> = vec![0.0; 4 * 4];
        let input: Tensor<4, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 4, 4], &input_data));

        // Weight: (out_channels=3, in_channels=1, kH=3, kW=3) — all zeros
        let weight_data = vec![0.0f32; 3 * 3 * 3];
        let weight: Tensor<4, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([3, 1, 3, 3], &weight_data));

        // Bias: [10, 20, 30] — one per output channel
        let bias: Tensor<1, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([3], &[10.0f32, 20.0, 30.0]));

        let output = input.conv(&weight, Some(&bias), [0, 0], [1, 1]);
        // Output shape: (1, 3, 2, 2)
        assert_eq!(output.shape(), [1, 3, 2, 2]);

        let result = output.as_slice().await.unwrap();

        // With zero input and zero weights, output should be just the bias
        // per channel, broadcast over all spatial positions.
        for h in 0..2 {
            for w in 0..2 {
                assert!(
                    (result[[0, 0, h, w]] - 10.0).abs() < 1e-5,
                    "ch0[{},{}] got {} expected 10",
                    h,
                    w,
                    result[[0, 0, h, w]]
                );
                assert!(
                    (result[[0, 1, h, w]] - 20.0).abs() < 1e-5,
                    "ch1[{},{}] got {} expected 20",
                    h,
                    w,
                    result[[0, 1, h, w]]
                );
                assert!(
                    (result[[0, 2, h, w]] - 30.0).abs() < 1e-5,
                    "ch2[{},{}] got {} expected 30",
                    h,
                    w,
                    result[[0, 2, h, w]]
                );
            }
        }
    }

    /// Compares `grouped_conv` on GPU against the per-group narrow + conv + cat
    /// reference on CPU. Mirrors TinyViT MBConv conv2: depthwise (groups=channels),
    /// 3x3 kernel, padding=1, stride=1.
    #[tokio::test]
    async fn test_grouped_conv_gpu_vs_per_group_reference() {
        use crate::Device;

        let groups = 8usize;
        let ipg = 1usize;
        let opg = 1usize;
        let in_channels = groups * ipg;
        let out_channels = groups * opg;
        let h = 7usize;
        let w = 9usize;
        let kh = 3usize;
        let kw = 3usize;

        let weight_data: Vec<f32> = (0..out_channels * ipg * kh * kw)
            .map(|i| (i as f32 * 0.05).sin() * 0.4)
            .collect();
        let bias_data: Vec<f32> = (0..out_channels).map(|i| i as f32 * 0.1 - 0.3).collect();
        let input_data: Vec<f32> = (0..in_channels * h * w)
            .map(|i| (i as f32 * 0.07).cos() * 0.5)
            .collect();

        // Reference on CPU using per-group narrow + conv + cat.
        let weight_cpu: Tensor<4, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice(
            [out_channels, ipg, kh, kw],
            &weight_data,
        ));
        let bias_cpu: Tensor<1, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([out_channels], &bias_data));
        let input_cpu: Tensor<4, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice(
            [1, in_channels, h, w],
            &input_data,
        ));
        let mut group_outputs: Vec<Tensor<4, f32, ConcreteTensor<f32, 4>>> =
            Vec::with_capacity(groups);
        for g in 0..groups {
            let input_slice = input_cpu.narrow(1, g * ipg, ipg).to_concrete();
            let weight_slice = weight_cpu.narrow(0, g * opg, opg).to_concrete();
            let group_out: Tensor<4, f32, ConcreteTensor<f32, 4>> = input_slice.conv(
                &weight_slice,
                None::<&Tensor<1, f32, ConcreteTensor<f32, 1>>>,
                [1, 1],
                [1, 1],
            );
            group_outputs.push(group_out);
        }
        let cat: Tensor<4, f32> = Tensor::cat(group_outputs, 1);
        let cat_shape = cat.shape();
        let bias_reshaped = bias_cpu.reshape([1, out_channels, 1, 1]);
        let bias_b = bias_reshaped.broadcast_as(cat_shape);
        let reference: Tensor<4, f32, ConcreteTensor<f32, 4>> = (cat + bias_b).to_concrete();
        let reference_slice = reference.as_slice().await.unwrap();

        // Now run grouped_conv on GPU.
        let gpu = Device::new().await.expect("GPU required for this test");
        let weight_gpu: Tensor<4, f32> =
            Tensor::from_slice(&gpu, [out_channels, ipg, kh, kw], &weight_data);
        let bias_gpu: Tensor<1, f32> = Tensor::from_slice(&gpu, [out_channels], &bias_data);
        let input_gpu: Tensor<4, f32> =
            Tensor::from_slice(&gpu, [1, in_channels, h, w], &input_data);
        let actual: Tensor<4, f32, ConcreteTensor<f32, 4>> =
            input_gpu.grouped_conv(&weight_gpu, Some(&bias_gpu), [1, 1], [1, 1], groups);
        let actual_slice = actual.as_slice().await.unwrap();

        assert_eq!(actual.shape(), reference.shape());
        let [_, oc, oh, ow] = actual.shape();
        let mut max_diff = 0.0f32;
        for c in 0..oc {
            for i in 0..oh {
                for j in 0..ow {
                    let a: f32 = actual_slice[[0, c, i, j]].into();
                    let r: f32 = reference_slice[[0, c, i, j]].into();
                    let d = (a - r).abs();
                    if d > max_diff {
                        eprintln!("[{c},{i},{j}] gpu={a} cpu_ref={r} diff={d}");
                    }
                    max_diff = max_diff.max(d);
                }
            }
        }
        assert!(
            max_diff < 1e-3,
            "grouped_conv GPU vs per-group reference diverged: max_diff={max_diff}"
        );
    }

    /// Larger depthwise case mirroring SAM TinyViT MBConv ConvLayer0 dims.
    /// Uses groups=256 (matching MBCONV_EXPAND_RATIO * 64) but a smaller
    /// spatial extent to keep the test fast.
    #[tokio::test]
    async fn test_grouped_conv_gpu_vs_per_group_reference_large_groups() {
        use crate::Device;

        let groups = 256usize;
        let ipg = 1usize;
        let opg = 1usize;
        let in_channels = groups * ipg;
        let out_channels = groups * opg;
        let h = 16usize;
        let w = 16usize;
        let kh = 3usize;
        let kw = 3usize;

        let weight_data: Vec<f32> = (0..out_channels * ipg * kh * kw)
            .map(|i| (i as f32 * 0.013).sin() * 0.4)
            .collect();
        let input_data: Vec<f32> = (0..in_channels * h * w)
            .map(|i| (i as f32 * 0.0091).cos() * 0.5)
            .collect();

        let weight_cpu: Tensor<4, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice(
            [out_channels, ipg, kh, kw],
            &weight_data,
        ));
        let input_cpu: Tensor<4, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice(
            [1, in_channels, h, w],
            &input_data,
        ));
        let mut group_outputs: Vec<Tensor<4, f32, ConcreteTensor<f32, 4>>> =
            Vec::with_capacity(groups);
        for g in 0..groups {
            let input_slice = input_cpu.narrow(1, g * ipg, ipg).to_concrete();
            let weight_slice = weight_cpu.narrow(0, g * opg, opg).to_concrete();
            let group_out: Tensor<4, f32, ConcreteTensor<f32, 4>> = input_slice.conv(
                &weight_slice,
                None::<&Tensor<1, f32, ConcreteTensor<f32, 1>>>,
                [1, 1],
                [1, 1],
            );
            group_outputs.push(group_out);
        }
        let reference: Tensor<4, f32, ConcreteTensor<f32, 4>> =
            Tensor::cat(group_outputs, 1).to_concrete();
        let reference_slice = reference.as_slice().await.unwrap();

        let gpu = Device::new().await.expect("GPU required for this test");
        let weight_gpu: Tensor<4, f32> =
            Tensor::from_slice(&gpu, [out_channels, ipg, kh, kw], &weight_data);
        let input_gpu: Tensor<4, f32> =
            Tensor::from_slice(&gpu, [1, in_channels, h, w], &input_data);
        let actual: Tensor<4, f32, ConcreteTensor<f32, 4>> = input_gpu.grouped_conv(
            &weight_gpu,
            None::<&Tensor<1, f32, ConcreteTensor<f32, 1>>>,
            [1, 1],
            [1, 1],
            groups,
        );
        let actual_slice = actual.as_slice().await.unwrap();

        assert_eq!(actual.shape(), reference.shape());
        let [_, oc, oh, ow] = actual.shape();
        let mut max_diff = 0.0f32;
        let mut first_mismatch: Option<(usize, usize, usize, f32, f32)> = None;
        for c in 0..oc {
            for i in 0..oh {
                for j in 0..ow {
                    let a: f32 = actual_slice[[0, c, i, j]].into();
                    let r: f32 = reference_slice[[0, c, i, j]].into();
                    let d = (a - r).abs();
                    if d > 1e-3 && first_mismatch.is_none() {
                        first_mismatch = Some((c, i, j, a, r));
                    }
                    if d > max_diff {
                        max_diff = d;
                    }
                }
            }
        }
        assert!(
            max_diff < 1e-3,
            "large-groups grouped_conv GPU vs reference diverged: max_diff={max_diff}, first_mismatch={first_mismatch:?}"
        );
    }

    /// Bisects whether the bug is in the bmm itself with strided A, or in
    /// the surrounding conv layout chain.
    #[tokio::test]
    async fn test_batched_matmul_strided_a_at_scale() {
        use crate::Device;
        let gpu = Device::new().await.expect("GPU required");

        let groups = 64usize;
        let k = 9usize;
        for m in [256usize, 1024, 4096, 16384, 65536] {
            let a_data: Vec<f32> = (0..m * groups * k)
                .map(|i| (i as f32 * 0.0091).cos() * 0.5)
                .collect();
            let b_data: Vec<f32> = (0..groups * k)
                .map(|i| (i as f32 * 0.013).sin() * 0.4)
                .collect();

            let a_flat_gpu: Tensor<2, f32> = Tensor::from_slice(&gpu, [m, groups * k], &a_data);
            let a_3d_gpu = a_flat_gpu.reshape([m, groups, k]).to_concrete();
            let a_grouped_gpu = a_3d_gpu.transpose(0, 1).to_concrete();

            let b_gpu: Tensor<3, f32> = Tensor::from_slice(&gpu, [groups, k, 1], &b_data);
            let out_gpu = a_grouped_gpu.mat_mul(&b_gpu).to_concrete();
            let out_slice = out_gpu.as_slice().await.unwrap();

            let a_flat_cpu: Tensor<2, f32> =
                Tensor::Cpu(fusor_cpu::Tensor::from_slice([m, groups * k], &a_data));
            let a_3d_cpu: Tensor<3, f32> = a_flat_cpu.reshape([m, groups, k]).to_concrete();
            let a_grouped_cpu = a_3d_cpu.transpose(0, 1).to_concrete();
            let b_cpu: Tensor<3, f32> =
                Tensor::Cpu(fusor_cpu::Tensor::from_slice([groups, k, 1], &b_data));
            let out_cpu = a_grouped_cpu.mat_mul(&b_cpu).to_concrete();
            let ref_slice = out_cpu.as_slice().await.unwrap();

            let mut max_diff = 0.0f32;
            let mut first: Option<(usize, usize, f32, f32)> = None;
            for g in 0..groups {
                for mi in 0..m {
                    let a: f32 = out_slice[[g, mi, 0]].into();
                    let r: f32 = ref_slice[[g, mi, 0]].into();
                    let d = (a - r).abs();
                    if d > 1e-3 && first.is_none() {
                        first = Some((g, mi, a, r));
                    }
                    max_diff = max_diff.max(d);
                }
            }
            eprintln!("m={m} max_diff={max_diff} first_mismatch={first:?}");
            assert!(
                max_diff < 1e-3,
                "bmm strided a failed at m={m}: max_diff={max_diff}, first={first:?}"
            );
        }
    }

    /// Same chain as grouped_conv up to the matmul, then matmuls with a tensor
    /// constructed exactly like weight_grouped_t. Isolates whether the bug is in
    /// the matmul's handling of an A input that's deep in a lazy graph.
    #[tokio::test]
    async fn test_grouped_conv_bmm_isolated() {
        use crate::Device;

        let groups = 64usize;
        let h = 200usize;
        let w = 200usize;
        let kh = 3usize;
        let kw = 3usize;
        let pad = 1usize;
        let in_channels = groups;
        let kernel_size = kh * kw;
        let out_h = h + 2 * pad - kh + 1;
        let out_w = w + 2 * pad - kw + 1;
        let out_spatial = out_h * out_w;

        let input_data: Vec<f32> = (0..in_channels * h * w)
            .map(|i| (i as f32 * 0.0091).cos() * 0.5)
            .collect();
        let weight_data: Vec<f32> = (0..groups * kernel_size)
            .map(|i| (i as f32 * 0.013).sin() * 0.4)
            .collect();

        let gpu = Device::new().await.expect("GPU required");
        let input_gpu: Tensor<4, f32> =
            Tensor::from_slice(&gpu, [1, in_channels, h, w], &input_data);
        let weight_gpu: Tensor<4, f32> =
            Tensor::from_slice(&gpu, [groups, 1, kh, kw], &weight_data);

        let padded_gpu = input_gpu.pad_axis(2, pad).pad_axis(3, pad);
        let windows_gpu = padded_gpu
            .sliding_window_view([SlidingWindow::new(2, kh, 1), SlidingWindow::new(3, kw, 1)]);
        let permuted_gpu = windows_gpu.permute([0, 2, 3, 1, 4, 5]);
        let flat_gpu: Tensor<2, f32> = permuted_gpu
            .reshape([out_spatial, in_channels * kernel_size])
            .to_concrete();
        let windows_3d_gpu = flat_gpu
            .reshape([out_spatial, groups, kernel_size])
            .to_concrete();
        let windows_grouped_gpu = windows_3d_gpu.transpose(0, 1).to_concrete();
        let weight_grouped_gpu = weight_gpu.reshape([groups, 1, kernel_size]).to_concrete();
        let weight_grouped_t_gpu = weight_grouped_gpu.transpose(1, 2).to_concrete();

        let bmm_out_gpu = windows_grouped_gpu
            .mat_mul(&weight_grouped_t_gpu)
            .to_concrete();
        // Mirror the post-matmul rearrangement that grouped_conv does.
        let output_t_gpu = bmm_out_gpu.transpose(0, 1).to_concrete();
        let output_2d_gpu: Tensor<2, f32> =
            output_t_gpu.reshape([out_spatial, groups]).to_concrete();
        let output_4d_gpu: Tensor<4, f32> = output_2d_gpu
            .reshape([1, out_h, out_w, groups])
            .to_concrete();
        let output_permuted_gpu = output_4d_gpu.permute([0, 3, 1, 2]).to_concrete();
        // grouped_conv reshapes the permuted output to its target shape (which
        // is identical) — replicate that here.
        let output_final_gpu: Tensor<4, f32> = output_permuted_gpu
            .reshape([1, groups, out_h, out_w])
            .to_concrete();
        let bmm_flat_gpu: Tensor<1, f32> = output_final_gpu
            .reshape([groups * out_spatial])
            .to_concrete();
        let actual_slice = bmm_flat_gpu.as_slice().await.unwrap();
        let actual = actual_slice.as_slice();

        // CPU reference via per-group narrow + conv.
        let weight_cpu: Tensor<4, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice(
            [groups, 1, kh, kw],
            &weight_data,
        ));
        let input_cpu: Tensor<4, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice(
            [1, in_channels, h, w],
            &input_data,
        ));
        let mut group_outputs: Vec<Tensor<4, f32, ConcreteTensor<f32, 4>>> =
            Vec::with_capacity(groups);
        for g in 0..groups {
            let input_slice = input_cpu.narrow(1, g, 1).to_concrete();
            let weight_slice = weight_cpu.narrow(0, g, 1).to_concrete();
            let group_out = input_slice.conv(
                &weight_slice,
                None::<&Tensor<1, f32, ConcreteTensor<f32, 1>>>,
                [pad, pad],
                [1, 1],
            );
            group_outputs.push(group_out);
        }
        let cat: Tensor<4, f32> = Tensor::cat(group_outputs, 1).to_concrete();
        let cat_slice = cat.as_slice().await.unwrap();

        let mut max_diff = 0.0f32;
        let mut first: Option<(usize, usize, usize, f32, f32)> = None;
        for c in 0..groups {
            for hi in 0..out_h {
                for wi in 0..out_w {
                    let mi = hi * out_w + wi;
                    let actual_idx = c * out_spatial + mi;
                    let a = actual[actual_idx];
                    let r: f32 = cat_slice[[0, c, hi, wi]].into();
                    let d = (a - r).abs();
                    if d > 1e-3 && first.is_none() {
                        first = Some((c, hi, wi, a, r));
                    }
                    max_diff = max_diff.max(d);
                }
            }
        }
        assert!(
            max_diff < 1e-3,
            "isolated bmm chain diverged: max_diff={max_diff}, first={first:?}"
        );

        // Now run the actual grouped_conv on the SAME inputs and compare to
        // the manual chain we just verified. This pins down where grouped_conv
        // diverges from the manual decomposition.
        let gc_out: Tensor<4, f32> = input_gpu.grouped_conv(
            &weight_gpu,
            None::<&Tensor<1, f32, ConcreteTensor<f32, 1>>>,
            [pad, pad],
            [1, 1],
            groups,
        );
        let gc_flat: Tensor<1, f32> = gc_out.reshape([groups * out_spatial]).to_concrete();
        let gc_slice = gc_flat.as_slice().await.unwrap();
        let gc = gc_slice.as_slice();

        let mut gc_max = 0.0f32;
        let mut gc_first: Option<(usize, usize, usize, f32, f32)> = None;
        for c in 0..groups {
            for hi in 0..out_h {
                for wi in 0..out_w {
                    let actual_idx = c * out_spatial + hi * out_w + wi;
                    let manual = actual[actual_idx];
                    let gc_val = gc[actual_idx];
                    let d = (gc_val - manual).abs();
                    if d > 1e-3 && gc_first.is_none() {
                        gc_first = Some((c, hi, wi, gc_val, manual));
                    }
                    gc_max = gc_max.max(d);
                }
            }
        }
        assert!(
            gc_max < 1e-3,
            "grouped_conv diverges from manual chain: max_diff={gc_max}, first={gc_first:?}"
        );
    }

    /// Materializes `windows_grouped` (the rank-3 BMM A input from grouped_conv)
    /// to CPU and compares element-wise with a CPU-side construction. Isolates
    /// whether the bug is in the windows pipeline (sliding_window_view +
    /// permute + reshape + transpose) or in the matmul itself.
    #[tokio::test]
    async fn test_windows_grouped_at_sam_scale() {
        use crate::Device;

        let groups = 64usize;
        let h = 200usize;
        let w = 200usize;
        let kh = 3usize;
        let kw = 3usize;
        let pad = 1usize;
        let in_channels = groups;
        let out_h = h + 2 * pad - kh + 1;
        let out_w = w + 2 * pad - kw + 1;
        let out_spatial = out_h * out_w;

        let input_data: Vec<f32> = (0..in_channels * h * w)
            .map(|i| (i as f32 * 0.0091).cos() * 0.5)
            .collect();

        let gpu = Device::new().await.expect("GPU required");
        let input_gpu: Tensor<4, f32> =
            Tensor::from_slice(&gpu, [1, in_channels, h, w], &input_data);
        let padded_gpu = input_gpu.pad_axis(2, pad).pad_axis(3, pad);
        let windows_gpu = padded_gpu
            .sliding_window_view([SlidingWindow::new(2, kh, 1), SlidingWindow::new(3, kw, 1)]);
        let permuted_gpu = windows_gpu.permute([0, 2, 3, 1, 4, 5]);
        let flat_gpu: Tensor<2, f32> = permuted_gpu
            .reshape([out_spatial, in_channels * kh * kw])
            .to_concrete();
        let windows_3d_gpu = flat_gpu
            .reshape([out_spatial, groups, kh * kw])
            .to_concrete();
        let windows_grouped_gpu = windows_3d_gpu.transpose(0, 1).to_concrete();
        // Materialize via as_slice — the only way to actually read GPU data.
        let windows_grouped_flat: Tensor<1, f32> = windows_grouped_gpu
            .reshape([groups * out_spatial * kh * kw])
            .to_concrete();
        let actual_slice = windows_grouped_flat.as_slice().await.unwrap();
        let actual = actual_slice.as_slice();

        // CPU reference: sliding window indexed manually.
        let mut expected = vec![0.0f32; groups * out_spatial * kh * kw];
        for c in 0..in_channels {
            for oh_i in 0..out_h {
                for ow_i in 0..out_w {
                    for ki in 0..kh {
                        for kj in 0..kw {
                            let in_h = oh_i + ki;
                            let in_w = ow_i + kj;
                            let in_h_p = in_h as isize - pad as isize;
                            let in_w_p = in_w as isize - pad as isize;
                            let val = if in_h_p >= 0
                                && in_h_p < h as isize
                                && in_w_p >= 0
                                && in_w_p < w as isize
                            {
                                input_data[c * h * w + in_h_p as usize * w + in_w_p as usize]
                            } else {
                                0.0
                            };
                            // After transpose to (groups, out_spatial, kh*kw):
                            let out_idx = c * out_spatial * kh * kw
                                + (oh_i * out_w + ow_i) * kh * kw
                                + ki * kw
                                + kj;
                            expected[out_idx] = val;
                        }
                    }
                }
            }
        }

        let mut max_diff = 0.0f32;
        let mut first_mismatch: Option<(usize, f32, f32)> = None;
        for i in 0..actual.len() {
            let a = actual[i];
            let e = expected[i];
            let d = (a - e).abs();
            if d > 1e-3 && first_mismatch.is_none() {
                first_mismatch = Some((i, a, e));
            }
            max_diff = max_diff.max(d);
        }
        assert!(
            max_diff < 1e-3,
            "windows_grouped at SAM scale diverged: max_diff={max_diff}, first_mismatch={first_mismatch:?}"
        );
    }

    /// SAM-scale depthwise: 256 channels at 64×64 spatial. Mirrors what SAM's
    /// TinyViT MBConv depthwise conv actually runs. Reproduces the bug that
    /// breaks SAM masks while smaller-scale tests above pass.
    #[tokio::test]
    async fn test_grouped_conv_gpu_sam_scale_depthwise() {
        use crate::Device;

        let groups = 64usize;
        let h = 200usize;
        let w = 200usize;
        let kh = 3usize;
        let kw = 3usize;

        let weight_data: Vec<f32> = (0..groups * kh * kw)
            .map(|i| (i as f32 * 0.013).sin() * 0.4)
            .collect();
        let input_data: Vec<f32> = (0..groups * h * w)
            .map(|i| (i as f32 * 0.0091).cos() * 0.5)
            .collect();

        let weight_cpu: Tensor<4, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice(
            [groups, 1, kh, kw],
            &weight_data,
        ));
        let input_cpu: Tensor<4, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice(
            [1, groups, h, w],
            &input_data,
        ));
        let mut group_outputs: Vec<Tensor<4, f32, ConcreteTensor<f32, 4>>> =
            Vec::with_capacity(groups);
        for g in 0..groups {
            let input_slice = input_cpu.narrow(1, g, 1).to_concrete();
            let weight_slice = weight_cpu.narrow(0, g, 1).to_concrete();
            let group_out: Tensor<4, f32, ConcreteTensor<f32, 4>> = input_slice.conv(
                &weight_slice,
                None::<&Tensor<1, f32, ConcreteTensor<f32, 1>>>,
                [1, 1],
                [1, 1],
            );
            group_outputs.push(group_out);
        }
        let reference: Tensor<4, f32, ConcreteTensor<f32, 4>> =
            Tensor::cat(group_outputs, 1).to_concrete();
        let reference_slice = reference.as_slice().await.unwrap();

        let gpu = Device::new().await.expect("GPU required for this test");

        let weight_gpu: Tensor<4, f32> =
            Tensor::from_slice(&gpu, [groups, 1, kh, kw], &weight_data);
        let input_gpu: Tensor<4, f32> = Tensor::from_slice(&gpu, [1, groups, h, w], &input_data);
        let actual: Tensor<4, f32, ConcreteTensor<f32, 4>> = input_gpu.grouped_conv(
            &weight_gpu,
            None::<&Tensor<1, f32, ConcreteTensor<f32, 1>>>,
            [1, 1],
            [1, 1],
            groups,
        );
        let actual_slice = actual.as_slice().await.unwrap();

        assert_eq!(actual.shape(), reference.shape());
        let [_, oc, oh, ow] = actual.shape();
        let mut max_diff = 0.0f32;
        let mut first_mismatch: Option<(usize, usize, usize, f32, f32)> = None;
        for c in 0..oc {
            for i in 0..oh {
                for j in 0..ow {
                    let a: f32 = actual_slice[[0, c, i, j]].into();
                    let r: f32 = reference_slice[[0, c, i, j]].into();
                    let d = (a - r).abs();
                    if d > 1e-3 && first_mismatch.is_none() {
                        first_mismatch = Some((c, i, j, a, r));
                    }
                    if d > max_diff {
                        max_diff = d;
                    }
                }
            }
        }
        assert!(
            max_diff < 1e-3,
            "SAM-scale grouped_conv GPU vs reference diverged: max_diff={max_diff}, first_mismatch={first_mismatch:?}"
        );
    }
}
