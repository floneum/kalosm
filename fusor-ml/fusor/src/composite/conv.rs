//! Convolution operations that work on both CPU and GPU backends.

use crate::{ConcreteTensor, FloatOps, MatmulImpl, SimdElement, Tensor};
use fusor_core::{DataType, FloatDataType};
use fusor_types::SlidingWindow;

impl<const R: usize, D> Tensor<R, D>
where
    D: SimdElement + DataType + FloatDataType + FloatOps + Default,
{
    /// Pad a specific axis with zeros on both sides.
    fn pad_axis(&self, axis: usize, padding: usize) -> Self {
        if padding == 0 {
            return self.clone();
        }

        let shape = self.shape();

        // Create left padding shape
        let mut pad_shape = shape;
        pad_shape[axis] = padding;
        let pad_left = Self::zeros(&self.device(), pad_shape);
        let pad_right = Self::zeros(&self.device(), pad_shape);

        // Concatenate: [pad_left, self, pad_right] along the axis
        super::cat([pad_left, self.clone(), pad_right], axis)
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
    /// Unified convolution method that handles different tensor formats:
    /// - Multi-channel convolution (R = 2 + DIFF): (batch, channels, ...spatial) format
    ///
    /// For Conv1d: R=3, DIFF=1 gives (batch, in_channels, length) -> (batch, out_channels, out_length)
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

        // Transpose to move in_channels after spatial
        let windows_transposed = windows_tensor.transpose(in_channels_axis, spatial_start);

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

        // Step 6: Reshape and transpose back to (batch, out_channels, ...out_spatial...)
        let output_reshaped: Tensor<3, D, _> =
            output.reshape([batch, out_spatial_size, out_channels]);
        let output_transposed = output_reshaped.transpose(in_channels_axis, spatial_start);

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
            // Need to broadcast to (batch, out_channels, ...spatial...)
            // Broadcast bias to the FULL output shape for correct addition
            let bias_broadcast: Tensor<R, D, _> = bias.broadcast_as(output_shape);
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
}
