//! Convolution operations that work on both CPU and GPU backends.

use crate::gpu::{DataType, FloatDataType};
use crate::{ConcreteTensor, FloatOps, MatmulImpl, SimdElement, Tensor};
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
    /// For Conv1d: R=3, DIFF=1 gives (batch, in_channels, length) -> (batch, out_channels, out_length)
    pub fn conv<const WEIGHT_RANK: usize, const DIFF: usize, const R2: usize>(
        &self,
        weight: &Tensor<WEIGHT_RANK, D, ConcreteTensor<D, WEIGHT_RANK>>,
        bias: Option<&Tensor<1, D, ConcreteTensor<D, 1>>>,
        padding: [usize; DIFF],
        strides: [usize; DIFF],
    ) -> Self
    where
        ConcreteTensor<D, R>: crate::cpu::LargerRank<R2, DIFF, D>,
        crate::gpu::Tensor<R, D>: crate::gpu::LargerRank<DIFF, R2, D>,
        crate::MulOp: crate::cpu::SimdBinaryOp<D>,
        crate::AddOp: crate::cpu::SimdBinaryOp<D>,
        crate::cpu::SumOp: crate::cpu::SimdReduceOp<D>,
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
}
