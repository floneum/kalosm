use std::{fmt::Write, sync::Arc};

use crate::{
    DataType, DataTypeEnum, LargerRank, Layout, LazyTensorData, TILE_SIZE, Tensor, TensorData,
    TensorInfo, TensorLayoutInfo,
    compute_graph::{ComputeGraphInner, NodeIndex},
    mir::{
        inputs::MirValue,
        kernel::GenericKernel,
        operation::Operation,
        workgroup_shape::{WorkgroupShape, WorkgroupShapeConstraints},
    },
    visit_tiled::{
        MaybeQTensorInput, VisitTiledInput, build_visit_tiled_kernel, titled_map_dispatch_size,
        titled_map_workgroup_size_constraints,
    },
};

#[derive(Debug, Clone)]
struct GroupedConv1dOperation {
    input: NodeIndex,
    weight: NodeIndex,
    bias: Option<NodeIndex>,
    datatype: DataTypeEnum,
    input_shape: [usize; 3],
    weight_shape: [usize; 3],
    output_shape: [usize; 3],
    padding: usize,
    stride: usize,
    groups: usize,
}

impl GroupedConv1dOperation {
    #[allow(clippy::too_many_arguments)]
    fn new(
        input: NodeIndex,
        weight: NodeIndex,
        bias: Option<NodeIndex>,
        datatype: DataTypeEnum,
        input_shape: [usize; 3],
        weight_shape: [usize; 3],
        output_shape: [usize; 3],
        padding: usize,
        stride: usize,
        groups: usize,
    ) -> Self {
        Self {
            input,
            weight,
            bias,
            datatype,
            input_shape,
            weight_shape,
            output_shape,
            padding,
            stride,
            groups,
        }
    }

    fn zero_literal(&self) -> &'static str {
        match self.datatype {
            DataTypeEnum::F32 => "0.0",
            DataTypeEnum::F16 => "f16(0.0)",
            DataTypeEnum::U32 => "0u",
        }
    }
}

impl Operation for GroupedConv1dOperation {
    fn workgroup_shape_constraints(&self, device: &crate::Device) -> WorkgroupShapeConstraints {
        titled_map_workgroup_size_constraints(&self.output_shape, device)
    }

    fn dispatch_size(&self, workgroup_shape: &WorkgroupShape, _inputs: &[MirValue]) -> [u32; 3] {
        titled_map_dispatch_size(TILE_SIZE, *workgroup_shape, &self.output_shape)
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        f(self.input);
        f(self.weight);
        if let Some(bias) = self.bias {
            f(bias);
        }
    }

    fn inputs(&self, nodes: &ComputeGraphInner) -> Vec<MirValue> {
        let input = nodes.get_cached_result(self.input).unwrap();
        let weight = nodes.get_cached_result(self.weight).unwrap();
        let output = TensorData::new_for_shape(input.device(), &self.output_shape, self.datatype);

        let mut inputs = Vec::with_capacity(if self.bias.is_some() { 4 } else { 3 });
        inputs.push(input.clone().into());
        inputs.push(weight.clone().into());
        if let Some(bias) = self.bias {
            inputs.push(nodes.get_cached_result(bias).unwrap().clone().into());
        }
        inputs.push(output.into());
        inputs
    }

    fn build_kernel(
        &self,
        graph: &ComputeGraphInner,
        _workgroup_shape: &WorkgroupShape,
        inputs: &[MirValue],
        kernel: &mut GenericKernel,
    ) {
        let output_tensor_idx = inputs.len() - 1;
        let has_bias = self.bias.is_some();
        let input_rank = self.input_shape.len() as u32;
        let weight_rank = self.weight_shape.len() as u32;
        let output_rank = self.output_shape.len() as u32;
        let input_len = self.input_shape[2];
        let in_channels_per_group = self.weight_shape[1];
        let out_channels_per_group = self.output_shape[1] / self.groups;
        let kernel_size = self.weight_shape[2];
        let dtype = self.datatype;
        let zero = self.zero_literal();

        let mut tiled_inputs = vec![
            VisitTiledInput::new(dtype.into(), input_rank),
            VisitTiledInput::new(dtype.into(), weight_rank),
        ];
        if has_bias {
            tiled_inputs.push(VisitTiledInput::new(dtype.into(), 1));
        }
        tiled_inputs.push(VisitTiledInput::new(dtype.into(), output_rank));

        build_visit_tiled_kernel(
            &graph.device(),
            &self.output_shape,
            TILE_SIZE,
            tiled_inputs,
            output_tensor_idx,
            |kernel, indexes, tensors, _values| {
                let input_tensor = match &tensors[0] {
                    MaybeQTensorInput::Tensor(tensor) => tensor,
                    MaybeQTensorInput::QTensor(_) => {
                        panic!("Grouped conv input cannot be quantized")
                    }
                };
                let weight_tensor = match &tensors[1] {
                    MaybeQTensorInput::Tensor(tensor) => tensor,
                    MaybeQTensorInput::QTensor(_) => {
                        panic!("Grouped conv weight cannot be quantized")
                    }
                };
                let bias_tensor = if has_bias {
                    match &tensors[2] {
                        MaybeQTensorInput::Tensor(tensor) => Some(tensor),
                        MaybeQTensorInput::QTensor(_) => {
                            panic!("Grouped conv bias cannot be quantized")
                        }
                    }
                } else {
                    None
                };
                let output_tensor = match &tensors[output_tensor_idx] {
                    MaybeQTensorInput::Tensor(tensor) => tensor,
                    MaybeQTensorInput::QTensor(_) => {
                        panic!("Grouped conv output cannot be quantized")
                    }
                };

                if let Some(bias_tensor) = bias_tensor {
                    write!(kernel, "var acc: {dtype} = {bias_tensor}[").unwrap();
                    bias_tensor.strided_index(kernel, ["dim_1"]);
                    writeln!(kernel, "];").unwrap();
                } else {
                    writeln!(kernel, "var acc: {dtype} = {zero};").unwrap();
                }

                writeln!(
                    kernel,
                    "let conv_group = dim_1 / {out_channels_per_group}u;"
                )
                .unwrap();
                writeln!(
                    kernel,
                    "for (var ic_local = 0u; ic_local < {in_channels_per_group}u; ic_local++) {{"
                )
                .unwrap();
                writeln!(
                    kernel,
                    "    let input_channel = conv_group * {in_channels_per_group}u + ic_local;"
                )
                .unwrap();
                writeln!(
                    kernel,
                    "    for (var kernel_index = 0u; kernel_index < {kernel_size}u; kernel_index++) {{"
                )
                .unwrap();
                writeln!(
                    kernel,
                    "        let padded_pos = dim_2 * {}u + kernel_index;",
                    self.stride
                )
                .unwrap();
                writeln!(
                    kernel,
                    "        if (padded_pos >= {}u && padded_pos < {}u) {{",
                    self.padding,
                    self.padding + input_len
                )
                .unwrap();
                writeln!(
                    kernel,
                    "            let input_pos = padded_pos - {}u;",
                    self.padding
                )
                .unwrap();
                write!(kernel, "            let input_value = {input_tensor}[").unwrap();
                input_tensor.strided_index(kernel, ["dim_0", "input_channel", "input_pos"]);
                writeln!(kernel, "];").unwrap();
                write!(kernel, "            let weight_value = {weight_tensor}[").unwrap();
                weight_tensor.strided_index(kernel, ["dim_1", "ic_local", "kernel_index"]);
                writeln!(kernel, "];").unwrap();
                writeln!(kernel, "            acc += input_value * weight_value;").unwrap();
                writeln!(kernel, "        }}").unwrap();
                writeln!(kernel, "    }}").unwrap();
                writeln!(kernel, "}}").unwrap();

                format!("{output_tensor}[{}] = acc;", indexes[output_tensor_idx])
            },
            kernel,
        );
    }

    fn output(&self, _nodes: &ComputeGraphInner, inputs: &[MirValue]) -> MirValue {
        inputs.last().unwrap().clone()
    }

    fn name(&self) -> String {
        format!(
            "conv1d_grouped_{}_{}x{}x{}_groups_{}",
            self.datatype,
            self.input_shape[1],
            self.output_shape[1],
            self.weight_shape[2],
            self.groups
        )
    }

    fn output_layout(
        &self,
        _map: &rustc_hash::FxHashMap<NodeIndex, TensorLayoutInfo>,
    ) -> TensorLayoutInfo {
        TensorLayoutInfo::new(Layout::contiguous(&self.output_shape), self.datatype)
    }
}

impl<D: DataType> Tensor<3, D> {
    pub fn conv1d_grouped(
        &self,
        weight: &Tensor<3, D>,
        bias: Option<&Tensor<1, D>>,
        padding: usize,
        stride: usize,
        groups: usize,
    ) -> Self {
        assert!(groups > 0, "groups must be greater than zero");
        assert!(stride > 0, "stride must be greater than zero");

        let input_shape = *self.shape();
        let weight_shape = *weight.shape();
        let in_channels = input_shape[1];
        let out_channels = weight_shape[0];
        let in_channels_per_group = weight_shape[1];
        let kernel_size = weight_shape[2];

        assert_eq!(
            in_channels,
            in_channels_per_group * groups,
            "weight in_channels per group must match input channels / groups"
        );
        assert_eq!(
            out_channels % groups,
            0,
            "out_channels ({out_channels}) must be divisible by groups ({groups})"
        );
        assert!(
            input_shape[2] + 2 * padding >= kernel_size,
            "kernel size ({kernel_size}) cannot exceed padded input length ({})",
            input_shape[2] + 2 * padding
        );

        if let Some(bias) = bias {
            assert_eq!(
                bias.shape()[0],
                out_channels,
                "bias shape must match out_channels"
            );
        }

        let output_length = (input_shape[2] + 2 * padding - kernel_size) / stride + 1;
        let output_shape = [input_shape[0], out_channels, output_length];
        let operation = GroupedConv1dOperation::new(
            self.key(),
            weight.key(),
            bias.map(|bias| bias.key()),
            self.datatype(),
            input_shape,
            weight_shape,
            output_shape,
            padding,
            stride,
            groups,
        );
        let device = self.device().clone();
        let key = device.compute_graph().create_custom(Arc::new(operation));

        Tensor::from_parts(LazyTensorData::from_parts(
            device,
            TensorInfo::new(output_shape.to_vec().into_boxed_slice(), self.datatype()),
            key,
        ))
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    /// Pad a specific axis with zeros on both sides
    fn pad_axis(&self, axis: usize, padding: usize) -> Self {
        if padding == 0 {
            return self.clone();
        }

        let shape = self.shape();
        let device = self.device();

        // Create left padding shape
        let mut pad_shape = *shape;
        pad_shape[axis] = padding;
        let pad_left = Tensor::zeros(device, pad_shape);
        let pad_right = Tensor::zeros(device, pad_shape);

        // Concatenate: [pad_left, self, pad_right] along the axis
        Tensor::cat([pad_left, self.clone(), pad_right], axis)
    }

    /// Unified convolution method that handles different tensor formats:
    /// - Simple convolution (R = DIFF): element-wise convolution without channels
    /// - Multi-channel convolution (R = 2 + DIFF): (batch, channels, ...spatial) format
    ///
    /// For Conv1d: R=3, DIFF=1 gives (batch, in_channels, length) -> (batch, out_channels, out_length)
    /// For simple 1D conv: R=1, DIFF=1 gives (length) -> (out_length)
    pub fn conv<const WEIGHT_RANK: usize, const DIFF: usize, const R2: usize>(
        &self,
        weight: &Tensor<WEIGHT_RANK, D>,
        bias: Option<&Tensor<1, D>>,
        padding: [usize; DIFF],
        strides: [usize; DIFF],
    ) -> Self
    where
        Self: LargerRank<DIFF, R2, D>,
    {
        // Extract dimensions
        let input_shape = self.shape();
        let weight_shape = weight.shape();
        let spatial_start = R - DIFF;

        // Multi-channel convolution: (batch, channels, ...spatial)
        // Note: This implementation expects R = 2 + DIFF format
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
        // This gives us shape: (batch, in_channels, ...out_spatial..., ...kernel...)
        let windows = padded.sliding_window_view(std::array::from_fn(|i| {
            let axis = R - DIFF + i;
            let kernel_size = weight_shape[spatial_start + i];
            [axis, kernel_size, strides[i]]
        }));

        // Step 3: Prepare for matmul by reshaping and transposing
        // Windows: (batch, in_channels, ...out_spatial..., ...kernel...)
        // We need: (batch, ...out_spatial..., in_channels, ...kernel...)
        // Then flatten to: (batch * out_spatial_size, in_channels * kernel_size)

        // First, calculate kernel size
        let kernel_size: usize = weight_shape[spatial_start..].iter().product();

        // Transpose to move in_channels after spatial:
        // (batch, in_channels, ...out_spatial..., ...kernel...) -> (batch, ...out_spatial..., in_channels, ...kernel...)
        let windows_transposed = windows.transpose(in_channels_axis, spatial_start);

        // Flatten to (batch * out_spatial_size, in_channels * kernel_size)
        let windows_flat =
            windows_transposed.reshape([batch * out_spatial_size, in_channels * kernel_size]);

        // Step 4: Reshape weight for matmul
        // Weight: (out_channels, in_channels, ...kernel...) -> (out_channels, in_channels * kernel_size)
        let weight_reshaped = weight.reshape([out_channels, in_channels * kernel_size]);
        // Transpose for matmul: (in_channels * kernel_size, out_channels)
        let weight_t = weight_reshaped.t();

        // Step 5: Matrix multiplication
        // (batch * out_spatial_size, in_channels * kernel_size) @ (in_channels * kernel_size, out_channels)
        // = (batch * out_spatial_size, out_channels)
        let output = windows_flat.mat_mul(&weight_t);

        // Step 6: Reshape and transpose back to (batch, out_channels, ...out_spatial...)
        // First reshape to (batch, out_spatial_size, out_channels)
        let output_reshaped = output.reshape([batch, out_spatial_size, out_channels]);
        // Transpose to (batch, out_channels, out_spatial_size)
        let output_transposed = output_reshaped.transpose(in_channels_axis, spatial_start);

        // Reshape to (batch, out_channels, ...out_spatial_dims...)
        let mut output_shape = *input_shape;
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
            // Reshape to (1, out_channels, 1, 1, ...) for broadcasting
            let mut bias_shape = [1; R];
            bias_shape[in_channels_axis] = out_channels;
            let bias_reshaped = bias.unsqueeze(0).reshape(bias_shape);
            output_final.add_(&bias_reshaped)
        } else {
            output_final
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    #[tokio::test]
    async fn test_conv_1d() {
        let device = Device::test_instance();

        // Input: (batch=1, in_channels=1, length=5)
        let input_data = [[[1.0f32, 2.0, 3.0, 4.0, 5.0]]];
        let input_tensor = Tensor::new(&device, &input_data);

        // Weight: (out_channels=1, in_channels=1, kernel_size=3)
        let weight_data = [[[0.2f32, 0.5, 0.3]]];
        let weight_tensor = Tensor::new(&device, &weight_data);

        let bias_val = 0.1f32;
        let bias = Some(Tensor::splat(&device, bias_val, [1]));

        // Perform convolution with stride 1 and no padding
        // Input: (batch=1, in_channels=1, length=5), Weight: (out_channels=1, in_channels=1, kernel_size=3)
        // For R=3, DIFF=1: R2=4 (after sliding window)
        let output_tensor = input_tensor.conv(&weight_tensor, bias.as_ref(), [0], [1]);

        // Expected values for the 1D convolution
        let input_flat = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let weight_flat = [0.2f32, 0.5, 0.3];
        let expected = input_flat
            .windows(weight_flat.len())
            .map(|window| {
                window
                    .iter()
                    .zip(weight_flat.iter())
                    .map(|(x, w)| x * w)
                    .sum::<f32>()
                    + bias_val
            })
            .collect::<Vec<f32>>();

        let output_data = output_tensor.as_slice().await.unwrap();
        assert_eq!(output_data.shape(), &[1, 1, expected.len()]);
        for i in 0..expected.len() {
            let val = output_data[[0, 0, i]];
            let expected_val = expected[i];
            assert!(
                (val - expected_val).abs() < 1e-6,
                "Mismatch at index {}: got {}, expected {}",
                i,
                val,
                expected_val
            );
        }
    }

    #[tokio::test]
    async fn test_conv_1d_strided() {
        let device = Device::test_instance();

        // Input: (batch=1, in_channels=1, length=5)
        let input_data = [[[1.0f32, 2.0, 3.0, 4.0, 5.0]]];
        let input_tensor = Tensor::new(&device, &input_data);

        // Weight: (out_channels=1, in_channels=1, kernel_size=3)
        let weight_data = [[[0.2f32, 0.5, 0.3]]];
        let weight_tensor = Tensor::new(&device, &weight_data);

        let bias_val = 0.1f32;
        let bias_tensor = Some(Tensor::splat(&device, bias_val, [1]));
        let stride = 2;
        // Input: (batch=1, in_channels=1, length=5), Weight: (out_channels=1, in_channels=1, kernel_size=3)
        // For R=3, DIFF=1: R2=4 (after sliding window)
        let output_tensor = input_tensor.conv(&weight_tensor, bias_tensor.as_ref(), [0], [stride]);

        // Expected values for the strided 1D convolution
        let input_flat = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let weight_flat = [0.2f32, 0.5, 0.3];
        let expected = input_flat
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
            .collect::<Vec<f32>>();

        let output_data = output_tensor.as_slice().await.unwrap();
        assert_eq!(output_data.shape(), &[1, 1, expected.len()]);
        for i in 0..expected.len() {
            let val = output_data[[0, 0, i]];
            let expected_val = expected[i];
            assert!(
                (val - expected_val).abs() < 1e-6,
                "Mismatch at index {}: got {}, expected {}",
                i,
                val,
                expected_val
            );
        }
    }

    #[tokio::test]
    async fn test_conv_1d_vs_candle() {
        use candle_core::{Device as CandleDevice, Tensor as CandleTensor};

        let device = Device::test_instance();
        let candle_device = CandleDevice::Cpu;

        // Input: (2, 3, 8) - batch=2, in_channels=3, length=8
        let mut input_data = vec![];
        let mut input_nested = vec![];
        for b in 0..2 {
            let mut batch = vec![];
            for c in 0..3 {
                let mut channel = vec![];
                for i in 0..8 {
                    let val = (b * 24 + c * 8 + i + 1) as f32 * 0.15;
                    input_data.push(val);
                    channel.push(val);
                }
                batch.push(channel);
            }
            input_nested.push(batch);
        }
        let input = Tensor::new(&device, &input_nested);

        // Weight: (5, 3, 4) - out_channels=5, in_channels=3, kernel_size=4
        let mut weight_data = vec![];
        let mut weight_nested = vec![];
        for o in 0..5 {
            let mut out_ch = vec![];
            for i in 0..3 {
                let mut in_ch = vec![];
                for k in 0..4 {
                    let val = ((o * 12 + i * 4 + k) % 11) as f32 * 0.1;
                    weight_data.push(val);
                    in_ch.push(val);
                }
                out_ch.push(in_ch);
            }
            weight_nested.push(out_ch);
        }
        let weight = Tensor::new(&device, &weight_nested);

        // Bias: (5,)
        let bias_data: Vec<f32> = (0..5).map(|i| i as f32 * 0.05).collect();
        let bias = Tensor::new(&device, &bias_data);

        // Fusor convolution with padding and stride
        let fusor_output = input.conv(&weight, Some(&bias), [1], [2]);
        let fusor_result = fusor_output.as_slice().await.unwrap();

        // Candle convolution
        let candle_input =
            CandleTensor::from_slice(&input_data, (2, 3, 8), &candle_device).unwrap();
        let candle_weight =
            CandleTensor::from_slice(&weight_data, (5, 3, 4), &candle_device).unwrap();
        let candle_bias = CandleTensor::from_slice(&bias_data, 5, &candle_device).unwrap();

        let candle_output = candle_input.conv1d(&candle_weight, 1, 2, 1, 1).unwrap();
        let candle_output = candle_output
            .broadcast_add(&candle_bias.reshape((1, 5, 1)).unwrap())
            .unwrap();
        let candle_result = candle_output.to_vec3::<f32>().unwrap();

        // Compare results
        let fusor_shape = fusor_result.shape();
        assert_eq!(fusor_shape[0], 2);
        assert_eq!(fusor_shape[1], 5);
        assert_eq!(candle_result.len(), 2);
        assert_eq!(candle_result[0].len(), 5);

        for b in 0..2 {
            for c in 0..5 {
                assert_eq!(
                    fusor_shape[2],
                    candle_result[b][c].len(),
                    "Output length mismatch at batch {} channel {}",
                    b,
                    c
                );
                for i in 0..fusor_shape[2] {
                    let fusor_val = fusor_result[[b, c, i]];
                    let candle_val = candle_result[b][c][i];
                    assert!(
                        (fusor_val - candle_val).abs() < 1e-3,
                        "Mismatch at [{}, {}, {}]: fusor={}, candle={}",
                        b,
                        c,
                        i,
                        fusor_val,
                        candle_val
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn test_grouped_conv_1d_depthwise_single_kernel() {
        let device = Device::test_instance();

        let input_data = [[[1.0f32, 2.0, 3.0, 4.0], [10.0f32, 20.0, 30.0, 40.0]]];
        let input = Tensor::new(&device, &input_data);
        let weight_data = [[[1.0f32, 0.0, -1.0]], [[0.5f32, 0.25, -0.5]]];
        let weight = Tensor::new(&device, &weight_data);
        let bias = Tensor::new(&device, &[0.5f32, -1.0]);

        let output = input.conv1d_grouped(&weight, Some(&bias), 1, 1, 2);
        assert_eq!(output.count_kernels_to_resolve(), 1);

        let result = output.as_slice().await.unwrap();
        assert_eq!(result.shape(), &[1, 2, 4]);
        let expected = [[[-1.5f32, -1.5, -1.5, 3.5], [-8.5f32, -6.0, -3.5, 24.0]]];
        for batch in 0..1 {
            for channel in 0..2 {
                for position in 0..4 {
                    let actual = result[[batch, channel, position]];
                    let expected = expected[batch][channel][position];
                    assert!(
                        (actual - expected).abs() < 1e-5,
                        "Mismatch at [{batch}, {channel}, {position}]: got {actual}, expected {expected}"
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn test_conv_1d_moonshine_conv2_shape_matches_cpu_reference() {
        let device = Device::test_instance();
        let batch = 1;
        let in_channels = 640;
        let out_channels = 320;
        let input_len = 204;
        let kernel = 5;
        let stride = 2;

        let input_data: Vec<f32> = (0..batch * in_channels * input_len)
            .map(|i| ((i % 101) as f32 - 50.0) * 0.002)
            .collect();
        let weight_data: Vec<f32> = (0..out_channels * in_channels * kernel)
            .map(|i| ((i % 103) as f32 - 51.0) * 0.001)
            .collect();
        let bias_data: Vec<f32> = (0..out_channels)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.001)
            .collect();

        let input_nested: Vec<Vec<Vec<f32>>> = input_data
            .chunks_exact(in_channels * input_len)
            .map(|batch| {
                batch
                    .chunks_exact(input_len)
                    .map(|channel| channel.to_vec())
                    .collect()
            })
            .collect();
        let weight_nested: Vec<Vec<Vec<f32>>> = weight_data
            .chunks_exact(in_channels * kernel)
            .map(|out_channel| {
                out_channel
                    .chunks_exact(kernel)
                    .map(|channel| channel.to_vec())
                    .collect()
            })
            .collect();

        let input = Tensor::new(&device, &input_nested);
        let weight = Tensor::new(&device, &weight_nested);
        let bias = Tensor::new(&device, &bias_data);
        let result = input
            .conv(&weight, Some(&bias), [0], [stride])
            .as_slice()
            .await
            .unwrap();

        let check_positions = [
            (0, 0, 0),
            (0, 0, result.shape()[2] - 1),
            (0, out_channels / 2, result.shape()[2] / 2),
            (0, out_channels - 1, 0),
            (0, out_channels - 1, result.shape()[2] - 1),
        ];

        for (batch_idx, out_channel, out_pos) in check_positions {
            let mut expected = bias_data[out_channel];
            for in_channel in 0..in_channels {
                for kernel_idx in 0..kernel {
                    let input_pos = out_pos * stride + kernel_idx;
                    expected += input_data
                        [batch_idx * in_channels * input_len + in_channel * input_len + input_pos]
                        * weight_data
                            [out_channel * in_channels * kernel + in_channel * kernel + kernel_idx];
                }
            }
            let actual = result[[batch_idx, out_channel, out_pos]];
            assert!(
                (actual - expected).abs() < 1e-3,
                "Mismatch at [{batch_idx}, {out_channel}, {out_pos}]: actual={actual}, expected={expected}"
            );
        }
    }

    #[tokio::test]
    async fn test_conv_1d_after_zero_cat_uses_materialized_layout() {
        let device = Device::test_instance();
        let batch = 1;
        let in_channels = 64;
        let out_channels = 32;
        let input_len = 17;
        let pad = 4;
        let padded_len = input_len + pad;
        let kernel = 5;
        let stride = 2;

        let input_data: Vec<f32> = (0..batch * in_channels * input_len)
            .map(|i| ((i % 101) as f32 - 50.0) * 0.002)
            .collect();
        let weight_data: Vec<f32> = (0..out_channels * in_channels * kernel)
            .map(|i| ((i % 103) as f32 - 51.0) * 0.001)
            .collect();
        let bias_data: Vec<f32> = (0..out_channels)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.001)
            .collect();

        let input_nested: Vec<Vec<Vec<f32>>> = input_data
            .chunks_exact(in_channels * input_len)
            .map(|batch| {
                batch
                    .chunks_exact(input_len)
                    .map(|channel| channel.to_vec())
                    .collect()
            })
            .collect();
        let weight_nested: Vec<Vec<Vec<f32>>> = weight_data
            .chunks_exact(in_channels * kernel)
            .map(|out_channel| {
                out_channel
                    .chunks_exact(kernel)
                    .map(|channel| channel.to_vec())
                    .collect()
            })
            .collect();

        let input = Tensor::new(&device, &input_nested);
        let padded = Tensor::cat(
            [Tensor::zeros(&device, [batch, in_channels, pad]), input],
            2,
        );
        let weight = Tensor::new(&device, &weight_nested);
        let bias = Tensor::new(&device, &bias_data);
        let result = padded
            .conv(&weight, Some(&bias), [0], [stride])
            .as_slice()
            .await
            .unwrap();

        for out_channel in [0, out_channels / 2, out_channels - 1] {
            for out_pos in [0, result.shape()[2] / 2, result.shape()[2] - 1] {
                let mut expected = bias_data[out_channel];
                for in_channel in 0..in_channels {
                    for kernel_idx in 0..kernel {
                        let padded_pos = out_pos * stride + kernel_idx;
                        let input_value = if padded_pos < pad {
                            0.0
                        } else {
                            input_data[in_channel * input_len + padded_pos - pad]
                        };
                        expected += input_value
                            * weight_data[out_channel * in_channels * kernel
                                + in_channel * kernel
                                + kernel_idx];
                    }
                }
                let actual = result[[0, out_channel, out_pos]];
                assert!(
                    (actual - expected).abs() < 1e-3,
                    "Mismatch at [0, {out_channel}, {out_pos}] with padded_len={padded_len}: actual={actual}, expected={expected}"
                );
            }
        }
    }
}
