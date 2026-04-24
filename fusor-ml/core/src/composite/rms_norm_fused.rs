use std::{fmt::Write, sync::Arc};

use crate::{
    CastTensor, DataType, DataTypeEnum, Layout, MaxRank, Tensor, TensorData,
    compute_graph::NodeIndex,
    mir::{
        globals::KernelGlobalSpace,
        inputs::MirValue,
        kernel::GenericKernel,
        operation::Operation,
        workgroup_shape::{Constraint, WorkgroupShape, WorkgroupShapeConstraints},
    },
    visit_tiled::distribute_workgroups,
};

impl<const R: usize, T: DataType> Tensor<R, T> {
    /// Fused RMSNorm kernel that performs the entire normalization in a single kernel launch.
    ///
    /// Formula: output = input / sqrt(mean(input^2) + eps) * weight + bias
    ///
    /// This is more efficient than the composite implementation which requires multiple
    /// kernel launches for cast, square, sum, division, sqrt, multiplication, and addition.
    pub fn rms_norm_fused<const W: usize>(
        &self,
        weight: &Tensor<W, T>,
        bias: Option<&Tensor<W, T>>,
        eps: f32,
    ) -> Self
    where
        T: CastTensor<f32>,
        f32: CastTensor<T>,
        (Tensor<R, T>, Tensor<W, T>): MaxRank<R, T>,
    {
        let operation = RmsNormOperation::new(
            self.key(),
            weight.key(),
            bias.map(|b| b.key()),
            self.datatype(),
            weight.datatype(),
            self.shape(),
            weight.shape(),
            eps,
            false,
        );
        let data = self.data();

        Self::from_parts(data.custom(Arc::new(operation)))
    }

    /// Fused LayerNorm kernel that performs mean, variance, scale, and bias in one launch.
    ///
    /// Formula: output = (input - mean(input)) / sqrt(var(input) + eps) * weight + bias
    pub fn layer_norm_fused<const W: usize>(
        &self,
        weight: &Tensor<W, T>,
        bias: Option<&Tensor<W, T>>,
        eps: f32,
    ) -> Self
    where
        T: CastTensor<f32>,
        f32: CastTensor<T>,
        (Tensor<R, T>, Tensor<W, T>): MaxRank<R, T>,
    {
        let operation = RmsNormOperation::new(
            self.key(),
            weight.key(),
            bias.map(|b| b.key()),
            self.datatype(),
            weight.datatype(),
            self.shape(),
            weight.shape(),
            eps,
            true,
        );
        let data = self.data();

        Self::from_parts(data.custom(Arc::new(operation)))
    }

    /// Fused RMSNorm without bias
    pub fn rms_norm_fused_no_bias<const W: usize>(&self, weight: &Tensor<W, T>, eps: f32) -> Self
    where
        T: CastTensor<f32>,
        f32: CastTensor<T>,
        (Tensor<R, T>, Tensor<W, T>): MaxRank<R, T>,
    {
        self.rms_norm_fused(weight, None, eps)
    }
}

#[derive(Debug, Clone)]
struct RmsNormOperation {
    /// Input tensor node
    input: NodeIndex,
    /// Weight tensor node
    weight: NodeIndex,
    /// Optional bias tensor node
    bias: Option<NodeIndex>,
    /// Input datatype
    input_dtype: DataTypeEnum,
    /// Weight datatype
    weight_dtype: DataTypeEnum,
    /// Input shape
    input_shape: Box<[usize]>,
    /// Epsilon for numerical stability
    eps: f32,
    /// Whether to subtract the mean and normalize by variance instead of RMS.
    remove_mean: bool,
}

impl RmsNormOperation {
    #[allow(clippy::too_many_arguments)]
    fn new(
        input: NodeIndex,
        weight: NodeIndex,
        bias: Option<NodeIndex>,
        input_dtype: DataTypeEnum,
        weight_dtype: DataTypeEnum,
        input_shape: &[usize],
        _weight_shape: &[usize],
        eps: f32,
        remove_mean: bool,
    ) -> Self {
        Self {
            input,
            weight,
            bias,
            input_dtype,
            weight_dtype,
            input_shape: input_shape.into(),
            eps,
            remove_mean,
        }
    }

    fn rank(&self) -> u32 {
        self.input_shape.len() as _
    }

    fn hidden_size(&self) -> usize {
        *self.input_shape.last().unwrap()
    }

    fn kernel(
        &self,
        workgroup_shape: &WorkgroupShape,
        blocksize: u32,
        kernel: &mut GenericKernel,
        device: &crate::Device,
    ) {
        let input_dtype = self.input_dtype;
        let weight_dtype = self.weight_dtype;
        let output_rank = self.rank() - 1;
        let hidden_size = self.hidden_size();
        let large_reduction = hidden_size > 256;
        let has_bias = self.bias.is_some();
        let remove_mean = self.remove_mean;

        // Input tensor (without the last dimension in the layout for workgroup indexing)
        let input_tensor = kernel.add_tensor_input(output_rank, false, input_dtype);
        // Output tensor
        let output_tensor = kernel.add_tensor_input(output_rank, true, input_dtype);
        // Weight tensor (1D)
        let weight_tensor = kernel.add_tensor_input(0, false, weight_dtype);
        // Optional bias tensor (1D)
        let bias_tensor = if has_bias {
            Some(kernel.add_tensor_input(0, false, weight_dtype))
        } else {
            None
        };
        // Hidden dimension size and stride
        let reduce_size = kernel.add_integer_input();
        let reduce_stride = kernel.add_integer_input();
        // Epsilon uniform
        let eps_input = kernel.add_float_input();

        let workgroup_local_index = kernel.workgroup_local_index();

        // Each workgroup works on a single row (all elements along the last dimension)
        let workgroup_index = workgroup_shape.linearized_workgroup_index(kernel);
        writeln!(
            kernel,
            "var workgroup_index_remainder = {};",
            workgroup_index
        )
        .unwrap();
        for i in (0..output_rank).rev() {
            let out_shape_i = output_tensor.shape_binding(i);
            writeln!(
                kernel,
                "let index_{i} = workgroup_index_remainder % {out_shape_i};",
            )
            .unwrap();
            writeln!(kernel, "workgroup_index_remainder /= {out_shape_i};").unwrap();
        }
        writeln!(kernel, "var in_start_offset = ").unwrap();
        input_tensor.strided_index(kernel, (0..).map(|i| format!("index_{i}")));
        writeln!(kernel, ";").unwrap();
        writeln!(kernel, "var out_start_offset = ").unwrap();
        output_tensor.strided_index(kernel, (0..).map(|i| format!("index_{i}")));
        writeln!(kernel, ";").unwrap();
        writeln!(kernel).unwrap();

        // Phase 1: Compute row statistics
        if remove_mean {
            writeln!(kernel, "var sum = f32(0.0);").unwrap();
        }
        writeln!(kernel, "var sum_sq = f32(0.0);").unwrap();

        // Divide work among threads in the workgroup
        writeln!(
            kernel,
            "let bucket_size = ({reduce_size} + {blocksize}u - 1) / {blocksize}u;"
        )
        .unwrap();
        writeln!(
            kernel,
            "let base_axis_index = {workgroup_local_index} * bucket_size;"
        )
        .unwrap();
        writeln!(
            kernel,
            "let end_axis_index = min({workgroup_local_index} * bucket_size + bucket_size, {reduce_size});"
        )
        .unwrap();
        writeln!(kernel, "var index = base_axis_index;").unwrap();

        // Process elements in groups of 4 for large reductions
        if large_reduction {
            writeln!(kernel, "while (index + 4u <= end_axis_index) {{").unwrap();
            write!(kernel, "let data = vec4<{input_dtype}>(").unwrap();
            for i in 0..4 {
                if i > 0 {
                    write!(kernel, ", ").unwrap();
                }
                write!(
                    kernel,
                    "{input_tensor}[in_start_offset + (index + {i}u) * {reduce_stride}]"
                )
                .unwrap();
            }
            writeln!(kernel, ");").unwrap();

            // Convert to f32 and compute squared values
            writeln!(kernel, "let f32_data = vec4<f32>(data);").unwrap();
            if remove_mean {
                writeln!(
                    kernel,
                    "sum += f32_data.x + f32_data.y + f32_data.z + f32_data.w;"
                )
                .unwrap();
            }
            writeln!(kernel, "let sq_data = f32_data * f32_data;").unwrap();
            writeln!(
                kernel,
                "sum_sq += sq_data.x + sq_data.y + sq_data.z + sq_data.w;"
            )
            .unwrap();

            writeln!(kernel, "index += 4u;").unwrap();
            writeln!(kernel, "}}").unwrap();
            writeln!(kernel).unwrap();
        }

        // Handle remaining elements
        writeln!(kernel, "while (index < end_axis_index) {{").unwrap();
        writeln!(
            kernel,
            "let data = f32({input_tensor}[in_start_offset + index * {reduce_stride}]);"
        )
        .unwrap();
        if remove_mean {
            writeln!(kernel, "sum += data;").unwrap();
        }
        writeln!(kernel, "sum_sq += data * data;").unwrap();
        writeln!(kernel, "index += 1u;").unwrap();
        writeln!(kernel, "}}").unwrap();
        writeln!(kernel).unwrap();

        // Phase 2: Reduce row statistics across the workgroup
        let global_rms = kernel.add_global_value(KernelGlobalSpace::Workgroup, DataTypeEnum::F32);
        let global_mean = if remove_mean {
            Some(kernel.add_global_value(KernelGlobalSpace::Workgroup, DataTypeEnum::F32))
        } else {
            None
        };
        if device.subgroups_supported() {
            let max_subgroup_size = device.max_subgroup_size();
            let local_data = kernel.add_global_array(
                KernelGlobalSpace::Workgroup,
                DataTypeEnum::F32,
                max_subgroup_size.to_string(),
            );
            let local_sum_data = if remove_mean {
                Some(kernel.add_global_array(
                    KernelGlobalSpace::Workgroup,
                    DataTypeEnum::F32,
                    max_subgroup_size.to_string(),
                ))
            } else {
                None
            };
            let subgroup_id = kernel.subgroup_index();
            let subgroup_local_id = kernel.subgroup_local_index();
            let subgroups_per_workgroup = kernel.subgroups_per_workgroup();

            // First: reduce within each subgroup
            writeln!(kernel, "sum_sq = subgroupAdd(sum_sq);").unwrap();
            if remove_mean {
                writeln!(kernel, "sum = subgroupAdd(sum);").unwrap();
            }

            // Write subgroup results to shared memory
            writeln!(kernel, "{local_data}[{subgroup_id}] = sum_sq;").unwrap();
            if let Some(local_sum_data) = &local_sum_data {
                writeln!(kernel, "{local_sum_data}[{subgroup_id}] = sum;").unwrap();
            }
            writeln!(kernel, "workgroupBarrier();").unwrap();

            // Final reduction across subgroups (only first subgroup participates)
            writeln!(
                kernel,
                "if {subgroup_local_id} < {subgroups_per_workgroup} {{"
            )
            .unwrap();
            writeln!(kernel, "sum_sq = {local_data}[{subgroup_local_id}];").unwrap();
            if let Some(local_sum_data) = &local_sum_data {
                writeln!(kernel, "sum = {local_sum_data}[{subgroup_local_id}];").unwrap();
            }
            writeln!(kernel, "}}").unwrap();
            writeln!(kernel, "else {{").unwrap();
            writeln!(kernel, "sum_sq = f32(0.0);").unwrap();
            if remove_mean {
                writeln!(kernel, "sum = f32(0.0);").unwrap();
            }
            writeln!(kernel, "}}").unwrap();
            writeln!(kernel, "sum_sq = subgroupAdd(sum_sq);").unwrap();
            if remove_mean {
                writeln!(kernel, "sum = subgroupAdd(sum);").unwrap();
            }

            // Thread 0 now has the final row statistics.
            writeln!(kernel, "if {subgroup_id} == 0u {{").unwrap();
            if let Some(global_mean) = &global_mean {
                writeln!(kernel, "let mean = sum / f32({reduce_size});").unwrap();
                writeln!(
                    kernel,
                    "let variance = max(sum_sq / f32({reduce_size}) - mean * mean, f32(0.0));"
                )
                .unwrap();
                writeln!(kernel, "{global_mean} = mean;").unwrap();
                writeln!(kernel, "{global_rms} = sqrt(variance + {eps_input});").unwrap();
            } else {
                writeln!(
                    kernel,
                    "{global_rms} = sqrt(sum_sq / f32({reduce_size}) + {eps_input});"
                )
                .unwrap();
            }
            writeln!(kernel, "}}").unwrap();
        } else {
            // Fallback: shared memory reduction
            let local_data = kernel.add_global_array(
                KernelGlobalSpace::Workgroup,
                DataTypeEnum::F32,
                blocksize.to_string(),
            );
            let local_sum_data = if remove_mean {
                Some(kernel.add_global_array(
                    KernelGlobalSpace::Workgroup,
                    DataTypeEnum::F32,
                    blocksize.to_string(),
                ))
            } else {
                None
            };
            let mut offset = blocksize;
            while offset > 1 {
                writeln!(kernel, "{local_data}[{workgroup_local_index}] = sum_sq;").unwrap();
                if let Some(local_sum_data) = &local_sum_data {
                    writeln!(kernel, "{local_sum_data}[{workgroup_local_index}] = sum;").unwrap();
                }
                writeln!(kernel, "workgroupBarrier();").unwrap();
                offset /= 2;
                // Only threads in the first half do the reduction to avoid OOB reads
                // (workgroup memory OOB is undefined behavior in WebGPU)
                writeln!(kernel, "if {workgroup_local_index} < {offset}u {{").unwrap();
                writeln!(
                    kernel,
                    "let neighbor = {local_data}[{workgroup_local_index} + {offset}u];"
                )
                .unwrap();
                writeln!(kernel, "sum_sq += neighbor;").unwrap();
                if let Some(local_sum_data) = &local_sum_data {
                    writeln!(
                        kernel,
                        "let neighbor_sum = {local_sum_data}[{workgroup_local_index} + {offset}u];"
                    )
                    .unwrap();
                    writeln!(kernel, "sum += neighbor_sum;").unwrap();
                }
                writeln!(kernel, "}}").unwrap();
            }

            // Compute denominator and store it in shared memory.
            writeln!(kernel, "if {workgroup_local_index} == 0u {{").unwrap();
            if let Some(global_mean) = &global_mean {
                writeln!(kernel, "let mean = sum / f32({reduce_size});").unwrap();
                writeln!(
                    kernel,
                    "let variance = max(sum_sq / f32({reduce_size}) - mean * mean, f32(0.0));"
                )
                .unwrap();
                writeln!(kernel, "{global_mean} = mean;").unwrap();
                writeln!(kernel, "{global_rms} = sqrt(variance + {eps_input});").unwrap();
            } else {
                writeln!(kernel, "let mean_sq = sum_sq / f32({reduce_size});").unwrap();
                writeln!(kernel, "{global_rms} = sqrt(mean_sq + {eps_input});").unwrap();
            }
            writeln!(kernel, "}}").unwrap();
        }
        writeln!(kernel, "workgroupBarrier();").unwrap();
        // Read normalization parameters from shared memory.
        if let Some(global_mean) = &global_mean {
            writeln!(kernel, "let mean = {global_mean};").unwrap();

            // Recompute variance from centered values. This preserves LayerNorm's
            // two-pass numerical behavior while still keeping the whole op in one kernel.
            writeln!(kernel, "var variance_sum = f32(0.0);").unwrap();
            writeln!(kernel, "var variance_index = base_axis_index;").unwrap();
            if large_reduction {
                writeln!(kernel, "while (variance_index + 4u <= end_axis_index) {{").unwrap();
                write!(kernel, "let variance_data = vec4<{input_dtype}>(").unwrap();
                for i in 0..4 {
                    if i > 0 {
                        write!(kernel, ", ").unwrap();
                    }
                    write!(
                        kernel,
                        "{input_tensor}[in_start_offset + (variance_index + {i}u) * {reduce_stride}]"
                    )
                    .unwrap();
                }
                writeln!(kernel, ");").unwrap();
                writeln!(
                    kernel,
                    "let variance_diff = vec4<f32>(variance_data) - vec4<f32>(mean);"
                )
                .unwrap();
                writeln!(kernel, "let variance_sq = variance_diff * variance_diff;").unwrap();
                writeln!(
                    kernel,
                    "variance_sum += variance_sq.x + variance_sq.y + variance_sq.z + variance_sq.w;"
                )
                .unwrap();
                writeln!(kernel, "variance_index += 4u;").unwrap();
                writeln!(kernel, "}}").unwrap();
            }
            writeln!(kernel, "while (variance_index < end_axis_index) {{").unwrap();
            writeln!(
                kernel,
                "let variance_data = f32({input_tensor}[in_start_offset + variance_index * {reduce_stride}]);"
            )
            .unwrap();
            writeln!(kernel, "let variance_diff = variance_data - mean;").unwrap();
            writeln!(kernel, "variance_sum += variance_diff * variance_diff;").unwrap();
            writeln!(kernel, "variance_index += 1u;").unwrap();
            writeln!(kernel, "}}").unwrap();

            if device.subgroups_supported() {
                let max_subgroup_size = device.max_subgroup_size();
                let local_variance_data = kernel.add_global_array(
                    KernelGlobalSpace::Workgroup,
                    DataTypeEnum::F32,
                    max_subgroup_size.to_string(),
                );
                let subgroup_id = kernel.subgroup_index();
                let subgroup_local_id = kernel.subgroup_local_index();
                let subgroups_per_workgroup = kernel.subgroups_per_workgroup();

                writeln!(kernel, "variance_sum = subgroupAdd(variance_sum);").unwrap();
                writeln!(
                    kernel,
                    "{local_variance_data}[{subgroup_id}] = variance_sum;"
                )
                .unwrap();
                writeln!(kernel, "workgroupBarrier();").unwrap();
                writeln!(
                    kernel,
                    "if {subgroup_local_id} < {subgroups_per_workgroup} {{"
                )
                .unwrap();
                writeln!(
                    kernel,
                    "variance_sum = {local_variance_data}[{subgroup_local_id}];"
                )
                .unwrap();
                writeln!(kernel, "}}").unwrap();
                writeln!(kernel, "else {{").unwrap();
                writeln!(kernel, "variance_sum = f32(0.0);").unwrap();
                writeln!(kernel, "}}").unwrap();
                writeln!(kernel, "variance_sum = subgroupAdd(variance_sum);").unwrap();
                writeln!(kernel, "if {subgroup_id} == 0u {{").unwrap();
                writeln!(
                    kernel,
                    "{global_rms} = sqrt(variance_sum / f32({reduce_size}) + {eps_input});"
                )
                .unwrap();
                writeln!(kernel, "}}").unwrap();
            } else {
                let local_variance_data = kernel.add_global_array(
                    KernelGlobalSpace::Workgroup,
                    DataTypeEnum::F32,
                    blocksize.to_string(),
                );
                let mut offset = blocksize;
                while offset > 1 {
                    writeln!(
                        kernel,
                        "{local_variance_data}[{workgroup_local_index}] = variance_sum;"
                    )
                    .unwrap();
                    writeln!(kernel, "workgroupBarrier();").unwrap();
                    offset /= 2;
                    writeln!(kernel, "if {workgroup_local_index} < {offset}u {{").unwrap();
                    writeln!(
                        kernel,
                        "let variance_neighbor = {local_variance_data}[{workgroup_local_index} + {offset}u];"
                    )
                    .unwrap();
                    writeln!(kernel, "variance_sum += variance_neighbor;").unwrap();
                    writeln!(kernel, "}}").unwrap();
                }
                writeln!(kernel, "if {workgroup_local_index} == 0u {{").unwrap();
                writeln!(
                    kernel,
                    "{global_rms} = sqrt(variance_sum / f32({reduce_size}) + {eps_input});"
                )
                .unwrap();
                writeln!(kernel, "}}").unwrap();
            }
            writeln!(kernel, "workgroupBarrier();").unwrap();
        }
        writeln!(kernel, "let rms = {global_rms};").unwrap();

        // Phase 3: Normalize and apply weight/bias
        writeln!(kernel, "var out_index = base_axis_index;").unwrap();

        // Process elements in groups of 4 for large reductions
        if large_reduction {
            writeln!(kernel, "while (out_index + 4u <= end_axis_index) {{").unwrap();

            // Load input data
            write!(kernel, "let data = vec4<{input_dtype}>(").unwrap();
            for i in 0..4 {
                if i > 0 {
                    write!(kernel, ", ").unwrap();
                }
                write!(
                    kernel,
                    "{input_tensor}[in_start_offset + (out_index + {i}u) * {reduce_stride}]"
                )
                .unwrap();
            }
            writeln!(kernel, ");").unwrap();

            // Load weight
            write!(kernel, "let w = vec4<{weight_dtype}>(").unwrap();
            for i in 0..4 {
                if i > 0 {
                    write!(kernel, ", ").unwrap();
                }
                write!(kernel, "{weight_tensor}[out_index + {i}u]").unwrap();
            }
            writeln!(kernel, ");").unwrap();

            if remove_mean {
                writeln!(
                    kernel,
                    "let normalized = (vec4<f32>(data) - vec4<f32>(mean)) / rms;"
                )
                .unwrap();
            } else {
                writeln!(kernel, "let normalized = vec4<f32>(data) / rms;").unwrap();
            }
            writeln!(kernel, "var result = normalized * vec4<f32>(w);").unwrap();

            // Add bias if present
            if let Some(bias_tensor) = &bias_tensor {
                write!(kernel, "let b = vec4<{weight_dtype}>(").unwrap();
                for i in 0..4 {
                    if i > 0 {
                        write!(kernel, ", ").unwrap();
                    }
                    write!(kernel, "{bias_tensor}[out_index + {i}u]").unwrap();
                }
                writeln!(kernel, ");").unwrap();
                writeln!(kernel, "result += vec4<f32>(b);").unwrap();
            }

            // Write output
            for i in 0..4 {
                let component = ["result.x", "result.y", "result.z", "result.w"][i];
                writeln!(
                    kernel,
                    "{output_tensor}[out_start_offset + (out_index + {i}u) * {reduce_stride}] = {input_dtype}({component});"
                )
                .unwrap();
            }

            writeln!(kernel, "out_index += 4u;").unwrap();
            writeln!(kernel, "}}").unwrap();
            writeln!(kernel).unwrap();
        }

        // Handle remaining elements
        writeln!(kernel, "while (out_index < end_axis_index) {{").unwrap();
        writeln!(
            kernel,
            "let data = f32({input_tensor}[in_start_offset + out_index * {reduce_stride}]);"
        )
        .unwrap();
        writeln!(kernel, "let w = f32({weight_tensor}[out_index]);").unwrap();
        if remove_mean {
            writeln!(kernel, "var result = ((data - mean) / rms) * w;").unwrap();
        } else {
            writeln!(kernel, "var result = (data / rms) * w;").unwrap();
        }

        if let Some(bias_tensor) = &bias_tensor {
            writeln!(kernel, "result += f32({bias_tensor}[out_index]);").unwrap();
        }

        writeln!(
            kernel,
            "{output_tensor}[out_start_offset + out_index * {reduce_stride}] = {input_dtype}(result);"
        )
        .unwrap();
        writeln!(kernel, "out_index += 1u;").unwrap();
        writeln!(kernel, "}}").unwrap();
    }
}

impl Operation for RmsNormOperation {
    fn workgroup_shape_constraints(&self, device: &crate::Device) -> WorkgroupShapeConstraints {
        let mut constraints = WorkgroupShapeConstraints::new();
        constraints.add_constraint(
            0,
            Constraint::less_than(device.limits().max_compute_workgroup_size_x + 1),
        );
        if device.subgroups_supported() {
            constraints.add_constraint(
                0,
                Constraint::more_than_or_equals(device.min_subgroup_size()),
            );
            constraints.add_constraint(
                0,
                Constraint::less_than_or_equals(device.max_subgroup_size()),
            );
        }
        constraints.add_constraint(1, Constraint::equals(1));
        constraints.add_constraint(2, Constraint::equals(1));
        constraints
    }

    fn dispatch_size(&self, _: &WorkgroupShape, inputs: &[MirValue]) -> [u32; 3] {
        // One workgroup per row (all dimensions except the last)
        let trimmed_tensor: TensorData = inputs[0].as_tensor().unwrap().clone();
        let total_workgroups = trimmed_tensor.layout().shape().iter().product::<usize>() as u32;
        distribute_workgroups(total_workgroups)
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        f(self.input);
        f(self.weight);
        if let Some(bias) = self.bias {
            f(bias);
        }
    }

    fn inputs(&self, nodes: &crate::compute_graph::ComputeGraphInner) -> Vec<MirValue> {
        let input_tensor = nodes.get_cached_result(self.input).unwrap();
        let weight_tensor = nodes.get_cached_result(self.weight).unwrap();
        let bias_tensor = self.bias.map(|b| nodes.get_cached_result(b).unwrap());

        let layout = input_tensor.layout();
        let shape = layout.shape();
        let hidden_dim = shape.len() - 1;

        // Output has same shape as input
        let output_tensor =
            TensorData::new_for_shape(input_tensor.device(), shape, self.input_dtype);

        // Create trimmed layout (all dims except last) for workgroup indexing
        let trimmed_tensor_layout = Layout::from_parts(
            input_tensor.layout().offset(),
            input_tensor
                .layout()
                .shape()
                .iter()
                .enumerate()
                .filter_map(|(i, x)| (i != hidden_dim).then_some(*x))
                .collect(),
            input_tensor
                .layout()
                .strides()
                .iter()
                .enumerate()
                .filter_map(|(i, x)| (i != hidden_dim).then_some(*x))
                .collect(),
        );
        let trimmed_input = TensorData::new_from_parts(
            input_tensor.device(),
            input_tensor.buffer().clone(),
            trimmed_tensor_layout.clone(),
            input_tensor.datatype(),
        );

        // Weight tensor layout (0-dimensional for uniform access)
        let weight_layout = Layout::from_parts(
            weight_tensor.layout().offset(),
            vec![].into(),
            vec![].into(),
        );
        let trimmed_weight = TensorData::new_from_parts(
            weight_tensor.device(),
            weight_tensor.buffer().clone(),
            weight_layout,
            weight_tensor.datatype(),
        );

        let mut inputs = vec![
            MirValue::Tensor(trimmed_input),
            MirValue::Tensor(output_tensor),
            MirValue::Tensor(trimmed_weight),
        ];

        // Add bias if present
        if let Some(bias) = bias_tensor {
            let bias_layout =
                Layout::from_parts(bias.layout().offset(), vec![].into(), vec![].into());
            let trimmed_bias = TensorData::new_from_parts(
                bias.device(),
                bias.buffer().clone(),
                bias_layout,
                bias.datatype(),
            );
            inputs.push(MirValue::Tensor(trimmed_bias));
        }

        // Add hidden size and stride
        inputs.push(MirValue::Integer(
            input_tensor.layout().shape()[hidden_dim] as u32,
        ));
        inputs.push(MirValue::Integer(
            input_tensor.layout().strides()[hidden_dim] as u32,
        ));
        // Add epsilon
        inputs.push(MirValue::Float(self.eps));

        inputs
    }

    fn build_kernel(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        workgroup_shape: &WorkgroupShape,
        _: &[MirValue],
        kernel: &mut GenericKernel,
    ) {
        let max_blocksize = workgroup_shape.x();
        self.kernel(workgroup_shape, max_blocksize, kernel, &graph.device());
    }

    fn output(&self, _: &crate::compute_graph::ComputeGraphInner, inputs: &[MirValue]) -> MirValue {
        let output_tensor: TensorData = inputs[1].as_tensor().unwrap().clone();
        output_tensor.into()
    }

    fn name(&self) -> String {
        format!(
            "{}_{}_{}{}",
            if self.remove_mean {
                "layer_norm_fused"
            } else {
                "rms_norm_fused"
            },
            self.rank(),
            self.input_dtype,
            if self.bias.is_some() { "_bias" } else { "" }
        )
    }

    fn output_layout(
        &self,
        map: &rustc_hash::FxHashMap<NodeIndex, crate::TensorLayoutInfo>,
    ) -> crate::TensorLayoutInfo {
        let input_layout = map.get(&self.input).unwrap();
        crate::TensorLayoutInfo::new(Layout::contiguous(input_layout.shape()), self.input_dtype)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_rms_norm_fused() {
    use crate::Device;

    let device = Device::test_instance();

    // Test data: 3x2 tensor
    let input = Tensor::new(&device, &[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    let weight = Tensor::new(&device, &[2.0f32, 3.0]);
    let eps = 1e-5;

    // Compute expected result manually
    // For row [1, 2]: rms = sqrt((1 + 4) / 2 + eps) = sqrt(2.5 + eps) ≈ 1.5811
    //   normalized = [1/1.5811, 2/1.5811] ≈ [0.6325, 1.2649]
    //   result = [0.6325 * 2, 1.2649 * 3] ≈ [1.2649, 3.7947]

    let result = input.rms_norm_fused(&weight, None, eps);
    let output = result.as_slice().await.unwrap();

    println!("Output: {:?}", output);

    // Verify against manually computed values
    assert!(
        (output[[0, 0]] - 1.2649).abs() < 0.01,
        "Expected ~1.2649, got {}",
        output[[0, 0]]
    );
    assert!(
        (output[[0, 1]] - 3.7947).abs() < 0.01,
        "Expected ~3.7947, got {}",
        output[[0, 1]]
    );
}

#[cfg(test)]
#[tokio::test]
async fn test_rms_norm_fused_vs_composite() {
    use crate::Device;

    let device = Device::test_instance();

    // Test with random-ish data
    let input = Tensor::new(&device, &[[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]);
    let weight = Tensor::new(&device, &[1.0f32, 1.0, 1.0, 1.0]);
    let eps = 1e-5;

    // Compute using composite implementation
    let composite_result = input.layer_norm(&weight, None, eps, false);
    let composite_output = composite_result.as_slice().await.unwrap();

    // Compute using fused implementation
    let fused_result = input.rms_norm_fused(&weight, None, eps);
    let fused_output = fused_result.as_slice().await.unwrap();

    println!("Composite: {:?}", composite_output);
    println!("Fused: {:?}", fused_output);

    // Verify they match
    for i in 0..2 {
        for j in 0..4 {
            assert!(
                (composite_output[[i, j]] - fused_output[[i, j]]).abs() < 0.001,
                "Mismatch at [{}, {}]: composite={}, fused={}",
                i,
                j,
                composite_output[[i, j]],
                fused_output[[i, j]]
            );
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_layer_norm_fused_vs_composite() {
    use crate::Device;

    let device = Device::test_instance();

    let input = Tensor::new(&device, &[[1.0f32, 2.0, 3.0, 4.0], [5.0, 7.0, 11.0, 13.0]]);
    let weight = Tensor::new(&device, &[1.0f32, 0.5, 2.0, -1.0]);
    let bias = Tensor::new(&device, &[0.0f32, 0.25, -0.5, 1.0]);
    let eps = 1e-5;

    let composite_result = input.layer_norm(&weight, Some(&bias), eps, true);
    let composite_output = composite_result.as_slice().await.unwrap();

    let fused_result = input.layer_norm_fused(&weight, Some(&bias), eps);
    let fused_output = fused_result.as_slice().await.unwrap();

    for i in 0..2 {
        for j in 0..4 {
            let diff = (composite_output[[i, j]] - fused_output[[i, j]]).abs();
            assert!(
                diff < 0.001,
                "Mismatch at [{}, {}]: composite={}, fused={}, diff={}",
                i,
                j,
                composite_output[[i, j]],
                fused_output[[i, j]],
                diff
            );
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_layer_norm_fused_large() {
    use crate::Device;

    let device = Device::test_instance();

    let hidden_size = 320;
    let rows = 5;
    let input_data: Vec<Vec<f32>> = (0..rows)
        .map(|row| {
            (0..hidden_size)
                .map(|col| (((row * 17 + col * 13) % 29) as f32 - 14.0) * 0.125)
                .collect()
        })
        .collect();
    let weight_data: Vec<f32> = (0..hidden_size)
        .map(|col| 0.75 + (col % 7) as f32 * 0.05)
        .collect();

    let input: Tensor<2, f32> = Tensor::new(&device, &input_data);
    let weight: Tensor<1, f32> = Tensor::new(&device, &weight_data);
    let eps = 1e-5;

    let composite_result = input.layer_norm(&weight, None, eps, true);
    let composite_output = composite_result.as_slice().await.unwrap();

    let fused_result = input.layer_norm_fused(&weight, None, eps);
    let fused_output = fused_result.as_slice().await.unwrap();

    for row in 0..rows {
        for col in 0..hidden_size {
            let diff = (composite_output[[row, col]] - fused_output[[row, col]]).abs();
            assert!(
                diff < 0.001,
                "Mismatch at [{}, {}]: composite={}, fused={}, diff={}",
                row,
                col,
                composite_output[[row, col]],
                fused_output[[row, col]],
                diff
            );
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_layer_norm_fused_non_contiguous_input_layout() {
    use crate::Device;

    let device = Device::test_instance();

    let input = Tensor::new(
        &device,
        &[
            [
                [1.0f32, 2.0, 3.0, 4.0],
                [5.0, 7.0, 11.0, 13.0],
                [-4.0, -1.0, 0.5, 3.0],
                [8.0, 6.0, 4.0, 2.0],
            ],
            [
                [0.25, 0.5, 1.0, 2.0],
                [3.0, 1.5, 0.75, 0.25],
                [9.0, 10.0, 12.0, 15.0],
                [-8.0, -4.0, -2.0, -1.0],
            ],
        ],
    )
    .narrow(1, 1, 2);
    let weight = Tensor::new(&device, &[1.0f32, 0.5, 2.0, -1.0]);
    let eps = 1e-5;

    let composite_result = input.layer_norm(&weight, None, eps, true);
    let composite_output = composite_result.as_slice().await.unwrap();

    let fused_result = input.layer_norm_fused(&weight, None, eps);
    let fused_output = fused_result.as_slice().await.unwrap();

    assert_eq!(fused_output.shape(), &[2, 2, 4]);
    for batch in 0..2 {
        for row in 0..2 {
            for col in 0..4 {
                let diff =
                    (composite_output[[batch, row, col]] - fused_output[[batch, row, col]]).abs();
                assert!(
                    diff < 0.001,
                    "Mismatch at [{}, {}, {}]: composite={}, fused={}, diff={}",
                    batch,
                    row,
                    col,
                    composite_output[[batch, row, col]],
                    fused_output[[batch, row, col]],
                    diff
                );
            }
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_layer_norm_fused_matmul_chain_matches_composite() {
    use crate::Device;

    let device = Device::test_instance();

    let rows = 7;
    let hidden = 320;
    let out = 13;
    let input_data: Vec<Vec<f32>> = (0..rows)
        .map(|row| {
            (0..hidden + 3)
                .map(|col| (((row * 19 + col * 11) % 37) as f32 - 18.0) * 0.03125)
                .collect()
        })
        .collect();
    let proj_data: Vec<Vec<f32>> = (0..hidden)
        .map(|row| {
            (0..out)
                .map(|col| (((row * 7 + col * 5) % 23) as f32 - 11.0) * 0.015625)
                .collect()
        })
        .collect();
    let norm_weight: Vec<f32> = (0..hidden)
        .map(|col| 0.9 + (col % 5) as f32 * 0.025)
        .collect();

    let input = Tensor::new(&device, &input_data).narrow(1, 0, hidden);
    let weight = Tensor::new(&device, &norm_weight);
    let proj = Tensor::new(&device, &proj_data);
    let eps = 1e-5;

    let composite = input.layer_norm(&weight, None, eps, true).mat_mul(&proj);
    let fused = input.layer_norm_fused(&weight, None, eps).mat_mul(&proj);
    let composite = composite.as_slice().await.unwrap();
    let fused = fused.as_slice().await.unwrap();

    for row in 0..rows {
        for col in 0..out {
            let diff = (composite[[row, col]] - fused[[row, col]]).abs();
            assert!(
                diff < 0.001,
                "Mismatch at [{}, {}]: composite={}, fused={}, diff={}",
                row,
                col,
                composite[[row, col]],
                fused[[row, col]],
                diff
            );
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_rms_norm_fused_with_bias() {
    use crate::Device;

    let device = Device::test_instance();

    let input = Tensor::new(&device, &[[1.0f32, 2.0], [3.0, 4.0]]);
    let weight = Tensor::new(&device, &[2.0f32, 3.0]);
    let bias = Tensor::new(&device, &[0.5f32, 0.5]);
    let eps = 1e-5;

    // Compute using composite implementation
    let composite_result = input.layer_norm(&weight, Some(&bias), eps, false);
    let composite_output = composite_result.as_slice().await.unwrap();

    // Compute using fused implementation
    let fused_result = input.rms_norm_fused(&weight, Some(&bias), eps);
    let fused_output = fused_result.as_slice().await.unwrap();

    println!("Composite with bias: {:?}", composite_output);
    println!("Fused with bias: {:?}", fused_output);

    // Verify they match
    for i in 0..2 {
        for j in 0..2 {
            assert!(
                (composite_output[[i, j]] - fused_output[[i, j]]).abs() < 0.001,
                "Mismatch at [{}, {}]: composite={}, fused={}",
                i,
                j,
                composite_output[[i, j]],
                fused_output[[i, j]]
            );
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_rms_norm_fused_large() {
    use crate::Device;

    let device = Device::test_instance();

    // Test with a larger hidden dimension (512)
    let hidden_size = 512;
    let batch_size = 4;

    // Create input as a 2D array
    let input_data: Vec<Vec<f32>> = (0..batch_size)
        .map(|b| {
            (0..hidden_size)
                .map(|h| (((b * hidden_size + h) % 10) as f32) * 0.1)
                .collect()
        })
        .collect();
    let weight_data: Vec<f32> = (0..hidden_size).map(|_| 1.0).collect();

    let input: Tensor<2, f32> = Tensor::new(&device, &input_data);
    let weight: Tensor<1, f32> = Tensor::new(&device, &weight_data);
    let eps = 1e-5;

    // Compute using composite implementation
    let composite_result = input.layer_norm(&weight, None, eps, false);
    let composite_output = composite_result.as_slice().await.unwrap();

    // Compute using fused implementation
    let fused_result = input.rms_norm_fused(&weight, None, eps);
    let fused_output = fused_result.as_slice().await.unwrap();

    // Verify they match
    for i in 0..batch_size {
        for j in 0..hidden_size {
            let diff = (composite_output[[i, j]] - fused_output[[i, j]]).abs();
            assert!(
                diff < 0.01,
                "Mismatch at [{}, {}]: composite={}, fused={}, diff={}",
                i,
                j,
                composite_output[[i, j]],
                fused_output[[i, j]],
                diff
            );
        }
    }

    println!("Large RMSNorm test passed!");
}
