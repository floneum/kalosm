use std::{
    fmt::{Display, Write},
    sync::Arc,
};

use crate::{
    D, DataType, DataTypeEnum, Dim, LastRank, Layout, Tensor, TensorData,
    compute_graph::NodeIndex,
    min_for_dtype,
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
    pub fn softmax_slow<const R2: usize>(&self, dim: impl Dim<R>) -> Self
    where
        Tensor<R, T>: LastRank<R2, T>,
    {
        let size = *self.shape();
        let max = self.max(dim);
        let normalized = self - &max.broadcast_as(size);
        let exp = normalized.exp();
        let sum = exp.sum(dim);
        exp / sum.broadcast_as(size)
    }

    pub fn softmax_slow_last_dim<const R2: usize>(&self) -> Self
    where
        Tensor<R, T>: LastRank<R2, T>,
    {
        self.softmax_slow(D::Minus1)
    }

    pub fn softmax<const R2: usize>(&self, axis: impl Dim<R>) -> Self
    where
        Tensor<R, T>: LastRank<R2, T>,
    {
        let operation =
            SoftmaxOperation::new(self.key(), self.datatype(), axis.resolve(), self.shape());
        let data = self.data();

        Self::from_parts(data.custom(Arc::new(operation)))
    }

    pub fn softmax_last_dim<const R2: usize>(&self) -> Self
    where
        Tensor<R, T>: LastRank<R2, T>,
    {
        self.softmax(D::Minus1)
    }
}

fn online_update(f: &mut impl Write, m: impl Display, d: impl Display, x: impl Display) {
    writeln!(f, "{{").unwrap();
    writeln!(f, "let original_m = {m};").unwrap();
    writeln!(f, "{m} = max({m}, {x});").unwrap();
    writeln!(f, "{d} = {d} * exp(original_m - {m}) + exp({x} - {m});").unwrap();
    writeln!(f, "}}").unwrap();
}

fn combine(
    f: &mut impl Write,
    m: impl Display,
    d: impl Display,
    m_peer: impl Display,
    d_peer: impl Display,
) {
    writeln!(f, "let original_m = {m};").unwrap();
    writeln!(f, "{m} = max(original_m, {m_peer});").unwrap();
    writeln!(
        f,
        "{d} = {d} * exp(original_m - {m}) + {d_peer} * exp({m_peer} - {m});"
    )
    .unwrap();
}

#[derive(Debug, Clone)]
struct SoftmaxOperation {
    pub(crate) value: NodeIndex,
    pub(crate) axis: usize,
    pub(crate) shape: Box<[usize]>,
    pub(crate) datatype: DataTypeEnum,
}

impl SoftmaxOperation {
    pub fn new(value: NodeIndex, datatype: DataTypeEnum, axis: usize, shape: &[usize]) -> Self {
        Self {
            value,
            axis,
            shape: shape.into(),
            datatype,
        }
    }

    pub fn reduce_datatype(&self) -> DataTypeEnum {
        self.datatype
    }

    pub fn out_datatype(&self) -> DataTypeEnum {
        self.datatype
    }

    fn rank(&self) -> u32 {
        self.shape.len() as _
    }

    fn kernel(
        &self,
        workgroup_shape: &WorkgroupShape,
        blocksize: u32,
        kernel: &mut GenericKernel,
        device: &crate::Device,
    ) {
        let dtype = self.datatype;
        let out_datatype = self.out_datatype();
        let output_rank = self.rank() - 1;
        let large_reduction = self.shape[self.axis] > 256;

        // Based on v7 of https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
        // And the mlx implementation https://github.com/ml-explore/mlx/blob/b05bcfd27f5f1293401b74dce02e38c8fd7ef66a/mlx/backend/metal/kernels/arg_reduce.metal
        // We can't query the warp size in WGSL, but we can use subgroup operations
        // https://github.com/gpuweb/gpuweb/issues/4437 would unlock a better equivalent to warp synchronization
        // We also can't synchronize among workgroups without atomics. storageBarrier() is a barrier for
        // the storage memory only inside the workgroup.
        // This kernel just uses one workgroup per reduction unit like the MLX kernel
        let input_tensor = kernel.add_tensor_input(output_rank, false, self.reduce_datatype());
        let output_tensor = kernel.add_tensor_input(output_rank, true, out_datatype);
        let reduce_size = kernel.add_integer_input();
        let reduce_stride = kernel.add_integer_input();

        let global_m_final = kernel.add_global_value(KernelGlobalSpace::Workgroup, dtype);
        let global_d_final = kernel.add_global_value(KernelGlobalSpace::Workgroup, dtype);

        let workgroup_local_index = kernel.workgroup_local_index();

        // Each workgroup group works on a single column in the input tensor. This code calculates the
        // start offset of the input and output tensors for each thread group.
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
            writeln!(kernel, "workgroup_index_remainder /= {out_shape_i};",).unwrap();
        }
        writeln!(kernel, "var in_start_offset = ",).unwrap();
        input_tensor.strided_index(kernel, (0..).map(|i| format!("index_{i}")));
        writeln!(kernel, ";").unwrap();
        writeln!(kernel, "var out_start_offset = ",).unwrap();
        output_tensor.strided_index(kernel, (0..).map(|i| format!("index_{i}")));
        writeln!(kernel, ";").unwrap();
        writeln!(kernel).unwrap();

        writeln!(kernel, "var m_lane = {};", min_for_dtype(dtype)).unwrap();
        writeln!(kernel, "var d_lane = {dtype}(0.0);").unwrap();

        // First merge values on each thread individually. We divide the column allocated to the thread group into equal sized buckets
        // Round up
        writeln!(
            kernel,
            "let bucket_size = ({reduce_size} + {blocksize}u - 1) / {blocksize}u;"
        )
        .unwrap();
        // Then loop over this thread's portion of the column and merge the values
        // First load in groups of 4
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

        // If this is a large reduction, process elements in groups of 4
        if large_reduction {
            writeln!(kernel, "while (index + 4u <= end_axis_index) {{").unwrap();
            // Load the chunk of 4 elements at once
            write!(kernel, "let data = vec4<{dtype}>(").unwrap();
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

            // Apply reduction to the 4 elements
            let components = ["data.x", "data.y", "data.z", "data.w"];
            for component in components {
                online_update(kernel, "m_lane", "d_lane", component);
            }

            writeln!(kernel, "index += 4u;").unwrap();
            writeln!(kernel, "}}").unwrap();
            writeln!(kernel).unwrap();
        }

        // Merge the < 4 remaining elements if the bucket size is not a multiple of 4
        writeln!(kernel, "while (index < end_axis_index) {{").unwrap();
        // Load a single element
        writeln!(
            kernel,
            "let data = {input_tensor}[in_start_offset + index * {reduce_stride}];"
        )
        .unwrap();
        // Apply the online update function to merge the single element
        online_update(kernel, "m_lane", "d_lane", "data");
        writeln!(kernel, "index += 1u;").unwrap();
        writeln!(kernel, "}}").unwrap();
        writeln!(kernel).unwrap();

        // If subgroups are supported, use shuffle down reduction
        if device.subgroups_supported() {
            let subgroup_id = kernel.subgroup_index();
            let subgroup_local_id = kernel.subgroup_local_index();
            let subgroups_per_workgroup = kernel.subgroups_per_workgroup();
            let subgroup_size = kernel.subgroup_size();
            let max_subgroup_size = device.max_subgroup_size();
            let local_m_data = kernel.add_global_array(
                KernelGlobalSpace::Workgroup,
                dtype,
                max_subgroup_size.to_string(),
            );
            let local_d_data = kernel.add_global_array(
                KernelGlobalSpace::Workgroup,
                dtype,
                max_subgroup_size.to_string(),
            );
            // Optimized subgroup reduction with unrolled shuffle operations
            let mut offset = max_subgroup_size;
            while offset > 1 {
                writeln!(kernel, "if {subgroup_size} >= {offset}u {{").unwrap();
                offset /= 2;
                writeln!(
                    kernel,
                    "let m_peer = subgroupShuffleDown(m_lane, {offset}u);"
                )
                .unwrap();
                writeln!(
                    kernel,
                    "let d_peer = subgroupShuffleDown(d_lane, {offset}u);"
                )
                .unwrap();
                combine(kernel, "m_lane", "d_lane", "m_peer", "d_peer");
                writeln!(kernel, "}}").unwrap();
            }
            writeln!(kernel).unwrap();

            // Write the output to the workgroup memory if this is the first thread in the subgroup
            writeln!(kernel, "if {subgroup_local_id} == 0u {{").unwrap();
            writeln!(kernel, "{local_m_data}[{subgroup_id}] = m_lane;").unwrap();
            writeln!(kernel, "{local_d_data}[{subgroup_id}] = d_lane;").unwrap();
            writeln!(kernel, "}}").unwrap();

            // Wait until all threads have written to the workgroup shared memory
            writeln!(kernel, "workgroupBarrier();").unwrap();

            // Then if this is the first subgroup, do one final shuffle down reduction
            // Copy over the best value from each subgroup from the workgroup shared memory to the merged variable
            writeln!(
                kernel,
                "if {subgroup_local_id} < {subgroups_per_workgroup} {{"
            )
            .unwrap();
            writeln!(kernel, "m_lane = {local_m_data}[{subgroup_local_id}];").unwrap();
            writeln!(kernel, "d_lane = {local_d_data}[{subgroup_local_id}];").unwrap();
            writeln!(kernel, "}}").unwrap();
            writeln!(kernel, "else {{").unwrap();
            writeln!(kernel, "m_lane = {};", min_for_dtype(dtype)).unwrap();
            writeln!(kernel, "d_lane = {dtype}(0.0);").unwrap();
            writeln!(kernel, "}}").unwrap();

            // Optimized final subgroup reduction with unrolled operations
            let mut offset = max_subgroup_size;
            while offset > 1 {
                writeln!(kernel, "if {subgroup_size} >= {offset}u {{").unwrap();
                offset /= 2;
                writeln!(
                    kernel,
                    "let m_peer = subgroupShuffleDown(m_lane, {offset}u);"
                )
                .unwrap();
                writeln!(
                    kernel,
                    "let d_peer = subgroupShuffleDown(d_lane, {offset}u);"
                )
                .unwrap();
                combine(kernel, "m_lane", "d_lane", "m_peer", "d_peer");
                writeln!(kernel, "}}").unwrap();
            }
        } else {
            // Otherwise, do a simpler workgroup reduction
            let local_m_data =
                kernel.add_global_array(KernelGlobalSpace::Workgroup, dtype, blocksize.to_string());
            let local_d_data =
                kernel.add_global_array(KernelGlobalSpace::Workgroup, dtype, blocksize.to_string());
            let mut offset = blocksize;
            while offset > 1 {
                // Write this thread's values to the shared memory
                writeln!(kernel, "{local_m_data}[{workgroup_local_index}] = m_lane;").unwrap();
                writeln!(kernel, "{local_d_data}[{workgroup_local_index}] = d_lane;").unwrap();
                writeln!(kernel, "workgroupBarrier();").unwrap();
                offset /= 2;
                // Only threads in the first half do the reduction to avoid OOB reads
                // (workgroup memory OOB is undefined behavior in WebGPU)
                writeln!(kernel, "if {workgroup_local_index} < {offset}u {{").unwrap();
                writeln!(
                    kernel,
                    "let m_peer = {local_m_data}[{workgroup_local_index} + {offset}u];"
                )
                .unwrap();
                writeln!(
                    kernel,
                    "let d_peer = {local_d_data}[{workgroup_local_index} + {offset}u];"
                )
                .unwrap();
                combine(kernel, "m_lane", "d_lane", "m_peer", "d_peer");
                writeln!(kernel, "}}").unwrap();
            }
        }

        // Write the output to the output tensor if this is the first thread in the workgroup
        writeln!(kernel, "if {workgroup_local_index} == 0u {{").unwrap();
        writeln!(kernel, "{global_m_final} = m_lane;").unwrap();
        writeln!(kernel, "{global_d_final} = d_lane;").unwrap();
        writeln!(kernel, "}}").unwrap();

        writeln!(kernel, "workgroupBarrier();").unwrap();

        // Finally, write the normalized output to the output tensor
        writeln!(kernel, "let m_all = {global_m_final};").unwrap();
        writeln!(kernel, "let d_all = {global_d_final};").unwrap();
        writeln!(kernel, "var out_index = base_axis_index;").unwrap();

        // If this is a large reduction, process elements in groups of 4
        if large_reduction {
            writeln!(kernel, "while (out_index + 4u <= end_axis_index) {{").unwrap();
            // Load the chunk of 4 elements at once
            write!(kernel, "let data = vec4<{dtype}>(").unwrap();
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

            // Apply the softmax function to each component
            let components = ["data.x", "data.y", "data.z", "data.w"];
            for (i, component) in components.iter().enumerate() {
                writeln!(
                kernel,
                "{output_tensor}[out_start_offset + (out_index + {i}u) * {reduce_stride}] = exp({component} - m_all) / d_all;"
            )
            .unwrap();
            }
            writeln!(kernel, "out_index += 4u;").unwrap();
            writeln!(kernel, "}}").unwrap();
            writeln!(kernel).unwrap();
        }

        // Handle the < 4 remaining elements
        writeln!(kernel, "while (out_index < end_axis_index) {{").unwrap();
        writeln!(
            kernel,
            "let data = {input_tensor}[in_start_offset + out_index * {reduce_stride}];"
        )
        .unwrap();
        writeln!(
            kernel,
            "{output_tensor}[out_start_offset + out_index * {reduce_stride}] = exp(data - m_all) / d_all;"
        )
        .unwrap();
        writeln!(kernel, "out_index += 1u;").unwrap();
        writeln!(kernel, "}}").unwrap();
    }
}

impl Operation for SoftmaxOperation {
    fn workgroup_shape_constraints(
        &self,
        device: &crate::Device,
    ) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
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

    fn dispatch_size(
        &self,
        _: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[MirValue],
    ) -> [u32; 3] {
        let trimmed_tensor: TensorData = inputs[0].as_tensor().unwrap().clone();
        let total_workgroups = trimmed_tensor.layout().shape().iter().product::<usize>() as u32;

        distribute_workgroups(total_workgroups)
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        f(self.value);
    }

    fn inputs(&self, nodes: &crate::compute_graph::ComputeGraphInner) -> Vec<MirValue> {
        let dim = self.axis;
        let tensor = nodes.get_cached_result(self.value).unwrap();
        let layout = tensor.layout();
        let shape = layout.shape();
        let output_type = self.out_datatype();
        let output_tensor = TensorData::new_for_shape(tensor.device(), shape, output_type);

        let trimmed_tensor_layout = Layout::from_parts(
            tensor.layout().offset(),
            tensor
                .layout()
                .shape()
                .iter()
                .enumerate()
                .filter_map(|(i, x)| (i != dim).then_some(*x))
                .collect(),
            tensor
                .layout()
                .strides()
                .iter()
                .enumerate()
                .filter_map(|(i, x)| (i != dim).then_some(*x))
                .collect(),
        );
        let trimmed_tensor = TensorData::new_from_parts(
            tensor.device(),
            tensor.buffer().clone(),
            trimmed_tensor_layout,
            tensor.datatype(),
        );
        vec![
            MirValue::Tensor(trimmed_tensor.clone()),
            MirValue::Tensor(output_tensor.clone()),
            MirValue::Integer(tensor.layout().shape()[dim] as u32),
            MirValue::Integer(tensor.layout().strides()[dim] as u32),
        ]
    }

    fn build_kernel(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
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
        format!("softmax_{}_{}", self.rank(), self.datatype)
    }

    fn output_layout(
        &self,
        map: &rustc_hash::FxHashMap<NodeIndex, crate::TensorLayoutInfo>,
    ) -> crate::TensorLayoutInfo {
        let input_layout = map.get(&self.value).unwrap();
        input_layout.clone()
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_softmax_slow() {
    use crate::Device;

    let device = Device::test_instance();

    let data = [1f32, -2., -3., 4., 5., -6.];
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let diff: [f32; 6] = std::array::from_fn(|i| data[i] - max);
    let exp: [f32; 6] = std::array::from_fn(|i| diff[i].exp());
    let sum = exp.iter().sum::<f32>();
    let softmax_array: [f32; 6] = std::array::from_fn(|i| exp[i] / sum);

    println!("{softmax_array:?}");

    let tensor = Tensor::new(&device, &data);
    let tensor = tensor.softmax_slow(0);
    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert!((output[[0]] - softmax_array[0]).abs() < 0.001);
    assert!((output[[1]] - softmax_array[1]).abs() < 0.001);
    assert!((output[[2]] - softmax_array[2]).abs() < 0.001);
    assert!((output[[3]] - softmax_array[3]).abs() < 0.001);
    assert!((output[[4]] - softmax_array[4]).abs() < 0.001);
    assert!((output[[5]] - softmax_array[5]).abs() < 0.001);
}

#[cfg(test)]
#[tokio::test]
async fn test_softmax() {
    use crate::Device;

    let device = Device::test_instance();

    let data = [1f32, -2., -3., 4., 5., -6.];
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let diff: [f32; 6] = std::array::from_fn(|i| data[i] - max);
    let exp: [f32; 6] = std::array::from_fn(|i| diff[i].exp());
    let sum = exp.iter().sum::<f32>();
    let softmax_array: [f32; 6] = std::array::from_fn(|i| exp[i] / sum);

    println!("{softmax_array:?}");

    let tensor = Tensor::new(&device, &data);
    let tensor = tensor.softmax(0);
    let output = tensor.as_slice().await.unwrap();
    println!("output: {output:?}");
    println!("expect: {softmax_array:?}");
    assert!((output[[0]] - softmax_array[0]).abs() < 0.001);
    assert!((output[[1]] - softmax_array[1]).abs() < 0.001);
    assert!((output[[2]] - softmax_array[2]).abs() < 0.001);
    assert!((output[[3]] - softmax_array[3]).abs() < 0.001);
    assert!((output[[4]] - softmax_array[4]).abs() < 0.001);
    assert!((output[[5]] - softmax_array[5]).abs() < 0.001);
}

#[cfg(test)]
#[tokio::test]
async fn test_softmax_large() {
    use crate::Device;

    let device = Device::test_instance();

    let data: [f32; 1024] = std::array::from_fn(|_| rand::random::<f32>() * 10.0 - 5.0);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let diff: [f32; 1024] = std::array::from_fn(|i| data[i] - max);
    let exp: [f32; 1024] = std::array::from_fn(|i| diff[i].exp());
    let sum = exp.iter().sum::<f32>();
    let softmax_array: [f32; 1024] = std::array::from_fn(|i| exp[i] / sum);

    println!("{softmax_array:?}");

    let tensor = Tensor::new(&device, &data);
    let tensor = tensor.softmax(0);
    let output = tensor.as_slice().await.unwrap();
    println!("output: {output:?}");
    println!("expect: {softmax_array:?}");
    for i in 0..1024 {
        assert!(
            (output[[i]] - softmax_array[i]).abs() < 0.001,
            "Mismatch at index {i}"
        );
    }
}
