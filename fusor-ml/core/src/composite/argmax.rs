use std::{fmt::Write, sync::Arc};

use crate::{
    DataTypeEnum, Layout, Tensor,
    compute_graph::NodeIndex,
    min_for_dtype,
    mir::{
        globals::KernelGlobalSpace,
        inputs::MirValue,
        kernel::GenericKernel,
        operation::Operation,
        workgroup_shape::{Constraint, WorkgroupShape, WorkgroupShapeConstraints},
    },
    tensor::{LazyTensorData, TensorData, TensorInfo},
    visit_tiled::distribute_workgroups,
};

impl<const R: usize> Tensor<R, f32> {
    pub fn argmax_last_dim<const OUT_RANK: usize>(&self) -> Tensor<OUT_RANK, u32> {
        assert_eq!(R, OUT_RANK + 1);

        let operation = ArgmaxLastDimOperation::new(self.key(), self.shape());
        let device = self.device().clone();
        let output_shape = self.shape()[..R - 1].to_vec().into_boxed_slice();
        let info = TensorInfo::new(output_shape, DataTypeEnum::U32);
        let key = device.compute_graph().create_custom(Arc::new(operation));

        Tensor::from_parts(LazyTensorData::from_parts(device, info, key))
    }
}

#[derive(Debug, Clone)]
struct ArgmaxLastDimOperation {
    input: NodeIndex,
    shape: Box<[usize]>,
}

impl ArgmaxLastDimOperation {
    fn new(input: NodeIndex, shape: &[usize]) -> Self {
        Self {
            input,
            shape: shape.into(),
        }
    }

    fn rank(&self) -> u32 {
        self.shape.len() as u32
    }

    fn output_rank(&self) -> u32 {
        self.rank() - 1
    }

    fn kernel(
        &self,
        workgroup_shape: &WorkgroupShape,
        blocksize: u32,
        kernel: &mut GenericKernel,
        device: &crate::Device,
    ) {
        let output_rank = self.output_rank();
        let input_tensor = kernel.add_tensor_input(output_rank, false, DataTypeEnum::F32);
        let output_tensor = kernel.add_tensor_input(output_rank, true, DataTypeEnum::U32);
        let reduce_size = kernel.add_integer_input();
        let reduce_stride = kernel.add_integer_input();
        let workgroup_local_index = kernel.workgroup_local_index();

        let linearized_workgroup = workgroup_shape.linearized_workgroup_index(kernel);
        writeln!(
            kernel,
            "var workgroup_index_remainder = {};",
            linearized_workgroup
        )
        .unwrap();
        for i in (0..output_rank).rev() {
            let out_shape_i = output_tensor.shape_binding(i);
            writeln!(
                kernel,
                "let index_{i} = workgroup_index_remainder % {out_shape_i};"
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

        writeln!(
            kernel,
            "var best_value = f32({});",
            min_for_dtype(DataTypeEnum::F32)
        )
        .unwrap();
        writeln!(kernel, "var best_index = 0u;").unwrap();
        writeln!(
            kernel,
            "let bucket_size = ({reduce_size} + {blocksize}u - 1u) / {blocksize}u;"
        )
        .unwrap();
        writeln!(
            kernel,
            "let base_axis_index = {workgroup_local_index} * bucket_size;"
        )
        .unwrap();
        writeln!(
            kernel,
            "let end_axis_index = min(base_axis_index + bucket_size, {reduce_size});"
        )
        .unwrap();
        writeln!(kernel, "var index = base_axis_index;").unwrap();
        writeln!(kernel, "while (index < end_axis_index) {{").unwrap();
        writeln!(
            kernel,
            "let value = {input_tensor}[in_start_offset + index * {reduce_stride}];"
        )
        .unwrap();
        writeln!(
            kernel,
            "if value > best_value || (value == best_value && index < best_index) {{"
        )
        .unwrap();
        writeln!(kernel, "best_value = value;").unwrap();
        writeln!(kernel, "best_index = index;").unwrap();
        writeln!(kernel, "}}").unwrap();
        writeln!(kernel, "index += 1u;").unwrap();
        writeln!(kernel, "}}").unwrap();

        if device.subgroups_supported() {
            let max_subgroup_size = device.max_subgroup_size();
            let local_values = kernel.add_global_array(
                KernelGlobalSpace::Workgroup,
                DataTypeEnum::F32,
                max_subgroup_size.to_string(),
            );
            let local_indexes = kernel.add_global_array(
                KernelGlobalSpace::Workgroup,
                DataTypeEnum::U32,
                max_subgroup_size.to_string(),
            );
            let subgroup_id = kernel.subgroup_index();
            let subgroup_local_id = kernel.subgroup_local_index();
            let subgroups_per_workgroup = kernel.subgroups_per_workgroup();
            let subgroup_size = kernel.subgroup_size();

            let mut offset = max_subgroup_size;
            while offset > 1 {
                writeln!(kernel, "if {subgroup_size} >= {offset}u {{").unwrap();
                offset /= 2;
                writeln!(
                    kernel,
                    "let neighbor_value = subgroupShuffleDown(best_value, {offset}u);"
                )
                .unwrap();
                writeln!(
                    kernel,
                    "let neighbor_index = subgroupShuffleDown(best_index, {offset}u);"
                )
                .unwrap();
                writeln!(
                    kernel,
                    "if neighbor_value > best_value || (neighbor_value == best_value && neighbor_index < best_index) {{"
                )
                .unwrap();
                writeln!(kernel, "best_value = neighbor_value;").unwrap();
                writeln!(kernel, "best_index = neighbor_index;").unwrap();
                writeln!(kernel, "}}").unwrap();
                writeln!(kernel, "}}").unwrap();
            }

            writeln!(kernel, "if {subgroup_local_id} == 0u {{").unwrap();
            writeln!(kernel, "{local_values}[{subgroup_id}] = best_value;").unwrap();
            writeln!(kernel, "{local_indexes}[{subgroup_id}] = best_index;").unwrap();
            writeln!(kernel, "}}").unwrap();
            writeln!(kernel, "workgroupBarrier();").unwrap();

            writeln!(
                kernel,
                "if {subgroup_local_id} < {subgroups_per_workgroup} {{"
            )
            .unwrap();
            writeln!(kernel, "best_value = {local_values}[{subgroup_local_id}];").unwrap();
            writeln!(kernel, "best_index = {local_indexes}[{subgroup_local_id}];").unwrap();
            writeln!(kernel, "}} else {{").unwrap();
            writeln!(
                kernel,
                "best_value = f32({});",
                min_for_dtype(DataTypeEnum::F32)
            )
            .unwrap();
            writeln!(kernel, "best_index = 0u;").unwrap();
            writeln!(kernel, "}}").unwrap();

            offset = max_subgroup_size;
            while offset > 1 {
                writeln!(kernel, "if {subgroup_size} >= {offset}u {{").unwrap();
                offset /= 2;
                writeln!(
                    kernel,
                    "let neighbor_value = subgroupShuffleDown(best_value, {offset}u);"
                )
                .unwrap();
                writeln!(
                    kernel,
                    "let neighbor_index = subgroupShuffleDown(best_index, {offset}u);"
                )
                .unwrap();
                writeln!(
                    kernel,
                    "if neighbor_value > best_value || (neighbor_value == best_value && neighbor_index < best_index) {{"
                )
                .unwrap();
                writeln!(kernel, "best_value = neighbor_value;").unwrap();
                writeln!(kernel, "best_index = neighbor_index;").unwrap();
                writeln!(kernel, "}}").unwrap();
                writeln!(kernel, "}}").unwrap();
            }
        } else {
            let local_values = kernel.add_global_array(
                KernelGlobalSpace::Workgroup,
                DataTypeEnum::F32,
                blocksize.to_string(),
            );
            let local_indexes = kernel.add_global_array(
                KernelGlobalSpace::Workgroup,
                DataTypeEnum::U32,
                blocksize.to_string(),
            );
            let mut offset = blocksize;
            while offset > 1 {
                writeln!(
                    kernel,
                    "{local_values}[{workgroup_local_index}] = best_value;"
                )
                .unwrap();
                writeln!(
                    kernel,
                    "{local_indexes}[{workgroup_local_index}] = best_index;"
                )
                .unwrap();
                writeln!(kernel, "workgroupBarrier();").unwrap();
                offset /= 2;
                writeln!(kernel, "if {workgroup_local_index} < {offset}u {{").unwrap();
                writeln!(
                    kernel,
                    "let neighbor_value = {local_values}[{workgroup_local_index} + {offset}u];"
                )
                .unwrap();
                writeln!(
                    kernel,
                    "let neighbor_index = {local_indexes}[{workgroup_local_index} + {offset}u];"
                )
                .unwrap();
                writeln!(
                    kernel,
                    "if neighbor_value > best_value || (neighbor_value == best_value && neighbor_index < best_index) {{"
                )
                .unwrap();
                writeln!(kernel, "best_value = neighbor_value;").unwrap();
                writeln!(kernel, "best_index = neighbor_index;").unwrap();
                writeln!(kernel, "}}").unwrap();
                writeln!(kernel, "}}").unwrap();
            }
        }

        writeln!(kernel, "if {workgroup_local_index} == 0u {{").unwrap();
        writeln!(kernel, "{output_tensor}[out_start_offset] = best_index;").unwrap();
        writeln!(kernel, "}}").unwrap();
    }
}

impl Operation for ArgmaxLastDimOperation {
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
        let output_tensor: TensorData = inputs[1].as_tensor().unwrap().clone();
        let total_workgroups = output_tensor.layout().shape().iter().product::<usize>() as u32;

        distribute_workgroups(total_workgroups)
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        f(self.input);
    }

    fn inputs(&self, nodes: &crate::compute_graph::ComputeGraphInner) -> Vec<MirValue> {
        let tensor = nodes.get_cached_result(self.input).unwrap();
        let last_dim = tensor.layout().shape().len() - 1;
        let output_shape = tensor.layout().shape()[..last_dim].to_vec();
        let output_tensor =
            TensorData::new_for_shape(tensor.device(), &output_shape, DataTypeEnum::U32);
        let trimmed_layout = Layout::from_parts(
            tensor.layout().offset(),
            tensor.layout().shape()[..last_dim].to_vec().into(),
            tensor.layout().strides()[..last_dim].to_vec().into(),
        );
        let trimmed_tensor = TensorData::new_from_parts(
            tensor.device(),
            tensor.buffer().clone(),
            trimmed_layout,
            DataTypeEnum::F32,
        );

        vec![
            MirValue::Tensor(trimmed_tensor),
            MirValue::Tensor(output_tensor),
            MirValue::Integer(tensor.layout().shape()[last_dim] as u32),
            MirValue::Integer(tensor.layout().strides()[last_dim] as u32),
        ]
    }

    fn build_kernel(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        workgroup_shape: &WorkgroupShape,
        _: &[MirValue],
        kernel: &mut GenericKernel,
    ) {
        self.kernel(
            workgroup_shape,
            workgroup_shape.x(),
            kernel,
            &graph.device(),
        );
    }

    fn output(&self, _: &crate::compute_graph::ComputeGraphInner, inputs: &[MirValue]) -> MirValue {
        let output_tensor: TensorData = inputs[1].as_tensor().unwrap().clone();
        output_tensor.into()
    }

    fn name(&self) -> String {
        format!(
            "argmax_last_dim_f32_{}",
            self.shape
                .iter()
                .map(|dim| dim.to_string())
                .collect::<Vec<_>>()
                .join("x")
        )
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_argmax_last_dim_1d() {
    use crate::Device;

    let device = Device::test_instance();
    let tensor = Tensor::new(&device, &[1.0, 3.0, 2.0, 3.0]);
    let output: Tensor<0, u32> = tensor.argmax_last_dim();

    assert_eq!(output.to_scalar().await.unwrap(), 1);
}

#[cfg(test)]
#[tokio::test]
async fn test_argmax_last_dim_2d() {
    use crate::Device;

    let device = Device::test_instance();
    let tensor = Tensor::new(&device, &[[1.0, 5.0, 2.0], [3.0, 0.0, 4.0]]);
    let output: Tensor<1, u32> = tensor.argmax_last_dim();
    let output = output.as_slice().await.unwrap();

    assert_eq!(output[[0]], 1);
    assert_eq!(output[[1]], 2);
}

#[cfg(test)]
#[tokio::test]
async fn test_argmax_last_dim_non_contiguous() {
    use crate::Device;

    let device = Device::test_instance();
    let tensor = Tensor::new(
        &device,
        &[
            [1.0, 10.0, 3.0, 8.0],
            [7.0, 4.0, 9.0, 2.0],
            [5.0, 6.0, 0.0, 11.0],
        ],
    );
    let transposed = tensor.t();
    let output: Tensor<1, u32> = transposed.argmax_last_dim();
    let output = output.as_slice().await.unwrap();

    assert_eq!(output[[0]], 1);
    assert_eq!(output[[1]], 0);
    assert_eq!(output[[2]], 1);
    assert_eq!(output[[3]], 2);
}
