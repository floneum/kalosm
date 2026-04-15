use std::fmt::{Display, Write};

use crate::{
    Dim, ElementWiseFunctions, LastRank, LastRankInner, NextRankInner,
    mir::{
        globals::KernelGlobalSpace,
        operation::Operation,
        workgroup_shape::{Constraint, WorkgroupShape, WorkgroupShapeConstraints},
    },
    visit_tiled::distribute_workgroups,
};
use crate::{
    Layout, Tensor,
    compute_graph::NodeIndex,
    mir::{function::Function, inputs::MirValue, kernel::GenericKernel},
    tensor::{DataType, DataTypeEnum, TensorData},
};

#[derive(Debug, Clone)]
pub(crate) struct ReduceOperation {
    pub(crate) value: NodeIndex,
    pub(crate) pre_element_wise: ElementWiseFunctions,
    pub(crate) function: ReduceFunction,
    pub(crate) post_element_wise: ElementWiseFunctions,
    pub(crate) axis: usize,
    pub(crate) shape: Box<[usize]>,
}

impl ReduceOperation {
    pub fn new(value: NodeIndex, function: ReduceFunction, axis: usize, shape: &[usize]) -> Self {
        let datatype = function.datatype();
        Self {
            value,
            pre_element_wise: ElementWiseFunctions::empty(datatype),
            function,
            post_element_wise: ElementWiseFunctions::empty(datatype),
            axis,
            shape: shape.into(),
        }
    }

    pub fn add_pre_element_wise_functions(&self, kernel: &mut GenericKernel) -> Vec<Function> {
        self.pre_element_wise.add_functions(kernel)
    }

    pub fn add_post_element_wise_functions(&self, kernel: &mut GenericKernel) -> Vec<Function> {
        self.post_element_wise.add_functions(kernel)
    }

    pub fn add_function(&self, kernel: &mut GenericKernel) -> Function {
        kernel.add_function(
            self.function.datatype(),
            self.function.operation.clone(),
            [
                ("a".to_string(), self.function.datatype().to_string()),
                ("b".to_string(), self.function.datatype().to_string()),
            ],
        )
    }

    pub fn out_datatype(&self) -> DataTypeEnum {
        self.post_element_wise.out_datatype()
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
        let dtype = self.function.datatype();
        let input_dtype = self.pre_element_wise.input_datatype();
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
        let input_tensor = kernel.add_tensor_input(output_rank, false, input_dtype);
        let output_tensor = kernel.add_tensor_input(output_rank, true, out_datatype);
        let reduce_size = kernel.add_integer_input();
        let reduce_stride = kernel.add_integer_input();
        let reduce = self.add_function(kernel);
        let pre_element_wise = self.add_pre_element_wise_functions(kernel);
        let post_element_wise = self.add_post_element_wise_functions(kernel);
        let workgroup_local_index = kernel.workgroup_local_index();

        // Each workgroup group works on a single column in the input tensor. This code calculates the
        // start offset of the input and output tensors for each thread group.
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

        writeln!(
            kernel,
            "var merged = {dtype}({});",
            self.function.initial_value
        )
        .unwrap();

        // First merge values on each thread individually. We divide the column allocated to the thread group into equal sized buckets
        // Round up
        writeln!(
            kernel,
            "let bucket_size = ({reduce_size} + {blocksize}u - 1) / {blocksize}u;"
        )
        .unwrap();
        // Then loop over this thread's portion of the column and merge the values
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

        // Process elements in groups of 4 with optimized tree reduction if this is a large tensor
        if large_reduction {
            writeln!(kernel, "while (index + 4u <= end_axis_index) {{").unwrap();
            // Load the chunk of 4 elements at once
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

            // Apply pre-element-wise functions to the data
            let components = ["data.x", "data.y", "data.z", "data.w"];
            write!(kernel, "let after_element_wise = vec4<{dtype}>(").unwrap();
            for (i, component) in components.iter().enumerate() {
                if i > 0 {
                    write!(kernel, ", ").unwrap();
                }
                write!(
                    kernel,
                    "{}",
                    pre_element_wise
                        .iter()
                        .fold(component.to_string(), |acc, f| f.call(vec![acc]))
                )
                .unwrap();
            }
            writeln!(kernel, ");").unwrap();

            // Optimized tree reduction for vec4
            writeln!(
                kernel,
                "let vec4_reduced = {};",
                reduce.call(vec![
                    reduce.call(vec![
                        "after_element_wise.x".to_string(),
                        "after_element_wise.y".to_string()
                    ]),
                    reduce.call(vec![
                        "after_element_wise.z".to_string(),
                        "after_element_wise.w".to_string()
                    ])
                ])
            )
            .unwrap();
            writeln!(
                kernel,
                "merged = {};",
                reduce.call(vec!["vec4_reduced".to_string(), "merged".to_string()])
            )
            .unwrap();
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
        // Apply the pre-element-wise functions to the data
        writeln!(
            kernel,
            "let after_element_wise = {};",
            pre_element_wise
                .iter()
                .fold("data".to_string(), |acc, f| f.call(vec![acc]))
        )
        .unwrap();
        // Merge the result into the merged variable
        writeln!(
            kernel,
            "merged = {}; ",
            reduce.call(vec!["after_element_wise".to_string(), "merged".to_string()])
        )
        .unwrap();
        writeln!(kernel, "index += 1u;").unwrap();
        writeln!(kernel, "}}").unwrap();

        // If subgroups are supported, do the shuffle down reduction
        if device.subgroups_supported() {
            let max_subgroup_size = device.max_subgroup_size();
            let local_data = kernel.add_global_array(
                KernelGlobalSpace::Workgroup,
                dtype,
                max_subgroup_size.to_string(),
            );
            let subgroup_id = kernel.subgroup_index();
            let subgroup_local_id = kernel.subgroup_local_index();
            let subgroups_per_workgroup = kernel.subgroups_per_workgroup();
            let subgroup_size = kernel.subgroup_size();

            // Optimized subgroup reduction with unrolled shuffle operations
            let mut offset = max_subgroup_size;
            while offset > 1 {
                writeln!(kernel, "if {subgroup_size} >= {offset}u {{").unwrap();
                offset /= 2;
                writeln!(
                    kernel,
                    "let neighbor = subgroupShuffleDown(merged, {offset}u);"
                )
                .unwrap();
                writeln!(
                    kernel,
                    "merged = {};",
                    reduce.call(vec!["neighbor".to_string(), "merged".to_string()])
                )
                .unwrap();
                writeln!(kernel, "}}").unwrap();
            }

            // Write the output to the workgroup memory if this is the first thread in the subgroup
            writeln!(kernel, "if {subgroup_local_id} == 0u {{").unwrap();
            writeln!(kernel, "{local_data}[{subgroup_id}] = merged;").unwrap();
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
            writeln!(kernel, "merged = {local_data}[{subgroup_local_id}];").unwrap();
            writeln!(kernel, "}}").unwrap();
            writeln!(kernel, "else {{").unwrap();
            writeln!(kernel, "merged = {dtype}({});", self.function.initial_value,).unwrap();
            writeln!(kernel, "}}").unwrap();

            // Final unrolled subgroup reduction
            offset = max_subgroup_size;
            while offset > 1 {
                writeln!(kernel, "if {subgroup_size} >= {offset}u {{").unwrap();
                offset /= 2;
                writeln!(
                    kernel,
                    "let neighbor = subgroupShuffleDown(merged, {offset}u);"
                )
                .unwrap();
                writeln!(
                    kernel,
                    "merged = {};",
                    reduce.call(vec!["neighbor".to_string(), "merged".to_string()])
                )
                .unwrap();
                writeln!(kernel, "}}").unwrap();
            }
        } else {
            // Otherwise reduce using shared memory
            let local_data =
                kernel.add_global_array(KernelGlobalSpace::Workgroup, dtype, blocksize.to_string());
            let mut offset = blocksize;
            while offset > 1 {
                // Write this thread's value to the shared memory
                writeln!(kernel, "{local_data}[{workgroup_local_index}] = merged;").unwrap();
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
                writeln!(
                    kernel,
                    "merged = {};",
                    reduce.call(vec!["neighbor".to_string(), "merged".to_string()])
                )
                .unwrap();
                writeln!(kernel, "}}").unwrap();
            }
        }

        // Write the output to the output tensor if this is the first thread in the workgroup
        writeln!(kernel, "if {workgroup_local_index} == 0u {{").unwrap();
        writeln!(
            kernel,
            "let data = {};",
            post_element_wise
                .iter()
                .fold("merged".to_string(), |acc, f| f.call(vec![acc]))
        )
        .unwrap();
        writeln!(kernel, "{output_tensor}[out_start_offset] = data;").unwrap();
        writeln!(kernel, "}}").unwrap();
    }
}

impl Operation for ReduceOperation {
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
        let output_tensor: TensorData = inputs[1].as_tensor().unwrap().clone();
        let total_workgroups = output_tensor.layout().shape().iter().product::<usize>() as u32;

        distribute_workgroups(total_workgroups)
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        f(self.value);
    }

    fn inputs(&self, nodes: &crate::compute_graph::ComputeGraphInner) -> Vec<MirValue> {
        let dim = self.axis;
        let tensor = nodes.get_cached_result(self.value).unwrap();
        assert_eq!(self.pre_element_wise.input_datatype(), tensor.datatype());
        let layout = tensor.layout();
        let shape = layout.shape();
        let new_tensor_shape = shape
            .iter()
            .enumerate()
            .filter_map(|(i, x)| (i != dim).then_some(*x))
            .collect::<Vec<_>>();
        let output_type = self.out_datatype();
        let output_tensor =
            TensorData::new_for_shape(tensor.device(), &new_tensor_shape, output_type);

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
        format!("reduce_{}", self.function.name())
    }
}

#[derive(Clone, Debug)]
pub struct ReduceFunction {
    pub(crate) name: Option<String>,
    pub(crate) operation: String,
    pub(crate) initial_value: String,
    pub(crate) datatype: DataTypeEnum,
}

impl ReduceFunction {
    fn new(operation: impl Display, initial_value: impl Display, datatype: DataTypeEnum) -> Self {
        Self {
            name: None,
            operation: operation.to_string(),
            initial_value: initial_value.to_string(),
            datatype,
        }
    }

    pub fn name(&self) -> &str {
        self.name.as_deref().unwrap_or("reduce")
    }

    pub fn with_name(mut self, name: impl ToString) -> Self {
        self.name = Some(name.to_string());
        self
    }

    pub(crate) fn datatype(&self) -> DataTypeEnum {
        self.datatype
    }
}

impl<const N: usize, D: DataType> Tensor<N, D> {
    pub fn sum<const O: usize>(&self, dim: impl Dim<N>) -> Tensor<O, D>
    where
        Self: LastRank<O, D>,
    {
        self.reduce(sum_fn::<D>(), dim)
    }

    pub fn sum_keepdim<const O: usize>(&self, dim: impl Dim<N>) -> Self
    where
        Self: LastRank<O, D>,
        <Self as LastRankInner>::LastRank: NextRankInner<NextRank = Self>,
    {
        self.sum(dim).unsqueeze(dim)
    }
}

fn sum_fn<D: DataType>() -> ReduceFunction {
    ReduceFunction::new("let output = a + b;".to_string(), "0.0", D::WGSL_TYPE).with_name("sum")
}

#[cfg(test)]
#[tokio::test]
async fn test_reduce_sum() {
    use crate::Device;

    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let output = tensor.sum(0);

    let output = output.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0]], 9.);
    assert_eq!(output[[1]], 12.);

    let output = tensor.sum(1);

    let output = output.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0]], 3.);
    assert_eq!(output[[1]], 7.);
    assert_eq!(output[[2]], 11.);
}

#[cfg(test)]
#[tokio::test]
async fn test_reduce_sum_large() {
    use crate::Device;

    let device = Device::test_instance();

    let data: [f32; 1024] = std::array::from_fn(|_| rand::random::<f32>() * 10.0 - 5.0);
    let tensor = Tensor::new(&device, &data);

    let output = tensor.sum(0);

    let output = output.as_slice().await.unwrap();
    println!("{output:?}");

    let expected: f32 = data.iter().sum();
    println!("Expected sum: {expected}");

    assert!(
        (output[[]] - expected).abs() < 1e-3,
        "Expected sum to be close to {expected}"
    );
}

#[cfg(test)]
#[tokio::test]
async fn test_reduce_sum_f16() {
    use crate::Device;

    let device = Device::test_instance();
    if !device.f16_supported() {
        return;
    }

    let data = [
        [half::f16::from_f32(1.), half::f16::from_f32(2.)],
        [half::f16::from_f32(3.), half::f16::from_f32(4.)],
        [half::f16::from_f32(5.), half::f16::from_f32(6.)],
    ];
    let tensor = Tensor::new(&device, &data);

    let output = tensor.sum(0);

    let output = output.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0]], half::f16::from_f32(9.));
    assert_eq!(output[[1]], half::f16::from_f32(12.));

    let output = tensor.sum(1);

    let output = output.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0]], half::f16::from_f32(3.));
    assert_eq!(output[[1]], half::f16::from_f32(7.));
    assert_eq!(output[[2]], half::f16::from_f32(11.));
}

#[cfg(test)]
#[tokio::test]
async fn test_reduce_sliced_sum() {
    use crate::Device;

    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let tensor = tensor.slice([0..3, 0..1]);

    let output = tensor.sum(0);

    let output = output.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0]], 9.);

    let output = tensor.sum(1);

    let output = output.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0]], 1.);
    assert_eq!(output[[1]], 3.);
    assert_eq!(output[[2]], 5.);
}

#[cfg(test)]
#[tokio::test]
async fn test_reduce_transposed_sum() {
    use crate::Device;

    let device = Device::test_instance();

    let data = [[1., 3., 5.], [2., 4., 6.]];
    let tensor = Tensor::new(&device, &data).t();

    let output = tensor.sum(0);

    let output = output.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0]], 9.);
    assert_eq!(output[[1]], 12.);

    let output = tensor.sum(1);

    let output = output.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0]], 3.);
    assert_eq!(output[[1]], 7.);
    assert_eq!(output[[2]], 11.);
}

#[cfg(test)]
#[tokio::test]
async fn test_reduce_const_add_then_sum_fused() {
    use crate::Device;

    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let output = (tensor.clone() + 1.).sum(0);

    let output = output.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0]], 3. + 9.);
    assert_eq!(output[[1]], 3. + 12.);

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let output = (tensor + 1.).sum(1);

    let output = output.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0]], 2. + 3.);
    assert_eq!(output[[1]], 2. + 7.);
    assert_eq!(output[[2]], 2. + 11.);
}

#[cfg(test)]
#[tokio::test]
async fn test_reduce_const_sum_then_add_fused() {
    use crate::Device;

    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let output = tensor.sum(0) + 1.;

    let output = output.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0]], 1. + 9.);
    assert_eq!(output[[1]], 1. + 12.);

    let output = tensor.sum(1) + 1.;

    let output = output.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0]], 1. + 3.);
    assert_eq!(output[[1]], 1. + 7.);
    assert_eq!(output[[2]], 1. + 11.);
}

#[cfg(test)]
#[tokio::test]
async fn test_reduce_const_sum_then_cast_fused() {
    use crate::Device;

    let device = Device::test_instance();

    if !device.f16_supported() {
        return;
    }

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let output = tensor.sum(0).cast::<half::f16>();

    let output = output.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0]], half::f16::from_f32(9.));
    assert_eq!(output[[1]], half::f16::from_f32(12.));
}
#[cfg(test)]
#[tokio::test]
async fn test_cast_then_reduce_const_sum_fused() {
    use crate::Device;

    let device = Device::test_instance();

    if !device.f16_supported() {
        return;
    }

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data).cast::<half::f16>();

    let output = tensor.sum(0);

    let output = output.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0]], half::f16::from_f32(9.));
    assert_eq!(output[[1]], half::f16::from_f32(12.));
}

impl<const N: usize, T: DataType> Tensor<N, T> {
    pub fn max<const O: usize>(&self, dim: impl Dim<N>) -> Tensor<O, T>
    where
        Self: LastRank<O, T>,
    {
        self.reduce(max_fn::<T>(), dim)
    }

    pub fn max_keepdim<const O: usize>(&self, dim: impl Dim<N>) -> Self
    where
        Self: LastRank<O, T>,
        <Self as LastRankInner>::LastRank: NextRankInner<NextRank = Self>,
    {
        self.max(dim).unsqueeze(dim)
    }
}

fn max_fn<D: DataType>() -> ReduceFunction {
    ReduceFunction::new(
        "let output = max(a, b);".to_string(),
        min_for_dtype(D::WGSL_TYPE),
        D::WGSL_TYPE,
    )
    .with_name("max")
}

#[cfg(test)]
#[tokio::test]
async fn test_reduce_max() {
    use crate::Device;

    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let output = tensor.max(0);

    let output = output.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0]], 5.);
    assert_eq!(output[[1]], 6.);

    let output = tensor.max(1);

    let output = output.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0]], 2.);
    assert_eq!(output[[1]], 4.);
    assert_eq!(output[[2]], 6.);
}

fn min_fn<D: DataType>() -> ReduceFunction {
    ReduceFunction::new(
        "let output = min(a, b);".to_string(),
        max_for_dtype(D::WGSL_TYPE),
        D::WGSL_TYPE,
    )
    .with_name("min")
}

impl<const N: usize, D: DataType> Tensor<N, D> {
    pub fn min<const O: usize>(&self, dim: impl Dim<N>) -> Tensor<O, D>
    where
        Self: LastRank<O, D>,
    {
        self.reduce(min_fn::<D>(), dim)
    }

    pub fn min_keepdim<const O: usize>(&self, dim: impl Dim<N>) -> Self
    where
        Self: LastRank<O, D>,
        <Self as LastRankInner>::LastRank: NextRankInner<NextRank = Self>,
    {
        self.min(dim).unsqueeze(dim)
    }
}

pub(crate) fn min_for_dtype(dtype: DataTypeEnum) -> &'static str {
    match dtype {
        DataTypeEnum::F32 => "-3.40282e+38",
        DataTypeEnum::F16 => "f16(-65504.0)",
        DataTypeEnum::U32 => "0",
    }
}

pub(crate) fn max_for_dtype(dtype: DataTypeEnum) -> &'static str {
    match dtype {
        DataTypeEnum::F32 => "3.40282e+38",
        DataTypeEnum::F16 => "f16(65504.0)",
        DataTypeEnum::U32 => "4294967295",
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_reduce_min() {
    use crate::Device;

    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let output = tensor.min(0);

    let output = output.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0]], 1.);
    assert_eq!(output[[1]], 2.);

    let output = tensor.min(1);

    let output = output.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0]], 1.);
    assert_eq!(output[[1]], 3.);
    assert_eq!(output[[2]], 5.);
}

fn product_fn<D: DataType>() -> ReduceFunction {
    ReduceFunction::new("let output = a * b;".to_string(), "1.0", D::WGSL_TYPE).with_name("product")
}

impl<const N: usize, D: DataType> Tensor<N, D> {
    pub fn product<const O: usize>(&self, dim: impl Dim<N>) -> Tensor<O, D>
    where
        Self: LastRank<O, D>,
    {
        self.reduce(product_fn::<D>(), dim)
    }

    pub fn product_keepdim<const O: usize>(&self, dim: impl Dim<N>) -> Self
    where
        Self: LastRank<O, D>,
        <Self as LastRankInner>::LastRank: NextRankInner<NextRank = Self>,
    {
        self.product(dim).unsqueeze(dim)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_reduce_product() {
    use crate::Device;

    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let output = tensor.product(0);

    let output = output.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0]], 15.);
    assert_eq!(output[[1]], 48.);

    let output = tensor.product(1);

    let output = output.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0]], 2.);
    assert_eq!(output[[1]], 12.);
    assert_eq!(output[[2]], 30.);
}
