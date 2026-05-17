use std::hash::Hash;

use rustc_hash::FxHasher;

use crate::{
    Dim, LastRank, LastRankInner, NextRankInner,
    mir::{
        kernel_backend::DirectKernel,
        inputs::MirValue,
        operation::Operation,
        workgroup_shape::{Constraint, WorkgroupShapeConstraints},
    },
    nary_wise::{NaryScalar, UnaryFunctionChain},
    visit_tiled::distribute_workgroups,
};
use crate::{
    Layout, Tensor,
    compute_graph::NodeIndex,
    tensor::{DataType, DataTypeEnum, TensorData},
};

/// Unsqueeze a reduced tensor back to its original rank by inserting a size-1 dim.
/// This is equivalent to `tensor.unsqueeze(dim)` but implemented inline to avoid
/// depending on the removed composite unsqueeze operation.
fn unsqueeze_dim<const N: usize, const O: usize, D: DataType>(
    tensor: &Tensor<O, D>,
    dim_idx: usize,
) -> Tensor<N, D> {
    let old_shape = tensor.shape();
    let new_shape: [usize; N] = std::array::from_fn(|i| {
        if i < dim_idx {
            old_shape[i]
        } else if i == dim_idx {
            1
        } else {
            old_shape[i - 1]
        }
    });
    tensor.reshape(new_shape)
}

#[derive(Debug, Clone)]
pub(crate) struct ReduceOperation {
    pub(crate) value: NodeIndex,
    pub(crate) pre_element_wise: UnaryFunctionChain,
    pub(crate) function: ReduceFunction,
    pub(crate) post_element_wise: UnaryFunctionChain,
    pub(crate) axis: usize,
}

impl ReduceOperation {
    pub fn new(value: NodeIndex, function: ReduceFunction, axis: usize, _shape: &[usize]) -> Self {
        let datatype = function.datatype();
        Self {
            value,
            pre_element_wise: UnaryFunctionChain::empty(datatype),
            function,
            post_element_wise: UnaryFunctionChain::empty(datatype),
            axis,
        }
    }

    pub fn out_datatype(&self) -> DataTypeEnum {
        self.post_element_wise.out_datatype()
    }
}

impl Operation for ReduceOperation {
    fn hash_kernel_fields(&self, state: &mut FxHasher) {
        self.pre_element_wise.hash(state);
        self.function.hash(state);
        self.post_element_wise.hash(state);
        self.axis.hash(state);
    }

    fn workgroup_shape_constraints(
        &self,
        device: &crate::Device,
    ) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
        let mut constraints = WorkgroupShapeConstraints::new();
        let workgroup_size = device.limits().max_compute_workgroup_size_x.min(256);
        constraints.add_constraint(0, Constraint::equals(workgroup_size));
        constraints.add_constraint(1, Constraint::equals(1));
        constraints.add_constraint(2, Constraint::equals(1));
        constraints
    }

    fn dispatch_size(
        &self,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[MirValue],
    ) -> [u32; 3] {
        let output_tensor: TensorData = inputs[1].as_tensor().unwrap().clone();
        let total_outputs = output_tensor.layout().shape().iter().product::<usize>() as u32;
        let total_workgroups = total_outputs.div_ceil(workgroup_shape.x());

        distribute_workgroups(
            total_workgroups,
            output_tensor
                .device()
                .limits()
                .max_compute_workgroups_per_dimension,
        )
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

    fn build_direct_kernel(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[MirValue],
    ) -> Option<DirectKernel> {
        crate::reduce_direct::build_reduce_direct_kernel(self, graph, workgroup_shape, inputs)
    }

    fn requires_single_kernel_batch(&self) -> bool {
        true
    }

    fn output(&self, _: &crate::compute_graph::ComputeGraphInner, inputs: &[MirValue]) -> MirValue {
        let output_tensor: TensorData = inputs[1].as_tensor().unwrap().clone();
        output_tensor.into()
    }

    fn name(&self) -> String {
        format!("reduce_{}", self.function.name())
    }
}

#[derive(Clone, Debug, Hash)]
pub struct ReduceFunction {
    pub(crate) name: Option<String>,
    pub(crate) op: ReduceOp,
    pub(crate) initial_value: NaryScalar,
    pub(crate) datatype: DataTypeEnum,
}

impl ReduceFunction {
    fn new(op: ReduceOp, initial_value: NaryScalar, datatype: DataTypeEnum) -> Self {
        Self {
            name: None,
            op,
            initial_value,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ReduceOp {
    Sum,
    Max,
    Min,
    Product,
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
        let dim_idx = dim.resolve();
        let reduced = self.sum(dim);
        unsqueeze_dim::<N, O, D>(&reduced, dim_idx)
    }
}

fn sum_fn<D: DataType>() -> ReduceFunction {
    ReduceFunction::new(ReduceOp::Sum, zero_for_dtype(D::DATA_TYPE), D::DATA_TYPE).with_name("sum")
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
        let dim_idx = dim.resolve();
        let reduced = self.max(dim);
        unsqueeze_dim::<N, O, T>(&reduced, dim_idx)
    }
}

fn max_fn<D: DataType>() -> ReduceFunction {
    ReduceFunction::new(
        ReduceOp::Max,
        min_scalar_for_dtype(D::DATA_TYPE),
        D::DATA_TYPE,
    )
    .with_name("max")
}

fn min_fn<D: DataType>() -> ReduceFunction {
    ReduceFunction::new(
        ReduceOp::Min,
        max_scalar_for_dtype(D::DATA_TYPE),
        D::DATA_TYPE,
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
        let dim_idx = dim.resolve();
        let reduced = self.min(dim);
        unsqueeze_dim::<N, O, D>(&reduced, dim_idx)
    }
}

fn product_fn<D: DataType>() -> ReduceFunction {
    ReduceFunction::new(ReduceOp::Product, one_for_dtype(D::DATA_TYPE), D::DATA_TYPE)
        .with_name("product")
}

fn zero_for_dtype(dtype: DataTypeEnum) -> NaryScalar {
    match dtype {
        DataTypeEnum::F32 => NaryScalar::F32(0.0),
        DataTypeEnum::F16 => NaryScalar::F16(half::f16::from_f32(0.0)),
        DataTypeEnum::U32 => NaryScalar::U32(0),
    }
}

fn one_for_dtype(dtype: DataTypeEnum) -> NaryScalar {
    match dtype {
        DataTypeEnum::F32 => NaryScalar::F32(1.0),
        DataTypeEnum::F16 => NaryScalar::F16(half::f16::from_f32(1.0)),
        DataTypeEnum::U32 => NaryScalar::U32(1),
    }
}

fn min_scalar_for_dtype(dtype: DataTypeEnum) -> NaryScalar {
    match dtype {
        DataTypeEnum::F32 => NaryScalar::F32(-3.40282e38),
        DataTypeEnum::F16 => NaryScalar::F16(half::f16::from_f32(-65504.0)),
        DataTypeEnum::U32 => NaryScalar::U32(0),
    }
}

fn max_scalar_for_dtype(dtype: DataTypeEnum) -> NaryScalar {
    match dtype {
        DataTypeEnum::F32 => NaryScalar::F32(3.40282e38),
        DataTypeEnum::F16 => NaryScalar::F16(half::f16::from_f32(65504.0)),
        DataTypeEnum::U32 => NaryScalar::U32(u32::MAX),
    }
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
        let dim_idx = dim.resolve();
        let reduced = self.product(dim);
        unsqueeze_dim::<N, O, D>(&reduced, dim_idx)
    }
}
