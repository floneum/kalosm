use std::hash::Hash;

use rustc_hash::FxHasher;

use crate::{
    Dim, LastRank, LastRankInner, NextRankInner,
    mir::{
        inputs::MirValue,
        kernel_backend::DirectKernel,
        operation::Operation,
        workgroup_shape::{Constraint, WorkgroupShapeConstraints},
    },
    nary_wise::{NaryOp, NaryScalar, UnaryFunctionChain},
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

    fn as_reduce(&self) -> Option<&ReduceOperation> {
        Some(self)
    }
}

pub(crate) fn resolve_reduce_on_host(
    operation: &ReduceOperation,
    graph: &crate::compute_graph::ComputeGraphInner,
    inputs: &[MirValue],
) -> Option<TensorData> {
    if !operation.pre_element_wise.functions.is_empty() {
        return None;
    }
    let input = graph.get_cached_result(operation.value)?;
    let output = inputs.get(1)?.as_tensor()?.clone();
    let output_shape = output.layout().shape().to_vec();
    match operation.out_datatype() {
        DataTypeEnum::F32 => resolve_reduce_f32(operation, &input, &output_shape),
        DataTypeEnum::F16 => resolve_reduce_typed::<half::f16>(operation, &input, &output_shape),
        DataTypeEnum::U32 => resolve_reduce_typed::<u32>(operation, &input, &output_shape),
    }
}

fn resolve_reduce_f32(
    operation: &ReduceOperation,
    input: &TensorData,
    output_shape: &[usize],
) -> Option<TensorData> {
    if input.datatype() != DataTypeEnum::F32 || operation.out_datatype() != DataTypeEnum::F32 {
        return None;
    }
    let input_shape = input.layout().shape().to_vec();
    let values = input.to_host_vec::<f32>();
    let total = output_shape.iter().product::<usize>();
    let mut output = Vec::with_capacity(total);
    for flat in 0..total {
        let out_coords = coords_from_flat(flat, output_shape);
        let mut acc = f32::initial(operation.function.op)?;
        for reduce_coord in 0..input_shape[operation.axis] {
            let mut input_coords = Vec::with_capacity(input_shape.len());
            let mut out_axis = 0;
            for axis in 0..input_shape.len() {
                if axis == operation.axis {
                    input_coords.push(reduce_coord);
                } else {
                    input_coords.push(out_coords[out_axis]);
                    out_axis += 1;
                }
            }
            let value = values[row_major_index(&input_shape, &input_coords)];
            acc = f32::reduce(operation.function.op, acc, value);
        }
        output.push(apply_unary_chain_f32(acc, &operation.post_element_wise)?);
    }
    Some(TensorData::from_host_slice(
        input.device(),
        output_shape,
        &output,
    ))
}

fn apply_unary_chain_f32(mut value: f32, chain: &UnaryFunctionChain) -> Option<f32> {
    for function in &chain.functions {
        value = match function.op {
            NaryOp::Neg => -value,
            NaryOp::Cast => value,
            NaryOp::Exp | NaryOp::ApproximateExp | NaryOp::LessApproximateExp => value.exp(),
            NaryOp::Exp2 => value.exp2(),
            NaryOp::Log => value.ln(),
            NaryOp::Log2 => value.log2(),
            NaryOp::Sqrt => value.sqrt(),
            NaryOp::Sin => value.sin(),
            NaryOp::Cos => value.cos(),
            NaryOp::Tan => value.tan(),
            NaryOp::Tanh | NaryOp::TanhExact => value.tanh(),
            NaryOp::Abs => value.abs(),
            NaryOp::AddConst(scalar) => value + scalar_to_f32(&scalar),
            NaryOp::SubConst(scalar) => value - scalar_to_f32(&scalar),
            NaryOp::RSubConst(scalar) => scalar_to_f32(&scalar) - value,
            NaryOp::MulConst(scalar) => value * scalar_to_f32(&scalar),
            NaryOp::DivConst(scalar) => value / scalar_to_f32(&scalar),
            NaryOp::RDivConst(scalar) => scalar_to_f32(&scalar) / value,
            _ => return None,
        };
    }
    Some(value)
}

fn scalar_to_f32(value: &NaryScalar) -> f32 {
    match value {
        NaryScalar::F32(value) => *value,
        NaryScalar::F16(value) => value.to_f32(),
        NaryScalar::U32(value) => *value as f32,
    }
}

fn resolve_reduce_typed<D: DataType + ReduceHostValue>(
    operation: &ReduceOperation,
    input: &TensorData,
    output_shape: &[usize],
) -> Option<TensorData> {
    if input.datatype() != D::DATA_TYPE || operation.out_datatype() != D::DATA_TYPE {
        return None;
    }
    let input_shape = input.layout().shape().to_vec();
    let values = input.to_host_vec::<D>();
    let total = output_shape.iter().product::<usize>();
    let mut output = Vec::with_capacity(total);
    for flat in 0..total {
        let out_coords = coords_from_flat(flat, output_shape);
        let mut acc = D::initial(operation.function.op)?;
        for reduce_coord in 0..input_shape[operation.axis] {
            let mut input_coords = Vec::with_capacity(input_shape.len());
            let mut out_axis = 0;
            for axis in 0..input_shape.len() {
                if axis == operation.axis {
                    input_coords.push(reduce_coord);
                } else {
                    input_coords.push(out_coords[out_axis]);
                    out_axis += 1;
                }
            }
            let value = values[row_major_index(&input_shape, &input_coords)];
            acc = D::reduce(operation.function.op, acc, value);
        }
        output.push(D::apply_post(acc, &operation.post_element_wise)?);
    }
    Some(TensorData::from_host_slice(
        input.device(),
        output_shape,
        &output,
    ))
}

trait ReduceHostValue: DataType + Copy {
    fn initial(op: ReduceOp) -> Option<Self>;
    fn reduce(op: ReduceOp, acc: Self, value: Self) -> Self;
    fn apply_post(value: Self, chain: &UnaryFunctionChain) -> Option<Self>;
}

impl ReduceHostValue for f32 {
    fn initial(op: ReduceOp) -> Option<Self> {
        Some(match op {
            ReduceOp::Sum => 0.0,
            ReduceOp::Max => f32::NEG_INFINITY,
            ReduceOp::Min => f32::INFINITY,
            ReduceOp::Product => 1.0,
        })
    }

    fn reduce(op: ReduceOp, acc: Self, value: Self) -> Self {
        match op {
            ReduceOp::Sum => acc + value,
            ReduceOp::Max => acc.max(value),
            ReduceOp::Min => acc.min(value),
            ReduceOp::Product => acc * value,
        }
    }

    fn apply_post(value: Self, chain: &UnaryFunctionChain) -> Option<Self> {
        apply_unary_chain_f32(value, chain)
    }
}

impl ReduceHostValue for half::f16 {
    fn initial(op: ReduceOp) -> Option<Self> {
        f32::initial(op).map(half::f16::from_f32)
    }

    fn reduce(op: ReduceOp, acc: Self, value: Self) -> Self {
        half::f16::from_f32(f32::reduce(op, acc.to_f32(), value.to_f32()))
    }

    fn apply_post(value: Self, chain: &UnaryFunctionChain) -> Option<Self> {
        apply_unary_chain_f32(value.to_f32(), chain).map(half::f16::from_f32)
    }
}

impl ReduceHostValue for u32 {
    fn initial(op: ReduceOp) -> Option<Self> {
        Some(match op {
            ReduceOp::Sum => 0,
            ReduceOp::Max => u32::MIN,
            ReduceOp::Min => u32::MAX,
            ReduceOp::Product => 1,
        })
    }

    fn reduce(op: ReduceOp, acc: Self, value: Self) -> Self {
        match op {
            ReduceOp::Sum => acc.wrapping_add(value),
            ReduceOp::Max => acc.max(value),
            ReduceOp::Min => acc.min(value),
            ReduceOp::Product => acc.wrapping_mul(value),
        }
    }

    fn apply_post(value: Self, chain: &UnaryFunctionChain) -> Option<Self> {
        apply_unary_chain_u32(value, chain)
    }
}

fn apply_unary_chain_u32(mut value: u32, chain: &UnaryFunctionChain) -> Option<u32> {
    for function in &chain.functions {
        value = match function.op {
            NaryOp::Cast => value,
            NaryOp::AddConst(scalar) => value.wrapping_add(scalar_to_u32(&scalar)),
            NaryOp::SubConst(scalar) => value.wrapping_sub(scalar_to_u32(&scalar)),
            NaryOp::RSubConst(scalar) => scalar_to_u32(&scalar).wrapping_sub(value),
            NaryOp::MulConst(scalar) => value.wrapping_mul(scalar_to_u32(&scalar)),
            NaryOp::DivConst(scalar) => value / scalar_to_u32(&scalar),
            NaryOp::RDivConst(scalar) => scalar_to_u32(&scalar) / value,
            NaryOp::RemConst(scalar) => value % scalar_to_u32(&scalar),
            NaryOp::RRemConst(scalar) => scalar_to_u32(&scalar) % value,
            NaryOp::MinConst(scalar) => value.min(scalar_to_u32(&scalar)),
            NaryOp::MaxConst(scalar) => value.max(scalar_to_u32(&scalar)),
            NaryOp::EqualConst(scalar) => (value == scalar_to_u32(&scalar)) as u32,
            NaryOp::LessConst(scalar) => (value < scalar_to_u32(&scalar)) as u32,
            NaryOp::LessEqualConst(scalar) => (value <= scalar_to_u32(&scalar)) as u32,
            NaryOp::GreaterConst(scalar) => (value > scalar_to_u32(&scalar)) as u32,
            NaryOp::GreaterEqualConst(scalar) => (value >= scalar_to_u32(&scalar)) as u32,
            _ => return None,
        };
    }
    Some(value)
}

fn scalar_to_u32(value: &NaryScalar) -> u32 {
    match value {
        NaryScalar::F32(value) => *value as u32,
        NaryScalar::F16(value) => value.to_f32() as u32,
        NaryScalar::U32(value) => *value,
    }
}

fn coords_from_flat(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        coords[axis] = flat % shape[axis];
        flat /= shape[axis];
    }
    coords
}

fn row_major_index(shape: &[usize], coords: &[usize]) -> usize {
    let mut index = 0;
    for (axis, coord) in coords.iter().enumerate() {
        let stride = shape[axis + 1..].iter().product::<usize>();
        index += coord * stride;
    }
    index
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nary_wise::NaryFunction;

    #[test]
    fn typed_reduce_host_apply_f16_post_chain() {
        let chain = UnaryFunctionChain::new(
            vec![NaryFunction::unary(
                Some("abs".to_string()),
                NaryOp::Abs,
                DataTypeEnum::F16,
                DataTypeEnum::F16,
            )],
            DataTypeEnum::F16,
        );
        let actual =
            <half::f16 as ReduceHostValue>::apply_post(half::f16::from_f32(-9.0), &chain).unwrap();
        assert_eq!(actual, half::f16::from_f32(9.0));
    }

    #[test]
    fn typed_reduce_host_apply_u32_post_chain() {
        let chain = UnaryFunctionChain::new(
            vec![NaryFunction::unary(
                Some("add_const".to_string()),
                NaryOp::AddConst(NaryScalar::U32(7)),
                DataTypeEnum::U32,
                DataTypeEnum::U32,
            )],
            DataTypeEnum::U32,
        );
        let actual = <u32 as ReduceHostValue>::apply_post(9, &chain).unwrap();
        assert_eq!(actual, 16);
    }
}
