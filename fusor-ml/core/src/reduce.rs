use std::hash::Hash;

use rustc_hash::FxHasher;

use crate::{
    Tensor,
    compute_graph::NodeIndex,
    nary_wise::NaryExpr,
    tensor::{DataTypeEnum, TensorData},
};
use crate::{
    mir::{
        inputs::MirValue,
        kernel_backend::DirectKernel,
        operation::Operation,
        workgroup_shape::{Constraint, WorkgroupShapeConstraints},
    },
    nary_wise::{NaryScalar, UnaryFunctionChain},
    visit_tiled::distribute_workgroups,
};

/// Unsqueeze a reduced tensor back to its original rank by inserting a size-1 dim.
/// This is equivalent to `tensor.unsqueeze(dim)` but implemented inline to avoid
/// depending on the removed composite unsqueeze operation.
fn unsqueeze_dim(tensor: &Tensor, dim_idx: usize) -> Tensor {
    let old_shape = tensor.shape();
    assert!(
        dim_idx <= old_shape.len(),
        "cannot unsqueeze dim {dim_idx} for shape {old_shape:?}"
    );
    let mut new_shape = Vec::with_capacity(old_shape.len() + 1);
    new_shape.extend_from_slice(&old_shape[..dim_idx]);
    new_shape.push(1);
    new_shape.extend_from_slice(&old_shape[dim_idx..]);
    tensor.reshape(new_shape)
}

/// A reduction over one axis of an index space, with a fused n-ary producer.
///
/// `expression` is evaluated at every coordinate of `shape` (the full
/// pre-reduce index space, including the reduced `axis`) and folded with
/// `function` along `axis`. A plain tensor reduction is the trivial producer
/// `input(0, rank)`; the resolver widens it by inlining upstream elementwise
/// expressions, so composed map-reduce clusters (contractions included) lower
/// as a single kernel without materializing the intermediate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ReduceOperation {
    /// Producer inputs referenced by `expression`.
    pub(crate) inputs: Vec<NodeIndex>,
    /// Fused producer over the full index space `shape`.
    pub(crate) expression: NaryExpr,
    /// The full pre-reduce index space, including the reduced axis.
    pub(crate) shape: Box<[usize]>,
    pub(crate) function: ReduceFunction,
    pub(crate) post_element_wise: UnaryFunctionChain,
    pub(crate) axis: usize,
}

impl ReduceOperation {
    pub fn new(value: NodeIndex, function: ReduceFunction, axis: usize, shape: &[usize]) -> Self {
        let datatype = function.datatype();
        Self {
            inputs: vec![value],
            expression: NaryExpr::input(0, shape.len()),
            shape: shape.into(),
            function,
            post_element_wise: UnaryFunctionChain::empty(datatype),
            axis,
        }
    }

    pub fn out_datatype(&self) -> DataTypeEnum {
        self.post_element_wise.out_datatype()
    }

    /// The single input of a trivial (un-fused) reduction: the producer is
    /// still the bare `input(0, rank)` the tensor API emitted. Recognition
    /// runs before fusion, so the canonical clusters it matches always take
    /// this form.
    pub(crate) fn plain_input(&self) -> Option<NodeIndex> {
        (self.inputs.len() == 1 && self.expression == NaryExpr::input(0, self.shape.len()))
            .then(|| self.inputs[0])
    }

    /// The output shape: the index space with the reduced axis removed.
    pub(crate) fn out_shape(&self) -> Vec<usize> {
        self.shape
            .iter()
            .enumerate()
            .filter_map(|(i, x)| (i != self.axis).then_some(*x))
            .collect()
    }
}

impl Operation for ReduceOperation {
    fn hash_kernel_fields(&self, state: &mut FxHasher) {
        self.expression.hash(state);
        self.shape.hash(state);
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
        _workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[MirValue],
    ) -> [u32; 3] {
        let output_tensor: TensorData = inputs.last().unwrap().as_tensor().unwrap().clone();
        let rows = output_tensor.layout().shape().iter().product::<usize>() as u32;
        distribute_workgroups(
            rows,
            output_tensor
                .device()
                .limits()
                .max_compute_workgroups_per_dimension,
        )
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        for input in &self.inputs {
            f(*input);
        }
    }

    fn inputs(&self, nodes: &crate::compute_graph::ComputeGraphInner) -> Vec<MirValue> {
        let mut mir_inputs: Vec<MirValue> = self
            .inputs
            .iter()
            .enumerate()
            .map(|(i, idx)| {
                // Custom-indexed inputs need the dense (dequantized) tensor;
                // block-quantized data only supports the plain row/col path.
                if self.expression.uses_custom_indexing_for_input(i)
                    && let Some(cached) = nodes.get_result(*idx)
                {
                    return cached.into();
                }
                nodes.get_result_or_qmatrix(*idx).unwrap().into()
            })
            .collect();

        let device = match &mir_inputs[0] {
            MirValue::Tensor(tensor) => tensor.device().clone(),
            MirValue::QMatrix(matrix) => matrix.device().clone(),
            _ => unreachable!("reduce inputs are tensors or quantized matrices"),
        };
        let output_tensor =
            TensorData::new_for_shape(&device, &self.out_shape(), self.out_datatype());
        mir_inputs.push(output_tensor.into());
        mir_inputs
    }

    fn build_direct_kernel(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[MirValue],
    ) -> Option<DirectKernel> {
        crate::row_program::RowProgramOperation::from_reduce(self).build_direct_kernel(
            graph,
            workgroup_shape,
            inputs,
        )
    }

    fn output(&self, _: &crate::compute_graph::ComputeGraphInner, inputs: &[MirValue]) -> MirValue {
        let output_tensor: TensorData = inputs.last().unwrap().as_tensor().unwrap().clone();
        output_tensor.into()
    }

    fn name(&self) -> String {
        if self.plain_input().is_some() {
            format!("reduce_{}", self.function.name())
        } else {
            format!("reduce_{}_fused", self.function.name())
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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

impl Tensor {
    pub fn sum(&self, dim: usize) -> Tensor {
        self.reduce(sum_fn(self.datatype()), dim)
    }

    pub fn sum_keepdim(&self, dim: usize) -> Self {
        let reduced = self.sum(dim);
        unsqueeze_dim(&reduced, dim)
    }
}

pub(crate) fn sum_fn(datatype: DataTypeEnum) -> ReduceFunction {
    ReduceFunction::new(ReduceOp::Sum, zero_for_dtype(datatype), datatype).with_name("sum")
}

impl Tensor {
    pub fn max(&self, dim: usize) -> Tensor {
        self.reduce(max_fn(self.datatype()), dim)
    }

    pub fn max_keepdim(&self, dim: usize) -> Self {
        let reduced = self.max(dim);
        unsqueeze_dim(&reduced, dim)
    }
}

pub(crate) fn max_fn(datatype: DataTypeEnum) -> ReduceFunction {
    ReduceFunction::new(ReduceOp::Max, min_scalar_for_dtype(datatype), datatype).with_name("max")
}

fn min_fn(datatype: DataTypeEnum) -> ReduceFunction {
    ReduceFunction::new(ReduceOp::Min, max_scalar_for_dtype(datatype), datatype).with_name("min")
}

impl Tensor {
    pub fn min(&self, dim: usize) -> Tensor {
        self.reduce(min_fn(self.datatype()), dim)
    }

    pub fn min_keepdim(&self, dim: usize) -> Self {
        let reduced = self.min(dim);
        unsqueeze_dim(&reduced, dim)
    }
}

fn product_fn(datatype: DataTypeEnum) -> ReduceFunction {
    ReduceFunction::new(ReduceOp::Product, one_for_dtype(datatype), datatype).with_name("product")
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

impl Tensor {
    pub fn product(&self, dim: usize) -> Tensor {
        self.reduce(product_fn(self.datatype()), dim)
    }

    pub fn product_keepdim(&self, dim: usize) -> Self {
        let reduced = self.product(dim);
        unsqueeze_dim(&reduced, dim)
    }
}
