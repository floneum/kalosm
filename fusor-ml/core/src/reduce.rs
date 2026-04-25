use crate::{
    Dim, LastRank, LastRankInner, NextRankInner, mir::operation::Operation,
    nary_wise::UnaryFunctionChain,
};
use crate::{
    Layout, Tensor, TensorLayoutInfo,
    compute_graph::NodeIndex,
    mir::inputs::MirValue,
    tensor::{DataType, DataTypeEnum, TensorData},
};
use tensor_ir::ReduceOp;

/// Unsqueeze a reduced tensor back to its original rank by inserting a size-1 dim.
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
    pub(crate) shape: Box<[usize]>,
}

impl ReduceOperation {
    pub fn new(value: NodeIndex, function: ReduceFunction, axis: usize, shape: &[usize]) -> Self {
        let datatype = function.datatype();
        Self {
            value,
            pre_element_wise: UnaryFunctionChain::empty(datatype),
            function,
            post_element_wise: UnaryFunctionChain::empty(datatype),
            axis,
            shape: shape.into(),
        }
    }

    pub fn out_datatype(&self) -> DataTypeEnum {
        self.post_element_wise.out_datatype()
    }
}

impl Operation for ReduceOperation {
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

    fn name(&self) -> String {
        format!("reduce_{}", self.function.name())
    }

    fn output_layout(
        &self,
        map: &rustc_hash::FxHashMap<NodeIndex, TensorLayoutInfo>,
    ) -> TensorLayoutInfo {
        let input = map
            .get(&self.value)
            .expect("reduce input layout is available");
        let new_shape = input
            .layout()
            .shape()
            .iter()
            .enumerate()
            .filter_map(|(i, dim)| (i != self.axis).then_some(*dim))
            .collect::<Vec<_>>();
        TensorLayoutInfo::new(Layout::contiguous(&new_shape), self.out_datatype())
    }

    fn build_tensor_ir(
        &self,
        _nodes: &crate::compute_graph::ComputeGraphInner,
        inputs: &[MirValue],
    ) -> Result<crate::mir::operation::TensorIrLowering, String> {
        crate::tensor_ir_lowering::reduce(self, inputs)
    }
}

#[derive(Clone, Debug)]
pub struct ReduceFunction {
    pub(crate) name: Option<String>,
    pub(crate) op: ReduceOp,
    pub(crate) datatype: DataTypeEnum,
}

impl ReduceFunction {
    fn new(op: ReduceOp, datatype: DataTypeEnum) -> Self {
        Self {
            name: None,
            op,
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
        let dim_idx = dim.resolve();
        let reduced = self.sum(dim);
        unsqueeze_dim::<N, O, D>(&reduced, dim_idx)
    }
}

fn sum_fn<D: DataType>() -> ReduceFunction {
    ReduceFunction::new(ReduceOp::Add, D::WGSL_TYPE).with_name("sum")
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
    ReduceFunction::new(ReduceOp::Max, D::WGSL_TYPE).with_name("max")
}

fn min_fn<D: DataType>() -> ReduceFunction {
    ReduceFunction::new(ReduceOp::Min, D::WGSL_TYPE).with_name("min")
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
    ReduceFunction::new(ReduceOp::Mul, D::WGSL_TYPE).with_name("product")
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
