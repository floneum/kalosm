use std::sync::Arc;

use crate::{
    D, DataType, DataTypeEnum, Dim, LastRank, Tensor,
    compute_graph::NodeIndex,
    mir::{inputs::MirValue, operation::Operation},
};

impl<const R: usize, T: DataType> Tensor<R, T> {
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

#[derive(Debug, Clone)]
pub(crate) struct SoftmaxOperation {
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

    pub fn out_datatype(&self) -> DataTypeEnum {
        self.datatype
    }
}

impl Operation for SoftmaxOperation {
    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        f(self.value);
    }

    fn inputs(&self, nodes: &crate::compute_graph::ComputeGraphInner) -> Vec<MirValue> {
        vec![MirValue::Tensor(
            nodes.get_cached_result(self.value).unwrap().clone(),
        )]
    }

    fn name(&self) -> String {
        format!("softmax_{}_{}", self.shape.len(), self.datatype)
    }

    fn output_layout(
        &self,
        map: &rustc_hash::FxHashMap<NodeIndex, crate::TensorLayoutInfo>,
    ) -> crate::TensorLayoutInfo {
        map.get(&self.value).unwrap().clone()
    }

    fn build_tensor_ir(
        &self,
        _nodes: &crate::compute_graph::ComputeGraphInner,
        inputs: &[MirValue],
    ) -> Result<crate::mir::operation::TensorIrLowering, String> {
        crate::tensor_ir_lowering::softmax(self, inputs)
    }
}
