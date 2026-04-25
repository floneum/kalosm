use std::fmt::Debug;

use rustc_hash::FxHashMap;

use crate::{
    DataTypeEnum, TensorData, TensorLayoutInfo,
    compute_graph::{ComputeGraphInner, NodeIndex},
};

use super::inputs::MirValue;

pub(crate) trait Operation: Debug {
    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex));

    fn inputs(&self, nodes: &ComputeGraphInner) -> Vec<MirValue>;

    fn name(&self) -> String;

    fn output_layout(&self, _: &FxHashMap<NodeIndex, TensorLayoutInfo>) -> TensorLayoutInfo {
        todo!()
    }

    fn build_tensor_ir(
        &self,
        nodes: &ComputeGraphInner,
        inputs: &[MirValue],
    ) -> Result<TensorIrLowering, String>;
}

pub(crate) struct TensorIrLowering {
    pub(crate) program: tensor_ir::TensorExprProgram,
    pub(crate) inputs: Vec<TensorData>,
    pub(crate) output_shape: Box<[usize]>,
    pub(crate) output_datatype: DataTypeEnum,
}
