use std::ops::Range;

use crate::{
    Tensor, TensorData,
    compute_graph::{ComputeGraphInner, NodeIndex},
    mir::{inputs::MirValue, operation::Operation},
};

#[derive(Clone, Debug)]
pub(crate) struct SliceAssignOperation {
    pub(crate) input: NodeIndex,
    pub(crate) value: NodeIndex,
    pub(crate) slices: Box<[Range<usize>]>,
}

impl SliceAssignOperation {
    pub fn new(input: NodeIndex, value: NodeIndex, slices: Box<[Range<usize>]>) -> Self {
        Self {
            input,
            value,
            slices,
        }
    }
}

impl Operation for SliceAssignOperation {
    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        f(self.value);
        f(self.input);
    }

    fn inputs(&self, nodes: &ComputeGraphInner) -> Vec<MirValue> {
        // Pass the ORIGINAL input tensor (not sliced) and the value tensor
        let input = nodes.get_cached_result(self.input).unwrap();
        let value = nodes.get_cached_result(self.value).unwrap();

        // Create output buffer with the same shape as input
        let output =
            TensorData::new_for_shape(input.device(), input.layout().shape(), input.datatype());

        vec![input.clone().into(), value.clone().into(), output.into()]
    }

    fn name(&self) -> String {
        format!(
            "slice_assign_{}",
            self.slices
                .iter()
                .map(|slice| format!("{slice:?}"))
                .collect::<Vec<_>>()
                .join("_")
        )
    }

    fn output_layout(
        &self,
        map: &rustc_hash::FxHashMap<NodeIndex, crate::TensorLayoutInfo>,
    ) -> crate::TensorLayoutInfo {
        // Output has the same layout as input
        map.get(&self.input).unwrap().clone()
    }

    fn build_tensor_ir(
        &self,
        _nodes: &ComputeGraphInner,
        inputs: &[MirValue],
    ) -> Result<crate::mir::operation::TensorIrLowering, String> {
        crate::tensor_ir_lowering::slice_assign(self, inputs)
    }
}

impl<const R: usize, T: crate::DataType> Tensor<R, T> {
    pub fn slice_assign(&self, slices: [Range<usize>; R], value: &Self) -> Self {
        self.add_slice_assign(value, slices)
    }
}
