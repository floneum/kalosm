use std::fmt::Debug;

use rustc_hash::FxHashMap;

use crate::{
    Device, TensorLayoutInfo,
    compute_graph::{ComputeGraphInner, NodeIndex},
};

use super::{
    direct_kernel::DirectKernel,
    inputs::MirValue,
    workgroup_shape::{WorkgroupShape, WorkgroupShapeConstraints},
};

pub(crate) trait Operation: Debug {
    fn workgroup_shape_constraints(&self, device: &Device) -> WorkgroupShapeConstraints;

    fn dispatch_size(&self, workgroup_shape: &WorkgroupShape, inputs: &[MirValue]) -> [u32; 3];

    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex));

    fn inputs(&self, nodes: &ComputeGraphInner) -> Vec<MirValue>;

    fn output(&self, nodes: &ComputeGraphInner, inputs: &[MirValue]) -> MirValue;

    fn build_direct_kernel(
        &self,
        nodes: &ComputeGraphInner,
        workgroup_shape: &WorkgroupShape,
        inputs: &[MirValue],
    ) -> Option<DirectKernel>;

    fn requires_single_kernel_batch(&self) -> bool {
        false
    }

    fn name(&self) -> String;

    fn output_layout(&self, inputs: &FxHashMap<NodeIndex, TensorLayoutInfo>) -> TensorLayoutInfo;
}
