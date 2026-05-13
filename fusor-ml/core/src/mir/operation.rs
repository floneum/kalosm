use std::{fmt::Debug, hash::Hash};

use rustc_hash::FxHasher;

use crate::{
    Device,
    compute_graph::{ComputeGraphInner, NodeIndex},
};

use super::{
    direct_kernel::DirectKernel,
    inputs::MirValue,
    kernel_backend,
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

    #[allow(dead_code)]
    fn requires_single_kernel_batch(&self) -> bool {
        false
    }

    fn name(&self) -> String;

    /// Hash the structural operation state that affects generated kernel IR.
    ///
    /// The default stays on the object-safe `Operation` surface: operation
    /// name plus `kernel_module_key_with_dispatch`'s MIR inputs, dispatch, and
    /// workgroup shape. Implementations only override this when generated IR
    /// depends on fields not represented by that trait-level data.
    fn hash_kernel_signature(&self, state: &mut FxHasher) {
        self.name().hash(state);
    }

    fn kernel_module_key_with_dispatch(
        &self,
        label: &str,
        workgroup_shape: Option<&WorkgroupShape>,
        dispatch_size: [u32; 3],
        inputs: &[MirValue],
    ) -> [u64; 2] {
        kernel_backend::module_key_from(|hasher| {
            // Version the shared key layout so future changes cannot silently
            // collide with cache entries produced by an older hash recipe.
            1u64.hash(hasher);
            label.hash(hasher);
            self.hash_kernel_signature(hasher);
            workgroup_shape
                .map(|workgroup_shape| workgroup_shape.shape())
                .hash(hasher);
            dispatch_size.hash(hasher);
            inputs.len().hash(hasher);
            for input in inputs {
                hash_mir_value(hasher, input);
            }
        })
    }

    fn kernel_cache_key_with_dispatch(
        &self,
        label: &str,
        workgroup_shape: Option<&WorkgroupShape>,
        dispatch_size: [u32; 3],
        inputs: &[MirValue],
    ) -> String {
        kernel_backend::hashed_cache_key(
            label,
            self.kernel_module_key_with_dispatch(label, workgroup_shape, dispatch_size, inputs),
        )
    }
}

fn hash_mir_value(state: &mut FxHasher, value: &MirValue) {
    std::mem::discriminant(value).hash(state);
    match value {
        MirValue::QMatrix(matrix) => {
            matrix.datatype().hash(state);
            matrix.shape().hash(state);
        }
        MirValue::Tensor(tensor) => {
            tensor.datatype().hash(state);
            hash_layout(state, tensor.layout());
            layout_allocation_len(tensor.layout()).hash(state);
        }
        MirValue::Integer(value) => value.hash(state),
        MirValue::Float(value) => value.to_bits().hash(state),
    }
}

fn hash_layout(state: &mut FxHasher, layout: &crate::Layout) {
    layout.offset().hash(state);
    layout.shape().hash(state);
    layout.strides().hash(state);
}

fn layout_allocation_len(layout: &crate::Layout) -> Option<u32> {
    let max_index = layout
        .shape()
        .iter()
        .zip(layout.strides())
        .try_fold(layout.offset(), |acc, (dim, stride)| {
            acc.checked_add(dim.saturating_sub(1).checked_mul(*stride)?)
        })?;
    max_index.checked_add(1)?.try_into().ok()
}
