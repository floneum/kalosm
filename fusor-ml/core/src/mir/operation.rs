use std::{any::TypeId, fmt::Debug, hash::Hash};

use rustc_hash::FxHasher;

use crate::{
    Device,
    compute_graph::{ComputeGraphInner, NodeIndex},
};

use super::{
    inputs::MirValue,
    kernel_backend,
    kernel_backend::DirectKernel,
    workgroup_shape::{WorkgroupShape, WorkgroupShapeConstraints},
};

/// The complete direct-kernel lowering for one operation.
///
/// Most operations produce exactly one kernel. Some valid operations produce
/// no dispatch for an empty output, while others require an ordered sequence
/// of kernels. Keeping those cases in the operation interface lets executors
/// treat every operation uniformly.
pub(crate) struct DirectKernelPlan {
    kernels: Vec<DirectKernel>,
}

impl DirectKernelPlan {
    pub(crate) fn empty() -> Self {
        Self {
            kernels: Vec::new(),
        }
    }

    pub(crate) fn single(kernel: DirectKernel) -> Self {
        Self {
            kernels: vec![kernel],
        }
    }

    pub(crate) fn many(kernels: Vec<DirectKernel>) -> Self {
        Self { kernels }
    }

    pub(crate) fn into_kernels(self) -> Vec<DirectKernel> {
        self.kernels
    }
}

#[derive(Debug, Clone)]
pub(crate) struct DirectKernelLoweringError {
    operation: String,
}

impl DirectKernelLoweringError {
    pub(crate) fn new(operation: String) -> Self {
        Self { operation }
    }
}

impl std::fmt::Display for DirectKernelLoweringError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "operation did not provide a direct kernel plan: {}",
            self.operation
        )
    }
}

impl std::error::Error for DirectKernelLoweringError {}

pub(crate) trait Operation: Debug + Send + Sync + 'static {
    fn workgroup_shape_constraints(&self, device: &Device) -> WorkgroupShapeConstraints;

    fn dispatch_size(&self, workgroup_shape: &WorkgroupShape, inputs: &[MirValue]) -> [u32; 3];

    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex));

    /// Visit the same dependency slots as [`Self::visit_dependencies`], in
    /// the same order, so callers can rebind them. Materializing an operation
    /// that was interned from a structurally identical instance rewrites
    /// every slot through this visitor.
    fn visit_dependencies_mut(&mut self, f: &mut dyn FnMut(&mut NodeIndex));

    fn inputs(&self, nodes: &ComputeGraphInner) -> Vec<MirValue>;

    fn output(&self, nodes: &ComputeGraphInner, inputs: &[MirValue]) -> MirValue;

    fn build_direct_kernel(
        &self,
        nodes: &ComputeGraphInner,
        workgroup_shape: &WorkgroupShape,
        inputs: &[MirValue],
    ) -> Option<DirectKernel>;

    /// Lower this operation to its complete zero-, one-, or many-kernel plan.
    /// Operations with a singular lowering only implement
    /// [`Self::build_direct_kernel`]; multi-dispatch and empty-output
    /// operations override this method.
    fn build_direct_kernel_plan(
        &self,
        nodes: &ComputeGraphInner,
        workgroup_shape: &WorkgroupShape,
        inputs: &[MirValue],
    ) -> Result<DirectKernelPlan, DirectKernelLoweringError> {
        self.build_direct_kernel(nodes, workgroup_shape, inputs)
            .map(DirectKernelPlan::single)
            .ok_or_else(|| DirectKernelLoweringError::new(self.name()))
    }

    fn name(&self) -> String;

    /// Hash structural operation fields that affect generated kernel IR.
    ///
    /// The concrete operation type is added by `kernel_cache_key_with_dispatch`;
    /// implementations only hash fields not represented by MIR inputs,
    /// dispatch, or workgroup shape.
    fn hash_kernel_fields(&self, state: &mut FxHasher);

    fn kernel_cache_key_with_dispatch(
        &self,
        variant: kernel_backend::KernelVariantKey,
        workgroup_shape: Option<&WorkgroupShape>,
        dispatch_size: [u32; 3],
        inputs: &[MirValue],
    ) -> kernel_backend::KernelCacheKey {
        kernel_backend::KernelCacheKey::from_hash_inputs(|hasher| {
            // Version the shared key layout so future changes cannot silently
            // collide with cache entries produced by an older hash recipe.
            1u64.hash(hasher);
            variant.hash(hasher);
            TypeId::of::<Self>().hash(hasher);
            self.hash_kernel_fields(hasher);
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
}

pub(crate) fn hash_mir_value(state: &mut FxHasher, value: &MirValue) {
    std::mem::discriminant(value).hash(state);
    match value {
        MirValue::QMatrix(matrix) => {
            matrix.datatype().hash(state);
            matrix.storage_layout().hash(state);
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

pub(crate) fn hash_layout(state: &mut FxHasher, layout: &crate::Layout) {
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
