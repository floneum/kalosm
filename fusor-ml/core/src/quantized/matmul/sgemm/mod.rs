use crate::{
    Device, QMatrix, dequantize_mat4x4_block_count,
    mir::{
        inputs::{QMatrixInput, TensorInput},
        kernel::GenericKernel,
        workgroup_shape::{Constraint, WorkgroupShape},
    },
    quantized::matmul::QMatMulOperation,
};

mod chunked;
mod general;

pub use chunked::{ChunkedSgemmConfig, chunked_sgemm_with_config};
pub use general::{GeneralSgemmConfig, general_sgemm_with_config};

#[allow(clippy::too_many_arguments)]
pub(crate) fn sgemm(
    op: &QMatMulOperation,
    generic_kernel: &mut GenericKernel,
    _: &WorkgroupShape,
    input_a: &TensorInput,
    input_b: &QMatrixInput,
    output: &TensorInput,
    _n_size: &str,
    // m size is always 1 for sgemv
    _m_size: &str,
    k_size: &str,
    graph: &crate::compute_graph::ComputeGraphInner,
) {
    // Use chunked sgemm for all types that support mat4x4 dequantization
    if dequantize_mat4x4_block_count(input_b.datatype) > 0 && graph.device().subgroups_supported() {
        let config = op.chunked_config.unwrap_or(ChunkedSgemmConfig::default());
        chunked_sgemm_with_config(op, generic_kernel, input_a, input_b, output, k_size, config);
    } else {
        let config = op.general_config.unwrap_or(GeneralSgemmConfig::default());
        general_sgemm_with_config(
            op,
            generic_kernel,
            input_a,
            input_b,
            output,
            _n_size,
            _m_size,
            k_size,
            config,
        );
    }
}

pub(crate) fn dispatch_size(
    op: &QMatMulOperation,
    workgroup_shape: &WorkgroupShape,
    matrix: &QMatrix,
    n: u32,
    m: u32,
    batch_size: u32,
) -> [u32; 3] {
    // Use chunked dispatch size for all types that support mat4x4 dequantization
    if dequantize_mat4x4_block_count(matrix.datatype()) > 0 && matrix.device().subgroups_supported()
    {
        let config = op.chunked_config.unwrap_or(ChunkedSgemmConfig::default());
        [
            m.div_ceil(workgroup_shape.y() * config.m_results_per_thread * 4),
            n.div_ceil(workgroup_shape.x() * config.n_results_per_thread * 4),
            batch_size.div_ceil(workgroup_shape.z()),
        ]
    } else {
        [
            n.div_ceil(workgroup_shape.x()),
            m.div_ceil(workgroup_shape.y()),
            batch_size.div_ceil(workgroup_shape.z()),
        ]
    }
}

pub(crate) fn workgroup_shape_constraints(
    matrix: &QMatrix,
    _device: &Device,
) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
    // Use chunked workgroup constraints for all types that support mat4x4 dequantization
    if dequantize_mat4x4_block_count(matrix.datatype()) > 0 {
        let mut constraints = crate::mir::workgroup_shape::WorkgroupShapeConstraints::default();
        constraints.add_constraint(0, Constraint::equals(16));
        constraints.add_constraint(1, Constraint::equals(16));
        constraints.add_constraint(2, Constraint::equals(1));
        return constraints;
    }
    let mut constraints = crate::mir::workgroup_shape::WorkgroupShapeConstraints::default();
    let second_dim = matrix.shape()[1];
    let second_dim_block_width = second_dim.div_ceil(matrix.datatype().block_size());
    constraints.add_constraint(
        1,
        Constraint::less_than_or_equals(second_dim_block_width as _),
    );
    constraints.add_constraint(2, Constraint::equals(1));
    constraints
}
