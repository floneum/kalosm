use fusor_gguf::GgmlType;

use std::fmt::Write;

use crate::{
    Device,
    mir::{
        inputs::{QMatrixInput, TensorInput},
        kernel::GenericKernel,
        workgroup_shape::WorkgroupShape,
    },
    quantized::matmul::{
        QMatMulOperation,
        sgemv::{
            general::general_sgemv, q_8_0::q_8_0_sgemv, q_n::q_n_sgemv, q4k::q4k_sgemv,
            q5k::q5k_sgemv, q6k::q6k_sgemv,
        },
    },
};
use crate::{
    QMatrix,
    quantized::matmul::sgemv::{
        q_8_0::Q_8_0_SGEMV_CHUNK_SIZE, q_n::Q_N_SGEMV_CHUNK_SIZE, q4k::Q4K_SGEMV_CHUNK_SIZE,
        q5k::Q5K_SGEMV_CHUNK_SIZE, q6k::Q6K_SGEMV_CHUNK_SIZE,
    },
    visit_tiled::distribute_workgroups,
};

mod general;
pub mod q4k;
pub mod q5k;
pub mod q6k;
pub mod q_8_0;
pub mod q_n;

pub(crate) const SGEMV_CHUNK_SIZE: u32 = 2; // This is the size of the chunk each thread will process at a time
pub(crate) const SGEMV_VECTOR_SIZE: u32 = 4; // This is the size of the chunk we will dot at a time

/// Decompose a linearized workgroup index into (n_workgroup_idx, m_idx, batch_idx)
/// The linearization order is: n_workgroup_idx + m_idx * n_workgroups + batch_idx * n_workgroups * m_size
/// This is used when workgroups are distributed across multiple dispatch dimensions to avoid
/// exceeding the max 65535 workgroups per dimension limit in WebGPU.
pub(crate) fn decompose_workgroup_index(
    kernel: &mut GenericKernel,
    workgroup_shape: &WorkgroupShape,
    m_size: &str,
    n_workgroups: &str,
) {
    let linearized = workgroup_shape.linearized_workgroup_index(kernel);

    // Decompose linearized index: batch_idx * (n_workgroups * m_size) + m_idx * n_workgroups + n_workgroup_idx
    writeln!(kernel, "let n_workgroups = {n_workgroups};").unwrap();
    writeln!(kernel, "let linearized_wg = {linearized};").unwrap();
    writeln!(
        kernel,
        "let batch_idx = linearized_wg / (n_workgroups * {m_size});"
    )
    .unwrap();
    writeln!(
        kernel,
        "let remaining = linearized_wg % (n_workgroups * {m_size});"
    )
    .unwrap();
    writeln!(kernel, "let m_idx = remaining / n_workgroups;").unwrap();
    writeln!(kernel, "let n_workgroup_idx = remaining % n_workgroups;").unwrap();
}

/// Check if the device can support specialized SGEMV kernels that require 2 subgroups per workgroup.
/// This requires the workgroup to be large enough to fit 2 subgroups, which means:
/// max_subgroup_size >= 2 * min_subgroup_size
fn can_use_specialized_sgemv(device: &Device) -> bool {
    if !device.subgroups_supported() {
        return false;
    }
    // The workgroup is constrained to be <= max_subgroup_size, so we need
    // max_subgroup_size >= 2 * min_subgroup_size to fit 2 subgroups
    device.max_subgroup_size() >= 2 * device.min_subgroup_size()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn sgemv(
    op: &QMatMulOperation,
    generic_kernel: &mut GenericKernel,
    workgroup_size: &WorkgroupShape,
    input_a: &TensorInput,
    input_b: &QMatrixInput,
    output: &TensorInput,
    _n_size: &str,
    _m_size: &str,
    k_size: &str,
    graph: &crate::compute_graph::ComputeGraphInner,
) {
    let device = graph.device();
    // Check if we can use specialized SGEMV (requires 2 subgroups per workgroup)
    let use_specialized = can_use_specialized_sgemv(&device);
    match op.matrix.datatype {
        GgmlType::Q6K if use_specialized => q6k_sgemv(
            op,
            generic_kernel,
            workgroup_size,
            input_a,
            input_b,
            output,
            _n_size,
            _m_size,
            k_size,
        ),
        GgmlType::Q4K if use_specialized => q4k_sgemv(
            op,
            generic_kernel,
            workgroup_size,
            input_a,
            input_b,
            output,
            _n_size,
            _m_size,
            k_size,
        ),
        GgmlType::Q5K if use_specialized => q5k_sgemv(
            op,
            generic_kernel,
            workgroup_size,
            input_a,
            input_b,
            output,
            _n_size,
            _m_size,
            k_size,
        ),
        GgmlType::Q4_0 | GgmlType::Q5_0 if use_specialized => q_n_sgemv(
            op,
            generic_kernel,
            workgroup_size,
            input_a,
            input_b,
            output,
            _n_size,
            _m_size,
            k_size,
        ),
        GgmlType::Q8_0 if use_specialized => q_8_0_sgemv(
            op,
            generic_kernel,
            workgroup_size,
            input_a,
            input_b,
            output,
            _n_size,
            _m_size,
            k_size,
        ),
        _ => general_sgemv(
            op,
            generic_kernel,
            workgroup_size,
            input_a,
            input_b,
            output,
            _n_size,
            _m_size,
            k_size,
            graph,
        ),
    }
}

/// Calculate the number of N-dimension workgroups based on matrix type
pub(crate) fn n_workgroups(matrix: &QMatrix, n: u32) -> u32 {
    // Only use specialized dispatch sizes if we can use specialized SGEMV
    if can_use_specialized_sgemv(&matrix.device) {
        if matrix.datatype == GgmlType::Q6K {
            n.div_ceil(Q6K_SGEMV_CHUNK_SIZE * 2)
        } else if matrix.datatype == GgmlType::Q4K {
            n.div_ceil(Q4K_SGEMV_CHUNK_SIZE * 2)
        } else if matrix.datatype == GgmlType::Q5K {
            n.div_ceil(Q5K_SGEMV_CHUNK_SIZE * 2)
        } else if matches!(matrix.datatype, GgmlType::Q4_0 | GgmlType::Q5_0) {
            n.div_ceil(Q_N_SGEMV_CHUNK_SIZE * 2)
        } else if matches!(matrix.datatype, GgmlType::Q8_0) {
            n.div_ceil(Q_8_0_SGEMV_CHUNK_SIZE * 2)
        } else {
            n.div_ceil(SGEMV_CHUNK_SIZE)
        }
    } else {
        n.div_ceil(SGEMV_CHUNK_SIZE)
    }
}

pub(crate) fn dispatch_size(matrix: &QMatrix, n: u32, m: u32, batch_size: u32) -> [u32; 3] {
    // Calculate total workgroups: n_workgroups * m * batch
    // Use distribute_workgroups to spread across all 3 dimensions if needed
    let n_wg = n_workgroups(matrix, n);
    let total_workgroups = n_wg * m * batch_size;
    distribute_workgroups(total_workgroups)
}

pub(crate) fn workgroup_shape_constraints(
    _: &QMatrix,
    device: &Device,
) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
    let mut constraints = crate::mir::workgroup_shape::WorkgroupShapeConstraints::default();
    if device.subgroups_supported() {
        constraints.add_constraint(
            0,
            crate::mir::workgroup_shape::Constraint::more_than_or_equals(
                device.min_subgroup_size(),
            ),
        );
        constraints.add_constraint(
            0,
            crate::mir::workgroup_shape::Constraint::less_than_or_equals(
                device.max_subgroup_size(),
            ),
        );
    }
    constraints.add_constraint(1, crate::mir::workgroup_shape::Constraint::Equals(1));
    constraints.add_constraint(2, crate::mir::workgroup_shape::Constraint::Equals(1));
    constraints
}
