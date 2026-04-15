use std::fmt::Write;
use std::sync::OnceLock;

use crate::{
    MatMulOperation,
    mir::{
        globals::KernelGlobalSpace, inputs::TensorInput, kernel::GenericKernel,
        workgroup_shape::WorkgroupShape,
    },
    util::{
        maybe_vec_dot, maybe_vec_storage_add, maybe_vec_storage_index,
        maybe_vec_storage_subgroup_add, maybe_vec_storage_type, maybe_vec_storage_type_enum,
    },
};

#[allow(clippy::too_many_arguments)]
pub(crate) fn sgemv(
    op: &MatMulOperation,
    kernel: &mut GenericKernel,
    workgroup_size: &WorkgroupShape,
    input_a: &TensorInput,
    input_b: &TensorInput,
    output: &TensorInput,
    n_size: &str,
    m_size: &str,
    k_size: &str,
    params: &SgemvParams,
    graph: &crate::compute_graph::ComputeGraphInner,
) {
    let blocksize = workgroup_size.x();
    let dtype = op.matmul_dtype();
    let workgroup_index = kernel.workgroup_index();
    let workgroup_local_index = kernel.workgroup_local_index();
    let device = graph.device();

    let chunk_size = params.chunk_size;
    let vector_size = params.vector_size;

    let input_a_val = graph.get_result(op.first).unwrap();
    let input_b_val = graph.get_result(op.second).unwrap();
    let input_a_datatype = input_a_val.datatype();
    let input_b_datatype = input_b_val.datatype();

    let pre_element_wise_functions: OnceLock<[Vec<_>; 2]> = OnceLock::new();
    let post_element_wise_functions = OnceLock::new();

    // Handle batch dimensions
    writeln!(kernel, "var block_batch = {workgroup_index}.z;").unwrap();

    // Decompose the batch index for higher-dimensional tensors
    for dim in (0..op.rank()).rev().skip(2) {
        let shape = input_a.shape_binding(dim);
        writeln!(kernel, "let block_batch_{dim} = block_batch % {shape};").unwrap();
        writeln!(kernel, "block_batch /= {shape};").unwrap();
    }

    // Find the batch offset for a, b and output
    for (name, tensor) in [("a", input_a), ("b", input_b), ("c", output)] {
        writeln!(kernel, "let {name}_start_index = ").unwrap();
        let offset = tensor.offset_binding();
        write!(kernel, "{offset}").unwrap();
        for dim in (0..op.rank()).rev().skip(2) {
            let stride = tensor.stride_binding(dim);
            write!(kernel, " + block_batch_{dim}*{stride}").unwrap();
        }
        writeln!(kernel, ";").unwrap();
    }

    // In index of the single element in the vector we are multiplying against
    writeln!(kernel, "let workgroup_offset = {workgroup_index}.x;").unwrap();

    // Column index to compute (for n > 1)
    writeln!(kernel, "let col_index = {workgroup_index}.y;").unwrap();

    let storage_type = maybe_vec_storage_type(chunk_size, dtype);

    // Get column strides for input_b and output
    let b_col_stride = input_b.stride_binding(op.rank() - 1);
    let b_row_stride = input_b.stride_binding(op.rank() - 2);
    let c_col_stride = output.stride_binding(op.rank() - 1);
    let c_row_stride = output.stride_binding(op.rank() - 2);

    writeln!(kernel, "var acc = {storage_type}();").unwrap();

    // Find this threads position in the workgroup
    writeln!(
        kernel,
        "var index = {workgroup_local_index} * {vector_size}u;"
    )
    .unwrap();

    let vec_storage = maybe_vec_storage_type(vector_size, dtype);

    // Initialize pre-element-wise functions once
    let pef = pre_element_wise_functions
        .get_or_init(|| std::array::from_fn(|i| op.pre_element_wise[i].add_functions(kernel)));

    // Loop over all of the vector chunks this thread is responsible for
    writeln!(kernel, "while (index < {k_size}) {{").unwrap();
    {
        // Load vector elements into b_reg from input_b. This will be reused for each offset in the chunk
        for i in 0..vector_size {
            writeln!(kernel, "let input_b_{i}_index = index + {i};").unwrap();
            let b_base = format!(
                "select({input_b_datatype}(0.0), {input_b}[b_start_index + input_b_{i}_index * {b_row_stride} + col_index * {b_col_stride}], input_b_{i}_index < {k_size})"
            );
            let b_val = pef[1].iter().fold(b_base, |acc, f| f.call(vec![acc]));
            writeln!(kernel, "let input_b_{i} = {b_val};").unwrap();
        }
        write!(kernel, "let reg_b = {vec_storage}(").unwrap();
        for i in 0..vector_size {
            if i > 0 {
                write!(kernel, ", ").unwrap();
            }
            write!(kernel, "input_b_{i}").unwrap();
        }
        writeln!(kernel, ");").unwrap();

        if chunk_size > 1 {
            writeln!(
                kernel,
                "for (var offset = 0u; offset < {chunk_size}; offset += 1u) {{"
            )
            .unwrap();
        }
        let row_index = if chunk_size > 1 {
            format!("(workgroup_offset * {chunk_size} + offset)")
        } else {
            "workgroup_offset".into()
        };

        // Load matrix row elements into a_reg from input_a
        for i in 0..vector_size {
            writeln!(kernel, "let input_a_{i}_index = index + {i};").unwrap();
            let a_row_stride = input_a.stride_binding(op.rank() - 2);
            let a_col_stride = input_a.stride_binding(op.rank() - 1);
            let a_base = format!(
                "select({input_a_datatype}(0.0), {input_a}[a_start_index + {row_index} * {a_row_stride} + input_a_{i}_index * {a_col_stride}], input_a_{i}_index < {k_size} && {row_index} < {m_size})"
            );
            let a_val = pef[0].iter().fold(a_base, |acc, f| f.call(vec![acc]));
            writeln!(kernel, "let input_a_{i} = {a_val};").unwrap();
        }
        // Then pack them into a vector and write to the cache
        write!(kernel, "let reg_a = {vec_storage}(").unwrap();
        for i in 0..vector_size {
            if i > 0 {
                write!(kernel, ", ").unwrap();
            }
            write!(kernel, "input_a_{i}").unwrap();
        }
        writeln!(kernel, ");").unwrap();

        // Compute dot product between the registers
        let acc_indexed = maybe_vec_storage_index(chunk_size, "acc", "offset");
        let dot = maybe_vec_dot(vector_size, "reg_a", "reg_b");
        writeln!(kernel, "{acc_indexed} += {dot};").unwrap();

        if chunk_size > 1 {
            writeln!(kernel, "}}").unwrap();
        }

        writeln!(kernel, "index += {blocksize}u * {vector_size}u;").unwrap();
    }

    writeln!(kernel, "}}").unwrap();

    // If subgroups are supported, perform a reduction with subgroup operations
    if device.subgroups_supported() {
        // Get the sum among all threads in the subgroup
        writeln!(
            kernel,
            "acc = {};",
            maybe_vec_storage_subgroup_add(chunk_size, "acc")
        )
        .unwrap();

        // We don't need to synchronize between the whole workgroup if there is only one subgroup
        let subgroup_size = device.min_subgroup_size();
        if blocksize > subgroup_size {
            let local_data = kernel.add_global_array(
                KernelGlobalSpace::Workgroup,
                maybe_vec_storage_type_enum(chunk_size, dtype),
                blocksize.to_string(),
            );
            let subgroup_id = kernel.subgroup_index();
            let subgroup_local_id = kernel.subgroup_local_index();
            let subgroups_per_workgroup = kernel.subgroups_per_workgroup();
            let subgroup_size = kernel.subgroup_size();
            let total_workgroup_size = workgroup_size.linearized();
            writeln!(kernel, "if {total_workgroup_size} > {subgroup_size} {{").unwrap();
            {
                // Write the output to the workgroup memory if this is the first thread in the subgroup
                writeln!(kernel, "if {subgroup_local_id} == 0u {{").unwrap();
                {
                    writeln!(kernel, "{local_data}[{subgroup_id}] = acc;").unwrap();
                }
                writeln!(kernel, "}}").unwrap();

                // Wait until all threads have written to the workgroup shared memory
                writeln!(kernel, "workgroupBarrier();").unwrap();

                // Then if this is the first subgroup, do one final shuffle down reduction
                writeln!(kernel, "if {subgroup_id} == 0u {{").unwrap();
                {
                    // Copy over the final value from each subgroup from the workgroup shared memory to the acc variable
                    writeln!(
                        kernel,
                        "if {subgroup_local_id} < {subgroups_per_workgroup} {{"
                    )
                    .unwrap();
                    {
                        writeln!(kernel, "acc = {local_data}[{subgroup_local_id}];").unwrap();
                    }
                    writeln!(kernel, "}}").unwrap();
                    writeln!(kernel, "else {{").unwrap();
                    {
                        writeln!(kernel, "acc = {storage_type}();").unwrap();
                    }
                    writeln!(kernel, "}}").unwrap();
                }
                writeln!(kernel, "}}").unwrap();

                // Finally get the final sum across all threads in the workgroup
                writeln!(
                    kernel,
                    "acc = {};",
                    maybe_vec_storage_subgroup_add(chunk_size, "acc")
                )
                .unwrap();
            }
            writeln!(kernel, "}}").unwrap();
        }
    }
    // Otherwise, perform the reduction in workgroup memory
    else {
        let local_data = kernel.add_global_array(
            KernelGlobalSpace::Workgroup,
            maybe_vec_storage_type_enum(chunk_size, dtype),
            blocksize.to_string(),
        );
        let mut offset = blocksize;
        while offset > 1 {
            // Write this thread's value to the shared memory
            writeln!(kernel, "{local_data}[{workgroup_local_index}] = acc;").unwrap();
            writeln!(kernel, "workgroupBarrier();").unwrap();
            offset /= 2;
            // Only threads in the first half do the reduction to avoid OOB reads
            // (workgroup memory OOB is undefined behavior in WebGPU)
            writeln!(kernel, "if {workgroup_local_index} < {offset}u {{").unwrap();
            writeln!(
                kernel,
                "let neighbor = {local_data}[{workgroup_local_index} + {offset}u];"
            )
            .unwrap();
            writeln!(
                kernel,
                "acc = {};",
                maybe_vec_storage_add(chunk_size, "acc", "neighbor")
            )
            .unwrap();
            writeln!(kernel, "}}").unwrap();
        }
    }

    // If this is not the first simd thread in the workgroup, we can return early
    writeln!(kernel, "if {workgroup_local_index} != 0u {{ return; }}").unwrap();

    // Write the output to the output tensor if this is the first thread in the workgroup
    if chunk_size > 1 {
        writeln!(
            kernel,
            "for (var offset = 0u; offset < {chunk_size}; offset += 1u) {{"
        )
        .unwrap();
    }
    {
        if chunk_size > 1 {
            writeln!(
                kernel,
                "let output_index = workgroup_offset * {chunk_size} + offset;"
            )
            .unwrap();
        } else {
            writeln!(kernel, "let output_index = workgroup_offset;").unwrap();
        }

        // Apply post element-wise operations
        let post_element_wise_functions =
            post_element_wise_functions.get_or_init(|| op.post_element_wise.add_functions(kernel));
        let result = post_element_wise_functions.iter().fold(
            maybe_vec_storage_index(chunk_size, "acc", "offset"),
            |acc, f| f.call(vec![acc]),
        );

        if chunk_size > 1 {
            writeln!(
                kernel,
                "if (output_index >= {m_size} || col_index >= {n_size}) {{ continue; }}"
            )
            .unwrap();
        }
        writeln!(
            kernel,
            "{output}[c_start_index + output_index * {c_row_stride} + col_index * {c_col_stride}] = {result};",
        )
        .unwrap();
    }
    if chunk_size > 1 {
        writeln!(kernel, "}}").unwrap();
    }
}

pub(crate) fn dispatch_size(
    m: u32,
    n: u32,
    batch_size: u32,
    _workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
    params: &SgemvParams,
) -> [u32; 3] {
    [m.div_ceil(params.chunk_size), n, batch_size]
}

pub(crate) fn workgroup_shape_constraints(
    _: &MatMulOperation,
    device: &crate::Device,
    params: &SgemvParams,
) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
    let mut constraints = crate::mir::workgroup_shape::WorkgroupShapeConstraints::default();
    constraints.add_constraint(
        0,
        crate::mir::workgroup_shape::Constraint::less_than(
            device.limits().max_compute_workgroup_size_x + 1,
        ),
    );
    constraints.add_constraint(
        0,
        crate::mir::workgroup_shape::Constraint::more_than_or_equals(device.min_subgroup_size()),
    );
    constraints.add_constraint(
        0,
        crate::mir::workgroup_shape::Constraint::less_than_or_equals(
            device.max_subgroup_size()
                * params
                    .subgroups_per_workgroup
                    .min(device.max_subgroup_size()),
        ),
    );
    constraints.add_constraint(1, crate::mir::workgroup_shape::Constraint::Equals(1));
    constraints.add_constraint(2, crate::mir::workgroup_shape::Constraint::Equals(1));
    constraints
}

#[derive(Debug, Clone)]
pub struct SgemvParams {
    chunk_size: u32,
    vector_size: u32,
    subgroups_per_workgroup: u32,
}

impl SgemvParams {
    pub fn new(chunk_size: u32, vector_size: u32, subgroups_per_workgroup: u32) -> Self {
        Self {
            chunk_size,
            vector_size,
            subgroups_per_workgroup,
        }
    }

    pub fn chunk_size(&self) -> u32 {
        self.chunk_size
    }

    pub fn vector_size(&self) -> u32 {
        self.vector_size
    }

    pub fn subgroups_per_workgroup(&self) -> u32 {
        self.subgroups_per_workgroup
    }
}

impl Default for SgemvParams {
    fn default() -> Self {
        SgemvParams::new(1, 4, 1)
    }
}
