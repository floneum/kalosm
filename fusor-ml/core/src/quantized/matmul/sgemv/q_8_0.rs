use crate::{
    DataTypeEnum,
    mir::{
        inputs::{QMatrixInput, TensorInput},
        kernel::GenericKernel,
        workgroup_shape::WorkgroupShape,
    },
    quantized::matmul::{QMatMulOperation, sgemv::decompose_workgroup_index},
    util::{maybe_vec_storage_index, maybe_vec_storage_subgroup_add, maybe_vec_storage_type},
};
use std::fmt::Write;
use std::sync::OnceLock;

pub(crate) const Q_8_0_SGEMV_CHUNK_SIZE: u32 = 4; // This is the size of the chunk each thread will process at a time
const STEP_SIZE: u32 = 8;
const SUBGROUP_COUNT: u32 = 2;
const SUBGROUP_SIZE: u32 = 32;

/// Generate WGSL for computing one row's contribution in the inner loop
fn generate_row_computation(
    kernel: &mut GenericKernel,
    offset: u32,
    dtype: DataTypeEnum,
    input_b: &QMatrixInput,
) {
    writeln!(kernel, "var local_sum = {dtype}();").unwrap();
    for data_offset in 0..(STEP_SIZE / 4) {
        writeln!(kernel, "{{").unwrap();
        writeln!(kernel, "let block = vec4<{dtype}>(unpack4xI8({input_b}[block_offset].data[thread_local_id * 2u + {data_offset}]));").unwrap();
        writeln!(kernel, "let float_block = vec4<{dtype}>(cached_a_values[{data_offset} * 4u + 0], cached_a_values[{data_offset} * 4u + 1], cached_a_values[{data_offset} * 4u + 2], cached_a_values[{data_offset} * 4u + 3]);").unwrap();
        writeln!(kernel, "local_sum += dot(block, float_block);").unwrap();
        writeln!(kernel, "}}").unwrap();
    }
    let indexed = maybe_vec_storage_index(Q_8_0_SGEMV_CHUNK_SIZE, "sum", offset);
    writeln!(
        kernel,
        "{indexed} += local_sum * {dtype}({input_b}[block_offset].scale);"
    )
    .unwrap();
    writeln!(kernel, "block_offset += k_block_size;").unwrap();
}

/// Generate WGSL for writing one row's output
fn generate_row_output(
    kernel: &mut GenericKernel,
    offset: u32,
    input_datatype: DataTypeEnum,
    output: &TensorInput,
    post_element_wise_functions: &[crate::mir::function::Function],
) {
    write!(kernel, "{output}[").unwrap();
    let mut output_indices = Vec::new();
    for dim in (0..output.rank()).rev().skip(2) {
        output_indices.push(format!("batch_idx_{dim}"));
    }
    output_indices.push("m_idx".to_string());
    output_indices.push("row_index".to_string());
    output.strided_index(kernel, output_indices);
    let indexed = maybe_vec_storage_index(Q_8_0_SGEMV_CHUNK_SIZE, "sum", offset);
    let result = post_element_wise_functions
        .iter()
        .fold(format!("{input_datatype}({indexed})"), |acc, f| {
            f.call(vec![acc])
        });
    writeln!(kernel, "] = {result};").unwrap();
}

// https://github.com/ggml-org/llama.cpp/blob/6efcd65945a98cf6883cdd9de4c8ccd8c79d219a/ggml/src/ggml-metal/ggml-metal.metal#L2452
#[allow(clippy::too_many_arguments)]
pub(crate) fn q_8_0_sgemv(
    op: &QMatMulOperation,
    kernel: &mut GenericKernel,
    workgroup_shape: &WorkgroupShape,
    input_a: &TensorInput,
    input_b: &QMatrixInput,
    output: &TensorInput,
    n_size: &str,
    m_size: &str,
    k_size: &str,
) {
    let input_datatype = op.input_datatype;
    let dtype = op.matmul_datatype();
    let subgroup_index = kernel.subgroup_index();
    let subgroup_local_index = kernel.subgroup_local_index();
    let elements_per_block = op.elements_per_block();
    let pre_element_wise_functions = OnceLock::new();
    let post_element_wise_functions = OnceLock::new();

    // Calculate n_workgroups for this kernel type (SUBGROUP_COUNT subgroups per workgroup, Q_8_0_SGEMV_CHUNK_SIZE per subgroup)
    let chunk_size = Q_8_0_SGEMV_CHUNK_SIZE * SUBGROUP_COUNT;
    let n_workgroups = format!("(({n_size} + {chunk_size} - 1) / {chunk_size})");

    // Decompose linearized workgroup index into (n_workgroup_idx, m_idx, batch_idx)
    decompose_workgroup_index(kernel, workgroup_shape, m_size, &n_workgroups);

    // Decompose the batch index for higher-dimensional tensors
    writeln!(kernel, "var batch_idx_remaining = batch_idx;").unwrap();
    for dim in (0..input_a.rank()).rev().skip(2) {
        let shape = input_a.shape_binding(dim);
        writeln!(
            kernel,
            "let batch_idx_{dim} = batch_idx_remaining % {shape};"
        )
        .unwrap();
        writeln!(
            kernel,
            "batch_idx_remaining = batch_idx_remaining / {shape};"
        )
        .unwrap();
    }

    // Find the reduce size in blocks rounded up
    writeln!(
        kernel,
        "let k_block_size = ({k_size} + {elements_per_block} - 1) / {elements_per_block};"
    )
    .unwrap();

    // Workgroup offset in the N dimension (from decomposed linearized index)
    writeln!(kernel, "let workgroup_offset = n_workgroup_idx;").unwrap();
    writeln!(
        kernel,
        "let row = ({SUBGROUP_COUNT} * workgroup_offset + {subgroup_index}) * {Q_8_0_SGEMV_CHUNK_SIZE};"
    )
    .unwrap();

    // Fast path check: if all rows in this tile are valid, skip per-row bounds checks
    writeln!(
        kernel,
        "let is_full_tile = row + {Q_8_0_SGEMV_CHUNK_SIZE} <= {n_size};"
    )
    .unwrap();

    writeln!(kernel, "let row_block_offset = row * k_block_size;").unwrap();

    writeln!(kernel, "let thread_id = {subgroup_local_index} / 4;").unwrap();
    writeln!(kernel, "let thread_local_id = {subgroup_local_index} % 4;").unwrap();

    writeln!(kernel, "let lane_index = thread_local_id * {STEP_SIZE};").unwrap();

    writeln!(
        kernel,
        "var y_offset = thread_id * {elements_per_block} + lane_index;"
    )
    .unwrap();

    let sum_storage_type = maybe_vec_storage_type(Q_8_0_SGEMV_CHUNK_SIZE, dtype);
    writeln!(kernel, "var sum = {sum_storage_type}();",).unwrap();

    writeln!(
        kernel,
        "var cached_a_values = array<{dtype}, {STEP_SIZE}>();",
    )
    .unwrap();

    // Loop over all of the blocks this thread is responsible for
    writeln!(
        kernel,
        "for (var i = thread_id; i < k_block_size; i += {SUBGROUP_SIZE}/4u) {{"
    )
    .unwrap();
    {
        // First load the values of a into cached_a_values
        for j in 0..STEP_SIZE {
            writeln!(kernel, "{{").unwrap();
            let pre_element_wise_functions = pre_element_wise_functions
                .get_or_init(|| op.pre_element_wise.add_functions(kernel));
            let mut raw_input = String::new();
            write!(&mut raw_input, "{input_a}[").unwrap();
            let mut indices = Vec::new();
            // Add batch indices first
            for dim in (0..input_a.rank()).rev().skip(2) {
                indices.push(format!("batch_idx_{dim}"));
            }
            // Then add M and K indices
            indices.push("m_idx".to_string());
            indices.push(format!("y_offset + {j}"));
            input_a.strided_index(&mut raw_input, indices);
            write!(&mut raw_input, "]").unwrap();
            let processed = pre_element_wise_functions
                .iter()
                .fold(raw_input, |acc, f| f.call(vec![acc]));
            writeln!(kernel, "cached_a_values[{j}] = {processed};").unwrap();

            writeln!(kernel, "}}").unwrap();
        }
        writeln!(kernel, "var block_offset = row_block_offset + i;").unwrap();

        // Fast path: all rows in tile are valid, no bounds checks needed
        writeln!(kernel, "if is_full_tile {{").unwrap();
        for offset in 0..Q_8_0_SGEMV_CHUNK_SIZE {
            writeln!(kernel, "{{").unwrap();
            generate_row_computation(kernel, offset, dtype, input_b);
            writeln!(kernel, "}}").unwrap();
        }
        writeln!(kernel, "}} else {{").unwrap();
        // Slow path: check bounds for each row
        for offset in 0..Q_8_0_SGEMV_CHUNK_SIZE {
            writeln!(kernel, "{{").unwrap();
            writeln!(kernel, "let row_index = row + {offset};").unwrap();
            writeln!(kernel, "if row_index < {n_size} {{").unwrap();
            generate_row_computation(kernel, offset, dtype, input_b);
            writeln!(kernel, "}}").unwrap();
            writeln!(kernel, "}}").unwrap();
        }
        writeln!(kernel, "}}").unwrap();

        writeln!(kernel, "y_offset += {elements_per_block} * {STEP_SIZE};").unwrap();
    }
    writeln!(kernel, "}}").unwrap();

    // Get the sum among all threads in the subgroup
    writeln!(
        kernel,
        "sum = {};",
        maybe_vec_storage_subgroup_add(Q_8_0_SGEMV_CHUNK_SIZE, "sum")
    )
    .unwrap();

    // Initialize post element-wise functions once before the loop
    let post_fns = post_element_wise_functions
        .get_or_init(|| op.post_element_wise.add_functions(kernel));

    // Fast path: all rows in tile are valid, no bounds checks needed
    writeln!(kernel, "if is_full_tile {{").unwrap();
    for offset in 0..Q_8_0_SGEMV_CHUNK_SIZE {
        writeln!(kernel, "{{").unwrap();
        writeln!(kernel, "if {subgroup_local_index} == 0u {{").unwrap();
        writeln!(kernel, "let row_index = row + {offset}u;").unwrap();
        generate_row_output(kernel, offset, input_datatype, output, post_fns);
        writeln!(kernel, "}}").unwrap();
        writeln!(kernel, "}}").unwrap();
    }
    writeln!(kernel, "}} else {{").unwrap();
    // Slow path: check bounds for each row
    for offset in 0..Q_8_0_SGEMV_CHUNK_SIZE {
        writeln!(kernel, "{{").unwrap();
        writeln!(kernel, "if {subgroup_local_index} == 0u {{").unwrap();
        writeln!(kernel, "let row_index = row + {offset}u;").unwrap();
        writeln!(kernel, "if row_index < {n_size} {{").unwrap();
        generate_row_output(kernel, offset, input_datatype, output, post_fns);
        writeln!(kernel, "}}").unwrap();
        writeln!(kernel, "}}").unwrap();
        writeln!(kernel, "}}").unwrap();
    }
    writeln!(kernel, "}}").unwrap();
}
