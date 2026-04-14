use crate::{
    DataTypeEnum,
    mir::{
        inputs::{QMatrixInput, TensorInput},
        kernel::GenericKernel,
        workgroup_shape::WorkgroupShape,
    },
    quantized::matmul::{QMatMulOperation, sgemv::decompose_workgroup_index},
    shift_right_scale,
    util::{maybe_vec_storage_index, maybe_vec_storage_subgroup_add, maybe_vec_storage_type},
};
use std::fmt::Write;
use std::sync::OnceLock;

pub(crate) const Q_N_SGEMV_CHUNK_SIZE: u32 = 4; // This is the size of the chunk each thread will process at a time
const SUBGROUP_COUNT: u32 = 2;
const SUBGROUP_SIZE: u32 = 32;

// https://github.com/ggml-org/llama.cpp/blob/6efcd65945a98cf6883cdd9de4c8ccd8c79d219a/ggml/src/ggml-metal/ggml-metal.metal#L2329
#[allow(clippy::too_many_arguments)]
pub(crate) fn q_n_sgemv(
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
    let dtype = op.input_datatype;
    let subgroup_index = kernel.subgroup_index();
    let subgroup_local_index = kernel.subgroup_local_index();
    let elements_per_block = op.elements_per_block();
    let pre_element_wise_functions = OnceLock::new();
    let post_element_wise_functions = OnceLock::new();

    // Calculate n_workgroups for this kernel type (SUBGROUP_COUNT subgroups per workgroup, Q_N_SGEMV_CHUNK_SIZE per subgroup)
    let chunk_size = Q_N_SGEMV_CHUNK_SIZE * SUBGROUP_COUNT;
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
        "let row = ({SUBGROUP_COUNT} * workgroup_offset + {subgroup_index}) * {Q_N_SGEMV_CHUNK_SIZE};"
    )
    .unwrap();

    // Fast path check: if all rows in this tile are valid, skip per-row bounds checks
    writeln!(
        kernel,
        "let is_full_tile = row + {Q_N_SGEMV_CHUNK_SIZE} <= {n_size};"
    )
    .unwrap();

    writeln!(kernel, "let row_block_offset = row * k_block_size;").unwrap();

    writeln!(kernel, "let thread_id = {subgroup_local_index} / 2;").unwrap();
    writeln!(kernel, "let thread_local_id = {subgroup_local_index} % 2;").unwrap();

    writeln!(kernel, "let lane_index = thread_local_id * 8;").unwrap();

    writeln!(
        kernel,
        "var y_offset = thread_id * {elements_per_block} + lane_index;"
    )
    .unwrap();

    // Always accumulate in f32 to avoid overflow, then convert to output dtype at the end
    let sum_storage_type = maybe_vec_storage_type(Q_N_SGEMV_CHUNK_SIZE, DataTypeEnum::F32);
    writeln!(kernel, "var sum = {sum_storage_type}();",).unwrap();

    writeln!(kernel, "var cached_a_values = array<f32, 16>();",).unwrap();

    // Loop over all of the blocks this thread is responsible for
    writeln!(
        kernel,
        "for (var i = thread_id; i < k_block_size; i += {SUBGROUP_SIZE}/2u) {{"
    )
    .unwrap();
    {
        // First load the values of a into cached_a_values
        writeln!(kernel, "var vector_sum = f32(0.0);").unwrap();

        for j in (0..8).step_by(2) {
            writeln!(kernel, "{{").unwrap();

            for (var_name, offset) in [
                ("a_val_0", j),
                ("a_val_1", j + 1),
                ("a_val_16", j + 16),
                ("a_val_17", j + 17),
            ] {
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
                indices.push(format!("y_offset + {offset}"));
                input_a.strided_index(&mut raw_input, indices);
                write!(&mut raw_input, "]").unwrap();
                let processed = pre_element_wise_functions
                    .iter()
                    .fold(raw_input, |acc, f| f.call(vec![acc]));
                writeln!(kernel, "let {var_name} = f32({processed});").unwrap();
            }

            writeln!(
                kernel,
                "vector_sum += a_val_0 + a_val_1 + a_val_16 + a_val_17;"
            )
            .unwrap();
            writeln!(kernel, "cached_a_values[{j}] = a_val_0;").unwrap();
            writeln!(
                kernel,
                "cached_a_values[{}] = a_val_1 * f32({});",
                j + 1,
                shift_right_scale(8)
            )
            .unwrap();
            writeln!(
                kernel,
                "cached_a_values[{}] = a_val_16 * f32({});",
                j + 8,
                shift_right_scale(4)
            )
            .unwrap();
            writeln!(
                kernel,
                "cached_a_values[{}] = a_val_17 * f32({});",
                j + 9,
                shift_right_scale(12)
            )
            .unwrap();

            writeln!(kernel, "}}").unwrap();
        }

        writeln!(kernel, "let vector_total = vector_sum;").unwrap();

        writeln!(kernel, "var block_offset = row_block_offset + i;").unwrap();

        // Fast path: all rows in tile are valid, no bounds checks needed
        writeln!(kernel, "if is_full_tile {{").unwrap();
        if Q_N_SGEMV_CHUNK_SIZE > 1 {
            writeln!(
                kernel,
                "for (var offset = 0u; offset < {Q_N_SGEMV_CHUNK_SIZE}; offset += 1u) {{"
            )
            .unwrap();
        }
        {
            writeln!(kernel, "let row_index = row + offset;").unwrap();
            block_dot(kernel, op, input_b);
            let indexed = maybe_vec_storage_index(Q_N_SGEMV_CHUNK_SIZE, "sum", "offset");
            writeln!(kernel, "{indexed} += product;").unwrap();
        }
        if Q_N_SGEMV_CHUNK_SIZE > 1 {
            writeln!(kernel, "block_offset += k_block_size;").unwrap();
            writeln!(kernel, "}}").unwrap();
        }
        writeln!(kernel, "}} else {{").unwrap();
        // Slow path: check bounds for each row
        if Q_N_SGEMV_CHUNK_SIZE > 1 {
            writeln!(
                kernel,
                "for (var offset = 0u; offset < {Q_N_SGEMV_CHUNK_SIZE}; offset += 1u) {{"
            )
            .unwrap();
        }
        {
            writeln!(kernel, "let row_index = row + offset;").unwrap();
            writeln!(kernel, "if row_index < {n_size} {{").unwrap();
            block_dot(kernel, op, input_b);
            let indexed = maybe_vec_storage_index(Q_N_SGEMV_CHUNK_SIZE, "sum", "offset");
            writeln!(kernel, "{indexed} += product;").unwrap();
            writeln!(kernel, "}}").unwrap();
        }
        if Q_N_SGEMV_CHUNK_SIZE > 1 {
            writeln!(kernel, "block_offset += k_block_size;").unwrap();
            writeln!(kernel, "}}").unwrap();
        }
        writeln!(kernel, "}}").unwrap();

        writeln!(kernel, "y_offset += {elements_per_block} * 16;").unwrap();
    }
    writeln!(kernel, "}}").unwrap();

    // Get the sum among all threads in the subgroup
    writeln!(
        kernel,
        "sum = {};",
        maybe_vec_storage_subgroup_add(Q_N_SGEMV_CHUNK_SIZE, "sum")
    )
    .unwrap();

    // Initialize post element-wise functions once before the loop
    let post_fns =
        post_element_wise_functions.get_or_init(|| op.post_element_wise.add_functions(kernel));

    // Generate the output body for one row
    let generate_row_output = |kernel: &mut GenericKernel| {
        write!(kernel, "{output}[").unwrap();
        let mut output_indices = Vec::new();
        // Add batch indices first
        for dim in (0..output.rank()).rev().skip(2) {
            output_indices.push(format!("batch_idx_{dim}"));
        }
        // Then add M and N indices
        output_indices.push("m_idx".to_string());
        output_indices.push("row_index".to_string());
        output.strided_index(kernel, output_indices);
        let indexed = maybe_vec_storage_index(Q_N_SGEMV_CHUNK_SIZE, "sum", "offset");
        let result = post_fns
            .iter()
            .fold(format!("{dtype}({indexed})"), |acc, f| f.call(vec![acc]));
        writeln!(kernel, "] = {result};").unwrap();
    };

    // Fast path: all rows in tile are valid, no bounds checks needed
    writeln!(kernel, "if is_full_tile {{").unwrap();
    if Q_N_SGEMV_CHUNK_SIZE > 1 {
        writeln!(
            kernel,
            "for (var offset = 0u; offset < {Q_N_SGEMV_CHUNK_SIZE}; offset += 1u) {{"
        )
        .unwrap();
    }
    {
        writeln!(kernel, "if {subgroup_local_index} == 0u {{").unwrap();
        let index = if Q_N_SGEMV_CHUNK_SIZE > 1 {
            "row + offset".to_string()
        } else {
            "row".to_string()
        };
        writeln!(kernel, "let row_index = {index};").unwrap();
        generate_row_output(kernel);
        writeln!(kernel, "}}").unwrap();
    }
    if Q_N_SGEMV_CHUNK_SIZE > 1 {
        writeln!(kernel, "}}").unwrap();
    }
    writeln!(kernel, "}} else {{").unwrap();
    // Slow path: check bounds for each row
    if Q_N_SGEMV_CHUNK_SIZE > 1 {
        writeln!(
            kernel,
            "for (var offset = 0u; offset < {Q_N_SGEMV_CHUNK_SIZE}; offset += 1u) {{"
        )
        .unwrap();
    }
    {
        writeln!(kernel, "if {subgroup_local_index} == 0u {{").unwrap();
        let index = if Q_N_SGEMV_CHUNK_SIZE > 1 {
            "row + offset".to_string()
        } else {
            "row".to_string()
        };
        writeln!(kernel, "let row_index = {index};").unwrap();
        writeln!(kernel, "if row_index < {n_size} {{").unwrap();
        generate_row_output(kernel);
        writeln!(kernel, "}}").unwrap();
        writeln!(kernel, "}}").unwrap();
    }
    if Q_N_SGEMV_CHUNK_SIZE > 1 {
        writeln!(kernel, "}}").unwrap();
    }
    writeln!(kernel, "}}").unwrap();
}

fn block_dot(kernel: &mut GenericKernel, op: &QMatMulOperation, input_b: &QMatrixInput) {
    match op.matrix.datatype {
        fusor_gguf::GgmlType::F32 => todo!(),
        fusor_gguf::GgmlType::F16 => todo!(),
        fusor_gguf::GgmlType::Q4_0 => block_dot_q4_0(kernel, input_b),
        fusor_gguf::GgmlType::Q4_1 => todo!(),
        fusor_gguf::GgmlType::Q5_0 => block_dot_q5_0(kernel, op, input_b),
        fusor_gguf::GgmlType::Q5_1 => todo!(),
        fusor_gguf::GgmlType::Q8_0 => todo!(),
        fusor_gguf::GgmlType::Q8_1 => todo!(),
        fusor_gguf::GgmlType::Q2K => todo!(),
        fusor_gguf::GgmlType::Q3K => todo!(),
        fusor_gguf::GgmlType::Q4K => todo!(),
        fusor_gguf::GgmlType::Q5K => todo!(),
        fusor_gguf::GgmlType::Q6K => todo!(),
        fusor_gguf::GgmlType::Q8K => todo!(),
    }
}

fn block_dot_q4_0(kernel: &mut GenericKernel, input_b: &QMatrixInput) {
    writeln!(kernel, "var chunk_sum = vec4<f32>();").unwrap();

    for j in (0..8).step_by(4) {
        let data_index = j / 4;
        writeln!(
            kernel,
            "{} data_u32 = {input_b}[block_offset].data[lane_index / 4 + {data_index}];",
            if j == 0 { "var" } else { "" }
        )
        .unwrap();
        writeln!(
            kernel,
            "chunk_sum += vec4<f32>(cached_a_values[{j}] * f32(data_u32 & 0x000F), cached_a_values[{}] * f32(data_u32 & 0x0F00), cached_a_values[{}] * f32(data_u32 & 0x00F0), cached_a_values[{}] * f32(data_u32 & 0xF000));",
            j + 1, j + 8, j + 9
        )
        .unwrap();
        writeln!(kernel, "data_u32 >>= 16u;").unwrap();
        writeln!(
            kernel,
            "chunk_sum += vec4<f32>(cached_a_values[{}] * f32(data_u32 & 0x000F), cached_a_values[{}] * f32(data_u32 & 0x0F00), cached_a_values[{}] * f32(data_u32 & 0x00F0), cached_a_values[{}] * f32(data_u32 & 0xF000));",
            j + 2, j + 3, j + 10, j + 11
        )
        .unwrap();
    }

    writeln!(kernel, "let product = f32({input_b}[block_offset].scale) * (chunk_sum.x + chunk_sum.y + chunk_sum.z + chunk_sum.w + vector_total * f32(-8.0));").unwrap();
}

fn block_dot_q5_0(kernel: &mut GenericKernel, op: &QMatMulOperation, input_b: &QMatrixInput) {
    let elements_per_block = op.elements_per_block();
    writeln!(kernel, "var chunk_sum = vec4<f32>();").unwrap();

    writeln!(
        kernel,
        "let high_u32 = {input_b}[block_offset].data_high_bits[0];"
    )
    .unwrap();

    for j in (0..8).step_by(4) {
        let data_index = j / 4;
        writeln!(
            kernel,
            "{} low_u32 = {input_b}[block_offset].data_low_bits[lane_index / 4 + {data_index}];",
            if j == 0 { "var" } else { "" }
        )
        .unwrap();

        writeln!(
            kernel,
            "chunk_sum += vec4<f32>(cached_a_values[{j}] * f32((low_u32 & 0x000F) | ((high_u32 >> ({j} + lane_index) << 4) & 0x00010)), cached_a_values[{}] * f32((low_u32 & 0x0F00) | ((high_u32 >> ({} + lane_index) << 12) & 0x01000)), cached_a_values[{}] * f32((low_u32 & 0x00F0) | ((high_u32 >> ({j} + lane_index + {elements_per_block}/2) << 8) & 0x00100)), cached_a_values[{}] * f32((low_u32 & 0xF000) | ((high_u32 >> ({} + lane_index + {elements_per_block}/2) << 16) & 0x10000)));",
            j + 1, j + 1, j + 8, j + 9, j + 1
        )
        .unwrap();

        writeln!(kernel, "low_u32 >>= 16u;").unwrap();

        writeln!(
            kernel,
            "chunk_sum += vec4<f32>(cached_a_values[{}] * f32((low_u32 & 0x000F) | ((high_u32 >> ({} + lane_index) << 4) & 0x00010)), cached_a_values[{}] * f32((low_u32 & 0x0F00) | ((high_u32 >> ({} + lane_index) << 12) & 0x01000)), cached_a_values[{}] * f32((low_u32 & 0x00F0) | ((high_u32 >> ({} + lane_index + {elements_per_block}/2) << 8) & 0x00100)), cached_a_values[{}] * f32((low_u32 & 0xF000) | ((high_u32 >> ({} + lane_index + {elements_per_block}/2) << 16) & 0x10000)));",
            j + 2, j + 2, j + 3, j + 3, j + 10, j + 2, j + 11, j + 3
        )
        .unwrap();
    }

    writeln!(kernel, "let product = f32({input_b}[block_offset].scale) * (chunk_sum.x + chunk_sum.y + chunk_sum.z + chunk_sum.w + vector_total * f32(-16.0));").unwrap();
}
