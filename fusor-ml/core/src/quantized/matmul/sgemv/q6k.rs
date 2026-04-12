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

pub(crate) const Q6K_SGEMV_CHUNK_SIZE: u32 = 2; // This is the size of the chunk each thread will process at a time
const PRELOAD: bool = false;

// https://github.com/ggml-org/llama.cpp/blob/6efcd65945a98cf6883cdd9de4c8ccd8c79d219a/ggml/src/ggml-metal/ggml-metal.metal#L5564
#[allow(clippy::too_many_arguments)]
pub(crate) fn q6k_sgemv(
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

    // Calculate n_workgroups for this kernel type (2 subgroups per workgroup, Q6K_SGEMV_CHUNK_SIZE per subgroup)
    let n_workgroups = format!(
        "(({n_size} + {} - 1) / {})",
        Q6K_SGEMV_CHUNK_SIZE * 2,
        Q6K_SGEMV_CHUNK_SIZE * 2
    );

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
        "let row = (2 * workgroup_offset + {subgroup_index}) * {Q6K_SGEMV_CHUNK_SIZE};"
    )
    .unwrap();

    writeln!(kernel, "let block_offset = row * k_block_size;").unwrap();

    writeln!(kernel, "let thread_id = {subgroup_local_index} / 2;").unwrap();
    writeln!(kernel, "let thread_local_id = {subgroup_local_index} % 2;").unwrap();
    writeln!(kernel, "let half_subgroup_id = thread_id / 8;").unwrap();
    writeln!(kernel, "let half_subgroup_local_id = thread_id % 8;").unwrap();
    const CHUNKS_PER_STRIPE: usize = 4;
    writeln!(
        kernel,
        "let stripe_offset = half_subgroup_local_id * {CHUNKS_PER_STRIPE};"
    )
    .unwrap();
    writeln!(kernel, "let scale_index_offset = half_subgroup_id * 2;").unwrap();

    writeln!(kernel, "let scale_pair_offset = stripe_offset / 16;").unwrap();

    // Every half subgroup handles 128 normal values
    writeln!(
        kernel,
        "let y_offset = 128 * half_subgroup_id + stripe_offset;"
    )
    .unwrap();
    // Every half subgroup handles 64 lower values (half bytes)
    writeln!(
        kernel,
        "let q_offset_l = 16 * half_subgroup_id + stripe_offset / 4;"
    )
    .unwrap();
    // Every half subgroup handles 32 upper values (quarter bytes)
    writeln!(
        kernel,
        "let q_offset_h = 8 * half_subgroup_id + stripe_offset / 4;"
    )
    .unwrap();

    let sum_storage_type = maybe_vec_storage_type(Q6K_SGEMV_CHUNK_SIZE, DataTypeEnum::F32);
    writeln!(kernel, "var sum = {sum_storage_type}();",).unwrap();

    if PRELOAD {
        writeln!(kernel, "var cached_a_values = array<f32, 16>();",).unwrap();
    }

    // Loop over all of the blocks this thread is responsible for
    writeln!(
        kernel,
        "for (var i = thread_local_id; i < k_block_size; i += 2) {{"
    )
    .unwrap();
    {
        // First load the values of a into cached_a_values
        writeln!(
            kernel,
            "let vector_offset = i * {elements_per_block} + y_offset;"
        )
        .unwrap();
        let load_value = |kernel: &mut GenericKernel, j: &str, offset: u32| {
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
            indices.push(format!("{j} + vector_offset + {}", offset * 32));
            input_a.strided_index(&mut raw_input, indices);
            write!(&mut raw_input, "]").unwrap();
            let processed = pre_element_wise_functions
                .iter()
                .fold(raw_input, |acc, f| f.call(vec![acc]));
            write!(kernel, "f32({processed})").unwrap();
        };
        if PRELOAD {
            writeln!(kernel, "for (var j = 0u; j < 4; j += 1u) {{").unwrap();
            for offset in 0..4 {
                write!(kernel, "cached_a_values[j*4u + {offset}] = ",).unwrap();
                load_value(kernel, "j", offset);
                writeln!(kernel, ";").unwrap();
            }
            writeln!(kernel, "}}").unwrap();
        }

        if Q6K_SGEMV_CHUNK_SIZE > 1 {
            writeln!(
                kernel,
                "for (var offset = 0u; offset < {Q6K_SGEMV_CHUNK_SIZE}; offset += 1u) {{"
            )
            .unwrap();
        }
        {
            let row_index = if Q6K_SGEMV_CHUNK_SIZE > 1 {
                "row + offset"
            } else {
                "row"
            };
            writeln!(kernel, "let row_index = {row_index};").unwrap();
            writeln!(kernel, "if row_index < {n_size} {{").unwrap();
            if Q6K_SGEMV_CHUNK_SIZE > 1 {
                writeln!(
                    kernel,
                    "let local_block_offset = i + block_offset + offset * k_block_size;"
                )
                .unwrap();
            } else {
                writeln!(kernel, "let local_block_offset = i + block_offset;").unwrap();
            }
            writeln!(kernel, "let low_offset_1 = q_offset_l;").unwrap();
            writeln!(
                kernel,
                "let low_bytes_1 = unpack4xU8({input_b}[local_block_offset].data_low_bits[low_offset_1]);"
            )
            .unwrap();
            writeln!(kernel, "let low_offset_2 = q_offset_l + 8;").unwrap();
            writeln!(kernel, "let low_bytes_2 = unpack4xU8({input_b}[local_block_offset].data_low_bits[low_offset_2]);").unwrap();
            writeln!(kernel, "let high_offset = q_offset_h;").unwrap();
            writeln!(
                kernel,
                "let high_bytes = unpack4xU8({input_b}[local_block_offset].data_high_bits[high_offset]);"
            )
            .unwrap();
            writeln!(kernel, "let scale_offset = scale_index_offset;").unwrap();
            writeln!(
                kernel,
                "let scale_chunk_1 = unpack4xI8({input_b}[local_block_offset].scales[scale_offset]);"
            )
            .unwrap();
            writeln!(
                kernel,
                "let scale_chunk_2 = unpack4xI8({input_b}[local_block_offset].scales[scale_offset + 1]);"
            )
            .unwrap();
            writeln!(
                kernel,
                "let scales = vec4<f32>(f32(scale_chunk_1[scale_pair_offset]), f32(scale_chunk_1[2 + scale_pair_offset]), f32(scale_chunk_2[scale_pair_offset]), f32(scale_chunk_2[2 + scale_pair_offset]));"
            )
            .unwrap();

            writeln!(
                kernel,
                "let scale = f32({input_b}[local_block_offset].scale);"
            )
            .unwrap();

            writeln!(kernel, "var sums = vec4<f32>();").unwrap();
            writeln!(kernel, "for (var j = 0u; j < 4u; j += 1u) {{").unwrap();
            {
                let first_four_bytes = 0b00001111u8;
                let first_two_bytes = 0b00000011u8;
                let second_two_bytes = 0b00001100u8;
                let third_two_bytes = 0b00110000u8;
                let fourth_two_bytes = 0b11000000u8;
                let get_value = |kernel: &mut GenericKernel, j: &str, offset: u32| {
                    if PRELOAD {
                        write!(kernel, "cached_a_values[{j} + {offset}]").unwrap();
                    } else {
                        load_value(kernel, j, offset);
                    }
                };
                write!(kernel, "sums[0] += ").unwrap();
                get_value(kernel, "j", 0);
                writeln!(kernel,"* f32(i32((low_bytes_1[j] & {first_four_bytes}) | ((high_bytes[j] & {first_two_bytes})  << 4)) - 32);").unwrap();
                write!(kernel, "sums[1] += ").unwrap();
                get_value(kernel, "j", 1);
                writeln!(kernel,"* f32(i32((low_bytes_2[j] & {first_four_bytes}) | ((high_bytes[j] & {second_two_bytes}) << 2)) - 32);").unwrap();
                write!(kernel, "sums[2] += ").unwrap();
                get_value(kernel, "j", 2);
                writeln!(kernel,"* f32(i32((low_bytes_1[j]                 >> 4) | ((high_bytes[j] & {third_two_bytes})  << 0)) - 32);").unwrap();
                write!(kernel, "sums[3] += ").unwrap();
                get_value(kernel, "j", 3);
                writeln!(kernel,"* f32(i32((low_bytes_2[j]                 >> 4) | ((high_bytes[j] & {fourth_two_bytes}) >> 2)) - 32);").unwrap();
            }
            writeln!(kernel, "}}").unwrap();
            let indexed = maybe_vec_storage_index(Q6K_SGEMV_CHUNK_SIZE, "sum", "offset");
            writeln!(kernel, "{indexed} += scale * dot(sums, scales);").unwrap();
            writeln!(kernel, "}}").unwrap();
        }
        if Q6K_SGEMV_CHUNK_SIZE > 1 {
            writeln!(kernel, "}}").unwrap();
        }
    }
    writeln!(kernel, "}}").unwrap();

    // Get the sum among all threads in the subgroup
    writeln!(
        kernel,
        "sum = {};",
        maybe_vec_storage_subgroup_add(Q6K_SGEMV_CHUNK_SIZE, "sum")
    )
    .unwrap();

    // If this is not the first simd thread in the workgroup, we can return early
    writeln!(kernel, "if {subgroup_local_index} != 0u {{ return; }}").unwrap();

    if Q6K_SGEMV_CHUNK_SIZE > 1 {
        writeln!(
            kernel,
            "for (var offset = 0u; offset < {Q6K_SGEMV_CHUNK_SIZE}; offset += 1u) {{"
        )
        .unwrap();
    }
    {
        // Write the output to the output tensor if this is the first thread in the workgroup
        let index = if Q6K_SGEMV_CHUNK_SIZE > 1 {
            "row + offset".to_string()
        } else {
            "row".to_string()
        };
        writeln!(kernel, "let row_index = {index};").unwrap();
        writeln!(kernel, "if row_index < {n_size} {{").unwrap();
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
        let indexed = maybe_vec_storage_index(Q6K_SGEMV_CHUNK_SIZE, "sum", "offset");
        let result = post_element_wise_functions
            .get_or_init(|| op.post_element_wise.add_functions(kernel))
            .iter()
            .fold(format!("{dtype}({indexed})"), |acc, f| f.call(vec![acc]));
        writeln!(kernel, "] = {result};").unwrap();
        writeln!(kernel, "}}").unwrap();
    }
    if Q6K_SGEMV_CHUNK_SIZE > 1 {
        writeln!(kernel, "}}").unwrap();
    }
}
