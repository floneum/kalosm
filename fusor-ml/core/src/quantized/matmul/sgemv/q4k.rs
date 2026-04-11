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

pub(crate) const Q4K_SGEMV_CHUNK_SIZE: u32 = 4; // This is the size of the chunk each thread will process at a time
const SUBGROUP_COUNT: u32 = 2;

const MASK1: u32 = 0b0011111100111111;
const MASK2: u32 = 0b0000111100001111;
const MASK3: u32 = 0b1100000011000000;

// https://github.com/ggml-org/llama.cpp/blob/6efcd65945a98cf6883cdd9de4c8ccd8c79d219a/ggml/src/ggml-metal/ggml-metal.metal#L5311
#[allow(clippy::too_many_arguments)]
pub(crate) fn q4k_sgemv(
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

    // Calculate n_workgroups for this kernel type (SUBGROUP_COUNT subgroups per workgroup, Q4K_SGEMV_CHUNK_SIZE per subgroup)
    let chunk_size = Q4K_SGEMV_CHUNK_SIZE * SUBGROUP_COUNT;
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
        "let k_block_size = {k_size} / {elements_per_block};"
    )
    .unwrap();

    // Workgroup offset in the N dimension (from decomposed linearized index)
    writeln!(kernel, "let workgroup_offset = n_workgroup_idx;").unwrap();
    writeln!(
        kernel,
        "let row = (workgroup_offset * {SUBGROUP_COUNT} + {subgroup_index}) * {Q4K_SGEMV_CHUNK_SIZE};"
    )
    .unwrap();

    writeln!(kernel, "let thread_id = {subgroup_local_index} >> 3;").unwrap();
    writeln!(kernel, "let thread_local_id = {subgroup_local_index} & 7;").unwrap();
    writeln!(kernel, "let half_subgroup_id = thread_local_id >> 2;").unwrap();
    writeln!(kernel, "let half_subgroup_local_id = thread_local_id & 3;").unwrap();

    writeln!(kernel, "let block_offset = row * k_block_size;").unwrap();
    writeln!(kernel, "var vector_offset = thread_id * {elements_per_block} + half_subgroup_id * 64 + half_subgroup_local_id * 8;").unwrap();

    // Always accumulate in f32 to avoid overflow, then convert to output dtype at the end
    let sum_storage_type = maybe_vec_storage_type(Q4K_SGEMV_CHUNK_SIZE, DataTypeEnum::F32);
    writeln!(kernel, "var sum = {sum_storage_type}();",).unwrap();

    writeln!(kernel, "var cached_a_low_values = array<f32, 16>();",).unwrap();
    writeln!(kernel, "var cached_a_high_values = array<f32, 16>();",).unwrap();

    // Loop over all of the blocks this thread is responsible for
    writeln!(
        kernel,
        "for (var i = thread_id; i < k_block_size; i += 4) {{"
    )
    .unwrap();
    {
        // Keep track of the sum of each scale chunk of the vector for the offset calculation later
        writeln!(kernel, "var vector_sum = vec4<f32>();").unwrap();

        // First load the values of a into the cache
        for j in 0..8 {
            // Load all 4 values using strided indexing
            for (idx, offset) in [(0, 0), (1, 32), (2, 128), (3, 160)] {
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
                indices.push(format!("vector_offset + {j} + {offset}"));
                input_a.strided_index(&mut raw_input, indices);
                write!(&mut raw_input, "]").unwrap();
                let processed = pre_element_wise_functions
                    .iter()
                    .fold(raw_input, |acc, f| f.call(vec![acc]));
                writeln!(kernel, "let a_val_{j}_{idx} = f32({processed});").unwrap();
            }

            writeln!(
                kernel,
                "let vec_{j} = vec4<f32>(f32(a_val_{j}_0), f32(a_val_{j}_1), f32(a_val_{j}_2), f32(a_val_{j}_3));"
            )
            .unwrap();
            writeln!(kernel, "vector_sum += vec_{j};").unwrap();
            writeln!(kernel, "cached_a_low_values[{j} + 0] = vec_{j}.x;").unwrap();
            writeln!(kernel, "cached_a_low_values[{j} + 8] = vec_{j}.y;").unwrap();
            writeln!(kernel, "cached_a_high_values[{j} + 0] = vec_{j}.z;").unwrap();
            writeln!(kernel, "cached_a_high_values[{j} + 8] = vec_{j}.w;").unwrap();
        }

        // Find the block value offsets
        writeln!(kernel, "let scale_offset = half_subgroup_id;").unwrap();
        writeln!(
            kernel,
            "let data_offset = half_subgroup_id * 8u + half_subgroup_local_id * 2u;"
        )
        .unwrap();

        writeln!(kernel, "var local_block_offset = block_offset + i;").unwrap();

        for offset in 0..Q4K_SGEMV_CHUNK_SIZE {
            writeln!(kernel, "{{").unwrap();
            // Fetch and unpack the two sets of values from the cache
            writeln!(kernel, "let first_values_offset = data_offset;").unwrap();
            writeln!(kernel, "let second_values_offset = data_offset + 16u;").unwrap();

            // Keep track of the sum of each chunk
            writeln!(kernel, "var first_sums = vec4<f32>();").unwrap();
            writeln!(kernel, "var second_sums = vec4<f32>();").unwrap();

            // Perform the dot product of the values and scales
            for j in 0..2 {
                for (sum, cache, values) in [
                    ("first_sums", "cached_a_low_values", "first_values_offset"),
                    (
                        "second_sums",
                        "cached_a_high_values",
                        "second_values_offset",
                    ),
                ] {
                    // Note: We add the values with a mask **without** shifting them
                    // this means the sums in the first_sums and second_sums
                    // will be scaled by different values. We correct this below
                    // by multiplying by the floating point values that correspond to the
                    // bit shifts.
                    writeln!(
                        kernel,
                        "let value_u32_{values}_{j} = {input_b}[local_block_offset].data[{values} + {j}];"
                    )
                    .unwrap();
                    writeln!(
                        kernel,
                        "let first_four_values_{values}_{j} = vec4<f32>({cache}[{j}*4 + 0], {cache}[{j}*4 + 1], {cache}[{j}*4 + 2], {cache}[{j}*4 + 3]);"
                    )
                    .unwrap();
                    writeln!(
                        kernel,
                        "let second_four_values_{values}_{j} = vec4<f32>({cache}[{j}*4 + 8], {cache}[{j}*4 + 9], {cache}[{j}*4 + 10], {cache}[{j}*4 + 11]);"
                    )
                    .unwrap();
                    writeln!(
                        kernel,
                        "{sum} += vec4<f32>(first_four_values_{values}_{j}.x * f32(value_u32_{values}_{j} & 0x000F), first_four_values_{values}_{j}.y * f32(value_u32_{values}_{j} & 0x0F00), second_four_values_{values}_{j}.x * f32(value_u32_{values}_{j} & 0x00F0), second_four_values_{values}_{j}.y * f32(value_u32_{values}_{j} & 0xF000));"
                    )
                    .unwrap();
                    let shift_right_16 = shift_right_scale(16);
                    writeln!(
                        kernel,
                        "{sum} += vec4<f32>(first_four_values_{values}_{j}.z * f32(value_u32_{values}_{j} & 0x000F0000), first_four_values_{values}_{j}.w * f32(value_u32_{values}_{j} & 0x0F000000), second_four_values_{values}_{j}.z * f32(value_u32_{values}_{j} & 0x00F00000), second_four_values_{values}_{j}.w * f32(value_u32_{values}_{j} & 0xF0000000)) * f32({shift_right_16});"
                    )
                    .unwrap();
                }
            }

            // Load the block scale and min
            writeln!(
                kernel,
                "let block_scale = f32({input_b}[local_block_offset].scale);"
            )
            .unwrap();
            writeln!(
                kernel,
                "let block_min = f32({input_b}[local_block_offset].min);"
            )
            .unwrap();
            // Load 8 scales into a cache
            writeln!(
                kernel,
                "let first_32_scale_bits = {input_b}[local_block_offset].scales[0] >> (16 * scale_offset);"
            )
            .unwrap();
            writeln!(
                kernel,
                "let second_32_scale_bits = {input_b}[local_block_offset].scales[1] >> (16 * scale_offset);"
            )
            .unwrap();
            writeln!(
                kernel,
                "let third_32_scale_bits = {input_b}[local_block_offset].scales[2] >> (16 * scale_offset);"
            )
            .unwrap();
            // Extract the scales from the bits into cached_scales
            writeln!(
                kernel,
                "let first_two_scales = first_32_scale_bits & {MASK1};"
            )
            .unwrap();
            writeln!(
                kernel,
                "let second_two_scales = second_32_scale_bits & {MASK1};"
            )
            .unwrap();

            writeln!(kernel, "let third_two_scales = ((third_32_scale_bits >> 0) & {MASK2}) | ((first_32_scale_bits & {MASK3}) >> 2);").unwrap();
            writeln!(kernel, "let fourth_two_scales = ((third_32_scale_bits >> 4) & {MASK2}) | ((second_32_scale_bits & {MASK3}) >> 2);").unwrap();

            writeln!(
                kernel,
                "let odd_scales_unpacked = vec4<f32>(unpack4xU8(first_two_scales | (third_two_scales << 16)));"
            )
            .unwrap();
            writeln!(
                kernel,
                "let even_scales_unpacked = vec4<f32>(unpack4xU8(second_two_scales | (fourth_two_scales << 16)));"
            )
            .unwrap();

            // Add the sums to the total sum
            let indexed_sum = maybe_vec_storage_index(Q4K_SGEMV_CHUNK_SIZE, "sum", offset);
            // *_sums[0] needs to be shifted by 0 bits
            // *_sums[1] needs to be shifted by 8 bits
            // *_sums[2] needs to be shifted by 4 bits
            // *_sums[3] needs to be shifted by 12 bits
            let shift_right_8 = shift_right_scale(8);
            let shift_right_4 = shift_right_scale(4);
            writeln!(
                kernel,
                "let small_shift_sums = vec4<f32>(first_sums[0], first_sums[2], second_sums[0], second_sums[2]);"
            )
            .unwrap();
            writeln!(
                kernel,
                "let large_shift_sums = vec4<f32>(first_sums[1], first_sums[3], second_sums[1], second_sums[3]);"
            )
            .unwrap();
            writeln!(
                kernel,
                "let shift_4 = vec4<f32>(1.0, {shift_right_4}, 1.0, {shift_right_4});"
            )
            .unwrap();
            writeln!(
                kernel,
                r#"{indexed_sum} += block_scale * dot((small_shift_sums + f32({shift_right_8}) * large_shift_sums) * odd_scales_unpacked, shift_4) -
                                                            block_min * dot(vector_sum, even_scales_unpacked);"#
            )
            .unwrap();
            // Move forward the block offset by one row
            writeln!(kernel, "local_block_offset += k_block_size;").unwrap();
            writeln!(kernel, "}}").unwrap();
        }

        // move forward the vector offset
        writeln!(kernel, "vector_offset += 4 * {elements_per_block};").unwrap();
    }
    writeln!(kernel, "}}").unwrap();

    // Get the sum among all threads in the subgroup
    writeln!(
        kernel,
        "sum = {};",
        maybe_vec_storage_subgroup_add(Q4K_SGEMV_CHUNK_SIZE, "sum")
    )
    .unwrap();

    for offset in 0..Q4K_SGEMV_CHUNK_SIZE {
        // If this is not the first simd thread in the workgroup, we can return early
        writeln!(kernel, "if {subgroup_local_index} == 0u {{").unwrap();
        {
            // Write the output to the output tensor if this is the first thread in the workgroup
            // Convert from f32 accumulator to output dtype
            write!(kernel, "{output}[").unwrap();
            let index = format!("row + {offset}");
            let mut output_indices = Vec::new();
            // Add batch indices first
            for dim in (0..output.rank()).rev().skip(2) {
                output_indices.push(format!("batch_idx_{dim}"));
            }
            // Then add M and N indices
            output_indices.push("m_idx".to_string());
            output_indices.push(index);
            output.strided_index(kernel, output_indices);
            let indexed = maybe_vec_storage_index(Q4K_SGEMV_CHUNK_SIZE, "sum", offset);
            let result = post_element_wise_functions
                .get_or_init(|| op.post_element_wise.add_functions(kernel))
                .iter()
                .fold(format!("{dtype}({indexed})"), |acc, f| f.call(vec![acc]));
            writeln!(kernel, "] = {result};").unwrap();
        }
        writeln!(kernel, "}}").unwrap();
    }
}
