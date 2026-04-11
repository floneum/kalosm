use crate::{
    DataTypeEnum, dequantize_mat4x4_block, dequantize_mat4x4_block_count,
    mir::{
        globals::{KernelGlobalSpace, MatrixType},
        inputs::{QMatrixInput, TensorInput},
        kernel::GenericKernel,
    },
    quantized::matmul::QMatMulOperation,
};
use std::fmt::Write;
use std::sync::OnceLock;

/// Size of the chunk we will dot at a time
const MATRIX_ELEMENTS: u32 = 16;

/// Configuration for chunked SGEMM algorithm
#[derive(Debug, Clone, Copy)]
pub struct ChunkedSgemmConfig {
    /// How many matrix_size-wide blocks we load from input_a and input_b into the cache every step
    pub input_k_chunks: u32,
    /// How deep is the cache for input_a (M dimension, must be divisible by 4)
    pub input_m_elements: u32,
    /// How deep is the cache for input_b (N dimension, must be divisible by 4)
    pub input_n_elements: u32,
    /// How many results does each thread compute (in 4x4 blocks)
    pub n_results_per_thread: u32,
    /// How many results does each thread compute (in 4x4 blocks)
    pub m_results_per_thread: u32,
    /// The datatype the kernel will use for computation (F16 or F32)
    pub cache_datatype: DataTypeEnum,
    /// Use shuffled loads to read from shared memory
    pub shuffled_loads: bool,
}

impl ChunkedSgemmConfig {
    /// Default configuration
    pub const fn default() -> Self {
        Self {
            input_k_chunks: 4,
            input_m_elements: 64,
            input_n_elements: 64,
            n_results_per_thread: 1,
            m_results_per_thread: 1,
            cache_datatype: DataTypeEnum::F32,
            shuffled_loads: false,
        }
    }

    /// Compute the total K elements loaded per step
    pub const fn input_k_elements(&self) -> u32 {
        self.input_k_chunks * MATRIX_ELEMENTS
    }

    /// Validate configuration parameters
    pub fn validate(&self, _: u32, _: usize) -> Result<(), String> {
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub fn chunked_sgemm_with_config(
    op: &QMatMulOperation,
    kernel: &mut GenericKernel,
    input_a: &TensorInput,
    input_b: &QMatrixInput,
    output: &TensorInput,
    k_size: &str,
    config: ChunkedSgemmConfig,
) {
    let workgroup_id = kernel.workgroup_index();
    let workgroup_local_index = kernel.workgroup_local_index();
    let elements_per_block = op.elements_per_block();
    let dtype = op.input_datatype;

    let sub_chunks = dequantize_mat4x4_block_count(op.matrix.datatype);

    // Validate configuration
    config.validate(elements_per_block, sub_chunks).unwrap();

    let sgemm_input_k_chunks = config.input_k_chunks;
    let sgemm_input_k_elements = config.input_k_elements();
    let sgemm_input_m_elements = config.input_m_elements;
    let sgemm_input_n_elements = config.input_n_elements;
    let sgemm_n_results_per_thread = config.n_results_per_thread;
    let sgemm_m_results_per_thread = config.m_results_per_thread;
    let cache_datatype = config.cache_datatype;
    let shuffled_loads = config.shuffled_loads;
    let pre_element_wise_functions = OnceLock::new();

    let cache_a_size = (sgemm_input_k_elements / 4) * (sgemm_input_m_elements / 4);
    let cache_a = kernel.add_global_array(
        KernelGlobalSpace::Workgroup,
        MatrixType::new(["4".into(), "4".into()], cache_datatype),
        cache_a_size.to_string(),
    );
    let cache_b_size = (sgemm_input_n_elements / 4) * (sgemm_input_k_elements / 4);
    let cache_b = kernel.add_global_array(
        KernelGlobalSpace::Workgroup,
        MatrixType::new(["4".into(), "4".into()], cache_datatype),
        cache_b_size.to_string(),
    );

    // Find the block index this workgroup is responsible for
    writeln!(kernel, "let x = {workgroup_id}.x;").unwrap();
    writeln!(kernel, "let y = {workgroup_id}.y;").unwrap();

    // Find the block output index this thread is responsible for within the workgroup
    let subgroup_m_size = 8;
    let subgroup_n_size = 4;
    assert_eq!(subgroup_m_size * subgroup_n_size, 32);
    let subgroup_m_results = subgroup_m_size * sgemm_m_results_per_thread * 4;
    let subgroup_n_results = subgroup_n_size * sgemm_n_results_per_thread * 4;
    let subgroup_blocks_per_m_workgroup = sgemm_input_m_elements / subgroup_m_results;
    // Where is this subgroup in the workgroup grid
    let subgroup_index = kernel.subgroup_index();
    writeln!(
        kernel,
        "let subgroup_pos = vec2({subgroup_index} % {subgroup_blocks_per_m_workgroup}, {subgroup_index} / {subgroup_blocks_per_m_workgroup});"
    )
    .unwrap();
    // Where is this thread in the subgroup
    let subgroup_local_index = kernel.subgroup_local_index();
    writeln!(kernel, "let subgroup_local_pos = vec2({subgroup_local_index} % {subgroup_m_size}, {subgroup_local_index} / {subgroup_m_size});").unwrap();

    // Find the output index this thread is responsible for
    let output_offset = "output_offset";
    writeln!(
        kernel,
        "let {output_offset} = vec2<u32>(x * {}u + subgroup_pos.x * {subgroup_m_results}u + subgroup_local_pos.x * 4, y * {}u + subgroup_pos.y * {subgroup_n_results}u + subgroup_local_pos.y * 4);",
        sgemm_input_m_elements,
        sgemm_input_n_elements,
    )
    .unwrap();

    // Handle batch dimensions
    writeln!(kernel, "var block_batch = {workgroup_id}.z;").unwrap();

    // Decompose the batch index for higher-dimensional tensors
    for dim in (0..input_a.rank()).rev().skip(2) {
        let shape = input_a.shape_binding(dim);
        writeln!(kernel, "let block_batch_{dim} = block_batch % {shape};").unwrap();
        writeln!(kernel, "block_batch = block_batch / {shape};").unwrap();
    }

    // Each thread is responsible for a grid of 4x4 sub-blocks
    writeln!(kernel, "var acc: array<array<mat4x4<{dtype}>, {sgemm_n_results_per_thread}>, {sgemm_m_results_per_thread}>;").unwrap();
    for tile_m in 0..sgemm_m_results_per_thread {
        for tile_n in 0..sgemm_n_results_per_thread {
            writeln!(kernel, "acc[{tile_m}][{tile_n}] = mat4x4<{dtype}>();").unwrap();
        }
    }

    // We subdivide the x dimension into pairs of threads in the same subgroup that will
    // collaboratively load one quantized block into shared memory
    writeln!(
        kernel,
        "let pair_index_row = {workgroup_local_index} % {sgemm_input_k_chunks};"
    )
    .unwrap();
    writeln!(
        kernel,
        "let pair_index_col = {workgroup_local_index} / {sgemm_input_k_chunks};"
    )
    .unwrap();

    // How many blocks do we have to process in the k dim
    writeln!(kernel, "let k_chunk_size = {k_size} / {MATRIX_ELEMENTS};").unwrap();
    writeln!(
        kernel,
        "let k_block_size = {k_size} / {elements_per_block};"
    )
    .unwrap();

    // This threads b_input offset in blocks
    writeln!(
        kernel,
        "let b_block_offset = (y * {sgemm_input_n_elements} + pair_index_col) * k_block_size;"
    )
    .unwrap();

    let chunks_per_block = elements_per_block / MATRIX_ELEMENTS;

    // Calculate one block sized group
    writeln!(
        kernel,
        "for (var k_start = 0u; k_start < k_chunk_size; k_start += {sgemm_input_k_chunks}u) {{"
    )
    .unwrap();
    {
        // This is seperated from k_start above to avoid divergence checks within each subgroup
        writeln!(kernel, "let k = k_start + pair_index_row;").unwrap();
        {
            // Load the b block into the cache
            writeln!(kernel, "let b_block_index = k / {chunks_per_block};").unwrap();
            writeln!(kernel, "let b_index_within_block = k % {chunks_per_block};").unwrap();
            let y_stride = sgemm_input_k_elements / 4;
            dequantize_mat4x4_block(
                kernel,
                op.matrix.datatype,
                "b_index_within_block",
                format!("{input_b}[b_block_index + b_block_offset]"),
                dtype,
                |data, kernel| {
                    writeln!(kernel, "let b_values = {data};").unwrap();
                    writeln!(kernel, "for (var index = 0u; index < 4u; index += 1u) {{").unwrap();
                    writeln!(
                        kernel,
                        "{cache_b}[(pair_index_col / 4) * {y_stride} + {sgemm_input_k_chunks} * pair_index_row + index][pair_index_col % 4] = vec4<{cache_datatype}>(b_values[index]);"
                    )
                    .unwrap();
                    writeln!(kernel, "}}").unwrap();
                },
            );

            // Load the a block into the cache
            writeln!(
                kernel,
                "for (var index = 0u; index < {MATRIX_ELEMENTS}u; index += 1u) {{"
            )
            .unwrap();
            {
                let y_stride = sgemm_input_m_elements / 4;
                let mut indices = vec![];
                // Add batch indices first
                for dim in (0..input_a.rank()).rev().skip(2) {
                    indices.push(format!("block_batch_{dim}"));
                }
                // Then add M and K indices
                indices.push(format!("x * {sgemm_input_m_elements} + pair_index_col"));
                indices.push(format!("k*{MATRIX_ELEMENTS} + index"));
                writeln!(
                    kernel,
                    "let chunk_index = (pair_index_row * 4 + index / 4) * {y_stride} + pair_index_col / 4;"
                )
                .unwrap();
                writeln!(kernel, "let col_index = pair_index_col % 4;").unwrap();
                writeln!(kernel, "let row_index = index % 4;").unwrap();
                input_a
                    .check_bounds(
                        kernel,
                        indices.iter().cloned(),
                        |kernel: &mut GenericKernel| {
                            let pre_element_wise_functions = pre_element_wise_functions
                                .get_or_init(|| op.pre_element_wise.add_functions(kernel));
                            let mut raw_input = String::new();
                            write!(&mut raw_input, "{input_a}[")?;
                            input_a.strided_index(&mut raw_input, indices.iter().cloned());
                            write!(&mut raw_input, "]")?;
                            let processed = pre_element_wise_functions
                                .iter()
                                .fold(raw_input, |acc, f| f.call(vec![acc]));
                            write!(
                                kernel,
                                "{cache_a}[chunk_index][col_index][row_index] = {cache_datatype}({processed});"
                            )?;
                            std::fmt::Result::Ok(())
                        },
                    )
                    .unwrap();
                writeln!(kernel, "else {{").unwrap();
                {
                    writeln!(
                        kernel,
                        "{cache_a}[chunk_index][col_index][row_index] = {cache_datatype}(0.0);"
                    )
                    .unwrap();
                }
                writeln!(kernel, "}}").unwrap();
            }
            writeln!(kernel, "}}").unwrap();
        }

        // Make sure the caches are ready
        writeln!(kernel, "workgroupBarrier();").unwrap();

        // Subgroup tiling
        // There are more n blocks than m blocks, so we iterate over n blocks in the outer loop
        // and add them to the register cache two blocks at time
        writeln!(
            kernel,
            "for (var subgroup_n_index = 0u; subgroup_n_index < {}u; subgroup_n_index += 1u) {{",
            sgemm_input_k_elements / (subgroup_m_size * 4)
        )
        .unwrap();
        {
            if shuffled_loads {
                // Load this block's value from the shared memory cache for b
                writeln!(kernel,
                "let cached_b_block = {cache_b}[(subgroup_pos.y * {subgroup_n_size} + subgroup_local_pos.y) * {} + subgroup_n_index * {subgroup_m_size} + subgroup_local_pos.x];",
                sgemm_input_k_elements / 4
            )
            .unwrap();
            }
            // Then we iterate over m blocks within the subgroup
            writeln!(
                kernel,
                "for (var subgroup_m_offset = 0u; subgroup_m_offset < {}u; subgroup_m_offset += 1u) {{",
                subgroup_m_size / subgroup_n_size,
            )
            .unwrap();
            {
                writeln!(
                    kernel,
                    "let subgroup_m_index = subgroup_n_index * {}u + subgroup_m_offset;",
                    subgroup_m_size / subgroup_n_size,
                )
                .unwrap();
                if shuffled_loads {
                    // Load the 4x4 b block this thread caches
                    writeln!(
                    kernel,
                    "let cached_a_block = {cache_a}[(subgroup_m_index * {subgroup_n_size} + subgroup_local_pos.y) * {} + subgroup_pos.x * {subgroup_m_size} + subgroup_local_pos.x];",
                    sgemm_input_m_elements / 4
                ).unwrap();
                }
                // Multiply and accumulate within the subgroup cached values
                writeln!(
                    kernel,
                    "for (var index = 0u; index < {}u; index += 1u) {{",
                    subgroup_n_size
                )
                .unwrap();
                {
                    if shuffled_loads {
                        // First shuffle the b value from the right thread in the subgroup
                        writeln!(kernel, "let b_value_thread_index = index + subgroup_m_offset * {subgroup_n_size} + subgroup_local_pos.y * {subgroup_m_size};",
                    ).unwrap();
                        write!(kernel, "let b_value = mat4x4(",).unwrap();
                        for y in 0..4 {
                            write!(
                                kernel,
                                "subgroupShuffle(cached_b_block[{y}], b_value_thread_index),"
                            )
                            .unwrap();
                        }
                        writeln!(kernel, ");").unwrap();
                        // Then shuffle the a value from the right thread in the subgroup
                        writeln!(
                            kernel,
                            "let a_value_thread_index = subgroup_local_pos.x + index * {subgroup_m_size};"
                        ).unwrap();
                        write!(kernel, "let a_value = mat4x4(",).unwrap();
                        for y in 0..4 {
                            write!(
                                kernel,
                                "subgroupShuffle(cached_a_block[{y}], a_value_thread_index),"
                            )
                            .unwrap();
                        }
                        writeln!(kernel, ");").unwrap();
                    } else {
                        writeln!(
                            kernel,
                            "let b_value = {cache_b}[(subgroup_pos.y * {subgroup_n_size} + subgroup_local_pos.y) * {} + subgroup_m_index * {subgroup_n_size} + index];",
                            sgemm_input_k_elements / 4
                        )
                        .unwrap();
                        writeln!(
                            kernel,
                            "let a_value = {cache_a}[subgroup_pos.x * {subgroup_m_size} + subgroup_local_pos.x + (subgroup_m_index * {subgroup_n_size} + index) * {}];",
                            sgemm_input_m_elements / 4
                        )
                        .unwrap();
                    }

                    // Compute the results
                    for tile_m in 0..sgemm_m_results_per_thread {
                        for tile_n in 0..sgemm_n_results_per_thread {
                            writeln!(
                                kernel,
                                "acc[{tile_m}][{tile_n}] = acc[{tile_m}][{tile_n}] + mat4x4<{dtype}>(transpose(a_value) * b_value);"
                            )
                            .unwrap();
                        }
                    }
                }
                writeln!(kernel, "}}").unwrap();
            }
            writeln!(kernel, "}}").unwrap();
        }
        writeln!(kernel, "}}").unwrap();

        // Make sure all threads are done reading from the cache before we overwrite it
        writeln!(kernel, "workgroupBarrier();").unwrap();
    }
    writeln!(kernel, "}}").unwrap();

    write_acc_back(kernel, op, output, output_offset, &config).unwrap();
}

fn write_acc_back(
    kernel: &mut GenericKernel,
    op: &QMatMulOperation,
    output: &TensorInput,
    output_offset: &str,
    config: &ChunkedSgemmConfig,
) -> std::fmt::Result {
    let sgemm_n_results_per_thread = config.n_results_per_thread;
    let sgemm_m_results_per_thread = config.m_results_per_thread;
    let post_element_wise_functions = op.post_element_wise.add_functions(kernel);
    for tile_m in 0..sgemm_m_results_per_thread {
        for tile_n in 0..sgemm_n_results_per_thread {
            writeln!(
                kernel,
                "for (var y_offset = 0u; y_offset < 4u; y_offset += 1u) {{"
            )?;
            writeln!(
                kernel,
                "for (var x_offset = 0u; x_offset < 4u; x_offset += 1u) {{"
            )?;
            // Then write the result

            let mut output_indices = vec![];
            // Add batch indices first
            for dim in (0..output.rank()).rev().skip(2) {
                output_indices.push(format!("block_batch_{dim}"));
            }
            // Then add M and N indices
            output_indices.push(format!("{output_offset}.x + {tile_m} * 4u + x_offset",));
            output_indices.push(format!("{output_offset}.y + {tile_n} * 4u + y_offset",));
            output.check_bounds(
                kernel,
                output_indices.iter().cloned(),
                |kernel: &mut GenericKernel| {
                    write!(kernel, "let output_index = ")?;
                    output.strided_index(kernel, output_indices.iter().cloned());
                    writeln!(kernel, ";")?;
                    let result = post_element_wise_functions.iter().fold(
                        format!(
                            "{}(acc[{tile_m}][{tile_n}][y_offset][x_offset])",
                            op.input_datatype
                        ),
                        |acc, f| f.call(vec![acc]),
                    );
                    writeln!(kernel, "{output}[output_index] = {result};")?;
                    std::fmt::Result::Ok(())
                },
            )?;

            writeln!(kernel, "}}")?;
            writeln!(kernel, "}}")?;
        }
    }

    Ok(())
}
