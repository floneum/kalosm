use crate::{
    DataTypeEnum, dequantize_vec4_block,
    mir::{
        globals::KernelGlobalSpace,
        inputs::{QMatrixInput, TensorInput},
        kernel::GenericKernel,
        workgroup_shape::WorkgroupShape,
    },
    quantized::matmul::{
        QMatMulOperation,
        sgemv::{SGEMV_CHUNK_SIZE, SGEMV_VECTOR_SIZE, decompose_workgroup_index},
    },
    util::{
        maybe_vec_storage_add, maybe_vec_storage_index, maybe_vec_storage_subgroup_add,
        maybe_vec_storage_type, maybe_vec_storage_type_enum,
    },
};
use std::fmt::Write;

#[allow(clippy::too_many_arguments)]
pub(crate) fn general_sgemv(
    op: &QMatMulOperation,
    kernel: &mut GenericKernel,
    workgroup_size: &WorkgroupShape,
    input_a: &TensorInput,
    input_b: &QMatrixInput,
    output: &TensorInput,
    n_size: &str,
    m_size: &str,
    k_size: &str,
    graph: &crate::compute_graph::ComputeGraphInner,
) {
    let blocksize = workgroup_size.x();
    let dtype = op.input_datatype;
    let workgroup_local_index = kernel.workgroup_local_index();
    let elements_per_block = op.elements_per_block();
    let device = graph.device();

    // Calculate n_workgroups for decomposing the linearized workgroup index
    let n_workgroups = format!("(({n_size} + {SGEMV_CHUNK_SIZE} - 1) / {SGEMV_CHUNK_SIZE})");

    // Decompose linearized workgroup index into (n_workgroup_idx, m_idx, batch_idx)
    // This handles cases where workgroups are distributed across multiple dispatch dimensions
    decompose_workgroup_index(kernel, workgroup_size, m_size, &n_workgroups);

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

    // Workgroup offset in the N dimension
    writeln!(
        kernel,
        "let workgroup_offset = n_workgroup_idx * {SGEMV_CHUNK_SIZE};"
    )
    .unwrap();

    // Always accumulate in f32 for precision, convert to output dtype at the end
    let acc_storage_type = maybe_vec_storage_type(SGEMV_CHUNK_SIZE, DataTypeEnum::F32);

    writeln!(kernel, "var acc = {acc_storage_type}();").unwrap();

    // Find the reduce size in blocks rounded up
    writeln!(
        kernel,
        "let k_block_size = ({k_size} + {elements_per_block} - 1) / {elements_per_block};"
    )
    .unwrap();

    // Find this threads position in the workgroup
    writeln!(kernel, "let base_axis_index = {workgroup_local_index};").unwrap();
    writeln!(kernel, "let end_axis_index = k_block_size;").unwrap();
    writeln!(kernel, "var index = base_axis_index;").unwrap();

    let chunk_blocks = elements_per_block / SGEMV_VECTOR_SIZE;
    debug_assert!(elements_per_block.is_multiple_of(SGEMV_VECTOR_SIZE));
    writeln!(
        kernel,
        "var a_cache = array<vec{SGEMV_VECTOR_SIZE}<{dtype}>, {chunk_blocks}>();"
    )
    .unwrap();

    // Loop over all of the blocks this thread is responsible for
    writeln!(kernel, "while (index < end_axis_index) {{").unwrap();
    {
        // Load all elements of a into a cache first
        writeln!(kernel, "for (var i = 0u; i < {chunk_blocks}; i += 1u) {{").unwrap();
        {
            // Get the values first
            for i in 0..SGEMV_VECTOR_SIZE {
                writeln!(kernel, "let input_a_{i}_index = index * {elements_per_block} + i * {SGEMV_VECTOR_SIZE} + {i};").unwrap();
                write!(kernel, "let input_a_{i} = {input_a}[").unwrap();
                let mut indices = Vec::new();
                // Add batch indices first
                for dim in (0..input_a.rank()).rev().skip(2) {
                    indices.push(format!("batch_idx_{dim}"));
                }
                // Then add M and K indices
                indices.push("m_idx".to_string());
                indices.push(format!("input_a_{i}_index"));
                input_a.strided_index(kernel, indices);
                writeln!(kernel, "];").unwrap();
            }
            // The pack them into a vector and write to the cache
            write!(kernel, "a_cache[i] = vec{SGEMV_VECTOR_SIZE}(").unwrap();
            for i in 0..SGEMV_VECTOR_SIZE {
                if i > 0 {
                    write!(kernel, ", ").unwrap();
                }
                write!(kernel, "input_a_{i}").unwrap();
            }
            writeln!(kernel, ");").unwrap();
        }
        writeln!(kernel, "}}").unwrap();

        if SGEMV_CHUNK_SIZE > 1 {
            writeln!(
                kernel,
                "for (var acc_offset = 0u; acc_offset < {SGEMV_CHUNK_SIZE}; acc_offset += 1u) {{"
            )
            .unwrap();
        }
        let index = if SGEMV_CHUNK_SIZE > 1 {
            "(workgroup_offset + acc_offset)"
        } else {
            "workgroup_offset"
        };
        writeln!(
            kernel,
            "let chunk = &{input_b}[{index} * k_block_size + index];"
        )
        .unwrap();

        let acc_indexed = maybe_vec_storage_index(SGEMV_CHUNK_SIZE, "acc", "acc_offset");
        // Always convert a_cache to f32 for the dot product since dequantize outputs f32
        // and we accumulate in f32 for precision
        dequantize_vec4_block(
            kernel,
            op.matrix.datatype,
            "chunk".to_string(),
            DataTypeEnum::F32,
            |index, data, code| {
                writeln!(
                    code,
                    "{acc_indexed} += dot(vec4<f32>(a_cache[{index}]), {data});"
                )
                .unwrap();
            },
        );

        if SGEMV_CHUNK_SIZE > 1 {
            writeln!(kernel, "}}").unwrap();
        }

        writeln!(kernel, "index += {blocksize}u;").unwrap();
    }

    writeln!(kernel, "}}").unwrap();

    // Reduce with subgroup operations if the device supports subgroups
    if device.subgroups_supported() {
        // Get the sum among all threads in the subgroup
        writeln!(
            kernel,
            "acc = {};",
            maybe_vec_storage_subgroup_add(SGEMV_CHUNK_SIZE, "acc")
        )
        .unwrap();

        // We don't need to synchronize between the whole workgroup if there is only one subgroup
        let subgroup_size = device.max_subgroup_size();
        if blocksize > subgroup_size {
            let local_data = kernel.add_global_array(
                KernelGlobalSpace::Workgroup,
                maybe_vec_storage_type_enum(SGEMV_CHUNK_SIZE, DataTypeEnum::F32),
                subgroup_size.to_string(),
            );
            let subgroup_id = kernel.subgroup_index();
            let subgroup_local_id = kernel.subgroup_local_index();
            let subgroups_per_workgroup = kernel.subgroups_per_workgroup();

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
                    writeln!(kernel, "acc = {acc_storage_type}();").unwrap();
                }
                writeln!(kernel, "}}").unwrap();
            }
            writeln!(kernel, "}}").unwrap();

            // Finally get the final sum across all threads in the workgroup
            writeln!(
                kernel,
                "acc = {};",
                maybe_vec_storage_subgroup_add(SGEMV_CHUNK_SIZE, "acc",)
            )
            .unwrap();
        }
    }
    // Otherwise, reduce using workgroup memory
    else {
        let local_data = kernel.add_global_array(
            KernelGlobalSpace::Workgroup,
            maybe_vec_storage_type_enum(SGEMV_CHUNK_SIZE, DataTypeEnum::F32),
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
                maybe_vec_storage_add(SGEMV_CHUNK_SIZE, "acc", "neighbor")
            )
            .unwrap();
            writeln!(kernel, "}}").unwrap();
        }
    }

    // If this is not the first simd thread in the workgroup, we can return early
    writeln!(kernel, "if {workgroup_local_index} != 0u {{ return; }}").unwrap();

    // Write the output to the output tensor if this is the first thread in the workgroup
    if SGEMV_CHUNK_SIZE > 1 {
        writeln!(
            kernel,
            "for (var acc_offset = 0u; acc_offset < {SGEMV_CHUNK_SIZE}; acc_offset += 1u) {{"
        )
        .unwrap();
    }
    {
        if SGEMV_CHUNK_SIZE > 1 {
            writeln!(kernel, "let output_index = workgroup_offset + acc_offset;").unwrap();
        } else {
            writeln!(kernel, "let output_index = workgroup_offset;").unwrap();
        }
        write!(kernel, "{output}[").unwrap();
        let mut output_indices = Vec::new();
        // Add batch indices first
        for dim in (0..output.rank()).rev().skip(2) {
            output_indices.push(format!("batch_idx_{dim}"));
        }
        // Then add M and N indices
        output_indices.push("m_idx".to_string());
        output_indices.push("output_index".to_string());
        output.strided_index(kernel, output_indices);
        // Convert from f32 accumulator to output dtype (single element per iteration)
        let acc_val = maybe_vec_storage_index(SGEMV_CHUNK_SIZE, "acc", "acc_offset");
        writeln!(kernel, "] = {dtype}({acc_val});").unwrap();
    }
    if SGEMV_CHUNK_SIZE > 1 {
        writeln!(kernel, "}}").unwrap();
    }
}
