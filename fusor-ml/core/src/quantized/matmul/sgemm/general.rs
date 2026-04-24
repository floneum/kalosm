use crate::{
    DataTypeEnum, dequantize_vec4_block,
    mir::{
        globals::KernelGlobalSpace,
        inputs::{QMatrixInput, TensorInput},
        kernel::GenericKernel,
    },
    quantized::matmul::QMatMulOperation,
    util::maybe_vec_storage_type,
};
use std::fmt::Write;
use std::sync::OnceLock;

/// Configuration for general SGEMM algorithm
#[derive(Debug, Clone, Copy)]
pub struct GeneralSgemmConfig {
    /// Size of the vector to dot at a time
    pub vector_size: u32,
}

impl GeneralSgemmConfig {
    /// Default configuration
    pub const fn default() -> Self {
        Self { vector_size: 4 }
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.vector_size == 0 {
            return Err("vector_size must be greater than 0".to_string());
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub fn general_sgemm_with_config(
    op: &QMatMulOperation,
    kernel: &mut GenericKernel,
    input_a: &TensorInput,
    input_b: &QMatrixInput,
    bias: Option<&TensorInput>,
    output: &TensorInput,
    n_size: &str,
    m_size: &str,
    k_size: &str,
    config: GeneralSgemmConfig,
) {
    let global_id = kernel.global_id();
    let elements_per_block = op.elements_per_block();
    let input_datatype = op.input_datatype;
    let dtype = op.matmul_datatype();
    let pre_element_wise_functions = OnceLock::new();
    let post_element_wise_functions = OnceLock::new();

    // Validate configuration
    config.validate().unwrap();

    let sgemm_vector_size = config.vector_size;

    writeln!(kernel, "let x = {global_id}.x;").unwrap();
    writeln!(kernel, "let y = {global_id}.y;").unwrap();

    // Handle batch dimensions
    writeln!(kernel, "var block_batch = {global_id}.z;").unwrap();

    // Decompose the batch index for higher-dimensional tensors
    for dim in (0..input_a.rank()).rev().skip(2) {
        let shape = input_a.shape_binding(dim);
        writeln!(kernel, "let block_batch_{dim} = block_batch % {shape};").unwrap();
        writeln!(kernel, "block_batch = block_batch / {shape};").unwrap();
    }

    let acc_storage_type = maybe_vec_storage_type(sgemm_vector_size, DataTypeEnum::F32);
    writeln!(
        kernel,
        "var acc: {acc_storage_type} = {acc_storage_type}();"
    )
    .unwrap();

    writeln!(
        kernel,
        "let k_block_size = ({k_size} + {elements_per_block} - 1u) / {elements_per_block};"
    )
    .unwrap();
    writeln!(kernel, "var a_index_offset = 0u;").unwrap();

    // Calculate one block sized group
    writeln!(kernel, "if x < {n_size} && y < {m_size} {{").unwrap();

    writeln!(kernel, "for (var k = 0u; k < k_block_size; k += 1u) {{").unwrap();

    // Pack the individual dequantized values into vectors
    writeln!(kernel, "let chunk = &{input_b}[k + x * k_block_size];").unwrap();

    dequantize_vec4_block(
        kernel,
        op.matrix.datatype,
        "chunk".to_string(),
        dtype,
        |index, data, code| {
            let pre_element_wise_functions =
                pre_element_wise_functions.get_or_init(|| op.pre_element_wise.add_functions(code));
            writeln!(code, "{{",).unwrap();
            writeln!(
                code,
                "let a_index_local_offset = a_index_offset + ({index})*{sgemm_vector_size};"
            )
            .unwrap();
            write!(code, "let a_values = vec{sgemm_vector_size}<{dtype}>(",).unwrap();
            for local in 0..sgemm_vector_size {
                if local > 0 {
                    write!(code, ", ").unwrap();
                }
                let mut raw_input = String::new();
                write!(&mut raw_input, "{input_a}[").unwrap();
                let mut indices = vec![];
                // Add batch indices first
                for dim in (0..input_a.rank()).rev().skip(2) {
                    indices.push(format!("block_batch_{dim}"));
                }
                // Then add M and K indices
                indices.push("y".to_string());
                indices.push(format!("a_index_local_offset + {local}"));
                input_a.strided_index(&mut raw_input, indices);
                write!(&mut raw_input, "]").unwrap();
                let processed = pre_element_wise_functions
                    .iter()
                    .fold(raw_input, |acc, f| f.call(vec![acc]));
                write!(code, "{processed}").unwrap();
            }
            writeln!(code, ");").unwrap();

            writeln!(code, "acc += {acc_storage_type}(a_values * {data});").unwrap();
            writeln!(code, "}}").unwrap();
        },
    );

    writeln!(kernel, "a_index_offset += {elements_per_block};").unwrap();

    writeln!(kernel, "}}").unwrap();

    writeln!(kernel, "}}").unwrap();

    // Then write the result
    writeln!(kernel, "if x < {n_size} && y < {m_size} {{").unwrap();
    write!(kernel, "let output_index = ").unwrap();
    let mut output_indices = vec![];
    // Add batch indices first
    for dim in (0..output.rank()).rev().skip(2) {
        output_indices.push(format!("block_batch_{dim}"));
    }
    // Then add M and N indices
    output_indices.push("y".to_string());
    output_indices.push("x".to_string());
    output.strided_index(kernel, output_indices);
    writeln!(kernel, ";").unwrap();
    let acc_sum = match sgemm_vector_size {
        1 => "acc".to_string(),
        2..=4 => (0..sgemm_vector_size)
            .map(|index| format!("acc[{index}]"))
            .collect::<Vec<_>>()
            .join(" + "),
        _ => (0..sgemm_vector_size)
            .map(|index| format!("acc[{index}]"))
            .collect::<Vec<_>>()
            .join(" + "),
    };
    let post_fns =
        post_element_wise_functions.get_or_init(|| op.post_element_wise.add_functions(kernel));
    let result =
        op.apply_bias_and_post(bias, "x", format!("{input_datatype}({acc_sum})"), post_fns);
    writeln!(kernel, "{output}[output_index] = {result};").unwrap();
    writeln!(kernel, "}}").unwrap();
}

#[allow(clippy::too_many_arguments)]
pub fn general_q4_0_sgemm(
    op: &QMatMulOperation,
    kernel: &mut GenericKernel,
    input_a: &TensorInput,
    input_b: &QMatrixInput,
    bias: Option<&TensorInput>,
    output: &TensorInput,
    n_size: &str,
    m_size: &str,
    k_size: &str,
    use_f16: bool,
) {
    let global_id = kernel.global_id();
    let elements_per_block = op.elements_per_block();
    let input_datatype = op.input_datatype;
    let pre_element_wise_functions = OnceLock::new();
    let post_element_wise_functions = OnceLock::new();

    writeln!(kernel, "let x = {global_id}.x;").unwrap();
    writeln!(kernel, "let y = {global_id}.y;").unwrap();
    writeln!(kernel, "var block_batch = {global_id}.z;").unwrap();

    for dim in (0..input_a.rank()).rev().skip(2) {
        let shape = input_a.shape_binding(dim);
        writeln!(kernel, "let block_batch_{dim} = block_batch % {shape};").unwrap();
        writeln!(kernel, "block_batch = block_batch / {shape};").unwrap();
    }

    writeln!(
        kernel,
        "let k_block_size = ({k_size} + {elements_per_block} - 1u) / {elements_per_block};"
    )
    .unwrap();
    writeln!(kernel, "if x < {n_size} && y < {m_size} {{").unwrap();
    writeln!(kernel, "var acc = f32(0.0);").unwrap();
    writeln!(kernel, "for (var k = 0u; k < k_block_size; k += 1u) {{").unwrap();
    writeln!(kernel, "let chunk = &{input_b}[k + x * k_block_size];").unwrap();
    writeln!(kernel, "let a_index_base = k * {elements_per_block};").unwrap();
    writeln!(kernel, "var max_abs = f32(0.0);").unwrap();

    for local in 0..32u32 {
        let pre_element_wise_functions =
            pre_element_wise_functions.get_or_init(|| op.pre_element_wise.add_functions(kernel));
        let mut raw_input = String::new();
        write!(&mut raw_input, "{input_a}[").unwrap();
        let mut indices = Vec::new();
        for dim in (0..input_a.rank()).rev().skip(2) {
            indices.push(format!("block_batch_{dim}"));
        }
        indices.push("y".to_string());
        indices.push(format!("a_index_base + {local}u"));
        input_a.strided_index(&mut raw_input, indices);
        write!(&mut raw_input, "]").unwrap();
        let processed = pre_element_wise_functions
            .iter()
            .fold(raw_input, |acc, f| f.call(vec![acc]));
        writeln!(kernel, "let a_{local} = f32({processed});").unwrap();
        writeln!(kernel, "max_abs = max(max_abs, abs(a_{local}));").unwrap();
    }

    writeln!(kernel, "var inv_scale = f32(0.0);").unwrap();
    writeln!(kernel, "if max_abs != f32(0.0) {{").unwrap();
    writeln!(kernel, "inv_scale = f32(127.0) / max_abs;").unwrap();
    writeln!(kernel, "}}").unwrap();
    writeln!(
        kernel,
        "let q8_scale_raw = select(f32(0.0), max_abs / f32(127.0), max_abs != f32(0.0));"
    )
    .unwrap();
    if use_f16 {
        writeln!(kernel, "let q8_scale = f32(f16(q8_scale_raw));").unwrap();
    } else {
        writeln!(kernel, "let q8_scale = q8_scale_raw;").unwrap();
    }
    writeln!(kernel, "var block_sum = i32(0);").unwrap();

    for local in 0..32u32 {
        let byte = if local < 16 { local } else { local - 16 };
        let word = byte / 4;
        let shift = (byte % 4) * 8 + if local < 16 { 0 } else { 4 };
        writeln!(
            kernel,
            "let q4_{local} = i32((chunk.data[{word}] >> {shift}u) & 0xFu) - 8;"
        )
        .unwrap();
        writeln!(
            kernel,
            "let q8_{local} = i32(clamp(round(a_{local} * inv_scale), -127.0, 127.0));"
        )
        .unwrap();
        writeln!(kernel, "block_sum += q4_{local} * q8_{local};").unwrap();
    }

    writeln!(
        kernel,
        "acc += f32(block_sum) * f32(chunk.scale) * q8_scale;"
    )
    .unwrap();
    writeln!(kernel, "}}").unwrap();

    write!(kernel, "let output_index = ").unwrap();
    let mut output_indices = Vec::new();
    for dim in (0..output.rank()).rev().skip(2) {
        output_indices.push(format!("block_batch_{dim}"));
    }
    output_indices.push("y".to_string());
    output_indices.push("x".to_string());
    output.strided_index(kernel, output_indices);
    writeln!(kernel, ";").unwrap();
    let post_fns =
        post_element_wise_functions.get_or_init(|| op.post_element_wise.add_functions(kernel));
    let result = op.apply_bias_and_post(bias, "x", format!("{input_datatype}(acc)"), post_fns);
    writeln!(kernel, "{output}[output_index] = {result};").unwrap();
    writeln!(kernel, "}}").unwrap();
}

#[allow(clippy::too_many_arguments)]
pub fn general_q4k_sgemm(
    op: &QMatMulOperation,
    kernel: &mut GenericKernel,
    input_a: &TensorInput,
    input_b: &QMatrixInput,
    bias: Option<&TensorInput>,
    output: &TensorInput,
    n_size: &str,
    m_size: &str,
    k_size: &str,
    use_f16: bool,
) {
    let elements_per_block = op.elements_per_block();
    let input_datatype = op.input_datatype;
    let pre_element_wise_functions = OnceLock::new();
    let post_element_wise_functions = OnceLock::new();
    let q8_cache = kernel.add_global_array(
        KernelGlobalSpace::Workgroup,
        DataTypeEnum::F32,
        elements_per_block.to_string(),
    );
    let max_cache = kernel.add_global_array(
        KernelGlobalSpace::Workgroup,
        DataTypeEnum::F32,
        "64".to_string(),
    );
    let scale_offset_fn = kernel.add_function(
        "vec2<f32>",
        r#"
var output = vec2<f32>(0.0, 0.0);
if group < 4u {
    let shift = group * 8u;
    output = vec2<f32>(
        f32((scale_bytes_0 >> shift) & 0x3Fu),
        f32((scale_bytes_1 >> shift) & 0x3Fu)
    );
} else {
    let shift = (group - 4u) * 8u;
    let high_shift = shift + 6u;
    output = vec2<f32>(
        f32(((scale_bytes_2 >> shift) & 0x0Fu) | (((scale_bytes_0 >> high_shift) & 0x03u) << 4u)),
        f32(((scale_bytes_2 >> (shift + 4u)) & 0x0Fu) | (((scale_bytes_1 >> high_shift) & 0x03u) << 4u))
    );
}
"#,
        [
            ("scale_bytes_0".to_string(), "u32".to_string()),
            ("scale_bytes_1".to_string(), "u32".to_string()),
            ("scale_bytes_2".to_string(), "u32".to_string()),
            ("group".to_string(), "u32".to_string()),
        ],
    );

    let workgroup_id = kernel.workgroup_index();
    let local_id = kernel.workgroup_local_index();
    writeln!(kernel, "let local_id = {local_id};").unwrap();
    writeln!(kernel, "let x = {workgroup_id}.x * 64u + local_id;").unwrap();
    writeln!(kernel, "let y = {workgroup_id}.y;").unwrap();
    writeln!(kernel, "var block_batch = {workgroup_id}.z;").unwrap();

    for dim in (0..input_a.rank()).rev().skip(2) {
        let shape = input_a.shape_binding(dim);
        writeln!(kernel, "let block_batch_{dim} = block_batch % {shape};").unwrap();
        writeln!(kernel, "block_batch = block_batch / {shape};").unwrap();
    }

    writeln!(
        kernel,
        "let k_block_size = ({k_size} + {elements_per_block} - 1u) / {elements_per_block};"
    )
    .unwrap();
    writeln!(kernel, "var acc = f32(0.0);").unwrap();
    writeln!(kernel, "for (var k = 0u; k < k_block_size; k += 1u) {{").unwrap();
    writeln!(kernel, "let a_index_base = k * {elements_per_block};").unwrap();
    writeln!(kernel, "var thread_max_abs = f32(0.0);").unwrap();
    writeln!(
        kernel,
        "for (var offset = 0u; offset < {elements_per_block}; offset += 64u) {{"
    )
    .unwrap();
    let pre_element_wise_functions =
        pre_element_wise_functions.get_or_init(|| op.pre_element_wise.add_functions(kernel));
    let a_value = q4k_input_value_expr(input_a, "offset + local_id", pre_element_wise_functions);
    writeln!(kernel, "let a_value = f32({a_value});").unwrap();
    writeln!(
        kernel,
        "thread_max_abs = max(thread_max_abs, abs(a_value));"
    )
    .unwrap();
    writeln!(kernel, "}}").unwrap();
    writeln!(kernel, "{max_cache}[local_id] = thread_max_abs;").unwrap();
    writeln!(kernel, "workgroupBarrier();").unwrap();
    writeln!(kernel, "if local_id == 0u {{").unwrap();
    writeln!(kernel, "var block_max_abs = f32(0.0);").unwrap();
    writeln!(kernel, "for (var index = 0u; index < 64u; index += 1u) {{").unwrap();
    writeln!(
        kernel,
        "block_max_abs = max(block_max_abs, {max_cache}[index]);"
    )
    .unwrap();
    writeln!(kernel, "}}").unwrap();
    writeln!(kernel, "{max_cache}[0] = block_max_abs;").unwrap();
    writeln!(kernel, "}}").unwrap();
    writeln!(kernel, "workgroupBarrier();").unwrap();
    writeln!(kernel, "let max_abs = {max_cache}[0];").unwrap();
    writeln!(kernel, "var inv_scale = f32(0.0);").unwrap();
    writeln!(kernel, "if max_abs != f32(0.0) {{").unwrap();
    writeln!(kernel, "inv_scale = f32(127.0) / max_abs;").unwrap();
    writeln!(kernel, "}}").unwrap();
    writeln!(
        kernel,
        "let q8_scale_raw = select(f32(0.0), max_abs / f32(127.0), max_abs != f32(0.0));"
    )
    .unwrap();
    if use_f16 {
        writeln!(kernel, "let q8_scale = f32(f16(q8_scale_raw));").unwrap();
    } else {
        writeln!(kernel, "let q8_scale = q8_scale_raw;").unwrap();
    }
    writeln!(
        kernel,
        "for (var offset = 0u; offset < {elements_per_block}; offset += 64u) {{"
    )
    .unwrap();
    let a_value = q4k_input_value_expr(input_a, "offset + local_id", pre_element_wise_functions);
    writeln!(kernel, "let a_value = f32({a_value});").unwrap();
    writeln!(
        kernel,
        "{q8_cache}[offset + local_id] = f32(i32(clamp(round(a_value * inv_scale), -127.0, 127.0)));"
    )
    .unwrap();
    writeln!(kernel, "}}").unwrap();
    writeln!(kernel, "workgroupBarrier();").unwrap();

    writeln!(kernel, "if x < {n_size} && y < {m_size} {{").unwrap();
    writeln!(kernel, "let chunk = &{input_b}[k + x * k_block_size];").unwrap();
    writeln!(kernel, "let scale_bytes_0 = chunk.scales[0];").unwrap();
    writeln!(kernel, "let scale_bytes_1 = chunk.scales[1];").unwrap();
    writeln!(kernel, "let scale_bytes_2 = chunk.scales[2];").unwrap();
    writeln!(kernel, "var scale_dot_sum = f32(0.0);").unwrap();
    writeln!(kernel, "var offset_y_sum = f32(0.0);").unwrap();
    writeln!(
        kernel,
        "for (var local_k = 0u; local_k < {elements_per_block}; local_k += 1u) {{"
    )
    .unwrap();
    writeln!(kernel, "let pair = local_k / 64u;").unwrap();
    writeln!(kernel, "let within_pair = local_k % 64u;").unwrap();
    writeln!(kernel, "let high = within_pair >= 32u;").unwrap();
    writeln!(kernel, "let data_byte = pair * 32u + (within_pair % 32u);").unwrap();
    writeln!(kernel, "let data_word = data_byte / 4u;").unwrap();
    writeln!(
        kernel,
        "let data_shift = (data_byte % 4u) * 8u + select(0u, 4u, high);"
    )
    .unwrap();
    writeln!(
        kernel,
        "let q4 = f32((chunk.data[data_word] >> data_shift) & 0xFu);"
    )
    .unwrap();
    writeln!(kernel, "let group = pair * 2u + select(0u, 1u, high);").unwrap();
    let scale_offset = scale_offset_fn.call(vec![
        "scale_bytes_0".to_string(),
        "scale_bytes_1".to_string(),
        "scale_bytes_2".to_string(),
        "group".to_string(),
    ]);
    writeln!(kernel, "let scale_offset = {scale_offset};").unwrap();
    writeln!(kernel, "let q8_f = {q8_cache}[local_k];").unwrap();
    writeln!(kernel, "scale_dot_sum += scale_offset.x * q4 * q8_f;").unwrap();
    writeln!(kernel, "offset_y_sum += scale_offset.y * q8_f;").unwrap();
    writeln!(kernel, "}}").unwrap();
    writeln!(
        kernel,
        "acc += q8_scale * (f32(chunk.scale) * scale_dot_sum - f32(chunk.min) * offset_y_sum);"
    )
    .unwrap();
    writeln!(kernel, "}}").unwrap();
    writeln!(kernel, "workgroupBarrier();").unwrap();
    writeln!(kernel, "}}").unwrap();

    writeln!(kernel, "if x < {n_size} && y < {m_size} {{").unwrap();
    write!(kernel, "let output_index = ").unwrap();
    let mut output_indices = Vec::new();
    for dim in (0..output.rank()).rev().skip(2) {
        output_indices.push(format!("block_batch_{dim}"));
    }
    output_indices.push("y".to_string());
    output_indices.push("x".to_string());
    output.strided_index(kernel, output_indices);
    writeln!(kernel, ";").unwrap();
    let post_fns =
        post_element_wise_functions.get_or_init(|| op.post_element_wise.add_functions(kernel));
    let result = op.apply_bias_and_post(bias, "x", format!("{input_datatype}(acc)"), post_fns);
    writeln!(kernel, "{output}[output_index] = {result};").unwrap();
    writeln!(kernel, "}}").unwrap();
}

fn q4k_input_value_expr(
    input_a: &TensorInput,
    local_expr: &str,
    pre_element_wise_functions: &[crate::mir::function::Function],
) -> String {
    let mut raw_input = String::new();
    write!(&mut raw_input, "{input_a}[").unwrap();
    let mut indices = Vec::new();
    for dim in (0..input_a.rank()).rev().skip(2) {
        indices.push(format!("block_batch_{dim}"));
    }
    indices.push("y".to_string());
    indices.push(format!("a_index_base + {local_expr}"));
    input_a.strided_index(&mut raw_input, indices);
    write!(&mut raw_input, "]").unwrap();
    pre_element_wise_functions
        .iter()
        .fold(raw_input, |acc, f| f.call(vec![acc]))
}
