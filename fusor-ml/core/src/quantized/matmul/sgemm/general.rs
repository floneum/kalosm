use crate::{
    dequantize_vec4_block,
    mir::{
        inputs::{QMatrixInput, TensorInput},
        kernel::GenericKernel,
    },
    quantized::matmul::QMatMulOperation,
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

    writeln!(kernel, "var acc = 0.0;").unwrap();

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

            writeln!(code, "acc += dot(a_values, {data});").unwrap();
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
    let result = post_element_wise_functions
        .get_or_init(|| op.post_element_wise.add_functions(kernel))
        .iter()
        .fold(format!("{input_datatype}(acc)"), |acc, f| f.call(vec![acc]));
    writeln!(kernel, "{output}[output_index] = {result};").unwrap();
    writeln!(kernel, "}}").unwrap();
}
