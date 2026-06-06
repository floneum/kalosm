//! Quantized matmul fusion conformance cases.

use crate::common::quantized::{
    deterministic_input, q_mat_mul_input_fuzz, q4k_raw_bytes, q6k_raw_bytes, q8_0_raw_bytes,
    qmatrix_from_raw_bytes,
};
use fusor::{Device, GgmlType, QMatrix, Tensor};
use fusor_conformance::{
    AssertionCase, AssertionCases, approx_compare, available_devices, cases_from_rows,
    exact_value_compare,
};
use rand::distr::Uniform;

async fn gpu_devices() -> Vec<Device> {
    available_devices()
        .await
        .into_iter()
        .filter(|device| device.as_gpu().is_some())
        .collect()
}

fn assert_gpu_kernel_property(
    name: impl Into<String>,
    property: impl Fn(Device) -> bool + Clone + Send + 'static,
) -> AssertionCase {
    fusor_conformance::assert(move |device: Device| {
        let property = property.clone();
        async move { property(device) }
    })
    .arg(|device: &Device| device.clone())
    .equal_to(move |_device: Device| async move { true })
    .compare_with(exact_value_compare())
    .baseline_on_test_device()
    .devices_async(gpu_devices())
    .runs(1)
    .into_case(name)
}

pub fn q4k_q6k_ffn_chain_matches_cpu_reference_for_decode_rows() -> AssertionCase {
    let hidden = 512usize;
    let intermediate = 512usize;
    let output = 128usize;
    let gate_bytes = q4k_raw_bytes([intermediate, hidden]);
    let up_bytes = q4k_raw_bytes([intermediate, hidden]);
    let down_bytes = q6k_raw_bytes([output, intermediate]);

    fusor_conformance::assert(move |input: Tensor<2, f32>| {
        let gate_bytes = gate_bytes.clone();
        let up_bytes = up_bytes.clone();
        let down_bytes = down_bytes.clone();
        async move {
            let device = input.device();
            let gate =
                qmatrix_from_raw_bytes(&device, [intermediate, hidden], &gate_bytes, GgmlType::Q4K);
            let up =
                qmatrix_from_raw_bytes(&device, [intermediate, hidden], &up_bytes, GgmlType::Q4K);
            let down =
                qmatrix_from_raw_bytes(&device, [output, intermediate], &down_bytes, GgmlType::Q6K);
            let gate_out = input.q_mat_mul(&gate).silu();
            let up_out = input.q_mat_mul(&up);
            (gate_out * up_out).q_mat_mul(&down).to_concrete()
        }
    })
    .arg(q_mat_mul_input_fuzz(
        1,
        [intermediate, hidden],
        834,
        Uniform::new(-0.25, 0.25).unwrap(),
    ))
    .compare_with(fusor_conformance::approx_compare::<2, f32>(5.0))
    .runs(3)
    .into_case("quantized_matmul_fusion::q4k_q6k_ffn_chain_matches_cpu_reference_for_decode_rows")
}

/// The fuser must collapse `rms_norm(...).relu()` (or any unary chain after
/// an RmsNorm) into a single RmsNorm kernel dispatch — the kernel applies
/// the chain in-register before the store. Without the rule, the unfused
/// source resolves to 2 dispatches.
pub fn rmsnorm_post_relu_resolves_to_single_kernel() -> AssertionCase {
    let cols = 64usize;
    let input_data = vec![vec![0.1f32; cols]; 4];
    let weight_data = vec![1.2f32; cols];
    assert_gpu_kernel_property(
        "quantized_matmul_fusion::rmsnorm_post_relu_resolves_to_single_kernel",
        move |device| {
            let Some(gpu) = device.as_gpu() else {
                return true;
            };
            if !gpu.subgroups_supported() {
                return true;
            }
            let input: Tensor<2, f32> = Tensor::new(&device, &input_data);
            let weight: Tensor<1, f32> = Tensor::new(&device, &weight_data);
            input
                .rms_norm_fused::<1, 1>(&weight, None, 1e-5)
                .relu()
                .to_concrete()
                .as_gpu()
                .is_some_and(|gpu_out| gpu_out.count_kernels_to_resolve() == 1)
        },
    )
}

/// All three Q4K QMatMul fusion shapes must each collapse to a single QMatMul
/// kernel dispatch:
///   * `relu(input).q_mat_mul(weights)` — qgemv applies the activation to each
///     loaded activation tile before the dot product (subgroup path only).
///   * `q_mat_mul(...).relu()` — qgemv applies the unary chain in-register
///     before the store (all GPU backends).
///   * concat/split SwiGLU via `q_mat_mul_paired_silu_product` — the dynamic
///     paired qmatmul fuses the gate/up product (subgroup path only).
///
/// Without the corresponding fuser rule each source resolves to 2 dispatches.
pub fn q4k_qmatmul_fusion_kernels() -> AssertionCases {
    let weight_shape = [4, 512];
    let pre_relu_bytes = q4k_raw_bytes(weight_shape);
    let pre_relu_input = vec![vec![0.1f32; weight_shape[1]]; 1];

    let post_relu_bytes = q4k_raw_bytes(weight_shape);
    let post_relu_input = vec![vec![0.1f32; weight_shape[1]]; 1];

    let swiglu_shape = [64, 512];
    let swiglu_bytes = q4k_raw_bytes(swiglu_shape);
    let swiglu_input = vec![vec![0.1f32; swiglu_shape[1]]; 1];

    cases_from_rows([
        // `relu(input).q_mat_mul(weights)` pre-activation fusion (subgroup path).
        assert_gpu_kernel_property(
            "quantized_matmul_fusion::q4k_qmatmul_pre_relu_resolves_to_single_kernel",
            move |device| {
                let Some(gpu) = device.as_gpu() else {
                    return true;
                };
                if !gpu.subgroups_supported() {
                    return true;
                }
                let weights =
                    qmatrix_from_raw_bytes(&device, weight_shape, &pre_relu_bytes, GgmlType::Q4K);
                let input: Tensor<2, f32> = Tensor::new(&device, &pre_relu_input);
                input
                    .relu()
                    .to_concrete()
                    .q_mat_mul(&weights)
                    .to_concrete()
                    .as_gpu()
                    .is_some_and(|gpu_out| gpu_out.count_kernels_to_resolve() == 1)
            },
        ),
        // `q_mat_mul(...).relu()` post-activation fusion (all GPU backends).
        assert_gpu_kernel_property(
            "quantized_matmul_fusion::q4k_qmatmul_post_relu_resolves_to_single_kernel",
            move |device| {
                let weights =
                    qmatrix_from_raw_bytes(&device, weight_shape, &post_relu_bytes, GgmlType::Q4K);
                let input: Tensor<2, f32> = Tensor::new(&device, &post_relu_input);
                input
                    .q_mat_mul(&weights)
                    .relu()
                    .to_concrete()
                    .as_gpu()
                    .is_some_and(|gpu_out| gpu_out.count_kernels_to_resolve() == 1)
            },
        ),
        // concat/split SwiGLU paired qmatmul fusion (subgroup path).
        assert_gpu_kernel_property(
            "quantized_matmul_fusion::q4k_concat_split_swiglu_resolves_to_single_dynamic_qmatmul_kernel",
            move |device| {
                let Some(gpu) = device.as_gpu() else {
                    return true;
                };
                if !gpu.subgroups_supported() {
                    return true;
                }
                let weights =
                    qmatrix_from_raw_bytes(&device, swiglu_shape, &swiglu_bytes, GgmlType::Q4K);
                let input: Tensor<2, f32> = Tensor::new(&device, &swiglu_input);
                match (&input, &weights) {
                    (Tensor::Gpu(input), QMatrix::Gpu(weights)) => {
                        Tensor::<2, f32>::Gpu(input.q_mat_mul_paired_silu_product(weights))
                            .to_concrete()
                            .as_gpu()
                            .is_some_and(|gpu_out| gpu_out.count_kernels_to_resolve() == 1)
                    }
                    _ => false,
                }
            },
        ),
    ])
}

/// Q8_0 QMatMul epilogues must apply post-matmul extras (bias add, residual
/// add) in the correct binding order:
///   * a column-vector bias add on a non-block-multiple output width, and
///   * a mixed bias-plus-residual chain, where the residual is pre-biased so
///     the fused epilogue must preserve the binding order.
///
/// Both are deterministic-seed Q8_0 matmuls checked against the CPU reference.
pub fn q8_0_qmatmul_epilogue_tests() -> AssertionCases {
    let weight_shape = [4, 64];

    // Column-add on a non-block-multiple output width.
    let column_add_bytes = q8_0_raw_bytes(weight_shape);
    let column_add_input_shape = [2, weight_shape[1]];
    let column_add_input = deterministic_input(&column_add_input_shape, 1_031);
    let column_add_bias = vec![0.25f32, -0.5, 0.75, -1.0];

    let column_add = fusor_conformance::assert(move |device: Device| {
        let raw_bytes = column_add_bytes.clone();
        let input_data = column_add_input.clone();
        let bias_data = column_add_bias.clone();
        async move {
            let weights = qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, GgmlType::Q8_0);
            let input: Tensor<2, f32> =
                Tensor::from_slice(&device, column_add_input_shape, &input_data);
            let bias: Tensor<1, f32> = Tensor::from_slice(&device, [weight_shape[0]], &bias_data);
            input.q_mat_mul(&weights).add_(&bias).to_concrete()
        }
    })
    .arg(|device: &Device| device.clone())
    .compare_with(approx_compare::<2, f32>(2.0))
    .runs(1)
    .into_case(
        "quantized_matmul_fusion::q8_0_qmatmul_post_column_add_nonmultiple_applies_epilogue",
    );

    // Mixed bias + residual chain that must preserve binding order.
    let mixed_bytes = q8_0_raw_bytes(weight_shape);
    let mixed_input_shape = [2, weight_shape[1]];
    let mixed_output_shape = [2, weight_shape[0]];
    let mixed_input = deterministic_input(&mixed_input_shape, 1_047);
    let mixed_residual = deterministic_input(&mixed_output_shape, 1_211);
    let mixed_bias = vec![0.4f32, -0.2, 0.1, -0.6];

    let mixed_extras = fusor_conformance::assert(move |device: Device| {
        let raw_bytes = mixed_bytes.clone();
        let input_data = mixed_input.clone();
        let residual_data = mixed_residual.clone();
        let bias_data = mixed_bias.clone();
        async move {
            let weights = qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, GgmlType::Q8_0);
            let input: Tensor<2, f32> = Tensor::from_slice(&device, mixed_input_shape, &input_data);
            let residual: Tensor<2, f32> =
                Tensor::from_slice(&device, mixed_output_shape, &residual_data);
            let bias: Tensor<1, f32> = Tensor::from_slice(&device, [weight_shape[0]], &bias_data);
            let residual_biased = residual.add_(&bias);
            input
                .q_mat_mul(&weights)
                .add_(&residual_biased)
                .to_concrete()
        }
    })
    .arg(|device: &Device| device.clone())
    .compare_with(approx_compare::<2, f32>(2.0))
    .runs(1)
    .into_case("quantized_matmul_fusion::q8_0_qmatmul_post_mixed_extras_preserves_binding_order");

    cases_from_rows([column_add, mixed_extras])
}
