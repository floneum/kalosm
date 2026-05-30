mod common;

use common::quantized::{
    deterministic_input, q_mat_mul_input_fuzz, q4k_raw_bytes, q6k_raw_bytes, q8_0_raw_bytes,
    qmatrix_from_raw_bytes,
};
use fusor::{Device, GgmlType, Tensor};
use fusor_conformance::{approx_compare, approx_eq, available_devices};
use rand::distr::Uniform;

#[tokio::test]
async fn q4k_q6k_ffn_chain_matches_cpu_reference_for_decode_rows() {
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
    .compare_with(approx_compare::<2, f32>(5.0))
    .runs(3)
    .await
    .unwrap();
}

/// The fuser must collapse `rms_norm(...).relu()` (or any unary chain after
/// an RmsNorm) into a single RmsNorm kernel dispatch — the kernel applies
/// the chain in-register before the store. Without the rule, the unfused
/// source resolves to 2 dispatches.
#[tokio::test]
async fn rmsnorm_post_relu_resolves_to_single_kernel() {
    let Ok(device) = fusor::Device::new().await else {
        return;
    };
    let Some(gpu_device) = device.as_gpu() else {
        panic!("expected GPU device");
    };
    if !gpu_device.subgroups_supported() {
        return;
    }
    let cols = 64usize;
    let input_data = vec![vec![0.1f32; cols]; 4];
    let weight_data = vec![1.2f32; cols];
    let input: Tensor<2, f32> = Tensor::new(&device, &input_data);
    let weight: Tensor<1, f32> = Tensor::new(&device, &weight_data);

    let output = input
        .rms_norm_fused::<1, 1>(&weight, None, 1e-5)
        .relu()
        .to_concrete();
    let Tensor::Gpu(gpu_out) = output else {
        panic!("expected GPU tensor");
    };
    let kernels = gpu_device.resolve_batch(&[gpu_out.key()]);
    assert_eq!(
        kernels, 1,
        "expected fuser to collapse rms_norm -> relu to 1 dispatch, got {kernels}"
    );
}

/// The fuser must collapse `relu(input).q_mat_mul(weights)` into a single
/// QMatMul kernel — qgemv applies the activation to each loaded activation
/// tile before the dot product. Without the pre-fusion rule, the unfused
/// source resolves to 2 dispatches (nary + matmul).
#[tokio::test]
async fn q4k_qmatmul_pre_relu_resolves_to_single_kernel() {
    let Ok(device) = fusor::Device::new().await else {
        return;
    };
    let weight_shape = [4, 512];
    let raw_bytes = q4k_raw_bytes(weight_shape);
    let input_data = vec![vec![0.1f32; weight_shape[1]]; 1];
    let Some(gpu_device) = device.as_gpu() else {
        panic!("expected GPU device");
    };
    if !gpu_device.subgroups_supported() {
        return;
    }

    let weights = qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, GgmlType::Q4K);
    let input: Tensor<2, f32> = Tensor::new(&device, &input_data);
    let output = input.relu().to_concrete().q_mat_mul(&weights).to_concrete();
    let Tensor::Gpu(gpu_out) = output else {
        panic!("expected GPU tensor");
    };
    let kernels = gpu_device.resolve_batch(&[gpu_out.key()]);
    assert_eq!(
        kernels, 1,
        "expected fuser to collapse relu -> q_mat_mul to 1 dispatch, got {kernels}"
    );
}

/// The fuser must collapse `q_mat_mul → unary chain` (e.g. relu, silu)
/// into a single QMatMul kernel dispatch — qgemv kernels apply the chain
/// in-register before storing. Without the fuser rule, the unfused source
/// resolves to 2 dispatches (matmul + nary).
#[tokio::test]
async fn q4k_qmatmul_post_relu_resolves_to_single_kernel() {
    let Ok(device) = fusor::Device::new().await else {
        return; // No GPU available.
    };
    let weight_shape = [4, 512];
    let raw_bytes = q4k_raw_bytes(weight_shape);
    let input_data = vec![vec![0.1f32; weight_shape[1]]; 1];
    let Some(gpu_device) = device.as_gpu() else {
        panic!("expected GPU device");
    };

    let weights = qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, GgmlType::Q4K);
    let input: Tensor<2, f32> = Tensor::new(&device, &input_data);

    // Natural unfused source: matmul + relu. Fuser should collapse to 1 kernel.
    let output = input.q_mat_mul(&weights).relu().to_concrete();
    let Tensor::Gpu(natural_gpu) = output else {
        panic!("expected GPU tensor");
    };
    let natural_kernels = gpu_device.resolve_batch(&[natural_gpu.key()]);
    assert_eq!(
        natural_kernels, 1,
        "expected fuser to collapse q_mat_mul -> relu to 1 dispatch, got {natural_kernels}"
    );
}

#[tokio::test]
async fn q8_0_qmatmul_post_column_add_nonmultiple_applies_epilogue() {
    let weight_shape = [4, 64];
    let raw_bytes = q8_0_raw_bytes(weight_shape);
    let input_shape = [2, weight_shape[1]];
    let input_data = deterministic_input(&input_shape, 1_031);
    let bias_data = vec![0.25f32, -0.5, 0.75, -1.0];

    let cpu_weights =
        qmatrix_from_raw_bytes(&Device::Cpu, weight_shape, &raw_bytes, GgmlType::Q8_0);
    let cpu_input: Tensor<2, f32> = Tensor::from_slice(&Device::Cpu, input_shape, &input_data);
    let cpu_bias: Tensor<1, f32> = Tensor::from_slice(&Device::Cpu, [weight_shape[0]], &bias_data);
    let expected = cpu_input
        .q_mat_mul(&cpu_weights)
        .add_(&cpu_bias)
        .to_concrete();

    for device in available_devices().await {
        let weights = qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, GgmlType::Q8_0);
        let input: Tensor<2, f32> = Tensor::from_slice(&device, input_shape, &input_data);
        let bias: Tensor<1, f32> = Tensor::from_slice(&device, [weight_shape[0]], &bias_data);
        let actual = input.q_mat_mul(&weights).add_(&bias).to_concrete();
        approx_eq(&actual, &expected, 2.0).await.unwrap();
    }
}

#[tokio::test]
async fn q8_0_qmatmul_post_mixed_extras_preserves_binding_order() {
    let weight_shape = [4, 64];
    let raw_bytes = q8_0_raw_bytes(weight_shape);
    let input_shape = [2, weight_shape[1]];
    let output_shape = [2, weight_shape[0]];
    let input_data = deterministic_input(&input_shape, 1_047);
    let residual_data = deterministic_input(&output_shape, 1_211);
    let bias_data = vec![0.4f32, -0.2, 0.1, -0.6];

    let cpu_weights =
        qmatrix_from_raw_bytes(&Device::Cpu, weight_shape, &raw_bytes, GgmlType::Q8_0);
    let cpu_input: Tensor<2, f32> = Tensor::from_slice(&Device::Cpu, input_shape, &input_data);
    let cpu_residual: Tensor<2, f32> =
        Tensor::from_slice(&Device::Cpu, output_shape, &residual_data);
    let cpu_bias: Tensor<1, f32> = Tensor::from_slice(&Device::Cpu, [weight_shape[0]], &bias_data);
    let cpu_residual_biased = cpu_residual.add_(&cpu_bias);
    let expected = cpu_input
        .q_mat_mul(&cpu_weights)
        .add_(&cpu_residual_biased)
        .to_concrete();

    for device in available_devices().await {
        let weights = qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, GgmlType::Q8_0);
        let input: Tensor<2, f32> = Tensor::from_slice(&device, input_shape, &input_data);
        let residual: Tensor<2, f32> = Tensor::from_slice(&device, output_shape, &residual_data);
        let bias: Tensor<1, f32> = Tensor::from_slice(&device, [weight_shape[0]], &bias_data);
        let residual_biased = residual.add_(&bias);
        let actual = input
            .q_mat_mul(&weights)
            .add_(&residual_biased)
            .to_concrete();
        approx_eq(&actual, &expected, 2.0).await.unwrap();
    }
}
