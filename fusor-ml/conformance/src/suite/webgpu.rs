//! The in-browser WebGPU conformance suite.
//!
//! This is the suite the kalosm-chat web app runs on its `/conformance` route.
//! Each case is expressed through the same [`crate::assert`] builder the native
//! conformance tests use: the builder runs the op once on the CPU device as a
//! baseline, then re-runs it on every GPU variant — the cross product of
//! `{subgroups, no subgroups} × {cold pool, poisoned pool}` (see
//! [`crate::builder::device_test_variants`]) — and compares. The web build never
//! requests `Features::SUBGROUP` and reuses a buffer pool, so the
//! no-subgroup + poisoned-pool variant is the one the browser actually hits;
//! pinning the builder to `[gpu]` exercises all four with no bespoke device
//! plumbing here.
//!
//! Cases whose output is not a tensor (top-k pairs, sampled tokens) cannot use
//! the builder's tensor comparator, so they loop over the same
//! `device_test_variants` directly and assert their own invariants.

use std::fmt::Display;

use fusor::{
    BlockQ4K, Device, GgmlType, GgufBlock, Mirostat2Sampler, Mirostat2SamplerParams, QMatrix,
    Tensor,
};
use thiserror::Error;

use crate::{
    approx_eq,
    builder::device_test_variants,
    common::{
        quantized::{q4k_raw_bytes, q8_0_raw_bytes, qmatrix_from_raw_bytes},
        silu,
    },
};

#[derive(Debug, Error)]
pub enum SuiteError {
    #[error("{case}: {message}")]
    Case { case: String, message: String },
}

impl SuiteError {
    fn case(case: impl Into<String>, message: impl Display) -> Self {
        Self::Case {
            case: case.into(),
            message: message.to_string(),
        }
    }
}

pub async fn run_webgpu_kernel_suite(device: &Device) -> Result<(), SuiteError> {
    run_webgpu_kernel_suite_with_progress(device, |_| {}).await
}

pub async fn run_webgpu_kernel_suite_with_progress(
    device: &Device,
    mut progress: impl FnMut(&str),
) -> Result<(), SuiteError> {
    // Bespoke cases report once per case. Registry cases below pass this
    // progress callback into the assert builder, so every run/device/variant is
    // reported as a child result.
    macro_rules! run {
        ($name:expr, $fut:expr) => {{
            progress($name);
            $fut.await?;
        }};
    }

    run!("top_k_pairs", check_top_k(device));
    run!("large_top_k_pairs", check_large_top_k(device));
    run!("mirostat2_sampler", check_mirostat(device));
    run!(
        "q8_0_qwen_logits_qgemv_topk",
        check_q8_0_large_qgemv_and_topk(device)
    );
    run!("q4k_fused_sampler", check_q4k_fused_sampler(device));
    run!(
        "q4k_non_block_aligned_k_qmatmul",
        check_q4k_non_block_aligned_k_qmatmul(device)
    );
    run!(
        "q4k_non_block_aligned_k_paired_silu",
        check_q4k_non_block_aligned_k_paired_silu(device)
    );

    // The full native conformance suite, driven from the shared registry. Each
    // case runs across `available_devices()` (CPU baseline + the GPU variant
    // matrix), so every tensor op is already exercised on the no-subgroup +
    // poisoned-pool variant the browser hits. The cases above are the only ones
    // the registry cannot express: samplers (non-tensor output) and
    // non-block-aligned-K quantized matmuls (rejected by the CPU baseline).
    for case in crate::suite::registry::assertions() {
        let name = case.name().to_string();
        if skip_browser_registry_case(&name) {
            continue;
        }
        let mut current_name = name.clone();
        progress(&name);
        {
            let mut case_progress = |variant: &str| {
                current_name = variant.to_string();
                progress(variant);
            };
            case.run_with_progress(&mut case_progress)
                .await
                .map_err(|err| SuiteError::Case {
                    case: current_name,
                    message: err.to_string(),
                })?;
        }
    }
    Ok(())
}

fn skip_browser_registry_case(name: &str) -> bool {
    name.starts_with("flash_attention_ops::flash_attention_decode_tiled_matches_cpu_reference::")
        || name.starts_with(
            "flash_attention_ops::flash_attention_decode_tiled_with_transposed_q_matches_cpu_reference::",
        )
        || name == "flash_attention_ops::flash_attention_subgroup_fallback_preserves_gpu_backend"
}

// ---------------------------------------------------------------------------
// Shared input builders.
// ---------------------------------------------------------------------------

fn deterministic_values(len: usize, seed: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let bucket = (index
                .wrapping_mul(37)
                .wrapping_add(seed.wrapping_mul(17))
                .wrapping_add(11))
                % 211;
            (bucket as f32 - 105.0) * scale
        })
        .collect()
}

fn perturb_q4k_payload(bytes: &mut [u8], seed: u8) {
    let block_size = std::mem::size_of::<fusor::BlockQ4K>();
    for (block_index, block) in bytes.chunks_exact_mut(block_size).enumerate() {
        // Leave the f16 block scales intact; alter only packed scale/weight payload.
        for (offset, byte) in block[4..].iter_mut().enumerate() {
            if (offset + block_index) % 17 == 0 {
                *byte = byte.wrapping_add(seed.wrapping_add((offset % 11) as u8));
            }
        }
    }
}

/// Dequantize raw GGUF bytes to a flat f32 row-major buffer. Independent of the
/// CPU tensor backend (which rejects non-block-aligned inner dims), so it is the
/// reference for the `non_block_aligned_k` cases.
fn dequantized_flat<B>(bytes: &[u8], element_count: usize) -> Vec<f32>
where
    B: GgufBlock,
    B::Dequantized: AsRef<[f32]>,
{
    let blocks: &[B] = bytemuck::cast_slice(bytes);
    let mut values = Vec::with_capacity(blocks.len() * B::BLOCK_SIZE);
    for block in blocks {
        values.extend_from_slice(block.dequantize().as_ref());
    }
    values.truncate(element_count);
    values
}

fn q_mat_mul_reference_3d(
    input: &[f32],
    input_shape: [usize; 3],
    weights: &[f32],
    weight_shape: [usize; 2],
) -> Vec<f32> {
    let rows = input_shape[0] * input_shape[1];
    let input_width = input_shape[2];
    let output_width = weight_shape[0];
    let mut output = vec![0.0; rows * output_width];
    for row in 0..rows {
        let input_row = &input[row * input_width..(row + 1) * input_width];
        for output_col in 0..output_width {
            let weight_row = &weights[output_col * input_width..(output_col + 1) * input_width];
            output[row * output_width + output_col] = input_row
                .iter()
                .zip(weight_row)
                .map(|(left, right)| left * right)
                .sum();
        }
    }
    output
}

fn paired_silu_reference_3d(
    input: &[f32],
    input_shape: [usize; 3],
    gate_weights: &[f32],
    up_weights: &[f32],
    weight_shape: [usize; 2],
) -> Vec<f32> {
    let gate = q_mat_mul_reference_3d(input, input_shape, gate_weights, weight_shape);
    let up = q_mat_mul_reference_3d(input, input_shape, up_weights, weight_shape);
    gate.into_iter()
        .zip(up)
        .map(|(gate, up)| silu(gate) * up)
        .collect()
}

fn assert_top_k(
    case: &'static str,
    actual: Vec<(u32, f32)>,
    values: &[f32],
    k: usize,
) -> Result<(), SuiteError> {
    let mut expected = values
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .collect::<Vec<_>>();
    expected.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| right.0.cmp(&left.0))
    });
    expected.truncate(k);
    let actual = actual
        .into_iter()
        .map(|(id, value)| (id as usize, value))
        .collect::<Vec<_>>();
    if actual != expected {
        return Err(SuiteError::case(
            case,
            format!(
                "top-k mismatch first_actual={:?} first_expected={:?}",
                actual.get(0..8),
                expected.get(0..8)
            ),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Sampler / top-k cases (non-tensor output → manual variant loop).
// ---------------------------------------------------------------------------

async fn check_top_k(device: &Device) -> Result<(), SuiteError> {
    let case = "top_k_pairs";
    let values = [
        0.25,
        f32::NAN,
        7.0,
        -3.0,
        f32::INFINITY,
        2.5,
        9.0,
        f32::NEG_INFINITY,
        8.5,
        9.0,
        6.0,
        -1.0,
    ];
    for run_device in device_test_variants(device) {
        let tensor = Tensor::new(&run_device, values.as_slice());
        let top = tensor
            .top_k_pairs(5)
            .await
            .map_err(|err| SuiteError::case(case, err))?;
        assert_top_k(case, top, &values, 5)?;
    }
    Ok(())
}

async fn check_large_top_k(device: &Device) -> Result<(), SuiteError> {
    let case = "large_top_k_pairs";
    let values = (0..65_537usize)
        .map(|index| {
            let base = ((index * 67 + 29) % 10_007) as f32 * 0.001;
            let bump = if index % 4099 == 0 { 20.0 } else { 0.0 };
            base + bump - (index % 13) as f32 * 0.0001
        })
        .collect::<Vec<_>>();
    for run_device in device_test_variants(device) {
        let tensor = Tensor::new(&run_device, values.as_slice());
        let top = tensor
            .top_k_pairs(64)
            .await
            .map_err(|err| SuiteError::case(case, err))?;
        assert_top_k(case, top, &values, 64)?;
    }
    Ok(())
}

async fn check_mirostat(device: &Device) -> Result<(), SuiteError> {
    let case = "mirostat2_sampler";
    let values = [9.0, 8.5, 7.0, 6.0, 2.5, 0.25, -1.0, -3.0];
    for run_device in device_test_variants(device) {
        let tensor = Tensor::new(&run_device, values.as_slice());
        let gpu = tensor
            .as_gpu()
            .ok_or_else(|| SuiteError::case(case, "tensor was not on GPU"))?;
        let gpu_device = gpu.device().clone();
        let mut sampler = Mirostat2Sampler::new(&gpu_device, 10.0);
        let token = tensor
            .sample_mirostat2_token(
                &mut sampler,
                &[],
                Mirostat2SamplerParams {
                    top_k: 4,
                    temperature: 1.0,
                    repetition_penalty: 1.0,
                    tau: 5.0,
                    eta: 0.1,
                    random: 0.0,
                },
            )
            .await
            .map_err(|err| SuiteError::case(case, err))?;
        if token != 0 {
            return Err(SuiteError::case(case, format!("token={token} expected=0")));
        }
    }
    Ok(())
}

async fn check_q4k_fused_sampler(device: &Device) -> Result<(), SuiteError> {
    let case = "q4k_fused_sampler";
    let weight_shape = [512usize, 256usize];
    let raw_bytes = q4k_raw_bytes(weight_shape);
    let input_values = deterministic_values(weight_shape[1], 23, 0.003);
    for run_device in device_test_variants(device) {
        let matrix = qmatrix_from_raw_bytes(&run_device, weight_shape, &raw_bytes, GgmlType::Q4K);
        let hidden: Tensor<1, f32> =
            Tensor::from_slice(&run_device, [weight_shape[1]], &input_values);
        let gpu = hidden
            .as_gpu()
            .ok_or_else(|| SuiteError::case(case, "hidden tensor was not on GPU"))?;
        let gpu_device = gpu.device().clone();
        let mut sampler = Mirostat2Sampler::new(&gpu_device, 10.0);
        let token = hidden
            .try_sample_mirostat2_token_q_mat(
                &matrix,
                &mut sampler,
                &[],
                Mirostat2SamplerParams {
                    top_k: 16,
                    temperature: 0.8,
                    repetition_penalty: 1.3,
                    tau: 5.0,
                    eta: 0.1,
                    random: 0.0,
                },
            )
            .await
            .map_err(|err| SuiteError::case(case, err))?
            .ok_or_else(|| SuiteError::case(case, "fused sampler returned None"))?;
        if token >= weight_shape[0] as u32 {
            return Err(SuiteError::case(
                case,
                format!("token out of range: {token}"),
            ));
        }
    }
    Ok(())
}

async fn check_q8_0_large_qgemv_and_topk(device: &Device) -> Result<(), SuiteError> {
    let case = "q8_0_qwen_logits_qgemv_topk";
    let weight_shape = [16_384usize, 896usize];
    let raw_bytes = q8_0_raw_bytes(weight_shape);
    let input_values = deterministic_values(weight_shape[1], 7, 0.0015);

    // CPU reference logits for the qgemv, computed once.
    let cpu_matrix = qmatrix_from_raw_bytes(&Device::Cpu, weight_shape, &raw_bytes, GgmlType::Q8_0);
    let cpu_input: Tensor<2, f32> =
        Tensor::from_slice(&Device::Cpu, [1, weight_shape[1]], &input_values);
    let cpu_logits = cpu_input.q_mat_mul(&cpu_matrix);
    let cpu_logits_flat = cpu_logits
        .reshape([weight_shape[0]])
        .as_slice()
        .await
        .map_err(|err| SuiteError::case(case, err))?
        .as_slice()
        .to_vec();

    for (variant_idx, run_device) in device_test_variants(device).into_iter().enumerate() {
        let variant = match variant_idx {
            0 => "subgroups_cold_pool",
            1 => "no_subgroups_cold_pool",
            2 => "subgroups_poisoned_pool",
            3 => "no_subgroups_poisoned_pool",
            _ => "unknown",
        };
        let matrix = qmatrix_from_raw_bytes(&run_device, weight_shape, &raw_bytes, GgmlType::Q8_0);
        let input: Tensor<2, f32> =
            Tensor::from_slice(&run_device, [1, weight_shape[1]], &input_values);
        let logits = input.q_mat_mul(&matrix);
        let logits_tensor: Tensor<1, f32> = logits.reshape([weight_shape[0]]).to_concrete();
        let gpu_logits = logits_tensor
            .as_slice()
            .await
            .map_err(|err| SuiteError::case(case, err))?
            .as_slice()
            .to_vec();

        // qgemv output must match the CPU kernel at a spread of columns.
        for &col in &[
            0usize, 1, 31, 32, 63, 64, 511, 1023, 4095, 8191, 12_345, 16_383,
        ] {
            let tolerance = 0.5f32.max(cpu_logits_flat[col].abs() * 5.0e-4);
            if (gpu_logits[col] - cpu_logits_flat[col]).abs() > tolerance {
                return Err(SuiteError::case(
                    case,
                    format!(
                        "{variant}: qgemv col={col} actual={} expected={}",
                        gpu_logits[col], cpu_logits_flat[col]
                    ),
                ));
            }
        }

        let top = logits_tensor
            .top_k_pairs(32)
            .await
            .map_err(|err| SuiteError::case(case, err))?;
        assert_top_k(case, top, &gpu_logits, 32)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Differential tensor cases (assert builder).
// ---------------------------------------------------------------------------

// The CPU quantized backend rejects a non-block-aligned inner dimension
// (`K = 896` is not a multiple of Q4K's 256-element block), so these two cases
// cannot use a CPU baseline. They compare every GPU variant against an
// independent host dequantize+matmul reference instead.
async fn check_q4k_non_block_aligned_k_qmatmul(device: &Device) -> Result<(), SuiteError> {
    let case = "q4k_non_block_aligned_k_qmatmul";
    let hidden = 896usize;
    let output = 512usize;
    let weight_shape = [output, hidden];
    let raw_bytes = q4k_raw_bytes(weight_shape);
    let weight_values = dequantized_flat::<BlockQ4K>(&raw_bytes, output * hidden);
    let input_shape = [2usize, 3usize, hidden];
    let input_values = deterministic_values(input_shape.iter().product(), 101, 0.0015);
    let expected_values =
        q_mat_mul_reference_3d(&input_values, input_shape, &weight_values, weight_shape);
    let output_shape = [input_shape[0], input_shape[1], output];

    for run_device in device_test_variants(device) {
        let input: Tensor<3, f32> = Tensor::from_slice(&run_device, input_shape, &input_values);
        let matrix = qmatrix_from_raw_bytes(&run_device, weight_shape, &raw_bytes, GgmlType::Q4K);
        let actual = input.q_mat_mul(&matrix);
        let expected: Tensor<3, f32> =
            Tensor::from_slice(&run_device, output_shape, &expected_values);
        approx_eq(&actual, &expected, 5.0)
            .await
            .map_err(|err| SuiteError::case(case, err))?;
    }
    Ok(())
}

async fn check_q4k_non_block_aligned_k_paired_silu(device: &Device) -> Result<(), SuiteError> {
    let case = "q4k_non_block_aligned_k_paired_silu";
    let hidden = 896usize;
    let intermediate = 512usize;
    let gate_bytes = q4k_raw_bytes([intermediate, hidden]);
    let mut up_bytes = q4k_raw_bytes([intermediate, hidden]);
    perturb_q4k_payload(&mut up_bytes, 17);
    let gate_weights = dequantized_flat::<BlockQ4K>(&gate_bytes, intermediate * hidden);
    let up_weights = dequantized_flat::<BlockQ4K>(&up_bytes, intermediate * hidden);
    let input_shape = [1usize, 2usize, hidden];
    let input_values = deterministic_values(input_shape.iter().product(), 107, 0.0018);
    let expected_values = paired_silu_reference_3d(
        &input_values,
        input_shape,
        &gate_weights,
        &up_weights,
        [intermediate, hidden],
    );
    let output_shape = [input_shape[0], input_shape[1], intermediate];

    for run_device in device_test_variants(device) {
        let input: Tensor<3, f32> = Tensor::from_slice(&run_device, input_shape, &input_values);
        let gate = qmatrix_from_raw_bytes(
            &run_device,
            [intermediate, hidden],
            &gate_bytes,
            GgmlType::Q4K,
        );
        let up = qmatrix_from_raw_bytes(
            &run_device,
            [intermediate, hidden],
            &up_bytes,
            GgmlType::Q4K,
        );
        let gate_up = QMatrix::concat_rows(&[&gate, &up])
            .ok_or_else(|| SuiteError::case(case, "Q4K concat_rows returned None"))?;
        let actual = input.q_mat_mul_paired_silu_product(&gate_up);
        let expected: Tensor<3, f32> =
            Tensor::from_slice(&run_device, output_shape, &expected_values);
        approx_eq(&actual, &expected, 5.0)
            .await
            .map_err(|err| SuiteError::case(case, err))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{run_webgpu_kernel_suite, run_webgpu_kernel_suite_with_progress};
    use crate::available_devices;
    use fusor::Device;

    /// Runs the in-browser WebGPU conformance suite natively against the GPU
    /// device — the same `run_webgpu_kernel_suite` the kalosm-chat web app runs
    /// on its `/conformance` route — so the cases can be iterated without a
    /// browser. The suite internally expands each case across the
    /// {subgroups, no subgroups} × {cold pool, poisoned pool} device matrix, so
    /// the no-subgroup kernel fallbacks the web build takes are covered natively.
    ///
    /// To approximate the browser quantized storage layout on a native GPU, set
    /// `FUSOR_Q_NATIVE=0` to force the `GpuF32Scales` layout (the web build
    /// disables `SHADER_F16`, so it never uses the native f16-scale layout).
    #[allow(clippy::await_holding_lock)]
    #[tokio::test]
    async fn webgpu_kernel_suite_runs_on_gpu() {
        let _gpu_guard = crate::suite::registry::gpu_test_guard();
        let mut ran_on_gpu = false;
        for device in available_devices().await {
            if let Device::Gpu(_) = device {
                ran_on_gpu = true;
                if std::env::var_os("FUSOR_CONFORMANCE_PROGRESS").is_some() {
                    run_webgpu_kernel_suite_with_progress(&device, |case| {
                        tracing::info!("webgpu_conformance {case}");
                    })
                    .await
                    .expect("webgpu kernel suite should pass on the GPU device");
                } else {
                    run_webgpu_kernel_suite(&device)
                        .await
                        .expect("webgpu kernel suite should pass on the GPU device");
                }
            }
        }
        assert!(
            ran_on_gpu,
            "no GPU device was available to run the webgpu kernel suite"
        );
    }
}
