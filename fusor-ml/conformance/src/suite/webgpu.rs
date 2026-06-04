use std::{fmt::Display, future::Future, mem::size_of};

use fusor::cache::AttentionMask;
use fusor::{
    BlockQ4K, BlockQ5_0, BlockQ8_0, Device, GgmlType, GgufBlock, MaskKind, Mirostat2Sampler,
    Mirostat2SamplerParams, QMatrix, Tensor,
};
use thiserror::Error;

use crate::{
    approx_eq, approx_or_relative_eq,
    common::{
        quantized::{
            deterministic_input, q4k_raw_bytes, q5_0_raw_bytes, q6k_raw_bytes, q8_0_raw_bytes,
            qmatrix_from_raw_bytes,
        },
        reshape3, reshape4, rms_norm_last_dim_3d, rope_interleaved_4d, rope_normal_4d,
    },
};

#[derive(Debug, Error)]
pub enum SuiteError {
    #[error("{case}: {message}")]
    Case { case: &'static str, message: String },
}

impl SuiteError {
    fn case(case: &'static str, message: impl Display) -> Self {
        Self::Case {
            case,
            message: message.to_string(),
        }
    }
}

#[derive(Clone, Copy)]
struct FlashCase {
    batch: usize,
    num_heads: usize,
    num_kv_heads: usize,
    q_seq_len: usize,
    kv_seq_len: usize,
    head_dim: usize,
}

pub async fn run_webgpu_kernel_suite(device: &Device) -> Result<(), SuiteError> {
    run_webgpu_kernel_suite_with_progress(device, |_| {}).await
}

pub async fn run_webgpu_kernel_suite_with_progress(
    device: &Device,
    mut progress: impl FnMut(&str),
) -> Result<(), SuiteError> {
    // Every case is run across the cross product of two device properties:
    //   * subgroups available vs a `without_subgroups()` sibling device. The web
    //     build never requests `Features::SUBGROUP`, so the no-subgroup fallback
    //     kernels are the only ones the browser ever runs. (When the device
    //     already lacks subgroups — i.e. in the browser — the sibling is skipped
    //     as redundant.)
    //   * a cold buffer pool vs a `with_poisoned_allocations()` sibling, whose
    //     kernel-output buffers are pre-filled with poison before use. This
    //     reproduces the app's reused buffer pool, surfacing kernels that rely on
    //     zero-initialized storage.
    // Both are properties of the constructed device / its allocations, not global
    // state. Variants are reported with `_nosubgroups` / `_dirty` suffixes.
    let has_subgroups = device.as_gpu().is_some_and(|gpu| gpu.subgroups_supported());

    let mut subgroup_variants: Vec<(&str, Device)> = vec![("", device.clone())];
    if has_subgroups {
        subgroup_variants.push(("_nosubgroups", device.without_subgroups()));
    }

    for (subgroup_label, subgroup_device) in subgroup_variants {
        let poisoned = subgroup_device.with_poisoned_allocations();
        for (dirty_label, run_device) in [("", &subgroup_device), ("_dirty", &poisoned)] {
            let suffix = format!("{subgroup_label}{dirty_label}");
            run_kernel_cases(run_device, |name: &str| {
                if suffix.is_empty() {
                    progress(name);
                } else {
                    progress(&format!("{name}{suffix}"));
                }
            })
            .await?;
        }
    }
    Ok(())
}

async fn run_kernel_cases(
    device: &Device,
    mut progress: impl FnMut(&str),
) -> Result<(), SuiteError> {
    run_case(&mut progress, "top_k_pairs", check_top_k(device)).await?;
    run_case(
        &mut progress,
        "large_top_k_pairs",
        check_large_top_k(device),
    )
    .await?;
    run_case(&mut progress, "mirostat2_sampler", check_mirostat(device)).await?;
    run_case(
        &mut progress,
        "q8_0_qwen_logits_qgemv_topk",
        check_q8_0_large_qgemv_and_topk(device),
    )
    .await?;
    run_case(
        &mut progress,
        "q8_0_multirow_prefill",
        check_q8_0_multirow_prefill(device),
    )
    .await?;
    run_case(
        &mut progress,
        "q4k_multirow_decode_shape",
        check_q4k_multirow_decode_shape(device),
    )
    .await?;
    run_case(
        &mut progress,
        "q4k_fused_sampler",
        check_q4k_fused_sampler(device),
    )
    .await?;
    run_case(
        &mut progress,
        "ffn_q4k_q6k_chain",
        check_q4k_q6k_ffn_chain(device),
    )
    .await?;
    run_case(
        &mut progress,
        "ffn_q4k_concat_paired_silu",
        check_q4k_concat_paired_silu(device),
    )
    .await?;
    run_case(
        &mut progress,
        "q6k_q_mat_mul_add2_residual_3d",
        check_q6k_q_mat_mul_add2_residual_3d(device),
    )
    .await?;
    run_case(
        &mut progress,
        "decode_ffn_logits_chain",
        check_decode_ffn_logits_chain(device),
    )
    .await?;
    run_case(
        &mut progress,
        "rms_norm_hidden_896",
        check_rms_norm_hidden_896(device),
    )
    .await?;
    run_case(
        &mut progress,
        "rms_norm_decode_single_row",
        check_rms_norm_decode_single_row(device),
    )
    .await?;
    run_case(&mut progress, "rope_hidden_64", check_rope(device)).await?;
    run_case(
        &mut progress,
        "flash_attention_qwen_decode",
        check_flash_attention_qwen_decode(device),
    )
    .await?;
    run_case(
        &mut progress,
        "flash_attention_qwen_prefill_causal",
        check_flash_attention_qwen_prefill_causal(device),
    )
    .await?;
    run_case(
        &mut progress,
        "flash_attention_qwen_prefill_offset_qk_mask",
        check_flash_attention_qwen_prefill_offset_qk_mask(device),
    )
    .await?;
    run_case(
        &mut progress,
        "q4k_non_block_aligned_k_qmatmul",
        check_q4k_non_block_aligned_k_qmatmul(device),
    )
    .await?;
    run_case(
        &mut progress,
        "q4k_non_block_aligned_k_paired_silu",
        check_q4k_non_block_aligned_k_paired_silu(device),
    )
    .await?;
    run_case(
        &mut progress,
        "q5_0_attn_proj_qgemv_decode",
        check_q5_0_attn_proj_qgemv_decode(device),
    )
    .await?;
    run_case(
        &mut progress,
        "q5_0_ffn_gate_qgemv_decode",
        check_q5_0_ffn_gate_qgemv_decode(device),
    )
    .await?;
    run_case(
        &mut progress,
        "q5_0_kv_proj_qgemv_decode",
        check_q5_0_kv_proj_qgemv_decode(device),
    )
    .await?;
    run_case(
        &mut progress,
        "q5_0_attn_proj_prefill_multirow",
        check_q5_0_attn_proj_prefill_multirow(device),
    )
    .await?;
    Ok(())
}

async fn run_case<Fut>(
    progress: &mut impl FnMut(&str),
    name: &str,
    fut: Fut,
) -> Result<(), SuiteError>
where
    Fut: Future<Output = Result<(), SuiteError>>,
{
    progress(name);
    fut.await
}

fn attention_data(len: usize, offset: f32) -> Vec<f32> {
    (0..len)
        .map(|i| (((i % 17) as f32) - 8.0) * 0.12 + offset)
        .collect()
}

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

fn offset_causal_mask(q_seq_len: usize, kv_seq_len: usize, offset: usize) -> Vec<f32> {
    let mut mask = vec![0.0; q_seq_len * kv_seq_len];
    for query in 0..q_seq_len {
        let max_key = offset + query;
        for key in (max_key + 1)..kv_seq_len {
            mask[query * kv_seq_len + key] = f32::NEG_INFINITY;
        }
    }
    mask
}

fn transpose_bqhd_to_bhqd(
    data: &[f32],
    batch: usize,
    q_seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut transposed = vec![0.0; data.len()];
    for b in 0..batch {
        for q in 0..q_seq_len {
            for h in 0..num_heads {
                for d in 0..head_dim {
                    let source = (((b * q_seq_len + q) * num_heads + h) * head_dim) + d;
                    let target = (((b * num_heads + h) * q_seq_len + q) * head_dim) + d;
                    transposed[target] = data[source];
                }
            }
        }
    }
    transposed
}

fn perturb_q4k_payload(bytes: &mut [u8], seed: u8) {
    let block_size = size_of::<BlockQ4K>();
    for (block_index, block) in bytes.chunks_exact_mut(block_size).enumerate() {
        // Leave the f16 block scales intact; alter only packed scale/weight payload.
        for (offset, byte) in block[4..].iter_mut().enumerate() {
            if (offset + block_index) % 17 == 0 {
                *byte = byte.wrapping_add(seed.wrapping_add((offset % 11) as u8));
            }
        }
    }
}

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
    assert_eq!(input_width, weight_shape[1]);
    assert_eq!(input.len(), rows * input_width);
    assert_eq!(weights.len(), output_width * input_width);

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

fn silu_scalar(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
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
        .map(|(gate, up)| silu_scalar(gate) * up)
        .collect()
}

fn assert_close(
    case: &'static str,
    label: impl Display,
    actual: f32,
    expected: f32,
    tolerance_floor: f32,
) -> Result<(), SuiteError> {
    let tolerance = tolerance_floor.max(expected.abs() * 5.0e-4);
    if (actual - expected).abs() > tolerance {
        return Err(SuiteError::case(
            case,
            format!("{label} actual={actual} expected={expected} tol={tolerance}"),
        ));
    }
    Ok(())
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
    let tensor = Tensor::new(device, values.as_slice());
    let top = tensor
        .top_k_pairs(5)
        .await
        .map_err(|err| SuiteError::case(case, err))?;
    assert_top_k(case, top, &values, 5)
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
    let tensor = Tensor::new(device, values.as_slice());
    let top = tensor
        .top_k_pairs(64)
        .await
        .map_err(|err| SuiteError::case(case, err))?;
    assert_top_k(case, top, &values, 64)
}

async fn check_mirostat(device: &Device) -> Result<(), SuiteError> {
    let case = "mirostat2_sampler";
    let values = [9.0, 8.5, 7.0, 6.0, 2.5, 0.25, -1.0, -3.0];
    let tensor = Tensor::new(device, values.as_slice());
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
    Ok(())
}

async fn check_q8_0_qgemv_shape(
    device: &Device,
    weight_shape: [usize; 2],
    sample_cols: &[usize],
) -> Result<Vec<f32>, SuiteError> {
    let case = "q8_0_qgemv";
    let blocks_per_row = weight_shape[1] / BlockQ8_0::BLOCK_SIZE;
    let raw_bytes = q8_0_raw_bytes(weight_shape);
    let matrix = qmatrix_from_raw_bytes(device, weight_shape, &raw_bytes, GgmlType::Q8_0);
    let blocks: &[BlockQ8_0] = bytemuck::cast_slice(&raw_bytes);
    let input_values = deterministic_values(weight_shape[1], 7, 0.0015);
    let input: Tensor<2, f32> = Tensor::from_slice(device, [1, weight_shape[1]], &input_values);
    let result = input
        .q_mat_mul(&matrix)
        .as_slice()
        .await
        .map_err(|err| SuiteError::case(case, err))?;

    for &col in sample_cols {
        let expected = (0..blocks_per_row)
            .map(|block_col| {
                let block = &blocks[col * blocks_per_row + block_col];
                let weights = block.dequantize();
                weights
                    .as_ref()
                    .iter()
                    .enumerate()
                    .map(|(offset, weight)| {
                        input_values[block_col * BlockQ8_0::BLOCK_SIZE + offset] * *weight
                    })
                    .sum::<f32>()
            })
            .sum::<f32>();
        let actual = result[[0, col]];
        assert_close(
            case,
            format!("shape={weight_shape:?} col={col}"),
            actual,
            expected,
            1e-2,
        )?;
    }

    Ok(result.as_slice().to_vec())
}

async fn check_q8_0_large_qgemv_and_topk(device: &Device) -> Result<(), SuiteError> {
    let case = "q8_0_qwen_logits_qgemv_topk";
    let weight_shape = [16_384usize, 896usize];
    let sample_cols = [
        0usize, 1, 31, 32, 63, 64, 511, 1023, 4095, 8191, 12_345, 16_383,
    ];
    let logits = check_q8_0_qgemv_shape(device, weight_shape, &sample_cols).await?;
    let tensor = Tensor::from_slice(device, [weight_shape[0]], &logits);
    let top = tensor
        .top_k_pairs(32)
        .await
        .map_err(|err| SuiteError::case(case, err))?;
    assert_top_k(case, top, &logits, 32)
}

async fn check_q8_0_multirow_prefill(device: &Device) -> Result<(), SuiteError> {
    let case = "q8_0_multirow_prefill";
    let weight_shape = [4096usize, 896usize];
    let rows = 48usize;
    let blocks_per_row = weight_shape[1] / BlockQ8_0::BLOCK_SIZE;
    let raw_bytes = q8_0_raw_bytes(weight_shape);
    let matrix = qmatrix_from_raw_bytes(device, weight_shape, &raw_bytes, GgmlType::Q8_0);
    let blocks: &[BlockQ8_0] = bytemuck::cast_slice(&raw_bytes);
    let input_values = deterministic_values(rows * weight_shape[1], 19, 0.001);
    let input: Tensor<2, f32> = Tensor::from_slice(device, [rows, weight_shape[1]], &input_values);
    let result = input
        .q_mat_mul(&matrix)
        .as_slice()
        .await
        .map_err(|err| SuiteError::case(case, err))?;

    for row in [0usize, 1, 7, 17, 31, 47] {
        for col in [0usize, 1, 63, 64, 511, 1023, 2047, 4095] {
            let expected = (0..blocks_per_row)
                .map(|block_col| {
                    let block = &blocks[col * blocks_per_row + block_col];
                    let weights = block.dequantize();
                    weights
                        .as_ref()
                        .iter()
                        .enumerate()
                        .map(|(offset, weight)| {
                            let input_index =
                                row * weight_shape[1] + block_col * BlockQ8_0::BLOCK_SIZE + offset;
                            input_values[input_index] * *weight
                        })
                        .sum::<f32>()
                })
                .sum::<f32>();
            assert_close(
                case,
                format!("row={row} col={col}"),
                result[[row, col]],
                expected,
                1e-2,
            )?;
        }
    }
    Ok(())
}

async fn check_q4k_multirow_decode_shape(device: &Device) -> Result<(), SuiteError> {
    let case = "q4k_multirow_decode_shape";
    let weight_shape = [5120usize, 4096usize];
    let input_shape = [1usize, 32usize, 48usize, 128usize];
    let hidden = input_shape[1] * input_shape[3];
    let selected_k = 777usize;
    let selected_head = selected_k / input_shape[3];
    let selected_dim = selected_k % input_shape[3];
    let selected_block_in_row = selected_k / BlockQ4K::BLOCK_SIZE;
    let selected_offset = selected_k % BlockQ4K::BLOCK_SIZE;
    let blocks_per_row = hidden / BlockQ4K::BLOCK_SIZE;
    let raw_bytes = q4k_raw_bytes(weight_shape);
    let matrix = qmatrix_from_raw_bytes(device, weight_shape, &raw_bytes, GgmlType::Q4K);

    let mut input_data = vec![-0.25f32; input_shape.iter().product()];
    let mut row_values = Vec::with_capacity(input_shape[2]);
    for row in 0..input_shape[2] {
        let row_value = 0.125 + row as f32 * 0.01;
        row_values.push(row_value);
        let index = ((selected_head * input_shape[2] + row) * input_shape[3]) + selected_dim;
        input_data[index] = row_value - 0.25;
    }

    let input: Tensor<4, f32> = Tensor::from_slice(device, input_shape, &input_data);
    let actual = (input + 0.25)
        .transpose(1, 2)
        .reshape([1, input_shape[2], hidden])
        .q_mat_mul(&matrix)
        .as_slice()
        .await
        .map_err(|err| SuiteError::case(case, err))?;

    for row in [0usize, 1, 7, 17, 31, 47] {
        for col in [0usize, 1, 63, 64, 511, 1024, 4095, 5119] {
            let block_index = col * blocks_per_row + selected_block_in_row;
            let block = &bytemuck::cast_slice::<_, BlockQ4K>(&raw_bytes)[block_index];
            let expected = row_values[row] * block.dequantize().as_ref()[selected_offset];
            assert_close(
                case,
                format!("row={row} col={col}"),
                actual[[0, row, col]],
                expected,
                1e-2,
            )?;
        }
    }
    Ok(())
}

async fn check_q4k_fused_sampler(device: &Device) -> Result<(), SuiteError> {
    let case = "q4k_fused_sampler";
    let weight_shape = [512usize, 256usize];
    let raw_bytes = q4k_raw_bytes(weight_shape);
    let matrix = qmatrix_from_raw_bytes(device, weight_shape, &raw_bytes, GgmlType::Q4K);
    let input_values = deterministic_values(weight_shape[1], 23, 0.003);
    let hidden: Tensor<1, f32> = Tensor::from_slice(device, [weight_shape[1]], &input_values);
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
    Ok(())
}

async fn check_q4k_q6k_ffn_chain(device: &Device) -> Result<(), SuiteError> {
    let case = "ffn_q4k_q6k_chain";
    let hidden = 512usize;
    let intermediate = 512usize;
    let output = 128usize;
    let gate_bytes = q4k_raw_bytes([intermediate, hidden]);
    let up_bytes = q4k_raw_bytes([intermediate, hidden]);
    let down_bytes = q6k_raw_bytes([output, intermediate]);
    let input_data = deterministic_input(&[1, hidden], 834);

    let cpu_input: Tensor<2, f32> = Tensor::from_slice(&Device::Cpu, [1, hidden], &input_data);
    let cpu_gate = qmatrix_from_raw_bytes(
        &Device::Cpu,
        [intermediate, hidden],
        &gate_bytes,
        GgmlType::Q4K,
    );
    let cpu_up = qmatrix_from_raw_bytes(
        &Device::Cpu,
        [intermediate, hidden],
        &up_bytes,
        GgmlType::Q4K,
    );
    let cpu_down = qmatrix_from_raw_bytes(
        &Device::Cpu,
        [output, intermediate],
        &down_bytes,
        GgmlType::Q6K,
    );
    let expected = (cpu_input.q_mat_mul(&cpu_gate).silu() * cpu_input.q_mat_mul(&cpu_up))
        .q_mat_mul(&cpu_down)
        .to_concrete();

    let input: Tensor<2, f32> = Tensor::from_slice(device, [1, hidden], &input_data);
    let gate = qmatrix_from_raw_bytes(device, [intermediate, hidden], &gate_bytes, GgmlType::Q4K);
    let up = qmatrix_from_raw_bytes(device, [intermediate, hidden], &up_bytes, GgmlType::Q4K);
    let down = qmatrix_from_raw_bytes(device, [output, intermediate], &down_bytes, GgmlType::Q6K);
    let actual = (input.q_mat_mul(&gate).silu() * input.q_mat_mul(&up))
        .q_mat_mul(&down)
        .to_concrete();
    approx_eq(&actual, &expected, 5.0)
        .await
        .map_err(|err| SuiteError::case(case, err))
}

async fn check_q4k_concat_paired_silu(device: &Device) -> Result<(), SuiteError> {
    let case = "ffn_q4k_concat_paired_silu";
    let hidden = 1024usize;
    let intermediate = 512usize;
    let gate_bytes = q4k_raw_bytes([intermediate, hidden]);
    let mut up_bytes = q4k_raw_bytes([intermediate, hidden]);
    perturb_q4k_payload(&mut up_bytes, 3);
    let input_data = deterministic_values(hidden, 61, 0.0025);

    let gate_weights = dequantized_flat::<BlockQ4K>(&gate_bytes, intermediate * hidden);
    let up_weights = dequantized_flat::<BlockQ4K>(&up_bytes, intermediate * hidden);
    let expected_values = paired_silu_reference_3d(
        &input_data,
        [1, 1, hidden],
        &gate_weights,
        &up_weights,
        [intermediate, hidden],
    );
    let expected: Tensor<3, f32> =
        Tensor::from_slice(&Device::Cpu, [1, 1, intermediate], &expected_values);

    let input: Tensor<3, f32> = Tensor::from_slice(device, [1, 1, hidden], &input_data);
    let gate = qmatrix_from_raw_bytes(device, [intermediate, hidden], &gate_bytes, GgmlType::Q4K);
    let up = qmatrix_from_raw_bytes(device, [intermediate, hidden], &up_bytes, GgmlType::Q4K);
    let gate_up = QMatrix::concat_rows(&[&gate, &up])
        .ok_or_else(|| SuiteError::case(case, "Q4K concat_rows returned None"))?;
    if gate_up.shape() != [intermediate * 2, hidden] {
        return Err(SuiteError::case(
            case,
            format!(
                "gate_up shape {:?} expected {:?}",
                gate_up.shape(),
                [intermediate * 2, hidden]
            ),
        ));
    }

    let actual = input.q_mat_mul_paired_silu_product(&gate_up);
    approx_eq(&actual, &expected, 5.0)
        .await
        .map_err(|err| SuiteError::case(case, err))
}

async fn check_q6k_q_mat_mul_add2_residual_3d(device: &Device) -> Result<(), SuiteError> {
    let case = "q6k_q_mat_mul_add2_residual_3d";
    let input_width = 512usize;
    let output_width = 896usize;
    let down_bytes = q6k_raw_bytes([output_width, input_width]);
    let input_data = deterministic_values(input_width, 67, 0.002);
    let first_data = deterministic_values(output_width, 71, 0.003);
    let second_data = deterministic_values(output_width, 73, 0.0025);

    let cpu_input: Tensor<3, f32> =
        Tensor::from_slice(&Device::Cpu, [1, 1, input_width], &input_data);
    let cpu_first: Tensor<3, f32> =
        Tensor::from_slice(&Device::Cpu, [1, 1, output_width], &first_data);
    let cpu_second: Tensor<3, f32> =
        Tensor::from_slice(&Device::Cpu, [1, 1, output_width], &second_data);
    let cpu_down = qmatrix_from_raw_bytes(
        &Device::Cpu,
        [output_width, input_width],
        &down_bytes,
        GgmlType::Q6K,
    );
    let expected = cpu_input
        .q_mat_mul_add2(&cpu_down, &cpu_first, &cpu_second)
        .to_concrete();

    let input: Tensor<3, f32> = Tensor::from_slice(device, [1, 1, input_width], &input_data);
    let first: Tensor<3, f32> = Tensor::from_slice(device, [1, 1, output_width], &first_data);
    let second: Tensor<3, f32> = Tensor::from_slice(device, [1, 1, output_width], &second_data);
    let down = qmatrix_from_raw_bytes(
        device,
        [output_width, input_width],
        &down_bytes,
        GgmlType::Q6K,
    );
    let actual = input.q_mat_mul_add2(&down, &first, &second);
    approx_eq(&actual, &expected, 5.0)
        .await
        .map_err(|err| SuiteError::case(case, err))
}

async fn check_decode_ffn_logits_chain(device: &Device) -> Result<(), SuiteError> {
    let case = "decode_ffn_logits_chain";
    let hidden = 1024usize;
    let intermediate = 512usize;
    let vocab = 2048usize;
    let gate_bytes = q4k_raw_bytes([intermediate, hidden]);
    let mut up_bytes = q4k_raw_bytes([intermediate, hidden]);
    perturb_q4k_payload(&mut up_bytes, 9);
    let down_bytes = q6k_raw_bytes([hidden, intermediate]);
    let output_bytes = q8_0_raw_bytes([vocab, hidden]);
    let input_data = deterministic_values(hidden, 79, 0.002);
    let first_residual = deterministic_values(hidden, 83, 0.0025);
    let second_residual = deterministic_values(hidden, 89, 0.002);
    let norm_weight = deterministic_values(hidden, 97, 0.001)
        .into_iter()
        .map(|value| 1.0 + value)
        .collect::<Vec<_>>();

    let cpu_first: Tensor<3, f32> =
        Tensor::from_slice(&Device::Cpu, [1, 1, hidden], &first_residual);
    let cpu_second: Tensor<3, f32> =
        Tensor::from_slice(&Device::Cpu, [1, 1, hidden], &second_residual);
    let cpu_norm: Tensor<1, f32> = Tensor::from_slice(&Device::Cpu, [hidden], &norm_weight);
    let cpu_down = qmatrix_from_raw_bytes(
        &Device::Cpu,
        [hidden, intermediate],
        &down_bytes,
        GgmlType::Q6K,
    );
    let cpu_output =
        qmatrix_from_raw_bytes(&Device::Cpu, [vocab, hidden], &output_bytes, GgmlType::Q8_0);
    let gate_weights = dequantized_flat::<BlockQ4K>(&gate_bytes, intermediate * hidden);
    let up_weights = dequantized_flat::<BlockQ4K>(&up_bytes, intermediate * hidden);
    let expected_activation_values = paired_silu_reference_3d(
        &input_data,
        [1, 1, hidden],
        &gate_weights,
        &up_weights,
        [intermediate, hidden],
    );
    let expected_activation: Tensor<3, f32> = Tensor::from_slice(
        &Device::Cpu,
        [1, 1, intermediate],
        &expected_activation_values,
    );
    let expected_hidden = expected_activation
        .q_mat_mul_add2(&cpu_down, &cpu_first, &cpu_second)
        .rms_norm_fused::<1, 2>(&cpu_norm, None, 1e-5);
    let expected_logits = expected_hidden.q_mat_mul(&cpu_output).to_concrete();

    let input: Tensor<3, f32> = Tensor::from_slice(device, [1, 1, hidden], &input_data);
    let first: Tensor<3, f32> = Tensor::from_slice(device, [1, 1, hidden], &first_residual);
    let second: Tensor<3, f32> = Tensor::from_slice(device, [1, 1, hidden], &second_residual);
    let norm: Tensor<1, f32> = Tensor::from_slice(device, [hidden], &norm_weight);
    let gate = qmatrix_from_raw_bytes(device, [intermediate, hidden], &gate_bytes, GgmlType::Q4K);
    let up = qmatrix_from_raw_bytes(device, [intermediate, hidden], &up_bytes, GgmlType::Q4K);
    let gate_up = QMatrix::concat_rows(&[&gate, &up])
        .ok_or_else(|| SuiteError::case(case, "Q4K concat_rows returned None"))?;
    let down = qmatrix_from_raw_bytes(device, [hidden, intermediate], &down_bytes, GgmlType::Q6K);
    let output = qmatrix_from_raw_bytes(device, [vocab, hidden], &output_bytes, GgmlType::Q8_0);
    let actual_hidden = input
        .q_mat_mul_paired_silu_product(&gate_up)
        .q_mat_mul_add2(&down, &first, &second)
        .rms_norm_fused::<1, 2>(&norm, None, 1e-5);
    let actual_logits = actual_hidden.q_mat_mul(&output);

    approx_eq(&actual_logits, &expected_logits, 20.0)
        .await
        .map_err(|err| SuiteError::case(case, err))
}

async fn check_rms_norm_hidden_896(device: &Device) -> Result<(), SuiteError> {
    let case = "rms_norm_hidden_896";
    let shape = [1usize, 48usize, 896usize];
    let input_data = deterministic_values(shape.iter().product(), 31, 0.002);
    let weight_data = deterministic_values(shape[2], 37, 0.003)
        .into_iter()
        .map(|value| value + 1.0)
        .collect::<Vec<_>>();

    let input_host = reshape3(&input_data, shape);
    let expected_host = rms_norm_last_dim_3d(&input_host, &weight_data, 1e-5);
    let input: Tensor<3, f32> = Tensor::from_slice(device, shape, &input_data);
    let weight: Tensor<1, f32> = Tensor::from_slice(device, [shape[2]], &weight_data);
    let expected = Tensor::new(device, &expected_host);
    let actual = input.rms_norm_fused::<1, 2>(&weight, None, 1e-5);
    approx_eq(&actual, &expected, 1e-3)
        .await
        .map_err(|err| SuiteError::case(case, err))
}

/// Single-row (decode-shape) `rms_norm` repeated many times.
///
/// On a device without subgroups (the web build always disables them) the
/// `rms_norm` reduction falls back to the workgroup-shared-memory tree reduce.
/// The model invokes this exact `[1, 1, 896]` reduction once per layer per
/// decode step — hundreds of times per generation — and a flaky reduction
/// surfaces as intermittently corrupt logits (the "fails to generate
/// reasonable text" bug). A single comparison is not enough to catch a
/// non-deterministic reduction, so this case runs the reduction many times with
/// a warm buffer pool and a range of input magnitudes (including residual-stream
/// outliers) and fails if any iteration disagrees with the host reference.
async fn check_rms_norm_decode_single_row(device: &Device) -> Result<(), SuiteError> {
    let case = "rms_norm_decode_single_row";
    let hidden = 896usize;
    let shape = [1usize, 1usize, hidden];
    let weight_data = deterministic_values(hidden, 37, 0.003)
        .into_iter()
        .map(|value| value + 1.0)
        .collect::<Vec<_>>();
    let weight: Tensor<1, f32> = Tensor::from_slice(device, [hidden], &weight_data);

    for iter in 0..512usize {
        let mut input_data =
            deterministic_values(hidden, 31 + iter, 0.01 + (iter % 7) as f32 * 0.01);
        // Mimic the transformer residual stream's large outlier dimensions: a
        // few entries are much larger than the rest, which dominate the sum of
        // squares and make a missing reduction term obvious.
        input_data[iter % hidden] = 20.0 + (iter % 13) as f32;
        input_data[(iter * 7 + 3) % hidden] = -30.0 - (iter % 11) as f32;

        let input_host = reshape3(&input_data, shape);
        let expected_host = rms_norm_last_dim_3d(&input_host, &weight_data, 1e-5);
        let input: Tensor<3, f32> = Tensor::from_slice(device, shape, &input_data);
        let expected = Tensor::new(device, &expected_host);
        let actual = input.rms_norm_fused::<1, 2>(&weight, None, 1e-5);
        approx_eq(&actual, &expected, 2e-3)
            .await
            .map_err(|err| SuiteError::case(case, format!("iteration {iter}: {err}")))?;
    }
    Ok(())
}

async fn check_rope(device: &Device) -> Result<(), SuiteError> {
    let case = "rope_hidden_64";
    let shape = [1usize, 14usize, 3usize, 64usize];
    let input_data = deterministic_values(shape.iter().product(), 41, 0.01);
    let cos_data = (0..shape[2] * (shape[3] / 2))
        .map(|index| ((index as f32 + 1.0) * 0.017).cos())
        .collect::<Vec<_>>();
    let sin_data = (0..shape[2] * (shape[3] / 2))
        .map(|index| ((index as f32 + 1.0) * 0.017).sin())
        .collect::<Vec<_>>();
    let input_host = reshape4(&input_data, shape);
    let cos_host = crate::common::reshape2(&cos_data, [shape[2], shape[3] / 2]);
    let sin_host = crate::common::reshape2(&sin_data, [shape[2], shape[3] / 2]);

    let input: Tensor<4, f32> = Tensor::from_slice(device, shape, &input_data);
    let cos: Tensor<2, f32> = Tensor::from_slice(device, [shape[2], shape[3] / 2], &cos_data);
    let sin: Tensor<2, f32> = Tensor::from_slice(device, [shape[2], shape[3] / 2], &sin_data);

    let expected_normal = Tensor::new(device, &rope_normal_4d(&input_host, &cos_host, &sin_host));
    let actual_normal = input.rope_normal_fused(&cos, &sin);
    approx_eq(&actual_normal, &expected_normal, 1e-4)
        .await
        .map_err(|err| SuiteError::case(case, err))?;

    let expected_interleaved = Tensor::new(
        device,
        &rope_interleaved_4d(&input_host, &cos_host, &sin_host),
    );
    let actual_interleaved = input.rope_fused(&cos, &sin);
    approx_eq(&actual_interleaved, &expected_interleaved, 1e-4)
        .await
        .map_err(|err| SuiteError::case(case, err))
}

async fn check_flash_attention_qwen_decode(device: &Device) -> Result<(), SuiteError> {
    for (num_heads, num_kv_heads, kv_seq_len) in [(14, 2, 129), (14, 2, 257), (14, 2, 569)] {
        let case = FlashCase {
            batch: 1,
            num_heads,
            num_kv_heads,
            q_seq_len: 1,
            kv_seq_len,
            head_dim: 64,
        };
        assert_flash_attention_case(device, case, 1e-3, false).await?;
        assert_flash_attention_case(device, case, 1e-3, true).await?;
    }
    Ok(())
}

async fn check_flash_attention_qwen_prefill_causal(device: &Device) -> Result<(), SuiteError> {
    let suite_case = "flash_attention_qwen_prefill_causal";
    for seq_len in [17usize, 33] {
        let case = FlashCase {
            batch: 1,
            num_heads: 14,
            num_kv_heads: 2,
            q_seq_len: seq_len,
            kv_seq_len: seq_len,
            head_dim: 64,
        };
        let cpu_mask = AttentionMask::<f32>::causal(&Device::Cpu, seq_len);
        let mask = AttentionMask::<f32>::causal(device, seq_len);
        assert_flash_attention_masked_case(
            device,
            suite_case,
            case,
            cpu_mask.mask(),
            mask.mask(),
            MaskKind::Causal,
            2e-3,
            false,
        )
        .await?;
        assert_flash_attention_masked_case(
            device,
            suite_case,
            case,
            cpu_mask.mask(),
            mask.mask(),
            MaskKind::Causal,
            2e-3,
            true,
        )
        .await?;
    }
    Ok(())
}

async fn check_flash_attention_qwen_prefill_offset_qk_mask(
    device: &Device,
) -> Result<(), SuiteError> {
    let suite_case = "flash_attention_qwen_prefill_offset_qk_mask";
    let case = FlashCase {
        batch: 1,
        num_heads: 14,
        num_kv_heads: 2,
        q_seq_len: 13,
        kv_seq_len: 47,
        head_dim: 64,
    };
    let mask_data = offset_causal_mask(
        case.q_seq_len,
        case.kv_seq_len,
        case.kv_seq_len - case.q_seq_len,
    );
    let cpu_mask: Tensor<2, f32> =
        Tensor::from_slice(&Device::Cpu, [case.q_seq_len, case.kv_seq_len], &mask_data);
    let mask: Tensor<2, f32> =
        Tensor::from_slice(device, [case.q_seq_len, case.kv_seq_len], &mask_data);
    assert_flash_attention_masked_case(
        device,
        suite_case,
        case,
        &cpu_mask,
        &mask,
        MaskKind::QKMask,
        2e-3,
        true,
    )
    .await
}

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
    let expected: Tensor<3, f32> = Tensor::from_slice(
        &Device::Cpu,
        [input_shape[0], input_shape[1], output],
        &expected_values,
    );

    let input: Tensor<3, f32> = Tensor::from_slice(device, input_shape, &input_values);
    let matrix = QMatrix::from_raw_bytes(
        device,
        weight_shape.to_vec().into_boxed_slice(),
        &raw_bytes,
        GgmlType::Q4K,
    )
    .map_err(|err| SuiteError::case(case, err))?;
    let actual = input.q_mat_mul(&matrix);
    approx_eq(&actual, &expected, 5.0)
        .await
        .map_err(|err| SuiteError::case(case, err))
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
    let expected: Tensor<3, f32> = Tensor::from_slice(
        &Device::Cpu,
        [input_shape[0], input_shape[1], intermediate],
        &expected_values,
    );

    let input: Tensor<3, f32> = Tensor::from_slice(device, input_shape, &input_values);
    let gate = qmatrix_from_raw_bytes(device, [intermediate, hidden], &gate_bytes, GgmlType::Q4K);
    let up = qmatrix_from_raw_bytes(device, [intermediate, hidden], &up_bytes, GgmlType::Q4K);
    let gate_up = QMatrix::concat_rows(&[&gate, &up])
        .ok_or_else(|| SuiteError::case(case, "Q4K concat_rows returned None"))?;
    let actual = input.q_mat_mul_paired_silu_product(&gate_up);
    approx_eq(&actual, &expected, 5.0)
        .await
        .map_err(|err| SuiteError::case(case, err))
}

/// Qwen2.5-0.5B q4_k_m stores most attention/FFN weights (and the token
/// embedding) as `Q5_0`. The native build keeps the f16-scale `Q5_0Native`
/// storage layout, but the web build forces `SHADER_F16` off, so it always
/// uses the expanded f32-scale `Q5_0` layout. None of the other suite cases
/// exercise `Q5_0`, so this is the dominant real-model format that goes
/// completely unverified on WebGPU. These cases compare a `q_mat_mul` against a
/// host dequantize+matmul reference at the exact decode/prefill shapes the
/// model uses (hidden=896).
async fn check_q5_0_qmatmul_case(
    device: &Device,
    case: &'static str,
    weight_shape: [usize; 2],
    seq_len: usize,
    seed: usize,
) -> Result<(), SuiteError> {
    let hidden = weight_shape[1];
    let raw_bytes = q5_0_raw_bytes(weight_shape);
    let weight_values = dequantized_flat::<BlockQ5_0>(&raw_bytes, weight_shape[0] * hidden);
    let input_shape = [1usize, seq_len, hidden];
    let input_values = deterministic_values(input_shape.iter().product(), seed, 0.0015);
    let expected_values =
        q_mat_mul_reference_3d(&input_values, input_shape, &weight_values, weight_shape);
    let expected: Tensor<3, f32> = Tensor::from_slice(
        &Device::Cpu,
        [1, seq_len, weight_shape[0]],
        &expected_values,
    );

    let input: Tensor<3, f32> = Tensor::from_slice(device, input_shape, &input_values);
    let matrix = qmatrix_from_raw_bytes(device, weight_shape, &raw_bytes, GgmlType::Q5_0);
    let actual = input.q_mat_mul(&matrix);
    approx_or_relative_eq(&actual, &expected, 1.0e-2, 2.0e-2)
        .await
        .map_err(|err| SuiteError::case(case, err))
}

/// attn_q / attn_output: [896, 896] Q5_0, single decode row (M=1 qgemv).
async fn check_q5_0_attn_proj_qgemv_decode(device: &Device) -> Result<(), SuiteError> {
    check_q5_0_qmatmul_case(device, "q5_0_attn_proj_qgemv_decode", [896, 896], 1, 211).await
}

/// ffn_gate / ffn_up: [4864, 896] Q5_0, single decode row (M=1 qgemv).
async fn check_q5_0_ffn_gate_qgemv_decode(device: &Device) -> Result<(), SuiteError> {
    check_q5_0_qmatmul_case(device, "q5_0_ffn_gate_qgemv_decode", [4864, 896], 1, 223).await
}

/// attn_k: [128, 896] Q5_0, single decode row (M=1 qgemv).
async fn check_q5_0_kv_proj_qgemv_decode(device: &Device) -> Result<(), SuiteError> {
    check_q5_0_qmatmul_case(device, "q5_0_kv_proj_qgemv_decode", [128, 896], 1, 227).await
}

/// attn_q / attn_output: [896, 896] Q5_0, multi-row prefill (M>1 GEMM).
async fn check_q5_0_attn_proj_prefill_multirow(device: &Device) -> Result<(), SuiteError> {
    check_q5_0_qmatmul_case(
        device,
        "q5_0_attn_proj_prefill_multirow",
        [896, 896],
        17,
        229,
    )
    .await
}

async fn assert_flash_attention_case(
    device: &Device,
    case: FlashCase,
    tol: f32,
    transposed_q: bool,
) -> Result<(), SuiteError> {
    let suite_case = "flash_attention_qwen_decode";
    let q_data = attention_data(
        case.batch * case.num_heads * case.q_seq_len * case.head_dim,
        0.1,
    );
    let k_data = attention_data(
        case.batch * case.num_kv_heads * case.kv_seq_len * case.head_dim,
        -0.15,
    );
    let v_data = attention_data(
        case.batch * case.num_kv_heads * case.kv_seq_len * case.head_dim,
        0.35,
    );
    let scale = 1.0 / (case.head_dim as f32).sqrt();

    let q_cpu: Tensor<4, f32> = Tensor::from_slice(
        &Device::Cpu,
        [case.batch, case.num_heads, case.q_seq_len, case.head_dim],
        &q_data,
    );
    let k_cpu: Tensor<4, f32> = Tensor::from_slice(
        &Device::Cpu,
        [
            case.batch,
            case.num_kv_heads,
            case.kv_seq_len,
            case.head_dim,
        ],
        &k_data,
    );
    let v_cpu: Tensor<4, f32> = Tensor::from_slice(
        &Device::Cpu,
        [
            case.batch,
            case.num_kv_heads,
            case.kv_seq_len,
            case.head_dim,
        ],
        &v_data,
    );
    let expected = q_cpu
        .flash_attention(&k_cpu, &v_cpu, scale, None)
        .to_concrete();

    let q = if transposed_q {
        let q_pre: Tensor<4, f32> = Tensor::from_slice(
            device,
            [case.batch, case.q_seq_len, case.num_heads, case.head_dim],
            &q_data,
        );
        q_pre.transpose(1, 2).to_concrete()
    } else {
        Tensor::from_slice(
            device,
            [case.batch, case.num_heads, case.q_seq_len, case.head_dim],
            &q_data,
        )
    };
    let k: Tensor<4, f32> = Tensor::from_slice(
        device,
        [
            case.batch,
            case.num_kv_heads,
            case.kv_seq_len,
            case.head_dim,
        ],
        &k_data,
    );
    let v: Tensor<4, f32> = Tensor::from_slice(
        device,
        [
            case.batch,
            case.num_kv_heads,
            case.kv_seq_len,
            case.head_dim,
        ],
        &v_data,
    );
    let actual = q.flash_attention(&k, &v, scale, None).to_concrete();
    approx_eq(&actual, &expected, tol).await.map_err(|err| {
        SuiteError::case(
            suite_case,
            format!(
                "heads={} kv_heads={} kv_seq={} transposed_q={}: {err}",
                case.num_heads, case.num_kv_heads, case.kv_seq_len, transposed_q
            ),
        )
    })
}

#[allow(clippy::too_many_arguments)]
async fn assert_flash_attention_masked_case(
    device: &Device,
    suite_case: &'static str,
    case: FlashCase,
    mask_cpu: &Tensor<2, f32>,
    mask: &Tensor<2, f32>,
    kind: MaskKind,
    tol: f32,
    transposed_q: bool,
) -> Result<(), SuiteError> {
    let q_source_data = attention_data(
        case.batch * case.num_heads * case.q_seq_len * case.head_dim,
        0.2,
    );
    let q_data = if transposed_q {
        transpose_bqhd_to_bhqd(
            &q_source_data,
            case.batch,
            case.q_seq_len,
            case.num_heads,
            case.head_dim,
        )
    } else {
        q_source_data.clone()
    };
    let k_data = attention_data(
        case.batch * case.num_kv_heads * case.kv_seq_len * case.head_dim,
        -0.25,
    );
    let v_data = attention_data(
        case.batch * case.num_kv_heads * case.kv_seq_len * case.head_dim,
        0.45,
    );
    let scale = 1.0 / (case.head_dim as f32).sqrt();

    let q_cpu: Tensor<4, f32> = Tensor::from_slice(
        &Device::Cpu,
        [case.batch, case.num_heads, case.q_seq_len, case.head_dim],
        &q_data,
    );
    let k_cpu: Tensor<4, f32> = Tensor::from_slice(
        &Device::Cpu,
        [
            case.batch,
            case.num_kv_heads,
            case.kv_seq_len,
            case.head_dim,
        ],
        &k_data,
    );
    let v_cpu: Tensor<4, f32> = Tensor::from_slice(
        &Device::Cpu,
        [
            case.batch,
            case.num_kv_heads,
            case.kv_seq_len,
            case.head_dim,
        ],
        &v_data,
    );
    let expected = q_cpu
        .flash_attention(&k_cpu, &v_cpu, scale, Some((mask_cpu, kind)))
        .to_concrete();

    let q = if transposed_q {
        let q_pre: Tensor<4, f32> = Tensor::from_slice(
            device,
            [case.batch, case.q_seq_len, case.num_heads, case.head_dim],
            &q_source_data,
        );
        q_pre.transpose(1, 2).to_concrete()
    } else {
        Tensor::from_slice(
            device,
            [case.batch, case.num_heads, case.q_seq_len, case.head_dim],
            &q_data,
        )
    };
    let k: Tensor<4, f32> = Tensor::from_slice(
        device,
        [
            case.batch,
            case.num_kv_heads,
            case.kv_seq_len,
            case.head_dim,
        ],
        &k_data,
    );
    let v: Tensor<4, f32> = Tensor::from_slice(
        device,
        [
            case.batch,
            case.num_kv_heads,
            case.kv_seq_len,
            case.head_dim,
        ],
        &v_data,
    );
    let actual = q
        .flash_attention(&k, &v, scale, Some((mask, kind)))
        .to_concrete();
    approx_eq(&actual, &expected, tol).await.map_err(|err| {
        SuiteError::case(
            suite_case,
            format!(
                "heads={} kv_heads={} q_seq={} kv_seq={} kind={kind:?} transposed_q={transposed_q}: {err}",
                case.num_heads, case.num_kv_heads, case.q_seq_len, case.kv_seq_len
            ),
        )
    })
}
