//! Per-benchmark size sweeps used by the web runner detail route.
//!
//! Each sweep point runs the shared case bodies in `webgpu.rs` / `burn.rs`
//! with sweep-derived sizes; only the size tables and the size-to-parameter
//! mapping live here.

use fusor::Device;

use super::{BenchmarkConfig, BenchmarkReport, BenchmarkResult};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BenchmarkSweepSize {
    pub label: &'static str,
    pub value: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BenchmarkSweepDescriptor {
    pub case: &'static str,
    pub title: &'static str,
    pub detail: &'static str,
    pub sizes: &'static [BenchmarkSweepSize],
}

#[derive(Clone, Debug, PartialEq)]
pub struct BenchmarkSweepPoint {
    pub label: String,
    pub value: usize,
    pub burn: Option<BenchmarkReport>,
    pub webgpu: Option<BenchmarkReport>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BenchmarkSweepEvent {
    Started {
        suite: &'static str,
        label: String,
    },
    Finished {
        suite: &'static str,
        label: String,
        report: BenchmarkReport,
    },
}

const fn size(label: &'static str, value: usize) -> BenchmarkSweepSize {
    BenchmarkSweepSize { label, value }
}

const SQUARE_SIZES: [BenchmarkSweepSize; 4] = [
    size("128x128", 128),
    size("256x256", 256),
    size("512x512", 512),
    size("768x768", 768),
];

const RANK4_SIZES: [BenchmarkSweepSize; 4] = [
    size("3x5x16x8", 16),
    size("6x8x24x12", 24),
    size("9x11x32x16", 32),
    size("12x16x48x24", 48),
];

const ROW_SIZES: [BenchmarkSweepSize; 4] = [
    size("64 rows", 64),
    size("128 rows", 128),
    size("256 rows", 256),
    size("512 rows", 512),
];

const MID_AXIS_SIZES: [BenchmarkSweepSize; 4] = [
    size("16x64x32", 64),
    size("24x96x48", 96),
    size("32x128x64", 128),
    size("48x192x96", 192),
];

const SEQ_SIZES: [BenchmarkSweepSize; 4] = [
    size("seq 32", 32),
    size("seq 64", 64),
    size("seq 128", 128),
    size("seq 256", 256),
];

const MATMUL_SIZES: [BenchmarkSweepSize; 4] = [
    size("64", 64),
    size("128", 128),
    size("256", 256),
    size("384", 384),
];

const TOPK_SIZES: [BenchmarkSweepSize; 4] = [
    size("8k", 8_192),
    size("32k", 32_768),
    size("64k", 65_537),
    size("128k", 131_072),
];

const QWEN_TOPK_SIZES: [BenchmarkSweepSize; 4] = [
    size("32k", 32_768),
    size("65k", 65_536),
    size("100k", 100_000),
    size("151936", 151_936),
];

const Q8_GEMV_SIZES: [BenchmarkSweepSize; 4] = [
    size("1024x512", 1024),
    size("2048x768", 2048),
    size("4096x896", 4096),
    size("6144x1024", 6144),
];

const Q4_GEMV_SIZES: [BenchmarkSweepSize; 4] = [
    size("512x512", 512),
    size("1024x768", 1024),
    size("2048x1024", 2048),
    size("4096x1024", 4096),
];

const PAIRED_SILU_SIZES: [BenchmarkSweepSize; 4] = [
    size("pair 256", 256),
    size("pair 512", 512),
    size("pair 1024", 1024),
    size("pair 1536", 1536),
];

const fn desc(
    case: &'static str,
    title: &'static str,
    detail: &'static str,
    sizes: &'static [BenchmarkSweepSize],
) -> BenchmarkSweepDescriptor {
    BenchmarkSweepDescriptor {
        case,
        title,
        detail,
        sizes,
    }
}

pub fn descriptor(case: &str) -> Option<BenchmarkSweepDescriptor> {
    let descriptor = match case {
        "elementwise_add_square" => desc(
            "elementwise_add_square",
            "Elementwise add",
            "F32 add over square tensors.",
            &SQUARE_SIZES,
        ),
        "elementwise_mul_rank4" => desc(
            "elementwise_mul_rank4",
            "Elementwise mul rank 4",
            "F32 multiply over rank-4 tensors.",
            &RANK4_SIZES,
        ),
        "unary_trig_chain" => desc(
            "unary_trig_chain",
            "Unary trig chain",
            "sin(x) + cos(x) over square tensors.",
            &SQUARE_SIZES,
        ),
        "activation_gelu" => desc(
            "activation_gelu",
            "GELU activation",
            "F32 GELU over square tensors.",
            &SQUARE_SIZES,
        ),
        "broadcast_add" => desc(
            "broadcast_add",
            "Broadcast add",
            "Matrix plus broadcast row vector.",
            &ROW_SIZES,
        ),
        "transpose_then_elementwise" => desc(
            "transpose_then_elementwise",
            "Transpose then elementwise",
            "Transpose followed by elementwise square.",
            &ROW_SIZES,
        ),
        "reduction_sum_last_dim" => desc(
            "reduction_sum_last_dim",
            "Reduction sum",
            "Sum over the last matrix axis.",
            &ROW_SIZES,
        ),
        "reduction_max_middle_axis" => desc(
            "reduction_max_middle_axis",
            "Reduction max",
            "Max over the middle rank-3 axis.",
            &MID_AXIS_SIZES,
        ),
        "softmax_last_dim" => desc(
            "softmax_last_dim",
            "Softmax last axis",
            "Softmax over the last matrix axis.",
            &ROW_SIZES,
        ),
        "softmax_middle_axis" => desc(
            "softmax_middle_axis",
            "Softmax middle axis",
            "Softmax over the middle rank-3 axis.",
            &MID_AXIS_SIZES,
        ),
        "layer_norm_last_dim" => desc(
            "layer_norm_last_dim",
            "Layer norm",
            "Layer normalization over the last dimension.",
            &SEQ_SIZES,
        ),
        "rms_norm_fused" => desc(
            "rms_norm_fused",
            "RMS norm",
            "RMS normalization over the last dimension.",
            &SEQ_SIZES,
        ),
        "dense_matmul_square" => desc(
            "dense_matmul_square",
            "Dense matmul",
            "Square F32 matrix multiplication.",
            &MATMUL_SIZES,
        ),
        "dense_batched_matmul" => desc(
            "dense_batched_matmul",
            "Batched matmul",
            "Batched F32 matrix multiplication.",
            &MATMUL_SIZES,
        ),
        "conv1d_small" => desc(
            "conv1d_small",
            "Conv1D",
            "Small 1D convolution with fixed channels.",
            &ROW_SIZES,
        ),
        "top_k_large" => desc(
            "top_k_large",
            "Top K",
            "Top-k selection over a logits vector.",
            &TOPK_SIZES,
        ),
        "top_k_qwen_vocab" => desc(
            "top_k_qwen_vocab",
            "Top K Qwen vocab",
            "Top-k selection over vocabulary-scale logits.",
            &QWEN_TOPK_SIZES,
        ),
        "q8_0_qgemv" => desc(
            "q8_0_qgemv",
            "Q8_0 GEMV",
            "Fusor Q8_0 GEMV against a Burn dense-f32 baseline.",
            &Q8_GEMV_SIZES,
        ),
        "q4k_qgemv" => desc(
            "q4k_qgemv",
            "Q4K GEMV",
            "Fusor Q4K GEMV against a Burn dense-f32 baseline.",
            &Q4_GEMV_SIZES,
        ),
        "q4k_paired_silu" => desc(
            "q4k_paired_silu",
            "Q4K paired SiLU",
            "Fusor fused paired SiLU GEMV against a Burn dense-f32 baseline.",
            &PAIRED_SILU_SIZES,
        ),
        "attention_small" => desc(
            "attention_small",
            "Attention",
            "Scaled dot-product attention across sequence lengths.",
            &SEQ_SIZES,
        ),
        "attention_causal_small" => desc(
            "attention_causal_small",
            "Causal attention",
            "Causal scaled dot-product attention across sequence lengths.",
            &SEQ_SIZES,
        ),
        "rope_fused_decode" => desc(
            "rope_fused_decode",
            "RoPE",
            "Rotary positional encoding across sequence lengths.",
            &SEQ_SIZES,
        ),
        _ => return None,
    };
    Some(descriptor)
}

pub async fn run_sweep(
    case: &str,
    device: &Device,
    config: BenchmarkConfig,
    mut progress: impl FnMut(BenchmarkSweepEvent),
) -> BenchmarkResult<Vec<BenchmarkSweepPoint>> {
    let descriptor = descriptor(case).ok_or_else(|| format!("unknown benchmark sweep: {case}"))?;
    let mut points = Vec::with_capacity(descriptor.sizes.len());

    for size in descriptor.sizes {
        let label = size.label.to_string();
        #[cfg(feature = "burn-bench")]
        let burn = {
            progress(BenchmarkSweepEvent::Started {
                suite: "burn",
                label: label.clone(),
            });
            let report = run_burn_case(case, *size, config).await?;
            progress(BenchmarkSweepEvent::Finished {
                suite: "burn",
                label: label.clone(),
                report: report.clone(),
            });
            Some(report)
        };

        #[cfg(not(feature = "burn-bench"))]
        let burn = None;

        progress(BenchmarkSweepEvent::Started {
            suite: "webgpu",
            label: label.clone(),
        });
        let webgpu = run_webgpu_case(case, device, *size, config).await?;
        progress(BenchmarkSweepEvent::Finished {
            suite: "webgpu",
            label: label.clone(),
            report: webgpu.clone(),
        });

        points.push(BenchmarkSweepPoint {
            label,
            value: size.value,
            burn,
            webgpu: Some(webgpu),
        });
    }

    Ok(points)
}

fn rank4_shape(size: usize) -> [usize; 4] {
    [size / 4, size / 3, size, size / 2]
}

fn middle_shape(size: usize) -> [usize; 3] {
    [size / 4, size, size / 2]
}

fn q8_shape(value: usize) -> [usize; 2] {
    let k = match value {
        1024 => 512,
        2048 => 768,
        4096 => 896,
        _ => 1024,
    };
    [value, k]
}

fn q4_shape(value: usize) -> [usize; 2] {
    let k = match value {
        512 => 512,
        1024 => 768,
        _ => 1024,
    };
    [value, k]
}

async fn run_webgpu_case(
    case: &str,
    device: &Device,
    size: BenchmarkSweepSize,
    config: BenchmarkConfig,
) -> BenchmarkResult<BenchmarkReport> {
    use crate::bench::webgpu as cases;
    let name = format!("webgpu::{case}@{}", size.label);
    let value = size.value;
    match case {
        "elementwise_add_square" => {
            cases::elementwise_add_square_case(device, config, name, value).await
        }
        "elementwise_mul_rank4" => {
            cases::elementwise_mul_rank4_case(device, config, name, rank4_shape(value)).await
        }
        "unary_trig_chain" => cases::unary_trig_chain_case(device, config, name, value).await,
        "activation_gelu" => {
            cases::activation_gelu_case(device, config, name, [value, value]).await
        }
        "broadcast_add" => cases::broadcast_add_case(device, config, name, value).await,
        "transpose_then_elementwise" => {
            cases::transpose_then_elementwise_case(device, config, name, [value, value + value / 2])
                .await
        }
        "reduction_sum_last_dim" => {
            cases::reduction_sum_last_dim_case(device, config, name, value).await
        }
        "reduction_max_middle_axis" => {
            cases::reduction_max_middle_axis_case(device, config, name, middle_shape(value)).await
        }
        "softmax_last_dim" => cases::softmax_last_dim_case(device, config, name, value).await,
        "softmax_middle_axis" => {
            cases::softmax_middle_axis_case(device, config, name, middle_shape(value)).await
        }
        "layer_norm_last_dim" => {
            cases::layer_norm_last_dim_case(device, config, name, [4, value, 512]).await
        }
        "rms_norm_fused" => cases::rms_norm_fused_case(device, config, name, [4, value, 512]).await,
        "dense_matmul_square" => cases::dense_matmul_square_case(device, config, name, value).await,
        "dense_batched_matmul" => {
            cases::dense_batched_matmul_case(device, config, name, 4, value, value + 32).await
        }
        "conv1d_small" => cases::conv1d_small_case(device, config, name, value).await,
        "top_k_large" => {
            cases::top_k_case(device, config, name, value, 64, cases::topk_values(value)).await
        }
        "top_k_qwen_vocab" => {
            cases::top_k_case(device, config, name, value, 40, cases::topk_values(value)).await
        }
        "q8_0_qgemv" => cases::q8_0_qgemv_case(device, config, name, q8_shape(value)).await,
        "q4k_qgemv" => cases::q4k_qgemv_case(device, config, name, q4_shape(value)).await,
        "q4k_paired_silu" => {
            cases::q4k_paired_silu_case(device, config, name, [value * 2, 1024]).await
        }
        "attention_small" => {
            cases::attention_case(
                device,
                config,
                name,
                value,
                [31, 32, 33],
                false,
                "attention",
            )
            .await
        }
        "attention_causal_small" => {
            cases::attention_case(device, config, name, value, [31, 32, 33], true, "attention")
                .await
        }
        "rope_fused_decode" => cases::rope_fused_decode_case(device, config, name, value).await,
        _ => Err(format!("unknown WebGPU benchmark sweep: {case}").into()),
    }
}

#[cfg(feature = "burn-bench")]
async fn run_burn_case(
    case: &str,
    size: BenchmarkSweepSize,
    config: BenchmarkConfig,
) -> BenchmarkResult<BenchmarkReport> {
    use crate::bench::{burn as cases, webgpu};
    let name = format!("burn::{case}@{}", size.label);
    let value = size.value;
    match case {
        "elementwise_add_square" => cases::elementwise_add_square_case(config, name, value).await,
        "elementwise_mul_rank4" => {
            cases::elementwise_mul_rank4_case(config, name, rank4_shape(value)).await
        }
        "unary_trig_chain" => cases::unary_trig_chain_case(config, name, value).await,
        "activation_gelu" => cases::activation_gelu_case(config, name, [value, value]).await,
        "broadcast_add" => cases::broadcast_add_case(config, name, value).await,
        "transpose_then_elementwise" => {
            cases::transpose_then_elementwise_case(config, name, [value, value + value / 2]).await
        }
        "reduction_sum_last_dim" => cases::reduction_sum_last_dim_case(config, name, value).await,
        "reduction_max_middle_axis" => {
            cases::reduction_max_middle_axis_case(config, name, middle_shape(value)).await
        }
        "softmax_last_dim" => cases::softmax_last_dim_case(config, name, value).await,
        "softmax_middle_axis" => {
            cases::softmax_middle_axis_case(config, name, middle_shape(value)).await
        }
        "layer_norm_last_dim" => {
            cases::layer_norm_last_dim_case(config, name, [4, value, 512]).await
        }
        "rms_norm_fused" => cases::rms_norm_fused_case(config, name, [4, value, 512]).await,
        "dense_matmul_square" => cases::dense_matmul_square_case(config, name, value).await,
        "dense_batched_matmul" => {
            cases::dense_batched_matmul_case(config, name, 4, value, value + 32).await
        }
        "conv1d_small" => cases::conv1d_small_case(config, name, value).await,
        "top_k_large" => {
            cases::top_k_case(config, name, value, 64, webgpu::topk_values(value)).await
        }
        "top_k_qwen_vocab" => {
            cases::top_k_case(config, name, value, 40, webgpu::topk_values(value)).await
        }
        "q8_0_qgemv" => cases::qgemv_dense_case(config, name, q8_shape(value), 8, 80, "").await,
        "q4k_qgemv" => cases::qgemv_dense_case(config, name, q4_shape(value), 29, 81, "").await,
        "q4k_paired_silu" => cases::q4k_paired_silu_case(config, name, [value * 2, 1024], "").await,
        "attention_small" => {
            cases::attention_case(config, name, value, [31, 32, 33], false, "attention").await
        }
        "attention_causal_small" => {
            cases::attention_case(config, name, value, [31, 32, 33], true, "attention").await
        }
        "rope_fused_decode" => cases::rope_fused_decode_case(config, name, value).await,
        _ => Err(format!("unknown Burn benchmark sweep: {case}").into()),
    }
}
