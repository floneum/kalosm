//! Per-benchmark size sweeps used by the web runner detail route.

use fusor::{Device, GgmlType, MaskKind, QMatrix, Tensor as FusorTensor};

use crate::common::quantized::{q4k_raw_bytes, q8_0_raw_bytes, qmatrix_from_raw_bytes};

use super::{BenchmarkConfig, BenchmarkReport, BenchmarkResult, time_samples};

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

const SQUARE_SIZES: [BenchmarkSweepSize; 4] = [
    BenchmarkSweepSize {
        label: "128x128",
        value: 128,
    },
    BenchmarkSweepSize {
        label: "256x256",
        value: 256,
    },
    BenchmarkSweepSize {
        label: "512x512",
        value: 512,
    },
    BenchmarkSweepSize {
        label: "768x768",
        value: 768,
    },
];

const RANK4_SIZES: [BenchmarkSweepSize; 4] = [
    BenchmarkSweepSize {
        label: "3x5x16x8",
        value: 16,
    },
    BenchmarkSweepSize {
        label: "6x8x24x12",
        value: 24,
    },
    BenchmarkSweepSize {
        label: "9x11x32x16",
        value: 32,
    },
    BenchmarkSweepSize {
        label: "12x16x48x24",
        value: 48,
    },
];

const ROW_SIZES: [BenchmarkSweepSize; 4] = [
    BenchmarkSweepSize {
        label: "64 rows",
        value: 64,
    },
    BenchmarkSweepSize {
        label: "128 rows",
        value: 128,
    },
    BenchmarkSweepSize {
        label: "256 rows",
        value: 256,
    },
    BenchmarkSweepSize {
        label: "512 rows",
        value: 512,
    },
];

const MID_AXIS_SIZES: [BenchmarkSweepSize; 4] = [
    BenchmarkSweepSize {
        label: "16x64x32",
        value: 64,
    },
    BenchmarkSweepSize {
        label: "24x96x48",
        value: 96,
    },
    BenchmarkSweepSize {
        label: "32x128x64",
        value: 128,
    },
    BenchmarkSweepSize {
        label: "48x192x96",
        value: 192,
    },
];

const SEQ_SIZES: [BenchmarkSweepSize; 4] = [
    BenchmarkSweepSize {
        label: "seq 32",
        value: 32,
    },
    BenchmarkSweepSize {
        label: "seq 64",
        value: 64,
    },
    BenchmarkSweepSize {
        label: "seq 128",
        value: 128,
    },
    BenchmarkSweepSize {
        label: "seq 256",
        value: 256,
    },
];

const MATMUL_SIZES: [BenchmarkSweepSize; 4] = [
    BenchmarkSweepSize {
        label: "64",
        value: 64,
    },
    BenchmarkSweepSize {
        label: "128",
        value: 128,
    },
    BenchmarkSweepSize {
        label: "256",
        value: 256,
    },
    BenchmarkSweepSize {
        label: "384",
        value: 384,
    },
];

const TOPK_SIZES: [BenchmarkSweepSize; 4] = [
    BenchmarkSweepSize {
        label: "8k",
        value: 8_192,
    },
    BenchmarkSweepSize {
        label: "32k",
        value: 32_768,
    },
    BenchmarkSweepSize {
        label: "64k",
        value: 65_537,
    },
    BenchmarkSweepSize {
        label: "128k",
        value: 131_072,
    },
];

const QWEN_TOPK_SIZES: [BenchmarkSweepSize; 4] = [
    BenchmarkSweepSize {
        label: "32k",
        value: 32_768,
    },
    BenchmarkSweepSize {
        label: "65k",
        value: 65_536,
    },
    BenchmarkSweepSize {
        label: "100k",
        value: 100_000,
    },
    BenchmarkSweepSize {
        label: "151936",
        value: 151_936,
    },
];

const Q8_GEMV_SIZES: [BenchmarkSweepSize; 4] = [
    BenchmarkSweepSize {
        label: "1024x512",
        value: 1024,
    },
    BenchmarkSweepSize {
        label: "2048x768",
        value: 2048,
    },
    BenchmarkSweepSize {
        label: "4096x896",
        value: 4096,
    },
    BenchmarkSweepSize {
        label: "6144x1024",
        value: 6144,
    },
];

const Q4_GEMV_SIZES: [BenchmarkSweepSize; 4] = [
    BenchmarkSweepSize {
        label: "512x512",
        value: 512,
    },
    BenchmarkSweepSize {
        label: "1024x768",
        value: 1024,
    },
    BenchmarkSweepSize {
        label: "2048x1024",
        value: 2048,
    },
    BenchmarkSweepSize {
        label: "4096x1024",
        value: 4096,
    },
];

const PAIRED_SILU_SIZES: [BenchmarkSweepSize; 4] = [
    BenchmarkSweepSize {
        label: "pair 256",
        value: 256,
    },
    BenchmarkSweepSize {
        label: "pair 512",
        value: 512,
    },
    BenchmarkSweepSize {
        label: "pair 1024",
        value: 1024,
    },
    BenchmarkSweepSize {
        label: "pair 1536",
        value: 1536,
    },
];

pub fn descriptor(case: &str) -> Option<BenchmarkSweepDescriptor> {
    let descriptor = match case {
        "elementwise_add_square" => BenchmarkSweepDescriptor {
            case: "elementwise_add_square",
            title: "Elementwise add",
            detail: "F32 add over square tensors.",
            sizes: &SQUARE_SIZES,
        },
        "elementwise_mul_rank4" => BenchmarkSweepDescriptor {
            case: "elementwise_mul_rank4",
            title: "Elementwise mul rank 4",
            detail: "F32 multiply over rank-4 tensors.",
            sizes: &RANK4_SIZES,
        },
        "unary_trig_chain" => BenchmarkSweepDescriptor {
            case: "unary_trig_chain",
            title: "Unary trig chain",
            detail: "sin(x) + cos(x) over square tensors.",
            sizes: &SQUARE_SIZES,
        },
        "activation_gelu" => BenchmarkSweepDescriptor {
            case: "activation_gelu",
            title: "GELU activation",
            detail: "F32 GELU over square tensors.",
            sizes: &SQUARE_SIZES,
        },
        "broadcast_add" => BenchmarkSweepDescriptor {
            case: "broadcast_add",
            title: "Broadcast add",
            detail: "Matrix plus broadcast row vector.",
            sizes: &ROW_SIZES,
        },
        "transpose_then_elementwise" => BenchmarkSweepDescriptor {
            case: "transpose_then_elementwise",
            title: "Transpose then elementwise",
            detail: "Transpose followed by elementwise square.",
            sizes: &ROW_SIZES,
        },
        "reduction_sum_last_dim" => BenchmarkSweepDescriptor {
            case: "reduction_sum_last_dim",
            title: "Reduction sum",
            detail: "Sum over the last matrix axis.",
            sizes: &ROW_SIZES,
        },
        "reduction_max_middle_axis" => BenchmarkSweepDescriptor {
            case: "reduction_max_middle_axis",
            title: "Reduction max",
            detail: "Max over the middle rank-3 axis.",
            sizes: &MID_AXIS_SIZES,
        },
        "softmax_last_dim" => BenchmarkSweepDescriptor {
            case: "softmax_last_dim",
            title: "Softmax last axis",
            detail: "Softmax over the last matrix axis.",
            sizes: &ROW_SIZES,
        },
        "softmax_middle_axis" => BenchmarkSweepDescriptor {
            case: "softmax_middle_axis",
            title: "Softmax middle axis",
            detail: "Softmax over the middle rank-3 axis.",
            sizes: &MID_AXIS_SIZES,
        },
        "layer_norm_last_dim" => BenchmarkSweepDescriptor {
            case: "layer_norm_last_dim",
            title: "Layer norm",
            detail: "Layer normalization over the last dimension.",
            sizes: &SEQ_SIZES,
        },
        "rms_norm_fused" => BenchmarkSweepDescriptor {
            case: "rms_norm_fused",
            title: "RMS norm",
            detail: "RMS normalization over the last dimension.",
            sizes: &SEQ_SIZES,
        },
        "dense_matmul_square" => BenchmarkSweepDescriptor {
            case: "dense_matmul_square",
            title: "Dense matmul",
            detail: "Square F32 matrix multiplication.",
            sizes: &MATMUL_SIZES,
        },
        "dense_batched_matmul" => BenchmarkSweepDescriptor {
            case: "dense_batched_matmul",
            title: "Batched matmul",
            detail: "Batched F32 matrix multiplication.",
            sizes: &MATMUL_SIZES,
        },
        "conv1d_small" => BenchmarkSweepDescriptor {
            case: "conv1d_small",
            title: "Conv1D",
            detail: "Small 1D convolution with fixed channels.",
            sizes: &ROW_SIZES,
        },
        "top_k_large" => BenchmarkSweepDescriptor {
            case: "top_k_large",
            title: "Top K",
            detail: "Top-k selection over a logits vector.",
            sizes: &TOPK_SIZES,
        },
        "top_k_qwen_vocab" => BenchmarkSweepDescriptor {
            case: "top_k_qwen_vocab",
            title: "Top K Qwen vocab",
            detail: "Top-k selection over vocabulary-scale logits.",
            sizes: &QWEN_TOPK_SIZES,
        },
        "q8_0_qgemv" => BenchmarkSweepDescriptor {
            case: "q8_0_qgemv",
            title: "Q8_0 GEMV",
            detail: "Fusor Q8_0 GEMV against a Burn dense-f32 baseline.",
            sizes: &Q8_GEMV_SIZES,
        },
        "q4k_qgemv" => BenchmarkSweepDescriptor {
            case: "q4k_qgemv",
            title: "Q4K GEMV",
            detail: "Fusor Q4K GEMV against a Burn dense-f32 baseline.",
            sizes: &Q4_GEMV_SIZES,
        },
        "q4k_paired_silu" => BenchmarkSweepDescriptor {
            case: "q4k_paired_silu",
            title: "Q4K paired SiLU",
            detail: "Fusor fused paired SiLU GEMV against a Burn dense-f32 baseline.",
            sizes: &PAIRED_SILU_SIZES,
        },
        "attention_small" => BenchmarkSweepDescriptor {
            case: "attention_small",
            title: "Attention",
            detail: "Scaled dot-product attention across sequence lengths.",
            sizes: &SEQ_SIZES,
        },
        "attention_causal_small" => BenchmarkSweepDescriptor {
            case: "attention_causal_small",
            title: "Causal attention",
            detail: "Causal scaled dot-product attention across sequence lengths.",
            sizes: &SEQ_SIZES,
        },
        "rope_fused_decode" => BenchmarkSweepDescriptor {
            case: "rope_fused_decode",
            title: "RoPE",
            detail: "Rotary positional encoding across sequence lengths.",
            sizes: &SEQ_SIZES,
        },
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

fn shape_label(shape: &[usize]) -> String {
    shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join("x")
}

fn elements(shape: &[usize]) -> usize {
    shape.iter().product()
}

async fn materialize_inputs<const R: usize>(inputs: &[&FusorTensor<R, f32>]) {
    for input in inputs {
        input.materialize().await;
    }
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

fn sweep_report(
    suite: &str,
    case: &str,
    size: BenchmarkSweepSize,
    config: BenchmarkConfig,
    samples: Vec<super::Duration>,
    detail: impl Into<String>,
) -> BenchmarkReport {
    BenchmarkReport::new(
        format!("{suite}::{case}@{}", size.label),
        config,
        samples,
        detail,
    )
}

async fn run_webgpu_case(
    case: &str,
    device: &Device,
    size: BenchmarkSweepSize,
    config: BenchmarkConfig,
) -> BenchmarkResult<BenchmarkReport> {
    match case {
        "elementwise_add_square" => {
            let shape = [size.value, size.value];
            let lhs: FusorTensor<2, f32> = FusorTensor::from_slice(
                device,
                shape,
                &deterministic_values(elements(&shape), 1, 0.01),
            );
            let rhs: FusorTensor<2, f32> = FusorTensor::from_slice(
                device,
                shape,
                &deterministic_values(elements(&shape), 2, 0.008),
            );
            materialize_inputs(&[&lhs, &rhs]).await;
            let samples = time_samples(config, || {
                let output = (&lhs + &rhs).to_concrete();
                async move {
                    output.materialize().await;
                    Ok(())
                }
            })
            .await?;
            Ok(sweep_report(
                "webgpu",
                case,
                size,
                config,
                samples,
                format!("{} f32 add", shape_label(&shape)),
            ))
        }
        "elementwise_mul_rank4" => {
            let shape = rank4_shape(size.value);
            let lhs: FusorTensor<4, f32> = FusorTensor::from_slice(
                device,
                shape,
                &deterministic_values(elements(&shape), 3, 0.012),
            );
            let rhs: FusorTensor<4, f32> = FusorTensor::from_slice(
                device,
                shape,
                &deterministic_values(elements(&shape), 4, 0.009),
            );
            materialize_inputs(&[&lhs, &rhs]).await;
            let samples = time_samples(config, || {
                let output = (&lhs * &rhs).to_concrete();
                async move {
                    output.materialize().await;
                    Ok(())
                }
            })
            .await?;
            Ok(sweep_report(
                "webgpu",
                case,
                size,
                config,
                samples,
                format!("{} f32 mul", shape_label(&shape)),
            ))
        }
        "unary_trig_chain" => {
            let shape = [size.value, size.value];
            let input: FusorTensor<2, f32> = FusorTensor::from_slice(
                device,
                shape,
                &deterministic_values(elements(&shape), 10, 0.01),
            );
            materialize_inputs(&[&input]).await;
            let samples = time_samples(config, || {
                let output = (input.sin() + input.cos()).to_concrete();
                async move {
                    output.materialize().await;
                    Ok(())
                }
            })
            .await?;
            Ok(sweep_report(
                "webgpu",
                case,
                size,
                config,
                samples,
                format!("{} sin+cos", shape_label(&shape)),
            ))
        }
        "activation_gelu" => {
            let shape = [size.value, size.value];
            let input: FusorTensor<2, f32> = FusorTensor::from_slice(
                device,
                shape,
                &deterministic_values(elements(&shape), 11, 0.015),
            );
            materialize_inputs(&[&input]).await;
            let samples = time_samples(config, || {
                let output = input.gelu();
                async move {
                    output.materialize().await;
                    Ok(())
                }
            })
            .await?;
            Ok(sweep_report(
                "webgpu",
                case,
                size,
                config,
                samples,
                format!("{} gelu", shape_label(&shape)),
            ))
        }
        "broadcast_add" => {
            let matrix_shape = [size.value, 512usize];
            let vector_shape = [512usize];
            let matrix: FusorTensor<2, f32> = FusorTensor::from_slice(
                device,
                matrix_shape,
                &deterministic_values(elements(&matrix_shape), 12, 0.006),
            );
            let vector: FusorTensor<1, f32> = FusorTensor::from_slice(
                device,
                vector_shape,
                &deterministic_values(elements(&vector_shape), 13, 0.01),
            );
            materialize_inputs(&[&matrix]).await;
            materialize_inputs(&[&vector]).await;
            let samples = time_samples(config, || {
                let vector_row = vector.reshape([1, vector_shape[0]]);
                let output = (&matrix + vector_row.broadcast_as(matrix_shape)).to_concrete();
                async move {
                    output.materialize().await;
                    Ok(())
                }
            })
            .await?;
            Ok(sweep_report(
                "webgpu",
                case,
                size,
                config,
                samples,
                format!("{} + broadcast 512", shape_label(&matrix_shape)),
            ))
        }
        "transpose_then_elementwise" => {
            let shape = [size.value, size.value + size.value / 2];
            let input: FusorTensor<2, f32> = FusorTensor::from_slice(
                device,
                shape,
                &deterministic_values(elements(&shape), 14, 0.01),
            );
            materialize_inputs(&[&input]).await;
            let samples = time_samples(config, || {
                let transposed = input.transpose(0, 1);
                let output = (transposed.clone() * transposed).to_concrete();
                async move {
                    output.materialize().await;
                    Ok(())
                }
            })
            .await?;
            Ok(sweep_report(
                "webgpu",
                case,
                size,
                config,
                samples,
                format!("{} transpose, square", shape_label(&shape)),
            ))
        }
        "reduction_sum_last_dim" => {
            let shape = [size.value, 512usize];
            let input: FusorTensor<2, f32> = FusorTensor::from_slice(
                device,
                shape,
                &deterministic_values(elements(&shape), 15, 0.004),
            );
            materialize_inputs(&[&input]).await;
            let samples = time_samples(config, || {
                let output = input.sum::<1>(1);
                async move {
                    output.materialize().await;
                    Ok(())
                }
            })
            .await?;
            Ok(sweep_report(
                "webgpu",
                case,
                size,
                config,
                samples,
                format!("{} sum axis 1", shape_label(&shape)),
            ))
        }
        "reduction_max_middle_axis" => {
            let shape = middle_shape(size.value);
            let input: FusorTensor<3, f32> = FusorTensor::from_slice(
                device,
                shape,
                &deterministic_values(elements(&shape), 16, 0.004),
            );
            materialize_inputs(&[&input]).await;
            let samples = time_samples(config, || {
                let output = input.max::<2>(1);
                async move {
                    output.materialize().await;
                    Ok(())
                }
            })
            .await?;
            Ok(sweep_report(
                "webgpu",
                case,
                size,
                config,
                samples,
                format!("{} max axis 1", shape_label(&shape)),
            ))
        }
        "softmax_last_dim" => {
            let shape = [size.value, 256usize];
            let input: FusorTensor<2, f32> = FusorTensor::from_slice(
                device,
                shape,
                &deterministic_values(elements(&shape), 5, 0.006),
            );
            materialize_inputs(&[&input]).await;
            let samples = time_samples(config, || {
                let output = input.softmax_last_dim::<1>();
                async move {
                    output.materialize().await;
                    Ok(())
                }
            })
            .await?;
            Ok(sweep_report(
                "webgpu",
                case,
                size,
                config,
                samples,
                format!("{} last-axis softmax", shape_label(&shape)),
            ))
        }
        "softmax_middle_axis" => {
            let shape = middle_shape(size.value);
            let input: FusorTensor<3, f32> = FusorTensor::from_slice(
                device,
                shape,
                &deterministic_values(elements(&shape), 17, 0.004),
            );
            materialize_inputs(&[&input]).await;
            let samples = time_samples(config, || {
                let output = input.softmax::<2>(1);
                async move {
                    output.materialize().await;
                    Ok(())
                }
            })
            .await?;
            Ok(sweep_report(
                "webgpu",
                case,
                size,
                config,
                samples,
                format!("{} softmax axis 1", shape_label(&shape)),
            ))
        }
        "layer_norm_last_dim" => {
            let shape = [4usize, size.value, 512usize];
            let last_dim = shape[2];
            let input: FusorTensor<3, f32> = FusorTensor::from_slice(
                device,
                shape,
                &deterministic_values(elements(&shape), 18, 0.01),
            );
            let weight_values = deterministic_values(last_dim, 19, 0.002)
                .into_iter()
                .map(|value| value + 1.0)
                .collect::<Vec<_>>();
            let bias_values = deterministic_values(last_dim, 20, 0.001);
            let weight: FusorTensor<1, f32> =
                FusorTensor::from_slice(device, [last_dim], &weight_values);
            let bias: FusorTensor<1, f32> =
                FusorTensor::from_slice(device, [last_dim], &bias_values);
            materialize_inputs(&[&input]).await;
            materialize_inputs(&[&weight, &bias]).await;
            let samples = time_samples(config, || {
                let output =
                    input.layer_norm_last_dim_fused::<2, 1, _, _>(&weight, Some(&bias), 1.0e-5);
                async move {
                    output.materialize().await;
                    Ok(())
                }
            })
            .await?;
            Ok(sweep_report(
                "webgpu",
                case,
                size,
                config,
                samples,
                format!("{} layer norm", shape_label(&shape)),
            ))
        }
        "rms_norm_fused" => {
            let shape = [4usize, size.value, 512usize];
            let last_dim = shape[2];
            let input: FusorTensor<3, f32> = FusorTensor::from_slice(
                device,
                shape,
                &deterministic_values(elements(&shape), 21, 0.01),
            );
            let weight_values = deterministic_values(last_dim, 22, 0.002)
                .into_iter()
                .map(|value| value + 1.0)
                .collect::<Vec<_>>();
            let weight: FusorTensor<1, f32> =
                FusorTensor::from_slice(device, [last_dim], &weight_values);
            materialize_inputs(&[&input]).await;
            materialize_inputs(&[&weight]).await;
            let samples = time_samples(config, || {
                let output = input.rms_norm_fused_no_bias::<1, 2>(&weight, 1.0e-5);
                async move {
                    output.materialize().await;
                    Ok(())
                }
            })
            .await?;
            Ok(sweep_report(
                "webgpu",
                case,
                size,
                config,
                samples,
                format!("{} rms norm", shape_label(&shape)),
            ))
        }
        "dense_matmul_square" => {
            let lhs_shape = [size.value, size.value];
            let rhs_shape = [size.value, size.value];
            let lhs: FusorTensor<2, f32> = FusorTensor::from_slice(
                device,
                lhs_shape,
                &deterministic_values(elements(&lhs_shape), 6, 0.004),
            );
            let rhs: FusorTensor<2, f32> = FusorTensor::from_slice(
                device,
                rhs_shape,
                &deterministic_values(elements(&rhs_shape), 7, 0.004),
            );
            materialize_inputs(&[&lhs, &rhs]).await;
            let samples = time_samples(config, || {
                let output = lhs.matmul(&rhs);
                async move {
                    output.materialize().await;
                    Ok(())
                }
            })
            .await?;
            Ok(sweep_report(
                "webgpu",
                case,
                size,
                config,
                samples,
                format!(
                    "{} @ {} f32",
                    shape_label(&lhs_shape),
                    shape_label(&rhs_shape)
                ),
            ))
        }
        "dense_batched_matmul" => {
            let batch = 4usize;
            let k = size.value + 32;
            let lhs_shape = [batch, size.value, k];
            let rhs_shape = [batch, k, size.value];
            let lhs: FusorTensor<3, f32> = FusorTensor::from_slice(
                device,
                lhs_shape,
                &deterministic_values(elements(&lhs_shape), 23, 0.004),
            );
            let rhs: FusorTensor<3, f32> = FusorTensor::from_slice(
                device,
                rhs_shape,
                &deterministic_values(elements(&rhs_shape), 24, 0.004),
            );
            materialize_inputs(&[&lhs, &rhs]).await;
            let samples = time_samples(config, || {
                let output = lhs.matmul(&rhs);
                async move {
                    output.materialize().await;
                    Ok(())
                }
            })
            .await?;
            Ok(sweep_report(
                "webgpu",
                case,
                size,
                config,
                samples,
                format!(
                    "{} @ {} f32",
                    shape_label(&lhs_shape),
                    shape_label(&rhs_shape)
                ),
            ))
        }
        "conv1d_small" => {
            let input_shape = [4usize, 8usize, size.value];
            let weight_shape = [16usize, 8usize, 5usize];
            let bias_shape = [16usize];
            let input: FusorTensor<3, f32> = FusorTensor::from_slice(
                device,
                input_shape,
                &deterministic_values(elements(&input_shape), 25, 0.01),
            );
            let weight: FusorTensor<3, f32> = FusorTensor::from_slice(
                device,
                weight_shape,
                &deterministic_values(elements(&weight_shape), 26, 0.01),
            );
            let bias: FusorTensor<1, f32> = FusorTensor::from_slice(
                device,
                bias_shape,
                &deterministic_values(elements(&bias_shape), 27, 0.001),
            );
            materialize_inputs(&[&input]).await;
            materialize_inputs(&[&weight]).await;
            materialize_inputs(&[&bias]).await;
            let samples = time_samples(config, || {
                let output = input.conv(&weight, Some(&bias), [2], [1]);
                async move {
                    output.materialize().await;
                    Ok(())
                }
            })
            .await?;
            Ok(sweep_report(
                "webgpu",
                case,
                size,
                config,
                samples,
                format!(
                    "{} conv {}",
                    shape_label(&input_shape),
                    shape_label(&weight_shape)
                ),
            ))
        }
        "top_k_large" | "top_k_qwen_vocab" => {
            let input_len = size.value;
            let k = if case == "top_k_qwen_vocab" { 40 } else { 64 };
            let input: FusorTensor<1, f32> =
                FusorTensor::from_slice(device, [input_len], &topk_values(input_len));
            materialize_inputs(&[&input]).await;
            let samples = time_samples(config, || async {
                let top = input.top_k_pairs(k).await?;
                if top.len() != k {
                    return Err(format!("top_k returned {} pairs, expected {k}", top.len()).into());
                }
                Ok(())
            })
            .await?;
            Ok(sweep_report(
                "webgpu",
                case,
                size,
                config,
                samples,
                format!("{input_len} logits, k={k}"),
            ))
        }
        "q8_0_qgemv" => {
            let weight_shape = q8_shape(size.value);
            let input_shape = [1usize, weight_shape[1]];
            let raw_bytes = q8_0_raw_bytes(weight_shape);
            let matrix: QMatrix =
                qmatrix_from_raw_bytes(device, weight_shape, &raw_bytes, GgmlType::Q8_0);
            let input: FusorTensor<2, f32> = FusorTensor::from_slice(
                device,
                input_shape,
                &deterministic_values(elements(&input_shape), 8, 0.003),
            );
            materialize_inputs(&[&input]).await;
            let samples = time_samples(config, || {
                let output = input.q_mat_mul(&matrix);
                async move {
                    output.materialize().await;
                    Ok(())
                }
            })
            .await?;
            Ok(sweep_report(
                "webgpu",
                case,
                size,
                config,
                samples,
                format!(
                    "1x{} @ Q8_0 {}",
                    weight_shape[1],
                    shape_label(&weight_shape)
                ),
            ))
        }
        "q4k_qgemv" => {
            let weight_shape = q4_shape(size.value);
            let input_shape = [1usize, weight_shape[1]];
            let raw_bytes = q4k_raw_bytes(weight_shape);
            let matrix: QMatrix =
                qmatrix_from_raw_bytes(device, weight_shape, &raw_bytes, GgmlType::Q4K);
            let input: FusorTensor<2, f32> = FusorTensor::from_slice(
                device,
                input_shape,
                &deterministic_values(elements(&input_shape), 29, 0.003),
            );
            materialize_inputs(&[&input]).await;
            let samples = time_samples(config, || {
                let output = input.q_mat_mul(&matrix);
                async move {
                    output.materialize().await;
                    Ok(())
                }
            })
            .await?;
            Ok(sweep_report(
                "webgpu",
                case,
                size,
                config,
                samples,
                format!("1x{} @ Q4K {}", weight_shape[1], shape_label(&weight_shape)),
            ))
        }
        "q4k_paired_silu" => {
            let weight_shape = [size.value * 2, 1024usize];
            let input_shape = [1usize, weight_shape[1]];
            let raw_bytes = q4k_raw_bytes(weight_shape);
            let matrix: QMatrix =
                qmatrix_from_raw_bytes(device, weight_shape, &raw_bytes, GgmlType::Q4K);
            let input: FusorTensor<2, f32> = FusorTensor::from_slice(
                device,
                input_shape,
                &deterministic_values(elements(&input_shape), 30, 0.003),
            );
            materialize_inputs(&[&input]).await;
            let samples = time_samples(config, || {
                let pair_len = weight_shape[0] / 2;
                let projected = input.q_mat_mul(&matrix);
                let gate = projected
                    .narrow(fusor::D::Minus1, 0, pair_len)
                    .to_concrete();
                let up = projected
                    .narrow(fusor::D::Minus1, pair_len, pair_len)
                    .to_concrete();
                let output = (gate.silu() * up).to_concrete();
                async move {
                    output.materialize().await;
                    Ok(())
                }
            })
            .await?;
            Ok(sweep_report(
                "webgpu",
                case,
                size,
                config,
                samples,
                format!("1x1024 @ paired Q4K {}", shape_label(&weight_shape)),
            ))
        }
        "attention_small" | "attention_causal_small" => {
            let seq_len = size.value;
            let shape = [1usize, 4usize, seq_len, 64usize];
            let q: FusorTensor<4, f32> = FusorTensor::from_slice(
                device,
                shape,
                &deterministic_values(elements(&shape), 31, 0.003),
            );
            let k: FusorTensor<4, f32> = FusorTensor::from_slice(
                device,
                shape,
                &deterministic_values(elements(&shape), 32, 0.003),
            );
            let v: FusorTensor<4, f32> = FusorTensor::from_slice(
                device,
                shape,
                &deterministic_values(elements(&shape), 33, 0.003),
            );
            let mask_shape = [seq_len, seq_len];
            let mask_values = vec![0.0f32; elements(&mask_shape)];
            let mask: FusorTensor<2, f32> =
                FusorTensor::from_slice(device, mask_shape, &mask_values);
            materialize_inputs(&[&q, &k, &v]).await;
            if case == "attention_causal_small" {
                materialize_inputs(&[&mask]).await;
            }
            let samples = time_samples(config, || {
                let mask_arg = if case == "attention_causal_small" {
                    Some((&mask, MaskKind::Causal))
                } else {
                    None
                };
                let output = q.attention(&k, &v, 1.0 / (64.0f32).sqrt(), mask_arg);
                async move {
                    output.materialize().await;
                    Ok(())
                }
            })
            .await?;
            Ok(sweep_report(
                "webgpu",
                case,
                size,
                config,
                samples,
                format!("{} attention", shape_label(&shape)),
            ))
        }
        "rope_fused_decode" => {
            let seq_len = size.value;
            let shape = [1usize, 8usize, seq_len, 64usize];
            let [batch, heads, _, head_dim] = shape;
            let pos_shape = [seq_len * 2, head_dim / 2];
            let cos_values = rope_values(pos_shape, head_dim, true);
            let sin_values = rope_values(pos_shape, head_dim, false);
            let input: FusorTensor<4, f32> = FusorTensor::from_slice(
                device,
                shape,
                &deterministic_values(batch * heads * seq_len * head_dim, 9, 0.01),
            );
            let cos: FusorTensor<2, f32> = FusorTensor::from_slice(device, pos_shape, &cos_values);
            let sin: FusorTensor<2, f32> = FusorTensor::from_slice(device, pos_shape, &sin_values);
            materialize_inputs(&[&input]).await;
            materialize_inputs(&[&cos, &sin]).await;
            let samples = time_samples(config, || {
                let output = input.rope_fused(&cos, &sin);
                async move {
                    output.materialize().await;
                    Ok(())
                }
            })
            .await?;
            Ok(sweep_report(
                "webgpu",
                case,
                size,
                config,
                samples,
                format!("{} fused rope", shape_label(&shape)),
            ))
        }
        _ => Err(format!("unknown WebGPU benchmark sweep: {case}").into()),
    }
}

fn topk_values(input_len: usize) -> Vec<f32> {
    (0..input_len)
        .map(|index| {
            let base = ((index * 67 + 29) % 10_007) as f32 * 0.001;
            let bump = if index % 4099 == 0 { 20.0 } else { 0.0 };
            base + bump - (index % 13) as f32 * 0.0001
        })
        .collect()
}

fn rope_values(shape: [usize; 2], head_dim: usize, cos: bool) -> Vec<f32> {
    (0..shape[0])
        .flat_map(|i| {
            (0..shape[1]).map(move |j| {
                let value = (i as f32) / 10000f32.powf((2 * (j / 2)) as f32 / head_dim as f32);
                if cos { value.cos() } else { value.sin() }
            })
        })
        .collect()
}

#[cfg(feature = "burn-bench")]
type BurnTensor<const R: usize> = ::burn::tensor::Tensor<::burn::backend::Wgpu, R>;

#[cfg(feature = "burn-bench")]
fn burn_tensor<const R: usize>(
    values: Vec<f32>,
    shape: [usize; R],
    device: &::burn::backend::wgpu::WgpuDevice,
) -> BurnTensor<R> {
    ::burn::tensor::Tensor::<::burn::backend::Wgpu, R>::from_data(
        ::burn::tensor::TensorData::new(values, shape),
        device,
    )
}

#[cfg(feature = "burn-bench")]
async fn burn_materialize<const R: usize>(tensor: BurnTensor<R>) -> BenchmarkResult<()> {
    let _ = tensor.into_data_async().await;
    Ok(())
}

#[cfg(feature = "burn-bench")]
async fn burn_materialize_inputs<const R: usize>(inputs: &[BurnTensor<R>]) -> BenchmarkResult<()> {
    for input in inputs {
        burn_materialize(input.clone()).await?;
    }
    Ok(())
}

#[cfg(feature = "burn-bench")]
async fn run_burn_case(
    case: &str,
    size: BenchmarkSweepSize,
    config: BenchmarkConfig,
) -> BenchmarkResult<BenchmarkReport> {
    use ::burn::{
        backend::Wgpu,
        nn::{LayerNormConfig, RmsNormConfig, RotaryEncodingConfig},
        tensor::{
            activation, module,
            ops::{AttentionModuleOptions, ConvOptions},
        },
    };

    let device = crate::bench::burn::initialized_device().await;
    match case {
        "elementwise_add_square" => {
            let shape = [size.value, size.value];
            let lhs = burn_tensor(
                deterministic_values(elements(&shape), 1, 0.01),
                shape,
                &device,
            );
            let rhs = burn_tensor(
                deterministic_values(elements(&shape), 2, 0.008),
                shape,
                &device,
            );
            burn_materialize_inputs(&[lhs.clone(), rhs.clone()]).await?;
            let samples = time_samples(config, || {
                let output = lhs.clone() + rhs.clone();
                async move { burn_materialize(output).await }
            })
            .await?;
            Ok(sweep_report(
                "burn",
                case,
                size,
                config,
                samples,
                format!("{} f32 add", shape_label(&shape)),
            ))
        }
        "elementwise_mul_rank4" => {
            let shape = rank4_shape(size.value);
            let lhs = burn_tensor(
                deterministic_values(elements(&shape), 3, 0.012),
                shape,
                &device,
            );
            let rhs = burn_tensor(
                deterministic_values(elements(&shape), 4, 0.009),
                shape,
                &device,
            );
            burn_materialize_inputs(&[lhs.clone(), rhs.clone()]).await?;
            let samples = time_samples(config, || {
                let output = lhs.clone() * rhs.clone();
                async move { burn_materialize(output).await }
            })
            .await?;
            Ok(sweep_report(
                "burn",
                case,
                size,
                config,
                samples,
                format!("{} f32 mul", shape_label(&shape)),
            ))
        }
        "unary_trig_chain" => {
            let shape = [size.value, size.value];
            let input = burn_tensor(
                deterministic_values(elements(&shape), 10, 0.01),
                shape,
                &device,
            );
            burn_materialize_inputs(std::slice::from_ref(&input)).await?;
            let samples = time_samples(config, || {
                let output = input.clone().sin() + input.clone().cos();
                async move { burn_materialize(output).await }
            })
            .await?;
            Ok(sweep_report(
                "burn",
                case,
                size,
                config,
                samples,
                format!("{} sin+cos", shape_label(&shape)),
            ))
        }
        "activation_gelu" => {
            let shape = [size.value, size.value];
            let input = burn_tensor(
                deterministic_values(elements(&shape), 11, 0.015),
                shape,
                &device,
            );
            burn_materialize_inputs(std::slice::from_ref(&input)).await?;
            let samples = time_samples(config, || {
                let output = activation::gelu(input.clone());
                async move { burn_materialize(output).await }
            })
            .await?;
            Ok(sweep_report(
                "burn",
                case,
                size,
                config,
                samples,
                format!("{} gelu", shape_label(&shape)),
            ))
        }
        "broadcast_add" => {
            let matrix_shape = [size.value, 512usize];
            let vector_shape = [512usize];
            let matrix = burn_tensor(
                deterministic_values(elements(&matrix_shape), 12, 0.006),
                matrix_shape,
                &device,
            );
            let vector = burn_tensor(
                deterministic_values(elements(&vector_shape), 13, 0.01),
                vector_shape,
                &device,
            );
            burn_materialize_inputs(std::slice::from_ref(&matrix)).await?;
            burn_materialize_inputs(std::slice::from_ref(&vector)).await?;
            let samples = time_samples(config, || {
                let output = matrix.clone() + vector.clone().reshape([1, vector_shape[0]]);
                async move { burn_materialize(output).await }
            })
            .await?;
            Ok(sweep_report(
                "burn",
                case,
                size,
                config,
                samples,
                format!("{} + broadcast 512", shape_label(&matrix_shape)),
            ))
        }
        "transpose_then_elementwise" => {
            let shape = [size.value, size.value + size.value / 2];
            let input = burn_tensor(
                deterministic_values(elements(&shape), 14, 0.01),
                shape,
                &device,
            );
            burn_materialize_inputs(std::slice::from_ref(&input)).await?;
            let samples = time_samples(config, || {
                let transposed = input.clone().transpose();
                let output = transposed.clone() * transposed;
                async move { burn_materialize(output).await }
            })
            .await?;
            Ok(sweep_report(
                "burn",
                case,
                size,
                config,
                samples,
                format!("{} transpose, square", shape_label(&shape)),
            ))
        }
        "reduction_sum_last_dim" => {
            let shape = [size.value, 512usize];
            let input = burn_tensor(
                deterministic_values(elements(&shape), 15, 0.004),
                shape,
                &device,
            );
            burn_materialize_inputs(std::slice::from_ref(&input)).await?;
            let samples = time_samples(config, || {
                let output = input.clone().sum_dim(1);
                async move { burn_materialize(output).await }
            })
            .await?;
            Ok(sweep_report(
                "burn",
                case,
                size,
                config,
                samples,
                format!("{} sum axis 1", shape_label(&shape)),
            ))
        }
        "reduction_max_middle_axis" => {
            let shape = middle_shape(size.value);
            let input = burn_tensor(
                deterministic_values(elements(&shape), 16, 0.004),
                shape,
                &device,
            );
            burn_materialize_inputs(std::slice::from_ref(&input)).await?;
            let samples = time_samples(config, || {
                let output = input.clone().max_dim(1);
                async move { burn_materialize(output).await }
            })
            .await?;
            Ok(sweep_report(
                "burn",
                case,
                size,
                config,
                samples,
                format!("{} max axis 1", shape_label(&shape)),
            ))
        }
        "softmax_last_dim" => {
            let shape = [size.value, 256usize];
            let input = burn_tensor(
                deterministic_values(elements(&shape), 5, 0.006),
                shape,
                &device,
            );
            burn_materialize_inputs(std::slice::from_ref(&input)).await?;
            let samples = time_samples(config, || {
                let output = activation::softmax(input.clone(), 1);
                async move { burn_materialize(output).await }
            })
            .await?;
            Ok(sweep_report(
                "burn",
                case,
                size,
                config,
                samples,
                format!("{} last-axis softmax", shape_label(&shape)),
            ))
        }
        "softmax_middle_axis" => {
            let shape = middle_shape(size.value);
            let input = burn_tensor(
                deterministic_values(elements(&shape), 17, 0.004),
                shape,
                &device,
            );
            burn_materialize_inputs(std::slice::from_ref(&input)).await?;
            let samples = time_samples(config, || {
                let output = activation::softmax(input.clone(), 1);
                async move { burn_materialize(output).await }
            })
            .await?;
            Ok(sweep_report(
                "burn",
                case,
                size,
                config,
                samples,
                format!("{} softmax axis 1", shape_label(&shape)),
            ))
        }
        "layer_norm_last_dim" => {
            let shape = [4usize, size.value, 512usize];
            let last_dim = shape[2];
            let input = burn_tensor(
                deterministic_values(elements(&shape), 18, 0.01),
                shape,
                &device,
            );
            let layer = LayerNormConfig::new(last_dim)
                .with_epsilon(1.0e-5)
                .init::<Wgpu>(&device);
            burn_materialize_inputs(std::slice::from_ref(&input)).await?;
            let samples = time_samples(config, || {
                let output = layer.clone().forward(input.clone());
                async move { burn_materialize(output).await }
            })
            .await?;
            Ok(sweep_report(
                "burn",
                case,
                size,
                config,
                samples,
                format!("{} layer norm", shape_label(&shape)),
            ))
        }
        "rms_norm_fused" => {
            let shape = [4usize, size.value, 512usize];
            let last_dim = shape[2];
            let input = burn_tensor(
                deterministic_values(elements(&shape), 21, 0.01),
                shape,
                &device,
            );
            let rms = RmsNormConfig::new(last_dim)
                .with_epsilon(1.0e-5)
                .init::<Wgpu>(&device);
            burn_materialize_inputs(std::slice::from_ref(&input)).await?;
            let samples = time_samples(config, || {
                let output = rms.clone().forward(input.clone());
                async move { burn_materialize(output).await }
            })
            .await?;
            Ok(sweep_report(
                "burn",
                case,
                size,
                config,
                samples,
                format!("{} rms norm", shape_label(&shape)),
            ))
        }
        "dense_matmul_square" => {
            let lhs_shape = [size.value, size.value];
            let rhs_shape = [size.value, size.value];
            let lhs = burn_tensor(
                deterministic_values(elements(&lhs_shape), 6, 0.004),
                lhs_shape,
                &device,
            );
            let rhs = burn_tensor(
                deterministic_values(elements(&rhs_shape), 7, 0.004),
                rhs_shape,
                &device,
            );
            burn_materialize_inputs(&[lhs.clone(), rhs.clone()]).await?;
            let samples = time_samples(config, || {
                let output = lhs.clone().matmul(rhs.clone());
                async move { burn_materialize(output).await }
            })
            .await?;
            Ok(sweep_report(
                "burn",
                case,
                size,
                config,
                samples,
                format!(
                    "{} @ {} f32",
                    shape_label(&lhs_shape),
                    shape_label(&rhs_shape)
                ),
            ))
        }
        "dense_batched_matmul" => {
            let batch = 4usize;
            let k = size.value + 32;
            let lhs_shape = [batch, size.value, k];
            let rhs_shape = [batch, k, size.value];
            let lhs = burn_tensor(
                deterministic_values(elements(&lhs_shape), 23, 0.004),
                lhs_shape,
                &device,
            );
            let rhs = burn_tensor(
                deterministic_values(elements(&rhs_shape), 24, 0.004),
                rhs_shape,
                &device,
            );
            burn_materialize_inputs(&[lhs.clone(), rhs.clone()]).await?;
            let samples = time_samples(config, || {
                let output = lhs.clone().matmul(rhs.clone());
                async move { burn_materialize(output).await }
            })
            .await?;
            Ok(sweep_report(
                "burn",
                case,
                size,
                config,
                samples,
                format!(
                    "{} @ {} f32",
                    shape_label(&lhs_shape),
                    shape_label(&rhs_shape)
                ),
            ))
        }
        "conv1d_small" => {
            let input_shape = [4usize, 8usize, size.value];
            let weight_shape = [16usize, 8usize, 5usize];
            let bias_shape = [16usize];
            let input = burn_tensor(
                deterministic_values(elements(&input_shape), 25, 0.01),
                input_shape,
                &device,
            );
            let weight = burn_tensor(
                deterministic_values(elements(&weight_shape), 26, 0.01),
                weight_shape,
                &device,
            );
            let bias = burn_tensor(
                deterministic_values(elements(&bias_shape), 27, 0.001),
                bias_shape,
                &device,
            );
            burn_materialize_inputs(std::slice::from_ref(&input)).await?;
            burn_materialize_inputs(std::slice::from_ref(&weight)).await?;
            burn_materialize_inputs(std::slice::from_ref(&bias)).await?;
            let samples = time_samples(config, || {
                let output = module::conv1d(
                    input.clone(),
                    weight.clone(),
                    Some(bias.clone()),
                    ConvOptions::new([2], [1], [1], 1),
                );
                async move { burn_materialize(output).await }
            })
            .await?;
            Ok(sweep_report(
                "burn",
                case,
                size,
                config,
                samples,
                format!(
                    "{} conv {}",
                    shape_label(&input_shape),
                    shape_label(&weight_shape)
                ),
            ))
        }
        "top_k_large" | "top_k_qwen_vocab" => {
            let input_len = size.value;
            let k = if case == "top_k_qwen_vocab" { 40 } else { 64 };
            let input = burn_tensor(topk_values(input_len), [input_len], &device);
            burn_materialize_inputs(std::slice::from_ref(&input)).await?;
            let samples = time_samples(config, || {
                let output = input.clone().topk(k, 0);
                async move { burn_materialize(output).await }
            })
            .await?;
            Ok(sweep_report(
                "burn",
                case,
                size,
                config,
                samples,
                format!("{input_len} logits, k={k}"),
            ))
        }
        "q8_0_qgemv" | "q4k_qgemv" => {
            let weight_shape = if case == "q8_0_qgemv" {
                q8_shape(size.value)
            } else {
                q4_shape(size.value)
            };
            let input_shape = [1usize, weight_shape[1]];
            let dense_weight_shape = [weight_shape[1], weight_shape[0]];
            let input = burn_tensor(
                deterministic_values(
                    elements(&input_shape),
                    if case == "q8_0_qgemv" { 8 } else { 29 },
                    0.003,
                ),
                input_shape,
                &device,
            );
            let weights = burn_tensor(
                deterministic_values(
                    elements(&dense_weight_shape),
                    if case == "q8_0_qgemv" { 80 } else { 81 },
                    0.003,
                ),
                dense_weight_shape,
                &device,
            );
            burn_materialize_inputs(std::slice::from_ref(&input)).await?;
            burn_materialize_inputs(std::slice::from_ref(&weights)).await?;
            let samples = time_samples(config, || {
                let output = input.clone().matmul(weights.clone());
                async move { burn_materialize(output).await }
            })
            .await?;
            Ok(sweep_report(
                "burn",
                case,
                size,
                config,
                samples,
                format!(
                    "1x{} @ dense f32 {}",
                    weight_shape[1],
                    shape_label(&dense_weight_shape)
                ),
            ))
        }
        "q4k_paired_silu" => {
            let weight_shape = [size.value * 2, 1024usize];
            let input_shape = [1usize, weight_shape[1]];
            let dense_weight_shape = [weight_shape[1], weight_shape[0]];
            let input = burn_tensor(
                deterministic_values(elements(&input_shape), 30, 0.003),
                input_shape,
                &device,
            );
            let weights = burn_tensor(
                deterministic_values(elements(&dense_weight_shape), 82, 0.003),
                dense_weight_shape,
                &device,
            );
            burn_materialize_inputs(std::slice::from_ref(&input)).await?;
            burn_materialize_inputs(std::slice::from_ref(&weights)).await?;
            let samples = time_samples(config, || {
                let projected = input.clone().matmul(weights.clone());
                let gate = projected.clone().narrow(1, 0, size.value);
                let up = projected.narrow(1, size.value, size.value);
                let output = activation::silu(gate) * up;
                async move { burn_materialize(output).await }
            })
            .await?;
            Ok(sweep_report(
                "burn",
                case,
                size,
                config,
                samples,
                format!(
                    "1x1024 @ dense f32 {} + paired SiLU",
                    shape_label(&dense_weight_shape)
                ),
            ))
        }
        "attention_small" | "attention_causal_small" => {
            let seq_len = size.value;
            let shape = [1usize, 4usize, seq_len, 64usize];
            let q = burn_tensor(
                deterministic_values(elements(&shape), 31, 0.003),
                shape,
                &device,
            );
            let k = burn_tensor(
                deterministic_values(elements(&shape), 32, 0.003),
                shape,
                &device,
            );
            let v = burn_tensor(
                deterministic_values(elements(&shape), 33, 0.003),
                shape,
                &device,
            );
            burn_materialize_inputs(&[q.clone(), k.clone(), v.clone()]).await?;
            let samples = time_samples(config, || {
                let output = module::attention(
                    q.clone(),
                    k.clone(),
                    v.clone(),
                    None,
                    None,
                    AttentionModuleOptions {
                        scale: Some(1.0 / (64.0f64).sqrt()),
                        softcap: None,
                        is_causal: case == "attention_causal_small",
                    },
                );
                async move { burn_materialize(output).await }
            })
            .await?;
            Ok(sweep_report(
                "burn",
                case,
                size,
                config,
                samples,
                format!("{} attention", shape_label(&shape)),
            ))
        }
        "rope_fused_decode" => {
            let seq_len = size.value;
            let shape = [1usize, 8usize, seq_len, 64usize];
            let [batch, heads, _, head_dim] = shape;
            let input = burn_tensor(
                deterministic_values(batch * heads * seq_len * head_dim, 9, 0.01),
                shape,
                &device,
            );
            let rope = RotaryEncodingConfig::new(seq_len * 2, head_dim).init::<Wgpu>(&device);
            burn_materialize_inputs(std::slice::from_ref(&input)).await?;
            let samples = time_samples(config, || {
                let output = rope.clone().forward(input.clone());
                async move { burn_materialize(output).await }
            })
            .await?;
            Ok(sweep_report(
                "burn",
                case,
                size,
                config,
                samples,
                format!("{} rotary encoding", shape_label(&shape)),
            ))
        }
        _ => Err(format!("unknown Burn benchmark sweep: {case}").into()),
    }
}
