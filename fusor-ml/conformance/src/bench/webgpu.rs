//! Concrete WebGPU benchmark cases.

use fusor::{Device, GgmlType, MaskKind, QMatrix, Tensor};

use crate::{
    bench::{BenchmarkCase, BenchmarkConfig, BenchmarkEvent, BenchmarkReport, BenchmarkResult},
    common::quantized::{q4k_raw_bytes, q8_0_raw_bytes, qmatrix_from_raw_bytes},
};

use super::time_samples;

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

async fn materialize_inputs<const R: usize>(inputs: &[&Tensor<R, f32>]) {
    for input in inputs {
        input.materialize().await;
    }
}

fn bench_case(
    name: &'static str,
    run: impl for<'a> FnOnce(&'a Device, BenchmarkConfig) -> super::CaseFuture<'a> + 'static,
) -> BenchmarkCase {
    BenchmarkCase::new(name, run)
}

pub async fn run_webgpu_bench_suite(device: &Device) -> BenchmarkResult<Vec<BenchmarkReport>> {
    run_webgpu_bench_suite_with_progress(device, BenchmarkConfig::default(), |_| {}).await
}

pub async fn run_webgpu_bench_suite_with_progress(
    device: &Device,
    config: BenchmarkConfig,
    progress: impl FnMut(BenchmarkEvent),
) -> BenchmarkResult<Vec<BenchmarkReport>> {
    crate::bench::registry::run_cases(device, config, crate::bench::registry::cases(), progress)
        .await
}

pub fn elementwise_add_square() -> BenchmarkCase {
    bench_case("webgpu::elementwise_add_square", |device, config| {
        Box::pin(async move {
            let shape = [512usize, 512usize];
            let lhs_values = deterministic_values(elements(&shape), 1, 0.01);
            let rhs_values = deterministic_values(elements(&shape), 2, 0.008);
            let lhs: Tensor<2, f32> = Tensor::from_slice(device, shape, &lhs_values);
            let rhs: Tensor<2, f32> = Tensor::from_slice(device, shape, &rhs_values);
            materialize_inputs(&[&lhs, &rhs]).await;

            let samples = time_samples(config, || {
                let output = (&lhs + &rhs).to_concrete();
                async move {
                    output.materialize().await;
                    Ok(())
                }
            })
            .await?;

            Ok(BenchmarkReport::new(
                "webgpu::elementwise_add_square",
                config,
                samples,
                format!("{} f32 add", shape_label(&shape)),
            ))
        })
    })
}

pub fn elementwise_mul_rank4() -> BenchmarkCase {
    bench_case("webgpu::elementwise_mul_rank4", |device, config| {
        Box::pin(async move {
            let shape = [9usize, 11usize, 32usize, 16usize];
            let lhs_values = deterministic_values(elements(&shape), 3, 0.012);
            let rhs_values = deterministic_values(elements(&shape), 4, 0.009);
            let lhs: Tensor<4, f32> = Tensor::from_slice(device, shape, &lhs_values);
            let rhs: Tensor<4, f32> = Tensor::from_slice(device, shape, &rhs_values);
            materialize_inputs(&[&lhs, &rhs]).await;

            let samples = time_samples(config, || {
                let output = (&lhs * &rhs).to_concrete();
                async move {
                    output.materialize().await;
                    Ok(())
                }
            })
            .await?;

            Ok(BenchmarkReport::new(
                "webgpu::elementwise_mul_rank4",
                config,
                samples,
                format!("{} f32 mul", shape_label(&shape)),
            ))
        })
    })
}

pub fn unary_trig_chain() -> BenchmarkCase {
    bench_case("webgpu::unary_trig_chain", |device, config| {
        Box::pin(async move {
            let shape = [384usize, 384usize];
            let values = deterministic_values(elements(&shape), 10, 0.01);
            let input: Tensor<2, f32> = Tensor::from_slice(device, shape, &values);
            materialize_inputs(&[&input]).await;

            let samples = time_samples(config, || {
                let output = (input.sin() + input.cos()).to_concrete();
                async move {
                    output.materialize().await;
                    Ok(())
                }
            })
            .await?;

            Ok(BenchmarkReport::new(
                "webgpu::unary_trig_chain",
                config,
                samples,
                format!("{} sin+cos", shape_label(&shape)),
            ))
        })
    })
}

pub fn activation_gelu() -> BenchmarkCase {
    bench_case("webgpu::activation_gelu", |device, config| {
        Box::pin(async move {
            let shape = [512usize, 256usize];
            let values = deterministic_values(elements(&shape), 11, 0.015);
            let input: Tensor<2, f32> = Tensor::from_slice(device, shape, &values);
            materialize_inputs(&[&input]).await;

            let samples = time_samples(config, || {
                let output = input.gelu();
                async move {
                    output.materialize().await;
                    Ok(())
                }
            })
            .await?;

            Ok(BenchmarkReport::new(
                "webgpu::activation_gelu",
                config,
                samples,
                format!("{} gelu", shape_label(&shape)),
            ))
        })
    })
}

pub fn broadcast_add() -> BenchmarkCase {
    bench_case("webgpu::broadcast_add", |device, config| {
        Box::pin(async move {
            let matrix_shape = [256usize, 512usize];
            let vector_shape = [512usize];
            let matrix_values = deterministic_values(elements(&matrix_shape), 12, 0.006);
            let vector_values = deterministic_values(elements(&vector_shape), 13, 0.01);
            let matrix: Tensor<2, f32> = Tensor::from_slice(device, matrix_shape, &matrix_values);
            let vector: Tensor<1, f32> = Tensor::from_slice(device, vector_shape, &vector_values);
            materialize_inputs(&[&matrix]).await;
            materialize_inputs(&[&vector]).await;

            let samples = time_samples(config, || {
                let vector_row = vector.reshape([1, vector_shape[0]]);
                let vector_broadcast = vector_row.broadcast_as(matrix_shape);
                let output = (&matrix + vector_broadcast).to_concrete();
                async move {
                    output.materialize().await;
                    Ok(())
                }
            })
            .await?;

            Ok(BenchmarkReport::new(
                "webgpu::broadcast_add",
                config,
                samples,
                "256x512 + broadcast 512",
            ))
        })
    })
}

pub fn transpose_then_elementwise() -> BenchmarkCase {
    bench_case("webgpu::transpose_then_elementwise", |device, config| {
        Box::pin(async move {
            let shape = [256usize, 384usize];
            let values = deterministic_values(elements(&shape), 14, 0.01);
            let input: Tensor<2, f32> = Tensor::from_slice(device, shape, &values);
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

            Ok(BenchmarkReport::new(
                "webgpu::transpose_then_elementwise",
                config,
                samples,
                "256x384 transpose, square",
            ))
        })
    })
}

pub fn reduction_sum_last_dim() -> BenchmarkCase {
    bench_case("webgpu::reduction_sum_last_dim", |device, config| {
        Box::pin(async move {
            let shape = [256usize, 512usize];
            let values = deterministic_values(elements(&shape), 15, 0.004);
            let input: Tensor<2, f32> = Tensor::from_slice(device, shape, &values);
            materialize_inputs(&[&input]).await;

            let samples = time_samples(config, || {
                let output = input.sum::<1>(1);
                async move {
                    output.materialize().await;
                    Ok(())
                }
            })
            .await?;

            Ok(BenchmarkReport::new(
                "webgpu::reduction_sum_last_dim",
                config,
                samples,
                format!("{} sum axis 1", shape_label(&shape)),
            ))
        })
    })
}

pub fn reduction_max_middle_axis() -> BenchmarkCase {
    bench_case("webgpu::reduction_max_middle_axis", |device, config| {
        Box::pin(async move {
            let shape = [64usize, 128usize, 64usize];
            let values = deterministic_values(elements(&shape), 16, 0.004);
            let input: Tensor<3, f32> = Tensor::from_slice(device, shape, &values);
            materialize_inputs(&[&input]).await;

            let samples = time_samples(config, || {
                let output = input.max::<2>(1);
                async move {
                    output.materialize().await;
                    Ok(())
                }
            })
            .await?;

            Ok(BenchmarkReport::new(
                "webgpu::reduction_max_middle_axis",
                config,
                samples,
                format!("{} max axis 1", shape_label(&shape)),
            ))
        })
    })
}

pub fn softmax_last_dim() -> BenchmarkCase {
    bench_case("webgpu::softmax_last_dim", |device, config| {
        Box::pin(async move {
            let shape = [512usize, 256usize];
            let values = deterministic_values(elements(&shape), 5, 0.006);
            let input: Tensor<2, f32> = Tensor::from_slice(device, shape, &values);
            materialize_inputs(&[&input]).await;

            let samples = time_samples(config, || {
                let output = input.softmax_last_dim::<1>();
                async move {
                    output.materialize().await;
                    Ok(())
                }
            })
            .await?;

            Ok(BenchmarkReport::new(
                "webgpu::softmax_last_dim",
                config,
                samples,
                format!("{} last-axis softmax", shape_label(&shape)),
            ))
        })
    })
}

pub fn softmax_middle_axis() -> BenchmarkCase {
    bench_case("webgpu::softmax_middle_axis", |device, config| {
        Box::pin(async move {
            let shape = [32usize, 128usize, 64usize];
            let values = deterministic_values(elements(&shape), 17, 0.004);
            let input: Tensor<3, f32> = Tensor::from_slice(device, shape, &values);
            materialize_inputs(&[&input]).await;

            let samples = time_samples(config, || {
                let output = input.softmax::<2>(1);
                async move {
                    output.materialize().await;
                    Ok(())
                }
            })
            .await?;

            Ok(BenchmarkReport::new(
                "webgpu::softmax_middle_axis",
                config,
                samples,
                format!("{} softmax axis 1", shape_label(&shape)),
            ))
        })
    })
}

pub fn layer_norm_last_dim() -> BenchmarkCase {
    bench_case("webgpu::layer_norm_last_dim", |device, config| {
        Box::pin(async move {
            let shape = [8usize, 128usize, 512usize];
            let last_dim = shape[2];
            let values = deterministic_values(elements(&shape), 18, 0.01);
            let weight_values = deterministic_values(last_dim, 19, 0.002)
                .into_iter()
                .map(|value| value + 1.0)
                .collect::<Vec<_>>();
            let bias_values = deterministic_values(last_dim, 20, 0.001);
            let input: Tensor<3, f32> = Tensor::from_slice(device, shape, &values);
            let weight: Tensor<1, f32> = Tensor::from_slice(device, [last_dim], &weight_values);
            let bias: Tensor<1, f32> = Tensor::from_slice(device, [last_dim], &bias_values);
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

            Ok(BenchmarkReport::new(
                "webgpu::layer_norm_last_dim",
                config,
                samples,
                format!("{} layer norm", shape_label(&shape)),
            ))
        })
    })
}

pub fn rms_norm_fused() -> BenchmarkCase {
    bench_case("webgpu::rms_norm_fused", |device, config| {
        Box::pin(async move {
            let shape = [8usize, 128usize, 512usize];
            let last_dim = shape[2];
            let values = deterministic_values(elements(&shape), 21, 0.01);
            let weight_values = deterministic_values(last_dim, 22, 0.002)
                .into_iter()
                .map(|value| value + 1.0)
                .collect::<Vec<_>>();
            let input: Tensor<3, f32> = Tensor::from_slice(device, shape, &values);
            let weight: Tensor<1, f32> = Tensor::from_slice(device, [last_dim], &weight_values);
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

            Ok(BenchmarkReport::new(
                "webgpu::rms_norm_fused",
                config,
                samples,
                format!("{} rms norm", shape_label(&shape)),
            ))
        })
    })
}

pub fn dense_matmul_square() -> BenchmarkCase {
    bench_case("webgpu::dense_matmul_square", |device, config| {
        Box::pin(async move {
            let lhs_shape = [256usize, 256usize];
            let rhs_shape = [256usize, 256usize];
            let lhs_values = deterministic_values(elements(&lhs_shape), 6, 0.004);
            let rhs_values = deterministic_values(elements(&rhs_shape), 7, 0.004);
            let lhs: Tensor<2, f32> = Tensor::from_slice(device, lhs_shape, &lhs_values);
            let rhs: Tensor<2, f32> = Tensor::from_slice(device, rhs_shape, &rhs_values);
            materialize_inputs(&[&lhs, &rhs]).await;

            let samples = time_samples(config, || {
                let output = lhs.matmul(&rhs);
                async move {
                    output.materialize().await;
                    Ok(())
                }
            })
            .await?;

            Ok(BenchmarkReport::new(
                "webgpu::dense_matmul_square",
                config,
                samples,
                "256x256 @ 256x256 f32",
            ))
        })
    })
}

pub fn dense_batched_matmul() -> BenchmarkCase {
    bench_case("webgpu::dense_batched_matmul", |device, config| {
        Box::pin(async move {
            let lhs_shape = [8usize, 64usize, 96usize];
            let rhs_shape = [8usize, 96usize, 64usize];
            let lhs_values = deterministic_values(elements(&lhs_shape), 23, 0.004);
            let rhs_values = deterministic_values(elements(&rhs_shape), 24, 0.004);
            let lhs: Tensor<3, f32> = Tensor::from_slice(device, lhs_shape, &lhs_values);
            let rhs: Tensor<3, f32> = Tensor::from_slice(device, rhs_shape, &rhs_values);
            materialize_inputs(&[&lhs, &rhs]).await;

            let samples = time_samples(config, || {
                let output = lhs.matmul(&rhs);
                async move {
                    output.materialize().await;
                    Ok(())
                }
            })
            .await?;

            Ok(BenchmarkReport::new(
                "webgpu::dense_batched_matmul",
                config,
                samples,
                "8x64x96 @ 8x96x64 f32",
            ))
        })
    })
}

pub fn conv1d_small() -> BenchmarkCase {
    bench_case("webgpu::conv1d_small", |device, config| {
        Box::pin(async move {
            let input_shape = [4usize, 8usize, 256usize];
            let weight_shape = [16usize, 8usize, 5usize];
            let bias_shape = [16usize];
            let input_values = deterministic_values(elements(&input_shape), 25, 0.01);
            let weight_values = deterministic_values(elements(&weight_shape), 26, 0.01);
            let bias_values = deterministic_values(elements(&bias_shape), 27, 0.001);
            let input: Tensor<3, f32> = Tensor::from_slice(device, input_shape, &input_values);
            let weight: Tensor<3, f32> = Tensor::from_slice(device, weight_shape, &weight_values);
            let bias: Tensor<1, f32> = Tensor::from_slice(device, bias_shape, &bias_values);
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

            Ok(BenchmarkReport::new(
                "webgpu::conv1d_small",
                config,
                samples,
                "4x8x256 conv 16x8x5",
            ))
        })
    })
}

pub fn top_k_large() -> BenchmarkCase {
    bench_case("webgpu::top_k_large", |device, config| {
        Box::pin(async move {
            let input_len = 65_537usize;
            let k = 64usize;
            let values = (0..input_len)
                .map(|index| {
                    let base = ((index * 67 + 29) % 10_007) as f32 * 0.001;
                    let bump = if index % 4099 == 0 { 20.0 } else { 0.0 };
                    base + bump - (index % 13) as f32 * 0.0001
                })
                .collect::<Vec<_>>();
            let input: Tensor<1, f32> = Tensor::from_slice(device, [input_len], &values);
            materialize_inputs(&[&input]).await;

            let samples = time_samples(config, || async {
                let top = input.top_k_pairs(k).await?;
                if top.len() != k {
                    return Err(format!("top_k returned {} pairs, expected {k}", top.len()).into());
                }
                Ok(())
            })
            .await?;

            Ok(BenchmarkReport::new(
                "webgpu::top_k_large",
                config,
                samples,
                format!("{input_len} logits, k={k}"),
            ))
        })
    })
}

pub fn top_k_qwen_vocab() -> BenchmarkCase {
    bench_case("webgpu::top_k_qwen_vocab", |device, config| {
        Box::pin(async move {
            let input_len = 151_936usize;
            let k = 40usize;
            let values = deterministic_values(input_len, 28, 0.01);
            let input: Tensor<1, f32> = Tensor::from_slice(device, [input_len], &values);
            materialize_inputs(&[&input]).await;

            let samples = time_samples(config, || async {
                let top = input.top_k_pairs(k).await?;
                if top.len() != k {
                    return Err(format!("top_k returned {} pairs, expected {k}", top.len()).into());
                }
                Ok(())
            })
            .await?;

            Ok(BenchmarkReport::new(
                "webgpu::top_k_qwen_vocab",
                config,
                samples,
                format!("{input_len} logits, k={k}"),
            ))
        })
    })
}

pub fn q8_0_qgemv() -> BenchmarkCase {
    bench_case("webgpu::q8_0_qgemv", |device, config| {
        Box::pin(async move {
            let weight_shape = [4096usize, 896usize];
            let input_shape = [1usize, weight_shape[1]];
            let raw_bytes = q8_0_raw_bytes(weight_shape);
            let matrix: QMatrix =
                qmatrix_from_raw_bytes(device, weight_shape, &raw_bytes, GgmlType::Q8_0);
            let input_values = deterministic_values(elements(&input_shape), 8, 0.003);
            let input: Tensor<2, f32> = Tensor::from_slice(device, input_shape, &input_values);
            materialize_inputs(&[&input]).await;

            let samples = time_samples(config, || {
                let output = input.q_mat_mul(&matrix);
                async move {
                    output.materialize().await;
                    Ok(())
                }
            })
            .await?;

            Ok(BenchmarkReport::new(
                "webgpu::q8_0_qgemv",
                config,
                samples,
                "1x896 @ Q8_0 4096x896",
            ))
        })
    })
}

pub fn q4k_qgemv() -> BenchmarkCase {
    bench_case("webgpu::q4k_qgemv", |device, config| {
        Box::pin(async move {
            let weight_shape = [2048usize, 1024usize];
            let input_shape = [1usize, weight_shape[1]];
            let raw_bytes = q4k_raw_bytes(weight_shape);
            let matrix: QMatrix =
                qmatrix_from_raw_bytes(device, weight_shape, &raw_bytes, GgmlType::Q4K);
            let input_values = deterministic_values(elements(&input_shape), 29, 0.003);
            let input: Tensor<2, f32> = Tensor::from_slice(device, input_shape, &input_values);
            materialize_inputs(&[&input]).await;

            let samples = time_samples(config, || {
                let output = input.q_mat_mul(&matrix);
                async move {
                    output.materialize().await;
                    Ok(())
                }
            })
            .await?;

            Ok(BenchmarkReport::new(
                "webgpu::q4k_qgemv",
                config,
                samples,
                "1x1024 @ Q4K 2048x1024",
            ))
        })
    })
}

pub fn q4k_paired_silu() -> BenchmarkCase {
    bench_case("webgpu::q4k_paired_silu", |device, config| {
        Box::pin(async move {
            let weight_shape = [2048usize, 1024usize];
            let input_shape = [1usize, weight_shape[1]];
            let raw_bytes = q4k_raw_bytes(weight_shape);
            let matrix: QMatrix =
                qmatrix_from_raw_bytes(device, weight_shape, &raw_bytes, GgmlType::Q4K);
            let input_values = deterministic_values(elements(&input_shape), 30, 0.003);
            let input: Tensor<2, f32> = Tensor::from_slice(device, input_shape, &input_values);
            materialize_inputs(&[&input]).await;

            let samples = time_samples(config, || {
                let output = input.q_mat_mul_paired_silu_product(&matrix);
                async move {
                    output.materialize().await;
                    Ok(())
                }
            })
            .await?;

            Ok(BenchmarkReport::new(
                "webgpu::q4k_paired_silu",
                config,
                samples,
                "1x1024 @ paired Q4K 2048x1024",
            ))
        })
    })
}

pub fn flash_attention_small() -> BenchmarkCase {
    bench_case("webgpu::flash_attention_small", |device, config| {
        Box::pin(async move {
            let shape = [1usize, 4usize, 128usize, 64usize];
            let q_values = deterministic_values(elements(&shape), 31, 0.003);
            let k_values = deterministic_values(elements(&shape), 32, 0.003);
            let v_values = deterministic_values(elements(&shape), 33, 0.003);
            let q: Tensor<4, f32> = Tensor::from_slice(device, shape, &q_values);
            let k: Tensor<4, f32> = Tensor::from_slice(device, shape, &k_values);
            let v: Tensor<4, f32> = Tensor::from_slice(device, shape, &v_values);
            materialize_inputs(&[&q, &k, &v]).await;

            let samples = time_samples(config, || {
                let output = q.flash_attention(&k, &v, 1.0 / (64.0f32).sqrt(), None);
                async move {
                    output.materialize().await;
                    Ok(())
                }
            })
            .await?;

            Ok(BenchmarkReport::new(
                "webgpu::flash_attention_small",
                config,
                samples,
                format!("{} flash attention", shape_label(&shape)),
            ))
        })
    })
}

pub fn flash_attention_causal_small() -> BenchmarkCase {
    bench_case("webgpu::flash_attention_causal_small", |device, config| {
        Box::pin(async move {
            let shape = [1usize, 4usize, 128usize, 64usize];
            let mask_shape = [128usize, 128usize];
            let q_values = deterministic_values(elements(&shape), 34, 0.003);
            let k_values = deterministic_values(elements(&shape), 35, 0.003);
            let v_values = deterministic_values(elements(&shape), 36, 0.003);
            let mask_values = vec![0.0f32; elements(&mask_shape)];
            let q: Tensor<4, f32> = Tensor::from_slice(device, shape, &q_values);
            let k: Tensor<4, f32> = Tensor::from_slice(device, shape, &k_values);
            let v: Tensor<4, f32> = Tensor::from_slice(device, shape, &v_values);
            let mask: Tensor<2, f32> = Tensor::from_slice(device, mask_shape, &mask_values);
            materialize_inputs(&[&q, &k, &v]).await;
            materialize_inputs(&[&mask]).await;

            let samples = time_samples(config, || {
                let output = q.flash_attention(
                    &k,
                    &v,
                    1.0 / (64.0f32).sqrt(),
                    Some((&mask, MaskKind::Causal)),
                );
                async move {
                    output.materialize().await;
                    Ok(())
                }
            })
            .await?;

            Ok(BenchmarkReport::new(
                "webgpu::flash_attention_causal_small",
                config,
                samples,
                format!("{} causal flash attention", shape_label(&shape)),
            ))
        })
    })
}

pub fn rope_fused_decode() -> BenchmarkCase {
    bench_case("webgpu::rope_fused_decode", |device, config| {
        Box::pin(async move {
            let shape = [1usize, 8usize, 256usize, 64usize];
            let [batch, heads, seq_len, head_dim] = shape;
            let pos_shape = [seq_len * 2, head_dim / 2];
            let cos_values = (0..pos_shape[0])
                .flat_map(|i| {
                    (0..pos_shape[1]).map(move |j| {
                        ((i as f32) / 10000f32.powf((2 * (j / 2)) as f32 / head_dim as f32)).cos()
                    })
                })
                .collect::<Vec<_>>();
            let sin_values = (0..pos_shape[0])
                .flat_map(|i| {
                    (0..pos_shape[1]).map(move |j| {
                        ((i as f32) / 10000f32.powf((2 * (j / 2)) as f32 / head_dim as f32)).sin()
                    })
                })
                .collect::<Vec<_>>();
            let input_values = deterministic_values(batch * heads * seq_len * head_dim, 9, 0.01);
            let input: Tensor<4, f32> = Tensor::from_slice(device, shape, &input_values);
            let cos: Tensor<2, f32> = Tensor::from_slice(device, pos_shape, &cos_values);
            let sin: Tensor<2, f32> = Tensor::from_slice(device, pos_shape, &sin_values);
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

            Ok(BenchmarkReport::new(
                "webgpu::rope_fused_decode",
                config,
                samples,
                format!("{} fused rope", shape_label(&shape)),
            ))
        })
    })
}
