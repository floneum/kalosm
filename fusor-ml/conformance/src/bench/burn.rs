//! Burn WGPU benchmark cases that mirror Fusor WebGPU cases where possible.

use burn::{
    backend::{Wgpu, wgpu::WgpuDevice},
    nn::{RmsNormConfig, RotaryEncodingConfig},
    tensor::{
        Tensor, TensorData, activation, module,
        ops::{AttentionModuleOptions, ConvOptions},
    },
};

use fusor::Device;

use crate::bench::{BenchmarkCase, BenchmarkConfig, BenchmarkReport, BenchmarkResult};

use super::time_samples;

type BurnTensor<const R: usize> = Tensor<Wgpu, R>;

#[cfg(any(target_arch = "wasm32", windows))]
static BURN_WGPU_RUNTIME_INITIALIZED: core::sync::atomic::AtomicBool =
    core::sync::atomic::AtomicBool::new(false);

#[cfg(any(target_arch = "wasm32", windows))]
pub(crate) async fn initialized_device() -> WgpuDevice {
    use core::sync::atomic::Ordering;

    use burn::backend::wgpu::{RuntimeOptions, graphics, init_setup_async};

    #[cfg(target_arch = "wasm32")]
    type Api = graphics::WebGpu;
    // Cubecl's auto selection requests Vulkan on Windows, which is missing on
    // GPU-less runners; DX12 always has at least the WARP software adapter.
    #[cfg(windows)]
    type Api = graphics::Dx12;

    let device = WgpuDevice::default();
    if BURN_WGPU_RUNTIME_INITIALIZED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
    {
        init_setup_async::<Api>(&device, RuntimeOptions::default()).await;
    }
    device
}

#[cfg(not(any(target_arch = "wasm32", windows)))]
pub(crate) async fn initialized_device() -> WgpuDevice {
    WgpuDevice::default()
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

fn burn_tensor<const R: usize>(
    values: Vec<f32>,
    shape: [usize; R],
    device: &WgpuDevice,
) -> BurnTensor<R> {
    Tensor::<Wgpu, R>::from_data(TensorData::new(values, shape), device)
}

async fn materialize<const R: usize>(tensor: BurnTensor<R>) -> BenchmarkResult<()> {
    let _ = tensor.into_data_async().await;
    Ok(())
}

async fn materialize_inputs<const R: usize>(inputs: &[BurnTensor<R>]) -> BenchmarkResult<()> {
    for input in inputs {
        materialize(input.clone()).await?;
    }
    Ok(())
}

fn bench_case(
    name: &'static str,
    run: impl for<'a> FnOnce(&'a Device, BenchmarkConfig) -> super::CaseFuture<'a> + 'static,
) -> BenchmarkCase {
    BenchmarkCase::new(name, run)
}

pub fn elementwise_add_square() -> BenchmarkCase {
    bench_case("burn::elementwise_add_square", |_fusor_device, config| {
        Box::pin(async move {
            let device = initialized_device().await;
            let shape = [512usize, 512usize];
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
            materialize_inputs(&[lhs.clone(), rhs.clone()]).await?;

            let samples = time_samples(config, || {
                let output = lhs.clone() + rhs.clone();
                async move { materialize(output).await }
            })
            .await?;

            Ok(BenchmarkReport::new(
                "burn::elementwise_add_square",
                config,
                samples,
                format!("{} f32 add", shape_label(&shape)),
            ))
        })
    })
}

pub fn elementwise_mul_rank4() -> BenchmarkCase {
    bench_case("burn::elementwise_mul_rank4", |_fusor_device, config| {
        Box::pin(async move {
            let device = initialized_device().await;
            let shape = [9usize, 11usize, 32usize, 16usize];
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
            materialize_inputs(&[lhs.clone(), rhs.clone()]).await?;

            let samples = time_samples(config, || {
                let output = lhs.clone() * rhs.clone();
                async move { materialize(output).await }
            })
            .await?;

            Ok(BenchmarkReport::new(
                "burn::elementwise_mul_rank4",
                config,
                samples,
                format!("{} f32 mul", shape_label(&shape)),
            ))
        })
    })
}

pub fn unary_trig_chain() -> BenchmarkCase {
    bench_case("burn::unary_trig_chain", |_fusor_device, config| {
        Box::pin(async move {
            let device = initialized_device().await;
            let shape = [384usize, 384usize];
            let input = burn_tensor(
                deterministic_values(elements(&shape), 10, 0.01),
                shape,
                &device,
            );
            materialize_inputs(std::slice::from_ref(&input)).await?;

            let samples = time_samples(config, || {
                let output = input.clone().sin() + input.clone().cos();
                async move { materialize(output).await }
            })
            .await?;

            Ok(BenchmarkReport::new(
                "burn::unary_trig_chain",
                config,
                samples,
                format!("{} sin+cos", shape_label(&shape)),
            ))
        })
    })
}

pub fn activation_gelu() -> BenchmarkCase {
    bench_case("burn::activation_gelu", |_fusor_device, config| {
        Box::pin(async move {
            let device = initialized_device().await;
            let shape = [512usize, 256usize];
            let input = burn_tensor(
                deterministic_values(elements(&shape), 11, 0.015),
                shape,
                &device,
            );
            materialize_inputs(std::slice::from_ref(&input)).await?;

            let samples = time_samples(config, || {
                let output = activation::gelu(input.clone());
                async move { materialize(output).await }
            })
            .await?;

            Ok(BenchmarkReport::new(
                "burn::activation_gelu",
                config,
                samples,
                format!("{} gelu", shape_label(&shape)),
            ))
        })
    })
}

pub fn broadcast_add() -> BenchmarkCase {
    bench_case("burn::broadcast_add", |_fusor_device, config| {
        Box::pin(async move {
            let device = initialized_device().await;
            let matrix_shape = [256usize, 512usize];
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
            materialize_inputs(std::slice::from_ref(&matrix)).await?;
            materialize_inputs(std::slice::from_ref(&vector)).await?;

            let samples = time_samples(config, || {
                let output = matrix.clone() + vector.clone().reshape([1, vector_shape[0]]);
                async move { materialize(output).await }
            })
            .await?;

            Ok(BenchmarkReport::new(
                "burn::broadcast_add",
                config,
                samples,
                "256x512 + broadcast 512",
            ))
        })
    })
}

pub fn transpose_then_elementwise() -> BenchmarkCase {
    bench_case(
        "burn::transpose_then_elementwise",
        |_fusor_device, config| {
            Box::pin(async move {
                let device = initialized_device().await;
                let shape = [256usize, 384usize];
                let input = burn_tensor(
                    deterministic_values(elements(&shape), 14, 0.01),
                    shape,
                    &device,
                );
                materialize_inputs(std::slice::from_ref(&input)).await?;

                let samples = time_samples(config, || {
                    let transposed = input.clone().transpose();
                    let output = transposed.clone() * transposed;
                    async move { materialize(output).await }
                })
                .await?;

                Ok(BenchmarkReport::new(
                    "burn::transpose_then_elementwise",
                    config,
                    samples,
                    "256x384 transpose, square",
                ))
            })
        },
    )
}

pub fn reduction_sum_last_dim() -> BenchmarkCase {
    bench_case("burn::reduction_sum_last_dim", |_fusor_device, config| {
        Box::pin(async move {
            let device = initialized_device().await;
            let shape = [256usize, 512usize];
            let input = burn_tensor(
                deterministic_values(elements(&shape), 15, 0.004),
                shape,
                &device,
            );
            materialize_inputs(std::slice::from_ref(&input)).await?;

            let samples = time_samples(config, || {
                let output = input.clone().sum_dim(1);
                async move { materialize(output).await }
            })
            .await?;

            Ok(BenchmarkReport::new(
                "burn::reduction_sum_last_dim",
                config,
                samples,
                format!("{} sum axis 1", shape_label(&shape)),
            ))
        })
    })
}

pub fn reduction_max_middle_axis() -> BenchmarkCase {
    bench_case(
        "burn::reduction_max_middle_axis",
        |_fusor_device, config| {
            Box::pin(async move {
                let device = initialized_device().await;
                let shape = [64usize, 128usize, 64usize];
                let input = burn_tensor(
                    deterministic_values(elements(&shape), 16, 0.004),
                    shape,
                    &device,
                );
                materialize_inputs(std::slice::from_ref(&input)).await?;

                let samples = time_samples(config, || {
                    let output = input.clone().max_dim(1);
                    async move { materialize(output).await }
                })
                .await?;

                Ok(BenchmarkReport::new(
                    "burn::reduction_max_middle_axis",
                    config,
                    samples,
                    format!("{} max axis 1", shape_label(&shape)),
                ))
            })
        },
    )
}

pub fn softmax_last_dim() -> BenchmarkCase {
    bench_case("burn::softmax_last_dim", |_fusor_device, config| {
        Box::pin(async move {
            let device = initialized_device().await;
            let shape = [512usize, 256usize];
            let input = burn_tensor(
                deterministic_values(elements(&shape), 5, 0.006),
                shape,
                &device,
            );
            materialize_inputs(std::slice::from_ref(&input)).await?;

            let samples = time_samples(config, || {
                let output = activation::softmax(input.clone(), 1);
                async move { materialize(output).await }
            })
            .await?;

            Ok(BenchmarkReport::new(
                "burn::softmax_last_dim",
                config,
                samples,
                format!("{} last-axis softmax", shape_label(&shape)),
            ))
        })
    })
}

pub fn softmax_middle_axis() -> BenchmarkCase {
    bench_case("burn::softmax_middle_axis", |_fusor_device, config| {
        Box::pin(async move {
            let device = initialized_device().await;
            let shape = [32usize, 128usize, 64usize];
            let input = burn_tensor(
                deterministic_values(elements(&shape), 17, 0.004),
                shape,
                &device,
            );
            materialize_inputs(std::slice::from_ref(&input)).await?;

            let samples = time_samples(config, || {
                let output = activation::softmax(input.clone(), 1);
                async move { materialize(output).await }
            })
            .await?;

            Ok(BenchmarkReport::new(
                "burn::softmax_middle_axis",
                config,
                samples,
                format!("{} softmax axis 1", shape_label(&shape)),
            ))
        })
    })
}

pub fn layer_norm_last_dim() -> BenchmarkCase {
    bench_case("burn::layer_norm_last_dim", |_fusor_device, config| {
        Box::pin(async move {
            let device = initialized_device().await;
            let shape = [8usize, 128usize, 512usize];
            let last_dim = shape[2];
            let input = burn_tensor(
                deterministic_values(elements(&shape), 18, 0.01),
                shape,
                &device,
            );
            let layer = burn::nn::LayerNormConfig::new(last_dim)
                .with_epsilon(1.0e-5)
                .init::<Wgpu>(&device);
            materialize_inputs(std::slice::from_ref(&input)).await?;

            let samples = time_samples(config, || {
                let output = layer.clone().forward(input.clone());
                async move { materialize(output).await }
            })
            .await?;

            Ok(BenchmarkReport::new(
                "burn::layer_norm_last_dim",
                config,
                samples,
                format!("{} layer norm", shape_label(&shape)),
            ))
        })
    })
}

pub fn rms_norm_fused() -> BenchmarkCase {
    bench_case("burn::rms_norm_fused", |_fusor_device, config| {
        Box::pin(async move {
            let device = initialized_device().await;
            let shape = [8usize, 128usize, 512usize];
            let last_dim = shape[2];
            let input = burn_tensor(
                deterministic_values(elements(&shape), 21, 0.01),
                shape,
                &device,
            );
            let rms = RmsNormConfig::new(last_dim)
                .with_epsilon(1.0e-5)
                .init::<Wgpu>(&device);
            materialize_inputs(std::slice::from_ref(&input)).await?;

            let samples = time_samples(config, || {
                let output = rms.clone().forward(input.clone());
                async move { materialize(output).await }
            })
            .await?;

            Ok(BenchmarkReport::new(
                "burn::rms_norm_fused",
                config,
                samples,
                format!("{} rms norm", shape_label(&shape)),
            ))
        })
    })
}

pub fn dense_matmul_square() -> BenchmarkCase {
    bench_case("burn::dense_matmul_square", |_fusor_device, config| {
        Box::pin(async move {
            let device = initialized_device().await;
            let lhs_shape = [256usize, 256usize];
            let rhs_shape = [256usize, 256usize];
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
            materialize_inputs(&[lhs.clone(), rhs.clone()]).await?;

            let samples = time_samples(config, || {
                let output = lhs.clone().matmul(rhs.clone());
                async move { materialize(output).await }
            })
            .await?;

            Ok(BenchmarkReport::new(
                "burn::dense_matmul_square",
                config,
                samples,
                "256x256 @ 256x256 f32",
            ))
        })
    })
}

pub fn dense_batched_matmul() -> BenchmarkCase {
    bench_case("burn::dense_batched_matmul", |_fusor_device, config| {
        Box::pin(async move {
            let device = initialized_device().await;
            let lhs_shape = [8usize, 64usize, 96usize];
            let rhs_shape = [8usize, 96usize, 64usize];
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
            materialize_inputs(&[lhs.clone(), rhs.clone()]).await?;

            let samples = time_samples(config, || {
                let output = lhs.clone().matmul(rhs.clone());
                async move { materialize(output).await }
            })
            .await?;

            Ok(BenchmarkReport::new(
                "burn::dense_batched_matmul",
                config,
                samples,
                "8x64x96 @ 8x96x64 f32",
            ))
        })
    })
}

pub fn conv1d_small() -> BenchmarkCase {
    bench_case("burn::conv1d_small", |_fusor_device, config| {
        Box::pin(async move {
            let device = initialized_device().await;
            let input_shape = [4usize, 8usize, 256usize];
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
            materialize_inputs(std::slice::from_ref(&input)).await?;
            materialize_inputs(std::slice::from_ref(&weight)).await?;
            materialize_inputs(std::slice::from_ref(&bias)).await?;

            let samples = time_samples(config, || {
                let output = module::conv1d(
                    input.clone(),
                    weight.clone(),
                    Some(bias.clone()),
                    ConvOptions::new([2], [1], [1], 1),
                );
                async move { materialize(output).await }
            })
            .await?;

            Ok(BenchmarkReport::new(
                "burn::conv1d_small",
                config,
                samples,
                "4x8x256 conv 16x8x5",
            ))
        })
    })
}

pub fn top_k_large() -> BenchmarkCase {
    bench_case("burn::top_k_large", |_fusor_device, config| {
        Box::pin(async move {
            let device = initialized_device().await;
            let input_len = 65_537usize;
            let k = 64usize;
            let values = (0..input_len)
                .map(|index| {
                    let base = ((index * 67 + 29) % 10_007) as f32 * 0.001;
                    let bump = if index % 4099 == 0 { 20.0 } else { 0.0 };
                    base + bump - (index % 13) as f32 * 0.0001
                })
                .collect::<Vec<_>>();
            let input = burn_tensor(values, [input_len], &device);
            materialize_inputs(std::slice::from_ref(&input)).await?;

            let samples = time_samples(config, || {
                let output = input.clone().topk(k, 0);
                async move { materialize(output).await }
            })
            .await?;

            Ok(BenchmarkReport::new(
                "burn::top_k_large",
                config,
                samples,
                format!("{input_len} logits, k={k}"),
            ))
        })
    })
}

pub fn top_k_qwen_vocab() -> BenchmarkCase {
    bench_case("burn::top_k_qwen_vocab", |_fusor_device, config| {
        Box::pin(async move {
            let device = initialized_device().await;
            let input_len = 151_936usize;
            let k = 40usize;
            let input = burn_tensor(
                deterministic_values(input_len, 28, 0.01),
                [input_len],
                &device,
            );
            materialize_inputs(std::slice::from_ref(&input)).await?;

            let samples = time_samples(config, || {
                let output = input.clone().topk(k, 0);
                async move { materialize(output).await }
            })
            .await?;

            Ok(BenchmarkReport::new(
                "burn::top_k_qwen_vocab",
                config,
                samples,
                format!("{input_len} logits, k={k}"),
            ))
        })
    })
}

pub fn q8_0_qgemv() -> BenchmarkCase {
    bench_case("burn::q8_0_qgemv", |_fusor_device, config| {
        Box::pin(async move {
            let device = initialized_device().await;
            let weight_shape = [4096usize, 896usize];
            let input_shape = [1usize, weight_shape[1]];
            let dense_weight_shape = [weight_shape[1], weight_shape[0]];
            let input = burn_tensor(
                deterministic_values(elements(&input_shape), 8, 0.003),
                input_shape,
                &device,
            );
            let weights = burn_tensor(
                deterministic_values(elements(&dense_weight_shape), 80, 0.003),
                dense_weight_shape,
                &device,
            );
            materialize_inputs(std::slice::from_ref(&input)).await?;
            materialize_inputs(std::slice::from_ref(&weights)).await?;

            let samples = time_samples(config, || {
                let output = input.clone().matmul(weights.clone());
                async move { materialize(output).await }
            })
            .await?;

            Ok(BenchmarkReport::new(
                "burn::q8_0_qgemv",
                config,
                samples,
                "1x896 @ dense f32 896x4096 baseline",
            ))
        })
    })
}

pub fn q4k_qgemv() -> BenchmarkCase {
    bench_case("burn::q4k_qgemv", |_fusor_device, config| {
        Box::pin(async move {
            let device = initialized_device().await;
            let weight_shape = [2048usize, 1024usize];
            let input_shape = [1usize, weight_shape[1]];
            let dense_weight_shape = [weight_shape[1], weight_shape[0]];
            let input = burn_tensor(
                deterministic_values(elements(&input_shape), 29, 0.003),
                input_shape,
                &device,
            );
            let weights = burn_tensor(
                deterministic_values(elements(&dense_weight_shape), 81, 0.003),
                dense_weight_shape,
                &device,
            );
            materialize_inputs(std::slice::from_ref(&input)).await?;
            materialize_inputs(std::slice::from_ref(&weights)).await?;

            let samples = time_samples(config, || {
                let output = input.clone().matmul(weights.clone());
                async move { materialize(output).await }
            })
            .await?;

            Ok(BenchmarkReport::new(
                "burn::q4k_qgemv",
                config,
                samples,
                "1x1024 @ dense f32 1024x2048 baseline",
            ))
        })
    })
}

pub fn q4k_paired_silu() -> BenchmarkCase {
    bench_case("burn::q4k_paired_silu", |_fusor_device, config| {
        Box::pin(async move {
            let device = initialized_device().await;
            let weight_shape = [2048usize, 1024usize];
            let input_shape = [1usize, weight_shape[1]];
            let dense_weight_shape = [weight_shape[1], weight_shape[0]];
            let pair_len = weight_shape[0] / 2;
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
            materialize_inputs(std::slice::from_ref(&input)).await?;
            materialize_inputs(std::slice::from_ref(&weights)).await?;

            let samples = time_samples(config, || {
                let projected = input.clone().matmul(weights.clone());
                let gate = projected.clone().narrow(1, 0, pair_len);
                let up = projected.narrow(1, pair_len, pair_len);
                let output = activation::silu(gate) * up;
                async move { materialize(output).await }
            })
            .await?;

            Ok(BenchmarkReport::new(
                "burn::q4k_paired_silu",
                config,
                samples,
                "1x1024 @ dense f32 1024x2048 + paired SiLU baseline",
            ))
        })
    })
}

pub fn flash_attention_small() -> BenchmarkCase {
    bench_case("burn::flash_attention_small", |_fusor_device, config| {
        Box::pin(async move {
            let device = initialized_device().await;
            let shape = [1usize, 4usize, 128usize, 64usize];
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
            materialize_inputs(&[q.clone(), k.clone(), v.clone()]).await?;

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
                        is_causal: false,
                    },
                );
                async move { materialize(output).await }
            })
            .await?;

            Ok(BenchmarkReport::new(
                "burn::flash_attention_small",
                config,
                samples,
                format!("{} scaled dot-product attention", shape_label(&shape)),
            ))
        })
    })
}

pub fn flash_attention_causal_small() -> BenchmarkCase {
    bench_case(
        "burn::flash_attention_causal_small",
        |_fusor_device, config| {
            Box::pin(async move {
                let device = initialized_device().await;
                let shape = [1usize, 4usize, 128usize, 64usize];
                let q = burn_tensor(
                    deterministic_values(elements(&shape), 34, 0.003),
                    shape,
                    &device,
                );
                let k = burn_tensor(
                    deterministic_values(elements(&shape), 35, 0.003),
                    shape,
                    &device,
                );
                let v = burn_tensor(
                    deterministic_values(elements(&shape), 36, 0.003),
                    shape,
                    &device,
                );
                materialize_inputs(&[q.clone(), k.clone(), v.clone()]).await?;

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
                            is_causal: true,
                        },
                    );
                    async move { materialize(output).await }
                })
                .await?;

                Ok(BenchmarkReport::new(
                    "burn::flash_attention_causal_small",
                    config,
                    samples,
                    format!(
                        "{} causal scaled dot-product attention",
                        shape_label(&shape)
                    ),
                ))
            })
        },
    )
}

pub fn rope_fused_decode() -> BenchmarkCase {
    bench_case("burn::rope_fused_decode", |_fusor_device, config| {
        Box::pin(async move {
            let device = initialized_device().await;
            let shape = [1usize, 8usize, 256usize, 64usize];
            let [batch, heads, seq_len, head_dim] = shape;
            let input = burn_tensor(
                deterministic_values(batch * heads * seq_len * head_dim, 9, 0.01),
                shape,
                &device,
            );
            let rope = RotaryEncodingConfig::new(seq_len * 2, head_dim).init::<Wgpu>(&device);
            materialize_inputs(std::slice::from_ref(&input)).await?;

            let samples = time_samples(config, || {
                let output = rope.clone().forward(input.clone());
                async move { materialize(output).await }
            })
            .await?;

            Ok(BenchmarkReport::new(
                "burn::rope_fused_decode",
                config,
                samples,
                format!("{} rotary encoding", shape_label(&shape)),
            ))
        })
    })
}
