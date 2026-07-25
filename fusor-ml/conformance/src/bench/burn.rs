//! Burn WGPU benchmark cases that mirror Fusor WebGPU cases where possible.
//!
//! Like `webgpu.rs`, each case body is parameterized by the sizes that vary
//! between the fixed registry entry and the per-size sweep, and the
//! `fixed_cases!` invocation at the bottom pins the registry sizes.

use burn::{
    backend::{Wgpu, wgpu::WgpuDevice},
    nn::{LayerNormConfig, RmsNormConfig, RotaryEncodingConfig},
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

async fn values_input<const R: usize>(
    device: &WgpuDevice,
    shape: [usize; R],
    values: Vec<f32>,
) -> BenchmarkResult<BurnTensor<R>> {
    let tensor = burn_tensor(values, shape, device);
    materialize(tensor.clone()).await?;
    Ok(tensor)
}

async fn input_tensor<const R: usize>(
    device: &WgpuDevice,
    shape: [usize; R],
    seed: usize,
    scale: f32,
) -> BenchmarkResult<BurnTensor<R>> {
    values_input(
        device,
        shape,
        deterministic_values(elements(&shape), seed, scale),
    )
    .await
}

fn bench_case(
    name: &'static str,
    run: impl FnOnce(BenchmarkConfig, String) -> super::CaseFuture<'static> + 'static,
) -> BenchmarkCase {
    BenchmarkCase::new(name, move |_fusor_device: &Device, config| {
        run(config, name.to_string())
    })
}

pub(super) async fn elementwise_add_square_case(
    config: BenchmarkConfig,
    name: String,
    size: usize,
) -> BenchmarkResult<BenchmarkReport> {
    let device = initialized_device().await;
    let shape = [size, size];
    let lhs = input_tensor(&device, shape, 1, 0.01).await?;
    let rhs = input_tensor(&device, shape, 2, 0.008).await?;
    let samples = time_samples(config, || {
        let output = lhs.clone() + rhs.clone();
        async move { materialize(output).await }
    })
    .await?;
    Ok(BenchmarkReport::new(
        name,
        config,
        samples,
        format!("{} f32 add", shape_label(&shape)),
    ))
}

pub(super) async fn elementwise_mul_rank4_case(
    config: BenchmarkConfig,
    name: String,
    shape: [usize; 4],
) -> BenchmarkResult<BenchmarkReport> {
    let device = initialized_device().await;
    let lhs = input_tensor(&device, shape, 3, 0.012).await?;
    let rhs = input_tensor(&device, shape, 4, 0.009).await?;
    let samples = time_samples(config, || {
        let output = lhs.clone() * rhs.clone();
        async move { materialize(output).await }
    })
    .await?;
    Ok(BenchmarkReport::new(
        name,
        config,
        samples,
        format!("{} f32 mul", shape_label(&shape)),
    ))
}

pub(super) async fn unary_trig_chain_case(
    config: BenchmarkConfig,
    name: String,
    size: usize,
) -> BenchmarkResult<BenchmarkReport> {
    let device = initialized_device().await;
    let shape = [size, size];
    let input = input_tensor(&device, shape, 10, 0.01).await?;
    let samples = time_samples(config, || {
        let output = input.clone().sin() + input.clone().cos();
        async move { materialize(output).await }
    })
    .await?;
    Ok(BenchmarkReport::new(
        name,
        config,
        samples,
        format!("{} sin+cos", shape_label(&shape)),
    ))
}

pub(super) async fn activation_gelu_case(
    config: BenchmarkConfig,
    name: String,
    shape: [usize; 2],
) -> BenchmarkResult<BenchmarkReport> {
    let device = initialized_device().await;
    let input = input_tensor(&device, shape, 11, 0.015).await?;
    let samples = time_samples(config, || {
        let output = activation::gelu(input.clone());
        async move { materialize(output).await }
    })
    .await?;
    Ok(BenchmarkReport::new(
        name,
        config,
        samples,
        format!("{} gelu", shape_label(&shape)),
    ))
}

pub(super) async fn broadcast_add_case(
    config: BenchmarkConfig,
    name: String,
    rows: usize,
) -> BenchmarkResult<BenchmarkReport> {
    let device = initialized_device().await;
    let matrix_shape = [rows, 512usize];
    let vector_shape = [512usize];
    let matrix = input_tensor(&device, matrix_shape, 12, 0.006).await?;
    let vector = input_tensor(&device, vector_shape, 13, 0.01).await?;
    let samples = time_samples(config, || {
        let output = matrix.clone() + vector.clone().reshape([1, vector_shape[0]]);
        async move { materialize(output).await }
    })
    .await?;
    Ok(BenchmarkReport::new(
        name,
        config,
        samples,
        format!("{} + broadcast 512", shape_label(&matrix_shape)),
    ))
}

pub(super) async fn transpose_then_elementwise_case(
    config: BenchmarkConfig,
    name: String,
    shape: [usize; 2],
) -> BenchmarkResult<BenchmarkReport> {
    let device = initialized_device().await;
    let input = input_tensor(&device, shape, 14, 0.01).await?;
    let samples = time_samples(config, || {
        let transposed = input.clone().transpose();
        let output = transposed.clone() * transposed;
        async move { materialize(output).await }
    })
    .await?;
    Ok(BenchmarkReport::new(
        name,
        config,
        samples,
        format!("{} transpose, square", shape_label(&shape)),
    ))
}

pub(super) async fn reduction_sum_last_dim_case(
    config: BenchmarkConfig,
    name: String,
    rows: usize,
) -> BenchmarkResult<BenchmarkReport> {
    let device = initialized_device().await;
    let shape = [rows, 512usize];
    let input = input_tensor(&device, shape, 15, 0.004).await?;
    let samples = time_samples(config, || {
        let output = input.clone().sum_dim(1);
        async move { materialize(output).await }
    })
    .await?;
    Ok(BenchmarkReport::new(
        name,
        config,
        samples,
        format!("{} sum axis 1", shape_label(&shape)),
    ))
}

pub(super) async fn reduction_max_middle_axis_case(
    config: BenchmarkConfig,
    name: String,
    shape: [usize; 3],
) -> BenchmarkResult<BenchmarkReport> {
    let device = initialized_device().await;
    let input = input_tensor(&device, shape, 16, 0.004).await?;
    let samples = time_samples(config, || {
        let output = input.clone().max_dim(1);
        async move { materialize(output).await }
    })
    .await?;
    Ok(BenchmarkReport::new(
        name,
        config,
        samples,
        format!("{} max axis 1", shape_label(&shape)),
    ))
}

pub(super) async fn softmax_last_dim_case(
    config: BenchmarkConfig,
    name: String,
    rows: usize,
) -> BenchmarkResult<BenchmarkReport> {
    let device = initialized_device().await;
    let shape = [rows, 256usize];
    let input = input_tensor(&device, shape, 5, 0.006).await?;
    let samples = time_samples(config, || {
        let output = activation::softmax(input.clone(), 1);
        async move { materialize(output).await }
    })
    .await?;
    Ok(BenchmarkReport::new(
        name,
        config,
        samples,
        format!("{} last-axis softmax", shape_label(&shape)),
    ))
}

pub(super) async fn softmax_middle_axis_case(
    config: BenchmarkConfig,
    name: String,
    shape: [usize; 3],
) -> BenchmarkResult<BenchmarkReport> {
    let device = initialized_device().await;
    let input = input_tensor(&device, shape, 17, 0.004).await?;
    let samples = time_samples(config, || {
        let output = activation::softmax(input.clone(), 1);
        async move { materialize(output).await }
    })
    .await?;
    Ok(BenchmarkReport::new(
        name,
        config,
        samples,
        format!("{} softmax axis 1", shape_label(&shape)),
    ))
}

pub(super) async fn layer_norm_last_dim_case(
    config: BenchmarkConfig,
    name: String,
    shape: [usize; 3],
) -> BenchmarkResult<BenchmarkReport> {
    let device = initialized_device().await;
    let last_dim = shape[2];
    let input = input_tensor(&device, shape, 18, 0.01).await?;
    let layer = LayerNormConfig::new(last_dim)
        .with_epsilon(1.0e-5)
        .init::<Wgpu>(&device);
    let samples = time_samples(config, || {
        let output = layer.clone().forward(input.clone());
        async move { materialize(output).await }
    })
    .await?;
    Ok(BenchmarkReport::new(
        name,
        config,
        samples,
        format!("{} layer norm", shape_label(&shape)),
    ))
}

pub(super) async fn rms_norm_fused_case(
    config: BenchmarkConfig,
    name: String,
    shape: [usize; 3],
) -> BenchmarkResult<BenchmarkReport> {
    let device = initialized_device().await;
    let last_dim = shape[2];
    let input = input_tensor(&device, shape, 21, 0.01).await?;
    let rms = RmsNormConfig::new(last_dim)
        .with_epsilon(1.0e-5)
        .init::<Wgpu>(&device);
    let samples = time_samples(config, || {
        let output = rms.clone().forward(input.clone());
        async move { materialize(output).await }
    })
    .await?;
    Ok(BenchmarkReport::new(
        name,
        config,
        samples,
        format!("{} rms norm", shape_label(&shape)),
    ))
}

pub(super) async fn dense_matmul_square_case(
    config: BenchmarkConfig,
    name: String,
    size: usize,
) -> BenchmarkResult<BenchmarkReport> {
    let device = initialized_device().await;
    let lhs_shape = [size, size];
    let rhs_shape = [size, size];
    let lhs = input_tensor(&device, lhs_shape, 6, 0.004).await?;
    let rhs = input_tensor(&device, rhs_shape, 7, 0.004).await?;
    let samples = time_samples(config, || {
        let output = lhs.clone().matmul(rhs.clone());
        async move { materialize(output).await }
    })
    .await?;
    Ok(BenchmarkReport::new(
        name,
        config,
        samples,
        format!(
            "{} @ {} f32",
            shape_label(&lhs_shape),
            shape_label(&rhs_shape)
        ),
    ))
}

pub(super) async fn dense_batched_matmul_case(
    config: BenchmarkConfig,
    name: String,
    batch: usize,
    m: usize,
    k: usize,
) -> BenchmarkResult<BenchmarkReport> {
    let device = initialized_device().await;
    let lhs_shape = [batch, m, k];
    let rhs_shape = [batch, k, m];
    let lhs = input_tensor(&device, lhs_shape, 23, 0.004).await?;
    let rhs = input_tensor(&device, rhs_shape, 24, 0.004).await?;
    let samples = time_samples(config, || {
        let output = lhs.clone().matmul(rhs.clone());
        async move { materialize(output).await }
    })
    .await?;
    Ok(BenchmarkReport::new(
        name,
        config,
        samples,
        format!(
            "{} @ {} f32",
            shape_label(&lhs_shape),
            shape_label(&rhs_shape)
        ),
    ))
}

pub(super) async fn conv1d_small_case(
    config: BenchmarkConfig,
    name: String,
    len: usize,
) -> BenchmarkResult<BenchmarkReport> {
    let device = initialized_device().await;
    let input_shape = [4usize, 8usize, len];
    let weight_shape = [16usize, 8usize, 5usize];
    let bias_shape = [16usize];
    let input = input_tensor(&device, input_shape, 25, 0.01).await?;
    let weight = input_tensor(&device, weight_shape, 26, 0.01).await?;
    let bias = input_tensor(&device, bias_shape, 27, 0.001).await?;
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
        name,
        config,
        samples,
        format!(
            "{} conv {}",
            shape_label(&input_shape),
            shape_label(&weight_shape)
        ),
    ))
}

pub(super) async fn top_k_case(
    config: BenchmarkConfig,
    name: String,
    input_len: usize,
    k: usize,
    values: Vec<f32>,
) -> BenchmarkResult<BenchmarkReport> {
    let device = initialized_device().await;
    let input = values_input(&device, [input_len], values).await?;
    let samples = time_samples(config, || {
        let output = input.clone().topk(k, 0);
        async move { materialize(output).await }
    })
    .await?;
    Ok(BenchmarkReport::new(
        name,
        config,
        samples,
        format!("{input_len} logits, k={k}"),
    ))
}

pub(super) async fn qgemv_dense_case(
    config: BenchmarkConfig,
    name: String,
    weight_shape: [usize; 2],
    input_seed: usize,
    weight_seed: usize,
    detail_suffix: &'static str,
) -> BenchmarkResult<BenchmarkReport> {
    let device = initialized_device().await;
    let input_shape = [1usize, weight_shape[1]];
    let dense_weight_shape = [weight_shape[1], weight_shape[0]];
    let input = input_tensor(&device, input_shape, input_seed, 0.003).await?;
    let weights = input_tensor(&device, dense_weight_shape, weight_seed, 0.003).await?;
    let samples = time_samples(config, || {
        let output = input.clone().matmul(weights.clone());
        async move { materialize(output).await }
    })
    .await?;
    Ok(BenchmarkReport::new(
        name,
        config,
        samples,
        format!(
            "1x{} @ dense f32 {}{detail_suffix}",
            weight_shape[1],
            shape_label(&dense_weight_shape)
        ),
    ))
}

pub(super) async fn q4k_paired_silu_case(
    config: BenchmarkConfig,
    name: String,
    weight_shape: [usize; 2],
    detail_suffix: &'static str,
) -> BenchmarkResult<BenchmarkReport> {
    let device = initialized_device().await;
    let input_shape = [1usize, weight_shape[1]];
    let dense_weight_shape = [weight_shape[1], weight_shape[0]];
    let pair_len = weight_shape[0] / 2;
    let input = input_tensor(&device, input_shape, 30, 0.003).await?;
    let weights = input_tensor(&device, dense_weight_shape, 82, 0.003).await?;
    let samples = time_samples(config, || {
        let projected = input.clone().matmul(weights.clone());
        let gate = projected.clone().narrow(1, 0, pair_len);
        let up = projected.narrow(1, pair_len, pair_len);
        let output = activation::silu(gate) * up;
        async move { materialize(output).await }
    })
    .await?;
    Ok(BenchmarkReport::new(
        name,
        config,
        samples,
        format!(
            "1x{} @ dense f32 {} + paired SiLU{detail_suffix}",
            weight_shape[1],
            shape_label(&dense_weight_shape)
        ),
    ))
}

pub(super) async fn attention_case(
    config: BenchmarkConfig,
    name: String,
    seq_len: usize,
    seeds: [usize; 3],
    causal: bool,
    detail_op: &'static str,
) -> BenchmarkResult<BenchmarkReport> {
    let device = initialized_device().await;
    let shape = [1usize, 4usize, seq_len, 64usize];
    let q = input_tensor(&device, shape, seeds[0], 0.003).await?;
    let k = input_tensor(&device, shape, seeds[1], 0.003).await?;
    let v = input_tensor(&device, shape, seeds[2], 0.003).await?;
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
                is_causal: causal,
            },
        );
        async move { materialize(output).await }
    })
    .await?;
    Ok(BenchmarkReport::new(
        name,
        config,
        samples,
        format!("{} {detail_op}", shape_label(&shape)),
    ))
}

pub(super) async fn rope_fused_decode_case(
    config: BenchmarkConfig,
    name: String,
    seq_len: usize,
) -> BenchmarkResult<BenchmarkReport> {
    let device = initialized_device().await;
    let shape = [1usize, 8usize, seq_len, 64usize];
    let [_, _, _, head_dim] = shape;
    let input = input_tensor(&device, shape, 9, 0.01).await?;
    let rope = RotaryEncodingConfig::new(seq_len * 2, head_dim).init::<Wgpu>(&device);
    let samples = time_samples(config, || {
        let output = rope.clone().forward(input.clone());
        async move { materialize(output).await }
    })
    .await?;
    Ok(BenchmarkReport::new(
        name,
        config,
        samples,
        format!("{} rotary encoding", shape_label(&shape)),
    ))
}

macro_rules! fixed_cases {
    ($($case:ident => $body:ident($($arg:expr),* $(,)?);)*) => {
        $(
            pub fn $case() -> BenchmarkCase {
                bench_case(concat!("burn::", stringify!($case)), |config, name| {
                    Box::pin($body(config, name, $($arg),*))
                })
            }
        )*
    };
}

fixed_cases! {
    elementwise_add_square => elementwise_add_square_case(512);
    elementwise_mul_rank4 => elementwise_mul_rank4_case([9, 11, 32, 16]);
    unary_trig_chain => unary_trig_chain_case(384);
    activation_gelu => activation_gelu_case([512, 256]);
    broadcast_add => broadcast_add_case(256);
    transpose_then_elementwise => transpose_then_elementwise_case([256, 384]);
    reduction_sum_last_dim => reduction_sum_last_dim_case(256);
    reduction_max_middle_axis => reduction_max_middle_axis_case([64, 128, 64]);
    softmax_last_dim => softmax_last_dim_case(512);
    softmax_middle_axis => softmax_middle_axis_case([32, 128, 64]);
    layer_norm_last_dim => layer_norm_last_dim_case([8, 128, 512]);
    rms_norm_fused => rms_norm_fused_case([8, 128, 512]);
    dense_matmul_square => dense_matmul_square_case(256);
    dense_batched_matmul => dense_batched_matmul_case(8, 64, 96);
    conv1d_small => conv1d_small_case(256);
    top_k_large => top_k_case(65_537, 64, crate::bench::webgpu::topk_values(65_537));
    top_k_qwen_vocab => top_k_case(151_936, 40, deterministic_values(151_936, 28, 0.01));
    q8_0_qgemv => qgemv_dense_case([4096, 896], 8, 80, " baseline");
    q4k_qgemv => qgemv_dense_case([2048, 1024], 29, 81, " baseline");
    q4k_paired_silu => q4k_paired_silu_case([2048, 1024], " baseline");
    attention_small => attention_case(128, [31, 32, 33], false, "scaled dot-product attention");
    attention_causal_small => attention_case(128, [34, 35, 36], true, "causal scaled dot-product attention");
    rope_fused_decode => rope_fused_decode_case(256);
}
