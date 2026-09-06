//! Concrete WebGPU benchmark cases.
//!
//! Each case body is parameterized by the sizes that vary between the fixed
//! registry entry and the per-size sweep, so both are thin callers into one
//! implementation. The `fixed_cases!` invocation at the bottom pins the sizes
//! every registry entry runs at.
//!
//! A case builds its inputs once, then times `iterations` of "build the lazy
//! output, dispatch, fence". The fence is [`Device::wait_async`]: the queue's
//! own submitted-work signal, so no readback is on the clock.

use fusor::cache::MaskKind;
use fusor::{Device, Dim, QMatrix, Tensor};
use fusor_gguf::blocks::block_fields;
use fusor_ir::dtype::{QFmt, QLayout};
use half::f16;

use crate::bench::{
    BenchmarkCase, BenchmarkConfig, BenchmarkEvent, BenchmarkReport, BenchmarkResult,
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

pub(super) fn topk_values(input_len: usize) -> Vec<f32> {
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

/// Dispatch everything `value` needs and wait for the device to retire it.
async fn materialize<const R: usize, T: fusor::Element>(
    device: &Device,
    value: &Tensor<R, T>,
) -> BenchmarkResult<()> {
    device
        .session()
        .resolve(std::slice::from_ref(value.as_dyn()))?;
    device.wait_async().await?;
    Ok(())
}

async fn values_input<const R: usize>(
    device: &Device,
    shape: [usize; R],
    values: &[f32],
) -> BenchmarkResult<Tensor<R, f32>> {
    let tensor: Tensor<R, f32> = Tensor::from_slice(device, shape, values);
    materialize(device, &tensor).await?;
    Ok(tensor)
}

async fn input_tensor<const R: usize>(
    device: &Device,
    shape: [usize; R],
    seed: usize,
    scale: f32,
) -> BenchmarkResult<Tensor<R, f32>> {
    values_input(
        device,
        shape,
        &deterministic_values(elements(&shape), seed, scale),
    )
    .await
}

/// Well-formed quantized blocks for a `[rows, cols]` weight: a deterministic
/// payload under an explicit finite scale (a random f16 scale is NaN or Inf
/// about 1 time in 2000).
fn quantized_bytes(fmt: QFmt, layout: QLayout, shape: [usize; 2]) -> Vec<u8> {
    let fields = block_fields(fmt, layout);
    let block_bytes = fmt.block_bytes(layout) as usize;
    let block_elements = fmt.block_elements() as usize;
    let blocks = elements(&shape).div_ceil(block_elements);
    let mut out = Vec::with_capacity(blocks * block_bytes);
    let mut state = 0x9e37_79b9u32;
    for _ in 0..blocks {
        let start = out.len();
        for _ in 0..block_bytes {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            out.push((state >> 24) as u8);
        }
        let block = &mut out[start..];
        let mut write_scale = |at: u32, value: f32| {
            let at = at as usize;
            if fields.scale_is_f16 {
                block[at..at + 2].copy_from_slice(&f16::from_f32(value).to_le_bytes());
            } else {
                block[at..at + 4].copy_from_slice(&value.to_le_bytes());
            }
        };
        write_scale(fields.scale, 0.015_625);
        if let Some(min) = fields.min {
            write_scale(min, 0.003_906_25);
        }
    }
    out
}

fn qmatrix(device: &Device, fmt: QFmt, shape: [usize; 2]) -> BenchmarkResult<QMatrix> {
    let layout = QLayout::Native;
    let bytes = quantized_bytes(fmt, layout, shape);
    let rows = Dim::Const(shape[0] as u64);
    let cols = Dim::Const(shape[1] as u64);
    let tensor = device
        .graph()
        .quantized(fmt, layout, [rows, cols], &bytes)?;
    Ok(QMatrix {
        tensor,
        fmt,
        layout,
        rows,
        cols,
    })
}

fn bench_case(
    name: &'static str,
    run: impl for<'a> FnOnce(&'a Device, BenchmarkConfig, String) -> super::CaseFuture<'a> + 'static,
) -> BenchmarkCase {
    BenchmarkCase::new(name, move |device, config| {
        run(device, config, name.to_string())
    })
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

/// Time `iterations` of: build `output`, dispatch it, fence.
macro_rules! timed {
    ($device:expr, $config:expr, $output:expr) => {{
        let device = $device;
        time_samples($config, || {
            let output = $output;
            async move { materialize(device, &output).await }
        })
        .await?
    }};
}

pub(super) async fn elementwise_add_square_case(
    device: &Device,
    config: BenchmarkConfig,
    name: String,
    size: usize,
) -> BenchmarkResult<BenchmarkReport> {
    let shape = [size, size];
    let lhs = input_tensor(device, shape, 1, 0.01).await?;
    let rhs = input_tensor(device, shape, 2, 0.008).await?;
    let samples = timed!(device, config, lhs.add(&rhs));
    Ok(BenchmarkReport::new(
        name,
        config,
        samples,
        format!("{} f32 add", shape_label(&shape)),
    ))
}

pub(super) async fn elementwise_mul_rank4_case(
    device: &Device,
    config: BenchmarkConfig,
    name: String,
    shape: [usize; 4],
) -> BenchmarkResult<BenchmarkReport> {
    let lhs = input_tensor(device, shape, 3, 0.012).await?;
    let rhs = input_tensor(device, shape, 4, 0.009).await?;
    let samples = timed!(device, config, lhs.mul(&rhs));
    Ok(BenchmarkReport::new(
        name,
        config,
        samples,
        format!("{} f32 mul", shape_label(&shape)),
    ))
}

pub(super) async fn unary_trig_chain_case(
    device: &Device,
    config: BenchmarkConfig,
    name: String,
    size: usize,
) -> BenchmarkResult<BenchmarkReport> {
    let shape = [size, size];
    let input = input_tensor(device, shape, 10, 0.01).await?;
    let samples = timed!(device, config, input.sin().add(&input.cos()));
    Ok(BenchmarkReport::new(
        name,
        config,
        samples,
        format!("{} sin+cos", shape_label(&shape)),
    ))
}

pub(super) async fn activation_gelu_case(
    device: &Device,
    config: BenchmarkConfig,
    name: String,
    shape: [usize; 2],
) -> BenchmarkResult<BenchmarkReport> {
    let input = input_tensor(device, shape, 11, 0.015).await?;
    let samples = timed!(device, config, input.gelu());
    Ok(BenchmarkReport::new(
        name,
        config,
        samples,
        format!("{} gelu", shape_label(&shape)),
    ))
}

pub(super) async fn broadcast_add_case(
    device: &Device,
    config: BenchmarkConfig,
    name: String,
    rows: usize,
) -> BenchmarkResult<BenchmarkReport> {
    let matrix_shape = [rows, 512usize];
    let vector_shape = [512usize];
    let matrix = input_tensor(device, matrix_shape, 12, 0.006).await?;
    let vector = input_tensor(device, vector_shape, 13, 0.01).await?;
    let samples = timed!(device, config, {
        let vector_row = vector.reshape([1, vector_shape[0]]);
        matrix.add(&vector_row.broadcast_as(matrix_shape))
    });
    Ok(BenchmarkReport::new(
        name,
        config,
        samples,
        format!("{} + broadcast 512", shape_label(&matrix_shape)),
    ))
}

pub(super) async fn transpose_then_elementwise_case(
    device: &Device,
    config: BenchmarkConfig,
    name: String,
    shape: [usize; 2],
) -> BenchmarkResult<BenchmarkReport> {
    let input = input_tensor(device, shape, 14, 0.01).await?;
    let samples = timed!(device, config, {
        let transposed = input.transpose(0, 1);
        transposed.mul(&transposed)
    });
    Ok(BenchmarkReport::new(
        name,
        config,
        samples,
        format!("{} transpose, square", shape_label(&shape)),
    ))
}

pub(super) async fn reduction_sum_last_dim_case(
    device: &Device,
    config: BenchmarkConfig,
    name: String,
    rows: usize,
) -> BenchmarkResult<BenchmarkReport> {
    let shape = [rows, 512usize];
    let input = input_tensor(device, shape, 15, 0.004).await?;
    let samples = timed!(device, config, input.sum::<1>(1));
    Ok(BenchmarkReport::new(
        name,
        config,
        samples,
        format!("{} sum axis 1", shape_label(&shape)),
    ))
}

pub(super) async fn reduction_max_middle_axis_case(
    device: &Device,
    config: BenchmarkConfig,
    name: String,
    shape: [usize; 3],
) -> BenchmarkResult<BenchmarkReport> {
    let input = input_tensor(device, shape, 16, 0.004).await?;
    let samples = timed!(device, config, input.max::<2>(1));
    Ok(BenchmarkReport::new(
        name,
        config,
        samples,
        format!("{} max axis 1", shape_label(&shape)),
    ))
}

pub(super) async fn softmax_last_dim_case(
    device: &Device,
    config: BenchmarkConfig,
    name: String,
    rows: usize,
) -> BenchmarkResult<BenchmarkReport> {
    let shape = [rows, 256usize];
    let input = input_tensor(device, shape, 5, 0.006).await?;
    let samples = timed!(device, config, input.softmax_last_dim());
    Ok(BenchmarkReport::new(
        name,
        config,
        samples,
        format!("{} last-axis softmax", shape_label(&shape)),
    ))
}

pub(super) async fn softmax_middle_axis_case(
    device: &Device,
    config: BenchmarkConfig,
    name: String,
    shape: [usize; 3],
) -> BenchmarkResult<BenchmarkReport> {
    let input = input_tensor(device, shape, 17, 0.004).await?;
    let samples = timed!(device, config, input.softmax(1));
    Ok(BenchmarkReport::new(
        name,
        config,
        samples,
        format!("{} softmax axis 1", shape_label(&shape)),
    ))
}

pub(super) async fn layer_norm_last_dim_case(
    device: &Device,
    config: BenchmarkConfig,
    name: String,
    shape: [usize; 3],
) -> BenchmarkResult<BenchmarkReport> {
    let last_dim = shape[2];
    let input = input_tensor(device, shape, 18, 0.01).await?;
    let weight_values = deterministic_values(last_dim, 19, 0.002)
        .into_iter()
        .map(|value| value + 1.0)
        .collect::<Vec<_>>();
    let weight = values_input(device, [last_dim], &weight_values).await?;
    let bias = input_tensor(device, [last_dim], 20, 0.001).await?;
    let samples = timed!(
        device,
        config,
        input.layer_norm(&weight, Some(&bias), 1.0e-5, true)
    );
    Ok(BenchmarkReport::new(
        name,
        config,
        samples,
        format!("{} layer norm", shape_label(&shape)),
    ))
}

pub(super) async fn rms_norm_fused_case(
    device: &Device,
    config: BenchmarkConfig,
    name: String,
    shape: [usize; 3],
) -> BenchmarkResult<BenchmarkReport> {
    let last_dim = shape[2];
    let input = input_tensor(device, shape, 21, 0.01).await?;
    let weight_values = deterministic_values(last_dim, 22, 0.002)
        .into_iter()
        .map(|value| value + 1.0)
        .collect::<Vec<_>>();
    let weight = values_input(device, [last_dim], &weight_values).await?;
    let samples = timed!(device, config, input.rms_norm(&weight, 1.0e-5));
    Ok(BenchmarkReport::new(
        name,
        config,
        samples,
        format!("{} rms norm", shape_label(&shape)),
    ))
}

pub(super) async fn dense_matmul_square_case(
    device: &Device,
    config: BenchmarkConfig,
    name: String,
    size: usize,
) -> BenchmarkResult<BenchmarkReport> {
    let lhs_shape = [size, size];
    let rhs_shape = [size, size];
    let lhs = input_tensor(device, lhs_shape, 6, 0.004).await?;
    let rhs = input_tensor(device, rhs_shape, 7, 0.004).await?;
    let samples = timed!(device, config, lhs.matmul(&rhs));
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
    device: &Device,
    config: BenchmarkConfig,
    name: String,
    batch: usize,
    m: usize,
    k: usize,
) -> BenchmarkResult<BenchmarkReport> {
    let lhs_shape = [batch, m, k];
    let rhs_shape = [batch, k, m];
    let lhs = input_tensor(device, lhs_shape, 23, 0.004).await?;
    let rhs = input_tensor(device, rhs_shape, 24, 0.004).await?;
    let samples = timed!(device, config, lhs.matmul(&rhs));
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
    device: &Device,
    config: BenchmarkConfig,
    name: String,
    len: usize,
) -> BenchmarkResult<BenchmarkReport> {
    let input_shape = [4usize, 8usize, len];
    let weight_shape = [16usize, 8usize, 5usize];
    let bias_shape = [16usize];
    let input = input_tensor(device, input_shape, 25, 0.01).await?;
    let weight = input_tensor(device, weight_shape, 26, 0.01).await?;
    let bias = input_tensor(device, bias_shape, 27, 0.001).await?;
    let samples = timed!(
        device,
        config,
        input.conv::<3, 1, 4>(&weight, Some(&bias), [2], [1])
    );
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
    device: &Device,
    config: BenchmarkConfig,
    name: String,
    input_len: usize,
    k: usize,
    values: Vec<f32>,
) -> BenchmarkResult<BenchmarkReport> {
    let input = values_input(device, [input_len], &values).await?;
    let samples = time_samples(config, || {
        let (top_values, top_indices) = input.top_k(k as u32);
        async move {
            materialize(device, &top_values).await?;
            materialize(device, &top_indices).await?;
            let got = top_indices.elem_count().unwrap_or(0) as usize;
            if got != k {
                return Err(format!("top_k returned {got} pairs, expected {k}").into());
            }
            Ok(())
        }
    })
    .await?;
    Ok(BenchmarkReport::new(
        name,
        config,
        samples,
        format!("{input_len} logits, k={k}"),
    ))
}

pub(super) async fn q8_0_qgemv_case(
    device: &Device,
    config: BenchmarkConfig,
    name: String,
    weight_shape: [usize; 2],
) -> BenchmarkResult<BenchmarkReport> {
    let input_shape = [1usize, weight_shape[1]];
    let matrix = qmatrix(device, QFmt::Q8_0, weight_shape)?;
    let input = input_tensor(device, input_shape, 8, 0.003).await?;
    let samples = timed!(device, config, input.q_mat_mul(&matrix));
    Ok(BenchmarkReport::new(
        name,
        config,
        samples,
        format!(
            "1x{} @ Q8_0 {}",
            weight_shape[1],
            shape_label(&weight_shape)
        ),
    ))
}

pub(super) async fn q4k_qgemv_case(
    device: &Device,
    config: BenchmarkConfig,
    name: String,
    weight_shape: [usize; 2],
) -> BenchmarkResult<BenchmarkReport> {
    let input_shape = [1usize, weight_shape[1]];
    let matrix = qmatrix(device, QFmt::Q4K, weight_shape)?;
    let input = input_tensor(device, input_shape, 29, 0.003).await?;
    let samples = timed!(device, config, input.q_mat_mul(&matrix));
    Ok(BenchmarkReport::new(
        name,
        config,
        samples,
        format!("1x{} @ Q4K {}", weight_shape[1], shape_label(&weight_shape)),
    ))
}

pub(super) async fn q4k_paired_silu_case(
    device: &Device,
    config: BenchmarkConfig,
    name: String,
    weight_shape: [usize; 2],
) -> BenchmarkResult<BenchmarkReport> {
    let input_shape = [1usize, weight_shape[1]];
    let matrix = qmatrix(device, QFmt::Q4K, weight_shape)?;
    let input = input_tensor(device, input_shape, 30, 0.003).await?;
    let pair_len = weight_shape[0] / 2;
    let samples = timed!(device, config, {
        let projected = input.q_mat_mul(&matrix);
        let gate = projected.narrow(1, 0, pair_len);
        let up = projected.narrow(1, pair_len, pair_len);
        gate.silu().mul(&up)
    });
    Ok(BenchmarkReport::new(
        name,
        config,
        samples,
        format!(
            "1x{} @ paired Q4K {}",
            weight_shape[1],
            shape_label(&weight_shape)
        ),
    ))
}

pub(super) async fn attention_case(
    device: &Device,
    config: BenchmarkConfig,
    name: String,
    seq_len: usize,
    seeds: [usize; 3],
    causal: bool,
    detail_op: &'static str,
) -> BenchmarkResult<BenchmarkReport> {
    let shape = [1usize, 4usize, seq_len, 64usize];
    let q = input_tensor(device, shape, seeds[0], 0.003).await?;
    let k = input_tensor(device, shape, seeds[1], 0.003).await?;
    let v = input_tensor(device, shape, seeds[2], 0.003).await?;
    let mask = if causal {
        MaskKind::Causal
    } else {
        MaskKind::None
    };
    let samples = timed!(
        device,
        config,
        q.attention(&k, &v, mask, Some(1.0 / (64.0f32).sqrt()))
    );
    Ok(BenchmarkReport::new(
        name,
        config,
        samples,
        format!("{} {detail_op}", shape_label(&shape)),
    ))
}

pub(super) async fn rope_fused_decode_case(
    device: &Device,
    config: BenchmarkConfig,
    name: String,
    seq_len: usize,
) -> BenchmarkResult<BenchmarkReport> {
    let shape = [1usize, 8usize, seq_len, 64usize];
    let [_, _, _, head_dim] = shape;
    let pos_shape = [seq_len * 2, head_dim / 2];
    let input = input_tensor(device, shape, 9, 0.01).await?;
    let cos = values_input(device, pos_shape, &rope_values(pos_shape, head_dim, true)).await?;
    let sin = values_input(device, pos_shape, &rope_values(pos_shape, head_dim, false)).await?;
    let samples = timed!(device, config, input.rope(&cos, &sin, 0));
    Ok(BenchmarkReport::new(
        name,
        config,
        samples,
        format!("{} rope", shape_label(&shape)),
    ))
}

macro_rules! fixed_cases {
    ($($case:ident => $body:ident($($arg:expr),* $(,)?);)*) => {
        $(
            pub fn $case() -> BenchmarkCase {
                bench_case(
                    concat!("webgpu::", stringify!($case)),
                    |device, config, name| Box::pin($body(device, config, name, $($arg),*)),
                )
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
    top_k_large => top_k_case(65_537, 64, topk_values(65_537));
    top_k_qwen_vocab => top_k_case(151_936, 40, deterministic_values(151_936, 28, 0.01));
    q8_0_qgemv => q8_0_qgemv_case([4096, 896]);
    q4k_qgemv => q4k_qgemv_case([2048, 1024]);
    q4k_paired_silu => q4k_paired_silu_case([2048, 1024]);
    attention_small => attention_case(128, [31, 32, 33], false, "attention");
    attention_causal_small => attention_case(128, [34, 35, 36], true, "causal attention");
    rope_fused_decode => rope_fused_decode_case(256);
}
