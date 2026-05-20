use std::time::Duration;

use fusor_core::{Device, QMatrix, Tensor};
use fusor_gguf::{BlockQ4K, BlockQ6K, GgmlType};

const M: usize = 1;
const DEFAULT_K: usize = 4096;
const DEFAULT_N: usize = 28672;
const DEFAULT_WARMUP_BATCHES: usize = 3;
const DEFAULT_MEASURED_BATCHES: usize = 20;
const DEFAULT_DISPATCHES_PER_BATCH: usize = 16;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_format(name: &str, default: GgmlType) -> GgmlType {
    match std::env::var(name).as_deref() {
        Ok("Q4K") | Ok("q4k") => GgmlType::Q4K,
        Ok("Q6K") | Ok("q6k") => GgmlType::Q6K,
        _ => default,
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let k = env_usize("FUSOR_QMAT_BENCH_K", DEFAULT_K);
    let n = env_usize("FUSOR_QMAT_BENCH_N", DEFAULT_N);
    let format = env_format("FUSOR_QMAT_BENCH_FORMAT", GgmlType::Q4K);
    let warmup_batches = env_usize("FUSOR_QMAT_BENCH_WARMUP_BATCHES", DEFAULT_WARMUP_BATCHES);
    let measured_batches = env_usize(
        "FUSOR_QMAT_BENCH_MEASURED_BATCHES",
        DEFAULT_MEASURED_BATCHES,
    );
    let dispatches_per_batch = env_usize(
        "FUSOR_QMAT_BENCH_DISPATCHES_PER_BATCH",
        DEFAULT_DISPATCHES_PER_BATCH,
    )
    .max(1);

    let device = Device::new().await?;
    let input_data = vec![vec![0.25f32; k]; M];
    let input: Tensor<2, f32> = Tensor::new(&device, &input_data);
    input.materialize().await;

    let raw_weight = weight_bytes(n, k, format);
    let weight = QMatrix::from_parts(&device, &raw_weight, Box::new([n, k]), format)?;

    for _ in 0..warmup_batches {
        run_batch(&device, &input, &weight, dispatches_per_batch);
    }

    let mut samples = Vec::with_capacity(measured_batches);
    let mut kernels = 0usize;
    for _ in 0..measured_batches {
        let start = std::time::Instant::now();
        kernels = run_batch(&device, &input, &weight, dispatches_per_batch);
        samples.push(start.elapsed() / dispatches_per_batch as u32);
    }

    samples.sort_unstable();
    let mean = mean_duration(&samples);
    let p50 = percentile_duration(&samples, 50);
    let p90 = percentile_duration(&samples, 90);
    let min = samples.first().copied().unwrap_or_default();
    let max = samples.last().copied().unwrap_or_default();
    let flops = 2.0 * M as f64 * n as f64 * k as f64;
    let weight_bytes = raw_weight.len() as f64;

    println!("bench_llama_qmat_bottleneck");
    println!("shape: A={M}x{k} B={k}x{n} -> Y={M}x{n}");
    println!("kernel: q_mat_mul_f32_1x1x{k}_{format:?}_{n}x{k}");
    println!("format: {:?}", format);
    println!("warmup_batches: {warmup_batches}");
    println!("measured_batches: {measured_batches}");
    println!("dispatches_per_batch: {dispatches_per_batch}");
    println!("kernels_per_batch: {kernels}");
    println!(
        "kernels_per_dispatch: {:.3}",
        kernels as f64 / dispatches_per_batch as f64
    );
    println!("mean_dispatch_time_us: {:.3}", duration_us(mean));
    println!("p50_dispatch_time_us: {:.3}", duration_us(p50));
    println!("p90_dispatch_time_us: {:.3}", duration_us(p90));
    println!("min_dispatch_time_us: {:.3}", duration_us(min));
    println!("max_dispatch_time_us: {:.3}", duration_us(max));
    println!(
        "effective_gflops: {:.6}",
        flops / mean.as_secs_f64() / 1.0e9
    );
    println!(
        "packed_weight_bandwidth_gb_s: {:.6}",
        weight_bytes / mean.as_secs_f64() / 1.0e9
    );

    Ok(())
}

fn run_batch(
    device: &Device,
    input: &Tensor<2, f32>,
    weight: &QMatrix,
    dispatches: usize,
) -> usize {
    let mut outputs = Vec::with_capacity(dispatches);
    let mut keys = Vec::with_capacity(dispatches);
    for _ in 0..dispatches {
        let output = input.q_mat_mul(weight);
        keys.push(output.key());
        outputs.push(output);
    }
    let kernels = device.resolve_batch(&keys);
    device.poll_wait();
    drop(outputs);
    kernels
}

fn weight_bytes(rows: usize, cols: usize, format: GgmlType) -> Vec<u8> {
    match format {
        GgmlType::Q4K => block_weight_bytes::<BlockQ4K>(rows, cols, "Q4K"),
        GgmlType::Q6K => block_weight_bytes::<BlockQ6K>(rows, cols, "Q6K"),
        other => panic!("unsupported benchmark format {other:?}"),
    }
}

fn block_weight_bytes<B>(rows: usize, cols: usize, name: &str) -> Vec<u8>
where
    B: fusor_gguf::GgufBlock,
{
    let elements = rows
        .checked_mul(cols)
        .expect("qmat benchmark shape should not overflow");
    assert!(
        elements.is_multiple_of(B::BLOCK_SIZE),
        "{name} element count must be divisible by the block size"
    );
    vec![0; elements / B::BLOCK_SIZE * std::mem::size_of::<B>()]
}

fn mean_duration(samples: &[Duration]) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }
    samples.iter().copied().sum::<Duration>() / samples.len() as u32
}

fn percentile_duration(samples: &[Duration], percentile: usize) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }
    let index = ((samples.len() - 1) * percentile).div_ceil(100);
    samples[index]
}

fn duration_us(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1.0e6
}
