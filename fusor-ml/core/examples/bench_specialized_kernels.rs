use std::time::Duration;

use fusor_core::{Device, GpuMirostat2Sampler, GpuMirostat2SamplerParams, Tensor};

const DEFAULT_WARMUP_BATCHES: usize = 3;
const DEFAULT_MEASURED_BATCHES: usize = 20;
const DEFAULT_DISPATCHES_PER_BATCH: usize = 16;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let warmup_batches = env_usize(
        "FUSOR_SPECIALIZED_BENCH_WARMUP_BATCHES",
        DEFAULT_WARMUP_BATCHES,
    );
    let measured_batches = env_usize(
        "FUSOR_SPECIALIZED_BENCH_MEASURED_BATCHES",
        DEFAULT_MEASURED_BATCHES,
    );
    let dispatches_per_batch = env_usize(
        "FUSOR_SPECIALIZED_BENCH_DISPATCHES_PER_BATCH",
        DEFAULT_DISPATCHES_PER_BATCH,
    )
    .max(1);

    let device = Device::new().await?;
    println!("bench_specialized_kernels");
    println!("warmup_batches: {warmup_batches}");
    println!("measured_batches: {measured_batches}");
    println!("dispatches_per_batch: {dispatches_per_batch}");

    bench_rms_norm_vec4(
        &device,
        warmup_batches,
        measured_batches,
        dispatches_per_batch,
    );
    bench_flash_attention_streaming(
        &device,
        warmup_batches,
        measured_batches,
        dispatches_per_batch,
    );
    bench_flash_attention_decode(
        &device,
        warmup_batches,
        measured_batches,
        dispatches_per_batch,
    );
    bench_top_k_pairs(&device, warmup_batches, measured_batches).await?;
    bench_mirostat2(&device, warmup_batches, measured_batches).await?;

    Ok(())
}

fn bench_rms_norm_vec4(
    device: &Device,
    warmup_batches: usize,
    measured_batches: usize,
    dispatches_per_batch: usize,
) {
    let input = Tensor::<2, f32>::splat(device, 0.25, [64, 4096]);
    let weight = Tensor::<1, f32>::splat(device, 1.0, [4096]);
    input.materialize_sync();
    weight.materialize_sync();

    bench_tensor_case(
        "rms_norm_vec4_64x4096",
        warmup_batches,
        measured_batches,
        dispatches_per_batch,
        || input.rms_norm_fused::<1, 1>(&weight, None, 1e-5),
        device,
    );
}

fn bench_flash_attention_streaming(
    device: &Device,
    warmup_batches: usize,
    measured_batches: usize,
    dispatches_per_batch: usize,
) {
    let q = Tensor::<4, f32>::splat(device, 0.125, [1, 32, 48, 128]);
    let k = Tensor::<4, f32>::splat(device, 0.25, [1, 8, 48, 128]);
    let v = Tensor::<4, f32>::splat(device, 0.5, [1, 8, 48, 128]);
    q.materialize_sync();
    k.materialize_sync();
    v.materialize_sync();

    bench_tensor_case(
        "flash_attention_streaming_1x32x48x128_by_8x48",
        warmup_batches,
        measured_batches,
        dispatches_per_batch,
        || q.flash_attention(&k, &v, 1.0 / f32::sqrt(128.0), None),
        device,
    );
}

fn bench_flash_attention_decode(
    device: &Device,
    warmup_batches: usize,
    measured_batches: usize,
    dispatches_per_batch: usize,
) {
    let q = Tensor::<4, f32>::splat(device, 0.125, [1, 32, 1, 128]);
    let k = Tensor::<4, f32>::splat(device, 0.25, [1, 8, 512, 128]);
    let v = Tensor::<4, f32>::splat(device, 0.5, [1, 8, 512, 128]);
    q.materialize_sync();
    k.materialize_sync();
    v.materialize_sync();

    bench_tensor_case(
        "flash_attention_decode_1x32x1x128_by_8x512",
        warmup_batches,
        measured_batches,
        dispatches_per_batch,
        || q.flash_attention(&k, &v, 1.0 / f32::sqrt(128.0), None),
        device,
    );
}

fn bench_tensor_case<const R: usize, F>(
    name: &str,
    warmup_batches: usize,
    measured_batches: usize,
    dispatches_per_batch: usize,
    mut make_output: F,
    device: &Device,
) where
    F: FnMut() -> Tensor<R, f32>,
{
    for _ in 0..warmup_batches {
        run_tensor_batch(dispatches_per_batch, &mut make_output, device);
    }

    let mut samples = Vec::with_capacity(measured_batches);
    let mut kernels = 0usize;
    for _ in 0..measured_batches {
        let (elapsed, batch_kernels) =
            run_tensor_batch(dispatches_per_batch, &mut make_output, device);
        samples.push(elapsed / dispatches_per_batch as u32);
        kernels = batch_kernels;
    }

    print_summary(
        name,
        &samples,
        Some(kernels as f64 / dispatches_per_batch as f64),
    );
}

fn run_tensor_batch<const R: usize, F>(
    dispatches: usize,
    make_output: &mut F,
    device: &Device,
) -> (Duration, usize)
where
    F: FnMut() -> Tensor<R, f32>,
{
    let mut outputs = Vec::with_capacity(dispatches);
    let mut keys = Vec::with_capacity(dispatches);
    for _ in 0..dispatches {
        let output = make_output();
        keys.push(output.key());
        outputs.push(output);
    }

    let start = std::time::Instant::now();
    let kernels = device.resolve_batch(&keys);
    device.poll_wait();
    let elapsed = start.elapsed();
    drop(outputs);

    (elapsed, kernels)
}

async fn bench_top_k_pairs(
    device: &Device,
    warmup_batches: usize,
    measured_batches: usize,
) -> Result<(), wgpu::BufferAsyncError> {
    let logits_data = bench_logits(8192);
    let logits = Tensor::<1, f32>::new(device, &logits_data);
    logits.materialize_sync();

    for _ in 0..warmup_batches {
        let _ = logits.top_k_pairs(512).await?;
    }

    let mut samples = Vec::with_capacity(measured_batches);
    for _ in 0..measured_batches {
        let start = std::time::Instant::now();
        let _ = logits.top_k_pairs(512).await?;
        samples.push(start.elapsed());
    }

    print_summary("top_k_pairs_8192_k512", &samples, None);
    Ok(())
}

async fn bench_mirostat2(
    device: &Device,
    warmup_batches: usize,
    measured_batches: usize,
) -> Result<(), wgpu::BufferAsyncError> {
    let logits_data = bench_logits(8192);
    let logits = Tensor::<1, f32>::new(device, &logits_data);
    logits.materialize_sync();
    let params = GpuMirostat2SamplerParams {
        top_k: 512,
        temperature: 0.8,
        repetition_penalty: 1.05,
        tau: 5.0,
        eta: 0.1,
        random: 0.35,
    };

    for _ in 0..warmup_batches {
        let mut sampler = GpuMirostat2Sampler::new(device, 2.0 * params.tau);
        let _ = logits
            .sample_mirostat2_token(&mut sampler, &[], params)
            .await?;
    }

    let mut samples = Vec::with_capacity(measured_batches);
    for _ in 0..measured_batches {
        let mut sampler = GpuMirostat2Sampler::new(device, 2.0 * params.tau);
        let start = std::time::Instant::now();
        let _ = logits
            .sample_mirostat2_token(&mut sampler, &[], params)
            .await?;
        samples.push(start.elapsed());
    }

    print_summary("mirostat2_8192_k512", &samples, None);
    Ok(())
}

fn bench_logits(len: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let chunk_rank = index % 256;
            let chunk = index / 256;
            10.0 - chunk_rank as f32 * 0.01 - chunk as f32 * 0.0001
        })
        .collect()
}

fn print_summary(name: &str, samples: &[Duration], kernels_per_dispatch: Option<f64>) {
    let mut samples = samples.to_vec();
    samples.sort_unstable();
    let mean = mean_duration(&samples);
    let p50 = percentile_duration(&samples, 50);
    let p90 = percentile_duration(&samples, 90);
    let min = samples.first().copied().unwrap_or_default();
    let max = samples.last().copied().unwrap_or_default();

    println!("{name}:");
    if let Some(kernels_per_dispatch) = kernels_per_dispatch {
        println!("  kernels_per_dispatch: {kernels_per_dispatch:.3}");
    }
    println!("  mean_us: {:.3}", duration_us(mean));
    println!("  p50_us: {:.3}", duration_us(p50));
    println!("  p90_us: {:.3}", duration_us(p90));
    println!("  min_us: {:.3}", duration_us(min));
    println!("  max_us: {:.3}", duration_us(max));
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
