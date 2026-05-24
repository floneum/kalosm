use std::time::Duration;

use fusor_core::{Device, QMatrix, Tensor};
use fusor_gguf::{BlockQ4K, BlockQ6K, GgmlType};

const DEFAULT_M: usize = 1;
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    pollster::block_on(async {
        let k = env_usize("FUSOR_QMAT_BENCH_K", DEFAULT_K);
        let n = env_usize("FUSOR_QMAT_BENCH_N", DEFAULT_N);
        let m = env_usize("FUSOR_QMAT_BENCH_M", DEFAULT_M);
        let format = env_format("FUSOR_QMAT_BENCH_FORMAT", GgmlType::Q4K);
        let paired = std::env::var_os("FUSOR_QMAT_BENCH_PAIRED").is_some();
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
        let rank3 = std::env::var_os("FUSOR_QMAT_BENCH_RANK3").is_some();
        let add2 = std::env::var_os("FUSOR_QMAT_BENCH_ADD2").is_some();
        assert!(
            !add2 || !paired,
            "ADD2 and PAIRED modes are mutually exclusive"
        );

        let device = Device::new().await?;
        let raw_weight = weight_bytes(n, k, format);
        let weight = QMatrix::from_parts(&device, &raw_weight, Box::new([n, k]), format)?;

        let (samples, kernels) = if rank3 {
            let input_data = vec![vec![vec![0.25f32; k]; m]];
            let input: Tensor<3, f32> = Tensor::new(&device, &input_data);
            input.materialize().await;
            let first_data = vec![vec![vec![0.0f32; n]; m]];
            let first: Tensor<3, f32> = Tensor::new(&device, &first_data);
            first.materialize().await;
            let second_data = vec![vec![vec![0.0f32; n]; m]];
            let second: Tensor<3, f32> = Tensor::new(&device, &second_data);
            second.materialize().await;
            measure_qmat(
                &device,
                &input,
                &weight,
                add2.then_some((&first, &second)),
                warmup_batches,
                measured_batches,
                dispatches_per_batch,
                paired,
            )
        } else {
            let input_data = vec![vec![0.25f32; k]; m];
            let input: Tensor<2, f32> = Tensor::new(&device, &input_data);
            input.materialize().await;
            let first_data = vec![vec![0.0f32; n]; m];
            let first: Tensor<2, f32> = Tensor::new(&device, &first_data);
            first.materialize().await;
            let second_data = vec![vec![0.0f32; n]; m];
            let second: Tensor<2, f32> = Tensor::new(&device, &second_data);
            second.materialize().await;
            measure_qmat(
                &device,
                &input,
                &weight,
                add2.then_some((&first, &second)),
                warmup_batches,
                measured_batches,
                dispatches_per_batch,
                paired,
            )
        };

        let mut samples = samples;
        samples.sort_unstable();
        let mean = mean_duration(&samples);
        let p50 = percentile_duration(&samples, 50);
        let p90 = percentile_duration(&samples, 90);
        let min = samples.first().copied().unwrap_or_default();
        let max = samples.last().copied().unwrap_or_default();
        let output_cols = if paired { n / 2 } else { n };
        let flops = 2.0 * m as f64 * n as f64 * k as f64;
        let weight_bytes = raw_weight.len() as f64;

        println!("bench_llama_qmat_bottleneck");
        println!("shape: A={m}x{k} B={k}x{n} -> Y={m}x{output_cols}");
        println!("rank3: {rank3}");
        println!("add2: {add2}");
        println!("paired: {paired}");
        println!("kernel: q_mat_mul_f32_1x{m}x{k}_{format:?}_{n}x{k}");
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
    })
}

fn measure_qmat<const R: usize>(
    device: &Device,
    input: &Tensor<R, f32>,
    weight: &QMatrix,
    add2: Option<(&Tensor<R, f32>, &Tensor<R, f32>)>,
    warmup_batches: usize,
    measured_batches: usize,
    dispatches_per_batch: usize,
    paired: bool,
) -> (Vec<Duration>, usize) {
    for _ in 0..warmup_batches {
        run_batch(device, input, weight, add2, dispatches_per_batch, paired);
    }

    let mut samples = Vec::with_capacity(measured_batches);
    let mut kernels = 0usize;
    for _ in 0..measured_batches {
        let start = std::time::Instant::now();
        kernels = run_batch(device, input, weight, add2, dispatches_per_batch, paired);
        samples.push(start.elapsed() / dispatches_per_batch as u32);
    }

    (samples, kernels)
}

fn run_batch<const R: usize>(
    device: &Device,
    input: &Tensor<R, f32>,
    weight: &QMatrix,
    add2: Option<(&Tensor<R, f32>, &Tensor<R, f32>)>,
    dispatches: usize,
    paired: bool,
) -> usize {
    let mut outputs = Vec::with_capacity(dispatches);
    let mut keys = Vec::with_capacity(dispatches);
    for _ in 0..dispatches {
        let output = if let Some((first, second)) = add2 {
            input.q_mat_mul_add2(weight, first, second)
        } else if paired {
            input.q_mat_mul_paired_silu_product(weight)
        } else {
            input.q_mat_mul(weight)
        };
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
