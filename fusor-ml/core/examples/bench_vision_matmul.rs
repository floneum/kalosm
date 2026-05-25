// Microbench for the vision-encoder matmul shapes hit during Qwen2.5-VL
// prefill. The vision encoder runs ~133 dense f32 matmuls per image; this
// binary isolates one shape and reports per-call ms so we can compare the
// shared-tile fallback against any new kernel variant in isolation.
//
// Run with:
//   cargo run --package fusor-core --example bench_vision_matmul --release

use std::time::{Duration, Instant};

use fusor_core::{Device, Tensor};

const WARMUP_BATCHES: usize = 3;
const MEASURED_BATCHES: usize = 10;
const DISPATCHES_PER_BATCH: usize = 4;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    pollster::block_on(async {
        let device = Device::new().await?;
        println!("bench_vision_matmul");
        println!("warmup_batches: {WARMUP_BATCHES}");
        println!("measured_batches: {MEASURED_BATCHES}");
        println!("dispatches_per_batch: {DISPATCHES_PER_BATCH}");
        println!();

        // Shapes that show up in the Qwen2.5-VL vision encoder prefill for the
        // standard demo image (M=1944 = 1944 patches after merge). N varies by
        // projection; K is 1280 (embed dim) or 3420 (mlp inner / 2).
        let cases: &[(&str, usize, usize, usize)] = &[
            ("vision_qkv", 1944, 1280, 3840),         // fused qkv projection
            ("vision_o", 1944, 1280, 1280),           // attention output proj
            ("vision_mlp_gate_up", 1944, 1280, 6840), // gate+up fused
            ("vision_mlp_down", 1944, 3420, 1280),    // down proj
            // Aligned reference shape (M%128=0) so we can see how much faster the
            // coop tile path is for the same K/N when M is friendly.
            ("aligned_ref_1920_qkv", 1920, 1280, 3840),
            ("aligned_ref_2048_qkv", 2048, 1280, 3840),
        ];

        for &(name, m, k, n) in cases {
            bench_matmul(&device, name, m, k, n);
        }

        bench_flash_attention_vision(&device);

        Ok(())
    })
}

fn bench_flash_attention_vision(device: &Device) {
    // Mirror the vision attention shape: 16 heads, seq=1944, head_dim=80,
    // unmasked self-attention (the per-window mask is dense and irrelevant
    // to throughput).
    let q = Tensor::splat(device, 0.1f32, [1, 16, 1944, 80]);
    let k = Tensor::splat(device, 0.1f32, [1, 16, 1944, 80]);
    let v = Tensor::splat(device, 0.1f32, [1, 16, 1944, 80]);
    q.materialize_sync();
    k.materialize_sync();
    v.materialize_sync();

    // Cold-start
    let cold = Instant::now();
    {
        let y = q.flash_attention(&k, &v, 1.0 / (80f32).sqrt(), None);
        let key = y.key();
        device.resolve_batch(&[key]);
        device.poll_wait();
        drop(y);
    }
    let cold_elapsed = cold.elapsed();

    for _ in 0..WARMUP_BATCHES {
        let y = q.flash_attention(&k, &v, 1.0 / (80f32).sqrt(), None);
        let key = y.key();
        device.resolve_batch(&[key]);
        device.poll_wait();
        drop(y);
    }

    let mut samples = Vec::with_capacity(MEASURED_BATCHES);
    for _ in 0..MEASURED_BATCHES {
        let start = Instant::now();
        let y = q.flash_attention(&k, &v, 1.0 / (80f32).sqrt(), None);
        let key = y.key();
        device.resolve_batch(&[key]);
        device.poll_wait();
        samples.push(start.elapsed());
        drop(y);
    }

    let mean = mean_duration(&samples);
    let mut sorted = samples.clone();
    sorted.sort_unstable();
    let p50 = percentile_duration(&sorted, 50);
    let p90 = percentile_duration(&sorted, 90);
    let min = sorted.first().copied().unwrap_or_default();
    println!();
    println!("vision_flash_attention NO MASK (Q=K=V [1,16,1944,80]):");
    println!("  cold ms: {:.3}", cold_elapsed.as_secs_f64() * 1000.0);
    println!("  mean_ms: {:.3}", mean.as_secs_f64() * 1000.0);
    println!("  p50_ms:  {:.3}", p50.as_secs_f64() * 1000.0);
    println!("  p90_ms:  {:.3}", p90.as_secs_f64() * 1000.0);
    println!("  min_ms:  {:.3}", min.as_secs_f64() * 1000.0);

    // Now bench WITH a mask (matches vision attention call site).
    let mask: Tensor = Tensor::splat(device, 0.0f32, [1944, 1944]);
    mask.materialize_sync();

    // Bench with TRANSPOSED Q/K/V layout — the model produces Q via
    // `xs.transpose(0,1).unsqueeze(0)` so the underlying tensor has
    // non-contiguous strides. If that defeats a fast path in
    // `try_flash_attention_direct`, this case will be much slower.
    // Build a [1, 16, 1944, 80] tensor whose underlying layout is the
    // [1944, 16, 80] memory order — same as `q.transpose(0, 1).unsqueeze(0)`
    // in the model. We do this by allocating [1944, 16, 80] and then using
    // restride to expose it as [1, 16, 1944, 80] with the seq/head strides
    // swapped (head: 80, seq: 16*80=1280).
    // (the transposed-Q/K/V bench was removed — see kernel-level analysis
    //  in qwen_vision_block.rs: the issue is V's non-contiguous layout
    //  defeats coalesced GPU loads in the streaming flash kernel.)
    let _ = mask;

    for _ in 0..WARMUP_BATCHES {
        let y = q.flash_attention(&k, &v, 1.0 / (80f32).sqrt(), Some(&mask));
        let key = y.key();
        device.resolve_batch(&[key]);
        device.poll_wait();
        drop(y);
    }

    let mut masked_samples = Vec::with_capacity(MEASURED_BATCHES);
    for _ in 0..MEASURED_BATCHES {
        let start = Instant::now();
        let y = q.flash_attention(&k, &v, 1.0 / (80f32).sqrt(), Some(&mask));
        let key = y.key();
        device.resolve_batch(&[key]);
        device.poll_wait();
        masked_samples.push(start.elapsed());
        drop(y);
    }
    let mut sorted_m = masked_samples.clone();
    sorted_m.sort_unstable();
    println!();
    println!("vision_flash_attention WITH MASK (1944x1944):");
    println!(
        "  mean_ms: {:.3}",
        mean_duration(&masked_samples).as_secs_f64() * 1000.0
    );
    println!(
        "  p50_ms:  {:.3}",
        percentile_duration(&sorted_m, 50).as_secs_f64() * 1000.0
    );
    println!(
        "  p90_ms:  {:.3}",
        percentile_duration(&sorted_m, 90).as_secs_f64() * 1000.0
    );
    println!(
        "  min_ms:  {:.3}",
        sorted_m.first().copied().unwrap_or_default().as_secs_f64() * 1000.0
    );
}

fn bench_matmul(device: &Device, name: &str, m: usize, k: usize, n: usize) {
    let a = Tensor::splat(device, 0.001f32, [1, m, k]);
    let b = Tensor::splat(device, 0.001f32, [1, k, n]);
    a.materialize_sync();
    b.materialize_sync();

    // Cold-start: measure the first dispatch separately. Shader pipeline
    // creation (WGSL -> MSL -> Metal pipeline) happens lazily on first use
    // — if this is large relative to warm runs, shader compile is the
    // dominant cost in real prefill, not matmul math.
    let cold_start = Instant::now();
    {
        let y = a.mat_mul(&b);
        let key = y.key();
        device.resolve_batch(&[key]);
        device.poll_wait();
        drop(y);
    }
    let cold = cold_start.elapsed();
    println!("  COLD first-call ms: {:.3}", cold.as_secs_f64() * 1000.0);

    for _ in 0..WARMUP_BATCHES {
        run_batch(device, &a, &b);
    }

    let mut samples = Vec::with_capacity(MEASURED_BATCHES);
    let mut kernels = 0usize;
    for _ in 0..MEASURED_BATCHES {
        let (elapsed, k_count) = run_batch(device, &a, &b);
        samples.push(elapsed / DISPATCHES_PER_BATCH as u32);
        kernels = k_count;
    }

    let gflops = {
        // M * N * K * 2 (mul + add) per matmul
        let ops_per_call = (m as f64) * (n as f64) * (k as f64) * 2.0;
        let mean_secs = mean_duration(&samples).as_secs_f64();
        if mean_secs > 0.0 {
            ops_per_call / mean_secs / 1.0e9
        } else {
            0.0
        }
    };

    print_summary(name, m, k, n, &samples, kernels, gflops);
}

fn run_batch(device: &Device, a: &Tensor, b: &Tensor) -> (Duration, usize) {
    let mut keys = Vec::with_capacity(DISPATCHES_PER_BATCH);
    let mut outputs = Vec::with_capacity(DISPATCHES_PER_BATCH);
    for _ in 0..DISPATCHES_PER_BATCH {
        let y = a.mat_mul(b);
        keys.push(y.key());
        outputs.push(y);
    }
    let start = Instant::now();
    let kernels = device.resolve_batch(&keys);
    device.poll_wait();
    let elapsed = start.elapsed();
    drop(outputs);
    (elapsed, kernels)
}

fn print_summary(
    name: &str,
    m: usize,
    k: usize,
    n: usize,
    samples: &[Duration],
    kernels: usize,
    gflops: f64,
) {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let mean = mean_duration(samples);
    let p50 = percentile_duration(&sorted, 50);
    let p90 = percentile_duration(&sorted, 90);
    let min = sorted.first().copied().unwrap_or_default();
    let max = sorted.last().copied().unwrap_or_default();
    println!();
    println!("{name} (M={m}, K={k}, N={n}):");
    println!("  kernels_per_dispatch: {kernels}");
    println!("  mean_ms: {:.3}", mean.as_secs_f64() * 1000.0);
    println!("  p50_ms:  {:.3}", p50.as_secs_f64() * 1000.0);
    println!("  p90_ms:  {:.3}", p90.as_secs_f64() * 1000.0);
    println!("  min_ms:  {:.3}", min.as_secs_f64() * 1000.0);
    println!("  max_ms:  {:.3}", max.as_secs_f64() * 1000.0);
    println!("  gflops_mean: {gflops:.1}");
    println!();
}

fn mean_duration(samples: &[Duration]) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }
    samples.iter().copied().sum::<Duration>() / samples.len() as u32
}

fn percentile_duration(sorted: &[Duration], percentile: usize) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let index = ((sorted.len() - 1) * percentile) / 100;
    sorted[index]
}
