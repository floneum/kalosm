//! Microbenchmark for individual GPU kernel shapes — mirrors the transformer
//! training profile measurement mode. Runs batches of independent operations
//! per resolve; the resolver's GPU timestamp profile is read back
//! programmatically and printed as machine-diffable `kernel_profile` /
//! `kernel_profile_category` lines. Only the first two resolves per process
//! are profiled (later rounds replay the recorded materialization plan), so
//! run each case's process three times for six samples.
//!
//! Usage:
//! cargo run --release -p fusor --example kernel_bench -- [case]

use fusor::{Device, FusorConfig, Tensor};

const REPEATS: usize = 16;
const ROUNDS: usize = 6;

fn fill(seed: u32, len: usize) -> Vec<f32> {
    let mut state = seed as u64 | 1;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

fn print_profiles(case: &str, device: &Device) {
    let Some(gpu) = device.as_gpu() else {
        return;
    };
    for profile in gpu.take_kernel_profiles() {
        println!(
            "kernel_profile case={case} mode={} kernels={} accounted_ms={:.3} span_ms={:.3}",
            profile.timestamp_mode, profile.kernels, profile.accounted_ms, profile.span_ms
        );
        let mut categories = profile.categories;
        categories.sort_by(|a, b| a.name.cmp(&b.name));
        for row in categories {
            println!(
                "kernel_profile_category case={case} category={} count={} total_ms={:.3} avg_us={:.1} max_us={:.1}",
                row.name, row.count, row.total_ms, row.average_us, row.max_us
            );
        }
    }
}

async fn bench_matmul(case: &str, device: &Device, m: usize, k: usize, n: usize) {
    println!("=== matmul {m}x{k} by {k}x{n} ===");
    for _ in 0..ROUNDS {
        let outputs: Vec<Tensor<2, f32>> = (0..REPEATS)
            .map(|i| {
                let a = Tensor::from_slice(device, [m, k], &fill(3 + i as u32, m * k));
                let b = Tensor::from_slice(device, [k, n], &fill(77 + i as u32, k * n));
                a.mat_mul(&b)
            })
            .collect();
        device.flush();
        let mut total = 0.0f32;
        for out in outputs {
            let slice = out.as_slice().await.unwrap();
            total += slice[[0, 0]];
        }
        println!("checksum {total}");
        print_profiles(case, device);
    }
}

async fn bench_batched_matmul(
    case: &str,
    device: &Device,
    b0: usize,
    b1: usize,
    m: usize,
    k: usize,
    n: usize,
) {
    println!("=== batched matmul [{b0},{b1}] {m}x{k} by {k}x{n} ===");
    for _ in 0..ROUNDS {
        let outputs: Vec<Tensor<4, f32>> = (0..REPEATS)
            .map(|i| {
                let a = Tensor::from_slice(
                    device,
                    [b0, b1, m, k],
                    &fill(3 + i as u32, b0 * b1 * m * k),
                );
                let b = Tensor::from_slice(
                    device,
                    [b0, b1, k, n],
                    &fill(77 + i as u32, b0 * b1 * k * n),
                );
                a.mat_mul(&b)
            })
            .collect();
        device.flush();
        let mut total = 0.0f32;
        for out in outputs {
            let slice = out.as_slice().await.unwrap();
            total += slice[[0, 0, 0, 0]];
        }
        println!("checksum {total}");
        print_profiles(case, device);
    }
}

async fn bench_softmax(case: &str, device: &Device, b0: usize, b1: usize, m: usize, k: usize) {
    println!("=== softmax [{b0},{b1},{m},{k}] last dim ===");
    for _ in 0..ROUNDS {
        let outputs: Vec<Tensor<4, f32>> = (0..REPEATS)
            .map(|i| {
                let x = Tensor::from_slice(
                    device,
                    [b0, b1, m, k],
                    &fill(3 + i as u32, b0 * b1 * m * k),
                );
                x.softmax_last_dim()
            })
            .collect();
        device.flush();
        let mut total = 0.0f32;
        for out in outputs {
            let slice = out.as_slice().await.unwrap();
            total += slice[[0, 0, 0, 0]];
        }
        println!("checksum {total}");
        print_profiles(case, device);
    }
}

#[tokio::main]
async fn main() {
    if std::env::var_os("RUST_LOG").is_some() {
        tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .init();
    }
    let mut config = FusorConfig::from_env();
    config.trace_gpu_kernels = true;
    let device = Device::gpu_with_config(config).await.unwrap();
    let case = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());
    match case.as_str() {
        "wgrad" => bench_matmul("wgrad", &device, 64, 2048, 64).await,
        "wgrad256" => bench_matmul("wgrad256", &device, 64, 2048, 256).await,
        "wgrad256m" => bench_matmul("wgrad256m", &device, 256, 2048, 64).await,
        "fwd" => bench_matmul("fwd", &device, 2048, 64, 64).await,
        "fwd256" => bench_matmul("fwd256", &device, 2048, 256, 64).await,
        "fwdup" => bench_matmul("fwdup", &device, 2048, 64, 256).await,
        "attn" => bench_batched_matmul("attn", &device, 32, 4, 64, 64, 16).await,
        "softmax" => bench_softmax("softmax", &device, 32, 4, 64, 64).await,
        _ => {
            bench_matmul("wgrad", &device, 64, 2048, 64).await;
            bench_matmul("fwd", &device, 2048, 64, 64).await;
        }
    }
}
