//! Microbenchmark for individual GPU kernel shapes measured with
//! FUSOR_TRACE_GPU_KERNELS=1 — mirrors the transformer training profile
//! measurement mode. Runs batches of independent operations per resolve so
//! per-kernel GPU timestamps are printed by the resolver.
//!
//! Usage:
//! RUST_LOG=info FUSOR_TRACE_GPU_KERNELS=1 \
//!   cargo run --release -p fusor --example kernel_bench -- [case]

use fusor::{Device, Tensor};

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

async fn bench_matmul(device: &Device, m: usize, k: usize, n: usize) {
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
    }
}

async fn bench_batched_matmul(device: &Device, b0: usize, b1: usize, m: usize, k: usize, n: usize) {
    println!("=== batched matmul [{b0},{b1}] {m}x{k} by {k}x{n} ===");
    for _ in 0..ROUNDS {
        let outputs: Vec<Tensor<4, f32>> = (0..REPEATS)
            .map(|i| {
                let a =
                    Tensor::from_slice(device, [b0, b1, m, k], &fill(3 + i as u32, b0 * b1 * m * k));
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
    }
}

async fn bench_softmax(device: &Device, b0: usize, b1: usize, m: usize, k: usize) {
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
    }
}

#[tokio::main]
async fn main() {
    if std::env::var_os("RUST_LOG").is_some() {
        tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .init();
    }
    let device = Device::gpu().await.unwrap();
    let case = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());
    match case.as_str() {
        "wgrad" => bench_matmul(&device, 64, 2048, 64).await,
        "wgrad256" => bench_matmul(&device, 64, 2048, 256).await,
        "wgrad256m" => bench_matmul(&device, 256, 2048, 64).await,
        "fwd" => bench_matmul(&device, 2048, 64, 64).await,
        "fwd256" => bench_matmul(&device, 2048, 256, 64).await,
        "fwdup" => bench_matmul(&device, 2048, 64, 256).await,
        "attn" => bench_batched_matmul(&device, 32, 4, 64, 64, 16).await,
        "softmax" => bench_softmax(&device, 32, 4, 64, 64).await,
        _ => {
            bench_matmul(&device, 64, 2048, 64).await;
            bench_matmul(&device, 2048, 64, 64).await;
        }
    }
}
