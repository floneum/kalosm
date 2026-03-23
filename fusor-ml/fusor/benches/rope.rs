#![allow(unused)]
use std::time::Duration;

use candle_core::backend::BackendDevice;
use criterion::BatchSize;
use fusor::{Device, Tensor};
use futures::executor::block_on;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};

use criterion::async_executor::FuturesExecutor;

// Sizes: [batch_size, num_heads, seq_len, head_dim]
const SIZES: [[usize; 4]; 8] = [
    [1, 32, 128, 64],  // Small sequence
    [1, 32, 512, 64],  // Medium sequence
    [1, 32, 1024, 64], // Large sequence
    [1, 32, 2048, 64], // Very large sequence
    [2, 32, 512, 64],  // Batch of 2
    [4, 32, 512, 64],  // Batch of 4
    [1, 32, 128, 128], // Larger head dimension
    [1, 8, 1024, 128], // Fewer heads, larger dim
];

async fn setup_fusor_tensors(
    device: &Device,
    batch: usize,
    heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> (Tensor<4, f32>, Tensor<2, f32>, Tensor<2, f32>) {
    let pos_shape = [seq_len * 2, head_dim / 2];
    let cos_flat: Vec<f32> = (0..pos_shape[0])
        .flat_map(|i| {
            (0..pos_shape[1]).map(move |j| {
                ((i as f32) / 10000f32.powf((2 * (j / 2)) as f32 / head_dim as f32)).cos()
            })
        })
        .collect();
    let sin_flat: Vec<f32> = (0..pos_shape[0])
        .flat_map(|i| {
            (0..pos_shape[1]).map(move |j| {
                ((i as f32) / 10000f32.powf((2 * (j / 2)) as f32 / head_dim as f32)).sin()
            })
        })
        .collect();

    let cos: Tensor<2, f32> = Tensor::from_slice(device, [pos_shape[0], pos_shape[1]], &cos_flat);
    let sin: Tensor<2, f32> = Tensor::from_slice(device, [pos_shape[0], pos_shape[1]], &sin_flat);

    let numel = batch * heads * seq_len * head_dim;
    let input_flat: Vec<f32> = (0..numel).map(|_| 1.0f32).collect();
    let input: Tensor<4, f32> =
        Tensor::from_slice(device, [batch, heads, seq_len, head_dim], &input_flat);

    // Ensure materialization
    _ = input.as_slice().await.unwrap();
    _ = cos.as_slice().await.unwrap();
    _ = sin.as_slice().await.unwrap();

    (input, cos, sin)
}

fn rope_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("rope-fusor");

    for [batch, heads, seq_len, head_dim] in SIZES {
        // Composite interleaved (CPU or GPU composite path)
        let device = block_on(async { Device::new().await.unwrap() });
        let device_ref = device.clone();
        group.bench_with_input(
            BenchmarkId::new(
                "rope_interleaved",
                format!("{}x{}x{}x{}", batch, heads, seq_len, head_dim),
            ),
            &(batch, heads, seq_len, head_dim),
            move |b, &(batch, heads, seq_len, head_dim)| {
                let device = device_ref.clone();
                b.to_async(FuturesExecutor).iter_custom(|iters| {
                    let device = device.clone();
                    async move {
                        let (input, cos, sin) =
                            setup_fusor_tensors(&device, batch, heads, seq_len, head_dim).await;

                        let mut sum = Duration::ZERO;
                        while sum.is_zero() {
                            for _ in 0..iters {
                                let start = std::time::Instant::now();
                                let result = input.rope_interleaved(&cos, &sin);
                                result.materialize().await;
                                sum += start.elapsed();
                            }
                        }
                        sum
                    }
                });
            },
        );

        // Fused
        let device = block_on(async { Device::new().await.unwrap() });
        let device_ref = device.clone();
        group.bench_with_input(
            BenchmarkId::new(
                "rope_fused",
                format!("{}x{}x{}x{}", batch, heads, seq_len, head_dim),
            ),
            &(batch, heads, seq_len, head_dim),
            move |b, &(batch, heads, seq_len, head_dim)| {
                let device = device_ref.clone();
                b.to_async(FuturesExecutor).iter_custom(|iters| {
                    let device = device.clone();
                    async move {
                        let (input, cos, sin) =
                            setup_fusor_tensors(&device, batch, heads, seq_len, head_dim).await;

                        let mut sum = Duration::ZERO;
                        while sum.is_zero() {
                            for _ in 0..iters {
                                let start = std::time::Instant::now();
                                let result = input.rope_fused(&cos, &sin);
                                result.materialize().await;
                                sum += start.elapsed();
                            }
                        }
                        sum
                    }
                });
            },
        );
    }
    group.finish();

    // Benchmark candle's rope_i on Metal (macOS only)
    #[cfg(target_os = "macos")]
    {
        let candle_device = candle_core::Device::Metal(candle_core::MetalDevice::new(0).unwrap());
        bench_candle_rope(candle_device, "rope-candle-metal", c);
    }

    // Benchmark candle's rope_i on CPU
    {
        let candle_device = candle_core::Device::Cpu;
        bench_candle_rope(candle_device, "rope-candle-cpu", c);
    }
}

fn bench_candle_rope(candle_device: candle_core::Device, name: &str, c: &mut Criterion) {
    let mut group = c.benchmark_group(name);
    let group = group.sample_size(20);

    for [batch, heads, seq_len, head_dim] in SIZES {
        let candle_device = candle_device.clone();
        group.bench_with_input(
            BenchmarkId::new(
                "rope_i",
                format!("{}x{}x{}x{}", batch, heads, seq_len, head_dim),
            ),
            &(batch, heads, seq_len, head_dim),
            move |b, &(batch, heads, seq_len, head_dim)| {
                b.to_async(FuturesExecutor).iter_batched(
                    {
                        let candle_device = candle_device.clone();
                        move || {
                            let pos_shape = [seq_len * 2, head_dim / 2];
                            let cos_data = (0..pos_shape[0])
                                .map(|i| {
                                    (0..pos_shape[1])
                                        .map(|j| {
                                            ((i as f32)
                                                / 10000f32
                                                    .powf((2 * (j / 2)) as f32 / head_dim as f32))
                                            .cos()
                                        })
                                        .collect::<Vec<_>>()
                                })
                                .collect::<Vec<_>>();
                            let sin_data = (0..pos_shape[0])
                                .map(|i| {
                                    (0..pos_shape[1])
                                        .map(|j| {
                                            ((i as f32)
                                                / 10000f32
                                                    .powf((2 * (j / 2)) as f32 / head_dim as f32))
                                            .sin()
                                        })
                                        .collect::<Vec<_>>()
                                })
                                .collect::<Vec<_>>();

                            let cos = candle_core::Tensor::new(cos_data, &candle_device).unwrap();
                            let sin = candle_core::Tensor::new(sin_data, &candle_device).unwrap();

                            let input_data: Vec<Vec<Vec<Vec<f32>>>> = (0..batch)
                                .map(|_| {
                                    (0..heads)
                                        .map(|_| {
                                            (0..seq_len)
                                                .map(|_| (0..head_dim).map(|_| 1.0f32).collect())
                                                .collect()
                                        })
                                        .collect()
                                })
                                .collect();
                            let input =
                                candle_core::Tensor::new(input_data, &candle_device).unwrap();

                            (input, cos, sin)
                        }
                    },
                    {
                        let candle_device = candle_device.clone();
                        move |(input, cos, sin)| {
                            let candle_device = candle_device.clone();
                            async move {
                                let output =
                                    candle_nn::rotary_emb::rope_i(&input, &cos, &sin).unwrap();
                                candle_device.synchronize().unwrap();
                                output
                            }
                        }
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }
}

criterion_group!(benches, rope_benchmark);
criterion_main!(benches);
