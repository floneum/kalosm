#![allow(unused)]
use std::sync::Arc;
use std::time::Duration;

use criterion::BatchSize;
use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};
use fusor_core::{Device, Tensor};
use futures::executor::block_on;

use criterion::async_executor::FuturesExecutor;

fn candle_gpu_device() -> Option<candle_core::Device> {
    candle_core::Device::new_cuda(0)
        .or_else(|_| candle_core::Device::new_metal(0))
        .ok()
}

const SIZES: [usize; 3] = [100, 1000, 4000];

fn bench_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("add");
    group.sample_size(20);
    group.plot_config(
        criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic),
    );

    {
        for size in SIZES {
            let device = block_on(Device::new()).unwrap();

            group.bench_with_input(BenchmarkId::new("fusor-gpu", size), &size, move |b, &s| {
                let device = device.clone();
                b.to_async(FuturesExecutor).iter_custom(async |iters| {
                    let mut sum = Duration::ZERO;
                    while sum.is_zero() {
                        for _ in 0..iters {
                            let tensor = Tensor::new(&device, &vec![vec![1.; size]; size]);
                            _ = tensor.as_slice::<2, f32>().await.unwrap();
                            let new = &tensor + &tensor;
                            let start = std::time::Instant::now();
                            new.materialize().await;
                            sum += start.elapsed();
                        }
                    }
                    sum
                })
            });
        }
    }

    {
        for size in SIZES {
            group.bench_with_input(BenchmarkId::new("ndarray", size), &size, move |b, &s| {
                b.to_async(FuturesExecutor).iter_batched(
                    || ndarray::Array2::<f32>::ones((s, s)),
                    |tensor| async move { &tensor + &tensor },
                    BatchSize::LargeInput,
                );
            });
        }
    }
    group.finish();
}

fn bench_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("mul");
    group.sample_size(20);
    group.plot_config(
        criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic),
    );

    {
        {
            for size in SIZES {
                let device = block_on(Device::new()).unwrap();

                group.bench_with_input(BenchmarkId::new("fusor-gpu", size), &size, move |b, &s| {
                    let device = device.clone();
                    b.to_async(FuturesExecutor).iter_custom(async |iters| {
                        let mut sum = Duration::ZERO;
                        while sum.is_zero() {
                            for _ in 0..iters {
                                let tensor1 = Tensor::new(&device, &vec![vec![1.; size]; size]);
                                let tensor2 = Tensor::new(&device, &vec![vec![1.; size]; size]);
                                _ = tensor2.as_slice::<2, f32>().await.unwrap();
                                let new = &tensor1 * &tensor2;
                                let start = std::time::Instant::now();
                                new.materialize().await;
                                sum += start.elapsed();
                            }
                        }
                        sum
                    })
                });
            }
        }
        {
            let device = block_on(Device::new()).unwrap();

            group.bench_with_input(
                BenchmarkId::new("fusor-gpu", "9x11x32x1"),
                &(),
                move |b, _| {
                    let device = device.clone();
                    b.to_async(FuturesExecutor).iter_custom(async |iters| {
                        let mut sum = Duration::ZERO;
                        while sum.is_zero() {
                            for _ in 0..iters {
                                let tensor1 =
                                    Tensor::new(&device, &vec![vec![vec![vec![1.]; 32]; 11]; 9]);
                                let tensor2 =
                                    Tensor::new(&device, &vec![vec![vec![vec![1.]; 32]; 11]; 9]);
                                _ = tensor1.materialize().await;
                                _ = tensor2.materialize().await;
                                let new = &tensor1 * &tensor2;
                                let start = std::time::Instant::now();
                                new.materialize().await;
                                sum += start.elapsed();
                            }
                        }
                        sum
                    })
                },
            );
        }
    }

    {
        {
            for size in SIZES {
                group.bench_with_input(BenchmarkId::new("ndarray", size), &size, move |b, &s| {
                    b.to_async(FuturesExecutor).iter_batched(
                        || ndarray::Array2::<f32>::ones((s, s)),
                        |tensor| async move { &tensor * &tensor },
                        BatchSize::LargeInput,
                    );
                });
            }
        }
        {
            let tensor = ndarray::Array4::<f32>::ones((9, 11, 32, 1));
            group.bench_with_input(
                BenchmarkId::new("ndarray", "9x11x32x1"),
                &(),
                move |b, _| {
                    b.to_async(FuturesExecutor).iter_batched(
                        || tensor.clone(),
                        |tensor| async move { &tensor * &tensor },
                        BatchSize::LargeInput,
                    );
                },
            );
        }
    }

    {
        let candle_device = candle_core::Device::Cpu;
        bench_candle_with_device(candle_device, "candle-cpu", &mut group);
    }

    if let Some(candle_device) = candle_gpu_device() {
        bench_candle_with_device(candle_device, "candle-gpu", &mut group);
    }
    group.finish();
}

fn bench_candle_with_device(
    candle_device: candle_core::Device,
    name: &str,
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
) {
    {
        for size in SIZES {
            let candle_device = candle_device.clone();
            group.bench_with_input(BenchmarkId::new(name, size), &size, move |b, &s| {
                b.to_async(FuturesExecutor).iter_batched(
                    {
                        let candle_device = candle_device.clone();
                        let random_data: Vec<Vec<f32>> = (0..size)
                            .map(|_| (0..size).map(|_| 1.).collect::<Vec<f32>>())
                            .collect();
                        move || {
                            candle_core::Tensor::from_iter(
                                random_data.iter().flat_map(|x| x.iter().copied()),
                                &candle_device,
                            )
                            .unwrap()
                            .reshape(&[size, size])
                            .unwrap()
                        }
                    },
                    {
                        let candle_device = candle_device.clone();
                        move |tensor| {
                            let candle_device = candle_device.clone();
                            async move {
                                let output = tensor.mul(&tensor).unwrap();
                                candle_device.synchronize().unwrap();
                                output
                            }
                        }
                    },
                    BatchSize::LargeInput,
                );
            });
        }
    }
    {
        let random_data: Vec<f32> = (0..9 * 11 * 32).map(|_| 1.).collect();
        group.bench_with_input(BenchmarkId::new(name, "9x11x32x1"), &(), move |b, _| {
            b.to_async(FuturesExecutor).iter_batched(
                || {
                    candle_core::Tensor::from_iter(random_data.iter().copied(), &candle_device)
                        .unwrap()
                        .reshape(&[9, 11, 32, 1])
                        .unwrap()
                },
                |tensor| {
                    let candle_device = candle_device.clone();
                    async move {
                        let output = tensor.mul(&tensor).unwrap();
                        candle_device.synchronize().unwrap();
                        output
                    }
                },
                BatchSize::LargeInput,
            );
        });
    }
}

criterion_group!(benches, bench_add, bench_mul);
criterion_main!(benches);
