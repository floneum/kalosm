#![allow(unused)]
use std::sync::Arc;
use std::time::Duration;

use candle_core::backend::BackendDevice;
use criterion::BatchSize;
use fusor_core::{Device, Tensor};
use futures::executor::block_on;
use ndarray::Axis;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};

use criterion::async_executor::FuturesExecutor;

fn candle_gpu_device() -> Option<candle_core::Device> {
    candle_core::Device::new_cuda(0)
        .or_else(|_| candle_core::Device::new_metal(0))
        .ok()
}

const SIZES: [usize; 3] = [100, 1000, 4000];

fn bench_sum_reduce(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum");
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
                            let new = tensor.sum(0);
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
                    |tensor| async move { tensor.sum_axis(Axis(0)) },
                    BatchSize::LargeInput,
                );
            });
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
                            let output = tensor.sum(0).unwrap();
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

criterion_group!(benches, bench_sum_reduce);
criterion_main!(benches);
