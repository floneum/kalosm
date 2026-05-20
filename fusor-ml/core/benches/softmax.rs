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

fn softmax_slow(tensor: &Tensor<2, f32>) -> Tensor<2, f32> {
    let max = tensor.max_keepdim::<1>(1);
    let shifted = tensor - &max;
    let exp = shifted.exp();
    let sum = exp.sum_keepdim::<1>(1);
    &exp / &sum
}

fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax");
    group.sample_size(20);
    group.plot_config(
        criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic),
    );

    {
        for size in SIZES {
            let device = block_on(Device::new()).unwrap();

            let random_data: Vec<Vec<f32>> = (0..size)
                .map(|_| (0..size).map(|_| rand::random()).collect())
                .collect();
            group.bench_with_input(
                BenchmarkId::new("fusor-gpu-slow", size),
                &size,
                move |b, &s| {
                    let device = device.clone();
                    b.to_async(FuturesExecutor).iter_custom(async |iters| {
                        let mut sum = Duration::ZERO;
                        while sum.is_zero() {
                            for _ in 0..iters {
                                let tensor = Tensor::new(&device, &random_data);
                                _ = tensor.as_slice().await.unwrap();
                                let new = softmax_slow(&tensor);
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
        for size in SIZES {
            let device = block_on(Device::new()).unwrap();

            let random_data: Vec<Vec<f32>> = (0..size)
                .map(|_| (0..size).map(|_| rand::random()).collect())
                .collect();
            group.bench_with_input(BenchmarkId::new("fusor-gpu", size), &size, move |b, &s| {
                let device = device.clone();
                b.to_async(FuturesExecutor).iter_custom(async |iters| {
                    let mut sum = Duration::ZERO;
                    while sum.is_zero() {
                        for _ in 0..iters {
                            let tensor = Tensor::new(&device, &random_data);
                            _ = tensor.as_slice().await.unwrap();
                            let new = tensor.softmax_last_dim();
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
                        .map(|_| (0..size).map(|_| rand::random()).collect())
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
                            let output = candle_nn::ops::softmax_last_dim(&tensor);
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

criterion_group!(benches, bench_softmax);
criterion_main!(benches);
