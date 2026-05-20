#![allow(unused)]
use std::sync::Arc;
use std::time::Duration;

use criterion::BatchSize;
use fusor_core::{Device, Tensor};
use futures::executor::block_on;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};

use criterion::async_executor::FuturesExecutor;

const SIZES: [usize; 3] = [100, 1000, 4000];

fn fused(c: &mut Criterion) {
    let mut group = c.benchmark_group("add-const");
    group.sample_size(20);
    group.plot_config(
        criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic),
    );

    {
        for size in SIZES {
            let device = block_on(Device::new()).unwrap();
            let tensor = Tensor::new(&device, &vec![vec![1.; size]; size]);
            block_on(tensor.as_slice()).unwrap();

            group.bench_with_input(
                BenchmarkId::new("fusor-gpu-fused", size),
                &size,
                move |b, &s| {
                    let device = device.clone();
                    b.to_async(FuturesExecutor).iter_custom(async |iters| {
                        let mut sum = Duration::ZERO;
                        while sum.is_zero() {
                            for _ in 0..iters {
                                let tensor = Tensor::new(&device, &vec![vec![1.; size]; size]);
                                _ = tensor.as_slice().await.unwrap();
                                let new = (tensor + 1.) + 1.;
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
            let tensor = Tensor::new(&device, &vec![vec![1.; size]; size]);
            block_on(tensor.as_slice()).unwrap();

            group.bench_with_input(
                BenchmarkId::new("fusor-gpu-separate", size),
                &size,
                move |b, &s| {
                    let device = device.clone();
                    b.to_async(FuturesExecutor).iter_custom(async |iters| {
                        let mut sum = Duration::ZERO;
                        while sum.is_zero() {
                            for _ in 0..iters {
                                for _ in 0..2 {
                                    let tensor = Tensor::new(&device, &vec![vec![1.; size]; size]);
                                    _ = tensor.as_slice().await.unwrap();
                                    let new = tensor + 1.;
                                    let start = std::time::Instant::now();
                                    new.materialize().await;
                                    sum += start.elapsed();
                                }
                            }
                        }
                        sum
                    })
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, fused);
criterion_main!(benches);
