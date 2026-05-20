#![allow(unused)]
use std::time::Duration;

use candle_core::backend::BackendDevice;
use criterion::BatchSize;
use fusor_core::{Device, Tensor};
use futures::executor::block_on;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};

use criterion::async_executor::FuturesExecutor;

fn candle_gpu_device() -> Option<candle_core::Device> {
    candle_core::Device::new_cuda(0)
        .or_else(|_| candle_core::Device::new_metal(0))
        .ok()
}

const SIZES: [[usize; 3]; 12] = [
    [1, 1500, 64],
    [64, 1500, 1],
    [1, 2048, 64],
    [64, 2048, 1],
    [128, 128, 128],
    [1, 128, 1],
    [1024, 1024, 1024],
    [1, 1024, 1],
    [2048, 2048, 2048],
    [1, 2048, 1],
    [4096, 4096, 4096],
    [1, 4096, 1],
];

fn matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    group.sample_size(20);
    group.plot_config(
        criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic),
    );
    {
        let device = block_on(Device::new()).unwrap();

        for [m, k, n] in SIZES {
            let device = device.clone();
            group.bench_with_input(
                BenchmarkId::new("fusor-gpu", format!("{m}x{k} by {k}x{n}")),
                &(m, k, n),
                move |b, &(m, k, n)| {
                    let device = device.clone();
                    b.to_async(FuturesExecutor).iter_custom(async |iters| {
                        let mut sum = Duration::ZERO;
                        while sum.is_zero() {
                            for _ in 0..iters {
                                let tensor1 = Tensor::new(&device, &vec![vec![1.; k]; m]);
                                let tensor2 = Tensor::new(&device, &vec![vec![1.; n]; k]);
                                _ = tensor1.as_slice().await.unwrap();
                                _ = tensor2.as_slice().await.unwrap();
                                let new = tensor1.mat_mul(&tensor2);
                                let start = std::time::Instant::now();
                                new.materialize().await;
                                sum += start.elapsed();
                            }
                        }
                        sum
                    });
                },
            );
        }
    }

    if let Some(candle_device) = candle_gpu_device() {
        bench_candle_with_device(candle_device, "candle-gpu", &mut group);
    }

    {
        let candle_device = candle_core::Device::Cpu;
        bench_candle_with_device(candle_device, "candle-cpu", &mut group);
    }

    {
        for [m, k, n] in SIZES {
            group.bench_with_input(
                BenchmarkId::new("ndarray", format!("{m}x{k} by {k}x{n}")),
                &(m, k, n),
                move |b, &(m, k, n)| {
                    b.to_async(FuturesExecutor).iter_batched(
                        || {
                            let matrix1 = ndarray::Array2::<f32>::ones((m, k));
                            let matrix2 = ndarray::Array2::<f32>::ones((k, n));
                            (matrix1, matrix2)
                        },
                        |(tensor_a, tensor_b)| async move { tensor_a.dot(&tensor_b) },
                        BatchSize::LargeInput,
                    );
                },
            );
        }
    }
    group.finish();
}

fn bench_candle_with_device(
    candle_device: candle_core::Device,
    name: &str,
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
) {
    // Cap the comparison footprint so the macOS GitHub runner doesn't
    // SIGKILL the bench. Each iteration holds two host vectors, two GPU
    // tensors, and the output, so a 4096x4096 square alone is ~256 MB —
    // criterion's warmup + sample loop ramps that to several GB and the
    // 7 GB runner OOMs.
    const CANDLE_MAX_FOOTPRINT_ELEMENTS: usize = 1024 * 1024 * 4; // 16 MB f32 per tensor
    for [m, k, n] in SIZES {
        if m * k > CANDLE_MAX_FOOTPRINT_ELEMENTS
            || k * n > CANDLE_MAX_FOOTPRINT_ELEMENTS
            || m * n > CANDLE_MAX_FOOTPRINT_ELEMENTS
        {
            continue;
        }
        let candle_device = candle_device.clone();
        group.bench_with_input(
            BenchmarkId::new(name, format!("{m}x{k} by {k}x{n}")),
            &(m, k, n),
            move |b, &(m, k, n)| {
                b.to_async(FuturesExecutor).iter_batched(
                    {
                        let candle_device = candle_device.clone();
                        let random_data_1: Vec<Vec<f32>> = (0..m)
                            .map(|_| (0..k).map(|_| 1.).collect::<Vec<f32>>())
                            .collect();
                        let random_data_2: Vec<Vec<f32>> = (0..k)
                            .map(|_| (0..n).map(|_| 1.).collect::<Vec<f32>>())
                            .collect();
                        move || {
                            (
                                candle_core::Tensor::from_iter(
                                    random_data_1.iter().flat_map(|x| x.iter().copied()),
                                    &candle_device,
                                )
                                .unwrap()
                                .reshape(&[m, k])
                                .unwrap(),
                                candle_core::Tensor::from_iter(
                                    random_data_2.iter().flat_map(|x| x.iter().copied()),
                                    &candle_device,
                                )
                                .unwrap()
                                .reshape(&[k, n])
                                .unwrap(),
                            )
                        }
                    },
                    {
                        let candle_device = candle_device.clone();
                        move |(tensor1, tensor2)| {
                            let candle_device = candle_device.clone();
                            async move {
                                let output = tensor1.matmul(&tensor2).unwrap();
                                candle_device.synchronize().unwrap();
                                output
                            }
                        }
                    },
                    BatchSize::PerIteration,
                );
            },
        );
    }
}

criterion_group!(benches, matmul);
criterion_main!(benches);
