#![allow(unused)]
use std::time::Duration;

use candle_core::backend::BackendDevice;
use candle_nn::Module;
use criterion::BatchSize;
use fusor::layers::Linear;
use fusor::{Device, Tensor, VarBuilder};
use futures::executor::block_on;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};

use criterion::async_executor::FuturesExecutor;
use kalosm_common::Cache;
use kalosm_model_types::FileSource;

fn candle_gpu_device() -> Option<candle_core::Device> {
    candle_core::Device::new_cuda(0)
        .or_else(|_| candle_core::Device::new_metal(0))
        .ok()
}

fn linear(c: &mut Criterion) {
    use candle_core::Module;
    use fusor_gguf::GgufMetadata;

    let source = FileSource::HuggingFace {
        model_id: "CompendiumLabs/bge-large-en-v1.5-gguf".to_string(),
        revision: "main".to_string(),
        file: "bge-large-en-v1.5-q4_k_m.gguf".to_string(),
    };
    let bytes = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async move {
            let cache = Cache::default();
            let path = cache.get(&source, |_| {}).await.unwrap();
            tokio::fs::read(&path).await.unwrap()
        });

    for size in [1, 512] {
        for (width, name) in [(1024, "blk.0.attn_q"), (4096, "blk.0.ffn_down")] {
            let random_data: Vec<Vec<f32>> = (0..size)
                .map(|_| (0..width).map(|_| rand::random()).collect())
                .collect();

            {
                let mut reader = std::io::Cursor::new(&bytes);
                let mut var_builder = VarBuilder::from_gguf(&mut reader).unwrap();
                let device = block_on(async { Device::new().await.unwrap() });
                let linear = Linear::load(&device, &mut var_builder.pp(name)).unwrap();
                let quantization = linear.quantization();

                let group_name = format!("linear-{width}-{quantization}");
                let mut group = c.benchmark_group(&group_name);
                group.sample_size(20);
                group.plot_config(
                    criterion::PlotConfiguration::default()
                        .summary_scale(criterion::AxisScale::Logarithmic),
                );

                let device = device.clone();
                let fusor_random_data = random_data.clone();
                group.bench_with_input(BenchmarkId::new("fusor-gpu", size), &size, move |b, &s| {
                    let device = device.clone();
                    let random_data = fusor_random_data.clone();
                    b.to_async(FuturesExecutor).iter_custom(async |iters| {
                        let flat_data: Vec<f32> =
                            random_data.iter().flat_map(|r| r.iter().copied()).collect();
                        let tensor: Tensor<2, f32> =
                            Tensor::from_slice(&device, [size, width], &flat_data);
                        tensor.as_gpu().unwrap().materialize().await;
                        let mut sum = Duration::ZERO;
                        while sum.is_zero() {
                            for _ in 0..iters {
                                let start = std::time::Instant::now();
                                let new = linear.forward(&tensor.unsqueeze::<3>(0));
                                new.as_gpu().unwrap().materialize().await;
                                sum += start.elapsed();
                            }
                        }
                        sum
                    });
                });

                if let Some(candle_device) = candle_gpu_device() {
                    bench_candle_with_device(
                        &bytes,
                        size,
                        random_data.clone(),
                        candle_device,
                        "candle-gpu",
                        name,
                        width,
                        &mut group,
                    );
                }

                {
                    let candle_device = candle_core::Device::Cpu;
                    bench_candle_with_device(
                        &bytes,
                        size,
                        random_data.clone(),
                        candle_device,
                        "candle-cpu",
                        name,
                        width,
                        &mut group,
                    );
                }
                group.finish();
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn bench_candle_with_device(
    bytes: &[u8],
    size: usize,
    random_data: Vec<Vec<f32>>,
    candle_device: candle_core::Device,
    name: &str,
    matrix_name: &str,
    width: usize,
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
) {
    use candle_transformers::quantized_nn::{Linear, linear};
    let var_builder = candle_transformers::quantized_var_builder::VarBuilder::from_gguf_buffer(
        bytes,
        &candle_device,
    )
    .unwrap()
    .pp(matrix_name);
    let weight = var_builder.get_no_shape("weight").unwrap();
    let bias = var_builder
        .get_no_shape("bias")
        .unwrap()
        .dequantize(var_builder.device())
        .unwrap();
    let quantization = weight.dtype();
    let linear = Linear::from_arc(weight, Some(bias)).unwrap();
    group.bench_with_input(BenchmarkId::new(name, size), &size, move |b, &s| {
        b.to_async(FuturesExecutor).iter_batched(
            || {
                let candle_b = candle_core::Tensor::from_iter(
                    random_data.iter().flat_map(|x| x.iter().copied()),
                    &candle_device,
                )
                .unwrap()
                .reshape(&[size, width])
                .unwrap();
                candle_device.synchronize().unwrap();
                (candle_b.clone(), linear.clone(), candle_device.clone())
            },
            |(tensor_a, linear, candle_device)| async move {
                linear.forward(&tensor_a).unwrap();
                candle_device.synchronize().unwrap();
            },
            BatchSize::LargeInput,
        );
    });
}

criterion_group!(benches, linear);
criterion_main!(benches);
