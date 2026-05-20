#![allow(unused)]
use std::time::Duration;

use candle_core::backend::BackendDevice;
use candle_nn::Module;
use criterion::BatchSize;
use fusor_core::QMatrix;
use fusor_core::{Device, Tensor};
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

fn qmatmul(c: &mut Criterion) {
    use crate::Device;
    use crate::Tensor;
    use candle_core::Module;
    use fusor_gguf::GgufMetadata;

    let source = FileSource::HuggingFace {
        model_id: "unsloth/SmolLM2-135M-Instruct-GGUF".to_string(),
        revision: "main".to_string(),
        file: "SmolLM2-135M-Instruct-Q4_K_M.gguf".to_string(),
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
        for (width, name) in [
            (1536, "blk.3.ffn_down.weight"),
            (1536, "blk.0.ffn_down.weight"),
            (576, "token_embd.weight"),
            (576, "blk.0.ffn_up.weight"),
        ] {
            let random_data: Vec<Vec<f32>> = (0..size)
                .map(|_| (0..width).map(|_| rand::random()).collect())
                .collect();

            {
                let mut reader = std::io::Cursor::new(&bytes);
                let metadata = GgufMetadata::read(&mut reader).unwrap();
                let q_matrix_metadata = metadata.tensor_infos.get(name).unwrap();
                println!("Q matrix metadata: {q_matrix_metadata:?}");
                let quantization = q_matrix_metadata.ty;

                let group_name = format!("qmatmul-{width}-{quantization}");
                let mut group = c.benchmark_group(&group_name);
                group.sample_size(20);
                group.plot_config(
                    criterion::PlotConfiguration::default()
                        .summary_scale(criterion::AxisScale::Logarithmic),
                );

                let device = block_on(Device::new()).unwrap();

                let q_matrix = QMatrix::read(
                    &device,
                    q_matrix_metadata,
                    &mut reader,
                    metadata.tensor_data_offset,
                )
                .unwrap();
                let device = device.clone();
                let fusor_random_data = random_data.clone();
                group.bench_with_input(BenchmarkId::new("fusor-gpu", size), &size, move |b, &s| {
                    let device = device.clone();
                    let random_data = fusor_random_data.clone();
                    b.to_async(FuturesExecutor).iter_custom(async |iters| {
                        let tensor = Tensor::new(&device, &random_data);
                        tensor.materialize().await;
                        let mut sum = Duration::ZERO;
                        while sum.is_zero() {
                            for _ in 0..iters {
                                let start = std::time::Instant::now();
                                let new = tensor.q_mat_mul(&q_matrix);
                                new.materialize().await;
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
    let mut reader = std::io::Cursor::new(&bytes);
    let candle_metadata = candle_core::quantized::gguf_file::Content::read(&mut reader).unwrap();
    let candle_q_matrix_metadata = candle_metadata.tensor_infos.get(matrix_name).unwrap();
    let quantization = candle_q_matrix_metadata.ggml_dtype;
    let candle_q_tensor = candle_q_matrix_metadata
        .read(
            &mut reader,
            candle_metadata.tensor_data_offset,
            &candle_device,
        )
        .unwrap();
    let candle_q_matrix = candle_core::quantized::QMatMul::from_qtensor(candle_q_tensor).unwrap();
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
                (
                    candle_b.clone(),
                    candle_q_matrix.clone(),
                    candle_device.clone(),
                )
            },
            |(tensor_a, tensor_b, candle_device)| async move {
                tensor_b.forward(&tensor_a).unwrap();
                candle_device.synchronize().unwrap();
            },
            BatchSize::LargeInput,
        );
    });
}

criterion_group!(benches, qmatmul);
criterion_main!(benches);
