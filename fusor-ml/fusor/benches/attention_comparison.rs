use std::{hint::black_box, time::Duration};

use criterion::async_executor::FuturesExecutor;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use fusor::{Device, Tensor, TensorBacking};
use pollster::block_on;

const SIZES: [[usize; 4]; 8] = [
    [1, 8, 128, 64],
    [1, 8, 256, 64],
    [1, 8, 512, 64],
    [1, 8, 1024, 64],
    [2, 8, 128, 64],
    [4, 8, 128, 64],
    [1, 32, 128, 64],
    [1, 8, 128, 128],
];

fn make_input(numel: usize, freq: f32, phase: f32) -> Vec<f32> {
    (0..numel)
        .map(|i| {
            let x = i as f32;
            ((x * freq + phase).sin() + (x * freq * 0.37 + phase).cos()) * 0.5
        })
        .collect()
}

fn attention_inputs(shape: [usize; 4]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let numel = shape.iter().product();
    (
        make_input(numel, 0.013, 0.1),
        make_input(numel, 0.017, 0.7),
        make_input(numel, 0.019, 1.3),
    )
}

async fn setup_tensors(
    device: &Device,
    shape: [usize; 4],
    q_data: &[f32],
    k_data: &[f32],
    v_data: &[f32],
) -> (Tensor<4, f32>, Tensor<4, f32>, Tensor<4, f32>) {
    let q = Tensor::from_slice(device, shape, q_data);
    let k = Tensor::from_slice(device, shape, k_data);
    let v = Tensor::from_slice(device, shape, v_data);

    resolve_tensor(&q).await;
    resolve_tensor(&k).await;
    resolve_tensor(&v).await;

    (q, k, v)
}

async fn resolve_tensor<const R: usize, B>(tensor: &Tensor<R, f32, B>)
where
    B: TensorBacking<R, Elem = f32>,
{
    if let Some(gpu) = tensor.as_gpu() {
        gpu.materialize().await;
    } else {
        black_box(tensor.to_concrete());
    }
}

fn bench_backend(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    backend: &'static str,
    device: &Device,
    shape: [usize; 4],
    size_str: &str,
    q_data: &[f32],
    k_data: &[f32],
    v_data: &[f32],
) {
    let [_, _, _, head_dim] = shape;

    let device_standard = device.clone();
    let q_data_standard = q_data.to_vec();
    let k_data_standard = k_data.to_vec();
    let v_data_standard = v_data.to_vec();
    group.bench_with_input(
        BenchmarkId::new(format!("{backend}_standard"), size_str),
        &shape,
        move |b, &shape| {
            let device = device_standard.clone();
            let q_data = q_data_standard.clone();
            let k_data = k_data_standard.clone();
            let v_data = v_data_standard.clone();

            b.to_async(FuturesExecutor).iter_custom(move |iters| {
                let device = device.clone();
                let q_data = q_data.clone();
                let k_data = k_data.clone();
                let v_data = v_data.clone();
                async move {
                    let (q, k, v) = setup_tensors(&device, shape, &q_data, &k_data, &v_data).await;
                    let scale = 1.0 / (head_dim as f32).sqrt();

                    let mut total = Duration::ZERO;
                    while total.is_zero() {
                        for _ in 0..iters {
                            let start = std::time::Instant::now();
                            let scores = q.mat_mul(&k.t()).mul_scalar(scale);
                            let probs = scores.softmax_last_dim::<3>();
                            let output = probs.mat_mul(&v);
                            resolve_tensor(&output).await;
                            total += start.elapsed();
                        }
                    }
                    total
                }
            });
        },
    );

    let device_flash = device.clone();
    let q_data_flash = q_data.to_vec();
    let k_data_flash = k_data.to_vec();
    let v_data_flash = v_data.to_vec();
    group.bench_with_input(
        BenchmarkId::new(format!("{backend}_flash"), size_str),
        &shape,
        move |b, &shape| {
            let device = device_flash.clone();
            let q_data = q_data_flash.clone();
            let k_data = k_data_flash.clone();
            let v_data = v_data_flash.clone();

            b.to_async(FuturesExecutor).iter_custom(move |iters| {
                let device = device.clone();
                let q_data = q_data.clone();
                let k_data = k_data.clone();
                let v_data = v_data.clone();
                async move {
                    let (q, k, v) = setup_tensors(&device, shape, &q_data, &k_data, &v_data).await;
                    let scale = 1.0 / (head_dim as f32).sqrt();

                    let mut total = Duration::ZERO;
                    while total.is_zero() {
                        for _ in 0..iters {
                            let start = std::time::Instant::now();
                            let output = q.flash_attention(&k, &v, scale, None);
                            resolve_tensor(&output).await;
                            total += start.elapsed();
                        }
                    }
                    total
                }
            });
        },
    );
}

fn bench_attention_comparison(c: &mut Criterion) {
    let cpu_device = Device::cpu();
    let gpu_device = block_on(Device::gpu()).ok();

    let mut group = c.benchmark_group("attention_comparison");
    group.sample_size(10);

    for shape in SIZES {
        let size_str = format!("{}x{}x{}x{}", shape[0], shape[1], shape[2], shape[3]);
        let (q_data, k_data, v_data) = attention_inputs(shape);

        bench_backend(
            &mut group,
            "cpu",
            &cpu_device,
            shape,
            &size_str,
            &q_data,
            &k_data,
            &v_data,
        );

        if let Some(device) = gpu_device.as_ref() {
            bench_backend(
                &mut group, "gpu", device, shape, &size_str, &q_data, &k_data, &v_data,
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_attention_comparison);
criterion_main!(benches);
