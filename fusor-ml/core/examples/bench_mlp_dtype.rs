//! f32 vs f16 throughput on the training step's dominant matmul shapes.

use fusor_core::{DataTypeEnum, Device, Tensor};
use std::time::Instant;

const SHAPES: [(usize, usize, usize); 4] = [
    (16384, 384, 1536),
    (16384, 1536, 384),
    (384, 16384, 1536),
    (16384, 384, 384),
];

fn bench(device: &Device, datatype: DataTypeEnum, m: usize, k: usize, n: usize) -> f64 {
    let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 61) as f32) * 0.01 - 0.3).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 53) as f32) * 0.01 - 0.26).collect();
    let (a, b) = match datatype {
        DataTypeEnum::F32 => (
            Tensor::from_slice(device, [m, k], &a_data),
            Tensor::from_slice(device, [k, n], &b_data),
        ),
        DataTypeEnum::F16 => {
            let half = |data: &[f32]| data.iter().map(|&x| half::f16::from_f32(x)).collect::<Vec<_>>();
            (
                Tensor::from_slice(device, [m, k], &half(&a_data)),
                Tensor::from_slice(device, [k, n], &half(&b_data)),
            )
        }
        _ => unreachable!(),
    };
    // Warm.
    for _ in 0..3 {
        a.mat_mul(&b).materialize_sync();
    }
    let iters = 10;
    let mut best = f64::MAX;
    for _ in 0..3 {
        let start = Instant::now();
        for _ in 0..iters {
            a.mat_mul(&b).materialize_sync();
        }
        best = best.min(start.elapsed().as_secs_f64() / iters as f64);
    }
    best
}

fn main() {
    pollster::block_on(async {
        let device = Device::new().await.expect("gpu device");
        if !device.f16_supported() {
            println!("f16 unsupported");
        }
        for (m, k, n) in SHAPES {
            let flops = 2.0 * m as f64 * k as f64 * n as f64;
            let f32_s = bench(&device, DataTypeEnum::F32, m, k, n);
            let f16_s = bench(&device, DataTypeEnum::F16, m, k, n);
            println!(
                "{m}x{k}x{n}: f32 {:.3} ms ({:.2} TF/s) | f16 {:.3} ms ({:.2} TF/s) | speedup {:.2}x",
                f32_s * 1e3,
                flops / f32_s / 1e12,
                f16_s * 1e3,
                flops / f16_s / 1e12,
                f32_s / f16_s
            );
        }
    });
}
