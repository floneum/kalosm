use candle_core::{Device, Tensor as CandleTensor};
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use fusor_cpu::Tensor;

const SIZES: &[usize] = &[64, 256];

fn bench_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("add_f32");

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));

        // Fusor (reference-based API, no cloning)
        group.bench_with_input(BenchmarkId::new("fusor", size), &size, |b, &size| {
            let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
            let lhs = Tensor::from_slice([size], &data);
            let rhs = Tensor::from_slice([size], &data);
            b.iter(|| black_box((black_box(&lhs) + black_box(&rhs)).to_concrete()));
        });

        // Candle
        group.bench_with_input(BenchmarkId::new("candle", size), &size, |b, &size| {
            let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
            let lhs = CandleTensor::from_vec(data.clone(), size, &Device::Cpu).unwrap();
            let rhs = CandleTensor::from_vec(data, size, &Device::Cpu).unwrap();
            b.iter(|| black_box(black_box(&lhs).add(black_box(&rhs)).unwrap()));
        });
    }

    group.finish();
}

fn bench_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("creation_f32");

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));

        // Fusor zeros
        group.bench_with_input(BenchmarkId::new("fusor_zeros", size), &size, |b, &size| {
            b.iter(|| {
                black_box(Tensor::<1, fusor_cpu::ConcreteTensor<f32, 1>>::zeros([
                    size,
                ]))
            });
        });

        // Candle zeros
        group.bench_with_input(BenchmarkId::new("candle_zeros", size), &size, |b, &size| {
            b.iter(|| {
                black_box(CandleTensor::zeros(size, candle_core::DType::F32, &Device::Cpu).unwrap())
            });
        });

        // Fusor from_slice
        group.bench_with_input(
            BenchmarkId::new("fusor_from_slice", size),
            &size,
            |b, &size| {
                let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
                b.iter(|| black_box(Tensor::from_slice([size], &data)));
            },
        );

        // Candle from_vec
        group.bench_with_input(
            BenchmarkId::new("candle_from_vec", size),
            &size,
            |b, &size| {
                let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
                b.iter(|| {
                    black_box(CandleTensor::from_vec(data.clone(), size, &Device::Cpu).unwrap())
                });
            },
        );
    }

    group.finish();
}

fn bench_add_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("add_f32_2d");

    let matrix_sizes: &[(usize, usize)] = &[(32, 32), (64, 64), (128, 128), (256, 256), (512, 512)];

    for &(rows, cols) in matrix_sizes {
        let size = rows * cols;
        group.throughput(Throughput::Elements(size as u64));

        // Fusor (reference-based API, no cloning)
        group.bench_with_input(
            BenchmarkId::new("fusor", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |b, &(rows, cols)| {
                let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.1).collect();
                let lhs = Tensor::from_slice([rows, cols], &data);
                let rhs = Tensor::from_slice([rows, cols], &data);
                b.iter(|| black_box((black_box(&lhs) + black_box(&rhs)).to_concrete()));
            },
        );

        // Candle
        group.bench_with_input(
            BenchmarkId::new("candle", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |b, &(rows, cols)| {
                let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.1).collect();
                let lhs = CandleTensor::from_vec(data.clone(), (rows, cols), &Device::Cpu).unwrap();
                let rhs = CandleTensor::from_vec(data, (rows, cols), &Device::Cpu).unwrap();
                b.iter(|| black_box(black_box(&lhs).add(black_box(&rhs)).unwrap()));
            },
        );
    }

    group.finish();
}

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_f32");

    let sizes: &[(usize, usize, usize)] = &[
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
    ];

    for &(m, k, n) in sizes {
        let ops = (2 * m * n * k) as u64; // Each output element is dot product of k elements
        group.throughput(Throughput::Elements(ops));

        // Fusor matmul
        group.bench_with_input(
            BenchmarkId::new("fusor", format!("{}x{}x{}", m, k, n)),
            &(m, k, n),
            |b, &(m, k, n)| {
                let lhs_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
                let rhs_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();
                let lhs = Tensor::from_slice([m, k], &lhs_data);
                let rhs = Tensor::from_slice([k, n], &rhs_data);
                b.iter(|| black_box(lhs.clone().matmul(rhs.clone())));
            },
        );

        // Candle matmul
        group.bench_with_input(
            BenchmarkId::new("candle", format!("{}x{}x{}", m, k, n)),
            &(m, k, n),
            |b, &(m, k, n)| {
                let lhs_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
                let rhs_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();
                let lhs = CandleTensor::from_vec(lhs_data, (m, k), &Device::Cpu).unwrap();
                let rhs = CandleTensor::from_vec(rhs_data, (k, n), &Device::Cpu).unwrap();
                b.iter(|| black_box(black_box(&lhs).matmul(black_box(&rhs)).unwrap()));
            },
        );
    }

    group.finish();
}

fn bench_reduce_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum_f32");

    let sizes: &[usize] = &[1024, 4096];

    for &size in sizes {
        group.throughput(Throughput::Elements(size as u64));

        // Fusor sum
        group.bench_with_input(BenchmarkId::new("fusor", size), &size, |b, &size| {
            let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
            let tensor = Tensor::from_slice([size], &data);
            b.iter(|| black_box(tensor.clone().sum()));
        });

        // Candle sum
        group.bench_with_input(BenchmarkId::new("candle", size), &size, |b, &size| {
            let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
            let tensor = CandleTensor::from_vec(data, size, &Device::Cpu).unwrap();
            b.iter(|| black_box(black_box(&tensor).sum_all().unwrap()));
        });
    }

    group.finish();
}

fn bench_reduce_max(c: &mut Criterion) {
    let mut group = c.benchmark_group("max_f32");

    let sizes: &[usize] = &[1024, 4096];

    for &size in sizes {
        group.throughput(Throughput::Elements(size as u64));

        // Fusor max
        group.bench_with_input(BenchmarkId::new("fusor", size), &size, |b, &size| {
            let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
            let tensor = Tensor::from_slice([size], &data);
            b.iter(|| black_box(tensor.clone().max()));
        });

        // Candle max
        group.bench_with_input(BenchmarkId::new("candle", size), &size, |b, &size| {
            let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
            let tensor = CandleTensor::from_vec(data, size, &Device::Cpu).unwrap();
            b.iter(|| black_box(black_box(&tensor).max_all().unwrap()));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_add,
    bench_creation,
    bench_add_2d,
    bench_matmul,
    bench_reduce_sum,
    bench_reduce_max,
);
criterion_main!(benches);
