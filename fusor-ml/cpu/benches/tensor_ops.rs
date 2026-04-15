use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use fusor_cpu::{Abs, Add, ConcreteTensor, Div, Mul, Neg, Sqrt, Sub, TensorBacking};

const SIZES: &[usize] = &[64, 256];

fn create_f32_tensor_1d(size: usize) -> ConcreteTensor<f32, 1> {
    let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
    ConcreteTensor::from_slice([size], &data)
}

fn create_f64_tensor_1d(size: usize) -> ConcreteTensor<f64, 1> {
    let data: Vec<f64> = (0..size).map(|i| (i as f64) * 0.1).collect();
    ConcreteTensor::from_slice([size], &data)
}

fn create_f32_tensor_2d(rows: usize, cols: usize) -> ConcreteTensor<f32, 2> {
    let data: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.1).collect();
    ConcreteTensor::from_slice([rows, cols], &data)
}

fn create_i32_tensor_1d(size: usize) -> ConcreteTensor<i32, 1> {
    let data: Vec<i32> = (0..size).map(|i| i as i32).collect();
    ConcreteTensor::from_slice([size], &data)
}

fn bench_add_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("add_f32");

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("1d", size), &size, |b, &size| {
            let lhs = create_f32_tensor_1d(size);
            let rhs = create_f32_tensor_1d(size);
            b.iter(|| {
                let op = Add::new(black_box(lhs.clone()), black_box(rhs.clone()));
                black_box(op.to_concrete())
            });
        });
    }

    group.finish();
}

fn bench_add_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("add_f64");

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("1d", size), &size, |b, &size| {
            let lhs = create_f64_tensor_1d(size);
            let rhs = create_f64_tensor_1d(size);
            b.iter(|| {
                let op = Add::new(black_box(lhs.clone()), black_box(rhs.clone()));
                black_box(op.to_concrete())
            });
        });
    }

    group.finish();
}

fn bench_add_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("add_f32_2d");

    let matrix_sizes: &[(usize, usize)] = &[(8, 8), (16, 16), (32, 32), (64, 64), (128, 128)];

    for &(rows, cols) in matrix_sizes {
        let size = rows * cols;
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("matrix", format!("{}x{}", rows, cols)),
            &(rows, cols),
            |b, &(rows, cols)| {
                let lhs = create_f32_tensor_2d(rows, cols);
                let rhs = create_f32_tensor_2d(rows, cols);
                b.iter(|| {
                    let op = Add::new(black_box(lhs.clone()), black_box(rhs.clone()));
                    black_box(op.to_concrete())
                });
            },
        );
    }

    group.finish();
}

fn bench_sub_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("sub_f32");

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("1d", size), &size, |b, &size| {
            let lhs = create_f32_tensor_1d(size);
            let rhs = create_f32_tensor_1d(size);
            b.iter(|| {
                let op = Sub::new(black_box(lhs.clone()), black_box(rhs.clone()));
                black_box(op.to_concrete())
            });
        });
    }

    group.finish();
}

fn bench_mul_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("mul_f32");

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("1d", size), &size, |b, &size| {
            let lhs = create_f32_tensor_1d(size);
            let rhs = create_f32_tensor_1d(size);
            b.iter(|| {
                let op = Mul::new(black_box(lhs.clone()), black_box(rhs.clone()));
                black_box(op.to_concrete())
            });
        });
    }

    group.finish();
}

fn bench_mul_i32(c: &mut Criterion) {
    let mut group = c.benchmark_group("mul_i32");

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("1d", size), &size, |b, &size| {
            let lhs = create_i32_tensor_1d(size);
            let rhs = create_i32_tensor_1d(size);
            b.iter(|| {
                let op = Mul::new(black_box(lhs.clone()), black_box(rhs.clone()));
                black_box(op.to_concrete())
            });
        });
    }

    group.finish();
}

fn bench_div_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("div_f32");

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("1d", size), &size, |b, &size| {
            let lhs = create_f32_tensor_1d(size);
            // Avoid division by zero by starting from 1
            let data: Vec<f32> = (1..=size).map(|i| i as f32).collect();
            let rhs = ConcreteTensor::from_slice([size], &data);
            b.iter(|| {
                let op = Div::new(black_box(lhs.clone()), black_box(rhs.clone()));
                black_box(op.to_concrete())
            });
        });
    }

    group.finish();
}

fn bench_neg_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("neg_f32");

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("1d", size), &size, |b, &size| {
            let tensor = create_f32_tensor_1d(size);
            b.iter(|| {
                let op = Neg::new(black_box(tensor.clone()));
                black_box(op.to_concrete())
            });
        });
    }

    group.finish();
}

fn bench_abs_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("abs_f32");

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("1d", size), &size, |b, &size| {
            // Create tensor with negative values
            let data: Vec<f32> = (0..size)
                .map(|i| if i % 2 == 0 { i as f32 } else { -(i as f32) })
                .collect();
            let tensor = ConcreteTensor::from_slice([size], &data);
            b.iter(|| {
                let op = Abs::new(black_box(tensor.clone()));
                black_box(op.to_concrete())
            });
        });
    }

    group.finish();
}

fn bench_abs_i32(c: &mut Criterion) {
    let mut group = c.benchmark_group("abs_i32");

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("1d", size), &size, |b, &size| {
            let data: Vec<i32> = (0..size)
                .map(|i| if i % 2 == 0 { i as i32 } else { -(i as i32) })
                .collect();
            let tensor = ConcreteTensor::from_slice([size], &data);
            b.iter(|| {
                let op = Abs::new(black_box(tensor.clone()));
                black_box(op.to_concrete())
            });
        });
    }

    group.finish();
}

fn bench_sqrt_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("sqrt_f32");

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("1d", size), &size, |b, &size| {
            // Create tensor with positive values for sqrt
            let data: Vec<f32> = (1..=size).map(|i| (i * i) as f32).collect();
            let tensor = ConcreteTensor::from_slice([size], &data);
            b.iter(|| {
                let op = Sqrt::new(black_box(tensor.clone()));
                black_box(op.to_concrete())
            });
        });
    }

    group.finish();
}

fn bench_tensor_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_creation");

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("zeros_f32", size), &size, |b, &size| {
            b.iter(|| black_box(ConcreteTensor::<f32, 1>::zeros([size])));
        });

        group.bench_with_input(
            BenchmarkId::new("from_slice_f32", size),
            &size,
            |b, &size| {
                let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
                b.iter(|| black_box(ConcreteTensor::from_slice([size], &data)));
            },
        );
    }

    group.finish();
}

fn bench_chained_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("chained_ops");

    for &size in SIZES {
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("add_mul_f32", size), &size, |b, &size| {
            let a = create_f32_tensor_1d(size);
            let b_tensor = create_f32_tensor_1d(size);
            let c_tensor = create_f32_tensor_1d(size);
            b.iter(|| {
                // (a + b) * c
                let add_result =
                    Add::new(black_box(a.clone()), black_box(b_tensor.clone())).to_concrete();
                let op = Mul::new(black_box(add_result), black_box(c_tensor.clone()));
                black_box(op.to_concrete())
            });
        });

        group.bench_with_input(BenchmarkId::new("abs_sqrt_f32", size), &size, |b, &size| {
            let data: Vec<f32> = (1..=size)
                .map(|i| if i % 2 == 0 { -(i as f32) } else { i as f32 })
                .collect();
            let tensor = ConcreteTensor::from_slice([size], &data);
            b.iter(|| {
                // sqrt(abs(x))
                let abs_result = Abs::new(black_box(tensor.clone())).to_concrete();
                let op = Sqrt::new(black_box(abs_result));
                black_box(op.to_concrete())
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_add_f32,
    bench_add_f64,
    bench_add_2d,
    bench_sub_f32,
    bench_mul_f32,
    bench_mul_i32,
    bench_div_f32,
    bench_neg_f32,
    bench_abs_f32,
    bench_abs_i32,
    bench_sqrt_f32,
    bench_tensor_creation,
    bench_chained_ops,
);
criterion_main!(benches);
