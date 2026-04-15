//! Benchmarks comparing fused vs non-fused operations
//!
//! This benchmark demonstrates the performance benefit of lazy operation fusion.
//! Fused operations traverse memory once, while non-fused operations traverse
//! memory multiple times (once per operation).

use candle_core::{Device, Tensor as CandleTensor};
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use fusor_cpu::{ConcreteTensor, Tensor};

/// Benchmark fused operations (single memory pass) vs separate operations (multiple passes)
fn bench_fusion(c: &mut Criterion) {
    let sizes = [1024, 4096, 16384, 65536, 262144];

    let mut group = c.benchmark_group("fusion_mul_add_sqrt");

    for &size in &sizes {
        let x = Tensor::from_slice([size], &vec![2.0f32; size]);
        let y = Tensor::from_slice([size], &vec![3.0f32; size]);
        let z = Tensor::from_slice([size], &vec![1.0f32; size]);

        group.throughput(Throughput::Elements(size as u64));

        // Fused: (x * y + z).sqrt() - single memory traversal
        // The wrapper types implement Expr, and to_concrete() calls materialize_expr
        // which evaluates the entire expression tree in one SIMD loop
        group.bench_with_input(BenchmarkId::new("fused", size), &size, |b, _| {
            b.iter(|| {
                let x_ref = black_box(&x);
                let y_ref = black_box(&y);
                let z_ref = black_box(&z);

                // This creates: Sqrt<Add<Mul<&CT, &CT>, &CT>>
                // References avoid cloning - the expression tree holds refs to input data
                // When to_concrete() is called, the entire tree is evaluated in one pass
                let mul = x_ref * y_ref;
                let add = mul + z_ref;
                let sqrt = add.sqrt();
                let result = sqrt.to_concrete();
                black_box(result)
            })
        });

        // Non-fused: compute each step separately with intermediate allocations
        // This is what would happen without fusion - 3 memory traversals
        let x_tensor = Tensor::new(x.clone());
        let y_tensor = Tensor::new(y.clone());
        let z_tensor = Tensor::new(z.clone());
        group.bench_with_input(BenchmarkId::new("separate_ops", size), &size, |b, _| {
            b.iter(|| {
                let x_ref = black_box(&x_tensor);
                let y_ref = black_box(&y_tensor);
                let z_ref = black_box(&z_tensor);

                // Each eval() materializes a new tensor
                // This causes 3 separate memory traversals
                let mul_result = (x_ref * y_ref).to_concrete(); // Pass 1: read x,y, write temp1
                let add_result = (&mul_result + z_ref).to_concrete(); // Pass 2: read temp1,z, write temp2
                let sqrt_result = add_result.sqrt(); // Pass 3: read temp2, write result
                black_box(sqrt_result)
            })
        });

        // Candle comparison
        let candle_x = CandleTensor::new(vec![2.0f32; size].as_slice(), &Device::Cpu).unwrap();
        let candle_y = CandleTensor::new(vec![3.0f32; size].as_slice(), &Device::Cpu).unwrap();
        let candle_z = CandleTensor::new(vec![1.0f32; size].as_slice(), &Device::Cpu).unwrap();

        group.bench_with_input(BenchmarkId::new("candle", size), &size, |b, _| {
            b.iter(|| {
                let x_ref = black_box(&candle_x);
                let y_ref = black_box(&candle_y);
                let z_ref = black_box(&candle_z);

                let result = ((x_ref * y_ref).unwrap() + z_ref).unwrap().sqrt().unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

/// Benchmark longer chains to show more dramatic fusion benefits
fn bench_long_chain(c: &mut Criterion) {
    let sizes = [4096, 16384, 65536];

    let mut group = c.benchmark_group("fusion_long_chain");

    for &size in &sizes {
        let x = ConcreteTensor::<f32, 1>::from_slice([size], &vec![1.5f32; size]);
        let y = ConcreteTensor::<f32, 1>::from_slice([size], &vec![2.0f32; size]);

        group.throughput(Throughput::Elements(size as u64));

        // Fused: x * y + x * y + x - single traversal for 5 operations
        group.bench_with_input(BenchmarkId::new("fused_5ops", size), &size, |b, _| {
            b.iter(|| {
                let x_ref = black_box(&x);
                let y_ref = black_box(&y);

                // Build expression tree using references
                let mul1 = fusor_cpu::Mul::new(x_ref, y_ref);
                let mul2 = fusor_cpu::Mul::new(x_ref, y_ref);
                let add1 = fusor_cpu::Add::new(mul1, mul2);
                let add2 = fusor_cpu::Add::new(add1, x_ref);
                let result: ConcreteTensor<f32, 1> = fusor_cpu::TensorBacking::to_concrete(&add2);
                black_box(result)
            })
        });

        // Non-fused: 5 separate operations
        let x_tensor = Tensor::new(x.clone());
        let y_tensor = Tensor::new(y.clone());
        group.bench_with_input(BenchmarkId::new("separate_5ops", size), &size, |b, _| {
            b.iter(|| {
                let x_ref = black_box(&x_tensor);
                let y_ref = black_box(&y_tensor);

                let mul1 = (x_ref * y_ref).to_concrete(); // Pass 1
                let mul2 = (x_ref * y_ref).to_concrete(); // Pass 2
                let add1 = (&mul1 + &mul2).to_concrete(); // Pass 3
                let add2 = (&add1 + x_ref).to_concrete(); // Pass 4
                black_box(add2)
            })
        });

        // Candle comparison
        let candle_x = CandleTensor::new(vec![1.5f32; size].as_slice(), &Device::Cpu).unwrap();
        let candle_y = CandleTensor::new(vec![2.0f32; size].as_slice(), &Device::Cpu).unwrap();

        group.bench_with_input(BenchmarkId::new("candle_5ops", size), &size, |b, _| {
            b.iter(|| {
                let x_ref = black_box(&candle_x);
                let y_ref = black_box(&candle_y);

                // x * y + x * y + x
                let mul1 = (x_ref * y_ref).unwrap();
                let mul2 = (x_ref * y_ref).unwrap();
                let add1 = (&mul1 + &mul2).unwrap();
                let result = (&add1 + x_ref).unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

/// Benchmark unary chain (neg, abs, sqrt)
fn bench_unary_chain(c: &mut Criterion) {
    let sizes = [4096, 16384, 65536];

    let mut group = c.benchmark_group("fusion_unary_chain");

    for &size in &sizes {
        let x = ConcreteTensor::<f32, 1>::from_slice([size], &vec![-4.0f32; size]);

        group.throughput(Throughput::Elements(size as u64));

        // Fused: sqrt(abs(neg(x))) - single traversal
        group.bench_with_input(BenchmarkId::new("fused", size), &size, |b, _| {
            b.iter(|| {
                let x_ref = black_box(&x);

                let neg = fusor_cpu::Neg::new(x_ref);
                let abs = fusor_cpu::Abs::new(neg);
                let sqrt = fusor_cpu::Sqrt::new(abs);
                let result: ConcreteTensor<f32, 1> = fusor_cpu::TensorBacking::to_concrete(&sqrt);
                black_box(result)
            })
        });

        // Non-fused: 3 separate operations
        let x_tensor = Tensor::new(x.clone());
        group.bench_with_input(BenchmarkId::new("separate", size), &size, |b, _| {
            b.iter(|| {
                let x_ref = black_box(&x_tensor);

                let neg_result = (-x_ref).to_concrete(); // Pass 1
                let abs_result = neg_result.abs().to_concrete(); // Pass 2
                let sqrt_result = abs_result.sqrt().to_concrete(); // Pass 3
                black_box(sqrt_result)
            })
        });

        // Candle comparison
        let candle_x = CandleTensor::new(vec![-4.0f32; size].as_slice(), &Device::Cpu).unwrap();

        group.bench_with_input(BenchmarkId::new("candle", size), &size, |b, _| {
            b.iter(|| {
                let x_ref = black_box(&candle_x);

                // sqrt(abs(neg(x)))
                let result = x_ref.neg().unwrap().abs().unwrap().sqrt().unwrap();
                black_box(result)
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_fusion, bench_long_chain, bench_unary_chain);
criterion_main!(benches);
