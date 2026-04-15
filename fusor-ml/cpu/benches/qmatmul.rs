use aligned_vec::AVec;
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use fusor_cpu::{BlockQ8_0, ConcreteTensor, QuantizedTensor, Tensor};
use half::f16;

/// Helper to create a Q8_0 block from scale and data
fn make_q8_0_block(scale: f32, data: [i8; 32]) -> BlockQ8_0 {
    // Q8_0 layout: scale (f16, 2 bytes) + data (32 i8, 32 bytes) = 34 bytes
    let mut bytes = [0u8; 34];
    let scale_f16 = f16::from_f32(scale);
    bytes[0..2].copy_from_slice(&scale_f16.to_le_bytes());
    bytes[2..34].copy_from_slice(pulp::bytemuck::cast_slice(&data));
    *pulp::bytemuck::from_bytes(&bytes)
}

fn bench_qmatmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("qmatmul_q8_0");

    // Matrix sizes: (M, K, N) where LHS is M x K and RHS is K x N
    let sizes: &[(usize, usize, usize)] = &[
        (1, 256, 256), // Single row (typical for inference)
        (1, 512, 512),
        (1, 1024, 1024),
        (4, 256, 256), // Small batch
        (4, 512, 512),
        (32, 256, 256), // Larger batch
        (32, 512, 512),
    ];

    for &(m, k, n) in sizes {
        let ops = (2 * m * n * k) as u64;
        group.throughput(Throughput::Elements(ops));

        group.bench_with_input(
            BenchmarkId::new("matmul_quantized", format!("{}x{}x{}", m, k, n)),
            &(m, k, n),
            |b, &(m, k, n)| {
                // Create LHS f32 tensor
                let lhs_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
                let lhs = ConcreteTensor::<f32, 2>::from_slice([m, k], &lhs_data);

                // Create quantized RHS tensor
                let rhs_shape = [k, n];
                let num_blocks = (k * n) / 32;
                let mut blocks_vec: AVec<BlockQ8_0> = AVec::with_capacity(64, num_blocks);
                for i in 0..num_blocks {
                    let mut data = [0i8; 32];
                    for j in 0..32 {
                        data[j] = ((i + j) % 100) as i8 - 50;
                    }
                    blocks_vec.push(make_q8_0_block(0.1, data));
                }
                let blocks = blocks_vec.into_boxed_slice();
                let rhs = QuantizedTensor::from_blocks(rhs_shape, blocks);

                b.iter(|| black_box(black_box(&lhs).q_mat_mul(black_box(&rhs))));
            },
        );

        // Compare with dequantize + regular matmul
        group.bench_with_input(
            BenchmarkId::new("dequant_then_matmul", format!("{}x{}x{}", m, k, n)),
            &(m, k, n),
            |b, &(m, k, n)| {
                let lhs_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
                let lhs = Tensor::from_slice([m, k], &lhs_data);

                let rhs_shape = [k, n];
                let num_blocks = (k * n) / 32;
                let mut blocks_vec: AVec<BlockQ8_0> = AVec::with_capacity(64, num_blocks);
                for i in 0..num_blocks {
                    let mut data = [0i8; 32];
                    for j in 0..32 {
                        data[j] = ((i + j) % 100) as i8 - 50;
                    }
                    blocks_vec.push(make_q8_0_block(0.1, data));
                }
                let blocks = blocks_vec.into_boxed_slice();
                let rhs = QuantizedTensor::from_blocks(rhs_shape, blocks);

                b.iter(|| {
                    let rhs_dequant = rhs.dequantize();
                    let rhs_tensor = Tensor::new(rhs_dequant);
                    black_box(lhs.clone().matmul(black_box(rhs_tensor)))
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_qmatmul);
criterion_main!(benches);
