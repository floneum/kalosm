mod common;

use std::mem::size_of;

use common::{matmul2, transpose2};
use fusor::{Device, QMatrix, Tensor, ToVec2};
use fusor_conformance::{FuzzGenerator, approx_compare};
use fusor_cpu::{
    BlockQ4_0, BlockQ4K, BlockQ5_0, BlockQ5K, BlockQ6K, BlockQ8_0, ConcreteTensor, GgmlType,
    GgufBlock, QuantizedTensor,
};
use rand::distr::Uniform;

#[derive(Clone)]
struct QuantizedFixture {
    ty: GgmlType,
    weight_shape: [usize; 2],
    raw_bytes: Vec<u8>,
    input_row_count: usize,
    dequantized: Vec<Vec<f32>>,
    dequantize_tol: f32,
    q_mat_mul_tol: f32,
}

fn push_f16(bytes: &mut Vec<u8>, value: f32) {
    bytes.extend_from_slice(&half::f16::from_f32(value).to_le_bytes());
}

fn packed_nibble_byte(low: usize, high: usize) -> u8 {
    ((low & 0x0F) as u8) | (((high & 0x0F) as u8) << 4)
}

fn block_count(shape: [usize; 2], block_size: usize) -> usize {
    (shape[0] * shape[1]) / block_size
}

fn raw_bytes_buffer<B: GgufBlock>(shape: [usize; 2]) -> Vec<u8> {
    Vec::with_capacity(block_count(shape, B::BLOCK_SIZE) * size_of::<B>())
}

fn concrete_to_rows(tensor: &ConcreteTensor<f32, 2>, shape: [usize; 2]) -> Vec<Vec<f32>> {
    (0..shape[0])
        .map(|row| (0..shape[1]).map(|col| tensor.get([row, col])).collect())
        .collect()
}

fn build_fixture<B>(
    ty: GgmlType,
    weight_shape: [usize; 2],
    raw_bytes: Vec<u8>,
    input_row_count: usize,
    dequantize_tol: f32,
    q_mat_mul_tol: f32,
) -> QuantizedFixture
where
    B: GgufBlock + Sync,
    B::Dequantized: AsRef<[f32]>,
    B::ActivationBlock: Send + Sync,
{
    let weights = QuantizedTensor::<B>::from_raw_bytes(weight_shape, &raw_bytes);
    let dequantized = concrete_to_rows(&weights.dequantize::<2>(), weight_shape);

    QuantizedFixture {
        ty,
        weight_shape,
        raw_bytes,
        input_row_count,
        dequantized,
        dequantize_tol,
        q_mat_mul_tol,
    }
}

fn qmatrix_from_raw_bytes(
    device: &Device,
    weight_shape: [usize; 2],
    raw_bytes: &[u8],
    ty: GgmlType,
) -> QMatrix {
    QMatrix::from_raw_bytes(device, weight_shape, raw_bytes, ty).unwrap()
}

fn q_mat_mul_input_fuzz(
    input_row_count: usize,
    weight_shape: [usize; 2],
    seed: u64,
    distribution: Uniform<f32>,
) -> FuzzGenerator<2, f32> {
    FuzzGenerator::<2, f32>::new([input_row_count, weight_shape[1]])
        .with_seed(seed)
        .with_distribution(distribution)
}

async fn assert_dequantize_matches_host_reference(
    ty: GgmlType,
    weight_shape: [usize; 2],
    raw_bytes: Vec<u8>,
    dequantized: Vec<Vec<f32>>,
    dequantize_tol: f32,
) {
    fusor_conformance::assert(move |device: Device| {
        let raw_bytes = raw_bytes.clone();
        async move { qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, ty).dequantize::<2>() }
    })
    .arg(|device: &Device| device.clone())
    .equal_to(move |device: Device| {
        let dequantized = dequantized.clone();
        async move { Tensor::new(&device, &dequantized) }
    })
    .compare_with(approx_compare::<2, f32>(dequantize_tol))
    .await
    .unwrap();
}

/// Fuzz configuration for input rows in `assert_q_mat_mul_matches_host_reference`.
struct QMatMulFuzz {
    seed: u64,
    distribution: Uniform<f32>,
}

async fn assert_q_mat_mul_matches_host_reference(fixture: &QuantizedFixture, fuzz: QMatMulFuzz) {
    let ty = fixture.ty;
    let weight_shape = fixture.weight_shape;
    let raw_bytes = fixture.raw_bytes.clone();
    let expected_weights = fixture.dequantized.clone();
    let input_row_count = fixture.input_row_count;
    let q_mat_mul_tol = fixture.q_mat_mul_tol;
    let QMatMulFuzz { seed, distribution } = fuzz;

    fusor_conformance::assert(move |input: Tensor<2, f32>| {
        let raw_bytes = raw_bytes.clone();
        async move {
            let weights = qmatrix_from_raw_bytes(&input.device(), weight_shape, &raw_bytes, ty);
            input.q_mat_mul(&weights)
        }
    })
    .arg(q_mat_mul_input_fuzz(
        input_row_count,
        weight_shape,
        seed,
        distribution,
    ))
    .equal_to(move |input: Tensor<2, f32>| {
        let expected_weights = expected_weights.clone();
        async move {
            let device = input.device();
            let input_values = input.as_slice().await.unwrap().to_vec2();
            Tensor::new(
                &device,
                &matmul2(&input_values, &transpose2(&expected_weights)),
            )
        }
    })
    .compare_with(approx_compare::<2, f32>(q_mat_mul_tol))
    .runs(3)
    .await
    .unwrap();
}

fn q4_0_raw_bytes(shape: [usize; 2]) -> Vec<u8> {
    let mut bytes = raw_bytes_buffer::<BlockQ4_0>(shape);
    for block in 0..block_count(shape, BlockQ4_0::BLOCK_SIZE) {
        push_f16(&mut bytes, 0.025 + block as f32 * 0.003);
        for i in 0..16 {
            bytes.push(packed_nibble_byte(
                10 + ((block + i * 2) % 6),
                9 + ((block * 3 + i * 5) % 7),
            ));
        }
    }
    bytes
}

fn q5_0_raw_bytes(shape: [usize; 2]) -> Vec<u8> {
    let mut bytes = raw_bytes_buffer::<BlockQ5_0>(shape);
    for block in 0..block_count(shape, BlockQ5_0::BLOCK_SIZE) {
        push_f16(&mut bytes, 0.02 + block as f32 * 0.0025);
        bytes.extend(std::iter::repeat_n(0xFF, 4));
        for i in 0..16 {
            bytes.push(packed_nibble_byte(
                4 + ((block + i * 3) % 8),
                6 + ((block * 2 + i * 5) % 8),
            ));
        }
    }
    bytes
}

fn q8_0_raw_bytes(shape: [usize; 2]) -> Vec<u8> {
    let mut bytes = raw_bytes_buffer::<BlockQ8_0>(shape);
    for block in 0..block_count(shape, BlockQ8_0::BLOCK_SIZE) {
        push_f16(&mut bytes, 0.01 + block as f32 * 0.0015);
        for i in 0..32 {
            let value = (4 + ((block * 7 + i * 5) % 17)) as i8;
            bytes.push(value as u8);
        }
    }
    bytes
}

fn q4k_raw_bytes(shape: [usize; 2]) -> Vec<u8> {
    let mut bytes = raw_bytes_buffer::<BlockQ4K>(shape);
    for block in 0..block_count(shape, BlockQ4K::BLOCK_SIZE) {
        push_f16(&mut bytes, 0.004 + block as f32 * 0.0005);
        push_f16(&mut bytes, 0.0005 + block as f32 * 0.0001);
        for i in 0..BlockQ4K::SCALES_SIZE {
            bytes.push((((block * 5 + i * 3) % 24) + 1) as u8);
        }
        for i in 0..BlockQ4K::WEIGHTS_SIZE {
            bytes.push(packed_nibble_byte(
                10 + ((block + i * 2) % 6),
                11 + ((block * 3 + i) % 5),
            ));
        }
    }
    bytes
}

fn q5k_raw_bytes(shape: [usize; 2]) -> Vec<u8> {
    let mut bytes = raw_bytes_buffer::<BlockQ5K>(shape);
    for block in 0..block_count(shape, BlockQ5K::BLOCK_SIZE) {
        push_f16(&mut bytes, 0.0045 + block as f32 * 0.0004);
        push_f16(&mut bytes, 0.0005 + block as f32 * 0.0001);
        for i in 0..BlockQ5K::SCALES_SIZE {
            bytes.push((((block * 7 + i * 2) % 24) + 1) as u8);
        }
        bytes.extend(std::iter::repeat_n(0xFF, BlockQ5K::QH_SIZE));
        for i in 0..BlockQ5K::QS_SIZE {
            bytes.push(packed_nibble_byte(
                8 + ((block + i * 3) % 8),
                9 + ((block * 2 + i * 5) % 7),
            ));
        }
    }
    bytes
}

fn q6k_raw_bytes(shape: [usize; 2]) -> Vec<u8> {
    let mut bytes = raw_bytes_buffer::<BlockQ6K>(shape);
    for block in 0..block_count(shape, BlockQ6K::BLOCK_SIZE) {
        for i in 0..BlockQ6K::WEIGHTS_LOW_BITS_SIZE {
            bytes.push(packed_nibble_byte(
                block * 5 + i * 3 + 1,
                block * 7 + i * 11 + 2,
            ));
        }
        for i in 0..BlockQ6K::WEIGHTS_HIGH_BITS_SIZE {
            bytes.push(((block * 17 + i * 9 + 0x12) & 0xFF) as u8);
        }
        for i in 0..BlockQ6K::SCALES_SIZE {
            let scale = ((block * 5 + i * 2) % 7 + 1) as i8;
            bytes.push(scale as u8);
        }
        push_f16(&mut bytes, 0.0035 + block as f32 * 0.00035);
    }
    bytes
}

macro_rules! quantized_fixture_fn {
    ($fn_name:ident, $block:ty, $ty:expr, $shape:expr, $raw_bytes_fn:ident, $rows:expr, $deq_tol:expr, $q_tol:expr) => {
        fn $fn_name() -> QuantizedFixture {
            let shape = $shape;
            build_fixture::<$block>($ty, shape, $raw_bytes_fn(shape), $rows, $deq_tol, $q_tol)
        }
    };
}

quantized_fixture_fn!(
    q4_0_fixture,
    BlockQ4_0,
    GgmlType::Q4_0,
    [2, 64],
    q4_0_raw_bytes,
    3,
    1e-5,
    1.0
);
quantized_fixture_fn!(
    q5_0_fixture,
    BlockQ5_0,
    GgmlType::Q5_0,
    [2, 64],
    q5_0_raw_bytes,
    1,
    1e-5,
    1.0
);
quantized_fixture_fn!(
    q8_0_fixture,
    BlockQ8_0,
    GgmlType::Q8_0,
    [2, 64],
    q8_0_raw_bytes,
    1,
    1e-5,
    0.5
);
quantized_fixture_fn!(
    q4k_fixture,
    BlockQ4K,
    GgmlType::Q4K,
    [2, 512],
    q4k_raw_bytes,
    1,
    1e-4,
    2.0
);
quantized_fixture_fn!(
    q5k_fixture,
    BlockQ5K,
    GgmlType::Q5K,
    [2, 512],
    q5k_raw_bytes,
    1,
    1e-4,
    1.0
);
quantized_fixture_fn!(
    q6k_fixture,
    BlockQ6K,
    GgmlType::Q6K,
    [2, 512],
    q6k_raw_bytes,
    1,
    1e-4,
    1.0
);
quantized_fixture_fn!(
    q5_0_wide_fixture,
    BlockQ5_0,
    GgmlType::Q5_0,
    [2, 64],
    q5_0_raw_bytes,
    3,
    1e-5,
    1.0
);
quantized_fixture_fn!(
    q8_0_wide_fixture,
    BlockQ8_0,
    GgmlType::Q8_0,
    [2, 64],
    q8_0_raw_bytes,
    3,
    1e-5,
    0.5
);
quantized_fixture_fn!(
    q4k_wide_fixture,
    BlockQ4K,
    GgmlType::Q4K,
    [2, 512],
    q4k_raw_bytes,
    3,
    1e-4,
    2.0
);
quantized_fixture_fn!(
    q5k_wide_fixture,
    BlockQ5K,
    GgmlType::Q5K,
    [2, 512],
    q5k_raw_bytes,
    3,
    1e-4,
    1.0
);
quantized_fixture_fn!(
    q6k_wide_fixture,
    BlockQ6K,
    GgmlType::Q6K,
    [2, 512],
    q6k_raw_bytes,
    3,
    1e-4,
    1.0
);

macro_rules! quantized_dequantize_test {
    ($test_name:ident, $fixture_fn:ident) => {
        #[tokio::test]
        async fn $test_name() {
            let QuantizedFixture {
                ty,
                weight_shape,
                raw_bytes,
                dequantized,
                dequantize_tol,
                ..
            } = $fixture_fn();
            assert_dequantize_matches_host_reference(
                ty,
                weight_shape,
                raw_bytes,
                dequantized,
                dequantize_tol,
            )
            .await;
        }
    };
}

macro_rules! quantized_q_mat_mul_test {
    ($test_name:ident, $fixture_fn:ident, $seed:expr) => {
        #[tokio::test]
        async fn $test_name() {
            let fixture = $fixture_fn();
            assert_q_mat_mul_matches_host_reference(
                &fixture,
                QMatMulFuzz {
                    seed: $seed,
                    distribution: Uniform::new(-0.25, 0.25).unwrap(),
                },
            )
            .await;
        }
    };
}

macro_rules! quantized_fixture_tests {
    ($fixture_fn:ident, $dequantize_test:ident, $q_mat_mul_test:ident, $seed:expr) => {
        quantized_dequantize_test!($dequantize_test, $fixture_fn);
        quantized_q_mat_mul_test!($q_mat_mul_test, $fixture_fn, $seed);
    };
}

quantized_fixture_tests!(
    q4_0_fixture,
    q4_0_dequantize_matches_cpu_reference,
    q4_0_q_mat_mul_matches_cpu_reference,
    800
);
quantized_fixture_tests!(
    q5_0_fixture,
    q5_0_dequantize_matches_cpu_reference,
    q5_0_q_mat_mul_matches_cpu_reference,
    801
);
quantized_fixture_tests!(
    q8_0_fixture,
    q8_0_dequantize_matches_cpu_reference,
    q8_0_q_mat_mul_matches_cpu_reference,
    802
);
quantized_fixture_tests!(
    q4k_fixture,
    q4k_dequantize_matches_cpu_reference,
    q4k_q_mat_mul_matches_cpu_reference,
    803
);
quantized_fixture_tests!(
    q5k_fixture,
    q5k_dequantize_matches_cpu_reference,
    q5k_q_mat_mul_matches_cpu_reference,
    804
);
quantized_fixture_tests!(
    q6k_fixture,
    q6k_dequantize_matches_cpu_reference,
    q6k_q_mat_mul_matches_cpu_reference,
    805
);

quantized_q_mat_mul_test!(
    q5_0_q_mat_mul_multi_row_matches_cpu_reference,
    q5_0_wide_fixture,
    810
);
quantized_q_mat_mul_test!(
    q8_0_q_mat_mul_multi_row_matches_cpu_reference,
    q8_0_wide_fixture,
    811
);
quantized_q_mat_mul_test!(
    q4k_q_mat_mul_multi_row_matches_cpu_reference,
    q4k_wide_fixture,
    812
);
quantized_q_mat_mul_test!(
    q5k_q_mat_mul_multi_row_matches_cpu_reference,
    q5k_wide_fixture,
    813
);
quantized_q_mat_mul_test!(
    q6k_q_mat_mul_multi_row_matches_cpu_reference,
    q6k_wide_fixture,
    814
);

fn f32_weight_rows() -> Vec<Vec<f32>> {
    vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]]
}

fn f32_weight_bytes() -> Vec<u8> {
    f32_weight_rows()
        .into_iter()
        .flatten()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn f16_weight_bytes() -> Vec<u8> {
    let mut bytes = Vec::new();
    for value in f32_weight_rows().into_iter().flatten() {
        push_f16(&mut bytes, value);
    }
    bytes
}

#[tokio::test]
async fn f32_q_matrix_q_mat_mul_matches_host_reference() {
    let fixture = QuantizedFixture {
        ty: GgmlType::F32,
        weight_shape: [2, 4],
        raw_bytes: f32_weight_bytes(),
        input_row_count: 2,
        dequantized: f32_weight_rows(),
        dequantize_tol: 1e-6,
        q_mat_mul_tol: 1e-6,
    };
    assert_q_mat_mul_matches_host_reference(
        &fixture,
        QMatMulFuzz {
            seed: 820,
            distribution: Uniform::new(-0.5, 0.5).unwrap(),
        },
    )
    .await;
}

#[tokio::test]
async fn f16_q_matrix_q_mat_mul_matches_host_reference() {
    let fixture = QuantizedFixture {
        ty: GgmlType::F16,
        weight_shape: [2, 4],
        raw_bytes: f16_weight_bytes(),
        input_row_count: 2,
        dequantized: f32_weight_rows(),
        dequantize_tol: 1e-3,
        q_mat_mul_tol: 1e-3,
    };
    assert_q_mat_mul_matches_host_reference(
        &fixture,
        QMatMulFuzz {
            seed: 821,
            distribution: Uniform::new(-0.5, 0.5).unwrap(),
        },
    )
    .await;
}

// ---- Batched / transposed q_mat_mul regressions ----
//
// These restore coverage that was deleted with
// `core/src/quantized/matmul/mod.rs::test_fuzz_q_mat_mul_transposed`,
// `test_fuzz_q_mat_mul_gemv_transposed`,
// `cpu/src/quantized.rs::test_batched_q_mat_mul_3d/4d/_matches_unbatched`.

fn deterministic_input(shape: &[usize], seed: u32) -> Vec<f32> {
    let total: usize = shape.iter().product();
    (0..total)
        .map(|i| {
            let v = ((i + seed as usize) % 23) as f32;
            (v - 11.0) * 0.04
        })
        .collect()
}

async fn assert_q_mat_mul_3d_batch(input_rows: usize) {
    use fusor::Device;
    use fusor_conformance::available_devices;

    let weight_shape = [2usize, 64];
    let raw_bytes = q8_0_raw_bytes(weight_shape);
    let cpu_weights =
        qmatrix_from_raw_bytes(&Device::Cpu, weight_shape, &raw_bytes, GgmlType::Q8_0);
    let dequantized_rows = cpu_weights
        .dequantize::<2>()
        .as_slice()
        .await
        .unwrap()
        .to_vec2();
    let weights_t = transpose2(&dequantized_rows);

    for batch in [1usize, 2, 3] {
        let shape = [batch, input_rows, weight_shape[1]];
        let data = deterministic_input(&shape, 901 + batch as u32);

        let cpu_input: Tensor<3, f32> = Tensor::from_slice(&Device::Cpu, shape, &data);
        let cpu_result = cpu_input.q_mat_mul(&cpu_weights).to_concrete();

        // Reference: batched matmul against the dequantized weights.
        let mut expected_rows = Vec::with_capacity(batch);
        for b in 0..batch {
            let slice: Vec<Vec<f32>> = (0..input_rows)
                .map(|m| {
                    let start = ((b * input_rows) + m) * weight_shape[1];
                    data[start..start + weight_shape[1]].to_vec()
                })
                .collect();
            expected_rows.push(matmul2(&slice, &weights_t));
        }
        let expected = Tensor::new(&Device::Cpu, &expected_rows);
        fusor_conformance::approx_eq(&cpu_result, &expected, 5e-2)
            .await
            .unwrap();

        for device in available_devices().await {
            let weights = qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, GgmlType::Q8_0);
            let input: Tensor<3, f32> = Tensor::from_slice(&device, shape, &data);
            let actual = input.q_mat_mul(&weights).to_concrete();
            fusor_conformance::approx_eq(&actual, &cpu_result, 5e-2)
                .await
                .unwrap();
        }
    }
}

#[tokio::test]
async fn q_mat_mul_batched_3d_matches_host_reference() {
    assert_q_mat_mul_3d_batch(1).await;
    assert_q_mat_mul_3d_batch(3).await;
}

#[tokio::test]
async fn q_mat_mul_chunked_sgemm_edge_n_matches_host_reference() {
    use fusor::Device;
    use fusor_conformance::available_devices;

    let weight_shape = [34usize, 32];
    let raw_bytes = q8_0_raw_bytes(weight_shape);
    let cpu_weights =
        qmatrix_from_raw_bytes(&Device::Cpu, weight_shape, &raw_bytes, GgmlType::Q8_0);

    let shape = [2usize, 40, weight_shape[1]];
    let data = deterministic_input(&shape, 1701);
    let cpu_input: Tensor<3, f32> = Tensor::from_slice(&Device::Cpu, shape, &data);
    let cpu_result = cpu_input.q_mat_mul(&cpu_weights).to_concrete();

    for device in available_devices().await {
        let weights = qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, GgmlType::Q8_0);
        let input: Tensor<3, f32> = Tensor::from_slice(&device, shape, &data);
        let actual = input.q_mat_mul(&weights).to_concrete();
        fusor_conformance::approx_eq(&actual, &cpu_result, 5e-2)
            .await
            .unwrap();
    }
}

#[tokio::test]
async fn q_mat_mul_transposed_input_matches_host_reference() {
    use fusor::Device;
    use fusor_conformance::available_devices;

    // Build [N, M, B] and transpose(0, 2) -> [B, M, N], matching the deleted
    // `test_fuzz_q_mat_mul_transposed` topology.
    let weight_shape = [2usize, 64];
    let raw_bytes = q8_0_raw_bytes(weight_shape);
    let cpu_weights =
        qmatrix_from_raw_bytes(&Device::Cpu, weight_shape, &raw_bytes, GgmlType::Q8_0);
    let dequantized_rows = cpu_weights
        .dequantize::<2>()
        .as_slice()
        .await
        .unwrap()
        .to_vec2();
    let weights_t = transpose2(&dequantized_rows);

    for &(input_rows, batch) in &[(2usize, 2usize), (1, 3)] {
        let shape = [weight_shape[1], input_rows, batch];
        let data = deterministic_input(&shape, 1100 + batch as u32);

        let cpu_input: Tensor<3, f32> = Tensor::from_slice(&Device::Cpu, shape, &data);
        let cpu_result = cpu_input
            .transpose(0, 2)
            .q_mat_mul(&cpu_weights)
            .to_concrete();

        // Build expected via the transposed input layout.
        let mut expected_rows = Vec::with_capacity(batch);
        for b in 0..batch {
            let slice: Vec<Vec<f32>> = (0..input_rows)
                .map(|m| {
                    (0..weight_shape[1])
                        .map(|n| {
                            let idx = (n * input_rows + m) * batch + b;
                            data[idx]
                        })
                        .collect()
                })
                .collect();
            expected_rows.push(matmul2(&slice, &weights_t));
        }
        let expected = Tensor::new(&Device::Cpu, &expected_rows);
        fusor_conformance::approx_eq(&cpu_result, &expected, 5e-2)
            .await
            .unwrap();

        for device in available_devices().await {
            let weights = qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, GgmlType::Q8_0);
            let input: Tensor<3, f32> = Tensor::from_slice(&device, shape, &data);
            let actual = input.transpose(0, 2).q_mat_mul(&weights).to_concrete();
            fusor_conformance::approx_eq(&actual, &cpu_result, 5e-2)
                .await
                .unwrap();
        }
    }
}

#[tokio::test]
async fn q_mat_mul_batched_matches_unbatched_property() {
    // Batched 3D q_mat_mul produces the same per-batch slice as 2D q_mat_mul
    // applied independently. Replaces
    // `cpu/src/quantized.rs::test_batched_q_mat_mul_matches_unbatched`.
    use fusor_conformance::available_devices;

    let weight_shape = [2usize, 64];
    let raw_bytes = q8_0_raw_bytes(weight_shape);
    let batch = 3;
    let input_rows = 2;
    let shape = [batch, input_rows, weight_shape[1]];
    let data = deterministic_input(&shape, 1300);

    for device in available_devices().await {
        let weights = qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, GgmlType::Q8_0);
        let batched: Tensor<3, f32> = Tensor::from_slice(&device, shape, &data);
        let batched_result = batched.q_mat_mul(&weights).to_concrete();

        for b in 0..batch {
            let slice_data: Vec<f32> = data
                [b * input_rows * weight_shape[1]..(b + 1) * input_rows * weight_shape[1]]
                .to_vec();
            let unbatched: Tensor<2, f32> =
                Tensor::from_slice(&device, [input_rows, weight_shape[1]], &slice_data);
            let unbatched_result = unbatched.q_mat_mul(&weights).to_concrete();

            // Pull batched slice as 2D for comparison.
            let batched_slice = batched_result
                .clone()
                .slice([b..b + 1, 0..input_rows, 0..weight_shape[0]])
                .reshape([input_rows, weight_shape[0]])
                .to_concrete();
            fusor_conformance::approx_eq(&batched_slice, &unbatched_result, 1e-4)
                .await
                .unwrap();
        }
    }
}

#[tokio::test]
async fn q_mat_mul_broadcasted_batch_matches_unbatched_property() {
    use fusor_conformance::available_devices;

    async fn assert_fixture(fixture: QuantizedFixture) {
        use fusor_conformance::available_devices;

        let batch = 8;
        let input_rows = 6;
        let shape = [1, input_rows, fixture.weight_shape[1]];
        let data = deterministic_input(&shape, 1400 + fixture.weight_shape[1] as u32);

        for device in available_devices().await {
            let weights = qmatrix_from_raw_bytes(
                &device,
                fixture.weight_shape,
                &fixture.raw_bytes,
                fixture.ty,
            );
            let base: Tensor<3, f32> = Tensor::from_slice(&device, shape, &data);
            let broadcasted = base.broadcast_as([batch, input_rows, fixture.weight_shape[1]]);
            let batched_result = broadcasted.q_mat_mul(&weights).to_concrete();

            let unbatched: Tensor<2, f32> =
                Tensor::from_slice(&device, [input_rows, fixture.weight_shape[1]], &data);
            let unbatched_result = unbatched.q_mat_mul(&weights).to_concrete();

            for b in 0..batch {
                let batched_slice = batched_result
                    .clone()
                    .slice([b..b + 1, 0..input_rows, 0..fixture.weight_shape[0]])
                    .reshape([input_rows, fixture.weight_shape[0]])
                    .to_concrete();
                fusor_conformance::approx_eq(&batched_slice, &unbatched_result, 1e-4)
                    .await
                    .unwrap();
            }
        }
    }

    for fixture in [
        q4_0_fixture(),
        q5_0_fixture(),
        q8_0_fixture(),
        q4k_fixture(),
        q5k_fixture(),
        q6k_fixture(),
    ] {
        assert_fixture(fixture).await;
    }

    let batch = 8;
    let input_rows = 6;
    let weight_shape = [2usize, 4];
    let shape = [1, input_rows, weight_shape[1]];
    let data = deterministic_input(&shape, 1400);
    let f32_raw_bytes = f32_weight_bytes();
    let f16_raw_bytes = f16_weight_bytes();

    for (ty, raw_bytes) in [
        (GgmlType::F32, f32_raw_bytes),
        (GgmlType::F16, f16_raw_bytes),
    ] {
        for device in available_devices().await {
            let weights = qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, ty);
            let base: Tensor<3, f32> = Tensor::from_slice(&device, shape, &data);
            let broadcasted = base.broadcast_as([batch, input_rows, weight_shape[1]]);
            let batched_result = broadcasted.q_mat_mul(&weights).to_concrete();

            let unbatched: Tensor<2, f32> =
                Tensor::from_slice(&device, [input_rows, weight_shape[1]], &data);
            let unbatched_result = unbatched.q_mat_mul(&weights).to_concrete();

            for b in 0..batch {
                let batched_slice = batched_result
                    .clone()
                    .slice([b..b + 1, 0..input_rows, 0..weight_shape[0]])
                    .reshape([input_rows, weight_shape[0]])
                    .to_concrete();
                fusor_conformance::approx_eq(&batched_slice, &unbatched_result, 1e-4)
                    .await
                    .unwrap();
            }
        }
    }
}
