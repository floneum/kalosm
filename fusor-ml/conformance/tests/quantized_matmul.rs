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

async fn assert_q_mat_mul_matches_host_reference(
    ty: GgmlType,
    weight_shape: [usize; 2],
    raw_bytes: Vec<u8>,
    expected_weights: Vec<Vec<f32>>,
    input_row_count: usize,
    seed: u64,
    distribution: Uniform<f32>,
    q_mat_mul_tol: f32,
) {
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
        for _ in 0..4 {
            bytes.push(0xFF);
        }
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
        for _ in 0..BlockQ5K::QH_SIZE {
            bytes.push(0xFF);
        }
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
            let QuantizedFixture {
                ty,
                weight_shape,
                raw_bytes,
                input_row_count,
                dequantized,
                q_mat_mul_tol,
                ..
            } = $fixture_fn();
            assert_q_mat_mul_matches_host_reference(
                ty,
                weight_shape,
                raw_bytes,
                dequantized,
                input_row_count,
                $seed,
                Uniform::new(-0.25, 0.25).unwrap(),
                q_mat_mul_tol,
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

#[tokio::test]
async fn f32_q_matrix_q_mat_mul_matches_host_reference() {
    assert_q_mat_mul_matches_host_reference(
        GgmlType::F32,
        [2, 4],
        f32_weight_bytes(),
        f32_weight_rows(),
        2,
        820,
        Uniform::new(-0.5, 0.5).unwrap(),
        1e-6,
    )
    .await;
}
