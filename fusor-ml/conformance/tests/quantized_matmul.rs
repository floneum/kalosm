mod common;

use std::mem::size_of;

use common::{assert_approx_devices, flatten2, matmul2, transpose2};
use fusor::{QMatrix, Tensor};
use fusor_cpu::{
    BlockQ4_0, BlockQ4K, BlockQ5_0, BlockQ5K, BlockQ6K, BlockQ8_0, ConcreteTensor, GgmlType,
    GgufBlock, QuantizedTensor,
};

#[derive(Clone)]
struct QuantizedFixture {
    ty: GgmlType,
    weight_shape: [usize; 2],
    raw_bytes: Vec<u8>,
    input: Vec<Vec<f32>>,
    dequantized: Vec<Vec<f32>>,
    q_mat_mul: Vec<Vec<f32>>,
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

fn input_rows(cols: usize, rows: usize) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|row| {
            (0..cols)
                .map(|col| {
                    let base = ((row * cols + col * 3) % 17) as f32;
                    0.02 + base * 0.01
                })
                .collect()
        })
        .collect()
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
    let input = input_rows(weight_shape[1], input_row_count);
    let weights = QuantizedTensor::<B>::from_raw_bytes(weight_shape, &raw_bytes);
    let dequantized = concrete_to_rows(&weights.dequantize::<2>(), weight_shape);

    let lhs_shape = [input.len(), input[0].len()];
    let lhs = ConcreteTensor::<f32, 2>::from_slice(lhs_shape, &flatten2(&input));
    let q_mat_mul_shape = [lhs_shape[0], weight_shape[0]];
    let q_mat_mul = concrete_to_rows(&lhs.q_mat_mul(&weights), q_mat_mul_shape);

    QuantizedFixture {
        ty,
        weight_shape,
        raw_bytes,
        input,
        dequantized,
        q_mat_mul,
        dequantize_tol,
        q_mat_mul_tol,
    }
}

fn q4_0_raw_bytes(shape: [usize; 2]) -> Vec<u8> {
    let mut bytes =
        Vec::with_capacity(block_count(shape, BlockQ4_0::BLOCK_SIZE) * size_of::<BlockQ4_0>());
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
    let mut bytes =
        Vec::with_capacity(block_count(shape, BlockQ5_0::BLOCK_SIZE) * size_of::<BlockQ5_0>());
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
    let mut bytes =
        Vec::with_capacity(block_count(shape, BlockQ8_0::BLOCK_SIZE) * size_of::<BlockQ8_0>());
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
    let mut bytes =
        Vec::with_capacity(block_count(shape, BlockQ4K::BLOCK_SIZE) * size_of::<BlockQ4K>());
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
    let mut bytes =
        Vec::with_capacity(block_count(shape, BlockQ5K::BLOCK_SIZE) * size_of::<BlockQ5K>());
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
    let mut bytes =
        Vec::with_capacity(block_count(shape, BlockQ6K::BLOCK_SIZE) * size_of::<BlockQ6K>());
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

fn q4_0_fixture() -> QuantizedFixture {
    build_fixture::<BlockQ4_0>(
        GgmlType::Q4_0,
        [2, 64],
        q4_0_raw_bytes([2, 64]),
        3,
        1e-5,
        1.0,
    )
}

fn q5_0_fixture() -> QuantizedFixture {
    build_fixture::<BlockQ5_0>(
        GgmlType::Q5_0,
        [2, 64],
        q5_0_raw_bytes([2, 64]),
        1,
        1e-5,
        1.0,
    )
}

fn q8_0_fixture() -> QuantizedFixture {
    build_fixture::<BlockQ8_0>(
        GgmlType::Q8_0,
        [2, 64],
        q8_0_raw_bytes([2, 64]),
        1,
        1e-5,
        0.5,
    )
}

fn q4k_fixture() -> QuantizedFixture {
    build_fixture::<BlockQ4K>(
        GgmlType::Q4K,
        [2, 512],
        q4k_raw_bytes([2, 512]),
        1,
        1e-4,
        2.0,
    )
}

fn q5k_fixture() -> QuantizedFixture {
    build_fixture::<BlockQ5K>(
        GgmlType::Q5K,
        [2, 512],
        q5k_raw_bytes([2, 512]),
        1,
        1e-4,
        1.0,
    )
}

fn q6k_fixture() -> QuantizedFixture {
    build_fixture::<BlockQ6K>(
        GgmlType::Q6K,
        [2, 512],
        q6k_raw_bytes([2, 512]),
        1,
        1e-4,
        1.0,
    )
}

fn q5_0_wide_fixture() -> QuantizedFixture {
    build_fixture::<BlockQ5_0>(
        GgmlType::Q5_0,
        [2, 64],
        q5_0_raw_bytes([2, 64]),
        3,
        1e-5,
        1.0,
    )
}

fn q8_0_wide_fixture() -> QuantizedFixture {
    build_fixture::<BlockQ8_0>(
        GgmlType::Q8_0,
        [2, 64],
        q8_0_raw_bytes([2, 64]),
        3,
        1e-5,
        0.5,
    )
}

fn q4k_wide_fixture() -> QuantizedFixture {
    build_fixture::<BlockQ4K>(
        GgmlType::Q4K,
        [2, 512],
        q4k_raw_bytes([2, 512]),
        3,
        1e-4,
        2.0,
    )
}

fn q5k_wide_fixture() -> QuantizedFixture {
    build_fixture::<BlockQ5K>(
        GgmlType::Q5K,
        [2, 512],
        q5k_raw_bytes([2, 512]),
        3,
        1e-4,
        1.0,
    )
}

fn q6k_wide_fixture() -> QuantizedFixture {
    build_fixture::<BlockQ6K>(
        GgmlType::Q6K,
        [2, 512],
        q6k_raw_bytes([2, 512]),
        3,
        1e-4,
        1.0,
    )
}

async fn assert_dequantize_matches_cpu_reference(fixture: QuantizedFixture) {
    let QuantizedFixture {
        ty,
        weight_shape,
        raw_bytes,
        dequantized,
        dequantize_tol,
        ..
    } = fixture;

    assert_approx_devices(
        move |device| {
            QMatrix::from_raw_bytes(device, weight_shape, &raw_bytes, ty)
                .unwrap()
                .dequantize::<2>()
        },
        move |device| Tensor::new(device, &dequantized),
        dequantize_tol,
    )
    .await;
}

async fn assert_q_mat_mul_matches_cpu_reference(fixture: QuantizedFixture) {
    let QuantizedFixture {
        ty,
        weight_shape,
        raw_bytes,
        input,
        q_mat_mul,
        q_mat_mul_tol,
        ..
    } = fixture;

    assert_approx_devices(
        move |device| {
            let input: Tensor<2, f32> = Tensor::new(device, &input);
            let weights = QMatrix::from_raw_bytes(device, weight_shape, &raw_bytes, ty).unwrap();
            input.q_mat_mul(&weights)
        },
        move |device| Tensor::new(device, &q_mat_mul),
        q_mat_mul_tol,
    )
    .await;
}

macro_rules! quantized_fixture_tests {
    ($fixture_fn:ident, $dequantize_test:ident, $q_mat_mul_test:ident) => {
        #[tokio::test]
        async fn $dequantize_test() {
            assert_dequantize_matches_cpu_reference($fixture_fn()).await;
        }

        #[tokio::test]
        async fn $q_mat_mul_test() {
            assert_q_mat_mul_matches_cpu_reference($fixture_fn()).await;
        }
    };
}

quantized_fixture_tests!(
    q4_0_fixture,
    q4_0_dequantize_matches_cpu_reference,
    q4_0_q_mat_mul_matches_cpu_reference
);
quantized_fixture_tests!(
    q5_0_fixture,
    q5_0_dequantize_matches_cpu_reference,
    q5_0_q_mat_mul_matches_cpu_reference
);
quantized_fixture_tests!(
    q8_0_fixture,
    q8_0_dequantize_matches_cpu_reference,
    q8_0_q_mat_mul_matches_cpu_reference
);
quantized_fixture_tests!(
    q4k_fixture,
    q4k_dequantize_matches_cpu_reference,
    q4k_q_mat_mul_matches_cpu_reference
);
quantized_fixture_tests!(
    q5k_fixture,
    q5k_dequantize_matches_cpu_reference,
    q5k_q_mat_mul_matches_cpu_reference
);
quantized_fixture_tests!(
    q6k_fixture,
    q6k_dequantize_matches_cpu_reference,
    q6k_q_mat_mul_matches_cpu_reference
);

#[tokio::test]
async fn q5_0_q_mat_mul_multi_row_matches_cpu_reference() {
    assert_q_mat_mul_matches_cpu_reference(q5_0_wide_fixture()).await;
}

#[tokio::test]
async fn q8_0_q_mat_mul_multi_row_matches_cpu_reference() {
    assert_q_mat_mul_matches_cpu_reference(q8_0_wide_fixture()).await;
}

#[tokio::test]
async fn q4k_q_mat_mul_multi_row_matches_cpu_reference() {
    assert_q_mat_mul_matches_cpu_reference(q4k_wide_fixture()).await;
}

#[tokio::test]
async fn q5k_q_mat_mul_multi_row_matches_cpu_reference() {
    assert_q_mat_mul_matches_cpu_reference(q5k_wide_fixture()).await;
}

#[tokio::test]
async fn q6k_q_mat_mul_multi_row_matches_cpu_reference() {
    assert_q_mat_mul_matches_cpu_reference(q6k_wide_fixture()).await;
}

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

fn f32_input() -> Vec<Vec<f32>> {
    vec![vec![1.0, 1.0, 1.0, 1.0], vec![0.5, -1.0, 2.0, 3.5]]
}

#[tokio::test]
async fn f32_q_matrix_q_mat_mul_matches_host_reference() {
    let input = f32_input();
    let weights = f32_weight_rows();
    let raw_bytes = f32_weight_bytes();

    assert_approx_devices(
        |device| {
            let input: Tensor<2, f32> = Tensor::new(device, &input);
            let weights =
                QMatrix::from_raw_bytes(device, [2, 4], &raw_bytes, GgmlType::F32).unwrap();
            input.q_mat_mul(&weights)
        },
        |device| Tensor::new(device, &matmul2(&input, &transpose2(&weights))),
        1e-6,
    )
    .await;
}
