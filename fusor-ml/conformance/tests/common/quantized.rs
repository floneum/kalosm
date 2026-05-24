#![allow(dead_code)]

use std::mem::size_of;

use fusor::{
    BlockQ4_0, BlockQ4K, BlockQ5_0, BlockQ5K, BlockQ6K, BlockQ8_0, Device, GgmlType, GgufBlock,
    QMatrix, QuantizedTensor, Tensor, fusion::Concrete,
};
use fusor_conformance::{FuzzGenerator, approx_compare};
use rand::distr::Uniform;

#[derive(Clone)]
pub struct QuantizedFixture {
    pub ty: GgmlType,
    pub weight_shape: [usize; 2],
    pub raw_bytes: Vec<u8>,
    pub input_row_count: usize,
    pub dequantized: Vec<Vec<f32>>,
    pub dequantize_tol: f32,
    pub q_mat_mul_tol: f32,
}

pub fn push_f16(bytes: &mut Vec<u8>, value: f32) {
    bytes.extend_from_slice(&half::f16::from_f32(value).to_le_bytes());
}

pub fn packed_nibble_byte(low: usize, high: usize) -> u8 {
    ((low & 0x0F) as u8) | (((high & 0x0F) as u8) << 4)
}

pub fn block_count(shape: [usize; 2], block_size: usize) -> usize {
    (shape[0] * shape[1]) / block_size
}

pub fn raw_bytes_buffer<B: GgufBlock>(shape: [usize; 2]) -> Vec<u8> {
    Vec::with_capacity(block_count(shape, B::BLOCK_SIZE) * size_of::<B>())
}

pub fn concrete_to_rows(tensor: &Concrete<f32, 2>, shape: [usize; 2]) -> Vec<Vec<f32>> {
    (0..shape[0])
        .map(|row| (0..shape[1]).map(|col| tensor.get([row, col])).collect())
        .collect()
}

pub fn build_fixture<B>(
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

pub fn qmatrix_from_raw_bytes(
    device: &Device,
    weight_shape: [usize; 2],
    raw_bytes: &[u8],
    ty: GgmlType,
) -> QMatrix {
    QMatrix::from_raw_bytes(device, weight_shape, raw_bytes, ty).unwrap()
}

pub fn q_mat_mul_input_fuzz(
    input_row_count: usize,
    weight_shape: [usize; 2],
    seed: u64,
    distribution: Uniform<f32>,
) -> FuzzGenerator<2, f32> {
    FuzzGenerator::<2, f32>::new([input_row_count, weight_shape[1]])
        .with_seed(seed)
        .with_distribution(distribution)
}

pub async fn assert_dequantize_matches_host_reference(
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
pub struct QMatMulFuzz {
    pub seed: u64,
    pub distribution: Uniform<f32>,
}

pub async fn assert_q_mat_mul_matches_host_reference(
    fixture: &QuantizedFixture,
    fuzz: QMatMulFuzz,
) {
    use fusor::ToVec2;

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
                &super::matmul2(&input_values, &super::transpose2(&expected_weights)),
            )
        }
    })
    .compare_with(approx_compare::<2, f32>(q_mat_mul_tol))
    .runs(3)
    .await
    .unwrap();
}

pub fn deterministic_input(shape: &[usize], seed: u32) -> Vec<f32> {
    let total: usize = shape.iter().product();
    (0..total)
        .map(|i| {
            let v = ((i + seed as usize) % 23) as f32;
            (v - 11.0) * 0.04
        })
        .collect()
}

pub fn q4_0_raw_bytes(shape: [usize; 2]) -> Vec<u8> {
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

pub fn q5_0_raw_bytes(shape: [usize; 2]) -> Vec<u8> {
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

pub fn q8_0_raw_bytes(shape: [usize; 2]) -> Vec<u8> {
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

pub fn q4k_raw_bytes(shape: [usize; 2]) -> Vec<u8> {
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

pub fn q5k_raw_bytes(shape: [usize; 2]) -> Vec<u8> {
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

pub fn q6k_raw_bytes(shape: [usize; 2]) -> Vec<u8> {
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

pub fn f32_weight_rows() -> Vec<Vec<f32>> {
    vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]]
}

pub fn f32_weight_bytes() -> Vec<u8> {
    f32_weight_rows()
        .into_iter()
        .flatten()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

pub fn f16_weight_bytes() -> Vec<u8> {
    let mut bytes = Vec::new();
    for value in f32_weight_rows().into_iter().flatten() {
        push_f16(&mut bytes, value);
    }
    bytes
}

macro_rules! quantized_fixture_cases {
    ($($fn_name:ident: $block:ty, $ty:expr, $shape:expr, $raw_bytes_fn:ident, $rows:expr, $deq_tol:expr, $q_tol:expr, $seed:expr, $dequantize:expr;)*) => {
        $(
            pub fn $fn_name() -> QuantizedFixture {
                let shape = $shape;
                build_fixture::<$block>($ty, shape, $raw_bytes_fn(shape), $rows, $deq_tol, $q_tol)
            }
        )*

        pub const QUANTIZED_FIXTURE_CASES: &[(fn() -> QuantizedFixture, u64, bool)] = &[
            $(($fn_name, $seed, $dequantize),)*
        ];
    };
}

quantized_fixture_cases! {
    q4_0_fixture: BlockQ4_0, GgmlType::Q4_0, [2, 64], q4_0_raw_bytes, 3, 1e-5, 1.0, 800, true;
    q5_0_fixture: BlockQ5_0, GgmlType::Q5_0, [2, 64], q5_0_raw_bytes, 1, 1e-5, 1.0, 801, true;
    q8_0_fixture: BlockQ8_0, GgmlType::Q8_0, [2, 64], q8_0_raw_bytes, 1, 1e-5, 0.5, 802, true;
    q4k_fixture: BlockQ4K, GgmlType::Q4K, [2, 512], q4k_raw_bytes, 1, 1e-4, 2.0, 803, true;
    q5k_fixture: BlockQ5K, GgmlType::Q5K, [2, 512], q5k_raw_bytes, 1, 1e-4, 1.0, 804, true;
    q6k_fixture: BlockQ6K, GgmlType::Q6K, [2, 512], q6k_raw_bytes, 1, 1e-4, 1.0, 805, true;
    q5_0_wide_fixture: BlockQ5_0, GgmlType::Q5_0, [2, 64], q5_0_raw_bytes, 3, 1e-5, 1.0, 810, false;
    q8_0_wide_fixture: BlockQ8_0, GgmlType::Q8_0, [2, 64], q8_0_raw_bytes, 3, 1e-5, 0.5, 811, false;
    q8_0_single_row_wide_output_fixture: BlockQ8_0, GgmlType::Q8_0, [96, 64], q8_0_raw_bytes, 1, 1e-5, 0.5, 826, false;
    q4k_wide_fixture: BlockQ4K, GgmlType::Q4K, [2, 512], q4k_raw_bytes, 3, 1e-4, 2.0, 812, false;
    q4k_large_qgemv_fixture: BlockQ4K, GgmlType::Q4K, [8192, 512], q4k_raw_bytes, 1, 1e-4, 2.0, 827, false;
    q4k_mid_qgemv_fixture: BlockQ4K, GgmlType::Q4K, [4096, 512], q4k_raw_bytes, 1, 1e-4, 2.0, 830, false;
    q4k_tall_qgemv_fixture: BlockQ4K, GgmlType::Q4K, [128, 4608], q4k_raw_bytes, 1, 1e-4, 2.0, 828, false;
    q4k_tail_rows_wide_output_fixture: BlockQ4K, GgmlType::Q4K, [128, 512], q4k_raw_bytes, 48, 1e-4, 2.0, 832, false;
    q4k_tail_rows_llama_k_fixture: BlockQ4K, GgmlType::Q4K, [128, 4096], q4k_raw_bytes, 48, 1e-4, 2.0, 833, false;
    q5k_wide_fixture: BlockQ5K, GgmlType::Q5K, [2, 512], q5k_raw_bytes, 3, 1e-4, 1.0, 813, false;
    q6k_wide_fixture: BlockQ6K, GgmlType::Q6K, [2, 512], q6k_raw_bytes, 3, 1e-4, 1.0, 814, false;
    q6k_large_qgemv_fixture: BlockQ6K, GgmlType::Q6K, [8192, 512], q6k_raw_bytes, 1, 1e-4, 1.0, 831, false;
    q6k_tall_qgemv_fixture: BlockQ6K, GgmlType::Q6K, [128, 4608], q6k_raw_bytes, 1, 1e-4, 1.0, 829, false;
    q4_0_tiled_fixture: BlockQ4_0, GgmlType::Q4_0, [64, 64], q4_0_raw_bytes, 64, 1e-5, 1.0, 820, false;
    q5_0_tiled_fixture: BlockQ5_0, GgmlType::Q5_0, [64, 64], q5_0_raw_bytes, 64, 1e-5, 1.0, 821, false;
    q8_0_tiled_fixture: BlockQ8_0, GgmlType::Q8_0, [64, 64], q8_0_raw_bytes, 64, 1e-5, 0.5, 822, false;
    q4k_tiled_fixture: BlockQ4K, GgmlType::Q4K, [64, 512], q4k_raw_bytes, 64, 1e-4, 2.0, 823, false;
    q5k_tiled_fixture: BlockQ5K, GgmlType::Q5K, [64, 512], q5k_raw_bytes, 64, 1e-4, 1.0, 824, false;
    q6k_tiled_fixture: BlockQ6K, GgmlType::Q6K, [64, 512], q6k_raw_bytes, 64, 1e-4, 1.0, 825, false;
}
