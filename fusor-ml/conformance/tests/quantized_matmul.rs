mod common;

use std::mem::size_of;

use common::{matmul2, transpose2};
use fusor::{Device, QMatrix, Tensor, ToVec2};
use fusor_conformance::{FuzzGenerator, approx_compare, approx_eq, available_devices};
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

#[tokio::test]
async fn q4k_paired_natural_form_matches_cpu_reference() {
    for kind in [PairedKind::SwiGLU, PairedKind::GeGLU, PairedKind::ReGLU] {
        for rows in [1, 4] {
            paired_matches_cpu_for_rows(rows, kind).await;
        }
    }
}

#[tokio::test]
async fn q4k_paired_llama_shape_one_hot_matches_cpu_reference() {
    use fusor::D;

    let Ok(device) = Device::new().await else {
        return;
    };
    let weight_shape = [14336usize, 4096usize];
    let pair_len = weight_shape[0] / 2;
    let input_rows = 48usize;
    let selected_k = 777usize;
    let blocks_per_row = weight_shape[1] / BlockQ4K::BLOCK_SIZE;
    let selected_block_in_row = selected_k / BlockQ4K::BLOCK_SIZE;
    let selected_offset = selected_k % BlockQ4K::BLOCK_SIZE;
    let raw_bytes = q4k_raw_bytes(weight_shape);
    let weights = qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, GgmlType::Q4K);
    let selected_weight = |row: usize| {
        let block_index = row * blocks_per_row + selected_block_in_row;
        let offset = block_index * size_of::<BlockQ4K>();
        assert!(offset + size_of::<BlockQ4K>() <= raw_bytes.len());
        let block =
            unsafe { std::ptr::read_unaligned(raw_bytes.as_ptr().add(offset).cast::<BlockQ4K>()) };
        block.dequantize().as_ref()[selected_offset]
    };
    let mut input_data = vec![0.0f32; input_rows * weight_shape[1]];
    for row in 0..input_rows {
        input_data[row * weight_shape[1] + selected_k] = 0.125 + row as f32 * 0.01;
    }
    let input: Tensor<2, f32> =
        Tensor::from_slice(&device, [input_rows, weight_shape[1]], &input_data);

    let projected = input.q_mat_mul(&weights);
    let gate = projected.narrow(D::Minus1, 0, pair_len).to_concrete();
    let up = projected
        .narrow(D::Minus1, pair_len, pair_len)
        .to_concrete();
    let actual = (gate.silu() * up).to_concrete().as_slice().await.unwrap();

    assert_eq!(actual.shape(), &[input_rows, pair_len]);
    for row in 0..input_rows {
        let input_value = input_data[row * weight_shape[1] + selected_k];
        for col in [0usize, 1, 63, 64, 511, 1024, 4095, pair_len - 1] {
            let gate = input_value * selected_weight(col);
            let up = input_value * selected_weight(col + pair_len);
            let expected = (gate / (1.0 + (-gate).exp())) * up;
            let actual = actual[[row, col]];
            let tolerance = 2.0f32.max(expected.abs() * 1.0e-4);
            assert!(
                (actual - expected).abs() <= tolerance,
                "row={row} col={col} actual={actual} expected={expected} tolerance={tolerance}"
            );
        }
    }
}

#[tokio::test]
async fn q4k_paired_llama_shape_dense_sampled_columns_match_cpu_reference() {
    use fusor::D;

    let Ok(device) = Device::new().await else {
        return;
    };
    let weight_shape = [14336usize, 4096usize];
    let pair_len = weight_shape[0] / 2;
    let input_rows = 48usize;
    let blocks_per_row = weight_shape[1] / BlockQ4K::BLOCK_SIZE;
    let raw_bytes = q4k_raw_bytes(weight_shape);
    let weights = qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, GgmlType::Q4K);
    let block_at = |matrix_row: usize, block_col: usize| {
        let block_index = matrix_row * blocks_per_row + block_col;
        let offset = block_index * size_of::<BlockQ4K>();
        assert!(offset + size_of::<BlockQ4K>() <= raw_bytes.len());
        unsafe { std::ptr::read_unaligned(raw_bytes.as_ptr().add(offset).cast::<BlockQ4K>()) }
    };
    let mut input_data = vec![0.0f32; input_rows * weight_shape[1]];
    for (index, value) in input_data.iter_mut().enumerate() {
        let bucket = (index.wrapping_mul(37).wrapping_add(11)) % 101;
        *value = (bucket as f32 - 50.0) * 0.0025;
    }
    let input: Tensor<2, f32> =
        Tensor::from_slice(&device, [input_rows, weight_shape[1]], &input_data);

    let projected = input.q_mat_mul(&weights);
    let gate = projected.narrow(D::Minus1, 0, pair_len).to_concrete();
    let up = projected
        .narrow(D::Minus1, pair_len, pair_len)
        .to_concrete();
    let actual = (gate.silu() * up).to_concrete().as_slice().await.unwrap();

    assert_eq!(actual.shape(), &[input_rows, pair_len]);
    for row in [0usize, 1, 7, 17, 31, 47] {
        let input_row = &input_data[row * weight_shape[1]..(row + 1) * weight_shape[1]];
        for col in [0usize, 1, 63, 64, 511, 1024, 4095, pair_len - 1] {
            let dot = |matrix_row: usize| {
                (0..blocks_per_row)
                    .map(|block_col| {
                        let block = block_at(matrix_row, block_col);
                        let weights = block.dequantize();
                        weights
                            .as_ref()
                            .iter()
                            .enumerate()
                            .map(|(offset, weight)| {
                                input_row[block_col * BlockQ4K::BLOCK_SIZE + offset] * *weight
                            })
                            .sum::<f32>()
                    })
                    .sum::<f32>()
            };
            let gate = dot(col);
            let up = dot(col + pair_len);
            let expected = (gate / (1.0 + (-gate).exp())) * up;
            let actual = actual[[row, col]];
            let tolerance = 2.0f32.max(expected.abs() * 1.0e-4);
            assert!(
                (actual - expected).abs() <= tolerance,
                "row={row} col={col} actual={actual} expected={expected} tolerance={tolerance}"
            );
        }
    }
}

#[tokio::test]
async fn q4k_q6k_ffn_chain_matches_cpu_reference_for_decode_rows() {
    let hidden = 512usize;
    let intermediate = 512usize;
    let output = 128usize;
    let gate_bytes = q4k_raw_bytes([intermediate, hidden]);
    let up_bytes = q4k_raw_bytes([intermediate, hidden]);
    let down_bytes = q6k_raw_bytes([output, intermediate]);

    fusor_conformance::assert(move |input: Tensor<2, f32>| {
        let gate_bytes = gate_bytes.clone();
        let up_bytes = up_bytes.clone();
        let down_bytes = down_bytes.clone();
        async move {
            let device = input.device();
            let gate =
                qmatrix_from_raw_bytes(&device, [intermediate, hidden], &gate_bytes, GgmlType::Q4K);
            let up =
                qmatrix_from_raw_bytes(&device, [intermediate, hidden], &up_bytes, GgmlType::Q4K);
            let down =
                qmatrix_from_raw_bytes(&device, [output, intermediate], &down_bytes, GgmlType::Q6K);
            let gate_out = input.q_mat_mul(&gate).silu();
            let up_out = input.q_mat_mul(&up);
            (gate_out * up_out).q_mat_mul(&down).to_concrete()
        }
    })
    .arg(q_mat_mul_input_fuzz(
        1,
        [intermediate, hidden],
        834,
        Uniform::new(-0.25, 0.25).unwrap(),
    ))
    .compare_with(approx_compare::<2, f32>(5.0))
    .runs(3)
    .await
    .unwrap();
}

#[derive(Clone, Copy)]
enum PairedKind {
    SwiGLU,
    GeGLU,
    ReGLU,
}

impl PairedKind {
    fn cpu_activation(self, x: f32) -> f32 {
        match self {
            PairedKind::SwiGLU => x / (1.0 + (-x).exp()),
            PairedKind::GeGLU => {
                // tanh approximation matching the kernel-side helper
                0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh())
            }
            PairedKind::ReGLU => {
                if x > 0.0 {
                    x
                } else {
                    0.0
                }
            }
        }
    }
}

async fn paired_matches_cpu_for_rows(input_row_count: usize, kind: PairedKind) {
    use fusor::D;
    let ty = GgmlType::Q4K;
    let weight_shape = [4, 512];
    let raw_bytes = q4k_raw_bytes(weight_shape);
    let weights = QuantizedTensor::<BlockQ4K>::from_raw_bytes(weight_shape, &raw_bytes);
    let expected_weights = concrete_to_rows(&weights.dequantize::<2>(), weight_shape);

    // Author the natural unfused source — the resolver's paired-fusion rule
    // rewrites this into a single paired-mode QMatMul kernel. Correctness verifies
    // both the rewrite and the kernel's per-output epilogue evaluation.
    fusor_conformance::assert(move |input: Tensor<2, f32>| {
        let raw_bytes = raw_bytes.clone();
        async move {
            let weights = qmatrix_from_raw_bytes(&input.device(), weight_shape, &raw_bytes, ty);
            let projected = input.q_mat_mul(&weights);
            let gate = projected.narrow(D::Minus1, 0, 2).to_concrete();
            let up = projected.narrow(D::Minus1, 2, 2).to_concrete();
            let activated = match kind {
                PairedKind::SwiGLU => gate.silu(),
                PairedKind::GeGLU => gate.gelu(),
                PairedKind::ReGLU => gate.relu(),
            };
            (activated * up).to_concrete()
        }
    })
    .arg(q_mat_mul_input_fuzz(
        input_row_count,
        [2, weight_shape[1]],
        0x5A17_5516_6C75,
        Uniform::new(-0.25, 0.25).unwrap(),
    ))
    .equal_to(move |input: Tensor<2, f32>| {
        let expected_weights = expected_weights.clone();
        async move {
            let device = input.device();
            let input_values = input.as_slice().await.unwrap().to_vec2();
            let projected = matmul2(&input_values, &transpose2(&expected_weights));
            let expected = projected
                .iter()
                .map(|row| {
                    let gate0 = row[0];
                    let gate1 = row[1];
                    vec![
                        kind.cpu_activation(gate0) * row[2],
                        kind.cpu_activation(gate1) * row[3],
                    ]
                })
                .collect::<Vec<_>>();
            Tensor::new(&device, &expected)
        }
    })
    .compare_with(approx_compare::<2, f32>(2.0))
    .runs(3)
    .await
    .unwrap();
}

/// The fuser must collapse `rms_norm(...).relu()` (or any unary chain after
/// an RmsNorm) into a single RmsNorm kernel dispatch — the kernel applies
/// the chain in-register before the store. Without the rule, the unfused
/// source resolves to 2 dispatches.
#[tokio::test]
async fn rmsnorm_post_relu_resolves_to_single_kernel() {
    let Ok(device) = fusor::Device::new().await else {
        return;
    };
    let Some(gpu_device) = device.as_gpu() else {
        panic!("expected GPU device");
    };
    if !gpu_device.subgroup_kernels_supported() {
        return;
    }
    let cols = 64usize;
    let input_data = vec![vec![0.1f32; cols]; 4];
    let weight_data = vec![1.2f32; cols];
    let input: Tensor<2, f32> = Tensor::new(&device, &input_data);
    let weight: Tensor<1, f32> = Tensor::new(&device, &weight_data);

    let output = input
        .rms_norm_fused::<1, 1>(&weight, None, 1e-5)
        .relu()
        .to_concrete();
    let Tensor::Gpu(gpu_out) = output else {
        panic!("expected GPU tensor");
    };
    let kernels = gpu_device.resolve_batch(&[gpu_out.key()]);
    assert_eq!(
        kernels, 1,
        "expected fuser to collapse rms_norm -> relu to 1 dispatch, got {kernels}"
    );
}

/// The fuser must collapse `relu(input).q_mat_mul(weights)` into a single
/// QMatMul kernel — qgemv applies the activation to each loaded activation
/// tile before the dot product. Without the pre-fusion rule, the unfused
/// source resolves to 2 dispatches (nary + matmul).
#[tokio::test]
async fn q4k_qmatmul_pre_relu_resolves_to_single_kernel() {
    let Ok(device) = fusor::Device::new().await else {
        return;
    };
    let weight_shape = [4, 512];
    let raw_bytes = q4k_raw_bytes(weight_shape);
    let input_data = vec![vec![0.1f32; weight_shape[1]]; 1];
    let Some(gpu_device) = device.as_gpu() else {
        panic!("expected GPU device");
    };
    if !gpu_device.subgroup_kernels_supported() {
        return;
    }

    let weights = qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, GgmlType::Q4K);
    let input: Tensor<2, f32> = Tensor::new(&device, &input_data);
    let output = input.relu().to_concrete().q_mat_mul(&weights).to_concrete();
    let Tensor::Gpu(gpu_out) = output else {
        panic!("expected GPU tensor");
    };
    let kernels = gpu_device.resolve_batch(&[gpu_out.key()]);
    assert_eq!(
        kernels, 1,
        "expected fuser to collapse relu -> q_mat_mul to 1 dispatch, got {kernels}"
    );
}

/// The fuser must collapse `q_mat_mul → unary chain` (e.g. relu, silu)
/// into a single QMatMul kernel dispatch — qgemv kernels apply the chain
/// in-register before storing. Without the fuser rule, the unfused source
/// resolves to 2 dispatches (matmul + nary).
#[tokio::test]
async fn q4k_qmatmul_post_relu_resolves_to_single_kernel() {
    let Ok(device) = fusor::Device::new().await else {
        return; // No GPU available.
    };
    let weight_shape = [4, 512];
    let raw_bytes = q4k_raw_bytes(weight_shape);
    let input_data = vec![vec![0.1f32; weight_shape[1]]; 1];
    let Some(gpu_device) = device.as_gpu() else {
        panic!("expected GPU device");
    };
    if gpu_device.requires_host_fallbacks() {
        return;
    }

    let weights = qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, GgmlType::Q4K);
    let input: Tensor<2, f32> = Tensor::new(&device, &input_data);

    // Natural unfused source: matmul + relu. Fuser should collapse to 1 kernel.
    let output = input.q_mat_mul(&weights).relu().to_concrete();
    let Tensor::Gpu(natural_gpu) = output else {
        panic!("expected GPU tensor");
    };
    let natural_kernels = gpu_device.resolve_batch(&[natural_gpu.key()]);
    assert_eq!(
        natural_kernels, 1,
        "expected fuser to collapse q_mat_mul -> relu to 1 dispatch, got {natural_kernels}"
    );
}

#[tokio::test]
async fn q8_0_qmatmul_post_column_add_nonmultiple_applies_epilogue() {
    let weight_shape = [4, 64];
    let raw_bytes = q8_0_raw_bytes(weight_shape);
    let input_shape = [2, weight_shape[1]];
    let input_data = deterministic_input(&input_shape, 1_031);
    let bias_data = vec![0.25f32, -0.5, 0.75, -1.0];

    let cpu_weights =
        qmatrix_from_raw_bytes(&Device::Cpu, weight_shape, &raw_bytes, GgmlType::Q8_0);
    let cpu_input: Tensor<2, f32> = Tensor::from_slice(&Device::Cpu, input_shape, &input_data);
    let cpu_bias: Tensor<1, f32> = Tensor::from_slice(&Device::Cpu, [weight_shape[0]], &bias_data);
    let expected = cpu_input
        .q_mat_mul(&cpu_weights)
        .add_(&cpu_bias)
        .to_concrete();

    for device in available_devices().await {
        let weights = qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, GgmlType::Q8_0);
        let input: Tensor<2, f32> = Tensor::from_slice(&device, input_shape, &input_data);
        let bias: Tensor<1, f32> = Tensor::from_slice(&device, [weight_shape[0]], &bias_data);
        let actual = input.q_mat_mul(&weights).add_(&bias).to_concrete();
        approx_eq(&actual, &expected, 2.0).await.unwrap();
    }
}

#[tokio::test]
async fn q8_0_qmatmul_post_mixed_extras_preserves_binding_order() {
    let weight_shape = [4, 64];
    let raw_bytes = q8_0_raw_bytes(weight_shape);
    let input_shape = [2, weight_shape[1]];
    let output_shape = [2, weight_shape[0]];
    let input_data = deterministic_input(&input_shape, 1_047);
    let residual_data = deterministic_input(&output_shape, 1_211);
    let bias_data = vec![0.4f32, -0.2, 0.1, -0.6];

    let cpu_weights =
        qmatrix_from_raw_bytes(&Device::Cpu, weight_shape, &raw_bytes, GgmlType::Q8_0);
    let cpu_input: Tensor<2, f32> = Tensor::from_slice(&Device::Cpu, input_shape, &input_data);
    let cpu_residual: Tensor<2, f32> =
        Tensor::from_slice(&Device::Cpu, output_shape, &residual_data);
    let cpu_bias: Tensor<1, f32> = Tensor::from_slice(&Device::Cpu, [weight_shape[0]], &bias_data);
    let cpu_residual_biased = cpu_residual.add_(&cpu_bias);
    let expected = cpu_input
        .q_mat_mul(&cpu_weights)
        .add_(&cpu_residual_biased)
        .to_concrete();

    for device in available_devices().await {
        let weights = qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, GgmlType::Q8_0);
        let input: Tensor<2, f32> = Tensor::from_slice(&device, input_shape, &input_data);
        let residual: Tensor<2, f32> = Tensor::from_slice(&device, output_shape, &residual_data);
        let bias: Tensor<1, f32> = Tensor::from_slice(&device, [weight_shape[0]], &bias_data);
        let residual_biased = residual.add_(&bias);
        let actual = input
            .q_mat_mul(&weights)
            .add_(&residual_biased)
            .to_concrete();
        approx_eq(&actual, &expected, 2.0).await.unwrap();
    }
}

/// Biased FFN pattern: `silu(gate + gate_bias) * (up + up_bias)`.
/// The fuser detects this as a paired-with-extras pattern (2 matmul views +
/// 2 broadcast bias vectors) and rewrites it to a single `paired-mode QMatMul`
/// kernel that loads the biases per output column at epilogue time.
#[tokio::test]
async fn q4k_paired_with_bias_resolves_to_single_kernel() {
    use fusor::D;
    let Ok(device) = fusor::Device::new().await else {
        return;
    };
    let Some(gpu_device) = device.as_gpu() else {
        panic!("expected GPU device");
    };
    if !gpu_device.subgroup_kernels_supported() {
        return;
    }
    let weight_shape = [4, 512];
    let raw_bytes = q4k_raw_bytes(weight_shape);
    let input_data = vec![vec![0.1f32; weight_shape[1]]; 1];
    let weights = qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, GgmlType::Q4K);
    let input: Tensor<2, f32> = Tensor::new(&device, &input_data);
    let gate_bias: Tensor<1, f32> = Tensor::new(&device, &vec![0.05f32, -0.05]);
    let up_bias: Tensor<1, f32> = Tensor::new(&device, &vec![0.02f32, 0.03]);

    let projected = input.q_mat_mul(&weights);
    let gate = projected.narrow(D::Minus1, 0, 2).to_concrete();
    let up = projected.narrow(D::Minus1, 2, 2).to_concrete();
    let gate_biased = gate.add_(&gate_bias);
    let up_biased = up.add_(&up_bias);
    let output = (gate_biased.silu() * up_biased).to_concrete();

    let Tensor::Gpu(gpu_out) = output else {
        panic!("expected GPU tensor");
    };
    let kernels = gpu_device.resolve_batch(&[gpu_out.key()]);
    assert_eq!(
        kernels, 1,
        "expected fuser to collapse biased silu(gate+gb)*up_biased to 1 dispatch, got {kernels}"
    );
}

/// The fuser must collapse the natural `q_mat_mul → narrow → silu → mul(narrow)`
/// pattern into a single dispatch. Without the rule the source would emit
/// multiple kernels (matmul + nary).
#[tokio::test]
async fn q4k_paired_pattern_resolves_to_single_kernel() {
    use fusor::D;
    let Ok(device) = fusor::Device::new().await else {
        return; // No GPU available in this environment.
    };
    let weight_shape = [4, 512];
    let raw_bytes = q4k_raw_bytes(weight_shape);
    let input_data = vec![vec![0.1f32; weight_shape[1]]; 1];
    let Some(gpu_device) = device.as_gpu() else {
        panic!("expected GPU device");
    };
    if !gpu_device.subgroup_kernels_supported() {
        return;
    }

    let weights = qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, GgmlType::Q4K);
    let input: Tensor<2, f32> = Tensor::new(&device, &input_data);
    let projected = input.q_mat_mul(&weights);
    let gate = projected.narrow(D::Minus1, 0, 2).to_concrete();
    let up = projected.narrow(D::Minus1, 2, 2).to_concrete();
    let output = (gate.silu() * up).to_concrete();
    let Tensor::Gpu(natural_gpu) = output else {
        panic!("expected GPU tensor");
    };
    let natural_kernels = gpu_device.resolve_batch(&[natural_gpu.key()]);

    assert_eq!(
        natural_kernels, 1,
        "expected fuser to collapse paired pattern to 1 dispatch, got {natural_kernels}"
    );
}

/// Authoring the natural source (`q_mat_mul → narrow → silu → mul(narrow)`)
/// should produce results identical to the CPU reference, because the
/// compute-graph fuser rewrites the pattern to a `paired-mode QMatMul` kernel.
#[tokio::test]
async fn q4k_paired_pattern_auto_fuses_to_paired_kernel() {
    use fusor::D;
    let ty = GgmlType::Q4K;
    let weight_shape = [4, 512];
    let raw_bytes = q4k_raw_bytes(weight_shape);
    let weights = QuantizedTensor::<BlockQ4K>::from_raw_bytes(weight_shape, &raw_bytes);
    let expected_weights = concrete_to_rows(&weights.dequantize::<2>(), weight_shape);

    fusor_conformance::assert(move |input: Tensor<2, f32>| {
        let raw_bytes = raw_bytes.clone();
        async move {
            let weights = qmatrix_from_raw_bytes(&input.device(), weight_shape, &raw_bytes, ty);
            // Natural unfused authoring — the resolver's `try_fuse_paired_qmatmul`
            // rule rewrites this subgraph to a single paired-fused kernel.
            let projected = input.q_mat_mul(&weights);
            let gate = projected.narrow(D::Minus1, 0, 2).to_concrete();
            let up = projected.narrow(D::Minus1, 2, 2).to_concrete();
            (gate.silu() * up).to_concrete()
        }
    })
    .arg(q_mat_mul_input_fuzz(
        1,
        [2, weight_shape[1]],
        0x5A17_5516_6C76,
        Uniform::new(-0.25, 0.25).unwrap(),
    ))
    .equal_to(move |input: Tensor<2, f32>| {
        let expected_weights = expected_weights.clone();
        async move {
            let device = input.device();
            let input_values = input.as_slice().await.unwrap().to_vec2();
            let projected = matmul2(&input_values, &transpose2(&expected_weights));
            let expected = projected
                .iter()
                .map(|row| {
                    let gate0 = row[0];
                    let gate1 = row[1];
                    let silu = |x: f32| x / (1.0 + (-x).exp());
                    vec![silu(gate0) * row[2], silu(gate1) * row[3]]
                })
                .collect::<Vec<_>>();
            Tensor::new(&device, &expected)
        }
    })
    .compare_with(approx_compare::<2, f32>(2.0))
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

macro_rules! quantized_fixture_cases {
    ($($fn_name:ident: $block:ty, $ty:expr, $shape:expr, $raw_bytes_fn:ident, $rows:expr, $deq_tol:expr, $q_tol:expr, $seed:expr, $dequantize:expr;)*) => {
        $(
            fn $fn_name() -> QuantizedFixture {
                let shape = $shape;
                build_fixture::<$block>($ty, shape, $raw_bytes_fn(shape), $rows, $deq_tol, $q_tol)
            }
        )*

        const QUANTIZED_FIXTURE_CASES: &[(fn() -> QuantizedFixture, u64, bool)] = &[
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

#[tokio::test]
async fn quantized_dequantize_matches_cpu_reference() {
    for &(fixture, _, _) in QUANTIZED_FIXTURE_CASES
        .iter()
        .filter(|&&(_, _, dequantize)| dequantize)
    {
        let QuantizedFixture {
            ty,
            weight_shape,
            raw_bytes,
            dequantized,
            dequantize_tol,
            ..
        } = fixture();
        assert_dequantize_matches_host_reference(
            ty,
            weight_shape,
            raw_bytes,
            dequantized,
            dequantize_tol,
        )
        .await;
    }
}

#[tokio::test]
async fn quantized_q_mat_mul_matches_cpu_reference() {
    for &(fixture, seed, _) in QUANTIZED_FIXTURE_CASES {
        let fixture = fixture();
        assert_q_mat_mul_matches_host_reference(
            &fixture,
            QMatMulFuzz {
                seed,
                distribution: Uniform::new(-0.25, 0.25).unwrap(),
            },
        )
        .await;
    }
}

#[tokio::test]
async fn q8_0_dequantize_then_add_matches_cpu_reference() {
    let QuantizedFixture {
        ty,
        weight_shape,
        raw_bytes,
        dequantized,
        ..
    } = q8_0_fixture();
    let expected = dequantized
        .iter()
        .map(|row| row.iter().map(|value| value + 1.25).collect::<Vec<_>>())
        .collect::<Vec<_>>();

    fusor_conformance::assert(move |device: Device| {
        let raw_bytes = raw_bytes.clone();
        async move {
            (qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, ty).dequantize::<2>() + 1.25)
                .to_concrete()
        }
    })
    .arg(|device: &Device| device.clone())
    .equal_to(move |device: Device| {
        let expected = expected.clone();
        async move { Tensor::new(&device, &expected) }
    })
    .compare_with(approx_compare::<2, f32>(1e-5))
    .await
    .unwrap();
}

#[tokio::test]
async fn q5_0_q_mat_mul_single_row_splits_large_qgemv_dispatch() {
    const Q5_0_QGEMV_COLS_PER_WORKGROUP: usize = 8;
    const QMATMUL_MAX_WORKGROUPS_PER_DIMENSION: usize = 1_024;

    let mut exercised_subgroup_gpu = false;
    for device in available_devices().await {
        let Some(gpu) = device.as_gpu() else {
            continue;
        };
        if !gpu.subgroup_kernels_supported() {
            continue;
        }
        exercised_subgroup_gpu = true;

        let max_workgroups = (gpu.limits().max_compute_workgroups_per_dimension as usize)
            .min(QMATMUL_MAX_WORKGROUPS_PER_DIMENSION);
        let output_cols = max_workgroups * Q5_0_QGEMV_COLS_PER_WORKGROUP + 1;
        let weight_shape = [output_cols, BlockQ5_0::BLOCK_SIZE];
        let raw_bytes =
            vec![0u8; block_count(weight_shape, BlockQ5_0::BLOCK_SIZE) * size_of::<BlockQ5_0>()];
        let input_values = vec![0.25f32; weight_shape[1]];
        let weights = qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, GgmlType::Q5_0);
        let input: Tensor<2, f32> =
            Tensor::from_slice(&device, [1, weight_shape[1]], &input_values);

        let result = input.q_mat_mul(&weights).as_slice().await.unwrap();

        assert_eq!(result.shape(), &[1, output_cols]);
        assert!(
            result.as_slice().iter().all(|value| *value == 0.0),
            "zero Q5_0 weights should produce zero qgemv output"
        );
    }

    if !exercised_subgroup_gpu {
        return;
    }
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
async fn q_mat_mul_transposed_input_matches_host_reference() {
    use fusor::Device;

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
async fn q_mat_mul_consumes_transpose_reshape_copy_matches_cpu_reference() {
    let weight_shape = [4usize, 4096usize];
    let raw_bytes = q8_0_raw_bytes(weight_shape);
    let input_shape = [1usize, 32usize, 2usize, 128usize];
    let data = deterministic_input(&input_shape, 1401);

    let cpu_weights =
        qmatrix_from_raw_bytes(&Device::Cpu, weight_shape, &raw_bytes, GgmlType::Q8_0);
    let cpu_input: Tensor<4, f32> = Tensor::from_slice(&Device::Cpu, input_shape, &data);
    let produced = cpu_input + 0.25;
    let transposed = produced.transpose(1, 2);
    let reshaped = transposed.reshape([1, 2, 32 * 128]);
    let cpu_result = reshaped.q_mat_mul(&cpu_weights).to_concrete();

    for device in available_devices().await {
        let weights = qmatrix_from_raw_bytes(&device, weight_shape, &raw_bytes, GgmlType::Q8_0);
        let input: Tensor<4, f32> = Tensor::from_slice(&device, input_shape, &data);
        let produced = input + 0.25;
        let transposed = produced.transpose(1, 2);
        let reshaped = transposed.reshape([1, 2, 32 * 128]);
        let actual = reshaped.q_mat_mul(&weights).to_concrete();
        fusor_conformance::approx_eq(&actual, &cpu_result, 5e-2)
            .await
            .unwrap();
    }
}

#[tokio::test]
async fn q4k_llama_decode_transpose_reshape_qmatmul_matches_one_hot_reference() {
    let Some(device) = available_devices()
        .await
        .into_iter()
        .find(|device| device.as_gpu().is_some())
    else {
        return;
    };
    let Some(gpu_device) = device.as_gpu() else {
        return;
    };
    if !gpu_device.subgroup_kernels_supported() {
        return;
    }

    for (weight_shape, sample_cols) in [
        (
            [5120usize, 4096usize],
            &[0usize, 1, 63, 64, 511, 1024, 4095, 5119][..],
        ),
        (
            [14336usize, 4096usize],
            &[0usize, 1, 63, 64, 511, 1024, 4095, 8191, 14335][..],
        ),
    ] {
        assert_q4k_llama_decode_transpose_reshape_shape(&device, weight_shape, sample_cols).await;
    }
}

async fn assert_q4k_llama_decode_transpose_reshape_shape(
    device: &Device,
    weight_shape: [usize; 2],
    sample_cols: &[usize],
) {
    let [output_cols, hidden] = weight_shape;
    let input_shape = [1usize, 32usize, 48usize, 128usize];
    assert_eq!(hidden, input_shape[1] * input_shape[3]);
    let selected_k = 777usize;
    let selected_head = selected_k / input_shape[3];
    let selected_dim = selected_k % input_shape[3];
    let selected_block_in_row = selected_k / BlockQ4K::BLOCK_SIZE;
    let selected_offset = selected_k % BlockQ4K::BLOCK_SIZE;
    let blocks_per_row = hidden / BlockQ4K::BLOCK_SIZE;
    let raw_bytes = q4k_raw_bytes(weight_shape);
    let weights = qmatrix_from_raw_bytes(device, weight_shape, &raw_bytes, GgmlType::Q4K);

    let mut input_data = vec![-0.25f32; input_shape.iter().product()];
    let mut row_values = Vec::with_capacity(input_shape[2]);
    for row in 0..input_shape[2] {
        let row_value = 0.125 + row as f32 * 0.01;
        row_values.push(row_value);
        let index = ((selected_head * input_shape[2] + row) * input_shape[3]) + selected_dim;
        input_data[index] = row_value - 0.25;
    }

    let input: Tensor<4, f32> = Tensor::from_slice(device, input_shape, &input_data);
    let actual = (input + 0.25)
        .transpose(1, 2)
        .reshape([1, input_shape[2], hidden])
        .q_mat_mul(&weights)
        .as_slice()
        .await
        .unwrap();

    assert_eq!(actual.shape(), &[1, input_shape[2], output_cols]);
    for row in [0usize, 1, 7, 17, 31, 47] {
        for &col in sample_cols {
            let block_index = col * blocks_per_row + selected_block_in_row;
            let offset = block_index * size_of::<BlockQ4K>();
            assert!(offset + size_of::<BlockQ4K>() <= raw_bytes.len());
            let block = unsafe {
                std::ptr::read_unaligned(raw_bytes.as_ptr().add(offset).cast::<BlockQ4K>())
            };
            let expected = row_values[row] * block.dequantize().as_ref()[selected_offset];
            let actual = actual[[0, row, col]];
            let tolerance = 1e-2_f32.max(expected.abs() * 1.0e-4);
            assert!(
                (actual - expected).abs() <= tolerance,
                "shape={weight_shape:?} row={row} col={col} actual={actual} expected={expected} tolerance={tolerance}"
            );
        }
    }
}

#[tokio::test]
async fn q_mat_mul_batched_matches_unbatched_property() {
    // Batched 3D q_mat_mul produces the same per-batch slice as 2D q_mat_mul
    // applied independently. Replaces
    // `cpu/src/quantized.rs::test_batched_q_mat_mul_matches_unbatched`.
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
