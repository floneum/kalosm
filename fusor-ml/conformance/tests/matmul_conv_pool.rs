mod common;

use common::{conv1d_ncw, matmul2, pool1d_ncw};
use fusor::{Device, Tensor};
use fusor_conformance::{FuzzGenerator, approx_compare};
use rand::distr::Uniform;

#[tokio::test]
async fn matmul_match_host_reference() {
    const M: usize = 64;
    const K: usize = 128;
    const N: usize = 64;

    let gen_lhs = FuzzGenerator::<2, f32>::new([M, K])
        .with_seed(300)
        .with_distribution(Uniform::new(-3.0, 3.0).unwrap());
    let gen_rhs = FuzzGenerator::<2, f32>::new([K, N])
        .with_seed(301)
        .with_distribution(Uniform::new(-3.0, 3.0).unwrap());

    // matmul vs host reference
    fusor_conformance::assert(async |a: Tensor<2, f32>, b: Tensor<2, f32>| a.matmul(&b))
        .arg(gen_lhs.clone())
        .arg(gen_rhs.clone())
        .equal_to_resolved_with_device(
            async |a: Vec<Vec<f32>>, b: Vec<Vec<f32>>, device: Device| {
                Tensor::new(&device, &matmul2(&a, &b))
            },
        )
        .compare_with(approx_compare::<2, f32>(1e-2))
        .runs(3)
        .await
        .unwrap();
}

#[tokio::test]
async fn matmul_small_fixed_regression() {
    const LHS: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    const RHS: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    fusor_conformance::assert(async |a: Tensor<2, f32>, b: Tensor<2, f32>| a.matmul(&b))
        .arg(move |device: &Device| Tensor::from_slice(device, [2, 3], &LHS))
        .arg(move |device: &Device| Tensor::from_slice(device, [3, 2], &RHS))
        .equal_to_resolved_with_device(
            async |_a: Vec<Vec<f32>>, _b: Vec<Vec<f32>>, device: Device| {
                Tensor::new(&device, &[[22.0f32, 28.0], [49.0, 64.0]])
            },
        )
        .compare_with(approx_compare::<2, f32>(1e-6))
        .runs(1)
        .await
        .unwrap();
}

#[tokio::test]
async fn conv_and_pool_match_host_reference() {
    // Conv1D with fuzzed input
    let gen_conv = FuzzGenerator::<3, f32>::new([1..=1, 1..=1, 255..=257])
        .with_seed(310)
        .with_distribution(Uniform::new(-3.0, 3.0).unwrap());

    static CONV_WEIGHT: &[f32] = &[0.25, -0.5, 1.0];
    static CONV_BIAS: &[f32] = &[0.1];

    fusor_conformance::assert(async |input: Tensor<3, f32>| {
        let weight = Tensor::from_slice(&input.device(), [1, 1, 3], CONV_WEIGHT);
        let bias = Tensor::from_slice(&input.device(), [1], CONV_BIAS);
        input.conv(&weight, Some(&bias), [1], [2])
    })
    .arg(gen_conv)
    .equal_to_resolved_with_device(async |v: Vec<Vec<Vec<f32>>>, device: Device| {
        let expected = conv1d_ncw(&v, &[vec![vec![0.25, -0.5, 1.0]]], Some(&[0.1]), 1, 2);
        Tensor::new(&device, &expected)
    })
    .compare_with(approx_compare::<3, f32>(1e-4))
    .runs(3)
    .await
    .unwrap();

    // Pool: pool_max with fuzzed input
    let gen_pool = FuzzGenerator::<3, f32>::new([1..=1, 2..=2, 255..=257])
        .with_seed(320)
        .with_distribution(Uniform::new(-4.0, 12.0).unwrap());

    // pool_max vs host reference
    fusor_conformance::assert(async |x: Tensor<3, f32>| x.pool_max([(2, 2)]))
        .arg(gen_pool.clone())
        .equal_to_resolved_with_device(async |v: Vec<Vec<Vec<f32>>>, device: Device| {
            Tensor::new(&device, &pool1d_ncw(&v, 2, 2, f32::max, f32::NEG_INFINITY))
        })
        .compare_with(approx_compare::<3, f32>(1e-6))
        .runs(3)
        .await
        .unwrap();

    // pool_min vs host reference
    fusor_conformance::assert(async |x: Tensor<3, f32>| x.pool_min([(2, 2)]))
        .arg(gen_pool)
        .equal_to_resolved_with_device(async |v: Vec<Vec<Vec<f32>>>, device: Device| {
            Tensor::new(&device, &pool1d_ncw(&v, 2, 2, f32::min, f32::INFINITY))
        })
        .compare_with(approx_compare::<3, f32>(1e-6))
        .runs(3)
        .await
        .unwrap();
}

#[tokio::test]
async fn matmul_identity_matrix() {
    const M: usize = 2;
    const N: usize = 3;
    let fuzz = FuzzGenerator::<2, f32>::new([M, N])
        .with_seed(330)
        .with_distribution(Uniform::new(-5.0, 5.0).unwrap());

    // A @ I = A
    fusor_conformance::assert(async |a: Tensor<2, f32>| {
        let device = a.device();
        let identity = Tensor::from_slice(
            &device,
            [N, N],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        );
        a.matmul(&identity)
    })
    .arg(fuzz)
    .equal_to(async |a: Tensor<2, f32>| a)
    .compare_with(approx_compare::<2, f32>(1e-5))
    .runs(3)
    .await
    .unwrap();
}

fn batched3_matmul(lhs: &[Vec<Vec<f32>>], rhs: &[Vec<Vec<f32>>]) -> Vec<Vec<Vec<f32>>> {
    assert_eq!(lhs.len(), rhs.len(), "batch dim mismatch");
    lhs.iter()
        .zip(rhs.iter())
        .map(|(a, b)| matmul2(a, b))
        .collect()
}

#[tokio::test]
async fn matmul_batched_3d_matches_host_reference() {
    // Per-batch [M, K] @ [K, N] -> [M, N]. Replaces the deleted
    // `core/src/matmul/mod.rs::test_batched_matmul` and `fuzz_batched_matmul`.
    let gen_lhs = FuzzGenerator::<3, f32>::new([2, 3, 4])
        .with_seed(360)
        .with_distribution(Uniform::new(-2.0, 2.0).unwrap());
    let gen_rhs = FuzzGenerator::<3, f32>::new([2, 4, 5])
        .with_seed(361)
        .with_distribution(Uniform::new(-2.0, 2.0).unwrap());

    fusor_conformance::assert(async |a: Tensor<3, f32>, b: Tensor<3, f32>| a.matmul(&b))
        .arg(gen_lhs)
        .arg(gen_rhs)
        .equal_to_resolved_with_device(
            async |a: Vec<Vec<Vec<f32>>>, b: Vec<Vec<Vec<f32>>>, device: Device| {
                Tensor::new(&device, &batched3_matmul(&a, &b))
            },
        )
        .compare_with(approx_compare::<3, f32>(1e-3))
        .runs(3)
        .await
        .unwrap();
}

#[tokio::test]
async fn matmul_attention_4d_matches_host_reference() {
    // Attention-shaped 4D matmul: [B, H, M, K] @ [B, H, K, N] regression
    // for the deleted `fusor/src/lib.rs::test_matmul_cpu_vs_gpu`. Smaller than
    // the original [1, 8, 100, 64] to keep CI fast — the original was a
    // float-precision smoke test, not a timing regression.
    use fusor_conformance::available_devices;
    const SHAPE: [usize; 4] = [1, 2, 16, 16];

    fn data(seed: u32) -> Vec<f32> {
        let total: usize = SHAPE.iter().product();
        (0..total)
            .map(|i| {
                let v = ((i + seed as usize) % 31) as f32;
                (v - 15.0) * 0.05
            })
            .collect()
    }

    let lhs_data = data(0);
    let rhs_data = data(7);

    let cpu_lhs = Tensor::from_slice(&Device::Cpu, SHAPE, &lhs_data);
    let cpu_rhs = Tensor::from_slice(&Device::Cpu, SHAPE, &rhs_data);
    let expected = cpu_lhs.matmul(&cpu_rhs).to_concrete();

    for device in available_devices().await {
        let lhs = Tensor::from_slice(&device, SHAPE, &lhs_data);
        let rhs = Tensor::from_slice(&device, SHAPE, &rhs_data);
        let actual = lhs.matmul(&rhs).to_concrete();
        fusor_conformance::approx_eq(&actual, &expected, 1e-3)
            .await
            .unwrap();
    }
}

#[tokio::test]
async fn matmul_attention_batched_skinny_m_4d_matches_host_reference() {
    use fusor_conformance::available_devices;
    const LHS_SHAPE: [usize; 4] = [2, 4, 1, 16];
    const RHS_SHAPE: [usize; 4] = [2, 4, 16, 8];

    let lhs_data = (0..LHS_SHAPE.iter().product::<usize>())
        .map(|i| (((i % 19) as f32) - 9.0) * 0.07)
        .collect::<Vec<_>>();
    let rhs_data = (0..RHS_SHAPE.iter().product::<usize>())
        .map(|i| (((i % 23) as f32) - 11.0) * 0.05)
        .collect::<Vec<_>>();

    let cpu_lhs = Tensor::from_slice(&Device::Cpu, LHS_SHAPE, &lhs_data);
    let cpu_rhs = Tensor::from_slice(&Device::Cpu, RHS_SHAPE, &rhs_data);
    let expected = cpu_lhs.matmul(&cpu_rhs).to_concrete();

    for device in available_devices().await {
        let lhs = Tensor::from_slice(&device, LHS_SHAPE, &lhs_data);
        let rhs = Tensor::from_slice(&device, RHS_SHAPE, &rhs_data);
        let actual = lhs.matmul(&rhs).to_concrete();
        fusor_conformance::approx_eq(&actual, &expected, 1e-3)
            .await
            .unwrap();
    }
}

#[tokio::test]
async fn matmul_sgemv_variants_match_host_reference() {
    // [M, K] @ [K, 1] -> [M, 1] : single-output gemv
    let gen_mat = FuzzGenerator::<2, f32>::new([8, 12])
        .with_seed(370)
        .with_distribution(Uniform::new(-2.0, 2.0).unwrap());
    let gen_vec = FuzzGenerator::<2, f32>::new([12, 1])
        .with_seed(371)
        .with_distribution(Uniform::new(-2.0, 2.0).unwrap());

    fusor_conformance::assert(async |a: Tensor<2, f32>, b: Tensor<2, f32>| a.matmul(&b))
        .arg(gen_mat)
        .arg(gen_vec)
        .equal_to_resolved_with_device(
            async |a: Vec<Vec<f32>>, b: Vec<Vec<f32>>, device: Device| {
                Tensor::new(&device, &matmul2(&a, &b))
            },
        )
        .compare_with(approx_compare::<2, f32>(1e-4))
        .runs(3)
        .await
        .unwrap();

    // [M, K] @ [K, N] with N>1 multi-row variant — distinct GPU kernel path
    let gen_mat = FuzzGenerator::<2, f32>::new([8, 12])
        .with_seed(372)
        .with_distribution(Uniform::new(-2.0, 2.0).unwrap());
    let gen_rhs = FuzzGenerator::<2, f32>::new([12, 4])
        .with_seed(373)
        .with_distribution(Uniform::new(-2.0, 2.0).unwrap());

    fusor_conformance::assert(async |a: Tensor<2, f32>, b: Tensor<2, f32>| a.matmul(&b))
        .arg(gen_mat)
        .arg(gen_rhs)
        .equal_to_resolved_with_device(
            async |a: Vec<Vec<f32>>, b: Vec<Vec<f32>>, device: Device| {
                Tensor::new(&device, &matmul2(&a, &b))
            },
        )
        .compare_with(approx_compare::<2, f32>(1e-4))
        .runs(3)
        .await
        .unwrap();

    // Batched gemv: [B, M, K] @ [B, K, 1]
    let gen_mat = FuzzGenerator::<3, f32>::new([2, 6, 9])
        .with_seed(374)
        .with_distribution(Uniform::new(-2.0, 2.0).unwrap());
    let gen_vec = FuzzGenerator::<3, f32>::new([2, 9, 1])
        .with_seed(375)
        .with_distribution(Uniform::new(-2.0, 2.0).unwrap());

    fusor_conformance::assert(async |a: Tensor<3, f32>, b: Tensor<3, f32>| a.matmul(&b))
        .arg(gen_mat)
        .arg(gen_vec)
        .equal_to_resolved_with_device(
            async |a: Vec<Vec<Vec<f32>>>, b: Vec<Vec<Vec<f32>>>, device: Device| {
                Tensor::new(&device, &batched3_matmul(&a, &b))
            },
        )
        .compare_with(approx_compare::<3, f32>(1e-4))
        .runs(3)
        .await
        .unwrap();
}

#[tokio::test]
async fn matmul_transposed_operand_matches_host_reference() {
    // matmul where the right operand is the lazy transpose of a contiguous tensor.
    // Replaces the deleted `core/src/matmul/mod.rs::test_transposed_matmul`.
    let gen_lhs = FuzzGenerator::<2, f32>::new([6, 8])
        .with_seed(380)
        .with_distribution(Uniform::new(-1.0, 1.0).unwrap());
    let gen_rhs_pre_t = FuzzGenerator::<2, f32>::new([5, 8])
        .with_seed(381)
        .with_distribution(Uniform::new(-1.0, 1.0).unwrap());

    fusor_conformance::assert(async |a: Tensor<2, f32>, b: Tensor<2, f32>| {
        // Transpose b so the matmul sees a non-contiguous strided operand.
        let b_t = b.transpose(0, 1).to_concrete();
        a.matmul(&b_t)
    })
    .arg(gen_lhs)
    .arg(gen_rhs_pre_t)
    .equal_to_resolved_with_device(async |a: Vec<Vec<f32>>, b: Vec<Vec<f32>>, device: Device| {
        // Transpose the host reference too.
        let b_t: Vec<Vec<f32>> = (0..b[0].len())
            .map(|col| b.iter().map(|row| row[col]).collect())
            .collect();
        Tensor::new(&device, &matmul2(&a, &b_t))
    })
    .compare_with(approx_compare::<2, f32>(1e-4))
    .runs(3)
    .await
    .unwrap();
}

#[tokio::test]
async fn matmul_non_contiguous_input_matches_host_reference() {
    // matmul on a sliced (non-contiguous) input. Replaces
    // `core/src/matmul/mod.rs::test_matrix_vector_mul_non_contiguous`.
    let gen_lhs_padded = FuzzGenerator::<2, f32>::new([6, 12])
        .with_seed(390)
        .with_distribution(Uniform::new(-1.0, 1.0).unwrap());
    let gen_rhs = FuzzGenerator::<2, f32>::new([8, 4])
        .with_seed(391)
        .with_distribution(Uniform::new(-1.0, 1.0).unwrap());

    fusor_conformance::assert(async |a: Tensor<2, f32>, b: Tensor<2, f32>| {
        let a_slice = a.slice([0..6, 2..10]).to_concrete();
        a_slice.matmul(&b)
    })
    .arg(gen_lhs_padded)
    .arg(gen_rhs)
    .equal_to_resolved_with_device(async |a: Vec<Vec<f32>>, b: Vec<Vec<f32>>, device: Device| {
        let a_slice: Vec<Vec<f32>> = a.iter().map(|row| row[2..10].to_vec()).collect();
        Tensor::new(&device, &matmul2(&a_slice, &b))
    })
    .compare_with(approx_compare::<2, f32>(1e-4))
    .runs(3)
    .await
    .unwrap();
}

#[tokio::test]
async fn matmul_large_fuzzed() {
    const M: usize = 256;
    const K: usize = 256;
    const N: usize = 256;

    let gen_lhs = FuzzGenerator::<2, f32>::new([M, K])
        .with_seed(100)
        .with_distribution(Uniform::new(-2.0, 2.0).unwrap());
    let gen_rhs = FuzzGenerator::<2, f32>::new([K, N])
        .with_seed(101)
        .with_distribution(Uniform::new(-2.0, 2.0).unwrap());

    fusor_conformance::assert(async |a: Tensor<2, f32>, b: Tensor<2, f32>| a.matmul(&b))
        .arg(gen_lhs)
        .arg(gen_rhs)
        .equal_to_resolved_with_device(
            async |a: Vec<Vec<f32>>, b: Vec<Vec<f32>>, device: Device| {
                let expected = matmul2(&a, &b);
                Tensor::new(&device, &expected)
            },
        )
        .compare_with(approx_compare::<2, f32>(1e-2))
        .runs(3)
        .await
        .unwrap();
}
