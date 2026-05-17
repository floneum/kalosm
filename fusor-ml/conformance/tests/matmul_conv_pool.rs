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

fn conv2d_nchw_ref(
    input: &[Vec<Vec<Vec<f32>>>],
    weight: &[Vec<Vec<Vec<f32>>>],
    bias: Option<&[f32]>,
    padding: [usize; 2],
    stride: [usize; 2],
) -> Vec<Vec<Vec<Vec<f32>>>> {
    let batch = input.len();
    let in_ch = input[0].len();
    let in_h = input[0][0].len();
    let in_w = input[0][0][0].len();
    let out_ch = weight.len();
    let kh = weight[0][0].len();
    let kw = weight[0][0][0].len();
    let out_h = (in_h + 2 * padding[0] - kh) / stride[0] + 1;
    let out_w = (in_w + 2 * padding[1] - kw) / stride[1] + 1;
    let mut out = vec![vec![vec![vec![0.0f32; out_w]; out_h]; out_ch]; batch];
    for b in 0..batch {
        for oc in 0..out_ch {
            let b0 = bias.map_or(0.0, |bs| bs[oc]);
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut acc = b0;
                    for ic in 0..in_ch {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy = oh * stride[0] + ky;
                                let ix = ow * stride[1] + kx;
                                if iy >= padding[0]
                                    && iy < padding[0] + in_h
                                    && ix >= padding[1]
                                    && ix < padding[1] + in_w
                                {
                                    let v = input[b][ic][iy - padding[0]][ix - padding[1]];
                                    acc += v * weight[oc][ic][ky][kx];
                                }
                            }
                        }
                    }
                    out[b][oc][oh][ow] = acc;
                }
            }
        }
    }
    out
}

fn conv3d_ncdhw_ref(
    input: &[Vec<Vec<Vec<Vec<f32>>>>],
    weight: &[Vec<Vec<Vec<Vec<f32>>>>],
    bias: Option<&[f32]>,
    padding: [usize; 3],
    stride: [usize; 3],
) -> Vec<Vec<Vec<Vec<Vec<f32>>>>> {
    let batch = input.len();
    let in_ch = input[0].len();
    let in_d = input[0][0].len();
    let in_h = input[0][0][0].len();
    let in_w = input[0][0][0][0].len();
    let out_ch = weight.len();
    let kd = weight[0][0].len();
    let kh = weight[0][0][0].len();
    let kw = weight[0][0][0][0].len();
    let out_d = (in_d + 2 * padding[0] - kd) / stride[0] + 1;
    let out_h = (in_h + 2 * padding[1] - kh) / stride[1] + 1;
    let out_w = (in_w + 2 * padding[2] - kw) / stride[2] + 1;
    let mut out = vec![vec![vec![vec![vec![0.0f32; out_w]; out_h]; out_d]; out_ch]; batch];
    for b in 0..batch {
        for oc in 0..out_ch {
            let b0 = bias.map_or(0.0, |bs| bs[oc]);
            for od in 0..out_d {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut acc = b0;
                        for ic in 0..in_ch {
                            for kz in 0..kd {
                                for ky in 0..kh {
                                    for kx in 0..kw {
                                        let iz = od * stride[0] + kz;
                                        let iy = oh * stride[1] + ky;
                                        let ix = ow * stride[2] + kx;
                                        if iz >= padding[0]
                                            && iz < padding[0] + in_d
                                            && iy >= padding[1]
                                            && iy < padding[1] + in_h
                                            && ix >= padding[2]
                                            && ix < padding[2] + in_w
                                        {
                                            let v = input[b][ic][iz - padding[0]]
                                                [iy - padding[1]][ix - padding[2]];
                                            acc += v * weight[oc][ic][kz][ky][kx];
                                        }
                                    }
                                }
                            }
                        }
                        out[b][oc][od][oh][ow] = acc;
                    }
                }
            }
        }
    }
    out
}

fn input_data(total: usize, seed: u32) -> Vec<f32> {
    (0..total)
        .map(|i| (((i + seed as usize) % 23) as f32 - 11.0) * 0.13)
        .collect()
}

fn vec4_from_flat(flat: &[f32], shape: [usize; 4]) -> Vec<Vec<Vec<Vec<f32>>>> {
    let mut out = vec![vec![vec![vec![0.0f32; shape[3]]; shape[2]]; shape[1]]; shape[0]];
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                for l in 0..shape[3] {
                    out[i][j][k][l] = flat
                        [((i * shape[1] + j) * shape[2] + k) * shape[3] + l];
                }
            }
        }
    }
    out
}

fn vec5_from_flat(flat: &[f32], shape: [usize; 5]) -> Vec<Vec<Vec<Vec<Vec<f32>>>>> {
    let mut out =
        vec![vec![vec![vec![vec![0.0f32; shape[4]]; shape[3]]; shape[2]]; shape[1]]; shape[0]];
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            for k in 0..shape[2] {
                for l in 0..shape[3] {
                    for m in 0..shape[4] {
                        let idx = (((i * shape[1] + j) * shape[2] + k) * shape[3] + l)
                            * shape[4]
                            + m;
                        out[i][j][k][l][m] = flat[idx];
                    }
                }
            }
        }
    }
    out
}

fn flatten4(v: &[Vec<Vec<Vec<f32>>>]) -> Vec<f32> {
    v.iter()
        .flat_map(|a| a.iter().flat_map(|b| b.iter().flat_map(|c| c.iter().copied())))
        .collect()
}

fn flatten5(v: &[Vec<Vec<Vec<Vec<f32>>>>]) -> Vec<f32> {
    v.iter()
        .flat_map(|a| {
            a.iter().flat_map(|b| {
                b.iter()
                    .flat_map(|c| c.iter().flat_map(|d| d.iter().copied()))
            })
        })
        .collect()
}

#[tokio::test]
async fn conv2d_matches_host_reference() {
    use fusor_conformance::available_devices;
    const BATCH: usize = 1;
    const IN_CH: usize = 3;
    const OUT_CH: usize = 4;
    const H: usize = 8;
    const W: usize = 8;
    const KH: usize = 3;
    const KW: usize = 3;
    let input_flat = input_data(BATCH * IN_CH * H * W, 340);
    let weight_flat: Vec<f32> = (0..OUT_CH * IN_CH * KH * KW)
        .map(|i| ((i as i32 % 17) - 8) as f32 * 0.05)
        .collect();
    let bias_flat: Vec<f32> = [0.1, -0.2, 0.3, -0.4].to_vec();

    let input_nested = vec4_from_flat(&input_flat, [BATCH, IN_CH, H, W]);
    let weight_nested = vec4_from_flat(&weight_flat, [OUT_CH, IN_CH, KH, KW]);
    let expected_nested =
        conv2d_nchw_ref(&input_nested, &weight_nested, Some(&bias_flat), [1, 1], [1, 1]);
    let expected_flat = flatten4(&expected_nested);
    let out_shape = [BATCH, OUT_CH, H, W];

    for device in available_devices().await {
        let input = Tensor::from_slice(&device, [BATCH, IN_CH, H, W], &input_flat);
        let weight = Tensor::from_slice(&device, [OUT_CH, IN_CH, KH, KW], &weight_flat);
        let bias = Tensor::from_slice(&device, [OUT_CH], &bias_flat);
        let actual = input.conv(&weight, Some(&bias), [1, 1], [1, 1]).to_concrete();
        let expected = Tensor::from_slice(&device, out_shape, &expected_flat);
        fusor_conformance::approx_eq(&actual, &expected, 1e-3)
            .await
            .unwrap();
    }
}

#[tokio::test]
async fn conv3d_matches_host_reference() {
    use fusor_conformance::available_devices;
    const BATCH: usize = 1;
    const IN_CH: usize = 2;
    const OUT_CH: usize = 2;
    const DD: usize = 4;
    const H: usize = 4;
    const W: usize = 4;
    const KD: usize = 3;
    const KH: usize = 3;
    const KW: usize = 3;
    let input_flat = input_data(BATCH * IN_CH * DD * H * W, 350);
    let weight_flat: Vec<f32> = (0..OUT_CH * IN_CH * KD * KH * KW)
        .map(|i| ((i as i32 % 13) - 6) as f32 * 0.07)
        .collect();
    let bias_flat: Vec<f32> = [0.05, -0.05].to_vec();

    let input_nested = vec5_from_flat(&input_flat, [BATCH, IN_CH, DD, H, W]);
    let weight_nested = vec5_from_flat(&weight_flat, [OUT_CH, IN_CH, KD, KH, KW]);
    let expected_nested = conv3d_ncdhw_ref(
        &input_nested,
        &weight_nested,
        Some(&bias_flat),
        [1, 1, 1],
        [1, 1, 1],
    );
    let expected_flat = flatten5(&expected_nested);
    let out_shape = [BATCH, OUT_CH, DD, H, W];

    for device in available_devices().await {
        let input = Tensor::from_slice(&device, [BATCH, IN_CH, DD, H, W], &input_flat);
        let weight = Tensor::from_slice(&device, [OUT_CH, IN_CH, KD, KH, KW], &weight_flat);
        let bias = Tensor::from_slice(&device, [OUT_CH], &bias_flat);
        let actual = input
            .conv(&weight, Some(&bias), [1, 1, 1], [1, 1, 1])
            .to_concrete();
        let expected = Tensor::from_slice(&device, out_shape, &expected_flat);
        fusor_conformance::approx_eq(&actual, &expected, 1e-3)
            .await
            .unwrap();
    }
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
async fn f16_matmul_coop_tile_matches_host_reference() {
    // Pins the f16 cooperative-matrix path: shape divides cleanly into the
    // smallest coop tile (Tile64x64, BK=16). Without f16 coop support this
    // would fall back to `batched_matmul_with_epilogues<F16, ...>`; with it,
    // dispatch lands on `try_batched_coop_matmul::<F16, 64, 64, 16>`.
    use fusor_conformance::f16_capable_devices;
    use half::f16;

    const M: usize = 64;
    const N: usize = 64;
    const K: usize = 64;

    fn data(seed: u32, total: usize) -> Vec<f16> {
        (0..total)
            .map(|i| {
                let v = ((i + seed as usize) % 31) as f32;
                f16::from_f32((v - 15.0) * 0.05)
            })
            .collect()
    }

    let lhs_data = data(0, M * K);
    let rhs_data = data(7, K * N);

    let cpu_lhs: Tensor<2, f16> = Tensor::from_slice(&Device::Cpu, [M, K], &lhs_data);
    let cpu_rhs: Tensor<2, f16> = Tensor::from_slice(&Device::Cpu, [K, N], &rhs_data);
    let expected = cpu_lhs.matmul(&cpu_rhs).to_concrete();

    for device in f16_capable_devices().await {
        let lhs: Tensor<2, f16> = Tensor::from_slice(&device, [M, K], &lhs_data);
        let rhs: Tensor<2, f16> = Tensor::from_slice(&device, [K, N], &rhs_data);
        let actual = lhs.matmul(&rhs).to_concrete();
        fusor_conformance::approx_eq(&actual, &expected, f16::from_f32(5e-2))
            .await
            .unwrap();
    }
}

#[tokio::test]
async fn f16_matmul_multi_tile_matches_host_reference() {
    // Regression for the cooperative-load bug in
    // `batched_matmul_with_epilogues`: f16 matmul disables coop selection
    // (allow_coop is f32-only), and tile-aligned shapes with m>32 / n>64
    // route to the shared-tile kernel. Multi-tile in M and N is needed so
    // the per-lane offsets that leaked into the cooperative load actually
    // shift global_row/global_col away from the workgroup tile base.
    use fusor_conformance::f16_capable_devices;
    use half::f16;

    const M: usize = 64;
    const N: usize = 96;
    const K: usize = 64;

    fn data(seed: u32, total: usize) -> Vec<f16> {
        (0..total)
            .map(|i| {
                let v = ((i + seed as usize) % 31) as f32;
                f16::from_f32((v - 15.0) * 0.05)
            })
            .collect()
    }

    let lhs_data = data(0, M * K);
    let rhs_data = data(7, K * N);

    let cpu_lhs: Tensor<2, f16> = Tensor::from_slice(&Device::Cpu, [M, K], &lhs_data);
    let cpu_rhs: Tensor<2, f16> = Tensor::from_slice(&Device::Cpu, [K, N], &rhs_data);
    let expected = cpu_lhs.matmul(&cpu_rhs).to_concrete();

    for device in f16_capable_devices().await {
        let lhs: Tensor<2, f16> = Tensor::from_slice(&device, [M, K], &lhs_data);
        let rhs: Tensor<2, f16> = Tensor::from_slice(&device, [K, N], &rhs_data);
        let actual = lhs.matmul(&rhs).to_concrete();
        fusor_conformance::approx_eq(&actual, &expected, f16::from_f32(5e-2))
            .await
            .unwrap();
    }
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
