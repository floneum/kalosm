//! Temporary diagnostic: conv im2col case-2 repro (m=2048, k=576, n=128).
//! Runs the windowed (implicit-GEMM) matmul and the same contraction as a
//! plain dense matmul, many rounds, full-output check against a CPU
//! reference. Not part of the permanent suite.

use fusor_core::{Device, StrideSpec, Tensor};

fn cpu_reference(
    b: usize,
    c: usize,
    h: usize,
    w: usize,
    n: usize,
    kh: usize,
    kw: usize,
    input_host: &[f32],
    weight_host: &[f32],
) -> Vec<f32> {
    let (oh, ow) = (h - kh + 1, w - kw + 1);
    let (m, k) = (b * oh * ow, c * kh * kw);
    let mut out = vec![0f32; m * n];
    for mi in 0..m {
        let (bi, rest) = (mi / (oh * ow), mi % (oh * ow));
        let (ohi, owi) = (rest / ow, rest % ow);
        for ni in 0..n {
            let mut acc = 0f32;
            for ci in 0..c {
                for khi in 0..kh {
                    for kwi in 0..kw {
                        let iv = input_host[((bi * c + ci) * h + ohi + khi) * w + owi + kwi];
                        let wv = weight_host[ni * k + (ci * kh + khi) * kw + kwi];
                        acc += iv * wv;
                    }
                }
            }
            out[mi * n + ni] = acc;
        }
    }
    out
}

fn check(round: usize, label: &str, result: &[f32], expected: &[f32], n: usize) -> usize {
    let mut mismatches = 0usize;
    let mut worst = 0f32;
    let mut first = None;
    for (i, (&got, &exp)) in result.iter().zip(expected).enumerate() {
        let err = (got - exp).abs();
        if err > 1e-3 * exp.abs().max(1.0) {
            mismatches += 1;
            if err > worst {
                worst = err;
            }
            if first.is_none() {
                first = Some((i / n, i % n, got, exp));
            }
        }
    }
    if let Some((mi, ni, got, exp)) = first {
        println!(
            "round {round} {label}: {mismatches} mismatches, worst {worst:.6}, first [{mi},{ni}] got {got} exp {exp}"
        );
    } else {
        println!("round {round} {label}: clean");
    }
    mismatches
}

#[test]
fn conv_case2_repro() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };
        let (b, c, h, w, n, kh, kw) = (2usize, 64, 34, 34, 128, 3, 3);
        let (oh, ow) = (h - kh + 1, w - kw + 1);
        let (m, k) = (b * oh * ow, c * kh * kw);
        let input_host: Vec<f32> = (0..b * c * h * w).map(|i| (i % 13) as f32 * 0.1).collect();
        let weight_host: Vec<f32> = (0..n * k).map(|i| (i % 7) as f32 * 0.01).collect();
        let expected = cpu_reference(b, c, h, w, n, kh, kw, &input_host, &weight_host);

        // Host-side im2col for the plain dense variant.
        let mut a_dense = vec![0f32; m * k];
        for mi in 0..m {
            let (bi, rest) = (mi / (oh * ow), mi % (oh * ow));
            let (ohi, owi) = (rest / ow, rest % ow);
            for ci in 0..c {
                for khi in 0..kh {
                    for kwi in 0..kw {
                        a_dense[mi * k + (ci * kh + khi) * kw + kwi] =
                            input_host[((bi * c + ci) * h + ohi + khi) * w + owi + kwi];
                    }
                }
            }
        }

        let mut windowed_bad = 0usize;
        let mut dense_bad = 0usize;
        for round in 0..10 {
            let input = Tensor::from_slice(&device, [b, c, h, w], &input_host);
            let weight = Tensor::from_slice(&device, [n, k], &weight_host);
            let windows = input.restride([
                StrideSpec::dim(0, b),
                StrideSpec::dim_with(2, oh, 1),
                StrideSpec::dim_with(3, ow, 1),
                StrideSpec::dim(1, c),
                StrideSpec::dim(2, kh),
                StrideSpec::dim(3, kw),
            ]);
            let a = windows.reshape([m, k]);
            let b_mat = weight.restride([StrideSpec::dim(1, k), StrideSpec::dim(0, n)]);
            let out = a.mat_mul(&b_mat);
            let result = out.as_slice::<2, f32>().await.unwrap();
            let flat: Vec<f32> = (0..m)
                .flat_map(|mi| (0..n).map(move |ni| (mi, ni)))
                .map(|(mi, ni)| result[[mi, ni]])
                .collect();
            if check(round, "windowed", &flat, &expected, n) > 0 {
                windowed_bad += 1;
            }

            let a_t = Tensor::from_slice(&device, [m, k], &a_dense);
            let weight2 = Tensor::from_slice(&device, [n, k], &weight_host);
            let b_mat2 = weight2.restride([StrideSpec::dim(1, k), StrideSpec::dim(0, n)]);
            let out2 = a_t.mat_mul(&b_mat2);
            let result2 = out2.as_slice::<2, f32>().await.unwrap();
            let flat2: Vec<f32> = (0..m)
                .flat_map(|mi| (0..n).map(move |ni| (mi, ni)))
                .map(|(mi, ni)| result2[[mi, ni]])
                .collect();
            if check(round, "dense   ", &flat2, &expected, n) > 0 {
                dense_bad += 1;
            }
        }
        println!("SUMMARY windowed_bad={windowed_bad}/10 dense_bad={dense_bad}/10");
        assert!(windowed_bad == 0 && dense_bad == 0);
    });
}
