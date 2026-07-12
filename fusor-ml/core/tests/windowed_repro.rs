//! Temporary diagnostic: windowed (im2col) 2048x576x128 matmul only.

use fusor_core::{Device, StrideSpec, Tensor};

#[test]
fn windowed_2048_576_128_repro() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };
        let (b, c, h, w, n, kh, kw) = (2usize, 64, 34, 34, 128, 3, 3);
        let (oh, ow) = (h - kh + 1, w - kw + 1);
        let (m, k) = (b * oh * ow, c * kh * kw);
        let input_host: Vec<f32> = (0..b * c * h * w).map(|i| (i % 13) as f32 * 0.1).collect();
        let weight_host: Vec<f32> = (0..n * k).map(|i| (i % 7) as f32 * 0.01).collect();
        let mut expected = vec![0f32; m * n];
        for mi in 0..m {
            let (bi, rest) = (mi / (oh * ow), mi % (oh * ow));
            let (ohi, owi) = (rest / ow, rest % ow);
            for ni in 0..n {
                let mut acc = 0f32;
                for ci in 0..c {
                    for khi in 0..kh {
                        for kwi in 0..kw {
                            acc += input_host[((bi * c + ci) * h + ohi + khi) * w + owi + kwi]
                                * weight_host[ni * k + (ci * kh + khi) * kw + kwi];
                        }
                    }
                }
                expected[mi * n + ni] = acc;
            }
        }

        let mut bad = 0usize;
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
            let mut mismatches = 0usize;
            let mut first = None;
            for mi in 0..m {
                for ni in 0..n {
                    let got = result[[mi, ni]];
                    let exp = expected[mi * n + ni];
                    if (got - exp).abs() > 1e-3 * exp.abs().max(1.0) {
                        mismatches += 1;
                        if first.is_none() {
                            first = Some((mi, ni, got, exp));
                        }
                    }
                }
            }
            println!("round {round}: mismatches={mismatches} first={first:?}");
            if mismatches > 0 {
                bad += 1;
            }
        }
        println!("SUMMARY bad={bad}/10");
        assert_eq!(bad, 0);
    });
}
