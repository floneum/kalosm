//! Temporary diagnostic: plain dense 2048x576x128 matmul, many rounds,
//! full-output check. Not part of the permanent suite.

use fusor_core::{Device, StrideSpec, Tensor};

#[test]
fn dense_2048_576_128_repro() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };
        let (m, k, n) = (2048usize, 576, 128);
        let a_host: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 * 0.1).collect();
        let weight_host: Vec<f32> = (0..n * k).map(|i| (i % 7) as f32 * 0.01).collect();
        let mut expected = vec![0f32; m * n];
        for mi in 0..m {
            for ni in 0..n {
                let mut acc = 0f32;
                for ki in 0..k {
                    acc += a_host[mi * k + ki] * weight_host[ni * k + ki];
                }
                expected[mi * n + ni] = acc;
            }
        }

        let mut bad = 0usize;
        for round in 0..10 {
            let a = Tensor::from_slice(&device, [m, k], &a_host);
            let weight = Tensor::from_slice(&device, [n, k], &weight_host);
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
