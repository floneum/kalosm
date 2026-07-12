//! Automatic cooperative-matrix selector conformance over shapes derived
//! from every tile-table geometry.

use fusor_core::{Device, Layout, Tensor};

fn values(len: usize, freq: f32) -> Vec<f32> {
    (0..len).map(|i| ((i as f32) * freq).sin()).collect()
}

/// f64-accumulated reference for `a[batch, m, k] @ b[batch, k, n]`.
fn cpu_matmul(a: &[f32], b: &[f32], batch: usize, m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * m * n];
    for bi in 0..batch {
        for mi in 0..m {
            for ni in 0..n {
                let mut acc = 0.0f64;
                for ki in 0..k {
                    let a_val = a[(bi * m + mi) * k + ki] as f64;
                    let b_val = b[(bi * k + ki) * n + ni] as f64;
                    acc += a_val * b_val;
                }
                out[(bi * m + mi) * n + ni] = acc as f32;
            }
        }
    }
    out
}

async fn check_automatic(
    device: &Device,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    transpose_b: bool,
) {
    let a_data = values(batch * m * k, 0.13);
    let b_data = values(batch * k * n, 0.07);

    let a = Tensor::from_slice(device, [batch, m, k], &a_data);
    let b = if transpose_b {
        let b_t_data: Vec<f32> = (0..batch * n * k)
            .map(|i| {
                let (bi, rest) = (i / (n * k), i % (n * k));
                let (ni, ki) = (rest / k, rest % k);
                b_data[(bi * k + ki) * n + ni]
            })
            .collect();
        let b_t = Tensor::from_slice(device, [batch, n, k], &b_t_data);
        b_t.restride_layout(Layout::from_parts(
            0,
            vec![batch, k, n].into(),
            vec![n * k, 1, k].into(),
        ))
    } else {
        Tensor::from_slice(device, [batch, k, n], &b_data)
    };

    let out = a.mat_mul(&b);
    let actual = out.as_slice::<3, f32>().await.unwrap();
    let expected = cpu_matmul(&a_data, &b_data, batch, m, k, n);
    for bi in 0..batch {
        for mi in 0..m {
            for ni in 0..n {
                let want = expected[(bi * m + mi) * n + ni];
                let got = actual[[bi, mi, ni]];
                assert!(
                    (got - want).abs() < 1e-3 + want.abs() * 1e-3,
                    "batch={batch} m={m} k={k} n={n} transpose_b={transpose_b} \
                     [{bi}, {mi}, {ni}]: got {got}, expected {want}",
                );
            }
        }
    }
}

#[test]
fn automatic_coop_selection_computes_table_derived_shapes_correctly() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };
        let geometries: Vec<(u32, u32)> = fusor_tile_ir_kernels::coop_tile_entries()
            .iter()
            .map(|entry| (entry.tile.bm, entry.tile.bn))
            .collect();
        for (bm, bn) in geometries {
            let (m, n) = (2 * bm as usize, 2 * bn as usize);
            // Aligned, all edges masked at once, batched, and transposed-B.
            check_automatic(&device, 1, m, 64, n, false).await;
            check_automatic(&device, 1, m - 13, 50, n.saturating_sub(9).max(1), false).await;
            check_automatic(&device, 3, m, 64, n, false).await;
            check_automatic(&device, 2, m, 64, n, true).await;
        }
    });
}
