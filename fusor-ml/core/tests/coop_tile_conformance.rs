//! Per-tile conformance sweep over the full cooperative-matrix tile table.
//!
//! `FUSOR_FORCE_COOP_TILE` pins each geometry in turn and every tile must
//! produce reference-correct results over aligned, edge-masked, batched, and
//! transposed-B shapes. The (128, 128) entry shipped in the table for months
//! while being unreachable and miscomputing — a general tile-selection cost
//! model reaches every entry, so every entry has to earn its place here
//! before selection may consider it.

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

async fn check_forced(
    device: &Device,
    tile: (u32, u32),
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
                    "tile {}x{} batch={batch} m={m} k={k} n={n} transpose_b={transpose_b} \
                     [{bi}, {mi}, {ni}]: got {got}, expected {want}",
                    tile.0,
                    tile.1,
                );
            }
        }
    }
}

/// One sequential sweep: the tile-forcing env is process-global, so all
/// forced cases run inside a single test body.
#[test]
fn every_coop_tile_entry_computes_correctly() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };
        let geometries: Vec<(u32, u32)> = fusor_tile_ir_kernels::coop_tile_entries()
            .iter()
            .map(|entry| (entry.tile.bm, entry.tile.bn))
            .collect();
        for (bm, bn) in geometries {
            // SAFETY: this test mutates the process environment; the sweep
            // is one sequential test body, and the selection code re-reads
            // the variable per call.
            unsafe { std::env::set_var("FUSOR_FORCE_COOP_TILE", format!("{bm}x{bn}")) };
            let (m, n) = (2 * bm as usize, 2 * bn as usize);
            // Aligned, all edges masked at once, batched, and transposed-B.
            check_forced(&device, (bm, bn), 1, m, 64, n, false).await;
            check_forced(&device, (bm, bn), 1, m - 13, 50, n.saturating_sub(9).max(1), false)
                .await;
            check_forced(&device, (bm, bn), 3, m, 64, n, false).await;
            check_forced(&device, (bm, bn), 2, m, 64, n, true).await;
        }
        unsafe { std::env::remove_var("FUSOR_FORCE_COOP_TILE") };
    });
}
