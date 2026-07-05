//! Correctness gates for the small-side cooperative-matrix tiles
//! ((64, 16) and (16, 64)): contractions with a 16-wide (or 16-padded)
//! M or N side — the attention head_dim family and narrow-vocab lm_head
//! shapes — now route to the coop kernel instead of the generic reduce.
//! Each case A/Bs the GPU result against a CPU reference over exact-multiple
//! shapes, masked-edge shapes (M, N, and K edges), batched forms, and a
//! transposed-B operand. The (128, 128) precedent showed a tile table entry
//! can satisfy the fragment-size rules yet miscompute, so every new geometry
//! is exercised here.

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

fn check_matmul(batch: usize, m: usize, k: usize, n: usize, transpose_b: bool) {
    check_matmul_tol(batch, m, k, n, transpose_b, 1e-3);
}

fn check_matmul_tol(batch: usize, m: usize, k: usize, n: usize, transpose_b: bool, tol: f32) {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };
        let a_data = values(batch * m * k, 0.13);
        let b_data = values(batch * k * n, 0.07);

        let a = Tensor::from_slice(&device, [batch, m, k], &a_data);
        let b = if transpose_b {
            // Store B in [batch, n, k] order and view it as [batch, k, n]:
            // the transposed-operand geometry every dA/dB backward matmul
            // reads through.
            let b_t_data: Vec<f32> = (0..batch * n * k)
                .map(|i| {
                    let (bi, rest) = (i / (n * k), i % (n * k));
                    let (ni, ki) = (rest / k, rest % k);
                    b_data[(bi * k + ki) * n + ni]
                })
                .collect();
            let b_t = Tensor::from_slice(&device, [batch, n, k], &b_t_data);
            b_t.restride_layout(Layout::from_parts(
                0,
                vec![batch, k, n].into(),
                vec![n * k, 1, k].into(),
            ))
        } else {
            Tensor::from_slice(&device, [batch, k, n], &b_data)
        };

        let out = a.mat_mul(&b);
        assert_eq!(
            out.count_kernels_to_resolve(),
            1,
            "batch={batch} m={m} k={k} n={n} transpose_b={transpose_b}: \
             the contraction should resolve to one matmul kernel"
        );
        let actual = out.as_slice::<3, f32>().await.unwrap();
        let expected = cpu_matmul(&a_data, &b_data, batch, m, k, n);
        for bi in 0..batch {
            for mi in 0..m {
                for ni in 0..n {
                    let want = expected[(bi * m + mi) * n + ni];
                    let got = actual[[bi, mi, ni]];
                    assert!(
                        (got - want).abs() < tol + want.abs() * 1e-3,
                        "batch={batch} m={m} k={k} n={n} transpose_b={transpose_b} \
                         [{bi}, {mi}, {ni}]: got {got}, expected {want}"
                    );
                }
            }
        }
    });
}

// (64, 16): exact-multiple shapes — the P@V / dQ / dV attention family.
#[test]
fn small_tile_64x16_exact() {
    check_matmul(4, 64, 64, 16, false);
}

// (64, 16): every edge masked at once (M 120→128, N 14→16, K=50 tail tile).
#[test]
fn small_tile_64x16_masked_edges() {
    check_matmul(3, 120, 50, 14, false);
}

// (64, 16): N-edge padding on the vocab-65 lm_head geometry (65 → 80).
#[test]
fn small_tile_64x16_vocab_edge() {
    check_matmul(1, 192, 64, 65, false);
}

// (64, 16): transposed-B operand (the dS @ Kᵀ-style backward reads).
#[test]
fn small_tile_64x16_transposed_b() {
    check_matmul(2, 64, 64, 16, true);
}

// (16, 64): exact-multiple shapes — the Qᵀ@dS attention family.
#[test]
fn small_tile_16x64_exact() {
    check_matmul(4, 16, 64, 64, false);
}

// (16, 64): M-edge padding (65 → 80) with a masked K tail.
#[test]
fn small_tile_16x64_masked_edges() {
    check_matmul(2, 65, 50, 64, false);
}

// (16, 64): transposed-B operand.
#[test]
fn small_tile_16x64_transposed_b() {
    check_matmul(2, 16, 64, 64, true);
}

// (16, 64): long-K single-tile-grid shape (the lm_head gradient geometry);
// with K = 2048 this also routes through the split-K partials + combine
// sequence.
#[test]
fn small_tile_16x64_long_k() {
    check_matmul(1, 65, 2048, 64, false);
}

// Split-K A/B gates: starved tile grids with a long contraction run the
// partials + combine two-kernel sequence; the result must match the
// f64-accumulated reference within sum-reorder tolerance.

// The 64×2048×64 weight-gradient shape: one (64, 64) tile, 16 K-spans.
#[test]
fn split_k_weight_grad_square() {
    check_matmul_tol(1, 64, 2048, 64, false, 2e-3);
}

// The 64×2048×256 shape: two (64, 128) tiles across N, 16 K-spans.
#[test]
fn split_k_weight_grad_wide() {
    check_matmul_tol(1, 64, 2048, 256, false, 2e-3);
}

// K not a multiple of the span width: the last split's tiles read past the
// logical K extent and must fill zero (k = 1000 → 63 K-tiles over 16 spans).
#[test]
fn split_k_ragged_k() {
    check_matmul_tol(1, 64, 1000, 64, false, 2e-3);
}

// Barely past the split gate (k = 520 → 33 K-tiles, the last spans idle),
// with a transposed-B operand.
#[test]
fn split_k_short_spans_transposed_b() {
    check_matmul_tol(1, 64, 520, 64, true, 2e-3);
}

// Batched split-K: scratch rows interleave (split, batch) correctly.
#[test]
fn split_k_batched() {
    check_matmul_tol(3, 64, 640, 64, false, 2e-3);
}
