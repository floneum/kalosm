//! Contractions across all four families. Family is never stored on a node —
//! `lower_coop`, `lower_sgemm`, `lower_sgemv` and `lower_generic` coexist in
//! one chain — so a case here that produces the right numbers is evidence
//! that whichever family extraction picked is correct, on both backends.

use fusor2::{Dtype, Session};

use crate::harness::{CaseError, CaseResult, Cases, FuzzDim, dims, fuzz_case};
use crate::suite::support::{Domain, expect_values, gradient_of, graph_of, read, upload};

// Gradients here are analytic (all-ones seed row/column sums), not finite
// differences, so the sampled sizes can be moderate.
const MATMUL_SPEC: &[FuzzDim] = &[
    FuzzDim::Range(1, 16),
    FuzzDim::Range(1, 64),
    FuzzDim::Range(1, 16),
];
const MATMUL_RANK3_SPEC: &[FuzzDim] = &[
    FuzzDim::Range(1, 4),
    FuzzDim::Range(1, 16),
    FuzzDim::Range(1, 64),
    FuzzDim::Range(1, 16),
];
const MATMUL_RANK4_SPEC: &[FuzzDim] = &[
    FuzzDim::Range(1, 3),
    FuzzDim::Range(1, 3),
    FuzzDim::Range(1, 16),
    FuzzDim::Range(1, 64),
    FuzzDim::Range(1, 16),
];
const TRANSPOSED_RHS_SPEC: &[FuzzDim] = &[
    FuzzDim::Range(1, 8),
    FuzzDim::Range(1, 16),
    FuzzDim::Range(1, 8),
];
const BROADCAST_BIAS_SPEC: &[FuzzDim] = &[
    FuzzDim::Range(1, 8),
    FuzzDim::Range(1, 16),
    FuzzDim::Range(1, 8),
];
// `n` starts past one 64-wide output tile.
const WIDE_N_SPEC: &[FuzzDim] = &[
    FuzzDim::Range(1, 8),
    FuzzDim::Range(1, 32),
    FuzzDim::Range(65, 192),
];
const QKV_SPEC: &[FuzzDim] = &[
    FuzzDim::Range(1, 64),
    FuzzDim::Range(1, 96),
    FuzzDim::Range(1, 64),
];
// K is the split-K extent, chosen per case; only m and n are sampled.
const SPLIT_K_MN_SPEC: &[FuzzDim] = &[FuzzDim::Range(1, 8), FuzzDim::Range(1, 8)];
// K is the quantized block size, fixed by the format inside the body.
const QMATMUL_SPEC: &[FuzzDim] = &[FuzzDim::Range(1, 8), FuzzDim::Range(1, 4)];
const QMATMUL_RANK1_SPEC: &[FuzzDim] = &[FuzzDim::Range(1, 8), FuzzDim::Fixed(1)];

pub fn cases() -> Cases {
    let mut cases = Cases::new();
    cases.push_case(fuzz_case("matmul", "matmul", MATMUL_SPEC, batched));
    cases.push_case(fuzz_case(
        "matmul",
        "mat_mul_rank3",
        MATMUL_RANK3_SPEC,
        batched,
    ));
    cases.push_case(fuzz_case(
        "matmul",
        "mat_mul_rank4",
        MATMUL_RANK4_SPEC,
        batched,
    ));
    cases.push_case(fuzz_case(
        "matmul",
        "mat_mul_transposed_rhs",
        TRANSPOSED_RHS_SPEC,
        transposed_rhs,
    ));
    cases.push_case(fuzz_case(
        "matmul",
        "matmul_with_broadcast_bias",
        BROADCAST_BIAS_SPEC,
        broadcast_bias,
    ));
    cases.push_case(fuzz_case("matmul", "q_mat_mul", QMATMUL_SPEC, |s, sh, seed| {
        quantized_matmul(s, 2, sh, seed)
    }));
    cases.push_case(fuzz_case(
        "matmul",
        "q_mat_mul_rank1",
        QMATMUL_RANK1_SPEC,
        |s, sh, seed| quantized_matmul(s, 1, sh, seed),
    ));
    // Split-K at the extents the trainer and this suite actually use. The
    // shipped `extent.at_least(4096)` gate refuses every one of them, so
    // whether the reduction runs split or unsplit is a schedule decision
    // these four cases must not be able to tell apart.
    for k in SPLIT_K_EXTENTS {
        cases.push_case(fuzz_case(
            "matmul",
            split_k_name(k),
            SPLIT_K_MN_SPEC,
            move |s, sh, seed| split_k(s, k, sh, seed),
        ));
    }
    cases.push_case(fuzz_case("matmul", "wide_n_columns", WIDE_N_SPEC, wide_n));
    cases.push_case(fuzz_case(
        "matmul",
        "qkv_projection_triple",
        QKV_SPEC,
        qkv_triple,
    ));
    cases
}

/// A dense `[m,k] x [k,n]` with **more than 64 output columns**.
///
/// Most contraction cases in this file keep `n` small, so nothing else reads
/// past the first output tile. A blocked microkernel that writes a tile and
/// never revisits it produces `0.0` from column 64 on — a silent wrong answer
/// for every dense layer wider than 64 units — while passing every narrow
/// case.
fn wide_n(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let [m, k, n] = [shape[0], shape[1], shape[2]];
    let a = Domain::Wide.sample(seed, (m * k) as usize);
    let b = Domain::Wide.sample(seed ^ 0x9e37_79b9, (k * n) as usize);

    let graph = graph_of(session);
    let lhs = upload(graph.handle(), &dims(&[m, k]), &a)?;
    let rhs = upload(graph.handle(), &dims(&[k, n]), &b)?;
    let y = lhs
        .matmul(&rhs)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let actual = read(&y)?;
    let mut expected = vec![0.0f32; (m * n) as usize];
    for i in 0..m as usize {
        for j in 0..n as usize {
            let mut acc = 0.0f64;
            for p in 0..k as usize {
                acc += a[i * k as usize + p] as f64 * b[p * n as usize + j] as f64;
            }
            expected[i * n as usize + j] = acc as f32;
        }
    }
    expect_values(session, &[m, n], Dtype::F32, &actual, &expected)
}

/// Three projections of one activation, each with its own bias: `x@Wq + bq`,
/// `x@Wk + bk`, `x@Wv + bv` resolved together. Every projection's own
/// epilogue must survive whatever horizontal fusion the compiler applies — a
/// fusion that dropped one bias still produces three plausible matrices.
fn qkv_triple(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let [rows, cin, cout] = [shape[0], shape[1], shape[2]];
    let x = Domain::Wide.sample(seed, (rows * cin) as usize);
    let w: Vec<Vec<f32>> = (0..3)
        .map(|i| Domain::Wide.sample(seed.wrapping_add(1 + i), (cin * cout) as usize))
        .collect();
    let bias: Vec<Vec<f32>> = (0..3)
        .map(|i| Domain::Wide.sample(seed.wrapping_add(11 + i), cout as usize))
        .collect();

    let graph = graph_of(session);
    let a = upload(graph.handle(), &dims(&[rows, cin]), &x)?;
    let mut outs = Vec::new();
    for i in 0..3usize {
        let wi = upload(graph.handle(), &dims(&[cin, cout]), &w[i])?;
        let bi = upload(graph.handle(), &dims(&[1, cout]), &bias[i])?;
        outs.push(
            a.matmul(&wi)
                .and_then(|y| y.broadcast_add(&bi))
                .map_err(|e| -> CaseError { e.to_string().into() })?,
        );
    }

    for (i, out) in outs.iter().enumerate() {
        let actual = read(out)?;
        let mut expected = vec![0.0f32; (rows * cout) as usize];
        for r in 0..rows as usize {
            for c in 0..cout as usize {
                let mut acc = 0.0f64;
                for p in 0..cin as usize {
                    acc += x[r * cin as usize + p] as f64 * w[i][p * cout as usize + c] as f64;
                }
                expected[r * cout as usize + c] = acc as f32 + bias[i][c];
            }
        }
        expect_values(session, &[rows, cout], Dtype::F32, &actual, &expected)?;
    }
    Ok(())
}

/// The reduction extents every real matmul in this suite and in the trainer
/// lands on. All four are under the 4096 gate.
const SPLIT_K_EXTENTS: [u64; 4] = [512, 768, 1024, 2048];

fn split_k_name(k: u64) -> &'static str {
    match k {
        512 => "split_k_512",
        768 => "split_k_768",
        1024 => "split_k_1024",
        _ => "split_k_2048",
    }
}

/// A `[m, k] @ [k, n]` contraction at a long reduction axis.
///
/// The point is numeric, not structural: a split reduction and an unsplit one
/// are declared value-equal only where `NumericContract::reassoc` holds, and
/// the two orders differ in f32. The reference is accumulated in f64, so both
/// forms are measured against the true sum rather than against one particular
/// traversal order.
fn split_k(session: &Session, k: u64, shape: &[u64], seed: u32) -> CaseResult {
    let [m, n] = [shape[0], shape[1]];
    let a_data = Domain::Wide.sample(seed, (m * k) as usize);
    let b_data = Domain::Wide.sample(seed ^ 0x9e37_79b9, (k * n) as usize);

    let graph = graph_of(session);
    let a = upload(graph.handle(), &dims(&[m, k]), &a_data)?;
    let b = upload(graph.handle(), &dims(&[k, n]), &b_data)?;
    let y = a
        .matmul(&b)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let actual = read(&y)?;

    let mut expected = vec![0.0f32; (m * n) as usize];
    for i in 0..m as usize {
        for j in 0..n as usize {
            let mut acc = 0.0f64;
            for t in 0..k as usize {
                acc += a_data[i * k as usize + t] as f64 * b_data[t * n as usize + j] as f64;
            }
            expected[i * n as usize + j] = acc as f32;
        }
    }
    // Relative to the accumulated magnitude, not to the result: a k-long dot
    // of centred data cancels, so the answer can be near zero while every
    // partial sum is not.
    let magnitude: f32 = (0..k as usize)
        .map(|t| a_data[t].abs() * b_data[t * n as usize].abs())
        .sum::<f32>()
        .max(1.0);
    for (i, (got, want)) in actual.iter().zip(&expected).enumerate() {
        if (got - want).abs() > 1e-4 * magnitude {
            return Err(format!(
                "k={k} element {i}: got {got}, want {want}. A split and an unsplit \
                 reduction must agree to reassociation tolerance; a disagreement this \
                 large is a wrong outer level, not a rounding order."
            )
            .into());
        }
    }
    Ok(())
}

/// Host reference: `[batch..., m, k] @ [batch..., k, n]`.
fn host_matmul(a: &[f32], b: &[f32], batch: usize, m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * m * n];
    for bi in 0..batch {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for t in 0..k {
                    acc += a[bi * m * k + i * k + t] * b[bi * k * n + t * n + j];
                }
                out[bi * m * n + i * n + j] = acc;
            }
        }
    }
    out
}

/// One contraction at an arbitrary batch prefix (`shape` is
/// `[prefix..., m, k, n]`). Batch dims must already match: there is no
/// implicit broadcast, the frontend emits the restride.
fn batched(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let (prefix, mkn) = shape.split_at(shape.len() - 3);
    let [m, k, n] = [mkn[0], mkn[1], mkn[2]];
    let batch: u64 = prefix.iter().product::<u64>().max(1);
    let a_shape: Vec<u64> = prefix.iter().copied().chain([m, k]).collect();
    let b_shape: Vec<u64> = prefix.iter().copied().chain([k, n]).collect();
    let out_shape: Vec<u64> = prefix.iter().copied().chain([m, n]).collect();

    let a_data = Domain::Wide.sample(seed, (batch * m * k) as usize);
    let b_data = Domain::Wide.sample(seed ^ 0x9e37_79b9, (batch * k * n) as usize);

    let graph = graph_of(session);
    let a = upload(graph.handle(), &dims(&a_shape), &a_data)?;
    let b = upload(graph.handle(), &dims(&b_shape), &b_data)?;
    let y = a
        .matmul(&b)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let actual = read(&y)?;
    let expected = host_matmul(
        &a_data,
        &b_data,
        batch as usize,
        m as usize,
        k as usize,
        n as usize,
    );
    expect_values(session, &out_shape, Dtype::F32, &actual, &expected)?;

    // dA = grad @ B^T. Under `sum_all` the incoming gradient is all ones, so
    // dA[i, t] is the row sum of B[t, :] — checked analytically rather than
    // only against finite differences, because a transposed contraction spec
    // that got its labels backwards still passes a symmetric shape.
    let d_a = gradient_of(&graph, &y, &a)?;
    let mut want_a = vec![0.0f32; (batch * m * k) as usize];
    for bi in 0..batch as usize {
        for i in 0..m as usize {
            for t in 0..k as usize {
                let row: f32 = (0..n as usize)
                    .map(|j| b_data[bi * (k * n) as usize + t * n as usize + j])
                    .sum();
                want_a[bi * (m * k) as usize + i * k as usize + t] = row;
            }
        }
    }
    let backend = if crate::harness::is_gpu(session) {
        "gpu"
    } else {
        "cpu"
    };
    crate::compare::approx_or_relative_eq(backend, &[want_a.len()], &want_a, &d_a, 1e-4, 1e-4)?;
    Ok(())
}

/// `a @ b^T`, whose backward must land `d_rhs` **contiguous in rhs's own
/// layout** — that is the whole reason the op exists separately from
/// `matmul(a, b.t())`. Optimizer-side flattens stay zero-cost views instead of
/// gather kernels only if the gradient comes out in the weight's layout.
fn transposed_rhs(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let [m, k, n] = [shape[0] as usize, shape[1] as usize, shape[2] as usize];
    let a_data = Domain::Wide.sample(seed, m * k);
    let b_data = Domain::Wide.sample(seed ^ 0x9e37_79b9, n * k);

    let graph = graph_of(session);
    let a = upload(graph.handle(), &dims(&[m as u64, k as u64]), &a_data)?;
    let b = upload(graph.handle(), &dims(&[n as u64, k as u64]), &b_data)?;
    let y = a
        .matmul_t(&b)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let mut expected = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            expected[i * n + j] = (0..k).map(|t| a_data[i * k + t] * b_data[j * k + t]).sum();
        }
    }
    expect_values(
        session,
        &[m as u64, n as u64],
        Dtype::F32,
        &read(&y)?,
        &expected,
    )?;

    let d_rhs = gradient_of(&graph, &y, &b)?;
    if d_rhs.len() != n * k {
        return Err(format!(
            "d_rhs has {} elements, want {}: it must land in rhs's own [N, K] layout, \
             not in a transposed view of it",
            d_rhs.len(),
            n * k
        )
        .into());
    }
    // d_rhs[j, t] = sum_i a[i, t], independent of j under an all-ones seed.
    for j in 0..n {
        for t in 0..k {
            let want: f32 = (0..m).map(|i| a_data[i * k + t]).sum();
            let got = d_rhs[j * k + t];
            if (got - want).abs() > 1e-4 * want.abs().max(1.0) {
                return Err(format!("d_rhs[{j}, {t}] is {got}, want {want}").into());
            }
        }
    }
    Ok(())
}

/// A contraction with a rank-1 bias broadcast over its rows: the epilogue that
/// `sink_epilogue` folds into the contraction's `post`. The bias gradient must
/// be summed over the broadcast axis.
fn broadcast_bias(session: &Session, shape: &[u64], seed: u32) -> CaseResult {
    let [m, k, n] = [shape[0] as usize, shape[1] as usize, shape[2] as usize];
    let a_data = Domain::Wide.sample(seed, m * k);
    let b_data = Domain::Wide.sample(seed ^ 0x9e37_79b9, k * n);
    let bias = Domain::Wide.sample(seed.wrapping_add(1), n);

    let graph = graph_of(session);
    let a = upload(graph.handle(), &dims(&[m as u64, k as u64]), &a_data)?;
    let b = upload(graph.handle(), &dims(&[k as u64, n as u64]), &b_data)?;
    let c = upload(graph.handle(), &dims(&[n as u64]), &bias)?;
    let y = a
        .matmul(&b)
        .and_then(|p| p.broadcast_add(&c))
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let product = host_matmul(&a_data, &b_data, 1, m, k, n);
    let expected: Vec<f32> = product
        .iter()
        .enumerate()
        .map(|(i, v)| v + bias[i % n])
        .collect();
    expect_values(
        session,
        &[m as u64, n as u64],
        Dtype::F32,
        &read(&y)?,
        &expected,
    )?;

    let d_bias = gradient_of(&graph, &y, &c)?;
    if d_bias.len() != n {
        return Err(format!("the bias gradient has {} elements, want {n}", d_bias.len()).into());
    }
    for (j, v) in d_bias.iter().enumerate() {
        if (v - m as f32).abs() > 1e-4 * (m as f32) {
            return Err(format!(
                "bias gradient {j} is {v}, want {m}: a stride-0 axis's adjoint sums \
                 over that axis"
            )
            .into());
        }
    }
    Ok(())
}

/// `q_mat_mul` against a quantized weight. `shape` is `[rows, batch]`; K is
/// fixed at the format's block size.
///
/// Gradients flow to the **activation only**: the weight is quantized and
/// non-trainable through this route, which is what makes QAT a separate master
/// copy rather than a quantized backward kernel.
fn quantized_matmul(session: &Session, act_rank: usize, shape: &[u64], seed: u32) -> CaseResult {
    use fusor2_ir::dtype::{QFmt, QLayout};

    let rows = shape[0];
    let fmt = QFmt::Q8_0;
    let layout = QLayout::Native;
    let k = fmt.block_elements() as usize;
    let batch = if act_rank == 1 { 1 } else { shape[1] as usize };

    // One block per weight row, so the host reference is a straight
    // `chunks(block_bytes)` walk over the same bytes the device reads.
    let block_bytes = fmt.block_bytes(layout) as usize;
    let mut bytes = Vec::with_capacity(rows as usize * block_bytes);
    let mut weights = vec![0.0f32; rows as usize * k];
    for r in 0..rows as usize {
        let mut block = vec![0u8; block_bytes];
        let mut state = seed.wrapping_add(7919 + r as u32).wrapping_mul(2_654_435_761);
        for slot in block.iter_mut() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *slot = (state >> 24) as u8;
        }
        // An explicit finite scale: a random f16 is NaN about 1 time in 2000.
        block[0..2].copy_from_slice(&half::f16::from_f32(0.015_625).to_le_bytes());
        fusor2_gguf::blocks::cpu_dequantize_block(
            fmt,
            layout,
            &block,
            &mut weights[r * k..(r + 1) * k],
        );
        bytes.extend_from_slice(&block);
    }

    let act = Domain::Wide.sample(seed ^ 0x9e37_79b9, batch * k);
    let graph = graph_of(session);
    let qm = fusor2::QMatrix::from_raw_bytes(
        &graph,
        fmt,
        layout,
        [fusor2::Dim::Const(rows), fusor2::Dim::Const(k as u64)],
        &bytes,
    )
    .map_err(|e| -> CaseError { e.to_string().into() })?;

    let act_shape: Vec<u64> = if act_rank == 1 {
        vec![k as u64]
    } else {
        vec![batch as u64, k as u64]
    };
    let a = upload(graph.handle(), &dims(&act_shape), &act)?;
    let y = qm
        .q_mat_mul(&a)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let mut expected = vec![0.0f32; batch * rows as usize];
    for b in 0..batch {
        for r in 0..rows as usize {
            expected[b * rows as usize + r] =
                (0..k).map(|t| act[b * k + t] * weights[r * k + t]).sum();
        }
    }
    let out_shape: Vec<usize> = if act_rank == 1 {
        vec![rows as usize]
    } else {
        vec![batch, rows as usize]
    };
    let got = read(&y)?;
    // A quantized dot accumulates in f32 over 32 terms, so the bar is
    // relative to the result rather than absolute.
    crate::compare::approx_or_relative_eq(
        if crate::harness::is_gpu(session) {
            "gpu"
        } else {
            "cpu"
        },
        &out_shape,
        &expected,
        &got,
        1e-3,
        1e-3,
    )?;

    // Gradients flow to the activation only.
    let d_a = gradient_of(&graph, &y, &a)?;
    let want: Vec<f32> = (0..batch * k)
        .map(|n| (0..rows as usize).map(|r| weights[r * k + n % k]).sum())
        .collect();
    crate::compare::approx_or_relative_eq(
        if crate::harness::is_gpu(session) {
            "gpu"
        } else {
            "cpu"
        },
        &[batch * k],
        &want,
        &d_a,
        1e-3,
        1e-3,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_named_contraction_is_registered() {
        let names: Vec<String> = cases().names().iter().map(|n| (*n).to_string()).collect();
        for wanted in [
            "matmul",
            "mat_mul_rank3",
            "mat_mul_rank4",
            "matmul_with_broadcast_bias",
            "mat_mul_transposed_rhs",
            "q_mat_mul",
            "q_mat_mul_rank1",
            "split_k_512",
            "split_k_768",
            "split_k_1024",
            "split_k_2048",
        ] {
            assert!(
                names.iter().any(|n| n == &format!("matmul::{wanted}")),
                "{wanted} is missing"
            );
        }
        for wanted in ["wide_n_columns", "qkv_projection_triple"] {
            assert!(
                names.iter().any(|n| n == &format!("matmul::{wanted}")),
                "{wanted} is missing"
            );
        }
        assert_eq!(names.len(), 7 + SPLIT_K_EXTENTS.len() + 2);
    }

    /// Every split-K extent is under the shipped `at_least(4096)` gate. If
    /// one drifts above it the case stops measuring what it was written for.
    #[test]
    fn the_split_k_extents_are_all_under_the_shipped_gate() {
        for k in SPLIT_K_EXTENTS {
            assert!(k < 4096, "{k} is not under the gate this case exists for");
            assert!(k.is_multiple_of(8), "{k} cannot be blocked at all");
        }
    }

    #[test]
    fn the_host_matmul_reference_is_right_on_a_hand_worked_case() {
        // [[1, 2], [3, 4]] @ [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        assert_eq!(
            host_matmul(&a, &b, 1, 2, 2, 2),
            vec![19.0, 22.0, 43.0, 50.0]
        );
    }

    #[test]
    fn the_host_matmul_reference_keeps_batches_independent() {
        let a = [1.0f32, 1.0, 2.0, 2.0];
        let b = [1.0f32, 0.0, 0.0, 1.0];
        // Two 1x2 @ 2x1 contractions: [1*1 + 1*0] and [2*0 + 2*1].
        assert_eq!(host_matmul(&a, &b, 2, 1, 2, 1), vec![1.0, 2.0]);
    }
}
