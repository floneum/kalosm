//! Contractions across all four families. Family is never stored on a node —
//! `lower_coop`, `lower_sgemm`, `lower_sgemv` and `lower_generic` coexist in
//! one chain — so a case here that produces the right numbers is evidence
//! that whichever family extraction picked is correct, on both backends.

use fusor2::{Dtype, Session};

use crate::harness::{CaseError, CaseResult, Cases, dims};
use crate::suite::support::{Domain, expect_values, gradient_of, graph_of, read, upload};

pub fn cases() -> Cases {
    let mut cases = Cases::new();
    cases.push("matmul", "matmul", |s| batched(s, &[], 3, 4, 5));
    cases.push("matmul", "mat_mul_rank3", |s| batched(s, &[2], 3, 4, 5));
    cases.push("matmul", "mat_mul_rank4", |s| batched(s, &[2, 2], 3, 4, 5));
    cases.push("matmul", "mat_mul_transposed_rhs", transposed_rhs);
    cases.push("matmul", "matmul_with_broadcast_bias", broadcast_bias);
    cases.push("matmul", "q_mat_mul", |s| quantized_matmul(s, 2));
    cases.push("matmul", "q_mat_mul_rank1", |s| quantized_matmul(s, 1));
    // Split-K at the extents the trainer and this suite actually use. The
    // shipped `extent.at_least(4096)` gate refuses every one of them, so
    // whether the reduction runs split or unsplit is a schedule decision
    // these four cases must not be able to tell apart.
    for k in SPLIT_K_EXTENTS {
        cases.push("matmul", split_k_name(k), move |s| split_k(s, k));
    }
    cases.extend(structural::cases());
    cases
}

/// The structural half: which law fired on a contraction, and what schedule
/// the extraction resolved for it.
///
/// A contraction is the one place where three separately hand-written
/// register-tiling mechanisms met — `SgemmParams{tm,tn}`, `CoopGeom{rg,cg}`
/// and `MapTiling{vector}` — and the claim of `PROMOTE` is that they are one
/// law at a schedule point. These cases are stated on a plain
/// `[m,k] x [k,n]`: no attention, no multi-slot carrier, no `exp` anywhere.
mod structural {
    use fusor2::{Session, };
use fusor2::tensor::Dyn as Tensor;

    use crate::harness::{CaseError, CaseResult, Cases, dims};
    use crate::suite::reductions::generality::structure;
    use crate::suite::support::{Domain, expect_shaped, graph_of, read, upload};

    pub fn cases() -> Cases {
        let mut cases = Cases::new();
        cases.push("matmul", "contraction_promotes_a_free_axis", promotes);
        cases.push("matmul", "wide_n_columns", wide_n);
        cases.push(
            "matmul",
            "contraction_resolves_a_schedule_point",
            resolves_theta,
        );
        cases.push("matmul", "qkv_projection_triple_plan", qkv_triple);
        cases
    }

    fn err(e: impl std::fmt::Display) -> CaseError {
        e.to_string().into()
    }

    fn build_matmul(
        session: &Session,
        m: u64,
        k: u64,
        n: u64,
        a: &[f32],
        b: &[f32],
    ) -> Result<Vec<Tensor>, CaseError> {
        let g = graph_of(session);
        let lhs = upload(g.handle(), &dims(&[m, k]), a)?;
        let rhs = upload(g.handle(), &dims(&[k, n]), b)?;
        Ok(vec![lhs.matmul(&rhs).map_err(err)?])
    }

    /// `PROMOTE` on an SGEMM-shaped contraction's nest.
    ///
    /// The `n` free axis moving into the accumulator's data space **is** `TN`;
    /// promoting `m` too is `TM x TN`. Nothing in this program is a fold law's
    /// motivating example — it is a dense matmul — so a `PROMOTE` that only
    /// fires on flash's output accumulator would report zero here.
    fn promotes(session: &Session) -> CaseResult {
        const M: u64 = 64;
        const K: u64 = 128;
        // 64 columns, not 96: `wide_n_columns` below owns the wider shape and
        // is red on CPU for a reason that has nothing to do with this law.
        const N: u64 = 64;
        let a = Domain::Wide.sample(901, (M * K) as usize);
        let b = Domain::Wide.sample(902, (K * N) as usize);
        let build = |s: &Session| build_matmul(s, M, K, N, &a, &b);

        // The value first: a promoted accumulator that lost a lane still
        // produces a plausible-looking matrix.
        let outs = build(session)?;
        let actual = read(&outs[0])?;
        let mut expected = vec![0.0f32; (M * N) as usize];
        for i in 0..M as usize {
            for j in 0..N as usize {
                let mut acc = 0.0f64;
                for p in 0..K as usize {
                    acc += a[i * K as usize + p] as f64 * b[p * N as usize + j] as f64;
                }
                expected[i * N as usize + j] = acc as f32;
            }
        }
        expect_shaped(session, &[M, N], &actual, &expected)?;

        structure::must_fire(
            session,
            &build,
            &[(
                "PROMOTE",
                "a free axis of the contraction's nest must be able to move into the \
                 accumulator's data space. That move IS register tiling; if it only fires \
                 on a multi-slot carrier it is a flash rule wearing a law's name",
            )],
        )
    }

    /// A dense `[m,k] x [k,n]` with **more than 64 output columns**.
    ///
    /// Every other contraction case in this file is `n <= 5`, so nothing in
    /// the suite reads past the first output tile. On the CPU backend
    /// everything from column 64 on comes back `0.0` — at `m >= 2` and any
    /// `k`, including `[4,3] x [3,96]` — while the GPU is correct at the same
    /// shapes. That is a silent wrong answer for every dense layer wider than
    /// 64 units, which is most of them, and it passes every existing case.
    ///
    /// `m == 1` is correct (the sgemv path), so the boundary is a tile the
    /// blocked CPU microkernel writes and never revisits.
    fn wide_n(session: &Session) -> CaseResult {
        const M: u64 = 4;
        const K: u64 = 8;
        const N: u64 = 96;
        let a = Domain::Wide.sample(921, (M * K) as usize);
        let b = Domain::Wide.sample(922, (K * N) as usize);

        let outs = build_matmul(session, M, K, N, &a, &b)?;
        let actual = read(&outs[0])?;
        let mut expected = vec![0.0f32; (M * N) as usize];
        for i in 0..M as usize {
            for j in 0..N as usize {
                let mut acc = 0.0f64;
                for p in 0..K as usize {
                    acc += a[i * K as usize + p] as f64 * b[p * N as usize + j] as f64;
                }
                expected[i * N as usize + j] = acc as f32;
            }
        }
        expect_shaped(session, &[M, N], &actual, &expected)
    }

    /// Extraction must resolve a **named** schedule point for the contraction,
    /// not merely select a node that carries a domain.
    ///
    /// Admissibility and selection are different claims: a node can carry
    /// 8,300 legal points and reach `verify_plan` with none of them chosen, at
    /// which point section 4.2 makes the failure a hard assert rather than a
    /// fallback. This asserts the plan's `theta` is non-empty and that at
    /// least one resolved point is a real contraction geometry rather than
    /// `SchedPoint::Point`.
    fn resolves_theta(session: &Session) -> CaseResult {
        use fusor2_ir::ir::level1::SchedPoint;
        const M: u64 = 128;
        const K: u64 = 512;
        const N: u64 = 128;
        let a = Domain::Wide.sample(903, (M * K) as usize);
        let b = Domain::Wide.sample(904, (K * N) as usize);

        let p = structure::probe_fresh(session, &|s| build_matmul(s, M, K, N, &a, &b))?;
        let thetas = p.thetas();
        if thetas.is_empty() {
            return Err(format!(
                "a [{M},{K}] x [{K},{N}] contraction resolved no schedule point at all. \
                 Rules that fired: {:?}",
                p.fired_names()
            )
            .into());
        }
        let named = thetas.iter().find(|t| !matches!(t, SchedPoint::Point));
        let Some(named) = named else {
            return Err(format!(
                "every resolved point on a [{M},{K}] x [{K},{N}] contraction is \
                 `SchedPoint::Point` — the node reached extraction with no schedule \
                 decision to make. Resolved: {thetas:?}"
            )
            .into());
        };
        // The point must be one the domain could actually offer: a geometry,
        // a fold strategy or a tiling. `Point` above is already excluded, so
        // this pins that the variant is a *contraction-shaped* one rather than
        // an unrelated node's schedule leaking into the assert.
        if !matches!(
            named,
            SchedPoint::Coop { .. }
                | SchedPoint::Sgemm(_)
                | SchedPoint::Sgemv(_)
                | SchedPoint::Fold(_)
        ) {
            return Err(format!(
                "the contraction resolved to {named:?}, which is not a contraction \
                 geometry, a fold strategy or a split"
            )
            .into());
        }
        Ok(())
    }

    /// Three projections of one activation, each with its own bias.
    ///
    /// `x@Wq + bq`, `x@Wk + bk`, `x@Wv + bv` is three launches today.
    /// `TUPLE` does not touch `post` — each slot keeps its own — so the
    /// triple can become one k-loop reading `x` once. The target is one
    /// launch, and the ceiling is what makes that a diff.
    fn qkv_triple(session: &Session) -> CaseResult {
        const ROWS: u64 = 64;
        const IN: u64 = 96;
        const OUT: u64 = 48;
        let x = Domain::Wide.sample(905, (ROWS * IN) as usize);
        let w: Vec<Vec<f32>> = (0..3)
            .map(|i| Domain::Wide.sample(906 + i, (IN * OUT) as usize))
            .collect();
        let bias: Vec<Vec<f32>> = (0..3)
            .map(|i| Domain::Wide.sample(916 + i, OUT as usize))
            .collect();

        let build = |s: &Session| -> Result<Vec<Tensor>, CaseError> {
            let g = graph_of(s);
            let a = upload(g.handle(), &dims(&[ROWS, IN]), &x)?;
            let mut outs = Vec::new();
            for i in 0..3usize {
                let wi = upload(g.handle(), &dims(&[IN, OUT]), &w[i])?;
                let bi = upload(g.handle(), &dims(&[1, OUT]), &bias[i])?;
                outs.push(
                    a.matmul(&wi)
                        .and_then(|y| y.broadcast_add(&bi))
                        .map_err(err)?,
                );
            }
            Ok(outs)
        };

        // Every projection's own epilogue must survive: a horizontal fusion
        // that dropped one bias produces three plausible matrices.
        let outs = build(session)?;
        for (i, out) in outs.iter().enumerate() {
            let actual = read(out)?;
            let mut expected = vec![0.0f32; (ROWS * OUT) as usize];
            for r in 0..ROWS as usize {
                for c in 0..OUT as usize {
                    let mut acc = 0.0f64;
                    for p in 0..IN as usize {
                        acc += x[r * IN as usize + p] as f64 * w[i][p * OUT as usize + c] as f64;
                    }
                    expected[r * OUT as usize + c] = acc as f32 + bias[i][c];
                }
            }
            expect_shaped(session, &[ROWS, OUT], &actual, &expected)?;
        }

        structure::plan_ceiling(
            session,
            &build,
            "qkv_projection_triple",
            9,
            1,
            "TUPLE under the sibling rooting: three nests over the same k axis of the \
             same activation are one nest, each slot keeping its own post",
        )
    }
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
/// traversal order — a case that pinned a left-to-right f32 accumulation
/// would go red the moment the schedule changed, which is the opposite of
/// what it is for.
fn split_k(session: &Session, k: u64) -> CaseResult {
    const M: u64 = 3;
    const N: u64 = 4;
    let a_data = Domain::Wide.sample(613, (M * k) as usize);
    let b_data = Domain::Wide.sample(617, (k * N) as usize);

    let graph = graph_of(session);
    let a = upload(graph.handle(), &dims(&[M, k]), &a_data)?;
    let b = upload(graph.handle(), &dims(&[k, N]), &b_data)?;
    let y = a
        .matmul(&b)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let actual = read(&y)?;

    let mut expected = vec![0.0f32; (M * N) as usize];
    for i in 0..M as usize {
        for j in 0..N as usize {
            let mut acc = 0.0f64;
            for t in 0..k as usize {
                acc += a_data[i * k as usize + t] as f64 * b_data[t * N as usize + j] as f64;
            }
            expected[i * N as usize + j] = acc as f32;
        }
    }
    // Relative to the accumulated magnitude, not to the result: a k-long dot
    // of centred data cancels, so the answer can be near zero while every
    // partial sum is not.
    let magnitude: f32 = (0..k as usize)
        .map(|t| a_data[t].abs() * b_data[t * N as usize].abs())
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

/// One contraction at an arbitrary batch prefix. Batch dims must already
/// match: there is no implicit broadcast, the frontend emits the restride.
fn batched(session: &Session, prefix: &[u64], m: u64, k: u64, n: u64) -> CaseResult {
    let batch: u64 = prefix.iter().product::<u64>().max(1);
    let a_shape: Vec<u64> = prefix.iter().copied().chain([m, k]).collect();
    let b_shape: Vec<u64> = prefix.iter().copied().chain([k, n]).collect();
    let out_shape: Vec<u64> = prefix.iter().copied().chain([m, n]).collect();

    let a_data = Domain::Wide.sample(307, (batch * m * k) as usize);
    let b_data = Domain::Wide.sample(311, (batch * k * n) as usize);

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
fn transposed_rhs(session: &Session) -> CaseResult {
    const M: usize = 3;
    const K: usize = 4;
    const N: usize = 5;
    let a_data = Domain::Wide.sample(313, M * K);
    let b_data = Domain::Wide.sample(317, N * K);

    let graph = graph_of(session);
    let a = upload(graph.handle(), &dims(&[M as u64, K as u64]), &a_data)?;
    let b = upload(graph.handle(), &dims(&[N as u64, K as u64]), &b_data)?;
    let y = a
        .matmul_t(&b)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let mut expected = vec![0.0f32; M * N];
    for i in 0..M {
        for j in 0..N {
            expected[i * N + j] = (0..K).map(|t| a_data[i * K + t] * b_data[j * K + t]).sum();
        }
    }
    expect_values(
        session,
        &[M as u64, N as u64],
        Dtype::F32,
        &read(&y)?,
        &expected,
    )?;

    let d_rhs = gradient_of(&graph, &y, &b)?;
    if d_rhs.len() != N * K {
        return Err(format!(
            "d_rhs has {} elements, want {}: it must land in rhs's own [N, K] layout, \
             not in a transposed view of it",
            d_rhs.len(),
            N * K
        )
        .into());
    }
    // d_rhs[j, t] = sum_i a[i, t], independent of j under an all-ones seed.
    for j in 0..N {
        for t in 0..K {
            let want: f32 = (0..M).map(|i| a_data[i * K + t]).sum();
            let got = d_rhs[j * K + t];
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
fn broadcast_bias(session: &Session) -> CaseResult {
    const M: usize = 4;
    const K: usize = 3;
    const N: usize = 5;
    let a_data = Domain::Wide.sample(331, M * K);
    let b_data = Domain::Wide.sample(337, K * N);
    let bias = Domain::Wide.sample(347, N);

    let graph = graph_of(session);
    let a = upload(graph.handle(), &dims(&[M as u64, K as u64]), &a_data)?;
    let b = upload(graph.handle(), &dims(&[K as u64, N as u64]), &b_data)?;
    let c = upload(graph.handle(), &dims(&[N as u64]), &bias)?;
    let y = a
        .matmul(&b)
        .and_then(|p| p.broadcast_add(&c))
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let product = host_matmul(&a_data, &b_data, 1, M, K, N);
    let expected: Vec<f32> = product
        .iter()
        .enumerate()
        .map(|(i, v)| v + bias[i % N])
        .collect();
    expect_values(
        session,
        &[M as u64, N as u64],
        Dtype::F32,
        &read(&y)?,
        &expected,
    )?;

    let d_bias = gradient_of(&graph, &y, &c)?;
    if d_bias.len() != N {
        return Err(format!("the bias gradient has {} elements, want {N}", d_bias.len()).into());
    }
    for (j, v) in d_bias.iter().enumerate() {
        if (v - M as f32).abs() > 1e-4 {
            return Err(format!(
                "bias gradient {j} is {v}, want {M}: a stride-0 axis's adjoint sums \
                 over that axis"
            )
            .into());
        }
    }
    Ok(())
}

/// `q_mat_mul` against a quantized weight.
///
/// Gradients flow to the **activation only**: the weight is quantized and
/// non-trainable through this route, which is what makes QAT a separate master
/// copy rather than a quantized backward kernel.
fn quantized_matmul(session: &Session, act_rank: usize) -> CaseResult {
    use fusor2_ir::dtype::{QFmt, QLayout};

    const ROWS: u64 = 3;
    let fmt = QFmt::Q8_0;
    let layout = QLayout::Native;
    let k = fmt.block_elements() as usize;
    let batch = if act_rank == 1 { 1 } else { 2 };

    // One block per weight row, so the host reference is a straight
    // `chunks(block_bytes)` walk over the same bytes the device reads.
    let block_bytes = fmt.block_bytes(layout) as usize;
    let mut bytes = Vec::with_capacity(ROWS as usize * block_bytes);
    let mut weights = vec![0.0f32; ROWS as usize * k];
    for r in 0..ROWS as usize {
        let mut block = vec![0u8; block_bytes];
        let mut state = (7919u32 + r as u32).wrapping_mul(2_654_435_761);
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

    let act = Domain::Wide.sample(313, batch * k);
    let graph = graph_of(session);
    let qm = fusor2::QMatrix::from_raw_bytes(
        &graph,
        fmt,
        layout,
        [fusor2::Dim::Const(ROWS), fusor2::Dim::Const(k as u64)],
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

    let mut expected = vec![0.0f32; batch * ROWS as usize];
    for b in 0..batch {
        for r in 0..ROWS as usize {
            expected[b * ROWS as usize + r] =
                (0..k).map(|t| act[b * k + t] * weights[r * k + t]).sum();
        }
    }
    let out_shape: Vec<usize> = if act_rank == 1 {
        vec![ROWS as usize]
    } else {
        vec![batch, ROWS as usize]
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
        .map(|n| (0..ROWS as usize).map(|r| weights[r * k + n % k]).sum())
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
        // The four structural cases are registered beside the numeric ones:
        // a contraction that resolves no schedule point, or one whose free
        // axis cannot be promoted, is a live failure mode that no numeric
        // case can see.
        for wanted in [
            "contraction_promotes_a_free_axis",
            "wide_n_columns",
            "contraction_resolves_a_schedule_point",
            "qkv_projection_triple_plan",
        ] {
            assert!(
                names.iter().any(|n| n == &format!("matmul::{wanted}")),
                "{wanted} is missing"
            );
        }
        assert_eq!(names.len(), 7 + SPLIT_K_EXTENTS.len() + 4);
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
