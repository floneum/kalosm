//! Attention (dense, causal, masked, GQA/MQA, lse, grads) and the RoPE family.
//!
//! Every attention case is checked against a host implementation that spells
//! out the softmax explicitly. `MaskKind::Causal` is structural: no mask tensor
//! is uploaded.

use fusor2::composite::attention::{
    attention, attention_causal, attention_grads, attention_lse, attention_masked,
    attention_with_lse,
};
use fusor2::composite::rope::{
    base_inverse_frequency, rope, rope_fused, rope_interleaved, rope_interleaved_with_position,
    rope_normal_fused, rope_normal_pair_fused, rope_normal_pair_fused_with_position,
    rope_pair_fused, rope_pair_fused_with_position, rope_with_position, rotate_half,
};
use fusor2::graph::GraphRef;
use fusor2::{Dtype, Session, Tensor};
use fusor2_ir::ir::level1::MaskKind;

use crate::harness::{CaseError, CaseResult, Cases, dims, from_u32};
use crate::suite::support::{Domain, expect_values, gradient_of, graph_of, read, upload};

/// `[B, H, L, Dh]`. `Dh` is even because every RoPE pairing needs it to be,
/// and `Lq != Lk`.
const B: usize = 2;
const H: usize = 2;
const LQ: usize = 3;
const LK: usize = 4;
const DH: usize = 4;

const Q_LEN: usize = B * H * LQ * DH;

fn q_shape() -> Vec<u64> {
    vec![B as u64, H as u64, LQ as u64, DH as u64]
}

fn kv_shape(heads: usize) -> Vec<u64> {
    vec![B as u64, heads as u64, LK as u64, DH as u64]
}

fn kv_len(heads: usize) -> usize {
    B * heads * LK * DH
}

/// `1 / sqrt(Dh)`, the default scale.
fn default_scale() -> f32 {
    1.0 / (DH as f32).sqrt()
}

fn backend_of(session: &Session) -> &'static str {
    if crate::harness::is_gpu(session) {
        "gpu"
    } else {
        "cpu"
    }
}

/// `[B, H, Lq, Dh]` output and `[B, H, Lq]` log-sum-exp.
///
/// `heads_kv` may be smaller than `H`; query head `h` reads kv head
/// `h / (H / heads_kv)`, which is the GQA expansion. `mask(qi, ki)` is the
/// additive score bias.
fn host_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    heads_kv: usize,
    scale: f32,
    mask: &dyn Fn(usize, usize) -> f32,
) -> (Vec<f32>, Vec<f32>) {
    let groups = H / heads_kv;
    let mut out = vec![0.0f32; B * H * LQ * DH];
    let mut lse = vec![0.0f32; B * H * LQ];
    for b in 0..B {
        for h in 0..H {
            let hk = h / groups;
            for i in 0..LQ {
                let qbase = ((b * H + h) * LQ + i) * DH;
                let mut scores = vec![0.0f32; LK];
                for (j, s) in scores.iter_mut().enumerate() {
                    let kbase = ((b * heads_kv + hk) * LK + j) * DH;
                    let dot: f32 = (0..DH).map(|d| q[qbase + d] * k[kbase + d]).sum();
                    *s = dot * scale + mask(i, j);
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum: f32 = e.iter().sum();
                lse[(b * H + h) * LQ + i] = max + sum.ln();
                for d in 0..DH {
                    let mut acc = 0.0f32;
                    for (j, ej) in e.iter().enumerate() {
                        let vbase = ((b * heads_kv + hk) * LK + j) * DH;
                        acc += (ej / sum) * v[vbase + d];
                    }
                    out[qbase + d] = acc;
                }
            }
        }
    }
    (out, lse)
}

fn no_mask(_: usize, _: usize) -> f32 {
    0.0
}

/// Causal over `[Lq, Lk]`, right-aligned: query `i` sees keys up to
/// `i + (Lk - Lq)`.
fn causal_mask(i: usize, j: usize) -> f32 {
    if j <= i + (LK - LQ) {
        0.0
    } else {
        f32::NEG_INFINITY
    }
}

/// Host `(dq, dk, dv)` for unmasked attention at `heads_kv == H`.
fn host_attention_grads(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    scale: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut dq = vec![0.0f32; q.len()];
    let mut dk = vec![0.0f32; k.len()];
    let mut dv = vec![0.0f32; v.len()];
    for b in 0..B {
        for h in 0..H {
            for i in 0..LQ {
                let qb = ((b * H + h) * LQ + i) * DH;
                let mut p = vec![0.0f32; LK];
                for (j, s) in p.iter_mut().enumerate() {
                    let kb = ((b * H + h) * LK + j) * DH;
                    *s = (0..DH).map(|d| q[qb + d] * k[kb + d]).sum::<f32>() * scale;
                }
                let max = p.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in p.iter_mut() {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                for s in p.iter_mut() {
                    *s /= sum;
                }
                // dp[j] = <g_i, v_j>; ds = p * (dp - <p, dp>) * scale.
                let mut dp = vec![0.0f32; LK];
                for (j, dpj) in dp.iter_mut().enumerate() {
                    let vb = ((b * H + h) * LK + j) * DH;
                    *dpj = (0..DH).map(|d| g[qb + d] * v[vb + d]).sum();
                    for d in 0..DH {
                        dv[vb + d] += p[j] * g[qb + d];
                    }
                }
                let dot: f32 = p.iter().zip(&dp).map(|(a, b)| a * b).sum();
                for j in 0..LK {
                    let ds = p[j] * (dp[j] - dot) * scale;
                    let kb = ((b * H + h) * LK + j) * DH;
                    for d in 0..DH {
                        dq[qb + d] += ds * k[kb + d];
                        dk[kb + d] += ds * q[qb + d];
                    }
                }
            }
        }
    }
    (dq, dk, dv)
}

/// The rotation applied to one `[Dh]` head vector at position `p`.
/// `interleaved` pairs `(2i, 2i+1)`; otherwise pairs `(i, i + Dh/2)`.
fn host_rope_vec(x: &[f32], cos: &[f32], sin: &[f32], p: usize, interleaved: bool) -> Vec<f32> {
    let half = DH / 2;
    let mut out = vec![0.0f32; DH];
    for i in 0..half {
        let (a, b) = if interleaved {
            (2 * i, 2 * i + 1)
        } else {
            (i, i + half)
        };
        let (c, s) = (cos[p * half + i], sin[p * half + i]);
        out[a] = x[a] * c - x[b] * s;
        out[b] = x[a] * s + x[b] * c;
    }
    out
}

/// The whole `[B, H, L, Dh]` rope, row `l` reading table row `offset + l`.
fn host_rope(x: &[f32], cos: &[f32], sin: &[f32], len: usize, offset: usize, il: bool) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for b in 0..B {
        for h in 0..H {
            for l in 0..len {
                let base = ((b * H + h) * len + l) * DH;
                let rotated = host_rope_vec(&x[base..base + DH], cos, sin, offset + l, il);
                out[base..base + DH].copy_from_slice(&rotated);
            }
        }
    }
    out
}

/// The `[max_len, Dh/2]` sin/cos tables.
fn rope_tables(max_len: usize) -> (Vec<f32>, Vec<f32>) {
    let inv = base_inverse_frequency(DH as u32, 10_000.0);
    let mut cos = Vec::with_capacity(max_len * inv.len());
    let mut sin = Vec::with_capacity(max_len * inv.len());
    for p in 0..max_len {
        for f in &inv {
            cos.push((p as f32 * f).cos());
            sin.push((p as f32 * f).sin());
        }
    }
    (cos, sin)
}

pub fn cases() -> Cases {
    let mut cases = Cases::new();

    cases.push("attention_rope", "attention", |s| {
        attention_case(s, "attention", H, &no_mask, |q, k, v| {
            attention(q, k, v, MaskKind::None, None)
        })
    });
    cases.push("attention_rope", "attention_causal", |s| {
        attention_case(s, "attention_causal", H, &causal_mask, |q, k, v| {
            attention_causal(q, k, v, None)
        })
    });
    cases.push("attention_rope", "attention_causal_via_mask_kind", |s| {
        attention_case(
            s,
            "attention_causal_via_mask_kind",
            H,
            &causal_mask,
            |q, k, v| attention(q, k, v, MaskKind::Causal, None),
        )
    });
    cases.push("attention_rope", "attention_explicit_scale", |s| {
        attention_scale_case(s)
    });
    cases.push("attention_rope", "attention_gqa", |s| {
        attention_case(s, "attention_gqa", 1, &no_mask, |q, k, v| {
            attention(q, k, v, MaskKind::None, None)
        })
    });
    cases.push("attention_rope", "attention_mqa_single_kv_head", |s| {
        attention_case(
            s,
            "attention_mqa_single_kv_head",
            1,
            &causal_mask,
            |q, k, v| attention_causal(q, k, v, None),
        )
    });

    // Structural, on the chain `attention_defn` emits rather than a hand-built
    // graph.
    cases.push("attention_rope", "attention_defn_saturates", |s| {
        saturation_case(s, false)
    });
    cases.push("attention_rope", "attention_causal_defn_saturates", |s| {
        saturation_case(s, true)
    });

    // Launch counts, as ceilings.
    cases.push("attention_rope", "attention_forward_launch_ceiling", |s| {
        launch_ceiling_case(s, "attention_forward")
    });
    cases.push("attention_rope", "attention_with_lse_launch_ceiling", |s| {
        launch_ceiling_case(s, "attention_with_lse")
    });
    cases.push("attention_rope", "attention_causal_launch_ceiling", |s| {
        launch_ceiling_case(s, "attention_causal_forward")
    });

    cases.push("attention_rope", "attention_qk_mask", qk_mask_case);
    cases.push(
        "attention_rope",
        "attention_refuses_a_tensor_mask_kind_without_a_tensor",
        mask_arity,
    );
    cases.push("attention_rope", "attention_lse", lse_case);
    cases.push("attention_rope", "attention_with_lse", with_lse_case);
    cases.push("attention_rope", "attention_grads", grads_case);
    cases.push(
        "attention_rope",
        "attention_grads_launch_ceiling",
        grads_launch_ceiling,
    );
    cases.push(
        "attention_rope",
        "attention_grads_refuse_grouped_heads",
        grads_gqa_refused,
    );
    cases.push(
        "attention_rope",
        "attention_backward_matches_the_analytic_adjoints",
        attention_backward,
    );

    // RoPE. Every spelling is checked against the same host rotation.
    cases.push("attention_rope", "rope", |s| {
        rope_case(s, "rope", false, 0, rope)
    });
    cases.push("attention_rope", "rope_interleaved", |s| {
        rope_case(s, "rope_interleaved", true, 0, rope_interleaved)
    });
    cases.push("attention_rope", "rope_offset", |s| {
        rope_case(s, "rope_offset", false, 2, rope)
    });
    cases.push("attention_rope", "rope_fused", |s| {
        rope_case(s, "rope_fused", true, 0, rope_fused)
    });
    cases.push("attention_rope", "rope_normal_fused", |s| {
        rope_case(s, "rope_normal_fused", false, 0, rope_normal_fused)
    });
    cases.push("attention_rope", "rope_pair_fused", |s| {
        rope_pair_case(s, "rope_pair_fused", true, rope_pair_fused)
    });
    cases.push("attention_rope", "rope_normal_pair_fused", |s| {
        rope_pair_case(s, "rope_normal_pair_fused", false, rope_normal_pair_fused)
    });
    cases.push("attention_rope", "rope_pair_fused_with_position", |s| {
        rope_position_pair_case(
            s,
            "rope_pair_fused_with_position",
            true,
            rope_pair_fused_with_position,
        )
    });
    cases.push(
        "attention_rope",
        "rope_normal_pair_fused_with_position",
        |s| {
            rope_position_pair_case(
                s,
                "rope_normal_pair_fused_with_position",
                false,
                rope_normal_pair_fused_with_position,
            )
        },
    );
    cases.push("attention_rope", "rope_with_position", |s| {
        rope_position_case(s, "rope_with_position", false, rope_with_position)
    });
    cases.push("attention_rope", "rope_interleaved_with_position", |s| {
        rope_position_case(
            s,
            "rope_interleaved_with_position",
            true,
            rope_interleaved_with_position,
        )
    });
    cases.push("attention_rope", "rotate_half", rotate_half_case);
    cases.push(
        "attention_rope",
        "rope_is_norm_preserving",
        rope_norm_preserving,
    );
    cases.push(
        "attention_rope",
        "rope_backward_is_the_transposed_rotation",
        rope_backward,
    );
    cases.extend(materialization::cases());
    cases
}

/// Asserts on the extracted plan's materialized set that the `[Lq, Lk]` score,
/// probability and `dp` matrices stay out of it.
mod materialization {
    use fusor2::composite::attention::{
        attention, attention_causal, attention_grads, attention_with_lse,
    };
    use fusor2::{Session, Tensor};
    use fusor2_ir::ir::level1::MaskKind;

    use crate::harness::{CaseError, CaseResult, Cases, dims};
    use crate::suite::reductions::generality::structure;
    use crate::suite::support::{Domain, graph_of, upload};

    /// A shape whose score matrix has a distinct element count from every other
    /// tensor in the program: `B*H*Lq*Lk = 140`, `q` = 120, `k` and `v` = 168.
    const B: u64 = 2;
    const H: u64 = 2;
    const LQ: u64 = 5;
    const LK: u64 = 7;
    const DH: u64 = 6;

    /// The `[B, H, Lq, Lk]` score / probability / dp element count.
    const SCORES: u64 = B * H * LQ * LK;

    pub fn cases() -> Cases {
        let mut cases = Cases::new();
        cases.push(
            "attention_rope",
            "attention_forward_score_matrix_materialization",
            |s| forward(s),
        );
        cases.push(
            "attention_rope",
            "attention_backward_score_matrix_materialization",
            |s| backward(s),
        );
        cases.push(
            "attention_rope",
            "attention_causal_plan_is_no_worse_than_dense",
            |s| causal_ratio(s),
        );
        cases
    }

    fn err(e: impl std::fmt::Display) -> CaseError {
        e.to_string().into()
    }

    fn qkv(session: &Session) -> Result<(Tensor, Tensor, Tensor), CaseError> {
        let g = graph_of(session);
        let q = upload(
            g.handle(),
            &dims(&[B, H, LQ, DH]),
            &Domain::Wide.sample(931, (B * H * LQ * DH) as usize),
        )?;
        let k = upload(
            g.handle(),
            &dims(&[B, H, LK, DH]),
            &Domain::Wide.sample(932, (B * H * LK * DH) as usize),
        )?;
        let v = upload(
            g.handle(),
            &dims(&[B, H, LK, DH]),
            &Domain::Wide.sample(933, (B * H * LK * DH) as usize),
        )?;
        Ok((q, k, v))
    }

    /// Counts materialized `[Lq, Lk]` score and probability matrices against a
    /// ceiling. The fold-to-fold launch boundary forces some to exist.
    fn forward(session: &Session) -> CaseResult {
        let build = |s: &Session| -> Result<Vec<Tensor>, CaseError> {
            let (q, k, v) = qkv(s)?;
            Ok(vec![
                attention(&q, &k, &v, MaskKind::None, None).map_err(err)?,
            ])
        };
        let p = structure::probe_fresh(session, &build)?;
        let scores = p.buffer_elements().iter().filter(|n| **n == SCORES).count();
        if scores > 5 {
            return Err(format!(
                "attention forward materializes {scores} separate [B,H,Lq,Lk] buffers \
                 ({SCORES} elements each), ceiling 5, target 0. Every one of them is a \
                 score or probability matrix that the fold-to-fold launch boundary \
                 forces into the materialized set; more than the ceiling means another \
                 intermediate joined it. Buffers: {:?}",
                p.buffer_elements()
            )
            .into());
        }
        Ok(())
    }

    /// The same count for the score, probability and `dp` matrices across the
    /// whole `attention_grads` chain.
    fn backward(session: &Session) -> CaseResult {
        let build = |s: &Session| -> Result<Vec<Tensor>, CaseError> {
            let (q, k, v) = qkv(s)?;
            let g = graph_of(s);
            let d_out = upload(
                g.handle(),
                &dims(&[B, H, LQ, DH]),
                &Domain::Wide.sample(934, (B * H * LQ * DH) as usize),
            )?;
            let (o, lse) = attention_with_lse(&q, &k, &v, MaskKind::None, None).map_err(err)?;
            let (dq, dk, dv) =
                attention_grads(&q, &k, &v, &o, &d_out, &lse, MaskKind::None, None).map_err(err)?;
            Ok(vec![dq, dk, dv])
        };
        let p = structure::probe_fresh(session, &build)?;
        let scores = p.buffer_elements().iter().filter(|n| **n == SCORES).count();
        if scores > 15 {
            return Err(format!(
                "attention backward materializes {scores} separate [B,H,Lq,Lk] buffers \
                 ({SCORES} elements each), ceiling 15, target 0. The score, probability \
                 and dp matrices are the whole memory win of the kernel this design \
                 replaces; 15 of them is where the composed backward lands today and 0 is \
                 where PROMOTE + the reduction-nesting clause must put it. \
                 Buffers: {:?}",
                p.buffer_elements()
            )
            .into());
        }
        Ok(())
    }

    /// Causal attention must not cost more than the dense shape: buffer bytes
    /// and launch count are both pinned at parity.
    fn causal_ratio(session: &Session) -> CaseResult {
        let square = |s: &Session, causal: bool| -> Result<Vec<Tensor>, CaseError> {
            let g = graph_of(s);
            let n = B * H * LQ * DH;
            let q = upload(
                g.handle(),
                &dims(&[B, H, LQ, DH]),
                &Domain::Wide.sample(941, n as usize),
            )?;
            let k = upload(
                g.handle(),
                &dims(&[B, H, LQ, DH]),
                &Domain::Wide.sample(942, n as usize),
            )?;
            let v = upload(
                g.handle(),
                &dims(&[B, H, LQ, DH]),
                &Domain::Wide.sample(943, n as usize),
            )?;
            Ok(vec![if causal {
                attention_causal(&q, &k, &v, None).map_err(err)?
            } else {
                attention(&q, &k, &v, MaskKind::None, None).map_err(err)?
            }])
        };
        let dense = structure::probe_fresh(session, &|s| square(s, false))?;
        let causal = structure::probe_fresh(session, &|s| square(s, true))?;

        if causal.launches() > dense.launches() {
            return Err(format!(
                "causal attention plans {} launches against dense attention's {}. \
                 Causality is a predicate that rides into the carrier; it must never add \
                 a dispatch.",
                causal.launches(),
                dense.launches()
            )
            .into());
        }
        if causal.buffer_bytes() > dense.buffer_bytes() {
            return Err(format!(
                "causal attention allocates {} bytes against dense attention's {}. \
                 `STRIP`'s elide clause is supposed to make the causal shape cheaper; it \
                 must not make it more expensive first.",
                causal.buffer_bytes(),
                dense.buffer_bytes()
            )
            .into());
        }
        Ok(())
    }
}

type AttnBuild = fn(&Tensor, &Tensor, &Tensor) -> fusor2::Result<Tensor>;

fn attention_case(
    session: &Session,
    name: &'static str,
    heads_kv: usize,
    host_mask: &dyn Fn(usize, usize) -> f32,
    build: AttnBuild,
) -> CaseResult {
    let q_data = Domain::Wide.sample(701, Q_LEN);
    let k_data = Domain::Wide.sample(709, kv_len(heads_kv));
    let v_data = Domain::Wide.sample(719, kv_len(heads_kv));

    let graph = graph_of(session);
    let q = upload(graph.handle(), &dims(&q_shape()), &q_data)?;
    let k = upload(graph.handle(), &dims(&kv_shape(heads_kv)), &k_data)?;
    let v = upload(graph.handle(), &dims(&kv_shape(heads_kv)), &v_data)?;
    let o = build(&q, &k, &v).map_err(|e| -> CaseError { format!("{name}: {e}").into() })?;

    let (expected, _) = host_attention(
        &q_data,
        &k_data,
        &v_data,
        heads_kv,
        default_scale(),
        host_mask,
    );
    expect_values(session, &q_shape(), Dtype::F32, &read(&o)?, &expected)?;
    Ok(())
}

/// Measure one attention shape's dispatch count against its ceiling. The
/// values are resolved together, so `attention_with_lse` is charged for both
/// outputs.
fn launch_ceiling_case(session: &Session, name: &'static str) -> CaseResult {
    let q_data = Domain::Wide.sample(701, Q_LEN);
    let k_data = Domain::Wide.sample(709, kv_len(H));
    let v_data = Domain::Wide.sample(719, kv_len(H));

    let graph = graph_of(session);
    let q = upload(graph.handle(), &dims(&q_shape()), &q_data)?;
    let k = upload(graph.handle(), &dims(&kv_shape(H)), &k_data)?;
    let v = upload(graph.handle(), &dims(&kv_shape(H)), &v_data)?;
    let values = match name {
        "attention_with_lse" => {
            let (o, lse) = attention_with_lse(&q, &k, &v, MaskKind::None, None)
                .map_err(|e| -> CaseError { e.to_string().into() })?;
            vec![o, lse]
        }
        "attention_causal_forward" => vec![
            attention_causal(&q, &k, &v, None)
                .map_err(|e| -> CaseError { e.to_string().into() })?,
        ],
        _ => vec![
            attention(&q, &k, &v, MaskKind::None, None)
                .map_err(|e| -> CaseError { e.to_string().into() })?,
        ],
    };
    crate::launch_counts::check_ceiling(session, name, &values)?;

    let (expected, expected_lse) = host_attention(
        &q_data,
        &k_data,
        &v_data,
        H,
        default_scale(),
        if name == "attention_causal_forward" {
            &causal_mask
        } else {
            &no_mask
        },
    );
    expect_values(
        session,
        &q_shape(),
        Dtype::F32,
        &read(&values[0])?,
        &expected,
    )?;
    if let Some(lse) = values.get(1) {
        let shape = [B as u64, H as u64, LQ as u64];
        expect_values(session, &shape, Dtype::F32, &read(lse)?, &expected_lse)?;
    }
    Ok(())
}

/// Rules that must fire while saturating the chain `attention_defn` emits,
/// paired with what their absence would mean. Every entry is
/// backend-independent, so one table serves both sessions.
const REQUIRED_ON_THE_ATTENTION_CHAIN: &[(&str, &str)] = &[
    (
        "LOWER_FOLD",
        "the floor that guarantees the softmax's reductions reach a runnable form",
    ),
    (
        "LOWER_CONTRACT_GENERIC",
        "the floor that turns q.k and p.v into nests; without it there is no fold to fuse into",
    ),
    (
        "ABSORB",
        "the fusion law: a reduction absorbs a producer whose index space it covers. \
         This is what collapses the softmax chain into one lift",
    ),
    (
        "TILE_FOLD",
        "the reduction's schedule domain. Without it every fold in attention reaches \
         extraction with no schedule decision to make and the emitter's default stands",
    ),
];

/// Saturate the graph `attention_defn` emits and assert the report: it reached
/// a fixpoint, no class was truncated, and the application count is at or over
/// `FLOOR`.
fn saturation_case(session: &Session, causal: bool) -> CaseResult {
    use fusor2_ir::egraph::Saturate;
    use fusor2_ir::egraph::SaturationBudget;
    use fusor2_ir::saturate::Driver;

    /// Far under the measured application count, so it fires only when
    /// attention stops being rewritten at all.
    const FLOOR: u32 = 64;

    let q_data = Domain::Wide.sample(701, Q_LEN);
    let k_data = Domain::Wide.sample(709, kv_len(H));
    let v_data = Domain::Wide.sample(719, kv_len(H));

    let graph = graph_of(session);
    let q = upload(graph.handle(), &dims(&q_shape()), &q_data)?;
    let k = upload(graph.handle(), &dims(&kv_shape(H)), &k_data)?;
    let v = upload(graph.handle(), &dims(&kv_shape(H)), &v_data)?;
    let o = if causal {
        attention_causal(&q, &k, &v, None)
    } else {
        attention(&q, &k, &v, MaskKind::None, None)
    }
    .map_err(|e| -> CaseError { e.to_string().into() })?;

    let caps = session.caps();
    let rules = session.rules().to_vec();
    let report = graph
        .handle()
        .with_egraph(|g| {
            g.add_root(o.id());
            Driver::new()
                .saturate(g, &caps, &rules, SaturationBudget::default())
                .map_err(Into::into)
        })
        .map_err(|e| -> CaseError { format!("saturating attention: {e}").into() })?;

    if !report.saturated {
        return Err(format!(
            "attention{} did not saturate in {} rounds ({} applications, {} nodes). \
             Every structural claim below this is unreadable while it is false.",
            if causal { " (causal)" } else { "" },
            report.rounds,
            report.applications,
            report.final_nodes
        )
        .into());
    }
    if !report.truncated.is_empty() {
        return Err(format!(
            "attention{} truncated {} class(es) at {} nodes: {:?}. Truncation is never \
             silent, and on the frontend's own chain it must not happen at all.",
            if causal { " (causal)" } else { "" },
            report.truncated.len(),
            report.final_nodes,
            report.truncated
        )
        .into());
    }
    if report.applications < FLOOR {
        return Err(format!(
            "attention{} drew only {} rule applications over {} rounds, under the {FLOOR} \
             floor. A rule that silently stops matching the frontend's chain is how flash \
             attention was unreachable on both backends for a week.",
            if causal { " (causal)" } else { "" },
            report.applications,
            report.rounds
        )
        .into());
    }
    for (rule, why) in REQUIRED_ON_THE_ATTENTION_CHAIN {
        if report
            .fired
            .iter()
            .find(|(n, _)| n == rule)
            .is_none_or(|(_, n)| *n == 0)
        {
            return Err(format!(
                "`{rule}` never fired while saturating the chain `attention_defn` emits \
                 ({why}). {} applications over {} rounds, and this rule was not one of \
                 them: {:?}. A rule that silently stops matching the frontend is how flash \
                 attention was unreachable on both backends for a week while every numeric \
                 case still passed.",
                report.applications, report.rounds, report.fired
            )
            .into());
        }
    }
    // The graph must compute the right numbers after the extra saturation pass.
    let (expected, _) = host_attention(
        &q_data,
        &k_data,
        &v_data,
        H,
        default_scale(),
        if causal { &causal_mask } else { &no_mask },
    );
    expect_values(session, &q_shape(), Dtype::F32, &read(&o)?, &expected)
}

/// A scale that is not `1/sqrt(Dh)`. The scale is a runtime uniform read from
/// binding 0, not a baked literal.
fn attention_scale_case(session: &Session) -> CaseResult {
    const SCALE: f32 = 0.37;
    let q_data = Domain::Wide.sample(701, Q_LEN);
    let k_data = Domain::Wide.sample(709, kv_len(H));
    let v_data = Domain::Wide.sample(719, kv_len(H));

    let graph = graph_of(session);
    let q = upload(graph.handle(), &dims(&q_shape()), &q_data)?;
    let k = upload(graph.handle(), &dims(&kv_shape(H)), &k_data)?;
    let v = upload(graph.handle(), &dims(&kv_shape(H)), &v_data)?;
    let o = attention(&q, &k, &v, MaskKind::None, Some(SCALE))
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let (expected, _) = host_attention(&q_data, &k_data, &v_data, H, SCALE, &no_mask);
    expect_values(session, &q_shape(), Dtype::F32, &read(&o)?, &expected)?;
    Ok(())
}

/// A materialized additive `[Lq, Lk]` mask. `QkMask` is the one mask kind that
/// is not structural, so the tensor has to reach the kernel.
fn qk_mask_case(session: &Session) -> CaseResult {
    let q_data = Domain::Wide.sample(727, Q_LEN);
    let k_data = Domain::Wide.sample(733, kv_len(H));
    let v_data = Domain::Wide.sample(739, kv_len(H));
    // Banded and asymmetric, so a transposed index gives wrong numbers.
    let mask: Vec<f32> = (0..LQ * LK)
        .map(|n| {
            let (i, j) = (n / LK, n % LK);
            if j > i { -1.0e4 } else { 0.25 * (i as f32) }
        })
        .collect();

    let graph = graph_of(session);
    let q = upload(graph.handle(), &dims(&q_shape()), &q_data)?;
    let k = upload(graph.handle(), &dims(&kv_shape(H)), &k_data)?;
    let v = upload(graph.handle(), &dims(&kv_shape(H)), &v_data)?;
    let m = upload(graph.handle(), &dims(&[LQ as u64, LK as u64]), &mask)?;
    let o = attention_masked(&q, &k, &v, MaskKind::QkMask, Some(&m), None)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let (expected, _) = host_attention(&q_data, &k_data, &v_data, H, default_scale(), &|i, j| {
        mask[i * LK + j]
    });
    expect_values(session, &q_shape(), Dtype::F32, &read(&o)?, &expected)?;
    Ok(())
}

/// `QkMask` and `BatchKeyMask` without a mask tensor must be refused at
/// construction: only `None` and `Causal` are structural.
fn mask_arity(session: &Session) -> CaseResult {
    let graph = graph_of(session);
    let q = upload(
        graph.handle(),
        &dims(&q_shape()),
        &Domain::Wide.sample(743, Q_LEN),
    )?;
    let k = upload(
        graph.handle(),
        &dims(&kv_shape(H)),
        &Domain::Wide.sample(751, kv_len(H)),
    )?;
    let v = upload(
        graph.handle(),
        &dims(&kv_shape(H)),
        &Domain::Wide.sample(757, kv_len(H)),
    )?;
    for kind in [MaskKind::QkMask, MaskKind::BatchKeyMask] {
        if attention(&q, &k, &v, kind, None).is_ok() {
            return Err(format!("{kind:?} was accepted without a mask tensor").into());
        }
    }
    Ok(())
}

fn lse_case(session: &Session) -> CaseResult {
    let q_data = Domain::Wide.sample(761, Q_LEN);
    let k_data = Domain::Wide.sample(769, kv_len(H));

    let graph = graph_of(session);
    let q = upload(graph.handle(), &dims(&q_shape()), &q_data)?;
    let k = upload(graph.handle(), &dims(&kv_shape(H)), &k_data)?;
    let lse = attention_lse(&q, &k, MaskKind::None, None, None)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    // v is unused by lse; zeros satisfy the host helper's shapes.
    let v_data = vec![0.0f32; kv_len(H)];
    let (_, expected) = host_attention(&q_data, &k_data, &v_data, H, default_scale(), &no_mask);
    let shape = [B as u64, H as u64, LQ as u64];
    expect_values(session, &shape, Dtype::F32, &read(&lse)?, &expected)?;
    Ok(())
}

fn with_lse_case(session: &Session) -> CaseResult {
    let q_data = Domain::Wide.sample(773, Q_LEN);
    let k_data = Domain::Wide.sample(787, kv_len(H));
    let v_data = Domain::Wide.sample(797, kv_len(H));

    let graph = graph_of(session);
    let q = upload(graph.handle(), &dims(&q_shape()), &q_data)?;
    let k = upload(graph.handle(), &dims(&kv_shape(H)), &k_data)?;
    let v = upload(graph.handle(), &dims(&kv_shape(H)), &v_data)?;
    let (o, lse) = attention_with_lse(&q, &k, &v, MaskKind::None, None)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let (want_o, want_lse) =
        host_attention(&q_data, &k_data, &v_data, H, default_scale(), &no_mask);
    expect_values(session, &q_shape(), Dtype::F32, &read(&o)?, &want_o)?;
    let lse_shape = [B as u64, H as u64, LQ as u64];
    expect_values(session, &lse_shape, Dtype::F32, &read(&lse)?, &want_lse)?;
    Ok(())
}

/// `attention_grads` against the analytic adjoints. dk and dv are halves of one
/// `[B, H, 2*Lk, Dh]` buffer handed back as zero-cost views.
fn grads_case(session: &Session) -> CaseResult {
    let q_data = Domain::Wide.sample(809, Q_LEN);
    let k_data = Domain::Wide.sample(811, kv_len(H));
    let v_data = Domain::Wide.sample(821, kv_len(H));
    let g_data = Domain::Wide.sample(823, Q_LEN);

    let graph = graph_of(session);
    let q = upload(graph.handle(), &dims(&q_shape()), &q_data)?;
    let k = upload(graph.handle(), &dims(&kv_shape(H)), &k_data)?;
    let v = upload(graph.handle(), &dims(&kv_shape(H)), &v_data)?;
    let g = upload(graph.handle(), &dims(&q_shape()), &g_data)?;
    let (o, lse) = attention_with_lse(&q, &k, &v, MaskKind::None, None)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let (dq, dk, dv) = attention_grads(&q, &k, &v, &o, &g, &lse, MaskKind::None, None)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let (want_dq, want_dk, want_dv) =
        host_attention_grads(&q_data, &k_data, &v_data, &g_data, default_scale());
    expect_values(session, &q_shape(), Dtype::F32, &read(&dq)?, &want_dq)?;
    expect_values(session, &kv_shape(H), Dtype::F32, &read(&dk)?, &want_dk)?;
    expect_values(session, &kv_shape(H), Dtype::F32, &read(&dv)?, &want_dv)?;
    Ok(())
}

/// The derived backward's dispatch count against its ceiling, with the adjoint
/// values asserted alongside the count.
fn grads_launch_ceiling(session: &Session) -> CaseResult {
    let q_data = Domain::Wide.sample(809, Q_LEN);
    let k_data = Domain::Wide.sample(811, kv_len(H));
    let v_data = Domain::Wide.sample(821, kv_len(H));
    let g_data = Domain::Wide.sample(823, Q_LEN);

    let graph = graph_of(session);
    let q = upload(graph.handle(), &dims(&q_shape()), &q_data)?;
    let k = upload(graph.handle(), &dims(&kv_shape(H)), &k_data)?;
    let v = upload(graph.handle(), &dims(&kv_shape(H)), &v_data)?;
    let g = upload(graph.handle(), &dims(&q_shape()), &g_data)?;
    let (o, lse) = attention_with_lse(&q, &k, &v, MaskKind::None, None)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let (dq, dk, dv) = attention_grads(&q, &k, &v, &o, &g, &lse, MaskKind::None, None)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let values = [dq.clone(), dk.clone(), dv.clone()];
    crate::launch_counts::check_ceiling(session, "attention_grads_all_three", &values)?;

    let (want_dq, want_dk, want_dv) =
        host_attention_grads(&q_data, &k_data, &v_data, &g_data, default_scale());
    expect_values(session, &q_shape(), Dtype::F32, &read(&dq)?, &want_dq)?;
    expect_values(session, &kv_shape(H), Dtype::F32, &read(&dk)?, &want_dk)?;
    expect_values(session, &kv_shape(H), Dtype::F32, &read(&dv)?, &want_dv)?;
    Ok(())
}

/// Grouped queries must be expanded by the caller; `attention_grads` refuses
/// them rather than silently summing over the group.
fn grads_gqa_refused(session: &Session) -> CaseResult {
    let graph = graph_of(session);
    let q = upload(
        graph.handle(),
        &dims(&q_shape()),
        &Domain::Wide.sample(827, Q_LEN),
    )?;
    let k = upload(
        graph.handle(),
        &dims(&kv_shape(1)),
        &Domain::Wide.sample(829, kv_len(1)),
    )?;
    let v = upload(
        graph.handle(),
        &dims(&kv_shape(1)),
        &Domain::Wide.sample(839, kv_len(1)),
    )?;
    let g = upload(
        graph.handle(),
        &dims(&q_shape()),
        &Domain::Wide.sample(853, Q_LEN),
    )?;
    let o = upload(
        graph.handle(),
        &dims(&q_shape()),
        &Domain::Wide.sample(859, Q_LEN),
    )?;
    let lse = upload(
        graph.handle(),
        &dims(&[B as u64, H as u64, LQ as u64]),
        &Domain::Wide.sample(857, B * H * LQ),
    )?;
    if attention_grads(&q, &k, &v, &o, &g, &lse, MaskKind::None, None).is_ok() {
        return Err("attention_grads accepted H != Hkv; the caller must expand first".into());
    }
    Ok(())
}

/// The taped backward of the composed attention must agree with the analytic
/// adjoints.
fn attention_backward(session: &Session) -> CaseResult {
    let q_data = Domain::Wide.sample(863, Q_LEN);
    let k_data = Domain::Wide.sample(877, kv_len(H));
    let v_data = Domain::Wide.sample(881, kv_len(H));
    let ones = vec![1.0f32; Q_LEN];

    let graph = graph_of(session);
    let q = upload(graph.handle(), &dims(&q_shape()), &q_data)?;
    let k = upload(graph.handle(), &dims(&kv_shape(H)), &k_data)?;
    let v = upload(graph.handle(), &dims(&kv_shape(H)), &v_data)?;
    let o = attention(&q, &k, &v, MaskKind::None, None)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let (want_dq, want_dk, want_dv) =
        host_attention_grads(&q_data, &k_data, &v_data, &ones, default_scale());
    for (label, tensor, want) in [
        ("dq", &q, &want_dq),
        ("dk", &k, &want_dk),
        ("dv", &v, &want_dv),
    ] {
        let got = gradient_of(&graph, &o, tensor)?;
        crate::compare::approx_or_relative_eq(
            backend_of(session),
            &[want.len()],
            want,
            &got,
            1e-3,
            1e-3,
        )
        .map_err(|e| -> CaseError { format!("{label}: {e}").into() })?;
    }
    Ok(())
}

type RopeBuild = fn(&Tensor, &Tensor, &Tensor, u64) -> fusor2::Result<Tensor>;
type RopePairBuild =
    fn(&Tensor, &Tensor, &Tensor, &Tensor, u64) -> fusor2::Result<(Tensor, Tensor)>;
type RopePosBuild = fn(&Tensor, &Tensor, &Tensor, &Tensor) -> fusor2::Result<Tensor>;
type RopePosPairBuild =
    fn(&Tensor, &Tensor, &Tensor, &Tensor, &Tensor) -> fusor2::Result<(Tensor, Tensor)>;

/// Upload the sin/cos tables covering `max_len` positions, returning both the
/// device tensors and the host copies.
fn upload_tables(
    graph: &GraphRef,
    max_len: usize,
) -> Result<(Tensor, Tensor, Vec<f32>, Vec<f32>), CaseError> {
    let (cos, sin) = rope_tables(max_len);
    let shape = dims(&[max_len as u64, (DH / 2) as u64]);
    let ct = upload(graph, &shape, &cos)?;
    let st = upload(graph, &shape, &sin)?;
    Ok((ct, st, cos, sin))
}

fn rope_case(
    session: &Session,
    name: &'static str,
    interleaved: bool,
    offset: u64,
    build: RopeBuild,
) -> CaseResult {
    let x_data = Domain::Wide.sample(883, Q_LEN);
    let graph = graph_of(session);
    let (ct, st, cos, sin) = upload_tables(graph.handle(), LQ + offset as usize)?;
    let x = upload(graph.handle(), &dims(&q_shape()), &x_data)?;
    let y =
        build(&x, &ct, &st, offset).map_err(|e| -> CaseError { format!("{name}: {e}").into() })?;

    let expected = host_rope(&x_data, &cos, &sin, LQ, offset as usize, interleaved);
    expect_values(session, &q_shape(), Dtype::F32, &read(&y)?, &expected)?;
    Ok(())
}

/// q and k rotated in one dispatch. Both outputs are checked.
fn rope_pair_case(
    session: &Session,
    name: &'static str,
    interleaved: bool,
    build: RopePairBuild,
) -> CaseResult {
    let q_data = Domain::Wide.sample(887, Q_LEN);
    let k_data = Domain::Wide.sample(907, Q_LEN);
    let graph = graph_of(session);
    let (ct, st, cos, sin) = upload_tables(graph.handle(), LQ)?;
    let q = upload(graph.handle(), &dims(&q_shape()), &q_data)?;
    let k = upload(graph.handle(), &dims(&q_shape()), &k_data)?;
    let (rq, rk) =
        build(&q, &k, &ct, &st, 0).map_err(|e| -> CaseError { format!("{name}: {e}").into() })?;

    let want_q = host_rope(&q_data, &cos, &sin, LQ, 0, interleaved);
    let want_k = host_rope(&k_data, &cos, &sin, LQ, 0, interleaved);
    expect_values(session, &q_shape(), Dtype::F32, &read(&rq)?, &want_q)?;
    expect_values(session, &q_shape(), Dtype::F32, &read(&rk)?, &want_k)?;
    Ok(())
}

/// Rotate each row by the position its `u32` entry names.
fn host_rope_at(data: &[f32], cos: &[f32], sin: &[f32], pos: &[u32], il: bool) -> Vec<f32> {
    let mut expected = vec![0.0f32; data.len()];
    for b in 0..B {
        for h in 0..H {
            for l in 0..LQ {
                let base = ((b * H + h) * LQ + l) * DH;
                let rot = host_rope_vec(&data[base..base + DH], cos, sin, pos[l] as usize, il);
                expected[base..base + DH].copy_from_slice(&rot);
            }
        }
    }
    expected
}

/// The decode form: positions live in a rank-1 `u32` tensor so the offset
/// never round-trips through the host. The positions are not `0..Lq`.
fn rope_position_case(
    session: &Session,
    name: &'static str,
    interleaved: bool,
    build: RopePosBuild,
) -> CaseResult {
    let x_data = Domain::Wide.sample(911, Q_LEN);
    let positions: Vec<u32> = vec![3, 0, 5];
    let graph = graph_of(session);
    let (ct, st, cos, sin) = upload_tables(graph.handle(), 8)?;
    let x = upload(graph.handle(), &dims(&q_shape()), &x_data)?;
    let p = from_u32(graph.handle(), &dims(&[LQ as u64]), &positions)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let y = build(&x, &ct, &st, &p).map_err(|e| -> CaseError { format!("{name}: {e}").into() })?;

    let expected = host_rope_at(&x_data, &cos, &sin, &positions, interleaved);
    expect_values(session, &q_shape(), Dtype::F32, &read(&y)?, &expected)?;
    Ok(())
}

fn rope_position_pair_case(
    session: &Session,
    name: &'static str,
    interleaved: bool,
    build: RopePosPairBuild,
) -> CaseResult {
    let q_data = Domain::Wide.sample(919, Q_LEN);
    let k_data = Domain::Wide.sample(929, Q_LEN);
    let positions: Vec<u32> = vec![1, 4, 2];
    let graph = graph_of(session);
    let (ct, st, cos, sin) = upload_tables(graph.handle(), 8)?;
    let q = upload(graph.handle(), &dims(&q_shape()), &q_data)?;
    let k = upload(graph.handle(), &dims(&q_shape()), &k_data)?;
    let p = from_u32(graph.handle(), &dims(&[LQ as u64]), &positions)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let (rq, rk) =
        build(&q, &k, &ct, &st, &p).map_err(|e| -> CaseError { format!("{name}: {e}").into() })?;

    for (data, got) in [(&q_data, &rq), (&k_data, &rk)] {
        let expected = host_rope_at(data, &cos, &sin, &positions, interleaved);
        expect_values(session, &q_shape(), Dtype::F32, &read(got)?, &expected)?;
    }
    Ok(())
}

/// `rotate_half(x) = cat(-x2, x1)` over the head axis.
fn rotate_half_case(session: &Session) -> CaseResult {
    let x_data = Domain::Wide.sample(937, Q_LEN);
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(&q_shape()), &x_data)?;
    let y = rotate_half(&x).map_err(|e| -> CaseError { e.to_string().into() })?;

    let half = DH / 2;
    let mut expected = vec![0.0f32; Q_LEN];
    for base in (0..Q_LEN).step_by(DH) {
        for i in 0..half {
            expected[base + i] = -x_data[base + half + i];
            expected[base + half + i] = x_data[base + i];
        }
    }
    expect_values(session, &q_shape(), Dtype::F32, &read(&y)?, &expected)?;
    Ok(())
}

/// A rotation preserves the norm of every `(a, b)` pair, hence of the whole
/// head vector. Independent of the sin/cos table.
fn rope_norm_preserving(session: &Session) -> CaseResult {
    let x_data = Domain::Wide.sample(941, Q_LEN);
    let graph = graph_of(session);
    let (ct, st, _, _) = upload_tables(graph.handle(), LQ)?;
    let x = upload(graph.handle(), &dims(&q_shape()), &x_data)?;
    let y = rope(&x, &ct, &st, 0).map_err(|e| -> CaseError { e.to_string().into() })?;
    let got = read(&y)?;
    for (head, chunk) in got.chunks(DH).enumerate() {
        let src = &x_data[head * DH..head * DH + DH];
        let a: f32 = chunk.iter().map(|v| v * v).sum();
        let b: f32 = src.iter().map(|v| v * v).sum();
        if (a - b).abs() > 1e-3 * b.max(1.0) {
            return Err(format!(
                "rope changed head {head}'s squared norm from {b} to {a}: a rotation cannot"
            )
            .into());
        }
    }
    Ok(())
}

/// The adjoint of a rotation by theta is the rotation by -theta. Under an
/// all-ones seed that gives `d/dx_a = cos + sin` and `d/dx_b = cos - sin`,
/// checked analytically rather than by finite difference.
fn rope_backward(session: &Session) -> CaseResult {
    let x_data = Domain::Wide.sample(947, Q_LEN);
    let graph = graph_of(session);
    let (ct, st, cos, sin) = upload_tables(graph.handle(), LQ)?;
    let x = upload(graph.handle(), &dims(&q_shape()), &x_data)?;
    let y = rope(&x, &ct, &st, 0).map_err(|e| -> CaseError { e.to_string().into() })?;
    let got = gradient_of(&graph, &y, &x)?;

    let half = DH / 2;
    let mut want = vec![0.0f32; Q_LEN];
    for b in 0..B {
        for h in 0..H {
            for l in 0..LQ {
                let base = ((b * H + h) * LQ + l) * DH;
                for i in 0..half {
                    let (c, s) = (cos[l * half + i], sin[l * half + i]);
                    want[base + i] = c + s;
                    want[base + half + i] = c - s;
                }
            }
        }
    }
    crate::compare::approx_or_relative_eq(backend_of(session), &[Q_LEN], &want, &got, 1e-4, 1e-4)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registered() -> Vec<String> {
        cases().names().iter().map(|n| (*n).to_string()).collect()
    }

    fn has(names: &[String], wanted: &str) -> bool {
        names
            .iter()
            .any(|n| n == &format!("attention_rope::{wanted}"))
    }

    #[test]
    fn every_attention_form_is_registered() {
        let names = registered();
        for wanted in [
            "attention",
            "attention_causal",
            "attention_gqa",
            "attention_mqa_single_kv_head",
            "attention_qk_mask",
            "attention_lse",
            "attention_with_lse",
            "attention_grads",
            "attention_explicit_scale",
        ] {
            assert!(has(&names, wanted), "{wanted} is missing");
        }
    }

    #[test]
    fn every_rope_form_is_registered() {
        let names = registered();
        for wanted in [
            "rope",
            "rope_interleaved",
            "rope_fused",
            "rope_normal_fused",
            "rope_pair_fused",
            "rope_normal_pair_fused",
            "rope_pair_fused_with_position",
            "rope_normal_pair_fused_with_position",
            "rope_with_position",
            "rope_interleaved_with_position",
            "rotate_half",
        ] {
            assert!(has(&names, wanted), "{wanted} is missing");
        }
    }

    #[test]
    fn the_host_attention_rows_are_convex_combinations_of_v() {
        // With q = 0 every score is equal, so each output row is the mean of v
        // and the log-sum-exp is ln(Lk).
        let q = vec![0.0f32; Q_LEN];
        let k = vec![1.0f32; kv_len(H)];
        let v: Vec<f32> = (0..kv_len(H)).map(|i| (i % 7) as f32).collect();
        let (out, lse) = host_attention(&q, &k, &v, H, default_scale(), &no_mask);
        for b in 0..B {
            for h in 0..H {
                for d in 0..DH {
                    let want: f32 = (0..LK)
                        .map(|j| v[((b * H + h) * LK + j) * DH + d])
                        .sum::<f32>()
                        / LK as f32;
                    let got = out[((b * H + h) * LQ) * DH + d];
                    assert!((got - want).abs() < 1e-5, "{got} vs {want}");
                }
            }
        }
        for value in lse {
            assert!((value - (LK as f32).ln()).abs() < 1e-5, "{value}");
        }
    }

    #[test]
    fn the_causal_mask_is_right_aligned() {
        // Lq = 3, Lk = 4: query 0 sees keys 0..=1, query 2 sees all four.
        assert_eq!(causal_mask(0, 1), 0.0);
        assert!(causal_mask(0, 2).is_infinite());
        assert_eq!(causal_mask(2, 3), 0.0);
    }

    #[test]
    fn the_gqa_expansion_shares_one_kv_head() {
        let q = vec![0.0f32; Q_LEN];
        let k = vec![1.0f32; kv_len(1)];
        let v: Vec<f32> = (0..kv_len(1)).map(|i| i as f32).collect();
        let (out, _) = host_attention(&q, &k, &v, 1, default_scale(), &no_mask);
        for b in 0..B {
            for d in 0..DH {
                let h0 = out[(b * H * LQ) * DH + d];
                let h1 = out[((b * H + 1) * LQ) * DH + d];
                assert!((h0 - h1).abs() < 1e-5, "GQA heads disagree: {h0} vs {h1}");
            }
        }
    }

    #[test]
    fn the_host_rope_is_a_rotation() {
        let (cos, sin) = rope_tables(4);
        let x: Vec<f32> = (0..DH).map(|i| (i + 1) as f32).collect();
        for il in [false, true] {
            for p in 0..4 {
                let y = host_rope_vec(&x, &cos, &sin, p, il);
                let a: f32 = x.iter().map(|v| v * v).sum();
                let b: f32 = y.iter().map(|v| v * v).sum();
                assert!((a - b).abs() < 1e-4, "p={p} il={il}: {a} vs {b}");
            }
        }
        // Position 0 is the identity: cos 0 = 1, sin 0 = 0.
        assert_eq!(host_rope_vec(&x, &cos, &sin, 0, false), x);
    }

    #[test]
    fn the_two_pairings_are_different_functions() {
        let (cos, sin) = rope_tables(3);
        let x: Vec<f32> = (0..DH).map(|i| (i + 1) as f32).collect();
        assert_ne!(
            host_rope_vec(&x, &cos, &sin, 1, false),
            host_rope_vec(&x, &cos, &sin, 1, true),
        );
    }

    #[test]
    fn the_inverse_frequency_table_is_decreasing_and_half_width() {
        let inv = base_inverse_frequency(DH as u32, 10_000.0);
        assert_eq!(inv.len(), DH / 2);
        assert!(inv.windows(2).all(|w| w[0] > w[1]), "{inv:?}");
        assert!((inv[0] - 1.0).abs() < 1e-6, "the first frequency is 1");
    }

    #[test]
    fn host_rope_at_reads_the_position_vector() {
        let (cos, sin) = rope_tables(8);
        let data: Vec<f32> = (0..Q_LEN).map(|i| i as f32).collect();
        // Position 0 everywhere is the identity.
        assert_eq!(host_rope_at(&data, &cos, &sin, &[0, 0, 0], false), data);
        // A different position vector must change the answer.
        assert_ne!(host_rope_at(&data, &cos, &sin, &[3, 0, 5], false), data);
    }

    #[test]
    fn the_host_grads_agree_with_a_finite_difference_of_the_host_forward() {
        let q: Vec<f32> = (0..Q_LEN).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect();
        let k: Vec<f32> = (0..kv_len(H))
            .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
            .collect();
        let v: Vec<f32> = (0..kv_len(H))
            .map(|i| ((i % 3) as f32 - 1.0) * 0.1)
            .collect();
        let g = vec![1.0f32; Q_LEN];
        let s = default_scale();
        let (dq, dk, dv) = host_attention_grads(&q, &k, &v, &g, s);
        let eps = 1e-3f32;
        let sum = |q: &[f32], k: &[f32], v: &[f32]| -> f32 {
            host_attention(q, k, v, H, s, &no_mask).0.iter().sum()
        };
        for probe in [0usize, 5, 17] {
            for (label, base, analytic) in [("dq", &q, &dq), ("dk", &k, &dk), ("dv", &v, &dv)] {
                let (mut hi, mut lo) = (base.clone(), base.clone());
                hi[probe] += eps;
                lo[probe] -= eps;
                let numeric = match label {
                    "dq" => (sum(&hi, &k, &v) - sum(&lo, &k, &v)) / (2.0 * eps),
                    "dk" => (sum(&q, &hi, &v) - sum(&q, &lo, &v)) / (2.0 * eps),
                    _ => (sum(&q, &k, &hi) - sum(&q, &k, &lo)) / (2.0 * eps),
                };
                assert!(
                    (numeric - analytic[probe]).abs() < 1e-2,
                    "{label}[{probe}]: analytic {} vs numeric {numeric}",
                    analytic[probe]
                );
            }
        }
    }
}
