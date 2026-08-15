//! Attention (dense, causal, masked, GQA/MQA, lse, grads) and the RoPE family.
//!
//! Every attention case is checked against a host implementation that spells
//! out the softmax explicitly, so which nest extraction picked is invisible to
//! the case and visible only in the numbers. `MaskKind::Causal` is *structural*: no mask tensor is
//! uploaded, so a lowering that silently needs one fails here rather than
//! reading garbage.

use fusor2::composite::attention::{
    attention, attention_causal, attention_grads, attention_lse, attention_masked,
    attention_with_lse,
};
use fusor2::composite::rope::{
    base_inverse_frequency, rope, rope_interleaved, rope_interleaved_pair,
    rope_interleaved_pair_with_position, rope_interleaved_with_position, rope_pair,
    rope_pair_with_position, rope_with_position, rotate_half,
};
use fusor2::graph::GraphRef;
use fusor2::{Dtype, Session, };
use fusor2::tensor::Dyn as Tensor;
use fusor2_ir::ir::level1::MaskKind;

use crate::harness::{CaseError, CaseResult, Cases, FuzzDim, Rng, dims, fill_indices, from_u32, fuzz_case};
use crate::suite::support::{Domain, expect_values, gradient_of, graph_of, read, upload};

/// One sampled attention problem. `dh` is even because every RoPE pairing
/// needs it to be, and `lq` and `lk` are sampled independently where legal so
/// a transposed score index cannot pass.
#[derive(Copy, Clone)]
struct AttnDims {
    b: usize,
    h: usize,
    heads_kv: usize,
    lq: usize,
    lk: usize,
    dh: usize,
}

impl AttnDims {
    fn q_shape(&self) -> Vec<u64> {
        vec![self.b as u64, self.h as u64, self.lq as u64, self.dh as u64]
    }

    fn kv_shape(&self) -> Vec<u64> {
        vec![
            self.b as u64,
            self.heads_kv as u64,
            self.lk as u64,
            self.dh as u64,
        ]
    }

    fn lse_shape(&self) -> Vec<u64> {
        vec![self.b as u64, self.h as u64, self.lq as u64]
    }

    fn q_len(&self) -> usize {
        self.b * self.h * self.lq * self.dh
    }

    fn kv_len(&self) -> usize {
        self.b * self.heads_kv * self.lk * self.dh
    }

    /// `1 / sqrt(Dh)`, the scale every case leaves to the default.
    fn default_scale(&self) -> f32 {
        1.0 / (self.dh as f32).sqrt()
    }
}

/// `[B, H, Lq, Lk, Dh]`, all heads shared with kv. `Lq` and `Lk` are
/// independent, which is legal for every non-causal mask.
const ATTN_SPEC: &[FuzzDim] = &[
    FuzzDim::Range(1, 3),
    FuzzDim::Choices(&[1, 2, 4]),
    FuzzDim::Range(1, 12),
    FuzzDim::Range(1, 12),
    FuzzDim::Mult(2, 2, 12),
];

/// `[B, H, Lq, Lk - Lq, Dh]`: the right-aligned causal mask needs
/// `Lk >= Lq` or the first query would see no keys.
const CAUSAL_SPEC: &[FuzzDim] = &[
    FuzzDim::Range(1, 3),
    FuzzDim::Choices(&[1, 2, 4]),
    FuzzDim::Range(1, 12),
    FuzzDim::Range(0, 8),
    FuzzDim::Mult(2, 2, 12),
];

/// `[B, Hkv, groups, Lq, Lk, Dh]`: `H = Hkv * groups`, so the kv head count
/// always divides the query head count.
const GQA_SPEC: &[FuzzDim] = &[
    FuzzDim::Range(1, 3),
    FuzzDim::Choices(&[1, 2]),
    FuzzDim::Choices(&[2, 4]),
    FuzzDim::Range(1, 12),
    FuzzDim::Range(1, 12),
    FuzzDim::Mult(2, 2, 12),
];

/// The analytic-gradient cases stay small: the host adjoint is O(everything)
/// and the tolerances tighten as the sums grow.
const GRADS_SPEC: &[FuzzDim] = &[
    FuzzDim::Range(1, 2),
    FuzzDim::Choices(&[1, 2]),
    FuzzDim::Range(1, 4),
    FuzzDim::Range(1, 6),
    FuzzDim::Mult(2, 2, 8),
];

fn dense_dims(shape: &[u64]) -> AttnDims {
    AttnDims {
        b: shape[0] as usize,
        h: shape[1] as usize,
        heads_kv: shape[1] as usize,
        lq: shape[2] as usize,
        lk: shape[3] as usize,
        dh: shape[4] as usize,
    }
}

/// `CAUSAL_SPEC`'s fourth entry is the key surplus, not `Lk` itself.
fn causal_dims(shape: &[u64], heads_kv_is_one: bool) -> AttnDims {
    let h = shape[1] as usize;
    AttnDims {
        b: shape[0] as usize,
        h,
        heads_kv: if heads_kv_is_one { 1 } else { h },
        lq: shape[2] as usize,
        lk: (shape[2] + shape[3]) as usize,
        dh: shape[4] as usize,
    }
}

fn gqa_dims(shape: &[u64]) -> AttnDims {
    AttnDims {
        b: shape[0] as usize,
        h: (shape[1] * shape[2]) as usize,
        heads_kv: shape[1] as usize,
        lq: shape[3] as usize,
        lk: shape[4] as usize,
        dh: shape[5] as usize,
    }
}

/// The fixed problem the refusal cases use: refusals are about arity, not
/// extents, so nothing is sampled.
const REFUSAL_DIMS: AttnDims = AttnDims {
    b: 2,
    h: 2,
    heads_kv: 2,
    lq: 3,
    lk: 4,
    dh: 4,
};

fn backend_of(session: &Session) -> &'static str {
    if crate::harness::is_gpu(session) {
        "gpu"
    } else {
        "cpu"
    }
}

// ---------------------------------------------------------------------------
// Host attention
// ---------------------------------------------------------------------------

/// `[B, H, Lq, Dh]` output and `[B, H, Lq]` log-sum-exp.
///
/// `heads_kv` may be smaller than `H`; query head `h` reads kv head
/// `h / (H / heads_kv)`, which is the GQA expansion. `mask(qi, ki)` is the
/// additive score bias.
fn host_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d: AttnDims,
    scale: f32,
    mask: &dyn Fn(usize, usize) -> f32,
) -> (Vec<f32>, Vec<f32>) {
    let AttnDims {
        b: bs,
        h: hs,
        heads_kv,
        lq,
        lk,
        dh,
        ..
    } = d;
    let groups = hs / heads_kv;
    let mut out = vec![0.0f32; bs * hs * lq * dh];
    let mut lse = vec![0.0f32; bs * hs * lq];
    for b in 0..bs {
        for h in 0..hs {
            let hk = h / groups;
            for i in 0..lq {
                let qbase = ((b * hs + h) * lq + i) * dh;
                let mut scores = vec![0.0f32; lk];
                for (j, s) in scores.iter_mut().enumerate() {
                    let kbase = ((b * heads_kv + hk) * lk + j) * dh;
                    let dot: f32 = (0..dh).map(|x| q[qbase + x] * k[kbase + x]).sum();
                    *s = dot * scale + mask(i, j);
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum: f32 = e.iter().sum();
                lse[(b * hs + h) * lq + i] = max + sum.ln();
                for x in 0..dh {
                    let mut acc = 0.0f32;
                    for (j, ej) in e.iter().enumerate() {
                        let vbase = ((b * heads_kv + hk) * lk + j) * dh;
                        acc += (ej / sum) * v[vbase + x];
                    }
                    out[qbase + x] = acc;
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
/// `i + (Lk - Lq)`, which is the decode-time convention. Needs `Lk >= Lq`.
fn causal_mask(lq: usize, lk: usize, i: usize, j: usize) -> f32 {
    if j + lq <= i + lk {
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
    d: AttnDims,
    scale: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let AttnDims {
        b: bs,
        h: hs,
        lq,
        lk,
        dh,
        ..
    } = d;
    let mut dq = vec![0.0f32; q.len()];
    let mut dk = vec![0.0f32; k.len()];
    let mut dv = vec![0.0f32; v.len()];
    for b in 0..bs {
        for h in 0..hs {
            for i in 0..lq {
                let qb = ((b * hs + h) * lq + i) * dh;
                let mut p = vec![0.0f32; lk];
                for (j, s) in p.iter_mut().enumerate() {
                    let kb = ((b * hs + h) * lk + j) * dh;
                    *s = (0..dh).map(|x| q[qb + x] * k[kb + x]).sum::<f32>() * scale;
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
                let mut dp = vec![0.0f32; lk];
                for (j, dpj) in dp.iter_mut().enumerate() {
                    let vb = ((b * hs + h) * lk + j) * dh;
                    *dpj = (0..dh).map(|x| g[qb + x] * v[vb + x]).sum();
                    for x in 0..dh {
                        dv[vb + x] += p[j] * g[qb + x];
                    }
                }
                let dot: f32 = p.iter().zip(&dp).map(|(a, b)| a * b).sum();
                for j in 0..lk {
                    let ds = p[j] * (dp[j] - dot) * scale;
                    let kb = ((b * hs + h) * lk + j) * dh;
                    for x in 0..dh {
                        dq[qb + x] += ds * k[kb + x];
                        dk[kb + x] += ds * q[qb + x];
                    }
                }
            }
        }
    }
    (dq, dk, dv)
}

// ---------------------------------------------------------------------------
// Host rope
// ---------------------------------------------------------------------------

/// The sampled `[B, H, L, Dh]` a rope case rotates. `dh` is even.
#[derive(Copy, Clone)]
struct RopeDims {
    b: usize,
    h: usize,
    l: usize,
    dh: usize,
}

impl RopeDims {
    fn shape(&self) -> Vec<u64> {
        vec![self.b as u64, self.h as u64, self.l as u64, self.dh as u64]
    }

    fn len(&self) -> usize {
        self.b * self.h * self.l * self.dh
    }
}

const ROPE_SPEC: &[FuzzDim] = &[
    FuzzDim::Range(1, 3),
    FuzzDim::Range(1, 4),
    FuzzDim::Range(1, 8),
    FuzzDim::Mult(2, 2, 12),
];

fn rope_dims(shape: &[u64]) -> RopeDims {
    RopeDims {
        b: shape[0] as usize,
        h: shape[1] as usize,
        l: shape[2] as usize,
        dh: shape[3] as usize,
    }
}

/// The rotation applied to one `[Dh]` head vector at position `p`.
/// `interleaved` pairs `(2i, 2i+1)`; otherwise pairs `(i, i + Dh/2)`.
fn host_rope_vec(
    x: &[f32],
    cos: &[f32],
    sin: &[f32],
    p: usize,
    dh: usize,
    interleaved: bool,
) -> Vec<f32> {
    let half = dh / 2;
    let mut out = vec![0.0f32; dh];
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
fn host_rope(x: &[f32], cos: &[f32], sin: &[f32], d: RopeDims, offset: usize, il: bool) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for b in 0..d.b {
        for h in 0..d.h {
            for l in 0..d.l {
                let base = ((b * d.h + h) * d.l + l) * d.dh;
                let rotated =
                    host_rope_vec(&x[base..base + d.dh], cos, sin, offset + l, d.dh, il);
                out[base..base + d.dh].copy_from_slice(&rotated);
            }
        }
    }
    out
}

/// The `[max_len, Dh/2]` sin/cos tables the rope cases upload.
fn rope_tables(dh: usize, max_len: usize) -> (Vec<f32>, Vec<f32>) {
    let inv = base_inverse_frequency(dh as u32, 10_000.0);
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

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn cases() -> Cases {
    let mut cases = Cases::new();

    cases.push_case(fuzz_case("attention_rope", "attention", ATTN_SPEC, |s, shape, seed| {
        attention_case(s, seed, "attention", dense_dims(shape), &no_mask, |q, k, v| {
            attention(q, k, v, MaskKind::None, None)
        })
    }));
    cases.push_case(fuzz_case(
        "attention_rope",
        "attention_causal",
        CAUSAL_SPEC,
        |s, shape, seed| {
            let d = causal_dims(shape, false);
            attention_case(
                s,
                seed,
                "attention_causal",
                d,
                &|i, j| causal_mask(d.lq, d.lk, i, j),
                |q, k, v| attention_causal(q, k, v, None),
            )
        },
    ));
    cases.push_case(fuzz_case(
        "attention_rope",
        "attention_causal_via_mask_kind",
        CAUSAL_SPEC,
        |s, shape, seed| {
            let d = causal_dims(shape, false);
            attention_case(
                s,
                seed,
                "attention_causal_via_mask_kind",
                d,
                &|i, j| causal_mask(d.lq, d.lk, i, j),
                |q, k, v| attention(q, k, v, MaskKind::Causal, None),
            )
        },
    ));
    cases.push_case(fuzz_case(
        "attention_rope",
        "attention_explicit_scale",
        ATTN_SPEC,
        |s, shape, seed| attention_scale_case(s, dense_dims(shape), seed),
    ));
    cases.push_case(fuzz_case(
        "attention_rope",
        "attention_gqa",
        GQA_SPEC,
        |s, shape, seed| {
            attention_case(s, seed, "attention_gqa", gqa_dims(shape), &no_mask, |q, k, v| {
                attention(q, k, v, MaskKind::None, None)
            })
        },
    ));
    cases.push_case(fuzz_case(
        "attention_rope",
        "attention_mqa_single_kv_head",
        CAUSAL_SPEC,
        |s, shape, seed| {
            let d = causal_dims(shape, true);
            attention_case(
                s,
                seed,
                "attention_mqa_single_kv_head",
                d,
                &|i, j| causal_mask(d.lq, d.lk, i, j),
                |q, k, v| attention_causal(q, k, v, None),
            )
        },
    ));

    cases.push_case(fuzz_case(
        "attention_rope",
        "attention_qk_mask",
        ATTN_SPEC,
        |s, shape, seed| qk_mask_case(s, dense_dims(shape), seed),
    ));
    cases.push(
        "attention_rope",
        "attention_refuses_a_tensor_mask_kind_without_a_tensor",
        mask_arity,
    );
    cases.push_case(fuzz_case(
        "attention_rope",
        "attention_lse",
        ATTN_SPEC,
        |s, shape, seed| lse_case(s, dense_dims(shape), seed),
    ));
    cases.push_case(fuzz_case(
        "attention_rope",
        "attention_with_lse",
        ATTN_SPEC,
        |s, shape, seed| with_lse_case(s, dense_dims(shape), seed),
    ));
    cases.push_case(fuzz_case(
        "attention_rope",
        "attention_grads",
        GRADS_SPEC,
        |s, shape, seed| grads_case(s, dense_dims(shape), seed),
    ));
    cases.push(
        "attention_rope",
        "attention_grads_refuse_grouped_heads",
        grads_gqa_refused,
    );
    cases.push_case(fuzz_case(
        "attention_rope",
        "attention_backward_matches_the_analytic_adjoints",
        GRADS_SPEC,
        |s, shape, seed| attention_backward(s, dense_dims(shape), seed),
    ));

    // RoPE. Every spelling is checked against the same host rotation, so an
    // alias that quietly picked the other pairing is a value failure.
    cases.push_case(fuzz_case("attention_rope", "rope", ROPE_SPEC, |s, shape, seed| {
        rope_case(s, seed, "rope", rope_dims(shape), false, 0, rope)
    }));
    cases.push_case(fuzz_case(
        "attention_rope",
        "rope_interleaved",
        ROPE_SPEC,
        |s, shape, seed| {
            rope_case(s, seed, "rope_interleaved", rope_dims(shape), true, 0, rope_interleaved)
        },
    ));
    cases.push_case(fuzz_case(
        "attention_rope",
        "rope_offset",
        ROPE_SPEC,
        |s, shape, seed| {
            // The offset is sampled apart from the shape stream, and nonzero
            // so the case never degenerates into plain `rope`.
            let offset = Rng::new(seed ^ 0x5eed).range(1, 6);
            rope_case(s, seed, "rope_offset", rope_dims(shape), false, offset, rope)
        },
    ));
    cases.push_case(fuzz_case(
        "attention_rope",
        "rope_pair",
        ROPE_SPEC,
        |s, shape, seed| rope_pair_case(s, seed, "rope_pair", rope_dims(shape), false, rope_pair),
    ));
    cases.push_case(fuzz_case(
        "attention_rope",
        "rope_interleaved_pair",
        ROPE_SPEC,
        |s, shape, seed| {
            rope_pair_case(
                s,
                seed,
                "rope_interleaved_pair",
                rope_dims(shape),
                true,
                rope_interleaved_pair,
            )
        },
    ));
    cases.push_case(fuzz_case(
        "attention_rope",
        "rope_pair_with_position",
        ROPE_SPEC,
        |s, shape, seed| {
            rope_position_pair_case(
                s,
                seed,
                "rope_pair_with_position",
                rope_dims(shape),
                false,
                rope_pair_with_position,
            )
        },
    ));
    cases.push_case(fuzz_case(
        "attention_rope",
        "rope_interleaved_pair_with_position",
        ROPE_SPEC,
        |s, shape, seed| {
            rope_position_pair_case(
                s,
                seed,
                "rope_interleaved_pair_with_position",
                rope_dims(shape),
                true,
                rope_interleaved_pair_with_position,
            )
        },
    ));
    cases.push_case(fuzz_case(
        "attention_rope",
        "rope_with_position",
        ROPE_SPEC,
        |s, shape, seed| {
            rope_position_case(
                s,
                seed,
                "rope_with_position",
                rope_dims(shape),
                false,
                rope_with_position,
            )
        },
    ));
    cases.push_case(fuzz_case(
        "attention_rope",
        "rope_interleaved_with_position",
        ROPE_SPEC,
        |s, shape, seed| {
            rope_position_case(
                s,
                seed,
                "rope_interleaved_with_position",
                rope_dims(shape),
                true,
                rope_interleaved_with_position,
            )
        },
    ));
    cases.push_case(fuzz_case(
        "attention_rope",
        "rotate_half",
        ROPE_SPEC,
        |s, shape, seed| rotate_half_case(s, rope_dims(shape), seed),
    ));
    cases.push_case(fuzz_case(
        "attention_rope",
        "rope_is_norm_preserving",
        ROPE_SPEC,
        |s, shape, seed| rope_norm_preserving(s, rope_dims(shape), seed),
    ));
    cases.push_case(fuzz_case(
        "attention_rope",
        "rope_backward_is_the_transposed_rotation",
        ROPE_SPEC,
        |s, shape, seed| rope_backward(s, rope_dims(shape), seed),
    ));
    cases
}

// ---------------------------------------------------------------------------
// Attention cases
// ---------------------------------------------------------------------------

type AttnBuild = fn(&Tensor, &Tensor, &Tensor) -> fusor2::Result<Tensor>;

fn attention_case(
    session: &Session,
    seed: u32,
    name: &'static str,
    d: AttnDims,
    host_mask: &dyn Fn(usize, usize) -> f32,
    build: AttnBuild,
) -> CaseResult {
    let q_data = Domain::Wide.sample(seed, d.q_len());
    let k_data = Domain::Wide.sample(seed ^ 0x9e37_79b9, d.kv_len());
    let v_data = Domain::Wide.sample(seed.wrapping_add(1), d.kv_len());

    let graph = graph_of(session);
    let q = upload(graph.handle(), &dims(&d.q_shape()), &q_data)?;
    let k = upload(graph.handle(), &dims(&d.kv_shape()), &k_data)?;
    let v = upload(graph.handle(), &dims(&d.kv_shape()), &v_data)?;
    let o = build(&q, &k, &v).map_err(|e| -> CaseError { format!("{name}: {e}").into() })?;

    let (expected, _) = host_attention(&q_data, &k_data, &v_data, d, d.default_scale(), host_mask);
    expect_values(session, &d.q_shape(), Dtype::F32, &read(&o)?, &expected)?;
    Ok(())
}

/// A scale that is not `1/sqrt(Dh)`. The scale is a runtime uniform read from
/// binding 0, not a baked literal, so passing a different one must change the
/// numbers without rebuilding a kernel.
fn attention_scale_case(session: &Session, d: AttnDims, seed: u32) -> CaseResult {
    const SCALE: f32 = 0.37;
    let q_data = Domain::Wide.sample(seed, d.q_len());
    let k_data = Domain::Wide.sample(seed ^ 0x9e37_79b9, d.kv_len());
    let v_data = Domain::Wide.sample(seed.wrapping_add(1), d.kv_len());

    let graph = graph_of(session);
    let q = upload(graph.handle(), &dims(&d.q_shape()), &q_data)?;
    let k = upload(graph.handle(), &dims(&d.kv_shape()), &k_data)?;
    let v = upload(graph.handle(), &dims(&d.kv_shape()), &v_data)?;
    let o = attention(&q, &k, &v, MaskKind::None, Some(SCALE))
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let (expected, _) = host_attention(&q_data, &k_data, &v_data, d, SCALE, &no_mask);
    expect_values(session, &d.q_shape(), Dtype::F32, &read(&o)?, &expected)?;
    Ok(())
}

/// A materialized additive `[Lq, Lk]` mask. `QkMask` is the one mask kind that
/// is *not* structural, so the tensor has to reach the kernel.
fn qk_mask_case(session: &Session, d: AttnDims, seed: u32) -> CaseResult {
    let q_data = Domain::Wide.sample(seed, d.q_len());
    let k_data = Domain::Wide.sample(seed ^ 0x9e37_79b9, d.kv_len());
    let v_data = Domain::Wide.sample(seed.wrapping_add(1), d.kv_len());
    // Banded and asymmetric, so a transposed index shows up as wrong numbers.
    let mask: Vec<f32> = (0..d.lq * d.lk)
        .map(|n| {
            let (i, j) = (n / d.lk, n % d.lk);
            if j > i { -1.0e4 } else { 0.25 * (i as f32) }
        })
        .collect();

    let graph = graph_of(session);
    let q = upload(graph.handle(), &dims(&d.q_shape()), &q_data)?;
    let k = upload(graph.handle(), &dims(&d.kv_shape()), &k_data)?;
    let v = upload(graph.handle(), &dims(&d.kv_shape()), &v_data)?;
    let m = upload(graph.handle(), &dims(&[d.lq as u64, d.lk as u64]), &mask)?;
    let o = attention_masked(&q, &k, &v, MaskKind::QkMask, Some(&m), None)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let (expected, _) = host_attention(&q_data, &k_data, &v_data, d, d.default_scale(), &|i, j| {
        mask[i * d.lk + j]
    });
    expect_values(session, &d.q_shape(), Dtype::F32, &read(&o)?, &expected)?;
    Ok(())
}

/// `QkMask` and `BatchKeyMask` without a mask tensor must be refused at
/// construction: only `None` and `Causal` are structural.
fn mask_arity(session: &Session) -> CaseResult {
    let d = REFUSAL_DIMS;
    let graph = graph_of(session);
    let q = upload(
        graph.handle(),
        &dims(&d.q_shape()),
        &Domain::Wide.sample(743, d.q_len()),
    )?;
    let k = upload(
        graph.handle(),
        &dims(&d.kv_shape()),
        &Domain::Wide.sample(751, d.kv_len()),
    )?;
    let v = upload(
        graph.handle(),
        &dims(&d.kv_shape()),
        &Domain::Wide.sample(757, d.kv_len()),
    )?;
    for kind in [MaskKind::QkMask, MaskKind::BatchKeyMask] {
        if attention(&q, &k, &v, kind, None).is_ok() {
            return Err(format!("{kind:?} was accepted without a mask tensor").into());
        }
    }
    Ok(())
}

fn lse_case(session: &Session, d: AttnDims, seed: u32) -> CaseResult {
    let q_data = Domain::Wide.sample(seed, d.q_len());
    let k_data = Domain::Wide.sample(seed ^ 0x9e37_79b9, d.kv_len());

    let graph = graph_of(session);
    let q = upload(graph.handle(), &dims(&d.q_shape()), &q_data)?;
    let k = upload(graph.handle(), &dims(&d.kv_shape()), &k_data)?;
    let lse = attention_lse(&q, &k, MaskKind::None, None, None)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    // v is unused by lse; zeros keep the host helper's shapes honest.
    let v_data = vec![0.0f32; d.kv_len()];
    let (_, expected) = host_attention(&q_data, &k_data, &v_data, d, d.default_scale(), &no_mask);
    expect_values(session, &d.lse_shape(), Dtype::F32, &read(&lse)?, &expected)?;
    Ok(())
}

fn with_lse_case(session: &Session, d: AttnDims, seed: u32) -> CaseResult {
    let q_data = Domain::Wide.sample(seed, d.q_len());
    let k_data = Domain::Wide.sample(seed ^ 0x9e37_79b9, d.kv_len());
    let v_data = Domain::Wide.sample(seed.wrapping_add(1), d.kv_len());

    let graph = graph_of(session);
    let q = upload(graph.handle(), &dims(&d.q_shape()), &q_data)?;
    let k = upload(graph.handle(), &dims(&d.kv_shape()), &k_data)?;
    let v = upload(graph.handle(), &dims(&d.kv_shape()), &v_data)?;
    let (o, lse) = attention_with_lse(&q, &k, &v, MaskKind::None, None)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let (want_o, want_lse) =
        host_attention(&q_data, &k_data, &v_data, d, d.default_scale(), &no_mask);
    expect_values(session, &d.q_shape(), Dtype::F32, &read(&o)?, &want_o)?;
    expect_values(session, &d.lse_shape(), Dtype::F32, &read(&lse)?, &want_lse)?;
    Ok(())
}

/// `attention_grads` against the analytic adjoints.
///
/// dk and dv are halves of one `[B, H, 2*Lk, Dh]` buffer handed back as
/// zero-cost views, so the element counts prove the halves were sliced the
/// right way round and the values prove they were not swapped.
fn grads_case(session: &Session, d: AttnDims, seed: u32) -> CaseResult {
    let q_data = Domain::Wide.sample(seed, d.q_len());
    let k_data = Domain::Wide.sample(seed ^ 0x9e37_79b9, d.kv_len());
    let v_data = Domain::Wide.sample(seed.wrapping_add(1), d.kv_len());
    let g_data = Domain::Wide.sample(seed ^ 0x5eed, d.q_len());

    let graph = graph_of(session);
    let q = upload(graph.handle(), &dims(&d.q_shape()), &q_data)?;
    let k = upload(graph.handle(), &dims(&d.kv_shape()), &k_data)?;
    let v = upload(graph.handle(), &dims(&d.kv_shape()), &v_data)?;
    let g = upload(graph.handle(), &dims(&d.q_shape()), &g_data)?;
    let (o, lse) = attention_with_lse(&q, &k, &v, MaskKind::None, None)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let (dq, dk, dv) = attention_grads(&q, &k, &v, &o, &g, &lse, MaskKind::None, None)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let (want_dq, want_dk, want_dv) =
        host_attention_grads(&q_data, &k_data, &v_data, &g_data, d, d.default_scale());
    expect_values(session, &d.q_shape(), Dtype::F32, &read(&dq)?, &want_dq)?;
    expect_values(session, &d.kv_shape(), Dtype::F32, &read(&dk)?, &want_dk)?;
    expect_values(session, &d.kv_shape(), Dtype::F32, &read(&dv)?, &want_dv)?;
    Ok(())
}

/// Grouped queries must be expanded by the caller; `attention_grads` refuses
/// them rather than silently summing over the group.
fn grads_gqa_refused(session: &Session) -> CaseResult {
    let d = AttnDims {
        heads_kv: 1,
        ..REFUSAL_DIMS
    };
    let graph = graph_of(session);
    let q = upload(
        graph.handle(),
        &dims(&d.q_shape()),
        &Domain::Wide.sample(827, d.q_len()),
    )?;
    let k = upload(
        graph.handle(),
        &dims(&d.kv_shape()),
        &Domain::Wide.sample(829, d.kv_len()),
    )?;
    let v = upload(
        graph.handle(),
        &dims(&d.kv_shape()),
        &Domain::Wide.sample(839, d.kv_len()),
    )?;
    let g = upload(
        graph.handle(),
        &dims(&d.q_shape()),
        &Domain::Wide.sample(853, d.q_len()),
    )?;
    let o = upload(
        graph.handle(),
        &dims(&d.q_shape()),
        &Domain::Wide.sample(859, d.q_len()),
    )?;
    let lse = upload(
        graph.handle(),
        &dims(&d.lse_shape()),
        &Domain::Wide.sample(857, d.b * d.h * d.lq),
    )?;
    if attention_grads(&q, &k, &v, &o, &g, &lse, MaskKind::None, None).is_ok() {
        return Err("attention_grads accepted H != Hkv; the caller must expand first".into());
    }
    Ok(())
}

/// The taped backward of the composed attention must agree with the analytic
/// adjoints. That agreement is what makes `attention_grads` an optimization
/// rather than a second rule to keep in sync by hand.
fn attention_backward(session: &Session, d: AttnDims, seed: u32) -> CaseResult {
    let q_data = Domain::Wide.sample(seed, d.q_len());
    let k_data = Domain::Wide.sample(seed ^ 0x9e37_79b9, d.kv_len());
    let v_data = Domain::Wide.sample(seed.wrapping_add(1), d.kv_len());
    let ones = vec![1.0f32; d.q_len()];

    let graph = graph_of(session);
    let q = upload(graph.handle(), &dims(&d.q_shape()), &q_data)?;
    let k = upload(graph.handle(), &dims(&d.kv_shape()), &k_data)?;
    let v = upload(graph.handle(), &dims(&d.kv_shape()), &v_data)?;
    let o = attention(&q, &k, &v, MaskKind::None, None)
        .map_err(|e| -> CaseError { e.to_string().into() })?;

    let (want_dq, want_dk, want_dv) =
        host_attention_grads(&q_data, &k_data, &v_data, &ones, d, d.default_scale());
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

// ---------------------------------------------------------------------------
// RoPE cases
// ---------------------------------------------------------------------------

type RopeBuild = fn(&Tensor, &Tensor, &Tensor, u64) -> fusor2::Result<Tensor>;
type RopePairBuild =
    fn(&Tensor, &Tensor, &Tensor, &Tensor, u64) -> fusor2::Result<(Tensor, Tensor)>;
type RopePosBuild = fn(&Tensor, &Tensor, &Tensor, &Tensor) -> fusor2::Result<Tensor>;
type RopePosPairBuild =
    fn(&Tensor, &Tensor, &Tensor, &Tensor, &Tensor) -> fusor2::Result<(Tensor, Tensor)>;

/// Upload the sin/cos tables covering `max_len` positions, returning both the
/// device tensors and the host copies the reference reads.
fn upload_tables(
    graph: &GraphRef,
    dh: usize,
    max_len: usize,
) -> Result<(Tensor, Tensor, Vec<f32>, Vec<f32>), CaseError> {
    let (cos, sin) = rope_tables(dh, max_len);
    let shape = dims(&[max_len as u64, (dh / 2) as u64]);
    let ct = upload(graph, &shape, &cos)?;
    let st = upload(graph, &shape, &sin)?;
    Ok((ct, st, cos, sin))
}

fn rope_case(
    session: &Session,
    seed: u32,
    name: &'static str,
    d: RopeDims,
    interleaved: bool,
    offset: u64,
    build: RopeBuild,
) -> CaseResult {
    let x_data = Domain::Wide.sample(seed, d.len());
    let graph = graph_of(session);
    let (ct, st, cos, sin) = upload_tables(graph.handle(), d.dh, d.l + offset as usize)?;
    let x = upload(graph.handle(), &dims(&d.shape()), &x_data)?;
    let y =
        build(&x, &ct, &st, offset).map_err(|e| -> CaseError { format!("{name}: {e}").into() })?;

    let expected = host_rope(&x_data, &cos, &sin, d, offset as usize, interleaved);
    expect_values(session, &d.shape(), Dtype::F32, &read(&y)?, &expected)?;
    Ok(())
}

/// q and k rotated in one dispatch. Both outputs are checked: a fused pair
/// that rotates q twice and leaves k alone still returns two tensors.
fn rope_pair_case(
    session: &Session,
    seed: u32,
    name: &'static str,
    d: RopeDims,
    interleaved: bool,
    build: RopePairBuild,
) -> CaseResult {
    let q_data = Domain::Wide.sample(seed, d.len());
    let k_data = Domain::Wide.sample(seed ^ 0x9e37_79b9, d.len());
    let graph = graph_of(session);
    let (ct, st, cos, sin) = upload_tables(graph.handle(), d.dh, d.l)?;
    let q = upload(graph.handle(), &dims(&d.shape()), &q_data)?;
    let k = upload(graph.handle(), &dims(&d.shape()), &k_data)?;
    let (rq, rk) =
        build(&q, &k, &ct, &st, 0).map_err(|e| -> CaseError { format!("{name}: {e}").into() })?;

    let want_q = host_rope(&q_data, &cos, &sin, d, 0, interleaved);
    let want_k = host_rope(&k_data, &cos, &sin, d, 0, interleaved);
    expect_values(session, &d.shape(), Dtype::F32, &read(&rq)?, &want_q)?;
    expect_values(session, &d.shape(), Dtype::F32, &read(&rk)?, &want_k)?;
    Ok(())
}

/// Rotate each row by the position its `u32` entry names.
fn host_rope_at(data: &[f32], cos: &[f32], sin: &[f32], pos: &[u32], d: RopeDims, il: bool) -> Vec<f32> {
    let mut expected = vec![0.0f32; data.len()];
    for b in 0..d.b {
        for h in 0..d.h {
            for l in 0..d.l {
                let base = ((b * d.h + h) * d.l + l) * d.dh;
                let rot = host_rope_vec(
                    &data[base..base + d.dh],
                    cos,
                    sin,
                    pos[l] as usize,
                    d.dh,
                    il,
                );
                expected[base..base + d.dh].copy_from_slice(&rot);
            }
        }
    }
    expected
}

/// A position per row, deliberately not `0..L`, so an implementation that
/// ignores the tensor fails. Every position stays inside the uploaded table.
fn sample_positions(seed: u32, l: usize, max_len: usize) -> Vec<u32> {
    fill_indices(seed ^ 0x5eed, l, max_len as u32)
}

/// The decode form: positions live in a rank-1 `u32` tensor so the offset
/// never round-trips through the host.
fn rope_position_case(
    session: &Session,
    seed: u32,
    name: &'static str,
    d: RopeDims,
    interleaved: bool,
    build: RopePosBuild,
) -> CaseResult {
    let x_data = Domain::Wide.sample(seed, d.len());
    let max_len = d.l + 8;
    let positions = sample_positions(seed, d.l, max_len);
    let graph = graph_of(session);
    let (ct, st, cos, sin) = upload_tables(graph.handle(), d.dh, max_len)?;
    let x = upload(graph.handle(), &dims(&d.shape()), &x_data)?;
    let p = from_u32(graph.handle(), &dims(&[d.l as u64]), &positions)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let y = build(&x, &ct, &st, &p).map_err(|e| -> CaseError { format!("{name}: {e}").into() })?;

    let expected = host_rope_at(&x_data, &cos, &sin, &positions, d, interleaved);
    expect_values(session, &d.shape(), Dtype::F32, &read(&y)?, &expected)?;
    Ok(())
}

fn rope_position_pair_case(
    session: &Session,
    seed: u32,
    name: &'static str,
    d: RopeDims,
    interleaved: bool,
    build: RopePosPairBuild,
) -> CaseResult {
    let q_data = Domain::Wide.sample(seed, d.len());
    let k_data = Domain::Wide.sample(seed ^ 0x9e37_79b9, d.len());
    let max_len = d.l + 8;
    let positions = sample_positions(seed, d.l, max_len);
    let graph = graph_of(session);
    let (ct, st, cos, sin) = upload_tables(graph.handle(), d.dh, max_len)?;
    let q = upload(graph.handle(), &dims(&d.shape()), &q_data)?;
    let k = upload(graph.handle(), &dims(&d.shape()), &k_data)?;
    let p = from_u32(graph.handle(), &dims(&[d.l as u64]), &positions)
        .map_err(|e| -> CaseError { e.to_string().into() })?;
    let (rq, rk) =
        build(&q, &k, &ct, &st, &p).map_err(|e| -> CaseError { format!("{name}: {e}").into() })?;

    for (data, got) in [(&q_data, &rq), (&k_data, &rk)] {
        let expected = host_rope_at(data, &cos, &sin, &positions, d, interleaved);
        expect_values(session, &d.shape(), Dtype::F32, &read(got)?, &expected)?;
    }
    Ok(())
}

/// `rotate_half(x) = cat(-x2, x1)` over the head axis.
fn rotate_half_case(session: &Session, d: RopeDims, seed: u32) -> CaseResult {
    let x_data = Domain::Wide.sample(seed, d.len());
    let graph = graph_of(session);
    let x = upload(graph.handle(), &dims(&d.shape()), &x_data)?;
    let y = rotate_half(&x).map_err(|e| -> CaseError { e.to_string().into() })?;

    let half = d.dh / 2;
    let mut expected = vec![0.0f32; d.len()];
    for base in (0..d.len()).step_by(d.dh) {
        for i in 0..half {
            expected[base + i] = -x_data[base + half + i];
            expected[base + half + i] = x_data[base + i];
        }
    }
    expect_values(session, &d.shape(), Dtype::F32, &read(&y)?, &expected)?;
    Ok(())
}

/// A rotation preserves the norm of every `(a, b)` pair, hence of the whole
/// head vector. Independent of the table, so it catches a sin/cos swap that a
/// self-consistent host reference would agree with.
fn rope_norm_preserving(session: &Session, d: RopeDims, seed: u32) -> CaseResult {
    let x_data = Domain::Wide.sample(seed, d.len());
    let graph = graph_of(session);
    let (ct, st, _, _) = upload_tables(graph.handle(), d.dh, d.l)?;
    let x = upload(graph.handle(), &dims(&d.shape()), &x_data)?;
    let y = rope(&x, &ct, &st, 0).map_err(|e| -> CaseError { e.to_string().into() })?;
    let got = read(&y)?;
    for (head, chunk) in got.chunks(d.dh).enumerate() {
        let src = &x_data[head * d.dh..head * d.dh + d.dh];
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
/// all-ones seed that gives `d/dx_a = cos + sin` and `d/dx_b = cos - sin` —
/// checked analytically, because a mis-signed sin term is exactly what
/// survives a symmetric finite-difference probe at small angles.
fn rope_backward(session: &Session, d: RopeDims, seed: u32) -> CaseResult {
    let x_data = Domain::Wide.sample(seed, d.len());
    let graph = graph_of(session);
    let (ct, st, cos, sin) = upload_tables(graph.handle(), d.dh, d.l)?;
    let x = upload(graph.handle(), &dims(&d.shape()), &x_data)?;
    let y = rope(&x, &ct, &st, 0).map_err(|e| -> CaseError { e.to_string().into() })?;
    let got = gradient_of(&graph, &y, &x)?;

    let half = d.dh / 2;
    let mut want = vec![0.0f32; d.len()];
    for b in 0..d.b {
        for h in 0..d.h {
            for l in 0..d.l {
                let base = ((b * d.h + h) * d.l + l) * d.dh;
                for i in 0..half {
                    let (c, s) = (cos[l * half + i], sin[l * half + i]);
                    want[base + i] = c + s;
                    want[base + half + i] = c - s;
                }
            }
        }
    }
    crate::compare::approx_or_relative_eq(backend_of(session), &[d.len()], &want, &got, 1e-4, 1e-4)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fixed problem every host self-check runs at.
    const TD: AttnDims = REFUSAL_DIMS;

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
            "rope_pair",
            "rope_interleaved_pair",
            "rope_pair_with_position",
            "rope_interleaved_pair_with_position",
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
        let q = vec![0.0f32; TD.q_len()];
        let k = vec![1.0f32; TD.kv_len()];
        let v: Vec<f32> = (0..TD.kv_len()).map(|i| (i % 7) as f32).collect();
        let (out, lse) = host_attention(&q, &k, &v, TD, TD.default_scale(), &no_mask);
        for b in 0..TD.b {
            for h in 0..TD.h {
                for d in 0..TD.dh {
                    let want: f32 = (0..TD.lk)
                        .map(|j| v[((b * TD.h + h) * TD.lk + j) * TD.dh + d])
                        .sum::<f32>()
                        / TD.lk as f32;
                    let got = out[((b * TD.h + h) * TD.lq) * TD.dh + d];
                    assert!((got - want).abs() < 1e-5, "{got} vs {want}");
                }
            }
        }
        for value in lse {
            assert!((value - (TD.lk as f32).ln()).abs() < 1e-5, "{value}");
        }
    }

    #[test]
    fn the_causal_mask_is_right_aligned() {
        // Lq = 3, Lk = 4: query 0 sees keys 0..=1, query 2 sees all four.
        assert_eq!(causal_mask(3, 4, 0, 1), 0.0);
        assert!(causal_mask(3, 4, 0, 2).is_infinite());
        assert_eq!(causal_mask(3, 4, 2, 3), 0.0);
    }

    #[test]
    fn the_gqa_expansion_shares_one_kv_head() {
        let d = AttnDims {
            heads_kv: 1,
            ..TD
        };
        let q = vec![0.0f32; d.q_len()];
        let k = vec![1.0f32; d.kv_len()];
        let v: Vec<f32> = (0..d.kv_len()).map(|i| i as f32).collect();
        let (out, _) = host_attention(&q, &k, &v, d, d.default_scale(), &no_mask);
        for b in 0..d.b {
            for x in 0..d.dh {
                let h0 = out[(b * d.h * d.lq) * d.dh + x];
                let h1 = out[((b * d.h + 1) * d.lq) * d.dh + x];
                assert!((h0 - h1).abs() < 1e-5, "GQA heads disagree: {h0} vs {h1}");
            }
        }
    }

    #[test]
    fn the_host_rope_is_a_rotation() {
        const DH: usize = 4;
        let (cos, sin) = rope_tables(DH, 4);
        let x: Vec<f32> = (0..DH).map(|i| (i + 1) as f32).collect();
        for il in [false, true] {
            for p in 0..4 {
                let y = host_rope_vec(&x, &cos, &sin, p, DH, il);
                let a: f32 = x.iter().map(|v| v * v).sum();
                let b: f32 = y.iter().map(|v| v * v).sum();
                assert!((a - b).abs() < 1e-4, "p={p} il={il}: {a} vs {b}");
            }
        }
        // Position 0 is the identity: cos 0 = 1, sin 0 = 0.
        assert_eq!(host_rope_vec(&x, &cos, &sin, 0, DH, false), x);
    }

    #[test]
    fn the_two_pairings_are_different_functions() {
        const DH: usize = 4;
        let (cos, sin) = rope_tables(DH, 3);
        let x: Vec<f32> = (0..DH).map(|i| (i + 1) as f32).collect();
        assert_ne!(
            host_rope_vec(&x, &cos, &sin, 1, DH, false),
            host_rope_vec(&x, &cos, &sin, 1, DH, true),
        );
    }

    #[test]
    fn the_inverse_frequency_table_is_decreasing_and_half_width() {
        const DH: usize = 4;
        let inv = base_inverse_frequency(DH as u32, 10_000.0);
        assert_eq!(inv.len(), DH / 2);
        assert!(inv.windows(2).all(|w| w[0] > w[1]), "{inv:?}");
        assert!((inv[0] - 1.0).abs() < 1e-6, "the first frequency is 1");
    }

    #[test]
    fn host_rope_at_reads_the_position_vector() {
        let d = RopeDims {
            b: 2,
            h: 2,
            l: 3,
            dh: 4,
        };
        let (cos, sin) = rope_tables(d.dh, 8);
        let data: Vec<f32> = (0..d.len()).map(|i| i as f32).collect();
        // Position 0 everywhere is the identity.
        assert_eq!(host_rope_at(&data, &cos, &sin, &[0, 0, 0], d, false), data);
        // A different position vector must change the answer.
        assert_ne!(host_rope_at(&data, &cos, &sin, &[3, 0, 5], d, false), data);
    }

    #[test]
    fn the_host_grads_agree_with_a_finite_difference_of_the_host_forward() {
        let q: Vec<f32> = (0..TD.q_len()).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect();
        let k: Vec<f32> = (0..TD.kv_len())
            .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
            .collect();
        let v: Vec<f32> = (0..TD.kv_len())
            .map(|i| ((i % 3) as f32 - 1.0) * 0.1)
            .collect();
        let g = vec![1.0f32; TD.q_len()];
        let s = TD.default_scale();
        let (dq, dk, dv) = host_attention_grads(&q, &k, &v, &g, TD, s);
        let eps = 1e-3f32;
        let sum = |q: &[f32], k: &[f32], v: &[f32]| -> f32 {
            host_attention(q, k, v, TD, s, &no_mask).0.iter().sum()
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
