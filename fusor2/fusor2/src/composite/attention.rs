//! Attention: the macro node carrying `MaskKind`, plus its `defn` expansion.
//!
//! `causal` is structural on the sugar node, so the compiler skips
//! upper-triangle Q.K work without loading a mask tensor.
//!
//! Grouped-query attention splits `q`'s head axis into `(Hkv, g)` — a legal
//! restride at any strides — and `g` becomes a free axis of the contraction.

use fusor2_autograd::tape::{GraphTape, TapeExt, accum_dtype, splat_of};
use fusor2_ir::autograd::{Tape, Val};
use fusor2_ir::ir::logical::{EinSpec, Label};
/// Which structural mask an attention node carries.
use fusor2_ir::ir::launch::MaskKind;
use fusor2_ir::scalar::{BinOp, CmpOp, ScalarExpr, UnOp};
use fusor2_ir::shape::{Dim, StrideSpec, SymId};
use fusor2_ir::{Error, Result};
use smallvec::SmallVec;

use crate::composite::normalization::softmax_defn;
use crate::composite::{AttentionOut, MacroAttr, MacroOp, const_dim, macro_op};
use crate::graph::GraphRef;
use crate::tensor::Tensor;

const B: Label = Label(0);
const HKV: Label = Label(1);
const G: Label = Label(2);
const LQ: Label = Label(3);
const LK: Label = Label(4);
const DH: Label = Label(5);

fn labels(list: &[Label]) -> SmallVec<[Label; 6]> {
    list.iter().copied().collect()
}

/// A uniform for one scale value, shared between calls that use the same one.
fn scale_uniform(graph: &GraphRef, scale: f32) -> SymId {
    let sym = graph.named_sym(&format!("attn_scale#{:08x}", scale.to_bits()));
    graph.set_uniform(sym, scale);
    sym
}

/// `1 / sqrt(head_dim)` unless the caller names one.
fn resolved_scale(graph: &GraphRef, q: &Tensor, scale: Option<f32>) -> Result<f32> {
    match scale {
        Some(s) => Ok(s),
        None => {
            let facts = graph.facts(q.id);
            let dh = const_dim(
                *facts
                    .shape
                    .last()
                    .ok_or_else(|| Error::Shape("attention q has no head dim".into()))?,
                "attention head_dim",
            )?;
            Ok(1.0 / (dh.max(1) as f32).sqrt())
        }
    }
}

/// `H / Hkv`, checked.
fn group_factor(graph: &GraphRef, q: &Tensor, k: &Tensor) -> Result<u64> {
    let (qf, kf) = (graph.facts(q.id), graph.facts(k.id));
    if qf.rank() != 4 || kf.rank() != 4 {
        return Err(Error::Shape(
            "attention operands are [batch, heads, len, head_dim]".into(),
        ));
    }
    let h = const_dim(qf.shape[1], "attention q heads")?;
    let hkv = const_dim(kf.shape[1], "attention kv heads")?;
    if hkv == 0 || h % hkv != 0 {
        return Err(Error::Shape(format!(
            "grouped-query attention needs {hkv} to divide {h}"
        )));
    }
    Ok(h / hkv)
}

/// Split `q`'s head axis into `(Hkv, g)` when the query has more heads than
/// the key. Always a legal restride: `Restride` composes relative to the
/// current strides.
fn split_query_heads(t: &mut GraphTape<'_>, q: Val, groups: u64) -> Result<Val> {
    if groups == 1 {
        return Ok(q);
    }
    let shape = t.shape_of(q);
    let h = shape[1]
        .as_const()
        .ok_or_else(|| Error::Shape("attention head count must be decidable".into()))?;
    let hkv = h / groups;
    let specs: SmallVec<[StrideSpec; 6]> = smallvec::smallvec![
        StrideSpec::dim(0, shape[0]),
        StrideSpec::dim_with(1, Dim::Const(hkv), groups as u32),
        StrideSpec::dim(1, Dim::Const(groups)),
        StrideSpec::dim(2, shape[2]),
        StrideSpec::dim(3, shape[3]),
    ];
    t.restride(&specs, q)
}

/// Merge the `(Hkv, g)` pair a grouped contraction output carries back into
/// one head axis. The output is contiguous by construction.
fn merge_heads(t: &mut GraphTape<'_>, v: Val, groups: u64) -> Result<Val> {
    if groups == 1 {
        return Ok(v);
    }
    let shape = t.shape_of(v);
    let hkv = shape[1]
        .as_const()
        .ok_or_else(|| Error::Shape("attention head count must be decidable".into()))?;
    let mut specs: SmallVec<[StrideSpec; 6]> = SmallVec::new();
    specs.push(StrideSpec::dim(0, shape[0]));
    specs.push(StrideSpec::dim(2, Dim::Const(hkv * groups)));
    for (i, d) in shape.iter().copied().enumerate().skip(3) {
        specs.push(StrideSpec::dim(i as u32, d));
    }
    t.restride(&specs, v)
}

/// `q . k^T * scale`, plus whatever the mask contributes.
///
/// `MaskKind::Causal` compiles to an `IndexOf` comparison inside the scaling
/// `Map`, so no mask tensor is loaded and no buffer is bound for it.
#[allow(clippy::too_many_arguments)]
fn scores(
    t: &mut GraphTape<'_>,
    q: Val,
    k: Val,
    groups: u64,
    scale: SymId,
    mask: MaskKind,
    mask_tensor: Option<Val>,
) -> Result<Val> {
    let dtype = t.dtype_of(q);
    let q = split_query_heads(t, q, groups)?;
    let grouped = groups > 1;

    let mut a = labels(&[B, HKV]);
    if grouped {
        a.push(G);
    }
    a.extend([LQ, DH]);
    let b = labels(&[B, HKV, LK, DH]);
    let mut out = labels(&[B, HKV]);
    if grouped {
        out.push(G);
    }
    out.extend([LQ, LK]);

    let acc = accum_dtype(dtype);
    let s = t.contract(q, k, EinSpec { a, b, out }, acc)?;
    let s = t.cast(dtype, s)?;

    // Scale and causal masking are one `Map`.
    let rank = t.rank_of(s);
    let (lq_axis, lk_axis) = ((rank - 2) as u32, (rank - 1) as u32);
    let score_shape = t.shape_of(s);
    let (shape_lq, shape_lk) = (score_shape[rank - 2], score_shape[rank - 1]);
    let scaled = ScalarExpr::bin(
        BinOp::Mul,
        ScalarExpr::arg(0, dtype),
        ScalarExpr::uniform(scale, dtype),
    );
    let body = if matches!(mask, MaskKind::Causal) {
        // Right-aligned: query `i` sees keys up to `i + (Lk - Lq)`. When
        // `Lq != Lk` the Lq queries are the last Lq of the Lk keys (decode
        // against a KV cache).
        let bound = match (shape_lq.as_const(), shape_lk.as_const()) {
            (Some(lq), Some(lk)) if lk > lq => ScalarExpr::bin(
                BinOp::Add,
                ScalarExpr::index_of(lq_axis),
                ScalarExpr::lit(fusor2_ir::dtype::Splat::U32((lk - lq) as u32)),
            ),
            _ => ScalarExpr::index_of(lq_axis),
        };
        ScalarExpr::select(
            ScalarExpr::cmp(CmpOp::Le, ScalarExpr::index_of(lk_axis), bound),
            scaled,
            ScalarExpr::lit(splat_of(dtype, f32::NEG_INFINITY)?),
        )
    } else {
        scaled
    };
    let s = t.map(body, &[s])?;

    match (mask, mask_tensor) {
        (MaskKind::QkMask, Some(m)) => {
            // `[Lq, Lk]` broadcasts right-aligned onto `[.., Lq, Lk]`.
            let shape = t.shape_of(s);
            let m = t.broadcast_to(m, &shape)?;
            t.binary(BinOp::Add, s, m)
        }
        (MaskKind::BatchKeyMask, Some(m)) => {
            // `[B, Lk]` needs explicit strides: the batch axis is leading, so
            // the right-aligned rule cannot place it.
            let shape = t.shape_of(s);
            let mut specs: SmallVec<[StrideSpec; 6]> = SmallVec::new();
            specs.push(StrideSpec::dim(0, shape[0]));
            for d in shape[1..shape.len() - 1].iter().copied() {
                specs.push(StrideSpec::broadcast(d));
            }
            specs.push(StrideSpec::dim(1, shape[shape.len() - 1]));
            let m = t.restride(&specs, m)?;
            t.binary(BinOp::Add, s, m)
        }
        _ => Ok(s),
    }
}

#[allow(clippy::too_many_arguments)]
fn attention_defn(
    t: &mut GraphTape<'_>,
    q: Val,
    k: Val,
    v: Val,
    groups: u64,
    scale: SymId,
    mask: MaskKind,
    mask_tensor: Option<Val>,
) -> Result<Val> {
    let dtype = t.dtype_of(q);
    let s = scores(t, q, k, groups, scale, mask, mask_tensor)?;
    let rank = t.rank_of(s);
    let p = softmax_defn(t, s, (rank - 1) as u32)?;

    let grouped = groups > 1;
    let mut a = labels(&[B, HKV]);
    if grouped {
        a.push(G);
    }
    a.extend([LQ, LK]);
    let b = labels(&[B, HKV, LK, DH]);
    let mut out = labels(&[B, HKV]);
    if grouped {
        out.push(G);
    }
    out.extend([LQ, DH]);

    let acc = accum_dtype(dtype);
    let o = t.contract(p, v, EinSpec { a, b, out }, acc)?;
    let o = t.cast(dtype, o)?;
    merge_heads(t, o, groups)
}

/// Scaled dot-product attention.
pub fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: MaskKind,
    scale: Option<f32>,
) -> Result<Tensor> {
    attention_masked(q, k, v, mask, None, scale)
}

/// Causal attention, with causality encoded structurally.
pub fn attention_causal(q: &Tensor, k: &Tensor, v: &Tensor, scale: Option<f32>) -> Result<Tensor> {
    attention_masked(q, k, v, MaskKind::Causal, None, scale)
}

/// Attention against a materialized additive mask.
pub fn attention_masked(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: MaskKind,
    mask_tensor: Option<&Tensor>,
    scale: Option<f32>,
) -> Result<Tensor> {
    let graph = &q.graph;
    let groups = group_factor(graph, q, k)?;
    let scale_value = resolved_scale(graph, q, scale)?;
    let sym = scale_uniform(graph, scale_value);
    if matches!(mask, MaskKind::QkMask | MaskKind::BatchKeyMask) && mask_tensor.is_none() {
        return Err(Error::Shape(format!(
            "{mask:?} needs a mask tensor; only None and Causal are structural"
        )));
    }

    let (qi, ki, vi) = (q.id, k.id, v.id);
    let mi = mask_tensor.map(|m| m.id);
    let mut ops = vec![qi, ki, vi];
    ops.extend(mi);
    macro_op(
        graph,
        MacroOp::Attention,
        MacroAttr::Attention {
            mask,
            causal: matches!(mask, MaskKind::Causal),
            groups: groups as u32,
            produce: AttentionOut::Output,
            scale: sym,
        },
        &ops,
        move |t| attention_defn(t, qi, ki, vi, groups, sym, mask, mi),
    )
}

/// The row log-sum-exp of the attention scores: `m + ln sum exp(s - m)`,
/// shaped `[.., Lq]`.
///
/// This is the statistic that lets probabilities be recomputed as
/// `exp(s - lse)` without ever storing the `[Lq, Lk]` matrix.
pub fn attention_lse(
    q: &Tensor,
    k: &Tensor,
    mask: MaskKind,
    mask_tensor: Option<&Tensor>,
    scale: Option<f32>,
) -> Result<Tensor> {
    let graph = &q.graph;
    let groups = group_factor(graph, q, k)?;
    let scale_value = resolved_scale(graph, q, scale)?;
    let sym = scale_uniform(graph, scale_value);
    let (qi, ki) = (q.id, k.id);
    let mi = mask_tensor.map(|m| m.id);
    let mut ops = vec![qi, ki];
    ops.extend(mi);
    macro_op(
        graph,
        MacroOp::Attention,
        MacroAttr::Attention {
            mask,
            causal: matches!(mask, MaskKind::Causal),
            groups: groups as u32,
            produce: AttentionOut::LogSumExp,
            scale: sym,
        },
        &ops,
        move |t| {
            let s = scores(t, qi, ki, groups, sym, mask, mi)?;
            let dtype = t.dtype_of(s);
            let rank = t.rank_of(s);
            let axis = (rank - 1) as u32;
            let extent = t.shape_of(s)[axis as usize];
            let m = t.fold_binop(BinOp::Max, axis, dtype, s)?;
            let mb = t.broadcast_axis(m, axis, extent)?;
            let centered = t.binary(BinOp::Sub, s, mb)?;
            let e = t.unary(UnOp::Exp, centered)?;
            let sum = t.fold_binop(BinOp::Add, axis, accum_dtype(dtype), e)?;
            let sum = t.cast(dtype, sum)?;
            let ln = t.unary(UnOp::Log, sum)?;
            let lse = t.binary(BinOp::Add, m, ln)?;
            merge_lse_heads(t, lse, groups)
        },
    )
}

/// The `[.., Hkv, g, Lq]` head pair of an lse, merged back to `[.., H, Lq]`.
fn merge_lse_heads(t: &mut GraphTape<'_>, v: Val, groups: u64) -> Result<Val> {
    if groups == 1 {
        return Ok(v);
    }
    let shape = t.shape_of(v);
    let hkv = shape[1]
        .as_const()
        .ok_or_else(|| Error::Shape("attention head count must be decidable".into()))?;
    let specs: SmallVec<[StrideSpec; 6]> = smallvec::smallvec![
        StrideSpec::dim(0, shape[0]),
        StrideSpec::dim(2, Dim::Const(hkv * groups)),
        StrideSpec::dim(3, shape[3]),
    ];
    t.restride(&specs, v)
}

/// Attention and its row log-sum-exp together.
pub fn attention_with_lse(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: MaskKind,
    scale: Option<f32>,
) -> Result<(Tensor, Tensor)> {
    let o = attention_masked(q, k, v, mask, None, scale)?;
    let lse = attention_lse(q, k, mask, None, scale)?;
    Ok((o, lse))
}

/// `(dq, dk, dv)`.
///
/// `dk` and `dv` are `Restride` views of one `[B, H, 2*Lk, Dh]` buffer —
/// dk rows then dv rows — so a paired streaming kernel can share the
/// probability recomputation. Requires `H == Hkv`: grouped-query attention
/// is expanded by the caller.
#[allow(clippy::too_many_arguments)]
pub fn attention_grads(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    o: &Tensor,
    grad_out: &Tensor,
    lse: &Tensor,
    mask: MaskKind,
    scale: Option<f32>,
) -> Result<(Tensor, Tensor, Tensor)> {
    let graph = &q.graph;
    if group_factor(graph, q, k)? != 1 {
        return Err(Error::Shape(
            "attention_grads needs matching head counts; expand grouped queries first".into(),
        ));
    }
    let scale_value = resolved_scale(graph, q, scale)?;
    let sym = scale_uniform(graph, scale_value);
    let kf = graph.facts(k.id);
    let lk = const_dim(kf.shape[2], "attention key length")?;

    let (qi, ki, vi, oi, gi, li) = (q.id, k.id, v.id, o.id, grad_out.id, lse.id);
    let index = crate::composite::index_run(graph, 0, lk)?;
    let index_upper = crate::composite::index_run(graph, lk, lk)?;

    let dq = macro_op(
        graph,
        MacroOp::Attention,
        MacroAttr::Attention {
            mask,
            causal: matches!(mask, MaskKind::Causal),
            groups: 1,
            produce: AttentionOut::GradQ,
            scale: sym,
        },
        &[qi, ki, vi, oi, gi, li],
        move |t| {
            let ds = grad_scores(t, qi, ki, vi, oi, gi, li, sym)?;
            let acc = accum_dtype(t.dtype_of(ds));
            let dtype = t.dtype_of(qi);
            let dq = t.contract(
                ds,
                ki,
                EinSpec {
                    a: labels(&[B, HKV, LQ, LK]),
                    b: labels(&[B, HKV, LK, DH]),
                    out: labels(&[B, HKV, LQ, DH]),
                },
                acc,
            )?;
            t.cast(dtype, dq)
        },
    )?;

    let combined = macro_op(
        graph,
        MacroOp::Attention,
        MacroAttr::Attention {
            mask,
            causal: matches!(mask, MaskKind::Causal),
            groups: 1,
            produce: AttentionOut::GradKV,
            scale: sym,
        },
        &[qi, ki, vi, oi, gi, li, index, index_upper],
        move |t| {
            let dtype = t.dtype_of(qi);
            let acc = accum_dtype(dtype);
            let ds = grad_scores(t, qi, ki, vi, oi, gi, li, sym)?;
            let dk = t.contract(
                ds,
                qi,
                EinSpec {
                    a: labels(&[B, HKV, LQ, LK]),
                    b: labels(&[B, HKV, LQ, DH]),
                    out: labels(&[B, HKV, LK, DH]),
                },
                acc,
            )?;
            let dk = t.cast(dtype, dk)?;
            let p = probabilities(t, qi, ki, oi, li, sym)?;
            let dv = t.contract(
                p,
                gi,
                EinSpec {
                    a: labels(&[B, HKV, LQ, LK]),
                    b: labels(&[B, HKV, LQ, DH]),
                    out: labels(&[B, HKV, LK, DH]),
                },
                acc,
            )?;
            let dv = t.cast(dtype, dv)?;

            // One buffer, dk rows then dv rows.
            let mut shape = t.shape_of(dk);
            shape[2] = Dim::Const(lk * 2);
            let base = t.zeros_shaped(dtype, &shape)?;
            let base = t.scatter_set(2, base, index, dk, true)?;
            t.scatter_set(2, base, index_upper, dv, true)
        },
    )?;

    let dk = narrow_axis(&combined, 2, 0, lk)?;
    let dv = narrow_axis(&combined, 2, lk, lk)?;
    Ok((dq, dk, dv))
}

/// `p = exp(s - lse)` — the probability matrix recomputed from the forward
/// output rather than stored.
fn probabilities(
    t: &mut GraphTape<'_>,
    q: Val,
    k: Val,
    _o: Val,
    lse: Val,
    scale: SymId,
) -> Result<Val> {
    let s = scores(t, q, k, 1, scale, MaskKind::None, None)?;
    let axis = (t.rank_of(s) - 1) as u32;
    let extent = t.shape_of(s)[axis as usize];
    let l = t.broadcast_axis(lse, axis, extent)?;
    let centered = t.binary(BinOp::Sub, s, l)?;
    t.unary(UnOp::Exp, centered)
}

/// `ds = scale * p * (dp - rowsum(grad_o * o))`.
#[allow(clippy::too_many_arguments)]
fn grad_scores(
    t: &mut GraphTape<'_>,
    q: Val,
    k: Val,
    v: Val,
    o: Val,
    grad_o: Val,
    lse: Val,
    scale: SymId,
) -> Result<Val> {
    let dtype = t.dtype_of(q);
    let acc = accum_dtype(dtype);
    let p = probabilities(t, q, k, o, lse, scale)?;
    let dp = t.contract(
        grad_o,
        v,
        EinSpec {
            a: labels(&[B, HKV, LQ, DH]),
            b: labels(&[B, HKV, LK, DH]),
            out: labels(&[B, HKV, LQ, LK]),
        },
        acc,
    )?;
    let dp = t.cast(dtype, dp)?;

    let go = t.binary(BinOp::Mul, grad_o, o)?;
    let dsum = t.fold_binop(BinOp::Add, 3, acc, go)?;
    let dsum = t.cast(dtype, dsum)?;
    let lk = t.shape_of(dp)[3];
    let dsum = t.broadcast_axis(dsum, 3, lk)?;

    let delta = t.binary(BinOp::Sub, dp, dsum)?;
    let scaled = t.map(
        ScalarExpr::bin(
            BinOp::Mul,
            ScalarExpr::arg(0, dtype),
            ScalarExpr::uniform(scale, dtype),
        ),
        &[delta],
    )?;
    t.binary(BinOp::Mul, p, scaled)
}

/// A zero-cost strided view of `len` positions of `axis` starting at `start`.
fn narrow_axis(x: &Tensor, axis: u32, start: u64, len: u64) -> Result<Tensor> {
    let shape = x.graph.facts(x.id).shape.clone();
    let specs: SmallVec<[StrideSpec; 6]> = shape
        .iter()
        .copied()
        .enumerate()
        .map(|(i, d)| {
            if i == axis as usize {
                StrideSpec::dim(i as u32, Dim::Const(len)).with_offset(Dim::Const(start))
            } else {
                StrideSpec::dim(i as u32, d)
            }
        })
        .collect();
    let xid = x.id;
    let id = x.graph.build(|t| t.restride(&specs, xid))?;
    Ok(x.graph.tensor(id))
}
