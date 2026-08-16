//! Rotary embeddings, all macro ops. Two pairings (`rope` /
//! `rope_interleaved`), each with a paired form that rotates `q` and `k` under
//! one node and a `_with_position` form that keeps the offset on device.
//!
//! Both pairings are the same expression, `x*cos + rot(x)*sin`, and differ
//! only in two index vectors: `rot` is one `Gather` along the head axis times
//! a sign vector.
//!
//! Sequence length is a `Dim::Sym` narrow or a position gather — never a host
//! bucket, so a decode loop recompiles nothing.

use fusor2_autograd::tape::{GraphTape, TapeExt};
use fusor2_ir::autograd::{Tape, Val};
use fusor2_ir::egraph::Id;
use fusor2_ir::scalar::BinOp;
use fusor2_ir::shape::{Dim, StrideSpec};
use fusor2_ir::{Error, Result};
use smallvec::SmallVec;

use crate::composite::{MacroAttr, MacroOp, const_dim, index_leaf, index_run, macro_op};
use crate::graph::GraphRef;
use crate::tensor::Tensor;

/// `1 / theta^(2i/dim)` for `i in 0..dim/2` — the shared RoPE frequency.
pub fn base_inverse_frequency(dim: u32, theta: f32) -> Vec<f32> {
    (0..dim / 2)
        .map(|i| 1.0 / theta.powf(2.0 * i as f32 / dim as f32))
        .collect()
}

/// Which elements pair with which.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Pairing {
    /// `(i, i + Dh/2)`.
    Halves,
    /// `(2i, 2i + 1)`.
    Interleaved,
}

impl Pairing {
    /// The head-axis permutation `rot` gathers with.
    fn permutation(self, dh: u64) -> Vec<u32> {
        let half = dh / 2;
        match self {
            Self::Halves => (0..dh)
                .map(|i| if i < half { i + half } else { i - half } as u32)
                .collect(),
            Self::Interleaved => (0..dh).map(|i| (i ^ 1) as u32).collect(),
        }
    }

    /// The sign each rotated element carries.
    fn signs(self, dh: u64) -> Vec<f32> {
        let half = dh / 2;
        match self {
            Self::Halves => (0..dh)
                .map(|i| if i < half { -1.0 } else { 1.0 })
                .collect(),
            Self::Interleaved => (0..dh)
                .map(|i| if i % 2 == 0 { -1.0 } else { 1.0 })
                .collect(),
        }
    }

    /// How a `[L, Dh/2]` table is expanded to `[L, Dh]`.
    fn table_expansion(self, dh: u64) -> Vec<u32> {
        let half = dh / 2;
        match self {
            Self::Halves => (0..dh).map(|i| (i % half) as u32).collect(),
            Self::Interleaved => (0..dh).map(|i| (i / 2) as u32).collect(),
        }
    }
}

/// Where the sequence offset comes from.
#[derive(Copy, Clone, Debug)]
enum Rows {
    /// A `narrow` of the table by a host-known offset.
    Offset(u64),
    /// A rank-1 `u32` position tensor: the offset stays on device, so a decode
    /// loop never re-slices the cache.
    Positions(Id),
}

/// Everything the defn needs, all created before the tape opens because index
/// and sign leaves carry host bytes.
struct RopeOperands {
    perm: Id,
    signs: Id,
    expand: Id,
    rows: Rows,
    seq: Dim,
}

fn prepare(
    graph: &GraphRef,
    x: &Tensor,
    cos: &Tensor,
    pairing: Pairing,
    rows: Rows,
) -> Result<RopeOperands> {
    let xf = graph.facts(x.id);
    if xf.rank() != 4 {
        return Err(Error::Shape(format!(
            "rope operates on [batch, heads, len, head_dim], got rank {}",
            xf.rank()
        )));
    }
    let dh = const_dim(xf.shape[3], "rope head_dim")?;
    if dh % 2 != 0 {
        return Err(Error::Shape(format!("rope needs an even head_dim, got {dh}")));
    }
    let table = graph.facts(cos.id);
    if table.rank() != 2 || !table.shape[1].known_eq(Dim::Const(dh / 2)) {
        return Err(Error::Shape(format!(
            "a rope table is [context, head_dim/2]; got {:?}",
            table.shape
        )));
    }
    Ok(RopeOperands {
        perm: index_leaf(graph, &pairing.permutation(dh))?,
        signs: sign_leaf(graph, x, &pairing.signs(dh))?,
        expand: index_leaf(graph, &pairing.table_expansion(dh))?,
        rows,
        seq: xf.shape[2],
    })
}

/// A rank-1 leaf of `+/-1` in the value's dtype.
fn sign_leaf(graph: &GraphRef, like: &Tensor, signs: &[f32]) -> Result<Id> {
    let dtype = graph.facts(like.id).dtype;
    let mut bytes = Vec::with_capacity(signs.len() * 4);
    for v in signs {
        match dtype {
            fusor2_ir::dtype::Dtype::F16 => {
                bytes.extend_from_slice(&half::f16::from_f32(*v).to_bits().to_le_bytes())
            }
            fusor2_ir::dtype::Dtype::BF16 => {
                bytes.extend_from_slice(&half::bf16::from_f32(*v).to_bits().to_le_bytes())
            }
            _ => bytes.extend_from_slice(&v.to_le_bytes()),
        }
    }
    graph.constant_leaf(dtype, &[Dim::Const(signs.len() as u64)], bytes)
}

/// `[L, Dh]` broadcast to the value's `[B, H, L, Dh]`.
fn broadcast_table(t: &mut GraphTape<'_>, table: Val, like: Val) -> Result<Val> {
    let shape = t.shape_of(like);
    let specs: SmallVec<[StrideSpec; 6]> = smallvec::smallvec![
        StrideSpec::broadcast(shape[0]),
        StrideSpec::broadcast(shape[1]),
        StrideSpec::dim(0, shape[2]),
        StrideSpec::dim(1, shape[3]),
    ];
    t.restride(&specs, table)
}

/// The `[L, Dh]` slice of one table this call uses.
fn table_rows(t: &mut GraphTape<'_>, table: Val, ops: &RopeOperands) -> Result<Val> {
    let expanded = t.gather(1, table, ops.expand)?;
    match ops.rows {
        Rows::Offset(0) if t.shape_of(expanded)[0].known_eq(ops.seq) => Ok(expanded),
        Rows::Offset(off) => {
            let shape = t.shape_of(expanded);
            let specs: SmallVec<[StrideSpec; 6]> = smallvec::smallvec![
                StrideSpec::dim(0, ops.seq).with_offset(Dim::Const(off)),
                StrideSpec::dim(1, shape[1]),
            ];
            t.restride(&specs, expanded)
        }
        Rows::Positions(p) => t.gather(0, expanded, p),
    }
}

/// `x * cos + rot(x) * sin`.
fn rope_defn(
    t: &mut GraphTape<'_>,
    x: Val,
    cos: Val,
    sin: Val,
    ops: &RopeOperands,
) -> Result<Val> {
    let cos = table_rows(t, cos, ops)?;
    let sin = table_rows(t, sin, ops)?;
    let cos = broadcast_table(t, cos, x)?;
    let sin = broadcast_table(t, sin, x)?;

    let rotated = rotate_defn(t, x, ops)?;
    let a = t.binary(BinOp::Mul, x, cos)?;
    let b = t.binary(BinOp::Mul, rotated, sin)?;
    t.binary(BinOp::Add, a, b)
}

/// One `Gather` along the head axis, times a sign vector. Both pairings are
/// this; only the two vectors differ.
fn rotate_defn(t: &mut GraphTape<'_>, x: Val, ops: &RopeOperands) -> Result<Val> {
    let swapped = t.gather(3, x, ops.perm)?;
    let shape = t.shape_of(x);
    let specs: SmallVec<[StrideSpec; 6]> = smallvec::smallvec![
        StrideSpec::broadcast(shape[0]),
        StrideSpec::broadcast(shape[1]),
        StrideSpec::broadcast(shape[2]),
        StrideSpec::dim(0, shape[3]),
    ];
    let signs = t.restride(&specs, ops.signs)?;
    t.binary(BinOp::Mul, swapped, signs)
}

fn rope_with(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    pairing: Pairing,
    rows: Rows,
) -> Result<Tensor> {
    let graph = &x.graph;
    let ops = prepare(graph, x, cos, pairing, rows)?;
    let (xi, ci, si) = (x.id, cos.id, sin.id);
    let mut operands = vec![xi, ci, si, ops.perm, ops.signs, ops.expand];
    if let Rows::Positions(p) = ops.rows {
        operands.push(p);
    }
    macro_op(
        graph,
        MacroOp::Rope,
        MacroAttr::Rope {
            interleaved: matches!(pairing, Pairing::Interleaved),
            paired: false,
            with_position: matches!(ops.rows, Rows::Positions(_)),
        },
        &operands,
        move |t| rope_defn(t, xi, ci, si, &ops),
    )
}

/// Non-interleaved rope: pairs `(i, i + Dh/2)`.
pub fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor, offset: u64) -> Result<Tensor> {
    rope_with(x, cos, sin, Pairing::Halves, Rows::Offset(offset))
}

/// Interleaved rope: pairs `(2i, 2i + 1)`.
pub fn rope_interleaved(x: &Tensor, cos: &Tensor, sin: &Tensor, offset: u64) -> Result<Tensor> {
    rope_with(x, cos, sin, Pairing::Interleaved, Rows::Offset(offset))
}

/// `cat(-x2, x1)`, exposed because callers spell it directly.
pub fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let graph = &x.graph;
    let facts = graph.facts(x.id);
    let dh = const_dim(
        *facts
            .shape
            .last()
            .ok_or_else(|| Error::Shape("rotate_half needs a head axis".into()))?,
        "rotate_half head_dim",
    )?;
    let axis = (facts.rank() - 1) as u32;
    let perm = index_leaf(graph, &Pairing::Halves.permutation(dh))?;
    let signs = sign_leaf(graph, x, &Pairing::Halves.signs(dh))?;
    let xid = x.id;
    let id = graph.build(|t| {
        let swapped = t.gather(axis, xid, perm)?;
        let shape = t.shape_of(xid);
        let mut specs: SmallVec<[StrideSpec; 6]> = SmallVec::new();
        for (i, d) in shape.iter().copied().enumerate() {
            if i == axis as usize {
                specs.push(StrideSpec::dim(0, d));
            } else {
                specs.push(StrideSpec::broadcast(d));
            }
        }
        let signs = t.restride(&specs, signs)?;
        t.binary(BinOp::Mul, swapped, signs)
    })?;
    Ok(graph.tensor(id))
}

/// Rotate `q` and `k` in one node, handed back as two views.
///
/// The heads are concatenated along the head axis, rotated once and narrowed
/// apart, so there is exactly one producer for a rule to mint a paired kernel
/// over and the two results cost a `Restride` each. Requires matching batch,
/// sequence and head dims — the reference asserts the same.
fn rope_pair_with(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    pairing: Pairing,
    rows: Rows,
) -> Result<(Tensor, Tensor)> {
    let graph = &q.graph;
    let (qf, kf) = (graph.facts(q.id), graph.facts(k.id));
    if qf.rank() != 4 || kf.rank() != 4 {
        return Err(Error::Shape("paired rope needs two rank-4 values".into()));
    }
    if !qf.shape[2].known_eq(kf.shape[2]) || !qf.shape[3].known_eq(kf.shape[3]) {
        return Err(Error::Shape(
            "paired rope needs one sequence length and one head dim".into(),
        ));
    }
    let hq = const_dim(qf.shape[1], "paired rope q heads")?;
    let hk = const_dim(kf.shape[1], "paired rope k heads")?;

    let ops = prepare(graph, q, cos, pairing, rows)?;
    let lower = index_run(graph, 0, hq)?;
    let upper = index_run(graph, hq, hk)?;
    let (qi, ki, ci, si) = (q.id, k.id, cos.id, sin.id);
    let mut operands = vec![qi, ki, ci, si, ops.perm, ops.signs, ops.expand, lower, upper];
    if let Rows::Positions(p) = ops.rows {
        operands.push(p);
    }

    let joined = macro_op(
        graph,
        MacroOp::Rope,
        MacroAttr::Rope {
            interleaved: matches!(pairing, Pairing::Interleaved),
            paired: true,
            with_position: matches!(ops.rows, Rows::Positions(_)),
        },
        &operands,
        move |t| {
            let dtype = t.dtype_of(qi);
            let mut shape = t.shape_of(qi);
            shape[1] = Dim::Const(hq + hk);
            let base = t.zeros_shaped(dtype, &shape)?;
            let base = t.scatter_set(1, base, lower, qi, true)?;
            let both = t.scatter_set(1, base, upper, ki, true)?;
            rope_defn(t, both, ci, si, &ops)
        },
    )?;

    Ok((
        narrow_heads(&joined, 0, hq)?,
        narrow_heads(&joined, hq, hk)?,
    ))
}

fn narrow_heads(x: &Tensor, start: u64, len: u64) -> Result<Tensor> {
    let shape = x.graph.facts(x.id).shape.clone();
    let specs: SmallVec<[StrideSpec; 6]> = shape
        .iter()
        .copied()
        .enumerate()
        .map(|(i, d)| {
            if i == 1 {
                StrideSpec::dim(1, Dim::Const(len)).with_offset(Dim::Const(start))
            } else {
                StrideSpec::dim(i as u32, d)
            }
        })
        .collect();
    let xid = x.id;
    let id = x.graph.build(|t| t.restride(&specs, xid))?;
    Ok(x.graph.tensor(id))
}

/// Rotate `q` and `k` under one node, pairing `(i, i + Dh/2)`.
///
/// The pair node names both operands, so the rotation is built once and read
/// back as two views.
pub fn rope_pair(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    offset: u64,
) -> Result<(Tensor, Tensor)> {
    rope_pair_with(q, k, cos, sin, Pairing::Halves, Rows::Offset(offset))
}

/// [`rope_pair`] pairing `(2i, 2i + 1)`.
pub fn rope_interleaved_pair(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    offset: u64,
) -> Result<(Tensor, Tensor)> {
    rope_pair_with(q, k, cos, sin, Pairing::Interleaved, Rows::Offset(offset))
}

/// The decode-loop form: `positions` is a rank-1 `u32` tensor, so the offset
/// stays on device and the table is never re-sliced on the host.
pub fn rope_pair_with_position(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: &Tensor,
) -> Result<(Tensor, Tensor)> {
    rope_pair_with(q, k, cos, sin, Pairing::Halves, Rows::Positions(positions.id))
}

/// [`rope_pair_with_position`] pairing `(2i, 2i + 1)`.
pub fn rope_interleaved_pair_with_position(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: &Tensor,
) -> Result<(Tensor, Tensor)> {
    rope_pair_with(
        q,
        k,
        cos,
        sin,
        Pairing::Interleaved,
        Rows::Positions(positions.id),
    )
}

/// Single-value forms against a device-side position vector.
pub fn rope_with_position(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: &Tensor,
) -> Result<Tensor> {
    rope_with(x, cos, sin, Pairing::Halves, Rows::Positions(positions.id))
}

/// Interleaved rotary embedding using a device-side position per row.
pub fn rope_interleaved_with_position(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    positions: &Tensor,
) -> Result<Tensor> {
    rope_with(
        x,
        cos,
        sin,
        Pairing::Interleaved,
        Rows::Positions(positions.id),
    )
}
