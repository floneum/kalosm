//! Rotary embeddings, all macro ops: one [`rope`] and one paired
//! [`rope_pair`], each taking a [`RopeLayout`] and a [`RopePos`].
//!
//! Both layouts are the same expression, `x*cos + rot(x)*sin`, and differ
//! only in two index vectors: `rot` is one `Gather` along the head axis times
//! a sign vector.
//!
//! Sequence length is a `Dim::Sym` narrow or a position gather — never a host
//! bucket, so a decode loop recompiles nothing.

use fusor_autograd::tape::{GraphTape, TapeExt};
use fusor_ir::autograd::{Tape, Val};
use fusor_ir::egraph::Id;
use fusor_ir::scalar::BinOp;
use fusor_ir::shape::{Dim, StrideSpec};
use fusor_ir::{Error, Result};
use smallvec::SmallVec;

use crate::composite::{const_dim, core_op, float_leaf, index_leaf, index_run};
use crate::graph::GraphRef;
use crate::tensor::Tensor;

/// `1 / theta^(2i/dim)` for `i in 0..dim/2` — the shared RoPE frequency.
pub fn base_inverse_frequency(dim: u32, theta: f32) -> Vec<f32> {
    (0..dim / 2)
        .map(|i| 1.0 / theta.powf(2.0 * i as f32 / dim as f32))
        .collect()
}

/// Which elements of a head rotate together.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RopeLayout {
    /// `(i, i + Dh/2)`, the "normal" convention.
    Halves,
    /// `(2i, 2i + 1)`.
    Interleaved,
}

impl RopeLayout {
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
            Self::Halves => (0..dh).map(|i| if i < half { -1.0 } else { 1.0 }).collect(),
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

/// Which table rows a call reads.
#[derive(Copy, Clone, Debug)]
pub enum RopePos<P> {
    /// A `narrow` of the table by a host-known offset.
    Offset(u64),
    /// A rank-1 `u32` position tensor: the offset stays on device, so a decode
    /// loop never re-slices the table.
    Positions(P),
}

impl<P> RopePos<P> {
    /// The same position with its tensor converted.
    pub fn map<Q>(self, f: impl FnOnce(P) -> Q) -> RopePos<Q> {
        match self {
            Self::Offset(off) => RopePos::Offset(off),
            Self::Positions(p) => RopePos::Positions(f(p)),
        }
    }
}

/// Everything the defn needs, all created before the tape opens because index
/// and sign leaves carry host bytes.
struct RopeOperands {
    perm: Id,
    signs: Id,
    expand: Id,
    rows: RopePos<Id>,
    seq: Dim,
}

fn prepare(
    graph: &GraphRef,
    x: &Tensor,
    cos: &Tensor,
    layout: RopeLayout,
    rows: RopePos<&Tensor>,
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
        return Err(Error::Shape(format!(
            "rope needs an even head_dim, got {dh}"
        )));
    }
    let table = graph.facts(cos.id);
    if table.rank() != 2 || !table.shape[1].known_eq(Dim::Const(dh / 2)) {
        return Err(Error::Shape(format!(
            "a rope table is [context, head_dim/2]; got {:?}",
            table.shape
        )));
    }
    Ok(RopeOperands {
        perm: index_leaf(graph, &layout.permutation(dh))?,
        signs: float_leaf(graph, xf.dtype, &layout.signs(dh))?,
        expand: index_leaf(graph, &layout.table_expansion(dh))?,
        rows: rows.map(|p| p.id),
        seq: xf.shape[2],
    })
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
    let rows = match ops.rows {
        RopePos::Offset(0) if t.shape_of(table)[0].known_eq(ops.seq) => table,
        RopePos::Offset(off) => {
            let shape = t.shape_of(table);
            let specs: SmallVec<[StrideSpec; 6]> = smallvec::smallvec![
                StrideSpec::dim(0, ops.seq).with_offset(Dim::Const(off)),
                StrideSpec::dim(1, shape[1]),
            ];
            t.restride(&specs, table)?
        }
        RopePos::Positions(p) => t.gather(0, table, p)?,
    };
    t.gather(1, rows, ops.expand)
}

/// `x * cos + rot(x) * sin`.
fn rope_defn(t: &mut GraphTape<'_>, x: Val, cos: Val, sin: Val, ops: &RopeOperands) -> Result<Val> {
    let cos = table_rows(t, cos, ops)?;
    let sin = table_rows(t, sin, ops)?;
    let cos = broadcast_table(t, cos, x)?;
    let sin = broadcast_table(t, sin, x)?;

    let rotated = rotate(t, x, 3, ops.perm, ops.signs)?;
    let a = t.binary(BinOp::Mul, x, cos)?;
    let b = t.binary(BinOp::Mul, rotated, sin)?;
    t.binary(BinOp::Add, a, b)
}

/// One `Gather` along `axis`, times a sign vector broadcast over the rest.
/// Both layouts are this; only the two vectors differ.
fn rotate(t: &mut GraphTape<'_>, x: Val, axis: u32, perm: Id, signs: Id) -> Result<Val> {
    let swapped = t.gather(axis, x, perm)?;
    let specs: SmallVec<[StrideSpec; 6]> = t
        .shape_of(x)
        .iter()
        .enumerate()
        .map(|(i, &d)| {
            if i == axis as usize {
                StrideSpec::dim(0, d)
            } else {
                StrideSpec::broadcast(d)
            }
        })
        .collect();
    let signs = t.restride(&specs, signs)?;
    t.binary(BinOp::Mul, swapped, signs)
}

/// Rotary embedding of a `[batch, heads, len, head_dim]` value against
/// `[context, head_dim/2]` tables.
pub fn rope(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    layout: RopeLayout,
    pos: RopePos<&Tensor>,
) -> Result<Tensor> {
    let graph = &x.graph;
    let ops = prepare(graph, x, cos, layout, pos)?;
    let (xi, ci, si) = (x.id, cos.id, sin.id);
    core_op(graph, move |t| rope_defn(t, xi, ci, si, &ops))
}

/// `cat(-x2, x1)` over the last axis, exposed because callers spell it
/// directly.
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
    let perm = index_leaf(graph, &RopeLayout::Halves.permutation(dh))?;
    let signs = float_leaf(graph, facts.dtype, &RopeLayout::Halves.signs(dh))?;
    let xid = x.id;
    core_op(graph, |t| rotate(t, xid, axis, perm, signs))
}

/// Rotate `q` and `k` in one node, handed back as two views.
///
/// The heads are concatenated along the head axis, rotated once and narrowed
/// apart, so there is exactly one producer for a rule to mint a paired kernel
/// over and the two results cost a `Restride` each. Requires matching batch,
/// sequence and head dims — the reference asserts the same.
pub fn rope_pair(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    layout: RopeLayout,
    pos: RopePos<&Tensor>,
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

    let ops = prepare(graph, q, cos, layout, pos)?;
    let lower = index_run(graph, 0, hq)?;
    let upper = index_run(graph, hq, hk)?;
    let (qi, ki, ci, si) = (q.id, k.id, cos.id, sin.id);

    let joined = core_op(graph, move |t| {
        let dtype = t.dtype_of(qi);
        let mut shape = t.shape_of(qi);
        shape[1] = Dim::Const(hq + hk);
        let base = t.zeros_shaped(dtype, &shape)?;
        let base = t.scatter_set(1, base, lower, qi, true)?;
        let both = t.scatter_set(1, base, upper, ki, true)?;
        rope_defn(t, both, ci, si, &ops)
    })?;

    Ok((
        joined.narrow(1, 0, hq as usize)?,
        joined.narrow(1, hq as usize, hk as usize)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use fusor_ir::dtype::Dtype;
    use fusor_ir::egraph::EGraph;
    use std::sync::Arc;

    #[test]
    fn rope_table_work_is_bounded_by_requested_rows() -> Result<()> {
        let mut graph = EGraph::new(fusor_ir::CoreSemantics::new(Arc::new(
            fusor_tile::Planner::new(),
        )));
        let mut tape = GraphTape::new(&mut graph);
        let table = tape.zeros_shaped(Dtype::F32, &[Dim::Const(131072), Dim::Const(64)])?;
        let expand = tape.zeros_shaped(Dtype::U32, &[Dim::Const(128)])?;
        let positions = tape.zeros_shaped(Dtype::U32, &[Dim::Const(3)])?;
        for rows in [RopePos::Offset(0), RopePos::Offset(7), RopePos::Positions(positions)] {
            let first = tape.graph().len();
            let result = table_rows(
                &mut tape,
                table,
                &RopeOperands {
                    perm: expand,
                    signs: table,
                    expand,
                    rows,
                    seq: Dim::Const(3),
                },
            )?;
            assert_eq!(
                tape.shape_of(result).as_slice(),
                &[Dim::Const(3), Dim::Const(128)]
            );
            for i in first..tape.graph().len() {
                let elements: u64 = tape
                    .graph()
                    .facts(Id(i as u32))
                    .shape
                    .iter()
                    .map(|d| d.as_const().unwrap())
                    .product();
                assert!(elements <= 3 * 128, "RoPE materializes unused context rows");
            }
        }
        Ok(())
    }
}
