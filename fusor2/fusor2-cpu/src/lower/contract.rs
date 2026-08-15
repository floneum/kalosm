//! `KContract` on CPU: real blocking and a register microkernel, so
//! bias/gelu/dequant epilogues fuse into the k-loop. No external BLAS — one in
//! the critical path makes epilogue fusion structurally impossible.
//!
//! The nest is the shape of betlang's `conv1d_block4_group16`: a `TM x TN`
//! register tile whose accumulators are `Stmt::Loop` accumulators, so they stay
//! resident across the whole k nest and never reload. Because the accumulators
//! are **in the IR**, `pre_a`, `pre_b` and `post` fuse into the k-loop
//! epilogue, which is exactly what delegating to `gemm` cannot do.
//!
//! **The tile is `theta`'s and the grid covers the whole output.** What was
//! here took the lane count from a written-in `CONTRACT_BLOCK = 64` and
//! launched one workgroup per `(batch, m block)` with the lanes covering the
//! whole n axis, so nothing ever wrote output column 64 and up: `[4,8] x
//! [8,96]` came back with 127 of its 384 entries at exactly 0.0 and
//! `[128,512] x [512,128]` with half of them wrong — every dense CPU layer
//! wider than 64 units was silently wrong. The column block is a grid axis
//! now, so coverage is `ceil(m / rows) * ceil(n / cols)` blocks whatever the
//! tile is, and the tile itself is read out of the resolved schedule point
//! rather than written here.
//!
//! Owned by W10.

use fusor2_ir::device::Caps;
use fusor2_ir::error::Error;
use fusor2_ir::ir::level1::{ContractSide, L1, SchedPoint};
use fusor2_ir::ir::level2::{
    Accumulator, Addr, Builtin, ElementType, KernelIr, LocalDecl, ScalarElement,
    StorageView, Stmt, TileExpr, TileExprKind, WorkgroupAxis,
};
use fusor2_ir::ir::{Node, Op};
use fusor2_ir::scalar::{BinOp, CmpOp, ScalarExpr};
use fusor2_ir::shape::{Dim, Layout};
use fusor2_ir::target::LowerCtx;
use fusor2_ir::Result;
use std::sync::Arc;

use super::{bin, cmp, lit_f32, lit_u32, load, u32_ty, Binds, Translate};

/// The output tile one workgroup owns: `tm x tn` accumulators held by each of
/// `row_groups x col_groups` lanes.
///
/// **Coverage never depends on it.** The grid takes
/// `batch * ceil(m / rows) * ceil(n / cols)` workgroups, so a tile that is
/// wider than the matrix, narrower than it, or shaped nothing like it still
/// computes every output element. That is what makes reading the tile off
/// `theta` safe: a schedule point moves the launch shape and the register
/// reuse, never the answer.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct Tile {
    /// Output rows one lane accumulates.
    tm: u32,
    /// Output columns one lane accumulates.
    tn: u32,
    /// Lane groups down the m axis.
    row_groups: u32,
    /// Lane groups across the n axis.
    col_groups: u32,
}

impl Tile {
    /// Lanes the workgroup launches.
    fn lanes(self) -> u32 {
        self.row_groups.saturating_mul(self.col_groups).max(1)
    }

    /// Output rows one workgroup covers.
    fn rows(self) -> u32 {
        self.row_groups.saturating_mul(self.tm).max(1)
    }

    /// Output columns one workgroup covers.
    fn cols(self) -> u32 {
        self.col_groups.saturating_mul(self.tn).max(1)
    }

    /// Narrow onto this shape and this device.
    ///
    /// A lane group starting past the last row or column runs only masked
    /// lanes, and a workgroup may not exceed the device's own lane limit.
    /// Both are narrowings of the *tile*; the grid picks up whatever the tile
    /// stops covering, so neither can change a result. Columns are dropped
    /// before rows because a dropped column group becomes one more grid block
    /// on an axis that already loops.
    fn fit(mut self, m: u32, n: u32, max_lanes: u32) -> Self {
        let m = m.max(1);
        let n = n.max(1);
        self.tm = self.tm.clamp(1, m);
        self.tn = self.tn.clamp(1, n);
        self.row_groups = self.row_groups.clamp(1, m.div_ceil(self.tm));
        self.col_groups = self.col_groups.clamp(1, n.div_ceil(self.tn));
        let max_lanes = max_lanes.max(1);
        self.row_groups = self.row_groups.clamp(1, max_lanes);
        self.col_groups = self
            .col_groups
            .clamp(1, (max_lanes / self.row_groups).max(1));
        self
    }

}

/// The tile a resolved schedule point names.
///
/// * [`SchedPoint::Sgemm`] is the register tiling directly: `bm / tm` lane
///   groups down m, `bn / tn` across n, `tm x tn` accumulators each. `bk` and
///   `double_buffer` size a staged workgroup tile this nest does not have — it
///   reads A and B straight from storage and keeps the k reduction in
///   registers — exactly as `fusor2-gpu`'s `lower_sgemm` does.
/// * [`SchedPoint::Sgemv`] names the workgroup's width in subgroups; k is
///   walked sequentially here, so what it contributes is
///   `subgroups * subgroup_width` lanes each owning `vector` adjacent
///   columns.
/// Anything else names no contraction geometry at all. [`SchedPoint::Point`]
/// in particular means no schedule decision was ever made for this node, which
/// is a plan answer, not a tile to invent.
fn tile_of(theta: SchedPoint, caps: &Caps) -> Result<Tile> {
    let width = caps.subgroup_width().max(1);
    Ok(match theta {
        SchedPoint::Sgemm(p) => {
            let tm = p.tm.max(1);
            let tn = p.tn.max(1);
            Tile {
                tm,
                tn,
                row_groups: (p.bm / tm).max(1),
                col_groups: (p.bn / tn).max(1),
            }
        }
        SchedPoint::Sgemv(v) => Tile {
            tm: 1,
            tn: v.vector.max(1),
            row_groups: 1,
            col_groups: v.subgroups.max(1).saturating_mul(width),
        },
        other => {
            return Err(Error::Legality(format!(
                "the CPU contraction nest needs a schedule point that names a \
                 contraction geometry; {other:?} names none, so there is no tile \
                 to read and nothing legal to invent"
            )));
        }
    })
}

pub fn lower(caps: &Caps, node: &Node, theta: SchedPoint, cx: &LowerCtx<'_>) -> Result<KernelIr> {
    let Op::L1(op) = &node.op else {
        return Err(Error::Legality("not an L1 node".into()));
    };
    match op {
        L1::KContract {
            m,
            n,
            k,
            batch,
            post,
            a,
            b,
            ..
        } => build(
            cx,
            caps,
            Dims {
                m: *m,
                n: *n,
                k: *k,
                batch: *batch,
            },
            tile_of(theta, caps)?,
            a,
            b,
            post,
        ),
        _ => Err(Error::Legality("contract got a foreign node".into())),
    }
}

struct Dims {
    m: Dim,
    n: Dim,
    k: Dim,
    batch: Dim,
}

#[allow(clippy::too_many_arguments)]

/// The three strides this kernel indexes an operand with, collapsed out of the
/// operand's own per-axis layout.
///
/// # Why this exists
///
/// `build` addressed A as `((batch*m + row)*k) + kk` and B as
/// `((batch*k + kk)*n) + col` — dense `[batch, m, k]` and `[batch, k, n]`,
/// hardcoded. That silently ignores `Operand::layout`, which is a *contract
/// violation*: `L1::KContract` carries a full strided `Layout` per operand
/// precisely so a contraction whose spec is not in kernel axis order can be
/// read by permuting strides instead of by copying. `fusor2-tile`'s
/// `permuted_alias` mints exactly that, and this kernel read it densely and
/// computed wrong values — 22 CPU conformance rows, `matmul [cpu]` returning
/// `0.4009152` for `0.9157541`.
///
/// # The collapse
///
/// The kernel's three loop indices are the *products* `batch`, `m`/`k` and
/// `k`/`n`, while a layout has one axis per einsum label — `bhqd` is four axes
/// collapsing to `batch=b*h, m=q, k=d`. Group boundaries are recovered from the
/// extents alone (no labels needed here): walk the axes outermost-first and cut
/// each group when its running extent product reaches that group's total.
///
/// A group collapses to a single stride only when its own axes are internally
/// dense — `stride[i] == stride[i+1] * extent[i+1]` — and then the collapsed
/// stride is the innermost axis's. `None` when they are not, which is a layout
/// this three-stride kernel genuinely cannot address.
///
/// # Extent-1 axes are not part of the structure
///
/// An axis of extent 1 has one coordinate, always `0`, so it contributes `0` to
/// every address whatever its stride: neither its stride nor its position among
/// the other axes is observable. Walking it as if it were structure was wrong in
/// both directions. A *trailing* unit axis was consumed by nobody — a group
/// whose `want` is 1 enters with `prod == 1` and never advances the cursor — so
/// the "every axis accounted for" test refused `[m, 1]` presented as
/// `[batch=1, m, k=1]`, which is every `[n, 1]` column operand in
/// `fusor2::sampling::row`. An *interior* unit axis was worse: it entered the
/// density test as a real neighbour, and `stride[i] == stride[i+1] * 1` compares
/// a stride nothing reads, which is how a broadcast KV head
/// (`[2, 1, 4, 4]` at `[16, 16, 1, 4]`, GQA and MQA) was refused for a
/// "gap" between two axes that address the same byte.
///
/// So the unit axes are dropped up front and the walk sees only observable
/// structure. This is a widening: a layout accepted before is accepted with
/// byte-identical strides, since a group's collapsed stride is its innermost
/// axis's and unit axes can only have been interior padding in that group.
fn collapsed_strides(layout: &Layout, groups: [u32; 3]) -> Option<[u32; 3]> {
    let shape = layout.shape();
    let strides = layout.strides();
    if shape.len() != strides.len() {
        return None;
    }
    let ext_all: Vec<u32> = shape.iter().map(|d| d.as_const().map(|v| v as u32)).collect::<Option<_>>()?;
    let str_all: Vec<u32> = strides.iter().map(|d| d.as_const().map(|v| v as u32)).collect::<Option<_>>()?;

    let (ext, str_): (Vec<u32>, Vec<u32>) = ext_all
        .iter()
        .zip(&str_all)
        .filter(|(e, _)| **e != 1)
        .map(|(e, s)| (*e, *s))
        .unzip();

    let mut out = [0u32; 3];
    let mut axis = 0usize;
    for (gi, want) in groups.iter().copied().enumerate() {
        let want = want.max(1);
        let start = axis;
        let mut prod: u64 = 1;
        while prod < u64::from(want) && axis < ext.len() {
            prod = prod.saturating_mul(u64::from(ext[axis]));
            axis += 1;
        }
        if prod != u64::from(want) {
            return None;
        }
        if axis == start {
            // A group of extent 1 spans no axis and never advances an index.
            out[gi] = 0;
            continue;
        }
        for i in start..axis - 1 {
            if u64::from(str_[i]) != u64::from(str_[i + 1]) * u64::from(ext[i + 1]) {
                return None;
            }
        }
        out[gi] = str_[axis - 1];
    }
    (axis == ext.len()).then_some(out)
}

/// `base + a*sa + b*sb + c*sc`, dropping the zero-stride terms.
fn strided_index(
    x: (&TileExpr, u32),
    y: (&TileExpr, u32),
    z: (&TileExpr, u32),
) -> TileExpr {
    let mut acc: Option<TileExpr> = None;
    for (e, s) in [x, y, z] {
        if s == 0 {
            continue;
        }
        let term = if s == 1 {
            e.clone()
        } else {
            bin(BinOp::Mul, e.clone(), lit_u32(s), u32_ty())
        };
        acc = Some(match acc {
            Some(a) => bin(BinOp::Add, a, term, u32_ty()),
            None => term,
        });
    }
    acc.unwrap_or_else(|| lit_u32(0))
}

fn build(
    cx: &LowerCtx<'_>,
    caps: &Caps,
    d: Dims,
    tile: Tile,
    a: &ContractSide,
    b: &ContractSide,
    post: &ScalarExpr,
) -> Result<KernelIr> {
    let m = konst(d.m, "m")?.max(1);
    let n = konst(d.n, "n")?.max(1);
    let k = konst(d.k, "k")?.max(1);
    let batch = konst(d.batch, "batch")?.max(1);

    let binds = Binds::build(cx)?;
    let uniforms = binds.buffers.first().cloned();
    let out_buf = binds.of(cx.launch.root)?;

    let tile = tile.fit(
        m,
        n,
        caps.limits.max_compute_invocations_per_workgroup,
    );
    let (tm, tn) = (tile.tm, tile.tn);
    let block = tile.lanes();

    // One workgroup per `(batch, m block, n block)`. The n block is what was
    // missing: lanes covered the whole n axis and were then clamped, so every
    // column past the clamp went unwritten.
    let m_blocks = m.div_ceil(tile.rows()).max(1);
    let n_blocks = n.div_ceil(tile.cols()).max(1);
    let grid = [
        batch
            .saturating_mul(m_blocks)
            .saturating_mul(n_blocks)
            .max(1),
        1,
        1,
    ];

    let pid = TileExpr::new(
        TileExprKind::Builtin(Builtin::ProgramId(WorkgroupAxis::X)),
        u32_ty(),
    );
    let lane = TileExpr::new(TileExprKind::Builtin(Builtin::Lane), u32_ty());
    let nblk = bin(BinOp::Rem, pid.clone(), lit_u32(n_blocks), u32_ty());
    let rest = bin(BinOp::Div, pid, lit_u32(n_blocks), u32_ty());
    let mblk = bin(BinOp::Rem, rest.clone(), lit_u32(m_blocks), u32_ty());
    let bidx = bin(BinOp::Div, rest, lit_u32(m_blocks), u32_ty());

    // The lane's own corner of the block: `lane / col_groups` down m,
    // `lane % col_groups` across n, `tm x tn` outputs from there.
    let row0 = bin(
        BinOp::Add,
        bin(BinOp::Mul, mblk, lit_u32(tile.rows()), u32_ty()),
        bin(
            BinOp::Mul,
            bin(
                BinOp::Div,
                lane.clone(),
                lit_u32(tile.col_groups),
                u32_ty(),
            ),
            lit_u32(tm),
            u32_ty(),
        ),
        u32_ty(),
    );
    let col0 = bin(
        BinOp::Add,
        bin(BinOp::Mul, nblk, lit_u32(tile.cols()), u32_ty()),
        bin(
            BinOp::Mul,
            bin(BinOp::Rem, lane, lit_u32(tile.col_groups), u32_ty()),
            lit_u32(tn),
            u32_ty(),
        ),
        u32_ty(),
    );

    let f32_ty = ElementType::Scalar(ScalarElement::F32);
    let bool_ty = ElementType::Scalar(ScalarElement::Bool);
    let kk = Arc::new(LocalDecl::new(u32_ty()));
    let k_idx = TileExpr::new(TileExprKind::LoadLocal(Arc::clone(&kk)), u32_ty());

    // `TM * TN` accumulator slots resident across the whole k nest. They are IR
    // accumulators, not buffer traffic, which is what lets the epilogue fuse.
    let accs: Vec<Arc<LocalDecl>> = (0..tm * tn)
        .map(|_| Arc::new(LocalDecl::new(f32_ty)))
        .collect();

    let cols: Vec<TileExpr> = (0..tn)
        .map(|j| bin(BinOp::Add, col0.clone(), lit_u32(j), u32_ty()))
        .collect();
    let col_oks: Vec<TileExpr> = cols
        .iter()
        .map(|c| cmp(CmpOp::Lt, c.clone(), lit_u32(n)))
        .collect();
    let rows: Vec<TileExpr> = (0..tm)
        .map(|i| bin(BinOp::Add, row0.clone(), lit_u32(i), u32_ty()))
        .collect();
    let row_oks: Vec<TileExpr> = rows
        .iter()
        .map(|r| cmp(CmpOp::Lt, r.clone(), lit_u32(m)))
        .collect();

    // `tn` B loads and `tm` broadcast A loads per k step, then `tm * tn` FMAs
    // — the register-tile shape, with zero accumulator spill. The emitter
    // memoizes identical expressions, so each operand element is read once
    // however many accumulators consume it.
    // Address both operands through their own layouts. `permuted_alias` in
    // `fusor2-tile` mints a non-contiguous `Alias` for any contraction whose
    // spec is not already in kernel axis order, and reading that densely is a
    // miscompile, not a slowdown. A contiguous layout collapses to exactly the
    // dense strides this kernel used to hardcode, so every previously-working
    // plan emits byte-identical text.
    //
    // A side is a list of operands, each with its own buffer and its own
    // layout, all addressed by that side's `(batch, row, k)` or
    // `(batch, k, col)` triple. One entry is the ordinary dense contraction;
    // several is a side that absorbed a multi-buffer producer, and the only
    // difference downstream is how many `Arg`s the side's `pre` reads.
    // Each entry is either a bound buffer with its collapsed strides, or a
    // `Const` leaf already folded to its literal — those have no binding.
    let bind_side = |side: &ContractSide, groups: [u32; 3], which: &str| {
        side.ops
            .iter()
            .map(|o| {
                if let Some(lit) = crate::lower::const_operand(cx, o.src) {
                    return Ok(Err(lit));
                }
                let strides = collapsed_strides(&o.layout, groups).ok_or_else(|| {
                    Error::Plan(format!(
                        "cpu contraction cannot address operand {which} at layout {:?}",
                        o.layout
                    ))
                })?;
                Ok(Ok((binds.of(o.src)?, strides)))
            })
            .collect::<Result<Vec<std::result::Result<_, TileExpr>>>>()
    };
    let a_binds = bind_side(a, [batch, m, k], "a")?;
    let b_binds = bind_side(b, [batch, k, n], "b")?;

    // The per-axis coordinates a side's `pre` may read (an absorbed causal
    // mask does), reconstructed from the three collapsed group indices the
    // kernel actually loops over. Each operand axis belongs to exactly one
    // group — the same factorization `collapsed_strides` proved — so its
    // coordinate is a divmod of that group's flat index.
    let side_coords = |side: &ContractSide,
                       groups: [u32; 3],
                       flats: [&TileExpr; 3]|
     -> Option<Vec<TileExpr>> {
        if !side.pre.reads_index_of() {
            return Some(Vec::new());
        }
        let shape = side.primary().layout.shape();
        let exts: Vec<u32> = shape
            .iter()
            .map(|d| d.as_const().map(|v| v as u32))
            .collect::<Option<_>>()?;
        let mut coords = vec![lit_u32(0); exts.len()];
        let mut axis = 0usize;
        for (gi, want) in groups.iter().copied().enumerate() {
            let want = want.max(1);
            let start = axis;
            let mut prod: u64 = 1;
            while prod < u64::from(want) && axis < exts.len() {
                // Unit axes carry no structure; fold them into whichever
                // group the walk is in, coordinate zero.
                prod = prod.saturating_mul(u64::from(exts[axis].max(1)));
                axis += 1;
            }
            if prod != u64::from(want) {
                return None;
            }
            let mut rest = flats[gi].clone();
            for i in (start..axis).rev() {
                let e = lit_u32(exts[i].max(1));
                coords[i] = bin(BinOp::Rem, rest.clone(), e.clone(), u32_ty());
                rest = bin(BinOp::Div, rest, e, u32_ty());
            }
        }
        Some(coords)
    };

    let mut b_vals = Vec::with_capacity(tn as usize);
    for j in 0..tn as usize {
        let args: Vec<TileExpr> = b_binds
            .iter()
            .map(|entry| match entry {
                Err(lit) => lit.clone(),
                Ok((buf, str_)) => {
                    let index =
                        strided_index((&bidx, str_[0]), (&k_idx, str_[1]), (&cols[j], str_[2]));
                    load(Arc::clone(buf), index, col_oks[j].clone())
                }
            })
            .collect();
        let coords = side_coords(b, [batch, k, n], [&bidx, &k_idx, &cols[j]])
            .ok_or_else(|| Error::Plan("cpu contraction cannot state side coordinates".into()))?;
        b_vals.push(
            Translate {
                args: &args,
                coords: &coords,
                uniforms: uniforms.clone(),
            }
            .run(&b.pre)?,
        );
    }

    let mut updates = Vec::with_capacity(accs.len());
    for i in 0..tm as usize {
        let args: Vec<TileExpr> = a_binds
            .iter()
            .map(|entry| match entry {
                Err(lit) => lit.clone(),
                Ok((buf, str_)) => {
                    let index =
                        strided_index((&bidx, str_[0]), (&rows[i], str_[1]), (&k_idx, str_[2]));
                    load(Arc::clone(buf), index, row_oks[i].clone())
                }
            })
            .collect();
        let coords = side_coords(a, [batch, m, k], [&bidx, &rows[i], &k_idx])
            .ok_or_else(|| Error::Plan("cpu contraction cannot state side coordinates".into()))?;
        let a_val = Translate {
            args: &args,
            coords: &coords,
            uniforms: uniforms.clone(),
        }
        .run(&a.pre)?;
        for (j, b_val) in b_vals.iter().enumerate() {
            let slot = i * tn as usize + j;
            let prev = TileExpr::new(TileExprKind::LoadLocal(Arc::clone(&accs[slot])), f32_ty);
            let prod = bin(BinOp::Mul, a_val.clone(), b_val.clone(), f32_ty);
            updates.push(Accumulator {
                local: Arc::clone(&accs[slot]),
                init: lit_f32(0.0),
                update: bin(BinOp::Add, prev, prod, f32_ty),
            });
        }
    }

    let mut body = vec![Stmt::Loop {
        count: Some(lit_u32(k)),
        index: Some(kk),
        accumulators: updates,
        body: vec![],
    }];

    // Epilogue, fused straight onto the resident accumulators: no intermediate
    // buffer, one store per accumulator slot.
    for i in 0..tm as usize {
        for j in 0..tn as usize {
            let acc = TileExpr::new(
                TileExprKind::LoadLocal(Arc::clone(&accs[i * tn as usize + j])),
                f32_ty,
            );
            let value = Translate {
                args: &[acc],
                coords: &[],
                uniforms: uniforms.clone(),
            }
            .run(post)?;
            let index = bin(
                BinOp::Add,
                bin(
                    BinOp::Mul,
                    bin(
                        BinOp::Add,
                        bin(BinOp::Mul, bidx.clone(), lit_u32(m), u32_ty()),
                        rows[i].clone(),
                        u32_ty(),
                    ),
                    lit_u32(n),
                    u32_ty(),
                ),
                cols[j].clone(),
                u32_ty(),
            );
            body.push(Stmt::Store {
                dst: StorageView {
                    buffer: Arc::clone(&out_buf),
                    offset: 0,
                    layout: out_buf.layout.clone(),
                },
                addr: Addr::Linear(index),
                value,
                mask: bin(
                    BinOp::LogicalAnd,
                    row_oks[i].clone(),
                    col_oks[j].clone(),
                    bool_ty,
                ),
            });
        }
    }

    Ok(KernelIr {
        buffers: binds.buffers,
        grid,
        block,
        body,
        byte_arena: None,
        name: "cpu_contract",
    })
}


fn konst(d: Dim, what: &str) -> Result<u32> {
    d.as_const().map(|v| v as u32).ok_or_else(|| {
        Error::Legality(format!(
            "the CPU contraction nest needs a concrete {what}; specialize the symbolic dim first"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use fusor2_ir::ir::level1::{SgemmParams, SgemvParams};

    /// An extent-1 axis is not structure. A *trailing* one belongs to no group
    /// (`ones([16, 1])` as the A operand of a `k = 1` contraction is
    /// `[batch=1, m=16, k=1]`, the shape of every `[n, 1]` column operand in
    /// `fusor2::sampling::row`), and an *interior* one sits between two axes
    /// that address the same byte, so it may not be asked to bridge them.
    #[test]
    fn extent_one_axes_are_not_part_of_the_addressable_structure() {
        let col = Layout::contiguous(&[Dim::Const(16), Dim::Const(1)]);
        assert_eq!(collapsed_strides(&col, [1, 16, 1]), Some([0, 1, 0]));
        let scalar = Layout::contiguous(&[Dim::Const(1), Dim::Const(1)]);
        assert_eq!(collapsed_strides(&scalar, [1, 1, 1]), Some([0, 0, 0]));
        let row = Layout::contiguous(&[Dim::Const(1), Dim::Const(16)]);
        assert_eq!(collapsed_strides(&row, [1, 1, 16]), Some([0, 0, 1]));

        // A broadcast KV head between the batch axis and a transposed
        // `[d, s]`: `k` is `(h=1, d=4)` at stride 1, `n` is `s=4` at stride 4.
        // The unit axis's stride of 16 describes nothing and must not be
        // compared against `d`'s.
        let gqa = Layout::from_parts(
            Dim::Const(0),
            &[Dim::Const(2), Dim::Const(1), Dim::Const(4), Dim::Const(4)],
            &[Dim::Const(16), Dim::Const(16), Dim::Const(1), Dim::Const(4)],
        )
        .unwrap();
        assert_eq!(collapsed_strides(&gqa, [2, 4, 4]), Some([16, 1, 4]));
    }

    /// The widening is exactly the unit axes: a genuine gap between two axes
    /// that address different bytes is still refused, and every layout that
    /// collapsed before collapses to the same three strides.
    #[test]
    fn a_genuinely_unaddressable_split_is_still_refused() {
        // `[4, 4]` rows 8 apart cannot be one `m = 16` stride.
        let gapped = Layout::from_parts(
            Dim::Const(0),
            &[Dim::Const(4), Dim::Const(4)],
            &[Dim::Const(8), Dim::Const(1)],
        )
        .unwrap();
        assert_eq!(collapsed_strides(&gapped, [1, 16, 1]), None);
        // Extents that do not cover the geometry are refused, not padded.
        let small = Layout::contiguous(&[Dim::Const(4), Dim::Const(4)]);
        assert_eq!(collapsed_strides(&small, [1, 8, 4]), None);
        // Unchanged: the dense cases this kernel used to hardcode.
        let dense_a = Layout::contiguous(&[Dim::Const(3), Dim::Const(8), Dim::Const(5)]);
        assert_eq!(collapsed_strides(&dense_a, [3, 8, 5]), Some([40, 5, 1]));
        assert_eq!(collapsed_strides(&dense_a, [24, 5, 1]), Some([5, 1, 0]));
    }

    #[test]
    fn a_symbolic_extent_is_refused_rather_than_guessed() {
        assert!(matches!(
            konst(Dim::Sym(fusor2_ir::shape::SymId(1)), "k"),
            Err(Error::Legality(_))
        ));
        assert_eq!(konst(Dim::Const(96), "n").unwrap(), 96);
    }

    /// A point that names no contraction geometry is a legality answer. The
    /// arm this replaces invented `bn = 64`, which is how a 96-column matmul
    /// came back with 32 columns of zeros.
    #[test]
    fn a_point_with_no_geometry_is_refused_rather_than_invented() {
        let caps = crate::caps::cpu_caps();
        assert!(matches!(
            tile_of(SchedPoint::Point, caps),
            Err(Error::Legality(_))
        ));
        assert!(tile_of(SchedPoint::Sgemv(SgemvParams { vector: 4, subgroups: 1, cols: 1, parts: 1, gap: 0 }), caps).is_ok());
    }

    /// `fit` narrows the tile and never the coverage: at every shape and
    /// point, `ceil(m / rows) * ceil(n / cols)` blocks of `rows x cols` cover
    /// the whole output and the workgroup stays inside the device's lanes.
    #[test]
    fn every_fitted_tile_covers_the_whole_output() {
        let caps = crate::caps::cpu_caps();
        let max_lanes = caps.limits.max_compute_invocations_per_workgroup;
        for m in [1u32, 2, 3, 4, 8, 33, 64, 128] {
            for n in [1u32, 3, 32, 64, 65, 96, 128, 192, 4096] {
                for theta in thetas() {
                    let t = tile_of(theta, caps).unwrap().fit(m, n, max_lanes);
                    assert!(t.lanes() <= max_lanes, "{theta:?} at [{m},{n}]");
                    assert!(t.lanes() == t.row_groups * t.col_groups);
                    assert!(
                        m.div_ceil(t.rows()) * t.rows() >= m,
                        "rows {} miss m {m}",
                        t.rows()
                    );
                    assert!(
                        n.div_ceil(t.cols()) * t.cols() >= n,
                        "cols {} miss n {n} for {theta:?}",
                        t.cols()
                    );
                    assert!(t.tm <= m && t.tn <= n);
                }
            }
        }
    }

    fn thetas() -> Vec<SchedPoint> {
        let mut out = vec![
            SchedPoint::Sgemv(SgemvParams { vector: 4, subgroups: 1, cols: 1, parts: 1, gap: 0 }),
            SchedPoint::Sgemv(SgemvParams { vector: 1, subgroups: 4, cols: 1, parts: 1, gap: 0 }),
        ];
        for bm in [16u32, 32, 64, 128, 256] {
            for bn in [16u32, 32, 64, 128, 256] {
                for tm in [1u32, 2, 4, 8] {
                    for tn in [1u32, 2, 4, 8] {
                        out.push(SchedPoint::Sgemm(SgemmParams {
                            double_buffer: false,
                            bm,
                            bn,
                            bk: 8,
                            tm,
                            tn,
                        }));
                    }
                }
            }
        }
        out
    }
}

/// The contraction nest against an f64 host reference, run on the worker pool.
///
/// The shapes sweep `n` across the old 64-lane boundary — `matmul::wide_n`
/// reads exactly one of them — and every case runs at **every** schedule point
/// the domain can hand this lowering, because a lowering whose answer depends
/// on which legal point extraction picked is the bug this file just had.
#[cfg(test)]
mod exec_tests {
    use super::*;
    use fusor2_ir::device::Caps;
    use fusor2_ir::dtype::{Dtype, Persistence};
    use fusor2_ir::egraph::{EGraph, Id};
    use fusor2_ir::extract::{BindKind, BindingPlan, Extraction, Launch, Plan, PlanHash};
    use fusor2_ir::ir::level0::{BufferId, L0, LeafKind};
    use fusor2_ir::ir::level1::{
        AccessPlan, ContractSide, Family, Operand, ScheduleDomain, SgemmParams, SgemvParams,
    };
    use fusor2_ir::ir::{Level, Node};
    use fusor2_ir::semantics::{CoreSemantics, SumArenaPlanner};
    use fusor2_ir::shape::Layout;
    use fusor2_ir::target::{Buf, Target};

    use crate::alloc::AlignedBuf;
    use crate::target::CpuTarget;

    /// A one-point domain for the node under test. `lower` reads the plan's
    /// resolved point, never the node's domain, so this only has to be a real
    /// domain rather than a schedule-less `Point`.
    fn sgemm_domain_of(theta: SchedPoint) -> ScheduleDomain {
        let p = match theta {
            SchedPoint::Sgemm(p) => p,
            _ => SgemmParams {
                double_buffer: false,
                bm: 16,
                bn: 32,
                bk: 8,
                tm: 2,
                tn: 2,
            },
        };
        ScheduleDomain::Sgemm(fusor2_ir::ir::level1::SgemmDomain {
            params: smallvec::smallvec![p],
        })
    }

    fn graph() -> EGraph {
        EGraph::new(CoreSemantics::new(Arc::new(SumArenaPlanner)))
    }

    fn buffer(g: &mut EGraph, shape: &[u64]) -> Id {
        let next = g.len() as u32;
        g.add(Op::L0(L0::Leaf(LeafKind::Buffer {
            name: BufferId(next),
            dtype: Dtype::F32,
            shape: shape.iter().map(|d| Dim::Const(*d)).collect(),
        })))
        .unwrap()
    }

    fn alias(g: &EGraph, src: Id) -> Operand {
        Operand {
            src,
            layout: Layout::contiguous(&g.facts(src).shape),
            access: AccessPlan::Alias,
        }
    }

    fn upload(target: &CpuTarget, data: &[f32]) -> Buf {
        let buf = target
            .alloc((data.len() * 4).max(4) as u64, Persistence::Step)
            .unwrap();
        let raw = buf.downcast_ref::<AlignedBuf>().unwrap();
        // SAFETY: nothing else holds this buffer yet; the pool handed it back
        // because its refcount was one.
        let slice = unsafe { std::slice::from_raw_parts_mut(raw.as_mut_ptr(), raw.len()) };
        slice.fill(0);
        for (i, v) in data.iter().enumerate() {
            slice[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        buf
    }

    fn download(buf: &Buf, n: usize) -> Vec<f32> {
        let raw = buf.downcast_ref::<AlignedBuf>().unwrap();
        raw.as_slice()[..n * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// `out[b, i, j] = sum_p a[b, i, p] * b[b, p, j]`, in f64.
    fn reference(a: &[f32], b: &[f32], batch: usize, m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; batch * m * n];
        for bi in 0..batch {
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f64;
                    for p in 0..k {
                        acc += a[(bi * m + i) * k + p] as f64 * b[(bi * k + p) * n + j] as f64;
                    }
                    out[(bi * m + i) * n + j] = acc as f32;
                }
            }
        }
        out
    }

    fn sample(seed: u32, len: usize) -> Vec<f32> {
        let mut state = seed.wrapping_mul(2_654_435_761).max(1);
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 8) as f32 / (1 << 24) as f32) - 0.5
            })
            .collect()
    }

    /// Lower one `KContract` at `theta`, run it, and return the output.
    fn run_contract(
        theta: SchedPoint,
        batch: u32,
        m: u32,
        n: u32,
        k: u32,
        a: &[f32],
        b: &[f32],
    ) -> Vec<f32> {
        let mut g = graph();
        let a_id = buffer(&mut g, &[u64::from(batch), u64::from(m), u64::from(k)]);
        let b_id = buffer(&mut g, &[u64::from(batch), u64::from(k), u64::from(n)]);
        let out_id = buffer(&mut g, &[u64::from(batch), u64::from(m), u64::from(n)]);

        let op = L1::KContract {
            m: Dim::Const(u64::from(m)),
            n: Dim::Const(u64::from(n)),
            k: Dim::Const(u64::from(k)),
            batch: Dim::Const(u64::from(batch)),
            family: Family::Sgemm,
            post: ScalarExpr::arg(0, Dtype::F32),
            acc: Dtype::F32,
            a: ContractSide::one(ScalarExpr::arg(0, Dtype::F32), alias(&g, a_id)),
            b: ContractSide::one(ScalarExpr::arg(0, Dtype::F32), alias(&g, b_id)),
            // A real domain rather than `Point`, so nothing in this file
            // reads as a schedule-less mint even in a test.
            sched: sgemm_domain_of(theta),
        };
        // The node is built rather than added: `lower` reads the op and the
        // launch, and the launch's root is the output buffer, so nothing here
        // needs the contraction to have an e-class of its own.
        let node = Node {
            op: Op::L1(op),
            level: Level::L1,
            children: smallvec::smallvec![a_id, b_id],
        };
        let plan = Plan {
            extraction: Extraction::default(),
            launches: vec![Launch {
                root: out_id,
                members: smallvec::smallvec![out_id],
                bindings: vec![
                    BindingPlan { binding: 1, value: a_id, kind: BindKind::Read },
                    BindingPlan { binding: 2, value: b_id, kind: BindKind::Read },
                    BindingPlan { binding: 3, value: out_id, kind: BindKind::Write },
                ],
                grid: [1, 1, 1],
                block: 1,
            }],
            buffers: Vec::new(),
            symbols: Vec::new(),
            hash: PlanHash(0),
            cost: fusor2_ir::cost::Picoseconds(0),
        };
        let cx = LowerCtx {
            plan: &plan,
            launch: &plan.launches[0],
            graph: &g,
            symbols: &[],
        };
        let caps = Caps::clone(crate::caps::cpu_caps());
        let ir = lower(&caps, &node, theta, &cx).unwrap();

        let target = CpuTarget::new().unwrap();
        let a_buf = upload(&target, a);
        let b_buf = upload(&target, b);
        let out = upload(&target, &vec![0.0; (batch * m * n) as usize]);
        let artifact = target.emit(&ir).unwrap();
        target
            .launch(
                &artifact,
                ir.grid,
                &[a_buf, b_buf, out.clone()],
                &Default::default(),
            )
            .unwrap();
        download(&out, (batch * m * n) as usize)
    }

    fn check(theta: SchedPoint, batch: u32, m: u32, n: u32, k: u32) {
        let a = sample(m * k + 7, (batch * m * k) as usize);
        let b = sample(n * k + 13, (batch * k * n) as usize);
        let got = run_contract(theta, batch, m, n, k, &a, &b);
        let want = reference(
            &a,
            &b,
            batch as usize,
            m as usize,
            n as usize,
            k as usize,
        );
        // A reference of zeros would make every assertion below vacuous, and
        // the defect this file had produced exactly zeros.
        assert!(
            want.iter().filter(|w| w.abs() > 1e-3).count() * 4 >= want.len(),
            "the reference is degenerate at [{batch},{m},{n},{k}]"
        );
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            let tol = 1e-4 * w.abs().max(1.0);
            assert!(
                (g - w).abs() <= tol,
                "{theta:?} at [{batch},{m},{n},{k}] element {i} \
                 (row {}, col {}): got {g}, want {w}",
                i / n as usize,
                i % n as usize,
            );
        }
    }

    /// The regression this file exists for: every column past 64 came back
    /// 0.0 on the shape `matmul::wide_n_columns` states, and wrong (not zero)
    /// at wider n where a different geometry was selected.
    #[test]
    fn the_whole_output_is_written_past_column_64() {
        let theta = SchedPoint::Sgemv(SgemvParams { vector: 4, subgroups: 1, cols: 1, parts: 1, gap: 0 });
        for n in [32u32, 64, 65, 96, 128, 192, 256] {
            for m in [1u32, 2, 4, 8, 32] {
                check(theta, 1, m, n, 8);
            }
        }
    }

    /// The answer may not depend on which legal point extraction picked.
    #[test]
    fn every_schedule_point_computes_the_same_matrix() {
        let mut thetas = vec![
            SchedPoint::Sgemv(SgemvParams { vector: 4, subgroups: 1, cols: 1, parts: 1, gap: 0 }),
            SchedPoint::Sgemv(SgemvParams { vector: 1, subgroups: 8, cols: 1, parts: 1, gap: 0 }),
        ];
        for bm in [16u32, 64, 256] {
            for bn in [16u32, 32, 256] {
                for tm in [1u32, 2, 8] {
                    for tn in [1u32, 4, 8] {
                        thetas.push(SchedPoint::Sgemm(SgemmParams {
                            double_buffer: false,
                            bm,
                            bn,
                            bk: 8,
                            tm,
                            tn,
                        }));
                    }
                }
            }
        }
        for theta in thetas {
            check(theta, 1, 4, 96, 8);
            check(theta, 2, 3, 65, 5);
            check(theta, 1, 33, 33, 17);
        }
    }

    /// leave the batch index recoverable.
    #[test]
    fn batched_contractions_stay_separate() {
        let theta = SchedPoint::Sgemm(SgemmParams {
            double_buffer: false,
            bm: 16,
            bn: 32,
            bk: 8,
            tm: 2,
            tn: 2,
        });
        check(theta, 4, 5, 96, 7);
        check(theta, 3, 1, 129, 2);
    }
}
