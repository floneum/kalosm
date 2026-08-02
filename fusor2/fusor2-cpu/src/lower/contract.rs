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
use fusor2_ir::ir::level1::{FoldStrat, L1, Operand, SchedPoint};
use fusor2_ir::ir::level2::{
    Accumulator, Addr, Builtin, ElementType, KernelIr, LocalDecl, QuantizedView, ScalarElement,
    StorageView, Stmt, TileExpr, TileExprKind, WorkgroupAxis,
};
use fusor2_ir::ir::{Node, Op};
use fusor2_ir::scalar::{BinOp, CmpOp, ScalarExpr};
use fusor2_ir::shape::Dim;
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

    /// The one-column-per-lane form the quantized body is written in: its
    /// decode program addresses one weight row per lane, so a `tn`-wide
    /// register tile has nowhere to go and what survives of the point is its
    /// lane count.
    fn columns_only(self) -> Self {
        Self {
            tm: 1,
            tn: 1,
            row_groups: 1,
            col_groups: self.lanes(),
        }
    }
}

/// The tile a resolved schedule point names.
///
/// * [`SchedPoint::Sgemm`] is the register tiling directly: `bm / tm` lane
///   groups down m, `bn / tn` across n, `tm x tn` accumulators each. `bk` and
///   `double_buffer` size a staged workgroup tile this nest does not have — it
///   reads A and B straight from storage and keeps the k reduction in
///   registers — exactly as `fusor2-gpu`'s `lower_sgemm` does.
/// * [`SchedPoint::Sgemv`] decomposes the **k** axis (`chunk`) and names the
///   workgroup's width in subgroups; k is walked sequentially here, so what it
///   contributes is `subgroups * subgroup_width` lanes each owning `vector`
///   adjacent columns.
/// * [`SchedPoint::Fold`] is what a `Family::GenericFold` contraction carries.
///   Its lane-group width becomes the workgroup width; the reduction it
///   describes is in-register in this nest, so no cross-lane tree is emitted
///   and `iterations` prices a k loop that is already sequential.
///
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
        SchedPoint::Fold(s) => {
            let lanes = match s {
                FoldStrat::Subgroup => width,
                FoldStrat::WgTree { lane_group }
                | FoldStrat::LoopThenTree { lane_group, .. } => lane_group.max(1),
            };
            Tile {
                tm: 1,
                tn: 1,
                row_groups: 1,
                col_groups: lanes,
            }
        }
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
            pre_a,
            pre_b,
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
            pre_a,
            pre_b,
            post,
        ),
        L1::KQContract {
            fmt,
            layout,
            act,
            m,
            n,
            k,
            post,
            a,
            b,
            ..
        } => build_quantized(
            cx,
            caps,
            tile_of(theta, caps)?.columns_only(),
            QDims {
                fmt: *fmt,
                layout: *layout,
                act: *act,
                m: *m,
                n: *n,
                k: *k,
            },
            post,
            a,
            b,
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
fn build(
    cx: &LowerCtx<'_>,
    caps: &Caps,
    d: Dims,
    tile: Tile,
    a: &Operand,
    b: &Operand,
    pre_a: &ScalarExpr,
    pre_b: &ScalarExpr,
    post: &ScalarExpr,
) -> Result<KernelIr> {
    let m = konst(d.m, "m")?.max(1);
    let n = konst(d.n, "n")?.max(1);
    let k = konst(d.k, "k")?.max(1);
    let batch = konst(d.batch, "batch")?.max(1);

    let binds = Binds::build(cx)?;
    let uniforms = binds.buffers.first().cloned();
    let a_buf = binds.of(a.src)?;
    let b_buf = binds.of(b.src)?;
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
    let mut b_vals = Vec::with_capacity(tn as usize);
    for j in 0..tn as usize {
        let b_index = bin(
            BinOp::Add,
            bin(
                BinOp::Mul,
                bin(
                    BinOp::Add,
                    bin(BinOp::Mul, bidx.clone(), lit_u32(k), u32_ty()),
                    k_idx.clone(),
                    u32_ty(),
                ),
                lit_u32(n),
                u32_ty(),
            ),
            cols[j].clone(),
            u32_ty(),
        );
        b_vals.push(
            Translate {
                args: &[load(Arc::clone(&b_buf), b_index, col_oks[j].clone())],
                coords: &[],
                uniforms: uniforms.clone(),
            }
            .run(pre_b)?,
        );
    }

    let mut updates = Vec::with_capacity(accs.len());
    for i in 0..tm as usize {
        let a_index = bin(
            BinOp::Add,
            bin(
                BinOp::Mul,
                bin(
                    BinOp::Add,
                    bin(BinOp::Mul, bidx.clone(), lit_u32(m), u32_ty()),
                    rows[i].clone(),
                    u32_ty(),
                ),
                lit_u32(k),
                u32_ty(),
            ),
            k_idx.clone(),
            u32_ty(),
        );
        let a_val = Translate {
            args: &[load(Arc::clone(&a_buf), a_index, row_oks[i].clone())],
            coords: &[],
            uniforms: uniforms.clone(),
        }
        .run(pre_a)?;
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

struct QDims {
    fmt: fusor2_ir::dtype::QFmt,
    layout: fusor2_ir::dtype::QLayout,
    act: fusor2_ir::dtype::QAct,
    m: Dim,
    n: Dim,
    k: Dim,
}

/// `KQContract`: `out[row, col] = sum_k act[row, k] * W[col, k]`, with `W`
/// still block-quantized and decoded inside the k nest.
///
/// One lane per output column, one workgroup per `(output row, column block)`,
/// and a loop over the weight's blocks. The two activation packings differ
/// only in the node the k-loop body builds — `Dequantize` + `LaneOf` +
/// `mul_add` for [`QAct::F32`], one `QuantizedDot` for [`QAct::Q8Dp4a`] — and
/// both are expanded into ordinary `TileExpr`s by `emit::quantized`, so there
/// is no per-format code here either.
///
/// **No cross-lane reduction.** Each lane loops over every block of its own
/// row, so its accumulator is already the whole dot product.
///
/// The lane count is the schedule point's, and the column block is a grid
/// axis: this body had the same written-in 64 the dense one did, and with one
/// workgroup per output row it could not reach a weight matrix with more than
/// 64 rows at all.
#[allow(clippy::too_many_arguments)]
fn build_quantized(
    cx: &LowerCtx<'_>,
    caps: &Caps,
    tile: Tile,
    d: QDims,
    post: &ScalarExpr,
    a: &Operand,
    b: &Operand,
) -> Result<KernelIr> {
    let m = konst(d.m, "m")?.max(1);
    let n = konst(d.n, "n")?.max(1);
    let k = konst(d.k, "k")?.max(1);
    let (fmt, layout, act) = (d.fmt, d.layout, d.act);

    let spec = fusor2_gguf::block_spec(fmt, layout);
    if !spec.activation.contains(&act) {
        return Err(Error::Legality(format!(
            "{fmt:?}/{layout:?} does not support activation packing {act:?}"
        )));
    }
    let block_elems = u32::from(spec.elements).max(1);
    let blocks_per_row = k.div_ceil(block_elems).max(1);

    let binds = Binds::build(cx)?;
    let uniforms = binds.buffers.first().cloned();
    let a_buf = binds.of(a.src)?;
    let w_buf = binds.of(b.src)?;
    let out_buf = binds.of(cx.launch.root)?;

    let weights = QuantizedView {
        data: StorageView {
            buffer: Arc::clone(&w_buf),
            offset: 0,
            layout: w_buf.layout.clone(),
        },
        fmt,
        layout,
        rows: n,
        cols: k,
    };

    // One row per workgroup, `block` columns of it per workgroup, every column
    // reached by `ceil(n / block)` blocks of the grid.
    let block = tile
        .fit(1, n, caps.limits.max_compute_invocations_per_workgroup)
        .cols();
    let n_blocks = n.div_ceil(block).max(1);
    let f32_ty = ElementType::Scalar(ScalarElement::F32);
    let bool_ty = ElementType::Scalar(ScalarElement::Bool);

    let pid = TileExpr::new(
        TileExprKind::Builtin(Builtin::ProgramId(WorkgroupAxis::X)),
        u32_ty(),
    );
    let lane = TileExpr::new(TileExprKind::Builtin(Builtin::Lane), u32_ty());
    let row = bin(BinOp::Div, pid.clone(), lit_u32(n_blocks), u32_ty());
    let col = bin(
        BinOp::Add,
        bin(
            BinOp::Mul,
            bin(BinOp::Rem, pid, lit_u32(n_blocks), u32_ty()),
            lit_u32(block),
            u32_ty(),
        ),
        lane,
        u32_ty(),
    );
    let live = cmp(CmpOp::Lt, col.clone(), lit_u32(n));

    let blk = Arc::new(LocalDecl::new(u32_ty()));
    let blk_read = TileExpr::new(TileExprKind::LoadLocal(Arc::clone(&blk)), u32_ty());
    let k_base = bin(BinOp::Mul, blk_read, lit_u32(block_elems), u32_ty());

    // The activation elements of this block, read once and shared by the
    // column this lane owns.
    let mut activations = Vec::with_capacity(block_elems as usize);
    for e in 0..block_elems {
        let kk = bin(BinOp::Add, k_base.clone(), lit_u32(e), u32_ty());
        let in_k = cmp(CmpOp::Lt, kk.clone(), lit_u32(k));
        let index = bin(
            BinOp::Add,
            bin(BinOp::Mul, row.clone(), lit_u32(k), u32_ty()),
            kk,
            u32_ty(),
        );
        activations.push(load(Arc::clone(&a_buf), index, in_k));
    }

    // The decode program's flat element index is `k_base + col + lane`
    // (`fusor2-gguf/src/decode.rs`, "Addressing convention"): the *caller*
    // folds the row stride into `col`. The bare weight-row index would read `k`
    // elements into row 0 instead of the start of row `col`.
    let row_start = bin(BinOp::Mul, col.clone(), lit_u32(k), u32_ty());

    let acc = Arc::new(LocalDecl::new(f32_ty));
    let prev = TileExpr::new(TileExprKind::LoadLocal(Arc::clone(&acc)), f32_ty);
    let fill = lit_f32(0.0);
    let contribution = match act {
        fusor2_ir::dtype::QAct::F32 => {
            let decoded = TileExpr::new(
                TileExprKind::Dequantize {
                    src: weights.clone(),
                    k_base: k_base.clone(),
                    col: row_start.clone(),
                    mask: live.clone(),
                    fill: fill.clone(),
                    lanes: block_elems,
                },
                f32_ty,
            );
            let mut sum = lit_f32(0.0);
            for (e, a_v) in activations.iter().enumerate() {
                let w = TileExpr::new(
                    TileExprKind::LaneOf {
                        block: decoded.clone(),
                        lane: e as u32,
                    },
                    f32_ty,
                );
                sum = bin(
                    BinOp::Add,
                    sum,
                    bin(BinOp::Mul, a_v.clone(), w, f32_ty),
                    f32_ty,
                );
            }
            sum
        }
        packing => TileExpr::new(
            TileExprKind::QuantizedDot {
                src: weights.clone(),
                packing,
                activations,
                k_base: k_base.clone(),
                col: row_start.clone(),
                mask: live.clone(),
                fill: fill.clone(),
            },
            f32_ty,
        ),
    };

    let mut body = vec![Stmt::Loop {
        count: Some(lit_u32(blocks_per_row)),
        index: Some(blk),
        accumulators: vec![Accumulator {
            local: Arc::clone(&acc),
            init: lit_f32(0.0),
            update: bin(BinOp::Add, prev, contribution, f32_ty),
        }],
        body: vec![],
    }];

    let total = TileExpr::new(TileExprKind::LoadLocal(acc), f32_ty);
    let value = Translate {
        args: &[total],
        coords: &[],
        uniforms,
    }
    .run(post)?;
    let index = bin(
        BinOp::Add,
        bin(BinOp::Mul, row.clone(), lit_u32(n), u32_ty()),
        col,
        u32_ty(),
    );
    let row_ok = cmp(CmpOp::Lt, row, lit_u32(m));
    body.push(Stmt::Store {
        dst: StorageView {
            buffer: Arc::clone(&out_buf),
            offset: 0,
            layout: out_buf.layout.clone(),
        },
        addr: Addr::Linear(index),
        value,
        mask: bin(BinOp::LogicalAnd, row_ok, live, bool_ty),
    });

    Ok(KernelIr {
        buffers: binds.buffers,
        grid: [m.saturating_mul(n_blocks).max(1), 1, 1],
        block,
        body,
        byte_arena: None,
        name: "cpu_qcontract",
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
        assert!(tile_of(SchedPoint::Sgemv(SgemvParams { chunk: 2, vector: 4, subgroups: 1 }), caps).is_ok());
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
            SchedPoint::Sgemv(SgemvParams { chunk: 2, vector: 4, subgroups: 1 }),
            SchedPoint::Sgemv(SgemvParams { chunk: 8, vector: 1, subgroups: 4 }),
            SchedPoint::Fold(FoldStrat::Subgroup),
            SchedPoint::Fold(FoldStrat::WgTree { lane_group: 64 }),
            SchedPoint::Fold(FoldStrat::LoopThenTree { iterations: 4, lane_group: 256 }),
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
    use fusor2_ir::ir::level1::{AccessPlan, Family, ScheduleDomain, SgemmParams, SgemvParams};
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
            pre_a: ScalarExpr::arg(0, Dtype::F32),
            pre_b: ScalarExpr::arg(0, Dtype::F32),
            post: ScalarExpr::arg(0, Dtype::F32),
            acc: Dtype::F32,
            a: alias(&g, a_id),
            b: alias(&g, b_id),
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
        let theta = SchedPoint::Sgemv(SgemvParams { chunk: 2, vector: 4, subgroups: 1 });
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
            SchedPoint::Sgemv(SgemvParams { chunk: 2, vector: 4, subgroups: 1 }),
            SchedPoint::Sgemv(SgemvParams { chunk: 1, vector: 1, subgroups: 8 }),
            SchedPoint::Fold(FoldStrat::Subgroup),
            SchedPoint::Fold(FoldStrat::WgTree { lane_group: 64 }),
            SchedPoint::Fold(FoldStrat::LoopThenTree { iterations: 2, lane_group: 128 }),
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

    // -- KQContract ---------------------------------------------------------

    /// One Q8_0 block per weight row, filled from a cheap LCG with an explicit
    /// finite scale, plus the rows it decodes to.
    fn q8_weights(n: u32, k: u32) -> (Vec<u8>, Vec<f32>) {
        let fmt = fusor2_ir::dtype::QFmt::Q8_0;
        let layout = fusor2_ir::dtype::QLayout::Native;
        let block_bytes = fmt.block_bytes(layout) as usize;
        let elems = fmt.block_elements() as usize;
        assert_eq!(k as usize, elems, "one block per weight row keeps the rows aligned");
        let mut bytes = Vec::with_capacity(n as usize * block_bytes);
        let mut decoded = vec![0.0f32; (n * k) as usize];
        for r in 0..n as usize {
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
                &mut decoded[r * k as usize..(r + 1) * k as usize],
            );
            bytes.extend_from_slice(&block);
        }
        (bytes, decoded)
    }

    fn upload_bytes(target: &CpuTarget, data: &[u8]) -> Buf {
        let buf = target
            .alloc(data.len().max(4) as u64, Persistence::Step)
            .unwrap();
        let raw = buf.downcast_ref::<AlignedBuf>().unwrap();
        // SAFETY: nothing else holds this buffer yet; the pool handed it back
        // because its refcount was one.
        let slice = unsafe { std::slice::from_raw_parts_mut(raw.as_mut_ptr(), raw.len()) };
        slice.fill(0);
        slice[..data.len()].copy_from_slice(data);
        buf
    }

    /// `out[row, col] = sum_k act[row, k] * W[col, k]` with `W` block-decoded.
    ///
    /// **`n` here is the weight's row count**, and nothing in the conformance
    /// suite gives a CPU quantized contraction more than 32 of them, so this
    /// is the only place the column-block grid of the quantized body is
    /// exercised at all.
    fn run_qcontract(theta: SchedPoint, act: fusor2_ir::dtype::QAct, m: u32, n: u32, k: u32) {
        let fmt = fusor2_ir::dtype::QFmt::Q8_0;
        let layout = fusor2_ir::dtype::QLayout::Native;
        let (bytes, weights) = q8_weights(n, k);
        let a = sample(m * k + 3, (m * k) as usize);

        let mut g = graph();
        let a_id = buffer(&mut g, &[u64::from(m), u64::from(k)]);
        let next = g.len() as u32;
        let w_id = g
            .add(Op::L0(L0::Leaf(LeafKind::Quantized {
                name: BufferId(next),
                fmt,
                layout,
                shape: smallvec::smallvec![Dim::Const(u64::from(n)), Dim::Const(u64::from(k))],
            })))
            .unwrap();
        let out_id = buffer(&mut g, &[u64::from(m), u64::from(n)]);

        let node = Node {
            op: Op::L1(L1::KQContract {
                fmt,
                layout,
                act,
                m: Dim::Const(u64::from(m)),
                n: Dim::Const(u64::from(n)),
                k: Dim::Const(u64::from(k)),
                acc: Dtype::F32,
                post: ScalarExpr::arg(0, Dtype::F32),
                a: alias(&g, a_id),
                b: alias(&g, w_id),
                sched: sgemm_domain_of(theta),
            }),
            level: Level::L1,
            children: smallvec::smallvec![a_id, w_id],
        };
        let plan = Plan {
            extraction: Extraction::default(),
            launches: vec![Launch {
                root: out_id,
                members: smallvec::smallvec![out_id],
                bindings: vec![
                    BindingPlan { binding: 1, value: a_id, kind: BindKind::Read },
                    BindingPlan { binding: 2, value: w_id, kind: BindKind::Read },
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
        let a_buf = upload(&target, &a);
        let w_buf = upload_bytes(&target, &bytes);
        let out = upload(&target, &vec![0.0; (m * n) as usize]);
        let artifact = target.emit(&ir).unwrap();
        target
            .launch(
                &artifact,
                ir.grid,
                &[a_buf, w_buf, out.clone()],
                &Default::default(),
            )
            .unwrap();
        let got = download(&out, (m * n) as usize);

        // `QAct::Q8Dp4a` rounds the activations through an int8 grid with a
        // per-block scale of `max|a| / 127` before accumulating
        // (`emit::quantized`), so each element carries up to half a step of
        // error and the dot carries `sum|w| * max|a| / 254` of it. That is the
        // packing's own arithmetic, not slack: the coverage claim this case
        // exists for — a column the grid never reaches comes back 0.0 — is off
        // by the whole dot product, an order of magnitude above the bound.
        let amax = a
            .iter()
            .fold(0.0f64, |acc, v| acc.max(f64::from(v.abs())));
        let mut informative = 0usize;
        for row in 0..m as usize {
            for col in 0..n as usize {
                let w = &weights[col * k as usize..(col + 1) * k as usize];
                let want: f64 = (0..k as usize)
                    .map(|t| a[row * k as usize + t] as f64 * w[t] as f64)
                    .sum();
                let mut tol = 2e-3 * want.abs().max(1.0);
                if matches!(act, fusor2_ir::dtype::QAct::Q8Dp4a) {
                    let sum_w: f64 = w.iter().map(|v| f64::from(v.abs())).sum();
                    tol += sum_w * amax / 254.0;
                }
                let g = got[row * n as usize + col];
                assert!(
                    (g as f64 - want).abs() <= tol,
                    "{theta:?}/{act:?} at [{m},{n},{k}] row {row} col {col}: \
                     got {g}, want {want} (tolerance {tol})"
                );
                informative += usize::from(want.abs() > 4.0 * tol);
            }
        }
        // An output an unreached column could match by accident proves
        // nothing: most entries have to sit well outside the tolerance.
        assert!(
            informative * 2 >= (m * n) as usize,
            "only {informative} of {} entries are far enough from zero to \
             witness coverage",
            m * n
        );
    }

    /// The quantized body had the same written-in 64 and one workgroup per
    /// output row, so a weight matrix with more than 64 rows was unreachable.
    #[test]
    fn the_quantized_body_reaches_every_weight_row() {
        for act in [fusor2_ir::dtype::QAct::F32, fusor2_ir::dtype::QAct::Q8Dp4a] {
            for theta in [
                SchedPoint::Fold(FoldStrat::Subgroup),
                SchedPoint::Fold(FoldStrat::WgTree { lane_group: 256 }),
                SchedPoint::Sgemv(SgemvParams { chunk: 2, vector: 4, subgroups: 1 }),
                SchedPoint::Sgemm(SgemmParams {
                    double_buffer: false,
                    bm: 16,
                    bn: 32,
                    bk: 8,
                    tm: 2,
                    tn: 2,
                }),
            ] {
                for n in [3u32, 64, 96, 130] {
                    run_qcontract(theta, act, 2, n, 32);
                }
            }
        }
    }

    /// Batches are not aliased: the m-block and n-block decomposition has to
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
