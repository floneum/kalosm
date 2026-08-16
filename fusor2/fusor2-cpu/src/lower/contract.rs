//! `Contract` on CPU: real blocking and a register microkernel, so
//! bias/gelu/dequant epilogues fuse into the k-loop.
//!
//! The nest is a `TM x TN` register tile whose accumulators are `Stmt::Loop`
//! accumulators, staying resident across the whole k nest and never reloading.
//! Because the accumulators are in the IR, `pre_a`, `pre_b` and `post` fuse
//! into the k-loop epilogue.
//!
//! The tile shape comes from `theta` (the schedule point), and the grid covers
//! the whole output with coverage `ceil(m / rows) * ceil(n / cols)` blocks
//! whatever the tile is.

use fusor2_ir::device::Caps;
use fusor2_ir::error::Error;
use fusor2_ir::ir::launch::{ContractSide, Launch, SchedPoint};
use fusor2_ir::ir::kernel::{
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
/// `batch * ceil(m / rows) * ceil(n / cols)` workgroups, so a tile of any
/// shape still computes every output element: a schedule point moves the
/// launch shape and the register reuse, never the answer.
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
///   `double_buffer` size a staged workgroup tile this nest does not have.
/// * [`SchedPoint::Sgemv`] names the workgroup's width in subgroups; k is
///   walked sequentially, so it contributes `subgroups * subgroup_width`
///   lanes each owning `vector` adjacent columns.
/// Anything else names no contraction geometry: [`SchedPoint::Point`] means
/// no schedule decision was made for this node, which is a plan answer, not a
/// tile to invent.
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

pub(crate) fn lower(caps: &Caps, node: &Node, theta: SchedPoint, cx: &LowerCtx<'_>) -> Result<KernelIr> {
    let Op::Launch(op) = &node.op else {
        return Err(Error::Legality("not a Launch node".into()));
    };
    match op {
        Launch::Contract {
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

/// The three strides this kernel indexes an operand with, collapsed out of the
/// operand's own per-axis layout.
///
/// The kernel's three loop indices are the *products* `batch`, `m`/`k` and
/// `k`/`n`, while a layout has one axis per einsum label — `bhqd` is four axes
/// collapsing to `batch=b*h, m=q, k=d`. Group boundaries are recovered from the
/// extents alone: walk the axes outermost-first and cut each group when its
/// running extent product reaches that group's total.
///
/// A group collapses to a single stride only when its own axes are internally
/// dense — `stride[i] == stride[i+1] * extent[i+1]` — and then the collapsed
/// stride is the innermost axis's. `None` when they are not, which is a layout
/// this three-stride kernel cannot address.
///
/// Extent-1 axes are dropped up front: an axis of extent 1 always contributes
/// `0` to every address, so neither its stride nor its position among the
/// other axes is observable, and letting one enter the density test compares
/// a stride nothing reads (a broadcast KV head would be refused for a "gap"
/// between two axes that address the same byte).
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

    // One workgroup per `(batch, m block, n block)`.
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

    // `tn` B loads and `tm` broadcast A loads per k step, then `tm * tn` FMAs.
    // The emitter memoizes identical expressions, so each operand element is
    // read once however many accumulators consume it.
    //
    // Both operands are addressed through their own layouts: `permuted_alias`
    // in `fusor2-tile` mints a non-contiguous `Alias` for any contraction
    // whose spec is not already in kernel axis order, and reading that densely
    // is a miscompile.
    //
    // A side is a list of operands, each with its own buffer and layout, all
    // addressed by that side's `(batch, row, k)` or `(batch, k, col)` triple;
    // several entries is a side that absorbed a multi-buffer producer. Each
    // entry is either a bound buffer with its collapsed strides, or a `Const`
    // leaf already folded to its literal — those have no binding.
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
