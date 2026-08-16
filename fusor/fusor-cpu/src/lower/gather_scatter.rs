//! `Gather` and `Scatter`.
//!
//! Both `ScatterMode`s name one map and differ only in strategy. On a target
//! with no f32 atomic they share one nest: one lane per output element, a
//! counted loop over the updates. Every output element is written by exactly one
//! lane, so no atomic is needed and the result is bit-reproducible.
//!
//! Both nests read their lane tiling off `theta`. `Gather` and `Scatter` carry
//! the same elementwise `ScheduleDomain::Map` a `Map` carries, and can use
//! `tm` elements per lane like the grid-strided register tile in `map_fold`,
//! amortizing the index read in scatter workloads.

use fusor_ir::Result;
use fusor_ir::device::Caps;
use fusor_ir::error::Error;
use fusor_ir::ir::kernel::{
    Accumulator, Addr, KernelIr, Local, LocalDecl, Stmt, StorageView, TileExpr, TileExprKind,
};
use fusor_ir::ir::launch::{Launch, ScatterMode, SchedPoint};
use fusor_ir::ir::logical::ScatterCombine;
use fusor_ir::ir::{Node, Op};
use fusor_ir::scalar::{BinOp, CmpOp};
use fusor_ir::target::LowerCtx;
use std::sync::Arc;

use super::{
    Binds, bin, cmp, const_extents, default_block, global_lane, grid_for, lit_u32, u32_ty,
};

pub(crate) fn lower(
    caps: &Caps,
    node: &Node,
    theta: SchedPoint,
    cx: &LowerCtx<'_>,
) -> Result<KernelIr> {
    let Op::Launch(op) = &node.op else {
        return Err(Error::Legality("not a Launch node".into()));
    };
    let tm = lane_tile(theta)?;
    match op {
        Launch::Gather {
            space, axis, ops, ..
        } => gather(caps, cx, space, *axis, ops, tm),
        Launch::Scatter {
            space,
            axis,
            mode,
            combine,
            ops,
            ..
        } => scatter(caps, cx, space, *axis, *mode, *combine, ops, tm),
        _ => Err(Error::Legality("gather_scatter got a foreign node".into())),
    }
}

/// How many output elements one lane owns, read off `theta`.
///
/// [`SchedPoint::Point`] is the floor lowering's untiled point, so it is
/// answered with 1 rather than refused. Any other family on these nodes is a
/// planner bug.
///
/// `MapTiling::dim` is ignored: this backend tiles with a grid stride
/// (`flat + t * grid.x * block`), exactly as `lower_map` does, so one lane's
/// elements are a fixed distance apart whatever axis the domain named and
/// coverage stays a bijection with no divisibility side condition.
/// `MapTiling::vector` is ignored too: `emit::pick_width` chooses the SIMD
/// instantiation from `caps.simd_widths` and the block width.
fn lane_tile(theta: SchedPoint) -> Result<u32> {
    match theta {
        SchedPoint::Map(t) => Ok(t.tm.max(1)),
        SchedPoint::Point => Ok(1),
        other => Err(Error::Legality(format!(
            "a gather or scatter needs SchedPoint::Map, got {other:?}"
        ))),
    }
}

fn view(buf: &Arc<fusor_ir::ir::kernel::BufferDecl>) -> StorageView {
    StorageView {
        buffer: Arc::clone(buf),
        offset: 0,
        layout: buf.layout.clone(),
    }
}

/// `out[i, rest] = src[idx[i], rest]`, one lane per output element.
///
/// Both `GatherMode`s share this nest; they differ only in how many output
/// elements one lane owns, which is a schedule attribute rather than a
/// different kernel.
fn gather(
    caps: &Caps,
    cx: &LowerCtx<'_>,
    space: &fusor_ir::ir::launch::IndexSpace,
    axis: u32,
    ops: &[fusor_ir::ir::launch::Operand],
    tm: u32,
) -> Result<KernelIr> {
    if ops.len() < 2 {
        return Err(Error::Legality(
            "a gather needs a source and an index operand".into(),
        ));
    }
    let binds = Binds::build(cx)?;
    let extents = const_extents(&space.dims)?;
    let n: u64 = extents.iter().map(|e| *e as u64).product::<u64>().max(1);
    let axis = axis as usize;
    if axis >= extents.len() {
        return Err(Error::Legality("gather axis is out of range".into()));
    }
    let inner: u32 = extents[axis + 1..].iter().product::<u32>().max(1);
    let out_stride = extents[axis].max(1) * inner;
    // The source's extent along the gathered axis, the only axis where source
    // and output disagree. Scaling the source's outer coordinate by the
    // output's stride reads the wrong row whenever the index vector is not
    // exactly as long as the axis it indexes.
    let src_shape = const_extents(ops[0].layout.shape())?;
    let src_axis = *src_shape
        .get(axis)
        .ok_or_else(|| Error::Legality("gather axis is out of range for the source".into()))?;
    let src_stride = src_axis.max(1) * inner;

    let src = super::operand_src(cx, &binds, ops[0].src)?;
    let idx = super::operand_src(cx, &binds, ops[1].src)?;
    let out = binds.of(cx.launch.root)?;

    // `tm` elements per lane, a whole grid apart, so lane 0..stride covers
    // [0, tm*stride) >= [0, n) exactly once with no divisibility condition.
    let block = default_block(caps);
    let grid = grid_for(n.div_ceil(u64::from(tm)), block);
    let stride = grid[0].saturating_mul(block);

    let mut body = Vec::with_capacity(tm as usize);
    for t in 0..tm {
        let flat = if t == 0 {
            global_lane(block)
        } else {
            bin(
                BinOp::Add,
                global_lane(block),
                lit_u32(t.saturating_mul(stride)),
                u32_ty(),
            )
        };
        let mask = cmp(CmpOp::Lt, flat.clone(), lit_u32(n as u32));
        // Split the flat output index into (outer, gathered, inner).
        let outer = bin(BinOp::Div, flat.clone(), lit_u32(out_stride), u32_ty());
        let rest = bin(BinOp::Rem, flat.clone(), lit_u32(out_stride), u32_ty());
        let g = bin(BinOp::Div, rest.clone(), lit_u32(inner), u32_ty());
        let within = bin(BinOp::Rem, rest, lit_u32(inner), u32_ty());

        let row = idx.at(g, mask.clone());
        // The gathered coordinate replaces `g`; everything else is unchanged —
        // but the outer coordinate steps by the *source's* stride.
        let src_index = bin(
            BinOp::Add,
            bin(
                BinOp::Add,
                bin(BinOp::Mul, outer, lit_u32(src_stride), u32_ty()),
                bin(BinOp::Mul, row, lit_u32(inner), u32_ty()),
                u32_ty(),
            ),
            within,
            u32_ty(),
        );
        let value = src.at(src_index, mask.clone());
        body.push(Stmt::Store {
            dst: view(&out),
            addr: Addr::Linear(flat),
            value,
            mask,
        });
    }

    Ok(KernelIr {
        buffers: binds.buffers,
        grid,
        block,
        body,
        byte_arena: None,
        name: "cpu_gather",
    })
}

/// `out = base` with `out[.., idx[u], ..] (combine)= upd[.., u, ..]`.
///
/// The nest walks the output, not the updates: a `Scatter`'s value is its
/// *base* with the updates applied, and the plan gives that value its own
/// buffer — nothing copies the base in beforehand — so a kernel that only
/// visits the written elements leaves every other one undefined.
///
/// One lane per output element, a counted loop over the updates, and the
/// accumulator carried in a register: the write map is not injective, so the
/// nest declares an associative `combine` (`verify_launch` invariant 3) and
/// discharges it by making each output element the *only* writer of itself.
/// The accumulation order is therefore fixed and the result bit-reproducible
/// at any thread count — no atomic, on a target that has none for f32.
///
/// `tm` output elements per lane, in one loop: the loop costs one `idx[u]`
/// read per output element per update, and `tm` accumulators in the same loop
/// share that read.
fn scatter(
    caps: &Caps,
    cx: &LowerCtx<'_>,
    space: &fusor_ir::ir::launch::IndexSpace,
    axis: u32,
    _mode: ScatterMode,
    combine: ScatterCombine,
    ops: &[fusor_ir::ir::launch::Operand],
    tm: u32,
) -> Result<KernelIr> {
    // Either mode names a *strategy* for the same map. This nest needs no
    // atomic, so `Atomic{Add}` is legal here even though `caps.atomic_f32`
    // is false.
    if ops.len() < 3 {
        return Err(Error::Legality(
            "a scatter needs base, index and update operands".into(),
        ));
    }
    let _ = caps;
    let binds = Binds::build(cx)?;
    let geom = super::scatter_geometry(cx, space, axis, ops)?;
    let (outer, bins, inner, updates) = (geom.outer, geom.bins, geom.inner, geom.updates);
    let total = outer as u64 * bins as u64 * inner as u64;

    let base = super::operand_src(cx, &binds, ops[0].src)?;
    let idx = super::operand_src(cx, &binds, ops[1].src)?;
    let upd = super::operand_src(cx, &binds, ops[2].src)?;
    let out = binds.of(cx.launch.root)?;
    let elem = out.element;

    let block = default_block(caps);
    let grid = grid_for(total.div_ceil(u64::from(tm)), block);
    let lane_stride = grid[0].saturating_mul(block);

    let u_local: Local = Arc::new(LocalDecl::new(u32_ty()));
    let u = TileExpr::new(TileExprKind::LoadLocal(Arc::clone(&u_local)), u32_ty());

    // The lowest offset is live whenever any of this lane's offsets is, so it
    // is the right mask for the one index read they share.
    let first_live = cmp(CmpOp::Lt, global_lane(block), lit_u32(total as u32));
    let u_bin = idx.at(u.clone(), first_live);

    let mut accumulators = Vec::with_capacity(tm as usize);
    let mut stores = Vec::with_capacity(tm as usize);
    for t in 0..tm {
        let flat = if t == 0 {
            global_lane(block)
        } else {
            bin(
                BinOp::Add,
                global_lane(block),
                lit_u32(t.saturating_mul(lane_stride)),
                u32_ty(),
            )
        };
        let live = cmp(CmpOp::Lt, flat.clone(), lit_u32(total as u32));
        // (outer, destination bin, inner) of this output element.
        let o = bin(BinOp::Div, flat.clone(), lit_u32(bins * inner), u32_ty());
        let dest = bin(
            BinOp::Rem,
            bin(BinOp::Div, flat.clone(), lit_u32(inner), u32_ty()),
            lit_u32(bins),
            u32_ty(),
        );
        let within = bin(BinOp::Rem, flat.clone(), lit_u32(inner), u32_ty());

        let acc_local: Local = Arc::new(LocalDecl::new(elem));
        let acc = TileExpr::new(TileExprKind::LoadLocal(Arc::clone(&acc_local)), elem);

        let hit = cmp(CmpOp::Eq, u_bin.clone(), dest);
        // `upd[o, u, within]` in the update's own flat space.
        let upd_index = bin(
            BinOp::Add,
            bin(
                BinOp::Mul,
                bin(
                    BinOp::Add,
                    bin(BinOp::Mul, o, lit_u32(updates), u32_ty()),
                    u.clone(),
                    u32_ty(),
                ),
                lit_u32(inner),
                u32_ty(),
            ),
            within,
            u32_ty(),
        );
        let contribution = upd.at(upd_index, live.clone());
        let combined = match combine {
            // `Add` duplicates accumulate — normative: an embedding table
            // receiving one token twice gets the summed gradient. `Set` is only
            // reachable when the node proved its indices unique.
            ScatterCombine::Add => bin(BinOp::Add, acc.clone(), contribution, elem),
            ScatterCombine::Set => contribution,
        };
        let update = TileExpr::new(
            TileExprKind::Select {
                condition: hit,
                accept: combined,
                reject: acc.clone(),
            },
            elem,
        );

        accumulators.push(Accumulator {
            local: Arc::clone(&acc_local),
            init: base.at(flat.clone(), live.clone()),
            update,
        });
        stores.push(Stmt::Store {
            dst: view(&out),
            addr: Addr::Linear(flat),
            value: acc,
            mask: live,
        });
    }

    let mut body = vec![Stmt::Loop {
        count: Some(lit_u32(updates)),
        index: Some(u_local),
        accumulators,
        body: Vec::new(),
    }];
    body.extend(stores);

    Ok(KernelIr {
        buffers: binds.buffers,
        grid,
        block,
        body,
        byte_arena: None,
        name: "cpu_scatter",
    })
}
