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
use fusor_ir::error::Error;
use fusor_ir::ir::kernel::{
    Accumulator, Addr, KernelIr, ScalarElement, Stmt, TileExpr, TileExprKind,
};
use fusor_ir::ir::launch::{IndexSpace, Launch, Operand, SchedPoint};
use fusor_ir::ir::logical::ScatterCombine;
use fusor_ir::ir::{Node, Op};
use fusor_ir::target::LowerCtx;
use fusor_tile::build::{Kernel, ScatterGeometry};

use super::{Binds, DEFAULT_BLOCK, const_extents, global_lane, grid_for, operand_src, view};

pub(crate) fn lower(node: &Node, theta: SchedPoint, cx: &LowerCtx<'_>) -> Result<KernelIr> {
    let Op::Launch(op) = &node.op else {
        return Err(Error::Legality("not a Launch node".into()));
    };
    let tm = lane_tile(theta)?;
    match op {
        Launch::Gather {
            space, axis, ops, ..
        } => gather(cx, space, *axis, ops, tm),
        Launch::Scatter {
            axis, combine, ops, ..
        } => scatter(cx, *axis, *combine, ops, tm),
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

/// The `tm` flat output indices one lane owns, a whole grid apart, so lanes
/// `0..stride` cover `[0, tm * stride) >= [0, n)` exactly once with no
/// divisibility condition; and the grid that makes it so.
fn lane_offsets(b: &Kernel, n: u64, tm: u32) -> ([u32; 3], Vec<TileExpr>) {
    let grid = grid_for(n.div_ceil(u64::from(tm)), DEFAULT_BLOCK);
    let stride = grid[0].saturating_mul(DEFAULT_BLOCK);
    let lane = global_lane(b, DEFAULT_BLOCK);
    let offsets = (0..tm)
        .map(|t| match t {
            0 => lane.clone(),
            _ => b.add(lane.clone(), b.u32(t.saturating_mul(stride))),
        })
        .collect();
    (grid, offsets)
}

/// `out[i, rest] = src[idx[i], rest]`, one lane per output element.
///
/// Both `GatherMode`s share this nest; they differ only in how many output
/// elements one lane owns, which is a schedule attribute rather than a
/// different kernel.
fn gather(
    cx: &LowerCtx<'_>,
    space: &IndexSpace,
    axis: u32,
    ops: &[Operand],
    tm: u32,
) -> Result<KernelIr> {
    let [src, idx, ..] = ops else {
        return Err(Error::Legality(
            "a gather needs a source and an index operand".into(),
        ));
    };
    let b = Kernel::new();
    let binds = Binds::build(cx)?;
    let extents = const_extents(cx, &space.dims)?;
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
    let src_shape = const_extents(cx, src.layout.shape())?;
    let src_axis = *src_shape
        .get(axis)
        .ok_or_else(|| Error::Legality("gather axis is out of range for the source".into()))?;
    let src_stride = src_axis.max(1) * inner;

    let src = operand_src(&b, cx, &binds, src.src)?;
    let idx = operand_src(&b, cx, &binds, idx.src)?;
    let out = view(&binds.of(cx.launch.root)?);
    let (grid, offsets) = lane_offsets(&b, n, tm);
    let mut body = Vec::with_capacity(tm as usize);
    for flat in offsets {
        let mask = b.lt(flat.clone(), b.u32(n as u32));
        // Split the flat output index into (outer, gathered, inner).
        let (outer, rest) = b.divrem(flat.clone(), b.u32(out_stride));
        let (g, within) = b.divrem(rest, b.u32(inner));
        // The gathered coordinate replaces `g`; everything else is unchanged —
        // but the outer coordinate steps by the *source's* stride.
        let row = idx.at(&b, g, mask.clone());
        let src_index = b.add(
            b.add(b.mul(outer, b.u32(src_stride)), b.mul(row, b.u32(inner))),
            within,
        );
        body.push(Stmt::Store {
            dst: out.clone(),
            addr: Addr::Linear(flat),
            value: src.at(&b, src_index, mask.clone()),
            mask,
        });
    }

    Ok(KernelIr {
        buffers: binds.buffers,
        grid,
        block: DEFAULT_BLOCK,
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
/// at any thread count — no atomic, on a target that has none for f32, so
/// either `ScatterMode` lowers here.
///
/// `tm` output elements per lane, in one loop: the loop costs one `idx[u]`
/// read per output element per update, and `tm` accumulators in the same loop
/// share that read.
fn scatter(
    cx: &LowerCtx<'_>,
    axis: u32,
    combine: ScatterCombine,
    ops: &[Operand],
    tm: u32,
) -> Result<KernelIr> {
    let [base, idx, upd, ..] = ops else {
        return Err(Error::Legality(
            "a scatter needs base, index and update operands".into(),
        ));
    };
    let b = Kernel::new();
    let binds = Binds::build(cx)?;
    let resolve = |d| super::resolve_dim(cx, d).map(u64::from);
    let geom = ScatterGeometry::of(ops, axis as usize, resolve, Error::Legality)?;
    let (bins, inner, updates) = (geom.bins, geom.inner, geom.updates);
    let total = geom.total();

    let base = operand_src(&b, cx, &binds, base.src)?;
    let idx = operand_src(&b, cx, &binds, idx.src)?;
    let upd = operand_src(&b, cx, &binds, upd.src)?;
    let out_buf = binds.of(cx.launch.root)?;
    let elem = out_buf.element;
    let out = view(&out_buf);

    let (grid, offsets) = lane_offsets(&b, total, tm);
    let u_local = b.local(ScalarElement::U32.element());
    let u = b.load_local(u_local.clone());
    // The lowest offset is live whenever any of this lane's offsets is, so it
    // is the right mask for the one index read they share.
    let first_live = b.lt(global_lane(&b, DEFAULT_BLOCK), b.u32(total as u32));
    let u_bin = idx.at(&b, u.clone(), first_live);

    let mut accumulators = Vec::with_capacity(tm as usize);
    let mut stores = Vec::with_capacity(tm as usize);
    for flat in offsets {
        let live = b.lt(flat.clone(), b.u32(total as u32));
        // (outer, destination bin, inner) of this output element.
        let o = b.div(flat.clone(), b.u32(bins * inner));
        let dest = b.rem(b.div(flat.clone(), b.u32(inner)), b.u32(bins));
        let within = b.rem(flat.clone(), b.u32(inner));
        let acc_local = b.local(elem);
        let acc = b.load_local(acc_local.clone());
        // `upd[o, u, within]` in the update's own flat space.
        let upd_index = b.add(
            b.mul(b.add(b.mul(o, b.u32(updates)), u.clone()), b.u32(inner)),
            within,
        );
        let contribution = upd.at(&b, upd_index, live.clone());
        let combined = match combine {
            // `Add` duplicates accumulate — normative: an embedding table
            // receiving one token twice gets the summed gradient. `Set` is only
            // reachable when the node proved its indices unique.
            ScatterCombine::Add => b.add(acc.clone(), contribution),
            ScatterCombine::Set => contribution,
        };
        let update = TileExpr::new(
            TileExprKind::Select {
                condition: b.eq(u_bin.clone(), dest),
                accept: combined,
                reject: acc.clone(),
            },
            elem,
        );
        accumulators.push(Accumulator {
            local: acc_local,
            init: base.at(&b, flat.clone(), live.clone()),
            update,
        });
        stores.push(Stmt::Store {
            dst: out.clone(),
            addr: Addr::Linear(flat),
            value: acc,
            mask: live,
        });
    }

    let mut body = vec![Stmt::Loop {
        count: Some(b.u32(updates)),
        index: Some(u_local),
        accumulators,
        body: Vec::new(),
    }];
    body.extend(stores);

    Ok(KernelIr {
        buffers: binds.buffers,
        grid,
        block: DEFAULT_BLOCK,
        body,
        byte_arena: None,
        name: "cpu_scatter",
    })
}
