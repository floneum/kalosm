//! `Gather` and `Scatter`: one lane per output element (`tm` with a grid-stride
//! tiling), a scatter looping over the updates, so no atomic is needed.

use fusor_ir::Result;
use fusor_ir::error::Error;
use fusor_ir::ir::kernel::{
    Accumulator, Addr, ElementType, KernelIr, ScalarElement, Stmt, TileExpr, TileExprKind,
};
use fusor_ir::ir::launch::ScatterGeometry;
use fusor_ir::ir::launch::{IndexSpace, Launch, Operand, SchedPoint};
use fusor_ir::ir::logical::ScatterCombine;
use fusor_ir::ir::{Node, Op};
use fusor_ir::target::LowerCtx;
use fusor_tile::build::Kernel;

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

/// Output elements per lane, read off `theta` (`Point` is 1). Tiling is a grid
/// stride, so `MapTiling::dim` and `vector` are ignored.
fn lane_tile(theta: SchedPoint) -> Result<u32> {
    match theta {
        SchedPoint::Map(t) => Ok(t.tm.max(1)),
        SchedPoint::Point => Ok(1),
        other => Err(Error::Legality(format!(
            "a gather or scatter needs SchedPoint::Map, got {other:?}"
        ))),
    }
}

/// The `tm` flat indices one lane owns, a grid apart, and that grid.
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
    // The source's extent on the gathered axis: the outer coordinate must step by
    // the source's stride, not the output's.
    let src_shape = const_extents(cx, src.layout.shape())?;
    let src_axis = *src_shape
        .get(axis)
        .ok_or_else(|| Error::Legality("gather axis is out of range for the source".into()))?;
    let src_stride = src_axis.max(1) * inner;

    let src = operand_src(&b, cx, &binds, src.src)?;
    let idx = operand_src(&b, cx, &binds, idx.src)?;
    let out = binds.of(cx.launch.root)?;
    // Whole rows of 4-byte elements out of plain buffers are block copies.
    if let (super::OperandSrc::Buffer(source), super::OperandSrc::Buffer(index)) = (&src, &idx)
        && out.element.byte_size() == 4
        && source.element == out.element
        && matches!(
            index.element,
            ElementType::Scalar(ScalarElement::U32 | ScalarElement::I32)
        )
    {
        let rows = crate::rows::GatherRows {
            out: out.binding as usize,
            src: source.binding as usize,
            idx: index.binding as usize,
            outer: extents[..axis]
                .iter()
                .map(|e| *e as usize)
                .product::<usize>()
                .max(1),
            count: extents[axis] as usize,
            inner: inner as usize,
            src_axis: src_axis.max(1) as usize,
        };
        return Ok(binds.finish(
            Box::leak(rows.name().into_boxed_str()),
            [1, 1, 1],
            1,
            Vec::new(),
        ));
    }
    let out = view(&out);
    let (grid, offsets) = lane_offsets(&b, n, tm);
    let mut body = Vec::with_capacity(tm as usize);
    for flat in offsets {
        let mask = b.lt(flat.clone(), b.u32(n as u32));
        // Split the flat output index into (outer, gathered, inner).
        let (outer, rest) = b.divrem(flat.clone(), b.u32(out_stride));
        let (g, within) = b.divrem(rest, b.u32(inner));
        // Replace the gathered coordinate; outer steps by the source's stride.
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

    Ok(binds.finish("cpu_gather", grid, DEFAULT_BLOCK, body))
}

/// `out = base` with `out[.., idx[u], ..] (combine)= upd[.., u, ..]`. Walks the
/// output (the plan never copies the base in), one lane per element over a
/// counted loop of updates: each element is its own sole writer, so the result
/// is bit-reproducible without atomics; `tm` accumulators share one `idx[u]` read.
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
    // The lowest offset is live whenever any is: the mask for the shared index read.
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
            // `Add` accumulates duplicates; `Set` only when indices are proven unique.
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

    Ok(binds.finish("cpu_scatter", grid, DEFAULT_BLOCK, body))
}
