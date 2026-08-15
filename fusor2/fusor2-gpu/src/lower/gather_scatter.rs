//! `KGather`'s two modes and `KScatter`'s two.
//!
//! Both nests read their lane tiling off `theta`. Currently the cost model does
//! not select tiled points, so `theta` is typically `SchedPoint::Point` and
//! bodies run one element per lane.

use fusor2_ir::Result;
use fusor2_ir::device::Caps;
use fusor2_ir::dtype::NumericContract;
use fusor2_ir::error::Error;
use fusor2_ir::ir::Node;
use fusor2_ir::ir::level0::ScatterCombine;
use fusor2_ir::ir::level1::{GatherMode, L1, MapTiling, Operand, SchedPoint};
use fusor2_ir::ir::level2::{
    Accumulator, Addr, ElementType, KernelIr, ScalarElement, Stmt, TileBinaryOp,
    TileCompareOp, TileExpr,
};
use fusor2_ir::target::LowerCtx;

use crate::lower::{Ctx, DimBinding, distribute_workgroups};
use fusor2_tile::domains::emitted_block;

/// Lowering entry point.
pub fn lower(caps: &Caps, node: &Node, theta: SchedPoint, cx: &LowerCtx<'_>) -> Result<KernelIr> {
    let fusor2_ir::ir::Op::L1(op) = &node.op else {
        return Err(Error::Plan("gather_scatter got a foreign node".into()));
    };
    let ctx = Ctx::new(caps, cx, DimBinding::new())?;
    match op {
        L1::KGather { .. } => lower_kgather(ctx, op, theta),
        L1::KScatter { .. } => {
            let mut ks = lower_kscatter(ctx, op, theta)?;
            if ks.len() == 1 {
                Ok(ks.remove(0))
            } else {
                Err(Error::Plan(
                    "sort-segment scatter lowers to two kernels; call lower_kscatter".into(),
                ))
            }
        }
        _ => Err(Error::Plan("gather_scatter got a foreign node".into())),
    }
}

// ---------------------------------------------------------------------------
// The lane tiling, read off theta
// ---------------------------------------------------------------------------

/// The register-reuse tiling this launch runs at.
///
/// [`SchedPoint::Point`] is the floor lowering's untiled point and resolves to
/// one element per lane. That is `rules::lower_floor`'s guarantee that every
/// chain reaches a valid L1 form when the saturation budget is exhausted, not
/// a missing case — so it is answered, not refused. Any other family is a
/// planner bug and says so.
fn tiling(theta: SchedPoint) -> Result<MapTiling> {
    match theta {
        SchedPoint::Map(t) => Ok(MapTiling {
            dim: t.dim,
            tm: t.tm.max(1),
            vector: t.vector.max(1),
        }),
        SchedPoint::Point => Ok(MapTiling {
            dim: None,
            tm: 1,
            vector: 1,
        }),
        other => Err(Error::Plan(format!(
            "a gather or scatter needs SchedPoint::Map, got {other:?}"
        ))),
    }
}

/// How far apart one lane's `tm` elements sit, and whether the tiling is
/// legal at all on this shape.
///
/// A lane owns `tm` elements one step of the tiled axis apart, which is
/// `stride = prod(extents[axis+1..])` elements in the flattened space. Two
/// conditions make the map from lanes to elements a bijection, and both are
/// decided here on constants rather than assumed:
///
/// * `extents[..=axis].product() >= tm`, or the tiled axis has fewer blocks
///   than the tile and the lanes stop short of the space (the `bins == 1`
///   scatter, where a naive tiling silently drops every element past
///   `total/tm`).
/// * `stride % run == 0` when a lane also owns a `run`-wide contiguous group,
///   or the run's residues drift across the tile and elements are written
///   twice or not at all.
///
/// Failing either, the tile degrades to 1 rather than the plan failing: the
/// schedule point stays selectable and simply buys nothing.
fn tile_stride(extents: &[u64], axis: usize, tm: u32, run: u32) -> Option<u64> {
    if tm <= 1 || axis + 1 >= extents.len() {
        return None;
    }
    let stride: u64 = extents[axis + 1..].iter().product::<u64>().max(1);
    let blocks: u64 = extents[..=axis].iter().product::<u64>().max(1);
    if blocks < u64::from(tm) || stride % u64::from(run.max(1)) != 0 {
        return None;
    }
    Some(stride)
}

/// The flat element offsets one lane owns, as integers — the arithmetic
/// [`lane_offset_exprs`] builds in `TileExpr`s, in a form a test can iterate.
///
/// `thread` is the global lane index; `stride` and `tm` come from
/// [`tile_stride`]; `run` is the contiguous group a vectorized mode reads in
/// one go. Read by the coverage test, which is the only place the scheme can
/// be checked as a whole: an expression tree cannot be iterated over threads.
#[cfg_attr(not(test), allow(dead_code))]
fn lane_offsets(thread: u64, stride: u64, tm: u32, run: u32) -> Vec<u64> {
    let run = u64::from(run.max(1));
    let base = thread * run;
    let tile_base = if tm > 1 {
        (base / stride) * stride * u64::from(tm) + base % stride
    } else {
        base
    };
    let mut out = Vec::with_capacity(tm as usize * run as usize);
    for t in 0..u64::from(tm.max(1)) {
        for r in 0..run {
            out.push(tile_base + t * stride + r);
        }
    }
    out
}

/// How many lanes the tile needs to cover a space of `n` elements.
///
/// **Not `n / (tm * run)`.** The tiled axis is blocked, so the lane index has
/// to reach `ceil(blocks / tm)` whole tiles even when the last one is partly
/// masked: at `[13, 8]` with `tm = 2` the naive count is 52 lanes and element
/// 100 is then written by nobody. Rounding up per block instead costs one
/// extra masked tile and covers the space.
fn lane_count(n: u64, stride: Option<u64>, tm: u32, run: u32) -> u64 {
    let run = u64::from(run.max(1));
    match stride {
        Some(s) if tm > 1 => {
            let blocks = n.div_ceil(s.max(1));
            blocks.div_ceil(u64::from(tm)) * s.div_ceil(run)
        }
        _ => n.div_ceil(run),
    }
    .max(1)
}

/// [`lane_offsets`] as `TileExpr`s. `thread` is `global_index`'s value.
fn lane_offset_exprs(
    ctx: &mut Ctx<'_>,
    thread: TileExpr,
    stride: Option<u64>,
    tm: u32,
    run: u32,
) -> Vec<TileExpr> {
    let base = if run > 1 {
        let run_e = ctx.b.u32(run);
        ctx.b.mul(thread, run_e)
    } else {
        thread
    };
    let (tile_base, stride_e, tm) = match stride {
        Some(s) if tm > 1 => {
            let stride_e = ctx.b.u32(u32::try_from(s).unwrap_or(u32::MAX));
            let outer = ctx.b.binary(
                TileBinaryOp::Div,
                base.clone(),
                stride_e.clone(),
                NumericContract::RELAXED,
            );
            let within = ctx.b.binary(
                TileBinaryOp::Rem,
                base,
                stride_e.clone(),
                NumericContract::RELAXED,
            );
            let step = {
                let tm_e = ctx.b.u32(tm);
                ctx.b.mul(stride_e.clone(), tm_e)
            };
            let scaled = ctx.b.mul(outer, step);
            (ctx.b.add(scaled, within), Some(stride_e), tm)
        }
        _ => (base, None, 1),
    };

    let mut out = Vec::with_capacity(tm as usize * run.max(1) as usize);
    for t in 0..tm {
        let row = match (&stride_e, t) {
            (_, 0) => tile_base.clone(),
            (Some(s), _) => {
                let t_e = ctx.b.u32(t);
                let off = ctx.b.mul(s.clone(), t_e);
                ctx.b.add(tile_base.clone(), off)
            }
            (None, _) => tile_base.clone(),
        };
        for r in 0..run.max(1) {
            if r == 0 {
                out.push(row.clone());
            } else {
                let r_e = ctx.b.u32(r);
                out.push(ctx.b.add(row.clone(), r_e));
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Gather
// ---------------------------------------------------------------------------

/// `RowPerGroup` and `Vectorized`, each at the lane tiling `theta` selected.
///
/// The mode fixes the *contiguous* group one lane reads (`Vectorized`'s four
/// f32 are one `dwordx4`); `theta` fixes how many such groups a lane owns and
/// how far apart they sit. They are independent, so a vectorized gather is
/// still register-tiled and a scalar one is not forced to be untiled.
pub fn lower_kgather(mut ctx: Ctx<'_>, op: &L1, theta: SchedPoint) -> Result<KernelIr> {
    let L1::KGather {
        space,
        axis,
        mode,
        ops,
        ..
    } = op
    else {
        return Err(Error::Plan("lower_kgather on a non-KGather node".into()));
    };
    let axis = *axis as usize;
    let [src, idx, ..] = ops.as_slice() else {
        return Err(Error::Plan(
            "a gather needs a source and an index operand".into(),
        ));
    };

    let out = ctx.output()?;
    let out_view = ctx.linear_view(out)?;
    let out_elem = out_view.buffer.element;
    let block = emitted_block(1, ctx.caps);
    let limits = ctx.caps.limits;

    // The output index space, and the one axis on which the source differs
    // from it. Addressing every gather as if `axis` were 0 is only right
    // when the gathered axis is outermost.
    let mut extents = Vec::with_capacity(space.dims.len());
    for d in &space.dims {
        extents.push(ctx.binding.require(*d)?);
    }
    if axis >= extents.len() {
        return Err(Error::Plan("gather axis is out of range".into()));
    }
    let n_u64: u64 = extents.iter().copied().product::<u64>().max(1);
    // `inner`: the elements one gathered coordinate spans. `width` keeps its
    // old name for the quantized branch, where it is the row width.
    let inner_u64: u64 = extents[axis + 1..].iter().copied().product::<u64>().max(1);
    let width = u32::try_from(inner_u64)
        .map_err(|_| Error::Plan("gather row width exceeds a u32".into()))?
        .max(1);
    let rows = u32::try_from(extents[axis])
        .map_err(|_| Error::Plan("gather row count exceeds a u32".into()))?;
    let out_stride = rows.max(1) * width;
    // The source's extent on the gathered axis. It is the output's only when
    // the index vector is exactly as long as the axis it indexes.
    let src_shape = src.layout.shape();
    let src_axis_dim = src_shape
        .get(axis)
        .ok_or_else(|| Error::Plan("gather axis is out of range for the source".into()))?;
    let src_axis = u32::try_from(ctx.binding.require(*src_axis_dim)?)
        .map_err(|_| Error::Plan("gather source extent exceeds a u32".into()))?;
    let src_stride = src_axis.max(1) * width;

    let run = 1;
    // ... and `theta` says how many such groups one lane owns, one step of
    // the tiled axis apart. `dim` names an axis of this node's own `space`,
    // which is the output space every address below is decomposed against.
    let tiling = tiling(theta)?;
    let stride = tiling
        .dim
        .and_then(|d| tile_stride(&extents, d as usize, tiling.tm, run));
    let tm = if stride.is_some() { tiling.tm } else { 1 };

    // The dispatch grid up front: `global_index` linearizes against it.
    let lanes = lane_count(n_u64, stride, tm, run);
    let grid = distribute_workgroups(
        u32::try_from(lanes.div_ceil(u64::from(block)).max(1)).unwrap_or(u32::MAX),
        limits.max_compute_workgroups_per_dimension,
    );

    let thread = ctx.global_index(block, grid);
    // A vectorized lane owns `run` *consecutive* output elements, so the flat
    // base steps by `run`. Adding the lane to the column instead made four
    // threads write the same four elements.
    let offsets = lane_offset_exprs(&mut ctx, thread, stride, tm, run);

    let width_e = ctx.b.u32(width);
    let out_stride_e = ctx.b.u32(out_stride);
    let src_stride_e = ctx.b.u32(src_stride);
    let n_e = ctx.b.u32(u32::try_from(n_u64).unwrap_or(u32::MAX));

    // Split the flat output index into (outer, gathered, within).
    let decompose = |ctx: &mut Ctx<'_>, flat: TileExpr| {
        let outer = ctx.b.binary(
            TileBinaryOp::Div,
            flat.clone(),
            out_stride_e.clone(),
            NumericContract::RELAXED,
        );
        let rest = ctx.b.binary(
            TileBinaryOp::Rem,
            flat.clone(),
            out_stride_e.clone(),
            NumericContract::RELAXED,
        );
        let g = ctx.b.binary(
            TileBinaryOp::Div,
            rest.clone(),
            width_e.clone(),
            NumericContract::RELAXED,
        );
        let within = ctx.b.binary(
            TileBinaryOp::Rem,
            rest,
            width_e.clone(),
            NumericContract::RELAXED,
        );
        (outer, g, within)
    };

    let mut body = Vec::new();
    match mode {
        // `QuantizedRows` shares the scalar nest: `src_addr` is a flat index
        // into the source's dense logical space either way, and
        // `load_operand` runs the format's decode program there because the
        // operand *is* the quantized leaf. Only gathered rows ever decode.
        GatherMode::RowPerGroup | GatherMode::QuantizedRows => {
            for flat in &offsets {
                let flat = flat.clone();
                let (outer, g, within) = decompose(&mut ctx, flat.clone());
                let picked = ctx.load_operand(idx, g)?;
                let picked = ctx.b.cast(picked, ElementType::Scalar(ScalarElement::U32));
                let mask = ctx
                    .b
                    .compare(TileCompareOp::Lt, flat.clone(), n_e.clone());
                let src_addr = {
                    let outer_off = ctx.b.mul(outer, src_stride_e.clone());
                    let row_off = ctx.b.mul(picked, width_e.clone());
                    let base = ctx.b.add(outer_off, row_off);
                    ctx.b.add(base, within)
                };
                let value = ctx.load_operand(src, src_addr)?;
                let value = ctx.b.cast(value, out_elem);
                body.push(Stmt::Store {
                    dst: out_view.clone(),
                    addr: Addr::Linear(flat),
                    value,
                    mask,
                });
            }
        }
    }

    Ok(ctx.finish("kgather", grid, block, body))
}

// ---------------------------------------------------------------------------
// Scatter
// ---------------------------------------------------------------------------

/// Both `ScatterMode`s name one map and differ only in *strategy*, and on
/// this runtime they lower through one nest: **one lane per output
/// element, a counted loop over the updates**.
///
/// The update-parallel forms (one lane per update, `atomicAdd` or a
/// workgroup-private histogram) are correct only when the output buffer
/// already holds the base. It does not: `derive_bindings` gives a `KScatter`'s
/// value its own buffer and nothing copies the base in, so an update-parallel
/// kernel left every unwritten element undefined — which is why `cat`,
/// `stack`, `pad`, `repeat` and `slice_assign` read back as zeros. Reinstating
/// them is a **planner** change: an `Effect::InPlace(BufferRole(0))` scatter
/// has to be given its base operand's buffer, and `fusor2-cost` does not do
/// that today. Until it does, one nest that reads the base is the only correct
/// answer, and it costs `O(out x updates)` index comparisons.
pub fn lower_kscatter(ctx: Ctx<'_>, op: &L1, theta: SchedPoint) -> Result<Vec<KernelIr>> {
    let L1::KScatter { .. } = op else {
        return Err(Error::Plan("lower_kscatter on a non-KScatter node".into()));
    };
    scatter_dense(ctx, op, tiling(theta)?).map(|k| vec![k])
}

/// A scatter's destination geometry, read off the **base operand** rather than
/// off `space`.
///
/// `space` is minted two ways — `rules::lower_floor` hands the output space,
/// `fusor2_tile::rules::scatter` hands the update space — so neither the bin
/// count nor the update count can be read off it. The base operand's layout
/// gives the destination shape and the index operand's gives the update count,
/// under either convention.
struct ScatterShape {
    /// Product of the base extents before the scattered axis.
    outer: u32,
    /// Extent of the scattered axis in the base — the destination bins.
    bins: u32,
    /// Product of the base extents after the scattered axis.
    inner: u32,
    /// Index count.
    updates: u32,
}

fn scatter_shape(ctx: &Ctx<'_>, op: &L1) -> Result<ScatterShape> {
    let L1::KScatter { axis, ops, .. } = op else {
        return Err(Error::Plan("scatter shape on a non-KScatter node".into()));
    };
    let axis = *axis as usize;
    let base = ops
        .first()
        .ok_or_else(|| Error::Plan("a scatter needs a base operand".into()))?;
    let mut dest: Vec<u32> = Vec::with_capacity(base.layout.rank());
    for d in base.layout.shape() {
        dest.push(
            u32::try_from(ctx.binding.require(*d)?)
                .map_err(|_| Error::Plan("scatter extent exceeds a u32".into()))?,
        );
    }
    if axis >= dest.len() {
        return Err(Error::Plan(format!(
            "scatter axis {axis} is outside a rank-{} base",
            dest.len()
        )));
    }
    let idx = ops
        .get(1)
        .ok_or_else(|| Error::Plan("a scatter needs an index operand".into()))?;
    let mut updates = 1u64;
    for d in idx.layout.shape() {
        updates = updates.saturating_mul(ctx.binding.require(*d)?);
    }
    Ok(ScatterShape {
        outer: dest[..axis].iter().product::<u32>().max(1),
        bins: dest[axis].max(1),
        inner: dest[axis + 1..].iter().product::<u32>().max(1),
        updates: u32::try_from(updates)
            .map_err(|_| Error::Plan("scatter update count exceeds a u32".into()))?
            .max(1),
    })
}

/// `out = base` with `out[.., idx[u], ..] (combine)= upd[.., u, ..]`, at the
/// lane tiling `theta` selected.
///
/// Every output element is written by exactly one lane, so no atomic is needed
/// and the accumulation order is fixed: the result is bit-reproducible at any
/// occupancy, which is what `verify_l1`'s associativity obligation asks for.
///
/// **`tm` is the number of destination bins one lane owns**, and it is what
/// makes this nest affordable on the embedding gradient. The loop over the
/// updates costs one `idx[u]` read per output element per update; with `tm`
/// bins in one lane the read is hash-consed to *one* expression serving `tm`
/// accumulators, so the index traffic — the dominant term at
/// `bins * inner >> updates` — falls by `tm`. The bins axis is also the only
/// one that can be tiled without breaking store coalescing: consecutive lanes
/// still write consecutive `inner` positions.
///
/// `theta.dim` is deliberately **not** read here. `space` is minted two ways
/// (`rules::lower_floor` hands the output space, `fusor2_tile::rules::scatter`
/// hands the update space), so an axis index taken from it cannot be
/// identified with an axis of the destination this nest walks. `tm` needs no
/// such identification.
fn scatter_dense(mut ctx: Ctx<'_>, op: &L1, tiling: MapTiling) -> Result<KernelIr> {
    let L1::KScatter { combine, ops, .. } = op else {
        return Err(Error::Plan("scatter_dense on a non-KScatter node".into()));
    };
    let shape = scatter_shape(&ctx, op)?;
    let base = ops
        .first()
        .ok_or_else(|| Error::Plan("a scatter needs a base operand".into()))?
        .clone();
    let (idx, upd) = index_and_update(ops)?;
    let (idx, upd) = (idx.clone(), upd.clone());
    let out = ctx.output()?;
    let out_view = ctx.linear_view(out)?;
    let out_elem = out_view.buffer.element;
    let acc_elem = match out_elem {
        ElementType::Scalar(s) => s,
        _ => ScalarElement::F32,
    };
    let block = emitted_block(1, ctx.caps);
    let limits = ctx.caps.limits;

    let total = (shape.outer as u64)
        .saturating_mul(shape.bins as u64)
        .saturating_mul(shape.inner as u64)
        .max(1);

    // The destination nest, as extents: [outer, bins, inner]. The tile runs
    // along the bins axis, so its stride is `inner`.
    let dest_extents = [
        u64::from(shape.outer),
        u64::from(shape.bins),
        u64::from(shape.inner),
    ];
    let stride = tile_stride(&dest_extents, 1, tiling.tm, 1);
    let tm = if stride.is_some() { tiling.tm } else { 1 };

    let lanes = lane_count(total, stride, tm, 1);
    let grid = distribute_workgroups(
        u32::try_from(lanes.div_ceil(u64::from(block)).max(1)).unwrap_or(u32::MAX),
        limits.max_compute_workgroups_per_dimension,
    );

    let thread = ctx.global_index(block, grid);
    let offsets = lane_offset_exprs(&mut ctx, thread, stride, tm, 1);
    let bound = ctx.b.u32(u32::try_from(total).unwrap_or(u32::MAX));

    let inner_e = ctx.b.u32(shape.inner);
    let bins_e = ctx.b.u32(shape.bins);
    let row_span = ctx.b.u32(shape.bins.saturating_mul(shape.inner).max(1));
    let updates_e = ctx.b.u32(shape.updates);

    // One index read per update, shared by every accumulator this lane
    // carries: `u_bin` does not depend on the output element, so the `tm`
    // slots hash-cons onto one load.
    let u_local = ctx.b.local(ElementType::Scalar(ScalarElement::U32));
    let u = ctx.b.load_local(u_local.clone());
    let u_bin = ctx.load_operand(&idx, u.clone())?;
    let u_bin = ctx.b.cast(u_bin, ElementType::Scalar(ScalarElement::U32));

    let mut accumulators = Vec::with_capacity(offsets.len());
    let mut stores = Vec::with_capacity(offsets.len());
    for flat in &offsets {
        let flat = flat.clone();
        let live = ctx.b.compare(TileCompareOp::Lt, flat.clone(), bound.clone());

        // (outer, destination bin, inner) of this output element.
        let o = ctx.b.binary(
            TileBinaryOp::Div,
            flat.clone(),
            row_span.clone(),
            NumericContract::RELAXED,
        );
        let dest = {
            let q = ctx.b.binary(
                TileBinaryOp::Div,
                flat.clone(),
                inner_e.clone(),
                NumericContract::RELAXED,
            );
            ctx.b
                .binary(TileBinaryOp::Rem, q, bins_e.clone(), NumericContract::RELAXED)
        };
        let within = ctx.b.binary(
            TileBinaryOp::Rem,
            flat.clone(),
            inner_e.clone(),
            NumericContract::RELAXED,
        );

        // The accumulator starts at the base, so every element the updates
        // never touch still lands.
        let seed = ctx.load_mapped(&base, flat.clone(), total)?;
        let init = ctx.b.cast(seed, ElementType::Scalar(acc_elem));
        let acc_local = ctx.b.local(ElementType::Scalar(acc_elem));
        let acc_read = ctx.b.load_local(acc_local.clone());

        let hit = ctx.b.compare(TileCompareOp::Eq, u_bin.clone(), dest);
        let upd_index = {
            let row = ctx.b.mul(o, updates_e.clone());
            let row = ctx.b.add(row, u.clone());
            let scaled = ctx.b.mul(row, inner_e.clone());
            ctx.b.add(scaled, within)
        };
        let v = ctx.load_operand(&upd, upd_index)?;
        let v = ctx.b.cast(v, ElementType::Scalar(acc_elem));
        // `Add` duplicates accumulate — normative: an embedding table receiving
        // one token twice gets the summed gradient. `Set` is only reachable when
        // the node proved its indices unique.
        let combined = match combine {
            ScatterCombine::Add => ctx.b.add(acc_read.clone(), v),
            ScatterCombine::Set => v,
        };
        let update = ctx.b.select(hit, combined, acc_read);
        accumulators.push(Accumulator {
            local: acc_local.clone(),
            init,
            update,
        });

        let value = ctx.b.load_local(acc_local);
        let value = ctx.b.cast(value, out_elem);
        stores.push(Stmt::Store {
            dst: out_view.clone(),
            addr: Addr::Linear(flat),
            value,
            mask: live,
        });
    }

    let count = ctx.b.u32(shape.updates);
    let mut body = vec![Stmt::Loop {
        count: Some(count),
        index: Some(u_local),
        accumulators,
        body: Vec::new(),
    }];
    body.extend(stores);

    Ok(ctx.finish("scatter_dense", grid, block, body))
}

/// A scatter's operands are `(base, idx, upd)`; the base is bound as the
/// output, so lowering reads slots 1 and 2.
fn index_and_update(ops: &[Operand]) -> Result<(&Operand, &Operand)> {
    match ops {
        [_base, idx, upd, ..] => Ok((idx, upd)),
        [idx, upd] => Ok((idx, upd)),
        _ => Err(Error::Plan(format!(
            "a scatter needs base/index/update operands, got {}",
            ops.len()
        ))),
    }
}

/// A host reference scatter-add, used by the conformance case that pits the
/// three `Add` lowerings against each other. Duplicate indices accumulate.
pub fn reference_scatter_add(
    bins: usize,
    elem: usize,
    indices: &[u32],
    updates: &[f32],
) -> Vec<f32> {
    let mut out = vec![0.0f32; bins * elem];
    for (row, &bin) in indices.iter().enumerate() {
        let bin = bin as usize;
        if bin >= bins {
            continue;
        }
        for c in 0..elem {
            out[bin * elem + c] += updates[row * elem + c];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use fusor2_ir::ir::level1::ScatterMode;
    use fusor2_ir::cost::Picoseconds;
    use fusor2_ir::dtype::Dtype;
    use fusor2_ir::egraph::{EGraph, Id};
    use fusor2_ir::extract::{BindKind, BindingPlan, Extraction, Launch, Plan, PlanHash};
    use fusor2_ir::ir::Op;
    use fusor2_ir::ir::level0::{BufferId, L0, LeafKind};
    use fusor2_ir::ir::level1::{
        AccessPlan, FoldStrat, IndexSpace, MapTiling, ScheduleDomain,
    };
    use fusor2_ir::ir::level2::Stmt;
    use fusor2_ir::semantics::{CoreSemantics, SumArenaPlanner};
    use fusor2_ir::shape::{Dim, Layout};
    use smallvec::SmallVec;
    use std::sync::Arc;

    // -----------------------------------------------------------------------
    // The address scheme
    // -----------------------------------------------------------------------

    /// Every element of the space, written by exactly one lane, at every
    /// tiling the domain can offer. This is the property the whole change
    /// rests on: a tiling that covers the space twice or not at all is a
    /// wrong answer that no shape test would notice.
    fn covers_exactly_once(extents: &[u64], axis: usize, tm: u32, run: u32) {
        let n: u64 = extents.iter().product();
        let Some(stride) = tile_stride(extents, axis, tm, run) else {
            return;
        };
        let lanes = lane_count(n, Some(stride), tm, run);
        let mut seen = vec![0u32; n as usize];
        for thread in 0..lanes {
            for off in lane_offsets(thread, stride, tm, run) {
                if off < n {
                    seen[off as usize] += 1;
                }
            }
        }
        for (i, c) in seen.iter().enumerate() {
            assert_eq!(
                *c, 1,
                "element {i} of {extents:?} written {c} times at tm={tm} run={run}"
            );
        }
    }

    #[test]
    fn a_lane_tile_covers_the_index_space_exactly_once() {
        for tm in [2u32, 4, 8] {
            for run in [1u32, 4] {
                covers_exactly_once(&[64, 8], 0, tm, run);
                covers_exactly_once(&[4, 32, 8], 1, tm, run);
                covers_exactly_once(&[4, 32, 8], 0, tm, run);
                // extents that do not divide the tile: the tail is masked,
                // never doubled.
                covers_exactly_once(&[13, 8], 0, tm, run);
                covers_exactly_once(&[7, 5, 12], 1, tm, run);
            }
        }
    }

    /// The two conditions that make the tile a bijection are checked, not
    /// assumed. Both decline to 1 rather than failing the plan.
    #[test]
    fn a_tile_that_would_drop_elements_declines() {
        // fewer blocks than the tile: 3 rows cannot host a 4-wide tile
        assert_eq!(tile_stride(&[3, 8], 0, 4, 1), None);
        // one bin: the scatter shape where a naive tile drops everything past
        // total/tm
        assert_eq!(tile_stride(&[1, 1, 768], 1, 4, 1), None);
        // a 4-wide contiguous run inside a stride that is not a multiple of 4
        assert_eq!(tile_stride(&[64, 6], 0, 4, 4), None);
        assert_eq!(tile_stride(&[64, 8], 0, 4, 4), Some(8));
        // the innermost axis is never tiled, and tm = 1 is untiled
        assert_eq!(tile_stride(&[64, 8], 1, 4, 1), None);
        assert_eq!(tile_stride(&[64, 8], 0, 1, 1), None);
    }

    // -----------------------------------------------------------------------
    // The lowering reads it
    // -----------------------------------------------------------------------

    fn dims(v: &[u64]) -> SmallVec<[Dim; 6]> {
        v.iter().map(|d| Dim::Const(*d)).collect()
    }

    fn leaf(g: &mut EGraph, id: u32, dtype: Dtype, shape: &[u64]) -> Id {
        g.add(Op::L0(L0::Leaf(LeafKind::Buffer {
            name: BufferId(id),
            dtype,
            shape: dims(shape),
        })))
        .expect("leaf")
    }

    fn operand(g: &EGraph, src: Id) -> Operand {
        Operand {
            src,
            layout: Layout::contiguous(&g.facts(src).shape),
            access: AccessPlan::Alias,
        }
    }

    fn plan_for(root: Id, reads: &[Id]) -> Plan {
        let mut bindings: Vec<BindingPlan> = reads
            .iter()
            .enumerate()
            .map(|(i, v)| BindingPlan {
                binding: i as u32 + 1,
                value: *v,
                kind: BindKind::Read,
            })
            .collect();
        bindings.push(BindingPlan {
            binding: reads.len() as u32 + 1,
            value: root,
            kind: BindKind::Write,
        });
        Plan {
            extraction: Extraction::default(),
            launches: vec![Launch {
                root,
                members: smallvec::smallvec![root],
                bindings,
                grid: [1, 1, 1],
                // A placeholder: `lower` recomputes the width off `caps`, so
                // the plan's value is never what the kernel is built with.
                block: 256,
            }],
            buffers: Vec::new(),
            symbols: Vec::new(),
            hash: PlanHash(0),
            cost: Picoseconds(0),
        }
    }

    fn graph() -> EGraph {
        EGraph::new(CoreSemantics::new(Arc::new(SumArenaPlanner)))
    }

    fn scatter_ir(theta: SchedPoint, bins: u64, inner: u64, updates: u64) -> Result<KernelIr> {
        let mut g = graph();
        let base = leaf(&mut g, 0, Dtype::F32, &[bins, inner]);
        let idx = leaf(&mut g, 1, Dtype::U32, &[updates]);
        let upd = leaf(&mut g, 2, Dtype::F32, &[updates, inner]);
        let k = g
            .add(Op::L1(L1::KScatter {
                space: IndexSpace::new(dims(&[updates, inner]).into_iter()),
                axis: 0,
                mode: ScatterMode::SortSegment,
                combine: ScatterCombine::Add,
                ops: vec![operand(&g, base), operand(&g, idx), operand(&g, upd)],
                sched: ScheduleDomain::Point,
            }))
            .expect("kscatter");
        let plan = plan_for(k, &[base, idx, upd]);
        let cx = LowerCtx {
            plan: &plan,
            launch: &plan.launches[0],
            graph: &g,
            symbols: &[],
        };
        lower(&crate::emit::testkit::caps(false, true), g.node(k), theta, &cx)
    }

    fn gather_ir(
        theta: SchedPoint,
        mode: GatherMode,
        rows: u64,
        src_rows: u64,
        width: u64,
    ) -> Result<KernelIr> {
        let mut g = graph();
        let src = leaf(&mut g, 0, Dtype::F32, &[src_rows, width]);
        let idx = leaf(&mut g, 1, Dtype::U32, &[rows]);
        let k = g
            .add(Op::L1(L1::KGather {
                space: IndexSpace::new(dims(&[rows, width]).into_iter()),
                axis: 0,
                mode,
                ops: vec![operand(&g, src), operand(&g, idx)],
                sched: ScheduleDomain::Point,
            }))
            .expect("kgather");
        let plan = plan_for(k, &[src, idx]);
        let cx = LowerCtx {
            plan: &plan,
            launch: &plan.launches[0],
            graph: &g,
            symbols: &[],
        };
        lower(&crate::emit::testkit::caps(false, true), g.node(k), theta, &cx)
    }

    fn map_point(dim: Option<u32>, tm: u32, vector: u32) -> SchedPoint {
        SchedPoint::Map(MapTiling { dim, tm, vector })
    }

    fn accumulators(ir: &KernelIr) -> usize {
        match &ir.body[0] {
            Stmt::Loop { accumulators, .. } => accumulators.len(),
            other => panic!("expected the update loop, got {other:?}"),
        }
    }

    /// `theta` is read: `tm` bins per lane is `tm` accumulators in the one
    /// update loop, `tm` stores, and `tm` times fewer lanes launched. Every
    /// one of those was a constant before.
    #[test]
    fn the_scatter_reads_its_lane_tile_off_theta() {
        let untiled = scatter_ir(map_point(None, 1, 1), 1024, 8, 16).expect("lowers");
        let tiled = scatter_ir(map_point(Some(0), 4, 1), 1024, 8, 16).expect("lowers");
        assert_eq!(accumulators(&untiled), 1);
        assert_eq!(accumulators(&tiled), 4);
        assert_eq!(untiled.body.len(), 2);
        assert_eq!(tiled.body.len(), 5);
        assert_eq!(untiled.grid[0], tiled.grid[0] * 4);
        // ... and the floor's own point is still the untiled body, unchanged.
        let floor = scatter_ir(SchedPoint::Point, 1024, 8, 16).expect("lowers");
        assert_eq!(floor.grid, untiled.grid);
        assert_eq!(floor.body.len(), untiled.body.len());
    }

    #[test]
    fn the_gather_reads_its_lane_tile_off_theta() {
        let untiled = gather_ir(map_point(None, 1, 1), GatherMode::RowPerGroup, 512, 128, 8)
            .expect("lowers");
        let tiled =
            gather_ir(map_point(Some(0), 4, 1), GatherMode::RowPerGroup, 512, 128, 8)
                .expect("lowers");
        assert_eq!(untiled.body.len(), 1);
        assert_eq!(tiled.body.len(), 4);
        assert_eq!(untiled.grid[0], tiled.grid[0] * 4);
    }

    /// A tile the shape cannot host degrades to the untiled body instead of
    /// failing the plan: the point stays selectable and simply buys nothing.
    /// A `bins == 1` scatter is the case that matters — a naive tiling drops
    /// every element past `total / tm`.
    #[test]
    fn an_unhostable_tile_degrades_instead_of_failing() {
        let one_bin = scatter_ir(map_point(Some(0), 8, 1), 1, 768, 4).expect("lowers");
        assert_eq!(accumulators(&one_bin), 1);
        let short = gather_ir(map_point(Some(0), 8, 1), GatherMode::RowPerGroup, 3, 8, 8)
            .expect("lowers");
        assert_eq!(short.body.len(), 1);
    }

    /// The domain these nodes carry is non-trivial on the trainer's embedding
    /// gradient — 1,024 bins x 768 units, 384 updates — and every point of it
    /// lowers. A selectable point that cannot be lowered is worse than no
    /// domain: it makes extraction *prefer* a plan that then fails.
    #[test]
    fn the_embedding_gradient_scatter_carries_a_real_domain_and_all_of_it_lowers() {
        let caps = crate::emit::testkit::caps(false, true);
        let cx = fusor2_tile::domains::DomainCtx::new(&caps, fusor2_tile::Planner::global());
        let dom = fusor2_tile::domains::map_domain(&dims(&[384, 768]), &[], &cx);
        assert!(
            dom.tilings.len() > 1,
            "a one-point domain is a decision already made: {:?}",
            dom.tilings
        );
        assert!(dom.tilings.iter().any(|t| t.tm > 1));
        for t in &dom.tilings {
            scatter_ir(SchedPoint::Map(*t), 1024, 24, 8).expect("every point lowers");
            gather_ir(SchedPoint::Map(*t), GatherMode::RowPerGroup, 384, 1024, 24)
                .expect("every point lowers");
        }
    }

    /// A point from another family is a planner bug, not a shrug.
    #[test]
    fn a_foreign_schedule_family_is_refused() {
        assert!(tiling(SchedPoint::Point).is_ok());
        assert!(tiling(map_point(Some(1), 8, 1)).is_ok());
        assert!(tiling(SchedPoint::Fold(FoldStrat::Subgroup)).is_err());
    }

    /// The semantic half of the `wg_private_merge_scatter_matches_reference`
    /// case, checkable without an adapter: duplicate indices accumulate, and a
    /// padding row naming an out-of-range bin contributes nothing.
    #[test]
    fn reference_scatter_add_accumulates_duplicates() {
        let indices = [3u32, 3, 0, 9999];
        let updates = [1.0f32, 2.0, 10.0, 20.0, 100.0, 200.0, 5.0, 5.0];
        let out = reference_scatter_add(4, 2, &indices, &updates);
        assert_eq!(out[6], 11.0);
        assert_eq!(out[7], 22.0);
        assert_eq!(out[0], 100.0);
        assert_eq!(out[1], 200.0);
        assert_eq!(&out[2..6], &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn a_pure_padding_batch_scatters_nothing() {
        let out = reference_scatter_add(4, 2, &[9999, 9999], &[1.0, 2.0, 3.0, 4.0]);
        assert!(out.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn skewed_bins_still_sum_exactly() {
        // 7% of the indices forced into one bin, the trainer's worst case.
        let indices: Vec<u32> = (0..1024)
            .map(|i| if i % 14 == 0 { 7 } else { i as u32 })
            .collect();
        let updates: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let out = reference_scatter_add(1024, 1, &indices, &updates);
        let expected: f32 = indices
            .iter()
            .zip(&updates)
            .filter(|(b, _)| **b == 7)
            .map(|(_, v)| *v)
            .sum();
        assert_eq!(out[7], expected);
        assert!((out.iter().sum::<f32>() - updates.iter().sum::<f32>()).abs() < 1e-1);
    }
}
