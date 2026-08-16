//! `Gather`'s two modes and `Scatter`'s two.
//!
//! Both nests read their lane tiling off `theta`. Currently the cost model does
//! not select tiled points, so `theta` is typically `SchedPoint::Point` and
//! bodies run one element per lane.

use fusor2_ir::Result;
use fusor2_ir::dtype::NumericContract;
use fusor2_ir::error::Error;
use fusor2_ir::ir::logical::ScatterCombine;
use fusor2_ir::ir::launch::{GatherMode, Launch, MapTiling, Operand, SchedPoint};
use fusor2_ir::ir::kernel::{
    Accumulator, Addr, ElementType, KernelIr, ScalarElement, Stmt, TileBinaryOp,
    TileCompareOp, TileExpr,
};

use crate::lower::{Ctx, distribute_workgroups};
use fusor2_tile::domains::emitted_block;


/// The register-reuse tiling this launch runs at.
///
/// [`SchedPoint::Point`] is the floor lowering's untiled point and resolves to
/// one element per lane. Any other family is a planner bug.
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
/// conditions make the map from lanes to elements a bijection:
///
/// * `extents[..=axis].product() >= tm`, or the tiled axis has fewer blocks
///   than the tile and the lanes stop short of the space.
/// * `stride % run == 0` when a lane also owns a `run`-wide contiguous group,
///   or the run's residues drift across the tile and elements are written
///   twice or not at all.
///
/// Failing either, the tile degrades to 1 rather than the plan failing.
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

/// How many lanes the tile needs to cover a space of `n` elements.
///
/// **Not `n / (tm * run)`.** The tiled axis is blocked, so the lane index has
/// to reach `ceil(blocks / tm)` whole tiles even when the last one is partly
/// masked: at `[13, 8]` with `tm = 2` the naive count is 52 lanes and element
/// 100 is then written by nobody.
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

/// `RowPerGroup` and `Vectorized`, each at the lane tiling `theta` selected.
///
/// The mode fixes the *contiguous* group one lane reads; `theta` fixes how
/// many such groups a lane owns and how far apart they sit. They are
/// independent.
pub(crate) fn lower_kgather(mut ctx: Ctx<'_>, op: &Launch, theta: SchedPoint) -> Result<KernelIr> {
    let Launch::Gather {
        space,
        axis,
        mode,
        ops,
        ..
    } = op
    else {
        return Err(Error::Plan("lower_kgather on a non-Gather node".into()));
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
    // `inner`: the elements one gathered coordinate spans; for the quantized
    // branch it is the row width.
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
    // `theta.dim` names an axis of this node's own `space`, which is the
    // output space every address below is decomposed against.
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
    // base steps by `run`.
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
        // into the source's dense logical space either way, and `load_operand`
        // runs the format's decode program there. Only gathered rows decode.
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

/// Both `ScatterMode`s lower through one nest: **one lane per output element,
/// a counted loop over the updates**, costing `O(out x updates)` index
/// comparisons.
///
/// The update-parallel forms (one lane per update, `atomicAdd` or a
/// workgroup-private histogram) are correct only when the output buffer
/// already holds the base; `derive_bindings` gives a `Scatter`'s value its own
/// buffer and nothing copies the base in, so this nest must read the base.
pub(crate) fn lower_kscatter(ctx: Ctx<'_>, op: &Launch, theta: SchedPoint) -> Result<Vec<KernelIr>> {
    let Launch::Scatter { .. } = op else {
        return Err(Error::Plan("lower_kscatter on a non-Scatter node".into()));
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

fn scatter_shape(ctx: &Ctx<'_>, op: &Launch) -> Result<ScatterShape> {
    let Launch::Scatter { axis, ops, .. } = op else {
        return Err(Error::Plan("scatter shape on a non-Scatter node".into()));
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
/// occupancy, which is what `verify_launch`'s associativity obligation asks for.
///
/// **`tm` is the number of destination bins one lane owns.** With `tm` bins in
/// one lane the `idx[u]` read is hash-consed to one expression serving `tm`
/// accumulators. The bins axis is the only one that can be tiled without
/// breaking store coalescing: consecutive lanes still write consecutive
/// `inner` positions.
///
/// `theta.dim` is not read here: `space` is minted two ways
/// (`rules::lower_floor` hands the output space, `fusor2_tile::rules::scatter`
/// hands the update space), so an axis index taken from it cannot be
/// identified with an axis of the destination this nest walks.
fn scatter_dense(mut ctx: Ctx<'_>, op: &Launch, tiling: MapTiling) -> Result<KernelIr> {
    let Launch::Scatter { combine, ops, .. } = op else {
        return Err(Error::Plan("scatter_dense on a non-Scatter node".into()));
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
        // `Add` duplicates accumulate: an embedding table receiving one token
        // twice gets the summed gradient. `Set` is only reachable when the
        // node proved its indices unique.
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
