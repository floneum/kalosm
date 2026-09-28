//! `Gather`'s two modes and `Scatter`'s two, both at the lane tiling `theta`
//! selected.

use fusor_ir::Result;
use fusor_ir::error::Error;
use fusor_ir::ir::kernel::{
    Accumulator, Addr, Builtin, ElementType, KernelIr, ScalarElement, Stmt, TileExpr,
};
use fusor_ir::ir::launch::{Launch, MapTiling, ScatterGeometry, SchedPoint};
use fusor_ir::ir::logical::ScatterCombine;

use crate::lower::{Ctx, distribute_workgroups};
use fusor_tile::domains::emitted_block;

/// The register-reuse tiling this launch runs at; [`SchedPoint::Point`] is one
/// element per lane.
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

/// How far apart one lane's `tm` elements sit (`prod(extents[axis+1..])`), or
/// `None` when the tiled axis has fewer blocks than `tm` and the tile degrades
/// to 1.
fn tile_stride(extents: &[u64], axis: usize, tm: u32) -> Option<u64> {
    if tm <= 1 || axis + 1 >= extents.len() {
        return None;
    }
    let stride: u64 = extents[axis + 1..].iter().product::<u64>().max(1);
    let blocks: u64 = extents[..=axis].iter().product::<u64>().max(1);
    (blocks >= u64::from(tm)).then_some(stride)
}

/// The lanes a tile needs over `n` elements and the offsets each owns. The
/// lane count covers `ceil(blocks / tm)` whole tiles: a naive `n / tm` leaves
/// trailing elements unwritten.
struct LaneTile {
    lanes: u64,
    stride: Option<u64>,
    tm: u32,
}

impl LaneTile {
    fn new(n: u64, stride: Option<u64>, tm: u32) -> Self {
        let (stride, tm) = match stride {
            Some(s) if tm > 1 => (Some(s), tm),
            _ => (None, 1),
        };
        let lanes = match stride {
            Some(s) => n.div_ceil(s.max(1)).div_ceil(u64::from(tm)) * s,
            None => n,
        };
        Self {
            lanes: lanes.max(1),
            stride,
            tm,
        }
    }

    fn grid(&self, ctx: &Ctx<'_>, block: u32) -> [u32; 3] {
        distribute_workgroups(
            u32::try_from(self.lanes.div_ceil(u64::from(block)).max(1)).unwrap_or(u32::MAX),
            ctx.caps.limits.max_compute_workgroups_per_dimension,
        )
    }

    /// The offsets lane `thread` owns.
    fn offsets(&self, ctx: &Ctx<'_>, thread: TileExpr) -> Vec<TileExpr> {
        let b = &ctx.b;
        let Some(s) = self.stride else {
            return vec![thread];
        };
        let stride = b.u32(u32::try_from(s).unwrap_or(u32::MAX));
        let base = b.tile_origin(thread, stride.clone(), self.tm);
        (0..self.tm)
            .map(|t| match t {
                0 => base.clone(),
                _ => b.at(base.clone(), stride.clone(), b.u32(t)),
            })
            .collect()
    }
}

/// `RowPerGroup` and `QuantizedRows` at the lane tiling `theta` selected.
/// `QuantizedRows` shares the scalar nest; `load_operand` decodes only the
/// gathered rows.
pub(crate) fn lower_kgather(ctx: Ctx<'_>, op: &Launch, theta: SchedPoint) -> Result<KernelIr> {
    let Launch::Gather {
        space, axis, ops, ..
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
    let b = &ctx.b;
    let out = ctx.linear_view(ctx.output()?)?;
    let block = ctx.block(emitted_block(1, ctx.caps));

    // The output index space and the one axis on which the source differs from
    // it.
    let extents = space
        .dims
        .iter()
        .map(|d| ctx.binding.require(*d))
        .collect::<Result<Vec<u64>>>()?;
    if axis >= extents.len() {
        return Err(Error::Plan("gather axis is out of range".into()));
    }
    let n: u64 = extents.iter().copied().product::<u64>().max(1);
    // The elements one gathered coordinate spans.
    let width = u32::try_from(extents[axis + 1..].iter().product::<u64>().max(1))
        .map_err(|_| Error::Plan("gather row width exceeds a u32".into()))?
        .max(1);
    let rows = u32::try_from(extents[axis])
        .map_err(|_| Error::Plan("gather row count exceeds a u32".into()))?;
    // The source's extent on the gathered axis. It is the output's only when
    // the index vector is exactly as long as the axis it indexes.
    let src_axis_dim = src
        .layout
        .shape()
        .get(axis)
        .ok_or_else(|| Error::Plan("gather axis is out of range for the source".into()))?;
    let src_axis = u32::try_from(ctx.binding.require(*src_axis_dim)?)
        .map_err(|_| Error::Plan("gather source extent exceeds a u32".into()))?;

    // `theta.dim` names an axis of this node's own `space`, which is the
    // output space every address below is decomposed against.
    let tiling = tiling(theta)?;
    let stride = tiling
        .dim
        .and_then(|d| tile_stride(&extents, d as usize, tiling.tm));
    let tile = LaneTile::new(n, stride, tiling.tm);
    let grid = tile.grid(&ctx, block);
    let offsets = tile.offsets(&ctx, ctx.global_index(block));

    let width_e = b.u32(width);
    let out_stride = b.u32(rows.max(1) * width);
    let src_stride = b.u32(src_axis.max(1) * width);
    let n_e = b.u32(u32::try_from(n).unwrap_or(u32::MAX));
    let mut body = Vec::new();
    for flat in offsets {
        // Split the flat output index into (outer, gathered, within).
        let (outer, rest) = b.divrem(flat.clone(), out_stride.clone());
        let (g, within) = b.divrem(rest, width_e.clone());
        let picked = b.cast(ctx.load_operand(idx, g)?, ScalarElement::U32.element());
        let src_addr = b.add(
            b.add(
                b.mul(outer, src_stride.clone()),
                b.mul(picked, width_e.clone()),
            ),
            within,
        );
        let value = b.cast(ctx.load_operand(src, src_addr)?, out.buffer.element);
        body.push(Stmt::Store {
            dst: out.clone(),
            addr: Addr::Linear(flat.clone()),
            value,
            mask: b.lt(flat, n_e.clone()),
        });
    }
    Ok(ctx.finish("kgather", grid, block, body))
}

fn total_of(shape: &ScatterGeometry) -> u64 {
    shape.total().max(1)
}

/// `out = base` with `out[.., idx[u], ..] (combine)= upd[.., u, ..]`: one lane
/// per output element, a counted loop over the updates. The output buffer does
/// not hold the base, so this nest reads it; one writer per element keeps the
/// accumulation order fixed and bit-reproducible.
///
/// `tm` is the number of destination bins one lane owns, sharing one `idx[u]`
/// read; `theta.dim` is not read.
pub(crate) fn lower_kscatter(ctx: Ctx<'_>, op: &Launch, theta: SchedPoint) -> Result<KernelIr> {
    let Launch::Scatter {
        axis, combine, ops, ..
    } = op
    else {
        return Err(Error::Plan("lower_kscatter on a non-Scatter node".into()));
    };
    let tiling = tiling(theta)?;
    let shape = ScatterGeometry::of(ops, *axis as usize, |d| ctx.binding.require(d), Error::Plan)?;
    // The operands are `(base, idx, upd)`; the base seeds the accumulator.
    let [base, idx, upd, ..] = ops.as_slice() else {
        return Err(Error::Plan(format!(
            "a scatter needs base/index/update operands, got {}",
            ops.len()
        )));
    };
    let b = &ctx.b;
    let out = ctx.linear_view(ctx.output()?)?;
    let out_elem = out.buffer.element;
    let acc_ty = match out_elem {
        ElementType::Scalar(s) => s,
        _ => ScalarElement::F32,
    }
    .element();
    let max_block = ctx.caps.limits.max_compute_invocations_per_workgroup;
    if let Some(row_block) = shape.row_block(max_block) {
        let block = ctx.block(row_block);
        let rows = shape.rows();
        let grid =
            distribute_workgroups(rows, ctx.caps.limits.max_compute_workgroups_per_dimension);
        let row = ctx.linear_workgroup();
        let lane = b.builtin(Builtin::Lane);
        let bins_e = b.u32(shape.bins);
        let inner_e = b.u32(shape.inner);
        let updates = shape.updates;
        let (o, bin) = b.divrem(row.clone(), bins_e);
        let live_row = b.lt(row.clone(), b.u32(rows));

        // Each lane owns every `block`-th column of its row; a column past
        // the row reads its last element and never stores.
        let slots = shape.inner.div_ceil(block);
        let mut init = Vec::new();
        let mut stores = Vec::new();
        let mut columns = Vec::new();
        for slot in 0..slots {
            let col = b.add(lane.clone(), b.u32(slot * block));
            let live = b.and(live_row.clone(), b.lt(col.clone(), inner_e.clone()));
            let clamped = b.min(col, b.u32(shape.inner - 1));
            let flat = b.add(b.mul(row.clone(), inner_e.clone()), clamped.clone());
            let acc = b.local(acc_ty);
            init.push(Stmt::StoreLocal {
                dst: acc.clone(),
                value: b.cast(
                    ctx.load_mapped(base, flat.clone(), total_of(&shape))?,
                    acc_ty,
                ),
            });
            stores.push(Stmt::Store {
                dst: out.clone(),
                addr: Addr::Linear(flat),
                value: b.cast(b.load_local(acc.clone()), out_elem),
                mask: live,
            });
            columns.push((acc, clamped));
        }

        // Stage `block` indices, then walk them in update order; a lane
        // loads an update only for a match, so the sum order is the dense
        // nest's and the result is bit-identical.
        let bins_tile = b.tile("scatter_bins", ScalarElement::U32.element(), &[block]);
        let chunk_local = b.local(ScalarElement::U32.element());
        let chunk_base = b.mul(b.load_local(chunk_local.clone()), b.u32(block));
        let u = b.add(chunk_base.clone(), lane.clone());
        let picked = b.cast(
            ctx.load_operand(idx, b.min(u.clone(), b.u32(updates - 1)))?,
            ScalarElement::U32.element(),
        );
        let j_local = b.local(ScalarElement::U32.element());
        let j = b.load_local(j_local.clone());
        let at = b.add(chunk_base, j.clone());
        let mut hit = Vec::new();
        for (acc, col) in &columns {
            let upd_index = b.add(
                b.mul(
                    b.add(b.mul(o.clone(), b.u32(updates)), at.clone()),
                    inner_e.clone(),
                ),
                col.clone(),
            );
            let v = b.cast(ctx.load_operand(upd, upd_index)?, acc_ty);
            let value = match combine {
                ScatterCombine::Add => b.add(b.load_local(acc.clone()), v),
                ScatterCombine::Set => v,
            };
            hit.push(Stmt::StoreLocal {
                dst: acc.clone(),
                value,
            });
        }
        let scan = Stmt::Loop {
            count: Some(b.min(
                b.u32(block),
                b.sub(
                    b.u32(updates),
                    b.mul(b.load_local(chunk_local.clone()), b.u32(block)),
                ),
            )),
            index: Some(j_local),
            accumulators: Vec::new(),
            body: vec![Stmt::If {
                condition: b.eq(b.load_tile(bins_tile.clone(), j), bin),
                accept: hit,
                reject: Vec::new(),
            }],
        };
        let mut body = init;
        body.push(Stmt::Loop {
            count: Some(b.u32(updates.div_ceil(block))),
            index: Some(chunk_local),
            accumulators: Vec::new(),
            body: vec![
                Stmt::Barrier,
                Stmt::StoreTile {
                    dst: bins_tile,
                    index: lane,
                    value: b.select(b.lt(u, b.u32(updates)), picked, b.u32(u32::MAX)),
                },
                Stmt::Barrier,
                scan,
            ],
        });
        body.extend(stores);
        return Ok(ctx.finish("scatter_rows", grid, block, body));
    }
    let block = ctx.block(emitted_block(1, ctx.caps));
    let total = shape.total().max(1);

    // The destination nest, as extents: [outer, bins, inner]. The tile runs
    // along the bins axis, so its stride is `inner`.
    let dest = [shape.outer, shape.bins, shape.inner].map(u64::from);
    let tile = LaneTile::new(total, tile_stride(&dest, 1, tiling.tm), tiling.tm);
    let grid = tile.grid(&ctx, block);
    let offsets = tile.offsets(&ctx, ctx.global_index(block));
    let bound = b.u32(u32::try_from(total).unwrap_or(u32::MAX));
    let inner_e = b.u32(shape.inner);
    let bins_e = b.u32(shape.bins);
    let row_span = b.u32(shape.bins.saturating_mul(shape.inner).max(1));
    let updates_e = b.u32(shape.updates);

    // One index read per update, hash-consed across the `tm` accumulators.
    let u_local = b.local(ScalarElement::U32.element());
    let u = b.load_local(u_local.clone());
    let u_bin = b.cast(
        ctx.load_operand(idx, u.clone())?,
        ScalarElement::U32.element(),
    );

    let mut accumulators = Vec::with_capacity(offsets.len());
    let mut stores = Vec::with_capacity(offsets.len());
    for flat in offsets {
        // (outer, destination bin, inner) of this output element.
        let o = b.div(flat.clone(), row_span.clone());
        let dest = b.rem(b.div(flat.clone(), inner_e.clone()), bins_e.clone());
        let within = b.rem(flat.clone(), inner_e.clone());

        // The accumulator starts at the base, so every element the updates
        // never touch still lands.
        let init = b.cast(ctx.load_mapped(base, flat.clone(), total)?, acc_ty);
        let acc_local = b.local(acc_ty);
        let acc_read = b.load_local(acc_local.clone());
        let upd_index = b.add(
            b.mul(
                b.add(b.mul(o, updates_e.clone()), u.clone()),
                inner_e.clone(),
            ),
            within,
        );
        let v = b.cast(ctx.load_operand(upd, upd_index)?, acc_ty);
        // `Add` accumulates duplicates; `Set` is only reachable with proven-unique
        // indices.
        let combined = match combine {
            ScatterCombine::Add => b.add(acc_read.clone(), v),
            ScatterCombine::Set => v,
        };
        accumulators.push(Accumulator {
            local: acc_local.clone(),
            init,
            update: b.select(b.eq(u_bin.clone(), dest), combined, acc_read),
        });
        stores.push(Stmt::Store {
            dst: out.clone(),
            addr: Addr::Linear(flat.clone()),
            value: b.cast(b.load_local(acc_local), out_elem),
            mask: b.lt(flat, bound.clone()),
        });
    }

    let mut body = vec![Stmt::Loop {
        count: Some(b.u32(shape.updates)),
        index: Some(u_local),
        accumulators,
        body: Vec::new(),
    }];
    body.extend(stores);
    Ok(ctx.finish("scatter_dense", grid, block, body))
}
