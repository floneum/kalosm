//! `Gather`'s two modes and `Scatter`'s two.
//!
//! Both nests read their lane tiling off `theta`. Currently the cost model does
//! not select tiled points, so `theta` is typically `SchedPoint::Point` and
//! bodies run one element per lane.

use fusor_ir::Result;
use fusor_ir::error::Error;
use fusor_ir::ir::kernel::{
    Accumulator, Addr, ElementType, KernelIr, ScalarElement, Stmt, TileExpr,
};
use fusor_ir::ir::launch::{Launch, MapTiling, SchedPoint};
use fusor_ir::ir::logical::ScatterCombine;
use fusor_tile::build::ScatterGeometry;

use crate::lower::{Ctx, distribute_workgroups};
use fusor_tile::domains::emitted_block;

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
/// `stride = prod(extents[axis+1..])` elements in the flattened space. The
/// map from lanes to elements is a bijection only when
/// `extents[..=axis].product() >= tm`; otherwise the tiled axis has fewer
/// blocks than the tile and the tile degrades to 1 rather than the plan
/// failing.
fn tile_stride(extents: &[u64], axis: usize, tm: u32) -> Option<u64> {
    if tm <= 1 || axis + 1 >= extents.len() {
        return None;
    }
    let stride: u64 = extents[axis + 1..].iter().product::<u64>().max(1);
    let blocks: u64 = extents[..=axis].iter().product::<u64>().max(1);
    (blocks >= u64::from(tm)).then_some(stride)
}

/// The lanes a tile needs to cover a space of `n` elements, and the flat
/// element offsets each lane owns. The tiled axis is blocked, so the lane
/// count reaches `ceil(blocks / tm)` whole tiles even when the last one is
/// partly masked: at `[13, 8]` with `tm = 2` a naive `n / tm` is 52 lanes
/// and element 100 is then written by nobody.
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

/// `RowPerGroup` and `QuantizedRows`, each at the lane tiling `theta`
/// selected. `QuantizedRows` shares the scalar nest: the source address is a
/// flat index into the source's dense logical space either way, and
/// `load_operand` runs the format's decode program there, so only gathered
/// rows decode.
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

    // The output index space, and the one axis on which the source differs
    // from it. Addressing every gather as if `axis` were 0 is only right
    // when the gathered axis is outermost.
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

/// `out = base` with `out[.., idx[u], ..] (combine)= upd[.., u, ..]`, at the
/// lane tiling `theta` selected. Both `ScatterMode`s lower through this one
/// nest: **one lane per output element, a counted loop over the updates**,
/// costing `O(out x updates)` index comparisons.
///
/// The update-parallel forms (one lane per update, `atomicAdd` or a
/// workgroup-private histogram) are correct only when the output buffer
/// already holds the base; `derive_bindings` gives a `Scatter`'s value its own
/// buffer and nothing copies the base in, so this nest must read the base.
///
/// Every output element is written by exactly one lane, so no atomic is needed
/// and the accumulation order is fixed: the result is bit-reproducible at any
/// occupancy, which is what `verify_launch`'s associativity obligation asks for.
///
/// **`tm` is the number of destination bins one lane owns.** With `tm` bins in
/// one lane the `idx[u]` read is hash-consed to one expression serving `tm`
/// accumulators. The bins axis is the only one that can be tiled without
/// breaking store coalescing: consecutive lanes still write consecutive
/// `inner` positions. `theta.dim` is not read: `space` is minted two ways, so
/// an axis index taken from it cannot be identified with a destination axis.
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

    // One index read per update, shared by every accumulator this lane
    // carries: `u_bin` does not depend on the output element, so the `tm`
    // slots hash-cons onto one load.
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
        // `Add` duplicates accumulate: an embedding table receiving one token
        // twice gets the summed gradient. `Set` is only reachable when the
        // node proved its indices unique.
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
