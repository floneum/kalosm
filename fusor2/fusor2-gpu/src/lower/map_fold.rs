//! `KMap` and `KFold`: the elementwise and reduction loop nests.
//!
//! Both read their geometry **off `theta`**: the schedule domain is already
//! scored against the realized DAG, so there is nothing left to decide here.

use fusor2_ir::Result;
#[cfg(test)]
use fusor2_ir::device::Caps;
use fusor2_ir::dtype::NumericContract;
use fusor2_ir::error::Error;
#[cfg(test)]
use fusor2_ir::ir::Node;
use fusor2_ir::carrier::Carrier;
use fusor2_ir::dtype::Splat;
#[cfg(test)]
use fusor2_ir::scalar::BinOp;
use fusor2_ir::ir::level1::{FoldStrat, L1, MapTiling, SchedPoint};
use fusor2_ir::ir::level2::{
    Accumulator, Addr, ElementType, KernelIr, ReduceKind, ScalarElement, Stmt, TileBinaryOp,
    TileCompareOp, TileExpr, TileReduceOp,
};
#[cfg(test)]
use fusor2_ir::target::LowerCtx;

use crate::lower::{Ctx, DimBinding, grid_for, scalar_element};
use fusor2_tile::domains::emitted_block;

/// Single-node entry point used by the tests below; production dispatch
/// goes through `lower::lower_node`.
#[cfg(test)]
pub fn lower(caps: &Caps, node: &Node, theta: SchedPoint, cx: &LowerCtx<'_>) -> Result<KernelIr> {
    let ctx = Ctx::new(caps, cx, DimBinding::new())?;
    match &node.op {
        fusor2_ir::ir::Op::L1(op @ L1::KMap { .. }) => lower_kmap(ctx, op, theta),
        fusor2_ir::ir::Op::L1(op @ L1::KFold { .. }) => lower_kfold(ctx, op, theta),
        _ => Err(Error::Plan("map_fold was handed a foreign node".into())),
    }
}

// KMap

/// Lower a `KMap` at a [`MapTiling`].
///
/// `dim: None` is the untiled body: one output per lane. Otherwise each lane
/// computes `tm` outputs along `dim` and every operand that does *not* vary
/// with `dim` is hoisted into a `Local` before the loop, so it is read once
/// per lane instead of `tm` times.
pub fn lower_kmap(mut ctx: Ctx<'_>, op: &L1, theta: SchedPoint) -> Result<KernelIr> {
    let L1::KMap { space, body, ops, .. } = op else {
        return Err(Error::Plan("lower_kmap on a non-KMap node".into()));
    };
    let tiling = match theta {
        SchedPoint::Map(t) => t,
        SchedPoint::Point => MapTiling {
            dim: None,
            tm: 1,
            vector: 1,
        },
        other => {
            return Err(Error::Plan(format!(
                "KMap needs SchedPoint::Map, got {other:?}"
            )));
        }
    };

    let out = ctx.output()?;
    let out_view = ctx.linear_view(out)?;
    let out_elem = out_view.buffer.element;
    let block = emitted_block(1, ctx.caps);
    let limits = ctx.caps.limits;

    let mut body_stmts: Vec<Stmt> = Vec::new();
    let total = space_extent_expr(&mut ctx, space)?;
    let space_total = space.iterations().unwrap_or(0);

    // The dispatch grid, computed before the body so `global_index`
    // linearizes against the grid this kernel is actually launched with.
    let tm = tiling.tm.max(1);
    let grid = tiled_grid(space, block, tm, &ctx.binding, &limits)?;

    match tiling.dim {
        None => {
            let index = ctx.global_index(block, grid);
            let mask = ctx.b.compare(TileCompareOp::Lt, index.clone(), total);
            let coords = ctx.coords_from_linear(index.clone(), space)?;
            let mut args = Vec::with_capacity(ops.len());
            for operand in ops {
                args.push(ctx.load_mapped(operand, index.clone(), space_total)?);
            }
            let value = ctx.eval_scalar(body, &args, &coords)?;
            let value = ctx.b.cast(value, out_elem);
            body_stmts.push(Stmt::Store {
                dst: out_view,
                addr: Addr::Linear(index),
                value,
                mask,
            });
        }
        Some(dim) => {
            let tm = tiling.tm.max(1);
            let axis = dim as usize;
            if axis >= space.rank() {
                return Err(Error::Plan(format!(
                    "map tiling names axis {axis} of a rank-{} space",
                    space.rank()
                )));
            }
            // A thread-local run along the innermost axis breaks inter-thread
            // store coalescing, so the schedule domain never offers it.
            if axis + 1 == space.rank() {
                return Err(Error::Plan(
                    "map tiling on the innermost axis destroys store coalescing".into(),
                ));
            }

            let base = ctx.global_index(block, grid);
            let stride = inner_extent_expr(&mut ctx, op, axis)?;
            let tm_e = ctx.b.lit_u32(tm);
            let step = ctx.b.mul(stride.clone(), tm_e);
            let tile_base = {
                let outer = ctx
                    .b
                    .binary(TileBinaryOp::Div, base.clone(), stride.clone(), NumericContract::RELAXED);
                let inner = ctx
                    .b
                    .binary(TileBinaryOp::Rem, base, stride.clone(), NumericContract::RELAXED);
                let scaled = ctx.b.mul(outer, step);
                ctx.b.add(scaled, inner)
            };

            // Hoist every operand whose access does not vary along `dim`; the
            // reuse it buys is already priced by the schedule domain.
            let mut hoisted: Vec<Option<TileExpr>> = Vec::with_capacity(ops.len());
            for operand in ops {
                if operand_is_invariant(operand, axis) {
                    let v = ctx.load_mapped(operand, tile_base.clone(), space_total)?;
                    let local = ctx.b.local(v.element());
                    body_stmts.push(Stmt::StoreLocal {
                        dst: local.clone(),
                        value: v,
                    });
                    hoisted.push(Some(ctx.b.load_local(local)));
                } else {
                    hoisted.push(None);
                }
            }

            for t in 0..tm {
                let off = {
                    let t_e = ctx.b.lit_u32(t);
                    let scaled = ctx.b.mul(stride.clone(), t_e);
                    ctx.b.add(tile_base.clone(), scaled)
                };
                let mask = ctx.b.compare(TileCompareOp::Lt, off.clone(), total.clone());
                let coords = ctx.coords_from_linear(off.clone(), space)?;
                let mut args = Vec::with_capacity(ops.len());
                for (operand, cached) in ops.iter().zip(&hoisted) {
                    match cached {
                        Some(v) => args.push(v.clone()),
                        None => args.push(ctx.load_mapped(operand, off.clone(), space_total)?),
                    }
                }
                let value = ctx.eval_scalar(body, &args, &coords)?;
                let value = ctx.b.cast(value, out_elem);
                body_stmts.push(Stmt::Store {
                    dst: out_view.clone(),
                    addr: Addr::Linear(off),
                    value,
                    mask,
                });
            }
        }
    }

    Ok(ctx.finish("kmap", grid, block, body_stmts))
}

/// An operand is loop-invariant along `axis` when its layout gives that axis
/// stride 0 or extent 1 — `layout_index` drops both, so the address does not
/// move as the tiled coordinate advances.
fn operand_is_invariant(operand: &fusor2_ir::ir::level1::Operand, axis: usize) -> bool {
    let layout = &operand.layout;
    if axis >= layout.rank() {
        return true;
    }
    layout.strides()[axis].known_eq(fusor2_ir::shape::Dim::Const(0))
        || layout.shape()[axis].known_eq(fusor2_ir::shape::Dim::Const(1))
}

fn space_extent_expr(ctx: &mut Ctx<'_>, space: &fusor2_ir::ir::level1::IndexSpace) -> Result<TileExpr> {
    let mut acc = ctx.b.lit_u32(1);
    for dim in &space.dims {
        let e = ctx.dim_expr(*dim)?;
        acc = ctx.b.mul(acc, e);
    }
    Ok(acc)
}

/// Product of the extents strictly inside `axis` — the element distance one
/// step along `axis` covers in the flattened index space.
fn inner_extent_expr(ctx: &mut Ctx<'_>, op: &L1, axis: usize) -> Result<TileExpr> {
    let L1::KMap { space, .. } = op else {
        return Err(Error::Plan("inner_extent_expr on a non-KMap node".into()));
    };
    let mut acc = ctx.b.lit_u32(1);
    for dim in space.dims.iter().skip(axis + 1) {
        let e = ctx.dim_expr(*dim)?;
        acc = ctx.b.mul(acc, e);
    }
    Ok(acc)
}

fn tiled_grid(
    space: &fusor2_ir::ir::level1::IndexSpace,
    block: u32,
    tm: u32,
    binding: &DimBinding,
    limits: &fusor2_ir::device::Limits,
) -> Result<[u32; 3]> {
    let full = grid_for(space, block.saturating_mul(tm.max(1)), binding, limits)?;
    Ok(full)
}

// KFold

/// Lower a `KFold` at a [`FoldStrat`].
///
/// Three bodies, one carrier shape each:
/// * [`FoldStrat::Subgroup`] — a subgroup collective, no scratch, no barrier.
/// * [`FoldStrat::WgTree`] — a shared-memory tree over one scratch tile.
/// * [`FoldStrat::LoopThenTree`] — a per-lane accumulate loop, then the tree.
///
/// The carrier's `lift` runs before the merge and `post` after it, so a
/// softmax's `exp` and a mean's divide fuse into the same launch.
///
/// **Two bodies, one dispatch.** One scalar slot merged by a hardware operator
/// takes the collective path below; anything wider goes to
/// [`lower_kfold_carrier`], which carries one accumulator per lane and closes
/// with `Stmt::Reduce`'s N-ary merge, so a carrier like `Fold{(max, sum)}`
/// keeps every slot.
pub fn lower_kfold(mut ctx: Ctx<'_>, op: &L1, theta: SchedPoint) -> Result<KernelIr> {
    let L1::KFold {
        space,
        axis,
        vec_axes,
        carrier,
        acc,
        post,
        ops,
        ..
    } = op
    else {
        return Err(Error::Plan("lower_kfold on a non-KFold node".into()));
    };
    if !vec_axes.is_empty() || fusor2_ir::ir::level2::fast_reduce_op(carrier).is_none() {
        return lower_kfold_carrier(ctx, op, theta);
    }
    let reduce_op = single_slot_reduce_op(carrier)?;
    let pre = &carrier.lift[0];
    let post = &post[0];
    // `ScheduleDomain::Point` means "this node has no schedule parameters",
    // which is exactly what the floor lowering (`lower_fold`,
    // `lower_contract_generic`) mints. It is a legal domain, so it needs a
    // legal body, not an error: the subgroup collective where the device has
    // subgroups and the shared-memory tree where it does not. The CPU
    // emitter already defaults the same way.
    let strat = match theta {
        SchedPoint::Fold(s) => s,
        _ if ctx.caps.subgroups.is_some() => FoldStrat::Subgroup,
        _ => FoldStrat::WgTree {
            lane_group: emitted_block(1, ctx.caps),
        },
    };

    let axis = *axis as usize;
    if axis >= space.rank() {
        return Err(Error::Plan(format!(
            "fold axis {axis} is outside a rank-{} space",
            space.rank()
        )));
    }

    let space_total = space.iterations().unwrap_or(0);
    let acc_elem = scalar_element(*acc);

    let (block, lane_group) = match strat {
        FoldStrat::Subgroup => (
            ctx.caps
                .subgroup_width()
                .max(1)
                .min(ctx.caps.limits.max_compute_invocations_per_workgroup),
            ctx.caps.subgroup_width().max(1),
        ),
        FoldStrat::WgTree { lane_group } | FoldStrat::LoopThenTree { lane_group, .. } => {
            let lg = lane_group.max(1);
            (emitted_block(lg, ctx.caps), lg)
        }
    };
    let limits = ctx.caps.limits;

    // One row per output element; the fold axis is consumed by the lanes.
    let mut row_space = space.clone();
    row_space.dims.remove(axis);
    let rows = space_extent_expr(&mut ctx, &row_space)?;
    let axis_extent = ctx.dim_expr(space.dims[axis])?;
    let inner: TileExpr = {
        let mut acc_e = ctx.b.lit_u32(1);
        for dim in space.dims.iter().skip(axis + 1) {
            let e = ctx.dim_expr(*dim)?;
            acc_e = ctx.b.mul(acc_e, e);
        }
        acc_e
    };

    let out = ctx.output()?;
    let out_view = ctx.linear_view(out)?;
    let out_elem = out_view.buffer.element;

    let mut stmts: Vec<Stmt> = Vec::new();

    // The dispatch grid up front: `global_index` linearizes the workgroup id
    // against it, so it must be the grid this kernel is launched with.
    let grid = grid_for(&row_space, block / lane_group.max(1), &ctx.binding, &limits)?;

    // Row identity: one lane group per output row.
    let group = ctx.global_index(block, grid);
    let lg_e = ctx.b.lit_u32(lane_group);
    let row = ctx
        .b
        .binary(TileBinaryOp::Div, group.clone(), lg_e.clone(), NumericContract::RELAXED);
    let lane = ctx
        .b
        .binary(TileBinaryOp::Rem, group, lg_e.clone(), NumericContract::RELAXED);
    let row_live = ctx.b.compare(TileCompareOp::Lt, row.clone(), rows);

    // Element index of `(row, k)` in the flattened space.
    let outer = ctx
        .b
        .binary(TileBinaryOp::Div, row.clone(), inner.clone(), NumericContract::RELAXED);
    let within = ctx
        .b
        .binary(TileBinaryOp::Rem, row.clone(), inner.clone(), NumericContract::RELAXED);
    let row_stride = ctx.b.mul(inner.clone(), axis_extent.clone());
    let row_base = {
        let hi = ctx.b.mul(outer, row_stride);
        ctx.b.add(hi, within)
    };

    // One accumulator per slot, seeded from the carrier's own identity.
    let mut accs: Vec<Accumulator> = Vec::with_capacity(carrier.width());
    let mut acc_reads: Vec<TileExpr> = Vec::with_capacity(carrier.width());
    for slot in 0..carrier.width() {
        let local = ctx.b.local(ElementType::Scalar(acc_elem));
        let init = identity_expr(&mut ctx, carrier.identity[slot], acc_elem);
        let read = ctx.b.load_local(local.clone());
        acc_reads.push(read.clone());
        accs.push(Accumulator {
            local,
            init,
            update: read,
        });
    }

    // The strided element read for one k, with `pre` applied.
    let read_k = |ctx: &mut Ctx<'_>, k: TileExpr| -> Result<TileExpr> {
        let idx = {
            let off = ctx.b.mul(k, inner.clone());
            ctx.b.add(row_base.clone(), off)
        };
        let mut args = Vec::with_capacity(ops.len());
        for operand in ops {
            args.push(ctx.load_mapped(operand, idx.clone(), space_total)?);
        }
        let coords = ctx.coords_from_linear(idx, space)?;
        ctx.eval_scalar(pre, &args, &coords)
    };

    // A lane past the reduced extent contributes the combine's identity. The
    // collective spans the whole lane group whatever the extent is, so without
    // this a `[2, 4]` row fold on a 32-wide subgroup would sum the *next* row
    // into row 0.
    let guard = |ctx: &mut Ctx<'_>, k: &TileExpr, v: TileExpr| -> TileExpr {
        let in_range = ctx
            .b
            .compare(TileCompareOp::Lt, k.clone(), axis_extent.clone());
        let ident = identity_expr(ctx, carrier.identity[0], acc_elem);
        ctx.b.select(in_range, v, ident)
    };

    // A collective read at `k = lane` and nothing else would reduce only the
    // first `lane_group` elements, so anything longer than one pass needs the
    // per-lane strided loop first, whichever collective closes it.
    let one_pass = space.dims[axis]
        .as_const()
        .is_some_and(|k| k <= u64::from(lane_group.max(1)));

    let lane_value = if one_pass {
        let v = read_k(&mut ctx, lane.clone())?;
        let v = ctx.b.cast(v, ElementType::Scalar(acc_elem));
        guard(&mut ctx, &lane, v)
    } else {
        // Per-lane loop accumulate. The loop's accumulator is SSA-carried,
        // never reloaded per iteration. The trip count comes from the *runtime*
        // extent rather than from `FoldStrat::LoopThenTree`'s `iterations`, so
        // one formula covers a symbolic extent and the two collective
        // strategies too; `fold_domain` derives the same number from a constant
        // `k`, so a priced `LoopThenTree` emits exactly what it costed.
        let index = ctx.b.local(ElementType::Scalar(ScalarElement::U32));
        let idx_read = ctx.b.load_local(index.clone());
        let k = {
            let scaled = ctx.b.mul(idx_read, lg_e.clone());
            ctx.b.add(scaled, lane.clone())
        };
        let v = read_k(&mut ctx, k.clone())?;
        let v = ctx.b.cast(v, ElementType::Scalar(acc_elem));
        let guarded = guard(&mut ctx, &k, v);
        let partial = ctx.b.binary(
            reduce_op.binary(),
            acc_reads[0].clone(),
            guarded,
            NumericContract::RELAXED,
        );
        accs[0].update = partial;
        let count = {
            let lg_minus_1 = ctx.b.lit_u32(lane_group.max(1) - 1);
            let numerator = ctx.b.add(axis_extent.clone(), lg_minus_1);
            ctx.b.binary(
                TileBinaryOp::Div,
                numerator,
                lg_e.clone(),
                NumericContract::RELAXED,
            )
        };
        stmts.push(Stmt::Loop {
            count: Some(count),
            index: Some(index),
            accumulators: accs.clone(),
            body: Vec::new(),
        });
        ctx.b.load_local(accs[0].local.clone())
    };

    let reduced: Vec<TileExpr> = match strat {
        // One collective over the subgroup: no scratch and no barrier.
        FoldStrat::Subgroup => vec![ctx.b.reduce(reduce_op, ReduceKind::Subgroup, lane_value)],
        // A one-lane group owns its whole row, so the close is the identity
        // and stages nothing. `fold_scratch_bytes` reports 0 for the same
        // strategy; the two have to agree or the arena admits a plan this
        // emitter cannot lay out.
        FoldStrat::WgTree { .. } | FoldStrat::LoopThenTree { .. } if lane_group <= 1 => {
            vec![lane_value]
        }
        FoldStrat::WgTree { .. } | FoldStrat::LoopThenTree { .. } => {
            let scratch = ctx
                .b
                .tile("fold_scratch", ElementType::Scalar(acc_elem), &[block]);
            vec![ctx.b.reduce(
                reduce_op,
                ReduceKind::Workgroup {
                    scratch,
                    group_size: lane_group,
                },
                lane_value,
            )]
        }
    };

    let value = ctx.eval_scalar(post, &reduced, &[row.clone()])?;
    let value = ctx.b.cast(value, out_elem);
    let lane_zero = {
        let z = ctx.b.lit_u32(0);
        ctx.b.compare(TileCompareOp::Eq, lane, z)
    };
    let mask = ctx.b.and(row_live, lane_zero);
    stmts.push(Stmt::Store {
        dst: out_view,
        addr: Addr::Linear(row),
        value,
        mask,
    });

    Ok(ctx.finish("kfold", grid, block, stmts))
}

/// Lower a `KFold` whose carrier is **wider than one hardware operator**.
///
/// One accumulator per carrier lane, seeded from that lane's own identity,
/// absorbed with the carrier's own `merge`, and closed by `Stmt::Reduce`'s N-ary
/// tree. The output carries `carrier.lanes()` values per row — the trailing
/// carrier axis `infer_l1` already appends — so slot readback downstream is an
/// ordinary `Restride`.
///
/// There is no subgroup collective for a multi-lane merge, so this always closes
/// with the workgroup tree. `theta`'s `FoldStrat` still chooses the lane group,
/// and a `Subgroup` point is honoured as a tree *at the subgroup width*: the same
/// lanes, the same value, one strategy the hardware can actually run.
fn lower_kfold_carrier(mut ctx: Ctx<'_>, op: &L1, theta: SchedPoint) -> Result<KernelIr> {
    let L1::KFold {
        space,
        axis,
        vec_axes,
        carrier,
        acc,
        post,
        ops,
        ..
    } = op
    else {
        return Err(Error::Plan("lower_kfold_carrier on a non-KFold node".into()));
    };
    let axis = *axis as usize;
    if axis >= space.rank() {
        return Err(Error::Plan(format!(
            "fold axis {axis} is outside a rank-{} space",
            space.rank()
        )));
    }
    // The shared expand/legality prologue: one output row spans
    // `vec_extent * axis_extent` consecutive elements, which is what makes
    // the address below one multiply.
    let fusor2_ir::carrier::CarrierNest {
        merges,
        posts,
        lanes,
        vec_extent,
        iter_axes,
    } = fusor2_ir::carrier::CarrierNest::validate(carrier, space, axis, vec_axes, post)?;
    let space_total = space.iterations().unwrap_or(0);
    let acc_elem = scalar_element(*acc);
    let acc_ty = ElementType::Scalar(acc_elem);
    let limits = ctx.caps.limits;
    let max_block = emitted_block(1, ctx.caps);
    let lane_group = match theta {
        SchedPoint::Fold(FoldStrat::WgTree { lane_group })
        | SchedPoint::Fold(FoldStrat::LoopThenTree { lane_group, .. }) => lane_group.max(1),
        SchedPoint::Fold(FoldStrat::Subgroup) => ctx.caps.subgroup_width().max(1),
        _ => max_block,
    };
    let block = lane_group.max(max_block);

    // Output rows are `space` minus the reduced axis AND minus every promoted
    // axis: a promoted extent lives in the carrier's lanes, not in the write
    // map, which is exactly what makes the output shape `free ++ [lanes]`.
    let mut row_space = space.clone();
    row_space.dims.remove(axis);
    for i in vec_axes.iter().rev() {
        row_space.dims.remove(*i as usize);
    }
    let rows = space_extent_expr(&mut ctx, &row_space)?;
    let axis_extent = ctx.dim_expr(space.dims[axis])?;
    let inner: TileExpr = {
        let mut acc_e = ctx.b.lit_u32(1);
        for dim in space.dims.iter().skip(axis + 1) {
            let e = ctx.dim_expr(*dim)?;
            acc_e = ctx.b.mul(acc_e, e);
        }
        acc_e
    };

    let out = ctx.output()?;
    let out_view = ctx.linear_view(out)?;
    let out_elem = out_view.buffer.element;
    let mut stmts: Vec<Stmt> = Vec::new();

    let grid = grid_for(&row_space, block / lane_group, &ctx.binding, &limits)?;
    let group = ctx.global_index(block, grid);
    let lg_e = ctx.b.lit_u32(lane_group);
    let row = ctx
        .b
        .binary(TileBinaryOp::Div, group.clone(), lg_e.clone(), NumericContract::RELAXED);
    let lane = ctx
        .b
        .binary(TileBinaryOp::Rem, group, lg_e.clone(), NumericContract::RELAXED);
    let row_live = ctx.b.compare(TileCompareOp::Lt, row.clone(), rows);

    let outer = ctx
        .b
        .binary(TileBinaryOp::Div, row.clone(), inner.clone(), NumericContract::RELAXED);
    let within = ctx
        .b
        .binary(TileBinaryOp::Rem, row.clone(), inner.clone(), NumericContract::RELAXED);
    // One output row spans every promoted position of every reduced element,
    // so its stride carries `vec_extent`.
    let pos_stride = ctx.b.mul(inner.clone(), axis_extent.clone());
    let row_stride = {
        let ve = ctx.b.lit_u32(vec_extent as u32);
        ctx.b.mul(pos_stride.clone(), ve)
    };
    let row_base = {
        let hi = ctx.b.mul(outer, row_stride);
        ctx.b.add(hi, within)
    };

    // One lifted value per lane at element `k`, each guarded to **its own**
    // identity outside the reduced extent: a lane past the extent must
    // contribute nothing to every slot, and Welford's constant `1` lift is
    // exactly the slot where a shared identity would count a padding lane.
    //
    // **Per-position operand addressing.** A `Vector` slot is `vec_extent`
    // registers, and lane `(slot, p)` reads every operand at promoted position
    // `p`. An operand invariant in the promoted axes — the score row, in an
    // attention nest — has stride 0 there and its load is hash-consed back to
    // one read reused across all positions, which is the whole inner-loop win;
    // an operand that varies (the value matrix) is read once per position, as
    // it must be.
    let lift_at = |ctx: &mut Ctx<'_>, k: &TileExpr| -> Result<Vec<TileExpr>> {
        let in_range = ctx
            .b
            .compare(TileCompareOp::Lt, k.clone(), axis_extent.clone());
        let mut per_pos: Vec<(Vec<TileExpr>, Vec<TileExpr>)> =
            Vec::with_capacity(vec_extent as usize);
        for p in 0..vec_extent {
            let idx = {
                let off = ctx.b.mul(k.clone(), inner.clone());
                let base = ctx.b.add(row_base.clone(), off);
                if p == 0 {
                    base
                } else {
                    let pe = ctx.b.lit_u32(p as u32);
                    let shift = ctx.b.mul(pos_stride.clone(), pe);
                    ctx.b.add(base, shift)
                }
            };
            let mut args = Vec::with_capacity(ops.len());
            for operand in ops {
                args.push(ctx.load_mapped(operand, idx.clone(), space_total)?);
            }
            // `IndexOf` on this node names an ITERATION axis; resolve it
            // through `iter_axes` rather than against `space` directly.
            let full = ctx.coords_from_linear(idx, space)?;
            let coords: Vec<TileExpr> = iter_axes.iter().map(|i| full[*i].clone()).collect();
            per_pos.push((args, coords));
        }
        let lane_slots = carrier.lane_slots().ok_or_else(|| {
            Error::Plan("this carrier has a symbolic Vector extent".into())
        })?;
        let mut out = Vec::with_capacity(lanes);
        for (slot, p) in lane_slots {
            let (args, coords) = &per_pos[p as usize];
            let (args, coords) = (args.clone(), coords.clone());
            let v = ctx.eval_scalar(&carrier.lift[slot], &args, &coords)?;
            let v = ctx.b.cast(v, acc_ty);
            let ident = identity_expr(ctx, carrier.identity[slot], acc_elem);
            out.push(ctx.b.select(in_range.clone(), v, ident));
        }
        Ok(out)
    };

    let one_pass = space.dims[axis]
        .as_const()
        .is_some_and(|k| k <= u64::from(lane_group));

    let partials: Vec<TileExpr> = if one_pass {
        lift_at(&mut ctx, &lane)?
    } else {
        // The per-lane strided loop, carrying `lanes` SSA accumulators seeded
        // from the carrier's identities and absorbed with its own `merge`.
        let index = ctx.b.local(ElementType::Scalar(ScalarElement::U32));
        let idx_read = ctx.b.load_local(index.clone());
        let k = {
            let scaled = ctx.b.mul(idx_read, lg_e.clone());
            ctx.b.add(scaled, lane.clone())
        };
        let mut accs: Vec<Accumulator> = Vec::with_capacity(lanes);
        let mut acc_reads: Vec<TileExpr> = Vec::with_capacity(lanes);
        let lane_ident = carrier.identity_lanes().ok_or_else(|| {
            Error::Plan("this carrier has a symbolic Vector extent".into())
        })?;
        for slot in 0..lanes {
            let local = ctx.b.local(acc_ty);
            let init = identity_expr(&mut ctx, lane_ident[slot], acc_elem);
            let read = ctx.b.load_local(local.clone());
            acc_reads.push(read.clone());
            accs.push(Accumulator {
                local,
                init,
                update: read,
            });
        }
        let values = lift_at(&mut ctx, &k)?;
        let mut args = acc_reads.clone();
        args.extend(values);
        for slot in 0..lanes {
            accs[slot].update = ctx.eval_scalar(&merges[slot], &args, &[])?;
        }
        let count = {
            let lg_minus_1 = ctx.b.lit_u32(lane_group - 1);
            let numerator = ctx.b.add(axis_extent.clone(), lg_minus_1);
            ctx.b
                .binary(TileBinaryOp::Div, numerator, lg_e.clone(), NumericContract::RELAXED)
        };
        stmts.push(Stmt::Loop {
            count: Some(count),
            index: Some(index),
            accumulators: accs.clone(),
            body: Vec::new(),
        });
        accs.iter()
            .map(|a| ctx.b.load_local(a.local.clone()))
            .collect()
    };

    // The cross-lane close: one scratch tile per lane, one merge per lane.
    //
    // **Skipped entirely at a one-lane group.** `row = group / lane_group` and
    // the accumulation loop runs `(axis_extent + lane_group - 1) / lane_group`
    // times, so at `lane_group == 1` this invocation already reduced the whole
    // axis for its own row and there is no partner to merge with. Emitting the
    // close anyway would stage `lanes * block * acc_bytes` bytes to compute an
    // identity — which is exactly the footprint that makes a wide promoted
    // carrier unschedulable, and `fold_scratch_bytes` reports 0 here for the
    // same reason. The two must agree or the arena admits a plan the emitter
    // cannot lay out.
    let reduced: Vec<TileExpr> = if lane_group <= 1 {
        partials
    } else {
        let scratch: smallvec::SmallVec<[fusor2_ir::ir::level2::Tile; 4]> = (0..lanes)
            .map(|_| ctx.b.tile("fold_scratch", acc_ty, &[block]))
            .collect();
        let lhs: smallvec::SmallVec<[fusor2_ir::ir::level2::Local; 4]> =
            (0..lanes).map(|_| ctx.b.local(acc_ty)).collect();
        let rhs: smallvec::SmallVec<[fusor2_ir::ir::level2::Local; 4]> =
            (0..lanes).map(|_| ctx.b.local(acc_ty)).collect();
        let outs: smallvec::SmallVec<[fusor2_ir::ir::level2::Local; 4]> =
            (0..lanes).map(|_| ctx.b.local(acc_ty)).collect();
        let mut merge_args: Vec<TileExpr> = Vec::with_capacity(2 * lanes);
        for l in lhs.iter().chain(rhs.iter()) {
            merge_args.push(ctx.b.load_local(l.clone()));
        }
        let mut body: smallvec::SmallVec<[TileExpr; 4]> = smallvec::SmallVec::new();
        for slot in 0..lanes {
            body.push(ctx.eval_scalar(&merges[slot], &merge_args, &[])?);
        }
        stmts.push(Stmt::Reduce {
            kind: Box::new(ReduceKind::Workgroup {
                scratch: scratch[0].clone(),
                group_size: lane_group,
            }),
            values: partials.into_iter().collect(),
            merge: Box::new(fusor2_ir::ir::level2::MergeBody { lhs, rhs, body }),
            fast: None,
            outs: outs.clone(),
            scratch,
        });
        // One output per slot, at the trailing carrier axis.
        outs.iter().map(|l| ctx.b.load_local(l.clone())).collect()
    };
    let lane_zero = {
        let z = ctx.b.lit_u32(0);
        ctx.b.compare(TileCompareOp::Eq, lane, z)
    };
    let mask = ctx.b.and(row_live, lane_zero);
    let lanes_e = ctx.b.lit_u32(lanes as u32);
    let base = ctx.b.mul(row.clone(), lanes_e);
    for slot in 0..lanes {
        let value = ctx.eval_scalar(&posts[slot], &reduced, &[row.clone()])?;
        let value = ctx.b.cast(value, out_elem);
        let off = ctx.b.lit_u32(slot as u32);
        let addr = ctx.b.add(base.clone(), off);
        stmts.push(Stmt::Store {
            dst: out_view.clone(),
            addr: Addr::Linear(addr),
            value,
            mask: mask.clone(),
        });
    }

    Ok(ctx.finish("kfold_carrier", grid, block, stmts))
}

/// The hardware collective this carrier reduces with, or an honest `Err`.
///
/// This is `Carrier::kind()` — one scalar slot whose merge is a binop — mapped
/// onto `TileReduceOp`. Everything wider needs the N-lane `Stmt::Reduce` and
/// says so instead of computing slot 0 and dropping the rest.
fn single_slot_reduce_op(c: &Carrier) -> Result<TileReduceOp> {
    fusor2_ir::ir::level2::single_slot_reduce_op(c).map_err(Error::Plan)
}

/// A carrier identity as a tile literal. The infinities go through the
/// builder's own spellings rather than a formatted float.
fn identity_expr(ctx: &mut Ctx<'_>, s: Splat, elem: ScalarElement) -> TileExpr {
    let f = match s {
        Splat::F32(v) => v,
        Splat::F16(b) => half::f16::from_bits(b).to_f32(),
        Splat::BF16(b) => half::bf16::from_bits(b).to_f32(),
        Splat::U32(v) => return if v == 0 { ctx.b.zero_scalar(elem) } else if v == u32::MAX { ctx.b.pos_inf(elem) } else { ctx.b.lit_u32(v) },
        Splat::I32(v) => return if v == 0 { ctx.b.zero_scalar(elem) } else if v == i32::MIN { ctx.b.neg_inf(elem) } else if v == i32::MAX { ctx.b.pos_inf(elem) } else { ctx.b.lit_i32(v) },
    };
    if f == f32::NEG_INFINITY {
        ctx.b.neg_inf(elem)
    } else if f == f32::INFINITY {
        ctx.b.pos_inf(elem)
    } else if f == 0.0 {
        ctx.b.zero_scalar(elem)
    } else if f == 1.0 {
        match elem {
            ScalarElement::U32 => ctx.b.lit_u32(1),
            ScalarElement::I32 => ctx.b.lit_i32(1),
            _ => ctx.b.lit_f32(1.0),
        }
    } else {
        ctx.b.lit_f32(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fusor2_ir::ir::level1::{AccessPlan, IndexSpace, Operand, ScheduleDomain};
    use fusor2_ir::shape::{Dim, Layout};

    fn operand(shape: &[Dim], strides: &[Dim]) -> Operand {
        Operand {
            src: fusor2_ir::egraph::Id(0),
            layout: Layout::from_parts(Dim::Const(0), shape, strides).unwrap(),
            access: AccessPlan::Alias,
        }
    }

    #[test]
    fn stride_zero_operands_hoist() {
        let bias = operand(
            &[Dim::Const(8), Dim::Const(64)],
            &[Dim::Const(0), Dim::Const(1)],
        );
        assert!(operand_is_invariant(&bias, 0));
        assert!(!operand_is_invariant(&bias, 1));
    }

    #[test]
    fn extent_one_operands_hoist() {
        let x = operand(
            &[Dim::Const(1), Dim::Const(64)],
            &[Dim::Const(64), Dim::Const(1)],
        );
        assert!(operand_is_invariant(&x, 0));
    }

    fn binop(op: BinOp) -> Carrier {
        Carrier::binop(
            op,
            Carrier::binop_identity(op, fusor2_ir::dtype::Dtype::F32).unwrap(),
            fusor2_ir::dtype::Dtype::F32,
        )
    }

    #[test]
    fn every_single_slot_binop_carrier_has_a_collective() {
        for (op, want) in [
            (BinOp::Add, TileReduceOp::Sum),
            (BinOp::Mul, TileReduceOp::Product),
            (BinOp::Max, TileReduceOp::Max),
            (BinOp::Min, TileReduceOp::Min),
        ] {
            assert_eq!(single_slot_reduce_op(&binop(op)).unwrap(), want);
        }
        // A fused lift does not change the collective — every one of the
        // passing folds goes down this path with `pre` inlined.
        let fused = binop(BinOp::Add).with_lift([fusor2_ir::scalar::ScalarExpr::bin(
            BinOp::Mul,
            fusor2_ir::scalar::ScalarExpr::arg(0, fusor2_ir::dtype::Dtype::F32),
            fusor2_ir::scalar::ScalarExpr::arg(1, fusor2_ir::dtype::Dtype::F32),
        )]);
        assert_eq!(single_slot_reduce_op(&fused).unwrap(), TileReduceOp::Sum);
    }

    // The single-slot shader golden

    /// Lower one `KFold` at `carrier` over `[3, 8]` axis 1.
    fn fold_ir_result(carrier: Carrier, theta: SchedPoint) -> Result<KernelIr> {
        fold_ir_in(carrier, theta, &[3, 8], 1, &[])
    }

    /// [`fold_ir_result`] over an explicit space, so a **promoted** nest — one
    /// whose accumulator-resident axes sit immediately before the reduced one —
    /// can be lowered here too.
    fn fold_ir_in(
        carrier: Carrier,
        theta: SchedPoint,
        dims: &[u64],
        axis: u32,
        vec_axes: &[u32],
    ) -> Result<KernelIr> {
        use fusor2_ir::cost::Picoseconds;
        use fusor2_ir::dtype::Dtype;
        use fusor2_ir::egraph::EGraph;
        use fusor2_ir::extract::{Extraction, PlanHash};
        use fusor2_ir::ir::level0::{BufferId, L0, LeafKind};
        use fusor2_ir::ir::level1::L1;
        use fusor2_ir::ir::Op;
        use fusor2_ir::extract::{BindKind, BindingPlan, Launch, Plan};
        use fusor2_ir::scalar::ScalarExpr;
        use fusor2_ir::semantics::{CoreSemantics, SumArenaPlanner};
        use std::sync::Arc;

        let mut g = EGraph::new(CoreSemantics::new(Arc::new(SumArenaPlanner)));
        let shape: smallvec::SmallVec<[Dim; 6]> =
            dims.iter().map(|d| Dim::Const(*d)).collect();
        let x = g
            .add(Op::L0(L0::Leaf(LeafKind::Buffer {
                name: BufferId(0),
                dtype: Dtype::F32,
                shape,
            })))
            .unwrap();
        let width = carrier.width();
        let k = g
            .add(Op::L1(L1::KFold {
                space: IndexSpace::new(dims.iter().map(|d| Dim::Const(*d))),
                axis,
                vec_axes: vec_axes.iter().copied().collect(),
                carrier,
                acc: Dtype::F32,
                post: (0..width).map(|i| ScalarExpr::arg(i as u32, Dtype::F32)).collect(),
                ops: vec![Operand {
                    src: x,
                    layout: Layout::contiguous(&g.facts(x).shape),
                    access: AccessPlan::Alias,
                }],
                sched: ScheduleDomain::Point,
            }))
            .unwrap();
        let plan = Plan {
            extraction: Extraction::default(),
            launches: vec![Launch {
                root: k,
                members: smallvec::smallvec![k],
                bindings: vec![
                    BindingPlan { binding: 1, value: x, kind: BindKind::Read },
                    BindingPlan { binding: 2, value: k, kind: BindKind::Write },
                ],
                grid: [1, 1, 1],
                block: 32,
            }],
            buffers: Vec::new(),
            symbols: Vec::new(),
            hash: PlanHash(0),
            cost: Picoseconds(0),
        };
        let cx = LowerCtx {
            plan: &plan,
            launch: &plan.launches[0],
            graph: &g,
            symbols: &[],
        };
        let caps = crate::emit::testkit::caps(false, true);
        lower(&caps, g.node(k), theta, &cx)
    }

    fn fold_ir(carrier: Carrier, theta: SchedPoint) -> KernelIr {
        fold_ir_result(carrier, theta).expect("lowers")
    }

    /// [`fold_ir`], emitted as WGSL text.
    fn fold_wgsl(name: &'static str, carrier: Carrier, theta: SchedPoint) -> String {
        let caps = crate::emit::testkit::caps(false, true);
        let ir = fold_ir(carrier, theta);
        let emitted =
            crate::emit::emit_module(&ir, &caps, &crate::emit::testkit::no_plan()).expect("emits");
        let mut flags = naga::back::wgsl::WriterFlags::empty();
        flags.set(naga::back::wgsl::WriterFlags::EXPLICIT_TYPES, true);
        let text =
            naga::back::wgsl::write_string(&emitted.module, &emitted.info, flags).expect("wgsl");
        if let Ok(dir) = std::env::var("FUSOR2_WGSL_DUMP") {
            let _ = std::fs::write(format!("{dir}/lowered_{name}.wgsl"), &text);
        }
        text
    }

    fn fnv1a(bytes: &[u8]) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    /// The single-slot fast path, pinned: a plain `Fold{Add}` and a plain
    /// `Fold{Max}` emit the subgroup collective, not the N-ary `Stmt::Reduce`.
    /// The N-slot form is a new `Stmt` *beside* the collective, never in place
    /// of it. The failure message prints the shader, so a deliberate change is
    /// re-recorded by copying one line.
    #[test]
    fn single_slot_fold_wgsl_is_unchanged() {
        let cases: [(&'static str, BinOp, u64, usize); 4] = [
            ("add_subgroup", BinOp::Add, 0x1fa7_abfd_a91a_cd43, 850),
            ("max_subgroup", BinOp::Max, 0x41dc_217d_1c7a_9838, 889),
            ("add_wgtree", BinOp::Add, 0xe393_a7c7_3d68_a13f, 2041),
            ("max_wgtree", BinOp::Max, 0x3f7c_18bc_3e7d_7171, 2090),
        ];
        let texts: Vec<(&'static str, String)> = cases
            .iter()
            .map(|(name, op, _, _)| {
                let theta = if name.ends_with("wgtree") {
                    SchedPoint::Fold(FoldStrat::WgTree { lane_group: 32 })
                } else {
                    SchedPoint::Fold(FoldStrat::Subgroup)
                };
                (*name, fold_wgsl(name, binop(*op), theta))
            })
            .collect();
        for ((name, _, want_hash, want_len), (_, text)) in cases.iter().zip(&texts) {
            assert_eq!(
                (fnv1a(text.as_bytes()), text.len()),
                (*want_hash, *want_len),
                "{name} shader moved:\n{text}"
            );
        }
    }

    /// A two-slot carrier has no single `TileReduceOp`, so the fast path
    /// declines instead of reducing slot 0 and discarding the rest.
    #[test]
    fn a_multi_slot_carrier_is_refused_not_silently_truncated() {
        let pair = binop(BinOp::Max).tuple(&binop(BinOp::Add), &fusor2_ir::carrier::ArgRemap::identity(1));
        assert_eq!(pair.carrier.width(), 2);
        assert!(single_slot_reduce_op(&pair.carrier).is_err());

        let promoted = binop(BinOp::Add).promote(Dim::Const(64)).unwrap();
        assert!(single_slot_reduce_op(&promoted).is_err());
    }
    /// **A firing test.** A two-slot carrier lowers to exactly one N-ary
    /// `Stmt::Reduce` with `fast: None`, one scratch tile and one output local
    /// per lane, and one store per slot. The single-slot fast path must not be
    /// reachable from here: a `fast` operator would take the collective and
    /// silently drop slot 1.
    #[test]
    fn a_two_slot_carrier_lowers_to_the_n_ary_reduction() {
        use fusor2_ir::carrier::{ArgRemap, Carrier};
        use fusor2_ir::dtype::Dtype;
        let pair = binop(BinOp::Max)
            .tuple(&binop(BinOp::Add), &ArgRemap::identity(1))
            .carrier;
        assert_eq!(pair.width(), 2);
        assert!(fusor2_ir::ir::level2::fast_reduce_op(&pair).is_none());
        let _ = Carrier::binop_identity(BinOp::Add, Dtype::F32);

        let ir = fold_ir(pair, SchedPoint::Point);
        assert_eq!(ir.name, "kfold_carrier");
        let reduces: Vec<&Stmt> = ir
            .body
            .iter()
            .filter(|s| matches!(s, Stmt::Reduce { .. }))
            .collect();
        assert_eq!(reduces.len(), 1, "one reduction closes the whole carrier");
        let Stmt::Reduce {
            values,
            merge,
            fast,
            outs,
            scratch,
            ..
        } = reduces[0]
        else {
            unreachable!()
        };
        assert_eq!(fast, &None, "a two-slot merge has no hardware operator");
        assert_eq!(values.len(), 2);
        assert_eq!(merge.body.len(), 2);
        assert_eq!(outs.len(), 2);
        assert_eq!(scratch.len(), 2, "one scratch tile per accumulator lane");
        // Two slots, two outputs at the trailing carrier axis.
        let stores = ir
            .body
            .iter()
            .filter(|s| matches!(s, Stmt::Store { .. }))
            .count();
        assert_eq!(stores, 2);
        // And the whole kernel verifies, including the arity clause.
        fusor2_tile::verify_l2(&ir, &crate::emit::testkit::caps(false, true)).unwrap();
    }

    /// A Welford carrier is three lanes end to end, with a per-lane loop when
    /// the axis outruns the lane group.
    #[test]
    fn a_three_slot_carrier_carries_three_accumulators_through_the_loop() {
        let welford = fusor2_ir::carrier::oracle::welford(fusor2_ir::dtype::Dtype::F32);
        let ir = fold_ir(welford, SchedPoint::Fold(FoldStrat::WgTree { lane_group: 4 }));
        let loops = ir
            .body
            .iter()
            .filter_map(|s| match s {
                Stmt::Loop { accumulators, .. } => Some(accumulators.len()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            loops,
            vec![3],
            "a lane group of 4 over an extent of 8 needs the strided loop, with one              accumulator per lane"
        );
        let Some(Stmt::Reduce { values, .. }) =
            ir.body.iter().find(|s| matches!(s, Stmt::Reduce { .. }))
        else {
            panic!("expected a reduction");
        };
        assert_eq!(values.len(), 3);
        fusor2_tile::verify_l2(&ir, &crate::emit::testkit::caps(false, true)).unwrap();
    }

    /// A `Vector` slot without a promoted axis has nowhere to read its
    /// positions from, and says so. The positive half is the next test: the
    /// same carrier *with* `vec_axes` lowers.
    #[test]
    fn a_vector_slot_without_a_promoted_axis_is_refused_with_a_reason() {
        let promoted = binop(BinOp::Add).promote(Dim::Const(4)).unwrap();
        let err = fold_ir_result(promoted, SchedPoint::Point).unwrap_err();
        assert!(
            format!("{err}").contains("needs a promoted axis"),
            "got {err}"
        );
    }

    ///
    /// `space = [3, 4, 8]` with `vec_axes = [1]` and the reduced axis last is
    /// the shape PROMOTE mints — the free axis `3`, four accumulator positions,
    /// eight reduced elements. The carrier is one `Vector(4)` slot, so the
    /// reduction carries four lanes and the output is `3 * 4` values at the
    /// trailing carrier axis, not `3`.
    #[test]
    fn a_promoted_nest_carries_one_accumulator_per_position() {
        let promoted = binop(BinOp::Add).promote(Dim::Const(4)).unwrap();
        assert_eq!(promoted.lanes(), Some(4));
        let ir = fold_ir_in(
            promoted,
            SchedPoint::Fold(FoldStrat::WgTree { lane_group: 4 }),
            &[3, 4, 8],
            2,
            &[1],
        )
        .expect("a promoted nest lowers");
        let Some(Stmt::Reduce { values, outs, scratch, .. }) =
            ir.body.iter().find(|s| matches!(s, Stmt::Reduce { .. }))
        else {
            panic!("expected a reduction");
        };
        assert_eq!(values.len(), 4, "one partial per promoted position");
        assert_eq!(outs.len(), 4);
        assert_eq!(scratch.len(), 4, "one scratch tile per lane");
        let stores = ir
            .body
            .iter()
            .filter(|s| matches!(s, Stmt::Store { .. }))
            .count();
        assert_eq!(stores, 4, "one store per lane at the trailing carrier axis");
        fusor2_tile::verify_l2(&ir, &crate::emit::testkit::caps(false, true)).unwrap();
    }
    /// The N-ary tree, as emitted: **one scratch array per lane, and a barrier
    /// between every level.** A merge that read a slot its sibling had already
    /// overwritten would still emit; the barrier count and the array count are
    /// what say the tree is a tree.
    #[test]
    fn a_two_slot_fold_emits_one_scratch_array_per_lane_and_a_barrier_per_level() {
        use fusor2_ir::carrier::ArgRemap;
        let pair = binop(BinOp::Max)
            .tuple(&binop(BinOp::Add), &ArgRemap::identity(1))
            .carrier;
        let text = fold_wgsl(
            "two_slot_wgtree",
            pair,
            SchedPoint::Fold(FoldStrat::WgTree { lane_group: 8 }),
        );
        assert_eq!(
            text.matches("var<workgroup>").count(),
            2,
            "one scratch array per accumulator lane:\n{text}"
        );
        // Two seeding barriers plus one after each of log2(8) = 3 levels.
        assert_eq!(
            text.matches("workgroupBarrier()").count(),
            5,
            "a barrier between every tree level:\n{text}"
        );
        assert!(
            !text.contains("subgroupAdd") && !text.contains("subgroupMax"),
            "a multi-lane merge has no hardware collective:\n{text}"
        );
    }
}
