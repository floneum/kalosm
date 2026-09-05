//! `Map` and `Fold`: the elementwise and reduction loop nests.
//! Both read their geometry off `theta`.

use fusor_ir::Result;
use fusor_ir::carrier::{Carrier, SlotTy};
use fusor_ir::dtype::NumericContract;
use fusor_ir::dtype::Splat;
use fusor_ir::error::Error;
use fusor_ir::ir::kernel::{
    Accumulator, Addr, ElementType, KernelIr, ReduceKind, ScalarElement, Stmt, TileBinaryOp,
    TileCompareOp, TileExpr, TileReduceOp,
};
use fusor_ir::ir::launch::{FoldStrat, Launch, MapTiling, SchedPoint};
use fusor_ir::scalar::BinOp;

use crate::lower::{Ctx, DimBinding, grid_for, scalar_element};
use fusor_tile::domains::emitted_block;

/// Lower a `Map` at a [`MapTiling`].
///
/// `dim: None` is the untiled body: one output per lane. Otherwise each lane
/// computes `tm` outputs along `dim` and every operand that does *not* vary
/// with `dim` is hoisted into a `Local` before the loop, so it is read once
/// per lane instead of `tm` times.
pub(crate) fn lower_kmap(mut ctx: Ctx<'_>, op: &Launch, theta: SchedPoint) -> Result<KernelIr> {
    let Launch::Map {
        space, body, ops, ..
    } = op
    else {
        return Err(Error::Plan("lower_kmap on a non-Map node".into()));
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
                "Map needs SchedPoint::Map, got {other:?}"
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
            // store coalescing, which is why the fold domain never offers it.
            if axis + 1 == space.rank() {
                return Err(Error::Plan(
                    "map tiling on the innermost axis destroys store coalescing".into(),
                ));
            }

            let base = ctx.global_index(block, grid);
            let stride = inner_extent_expr(&mut ctx, op, axis)?;
            let tm_e = ctx.b.u32(tm);
            let step = ctx.b.mul(stride.clone(), tm_e);
            let tile_base = {
                let outer = ctx.b.binary(
                    TileBinaryOp::Div,
                    base.clone(),
                    stride.clone(),
                    NumericContract::RELAXED,
                );
                let inner = ctx.b.binary(
                    TileBinaryOp::Rem,
                    base,
                    stride.clone(),
                    NumericContract::RELAXED,
                );
                let scaled = ctx.b.mul(outer, step);
                ctx.b.add(scaled, inner)
            };

            // Hoist every operand whose access does not vary along `dim`.
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
                    let t_e = ctx.b.u32(t);
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
fn operand_is_invariant(operand: &fusor_ir::ir::launch::Operand, axis: usize) -> bool {
    let layout = &operand.layout;
    if axis >= layout.rank() {
        return true;
    }
    layout.strides()[axis].known_eq(fusor_ir::shape::Dim::Const(0))
        || layout.shape()[axis].known_eq(fusor_ir::shape::Dim::Const(1))
}

fn space_extent_expr(
    ctx: &mut Ctx<'_>,
    space: &fusor_ir::ir::launch::IndexSpace,
) -> Result<TileExpr> {
    let mut acc = ctx.b.u32(1);
    for dim in &space.dims {
        let e = ctx.dim_expr(*dim)?;
        acc = ctx.b.mul(acc, e);
    }
    Ok(acc)
}

/// Product of the extents strictly inside `axis` — the element distance one
/// step along `axis` covers in the flattened index space.
fn inner_extent_expr(ctx: &mut Ctx<'_>, op: &Launch, axis: usize) -> Result<TileExpr> {
    let Launch::Map { space, .. } = op else {
        return Err(Error::Plan("inner_extent_expr on a non-Map node".into()));
    };
    let mut acc = ctx.b.u32(1);
    for dim in space.dims.iter().skip(axis + 1) {
        let e = ctx.dim_expr(*dim)?;
        acc = ctx.b.mul(acc, e);
    }
    Ok(acc)
}

fn tiled_grid(
    space: &fusor_ir::ir::launch::IndexSpace,
    block: u32,
    tm: u32,
    binding: &DimBinding,
    limits: &fusor_ir::device::Limits,
) -> Result<[u32; 3]> {
    let full = grid_for(space, block.saturating_mul(tm.max(1)), binding, limits)?;
    Ok(full)
}

/// Lower a `Fold` at a [`FoldStrat`].
///
/// Three bodies, one carrier shape each:
/// * [`FoldStrat::Subgroup`] — a subgroup collective, no scratch, no barrier.
/// * [`FoldStrat::WgTree`] — a shared-memory tree over one scratch tile.
/// * [`FoldStrat::LoopThenTree`] — a per-lane accumulate loop, then the tree.
///
/// The carrier's `lift` runs before the merge and `post` after it, so a
/// softmax's `exp` and a mean's divide fuse into the same launch.
///
/// One scalar slot merged by a hardware operator takes the collective path
/// below; anything wider goes to [`lower_kfold_carrier`], which carries one
/// accumulator per lane and closes with `Stmt::Reduce`'s N-ary merge.
pub(crate) fn lower_kfold(mut ctx: Ctx<'_>, op: &Launch, theta: SchedPoint) -> Result<KernelIr> {
    let Launch::Fold {
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
        return Err(Error::Plan("lower_kfold on a non-Fold node".into()));
    };
    if !vec_axes.is_empty() || fusor_ir::ir::kernel::fast_reduce_op(carrier).is_none() {
        return lower_kfold_carrier(ctx, op, theta);
    }
    let reduce_op = single_slot_reduce_op(carrier)?;
    let pre = &carrier.lift[0];
    let post = &post[0];
    // `SchedPoint::Point` means the node has no schedule parameters: default
    // to the subgroup collective where the device has subgroups, the
    // shared-memory tree where it does not.
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
        let mut acc_e = ctx.b.u32(1);
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
    let lg_e = ctx.b.u32(lane_group);
    let row = ctx.b.binary(
        TileBinaryOp::Div,
        group.clone(),
        lg_e.clone(),
        NumericContract::RELAXED,
    );
    let lane = ctx.b.binary(
        TileBinaryOp::Rem,
        group,
        lg_e.clone(),
        NumericContract::RELAXED,
    );
    let row_live = ctx.b.compare(TileCompareOp::Lt, row.clone(), rows);

    // Element index of `(row, k)` in the flattened space.
    let outer = ctx.b.binary(
        TileBinaryOp::Div,
        row.clone(),
        inner.clone(),
        NumericContract::RELAXED,
    );
    let within = ctx.b.binary(
        TileBinaryOp::Rem,
        row.clone(),
        inner.clone(),
        NumericContract::RELAXED,
    );
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

    // A lane past the reduced extent contributes the combine's identity; the
    // collective spans the whole lane group whatever the extent is, so an
    // unguarded lane would sum the next row into this one.
    let guard = |ctx: &mut Ctx<'_>, k: &TileExpr, v: TileExpr| -> TileExpr {
        let in_range = ctx
            .b
            .compare(TileCompareOp::Lt, k.clone(), axis_extent.clone());
        let ident = identity_expr(ctx, carrier.identity[0], acc_elem);
        ctx.b.select(in_range, v, ident)
    };

    // One pass of the lane group covers `lane_group` elements of the axis;
    // anything longer needs the per-lane strided loop first, whichever
    // collective closes it.
    let one_pass = space.dims[axis]
        .as_const()
        .is_some_and(|k| k <= u64::from(lane_group.max(1)));

    let lane_value = if one_pass {
        let v = read_k(&mut ctx, lane.clone())?;
        let v = ctx.b.cast(v, ElementType::Scalar(acc_elem));
        guard(&mut ctx, &lane, v)
    } else {
        // Per-lane loop accumulate. The loop's accumulator is SSA-carried,
        // never reloaded per iteration. The trip count comes from the runtime
        // extent, so one formula covers a symbolic extent and both collective
        // strategies.
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
            let lg_minus_1 = ctx.b.u32(lane_group.max(1) - 1);
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
        // strategy; the two must agree.
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

    let value = ctx.eval_scalar(post, &reduced, std::slice::from_ref(&row))?;
    let value = ctx.b.cast(value, out_elem);
    let lane_zero = {
        let z = ctx.b.u32(0);
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

/// Lower a `Fold` whose carrier is **wider than one hardware operator**.
///
/// One accumulator per carrier lane, seeded from that lane's own identity,
/// absorbed with the carrier's own `merge`, and closed by `Stmt::Reduce`'s N-ary
/// tree. The output carries `carrier.lanes()` values per row at the trailing
/// carrier axis `infer_launch` appends.
///
/// There is no subgroup collective for a multi-lane merge, so this always closes
/// with the workgroup tree; a `Subgroup` point is honoured as a tree at the
/// subgroup width.
fn lower_kfold_carrier(mut ctx: Ctx<'_>, op: &Launch, theta: SchedPoint) -> Result<KernelIr> {
    let Launch::Fold {
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
        return Err(Error::Plan("lower_kfold_carrier on a non-Fold node".into()));
    };
    let merges = carrier.merge_lanes().ok_or_else(|| {
        Error::Plan("this carrier's merge does not expand to one expression per lane".into())
    })?;
    let lanes = merges.len();
    let posts = carrier.expand_lanes(post).ok_or_else(|| {
        Error::Plan(format!(
            "a {}-slot carrier carries {} post expressions, or a slot's post reads \
             a sibling of a different width",
            carrier.width(),
            post.len()
        ))
    })?;

    let axis = *axis as usize;
    if axis >= space.rank() {
        return Err(Error::Plan(format!(
            "fold axis {axis} is outside a rank-{} space",
            space.rank()
        )));
    }
    // A promoted nest: the accumulator-resident axes are a contiguous block
    // immediately before the reduced axis, so `space` is `free.. ++ vec.. ++
    // [reduced]` and one output row spans `vec_extent * axis_extent`
    // consecutive elements. `verify_launch` establishes the block property.
    let vec_extent: u64 = vec_axes
        .iter()
        .map(|i| space.dims[*i as usize].as_const())
        .try_fold(1u64, |a, d| Some(a * d?))
        .ok_or_else(|| Error::Plan("a promoted axis has a symbolic extent".into()))?;
    if !vec_axes.is_empty() && axis + 1 != space.rank() {
        return Err(Error::Plan(
            "a promoted Fold whose reduced axis is not last is not lowered".into(),
        ));
    }
    if vec_axes.is_empty() && carrier.slots.iter().any(|s| *s != SlotTy::Scalar) {
        return Err(Error::Plan(
            "a Vector carrier slot needs a promoted axis to read its positions from".into(),
        ));
    }
    // Iteration axis `j` is space axis `iter_axes[j]`. Every `ScalarExpr` on
    // this node is written against the iteration space, so an `IndexOf` has to
    // be resolved through this map and not against `space` directly.
    let iter_axes: Vec<usize> = (0..space.rank())
        .filter(|i| !vec_axes.contains(&(*i as u32)))
        .collect();
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

    // Output rows are `space` minus the reduced axis and every promoted axis:
    // a promoted extent lives in the carrier's lanes, not in the write map.
    let mut row_space = space.clone();
    row_space.dims.remove(axis);
    for i in vec_axes.iter().rev() {
        row_space.dims.remove(*i as usize);
    }
    let rows = space_extent_expr(&mut ctx, &row_space)?;
    let axis_extent = ctx.dim_expr(space.dims[axis])?;
    let inner: TileExpr = {
        let mut acc_e = ctx.b.u32(1);
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
    let lg_e = ctx.b.u32(lane_group);
    let row = ctx.b.binary(
        TileBinaryOp::Div,
        group.clone(),
        lg_e.clone(),
        NumericContract::RELAXED,
    );
    let lane = ctx.b.binary(
        TileBinaryOp::Rem,
        group,
        lg_e.clone(),
        NumericContract::RELAXED,
    );
    let row_live = ctx.b.compare(TileCompareOp::Lt, row.clone(), rows);

    let outer = ctx.b.binary(
        TileBinaryOp::Div,
        row.clone(),
        inner.clone(),
        NumericContract::RELAXED,
    );
    let within = ctx.b.binary(
        TileBinaryOp::Rem,
        row.clone(),
        inner.clone(),
        NumericContract::RELAXED,
    );
    // One output row spans every promoted position of every reduced element,
    // so its stride carries `vec_extent`.
    let pos_stride = ctx.b.mul(inner.clone(), axis_extent.clone());
    let row_stride = {
        let ve = ctx.b.u32(vec_extent as u32);
        ctx.b.mul(pos_stride.clone(), ve)
    };
    let row_base = {
        let hi = ctx.b.mul(outer, row_stride);
        ctx.b.add(hi, within)
    };

    // One lifted value per lane at element `k`, each guarded to its own
    // identity outside the reduced extent: a lane past the extent must
    // contribute nothing to every slot (Welford's constant `1` lift would
    // count a padding lane under a shared identity).
    //
    // A `Vector` slot is `vec_extent` registers, and lane `(slot, p)` reads
    // every operand at promoted position `p`; an operand invariant in the
    // promoted axes is hash-consed back to one read reused across positions.
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
                    let pe = ctx.b.u32(p as u32);
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
        let lane_slots = carrier
            .lane_slots()
            .ok_or_else(|| Error::Plan("this carrier has a symbolic Vector extent".into()))?;
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
        let lane_ident = carrier
            .identity_lanes()
            .ok_or_else(|| Error::Plan("this carrier has a symbolic Vector extent".into()))?;
        for &ident in lane_ident.iter().take(lanes) {
            let local = ctx.b.local(acc_ty);
            let init = identity_expr(&mut ctx, ident, acc_elem);
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
            let lg_minus_1 = ctx.b.u32(lane_group - 1);
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
        accs.iter()
            .map(|a| ctx.b.load_local(a.local.clone()))
            .collect()
    };

    // The cross-lane close: one scratch tile per lane, one merge per lane.
    // Skipped at a one-lane group: that invocation already reduced the whole
    // axis for its own row and there is no partner to merge with.
    // `fold_scratch_bytes` reports 0 here; the two must agree.
    let reduced: Vec<TileExpr> = if lane_group <= 1 {
        partials
    } else {
        let scratch: smallvec::SmallVec<[fusor_ir::ir::kernel::Tile; 4]> = (0..lanes)
            .map(|_| ctx.b.tile("fold_scratch", acc_ty, &[block]))
            .collect();
        let lhs: smallvec::SmallVec<[fusor_ir::ir::kernel::Local; 4]> =
            (0..lanes).map(|_| ctx.b.local(acc_ty)).collect();
        let rhs: smallvec::SmallVec<[fusor_ir::ir::kernel::Local; 4]> =
            (0..lanes).map(|_| ctx.b.local(acc_ty)).collect();
        let outs: smallvec::SmallVec<[fusor_ir::ir::kernel::Local; 4]> =
            (0..lanes).map(|_| ctx.b.local(acc_ty)).collect();
        let mut merge_args: Vec<TileExpr> = Vec::with_capacity(2 * lanes);
        for l in lhs.iter().chain(rhs.iter()) {
            merge_args.push(ctx.b.load_local(l.clone()));
        }
        let mut body: smallvec::SmallVec<[TileExpr; 4]> = smallvec::SmallVec::new();
        for merge in merges.iter().take(lanes) {
            body.push(ctx.eval_scalar(merge, &merge_args, &[])?);
        }
        stmts.push(Stmt::Reduce {
            kind: Box::new(ReduceKind::Workgroup {
                scratch: scratch[0].clone(),
                group_size: lane_group,
            }),
            values: partials.into_iter().collect(),
            merge: Box::new(fusor_ir::ir::kernel::MergeBody { lhs, rhs, body }),
            fast: None,
            outs: outs.clone(),
            scratch,
        });
        // One output per slot, at the trailing carrier axis.
        outs.iter().map(|l| ctx.b.load_local(l.clone())).collect()
    };
    let lane_zero = {
        let z = ctx.b.u32(0);
        ctx.b.compare(TileCompareOp::Eq, lane, z)
    };
    let mask = ctx.b.and(row_live, lane_zero);
    let lanes_e = ctx.b.u32(lanes as u32);
    let base = ctx.b.mul(row.clone(), lanes_e);
    for (slot, post) in posts.iter().enumerate().take(lanes) {
        let value = ctx.eval_scalar(post, &reduced, std::slice::from_ref(&row))?;
        let value = ctx.b.cast(value, out_elem);
        let off = ctx.b.u32(slot as u32);
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

/// The hardware collective this carrier reduces with, or an `Err`. Only one
/// scalar slot whose merge is a binop maps onto `TileReduceOp`; everything
/// wider needs the N-lane `Stmt::Reduce`.
fn single_slot_reduce_op(c: &Carrier) -> Result<TileReduceOp> {
    if c.width() != 1 || c.slots[0] != SlotTy::Scalar {
        return Err(Error::Plan(format!(
            "a {}-slot carrier needs the N-lane collective; the GPU emitter \
             only lowers a single scalar slot",
            c.width()
        )));
    }
    match c.kind() {
        Some(BinOp::Add) => Ok(TileReduceOp::Sum),
        Some(BinOp::Mul) => Ok(TileReduceOp::Product),
        Some(BinOp::Max) => Ok(TileReduceOp::Max),
        Some(BinOp::Min) => Ok(TileReduceOp::Min),
        other => Err(Error::Plan(format!(
            "carrier merge {other:?} has no hardware collective; the generic \
             merge path is not built yet"
        ))),
    }
}

/// A carrier identity as a tile literal. The infinities go through the
/// builder's own spellings so the emitted text is unchanged.
fn identity_expr(ctx: &mut Ctx<'_>, s: Splat, elem: ScalarElement) -> TileExpr {
    let f = match s {
        Splat::F32(v) => v,
        Splat::F16(b) => half::f16::from_bits(b).to_f32(),
        Splat::BF16(b) => half::bf16::from_bits(b).to_f32(),
        Splat::U32(v) => {
            return if v == 0 {
                ctx.b.zero(elem)
            } else if v == u32::MAX {
                ctx.b.pos_inf(elem)
            } else {
                ctx.b.u32(v)
            };
        }
        Splat::I32(v) => {
            return if v == 0 {
                ctx.b.zero(elem)
            } else if v == i32::MIN {
                ctx.b.neg_inf(elem)
            } else if v == i32::MAX {
                ctx.b.pos_inf(elem)
            } else {
                ctx.b.i32(v)
            };
        }
    };
    if f == f32::NEG_INFINITY {
        ctx.b.neg_inf(elem)
    } else if f == f32::INFINITY {
        ctx.b.pos_inf(elem)
    } else if f == 0.0 {
        ctx.b.zero(elem)
    } else if f == 1.0 {
        match elem {
            ScalarElement::U32 => ctx.b.u32(1),
            ScalarElement::I32 => ctx.b.i32(1),
            _ => ctx.b.f32(1.0),
        }
    } else {
        ctx.b.f32(f)
    }
}
