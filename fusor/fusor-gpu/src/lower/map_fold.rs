//! `Map` and `Fold`: the elementwise and reduction loop nests.
//! Both read their geometry off `theta`.

use fusor_ir::Result;
use fusor_ir::error::Error;
use fusor_ir::ir::kernel::{
    Accumulator, Addr, ElementType, KernelIr, ReduceKind, ScalarElement, Stmt, TileExpr,
};
use fusor_ir::ir::launch::{
    AccessPlan, FoldStrat, IndexSpace, Launch, MapTiling, Operand, SchedPoint,
};
use fusor_ir::scalar::ScalarExpr;
use fusor_ir::shape::Dim;
use fusor_tile::build::FoldLanes;

use crate::lower::{Ctx, grid_for, scalar_element};
use fusor_tile::domains::emitted_block;

/// Lower a `Map` at a [`MapTiling`].
///
/// `dim: None` is the untiled body: one output per lane. Otherwise each lane
/// computes `tm` outputs along `dim` and every operand that does *not* vary
/// with `dim` is hoisted into a `Local` before the loop, so it is read once
/// per lane instead of `tm` times.
pub(crate) fn lower_kmap(ctx: Ctx<'_>, op: &Launch, theta: SchedPoint) -> Result<KernelIr> {
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

    let b = &ctx.b;
    let out = ctx.linear_view(ctx.output()?)?;
    let block = ctx.block(emitted_block(1, ctx.caps));
    let total = ctx.extent_product(&space.dims)?;
    let space_total = space.iterations().unwrap_or(0);
    let tm = tiling.tm.max(1);
    let grid = grid_for(
        space,
        block.saturating_mul(tm),
        &ctx.binding,
        &ctx.caps.limits,
    )?;
    let store = |at: TileExpr, args: Vec<TileExpr>| -> Result<Stmt> {
        let coords = ctx.coords_from_linear(at.clone(), space)?;
        let value = b.cast(ctx.eval_scalar(body, &args, &coords)?, out.buffer.element);
        Ok(Stmt::Store {
            dst: out.clone(),
            addr: Addr::Linear(at.clone()),
            value,
            mask: b.lt(at, total.clone()),
        })
    };

    let mut stmts: Vec<Stmt> = Vec::new();
    match tiling.dim {
        None => {
            let index = ctx.global_index(block);
            let args = ops
                .iter()
                .map(|o| ctx.load_mapped(o, index.clone(), space_total))
                .collect::<Result<_>>()?;
            stmts.push(store(index, args)?);
        }
        Some(dim) => {
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
            let stride = ctx.extent_product(&space.dims[axis + 1..])?;
            let tile_base = b.tile_origin(ctx.global_index(block), stride.clone(), tm);

            // Hoist every operand whose access does not vary along `dim`.
            let mut hoisted: Vec<Option<TileExpr>> = Vec::with_capacity(ops.len());
            for operand in ops {
                hoisted.push(if operand_is_invariant(operand, axis) {
                    let v = ctx.load_mapped(operand, tile_base.clone(), space_total)?;
                    let local = b.local(v.element());
                    stmts.push(Stmt::StoreLocal {
                        dst: local.clone(),
                        value: v,
                    });
                    Some(b.load_local(local))
                } else {
                    None
                });
            }
            for t in 0..tm {
                let off = b.at(tile_base.clone(), stride.clone(), b.u32(t));
                let args = ops
                    .iter()
                    .zip(&hoisted)
                    .map(|(operand, cached)| match cached {
                        Some(v) => Ok(v.clone()),
                        None => ctx.load_mapped(operand, off.clone(), space_total),
                    })
                    .collect::<Result<_>>()?;
                stmts.push(store(off, args)?);
            }
        }
    }
    Ok(ctx.finish("kmap", grid, block, stmts))
}

/// An operand is loop-invariant along `axis` when its layout gives that axis
/// stride 0 or extent 1 — `layout_index` drops both, so the address does not
/// move as the tiled coordinate advances.
fn operand_is_invariant(operand: &Operand, axis: usize) -> bool {
    let layout = &operand.layout;
    axis >= layout.rank()
        || layout.strides()[axis].known_eq(Dim::Const(0))
        || layout.shape()[axis].known_eq(Dim::Const(1))
}

/// Lower every carrier through one row/axis loop nest. A single scalar
/// hardware operator closes with a collective; wider carriers use their
/// expanded merge expressions in the workgroup tree.
pub(crate) fn lower_kfold(ctx: Ctx<'_>, op: &Launch, theta: SchedPoint) -> Result<KernelIr> {
    let (op, producer) = match op {
        Launch::StreamFold {
            producer,
            fold,
            operand,
            ..
        } => (fold.as_ref(), Some((producer.as_ref(), *operand as usize))),
        _ => (op, None),
    };
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
    let fast = vec_axes
        .is_empty()
        .then(|| fusor_ir::ir::kernel::fast_reduce_op(carrier))
        .flatten();
    let axis = *axis as usize;
    let lanes = FoldLanes::of(carrier, post, space.rank(), axis, vec_axes, Error::Plan)?;
    // A promoted nest: the accumulator-resident axes are a contiguous block
    // immediately before the reduced axis, so one output row spans
    // `vec_extent * axis_extent` consecutive elements.
    let vec_extent: u64 = vec_axes
        .iter()
        .map(|i| space.dims[*i as usize].as_const())
        .try_fold(1u64, |a, d| Some(a * d?))
        .ok_or_else(|| Error::Plan("a promoted axis has a symbolic extent".into()))?;
    let space_total = space.iterations().unwrap_or(0);
    let acc_elem = scalar_element(*acc);
    let acc_ty = ElementType::Scalar(acc_elem);
    let schedule = op.fold_schedule(Some(theta), ctx.caps).expect("GPU Fold");
    let strat = schedule.strategy;
    let lane_group = strat.lane_group(ctx.caps.subgroup_width()).max(1);
    let block = ctx.block(schedule.block);
    let b = &ctx.b;

    // Output rows are `space` minus the reduced axis and every promoted axis:
    // a promoted extent lives in the carrier's lanes, not in the write map.
    let mut row_space = space.clone();
    row_space.dims.remove(axis);
    for i in vec_axes.iter().rev() {
        row_space.dims.remove(*i as usize);
    }
    let rows = ctx.extent_product(&row_space.dims)?;
    let axis_extent = ctx.dim_expr(space.dims[axis])?;
    let inner = ctx.extent_product(&space.dims[axis + 1..])?;
    let out = ctx.linear_view(ctx.output()?)?;
    let mut stmts: Vec<Stmt> = Vec::new();

    let grid = grid_for(
        &row_space,
        block / lane_group,
        &ctx.binding,
        &ctx.caps.limits,
    )?;
    let lg_e = b.u32(lane_group);
    let (row, lane) = b.divrem(ctx.global_index(block), lg_e.clone());
    let row_live = b.lt(row.clone(), rows);
    // One output row spans every promoted position of every reduced element,
    // so its stride carries `vec_extent`.
    let pos_stride = b.mul(inner.clone(), axis_extent.clone());
    let row_stride = match fast {
        Some(_) => pos_stride.clone(),
        None => b.mul(pos_stride.clone(), b.u32(vec_extent as u32)),
    };
    let row_base = b.row_base(row.clone(), inner.clone(), row_stride);

    // One lifted value per lane at element `k`, each guarded to its own
    // identity outside the reduced extent: a lane past the extent must
    // contribute nothing to every slot (Welford's constant `1` lift would
    // count a padding lane under a shared identity).
    //
    // A `Vector` slot is `vec_extent` registers, and lane `(slot, p)` reads
    // every operand at promoted position `p`; an operand invariant in the
    // promoted axes is hash-consed back to one read reused across positions.
    let lift_at = |k: &TileExpr, body: &mut Vec<Stmt>| -> Result<Vec<TileExpr>> {
        let in_range = b.lt(k.clone(), axis_extent.clone());
        let mut per_pos: Vec<(Vec<TileExpr>, Vec<TileExpr>)> =
            Vec::with_capacity(vec_extent as usize);
        let mut produced: rustc_hash::FxHashMap<TileExpr, TileExpr> = Default::default();
        let first = b.at(row_base.clone(), k.clone(), inner.clone());
        for p in 0..vec_extent {
            let idx = match p {
                0 => first.clone(),
                _ => b.at(first.clone(), pos_stride.clone(), b.u32(p as u32)),
            };
            let mut args = Vec::with_capacity(ops.len());
            for (slot, operand) in ops.iter().enumerate() {
                args.push(match producer {
                    Some((source, source_slot)) if slot == source_slot => {
                        let invariant = !matches!(operand.access, AccessPlan::Unflatten(_))
                            && operand.layout.rank() == space.rank()
                            && vec_axes.iter().all(|axis| {
                                operand.layout.strides()[*axis as usize].known_eq(Dim::Const(0))
                            });
                        let at = ctx.operand_address(
                            operand,
                            if invariant { &first } else { &idx }.clone(),
                            space_total,
                        )?;
                        match produced.get(&at) {
                            Some(value) => value.clone(),
                            None => {
                                let value = fold_element(&ctx, source, at.clone(), body)?;
                                produced.insert(at, value.clone());
                                value
                            }
                        }
                    }
                    _ => ctx.load_mapped(operand, idx.clone(), space_total)?,
                });
            }
            // `IndexOf` on this node names an ITERATION axis; resolve it
            // through `iter_axes` rather than against `space` directly.
            let full = ctx.coords_from_linear(idx, space)?;
            let coords = lanes.iter_axes.iter().map(|i| full[*i].clone()).collect();
            per_pos.push((args, coords));
        }
        lanes
            .slots
            .iter()
            .map(|&(slot, p)| {
                let (args, coords) = &per_pos[p as usize];
                let v = b.cast(ctx.eval_scalar(&carrier.lift[slot], args, coords)?, acc_ty);
                Ok(b.select(
                    in_range.clone(),
                    v,
                    b.identity(carrier.identity[slot], acc_elem),
                ))
            })
            .collect()
    };

    let one_pass = space.dims[axis]
        .as_const()
        .is_some_and(|k| k <= u64::from(lane_group));
    let partials: Vec<TileExpr> = if one_pass {
        lift_at(&lane, &mut stmts)?
    } else {
        // The per-lane strided loop, carrying `lanes` SSA accumulators seeded
        // from the carrier's identities and absorbed with its own `merge`.
        let index = b.local(ScalarElement::U32.element());
        let k = b.add(
            b.mul(b.load_local(index.clone()), lg_e.clone()),
            lane.clone(),
        );
        let mut accs: Vec<Accumulator> = Vec::with_capacity(lanes.lanes());
        for &ident in lanes.identities.iter().take(lanes.lanes()) {
            let local = b.local(acc_ty);
            let read = b.load_local(local.clone());
            accs.push(Accumulator {
                local,
                init: b.identity(ident, acc_elem),
                update: read,
            });
        }
        let mut loop_body = Vec::new();
        let mut args: Vec<TileExpr> = accs.iter().map(|a| a.update.clone()).collect();
        args.extend(lift_at(&k, &mut loop_body)?);
        for (slot, acc) in accs.iter_mut().enumerate() {
            acc.update = match fast {
                Some(op) => b.binary(
                    op.binary(),
                    args[0].clone(),
                    args[1].clone(),
                    fusor_ir::dtype::NumericContract::RELAXED,
                ),
                None => ctx.eval_scalar(&lanes.merges[slot], &args, &[])?,
            };
        }
        let count = b.div(b.add(axis_extent.clone(), b.u32(lane_group - 1)), lg_e);
        let reads = accs.iter().map(|a| b.load_local(a.local.clone())).collect();
        stmts.push(Stmt::Loop {
            count: Some(count),
            index: Some(index),
            accumulators: accs,
            body: loop_body,
        });
        reads
    };

    // The cross-lane close: one scratch tile per lane, one merge per lane.
    // Skipped at a one-lane group: that invocation already reduced the whole
    // axis for its own row and there is no partner to merge with.
    // `fold_scratch_bytes` reports 0 here; the two must agree.
    let reduced: Vec<TileExpr> = match fast {
        Some(op) => {
            let value = partials[0].clone();
            vec![match strat {
                FoldStrat::Subgroup => b.reduce(op, ReduceKind::Subgroup, value),
                _ if lane_group <= 1 => value,
                _ => b.reduce(
                    op,
                    ReduceKind::Workgroup {
                        scratch: b.tile("fold_scratch", acc_ty, &[block]),
                        group_size: lane_group,
                    },
                    value,
                ),
            }]
        }
        None if lane_group <= 1 => partials,
        None => {
            let scratch = (0..lanes.lanes())
                .map(|_| b.tile("fold_scratch", acc_ty, &[block]))
                .collect();
            let (reduce, outs) =
                b.merge_tree(scratch, lane_group, partials, None, acc_ty, |args| {
                    merge_body(&ctx, &lanes.merges, args, None)
                })?;
            stmts.push(reduce);
            outs
        }
    };
    let mask = b.and(row_live, b.eq(lane, b.u32(0)));
    let base = b.mul(row.clone(), b.u32(lanes.lanes() as u32));
    for (slot, post) in lanes.posts.iter().enumerate() {
        let value = ctx.eval_scalar(post, &reduced, std::slice::from_ref(&row))?;
        stmts.push(Stmt::Store {
            dst: out.clone(),
            addr: Addr::Linear(match fast {
                Some(_) => row.clone(),
                None => b.add(base.clone(), b.u32(slot as u32)),
            }),
            value: b.cast(value, out.buffer.element),
            mask: mask.clone(),
        });
    }

    let name = match (producer, fast) {
        (Some(_), _) => "kstream_fold",
        (None, Some(_)) => "kfold",
        (None, None) => "kfold_carrier",
    };
    Ok(ctx.finish(name, grid, block, stmts))
}

/// A carrier's per-lane merge body over `lhs ++ rhs` reads, each cast to
/// `cast` when given.
pub(crate) fn merge_body(
    ctx: &Ctx<'_>,
    merges: &[ScalarExpr],
    args: &[TileExpr],
    cast: Option<ElementType>,
) -> Result<smallvec::SmallVec<[TileExpr; 4]>> {
    merges
        .iter()
        .map(|merge| {
            let v = ctx.eval_scalar(merge, args, &[])?;
            Ok(match cast {
                Some(ty) => ctx.b.cast(v, ty),
                None => v,
            })
        })
        .collect()
}

fn fold_element(
    ctx: &Ctx<'_>,
    source: &Launch,
    row: TileExpr,
    body: &mut Vec<Stmt>,
) -> Result<TileExpr> {
    let Launch::Fold {
        space,
        axis,
        carrier,
        acc,
        post,
        ops,
        ..
    } = source
    else {
        return Err(Error::Plan("a streamed producer must be a Fold".into()));
    };
    let b = &ctx.b;
    let axis = *axis as usize;
    let element = scalar_element(*acc);
    let ty = ElementType::Scalar(element);
    let extent = ctx.dim_expr(space.dims[axis])?;
    let inner = ctx.extent_product(&space.dims[axis + 1..])?;
    let base = b.row_base(
        row.clone(),
        inner.clone(),
        b.mul(inner.clone(), extent.clone()),
    );
    let index = b.local(ScalarElement::U32.element());
    let k = b.load_local(index.clone());
    let at = b.at(base, k.clone(), inner);
    let mut output_space: IndexSpace = space.clone();
    output_space.dims.remove(axis);
    let mut coords = ctx.coords_from_linear(row.clone(), &output_space)?;
    coords.insert(axis, k);
    let args = ops
        .iter()
        .map(|o| {
            if !matches!(o.access, AccessPlan::Unflatten(_))
                && o.layout.shape() == space.dims.as_slice()
            {
                let mut address = ctx.dim_expr(o.layout.offset())?;
                for (coordinate, stride) in coords.iter().zip(o.layout.strides()) {
                    if !stride.known_eq(Dim::Const(0)) {
                        address = b.at(address, coordinate.clone(), ctx.dim_expr(*stride)?);
                    }
                }
                ctx.load_operand(o, address)
            } else {
                ctx.load_mapped(o, at.clone(), space.iterations().unwrap_or(0))
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let lifted = b.cast(ctx.eval_scalar(&carrier.lift[0], &args, &coords)?, ty);
    let accumulator = b.local(ty);
    let value = b.load_local(accumulator.clone());
    let update = ctx.eval_scalar(&carrier.merge[0], &[value.clone(), lifted], &[])?;
    body.push(Stmt::Loop {
        count: Some(extent),
        index: Some(index),
        accumulators: vec![Accumulator {
            local: accumulator,
            init: b.identity(carrier.identity[0], element),
            update,
        }],
        body: Vec::new(),
    });
    let value = ctx.eval_scalar(&post[0], &[value], &[row])?;
    Ok(b.cast(value, ty))
}
