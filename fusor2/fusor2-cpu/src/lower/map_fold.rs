//! `KMap` and `KFold` as SIMD loop nests with a register accumulator tile.
//!
//! `KMap` reads its register-reuse tiling off `SchedPoint::Map(MapTiling)`
//! (`dim`, `tm`, `vector`); `KFold` reads its strategy off
//! `SchedPoint::Fold(FoldStrat)` and lowers all three: `Subgroup` to a
//! horizontal reduce, `WgTree` to a tree over a scratch tile, and
//! `LoopThenTree` to per-lane loop accumulation followed by that tree.
//!
//! Owned by W10.

use fusor2_ir::device::Caps;
use fusor2_ir::error::Error;
use fusor2_ir::carrier::{Carrier, SlotTy};
use fusor2_ir::ir::level1::{FoldStrat, L1, SchedPoint};
use fusor2_ir::ir::level2::{
    Addr, Builtin, ElementType, KernelIr, LocalDecl, MemoryLevel, ReduceKind, ScalarElement,
    StorageView, Stmt, TileDecl, TileExpr, TileExprKind, TileLayout, TileReduceOp, WorkgroupAxis,
};
use fusor2_ir::ir::{Node, Op};
use fusor2_ir::scalar::{BinOp, CmpOp};
use fusor2_ir::target::LowerCtx;
use fusor2_ir::Result;
use std::sync::Arc;

use super::{
    bin, cmp, const_extents, coords_of, default_block, global_lane, grid_for, lit_f32, lit_u32,
    u32_ty, Binds, Translate,
};

/// The workgroup width a fold allocates its scratch over.
///
/// The resolved point decides it: `FoldStrat::Subgroup` runs one SIMD group
/// wide, the two tree strategies run their own `lane_group` floored by the
/// domain's default width. It is then **narrowed** by the axis — a width past
/// `axis_extent` rounded up to a power of two is idle lanes and scratch bytes
/// that are never read — and floored at 4 so the tree always has levels to
/// walk. Narrowing is safe in both directions: the per-lane strided loop
/// (`passes`) covers whatever the width does not.
fn fold_block(strat: FoldStrat, caps: &Caps, axis_extent: u32) -> u32 {
    let wide = match strat {
        FoldStrat::Subgroup => caps
            .subgroup_width()
            .max(1)
            .min(caps.limits.max_compute_invocations_per_workgroup.max(1)),
        FoldStrat::WgTree { lane_group } | FoldStrat::LoopThenTree { lane_group, .. } => {
            fusor2_tile::domains::emitted_block(lane_group.max(1), caps)
        }
    };
    wide.min(axis_extent.next_power_of_two()).max(4)
}

pub fn lower(caps: &Caps, node: &Node, theta: SchedPoint, cx: &LowerCtx<'_>) -> Result<KernelIr> {
    let Op::L1(op) = &node.op else {
        return Err(Error::Legality("not an L1 node".into()));
    };
    match op {
        L1::KMap { .. } => lower_map(caps, node, theta, cx),
        L1::KFold { .. } => lower_fold(caps, node, theta, cx),
        _ => Err(Error::Legality("map_fold got a foreign node".into())),
    }
}

fn view(buf: &Arc<fusor2_ir::ir::level2::BufferDecl>) -> StorageView {
    StorageView {
        buffer: Arc::clone(buf),
        offset: 0,
        layout: buf.layout.clone(),
    }
}

/// One elementwise pass: one lane per output element, `tm` elements per lane
/// when the tiling says so, so the loop-invariant operands stay resident in
/// registers across the `tm` outputs.
fn lower_map(caps: &Caps, node: &Node, theta: SchedPoint, cx: &LowerCtx<'_>) -> Result<KernelIr> {
    let Op::L1(L1::KMap {
        space, body, ops, ..
    }) = &node.op
    else {
        return Err(Error::Legality("not a KMap".into()));
    };
    let binds = Binds::build(cx)?;
    let uniforms = binds.buffers.first().cloned();
    let extents = const_extents(&space.dims)?;
    let n = extents.iter().map(|e| *e as u64).product::<u64>().max(1);

    let tm = match theta {
        SchedPoint::Map(t) => t.tm.max(1),
        _ => 1,
    };
    let block = default_block(caps);
    let grid = grid_for(n.div_ceil(tm as u64), block);
    let stride = grid[0] * block;

    let out_buf = binds.of(cx.launch.root)?;
    let mut stmts = Vec::with_capacity(tm as usize);
    for t in 0..tm {
        let flat = bin(
            BinOp::Add,
            global_lane(block),
            lit_u32(t * stride),
            u32_ty(),
        );
        let mask = cmp(CmpOp::Lt, flat.clone(), lit_u32(n as u32));
        let coords = coords_of(&flat, &extents);
        let mut args = Vec::with_capacity(ops.len());
        for o in ops {
            args.push(super::operand_at(cx, &binds, o, flat.clone(), n, mask.clone())?);
        }
        let value = Translate {
            args: &args,
            coords: &coords,
            uniforms: uniforms.clone(),
        }
        .run(body)?;
        stmts.push(Stmt::Store {
            dst: view(&out_buf),
            addr: Addr::Linear(flat),
            value,
            mask,
        });
    }

    Ok(KernelIr {
        buffers: binds.buffers,
        grid,
        block,
        body: stmts,
        byte_arena: None,
        name: "cpu_map",
    })
}

/// One workgroup per output row; the reduced axis is walked with vector loads
/// and finished by the strategy `theta` selected. The epilogue fuses straight
/// onto the reduced value, so nothing is materialized in between.
fn lower_fold(caps: &Caps, node: &Node, theta: SchedPoint, cx: &LowerCtx<'_>) -> Result<KernelIr> {
    let Op::L1(L1::KFold {
        space,
        axis,
        vec_axes,
        carrier,
        post,
        ops,
        ..
    }) = &node.op
    else {
        return Err(Error::Legality("not a KFold".into()));
    };
    if !vec_axes.is_empty() || fusor2_ir::ir::level2::fast_reduce_op(carrier).is_none() {
        return lower_fold_carrier(caps, node, theta, cx);
    }
    let pre = &carrier.lift[0];
    let post = &post[0];
    let binds = Binds::build(cx)?;
    let uniforms = binds.buffers.first().cloned();
    let extents = const_extents(&space.dims)?;
    let axis = *axis as usize;
    if axis >= extents.len() {
        return Err(Error::Legality("fold axis is out of range".into()));
    }
    let rop = reduce_op(carrier)?;
    let axis_extent = extents[axis].max(1);
    let inner: u32 = extents[axis + 1..].iter().product::<u32>().max(1);
    let outer: u32 = extents[..axis].iter().product::<u32>().max(1);
    let rows = (outer as u64) * (inner as u64);
    // The width is the resolved point's, not a written-in tile. A point that
    // names no lane group (a floor `ScheduleDomain::Point`, or a composite's
    // point handed down by `compose`) falls back to the domain's own default
    // width — the same `emitted_block` the fold domain prices with, so the
    // number the cost model charged and the number this allocates agree.
    let strat = match theta {
        SchedPoint::Fold(s) => s,
        _ => FoldStrat::WgTree {
            lane_group: default_block(caps),
        },
    };
    let block = fold_block(strat, caps, axis_extent);

    // **One pass of the block covers `block` elements of the axis.** A body
    // that reads `k = lane` and nothing else therefore reduces only the first
    // `block` of them: an 8192-element sum came back as the sum of its first
    // 256. Longer axes need a per-lane strided loop, and its counter has to
    // enter the address — `ReduceKind::Loop` re-evaluates the staging
    // expression per iteration, so an index-free body just combined the same
    // element `iterations` times.
    let passes = axis_extent.div_ceil(block).max(1);
    let loop_index = (passes > 1).then(|| Arc::new(LocalDecl::new(u32_ty())));

    let row = TileExpr::new(
        TileExprKind::Builtin(Builtin::ProgramId(WorkgroupAxis::X)),
        u32_ty(),
    );
    let lane = TileExpr::new(TileExprKind::Builtin(Builtin::Lane), u32_ty());
    let outer_idx = bin(BinOp::Div, row.clone(), lit_u32(inner), u32_ty());
    let inner_idx = bin(BinOp::Rem, row.clone(), lit_u32(inner), u32_ty());

    let k = match &loop_index {
        None => lane.clone(),
        Some(index) => {
            let pass = TileExpr::new(TileExprKind::LoadLocal(Arc::clone(index)), u32_ty());
            bin(
                BinOp::Add,
                bin(BinOp::Mul, pass, lit_u32(block), u32_ty()),
                lane.clone(),
                u32_ty(),
            )
        }
    };

    // The reduced axis is strided by `inner`, so a lane walks it with vector
    // loads rather than a scalar index iterator.
    let flat = bin(
        BinOp::Add,
        bin(
            BinOp::Mul,
            bin(
                BinOp::Add,
                bin(BinOp::Mul, outer_idx, lit_u32(axis_extent), u32_ty()),
                k.clone(),
                u32_ty(),
            ),
            lit_u32(inner),
            u32_ty(),
        ),
        inner_idx,
        u32_ty(),
    );
    let mask = cmp(CmpOp::Lt, k, lit_u32(axis_extent));

    let space_total = extents.iter().map(|e| u64::from(*e)).product::<u64>().max(1);
    let mut args = Vec::with_capacity(ops.len());
    for o in ops {
        args.push(super::operand_at(
            cx,
            &binds,
            o,
            flat.clone(),
            space_total,
            mask.clone(),
        )?);
    }
    let coords = coords_of(&flat, &extents);
    let contribution = Translate {
        args: &args,
        coords: &coords,
        uniforms: uniforms.clone(),
    }
    .run(pre)?;
    // Inactive lanes contribute the identity, so a partial tail cannot skew a
    // max or a product.
    let f32_ty = ElementType::Scalar(ScalarElement::F32);
    let contribution = TileExpr::new(
        TileExprKind::Select {
            condition: mask,
            accept: contribution,
            reject: lit_f32(crate::emit::reduce::identity_f32(rop)),
        },
        f32_ty,
    );

    let scratch: Arc<TileDecl> = Arc::new(TileDecl::new(
        f32_ty,
        TileLayout::contiguous(MemoryLevel::Workgroup, &[block]),
        "fold_scratch",
    ));
    // The group is the whole block, never the strategy's `lane_group`: this
    // kernel launches one workgroup per output row and every lane of it walks
    // that row's axis, so a tree over fewer lanes would drop the rest. The
    // strategy chooses the *shape* — a plain tree or a loop then a tree — and
    // the trip count comes from the extent, so a schedule point cannot ask for
    // fewer passes than the axis needs.
    let kind = match (&loop_index, strat) {
        (Some(index), _) => ReduceKind::Loop {
            iterations: passes,
            index: Arc::clone(index),
            scratch: Arc::clone(&scratch),
            group_size: block,
        },
        (None, FoldStrat::Subgroup) => ReduceKind::Subgroup,
        (None, _) => ReduceKind::Workgroup {
            scratch: Arc::clone(&scratch),
            group_size: block,
        },
    };

    let reduced = TileExpr::new(
        TileExprKind::Reduce {
            op: rop,
            kind: Box::new(kind),
            value: contribution,
        },
        f32_ty,
    );
    let value = Translate {
        args: &[reduced],
        coords: &coords,
        uniforms,
    }
    .run(post)?;

    let out_buf = binds.of(cx.launch.root)?;
    let body = vec![Stmt::Store {
        dst: view(&out_buf),
        addr: Addr::Linear(row),
        value,
        mask: cmp(CmpOp::Eq, lane, lit_u32(0)),
    }];

    Ok(KernelIr {
        buffers: binds.buffers,
        grid: [rows.max(1) as u32, 1, 1],
        block,
        body,
        byte_arena: None,
        name: "cpu_fold",
    })
}

/// Lower a `KFold` whose carrier is **wider than one hardware operator**.
///
/// One accumulator per carrier lane, seeded from that lane's own identity,
/// absorbed with the carrier's own `merge`, and closed by `Stmt::Reduce`'s N-ary
/// tree over one scratch tile per lane. The output carries `carrier.lanes()`
/// values per row, matching the trailing carrier axis `infer_l1` appends.
///
/// The SIMD butterfly folds one register with one operator, so there is no
/// horizontal-reduce form for a multi-lane merge and this always closes with the
/// scratch tree — the barrier the tree needs is the segment split, which
/// `stmt::block` performs around every collective statement.
fn lower_fold_carrier(
    caps: &Caps,
    node: &Node,
    theta: SchedPoint,
    cx: &LowerCtx<'_>,
) -> Result<KernelIr> {
    let Op::L1(L1::KFold {
        space,
        axis,
        vec_axes,
        carrier,
        post,
        ops,
        ..
    }) = &node.op
    else {
        return Err(Error::Legality("not a KFold".into()));
    };
    let merges = carrier.merge_lanes().ok_or_else(|| {
        Error::Legality("this carrier's merge does not expand to one expression per lane".into())
    })?;
    let lanes = merges.len();
    let posts = carrier.expand_lanes(post).ok_or_else(|| {
        Error::Legality(format!(
            "a {}-slot carrier carries {} post expressions, or a slot's post reads \
             a sibling of a different width",
            carrier.width(),
            post.len()
        ))
    })?;
    let lane_slots = carrier
        .lane_slots()
        .ok_or_else(|| Error::Legality("this carrier has a symbolic Vector extent".into()))?;
    let lane_ident = carrier
        .identity_lanes()
        .ok_or_else(|| Error::Legality("this carrier has a symbolic Vector extent".into()))?;

    let binds = Binds::build(cx)?;
    let uniforms = binds.buffers.first().cloned();
    let extents = const_extents(&space.dims)?;
    let axis = *axis as usize;
    if axis >= extents.len() {
        return Err(Error::Legality("fold axis is out of range".into()));
    }
    // A promoted nest: `space` is `free.. ++ vec.. ++ [reduced]`, so one output
    // row spans `vec_extent * axis_extent` consecutive elements and a `Vector`
    // slot is `vec_extent` registers. `verify_l1` establishes the contiguous
    // block; the reduced axis being last is what makes the address below one
    // multiply.
    let vec_extent: u32 = vec_axes
        .iter()
        .map(|i| extents[*i as usize])
        .product::<u32>()
        .max(1);
    if !vec_axes.is_empty() && axis + 1 != extents.len() {
        return Err(Error::Legality(
            "a promoted KFold whose reduced axis is not last is not lowered".into(),
        ));
    }
    if vec_axes.is_empty() && carrier.slots.iter().any(|s| *s != SlotTy::Scalar) {
        return Err(Error::Legality(
            "a Vector carrier slot needs a promoted axis to read its positions from".into(),
        ));
    }
    // Iteration axis `j` is space axis `iter_axes[j]`; every `ScalarExpr` here
    // is written against the iteration space.
    let iter_axes: Vec<usize> = (0..extents.len())
        .filter(|i| !vec_axes.contains(&(*i as u32)))
        .collect();
    let axis_extent = extents[axis].max(1);
    let inner: u32 = extents[axis + 1..].iter().product::<u32>().max(1);
    let outer: u32 = extents[..axis]
        .iter()
        .enumerate()
        .filter(|(i, _)| !vec_axes.contains(&(*i as u32)))
        .map(|(_, e)| *e)
        .product::<u32>()
        .max(1);
    let rows = (outer as u64) * (inner as u64);
    // Same rule as the single-slot body: the width comes off the resolved
    // point. This used to take `_theta` and allocate a written-in 256, so a
    // multi-lane carrier ignored every schedule decision extraction made.
    let strat = match theta {
        SchedPoint::Fold(s) => s,
        _ => FoldStrat::WgTree {
            lane_group: default_block(caps),
        },
    };
    let block = fold_block(strat, caps, axis_extent);
    let passes = axis_extent.div_ceil(block).max(1);
    let f32_ty = ElementType::Scalar(ScalarElement::F32);
    let space_total = extents.iter().map(|e| u64::from(*e)).product::<u64>().max(1);

    let row = TileExpr::new(
        TileExprKind::Builtin(Builtin::ProgramId(WorkgroupAxis::X)),
        u32_ty(),
    );
    let lane = TileExpr::new(TileExprKind::Builtin(Builtin::Lane), u32_ty());
    let outer_idx = bin(BinOp::Div, row.clone(), lit_u32(inner), u32_ty());
    let inner_idx = bin(BinOp::Rem, row.clone(), lit_u32(inner), u32_ty());

    // One lifted value per lane at element `k`, each guarded to **its own**
    // identity outside the reduced extent. A shared identity would let a padding
    // lane count in Welford's constant `1` slot.
    //
    // **Per-position operand addressing.** Lane `(slot, p)` reads every operand
    // at promoted position `p`. An operand invariant in the promoted axes is
    // read at the same address for every position and the emitter's own CSE
    // collapses it to one load; one that varies is read once per position, as
    // it must be.
    let lift_at = |k: TileExpr| -> Result<Vec<TileExpr>> {
        let mask = cmp(CmpOp::Lt, k.clone(), lit_u32(axis_extent));
        let row_elems = axis_extent.saturating_mul(vec_extent);
        let mut per_pos: Vec<(Vec<TileExpr>, Vec<TileExpr>)> =
            Vec::with_capacity(vec_extent as usize);
        for p in 0..vec_extent {
            let within = bin(
                BinOp::Add,
                bin(BinOp::Mul, lit_u32(p), lit_u32(axis_extent), u32_ty()),
                k.clone(),
                u32_ty(),
            );
            let flat = bin(
                BinOp::Add,
                bin(
                    BinOp::Mul,
                    bin(
                        BinOp::Add,
                        bin(BinOp::Mul, outer_idx.clone(), lit_u32(row_elems), u32_ty()),
                        within,
                        u32_ty(),
                    ),
                    lit_u32(inner),
                    u32_ty(),
                ),
                inner_idx.clone(),
                u32_ty(),
            );
            let mut args = Vec::with_capacity(ops.len());
            for o in ops {
                args.push(super::operand_at(
                    cx,
                    &binds,
                    o,
                    flat.clone(),
                    space_total,
                    mask.clone(),
                )?);
            }
            let full = coords_of(&flat, &extents);
            let coords: Vec<TileExpr> = iter_axes.iter().map(|i| full[*i].clone()).collect();
            per_pos.push((args, coords));
        }
        let mut out = Vec::with_capacity(lanes);
        for (slot, p) in &lane_slots {
            let (args, coords) = &per_pos[*p as usize];
            let v = Translate {
                args,
                coords,
                uniforms: uniforms.clone(),
            }
            .run(&carrier.lift[*slot])?;
            out.push(TileExpr::new(
                TileExprKind::Select {
                    condition: mask.clone(),
                    accept: v,
                    reject: lit_f32(splat_f32(carrier.identity[*slot])),
                },
                f32_ty,
            ));
        }
        Ok(out)
    };

    let mut body: Vec<Stmt> = Vec::new();
    let partials: Vec<TileExpr> = if passes > 1 {
        // The per-lane strided loop, carrying `lanes` accumulators seeded from
        // the carrier's identities and absorbed with its own `merge`.
        let index = Arc::new(LocalDecl::new(u32_ty()));
        let pass = TileExpr::new(TileExprKind::LoadLocal(Arc::clone(&index)), u32_ty());
        let k = bin(
            BinOp::Add,
            bin(BinOp::Mul, pass, lit_u32(block), u32_ty()),
            lane.clone(),
            u32_ty(),
        );
        let values = lift_at(k)?;
        let locals: Vec<Arc<LocalDecl>> = (0..lanes)
            .map(|_| Arc::new(LocalDecl::new(f32_ty)))
            .collect();
        let reads: Vec<TileExpr> = locals
            .iter()
            .map(|l| TileExpr::new(TileExprKind::LoadLocal(Arc::clone(l)), f32_ty))
            .collect();
        let mut args = reads.clone();
        args.extend(values);
        let mut accumulators = Vec::with_capacity(lanes);
        for slot in 0..lanes {
            accumulators.push(fusor2_ir::ir::level2::Accumulator {
                local: Arc::clone(&locals[slot]),
                init: lit_f32(splat_f32(lane_ident[slot])),
                update: Translate {
                    args: &args,
                    coords: &[],
                    uniforms: uniforms.clone(),
                }
                .run(&merges[slot])?,
            });
        }
        body.push(Stmt::Loop {
            count: Some(lit_u32(passes)),
            index: Some(index),
            accumulators,
            body: Vec::new(),
        });
        reads
    } else {
        lift_at(lane.clone())?
    };

    let scratch: smallvec::SmallVec<[Arc<TileDecl>; 4]> = (0..lanes)
        .map(|_| {
            Arc::new(TileDecl::new(
                f32_ty,
                TileLayout::contiguous(MemoryLevel::Workgroup, &[block]),
                "fold_scratch",
            ))
        })
        .collect();
    let lhs: smallvec::SmallVec<[Arc<LocalDecl>; 4]> = (0..lanes)
        .map(|_| Arc::new(LocalDecl::new(f32_ty)))
        .collect();
    let rhs: smallvec::SmallVec<[Arc<LocalDecl>; 4]> = (0..lanes)
        .map(|_| Arc::new(LocalDecl::new(f32_ty)))
        .collect();
    let outs: smallvec::SmallVec<[Arc<LocalDecl>; 4]> = (0..lanes)
        .map(|_| Arc::new(LocalDecl::new(f32_ty)))
        .collect();
    let merge_args: Vec<TileExpr> = lhs
        .iter()
        .chain(rhs.iter())
        .map(|l| TileExpr::new(TileExprKind::LoadLocal(Arc::clone(l)), f32_ty))
        .collect();
    let mut merge_body: smallvec::SmallVec<[TileExpr; 4]> = smallvec::SmallVec::new();
    for slot in 0..lanes {
        merge_body.push(
            Translate {
                args: &merge_args,
                coords: &[],
                uniforms: uniforms.clone(),
            }
            .run(&merges[slot])?,
        );
    }
    body.push(Stmt::Reduce {
        kind: Box::new(ReduceKind::Workgroup {
            scratch: Arc::clone(&scratch[0]),
            group_size: block,
        }),
        values: partials.into_iter().collect(),
        merge: Box::new(fusor2_ir::ir::level2::MergeBody {
            lhs,
            rhs,
            body: merge_body,
        }),
        fast: None,
        outs: outs.clone(),
        scratch,
    });

    let reduced: Vec<TileExpr> = outs
        .iter()
        .map(|l| TileExpr::new(TileExprKind::LoadLocal(Arc::clone(l)), f32_ty))
        .collect();
    let out_buf = binds.of(cx.launch.root)?;
    let mask = cmp(CmpOp::Eq, lane, lit_u32(0));
    let base = bin(BinOp::Mul, row, lit_u32(lanes as u32), u32_ty());
    for slot in 0..lanes {
        let value = Translate {
            args: &reduced,
            coords: &[],
            uniforms: uniforms.clone(),
        }
        .run(&posts[slot])?;
        body.push(Stmt::Store {
            dst: view(&out_buf),
            addr: Addr::Linear(bin(
                BinOp::Add,
                base.clone(),
                lit_u32(slot as u32),
                u32_ty(),
            )),
            value,
            mask: mask.clone(),
        });
    }

    Ok(KernelIr {
        buffers: binds.buffers,
        grid: [rows.max(1) as u32, 1, 1],
        block,
        body,
        byte_arena: None,
        name: "cpu_fold_carrier",
    })
}

/// A carrier identity as a host float.
fn splat_f32(s: fusor2_ir::dtype::Splat) -> f32 {
    use fusor2_ir::dtype::Splat;
    match s {
        Splat::F32(v) => v,
        Splat::F16(b) => half::f16::from_bits(b).to_f32(),
        Splat::BF16(b) => half::bf16::from_bits(b).to_f32(),
        Splat::U32(v) => v as f32,
        Splat::I32(v) => v as f32,
    }
}

/// The hardware collective this carrier reduces with, or an honest `Err`.
///
/// `Carrier::kind()` — one scalar slot merged by a binop — mapped onto
/// `TileReduceOp`. A multi-slot or promoted carrier needs N accumulators and
/// an N-lane reduce, which this emitter does not have; refusing is the point,
/// because reducing slot 0 and dropping the rest is a wrong answer.
pub(crate) fn reduce_op(c: &Carrier) -> Result<TileReduceOp> {
    if c.width() != 1 || c.slots[0] != SlotTy::Scalar {
        return Err(Error::Legality(format!(
            "a {}-slot carrier needs the N-lane reduce; the CPU emitter only \
             lowers a single scalar slot",
            c.width()
        )));
    }
    Ok(match c.kind() {
        Some(BinOp::Add) => TileReduceOp::Sum,
        Some(BinOp::Mul) => TileReduceOp::Product,
        Some(BinOp::Max) => TileReduceOp::Max,
        Some(BinOp::Min) => TileReduceOp::Min,
        other => {
            return Err(Error::Legality(format!(
                "carrier merge {other:?} has no CPU collective; the generic \
                 merge path is not built yet"
            )));
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use fusor2_ir::carrier::ArgRemap;
    use fusor2_ir::dtype::Dtype;

    fn binop(op: BinOp) -> Carrier {
        Carrier::binop(op, Carrier::binop_identity(op, Dtype::F32).unwrap(), Dtype::F32)
    }

    #[test]
    fn every_single_slot_binop_carrier_maps_to_a_reduction_operator() {
        assert_eq!(reduce_op(&binop(BinOp::Add)).unwrap(), TileReduceOp::Sum);
        assert_eq!(reduce_op(&binop(BinOp::Mul)).unwrap(), TileReduceOp::Product);
        assert_eq!(reduce_op(&binop(BinOp::Max)).unwrap(), TileReduceOp::Max);
        assert_eq!(reduce_op(&binop(BinOp::Min)).unwrap(), TileReduceOp::Min);
    }

    /// A carrier the emitter cannot honour is refused, never truncated to its
    /// first slot.
    #[test]
    fn a_multi_slot_carrier_is_refused() {
        let pair = binop(BinOp::Max).tuple(&binop(BinOp::Add), &ArgRemap::identity(1));
        assert!(reduce_op(&pair.carrier).is_err());
        let promoted = binop(BinOp::Add)
            .promote(fusor2_ir::shape::Dim::Const(8))
            .unwrap();
        assert!(reduce_op(&promoted).is_err());
    }
    /// **A firing test on the runner, not just the lowering.** A three-slot
    /// Welford carrier compiled through the CPU emitter and executed against a
    /// two-pass variance: the merge tape has to run at every tree level, for
    /// every lane, or `n` comes back plausible and `m2` comes back wrong.
    #[test]
    fn a_three_slot_carrier_runs_and_matches_a_two_pass_variance() {
        use crate::alloc::AlignedBuf;
        use fusor2_ir::carrier::oracle;
        use fusor2_ir::target::{Buf, Uniforms};

        const ROWS: usize = 3;
        const AXIS: usize = 600;
        let data: Vec<f32> = (0..ROWS * AXIS)
            .map(|i| ((i * 37 % 101) as f32) * 0.25 - 12.0)
            .collect();

        let ir = fold_kernel(oracle::welford(Dtype::F32), ROWS as u32, AXIS as u32);
        let art = crate::emit::compile(&ir, crate::caps::cpu_caps(), None).unwrap();
        // The staged tree is collective, so `stmt::block` cuts the lane loop
        // around it: the per-lane accumulate loop, the tree, and the stores are
        // three segments. A `Barrier` mapped to a no-op would fuse them and the
        // tree would read tile slots a later lane chunk has not written.
        assert!(
            art.prog.segments.len() >= 3,
            "the reduction must split the lane loop, got {} segments",
            art.prog.segments.len()
        );
        let kernel = crate::emit::CpuKernel {
            name: art.name,
            block: art.block,
            vector_width: art.prog.width,
            artifact: art,
        };
        let mut binds = Vec::new();
        for v in [vec![0.0f32; 4], data.clone()] {
            let mut b = AlignedBuf::zeroed(v.len() * 4).unwrap();
            b.as_mut_slice()
                .copy_from_slice(bytemuck::cast_slice(v.as_slice()));
            binds.push(Buf::new(b));
        }
        let out = Buf::new(AlignedBuf::zeroed(ROWS * 3 * 4).unwrap());
        binds.push(out.clone());
        kernel
            .run(ir.grid, &binds, &Uniforms::default())
            .expect("launch");
        let ab = out.downcast_ref::<AlignedBuf>().unwrap();
        let got: Vec<f32> = bytemuck::cast_slice::<u8, f32>(ab.as_slice()).to_vec();

        for r in 0..ROWS {
            let row = &data[r * AXIS..(r + 1) * AXIS];
            let n = row.len() as f32;
            let mean = row.iter().sum::<f32>() / n;
            let m2: f32 = row.iter().map(|v| (v - mean) * (v - mean)).sum();
            assert!((got[r * 3] - n).abs() < 1e-3, "row {r} n: {:?}", got);
            assert!((got[r * 3 + 1] - mean).abs() < 1e-3, "row {r} mean: {:?}", got);
            assert!(
                (got[r * 3 + 2] - m2).abs() < 1e-2 * m2.abs().max(1.0),
                "row {r} m2: got {} want {m2}",
                got[r * 3 + 2]
            );
        }
    }

    /// **A promoted nest, compiled and executed, against a host sum.**
    ///
    /// `space = [rows, dh, k]` with `vec_axes = [1]` is the shape PROMOTE
    /// mints: the reduction runs over `k`, and the free axis `dh` has moved
    /// out of the iteration domain into `dh` live accumulators. The value it
    /// must produce is the ordinary `sum_k x[r, d, k]` — so this test says the
    /// promoted spelling and the obvious one agree, which is the only thing
    /// that makes the rebinding worth anything.
    ///
    /// The lift is `Arg(0)`, read **per position**: if the lowering addressed
    /// its operand once per iteration step instead of once per (step,
    /// position), every accumulator would receive position 0's data and `dh -
    /// 1` of the `dh` outputs would be wrong while the first stayed right.
    #[test]
    fn a_promoted_nest_runs_and_matches_a_host_sum() {
        use crate::alloc::AlignedBuf;
        use fusor2_ir::target::{Buf, Uniforms};

        const ROWS: usize = 3;
        const DH: usize = 4;
        const K: usize = 40;
        let data: Vec<f32> = (0..ROWS * DH * K)
            .map(|i| ((i * 29 % 97) as f32) * 0.125 - 6.0)
            .collect();

        let carrier = binop(BinOp::Add)
            .promote(fusor2_ir::shape::Dim::Const(DH as u64))
            .unwrap();
        assert_eq!(carrier.lanes(), Some(DH as u64));
        let ir = fold_kernel_in(carrier, &[ROWS as u64, DH as u64, K as u64], 2, &[1]);
        let art = crate::emit::compile(&ir, crate::caps::cpu_caps(), None).unwrap();
        let kernel = crate::emit::CpuKernel {
            name: art.name,
            block: art.block,
            vector_width: art.prog.width,
            artifact: art,
        };
        let mut binds = Vec::new();
        for v in [vec![0.0f32; 4], data.clone()] {
            let mut b = AlignedBuf::zeroed(v.len() * 4).unwrap();
            b.as_mut_slice()
                .copy_from_slice(bytemuck::cast_slice(v.as_slice()));
            binds.push(Buf::new(b));
        }
        let out = Buf::new(AlignedBuf::zeroed(ROWS * DH * 4).unwrap());
        binds.push(out.clone());
        kernel
            .run(ir.grid, &binds, &Uniforms::default())
            .expect("launch");
        let ab = out.downcast_ref::<AlignedBuf>().unwrap();
        let got: Vec<f32> = bytemuck::cast_slice::<u8, f32>(ab.as_slice()).to_vec();

        for r in 0..ROWS {
            for d in 0..DH {
                let want: f32 = (0..K).map(|k| data[(r * DH + d) * K + k]).sum();
                let g = got[r * DH + d];
                assert!(
                    (g - want).abs() < 1e-3 * want.abs().max(1.0),
                    "row {r} position {d}: got {g} want {want} (all: {got:?})"
                );
            }
        }
    }

    /// `[rows, axis]` folded over axis 1 at `carrier`, as a standalone kernel.
    fn fold_kernel(carrier: Carrier, rows: u32, axis: u32) -> KernelIr {
        fold_kernel_in(carrier, &[rows as u64, axis as u64], 1, &[])
    }

    /// [`fold_kernel`] over an explicit space, so a **promoted** nest can be
    /// compiled and run here too.
    fn fold_kernel_in(
        carrier: Carrier,
        dims: &[u64],
        axis: u32,
        vec_axes: &[u32],
    ) -> KernelIr {
        use fusor2_ir::egraph::EGraph;
        use fusor2_ir::extract::{BindKind, BindingPlan, Extraction, Launch, Plan, PlanHash};
        use fusor2_ir::ir::level0::{BufferId, L0, LeafKind};
        use fusor2_ir::ir::level1::{AccessPlan, IndexSpace, Operand, ScheduleDomain};
        use fusor2_ir::scalar::ScalarExpr;
        use fusor2_ir::semantics::{CoreSemantics, SumArenaPlanner};
        use fusor2_ir::shape::{Dim, Layout};

        let mut g = EGraph::new(CoreSemantics::new(Arc::new(SumArenaPlanner)));
        let x = g
            .add(fusor2_ir::ir::Op::L0(L0::Leaf(LeafKind::Buffer {
                name: BufferId(0),
                dtype: Dtype::F32,
                shape: dims.iter().map(|d| Dim::Const(*d)).collect(),
            })))
            .unwrap();
        let width = carrier.width();
        let k = g
            .add(fusor2_ir::ir::Op::L1(L1::KFold {
                space: IndexSpace::new(dims.iter().map(|d| Dim::Const(*d))),
                axis,
                vec_axes: vec_axes.iter().copied().collect(),
                carrier,
                acc: Dtype::F32,
                post: (0..width)
                    .map(|i| ScalarExpr::arg(i as u32, Dtype::F32))
                    .collect(),
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
            cost: fusor2_ir::cost::Picoseconds(0),
        };
        let cx = LowerCtx {
            plan: &plan,
            launch: &plan.launches[0],
            graph: &g,
            symbols: &[],
        };
        let ir = lower(crate::caps::cpu_caps(), g.node(k), SchedPoint::Point, &cx).unwrap();
        assert_eq!(ir.name, "cpu_fold_carrier");
        ir
    }
}
