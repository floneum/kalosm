//! `Map` and `Fold` as SIMD loop nests with a register accumulator tile.
//!
//! `Map` reads its register-reuse tiling off `SchedPoint::Map(MapTiling)`
//! (`dim`, `tm`, `vector`); `Fold` reads its strategy off
//! `SchedPoint::Fold(FoldStrat)` and lowers all three: `Subgroup` to a
//! horizontal reduce, `WgTree` to a tree over a scratch tile, and
//! `LoopThenTree` to per-lane loop accumulation followed by that tree.

use fusor_ir::Result;
use fusor_ir::carrier::{Carrier, SlotTy};
use fusor_ir::device::Caps;
use fusor_ir::error::Error;
use fusor_ir::ir::kernel::{
    Addr, Builtin, ElementType, KernelIr, LocalDecl, MemoryLevel, ReduceKind, ScalarElement, Stmt,
    StorageView, TileDecl, TileExpr, TileExprKind, TileLayout, TileReduceOp, WorkgroupAxis,
};
use fusor_ir::ir::launch::{FoldStrat, Launch, SchedPoint};
use fusor_ir::ir::{Node, Op};
use fusor_ir::scalar::{BinOp, CmpOp};
use fusor_ir::target::LowerCtx;
use std::sync::Arc;

use super::{
    Binds, Translate, bin, cmp, const_extents, coords_of, default_block, global_lane, grid_for,
    lit_f32, lit_u32, u32_ty,
};

/// The workgroup width a fold allocates its scratch over.
///
/// The resolved point decides it: `FoldStrat::Subgroup` runs one SIMD group
/// wide, the two tree strategies run their own `lane_group` floored by the
/// domain's default width. It is then narrowed by the axis and floored at 4
/// so the tree always has levels to walk. Narrowing is safe in both
/// directions: the per-lane strided loop (`passes`) covers whatever the width
/// does not.
fn fold_block(strat: FoldStrat, caps: &Caps, axis_extent: u32) -> u32 {
    let wide = match strat {
        FoldStrat::Subgroup => caps
            .subgroup_width()
            .max(1)
            .min(caps.limits.max_compute_invocations_per_workgroup.max(1)),
        FoldStrat::WgTree { lane_group } | FoldStrat::LoopThenTree { lane_group, .. } => {
            fusor_tile::domains::emitted_block(lane_group.max(1), caps)
        }
    };
    wide.min(axis_extent.next_power_of_two()).max(4)
}

pub(crate) fn lower(
    caps: &Caps,
    node: &Node,
    theta: SchedPoint,
    cx: &LowerCtx<'_>,
) -> Result<KernelIr> {
    let Op::Launch(op) = &node.op else {
        return Err(Error::Legality("not a Launch node".into()));
    };
    match op {
        Launch::Map { .. } => lower_map(caps, node, theta, cx),
        // The carrier tree is the one native fold representation. It handles
        // both scalar operators and multi-slot carriers, so lowering every
        // fold through it prevents the planner from selecting a second shape
        // that has no Cranelift implementation.
        Launch::Fold { .. } => lower_fold_carrier(caps, node, theta, cx),
        _ => Err(Error::Legality("map_fold got a foreign node".into())),
    }
}

fn view(buf: &Arc<fusor_ir::ir::kernel::BufferDecl>) -> StorageView {
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
    let Op::Launch(Launch::Map {
        space, body, ops, ..
    }) = &node.op
    else {
        return Err(Error::Legality("not a Map".into()));
    };
    let binds = Binds::build(cx)?;
    let uniforms = binds.buffers.first().cloned();
    let extents = const_extents(cx, &space.dims)?;
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
            args.push(super::operand_at(
                cx,
                &binds,
                o,
                flat.clone(),
                n,
                mask.clone(),
            )?);
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
#[allow(dead_code)]
fn lower_fold(caps: &Caps, node: &Node, theta: SchedPoint, cx: &LowerCtx<'_>) -> Result<KernelIr> {
    let Op::Launch(Launch::Fold {
        space,
        axis,
        vec_axes,
        carrier,
        post,
        ops,
        ..
    }) = &node.op
    else {
        return Err(Error::Legality("not a Fold".into()));
    };
    if !vec_axes.is_empty() || fusor_ir::ir::kernel::fast_reduce_op(carrier).is_none() {
        return lower_fold_carrier(caps, node, theta, cx);
    }
    let pre = &carrier.lift[0];
    let post = &post[0];
    let binds = Binds::build(cx)?;
    let uniforms = binds.buffers.first().cloned();
    let extents = const_extents(cx, &space.dims)?;
    let axis = *axis as usize;
    if axis >= extents.len() {
        return Err(Error::Legality("fold axis is out of range".into()));
    }
    let rop = reduce_op(carrier)?;
    let axis_extent = extents[axis].max(1);
    let inner: u32 = extents[axis + 1..].iter().product::<u32>().max(1);
    let outer: u32 = extents[..axis].iter().product::<u32>().max(1);
    let rows = (outer as u64) * (inner as u64);
    // A point that names no lane group falls back to the domain's own default
    // width — the same `emitted_block` the fold domain prices with, so the
    // number the cost model charged and the number this allocates agree.
    let strat = match theta {
        SchedPoint::Fold(s) => s,
        _ => FoldStrat::WgTree {
            lane_group: default_block(caps),
        },
    };
    let block = fold_block(strat, caps, axis_extent);

    // One pass of the block covers `block` elements of the axis, so longer
    // axes need a per-lane strided loop, and its counter has to enter the
    // address: `ReduceKind::Loop` re-evaluates the staging expression per
    // iteration, so an index-free body would combine the same element
    // `iterations` times.
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

    let space_total = extents
        .iter()
        .map(|e| u64::from(*e))
        .product::<u64>()
        .max(1);
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
    // strategy chooses the shape; the trip count comes from the extent.
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

/// Lower a `Fold` whose carrier is wider than one hardware operator.
///
/// One accumulator per carrier lane, seeded from that lane's own identity,
/// absorbed with the carrier's own `merge`, and closed by `Stmt::Reduce`'s N-ary
/// tree over one scratch tile per lane. The output carries `carrier.lanes()`
/// values per row, matching the trailing carrier axis `infer_launch` appends.
///
/// The SIMD butterfly folds one register with one operator, so there is no
/// horizontal-reduce form for a multi-lane merge and this always closes with
/// the scratch tree.
fn lower_fold_carrier(
    caps: &Caps,
    node: &Node,
    theta: SchedPoint,
    cx: &LowerCtx<'_>,
) -> Result<KernelIr> {
    let Op::Launch(Launch::Fold {
        space,
        axis,
        vec_axes,
        carrier,
        post,
        ops,
        ..
    }) = &node.op
    else {
        return Err(Error::Legality("not a Fold".into()));
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
    let extents = const_extents(cx, &space.dims)?;
    let axis = *axis as usize;
    if axis >= extents.len() {
        return Err(Error::Legality("fold axis is out of range".into()));
    }
    // A promoted nest: `space` is `free.. ++ vec.. ++ [reduced]`, so one output
    // row spans `vec_extent * axis_extent` consecutive elements and a `Vector`
    // slot is `vec_extent` registers. `verify_launch` establishes the contiguous
    // block; the reduced axis being last is what makes the address below one
    // multiply.
    let vec_extent: u32 = vec_axes
        .iter()
        .map(|i| extents[*i as usize])
        .product::<u32>()
        .max(1);
    if !vec_axes.is_empty() && axis + 1 != extents.len() {
        return Err(Error::Legality(
            "a promoted Fold whose reduced axis is not last is not lowered".into(),
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
    // The width comes off the resolved point, same as the single-slot body.
    let strat = match theta {
        SchedPoint::Fold(s) => s,
        _ => FoldStrat::WgTree {
            lane_group: default_block(caps),
        },
    };
    let block = fold_block(strat, caps, axis_extent);
    let passes = axis_extent.div_ceil(block).max(1);
    let f32_ty = ElementType::Scalar(ScalarElement::F32);
    let space_total = extents
        .iter()
        .map(|e| u64::from(*e))
        .product::<u64>()
        .max(1);

    let row = TileExpr::new(
        TileExprKind::Builtin(Builtin::ProgramId(WorkgroupAxis::X)),
        u32_ty(),
    );
    let lane = TileExpr::new(TileExprKind::Builtin(Builtin::Lane), u32_ty());
    let outer_idx = bin(BinOp::Div, row.clone(), lit_u32(inner), u32_ty());
    let inner_idx = bin(BinOp::Rem, row.clone(), lit_u32(inner), u32_ty());

    // One lifted value per lane at element `k`, each guarded to its own
    // identity outside the reduced extent: a shared identity would let a
    // padding lane count in Welford's constant `1` slot.
    //
    // Lane `(slot, p)` reads every operand at promoted position `p`. An
    // operand invariant in the promoted axes is read at the same address for
    // every position and the emitter's CSE collapses it to one load.
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
            accumulators.push(fusor_ir::ir::kernel::Accumulator {
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
    for merge in merges.iter().take(lanes) {
        merge_body.push(
            Translate {
                args: &merge_args,
                coords: &[],
                uniforms: uniforms.clone(),
            }
            .run(merge)?,
        );
    }
    body.push(Stmt::Reduce {
        kind: Box::new(ReduceKind::Workgroup {
            scratch: Arc::clone(&scratch[0]),
            group_size: block,
        }),
        values: partials.into_iter().collect(),
        merge: Box::new(fusor_ir::ir::kernel::MergeBody {
            lhs,
            rhs,
            body: merge_body,
        }),
        fast: fusor_ir::ir::kernel::fast_reduce_op(carrier),
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
    for (slot, post) in posts.iter().enumerate().take(lanes) {
        let value = Translate {
            args: &reduced,
            coords: &[],
            uniforms: uniforms.clone(),
        }
        .run(post)?;
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
fn splat_f32(s: fusor_ir::dtype::Splat) -> f32 {
    use fusor_ir::dtype::Splat;
    match s {
        Splat::F32(v) => v,
        Splat::F16(b) => half::f16::from_bits(b).to_f32(),
        Splat::BF16(b) => half::bf16::from_bits(b).to_f32(),
        Splat::U32(v) => v as f32,
        Splat::I32(v) => v as f32,
    }
}

/// The hardware collective this carrier reduces with.
///
/// `Carrier::kind()` — one scalar slot merged by a binop — mapped onto
/// `TileReduceOp`. A multi-slot or promoted carrier needs N accumulators and
/// an N-lane reduce, which this emitter does not have, so it refuses rather
/// than reducing slot 0 and dropping the rest.
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
