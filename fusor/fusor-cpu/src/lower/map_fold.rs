//! `Map` and `Fold` as SIMD loop nests with a register accumulator tile.
//!
//! `Map` reads its register-reuse tiling off `SchedPoint::Map(MapTiling)`
//! (`dim`, `tm`, `vector`); `Fold` reads its strategy off
//! `SchedPoint::Fold(FoldStrat)` and lowers all three: `Subgroup` to a
//! horizontal reduce, `WgTree` to a tree over a scratch tile, and
//! `LoopThenTree` to per-lane loop accumulation followed by that tree.

use fusor_ir::Result;
use fusor_ir::device::Caps;
use fusor_ir::error::Error;
use fusor_ir::ir::kernel::{
    Accumulator, Addr, Builtin, ElementType, KernelIr, ScalarElement, Stmt, TileExpr, TileExprKind,
    WorkgroupAxis,
};
use fusor_ir::ir::launch::{FoldStrat, Launch, SchedPoint};
use fusor_ir::ir::{Node, Op};
use fusor_ir::target::LowerCtx;
use fusor_tile::build::{FoldLanes, Kernel};

use super::{
    Binds, DEFAULT_BLOCK, Translate, const_extents, coords_of, global_lane, grid_for, operand_at,
    view,
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
        Launch::Map { .. } => lower_map(node, theta, cx),
        // The carrier tree is the one native fold representation. It handles
        // both scalar operators and multi-slot carriers, so lowering every
        // fold through it prevents the planner from selecting a second shape
        // that has no Cranelift implementation.
        Launch::Fold { .. } => lower_fold_carrier(caps, op, theta, cx, None),
        Launch::StreamFold {
            producer,
            fold,
            operand,
            ..
        } => lower_fold_carrier(caps, fold, theta, cx, Some((producer, *operand))),
        _ => Err(Error::Legality("map_fold got a foreign node".into())),
    }
}

/// One elementwise pass: one lane per output element, `tm` elements per lane
/// when the tiling says so, so the loop-invariant operands stay resident in
/// registers across the `tm` outputs.
fn lower_map(node: &Node, theta: SchedPoint, cx: &LowerCtx<'_>) -> Result<KernelIr> {
    let Op::Launch(Launch::Map {
        space, body, ops, ..
    }) = &node.op
    else {
        return Err(Error::Legality("not a Map".into()));
    };
    let b = Kernel::new();
    let binds = Binds::build(cx)?;
    let uniforms = binds.buffers.first().cloned();
    let extents = const_extents(cx, &space.dims)?;
    let n = extents.iter().map(|e| *e as u64).product::<u64>().max(1);

    let tm = match theta {
        SchedPoint::Map(t) => t.tm.max(1),
        _ => 1,
    };
    let block = DEFAULT_BLOCK;
    let grid = grid_for(n.div_ceil(tm as u64), block);
    let stride = grid[0] * block;

    let out = view(&binds.of(cx.launch.root)?);
    let mut stmts = Vec::with_capacity(tm as usize);
    for t in 0..tm {
        let flat = b.add(global_lane(&b, block), b.u32(t * stride));
        let mask = b.lt(flat.clone(), b.u32(n as u32));
        let coords = coords_of(&b, &flat, &extents);
        let args = ops
            .iter()
            .map(|o| operand_at(&b, cx, &binds, o, flat.clone(), n, mask.clone()))
            .collect::<Result<Vec<_>>>()?;
        let value = Translate {
            b: &b,
            args: &args,
            coords: &coords,
            uniforms: uniforms.clone(),
        }
        .run(body)?;
        stmts.push(Stmt::Store {
            dst: out.clone(),
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

/// Lower a `Fold` through its carrier.
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
    fold: &Launch,
    theta: SchedPoint,
    cx: &LowerCtx<'_>,
    stream: Option<(&Launch, u32)>,
) -> Result<KernelIr> {
    let Launch::Fold {
        space,
        axis,
        vec_axes,
        carrier,
        post,
        ops,
        ..
    } = fold
    else {
        return Err(Error::Legality("not a Fold".into()));
    };
    let axis = *axis as usize;
    let lanes = FoldLanes::of(carrier, post, space.rank(), axis, vec_axes, Error::Legality)?;
    let b = Kernel::new();
    let binds = Binds::build(cx)?;
    let uniforms = binds.buffers.first().cloned();
    let translate = |args: &[TileExpr], coords: &[TileExpr], e| {
        Translate {
            b: &b,
            args,
            coords,
            uniforms: uniforms.clone(),
        }
        .run(e)
    };
    let extents = const_extents(cx, &space.dims)?;
    // A promoted nest: `space` is `free.. ++ vec.. ++ [reduced]`, so one output
    // row spans `vec_extent * axis_extent` consecutive elements and a `Vector`
    // slot is `vec_extent` registers; the reduced axis being last is what makes
    // the address below one multiply.
    let vec_extent: u32 = vec_axes
        .iter()
        .map(|i| extents[*i as usize])
        .product::<u32>()
        .max(1);
    let axis_extent = extents[axis];
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
            lane_group: DEFAULT_BLOCK,
        },
    };
    let block = fold_block(strat, caps, axis_extent);
    let passes = axis_extent.div_ceil(block).max(1);
    let f32_ty = ScalarElement::F32.element();
    let space_total = extents
        .iter()
        .map(|e| u64::from(*e))
        .product::<u64>()
        .max(1);

    let row = b.builtin(Builtin::ProgramId(WorkgroupAxis::X));
    let lane = b.builtin(Builtin::Lane);
    let (outer_idx, inner_idx) = b.divrem(row.clone(), b.u32(inner));

    // One lifted value per lane at element `k`, each guarded to its own
    // identity outside the reduced extent: a shared identity would let a
    // padding lane count in Welford's constant `1` slot.
    //
    // Lane `(slot, p)` reads every operand at promoted position `p`. An
    // operand invariant in the promoted axes is read at the same address for
    // every position and the emitter's CSE collapses it to one load.
    let lift_at = |k: TileExpr, body: &mut Vec<Stmt>| -> Result<Vec<TileExpr>> {
        let mask = b.lt(k.clone(), b.u32(axis_extent));
        let row_elems = axis_extent.saturating_mul(vec_extent);
        let mut per_pos: Vec<(Vec<TileExpr>, Vec<TileExpr>)> =
            Vec::with_capacity(vec_extent as usize);
        let mut produced: rustc_hash::FxHashMap<TileExpr, TileExpr> = Default::default();
        let mut first_index = None;
        for p in 0..vec_extent {
            let within = b.add(b.mul(b.u32(p), b.u32(axis_extent)), k.clone());
            let flat = b.add(
                b.mul(
                    b.add(b.mul(outer_idx.clone(), b.u32(row_elems)), within),
                    b.u32(inner),
                ),
                inner_idx.clone(),
            );
            let first = first_index.get_or_insert_with(|| flat.clone()).clone();
            let mut args = Vec::with_capacity(ops.len());
            for (i, o) in ops.iter().enumerate() {
                args.push(match stream {
                    Some((producer, operand)) if i == operand as usize => {
                        let invariant = o.layout.rank() == space.rank()
                            && vec_axes.iter().all(|axis| {
                                o.layout.strides()[*axis as usize]
                                    .known_eq(fusor_ir::shape::Dim::Const(0))
                            });
                        let flat = if invariant { &first } else { &flat }.clone();
                        let row = super::address_of(&b, cx, o, flat, space_total)?;
                        match produced.get(&row) {
                            Some(value) => value.clone(),
                            None => {
                                let value = fold_element(
                                    &b,
                                    cx,
                                    &binds,
                                    producer,
                                    row.clone(),
                                    mask.clone(),
                                    body,
                                )?;
                                produced.insert(row, value.clone());
                                value
                            }
                        }
                    }
                    _ => operand_at(&b, cx, &binds, o, flat.clone(), space_total, mask.clone())?,
                });
            }
            let full = coords_of(&b, &flat, &extents);
            let coords = lanes.iter_axes.iter().map(|i| full[*i].clone()).collect();
            per_pos.push((args, coords));
        }
        lanes
            .slots
            .iter()
            .map(|&(slot, p)| {
                let (args, coords) = &per_pos[p as usize];
                let v = translate(args, coords, &carrier.lift[slot])?;
                Ok(TileExpr::new(
                    TileExprKind::Select {
                        condition: mask.clone(),
                        accept: v,
                        reject: b.f32(splat_f32(carrier.identity[slot])),
                    },
                    f32_ty,
                ))
            })
            .collect()
    };

    let mut body: Vec<Stmt> = Vec::new();
    let partials: Vec<TileExpr> = if passes > 1 {
        // The per-lane strided loop, carrying `lanes` accumulators seeded from
        // the carrier's identities and absorbed with its own `merge`.
        let index = b.local(ScalarElement::U32.element());
        let k = b.add(
            b.mul(b.load_local(index.clone()), b.u32(block)),
            lane.clone(),
        );
        let mut loop_body = Vec::new();
        let values = lift_at(k, &mut loop_body)?;
        let locals: Vec<_> = (0..lanes.lanes()).map(|_| b.local(f32_ty)).collect();
        let reads: Vec<TileExpr> = locals.iter().map(|l| b.load_local(l.clone())).collect();
        let mut args = reads.clone();
        args.extend(values);
        let mut accumulators = Vec::with_capacity(lanes.lanes());
        for (slot, local) in locals.into_iter().enumerate() {
            accumulators.push(Accumulator {
                local,
                init: b.f32(splat_f32(lanes.identities[slot])),
                update: translate(&args, &[], &lanes.merges[slot])?,
            });
        }
        body.push(Stmt::Loop {
            count: Some(b.u32(passes)),
            index: Some(index),
            accumulators,
            body: loop_body,
        });
        reads
    } else {
        lift_at(lane.clone(), &mut body)?
    };

    let scratch = (0..lanes.lanes())
        .map(|_| b.tile("fold_scratch", f32_ty, &[block]))
        .collect();
    let fast = fusor_ir::ir::kernel::fast_reduce_op(carrier);
    let (reduce, reduced) = b.merge_tree(scratch, block, partials, fast, f32_ty, |args| {
        lanes
            .merges
            .iter()
            .map(|m| translate(args, &[], m))
            .collect()
    })?;
    body.push(reduce);

    let out = view(&binds.of(cx.launch.root)?);
    let mask = b.eq(lane, b.u32(0));
    let base = b.mul(row, b.u32(lanes.lanes() as u32));
    for (slot, post) in lanes.posts.iter().enumerate() {
        body.push(Stmt::Store {
            dst: out.clone(),
            addr: Addr::Linear(b.add(base.clone(), b.u32(slot as u32))),
            value: translate(&reduced, &[], post)?,
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

fn fold_element(
    b: &Kernel,
    cx: &LowerCtx<'_>,
    binds: &Binds,
    producer: &Launch,
    row: TileExpr,
    mask: TileExpr,
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
    } = producer
    else {
        return Err(Error::Legality("a streamed producer must be a Fold".into()));
    };
    let extents = const_extents(cx, &space.dims)?;
    let axis = *axis as usize;
    let width = extents[axis];
    let inner = extents[axis + 1..].iter().product::<u32>().max(1);
    let total = extents.iter().map(|v| u64::from(*v)).product();
    let index = b.local(ScalarElement::U32.element());
    let k = b.load_local(index.clone());
    let flat = b.add(
        b.mul(
            b.add(b.mul(b.div(row.clone(), b.u32(inner)), b.u32(width)), k),
            b.u32(inner),
        ),
        b.rem(row.clone(), b.u32(inner)),
    );
    let coords = coords_of(b, &flat, &extents);
    let args = ops
        .iter()
        .map(|operand| operand_at(b, cx, binds, operand, flat.clone(), total, mask.clone()))
        .collect::<Result<Vec<_>>>()?;
    let uniforms = binds.buffers.first().cloned();
    let translate = |args: &[TileExpr], coords: &[TileExpr], e| {
        Translate {
            b,
            args,
            coords,
            uniforms: uniforms.clone(),
        }
        .run(e)
    };
    let lifted = translate(&args, &coords, &carrier.lift[0])?;
    let local = b.local(ScalarElement::F32.element());
    let value = b.load_local(local.clone());
    let update = translate(&[value.clone(), lifted], &[], &carrier.merge[0])?;
    body.push(Stmt::Loop {
        count: Some(b.u32(width)),
        index: Some(index),
        accumulators: vec![Accumulator {
            local,
            init: b.f32(splat_f32(carrier.identity[0])),
            update,
        }],
        body: Vec::new(),
    });
    let value = translate(&[value], std::slice::from_ref(&row), &post[0])?;
    let ty = ElementType::Scalar(super::elem_of(*acc)?);
    Ok(TileExpr::new(TileExprKind::Cast { value, to: ty }, ty))
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
