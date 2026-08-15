//! `KGather` and `KScatter`.
//!
//! Both `ScatterMode`s name one map and differ only in strategy. On a target
//! with no f32 atomic they share one nest: one lane per output element, a
//! counted loop over the updates. Every output element is written by exactly one
//! lane, so no atomic is needed and the result is bit-reproducible.
//!
//! Both nests read their lane tiling off `theta`. `KGather` and `KScatter` carry
//! the same elementwise `ScheduleDomain::Map` a `KMap` carries, and can use
//! `tm` elements per lane like the grid-strided register tile in `map_fold`,
//! amortizing the index read in scatter workloads.

use fusor2_ir::device::Caps;
use fusor2_ir::error::Error;
use fusor2_ir::ir::level0::ScatterCombine;
use fusor2_ir::ir::level1::{L1, SchedPoint, ScatterMode};
use fusor2_ir::ir::level2::{
    Accumulator, Addr, KernelIr, Local, LocalDecl, StorageView, Stmt, TileExpr, TileExprKind,
};
use fusor2_ir::ir::{Node, Op};
use fusor2_ir::scalar::{BinOp, CmpOp};
use fusor2_ir::target::LowerCtx;
use fusor2_ir::Result;
use std::sync::Arc;

use super::{
    bin, cmp, const_extents, default_block, global_lane, grid_for, lit_u32, u32_ty, Binds,
};

pub fn lower(caps: &Caps, node: &Node, theta: SchedPoint, cx: &LowerCtx<'_>) -> Result<KernelIr> {
    let Op::L1(op) = &node.op else {
        return Err(Error::Legality("not an L1 node".into()));
    };
    let tm = lane_tile(theta)?;
    match op {
        L1::KGather {
            space, axis, ops, ..
        } => gather(caps, cx, space, *axis, ops, tm),
        L1::KScatter {
            space,
            axis,
            mode,
            combine,
            ops,
            ..
        } => scatter(caps, cx, space, *axis, *mode, *combine, ops, tm),
        _ => Err(Error::Legality("gather_scatter got a foreign node".into())),
    }
}

/// How many output elements one lane owns, read off `theta`.
///
/// [`SchedPoint::Point`] is the floor lowering's untiled point — the fallback
/// that guarantees every chain reaches a valid L1 form when the saturation
/// budget is exhausted — so it is answered with 1 rather than refused. Any
/// other family on these nodes is a planner bug and says so.
///
/// `MapTiling::dim` does not enter: this backend tiles with a **grid stride**
/// (`flat + t * grid.x * block`), exactly as `lower_map` does, so one lane's
/// elements are a fixed distance apart whatever axis the domain named. That
/// keeps coverage a bijection with no divisibility side condition, at the
/// price of not distinguishing two points that differ only in `dim`.
/// `MapTiling::vector` does not enter either: `emit::pick_width` chooses the
/// SIMD instantiation from `caps.simd_widths` and the block width, so a width
/// asserted here would be a second, disagreeing decision rather than a
/// consumed one.
fn lane_tile(theta: SchedPoint) -> Result<u32> {
    match theta {
        SchedPoint::Map(t) => Ok(t.tm.max(1)),
        SchedPoint::Point => Ok(1),
        other => Err(Error::Legality(format!(
            "a gather or scatter needs SchedPoint::Map, got {other:?}"
        ))),
    }
}

fn view(buf: &Arc<fusor2_ir::ir::level2::BufferDecl>) -> StorageView {
    StorageView {
        buffer: Arc::clone(buf),
        offset: 0,
        layout: buf.layout.clone(),
    }
}

/// `out[i, rest] = src[idx[i], rest]`, one lane per output element.
///
/// Both `GatherMode`s share this nest; they differ only in how many output
/// elements one lane owns, which is a schedule attribute rather than a
/// different kernel.
fn gather(
    caps: &Caps,
    cx: &LowerCtx<'_>,
    space: &fusor2_ir::ir::level1::IndexSpace,
    axis: u32,
    ops: &[fusor2_ir::ir::level1::Operand],
    tm: u32,
) -> Result<KernelIr> {
    if ops.len() < 2 {
        return Err(Error::Legality(
            "a gather needs a source and an index operand".into(),
        ));
    }
    let binds = Binds::build(cx)?;
    let extents = const_extents(&space.dims)?;
    let n: u64 = extents.iter().map(|e| *e as u64).product::<u64>().max(1);
    let axis = axis as usize;
    if axis >= extents.len() {
        return Err(Error::Legality("gather axis is out of range".into()));
    }
    let inner: u32 = extents[axis + 1..].iter().product::<u32>().max(1);
    let out_stride = extents[axis].max(1) * inner;
    // The source's extent along the gathered axis, which is the *only* axis
    // where source and output disagree. Scaling the source's outer coordinate
    // by the output's stride reads the wrong row whenever the index vector is
    // not exactly as long as the axis it indexes — the common case for a
    // table expansion, an upsample run or a narrow.
    let src_shape = const_extents(ops[0].layout.shape())?;
    let src_axis = *src_shape
        .get(axis)
        .ok_or_else(|| Error::Legality("gather axis is out of range for the source".into()))?;
    let src_stride = src_axis.max(1) * inner;

    let src = super::operand_src(cx, &binds, ops[0].src)?;
    let idx = super::operand_src(cx, &binds, ops[1].src)?;
    let out = binds.of(cx.launch.root)?;

    // `tm` elements per lane, a whole grid apart, so lane 0..stride covers
    // [0, tm*stride) >= [0, n) exactly once with no divisibility condition.
    let block = default_block(caps);
    let grid = grid_for(n.div_ceil(u64::from(tm)), block);
    let stride = grid[0].saturating_mul(block);

    let mut body = Vec::with_capacity(tm as usize);
    for t in 0..tm {
        let flat = if t == 0 {
            global_lane(block)
        } else {
            bin(
                BinOp::Add,
                global_lane(block),
                lit_u32(t.saturating_mul(stride)),
                u32_ty(),
            )
        };
        let mask = cmp(CmpOp::Lt, flat.clone(), lit_u32(n as u32));
        // Split the flat output index into (outer, gathered, inner).
        let outer = bin(BinOp::Div, flat.clone(), lit_u32(out_stride), u32_ty());
        let rest = bin(BinOp::Rem, flat.clone(), lit_u32(out_stride), u32_ty());
        let g = bin(BinOp::Div, rest.clone(), lit_u32(inner), u32_ty());
        let within = bin(BinOp::Rem, rest, lit_u32(inner), u32_ty());

        let row = idx.at(g, mask.clone());
        // The gathered coordinate replaces `g`; everything else is unchanged —
        // but the outer coordinate steps by the *source's* stride.
        let src_index = bin(
            BinOp::Add,
            bin(
                BinOp::Add,
                bin(BinOp::Mul, outer, lit_u32(src_stride), u32_ty()),
                bin(BinOp::Mul, row, lit_u32(inner), u32_ty()),
                u32_ty(),
            ),
            within,
            u32_ty(),
        );
        let value = src.at(src_index, mask.clone());
        body.push(Stmt::Store {
            dst: view(&out),
            addr: Addr::Linear(flat),
            value,
            mask,
        });
    }

    Ok(KernelIr {
        buffers: binds.buffers,
        grid,
        block,
        body,
        byte_arena: None,
        name: "cpu_gather",
    })
}

/// `out = base` with `out[.., idx[u], ..] (combine)= upd[.., u, ..]`.
///
/// **The nest walks the output, not the updates.** A `KScatter`'s value is its
/// *base* with the updates applied, and the plan gives that value its own
/// buffer — nothing copies the base in beforehand — so a kernel that only
/// visits the written elements leaves every other one undefined. That is what
/// made `cat`, `stack`, `pad`, `repeat` and `slice_assign` come back as zeros.
///
/// One lane per output element, a counted loop over the updates, and the
/// accumulator carried in a register: the write map is not injective, so the
/// nest declares an associative `combine` (`verify_l1` invariant 3) and
/// discharges it by making each output element the *only* writer of itself.
/// The accumulation order is therefore fixed and the result bit-reproducible
/// at any thread count — no atomic, on a target that has none for f32.
///
/// **`tm` output elements per lane, in one loop.** The counted loop costs one
/// `idx[u]` read per output element per update; `tm` accumulators in the same
/// loop share that read, so the index traffic that dominates the embedding
/// gradient falls by `tm` while the arithmetic is unchanged.
fn scatter(
    caps: &Caps,
    cx: &LowerCtx<'_>,
    space: &fusor2_ir::ir::level1::IndexSpace,
    axis: u32,
    _mode: ScatterMode,
    combine: ScatterCombine,
    ops: &[fusor2_ir::ir::level1::Operand],
    tm: u32,
) -> Result<KernelIr> {
    // Either mode names a *strategy* for the same map. This nest needs
    // no atomic, so `Atomic{Add}` is legal here even though `caps.atomic_f32`
    // is false: refusing it made every `Set` scatter unrunnable on the CPU,
    // and refusing `Add` made embedding backward unrunnable.
    if ops.len() < 3 {
        return Err(Error::Legality(
            "a scatter needs base, index and update operands".into(),
        ));
    }
    let _ = caps;
    let binds = Binds::build(cx)?;
    let geom = super::scatter_geometry(cx, space, axis, ops)?;
    let (outer, bins, inner, updates) = (geom.outer, geom.bins, geom.inner, geom.updates);
    let total = outer as u64 * bins as u64 * inner as u64;

    let base = super::operand_src(cx, &binds, ops[0].src)?;
    let idx = super::operand_src(cx, &binds, ops[1].src)?;
    let upd = super::operand_src(cx, &binds, ops[2].src)?;
    let out = binds.of(cx.launch.root)?;
    let elem = out.element;

    let block = default_block(caps);
    let grid = grid_for(total.div_ceil(u64::from(tm)), block);
    let lane_stride = grid[0].saturating_mul(block);

    let u_local: Local = Arc::new(LocalDecl::new(u32_ty()));
    let u = TileExpr::new(TileExprKind::LoadLocal(Arc::clone(&u_local)), u32_ty());

    // The lowest offset is live whenever any of this lane's offsets is, so it
    // is the right mask for the one index read they share.
    let first_live = cmp(CmpOp::Lt, global_lane(block), lit_u32(total as u32));
    let u_bin = idx.at(u.clone(), first_live);

    let mut accumulators = Vec::with_capacity(tm as usize);
    let mut stores = Vec::with_capacity(tm as usize);
    for t in 0..tm {
        let flat = if t == 0 {
            global_lane(block)
        } else {
            bin(
                BinOp::Add,
                global_lane(block),
                lit_u32(t.saturating_mul(lane_stride)),
                u32_ty(),
            )
        };
        let live = cmp(CmpOp::Lt, flat.clone(), lit_u32(total as u32));
        // (outer, destination bin, inner) of this output element.
        let o = bin(BinOp::Div, flat.clone(), lit_u32(bins * inner), u32_ty());
        let dest = bin(
            BinOp::Rem,
            bin(BinOp::Div, flat.clone(), lit_u32(inner), u32_ty()),
            lit_u32(bins),
            u32_ty(),
        );
        let within = bin(BinOp::Rem, flat.clone(), lit_u32(inner), u32_ty());

        let acc_local: Local = Arc::new(LocalDecl::new(elem));
        let acc = TileExpr::new(TileExprKind::LoadLocal(Arc::clone(&acc_local)), elem);

        let hit = cmp(CmpOp::Eq, u_bin.clone(), dest);
        // `upd[o, u, within]` in the update's own flat space.
        let upd_index = bin(
            BinOp::Add,
            bin(
                BinOp::Mul,
                bin(
                    BinOp::Add,
                    bin(BinOp::Mul, o, lit_u32(updates), u32_ty()),
                    u.clone(),
                    u32_ty(),
                ),
                lit_u32(inner),
                u32_ty(),
            ),
            within,
            u32_ty(),
        );
        let contribution = upd.at(upd_index, live.clone());
        let combined = match combine {
            // `Add` duplicates accumulate — normative: an embedding table
            // receiving one token twice gets the summed gradient. `Set` is only
            // reachable when the node proved its indices unique.
            ScatterCombine::Add => bin(BinOp::Add, acc.clone(), contribution, elem),
            ScatterCombine::Set => contribution,
        };
        let update = TileExpr::new(
            TileExprKind::Select {
                condition: hit,
                accept: combined,
                reject: acc.clone(),
            },
            elem,
        );

        accumulators.push(Accumulator {
            local: Arc::clone(&acc_local),
            init: base.at(flat.clone(), live.clone()),
            update,
        });
        stores.push(Stmt::Store {
            dst: view(&out),
            addr: Addr::Linear(flat),
            value: acc,
            mask: live,
        });
    }

    let mut body = vec![Stmt::Loop {
        count: Some(lit_u32(updates)),
        index: Some(u_local),
        accumulators,
        body: Vec::new(),
    }];
    body.extend(stores);

    Ok(KernelIr {
        buffers: binds.buffers,
        grid,
        block,
        body,
        byte_arena: None,
        name: "cpu_scatter",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use fusor2_ir::cost::Picoseconds;
    use fusor2_ir::dtype::Dtype;
    use fusor2_ir::egraph::{EGraph, Id};
    use fusor2_ir::extract::{BindKind, BindingPlan, Extraction, Launch, Plan, PlanHash};
    use fusor2_ir::ir::level0::{BufferId, LeafKind, L0};
    use fusor2_ir::ir::level1::{
        AccessPlan, IndexSpace, MapTiling, Operand, ScheduleDomain,
    };
    use fusor2_ir::semantics::CoreSemantics;
    use fusor2_ir::shape::{Dim, Layout};
    use smallvec::SmallVec;

    use crate::alloc::AlignedBuf;
    use crate::emit::CpuKernel;
    use fusor2_ir::target::{Buf, Uniforms};

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
                // Placeholder: `lower` recomputes the width off `caps`.
                block: 256,
            }],
            buffers: Vec::new(),
            symbols: Vec::new(),
            hash: PlanHash(0),
            cost: Picoseconds(0),
        }
    }

    fn run_kernel(ir: &KernelIr, ins: &[Vec<f32>], idx: &[u32], out_len: usize) -> Vec<f32> {
        let art = crate::emit::compile(ir, crate::caps::cpu_caps(), None).expect("compiles");
        let kernel = CpuKernel {
            name: art.name,
            block: art.block,
            vector_width: art.prog.width,
            artifact: art,
        };
        let mk_f32 = |v: &[f32]| {
            let mut b = AlignedBuf::zeroed(v.len() * 4).expect("alloc");
            b.as_mut_slice().copy_from_slice(bytemuck::cast_slice(v));
            Buf::new(b)
        };
        let mut ib = AlignedBuf::zeroed(idx.len() * 4).expect("alloc");
        ib.as_mut_slice()
            .copy_from_slice(bytemuck::cast_slice(idx));
        let outb = Buf::new(AlignedBuf::zeroed(out_len * 4).expect("alloc"));
        let mut binds: Vec<Buf> = Vec::new();
        for (i, v) in ins.iter().enumerate() {
            if i == 1 {
                binds.push(Buf::new(std::mem::replace(
                    &mut ib,
                    AlignedBuf::zeroed(4).expect("alloc"),
                )));
            } else {
                binds.push(mk_f32(v));
            }
        }
        binds.push(outb.clone());
        crate::launch::run(&kernel, ir.grid, &binds, &Uniforms::default()).expect("runs");
        let ab = outb.downcast_ref::<AlignedBuf>().expect("aligned");
        bytemuck::cast_slice::<u8, f32>(ab.as_slice()).to_vec()
    }

    /// `bins x inner` destination, `updates` rows of updates, run at one
    /// schedule point.
    fn scatter_at(theta: SchedPoint, bins: u64, inner: u64, updates: u64) -> (KernelIr, Vec<f32>) {
        let mut g = EGraph::new(CoreSemantics::new(std::sync::Arc::new(
            fusor2_ir::semantics::SumArenaPlanner,
        )));
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
        let ir = lower(&crate::caps::cpu_caps(), g.node(k), theta, &cx).expect("lowers");

        let indices: Vec<u32> = (0..updates)
            .map(|i| ((i * 7 + 3) % bins) as u32)
            .collect();
        let basev = vec![0.0f32; (bins * inner) as usize];
        let updv: Vec<f32> = (0..updates * inner).map(|i| (i % 13) as f32 * 0.5 - 3.0).collect();
        let got = run_kernel(
            &ir,
            &[basev, Vec::new(), updv.clone()],
            &indices,
            (bins * inner) as usize,
        );
        (ir, got)
    }

    fn reference_scatter(bins: u64, inner: u64, updates: u64) -> Vec<f32> {
        let mut want = vec![0.0f32; (bins * inner) as usize];
        for u in 0..updates {
            let b = (u * 7 + 3) % bins;
            for c in 0..inner {
                want[(b * inner + c) as usize] += (((u * inner + c) % 13) as f32) * 0.5 - 3.0;
            }
        }
        want
    }

    fn gather_at(theta: SchedPoint, rows: u64, src_rows: u64, width: u64) -> (KernelIr, Vec<f32>) {
        let mut g = EGraph::new(CoreSemantics::new(std::sync::Arc::new(
            fusor2_ir::semantics::SumArenaPlanner,
        )));
        let src = leaf(&mut g, 0, Dtype::F32, &[src_rows, width]);
        let idx = leaf(&mut g, 1, Dtype::U32, &[rows]);
        let k = g
            .add(Op::L1(L1::KGather {
                space: IndexSpace::new(dims(&[rows, width]).into_iter()),
                axis: 0,
                mode: fusor2_ir::ir::level1::GatherMode::RowPerGroup,
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
        let ir = lower(&crate::caps::cpu_caps(), g.node(k), theta, &cx).expect("lowers");

        let indices: Vec<u32> = (0..rows).map(|i| ((i * 5 + 1) % src_rows) as u32).collect();
        let srcv: Vec<f32> = (0..src_rows * width).map(|i| i as f32 * 0.25).collect();
        let got = run_kernel(
            &ir,
            &[srcv, Vec::new()],
            &indices,
            (rows * width) as usize,
        );
        (ir, got)
    }

    fn map_point(dim: Option<u32>, tm: u32, vector: u32) -> SchedPoint {
        SchedPoint::Map(MapTiling { dim, tm, vector })
    }

    // -- the schedule point is read, and reading it changes nothing numeric --

    /// The whole point of the change: `theta` is consumed, and every point of
    /// the real domain computes the same answer. A tiling that is *read* but
    /// wrong is worse than one that is ignored, so this runs the kernel.
    #[test]
    fn every_schedule_point_scatters_the_same_values() {
        let want = reference_scatter(64, 8, 32);
        for tm in [1u32, 2, 4, 8, 3] {
            let (_, got) = scatter_at(map_point(Some(0), tm, 1), 64, 8, 32);
            assert_eq!(got, want, "tm = {tm}");
        }
        // and the untiled floor point agrees with all of them
        let (_, floor) = scatter_at(SchedPoint::Point, 64, 8, 32);
        assert_eq!(floor, want);
    }

    #[test]
    fn every_schedule_point_gathers_the_same_values() {
        let (_, want) = gather_at(SchedPoint::Point, 64, 128, 8);
        for tm in [1u32, 2, 4, 8, 3] {
            let (_, got) = gather_at(map_point(Some(0), tm, 1), 64, 128, 8);
            assert_eq!(got, want, "tm = {tm}");
        }
        // and the values are the gather, not merely self-consistent
        let srcv: Vec<f32> = (0..128 * 8).map(|i| i as f32 * 0.25).collect();
        for r in 0..64usize {
            let row = (r * 5 + 1) % 128;
            for c in 0..8usize {
                assert_eq!(want[r * 8 + c], srcv[row * 8 + c]);
            }
        }
    }

    /// The tiling is not decoration: at `tm` the kernel carries `tm`
    /// accumulators in one loop and launches `tm` times fewer lanes.
    #[test]
    fn a_tiled_scatter_carries_one_accumulator_per_element_and_a_smaller_grid() {
        // 8,192 output elements: a whole number of 256-lane groups at both
        // tilings, so the grid ratio is the tile and not a rounding artifact.
        let (untiled, a) = scatter_at(map_point(None, 1, 1), 1024, 8, 8);
        let (tiled, b) = scatter_at(map_point(Some(0), 4, 1), 1024, 8, 8);
        assert_eq!(a, b);
        let accs = |ir: &KernelIr| match &ir.body[0] {
            Stmt::Loop { accumulators, .. } => accumulators.len(),
            other => panic!("expected the update loop, got {other:?}"),
        };
        assert_eq!(accs(&untiled), 1);
        assert_eq!(accs(&tiled), 4);
        assert_eq!(untiled.body.len(), 2, "one loop, one store");
        assert_eq!(tiled.body.len(), 5, "one loop, four stores");
        assert_eq!(
            untiled.grid[0],
            tiled.grid[0] * 4,
            "four elements per lane is four times fewer lanes"
        );
    }

    #[test]
    fn a_tiled_gather_stores_one_element_per_tile_step() {
        let (untiled, _) = gather_at(map_point(None, 1, 1), 512, 128, 8);
        let (tiled, _) = gather_at(map_point(Some(0), 4, 1), 512, 128, 8);
        assert_eq!(untiled.body.len(), 1);
        assert_eq!(tiled.body.len(), 4);
        assert_eq!(untiled.grid[0], tiled.grid[0] * 4);
    }

    /// `SchedPoint::Point` is the floor's guarantee and must stay answerable;
    /// a point from another family is a planner bug and must not be silently
    /// treated as untiled.
    #[test]
    fn the_floor_point_is_answered_and_a_foreign_family_is_refused() {
        use fusor2_ir::ir::level1::FoldStrat;
        assert_eq!(lane_tile(SchedPoint::Point).expect("floor point"), 1);
        assert_eq!(lane_tile(map_point(Some(0), 8, 4)).expect("map point"), 8);
        assert!(lane_tile(SchedPoint::Fold(FoldStrat::Subgroup)).is_err());
    }

    /// The domain these nodes carry is non-trivial on the trainer's embedding
    /// gradient — 1,024 bins x 768 units, 384 updates — so there is a real
    /// choice to make. `map_domain` is the generator the scatter rules feed
    /// `KScatter::sched` from, read here on that exact shape.
    #[test]
    fn the_embedding_gradient_scatter_carries_a_real_domain() {
        let caps = crate::caps::cpu_caps();
        let cx = fusor2_tile::domains::DomainCtx::new(caps, fusor2_tile::Planner::global());
        let dom = fusor2_tile::domains::map_domain(&dims(&[384, 768]), &[], &cx);
        assert!(
            dom.tilings.len() > 1,
            "a one-point domain is a decision already made: {:?}",
            dom.tilings
        );
        assert!(dom.tilings.iter().any(|t| t.tm > 1));
        // every point of it lowers, so no selectable point can fail the plan
        for t in &dom.tilings {
            let (_, got) = scatter_at(SchedPoint::Map(*t), 64, 8, 32);
            assert_eq!(got, reference_scatter(64, 8, 32), "point {t:?}");
        }
    }

    /// Different device caps generate different domains, so the point a plan
    /// resolves to is a function of the device rather than of this file. The
    /// CPU's SIMD widths multiply the domain; a device with none does not
    /// offer a vectorized point at all.
    #[test]
    fn device_caps_change_the_domain_the_scatter_is_scheduled_over() {
        let mut narrow = crate::caps::cpu_caps().clone();
        narrow.simd_widths = SmallVec::new();
        let wide = crate::caps::cpu_caps().clone();
        let planner = fusor2_tile::Planner::global();
        let shape = dims(&[384, 768]);
        let a = fusor2_tile::domains::map_domain(
            &shape,
            &[],
            &fusor2_tile::domains::DomainCtx::new(&narrow, planner),
        );
        let b = fusor2_tile::domains::map_domain(
            &shape,
            &[],
            &fusor2_tile::domains::DomainCtx::new(&wide, planner),
        );
        assert!(a.tilings.iter().all(|t| t.vector == 1));
        assert!(b.tilings.iter().any(|t| t.vector > 1));
        assert_ne!(a.tilings, b.tilings);
    }

    /// A shorter axis than the tile is a smaller domain: the generator does
    /// not offer a `tm` the shape cannot fill, so two shapes reach extraction
    /// with different candidate sets.
    #[test]
    fn a_short_axis_offers_fewer_points_than_a_long_one() {
        let caps = crate::caps::cpu_caps();
        let cx = fusor2_tile::domains::DomainCtx::new(caps, fusor2_tile::Planner::global());
        let long = fusor2_tile::domains::map_domain(&dims(&[384, 768]), &[], &cx);
        let short = fusor2_tile::domains::map_domain(&dims(&[3, 768]), &[], &cx);
        let tms = |d: &fusor2_ir::ir::level1::MapDomain| {
            let mut v: Vec<u32> = d.tilings.iter().filter(|t| t.dim.is_some()).map(|t| t.tm).collect();
            v.sort_unstable();
            v.dedup();
            v
        };
        assert_eq!(tms(&long), vec![2, 4, 8]);
        assert_eq!(tms(&short), vec![2]);
    }

}
