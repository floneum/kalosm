//! `KMerged`: one dispatch over several segments sharing a `MergeKey`, plus
//! the `KRegion` multi-output form.
//!
//! Segments are epilogue-free **by construction** — `KMerged::new` refuses a
//! segment carrying one — so there is no per-segment epilogue identity to
//! emit and no `merge_profile -> None` policy hiding in a categorizer. The
//! un-merged epilogue-carrying contraction stays a live alternative in the
//! same chain, so merging and epilogue fusion compete on cost.
//!
//! ## The schedule
//!
//! Both bodies ran at a hardcoded `BLOCK = 256` and took `_theta`, so the
//! compiler decided to fuse by cost and then executed the fused kernel at a
//! fixed geometry — the one node family the architecture calls its own
//! fusion primitive was the one opting out of late decision-making.
//!
//! Both variants now carry a `sched` field, so
//! [`fusor2_ir::ir::level1::L1::schedule`] returns their domain and
//! extraction resolves it like any other node's. [`merged_domain`] and
//! [`region_domain`] are this crate's spelling of that domain — both are
//! [`MapDomain::linear`], the one generator `fusor2-ir` mints the field with
//! and `verify_l1` checks it against — and both bodies consume the selected
//! [`MapTiling`]: `tm` outputs per lane at stride `block`, over a workgroup
//! width that is a whole number of subgroups and never wider than the work
//! there is. Nothing here is a constant any more: the last one, the `256`
//! ceiling, is `fusor2_tile::domains::emitted_block`, which is where the
//! fold domain that prices the width already reads it.
//!
//! Owned by W9.

use fusor2_ir::Result;
use fusor2_ir::device::Caps;
use fusor2_ir::dtype::NumericContract;
use fusor2_ir::error::Error;
use fusor2_ir::ir::Node;
use fusor2_ir::ir::level1::{KMerged, L1, MapDomain, MapTiling, SchedPoint, ScheduleDomain, WaveCat};
use fusor2_ir::ir::level2::{
    Accumulator, Addr, Builtin, ElementType, KernelIr, ScalarElement, Stmt, TileBinaryOp,
    TileCompareOp, TileExpr, WorkgroupAxis,
};
use fusor2_ir::shape::Dim;
use fusor2_ir::target::LowerCtx;

use crate::lower::{Ctx, DimBinding, distribute_workgroups};

// ---------------------------------------------------------------------------
// The schedule domain
// ---------------------------------------------------------------------------

/// The workgroup width for a linear body needing `lanes` lanes: a whole
/// number of subgroups, never wider than the work and never wider than the
/// device allows.
///
/// This is the term that used to be the constant 256. A 64-row merged wave
/// launched 256 lanes and idled three quarters of them on every workgroup;
/// an 8-element region launched 256 lanes for 8 stores.
///
/// The ceiling is [`fusor2_tile::domains::emitted_block`] at a lane group of
/// one — the single source of the default width, shared with the fold domain
/// that prices it and with the CPU backend, so the policy number lives beside
/// `BLOCK_CHOICES` rather than being re-spelled here. Past it the
/// guard-per-segment shape stops paying anyway: every workgroup evaluates
/// every segment's guard, so a wider group buys idle lanes, not occupancy.
pub fn block_for(caps: &Caps, lanes: u64) -> u32 {
    let cap = fusor2_tile::domains::emitted_block(1, caps)
        .min(caps.limits.max_compute_workgroup_size[0])
        .max(1);
    let sgw = caps.subgroup_width().max(1).min(cap);
    let want = u32::try_from(lanes.max(1).min(u64::from(cap))).unwrap_or(cap);
    want.div_ceil(sgw).max(1).saturating_mul(sgw).min(cap)
}

/// Every tiling worth scoring over `elements` outputs.
///
/// One call to [`MapDomain::linear`]: both bodies walk one linearized index,
/// which is exactly the shape that generator describes, and it is the same
/// call `rules::merge` mints the node's `sched` with and `verify_l1` checks
/// the node against. Two derivations of one domain is the drift this
/// delegation exists to prevent.
pub fn linear_domain(caps: &Caps, elements: u64) -> ScheduleDomain {
    ScheduleDomain::Map(MapDomain::linear(caps, elements))
}

/// Outputs one segment of `wave` writes.
///
/// Every segment shares the `MergeKey`, and that key *is* the members'
/// common index space — `rules::merge::segment_of` builds it out of each
/// candidate's own `space`/extents — so this is one number rather than a
/// per-segment table.
pub fn segment_elements(wave: &KMerged, binding: &DimBinding) -> Result<u64> {
    let key = wave.key();
    let get = |d: Dim| -> Result<u64> { binding.require(d).map(|v| v.max(1)) };
    let m = get(key.m)?;
    let n = get(key.n)?;
    let batch = get(key.batch)?;
    Ok(match wave.category() {
        WaveCat::Region => m.saturating_mul(n),
        WaveCat::Row => m.saturating_mul(batch),
        WaveCat::Matmul | WaveCat::MatmulSplitK => {
            m.saturating_mul(n).saturating_mul(batch)
        }
    }
    .max(1))
}

/// The schedule domain of a merged wave, derived from the segments' shared
/// index space and the device caps.
pub fn merged_domain(caps: &Caps, wave: &KMerged, binding: &DimBinding) -> Result<ScheduleDomain> {
    Ok(linear_domain(caps, segment_elements(wave, binding)?))
}

/// The schedule domain of a region: the members' shared index space is the
/// live-outs' element count, and every member writes one value per point of
/// it.
pub fn region_domain(caps: &Caps, elements: u64) -> ScheduleDomain {
    linear_domain(caps, elements)
}

/// Read the selected point.
///
/// `Point` is the untiled member of this node's own domain: it is what a
/// launch whose root extraction never scheduled falls back to
/// (`Session::run` supplies it when `Extraction::theta` has no entry), and it
/// is a real member of every domain [`MapDomain::linear`] generates, so the
/// floor is a point the search could itself have picked rather than a
/// geometry from outside the space. A `Map` point naming an axis is refused
/// rather than silently ignored: this body has no axis to tile.
fn tiling_of(theta: SchedPoint) -> Result<MapTiling> {
    match theta {
        SchedPoint::Point => Ok(MapTiling {
            dim: None,
            tm: 1,
            vector: 1,
        }),
        SchedPoint::Map(t) if t.dim.is_none() => Ok(MapTiling {
            tm: t.tm.max(1),
            ..t
        }),
        SchedPoint::Map(t) => Err(Error::Plan(format!(
            "a merged wave walks one linearized index and has no axis {:?} to tile",
            t.dim
        ))),
        other => Err(Error::Plan(format!(
            "a merged wave needs SchedPoint::Map, got {other:?}"
        ))),
    }
}

/// Contract-shaped entry point (see CONTRACTS.md §4.10).
pub fn lower(caps: &Caps, node: &Node, theta: SchedPoint, cx: &LowerCtx<'_>) -> Result<KernelIr> {
    let fusor2_ir::ir::Op::L1(op) = &node.op else {
        return Err(Error::Plan("merged got a foreign node".into()));
    };
    let ctx = Ctx::new(caps, cx, DimBinding::new())?;
    match op {
        L1::KMerged(m) => lower_kmerged(ctx, m, theta),
        L1::KRegion { .. } => lower_kregion(ctx, op, theta),
        _ => Err(Error::Plan("merged got a foreign node".into())),
    }
}

/// Lower a merged wave for one of the four [`WaveCat`]s.
///
/// One body, a per-segment guard on the linearized segment index, each
/// segment reading its own operand slice out of the launch's binding list.
///
/// The geometry is `theta`'s: `tm` outputs per lane at stride `block`, over a
/// workgroup sized to the segments' shared index space. A tile is walked at
/// stride `block` rather than contiguously so that consecutive lanes keep
/// writing consecutive addresses at every step of the tile.
pub fn lower_kmerged(mut ctx: Ctx<'_>, wave: &KMerged, theta: SchedPoint) -> Result<KernelIr> {
    let segments = wave.segments();
    if segments.is_empty() {
        return Err(Error::Plan("a merged wave has no segments".into()));
    }
    let key = wave.key();
    let k = u32::try_from(ctx.binding.require(key.k)?)
        .map_err(|_| Error::Plan("merged extent exceeds a u32".into()))?
        .max(1);
    let acc_elem = crate::lower::scalar_element(key.dtype);

    // Per-segment work, in elements. Every segment shares the merge key, so
    // this is one number rather than a per-segment table, and it is what the
    // schedule domain is derived from.
    let elements = segment_elements(wave, &ctx.binding)?;
    let per_segment = u32::try_from(elements)
        .map_err(|_| Error::Plan("merged extent exceeds a u32".into()))?;
    // A split-K wave asks for `splits` partial reductions, and this body has
    // nowhere to combine them: every split group iterates `k/splits` terms
    // from `k = 0` and stores the result as if it were the whole sum, while
    // the extra groups are masked off and do nothing. That is a partial sum
    // wearing an answer's clothes. `rules::merge::segment_of` mints
    // `splits: 1` on every segment it builds, so nothing reaches here;
    // refusing is honest where computing is not.
    if key.splits > 1 {
        return Err(Error::Plan(format!(
            "a merged wave asks for {} split-K partials and this body cannot combine \
             them; the split belongs in the contraction's own schedule domain",
            key.splits
        )));
    }

    let tm = tiling_of(theta)?.tm;
    let block = block_for(ctx.caps, elements.div_ceil(u64::from(tm)));
    // What one workgroup covers. At `tm == 1` this is `block`, so the whole
    // index expression below collapses to the one that shipped.
    let per_group = block.saturating_mul(tm);

    let groups_per_segment = per_segment.div_ceil(per_group).max(1);
    let total_groups = groups_per_segment.saturating_mul(segments.len() as u32);

    let mut body: Vec<Stmt> = Vec::new();
    let lane = ctx.b.builtin(Builtin::Lane);
    let max_dim = ctx.caps.limits.max_compute_workgroups_per_dimension;
    // The grid up front: the workgroup id is linearized against **this**
    // grid's `x` and `x*y`, not against `max_dim`. `distribute_workgroups`
    // picks the slab count first and sizes `x` to the slab, so `x` is
    // `max_dim` only when the launch happens to divide that way. One
    // linearization, shared with `Ctx::global_index`, so the two cannot drift.
    let grid = distribute_workgroups(total_groups, max_dim);
    let linear_group = {
        let gx = ctx.b.builtin(Builtin::ProgramId(WorkgroupAxis::X));
        let gy = ctx.b.builtin(Builtin::ProgramId(WorkgroupAxis::Y));
        let gz = ctx.b.builtin(Builtin::ProgramId(WorkgroupAxis::Z));
        let x_e = ctx.b.u32(grid[0].max(1));
        let xy_e = ctx.b.u32(grid[0].max(1).saturating_mul(grid[1].max(1)));
        let yx = ctx.b.mul(gy, x_e);
        let zxy = ctx.b.mul(gz, xy_e);
        let g = ctx.b.add(gx, yx);
        ctx.b.add(g, zxy)
    };
    let gps = ctx.b.u32(groups_per_segment);
    let segment = ctx.b.binary(
        TileBinaryOp::Div,
        linear_group.clone(),
        gps.clone(),
        NumericContract::RELAXED,
    );
    let within = ctx
        .b
        .binary(TileBinaryOp::Rem, linear_group, gps, NumericContract::RELAXED);
    let group_e = ctx.b.u32(per_group);
    let tile_base = {
        let base = ctx.b.mul(within, group_e);
        ctx.b.add(base, lane)
    };
    let bound = ctx.b.u32(per_segment);
    // The tile's own first element decides whether the workgroup does
    // anything at all: `tile_base + t*block` only grows with `t`.
    let tile_live = ctx
        .b
        .compare(TileCompareOp::Lt, tile_base.clone(), bound.clone());

    // Element `t` of this lane's tile, and the mask that keeps it in range.
    let mut offsets: Vec<TileExpr> = Vec::with_capacity(tm as usize);
    let mut in_range: Vec<TileExpr> = Vec::with_capacity(tm as usize);
    for t in 0..tm {
        let off = if t == 0 {
            tile_base.clone()
        } else {
            let step = ctx.b.u32(t.saturating_mul(block));
            ctx.b.add(tile_base.clone(), step)
        };
        in_range.push(ctx.b.compare(TileCompareOp::Lt, off.clone(), bound.clone()));
        offsets.push(off);
    }

    // Each segment gets its own guarded body. The guard is on the segment
    // index, not on a per-segment epilogue: `KMerged::new` already proved
    // every segment epilogue-free.
    for (slot, seg_id) in segments.iter().enumerate() {
        let slot_e = ctx.b.u32(slot as u32);
        let is_mine = ctx.b.compare(TileCompareOp::Eq, segment.clone(), slot_e);
        let guards: Vec<TileExpr> = in_range
            .iter()
            .map(|r| ctx.b.and(is_mine.clone(), r.clone()))
            .collect();

        let view = ctx
            .linear_view(*seg_id)
            .or_else(|_| ctx.linear_view(ctx.output()?))?;
        let out_elem = view.buffer.element;

        let values: Vec<TileExpr> = match wave.category() {
            // A region segment is an elementwise body; the operand slice is
            // the segment's own buffer.
            WaveCat::Region | WaveCat::Row => offsets
                .iter()
                .zip(&guards)
                .map(|(off, guard)| {
                    let fill = ctx.b.zero(acc_elem);
                    let v = ctx.b.load(
                        fusor2_ir::ir::level2::Source::Storage(view.clone()),
                        Addr::Linear(off.clone()),
                        guard.clone(),
                        fill,
                    );
                    ctx.b.cast(v, out_elem)
                })
                .collect(),
            // A matmul segment runs the shared k loop; `k` is a merge-key
            // field, so every segment iterates the same count. The whole tile
            // rides **one** loop with one accumulator per tile position —
            // that register reuse is the only reason `tm > 1` is worth
            // scoring here.
            WaveCat::Matmul | WaveCat::MatmulSplitK => {
                let k_index = ctx.b.local(ElementType::Scalar(ScalarElement::U32));
                let kk = ctx.b.load_local(k_index.clone());
                let mut accs: Vec<Accumulator> = Vec::with_capacity(tm as usize);
                let mut reads: Vec<TileExpr> = Vec::with_capacity(tm as usize);
                for (off, guard) in offsets.iter().zip(&guards) {
                    let acc_local = ctx.b.local(ElementType::Scalar(acc_elem));
                    let acc_read = ctx.b.load_local(acc_local.clone());
                    let init = ctx.b.zero(acc_elem);
                    let zero = ctx.b.zero(acc_elem);
                    let operand = ctx.b.load(
                        fusor2_ir::ir::level2::Source::Storage(view.clone()),
                        Addr::Rc2 {
                            row: off.clone(),
                            col: kk.clone(),
                        },
                        guard.clone(),
                        zero,
                    );
                    let operand = ctx.b.cast(operand, ElementType::Scalar(acc_elem));
                    let update = ctx.b.add(acc_read, operand);
                    reads.push(ctx.b.load_local(acc_local.clone()));
                    accs.push(Accumulator {
                        local: acc_local,
                        init,
                        update,
                    });
                }
                let count = ctx.b.u32(k);
                let live = ctx.b.and(is_mine.clone(), tile_live.clone());
                body.push(Stmt::If {
                    condition: live,
                    accept: vec![Stmt::Loop {
                        count: Some(count),
                        index: Some(k_index),
                        accumulators: accs,
                        body: Vec::new(),
                    }],
                    reject: Vec::new(),
                });
                reads
                    .into_iter()
                    .map(|total| ctx.b.cast(total, out_elem))
                    .collect()
            }
        };

        for ((off, guard), value) in offsets.iter().zip(guards).zip(values) {
            body.push(Stmt::Store {
                dst: view.clone(),
                addr: Addr::Linear(off.clone()),
                value,
                mask: guard,
            });
        }
    }

    let name = match wave.category() {
        WaveCat::Region => "merged_region",
        WaveCat::Row => "merged_row",
        WaveCat::Matmul => "merged_matmul",
        WaveCat::MatmulSplitK => "merged_matmul_split_k",
    };
    Ok(ctx.finish(name, grid, block, body))
}

/// A `KRegion` is the same rewrite as producer inlining, differing only in
/// that it emits an extra buffer per `live_out`. One pass over the shared
/// index space, several stores.
pub fn lower_kregion(mut ctx: Ctx<'_>, op: &L1, theta: SchedPoint) -> Result<KernelIr> {
    let L1::KRegion {
        members, live_outs, ..
    } = op
    else {
        return Err(Error::Plan("lower_kregion on a non-KRegion node".into()));
    };
    if members.is_empty() {
        return Err(Error::Plan("a region has no members".into()));
    }
    let limits = ctx.caps.limits;

    let out = ctx.output()?;
    let out_view = ctx.linear_view(out)?;
    let out_elem = out_view.buffer.element;
    let count = out_view.layout.element_count();
    let elements = u32::try_from(count)
        .map_err(|_| Error::Plan("region output exceeds a u32 element count".into()))?
        .max(1);

    // The members share one index space — that is what makes them a region —
    // so the live-outs' element count is the domain the tiling is derived
    // from.
    let tm = tiling_of(theta)?.tm;
    let block = block_for(ctx.caps, u64::from(elements).div_ceil(u64::from(tm)));
    let per_group = block.saturating_mul(tm);

    let grid = distribute_workgroups(
        elements.div_ceil(per_group),
        limits.max_compute_workgroups_per_dimension,
    );
    let base = ctx.global_index(block, grid);
    // At `tm == 1` a group covers exactly `block`, so the index expression is
    // the one that shipped, unwrapped.
    let tile_base = if tm == 1 {
        base
    } else {
        let block_e = ctx.b.u32(block);
        let group = ctx.b.binary(
            TileBinaryOp::Div,
            base.clone(),
            block_e.clone(),
            NumericContract::RELAXED,
        );
        let lane = ctx
            .b
            .binary(TileBinaryOp::Rem, base, block_e, NumericContract::RELAXED);
        let group_e = ctx.b.u32(per_group);
        let scaled = ctx.b.mul(group, group_e);
        ctx.b.add(scaled, lane)
    };
    let bound = ctx.b.u32(elements);

    let mut body = Vec::new();
    let zero_elem = match out_elem {
        ElementType::Scalar(s) => s,
        _ => ScalarElement::F32,
    };
    for t in 0..tm {
        let index = if t == 0 {
            tile_base.clone()
        } else {
            let step = ctx.b.u32(t.saturating_mul(block));
            ctx.b.add(tile_base.clone(), step)
        };
        let live = ctx
            .b
            .compare(TileCompareOp::Lt, index.clone(), bound.clone());

        // The shared value each member computes, read once into a register
        // and written to every live-out. That register reuse is the whole
        // point of a region: a later statement reads it without a second
        // load.
        let zero = ctx.b.zero(zero_elem);
        let shared = ctx.b.load(
            fusor2_ir::ir::level2::Source::Storage(out_view.clone()),
            Addr::Linear(index.clone()),
            live.clone(),
            zero,
        );
        let local = ctx.b.local(shared.element());
        body.push(Stmt::StoreLocal {
            dst: local.clone(),
            value: shared,
        });
        let value = ctx.b.load_local(local);

        for slot in live_outs {
            let member = members
                .get(*slot as usize)
                .copied()
                .ok_or_else(|| Error::Plan(format!("region live-out {slot} names no member")))?;
            let view = ctx.linear_view(member).unwrap_or_else(|_| out_view.clone());
            let v = ctx.b.cast(value.clone(), view.buffer.element);
            body.push(Stmt::Store {
                dst: view,
                addr: Addr::Linear(index.clone()),
                value: v,
                mask: live.clone(),
            });
        }
        if live_outs.is_empty() {
            let v = ctx.b.cast(value, out_elem);
            body.push(Stmt::Store {
                dst: out_view.clone(),
                addr: Addr::Linear(index),
                value: v,
                mask: live,
            });
        }
    }

    Ok(ctx.finish("kregion", grid, block, body))
}

#[cfg(test)]
mod tests {
    use super::*;
    use fusor2_ir::device::SubgroupWidths;
    use fusor2_ir::dtype::Dtype;
    use fusor2_ir::egraph::{EGraph, Id};
    use fusor2_ir::ir::level1::{Family, KMerged, MergeKey, MergeSegment, WaveCat};
    use fusor2_ir::ir::level2::{TileExprKind, TileLiteral};
    use fusor2_ir::shape::Dim;
    use smallvec::SmallVec;

    fn key() -> MergeKey {
        MergeKey {
            m: Dim::Const(128),
            n: Dim::Const(64),
            k: Dim::Const(256),
            batch: Dim::Const(1),
            splits: 1,
            dtype: Dtype::F32,
            family: Family::Coop,
        }
    }

    /// The domain [`key`]'s shape implies. The constructor tests below are
    /// about which segments are admissible, not about the geometry.
    fn dom() -> ScheduleDomain {
        linear_domain(&gpu_caps(32, 256), 128 * 64)
    }

    /// A merged body with per-segment epilogue identities is *unbuildable*:
    /// the illegal state is unrepresentable, so no categorizer policy is
    /// needed and epilogue fusion stays a live competing alternative.
    #[test]
    fn a_segment_with_an_epilogue_cannot_be_merged() {
        let segs = [
            MergeSegment {
                id: Id(1),
                key: key(),
                has_epilogue: false,
            },
            MergeSegment {
                id: Id(2),
                key: key(),
                has_epilogue: true,
            },
        ];
        assert!(KMerged::new(WaveCat::Matmul, segs, dom()).is_err());
    }

    #[test]
    fn segments_must_share_a_merge_key() {
        let other = MergeKey {
            n: Dim::Const(128),
            ..key()
        };
        let segs = [
            MergeSegment {
                id: Id(1),
                key: key(),
                has_epilogue: false,
            },
            MergeSegment {
                id: Id(2),
                key: other,
                has_epilogue: false,
            },
        ];
        assert!(KMerged::new(WaveCat::Matmul, segs, dom()).is_err());
    }

    #[test]
    fn a_three_segment_wave_keeps_its_segment_order() {
        let segs: Vec<_> = [1u32, 2, 3]
            .into_iter()
            .map(|i| MergeSegment {
                id: Id(i),
                key: key(),
                has_epilogue: false,
            })
            .collect();
        let wave = KMerged::new(WaveCat::Matmul, segs, dom()).unwrap();
        assert_eq!(wave.segments(), &[Id(1), Id(2), Id(3)]);
        assert_eq!(wave.category(), WaveCat::Matmul);
    }

    #[test]
    fn every_wave_category_names_a_kernel() {
        for cat in [
            WaveCat::Region,
            WaveCat::Row,
            WaveCat::Matmul,
            WaveCat::MatmulSplitK,
        ] {
            let wave = KMerged::new(
                cat,
                [MergeSegment {
                    id: Id(1),
                    key: key(),
                    has_epilogue: false,
                }],
                dom(),
            )
            .unwrap();
            assert_eq!(wave.segments().len(), 1);
        }
    }

    // -- the schedule domain ------------------------------------------------

    fn gpu_caps(subgroup: u32, max_invocations: u32) -> Caps {
        let mut c = crate::emit::testkit::caps(false, true);
        c.subgroups = Some(SubgroupWidths {
            min: subgroup,
            max: subgroup,
        });
        c.limits.max_compute_invocations_per_workgroup = max_invocations;
        c.limits.max_compute_workgroup_size[0] = max_invocations;
        c
    }

    fn tms(d: &ScheduleDomain) -> Vec<u32> {
        d.iter()
            .map(|p| match p {
                SchedPoint::Map(t) => {
                    assert_eq!(t.dim, None, "a merged body has no axis to name");
                    assert_eq!(t.vector, 1, "vector width is a CPU parameter");
                    t.tm
                }
                other => panic!("a merged wave offered {other:?}"),
            })
            .collect()
    }

    fn wave(cat: WaveCat, key: MergeKey, segments: usize) -> KMerged {
        let segs = || {
            (0..segments).map(move |i| MergeSegment {
                id: Id(i as u32 + 1),
                key,
                has_epilogue: false,
            })
        };
        // `segment_elements` reads the wave, so the domain is derived from a
        // probe rather than re-spelled from the key.
        let caps = gpu_caps(32, 256);
        let probe = KMerged::new(cat, segs(), linear_domain(&caps, 1)).unwrap();
        let d = merged_domain(&caps, &probe, &DimBinding::new()).unwrap();
        KMerged::new(cat, segs(), d).unwrap()
    }

    /// The whole point: a real wave has something to decide. `[128, 64]` is
    /// 8192 outputs per segment, which is four distinct register tiles, not
    /// one hardcoded 256-lane body.
    #[test]
    fn a_real_wave_has_a_non_trivial_domain() {
        let caps = gpu_caps(32, 256);
        let w = wave(WaveCat::Region, key(), 3);
        let d = merged_domain(&caps, &w, &DimBinding::new()).unwrap();
        assert_eq!(segment_elements(&w, &DimBinding::new()).unwrap(), 128 * 64);
        assert_eq!(tms(&d), vec![1, 2, 4, 8]);
        assert!(
            d.len() > 1,
            "a schedule domain of one is a node that opted out of the search"
        );
    }

    /// And the negative half: a wave with nothing to decide reports one
    /// point, rather than offering a tiling that would launch empty lanes.
    #[test]
    fn a_wave_too_small_to_tile_reports_one_point() {
        let caps = gpu_caps(32, 256);
        let tiny = MergeKey {
            m: Dim::Const(4),
            n: Dim::Const(1),
            ..key()
        };
        let d = merged_domain(&caps, &wave(WaveCat::Region, tiny, 2), &DimBinding::new()).unwrap();
        assert_eq!(tms(&d), vec![1]);
    }

    /// The domain is a function of the device, not only of the shape: a
    /// 64-wide subgroup needs twice the work before the same tile leaves a
    /// full subgroup busy.
    #[test]
    fn the_domain_follows_the_device() {
        let shape = MergeKey {
            m: Dim::Const(16),
            n: Dim::Const(16),
            ..key()
        };
        let w = wave(WaveCat::Region, shape, 2);
        let narrow = merged_domain(&gpu_caps(32, 256), &w, &DimBinding::new()).unwrap();
        let wide = merged_domain(&gpu_caps(64, 256), &w, &DimBinding::new()).unwrap();
        assert_eq!(tms(&narrow), vec![1, 2, 4, 8]);
        assert_eq!(tms(&wide), vec![1, 2, 4]);
        assert_ne!(narrow, wide);
    }

    /// The workgroup width was the constant this module existed to delete.
    /// It is now a whole number of subgroups, bounded by the work and by the
    /// device.
    #[test]
    fn the_width_is_derived_from_the_work_and_the_caps() {
        let caps = gpu_caps(32, 256);
        assert_eq!(block_for(&caps, 8), 32, "8 lanes do not want 256");
        assert_eq!(block_for(&caps, 100), 128);
        assert_eq!(block_for(&caps, 10_000), 256);
        // A device that cannot run 256 invocations gets a legal width, and a
        // 64-wide subgroup rounds up to whole subgroups.
        assert_eq!(block_for(&gpu_caps(32, 64), 10_000), 64);
        assert_eq!(block_for(&gpu_caps(64, 256), 100), 128);
        assert_eq!(block_for(&gpu_caps(64, 256), 8), 64);
    }

    /// A point this body cannot execute is refused, not silently rounded to
    /// something it can.
    #[test]
    fn a_tiling_naming_an_axis_is_refused() {
        assert!(
            tiling_of(SchedPoint::Map(MapTiling {
                dim: Some(0),
                tm: 4,
                vector: 1
            }))
            .is_err()
        );
        assert!(tiling_of(SchedPoint::Fold(fusor2_ir::ir::level1::FoldStrat::Subgroup)).is_err());
        // `Point` is the untiled member of the node's own domain.
        assert_eq!(tiling_of(SchedPoint::Point).unwrap().tm, 1);
    }

    // -- lowering at a point ------------------------------------------------

    fn graph_with(dims: &[u64], count: usize) -> (EGraph, Vec<Id>) {
        use fusor2_ir::ir::Op;
        use fusor2_ir::ir::level0::{BufferId, L0, LeafKind};
        use fusor2_ir::semantics::{CoreSemantics, SumArenaPlanner};
        use std::sync::Arc;

        let mut g = EGraph::new(CoreSemantics::new(Arc::new(SumArenaPlanner)));
        let ids = (0..count)
            .map(|i| {
                let shape: SmallVec<[Dim; 6]> = dims.iter().map(|d| Dim::Const(*d)).collect();
                g.add(Op::L0(L0::Leaf(LeafKind::Buffer {
                    name: BufferId(i as u32),
                    dtype: Dtype::F32,
                    shape,
                })))
                .unwrap()
            })
            .collect();
        (g, ids)
    }

    fn lower_root(g: &EGraph, root: Id, outs: &[Id], caps: &Caps, theta: SchedPoint) -> KernelIr {
        use fusor2_ir::cost::Picoseconds;
        use fusor2_ir::extract::{BindKind, BindingPlan, Extraction, Launch, Plan, PlanHash};

        let mut bindings = vec![BindingPlan {
            binding: 1,
            value: root,
            kind: BindKind::Write,
        }];
        for (i, out) in outs.iter().enumerate() {
            bindings.push(BindingPlan {
                binding: i as u32 + 2,
                value: *out,
                kind: BindKind::Write,
            });
        }
        let plan = Plan {
            extraction: Extraction::default(),
            launches: vec![Launch {
                root,
                members: smallvec::smallvec![root],
                bindings,
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
            graph: g,
            symbols: &[],
        };
        lower(caps, g.node(root), theta, &cx).expect("lowers")
    }

    /// A merged wave over `segments` buffers of `dims`, lowered at `theta`.
    fn merged_ir(cat: WaveCat, dims: &[u64], segments: usize, theta: SchedPoint) -> KernelIr {
        merged_ir_at(cat, dims, segments, theta, &gpu_caps(32, 256))
    }

    fn merged_ir_at(
        cat: WaveCat,
        dims: &[u64],
        segments: usize,
        theta: SchedPoint,
        caps: &Caps,
    ) -> KernelIr {
        use fusor2_ir::ir::Op;
        let (mut g, segs) = graph_with(dims, segments);
        let elems: u64 = dims.iter().product();
        let k = MergeKey {
            m: Dim::Const(elems),
            n: Dim::ONE,
            k: Dim::Const(8),
            batch: Dim::ONE,
            splits: 1,
            dtype: Dtype::F32,
            family: Family::GenericFold,
        };
        let w = KMerged::new(
            cat,
            segs.iter().map(|id| MergeSegment {
                id: *id,
                key: k,
                has_epilogue: false,
            }),
            linear_domain(caps, elems),
        )
        .unwrap();
        let root = g.add(Op::L1(L1::KMerged(w))).unwrap();
        lower_root(&g, root, &segs, caps, theta)
    }

    /// A region over `outs` live members of `dims`, lowered at `theta`.
    fn region_ir(dims: &[u64], outs: usize, theta: SchedPoint) -> KernelIr {
        use fusor2_ir::ir::Op;
        let (mut g, members) = graph_with(dims, outs);
        let root = g
            .add(Op::L1(L1::KRegion {
                members: members.iter().copied().collect(),
                live_outs: (0..outs as u32).collect(),
                sched: linear_domain(&gpu_caps(32, 256), dims.iter().product()),
            }))
            .unwrap();
        lower_root(&g, root, &members, &gpu_caps(32, 256), theta)
    }

    fn map_point(tm: u32) -> SchedPoint {
        SchedPoint::Map(MapTiling {
            dim: None,
            tm,
            vector: 1,
        })
    }

    // -- the write set, evaluated out of the emitted body -------------------

    #[derive(Copy, Clone)]
    struct Thread {
        gx: u32,
        gy: u32,
        gz: u32,
        lane: u32,
    }

    /// A closed evaluator over the index algebra these two bodies emit.
    /// Anything else returns `None` and the caller fails loudly, so a body
    /// that grows a term this cannot see is a test failure rather than a
    /// silently weaker assert.
    fn eval(e: &TileExpr, t: Thread) -> Option<u64> {
        use fusor2_ir::scalar::BinOp;
        Some(match e.kind() {
            TileExprKind::Literal(TileLiteral::U32(v)) => u64::from(*v),
            TileExprKind::Literal(TileLiteral::Bool(b)) => u64::from(*b),
            TileExprKind::Builtin(Builtin::Lane) => u64::from(t.lane),
            TileExprKind::Builtin(Builtin::ProgramId(WorkgroupAxis::X)) => u64::from(t.gx),
            TileExprKind::Builtin(Builtin::ProgramId(WorkgroupAxis::Y)) => u64::from(t.gy),
            TileExprKind::Builtin(Builtin::ProgramId(WorkgroupAxis::Z)) => u64::from(t.gz),
            TileExprKind::Binary {
                op, left, right, ..
            } => {
                let (l, r) = (eval(left, t)?, eval(right, t)?);
                match op {
                    BinOp::Add => l + r,
                    BinOp::Mul => l * r,
                    BinOp::Div => l.checked_div(r)?,
                    BinOp::Rem => l.checked_rem(r)?,
                    BinOp::LogicalAnd => u64::from(l != 0 && r != 0),
                    _ => return None,
                }
            }
            TileExprKind::Compare { op, left, right } => {
                let (l, r) = (eval(left, t)?, eval(right, t)?);
                u64::from(match op {
                    TileCompareOp::Lt => l < r,
                    TileCompareOp::Eq => l == r,
                    _ => return None,
                })
            }
            _ => return None,
        })
    }

    /// Every `(binding, address)` the kernel stores to, once per masked-in
    /// lane. This is read out of the emitted `KernelIr`, not recomputed from
    /// the parameters, so it is a statement about the shader.
    fn write_set(ir: &KernelIr) -> Vec<(u32, u64)> {
        let mut out = Vec::new();
        for gz in 0..ir.grid[2] {
            for gy in 0..ir.grid[1] {
                for gx in 0..ir.grid[0] {
                    for lane in 0..ir.block {
                        let t = Thread { gx, gy, gz, lane };
                        for stmt in &ir.body {
                            let Stmt::Store { dst, addr, mask, .. } = stmt else {
                                continue;
                            };
                            let Addr::Linear(index) = addr else {
                                panic!("a linear body stored through {addr:?}")
                            };
                            let live = eval(mask, t).expect("the mask is closed");
                            if live != 0 {
                                let a = eval(index, t).expect("the address is closed");
                                out.push((dst.buffer.binding, a));
                            }
                        }
                    }
                }
            }
        }
        out
    }

    fn one_write_per_element(ir: &KernelIr, bindings: &[u32], elements: u64) {
        let mut got = write_set(ir);
        got.sort_unstable();
        let mut want: Vec<(u32, u64)> = bindings
            .iter()
            .flat_map(|b| (0..elements).map(move |a| (*b, a)))
            .collect();
        want.sort_unstable();
        assert_eq!(
            got, want,
            "the write set moved: {} stores against {} elements",
            got.len(),
            want.len()
        );
    }

    /// **The numerics assert.** Each body writes one value per output
    /// element and the value at an element does not depend on the tiling, so
    /// "the same set of addresses, each written exactly once" is the whole
    /// of numeric equivalence here — and it is checked by evaluating the
    /// emitted address and mask expressions over the whole emitted grid, not
    /// by re-deriving them.
    #[test]
    fn every_point_writes_every_element_exactly_once() {
        // 200 elements: not a multiple of any candidate width, so a
        // fencepost in the tile, the group or the grid shows up.
        const N: u64 = 200;
        for cat in [
            WaveCat::Region,
            WaveCat::Row,
            WaveCat::Matmul,
            WaveCat::MatmulSplitK,
        ] {
            for tm in [1, 2, 4, 8] {
                let ir = merged_ir(cat, &[N], 3, map_point(tm));
                one_write_per_element(&ir, &[2, 3, 4], N);
            }
        }
        for tm in [1, 2, 4, 8] {
            let ir = region_ir(&[N], 2, map_point(tm));
            one_write_per_element(&ir, &[2, 3], N);
        }
    }

    /// The same, at `SchedPoint::Point` — the path every launch takes today,
    /// since neither variant carries a `sched` field for extraction to
    /// resolve.
    #[test]
    fn the_default_point_writes_every_element_exactly_once() {
        let ir = merged_ir(WaveCat::Row, &[200], 3, SchedPoint::Point);
        one_write_per_element(&ir, &[2, 3, 4], 200);
        let ir = region_ir(&[200], 2, SchedPoint::Point);
        one_write_per_element(&ir, &[2, 3], 200);
    }

    /// Different points are different kernels — the geometry reaches the
    /// dispatch, not just a comment. A wider tile means fewer workgroups and
    /// more stores per lane.
    #[test]
    fn every_point_is_a_different_dispatch() {
        let mut seen: Vec<([u32; 3], u32, usize)> = Vec::new();
        for tm in [1, 2, 4, 8] {
            let ir = merged_ir(WaveCat::Row, &[8192], 2, map_point(tm));
            let stores = ir
                .body
                .iter()
                .filter(|s| matches!(s, Stmt::Store { .. }))
                .count();
            assert_eq!(stores, 2 * tm as usize, "one store per segment per tile");
            seen.push((ir.grid, ir.block, stores));
        }
        let groups: Vec<u32> = seen.iter().map(|(g, _, _)| g[0]).collect();
        // 8192 elements over two segments: 32 groups per segment at tm=1.
        assert_eq!(groups, vec![64, 32, 16, 8], "a wider tile launches less");
        assert!(seen.iter().all(|(_, b, _)| *b == 256));
        seen.dedup();
        assert_eq!(seen.len(), 4, "every point is a distinct dispatch");
    }

    /// Small shapes stop launching a 256-lane workgroup for 8 stores. This
    /// is the constant that was `BLOCK`.
    #[test]
    fn a_small_wave_no_longer_launches_a_full_workgroup() {
        let ir = merged_ir(WaveCat::Row, &[8], 2, SchedPoint::Point);
        assert_eq!(ir.block, 32);
        assert_eq!(ir.grid, [2, 1, 1], "one group per segment");
        let ir = region_ir(&[8], 2, SchedPoint::Point);
        assert_eq!(ir.block, 32);
        assert_eq!(ir.grid, [1, 1, 1]);
    }

    /// The register tile is worth scoring on a matmul segment only because
    /// the whole tile rides **one** k loop. Two loops would be two passes
    /// over the same operand and the tiling would be a pessimization.
    #[test]
    fn a_matmul_tile_rides_one_k_loop() {
        for tm in [1, 2, 4] {
            let ir = merged_ir(WaveCat::Matmul, &[8192], 2, map_point(tm));
            let loops: Vec<&Vec<Accumulator>> = ir
                .body
                .iter()
                .filter_map(|s| match s {
                    Stmt::If { accept, .. } => accept.first(),
                    _ => None,
                })
                .filter_map(|s| match s {
                    Stmt::Loop { accumulators, .. } => Some(accumulators),
                    _ => None,
                })
                .collect();
            assert_eq!(
                loops.iter().map(|a| a.len()).collect::<Vec<_>>(),
                vec![tm as usize; 2],
                "one loop per segment, one accumulator per tile position"
            );
            // Distinct registers, not one read `tm` times — the hazard
            // `LocalDecl`'s identity exists to prevent.
            for accs in loops {
                let mut ids: Vec<u64> = accs.iter().map(|a| a.local.id()).collect();
                ids.sort_unstable();
                ids.dedup();
                assert_eq!(ids.len(), tm as usize, "the tile shares one accumulator");
            }
        }
    }

    /// Every point the domain offers lowers to a kernel that passes the L2
    /// verifier. A domain member that cannot be executed is worse than no
    /// domain at all.
    #[test]
    fn every_point_the_domain_offers_lowers_and_verifies() {
        let caps = gpu_caps(32, 256);
        let w = wave(WaveCat::Row, key(), 2);
        let domain = merged_domain(&caps, &w, &DimBinding::new()).unwrap();
        assert!(domain.len() > 1);
        for point in domain.iter() {
            for cat in [WaveCat::Region, WaveCat::Row, WaveCat::Matmul] {
                let ir = merged_ir(cat, &[1024], 2, point);
                fusor2_tile::verify_l2(&ir, &caps).expect("a domain member must be executable");
            }
            let ir = region_ir(&[1024], 2, point);
            fusor2_tile::verify_l2(&ir, &caps).expect("a domain member must be executable");
        }
    }

    /// A split-K wave stored `k/splits` terms as if it were the whole sum —
    /// a partial reduction that is nearly right, which is the worst kind of
    /// wrong. Nothing mints `splits > 1` (`rules::merge::segment_of` writes
    /// `splits: 1` on every segment), so refusing costs nothing and the
    /// silently-partial answer is gone.
    #[test]
    fn a_split_k_wave_is_refused_rather_than_summed_short() {
        use fusor2_ir::ir::Op;
        let (mut g, segs) = graph_with(&[64], 2);
        let split = MergeKey {
            m: Dim::Const(64),
            n: Dim::ONE,
            k: Dim::Const(256),
            batch: Dim::ONE,
            splits: 4,
            dtype: Dtype::F32,
            family: Family::Coop,
        };
        let w = KMerged::new(
            WaveCat::MatmulSplitK,
            segs.iter().map(|id| MergeSegment {
                id: *id,
                key: split,
                has_epilogue: false,
            }),
            linear_domain(&gpu_caps(32, 256), 64),
        )
        .unwrap();
        let root = g.add(Op::L1(L1::KMerged(w))).unwrap();

        use fusor2_ir::cost::Picoseconds;
        use fusor2_ir::extract::{BindKind, BindingPlan, Extraction, Launch, Plan, PlanHash};
        let plan = Plan {
            extraction: Extraction::default(),
            launches: vec![Launch {
                root,
                members: smallvec::smallvec![root],
                bindings: vec![BindingPlan {
                    binding: 1,
                    value: root,
                    kind: BindKind::Write,
                }],
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
        let err = lower(&gpu_caps(32, 256), g.node(root), SchedPoint::Point, &cx).unwrap_err();
        assert!(format!("{err}").contains("split-K"), "{err}");
        // And the same wave at `splits: 1` still lowers: the refusal is on
        // the split, not on the category.
        let ir = merged_ir(WaveCat::MatmulSplitK, &[64], 2, SchedPoint::Point);
        one_write_per_element(&ir, &[2, 3], 64);
    }

    /// The group index is linearized over all three dispatch axes, the way
    /// `distribute_workgroups` laid the grid out. Dropping `z` aliased every
    /// group past the second slab onto slab 0 — every one of them writing
    /// the same elements.
    #[test]
    fn the_group_index_reads_all_three_dispatch_axes() {
        // A device that only takes 4 workgroups per dimension folds 32 groups
        // onto `[4, 4, 2]`, which is the smallest grid that actually uses the
        // z axis. Without the z term every group in the second slab aliases
        // one in the first and the write set doubles up.
        let mut caps = gpu_caps(32, 256);
        caps.limits.max_compute_workgroups_per_dimension = 4;
        let ir = merged_ir_at(WaveCat::Row, &[4096], 2, SchedPoint::Point, &caps);
        assert_eq!(ir.grid, [4, 4, 2], "this shape must reach the third axis");
        one_write_per_element(&ir, &[2, 3], 4096);
    }

    /// `region_domain` and `merged_domain` are one function of one number, so
    /// the two nodes cannot drift into two schedule vocabularies.
    #[test]
    fn both_nodes_share_one_domain() {
        let caps = gpu_caps(32, 256);
        let w = wave(WaveCat::Region, key(), 2);
        assert_eq!(
            merged_domain(&caps, &w, &DimBinding::new()).unwrap(),
            region_domain(&caps, 128 * 64)
        );
    }
}
