//! The L1 invariants: schedule-point legality against lane limits and the
//! exact `arena_plan` workgroup footprint; an injective write map unless the
//! nest declares an associative `combine`; no fold dim with nonzero stride in
//! the write map; every operand's `AccessPlan` satisfying its own predicate;
//! `KMerged` segments sharing a `MergeKey` with no epilogue, and both
//! composite forms carrying the linear schedule domain their members' shared
//! index space implies; an `Effect` on every node; and no buffer offset,
//! since allocation is not described at L1.

use crate::carrier::SlotTy;
use crate::device::Caps;
use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::ir::{Op, VerifyCtx};
use crate::ir::level0::ScatterCombine;
use crate::ir::level1::{
    AccessPlan, CoopGeom, Effect, IndexSpace, L1, Operand, ScheduleDomain,
};
use crate::ir::level2::{
    ArenaPlanner, ElementType, MemoryLevel, ScalarElement, TileDecl, TileLayout, Tiles,
};
use crate::semantics::effect_of;
use crate::shape::{Dim, Layout};
use smallvec::SmallVec;
use std::sync::Arc;

/// Workgroup tiles a cooperative geometry declares, before packing.
///
/// A-tile `[bm, bk]` and B-tile `[bk, bn / n_passes]`, each replicated
/// `staging` times, plus one accumulator staging tile `[bm, bn / n_passes]`
/// when the store element is not `F32` (an f32 accumulator written into
/// narrower memory needs the staging pass unless the device supports a
/// mixed-precision cooperative store).
///
/// The single source of coop tile shapes: `verify_l1` and `fusor2-tile`'s
/// `domains::coop` both call it.
pub fn coop_tiles(geom: CoopGeom, elem: ScalarElement, staging: u8) -> Tiles {
    let mut tiles = Tiles::default();
    let n_passes = geom.n_passes.max(1);
    let bn_pass = geom.bn / n_passes;
    let element = ElementType::Scalar(elem);
    let depth = staging.max(1);

    // One decl per staging depth, and they are `depth` distinct tiles:
    // `TileDecl` is identity-bearing, so `staging: 2` is two `coop_a`
    // allocations the arena places separately rather than one name used twice.
    for _ in 0..depth {
        tiles.decls.push(Arc::new(TileDecl::new(
            element,
            TileLayout::contiguous(MemoryLevel::Workgroup, &[geom.bm, geom.bk]),
            "coop_a",
        )));
        tiles.decls.push(Arc::new(TileDecl::new(
            element,
            TileLayout::contiguous(MemoryLevel::Workgroup, &[geom.bk, bn_pass]),
            "coop_b",
        )));
    }
    if elem != ScalarElement::F32 {
        tiles.decls.push(Arc::new(TileDecl::new(
            ElementType::Scalar(ScalarElement::F32),
            TileLayout::contiguous(MemoryLevel::Workgroup, &[geom.bm, bn_pass]),
            "coop_acc",
        )));
    }
    tiles
}

/// Verify one L1 node against `caps` and the exact arena plan.
pub fn verify_l1(cx: &VerifyCtx<'_>, planner: &dyn ArenaPlanner) -> Result<()> {
    let Op::L1(op) = &cx.node.op else {
        return Err(Error::verify(
            crate::ir::Level::L1,
            cx.id,
            "verify_l1 applied to a node that is not L1",
        ));
    };

    // 1 + 2.
    if let Some(sched) = op.schedule() {
        check_schedule_domain(op, sched, cx.caps, planner)
            .map_err(|e| relabel(cx, format!("{e}")))?;
    }

    // 3.
    check_write_injective(cx)?;

    // 4.
    check_fold_axis_not_written(cx, op)?;

    // 5.
    check_operand_access(op).map_err(|e| relabel(cx, format!("{e}")))?;

    // 6.
    if let L1::KMerged(m) = op {
        if m.segments().is_empty() {
            return Err(relabel(cx, "a merged wave has no segments".into()));
        }
        if cx.operands.len() != m.segments().len() {
            return Err(relabel(
                cx,
                format!(
                    "merged wave has {} segments but {} operand facts",
                    m.segments().len(),
                    cx.operands.len()
                ),
            ));
        }
        // `KMerged`'s fields are private and `KMerged::new` already rejected
        // an epilogue-carrying or key-mismatched segment, so the wave's key
        // is the shared key by construction; re-assert the shape it implies.
        if m.key().splits == 0 {
            return Err(relabel(cx, "a merged wave declares zero splits".into()));
        }
    }

    // 6b.
    check_composite_domain(cx, op).map_err(|e| relabel(cx, format!("{e}")))?;

    // 7.
    let declared = effect_of(&cx.node.op);
    let expected = expected_effect(op);
    if declared != expected {
        return Err(relabel(
            cx,
            format!("effect {declared:?} disagrees with the classification {expected:?}"),
        ));
    }

    // 8.
    for (i, o) in operands_of(op).iter().enumerate() {
        if !o.layout.offset().known_eq(Dim::Const(0)) {
            return Err(relabel(
                cx,
                format!(
                    "operand {i} names a buffer offset ({}); allocation is not described at L1",
                    o.layout.offset()
                ),
            ));
        }
    }

    // The `verify_l0` constant-work tripwire, applied to the one L1 variant
    // whose row comes from outside the crate: an `OpDef` whose `work` does not
    // vary with shape is a placeholder, not a cost model.
    if let L1::Ext { def, .. } = op
        && let Some(d) = cx.registry.get(*def)
        && !crate::semantics::work::work_varies(d.work, cx.operands, cx.result)
    {
        return Err(relabel(
            cx,
            format!("OpDef `{}`: work() does not vary with shape", d.name),
        ));
    }

    Ok(())
}

/// Invariant 6b: a composite node's schedule domain is the one its members'
/// shared index space implies.
///
/// `KRegion` and `KMerged` are the same shape to both backends — a list of L1
/// nodes run in one dispatch over one linearized index — so their geometry is
/// [`crate::ir::level1::MapDomain::linear_over`] of the value they land, and
/// nothing else. Checking it against the *node's own inferred shape* rather
/// than against whatever the minting rule felt like is what makes the domain
/// a property of the node instead of a field a rule may drift.
///
/// The clause is exact rather than a bound: both mint sites call the same
/// generator on the same facts, so an inequality is a rule that stopped
/// deriving the domain, not a legal variation.
fn check_composite_domain(cx: &VerifyCtx<'_>, op: &L1) -> Result<()> {
    let sched = match op {
        L1::KRegion { sched, .. } => sched,
        L1::KMerged(m) => m.schedule(),
        _ => return Ok(()),
    };
    let want = ScheduleDomain::Map(crate::ir::level1::MapDomain::linear_over(
        cx.caps,
        &cx.result.shape,
    ));
    if *sched != want {
        return Err(Error::Legality(format!(
            "a composite node's schedule domain is the linear map domain of its \
             members' shared index space; got {sched:?}, want {want:?}"
        )));
    }
    Ok(())
}

/// Invariant 1+2: every point of `sched` is structurally legal and fits the
/// exact workgroup footprint. An empty resulting domain makes the node
/// unselectable, which extraction treats as "this alternative lost", never
/// as an error — but a domain *declared* empty on a node already in the
/// graph is a legality failure, because nothing could ever select it.
pub fn check_schedule_domain(
    op: &L1,
    sched: &ScheduleDomain,
    caps: &Caps,
    planner: &dyn ArenaPlanner,
) -> Result<()> {
    if sched.is_empty() {
        return Err(Error::Legality(
            "schedule domain is empty; this node is unselectable".into(),
        ));
    }

    let elem = store_dtype(op).scalar_element();
    let subgroup_width = caps.subgroup_width();
    let max_lanes = caps.limits.max_compute_invocations_per_workgroup;
    let max_storage = caps.limits.max_compute_workgroup_storage_size;

    match sched {
        ScheduleDomain::Coop(domain) => {
            for &geom in &domain.geoms {
                if !geom.legal(subgroup_width, max_lanes) {
                    return Err(Error::Legality(format!(
                        "coop geometry {geom:?} is illegal at subgroup width {subgroup_width} \
                         and {max_lanes} lanes"
                    )));
                }
                for &staging in &domain.staging {
                    // The exact planner value, never an estimator.
                    let bytes = planner.workgroup_bytes(&coop_tiles(geom, elem, staging), caps)?;
                    if bytes > max_storage {
                        return Err(Error::Legality(format!(
                            "coop geometry {geom:?} at staging {staging} needs {bytes} \
                             workgroup bytes, over the {max_storage} limit"
                        )));
                    }
                }
            }
            if domain.splits.contains(&0) {
                return Err(Error::Legality("a split-K count of 0 is illegal".into()));
            }
        }
        ScheduleDomain::Sgemm(domain) => {
            let elem_bytes = store_dtype(op).byte_size().max(1) as u32;
            for p in &domain.params {
                if !p.legal(elem_bytes, max_storage, max_lanes) {
                    return Err(Error::Legality(format!(
                        "sgemm params {p:?} are illegal at {elem_bytes}-byte elements"
                    )));
                }
            }
        }
        ScheduleDomain::Sgemv(domain) => {
            for p in &domain.params {
                if p.chunk == 0 || p.vector == 0 || p.subgroups == 0 {
                    return Err(Error::Legality(format!(
                        "sgemv params {p:?} have a zero term"
                    )));
                }
                if p.subgroups.saturating_mul(subgroup_width) > max_lanes {
                    return Err(Error::Legality(format!(
                        "sgemv params {p:?} want {} lanes, over the {max_lanes} limit",
                        p.subgroups.saturating_mul(subgroup_width)
                    )));
                }
            }
        }
        ScheduleDomain::Fold(domain) => {
            // A fold's accumulator lanes and width are on the node, so its
            // scratch footprint is decidable here even though the block is a
            // schedule choice: `fold_scratch_bytes` is a pure function of the
            // strategy and `caps`. Without this clause the lane-group check
            // below is the *only* admission test a fold domain faces, and a
            // promoted carrier — one whose accumulator holds a free axis, so
            // `lanes` is that axis's extent rather than 1 — slips a strategy
            // needing `lanes * block * acc_bytes` bytes past it. The domain
            // generator filters on exactly this, so a domain built there cannot
            // fail here; what this catches is a domain minted anywhere else,
            // which would otherwise be a `verify_plan` crash at extraction
            // rather than a lost alternative.
            let carrier_lanes = fold_carrier_lanes(op);
            for s in &domain.strategies {
                let group = s.lane_group(subgroup_width);
                if group == 0 || group > max_lanes {
                    return Err(Error::Legality(format!(
                        "fold strategy {s:?} wants a lane group of {group}, over {max_lanes}"
                    )));
                }
                if let Some((lanes, acc_bytes)) = carrier_lanes {
                    let bytes = crate::ir::level1::fold_scratch_bytes(
                        s,
                        lanes,
                        acc_bytes,
                        subgroup_width,
                        caps,
                    );
                    if bytes > u64::from(max_storage) {
                        return Err(Error::Legality(format!(
                            "fold strategy {s:?} over a {lanes}-lane carrier needs {bytes} \
                             workgroup bytes, over the {max_storage} limit"
                        )));
                    }
                }
            }
        }
        ScheduleDomain::Map(domain) => {
            for t in &domain.tilings {
                if t.tm == 0 || t.vector == 0 {
                    return Err(Error::Legality(format!("map tiling {t:?} has a zero term")));
                }
            }
        }
        ScheduleDomain::Point => {}
    }
    Ok(())
}

/// Invariant 3: the write map is injective, or the nest declares an
/// associative combine.
///
/// The write map is the operand-0 layout for a scatter (which writes through
/// its base) and the result's contiguous layout otherwise. Injectivity is
/// "every surviving output-axis stride is nonzero and distinct" after
/// dropping `Const(1)` axes, which cannot alias whatever their stride.
pub fn check_write_injective(cx: &VerifyCtx<'_>) -> Result<()> {
    let Op::L1(op) = &cx.node.op else {
        return Ok(());
    };
    if declares_associative_combine(op) {
        return Ok(());
    }
    let write = write_layout(op, cx);
    let mut strides: Vec<Dim> = Vec::with_capacity(write.rank());
    for (extent, stride) in write.shape().iter().zip(write.strides()) {
        if extent.known_eq(Dim::Const(1)) {
            continue;
        }
        if stride.known_eq(Dim::Const(0)) {
            return Err(relabel(
                cx,
                "write map has a stride-0 output axis and no associative combine".into(),
            ));
        }
        if strides.iter().any(|s| s.known_eq(*stride)) {
            return Err(relabel(
                cx,
                format!("write map repeats stride {stride}; it is not injective"),
            ));
        }
        strides.push(*stride);
    }
    Ok(())
}

/// Invariant 4: a `KFold`'s reduced axis must be absent from the write map.
/// A fold dim indexing the output is a scatter, not a reduction, so the
/// result rank has to be the space rank minus one, plus the carrier's axis.
fn check_fold_axis_not_written(cx: &VerifyCtx<'_>, op: &L1) -> Result<()> {
    let L1::KFold {
        space,
        axis,
        carrier,
        vec_axes,
        acc,
        post,
        ..
    } = op
    else {
        return Ok(());
    };
    let axis = *axis as usize;
    if axis >= space.rank() {
        return Err(relabel(
            cx,
            format!(
                "fold axis {axis} out of range for a rank-{} index space",
                space.rank()
            ),
        ));
    }
    crate::verify_l0::check_carrier(carrier, *acc).map_err(|e| relabel(cx, format!("{e}")))?;
    if post.len() != carrier.width() {
        return Err(relabel(
            cx,
            format!(
                "a rank-{} carrier carries {} post expressions",
                carrier.width(),
                post.len()
            ),
        ));
    }
    check_vec_axes(cx, space, axis, vec_axes, carrier)?;
    let carrier_axes = usize::from(carrier.out_dim().flatten().is_some());
    let expected = space.rank() - 1 - vec_axes.len() + carrier_axes;
    if cx.result.rank() != expected {
        return Err(relabel(
            cx,
            format!(
                "fold axis {axis} appears with nonzero stride in the write map: the result is \
                 rank {} where dropping the axis gives {expected}",
                cx.result.rank()
            ),
        ));
    }
    Ok(())
}

/// Invariant 5: each operand's `AccessPlan` satisfies its own predicate. A
/// failure names the operand, so it disqualifies only the rewrite that
/// produced it.
pub fn check_operand_access(op: &L1) -> Result<()> {
    // A contraction side holds a list, and an empty one would make its `pre`
    // a constant — a `KMap`, not a contraction, and a node the lowerings would
    // read `ops[0]` off. `ContractSide::primary` is written against this.
    if let L1::KContract { a, b, .. } = op {
        for (side, which) in [(a, "a"), (b, "b")] {
            if side.is_empty() {
                return Err(Error::Legality(format!(
                    "contraction side {which} reads no operand"
                )));
            }
        }
    }
    let space = index_space_of(op);
    for (i, o) in operands_of(op).iter().enumerate() {
        let fail = |msg: String| Error::Legality(format!("operand {i}: {msg}"));
        match &o.access {
            // Always legal: a gather derives its own addresses.
            AccessPlan::Gather => {}
            AccessPlan::Pack { into } => {
                if !into.is_contiguous() {
                    return Err(fail("Pack destination must be contiguous".into()));
                }
            }
            AccessPlan::Unflatten(map) => {
                // A contraction declares no index space; the map must then at
                // least match the operand's own layout rank.
                let rank = space.map_or_else(|| o.layout.rank(), IndexSpace::rank);
                if map.rank() != rank {
                    return Err(fail(format!(
                        "Unflatten map has rank {} but the index space is rank {rank}",
                        map.rank()
                    )));
                }
            }
            AccessPlan::Alias => {
                if let Some(s) = space {
                    if o.layout.rank() != s.rank() {
                        return Err(fail(format!(
                            "Alias layout is rank {} but the index space is rank {}",
                            o.layout.rank(),
                            s.rank()
                        )));
                    }
                    for (axis, (l, d)) in o.layout.shape().iter().zip(&s.dims).enumerate() {
                        // A `Const(1)` extent is a legal stride-0 broadcast.
                        if !l.known_eq(*d) && !l.known_eq(Dim::Const(1)) {
                            return Err(fail(format!(
                                "Alias layout axis {axis} is {l} but the index space is {d}"
                            )));
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Invariant 7's classification: `KScatter` writing through operand 0 with
/// atomics or a `Set` combine mutates state; everything else is pure.
fn expected_effect(op: &L1) -> Effect {
    match op {
        L1::KScatter { mode, combine, .. }
            if matches!(mode, crate::ir::level1::ScatterMode::Atomic)
                || matches!(combine, ScatterCombine::Set) =>
        {
            Effect::InPlace(crate::ir::level1::BufferRole(0))
        }
        _ => Effect::Pure,
    }
}

// Helpers

/// The promoted-axis invariants: `vec_axes` is a contiguous block immediately
/// before `axis`, every promoted extent is accounted for in the carrier's lane
/// count, and no expression on the node reads the coordinate of an axis that
/// no longer exists in the iteration domain.
///
/// The last clause is the one that catches a botched renumbering. A promoted
/// axis is gone from `iter_space`, so an `IndexOf` naming it would silently
/// read a *different* axis's coordinate — which is how an ALiBi or rope term
/// gets detached from its coordinate and the answer comes back nearly right.
fn check_vec_axes(
    cx: &VerifyCtx<'_>,
    space: &IndexSpace,
    axis: usize,
    vec_axes: &[u32],
    carrier: &crate::carrier::Carrier,
) -> Result<()> {
    if vec_axes.is_empty() {
        return Ok(());
    }
    let lo = axis - vec_axes.len();
    for (i, a) in vec_axes.iter().enumerate() {
        if *a as usize != lo + i {
            return Err(relabel(
                cx,
                format!(
                    "vec_axes {vec_axes:?} is not the contiguous block \
                     {lo}..{axis} immediately before the reduced axis"
                ),
            ));
        }
    }
    let promoted: Option<u64> = vec_axes
        .iter()
        .try_fold(1u64, |a, i| a.checked_mul(space.dims[*i as usize].as_const()?));
    let promoted = promoted.ok_or_else(|| {
        relabel(cx, "a promoted axis has a symbolic extent".to_string())
    })?;
    carrier
        .lanes()
        .ok_or_else(|| relabel(cx, "a promoted carrier has a symbolic lane count".into()))?;
    // Every **Vector** slot spans the promoted extent. A `Scalar` slot rides
    // through untouched: `Carrier::lanes` is the sum over slots, so a joint
    // carrier of `[Scalar rho, Vector(d) body]` legitimately has `1 + d` lanes.
    // Demanding `lanes == promoted * width` instead would require every slot to
    // be a Vector and would reject exactly the mixed accumulator a running
    // statistic beside a module-valued one produces — which is the shape the
    // retargeting law mints on a promoted nest.
    for (i, s) in carrier.slots.iter().enumerate() {
        let SlotTy::Vector(d) = s else { continue };
        let extent = d
            .as_const()
            .ok_or_else(|| relabel(cx, format!("slot {i} has a symbolic Vector extent")))?;
        if extent != promoted {
            return Err(relabel(
                cx,
                format!(
                    "slot {i} is Vector({extent}) but the promoted axes span {promoted} positions"
                ),
            ));
        }
    }
    //
    // It is updated once per iteration step, so its `lift` is evaluated at a
    // single promoted position. An operand that varies along a promoted axis
    // read there would contribute position 0's value at every position — a
    // wrong number, not a slow one, and invisible in any test whose promoted
    // extent is 1. A `Vector` slot has a register per position and may read
    // anything. This is the clause that makes a mixed `[Scalar, Vector]`
    // accumulator safe to mint at all.
    if let Op::L1(o) = &cx.node.op {
        let ops = operands_of(o);
        let varies: Vec<bool> = ops
            .iter()
            .copied()
            .map(|o| {
                vec_axes
                    .iter()
                    .any(|a| operand_varies_along(o, space, *a) != Some(false))
            })
            .collect();
        for (k, s) in carrier.slots.iter().enumerate() {
            if *s != SlotTy::Scalar {
                continue;
            }
            let mut used = SmallVec::new();
            collect_args(&carrier.lift[k], &mut used);
            if let Some(i) = used
                .iter()
                .find(|i| varies.get(**i as usize).copied() == Some(true))
            {
                return Err(relabel(
                    cx,
                    format!(
                        "scalar slot {k}'s lift reads operand {i}, which varies along a \
                         promoted axis; a scalar slot has one accumulator and would see \
                         only one of that operand's positions"
                    ),
                ));
            }
        }
    }

    // No expression may name a coordinate outside the ITERATION domain.
    //
    // Every `ScalarExpr` on a `KFold` is written against `iter_space()`, so the
    // legal indices are `0..iter_rank` and a promoted axis is simply not
    // nameable — that is the content of the rebinding. Asking instead whether
    // an expression reads `IndexOf(a)` for `a` a **space** index is one
    // renumbering behind: after one promotion the reduced axis's iteration
    // index equals the promoted axis's space index, so that spelling rejects
    // precisely the nests whose lift reads the reduction coordinate — a
    // max-pool's index slot, and a causal `select(IndexOf(lk) <= ..)`.
    let iter_rank = space.rank() - vec_axes.len();
    for a in iter_rank..space.rank() {
        let a = a as u32;
        if carrier.reads_index_of(a) || cx_post_reads(cx, a) {
            return Err(relabel(
                cx,
                format!(
                    "an expression reads IndexOf({a}), outside the rank-{iter_rank} \
                     iteration domain this node's expressions are written against"
                ),
            ));
        }
    }
    Ok(())
}

/// Every `Arg` index an expression names.
fn collect_args(e: &crate::scalar::ScalarExpr, out: &mut SmallVec<[u32; 4]>) {
    if let crate::scalar::ScalarKind::Arg(i) = e.kind()
        && !out.contains(i)
    {
        out.push(*i);
    }
    e.for_each_child(|c| collect_args(c, out));
}

/// Whether an operand's read moves as `axis`'s coordinate advances.
///
/// `Some(false)` is "provably invariant"; `None` is "cannot tell", which every
/// caller here must treat as "varies".
fn operand_varies_along(o: &Operand, space: &IndexSpace, axis: u32) -> Option<bool> {
    let a = axis as usize;
    if a >= space.rank() {
        return None;
    }
    // The flat-index window this axis occupies, row-major over `space`.
    let mut below = 1u64;
    for d in space.dims.iter().skip(a + 1) {
        below = below.checked_mul(d.as_const()?)?;
    }
    let hi = below.checked_mul(space.dims[a].as_const()?)?;
    let map = o.address_map()?;
    Some(map.terms.iter().any(|t| {
        let t_lo = u64::from(t.divisor);
        let t_hi = t_lo.saturating_mul(u64::from(t.modulus));
        t.stride != 0 && t_lo < hi && below < t_hi
    }))
}

fn cx_post_reads(cx: &VerifyCtx<'_>, axis: u32) -> bool {
    let Op::L1(L1::KFold { post, .. }) = &cx.node.op else {
        return false;
    };
    post.iter().any(|e| e.reads_index_of_axis(axis))
}

fn relabel(cx: &VerifyCtx<'_>, msg: String) -> Error {
    Error::verify(crate::ir::Level::L1, cx.id, msg)
}

fn declares_associative_combine(op: &L1) -> bool {
    match op {
        // Associativity is declared on the carrier, not derived from a name:
        // a non-associative carrier is legal but may not be tree-reduced.
        L1::KFold { carrier, .. } => carrier.associative,
        L1::KScatter { combine, .. } => matches!(combine, ScatterCombine::Add),
        _ => false,
    }
}

fn write_layout(op: &L1, cx: &VerifyCtx<'_>) -> Layout {
    match op {
        // A scatter writes through its base operand's layout.
        L1::KScatter { ops, .. } => ops
            .first()
            .map(|o| o.layout.clone())
            .unwrap_or_else(|| Layout::contiguous(&cx.result.shape)),
        _ => Layout::contiguous(&cx.result.shape),
    }
}

fn index_space_of(op: &L1) -> Option<&IndexSpace> {
    match op {
        L1::KMap { space, .. }
        | L1::KFold { space, .. }
        | L1::KGather { space, .. }
        | L1::KScatter { space, .. } => Some(space),
        _ => None,
    }
}

/// Every `Operand` a node carries, in `children_of` order.
fn operands_of(op: &L1) -> SmallVec<[&Operand; 4]> {
    match op {
        L1::KMap { ops, .. }
        | L1::KFold { ops, .. }
        | L1::KGather { ops, .. }
        | L1::KScatter { ops, .. }
        | L1::Ext { ops, .. } => ops.iter().collect(),
        L1::KContract { a, b, .. } => a.ops.iter().chain(b.ops.iter()).collect(),
        L1::KRegion { .. } | L1::KMerged(_) => SmallVec::new(),
    }
}

/// A `KFold`'s `(accumulator lanes, bytes per lane)`, or `None` when the node
/// is not a fold or its carrier's lane count is symbolic.
///
/// A symbolic `Vector` slot extent is allocatable on neither backend, and
/// `verify_l0` clause 3 already rejects one, so `None` here means "not a
/// fold" in every graph that got this far.
fn fold_carrier_lanes(op: &L1) -> Option<(u64, u64)> {
    match op {
        L1::KFold { carrier, acc, .. } => Some((carrier.lanes()?, acc.byte_size())),
        _ => None,
    }
}

fn store_dtype(op: &L1) -> Dtype {
    match op {
        L1::KMap { body, .. } => body.dtype(),
        L1::KFold { acc, .. } => *acc,
        L1::KContract { post, .. } => post.dtype(),
        _ => Dtype::F32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{DeviceKind, Limits, SubgroupWidths};
    use crate::egraph::Id;
    use crate::facts::{ValueFacts, Work};
    use crate::ir::OpDef;
    use crate::carrier::Carrier;
    use crate::scalar::BinOp;
    use crate::ir::level1::{
        BufferRole, ContractSide, Family, KMerged, MapDomain, MergeKey, MergeSegment, ScatterMode,
        WaveCat,
    };
    use crate::ir::level2::{ArenaMode, ArenaPlan, BarrierSuggestion, KernelIr};
    use crate::ir::{Level, Node, Op, OpDefRegistry};
    use crate::scalar::ScalarExpr;
    use smallvec::smallvec;

    /// An exact planner: the packed footprint is the sum of tile bytes. Real
    /// packing lives in `fusor2-tile`; this stands in for it in unit tests
    /// and is still the *only* source of the byte figure the verifier sees.
    struct SumPlanner;

    impl ArenaPlanner for SumPlanner {
        fn arena_plan(&self, _ir: &KernelIr, _caps: &Caps) -> Result<ArenaPlan> {
            Ok(ArenaPlan {
                mode: ArenaMode::Regions,
                total_bytes: 0,
                placements: Default::default(),
                barriers_inserted: Default::default(),
            })
        }
        fn workgroup_bytes(&self, tiles: &Tiles, _caps: &Caps) -> Result<u32> {
            Ok(tiles
                .decls
                .iter()
                .map(|t| (t.layout.element_count() * t.element.byte_size()) as u32)
                .sum())
        }
        fn barrier_suggestions(&self, _ir: &KernelIr) -> Vec<BarrierSuggestion> {
            Vec::new()
        }
        fn verify_arena(&self, _ir: &KernelIr, _plan: &ArenaPlan) -> Result<()> {
            Ok(())
        }
        fn verify_uniformity(&self, _ir: &KernelIr) -> Result<()> {
            Ok(())
        }
    }

    fn caps() -> Caps {
        Caps {
            kind: DeviceKind::Gpu,
            name: "test".into(),
            limits: Limits {
                max_compute_invocations_per_workgroup: 256,
                max_compute_workgroup_storage_size: 32 * 1024,
                ..Limits::default()
            },
            subgroups: Some(SubgroupWidths { min: 32, max: 32 }),
            f16: true,
            bf16: false,
            coop: Default::default(),
            atomic_f32: true,
            workgroup_alias: false,
            mixed_precision_coop_store: false,
            pipeline_cache: false,
            timestamp_query: false,
            simd_widths: Default::default(),
            threads: 1,
        }
    }

    fn f32s(shape: &[u64]) -> ValueFacts {
        ValueFacts::new(Dtype::F32, shape.iter().map(|&d| Dim::Const(d)))
    }

    fn operand(src: u32, shape: &[u64]) -> Operand {
        let dims: Vec<Dim> = shape.iter().map(|&d| Dim::Const(d)).collect();
        Operand {
            src: Id(src),
            layout: Layout::contiguous(&dims),
            access: AccessPlan::Alias,
        }
    }

    fn run(op: L1, operands: &[ValueFacts], result: &ValueFacts) -> Result<()> {
        run_with(op, operands, result, &OpDefRegistry::new())
    }

    fn run_with(
        op: L1,
        operands: &[ValueFacts],
        result: &ValueFacts,
        registry: &OpDefRegistry,
    ) -> Result<()> {
        let caps = caps();
        let node = Node {
            children: crate::semantics::children::children_l1(&op),
            op: Op::L1(op),
            level: Level::L1,
        };
        let cx = VerifyCtx {
            node: &node,
            id: Id(7),
            operands,
            result,
            caps: &caps,
            registry,
        };
        verify_l1(&cx, &SumPlanner)
    }

    /// **A `Scalar` slot may not read an operand that varies along a promoted
    /// axis.** It is one accumulator, updated once per iteration step, so its
    /// lift is evaluated at one promoted position; an operand that moves with
    /// the promoted coordinate would contribute position 0's value at every
    /// position. The positive half — a scalar slot beside a `Vector` one,
    /// reading only a promoted-invariant operand — must still verify, or the
    /// clause would reject the mixed accumulator it exists to make safe.
    #[test]
    fn a_scalar_slot_may_not_read_a_promoted_varying_operand() {
        use crate::carrier::{ArgRemap, SlotTy};
        use crate::dtype::Dtype;

        // space = [free 2, promoted 3, reduced 4].
        let dims = [2u64, 3, 4];
        let space = IndexSpace::new(dims.iter().map(|d| Dim::Const(*d)));
        // A `[Scalar Max, Vector(3) Add]` carrier: the shape a running
        // statistic beside a module-valued accumulator has.
        let joint = Carrier::binop(BinOp::Max, Carrier::binop_identity(BinOp::Max, Dtype::F32).unwrap(), Dtype::F32)
            .tuple(
                &Carrier::binop(BinOp::Add, Carrier::binop_identity(BinOp::Add, Dtype::F32).unwrap(), Dtype::F32)
                    .promote(Dim::Const(3))
                    .unwrap(),
                &ArgRemap::identity(1),
            )
            .carrier;
        assert_eq!(joint.slots[0], SlotTy::Scalar);
        assert_eq!(joint.slots[1], SlotTy::Vector(Dim::Const(3)));

        let facts = f32s;
        // Operand 0 is invariant along the promoted axis (stride 0 there);
        // operand 1 varies along it.
        let invariant = Operand {
            src: Id(1),
            layout: Layout::from_parts(
                Dim::Const(0),
                &[Dim::Const(2), Dim::Const(3), Dim::Const(4)],
                &[Dim::Const(4), Dim::Const(0), Dim::Const(1)],
            )
            .unwrap(),
            access: AccessPlan::Alias,
        };
        let varying = Operand {
            src: Id(2),
            layout: Layout::contiguous(&[Dim::Const(2), Dim::Const(3), Dim::Const(4)]),
            access: AccessPlan::Alias,
        };
        let build = |lift0: ScalarExpr, ops: Vec<Operand>| L1::KFold {
            space: space.clone(),
            axis: 2,
            vec_axes: smallvec![1],
            carrier: joint.clone().with_lift([lift0, ScalarExpr::arg(1, Dtype::F32)]),
            acc: Dtype::F32,
            post: smallvec![ScalarExpr::arg(0, Dtype::F32), ScalarExpr::arg(1, Dtype::F32)],
            ops,
            sched: ScheduleDomain::Point,
        };
        let out = facts(&[2, 4]);
        let ins = [facts(&[2, 3, 4]), facts(&[2, 3, 4])];

        // Positive: the scalar slot reads only the promoted-invariant operand.
        run(
            build(ScalarExpr::arg(0, Dtype::F32), vec![invariant.clone(), varying.clone()]),
            &ins,
            &out,
        )
        .expect("a scalar slot reading a promoted-invariant operand is legal");

        // Negative: it reads the one that varies along the promoted axis.
        let err = run(
            build(ScalarExpr::arg(1, Dtype::F32), vec![invariant, varying]),
            &ins,
            &out,
        )
        .expect_err("a scalar slot reading a promoted-varying operand must be refused");
        assert!(
            format!("{err}").contains("varies along a promoted axis"),
            "got {err}"
        );
    }

    #[test]
    fn coop_geometry_legality() {
        let geom = CoopGeom {
            bm: 64,
            bn: 64,
            bk: 32,
            n_passes: 1,
            subgroups: 4,
            rg: 2,
            cg: 2,
        };
        assert!(geom.legal(32, 256));
        assert_eq!(geom.lanes(32), 128);

        // rg: 3 makes bm % (8*3) = 64 % 24 != 0.
        let bad = CoopGeom { rg: 3, ..geom };
        assert!(!bad.legal(32, 256));
        assert_ne!(64 % 24, 0);
    }

    #[test]
    fn coop_tiles_are_the_single_source_of_tile_shapes() {
        let geom = CoopGeom {
            bm: 64,
            bn: 64,
            bk: 32,
            n_passes: 2,
            subgroups: 4,
            rg: 2,
            cg: 2,
        };
        // f32: two tiles per staging depth, no accumulator staging tile.
        let t = coop_tiles(geom, ScalarElement::F32, 1);
        assert_eq!(t.decls.len(), 2);
        assert_eq!(&t.decls[0].layout.extents[..], &[64, 32]);
        assert_eq!(&t.decls[1].layout.extents[..], &[32, 32]); // bn / n_passes

        let staged = coop_tiles(geom, ScalarElement::F32, 2);
        assert_eq!(staged.decls.len(), 4);

        // f16 store: one f32 accumulator staging tile joins.
        let mixed = coop_tiles(geom, ScalarElement::F16, 1);
        assert_eq!(mixed.decls.len(), 3);
        assert_eq!(mixed.decls[2].name, "coop_acc");
        assert_eq!(&mixed.decls[2].layout.extents[..], &[64, 32]);
    }

    #[test]
    fn footprint_comes_from_the_planner() {
        let caps = caps();
        let geom = CoopGeom {
            bm: 64,
            bn: 64,
            bk: 32,
            n_passes: 1,
            subgroups: 4,
            rg: 2,
            cg: 2,
        };
        let domain = crate::ir::level1::CoopDomain {
            geoms: smallvec![geom],
            splits: smallvec![1],
            staging: smallvec![1],
        };
        let op = L1::KContract {
            m: Dim::Const(64),
            n: Dim::Const(64),
            k: Dim::Const(64),
            batch: Dim::Const(1),
            family: Family::Coop,
            post: ScalarExpr::arg(0, Dtype::F32),
            acc: Dtype::F32,
            a: ContractSide::one(ScalarExpr::arg(0, Dtype::F32), operand(0, &[64, 64])),
            b: ContractSide::one(ScalarExpr::arg(0, Dtype::F32), operand(1, &[64, 64])),
            sched: ScheduleDomain::Coop(domain),
        };
        // (64*32 + 32*64) * 4 bytes = 16 KiB, inside the 32 KiB limit.
        assert!(check_schedule_domain(&op, op.schedule().unwrap(), &caps, &SumPlanner).is_ok());

        // Halve the limit and the same geometry is rejected by the *planner's*
        // number, not by an estimate.
        let mut tight = caps.clone();
        tight.limits.max_compute_workgroup_storage_size = 8 * 1024;
        let err =
            check_schedule_domain(&op, op.schedule().unwrap(), &tight, &SumPlanner).unwrap_err();
        assert!(matches!(err, Error::Legality(_)));
    }

    #[test]
    fn an_empty_schedule_domain_is_illegal() {
        let op = L1::KContract {
            m: Dim::Const(4),
            n: Dim::Const(4),
            k: Dim::Const(4),
            batch: Dim::Const(1),
            family: Family::Coop,
            post: ScalarExpr::arg(0, Dtype::F32),
            acc: Dtype::F32,
            a: ContractSide::one(ScalarExpr::arg(0, Dtype::F32), operand(0, &[4, 4])),
            b: ContractSide::one(ScalarExpr::arg(0, Dtype::F32), operand(1, &[4, 4])),
            sched: ScheduleDomain::Coop(Default::default()),
        };
        assert!(matches!(
            check_schedule_domain(&op, op.schedule().unwrap(), &caps(), &SumPlanner),
            Err(Error::Legality(_))
        ));
    }

    /// A promoted carrier's scratch is `lanes * emitted_block * acc_bytes`,
    /// and the block is a *schedule* choice — but a pure function of the
    /// strategy and `caps`, so it is decidable here. Both sides of the
    /// boundary are asserted: the widest carrier that fits is admitted, one
    /// lane more is refused. Without this the lane-group test is the only
    /// admission a fold domain faces, and the resulting `verify_plan` failure
    /// is a hard crash rather than a lost alternative.
    #[test]
    fn a_promoted_fold_carrier_over_the_workgroup_limit_is_refused() {
        let caps = caps();
        let max_storage = u64::from(caps.limits.max_compute_workgroup_storage_size);
        let strat = crate::ir::level1::FoldStrat::WgTree { lane_group: 32 };
        let block = u64::from(crate::ir::level1::emitted_block(32, &caps));
        // The exact number of f32 lanes that saturates the limit.
        let fits = max_storage / (block * 4);
        assert!(fits > 1, "the device must admit more than one lane");

        let build = |lanes: u64| L1::KFold {
            space: IndexSpace::new([Dim::Const(4), Dim::Const(lanes), Dim::Const(8)]),
            axis: 2,
            vec_axes: smallvec![1],
            carrier: Carrier::binop(
                BinOp::Add,
                Carrier::binop_identity(BinOp::Add, Dtype::F32).unwrap(),
                Dtype::F32,
            )
            .promote(Dim::Const(lanes))
            .expect("a constant extent promotes"),
            acc: Dtype::F32,
            post: smallvec::smallvec![ScalarExpr::arg(0, Dtype::F32)],
            ops: vec![operand(0, &[4, lanes, 8])],
            sched: ScheduleDomain::Fold(crate::ir::level1::FoldDomain {
                strategies: smallvec![strat],
            }),
        };

        let ok = build(fits);
        check_schedule_domain(&ok, ok.schedule().unwrap(), &caps, &SumPlanner)
            .expect("a carrier exactly at the limit is legal");

        let over = build(fits + 1);
        let err = check_schedule_domain(&over, over.schedule().unwrap(), &caps, &SumPlanner)
            .expect_err("one lane over the limit must be refused");
        let msg = format!("{err}");
        assert!(
            msg.contains("workgroup bytes") && msg.contains("carrier"),
            "the message must name the footprint and the carrier: {msg}"
        );
    }

    /// The scalar carrier every ordinary reduction uses is one lane, so this
    /// clause must not touch it — a `Fold{Add}` over f32 is admitted at every
    /// lane group the device allows.
    #[test]
    fn a_scalar_fold_carrier_is_unaffected_by_the_footprint_clause() {
        let caps = caps();
        for lane_group in [1u32, 32, 64, 256] {
            let op = L1::KFold {
                space: IndexSpace::new([Dim::Const(4), Dim::Const(8)]),
                axis: 1,
                vec_axes: smallvec::SmallVec::new(),
                carrier: Carrier::binop(
                    BinOp::Add,
                    Carrier::binop_identity(BinOp::Add, Dtype::F32).unwrap(),
                    Dtype::F32,
                ),
                acc: Dtype::F32,
                post: smallvec::smallvec![ScalarExpr::arg(0, Dtype::F32)],
                ops: vec![operand(0, &[4, 8])],
                sched: ScheduleDomain::Fold(crate::ir::level1::FoldDomain {
                    strategies: smallvec![crate::ir::level1::FoldStrat::WgTree { lane_group }],
                }),
            };
            check_schedule_domain(&op, op.schedule().unwrap(), &caps, &SumPlanner)
                .unwrap_or_else(|e| panic!("lane group {lane_group}: {e}"));
        }
    }

    #[test]
    fn a_fold_axis_in_the_write_map_is_rejected() {
        let good = L1::KFold {
            space: IndexSpace::new([Dim::Const(4), Dim::Const(8)]),
            axis: 1,
            vec_axes: smallvec::SmallVec::new(),
            carrier: Carrier::binop(
                BinOp::Add,
                Carrier::binop_identity(BinOp::Add, Dtype::F32).unwrap(),
                Dtype::F32,
            ),
            acc: Dtype::F32,
            post: smallvec::smallvec![ScalarExpr::arg(0, Dtype::F32)],
            ops: vec![operand(0, &[4, 8])],
            sched: ScheduleDomain::Point,
        };
        run(good.clone(), &[f32s(&[4, 8])], &f32s(&[4])).unwrap();

        // The result still carries the folded axis: the fold dim indexes the
        // output, which is a scatter, not a reduction.
        let err = run(good, &[f32s(&[4, 8])], &f32s(&[4, 8])).unwrap_err();
        assert!(format!("{err}").contains("write map"));
    }

    #[test]
    fn write_map_injectivity() {
        // A `KMap` whose result has two axes of the same extent is still
        // injective: contiguous strides differ.
        let op = L1::KMap {
            space: IndexSpace::new([Dim::Const(4), Dim::Const(4)]),
            body: ScalarExpr::arg(0, Dtype::F32),
            ops: vec![operand(0, &[4, 4])],
            sched: ScheduleDomain::Point,
        };
        run(op, &[f32s(&[4, 4])], &f32s(&[4, 4])).unwrap();

        // A `Set` scatter through a stride-0 base is not injective.
        let base = Operand {
            src: Id(0),
            layout: Layout::from_parts(
                Dim::Const(0),
                &[Dim::Const(4), Dim::Const(4)],
                &[Dim::Const(0), Dim::Const(1)],
            )
            .unwrap(),
            access: AccessPlan::Alias,
        };
        let scatter = L1::KScatter {
            space: IndexSpace::new([Dim::Const(4), Dim::Const(4)]),
            axis: 0,
            mode: ScatterMode::SortSegment,
            combine: ScatterCombine::Set,
            ops: vec![base.clone(), operand(1, &[4, 4]), operand(2, &[4, 4])],
            sched: ScheduleDomain::Point,
        };
        assert!(
            run(
                scatter,
                &[f32s(&[4, 4]), f32s(&[4, 4]), f32s(&[4, 4])],
                &f32s(&[4, 4])
            )
            .is_err()
        );

        // The same non-injective write is legal under an associative combine.
        let add = L1::KScatter {
            space: IndexSpace::new([Dim::Const(4), Dim::Const(4)]),
            axis: 0,
            mode: ScatterMode::SortSegment,
            combine: ScatterCombine::Add,
            ops: vec![base, operand(1, &[4, 4]), operand(2, &[4, 4])],
            sched: ScheduleDomain::Point,
        };
        run(
            add,
            &[f32s(&[4, 4]), f32s(&[4, 4]), f32s(&[4, 4])],
            &f32s(&[4, 4]),
        )
        .unwrap();
    }

    #[test]
    fn effect_classification() {
        let atomic = L1::KScatter {
            space: IndexSpace::new([Dim::Const(4)]),
            axis: 0,
            mode: ScatterMode::Atomic,
            combine: ScatterCombine::Add,
            ops: vec![operand(0, &[4]), operand(1, &[4]), operand(2, &[4])],
            sched: ScheduleDomain::Point,
        };
        assert_eq!(
            effect_of(&Op::L1(atomic.clone())),
            Effect::InPlace(BufferRole(0))
        );
        run(atomic, &[f32s(&[4]), f32s(&[4]), f32s(&[4])], &f32s(&[4])).unwrap();

        let map = L1::KMap {
            space: IndexSpace::new([Dim::Const(4)]),
            body: ScalarExpr::arg(0, Dtype::F32),
            ops: vec![operand(0, &[4])],
            sched: ScheduleDomain::Point,
        };
        assert_eq!(effect_of(&Op::L1(map)), Effect::Pure);
    }

    #[test]
    fn kmerged_constructor_is_the_only_way_in() {
        let key = MergeKey {
            m: Dim::Const(4),
            n: Dim::Const(4),
            k: Dim::Const(4),
            batch: Dim::Const(1),
            splits: 1,
            dtype: Dtype::F32,
            family: Family::Coop,
        };
        let dom = || ScheduleDomain::Map(MapDomain::linear_over(&caps(), &f32s(&[4, 4]).shape));
        // An epilogue-carrying segment is unbuildable.
        assert!(
            KMerged::new(
                WaveCat::Matmul,
                [
                    MergeSegment {
                        id: Id(1),
                        key,
                        has_epilogue: false
                    },
                    MergeSegment {
                        id: Id(2),
                        key,
                        has_epilogue: true
                    },
                ],
                dom(),
            )
            .is_err()
        );

        // So is a key mismatch.
        let other = MergeKey {
            n: Dim::Const(8),
            ..key
        };
        assert!(
            KMerged::new(
                WaveCat::Matmul,
                [
                    MergeSegment {
                        id: Id(1),
                        key,
                        has_epilogue: false
                    },
                    MergeSegment {
                        id: Id(2),
                        key: other,
                        has_epilogue: false
                    },
                ],
                dom(),
            )
            .is_err()
        );

        assert!(KMerged::new(WaveCat::Matmul, [], dom()).is_err());

        let good = KMerged::new(
            WaveCat::Matmul,
            [MergeSegment {
                id: Id(1),
                key,
                has_epilogue: false,
            }],
            dom(),
        )
        .unwrap();
        assert_eq!(good.key(), key);
        run(L1::KMerged(good), &[f32s(&[4, 4])], &f32s(&[4, 4])).unwrap();
    }

    /// A composite whose domain is not the one its members' shared index
    /// space implies is refused. Without this the field is a place a rule may
    /// write anything, and `SchedPoint::Point` — a node opting out of the
    /// search entirely — reads as legal.
    #[test]
    fn a_composite_carrying_a_foreign_domain_is_refused() {
        let members = smallvec::smallvec![Id(1), Id(2)];
        let good = L1::KRegion {
            members: members.clone(),
            live_outs: smallvec::smallvec![0],
            sched: ScheduleDomain::Map(MapDomain::linear_over(&caps(), &f32s(&[4, 4]).shape)),
        };
        run(good, &[f32s(&[4, 4]), f32s(&[4, 4])], &f32s(&[4, 4])).unwrap();

        let stale = L1::KRegion {
            members: members.clone(),
            live_outs: smallvec::smallvec![0],
            sched: ScheduleDomain::Point,
        };
        assert!(run(stale, &[f32s(&[4, 4]), f32s(&[4, 4])], &f32s(&[4, 4])).is_err());

        // And a domain derived from the wrong extent: `[4, 4]` is 16
        // elements, which at a 32-wide subgroup affords no tile at all, so a
        // domain generated for 8192 offers points this node cannot fill.
        let wrong_shape = L1::KRegion {
            members,
            live_outs: smallvec::smallvec![0],
            sched: ScheduleDomain::Map(MapDomain::linear(&caps(), 8192)),
        };
        assert!(run(wrong_shape, &[f32s(&[4, 4]), f32s(&[4, 4])], &f32s(&[4, 4])).is_err());
    }

    #[test]
    fn operand_access_predicates() {
        // `Pack` into a non-contiguous destination is illegal.
        let packed = Operand {
            src: Id(0),
            layout: Layout::contiguous(&[Dim::Const(4)]),
            access: AccessPlan::Pack {
                into: Layout::from_parts(Dim::Const(0), &[Dim::Const(4)], &[Dim::Const(2)])
                    .unwrap(),
            },
        };
        let op = L1::KMap {
            space: IndexSpace::new([Dim::Const(4)]),
            body: ScalarExpr::arg(0, Dtype::F32),
            ops: vec![packed],
            sched: ScheduleDomain::Point,
        };
        let err = run(op, &[f32s(&[4])], &f32s(&[4])).unwrap_err();
        assert!(format!("{err}").contains("operand 0"));

        // An `Unflatten` map of the wrong rank is illegal.
        let unflat = Operand {
            src: Id(0),
            layout: Layout::contiguous(&[Dim::Const(4)]),
            access: AccessPlan::Unflatten(crate::shape::MultiFlattenMap::affine(&[2, 2], &[2, 1])),
        };
        let op = L1::KMap {
            space: IndexSpace::new([Dim::Const(4)]),
            body: ScalarExpr::arg(0, Dtype::F32),
            ops: vec![unflat],
            sched: ScheduleDomain::Point,
        };
        assert!(run(op, &[f32s(&[4])], &f32s(&[4])).is_err());

        // A `Gather` is always legal, whatever its layout.
        let gather = Operand {
            src: Id(0),
            layout: Layout::contiguous(&[Dim::Const(9)]),
            access: AccessPlan::Gather,
        };
        let op = L1::KMap {
            space: IndexSpace::new([Dim::Const(4)]),
            body: ScalarExpr::arg(0, Dtype::F32),
            ops: vec![gather],
            sched: ScheduleDomain::Point,
        };
        run(op, &[f32s(&[9])], &f32s(&[4])).unwrap();
    }

    #[test]
    fn an_opdef_with_constant_work_is_rejected() {
        fn infer(ins: &[ValueFacts]) -> Result<ValueFacts> {
            ins.first()
                .cloned()
                .ok_or_else(|| Error::Shape("no operand".into()))
        }
        fn constant(_: &[ValueFacts], _: &ValueFacts) -> Work {
            Work {
                macs: 1,
                ..Work::default()
            }
        }
        fn real(_: &[ValueFacts], out: &ValueFacts) -> Work {
            Work {
                macs: out.elements().unwrap_or(1),
                ..Work::default()
            }
        }
        let def = |name, work: fn(&[ValueFacts], &ValueFacts) -> Work| OpDef {
            name,
            tag: crate::ir::OpTag::Ext,
            verify: |_| Ok(()),
            infer,
            work,
            adjoint: None,
            lower_per_target: &[],
            effect: Effect::Pure,
        };

        let mut registry = OpDefRegistry::new();
        let bad = registry.register(def("top_k_placeholder", constant));
        let good = registry.register(def("top_k", real));

        let ext = |d| L1::Ext {
            def: d,
            ops: vec![operand(0, &[4])],
            attrs: crate::ir::AttrId(0),
        };
        let facts = f32s(&[4]);
        let err = run_with(ext(bad), std::slice::from_ref(&facts), &facts, &registry).unwrap_err();
        assert!(format!("{err}").contains("does not vary with shape"));
        run_with(ext(good), std::slice::from_ref(&facts), &facts, &registry).unwrap();
    }

    #[test]
    fn no_l1_node_names_a_buffer_offset() {
        let offset = Operand {
            src: Id(0),
            layout: Layout::from_parts(Dim::Const(64), &[Dim::Const(4)], &[Dim::Const(1)]).unwrap(),
            access: AccessPlan::Alias,
        };
        let op = L1::KMap {
            space: IndexSpace::new([Dim::Const(4)]),
            body: ScalarExpr::arg(0, Dtype::F32),
            ops: vec![offset],
            sched: ScheduleDomain::Point,
        };
        let err = run(op, &[f32s(&[4])], &f32s(&[4])).unwrap_err();
        assert!(format!("{err}").contains("offset"));
    }
}
