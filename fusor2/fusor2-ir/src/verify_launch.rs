//! `verify_launch` — the eight Launch invariants.
//!
//! 1. `Geom::legal(caps)`: lane limits and whole-fragment divisibility.
//! 2. Workgroup footprint checked against the **exact** `arena_plan` value —
//!    the same pure memoized function the Kernel emitter uses, so there is no
//!    estimator and therefore no Launch/Kernel admission mismatch.
//! 3. A nest's write map must be injective unless the nest declares an
//!    associative `combine`. One invariant, three jobs: scatter-add
//!    legality, separating the four `Scatter{Add}` lowerings from an illegal
//!    in-place write, and proving a non-overlapping pool's adjoint is an
//!    elementwise mask.
//! 4. A fold dim may not appear with nonzero stride in the write map.
//! 5. Every operand's `AccessPlan` satisfies that operand's access
//!    predicate. A failed access analysis disqualifies **this rewrite only**.
//! 6. A composite node carries the linear schedule domain its members'
//!    shared index space implies, rather than an unsearchable point.
//! 7. Every node carries an `Effect`.
//! 8. Allocation is *not* described at Launch; a node claiming a buffer is an
//!    error.

use crate::carrier::SlotTy;
use crate::device::Caps;
use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::ir::{Op, VerifyCtx};
use crate::ir::logical::ScatterCombine;
use crate::ir::launch::{
    AccessPlan, CoopGeom, Effect, IndexSpace, Launch, Operand, ScheduleDomain,
};
use crate::ir::kernel::{
    ArenaPlanner, ElementType, MemoryLevel, ScalarElement, TileDecl, TileLayout, Tiles,
};
use crate::semantics::effect_of;
use crate::shape::{Dim, Layout};
use std::sync::Arc;

/// Workgroup tiles a cooperative geometry declares, before packing.
///
/// A-tile `[bm, bk]` and B-tile `[bk, bn / n_passes]`, each replicated
/// `staging` times, plus one accumulator staging tile `[bm, bn / n_passes]`
/// when the store element is not `F32` (an f32 accumulator written into
/// narrower memory needs the staging pass unless the device supports a
/// mixed-precision cooperative store).
///
/// **This is the single source of coop tile shapes.** `verify_launch` and
/// `fusor2-tile`'s `domains::coop` both call it, so an admitted geometry and
/// a planned one cannot disagree.
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

/// Verify one Launch node against `caps` and the exact arena plan.
pub fn verify_launch(cx: &VerifyCtx<'_>, planner: &dyn ArenaPlanner) -> Result<()> {
    let Op::Launch(op) = &cx.node.op else {
        return Err(Error::verify(
            crate::ir::Level::Launch,
            cx.id,
            "verify_launch applied to a node that is not Launch",
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
                    "operand {i} names a buffer offset ({}); allocation is not described at Launch",
                    o.layout.offset()
                ),
            ));
        }
    }

    // The `verify_l0` constant-work tripwire, applied to the one Launch variant
    // whose row comes from outside the crate. An `OpDef` registering
    // `Work { macs: 1, .. }` is exactly the reference's
    // `Attention { work: 1 }` placeholder wearing an extension hat.
    if let Launch::Ext { def, .. } = op
        && let Some(d) = cx.registry.get(*def)
    {
        let small = (d.work)(cx.operands, cx.result);
        let doubled_ins: Vec<crate::facts::ValueFacts> =
            cx.operands.iter().map(doubled).collect();
        let doubled_out = doubled(cx.result);
        let large = (d.work)(&doubled_ins, &doubled_out);
        let has_const = cx
            .operands
            .iter()
            .chain(std::iter::once(cx.result))
            .flat_map(|f| f.shape.iter())
            .any(|dim| dim.as_const().is_some());
        if has_const && small == large && small != crate::facts::Work::default() {
            return Err(relabel(
                cx,
                format!("OpDef `{}`: work() does not vary with shape", d.name),
            ));
        }
    }

    Ok(())
}

/// Every `Const` dim doubled — the second binding the work tripwire prices.
fn doubled(f: &crate::facts::ValueFacts) -> crate::facts::ValueFacts {
    let mut out = f.clone();
    for d in out.shape.iter_mut() {
        if let Dim::Const(v) = *d {
            *d = Dim::Const(v.saturating_mul(2));
        }
    }
    out
}

/// Invariant 6: a composite node's schedule domain is the one its members'
/// shared index space implies.
///
/// A `Region` is a list of Launch nodes run in one dispatch over one linearized
/// index to both backends, so its geometry is
/// [`crate::ir::launch::MapDomain::linear_over`] of the value they land, and
/// nothing else. Checking it against the *node's own inferred shape* rather
/// than against whatever the minting rule felt like is what makes the domain
/// a property of the node instead of a field a rule may drift.
///
/// The clause is exact rather than a bound: the mint site calls the same
/// generator on the same facts, so an inequality is a rule that stopped
/// deriving the domain, not a legal variation.
fn check_composite_domain(cx: &VerifyCtx<'_>, op: &Launch) -> Result<()> {
    let sched = match op {
        Launch::Region { sched, .. } => sched,
        _ => return Ok(()),
    };
    let want = ScheduleDomain::Map(crate::ir::launch::MapDomain::linear_over(
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
    op: &Launch,
    sched: &ScheduleDomain,
    caps: &Caps,
    planner: &dyn ArenaPlanner,
) -> Result<()> {
    if sched.is_empty() {
        return Err(Error::Legality(
            "schedule domain is empty; this node is unselectable".into(),
        ));
    }

    // Every lowering indexes the flattened iteration space in `u32` — flat
    // workgroup ids, `Addr::Linear`, loop counters. A space past `u32::MAX`
    // is therefore *unaddressable*, not merely slow: the fold spelling of a
    // 2048-cube matmul carries `[2048, 2048, 2048]` = 2^33 iterations, its
    // flat index wraps, and the member sweep caught it summing garbage while
    // every small shape stayed green. This is an addressing-capacity bound
    // exactly like `max_storage_buffers_per_shader_stage`, refused here so
    // extraction loses the member instead of the dispatch computing wrong.
    if let Some(iters) = op.iter_space().iterations()
        && iters > u64::from(u32::MAX)
    {
        return Err(Error::Legality(format!(
            "iteration space of {iters} elements exceeds u32 flat addressing"
        )));
    }

    let elem = element_of(store_dtype(op));
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
                if p.vector == 0 || p.subgroups == 0 || p.cols == 0 {
                    return Err(Error::Legality(format!(
                        "sgemv params {p:?} have a zero term"
                    )));
                }
                // A multi-column workgroup hands each subgroup an equal,
                // whole number of columns; a remainder would leave columns
                // no subgroup owns.
                if p.cols > 1 && p.cols % p.subgroups != 0 {
                    return Err(Error::Legality(format!(
                        "sgemv params {p:?} spread {} columns over {} subgroups unevenly",
                        p.cols, p.subgroups
                    )));
                }
                if p.subgroups.saturating_mul(subgroup_width) > max_lanes {
                    return Err(Error::Legality(format!(
                        "sgemv params {p:?} want {} lanes, over the {max_lanes} limit",
                        p.subgroups.saturating_mul(subgroup_width)
                    )));
                }
                // A split lane window re-tiles the subgroup's pass; the
                // arithmetic below is exactly what makes that a bijection
                // onto the same `width * vector` consecutive elements, so a
                // violation is a wrong-answer kernel, not a slow one.
                if p.parts <= 1 {
                    if p.gap != 0 {
                        return Err(Error::Legality(format!(
                            "sgemv params {p:?} carry a gap without a split window"
                        )));
                    }
                } else {
                    let run = p.vector / p.parts.max(1);
                    if p.cols <= 1
                        || p.vector % p.parts != 0
                        || run == 0
                        || p.gap % run.max(1) != 0
                        || p.gap <= run
                        || (subgroup_width * run) % p.gap.max(1) != 0
                    {
                        return Err(Error::Legality(format!(
                            "sgemv params {p:?} split the lane window illegally \
                             at subgroup width {subgroup_width}"
                        )));
                    }
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
            // generator already filters on exactly this, so a domain built
            // there cannot fail here; what this catches is a domain minted
            // anywhere else, which §4.2 would otherwise turn into a
            // `verify_plan` crash at extraction rather than a lost alternative.
            let carrier_lanes = fold_carrier_lanes(op);
            for s in &domain.strategies {
                let group = s.lane_group(subgroup_width);
                if group == 0 || group > max_lanes {
                    return Err(Error::Legality(format!(
                        "fold strategy {s:?} wants a lane group of {group}, over {max_lanes}"
                    )));
                }
                if let Some((lanes, acc_bytes)) = carrier_lanes {
                    let bytes = crate::ir::launch::fold_scratch_bytes(
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
    let Op::Launch(op) = &cx.node.op else {
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

/// Invariant 4: a `Fold`'s reduced axis must be absent from the write map.
/// A fold dim indexing the output is a scatter, not a reduction, so the
/// result rank has to be the space rank minus one, plus the carrier's axis.
fn check_fold_axis_not_written(cx: &VerifyCtx<'_>, op: &Launch) -> Result<()> {
    let Launch::Fold {
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
pub fn check_operand_access(op: &Launch) -> Result<()> {
    // A contraction side holds a list, and an empty one would make its `pre`
    // a constant — a `Map`, not a contraction, and a node the lowerings would
    // read `ops[0]` off. `ContractSide::primary` is written against this.
    if let Launch::Contract { a, b, .. } = op {
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

/// Invariant 7's classification: `Scatter` writing through operand 0 with
/// atomics or a `Set` combine mutates state; everything else is pure.
fn expected_effect(op: &Launch) -> Effect {
    match op {
        Launch::Scatter { mode, combine, .. }
            if matches!(mode, crate::ir::launch::ScatterMode::Atomic)
                || matches!(combine, ScatterCombine::Set) =>
        {
            Effect::InPlace(crate::ir::launch::BufferRole(0))
        }
        _ => Effect::Pure,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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
    // **A `Scalar` slot is one accumulator, not one per promoted position.**
    //
    // It is updated once per iteration step, so its `lift` is evaluated at a
    // single promoted position. An operand that varies along a promoted axis
    // read there would contribute position 0's value at every position — a
    // wrong number, not a slow one, and invisible in any test whose promoted
    // extent is 1. A `Vector` slot has a register per position and may read
    // anything. This is the clause that makes a mixed `[Scalar, Vector]`
    // accumulator safe to mint at all.
    if let Op::Launch(o) = &cx.node.op {
        let ops = operands_of(o);
        let varies: Vec<bool> = ops
            .iter()
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
            let mut used = Vec::new();
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
    // Every `ScalarExpr` on a `Fold` is written against `iter_space()`, so the
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
fn collect_args(e: &crate::scalar::ScalarExpr, out: &mut Vec<u32>) {
    use crate::scalar::ScalarKind as K;
    match e.kind() {
        K::Arg(i) => {
            if !out.contains(i) {
                out.push(*i);
            }
        }
        K::Un { x, .. } | K::Cast { x, .. } | K::Bitcast { x, .. } | K::Round { x, .. }
        | K::Splat { x, .. } => collect_args(x, out),
        K::Bin { a, b, .. } | K::Cmp { a, b, .. } | K::Dot { a, b } => {
            collect_args(a, out);
            collect_args(b, out);
        }
        K::Select { c, t, f } => {
            collect_args(c, out);
            collect_args(t, out);
            collect_args(f, out);
        }
        K::Lit(_) | K::Uniform(_) | K::IndexOf(_) => {}
    }
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
    let Op::Launch(Launch::Fold { post, .. }) = &cx.node.op else {
        return false;
    };
    post.iter().any(|e| reads_index_of(e, axis))
}

fn reads_index_of(e: &crate::scalar::ScalarExpr, axis: u32) -> bool {
    use crate::scalar::ScalarKind as K;
    match e.kind() {
        K::IndexOf(a) => *a == axis,
        K::Un { x, .. } | K::Cast { x, .. } | K::Bitcast { x, .. } | K::Round { x, .. } => {
            reads_index_of(x, axis)
        }
        K::Bin { a, b, .. } | K::Cmp { a, b, .. } | K::Dot { a, b } => {
            reads_index_of(a, axis) || reads_index_of(b, axis)
        }
        K::Select { c, t, f } => {
            reads_index_of(c, axis) || reads_index_of(t, axis) || reads_index_of(f, axis)
        }
        K::Splat { x, .. } => reads_index_of(x, axis),
        _ => false,
    }
}

fn relabel(cx: &VerifyCtx<'_>, msg: String) -> Error {
    Error::verify(crate::ir::Level::Launch, cx.id, msg)
}

fn declares_associative_combine(op: &Launch) -> bool {
    match op {
        // Associativity is declared on the carrier, not derived from a name:
        // a non-associative carrier is legal but may not be tree-reduced.
        Launch::Fold { carrier, .. } => carrier.associative,
        Launch::Scatter { combine, .. } => matches!(combine, ScatterCombine::Add),
        _ => false,
    }
}

fn write_layout(op: &Launch, cx: &VerifyCtx<'_>) -> Layout {
    match op {
        // A scatter writes through its base operand's layout.
        Launch::Scatter { ops, .. } => ops
            .first()
            .map(|o| o.layout.clone())
            .unwrap_or_else(|| Layout::contiguous(&cx.result.shape)),
        _ => Layout::contiguous(&cx.result.shape),
    }
}

fn index_space_of(op: &Launch) -> Option<&IndexSpace> {
    match op {
        Launch::Map { space, .. }
        | Launch::Fold { space, .. }
        | Launch::Gather { space, .. }
        | Launch::Scatter { space, .. } => Some(space),
        _ => None,
    }
}

/// Every `Operand` a node carries, in `children_of` order.
fn operands_of(op: &Launch) -> Vec<Operand> {
    match op {
        Launch::Map { ops, .. }
        | Launch::Fold { ops, .. }
        | Launch::Gather { ops, .. }
        | Launch::Scatter { ops, .. }
        | Launch::Ext { ops, .. } => ops.clone(),
        Launch::Contract { a, b, .. } => a.ops.iter().chain(b.ops.iter()).cloned().collect(),
        Launch::Region { .. } => Vec::new(),
    }
}

/// The element a node stores, which is what a staged coop tile holds.
/// A `Fold`'s `(accumulator lanes, bytes per lane)`, or `None` when the node
/// is not a fold or its carrier's lane count is symbolic.
///
/// A symbolic `Vector` slot extent is allocatable on neither backend, and
/// `verify_l0` clause 3 already rejects one, so `None` here means "not a
/// fold" in every graph that got this far.
fn fold_carrier_lanes(op: &Launch) -> Option<(u64, u64)> {
    match op {
        Launch::Fold { carrier, acc, .. } => Some((carrier.lanes()?, acc.byte_size())),
        _ => None,
    }
}

fn store_dtype(op: &Launch) -> Dtype {
    match op {
        Launch::Map { body, .. } => body.dtype(),
        Launch::Fold { acc, .. } => *acc,
        Launch::Contract { post, .. } => post.dtype(),
        _ => Dtype::F32,
    }
}

fn element_of(d: Dtype) -> ScalarElement {
    match d {
        Dtype::F16 => ScalarElement::F16,
        Dtype::BF16 => ScalarElement::BF16,
        Dtype::U32 => ScalarElement::U32,
        Dtype::I32 => ScalarElement::I32,
        _ => ScalarElement::F32,
    }
}
