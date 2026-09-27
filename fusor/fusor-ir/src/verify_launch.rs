//! `verify_launch` — the eight Launch invariants.
//!
//! 1. `Geom::legal(caps)`: lane limits and whole-fragment divisibility.
//! 2. Workgroup footprint against the exact `arena_plan` the emitter uses.
//! 3. The write map is injective unless the nest declares an associative
//!    `combine`.
//! 4. A fold dim may not appear with nonzero stride in the write map.
//! 5. Every operand's `AccessPlan` satisfies its access predicate.
//! 6. A composite sequences at least two independently scheduled members.
//! 7. Every node's `Effect` is derived by `semantics::effect_of`.
//! 8. Allocation is not described at Launch; an operand offset is an error.

use crate::carrier::SlotTy;
use crate::device::Caps;
use crate::dtype::Dtype;
use crate::error::{Error, Result};
use crate::ir::kernel::{ArenaPlanner, ScalarElement};
use crate::ir::launch::{AccessPlan, IndexSpace, Launch, ScheduleDomain};
use crate::ir::logical::ScatterCombine;
use crate::ir::{Op, VerifyCtx};
use crate::shape::{Dim, Layout};

/// Verify one Launch node against `caps` and the exact arena plan.
pub fn verify_launch(cx: &VerifyCtx<'_>, planner: &dyn ArenaPlanner) -> Result<()> {
    let Op::Launch(op) = &cx.node.op else {
        return Err(Error::verify(
            crate::ir::Level::Launch,
            cx.id,
            "verify_launch applied to a node that is not Launch",
        ));
    };

    if let Launch::StreamFold {
        producer,
        fold,
        operand,
        sched,
    } = op
    {
        if !op.stream_compatible() || !cx.result.numeric.reassoc {
            return Err(relabel(
                cx,
                "a streamed Fold needs a bounded scalar producer and reassociation".into(),
            ));
        }
        let (count, produced, inputs) =
            crate::semantics::infer_launch::stream_inputs(producer, fold, *operand, cx.operands)?;
        for (recipe, operands, result) in [
            (producer.as_ref(), &cx.operands[..count], &produced),
            (fold.as_ref(), inputs.as_slice(), cx.result),
        ] {
            let node = crate::ir::Node {
                op: Op::Launch(recipe.clone()),
                level: crate::ir::Level::Launch,
                children: crate::semantics::children::children_launch(recipe),
            };
            verify_launch(
                &VerifyCtx {
                    node: &node,
                    id: cx.id,
                    operands,
                    result,
                    caps: cx.caps,
                },
                planner,
            )?;
        }
        return check_schedule_domain(fold, sched, cx.caps, planner);
    }

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
    check_composite_domain(op).map_err(|e| relabel(cx, format!("{e}")))?;

    // 8.
    for (i, o) in op.operands().enumerate() {
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

    Ok(())
}

/// Composite members keep their own schedules.
fn check_composite_domain(op: &Launch) -> Result<()> {
    let (sched, members) = match op {
        Launch::Slab {
            sched,
            slabs,
            members,
        } => {
            if *slabs < 2 {
                return Err(Error::Legality(format!(
                    "a Slab needs at least two slabs, got {slabs}"
                )));
            }
            (sched, members)
        }
        Launch::Group { sched, members } => (sched, members),
        _ => return Ok(()),
    };
    if members.len() < 2 || *sched != ScheduleDomain::Point {
        return Err(Error::Legality(format!(
            "a composite needs at least two members and a Point domain, got {} and {sched:?}",
            members.len()
        )));
    }
    Ok(())
}

/// Invariant 1+2: every point of `sched` is structurally legal and fits the
/// exact workgroup footprint. A declared-empty domain is unselectable, so it
/// fails.
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

    // Lowerings index the flat iteration space in `u32`; past `u32::MAX` it
    // wraps and computes garbage, so the member is refused here.
    if let Some(iters) = op.iter_space().iterations()
        && iters > u64::from(u32::MAX)
    {
        return Err(Error::Legality(format!(
            "iteration space of {iters} elements exceeds u32 flat addressing"
        )));
    }

    let elem = element_of(match op {
        Launch::Contract { a, .. } => a.pre.dtype(),
        _ => store_dtype(op),
    });
    let subgroup_width = caps.subgroup_width();
    let max_lanes = caps.limits.max_compute_invocations_per_workgroup;
    let max_storage = caps.limits.max_compute_workgroup_storage_size;

    match sched {
        ScheduleDomain::Coop(domain) => {
            for point in &domain.schedules {
                let (geom, staging) = (point.geom, point.staging);
                if !geom.legal(subgroup_width, max_lanes) || !(1..=2).contains(&staging) {
                    return Err(Error::Legality(format!(
                        "illegal cooperative schedule {point:?}"
                    )));
                }
                let bytes = planner
                    .workgroup_bytes(&crate::ir::launch::coop_tiles(geom, elem, staging), caps)?;
                if bytes > max_storage {
                    return Err(Error::Legality(format!(
                        "coop {point:?} needs {bytes} workgroup bytes, limit {max_storage}"
                    )));
                }
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
                // A remainder would leave columns no subgroup owns.
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
                // A split window must biject onto the pass's `width * vector`
                // elements, or the kernel is wrong.
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
                        || !(subgroup_width * run).is_multiple_of(p.gap.max(1))
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
            // Scratch is decidable from the carrier's lanes; a promoted
            // carrier needs `lanes * block * acc_bytes`.
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

/// Invariant 3: the write map is injective (every non-unit axis stride nonzero
/// and distinct), or the nest declares an associative combine.
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

/// Invariant 4: a `Fold`'s reduced axis is absent from the write map, so the
/// result rank is the space rank minus one, plus the carrier's axis.
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

/// Invariant 5: each operand's `AccessPlan` satisfies its own predicate.
pub fn check_operand_access(op: &Launch) -> Result<()> {
    // `ContractSide::primary` reads `ops[0]`.
    if let Launch::Contract { a, b, .. } = op {
        for (side, which) in [(a, "a"), (b, "b")] {
            if side.is_empty() {
                return Err(Error::Legality(format!(
                    "contraction side {which} reads no operand"
                )));
            }
        }
    }
    let space = op.space();
    for (i, o) in op.operands().enumerate() {
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
                // A contraction has no space: match the layout rank instead.
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

/// The promoted-axis invariants: `vec_axes` is the contiguous block before
/// `axis`, every `Vector` slot spans the promoted extent, and no expression
/// reads a coordinate outside the iteration domain (a botched renumbering).
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
    let promoted: Option<u64> = vec_axes.iter().try_fold(1u64, |a, i| {
        a.checked_mul(space.dims[*i as usize].as_const()?)
    });
    let promoted =
        promoted.ok_or_else(|| relabel(cx, "a promoted axis has a symbolic extent".to_string()))?;
    carrier
        .lanes()
        .ok_or_else(|| relabel(cx, "a promoted carrier has a symbolic lane count".into()))?;
    // Every `Vector` slot spans the promoted extent; `Scalar` slots may mix in.
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
    // A `Scalar` slot is one accumulator, so its lift may not read an operand
    // varying along a promoted axis: it would see only one position.
    if let Op::Launch(o) = &cx.node.op {
        let varies: Vec<bool> = o
            .operands()
            .map(|o| {
                vec_axes
                    .iter()
                    .any(|a| o.varies_along(space, *a) != Some(false))
            })
            .collect();
        for (k, s) in carrier.slots.iter().enumerate() {
            if *s != SlotTy::Scalar {
                continue;
            }
            let mut used = Vec::new();
            carrier.lift[k].collect_args(&mut used);
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

    // Expressions are written against `iter_space()`: legal indices are
    // `0..iter_rank` (not space indices, which are one renumbering behind).
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

fn cx_post_reads(cx: &VerifyCtx<'_>, axis: u32) -> bool {
    let Op::Launch(Launch::Fold { post, .. }) = &cx.node.op else {
        return false;
    };
    post.iter().any(|e| e.reads_axis(axis))
}

fn relabel(cx: &VerifyCtx<'_>, msg: String) -> Error {
    Error::verify(crate::ir::Level::Launch, cx.id, msg)
}

fn declares_associative_combine(op: &Launch) -> bool {
    match op {
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

/// A `Fold`'s `(accumulator lanes, bytes per lane)`, or `None` when the node
/// is not a fold or its lane count is symbolic (rejected by `verify_l0`).
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
