//! CPU-exclusive lowering rules.
//!
//! `widen-compute` widens f16/bf16 to f32 registers, computes, and narrows on
//! store. Non-contiguous access is four lowering alternatives — contiguous,
//! broadcast/splat, unit-inner-stride sub-slice, general gather — plus a
//! `Pack` operand access.
//!
//! Every guard below reads only [`Facts`]: legality, never profitability.

use fusor2_ir::device::Caps;
use fusor2_ir::dtype::Dtype;
use fusor2_ir::egraph::{Builder, Facts, Id, Rule, RuleTag};
use fusor2_ir::ir::launch::{AccessPlan, Launch, MapDomain, MapTiling, Operand, ScheduleDomain};
use fusor2_ir::ir::{Level, Node, Op, OpTag};
use fusor2_ir::rule;
use fusor2_ir::scalar::ScalarExpr;
use fusor2_ir::shape::{Layout, MultiFlattenMap};

rule!(
    WIDEN_COMPUTE,
    level = Level::Launch,
    head = OpTag::LaunchMap,
    tag = RuleTag::Additive,
    apply = widen_compute,
);

rule!(
    SELECT_VECTOR_WIDTH,
    level = Level::Launch,
    head = OpTag::LaunchMap,
    tag = RuleTag::Additive,
    apply = select_vector_width,
);

rule!(
    ACCESS_CONTIGUOUS,
    level = Level::Launch,
    head = OpTag::LaunchMap,
    tag = RuleTag::Additive,
    apply = access_contiguous,
);

rule!(
    ACCESS_BROADCAST,
    level = Level::Launch,
    head = OpTag::LaunchMap,
    tag = RuleTag::Additive,
    apply = access_broadcast,
);

rule!(
    ACCESS_UNIT_INNER,
    level = Level::Launch,
    head = OpTag::LaunchMap,
    tag = RuleTag::Additive,
    apply = access_unit_inner,
);

rule!(
    ACCESS_GATHER,
    level = Level::Launch,
    head = OpTag::LaunchMap,
    tag = RuleTag::Additive,
    apply = access_gather,
);

rule!(
    PARALLEL_OUTER,
    level = Level::Launch,
    head = OpTag::LaunchMap,
    tag = RuleTag::Additive,
    apply = parallel_outer,
);

/// Every rule this backend contributes; the order carries no semantics.
pub static CPU_RULES: &[Rule] = &[
    WIDEN_COMPUTE,
    SELECT_VECTOR_WIDTH,
    ACCESS_CONTIGUOUS,
    ACCESS_BROADCAST,
    ACCESS_UNIT_INNER,
    ACCESS_GATHER,
    PARALLEL_OUTER,
];

fn as_kmap(node: &Node) -> Option<&Launch> {
    match &node.op {
        Op::Launch(l @ Launch::Map { .. }) => Some(l),
        _ => None,
    }
}

fn kmap_parts(node: &Node) -> Option<(&Launch, &Vec<Operand>, &ScheduleDomain, &ScalarExpr)> {
    match &node.op {
        Op::Launch(
            l @ Launch::Map {
                ops, sched, body, ..
            },
        ) => Some((l, ops, sched, body)),
        _ => None,
    }
}

fn rebuild(node: &Node, ops: Vec<Operand>, sched: ScheduleDomain, body: ScalarExpr) -> Option<Launch> {
    match &node.op {
        Op::Launch(Launch::Map { space, .. }) => Some(Launch::Map {
            space: space.clone(),
            body,
            ops,
            sched,
        }),
        _ => None,
    }
}

/// Widen f16/bf16 storage to f32 registers, compute, narrow on store.
pub(crate) fn widen_compute(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let (_, ops, sched, body) = kmap_parts(node)?;
    let narrow = |d: Dtype| matches!(d, Dtype::F16 | Dtype::BF16);
    let out = f.own().dtype;
    let any_narrow = narrow(out) || (0..ops.len()).any(|i| f.dtype(i).is_some_and(narrow));
    if !any_narrow {
        return None;
    }
    // Every argument reads as f32; the result narrows exactly once, at the
    // store.
    let widened: Vec<ScalarExpr> = (0..ops.len())
        .map(|i| {
            let d = f.dtype(i).unwrap_or(Dtype::F32);
            let a = ScalarExpr::arg(i as u32, d);
            if narrow(d) {
                ScalarExpr::cast(Dtype::F32, a)
            } else {
                a
            }
        })
        .collect();
    let mut new_body = body.compose(&widened);
    if narrow(out) {
        new_body = ScalarExpr::cast(out, new_body);
    }
    let alt = rebuild(node, ops.clone(), sched.clone(), new_body)?;
    let new_id = b.add_launch(alt).ok()?;
    b.union(id, new_id).ok()
}

/// Mint one `MapTiling` alternative per SIMD width the device reports.
///
/// The width is *not* chosen here — every legal width coexists on the node's
/// `ScheduleDomain` and one extraction picks the winner against the real
/// shapes and the real ISA level.
pub(crate) fn select_vector_width(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let (_, ops, sched, body) = kmap_parts(node)?;
    let ScheduleDomain::Map(dom) = sched else {
        // Give a schedule-less map a domain to start from.
        let dom = width_domain(f.caps(), None);
        let alt = rebuild(node, ops.clone(), ScheduleDomain::Map(dom), body.clone())?;
        let new_id = b.add_launch(alt).ok()?;
        return b.union(id, new_id).ok();
    };
    let widened = width_domain(f.caps(), Some(dom));
    if widened == *dom {
        return None;
    }
    let alt = rebuild(node, ops.clone(), ScheduleDomain::Map(widened), body.clone())?;
    let new_id = b.add_launch(alt).ok()?;
    b.union(id, new_id).ok()
}

fn width_domain(caps: &Caps, base: Option<&MapDomain>) -> MapDomain {
    let mut dom = base.cloned().unwrap_or_default();
    for w in caps.simd_widths.iter().copied() {
        let t = MapTiling {
            dim: None,
            tm: 1,
            vector: w,
        };
        if !dom.tilings.contains(&t) {
            dom.tilings.push(t);
        }
    }
    dom
}

/// A contiguous operand reads through its layout with no index arithmetic.
pub(crate) fn access_contiguous(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
    mint_access(b, id, node, |o| {
        (o.layout.is_contiguous() && o.access != AccessPlan::Alias).then_some(AccessPlan::Alias)
    })
}

/// A stride-0 operand is one scalar splatted across the register.
pub(crate) fn access_broadcast(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
    mint_access(b, id, node, |o| {
        (o.layout.overlaps() && o.access != AccessPlan::Alias).then_some(AccessPlan::Alias)
    })
}

/// A unit inner stride at an outer offset is a contiguous sub-slice, expressed
/// as the explicit index map so the emitter can prove the run length.
pub(crate) fn access_unit_inner(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
    mint_access(b, id, node, |o| {
        let strides = o.layout.strides();
        let unit_inner = strides
            .last()
            .is_some_and(|s| s.known_eq(fusor2_ir::shape::Dim::Const(1)));
        if !unit_inner || o.layout.is_contiguous() {
            return None;
        }
        let map = affine_map(&o.layout)?;
        let plan = AccessPlan::Unflatten(map);
        (o.access != plan).then_some(plan)
    })
}

/// The general form, plus the packed alternative beside it: pack the operand
/// into thread-local scratch once, then read it contiguous. Both stay live and
/// the cost model chooses.
pub(crate) fn access_gather(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
    let gathered = mint_access(b, id, node, |o| {
        (o.access != AccessPlan::Gather && !o.layout.is_contiguous())
            .then_some(AccessPlan::Gather)
    });
    let packed = mint_access(b, id, node, |o| {
        if o.layout.is_contiguous() {
            return None;
        }
        let into = Layout::contiguous(o.layout.shape());
        let plan = AccessPlan::Pack { into };
        (o.access != plan).then_some(plan)
    });
    packed.or(gathered)
}

fn affine_map(layout: &Layout) -> Option<MultiFlattenMap> {
    let mut extents = Vec::with_capacity(layout.rank());
    let mut strides = Vec::with_capacity(layout.rank());
    for (s, d) in layout.shape().iter().zip(layout.strides()) {
        extents.push(s.as_const()? as u32);
        strides.push(d.as_const()? as u32);
    }
    Some(MultiFlattenMap::affine(&extents, &strides))
}

fn mint_access(
    b: &mut Builder<'_>,
    id: Id,
    node: &Node,
    pick: impl Fn(&Operand) -> Option<AccessPlan>,
) -> Option<Id> {
    let (_, ops, sched, body) = kmap_parts(node)?;
    let mut changed = false;
    let mut next = ops.clone();
    for o in &mut next {
        if let Some(plan) = pick(o) {
            // Through `Operand::respell`, never a bare access swap: an
            // `Unflatten` map stated independently of the layout does not
            // survive a layout-derived re-spelling, and dropping it re-reads
            // the base densely. See `rules::layout::operand_alias` for related hazards.
            let Some(new) = o.respell(plan) else { continue };
            *o = new;
            changed = true;
        }
    }
    if !changed {
        return None;
    }
    let alt = rebuild(node, next, sched.clone(), body.clone())?;
    let new_id = b.add_launch(alt).ok()?;
    b.union(id, new_id).ok()
}

/// Mark an outer tile loop parallel.
///
/// Parallelism is a scheduling attribute, priced by the cost model against
/// `DeviceFacts::thread_wake_ps`.
///
/// The attribute rides on `MapTiling { dim: Some(0), tm > 1 }`, i.e. an
/// outermost tile loop of `tm` grid points — `launch::grain_for` turns that
/// into the `parallel_for` grain.
pub(crate) fn parallel_outer(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let (_, ops, sched, body) = kmap_parts(node)?;
    if f.caps().threads <= 1 {
        return None;
    }
    let Launch::Map { space, .. } = as_kmap(node)? else {
        return None;
    };
    if space.rank() == 0 {
        return None;
    }
    let mut dom = match sched {
        ScheduleDomain::Map(d) => d.clone(),
        _ => MapDomain::default(),
    };
    let vector = *f.caps().simd_widths.last().unwrap_or(&4);
    let t = MapTiling {
        dim: Some(0),
        tm: f.caps().threads.max(2),
        vector,
    };
    if dom.tilings.contains(&t) {
        return None;
    }
    dom.tilings.push(t);
    let alt = rebuild(node, ops.clone(), ScheduleDomain::Map(dom), body.clone())?;
    let new_id = b.add_launch(alt).ok()?;
    b.union(id, new_id).ok()
}
