//! Operand access alternatives. Access is an attribute of the *edge*, so one
//! reader may alias a strided parameter slice while another packs it — the
//! flat-parameter / gradient-concat case and the im2col operand case
//! coexisting in one graph.
//!
//! Each rule mints **one** alternative of the *reading* node with **one**
//! operand's access changed. `Rule::head` is a single tag, so the four rules
//! are spread across the two most common readers: three on `Map` and the
//! pack rule on `Contract`.

use crate::egraph::{Builder, Facts, Id, RuleTag};
use crate::ir::launch::{AccessPlan, ContractSide, Launch, Operand};
use crate::ir::{Level, Node, Op, OpTag};
use crate::rule;
use crate::shape::{AxisGroup, Layout, MultiFlattenMap, SubAxis};
use smallvec::SmallVec;

rule!(
    OPERAND_ALIAS,
    level = Level::Launch,
    head = OpTag::LaunchMap,
    tag = RuleTag::Additive,
    apply = operand_alias,
);

rule!(
    OPERAND_GATHER,
    level = Level::Launch,
    head = OpTag::LaunchMap,
    tag = RuleTag::Additive,
    apply = operand_gather,
);

rule!(
    OPERAND_PACK,
    level = Level::Launch,
    head = OpTag::LaunchContract,
    tag = RuleTag::Additive,
    apply = operand_pack,
);

rule!(
    OPERAND_UNFLATTEN,
    level = Level::Launch,
    head = OpTag::LaunchMap,
    tag = RuleTag::Additive,
    apply = operand_unflatten,
);

/// Rebuild a `Map` with the first operand that `pick` rewrites replaced.
fn remap_kmap(
    b: &mut Builder<'_>,
    id: Id,
    node: &Node,
    pick: impl Fn(&Operand) -> Option<Operand>,
) -> Option<Id> {
    let Op::Launch(Launch::Map {
        space,
        body,
        ops,
        sched,
    }) = &node.op
    else {
        return None;
    };
    let slot = ops.iter().position(|o| pick(o).is_some())?;
    let mut new_ops = ops.clone();
    new_ops[slot] = pick(&ops[slot])?;
    let alt = b
        .add_launch(Launch::Map {
            space: space.clone(),
            body: body.clone(),
            ops: new_ops,
            sched: sched.clone(),
        })
        .ok()?;
    b.union(id, alt).ok()
}

/// Read this operand straight through its own strides.
///
/// An `Alias` addresses through `layout`'s strides and nothing else, so
/// re-spelling an edge as one is sound only when the plan it replaces
/// addresses the same way. [`AccessPlan::Gather`] and [`AccessPlan::Pack`]
/// always do, and so does an [`AccessPlan::Unflatten`] whose map is
/// `decompose(layout)`. An `Unflatten` whose map was stated independently of
/// the layout (`rules::sink::fold_operand_views` mints those) carries the
/// view's index arithmetic while the layout carries only the base's shape;
/// dropping that map re-reads the base densely and loses the broadcast,
/// transpose or window the view expressed.
pub fn operand_alias(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
    remap_kmap(b, id, node, |o| {
        if matches!(o.access, AccessPlan::Alias) {
            return None;
        }
        // `Operand::respell` is the address-preservation judgement:
        // layout-derived plans move freely (which keeps a `Dim::Sym` edge
        // rewritable), an independently-stated `Unflatten` map requires
        // `AddressMap` equality and declines when undecidable.
        o.respell(AccessPlan::Alias)
    })
}

/// Read this operand through a per-element address computation.
///
/// Only minted for a layout that is not already dense row-major: over a
/// contiguous layout a gather and an alias name the *same* index map, so
/// minting both would put one access in the graph twice under two spellings.
pub fn operand_gather(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
    remap_kmap(b, id, node, |o| {
        if matches!(o.access, AccessPlan::Gather) || o.layout.is_contiguous() {
            return None;
        }
        // Through `respell`: a gather derives its addresses from the layout,
        // so re-spelling an independently-stated `Unflatten` map would
        // silently re-read the base densely (see `operand_alias`).
        o.respell(AccessPlan::Gather)
    })
}

/// Stage this operand into a dense tile first. Legal when the packed layout
/// is contiguous and holds exactly as many elements as the operand does.
pub fn operand_pack(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
    let Op::Launch(Launch::Contract {
        m,
        n,
        k,
        batch,
        family,
        post,
        acc,
        a,
        b: rhs,
        sched,
    }) = &node.op
    else {
        return None;
    };
    let repack = |o: &Operand| -> Option<Operand> {
        // Packing a layout that is already dense row-major stages it into a
        // byte-identical tile — the same access under two spellings.
        if matches!(o.access, AccessPlan::Pack { .. }) || o.layout.is_contiguous() {
            return None;
        }
        let into = Layout::contiguous(o.layout.shape());
        if !into.is_contiguous() || elements(&into)? != elements(&o.layout)? {
            return None;
        }
        // Packing stages the elements the *layout* addresses: an
        // independently-stated `Unflatten` map must survive the re-spelling
        // or the rule declines.
        o.respell(AccessPlan::Pack { into })
    };
    // Each operand of a side is loaded through its own access plan, so packing
    // one and aliasing its neighbour is sound. Exactly one alternative is
    // minted per fire — the first packable operand in `children_of` order.
    let pack_first = |side: &ContractSide| -> Option<ContractSide> {
        let (i, packed) = side
            .ops
            .iter()
            .enumerate()
            .find_map(|(i, o)| Some((i, repack(o)?)))?;
        let mut out = side.clone();
        out.ops[i] = packed;
        Some(out)
    };
    let (new_a, new_b) = match pack_first(a) {
        Some(pa) => (pa, rhs.clone()),
        None => (a.clone(), pack_first(rhs)?),
    };
    let alt = b
        .add_launch(Launch::Contract {
            m: *m,
            n: *n,
            k: *k,
            batch: *batch,
            family: *family,
            post: post.clone(),
            acc: *acc,
            a: new_a,
            b: new_b,
            sched: sched.clone(),
        })
        .ok()?;
    b.union(id, alt).ok()
}

/// Read this operand through an explicit index map. Legal only when the
/// operand's layout decomposes into decidable `AxisGroup`s; when it does not,
/// the alternative is simply not minted.
///
/// A dense row-major layout decomposes into exactly the map an alias already
/// implies, so it is skipped for the same canonicalization reason as
/// [`operand_gather`].
pub fn operand_unflatten(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
    remap_kmap(b, id, node, |o| {
        if matches!(o.access, AccessPlan::Unflatten(_)) || o.layout.is_contiguous() {
            return None;
        }
        let map = decompose(&o.layout)?;
        Some(Operand {
            src: o.src,
            layout: o.layout.clone(),
            access: AccessPlan::Unflatten(map),
        })
    })
}

fn elements(l: &Layout) -> Option<u64> {
    l.shape()
        .iter()
        .try_fold(1u64, |acc, d| acc.checked_mul(d.as_const()?))
}

/// One `AxisGroup` per logical axis of a decidable strided layout.
fn decompose(l: &Layout) -> Option<MultiFlattenMap> {
    let mut groups: SmallVec<[AxisGroup; 4]> = SmallVec::new();
    for (d, s) in l.shape().iter().zip(l.strides()) {
        let extent = u32::try_from(d.as_const()?).ok()?;
        let stride = u32::try_from(s.as_const()?).ok()?;
        groups.push(AxisGroup {
            sub_axes: smallvec::smallvec![SubAxis { extent, stride }],
        });
    }
    (!groups.is_empty()).then_some(MultiFlattenMap { groups })
}
