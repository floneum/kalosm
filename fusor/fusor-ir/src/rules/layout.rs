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
use crate::ir::launch::{AccessPlan, Operand};
use crate::ir::{Level, Node, Op, OpTag};
use crate::rule;
use crate::shape::{Layout, MultiFlattenMap, const_elements};

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

/// Mint the reader with the first operand `pick` re-spells replaced, in
/// `Launch::operands` order: a contraction's A side before its B side.
fn respell_first(
    b: &mut Builder<'_>,
    id: Id,
    node: &Node,
    pick: impl Fn(&Operand) -> Option<Operand>,
) -> Option<Id> {
    let Op::Launch(op) = &node.op else {
        return None;
    };
    let (slot, new) = op
        .operands()
        .enumerate()
        .find_map(|(i, o)| Some((i, pick(o)?)))?;
    let mut alt = op.clone();
    *alt.operands_mut().nth(slot)? = new;
    let alt = b.add_launch(alt).ok()?;
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
    respell_first(b, id, node, |o| {
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
    respell_first(b, id, node, |o| {
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
///
/// Each operand of a side is loaded through its own access plan, so packing
/// one and aliasing its neighbour is sound.
pub fn operand_pack(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
    respell_first(b, id, node, |o| {
        // Packing a layout that is already dense row-major stages it into a
        // byte-identical tile — the same access under two spellings.
        if matches!(o.access, AccessPlan::Pack { .. }) || o.layout.is_contiguous() {
            return None;
        }
        let into = Layout::contiguous(o.layout.shape());
        if !into.is_contiguous()
            || const_elements(into.shape())? != const_elements(o.layout.shape())?
        {
            return None;
        }
        // Packing stages the elements the *layout* addresses: an
        // independently-stated `Unflatten` map must survive the re-spelling
        // or the rule declines.
        o.respell(AccessPlan::Pack { into })
    })
}

/// Read this operand through an explicit index map. Legal only when the
/// operand's layout decomposes into decidable `AxisGroup`s; when it does not,
/// the alternative is simply not minted.
///
/// A dense row-major layout decomposes into exactly the map an alias already
/// implies, so it is skipped for the same canonicalization reason as
/// [`operand_gather`].
pub fn operand_unflatten(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
    respell_first(b, id, node, |o| {
        if matches!(o.access, AccessPlan::Unflatten(_)) || o.layout.is_contiguous() {
            return None;
        }
        // One group per axis of a decidable strided layout.
        let groups = o.layout.affine_groups().filter(|g| !g.is_empty())?;
        Some(Operand {
            src: o.src,
            layout: o.layout.clone(),
            access: AccessPlan::Unflatten(MultiFlattenMap { groups }),
        })
    })
}
