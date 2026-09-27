//! Operand access alternatives, an attribute of the edge. Each rule mints one
//! alternative of the reading node with one operand's access changed.

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

/// Read this operand straight through its own strides. `Operand::respell`
/// declines when an independently stated `Unflatten` map would be lost.
pub fn operand_alias(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
    respell_first(b, id, node, |o| {
        if matches!(o.access, AccessPlan::Alias) {
            return None;
        }
        o.respell(AccessPlan::Alias)
    })
}

/// Read this operand through a per-element address computation; never over a
/// contiguous layout, where it would duplicate the alias.
pub fn operand_gather(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
    respell_first(b, id, node, |o| {
        if matches!(o.access, AccessPlan::Gather) || o.layout.is_contiguous() {
            return None;
        }
        o.respell(AccessPlan::Gather)
    })
}

/// Stage this operand into a dense tile first, when the packed layout holds
/// exactly as many elements as the operand.
pub fn operand_pack(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
    respell_first(b, id, node, |o| {
        if matches!(o.access, AccessPlan::Pack { .. }) || o.layout.is_contiguous() {
            return None;
        }
        let into = Layout::contiguous(o.layout.shape());
        if !into.is_contiguous()
            || const_elements(into.shape())? != const_elements(o.layout.shape())?
        {
            return None;
        }
        o.respell(AccessPlan::Pack { into })
    })
}

/// Read this operand through an explicit index map, when its non-contiguous
/// layout decomposes into decidable `AxisGroup`s.
pub fn operand_unflatten(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
    respell_first(b, id, node, |o| {
        if matches!(o.access, AccessPlan::Unflatten(_)) || o.layout.is_contiguous() {
            return None;
        }
        let groups = o.layout.affine_groups().filter(|g| !g.is_empty())?;
        Some(Operand {
            src: o.src,
            layout: o.layout.clone(),
            access: AccessPlan::Unflatten(MultiFlattenMap { groups }),
        })
    })
}
