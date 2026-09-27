//! `FORM_GROUP_*`: independent launches as one dispatch — the chains a step
//! ends in (one optimizer update per parameter), and siblings reading a
//! common input (the q/k/v projections, a gradient's two products). A
//! `Launch::Group` runs each member on its own range of workgroups; nothing
//! synchronizes, since no member reads another. Minted on a launch with every
//! earlier candidate's best spelling before it, trimmed to the tail that fits
//! the device's bindings.

use crate::egraph::{Builder, ClassId, Facts, Id, RuleTag};
use crate::ir::launch::{Family, Launch, ScheduleDomain};
use crate::ir::{Level, Node, Op, OpTag};
use crate::rule;
use crate::rules::slab::{
    Deps, copy_operands, has_contract_spelling, is_contraction, outside_inputs, own_buffer,
    stage_rank,
};
use rustc_hash::FxHashSet;
use smallvec::SmallVec;

rule!(
    FORM_GROUP_MAP,
    level = Level::Launch,
    head = OpTag::LaunchMap,
    tag = RuleTag::Additive,
    apply = form_group
);
rule!(
    FORM_GROUP_FOLD,
    level = Level::Launch,
    head = OpTag::LaunchFold,
    tag = RuleTag::Additive,
    apply = form_group
);
rule!(
    FORM_GROUP_CONTRACT,
    level = Level::Launch,
    head = OpTag::LaunchContract,
    tag = RuleTag::Additive,
    apply = form_group
);
rule!(
    FORM_GROUP_SLAB,
    level = Level::Launch,
    head = OpTag::LaunchSlab,
    tag = RuleTag::Additive,
    apply = form_group
);

/// A member whose block the group lowering can set: a map, a fold, a slab
/// of those, or a cooperative contraction, whose extra subgroups mirror.
fn groupable(b: &Builder<'_>, m: Id) -> bool {
    let own = b.class_of(m);
    match &b.node(m).op {
        // A copy of its own class is the spelling the extractor never
        // selects.
        Op::Launch(Launch::Map { .. } | Launch::Fold { .. }) => {
            !is_contraction(b, m) && !b.node(m).children.iter().any(|c| b.class_of(*c) == own)
        }
        Op::Launch(Launch::Slab { .. }) => true,
        Op::Launch(Launch::Contract {
            family: Family::Coop,
            ..
        }) => true,
        _ => false,
    }
}

/// Copy-class operands over a slab's stages: each is a copy dispatch the
/// group would carry.
fn slab_copies(b: &Builder<'_>, members: &[Id]) -> isize {
    members
        .iter()
        .map(|m| match &b.node(*m).op {
            Op::Launch(Launch::Map { ops, .. } | Launch::Fold { ops, .. }) => {
                copy_operands(b, ops) as isize
            }
            _ => 0,
        })
        .sum()
}

/// Copy-class operands a contraction spelling reads.
fn contract_copies(b: &Builder<'_>, m: Id) -> usize {
    match &b.node(m).op {
        Op::Launch(op @ Launch::Contract { .. }) => copy_operands(b, op.operands()),
        _ => usize::MAX,
    }
}

/// The best groupable spelling of `class`: the slab reading the fewest
/// copies, then the most members, then the latest; else the best-ranked
/// stage.
fn best_spelling(b: &Builder<'_>, class: ClassId) -> Option<Id> {
    let mut best_slab: Option<((isize, usize), Id)> = None;
    let mut stage: Option<Id> = None;
    for m in b.class_members(class.0) {
        if !groupable(b, m) {
            continue;
        }
        match &b.node(m).op {
            Op::Launch(Launch::Slab { members, .. }) => {
                let key = (-slab_copies(b, members), members.len());
                if best_slab.is_none_or(|(k, _)| key >= k) {
                    best_slab = Some((key, m));
                }
            }
            // The spelling reading the fewest copies: one that reads its
            // views in place is a dispatch cheaper than one reading a copy.
            Op::Launch(Launch::Contract { .. }) => {
                let key = (contract_copies(b, m), m);
                if stage.is_none_or(|s| key < (contract_copies(b, s), s)) {
                    stage = Some(m);
                }
            }
            _ => {
                if stage.is_none_or(|s| stage_rank(b, m) > stage_rank(b, s)) {
                    stage = Some(m);
                }
            }
        }
    }
    best_slab.map(|(_, s)| s).or(stage)
}

/// Classes a member computes: itself and, for a slab, every stage.
fn covers(b: &Builder<'_>, m: Id, out: &mut FxHashSet<ClassId>) {
    out.insert(b.class_of(m));
    if let Op::Launch(Launch::Slab { members, .. }) = &b.node(m).op {
        for s in members {
            out.insert(b.class_of(*s));
        }
    }
}

/// Bindings a member needs beyond the uniform block and the arena: its
/// output and every slab stage a root reads back, and its outside inputs
/// that own a buffer.
fn bindings(
    b: &Builder<'_>,
    m: Id,
    roots: &FxHashSet<ClassId>,
    inputs: &mut FxHashSet<ClassId>,
) -> usize {
    let mut out = usize::from(roots.contains(&b.class_of(m)));
    match &b.node(m).op {
        Op::Launch(Launch::Slab { members, .. }) => {
            outside_inputs(b, members, inputs);
            out += members[..members.len() - 1]
                .iter()
                .filter(|s| roots.contains(&b.class_of(**s)))
                .count();
            inputs.retain(|c| own_buffer(b, *c, roots));
        }
        _ => {
            for c in b.node(m).children.iter() {
                let class = b.class_of(*c);
                if own_buffer(b, class, roots) {
                    inputs.insert(class);
                }
            }
        }
    }
    out
}

/// Whether `class` is one `head` computes itself.
fn covered_class(b: &Builder<'_>, head: Id, class: ClassId) -> bool {
    let mut own = FxHashSet::default();
    covers(b, head, &mut own);
    own.contains(&class)
}

pub fn form_group(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
    if b.caps().kind != crate::device::DeviceKind::Gpu {
        return None;
    }
    let Op::Launch(_) = &node.op else { return None };
    if !groupable(b, id) {
        return None;
    }
    let class = b.class_of(id);
    // A contraction class has many spellings (split-K, absorbed views); one
    // group per class is minted, from the spelling a group would carry.
    if matches!(node.op, Op::Launch(Launch::Contract { .. })) && best_spelling(b, class) != Some(id)
    {
        return None;
    }
    let roots: Vec<Id> = b.roots().to_vec();
    let root_classes: FxHashSet<ClassId> = roots.iter().map(|r| b.class_of(*r)).collect();
    // Earlier roots for a root map, fold or slab; earlier sibling
    // contractions for a contraction. "Earlier" orders a set of members one
    // way, so each group is minted once, from its last member.
    let head_root = roots
        .iter()
        .copied()
        .filter(|r| b.class_of(*r) == class)
        .min();
    let mut classes: Vec<ClassId> = Vec::new();
    // Siblings are contractions reading one of this contraction's inputs:
    // the products one activation or one gradient feeds.
    let contraction = matches!(node.op, Op::Launch(Launch::Contract { .. }));
    if !contraction && let Some(head) = head_root {
        classes.extend(roots.iter().filter(|r| **r < head).map(|r| b.class_of(*r)));
    } else if contraction {
        for c in node.children.iter() {
            for reader in b.readers_of(*c) {
                let rc = b.class_of(reader);
                if rc < class && has_contract_spelling(b, rc.0) {
                    classes.push(rc);
                }
            }
        }
    }
    // A class one of whose spellings reads `id`'s value (a view of it), or
    // that a spelling of `id`'s class reads (a split-K partial of it), is
    // not independent of `id`: grouping it makes the group its own producer.
    let read_by_head: FxHashSet<ClassId> = b
        .class_members(class.0)
        .into_iter()
        .flat_map(|m| b.node(m).children.to_vec())
        .map(|ch| b.class_of(ch))
        .collect();
    let mut seen = FxHashSet::default();
    classes.retain(|c| {
        *c != class
            && seen.insert(*c)
            && !read_by_head.contains(c)
            && !b
                .class_members(c.0)
                .into_iter()
                .any(|m| b.node(m).children.iter().any(|ch| b.class_of(*ch) == class))
    });
    let candidates: Vec<Id> = classes
        .into_iter()
        .filter_map(|rc| {
            (!covered_class(b, id, rc))
                .then(|| best_spelling(b, rc))
                .flatten()
        })
        .collect();

    let mut covered: FxHashSet<ClassId> = FxHashSet::default();
    covers(b, id, &mut covered);
    let mut members: Vec<Id> = Vec::new();
    let mut deps = Deps::new();
    deps.reset_classes(b, &covered);
    let mut back = Deps::new();
    for pick in candidates {
        let rc = b.class_of(pick);
        if covered.contains(&rc) {
            continue;
        }
        // A spelling reaching into a class already computed here would run
        // that stage twice; one reading a member, or read by one, would
        // need an order the group does not have.
        let mut own = FxHashSet::default();
        covers(b, pick, &mut own);
        if own.iter().any(|c| covered.contains(c)) || deps.depends(b, pick, &covered) {
            continue;
        }
        back.reset_classes(b, &own);
        if members
            .iter()
            .chain(std::iter::once(&id))
            .any(|m| back.depends(b, *m, &own))
        {
            continue;
        }
        covered.extend(own);
        deps.reset_classes(b, &covered);
        members.push(pick);
    }
    if members.is_empty() {
        return None;
    }
    members.push(id);

    // The longest tail that fits the bindings: the uniform block, each
    // member's own buffers, and every distinct outside input.
    let budget = b.caps().limits.max_storage_buffers_per_shader_stage as usize;
    let mut chosen: Option<Vec<Id>> = None;
    for cut in 0..members.len() - 1 {
        let tail = &members[cut..];
        let mut inputs: FxHashSet<ClassId> = FxHashSet::default();
        let mut outs = 0usize;
        for m in tail {
            outs += bindings(b, *m, &root_classes, &mut inputs);
        }
        let own: FxHashSet<ClassId> = tail.iter().map(|m| b.class_of(*m)).collect();
        let inputs = inputs.iter().filter(|c| !own.contains(c)).count();
        if 2 + outs + inputs <= budget {
            chosen = Some(tail.to_vec());
            break;
        }
    }
    let members = chosen?;
    let group = b
        .add_launch(Launch::Group {
            members: SmallVec::from_vec(members),
            sched: ScheduleDomain::Point,
        })
        .ok()?;
    b.union(id, group).ok()
}
