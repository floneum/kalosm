//! `FORM_GROUP_*`: the independent chains a step ends in — one optimizer
//! update per parameter — as one dispatch. A `Launch::Group` runs each
//! member on its own range of workgroups; nothing synchronizes, since no
//! member reads another. Minted on a root's launch spelling with every
//! earlier root's best spelling before it, trimmed to the tail that fits
//! the device's bindings.

use crate::egraph::{Builder, ClassId, Facts, Id, RuleTag};
use crate::ir::launch::{Launch, ScheduleDomain};
use crate::ir::{Level, Node, Op, OpTag};
use crate::rule;
use crate::rules::slab::{Deps, copy_operands, is_contraction, stage_rank};
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
    FORM_GROUP_SLAB,
    level = Level::Launch,
    head = OpTag::LaunchSlab,
    tag = RuleTag::Additive,
    apply = form_group
);

/// A member whose block the group lowering can set: a map, a fold, or a
/// slab of those. A tiled contraction fixes its own lane count.
fn groupable(b: &Builder<'_>, m: Id) -> bool {
    let own = b.class_of(m);
    match &b.node(m).op {
        // A copy of its own class is the spelling the extractor never
        // selects.
        Op::Launch(Launch::Map { .. } | Launch::Fold { .. }) => {
            !is_contraction(b, m) && !b.node(m).children.iter().any(|c| b.class_of(*c) == own)
        }
        Op::Launch(Launch::Slab { .. }) => true,
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
            _ => {
                if stage.is_none_or(|s| stage_rank(b, m) > stage_rank(b, s)) {
                    stage = Some(m);
                }
            }
        }
    }
    if std::env::var_os("FUSOR_GROUP_LOG").is_some()
        && let Some((_, s)) = best_slab
    {
        let all: Vec<String> = b
            .class_members(class.0)
            .into_iter()
            .filter_map(|m| match &b.node(m).op {
                Op::Launch(Launch::Slab { members, .. }) => Some(format!(
                    "{m}:c{}:n{}",
                    slab_copies(b, members),
                    members.len()
                )),
                _ => None,
            })
            .collect();
        eprintln!("PICK class {} -> {s} from {all:?}", class.0.index());
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

/// Whether a class binds its own buffer: an external leaf or a root the
/// caller reads back. Everything else shares the step arena's binding.
fn own_buffer(b: &Builder<'_>, class: ClassId, roots: &FxHashSet<ClassId>) -> bool {
    roots.contains(&class)
        || b.class_members(class.0).iter().any(|m| {
            matches!(
                b.node(*m).op,
                Op::Logical(crate::ir::logical::Logical::Leaf(_))
            )
        })
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
            let own: FxHashSet<ClassId> = members.iter().map(|s| b.class_of(*s)).collect();
            for s in members.iter() {
                for c in b.node(*s).children.iter() {
                    let class = b.class_of(*c);
                    if !own.contains(&class) {
                        inputs.insert(class);
                    }
                }
            }
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

/// Earlier stage launches of `id`'s kind over its index space, one per
/// class, in the order they were minted. A registry per graph, filled as
/// heads fire, so a firing scans only its own shape.
fn siblings(b: &Builder<'_>, id: Id) -> Vec<Id> {
    use std::cell::RefCell;
    type Siblings = rustc_hash::FxHashMap<(OpTag, Vec<crate::shape::Dim>), Vec<Id>>;
    thread_local! {
        static SEEN: RefCell<(u64, Siblings)> =
            RefCell::new((0, rustc_hash::FxHashMap::default()));
    }
    const MAX_SIBLINGS: usize = 24;
    let Op::Launch(op) = &b.node(id).op else {
        return Vec::new();
    };
    let (tag, space) = match op {
        Launch::Map { space, .. } | Launch::Fold { space, .. } => (op.tag(), space.dims.to_vec()),
        Launch::Slab { members, .. } => {
            let Some(last) = members.last() else {
                return Vec::new();
            };
            let Op::Launch(lop) = &b.node(*last).op else {
                return Vec::new();
            };
            (op.tag(), lop.iter_space().dims.to_vec())
        }
        _ => return Vec::new(),
    };
    let arena = b.arena_id();
    SEEN.with(|s| {
        let mut s = s.borrow_mut();
        if s.0 != arena {
            *s = (arena, rustc_hash::FxHashMap::default());
        }
        let list = s.1.entry((tag, space)).or_default();
        let class = b.class_of(id);
        let mut out: Vec<Id> = Vec::new();
        let mut seen_classes: FxHashSet<ClassId> = FxHashSet::default();
        for x in list.iter().rev() {
            if x.index() >= b.len() || b.class_of(*x) == class {
                continue;
            }
            let xc = b.class_of(*x);
            if !seen_classes.insert(xc) {
                continue;
            }
            // The class's current best spelling, not the one that fired.
            if let Some(best) = best_spelling(b, xc) {
                out.push(best);
            }
            if out.len() >= MAX_SIBLINGS {
                break;
            }
        }
        if !list.contains(&id) {
            list.push(id);
        }
        out.reverse();
        out
    })
}

pub fn form_group(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
    if b.caps().kind != crate::device::DeviceKind::Gpu
        || std::env::var_os("FUSOR_NO_GROUP").is_some()
    {
        return None;
    }
    let Op::Launch(_) = &node.op else { return None };
    if !groupable(b, id) {
        return None;
    }
    let class = b.class_of(id);
    let roots: Vec<Id> = b.roots().to_vec();
    let root_classes: FxHashSet<ClassId> = roots.iter().map(|r| b.class_of(*r)).collect();
    let head_root = roots
        .iter()
        .copied()
        .filter(|r| b.class_of(*r) == class)
        .min();

    // Candidates: for a root, every earlier root's best spelling; for
    // anything else, the earlier launches of the same kind over the same
    // index space — the sums of split-K partials, the bias gradients —
    // which run side by side whenever neither depends on the other.
    let candidates: Vec<Id> = match head_root {
        Some(head_root) => roots
            .iter()
            .copied()
            .filter(|r| *r < head_root)
            .filter_map(|r| {
                let rc = b.class_of(r);
                (!covered_class(b, id, rc))
                    .then(|| best_spelling(b, rc))
                    .flatten()
            })
            .collect(),
        // Same-shaped independent siblings (`siblings`) measured a net
        // loss: groups of plain spellings tie the fused ones on the bound
        // and the plan comes apart. Roots only until the bound prices that.
        None if std::env::var_os("FUSOR_GROUP_SIBLINGS").is_some() => siblings(b, id),
        None => Vec::new(),
    };

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
        if std::env::var_os("FUSOR_GROUP_LOG").is_some() {
            eprintln!(
                "GROUP {id}: cut {cut} tail {} outs {outs} inputs {inputs} budget {budget}",
                tail.len()
            );
        }
        if 2 + outs + inputs <= budget {
            chosen = Some(tail.to_vec());
            break;
        }
    }
    let members = chosen?;
    if std::env::var_os("FUSOR_GROUP_LOG").is_some() {
        eprintln!("GROUP {id}: {} members {members:?}", members.len());
    }
    let group = b
        .add_launch(Launch::Group {
            members: SmallVec::from_vec(members),
            sched: ScheduleDomain::Point,
        })
        .ok()?;
    b.union(id, group).ok()
}
