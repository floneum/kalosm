//! Groups: independent launches as one dispatch. Each member runs on its own
//! range of workgroups; the post-extraction wavefront pass mints them.

use crate::egraph::{Builder, ClassId, Id};
use crate::ir::Op;
use crate::ir::launch::{Family, Launch, ScheduleDomain};
use crate::rules::slab::{is_contraction, outside_inputs, own_buffer};
use rustc_hash::FxHashSet;
use smallvec::SmallVec;

/// A member whose block the group lowering can set: a map, a fold, a slab
/// of those, or a cooperative contraction.
fn groupable(b: &Builder<'_>, m: Id) -> bool {
    let own = b.class_of(m);
    match &b.node(m).op {
        // A copy of its own class is never selected.
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

/// Bindings a member needs beyond the uniform block and the arena: root
/// outputs it computes, plus (into `inputs`) its buffer-owning inputs.
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

/// Whether `members` bind within the device's storage buffers as one group.
fn fits(b: &Builder<'_>, members: &[Id], root_classes: &FxHashSet<ClassId>) -> bool {
    let budget = b.caps().limits.max_storage_buffers_per_shader_stage as usize;
    let mut inputs: FxHashSet<ClassId> = FxHashSet::default();
    let mut outs = 0usize;
    for m in members {
        outs += bindings(b, *m, root_classes, &mut inputs);
    }
    let own: FxHashSet<ClassId> = members.iter().map(|m| b.class_of(*m)).collect();
    let inputs = inputs.iter().filter(|c| !own.contains(c)).count();
    2 + outs + inputs <= budget
}

/// Whether independent `members` can run as one group: each groupable, and
/// their bindings within the device's budget.
pub fn group_fits(b: &Builder<'_>, members: &[Id]) -> bool {
    let root_classes: FxHashSet<ClassId> = b.roots().iter().map(|r| b.class_of(*r)).collect();
    members.len() > 1 && members.iter().all(|m| groupable(b, *m)) && fits(b, members, &root_classes)
}

/// Mint independent `members` as one group in the last member's class:
/// the group node and the union's result.
pub fn mint_group(b: &mut Builder<'_>, members: &[Id]) -> Option<(Id, Id)> {
    let head = *members.last()?;
    let group = b
        .add_launch(Launch::Group {
            members: SmallVec::from_slice(members),
            sched: ScheduleDomain::Point,
        })
        .ok()?;
    Some((group, b.union(head, group).ok()?))
}
