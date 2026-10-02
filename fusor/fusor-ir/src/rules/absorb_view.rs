//! `ABSORB_VIEW_INTO_CONTRACT`: a contraction operand that is a copy of a
//! view (head split, transpose, flatten) reads the view's source through the
//! view's strides instead, saving the copy's dispatch and round trip.

use crate::egraph::{Builder, Facts, Id, RuleTag};
use crate::ir::launch::{AccessPlan, Launch, Operand};
use crate::ir::{Level, Node, Op, OpTag};
use crate::rule;
use crate::rules::map_view;
use crate::scalar::ScalarKind;
use crate::shape::{Dim, Layout, const_elements};

rule!(
    ABSORB_VIEW_INTO_CONTRACT,
    level = Level::Launch,
    head = OpTag::LaunchContract,
    tag = RuleTag::Additive,
    apply = absorb_view,
);

rule!(
    ABSORB_BROADCAST,
    level = Level::Launch,
    heads = [OpTag::LaunchMap, OpTag::LaunchFold],
    tag = RuleTag::Additive,
    apply = absorb_broadcast,
);

/// A map or fold reading a broadcast copy reads the source with the
/// broadcast strides instead. Broadcasts only: absorbing every view mints a
/// head per reader and blows up saturation.
pub fn absorb_broadcast(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
    absorb(b, id, node, true)
}

pub fn absorb_view(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
    absorb(b, id, node, false)
}

/// Mint the reader with every operand [`through_copy`] accepts read through
/// its copy instead.
fn absorb(b: &mut Builder<'_>, id: Id, node: &Node, broadcast_only: bool) -> Option<Id> {
    let Op::Launch(op) = &node.op else {
        return None;
    };
    let seen: Vec<Option<Operand>> = op
        .operands()
        .map(|o| through_copy(b, o, broadcast_only))
        .collect();
    if seen.iter().all(Option::is_none) {
        return None;
    }
    let mut absorbed = op.clone();
    for (o, s) in absorbed.operands_mut().zip(seen) {
        if let Some(s) = s {
            *o = s;
        }
    }
    let minted = b.add_launch(absorbed).ok()?;
    // Reading a copy of one's own class through it changes nothing.
    if b.class_of(minted) == b.class_of(id) {
        return None;
    }
    b.union(id, minted).ok()
}

/// `o` read through the copy that produces it: over a dense (reshape) copy
/// `o` keeps its layout; a dense or permuted read takes the copy's strides.
fn through_copy(b: &Builder<'_>, o: &Operand, broadcast_only: bool) -> Option<Operand> {
    let own = b.class_of(o.src);
    if !matches!(o.access, AccessPlan::Alias | AccessPlan::Pack { .. }) {
        return None;
    }
    for m in b.class_members(o.src) {
        let Some(view) = map_view(b, m) else { continue };
        if view.ops.len() != 1
            || !matches!(view.body.kind(), ScalarKind::Arg(0))
            || !matches!(view.ops[0].access, AccessPlan::Alias)
        {
            continue;
        }
        let src = &view.ops[0];
        if b.facts_of(src.src).dtype != b.facts_of(o.src).dtype || b.class_of(src.src) == own {
            continue;
        }
        if broadcast_only && src.layout.strides().iter().any(|s| s.as_const() != Some(0)) {
            continue;
        }
        let out_shape = b.facts_of(o.src).shape.clone();
        if view.space.dims.as_slice() != out_shape.as_slice() {
            continue;
        }
        let out_elems = const_elements(&out_shape);
        if src.layout.is_contiguous()
            && out_elems.is_some()
            && out_elems == const_elements(src.layout.shape())
        {
            return Some(Operand {
                src: src.src,
                layout: o.layout.clone(),
                access: o.access.clone(),
            });
        }
        if o.layout.is_contiguous() && src.layout.shape() == o.layout.shape() {
            return Some(Operand {
                src: src.src,
                layout: src.layout.clone(),
                access: o.access.clone(),
            });
        }
        // `o` permutes the copy's axes (a transposed read of a head split).
        if let Some(layout) = permute_through(&o.layout, &out_shape, &src.layout) {
            return Some(Operand {
                src: src.src,
                layout,
                access: o.access.clone(),
            });
        }
    }
    None
}

/// `outer` over the copy's row-major output `shape`, composed with the
/// copy's read `inner`: each `outer` axis must be a broadcast or a run of
/// adjacent `shape` axes.
fn permute_through(outer: &Layout, shape: &[Dim], inner: &Layout) -> Option<Layout> {
    if inner.shape().len() != shape.len() || !matches!(outer.offset(), Dim::Const(0)) {
        return None;
    }
    let rs = Layout::row_major_strides(shape);
    let mut dims: smallvec::SmallVec<[Dim; 8]> = smallvec::SmallVec::new();
    let mut strides: smallvec::SmallVec<[Dim; 8]> = smallvec::SmallVec::new();
    for (d, st) in outer.shape().iter().zip(outer.strides()) {
        if matches!(st, Dim::Const(0)) {
            dims.push(*d);
            strides.push(Dim::Const(0));
            continue;
        }
        let j = (0..shape.len()).find(|k| rs[*k] == *st)?;
        let want = d.as_const()?;
        let mut k = j;
        let mut have = shape[j].as_const()?;
        while have < want {
            k = k.checked_sub(1)?;
            have = have.checked_mul(shape[k].as_const()?)?;
        }
        if have != want {
            return None;
        }
        for (axis, dim) in shape.iter().enumerate().take(j + 1).skip(k) {
            dims.push(*dim);
            strides.push(inner.strides()[axis]);
        }
    }
    Layout::from_parts(inner.offset(), &dims, &strides).ok()
}

/// Post-extraction forwarding: `id` with every operand whose class `copy`
/// accepts read through that copy's view, composite members rewritten in
/// place. Each `(old, new)` pair lands in `minted`, `id`'s own last.
pub fn forward_views(
    b: &mut Builder<'_>,
    id: Id,
    copy: &dyn Fn(&Builder<'_>, Id) -> bool,
    minted: &mut Vec<(Id, Id)>,
) -> Option<Id> {
    let Op::Launch(op) = b.node(id).op.clone() else {
        return None;
    };
    let mut changed = false;
    let mut rewritten = op.clone();
    match &mut rewritten {
        Launch::Map { .. } | Launch::Fold { .. } | Launch::Contract { .. } => {
            for o in rewritten.operands_mut() {
                if copy(b, o.src)
                    && let Some(seen) = through_copy(b, o, false)
                {
                    *o = seen;
                    changed = true;
                }
            }
        }
        Launch::Slab { members, .. } | Launch::Group { members, .. } => {
            let old = members.clone();
            for (slot, m) in old.iter().enumerate() {
                if let Some(new) = forward_views(b, *m, copy, minted) {
                    members[slot] = new;
                    changed = true;
                }
            }
        }
        _ => return None,
    }
    if !changed {
        return None;
    }
    let before = b.len();
    let new = b.add_launch(rewritten).ok()?;
    // Reject merging two selected classes, or a self-reading node (composite
    // members were checked when minted).
    let own = b.class_of(id);
    let composite = matches!(
        b.node(new).op,
        Op::Launch(Launch::Slab { .. } | Launch::Group { .. })
    );
    if new == id
        || (new.index() < before && b.class_of(new) != own)
        || (!composite && b.node(new).children.iter().any(|c| b.class_of(*c) == own))
    {
        return None;
    }
    b.union(id, new).ok()?;
    minted.push((id, new));
    Some(new)
}
