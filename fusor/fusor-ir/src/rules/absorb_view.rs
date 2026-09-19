//! `ABSORB_VIEW_INTO_CONTRACT`: a contraction operand that is a copy of a
//! view — a head split, a transpose, a flatten spelled as an identity map —
//! reads the view's source through the view's own strides instead. The
//! tiled loaders address a non-affine row group per element, so the copy
//! buys nothing but a dispatch and a round trip through memory.

use crate::egraph::{Builder, Facts, Id, RuleTag};
use crate::ir::launch::{AccessPlan, ContractSide, Launch, Operand};
use crate::ir::{Level, Node, Op, OpTag};
use crate::rule;
use crate::rules::map_view;
use crate::scalar::ScalarKind;
use crate::shape::{Dim, Layout};

rule!(
    ABSORB_VIEW_INTO_CONTRACT,
    level = Level::Launch,
    head = OpTag::LaunchContract,
    tag = RuleTag::Additive,
    apply = absorb_view,
);

rule!(
    ABSORB_BROADCAST_INTO_MAP,
    level = Level::Launch,
    head = OpTag::LaunchMap,
    tag = RuleTag::Additive,
    apply = absorb_view_stage,
);

rule!(
    ABSORB_BROADCAST_INTO_FOLD,
    level = Level::Launch,
    head = OpTag::LaunchFold,
    tag = RuleTag::Additive,
    apply = absorb_view_stage,
);

/// A map or fold reading a *broadcast* copy — one whose own read has no
/// varying axis, a scalar spread over a weight's shape — reads the source
/// with the broadcast strides instead. Restricted to broadcasts: absorbing
/// every view here minted a head per reader and saturation took minutes,
/// and a broadcast copy is what every optimizer chain shares, so it made
/// every chain's slab overlap every other's.
pub fn absorb_view_stage(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
    if std::env::var_os("FUSOR_NO_ABSORB").is_some() {
        return None;
    }
    let Op::Launch(op) = &node.op else {
        return None;
    };
    let ops = match op {
        Launch::Map { ops, .. } | Launch::Fold { ops, .. } => ops,
        _ => return None,
    };
    let mut changed = false;
    let new_ops: Vec<Operand> = ops
        .iter()
        .map(|o| match through_copy(b, o, true) {
            Some(seen) => {
                changed = true;
                seen
            }
            None => o.clone(),
        })
        .collect();
    if !changed {
        return None;
    }
    let mut absorbed = op.clone();
    match &mut absorbed {
        Launch::Map { ops, .. } | Launch::Fold { ops, .. } => *ops = new_ops,
        _ => return None,
    }
    let minted = b.add_launch(absorbed).ok()?;
    // Reading a copy of one's own class through it changes nothing, and a
    // rule that keeps answering is a pass that never ends.
    if b.class_of(minted) == b.class_of(id) {
        return None;
    }
    b.union(id, minted).ok()
}

pub fn absorb_view(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
    if std::env::var_os("FUSOR_NO_ABSORB").is_some() {
        return None;
    }
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
    let mut changed = false;
    let mut side = |side: &ContractSide| -> ContractSide {
        let ops = side
            .ops
            .iter()
            .map(|o| match through_copy(b, o, false) {
                Some(seen) => {
                    changed = true;
                    seen
                }
                None => o.clone(),
            })
            .collect();
        ContractSide {
            pre: side.pre.clone(),
            ops,
        }
    };
    let a2 = side(a);
    let b2 = side(rhs);
    if !changed {
        return None;
    }
    let absorbed = b
        .add_launch(Launch::Contract {
            m: *m,
            n: *n,
            k: *k,
            batch: *batch,
            family: *family,
            post: post.clone(),
            acc: *acc,
            a: a2,
            b: b2,
            sched: sched.clone(),
        })
        .ok()?;
    if b.class_of(absorbed) == b.class_of(id) {
        return None;
    }
    b.union(id, absorbed).ok()
}

/// `o` read through the copy that produces it. A copy whose own read is
/// dense is a reshape: its output's flat index is its source's, so `o`
/// keeps its layout over the source. Otherwise, when `o` reads the copy
/// densely in the copy's shape, `o` takes the copy's read layout.
fn through_copy(b: &Builder<'_>, o: &Operand, broadcast_only: bool) -> Option<Operand> {
    let log = std::env::var_os("FUSOR_ABSORB_LOG").is_some();
    let own = b.class_of(o.src);
    if !matches!(o.access, AccessPlan::Alias | AccessPlan::Pack { .. }) {
        if log {
            eprintln!("ABSORB {}: access {:?}", o.src, o.access);
        }
        return None;
    }
    if log {
        let kinds: Vec<String> = b
            .class_members(o.src)
            .iter()
            .map(|m| match map_view(b, *m) {
                Some(v) if v.ops.len() == 1 && matches!(v.body.kind(), ScalarKind::Arg(0)) => {
                    format!("{m}:copy:{}", access_name(&v.ops[0].access))
                }
                Some(_) => format!("{m}:map"),
                None => format!("{m}:{:?}", b.node(*m).op.tag()),
            })
            .collect();
        eprintln!(
            "ABSORB {} (class {}) read as {}: {kinds:?}",
            o.src,
            b.class_of(o.src).0.index(),
            access_name(&o.access)
        );
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
            if log {
                eprintln!(
                    "ABSORB {}: copy {m} space {:?} vs shape {:?}",
                    o.src, view.space.dims, out_shape
                );
            }
            continue;
        }
        if log {
            eprintln!(
                "ABSORB {}: copy {m} src layout {:?}/{:?} contiguous {} ; o layout {:?}/{:?} contiguous {}",
                o.src,
                src.layout.shape(),
                src.layout.strides(),
                src.layout.is_contiguous(),
                o.layout.shape(),
                o.layout.strides(),
                o.layout.is_contiguous()
            );
        }
        let out_elems: Option<u64> = out_shape
            .iter()
            .try_fold(1u64, |a, d| d.as_const().map(|d| a * d));
        let src_elems: Option<u64> = src
            .layout
            .shape()
            .iter()
            .try_fold(1u64, |a, d| d.as_const().map(|d| a * d));
        if src.layout.is_contiguous() && out_elems.is_some() && out_elems == src_elems {
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
        // `o` permutes the copy's axes (a transposed read of a head split):
        // each of its axes is one copy axis at that axis's row-major stride,
        // and takes that axis's stride in the copy's own read.
        if let Some(layout) = permute_through(&o.layout, &out_shape, &src.layout) {
            return Some(Operand {
                src: src.src,
                layout,
                access: o.access.clone(),
            });
        }
        if log {
            eprintln!(
                "ABSORB {}: no composition: o {:?}/{:?} over copy shape {:?}, copy reads {:?}/{:?}",
                o.src,
                o.layout.shape(),
                o.layout.strides(),
                out_shape,
                src.layout.shape(),
                src.layout.strides()
            );
        }
    }
    None
}

/// `outer` stated over the copy's row-major output `shape`, composed with
/// the copy's read `inner` over its source: defined when every `outer` axis
/// is a broadcast, one `shape` axis, or a run of adjacent `shape` axes
/// merged — which splits back into those axes at the copy's strides.
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
        // The innermost axis of the run has the outer stride; the run
        // extends outward while its extents multiply up to `d`.
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

fn access_name(a: &AccessPlan) -> &'static str {
    match a {
        AccessPlan::Alias => "Alias",
        AccessPlan::Gather => "Gather",
        AccessPlan::Pack { .. } => "Pack",
        AccessPlan::Unflatten(_) => "Unflatten",
    }
}
