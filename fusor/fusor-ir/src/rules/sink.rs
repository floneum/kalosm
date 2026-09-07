//! R5 — reader-rooted sinking. A pattern may match a *spine*
//! ([`crate::egraph::Builder::trace_pure_views`]).

use crate::dtype::Dtype;
use crate::egraph::{Builder, Facts, Id, RuleTag};
use crate::ir::launch::{AccessPlan, Launch, Operand};
use crate::ir::logical::Logical;
use crate::ir::{Level, Node, Op, OpTag};
use crate::rule;
use crate::shape::{AxisGroup, Dim, Layout, MultiFlattenMap, SubAxis};
use smallvec::SmallVec;

rule!(
    SINK_EPILOGUE,
    level = Level::Launch,
    head = OpTag::LaunchMap,
    tag = RuleTag::Additive,
    apply = sink_epilogue,
);

rule!(
    FOLD_VIEWS_INTO_INDEX,
    level = Level::Launch,
    head = OpTag::LaunchMap,
    tag = RuleTag::Additive,
    apply = fold_views_into_index,
);

rule!(
    FOLD_VIEWS_INTO_FOLD_INDEX,
    level = Level::Launch,
    head = OpTag::LaunchFold,
    tag = RuleTag::Additive,
    apply = fold_views_into_fold_index,
);

/// `f(view(x)) == view(f(x))` when `view` is pure: a single-operand `Map`
/// reading a contraction through a chain of restrides also equals that
/// contraction with a longer `post`, re-viewed.
///
/// The only guard is numeric: the epilogue must not round the accumulator
/// ahead of the chain, so its element type must be the accumulator's, or the
/// F16-accumulator/F32-epilogue widening pair. That is legality — whether
/// sinking pays is priced elsewhere.
pub fn sink_epilogue(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
    let Op::Launch(Launch::Map { body, ops, .. }) = &node.op else {
        return None;
    };
    if ops.len() != 1 || !matches!(ops[0].access, AccessPlan::Alias) {
        return None;
    }
    let spine = b.trace_pure_views(ops[0].src);
    let base = b.node(spine.base).op.clone();

    let sunk = match base {
        Op::Launch(Launch::Contract {
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
        }) => {
            if !epilogue_preserves_accum(body.dtype(), acc) {
                return None;
            }
            b.add_launch(Launch::Contract {
                m,
                n,
                k,
                batch,
                family,
                post: body.compose(&[post]),
                acc,
                a,
                b: rhs,
                sched,
            })
            .ok()?
        }
        _ => return None,
    };

    // Re-apply the spine, innermost first. Each view keeps its own relative
    // spec vector, which is what makes a multi-node spine compose correctly.
    let mut cursor = sunk;
    for view in spine.views.iter() {
        let Op::Logical(Logical::Restride { specs, bounds, .. }) = b.node(*view).op.clone() else {
            return None;
        };
        cursor = b
            .add_logical(Logical::Restride {
                specs,
                bounds,
                x: cursor,
            })
            .ok()?;
    }
    b.union(id, cursor).ok()
}

/// The accumulator must not be rounded ahead of the chain: an epilogue is
/// admissible when it preserves the accumulator's element type, or when it
/// is the F16-store / F32-compute widening pair.
fn epilogue_preserves_accum(epilogue: Dtype, acc: Dtype) -> bool {
    epilogue == acc
        || (acc == Dtype::F16 && epilogue == Dtype::F32)
        || (acc == Dtype::BF16 && epilogue == Dtype::F32)
}

/// Read a view through the operand's index map instead of through a
/// materialized copy. `MultiFlattenMap::divmod_ops` is the term the pricing
/// crate charges for the divmod chain, so this stays ungated here.
pub fn fold_views_into_index(
    b: &mut Builder<'_>,
    id: Id,
    node: &Node,
    _f: &Facts<'_>,
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
    let new_ops = fold_operand_views(b, ops, space)?;
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

/// The same law with a `Fold` in the consumer position.
///
/// A `Fold`'s operands are indexed over `space` exactly as a `Map`'s are, so
/// the rewrite is the same rewrite. `vec_axes` needs no special case: it
/// renumbers nothing, and `check_vec_axes` on the minted node refuses an
/// illegal spelling at `add_launch`.
pub fn fold_views_into_fold_index(
    b: &mut Builder<'_>,
    id: Id,
    node: &Node,
    _f: &Facts<'_>,
) -> Option<Id> {
    let Op::Launch(Launch::Fold {
        space,
        axis,
        vec_axes,
        carrier,
        acc,
        post,
        ops,
        sched,
    }) = &node.op
    else {
        return None;
    };
    let new_ops = fold_operand_views(b, ops, space)?;
    let alt = b
        .add_launch(Launch::Fold {
            space: space.clone(),
            axis: *axis,
            vec_axes: vec_axes.clone(),
            carrier: carrier.clone(),
            acc: *acc,
            post: post.clone(),
            ops: new_ops,
            sched: sched.clone(),
        })
        .ok()?;
    b.union(id, alt).ok()
}

/// Every operand whose source is a single pure view, restated as an index map
/// over the view's base. `None` when no slot moved, so the caller mints
/// nothing.
fn fold_operand_views(
    b: &Builder<'_>,
    ops: &[Operand],
    space: &crate::ir::launch::IndexSpace,
) -> Option<Vec<Operand>> {
    let mut new_ops = ops.to_vec();
    let mut changed = false;
    for slot in new_ops.iter_mut() {
        if !matches!(slot.access, AccessPlan::Alias) {
            continue;
        }
        // The rewrite replaces the operand's layout outright, which is sound
        // only when that layout was the dense read of the consuming space. A
        // permuted, broadcast or offset alias says something else, and every
        // one of those spellings reaches here.
        if !reads_its_view_densely(slot, space) {
            continue;
        }
        let spine = b.trace_pure_views(slot.src);
        if spine.views.is_empty() {
            continue;
        }
        if spine.views.len() > 1 {
            // A multi-node spine composes to one stride vector when every
            // stage is const, statically bounded and affine over the stage
            // below — `composed_spine_layout` states the conditions. This is
            // the narrow → reshape → transpose chain every rope operand and
            // attention head split arrives as.
            let Some(layout) = crate::rules::composed_spine_layout(b, &spine) else {
                continue;
            };
            if layout.rank() != space.dims.len()
                || !layout
                    .shape()
                    .iter()
                    .zip(&space.dims)
                    .all(|(l, d)| l.known_eq(*d))
            {
                continue;
            }
            *slot = Operand {
                src: spine.base,
                layout,
                access: AccessPlan::Alias,
            };
            changed = true;
            continue;
        }
        let Op::Logical(Logical::Restride { specs, .. }) = b.node(spine.views[0]).op.clone() else {
            continue;
        };
        // The view must span the consuming index space: a `[rows, 1]` view
        // read over `[rows, cols]` has its layout doing work the map cannot
        // express, and adopting the map reads `flat % rows` where
        // `flat / cols` belongs.
        if specs.len() != space.dims.len()
            || !specs
                .iter()
                .zip(&space.dims)
                .all(|(s, d)| s.size.known_eq(*d))
        {
            continue;
        }
        let base_shape = b.facts_of(spine.base).shape.clone();
        let Some((map, offset)) = unflatten_of(&specs, &base_shape) else {
            continue;
        };
        // `MultiFlattenMap` has nowhere to put a base offset, so
        // `Operand::address_map` takes it from the layout. Offset 0 here
        // silently turns a narrowed view (`table[2..]`) back into the whole
        // table.
        let layout = Layout::from_parts(
            Dim::Const(offset),
            &base_shape,
            &Layout::row_major_strides(&base_shape),
        )
        .ok()?;
        *slot = Operand {
            src: spine.base,
            layout,
            access: AccessPlan::Unflatten(map),
        };
        changed = true;
    }
    changed.then_some(new_ops)
}

/// Whether `o`'s own layout is the dense row-major read of `space` at offset
/// zero — the one layout [`fold_operand_views`] may discard, because it is the
/// one the replacement map reproduces.
///
/// `verify_launch::check_operand_access` pins an `Alias`'s rank and extents
/// only, so transposed, broadcast and windowed operands all arrive here;
/// replacing their layout with `unflatten_of`'s map addresses a different
/// element at every coordinate but the first.
fn reads_its_view_densely(o: &Operand, space: &crate::ir::launch::IndexSpace) -> bool {
    if !o.layout.offset().known_eq(Dim::Const(0)) {
        return false;
    }
    if o.layout.rank() != space.dims.len() {
        return false;
    }
    if !o
        .layout
        .shape()
        .iter()
        .zip(&space.dims)
        .all(|(l, d)| l.known_eq(*d))
    {
        return false;
    }
    let want = Layout::row_major_strides(&space.dims);
    o.layout
        .strides()
        .iter()
        .zip(&want)
        .all(|(s, w)| s.known_eq(*w))
}

/// The index map a relative spec vector induces over a dense base, and the
/// base offset it starts from. Declines when an extent, stride or offset is
/// not decidable — there is no contiguous fallback here, only the alternative
/// not being minted.
///
/// The offset is returned separately because `MultiFlattenMap` is a sum of
/// stride terms with no constant slot; the caller must put it on the
/// operand's layout, which is where [`Operand::address_map`] reads it from.
fn unflatten_of(
    specs: &[crate::shape::StrideSpec],
    base_shape: &[Dim],
) -> Option<(MultiFlattenMap, u64)> {
    let base_strides = Layout::row_major_strides(base_shape);
    let mut groups: SmallVec<[AxisGroup; 4]> = SmallVec::new();
    let mut offset: u64 = 0;
    for s in specs {
        let extent = u32::try_from(s.size.as_const()?).ok()?;
        let base = base_strides.get(s.input_dim as usize)?.as_const()?;
        // A spec's offset is in units of its own input axis, so it scales by
        // that axis's stride whether or not the axis is broadcast.
        offset = offset.checked_add(s.offset.as_const()?.checked_mul(base)?)?;
        let stride = if s.multiplier == 0 {
            0
        } else {
            u32::try_from(base.checked_mul(u64::from(s.multiplier))?).ok()?
        };
        groups.push(AxisGroup {
            sub_axes: smallvec::smallvec![SubAxis { extent, stride }],
        });
    }
    Some((MultiFlattenMap { groups }, offset))
}
