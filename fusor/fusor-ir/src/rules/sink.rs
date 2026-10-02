//! R5 — reader-rooted sinking. A pattern may match a *spine*
//! ([`crate::egraph::Builder::trace_pure_views`]).

use crate::dtype::Dtype;
use crate::egraph::{Builder, Facts, Id, RuleTag};
use crate::ir::launch::{AccessPlan, Launch, Operand};
use crate::ir::logical::Logical;
use crate::ir::{Level, Node, Op, OpTag};
use crate::rule;
use crate::shape::{AxisGroup, Dim, Layout, MultiFlattenMap};
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
    heads = [OpTag::LaunchMap, OpTag::LaunchFold],
    tag = RuleTag::Additive,
    apply = fold_views_into_index,
);

/// Compose a coordinate-independent map into a contraction's epilogue,
/// preserving accumulator precision and reapplying any pure views.
pub fn sink_epilogue(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
    let Op::Launch(Launch::Map { body, ops, .. }) = &node.op else {
        return None;
    };
    if ops.len() != 1 || !matches!(ops[0].access, AccessPlan::Alias) || body.reads_index_of() {
        return None;
    }
    let spine = b.trace_pure_views(ops[0].src);
    let Op::Launch(mut sunk) = b.node(spine.base).op.clone() else {
        return None;
    };
    let Launch::Contract { post, acc, .. } = &mut sunk else {
        return None;
    };
    if !epilogue_preserves_accum(body.dtype(), *acc) {
        return None;
    }
    *post = body.compose(std::slice::from_ref(post));
    let sunk = b.add_launch(sunk).ok()?;
    let cursor = crate::rules::rebuild_spine(b, &spine, sunk)?;
    b.union(id, cursor).ok()
}

/// An epilogue must not round the accumulator: it keeps its dtype or widens
/// a half accumulator to F32.
fn epilogue_preserves_accum(epilogue: Dtype, acc: Dtype) -> bool {
    epilogue == acc
        || (acc == Dtype::F16 && epilogue == Dtype::F32)
        || (acc == Dtype::BF16 && epilogue == Dtype::F32)
}

/// Read a view through the operand's index map instead of a materialized
/// copy; the divmod chain is priced by the cost model, not gated here.
pub fn fold_views_into_index(
    b: &mut Builder<'_>,
    id: Id,
    node: &Node,
    _f: &Facts<'_>,
) -> Option<Id> {
    let Op::Launch(op @ (Launch::Map { space, ops, .. } | Launch::Fold { space, ops, .. })) =
        &node.op
    else {
        return None;
    };
    let new_ops = fold_operand_views(b, ops, space)?;
    let mut alt = op.clone();
    for (o, new) in alt.operands_mut().zip(new_ops) {
        *o = new;
    }
    let alt = b.add_launch(alt).ok()?;
    b.union(id, alt).ok()
}

/// Every operand whose source is a pure view, restated over the view's base;
/// `None` when no slot moved.
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
        // Replacing the layout is sound only over a dense read of the space.
        if !slot.layout.is_contiguous() || slot.layout.shape() != space.dims.as_slice() {
            continue;
        }
        let spine = b.trace_pure_views(slot.src);
        if spine.views.is_empty() {
            continue;
        }
        if spine.views.len() > 1 {
            // A multi-node spine composes to one stride vector when
            // `composed_spine_layout` can state it.
            let Some(layout) = crate::rules::composed_spine_layout(b, &spine) else {
                continue;
            };
            if layout.shape() != space.dims.as_slice() {
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
        // The view must span the consuming index space exactly.
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
        // `MultiFlattenMap` has no base offset; the layout carries it.
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

/// The index map a spec vector induces over a dense base, and its base
/// offset (for the layout, where [`Operand::address_map`] reads it).
/// Declines on anything undecidable.
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
        // The offset scales by the input axis's stride even when broadcast.
        offset = offset.checked_add(s.offset.as_const()?.checked_mul(base)?)?;
        let stride = if s.multiplier == 0 {
            0
        } else {
            u32::try_from(base.checked_mul(u64::from(s.multiplier))?).ok()?
        };
        groups.push(AxisGroup::affine(extent, stride));
    }
    Some((MultiFlattenMap { groups }, offset))
}
