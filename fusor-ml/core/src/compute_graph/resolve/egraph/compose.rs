//! One expression-composition engine.
//!
//! Every fusion rewrite walks an n-ary expression the same way: a load's
//! index expressions are rewritten first, then the load itself is replaced,
//! folded into a producer's expression or left in place, while dimension
//! references go through a coordinate map. [`rewrite`] owns that walk and
//! each rule below is only its per-load decision, so an inlinability gate is
//! stated once and every rewrite composes coordinates identically.

use rustc_hash::{FxHashMap, FxHashSet};

use super::super::ExecutionVariant;
use crate::Layout;
use crate::compute_graph::NodeIndex;
use crate::nary_wise::{ElementwiseOperation, ExtractedUnaryChain, NaryExpr, NaryFunction};
use crate::view::ViewOperation;

/// Rewrite `expr` bottom-up: `coords` maps every dimension reference, and
/// `load` replaces every input read, receiving the read's slot, its original
/// index expressions and those same expressions after rewriting. `None` from
/// either hook aborts the whole rewrite.
fn rewrite(
    expr: &NaryExpr,
    coords: &mut impl FnMut(usize) -> Option<NaryExpr>,
    load: &mut impl FnMut(usize, &[NaryExpr], Vec<NaryExpr>) -> Option<NaryExpr>,
) -> Option<NaryExpr> {
    Some(match expr {
        NaryExpr::Op { children, function } => NaryExpr::Op {
            children: children
                .iter()
                .map(|child| rewrite(child, &mut *coords, &mut *load))
                .collect::<Option<Vec<_>>>()?,
            function: function.clone(),
        },
        NaryExpr::IndexedInput { input_idx, indices } => {
            let mapped = indices
                .iter()
                .map(|index| rewrite(index, &mut *coords, &mut *load))
                .collect::<Option<Vec<_>>>()?;
            load(*input_idx, indices, mapped)?
        }
        NaryExpr::DimIndex(dim) => coords(*dim)?,
        NaryExpr::Scalar(value) => NaryExpr::Scalar(*value),
    })
}

/// [`rewrite`] over loads alone, leaving the index space untouched.
pub(super) fn rewrite_loads(
    expr: &NaryExpr,
    load: &mut impl FnMut(usize, &[NaryExpr], Vec<NaryExpr>) -> Option<NaryExpr>,
) -> Option<NaryExpr> {
    rewrite(expr, &mut |dim| Some(NaryExpr::DimIndex(dim)), load)
}

/// [`rewrite_loads`] for rules that cannot decline, so neither can the walk.
pub(in super::super) fn map_loads(
    expr: &NaryExpr,
    load: &mut impl FnMut(usize, &[NaryExpr], Vec<NaryExpr>) -> NaryExpr,
) -> NaryExpr {
    rewrite_loads(expr, &mut |input_idx, indices, mapped| {
        Some(load(input_idx, indices, mapped))
    })
    .expect("a total load rule never aborts the walk")
}

/// Evaluate `expr` (written in its own index space) at the coordinates
/// given by `indices`: every `DimIndex(d)` becomes `indices[d]`. `None`
/// when `expr` references a dimension `indices` does not provide.
pub(super) fn compose_expr_with_indices(expr: &NaryExpr, indices: &[NaryExpr]) -> Option<NaryExpr> {
    rewrite(
        expr,
        &mut |dim| indices.get(dim).cloned(),
        &mut |input_idx, _, mapped| {
            Some(NaryExpr::IndexedInput {
                input_idx,
                indices: mapped,
            })
        },
    )
}

/// Renumber every input slot through `mapping`.
pub(super) fn remap_inputs(expr: &NaryExpr, mapping: &[usize]) -> NaryExpr {
    map_loads(expr, &mut |input_idx, _, indices| NaryExpr::IndexedInput {
        input_idx: mapping[input_idx],
        indices,
    })
}

/// Add offset to all input indices in an expression.
pub(super) fn offset_input_indices(expr: &NaryExpr, offset: usize) -> NaryExpr {
    map_loads(expr, &mut |input_idx, _, indices| NaryExpr::IndexedInput {
        input_idx: input_idx + offset,
        indices,
    })
}

/// Substitute IndexedInput(target_idx) with element-wise access with the
/// replacement expression. Returns (new_expression, success) where success is
/// true if all references to target_idx were successfully substituted. If
/// false, the input should NOT be removed from the graph.
pub(super) fn substitute_input_in_expr(
    expr: &NaryExpr,
    target_idx: usize,
    replacement: &NaryExpr,
) -> (NaryExpr, bool) {
    let elementwise_replacement = match replacement {
        NaryExpr::IndexedInput { input_idx, indices }
            if NaryExpr::is_elementwise_indices(indices) =>
        {
            Some(*input_idx)
        }
        _ => None,
    };
    let mut success = true;
    let expr = map_loads(expr, &mut |input_idx, indices, mapped| {
        if input_idx != target_idx {
            return NaryExpr::IndexedInput {
                input_idx,
                indices: mapped,
            };
        }
        if NaryExpr::is_elementwise_indices(indices) {
            // Element-wise can be fully replaced with any expression
            return replacement.clone();
        }
        // Custom indexing can only substitute if replacement is also
        // element-wise; anything else cannot fuse into the indexed load.
        let Some(input_idx) = elementwise_replacement else {
            success = false;
            return NaryExpr::IndexedInput {
                input_idx,
                indices: mapped,
            };
        };
        NaryExpr::IndexedInput {
            input_idx,
            indices: mapped,
        }
    });
    (expr, success)
}

/// Substitute every read of input `target_idx` — elementwise *or*
/// custom-indexed — with `replacement` evaluated at the read's index
/// expressions. Where [`substitute_input_in_expr`] declines custom-indexed
/// reads unless the replacement is a bare input, this composes the
/// replacement expression with the index list instead: `input_t[i0, i1]`
/// becomes `replacement` with `DimIndex(d)` rewritten to `i_d`. Returns
/// `None` when the composition is impossible (an index list shorter than the
/// replacement's rank).
pub(super) fn substitute_input_composed(
    expr: &NaryExpr,
    target_idx: usize,
    replacement: &NaryExpr,
) -> Option<NaryExpr> {
    rewrite_loads(expr, &mut |input_idx, _, mapped| {
        if input_idx == target_idx {
            compose_expr_with_indices(replacement, &mapped)
        } else {
            Some(NaryExpr::IndexedInput {
                input_idx,
                indices: mapped,
            })
        }
    })
}

/// Replace every element-wise read of an input carrying a replacement with
/// that expression. `None` when such an input is read through custom
/// indexing, which the replacement's index space cannot serve.
pub(super) fn replace_inputs_in_expr(
    expr: &NaryExpr,
    replacements: &[Option<NaryExpr>],
) -> Option<NaryExpr> {
    rewrite_loads(expr, &mut |input_idx, indices, mapped| match replacements
        .get(input_idx)
        .and_then(|r| r.as_ref())
    {
        Some(replacement) => NaryExpr::is_elementwise_indices(indices).then(|| replacement.clone()),
        None => Some(NaryExpr::IndexedInput {
            input_idx,
            indices: mapped,
        }),
    })
}

/// Rewrite every access to input `target_idx` through `view`'s
/// coordinate map: the original index expressions (the view's output
/// coordinates) walk down the stage stack to base coordinates, with
/// fill selects and in-bounds clamps where stages are partially defined
/// (both select branches evaluate).
pub(super) fn rewrite_view_input(
    expr: &NaryExpr,
    target_idx: usize,
    view: &ViewOperation,
) -> Option<NaryExpr> {
    rewrite_loads(expr, &mut |input_idx, _, mapped| {
        if input_idx == target_idx {
            Some(view.value_expression(input_idx, &mapped)?.0)
        } else {
            Some(NaryExpr::IndexedInput {
                input_idx,
                indices: mapped,
            })
        }
    })
}

/// Remove unused inputs and deduplicate, returning new inputs and remapped expression.
pub(in super::super) fn deduplicate_inputs(
    inputs: Vec<NodeIndex>,
    expr: NaryExpr,
) -> (Vec<NodeIndex>, NaryExpr) {
    // Collect which input indices are actually used
    let mut seen_indices = FxHashSet::default();
    let mut used_indices = Vec::new();
    collect_used_inputs(&expr, &mut seen_indices, &mut used_indices);

    // Build the input-index remap, collecting only used inputs.
    let mut new_inputs = Vec::new();
    let mut old_to_new = FxHashMap::default();

    for old_idx in used_indices {
        let node = inputs[old_idx];
        // Check if this node already exists in new_inputs (deduplication)
        let new_idx = if let Some(existing) = new_inputs.iter().position(|&n| n == node) {
            existing
        } else {
            let idx = new_inputs.len();
            new_inputs.push(node);
            idx
        };
        old_to_new.insert(old_idx, new_idx);
    }

    let new_expr = map_loads(&expr, &mut |input_idx, _, indices| NaryExpr::IndexedInput {
        input_idx: old_to_new[&input_idx],
        indices,
    });
    (new_inputs, new_expr)
}

fn collect_used_inputs(expr: &NaryExpr, seen: &mut FxHashSet<usize>, used: &mut Vec<usize>) {
    match expr {
        NaryExpr::Op { children, .. } => {
            for child in children {
                collect_used_inputs(child, seen, used);
            }
        }
        NaryExpr::IndexedInput { input_idx, indices } => {
            if seen.insert(*input_idx) {
                used.push(*input_idx);
            }
            for c in indices {
                collect_used_inputs(c, seen, used);
            }
        }
        NaryExpr::DimIndex(_) => {}
        NaryExpr::Scalar(_) => {}
    }
}

/// The worst re-read factor across this slot's loads: the product of
/// index-space dims a load's coordinates never reference — each such dim
/// re-reads the same element once per step.
pub(super) fn input_reread_factor(expr: &NaryExpr, shape: &[usize], slot: usize) -> usize {
    fn collect_dims(expr: &NaryExpr, referenced: &mut [bool]) {
        match expr {
            NaryExpr::Op { children, .. } => {
                for child in children {
                    collect_dims(child, referenced);
                }
            }
            NaryExpr::IndexedInput { indices, .. } => {
                for index in indices {
                    collect_dims(index, referenced);
                }
            }
            NaryExpr::DimIndex(dim) => referenced[*dim] = true,
            NaryExpr::Scalar(_) => {}
        }
    }
    fn visit_loads(expr: &NaryExpr, shape: &[usize], slot: usize, worst: &mut usize) {
        match expr {
            NaryExpr::Op { children, .. } => {
                for child in children {
                    visit_loads(child, shape, slot, worst);
                }
            }
            NaryExpr::IndexedInput { input_idx, indices } => {
                for index in indices {
                    visit_loads(index, shape, slot, worst);
                }
                if *input_idx == slot {
                    let mut referenced = vec![false; shape.len()];
                    for index in indices {
                        collect_dims(index, &mut referenced);
                    }
                    let factor: usize = shape
                        .iter()
                        .zip(&referenced)
                        .filter(|(_, referenced)| !**referenced)
                        .map(|(size, _)| *size)
                        .product();
                    *worst = (*worst).max(factor);
                }
            }
            NaryExpr::DimIndex(_) | NaryExpr::Scalar(_) => {}
        }
    }
    let mut worst = 1;
    visit_loads(expr, shape, slot, &mut worst);
    worst
}

/// Try to extract a unary function chain from a node variant.
/// Only Nary ops with a single input and element-wise access can be converted.
pub(super) fn try_get_unary_chain(variant: &ExecutionVariant) -> Option<ExtractedUnaryChain> {
    match variant {
        ExecutionVariant::Elementwise(nary) => nary.try_extract_unary_chain(),
        _ => None,
    }
}

/// Extract a (possibly empty) unary function chain over exactly one
/// read of input 0 with arbitrary index expressions, innermost function
/// first. The index expressions must not read any input themselves.
pub(super) fn extract_unary_chain_indexed(
    nary: &ElementwiseOperation,
) -> Option<(Vec<NaryFunction>, Vec<NaryExpr>)> {
    fn contains_input(expr: &NaryExpr) -> bool {
        match expr {
            NaryExpr::Op { children, .. } => children.iter().any(contains_input),
            NaryExpr::IndexedInput { .. } => true,
            NaryExpr::DimIndex(_) | NaryExpr::Scalar(_) => false,
        }
    }
    let mut functions = Vec::new();
    let mut expr = &nary.expression;
    loop {
        match expr {
            NaryExpr::Op { children, function }
                if children.len() == 1 && function.input_types.len() == 1 =>
            {
                functions.push(function.clone());
                expr = &children[0];
            }
            NaryExpr::IndexedInput {
                input_idx: 0,
                indices,
            } if !indices.iter().any(contains_input) => {
                functions.reverse();
                return Some((functions, indices.clone()));
            }
            _ => return None,
        }
    }
}

/// Walk through view nodes from `inner` down to the first non-view
/// node, composing each view's collapsed stage stack. Public tensor ops
/// collapse into single view nodes at construction, but composed
/// clusters (attention's attached GQA/transpose views) still layer view
/// nodes deliberately. `view_of` selects the chain — the collapsed layout and
/// input of the node's current view form, `None` where the walk must stop.
/// Returns the base node and the composed layout over the base's logical
/// value space; the layout is `None` when `inner` is not a view (identity).
/// Views that don't collapse or compose (or carry a fill region) act as chain
/// breaks: the walk stops without seeing through them.
pub(in super::super) fn walk_view_chain(
    mut inner: NodeIndex,
    mut view_of: impl FnMut(NodeIndex) -> Option<(Layout, NodeIndex)>,
) -> (NodeIndex, Option<Layout>) {
    let mut composed: Option<Layout> = None;
    loop {
        let Some((collapsed, input)) = view_of(inner) else {
            return (inner, composed);
        };
        let next = match &composed {
            None => collapsed,
            Some(outer) => match crate::view::compose_layouts(outer, &collapsed) {
                Some(layout) => layout,
                None => return (inner, composed),
            },
        };
        composed = Some(next);
        inner = input;
    }
}
