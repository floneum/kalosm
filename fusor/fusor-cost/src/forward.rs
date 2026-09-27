//! Post-extraction view forwarding: a selected reader of a selected identity
//! copy reads the copy's source through composed strides instead. Done here,
//! not in saturation, which would mint a spelling per reader of every copy.

use crate::nodes::is_view_copy;
use fusor_ir::device::Caps;
use fusor_ir::egraph::{ClassId, EGraph, Id};
use fusor_ir::extract::{Extraction, Plan};
use fusor_ir::rules::absorb_view::forward_views;
use rustc_hash::{FxHashMap, FxHashSet};

/// Rewrite `plan`'s selection into `ex` so no launch reads a non-root view
/// copy. Returns whether anything changed.
pub fn forward_selected_views(
    graph: &mut EGraph,
    caps: &Caps,
    roots: &[Id],
    plan: &Plan,
    ex: &mut Extraction,
) -> bool {
    let launched: FxHashSet<Id> = crate::nodes::plan_values(plan).collect();
    let root_classes: FxHashSet<ClassId> = roots.iter().map(|r| graph.class_of(*r)).collect();
    let copies: FxHashSet<ClassId> = ex
        .sigma
        .iter()
        .filter(|(c, n)| {
            launched.contains(n) && !root_classes.contains(c) && is_view_copy(&graph.node(**n).op)
        })
        .map(|(c, _)| *c)
        .collect();
    if copies.is_empty() {
        return false;
    }
    let mut selected: Vec<(ClassId, Id)> = ex
        .sigma
        .iter()
        .filter(|(c, n)| launched.contains(n) && !copies.contains(c))
        .map(|(c, n)| (*c, *n))
        .collect();
    selected.sort_unstable_by_key(|(c, _)| *c);

    let copy = |b: &fusor_ir::egraph::Builder<'_>, src: Id| copies.contains(&b.class_of(src));
    let mut minted: Vec<(Id, Id)> = Vec::new();
    let mut replaced: FxHashMap<Id, Id> = FxHashMap::default();
    {
        let mut b = graph.builder(caps);
        for (_, node) in &selected {
            let start = minted.len();
            if let Some(new) = forward_views(&mut b, *node, &copy, &mut minted) {
                replaced.insert(*node, new);
                for (old, new) in &minted[start..] {
                    if let Some(theta) = ex.theta.get(old).copied() {
                        ex.theta.insert(*new, theta);
                    }
                }
            }
        }
    }
    if replaced.is_empty() {
        return false;
    }
    ex.m.grow(graph.len());
    for (old, new) in &minted {
        if ex.m.contains(old.index()) {
            ex.m.insert(new.index());
        }
    }
    // A union may move a class's representative: rekey every selection.
    ex.sigma = ex
        .sigma
        .values()
        .map(|n| {
            let n = replaced.get(n).copied().unwrap_or(*n);
            (graph.class_of(n), n)
        })
        .collect();
    true
}
