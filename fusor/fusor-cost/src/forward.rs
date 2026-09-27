//! Post-extraction view forwarding. A selected identity copy of a strided
//! view is a dispatch and a round trip through memory; every selected reader
//! of it reads the view's source through the composed strides instead, and
//! the copy, read by nothing, drops out of the realized DAG.
//!
//! Saturation cannot carry this: minting a forwarded spelling for every
//! reader of every copy multiplies heads and extraction never settles. Here
//! one spelling is minted per selected reader of a selected copy.

use fusor_ir::device::Caps;
use fusor_ir::egraph::{ClassId, EGraph, Id};
use fusor_ir::extract::{Extraction, Plan};
use fusor_ir::ir::Op;
use fusor_ir::ir::launch::{AccessPlan, Launch};
use fusor_ir::rules::absorb_view::forward_views;
use fusor_ir::scalar::ScalarKind;
use rustc_hash::{FxHashMap, FxHashSet};

/// Rewrite `plan`'s selection into `ex` so no launch reads a view copy that
/// is not a root. Returns whether anything changed; the caller then replans.
pub fn forward_selected_views(
    graph: &mut EGraph,
    caps: &Caps,
    roots: &[Id],
    plan: &Plan,
    ex: &mut Extraction,
) -> bool {
    let launched: FxHashSet<Id> = plan
        .launches
        .iter()
        .flat_map(|l| std::iter::once(l.root).chain(l.members.iter().copied()))
        .collect();
    let root_classes: FxHashSet<ClassId> = roots.iter().map(|r| graph.class_of(*r)).collect();
    let copies: FxHashSet<ClassId> = ex
        .sigma
        .iter()
        .filter(|(c, n)| {
            launched.contains(n) && !root_classes.contains(c) && is_view_copy(graph, **n)
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

/// An identity map reading one strided view of another value.
fn is_view_copy(graph: &EGraph, id: Id) -> bool {
    matches!(
        &graph.node(id).op,
        Op::Launch(Launch::Map { body, ops, .. })
            if ops.len() == 1
                && matches!(body.kind(), ScalarKind::Arg(0))
                && matches!(ops[0].access, AccessPlan::Alias)
    )
}
