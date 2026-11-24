//! Post-extraction wavefront grouping. Launches at the same dependency level
//! are independent, so each level (split by block, within the binding budget)
//! runs as one group dispatch: a step costs its critical path in dispatches,
//! not its launch count.
//!
//! Saturation cannot carry this: every subset of a level is a candidate, and
//! which launches share a level is only known once a plan is selected.

use crate::nodes::{composite_members, is_group, resolved_children};
use fusor_ir::device::{Caps, DeviceKind};
use fusor_ir::egraph::{ClassId, EGraph, Id};
use fusor_ir::extract::{Extraction, Plan};
use fusor_ir::rules::group::{group_fits, mint_group};
use rustc_hash::{FxHashMap, FxHashSet};

/// Group each dependency level of groupable launches into one dispatch, one
/// pack at a time, keeping a pack only when it realizes and prices cheaper.
pub fn group_wavefronts(
    graph: &mut EGraph,
    caps: &Caps,
    plan: &Plan,
    replan: &mut dyn FnMut(&EGraph, &mut Extraction) -> Option<Plan>,
) -> Plan {
    if caps.kind != DeviceKind::Gpu || plan.launches.len() < 2 {
        return plan.clone();
    }
    let ex = &plan.extraction;
    let g: &EGraph = graph;
    let producer: FxHashMap<ClassId, usize> = plan
        .launches
        .iter()
        .enumerate()
        .flat_map(|(i, d)| {
            std::iter::once(d.root)
                .chain(d.members.iter().copied())
                .map(move |m| (g.class_of(m), i))
        })
        .collect();
    // Launches run in dependency order, so a level is one past its inputs'.
    let mut level = vec![0usize; plan.launches.len()];
    for (i, d) in plan.launches.iter().enumerate() {
        let mut stack: Vec<Id> = std::iter::once(d.root)
            .chain(d.members.iter().copied())
            .collect();
        let mut seen: FxHashSet<Id> = FxHashSet::default();
        while let Some(x) = stack.pop() {
            for (c, resolved) in resolved_children(graph, ex, x) {
                match producer.get(&graph.class_of(c)) {
                    Some(&p) if p != i => level[i] = level[i].max(level[p] + 1),
                    Some(_) => {}
                    None => {
                        if let Some(n) = resolved.filter(|n| seen.insert(*n)) {
                            stack.push(n);
                        }
                    }
                }
            }
        }
    }
    // Units per (level, block), in plan order: a launch's own members if it
    // is already a group.
    let mut buckets: FxHashMap<(usize, u32), Vec<Vec<Id>>> = FxHashMap::default();
    for (i, d) in plan.launches.iter().enumerate() {
        let unit = match composite_members(graph, d.root) {
            Some(m) if is_group(graph, d.root) => m.to_vec(),
            _ => vec![d.root],
        };
        buckets.entry((level[i], d.block)).or_default().push(unit);
    }
    let mut keys: Vec<_> = buckets.keys().copied().collect();
    keys.sort_unstable();

    // Greedy packs within the binding budget, in plan order, each minted.
    let mut minted: Vec<(Vec<Id>, Id)> = Vec::new();
    for key in keys {
        let units = &buckets[&key];
        if units.len() < 2 {
            continue;
        }
        let mut packs: Vec<Vec<Id>> = Vec::new();
        let mut current: Vec<Id> = Vec::new();
        {
            let b = graph.builder(caps);
            for unit in units {
                let mut next = current.clone();
                next.extend(unit.iter().copied());
                if group_fits(&b, &next) {
                    current = next;
                } else {
                    packs.push(std::mem::take(&mut current));
                    current = unit.clone();
                }
            }
        }
        packs.push(current);
        for members in packs.into_iter().filter(|m| m.len() > 1) {
            if let Some((group, _)) = mint_group(&mut graph.builder(caps), &members) {
                minted.push((members, group));
            }
        }
    }
    if minted.is_empty() {
        return plan.clone();
    }
    let select = |ex: &Extraction, packs: &[(Vec<Id>, Id)]| {
        let mut trial = rekeyed(graph, ex);
        for (members, group) in packs {
            for m in members {
                trial.sigma.insert(graph.class_of(*m), *m);
            }
            let head = *members.last().expect("a pack has members");
            trial.sigma.insert(graph.class_of(head), *group);
            if trial.m.contains(head.index()) {
                trial.m.insert(group.index());
            }
        }
        trial
    };
    // Every pack at once is one replan; only a refusal walks them singly.
    let mut all = select(&plan.extraction, &minted);
    if let Some(p) = replan(graph, &mut all).filter(|p| p.cost < plan.cost) {
        return p;
    }
    let mut ex = rekeyed(graph, &plan.extraction);
    let mut best = replan(graph, &mut ex).expect("the original selection realizes");
    for pack in &minted {
        let mut trial = select(&ex, std::slice::from_ref(pack));
        if let Some(p) = replan(graph, &mut trial).filter(|p| p.cost < best.cost) {
            best = p;
            ex = trial;
        }
    }
    best
}

/// `ex` with every selection keyed by its node's current class.
fn rekeyed(graph: &EGraph, ex: &Extraction) -> Extraction {
    let mut out = ex.clone();
    out.sigma = ex
        .sigma
        .values()
        .map(|n| (graph.class_of(*n), *n))
        .collect();
    out.m.grow(graph.len());
    out
}
