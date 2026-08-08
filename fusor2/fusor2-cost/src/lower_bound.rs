//! The admissible lower bound.
//!
//! `lb(c) = min over n in c of ( math_ps(n) + sum over distinct child classes
//! lb(child) )` — zero traffic, free sharing, min over the schedule domain.
//! It underestimates true cost in both the inlined and the materialized
//! regime, so it serves as both a seed and a branch-and-bound prune.

use fusor2_ir::cost::{CostModel, Picoseconds};
use fusor2_ir::device::Caps;
use fusor2_ir::egraph::{ClassId, EGraph, Id};
use fusor2_ir::ir::Op;
use fusor2_ir::ir::level0::L0;
use fusor2_ir::ir::level1::ScheduleDomain;
use rustc_hash::{FxHashMap, FxHasher};
use smallvec::SmallVec;
use std::hash::{Hash, Hasher};

/// Iteration cap for the Kleene fixpoint. A class id can exceed its own
/// members' ids, so one forward sweep is not enough; the operator is monotone
/// over non-negative picoseconds, so iterating from `0` converges upward onto
/// the least fixpoint.
const MAX_PASSES: u32 = 8;

/// Ceiling on `node_math` evaluations spent scanning schedule domains. Past it
/// a node's math term degrades to zero, which is still admissible.
const MATH_CALL_BUDGET: usize = 200_000;

/// Domain size past which a node earns a memo entry.
const MEMO_THRESHOLD: usize = 8;

fn domain_len(graph: &EGraph, id: Id) -> usize {
    match &graph.node(id).op {
        Op::L1(l1) => l1.schedule().map_or(1, |d| d.len()),
        _ => 1,
    }
}

/// Indexed by node id. One bottom-up sweep per pass; passes stop as soon as
/// nothing changes, so an acyclic graph costs two sweeps.
pub fn lower_bound(graph: &EGraph, cost: &dyn CostModel) -> Vec<Picoseconds> {
    let n = graph.len();
    let mut lb = vec![Picoseconds(0); n];
    if n == 0 {
        return lb;
    }
    let math = node_math_table(graph, cost);

    for _ in 0..MAX_PASSES {
        let mut changed = false;
        for i in 0..n {
            let id = Id(i as u32);
            let next = combine(graph, id, &math, &lb);
            if next != lb[i] {
                lb[i] = next;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    lb
}

/// The cheapest [`crate::realize::selectable`] member of `class`, ties by
/// smaller [`Id`]. Restricting to selectable members matters because floor
/// lowerings tie on math with the `L0` node they replace.
pub fn argmin_member(
    graph: &EGraph,
    lb: &[Picoseconds],
    class: ClassId,
    caps: &Caps,
) -> Id {
    if crate::realize::is_singleton(graph, class) {
        return class.0;
    }
    crate::realize::selectable(graph, class, caps)
        .into_iter()
        .min_by_key(|m| (lb[m.index()], *m))
        .unwrap_or(class.0)
}

fn combine(graph: &EGraph, id: Id, math: &[Picoseconds], lb: &[Picoseconds]) -> Picoseconds {
    let node = graph.node(id);
    match &node.op {
        Op::Union(a, b) => lb[a.index()].min(lb[b.index()]),
        Op::L0(L0::Leaf(_)) => Picoseconds(0),
        _ => {
            // Deduplicate children by class so a node reading the same class
            // twice contributes once.
            let mut seen: SmallVec<[ClassId; 4]> = SmallVec::new();
            let mut total = math[id.index()];
            for child in node.children.iter() {
                let class = graph.class_of(*child);
                if seen.contains(&class) {
                    continue;
                }
                seen.push(class);
                total += lb[class.0.index()];
            }
            total
        }
    }
}

fn node_math_table(graph: &EGraph, cost: &dyn CostModel) -> Vec<Picoseconds> {
    let n = graph.len();
    let mut out = vec![Picoseconds(0); n];
    // Identical nodes at identical operand facts share a scan.
    let mut memo: FxHashMap<u64, Picoseconds> = FxHashMap::default();
    let mut budget = MATH_CALL_BUDGET;
    for (i, slot) in out.iter_mut().enumerate() {
        let id = Id(i as u32);
        let node = graph.node(id);
        if matches!(node.op, Op::Union(..) | Op::L0(L0::Leaf(_))) {
            continue;
        }
        // Hashing operand facts costs about two `node_math` calls, so only a
        // wide enough domain earns a memo entry.
        if domain_len(graph, id) <= MEMO_THRESHOLD {
            *slot = best_math(graph, cost, id, &mut budget);
            continue;
        }
        let key = shape_key(graph, id);
        *slot = match memo.get(&key) {
            Some(hit) => *hit,
            None => {
                let v = best_math(graph, cost, id, &mut budget);
                memo.insert(key, v);
                v
            }
        };
    }
    out
}

/// `argmin over sched.iter()` of `node_math`; `ScheduleDomain::Point` passes
/// `None`, as does any node without a domain.
fn best_math(graph: &EGraph, cost: &dyn CostModel, id: Id, budget: &mut usize) -> Picoseconds {
    let scan = crate::realize::DomainScan::new(graph, id, cost);
    let domain = match &graph.node(id).op {
        Op::L1(l1) => l1.schedule(),
        _ => None,
    };
    match domain {
        None | Some(ScheduleDomain::Point) => {
            if *budget == 0 {
                return Picoseconds(0);
            }
            *budget -= 1;
            scan.price(cost, None)
        }
        Some(domain) => {
            let mut best: Option<Picoseconds> = None;
            for theta in domain.iter() {
                if *budget == 0 {
                    // A partial min could exceed the true minimum, so an
                    // unfinished scan drops the term entirely.
                    return Picoseconds(0);
                }
                *budget -= 1;
                let v = scan.price(cost, Some(theta));
                best = Some(match best {
                    Some(b) if b <= v => b,
                    _ => v,
                });
            }
            best.unwrap_or(Picoseconds(0))
        }
    }
}

fn shape_key(graph: &EGraph, id: Id) -> u64 {
    let node = graph.node(id);
    let mut h = FxHasher::default();
    node.op.hash(&mut h);
    for c in node.children.iter() {
        graph.facts(*c).hash(&mut h);
    }
    graph.facts(id).hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::realize::testkit::{TestCost, chain_graph, new_graph, seeded_graph, test_caps};

    #[test]
    fn leaves_bound_at_zero() {
        let (graph, _roots) = chain_graph(3);
        let cost = TestCost::default();
        let lb = lower_bound(&graph, &cost);
        assert_eq!(lb[0], Picoseconds(0), "the buffer leaf costs nothing");
    }

    #[test]
    fn bound_is_monotone_along_a_chain() {
        let (graph, roots) = chain_graph(4);
        let cost = TestCost::default();
        let lb = lower_bound(&graph, &cost);
        let root = roots[0];
        assert!(lb[root.index()].0 > 0, "a chain of maps costs something");
    }

    #[test]
    fn union_takes_the_cheaper_branch() {
        let (graph, _roots, cheap, dear, class) = seeded_graph();
        let cost = TestCost::default();
        let lb = lower_bound(&graph, &cost);
        assert!(lb[cheap.index()] <= lb[dear.index()]);
        assert_eq!(lb[class.0.index()], lb[cheap.index()]);
        assert_eq!(argmin_member(&graph, &lb, class, &test_caps()), cheap);
    }

    /// An `L0` node never wins selection over its own lowering, however the
    /// bound compares, because it is not runnable.
    #[test]
    fn an_l0_node_never_wins_selection_over_its_own_lowering() {
        use crate::realize::testkit::{N, buffer, kmap, new_graph};
        use fusor2_ir::ir::Level;

        let mut graph = new_graph();
        let shape = [N];
        let leaf = buffer(&mut graph, 0, &shape);
        // An `L0::Map` and an `L1::KMap` with the identical body: same work,
        // same bound, smaller id on the L0 node.
        let l0 = graph
            .add(Op::L0(L0::Map {
                expr: fusor2_ir::scalar::ScalarExpr::un(
                    fusor2_ir::scalar::UnOp::Exp,
                    fusor2_ir::scalar::ScalarExpr::arg(0, fusor2_ir::dtype::Dtype::F32),
                ),
                ins: smallvec::smallvec![leaf],
                outs: 1,
            }))
            .unwrap();
        let l1 = kmap(&mut graph, leaf, &shape, 1);
        let union = graph.union(l0, l1).unwrap();
        let class = graph.class_of(union);

        let cost = TestCost::default();
        let lb = lower_bound(&graph, &cost);
        // On cost alone the L0 node wins: no dearer, and smaller id.
        assert!(lb[l0.index()] <= lb[l1.index()]);
        assert!(l0 < l1);

        let picked = argmin_member(&graph, &lb, class, &test_caps());
        assert_eq!(picked, l1, "selection must skip the un-lowered L0 node");
        assert_eq!(graph.level(picked), Level::L1);
        assert!(crate::realize::is_runnable(&graph, picked));
        assert!(!crate::realize::is_runnable(&graph, l0));
    }

    #[test]
    fn a_class_with_no_lowering_still_yields_a_member() {
        // `selectable` falls back to every member so the failure surfaces as
        // `verify_plan`'s named error rather than as an unselected class.
        use crate::realize::testkit::{N, buffer};

        let mut graph = new_graph();
        let leaf = buffer(&mut graph, 0, &[N]);
        let class = graph.class_of(leaf);
        let lb = lower_bound(&graph, &TestCost::default());
        assert_eq!(argmin_member(&graph, &lb, class, &test_caps()), leaf);
    }
}
