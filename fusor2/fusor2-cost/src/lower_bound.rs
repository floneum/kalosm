//! The admissible lower bound.
//!
//! `lb(c) = min over n in c of ( math_ps(n) + sum over *distinct* child chains
//! lb(child) )` — zero traffic, free sharing, min over the schedule domain.
//! Admissible in both regimes, so it works as a seed and as a
//! branch-and-bound prune.

use fusor2_ir::cost::{CostModel, Picoseconds};
use fusor2_ir::device::Caps;
use fusor2_ir::egraph::{ClassId, EGraph, Id};
use fusor2_ir::facts::ValueFacts;
use fusor2_ir::ir::Op;
use fusor2_ir::ir::logical::Logical;
use fusor2_ir::ir::launch::ScheduleDomain;
use rustc_hash::{FxHashMap, FxHasher};
use smallvec::SmallVec;
use std::hash::{Hash, Hasher};

/// Kleene iteration from `0`: the operator is monotone, so passes converge
/// upward onto the least fixpoint — exact where the class graph is acyclic,
/// a safe underestimate where a class cycle exists.
const MAX_PASSES: u32 = 8;

/// Ceiling on `node_math` evaluations spent scanning schedule domains. Past
/// it a node's math term degrades to zero, which is still a *lower* bound and
/// therefore still admissible.
const MATH_CALL_BUDGET: usize = 200_000;

/// The budget for one graph. Scales with the node count so a 100k+ node model
/// graph keeps a meaningful bound; an exhausted budget zeroes every later
/// node's math term and selection then falls to the tie-break alone.
fn math_call_budget(nodes: usize) -> usize {
    MATH_CALL_BUDGET.max(nodes.saturating_mul(16))
}

/// Domain size past which a node earns a memo entry.
const MEMO_THRESHOLD: usize = 8;

fn domain_len(graph: &EGraph, id: Id) -> usize {
    match &graph.node(id).op {
        Op::Launch(l1) => l1.schedule().map_or(1, |d| d.len()),
        _ => 1,
    }
}

/// Indexed by node id. One bottom-up sweep per pass; passes stop as soon as
/// nothing changes, so an acyclic graph costs two sweeps.
pub fn lower_bound(graph: &EGraph, cost: &dyn CostModel) -> Vec<Picoseconds> {
    let ids: Vec<Id> = (0..graph.len()).map(|i| Id(i as u32)).collect();
    lower_bound_over(graph, cost, &ids)
}

/// [`lower_bound`] over the masked slots only. The vector is still indexed by
/// node id — unmasked slots stay `0`. The mask must come from
/// [`crate::realize::reachable`], which is closed under both class membership
/// and children, so every id the extractor can index is masked.
pub fn lower_bound_scoped(
    graph: &EGraph,
    cost: &dyn CostModel,
    mask: &fixedbitset::FixedBitSet,
) -> Vec<Picoseconds> {
    let ids: Vec<Id> = mask.ones().map(|i| Id(i as u32)).collect();
    lower_bound_over(graph, cost, &ids)
}

/// The fixpoint over `ids`, in dependency postorder. `ids` must be closed:
/// every child class root and every union operand of a listed node is itself
/// listed.
///
/// Postorder puts every child value before its consumers, so one pass
/// converges the acyclic graph exactly; the remaining passes only chase class
/// cycles, where the capped iteration keeps a safe underestimate.
fn lower_bound_over(graph: &EGraph, cost: &dyn CostModel, ids: &[Id]) -> Vec<Picoseconds> {
    let n = graph.len();
    let mut lb = vec![Picoseconds(0); n];
    if n == 0 || ids.is_empty() {
        return lb;
    }
    let math = node_math_table(graph, cost, ids);
    let order = postorder(graph, ids);

    for _ in 0..MAX_PASSES {
        let mut changed = false;
        for id in &order {
            let next = combine(graph, *id, &math, &lb);
            if next != lb[id.index()] {
                lb[id.index()] = next;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    lb
}

/// Dependency postorder over the masked ids: every edge a [`combine`] reads —
/// a union operand, or a child's class root — is visited before its reader
/// wherever the class graph is acyclic. Iterative, deterministic (roots
/// ascending, edges in operand order), and restricted to `ids`.
fn postorder(graph: &EGraph, ids: &[Id]) -> Vec<Id> {
    let n = graph.len();
    let mut masked = fixedbitset::FixedBitSet::with_capacity(n);
    for id in ids {
        masked.insert(id.index());
    }
    // 0 = unseen, 1 = open, 2 = done.
    let mut state = vec![0u8; n];
    let mut out: Vec<Id> = Vec::with_capacity(ids.len());
    let mut stack: Vec<(Id, bool)> = Vec::new();
    for root in ids {
        if state[root.index()] != 0 {
            continue;
        }
        stack.push((*root, false));
        while let Some((id, expanded)) = stack.pop() {
            if expanded {
                if state[id.index()] != 2 {
                    state[id.index()] = 2;
                    out.push(id);
                }
                continue;
            }
            if state[id.index()] != 0 {
                continue;
            }
            state[id.index()] = 1;
            stack.push((id, true));
            let node = graph.node(id);
            let push = |next: Id, stack: &mut Vec<(Id, bool)>| {
                if state[next.index()] == 0 && masked.contains(next.index()) {
                    stack.push((next, false));
                }
            };
            match &node.op {
                Op::Union(a, b) => {
                    push(*a, &mut stack);
                    push(*b, &mut stack);
                }
                _ => {
                    for child in node.children.iter() {
                        push(graph.class_of(*child).0, &mut stack);
                    }
                }
            }
        }
    }
    out
}

/// The cheapest **selectable** member of `class`, picosecond ties broken by
/// the launch bound, then by smaller [`Id`]. The seed selection is exactly
/// this, per class.
///
/// Selectable, not just cheapest: the floor lowerings tie with the `Logical` node
/// they replace on math, so an unrestricted `min_by_key` would return the
/// un-lowered original every time. See [`crate::realize::selectable`].
///
/// The launch bound is the tie-break because the relaxation erases exactly
/// the launch and traffic a fusion deletes, so fused and unfused spellings
/// tie on picoseconds; ranking ties by fewest launches adopts the merged
/// spelling where doing so is free.
pub fn argmin_member(
    graph: &EGraph,
    lb: &[Picoseconds],
    launches: &[u32],
    class: ClassId,
    caps: &Caps,
) -> Id {
    if crate::realize::is_singleton(graph, class) {
        return class.0;
    }
    // `FUSOR2_SEED_DEBUG=<id>` prints every selectable member's seed key for
    // that class.
    if let Ok(want) = std::env::var("FUSOR2_SEED_DEBUG")
        && want == class.0.index().to_string()
    {
        for m in crate::realize::selectable(graph, class, caps) {
            eprintln!(
                "[seed] class {} member {m:?} lb={} launches={} op={:?}",
                class.0.index(),
                lb[m.index()].0,
                launches[m.index()],
                std::mem::discriminant(&graph.node(m).op),
            );
        }
    }
    argmin_member_excluding(graph, lb, launches, class, caps, &Default::default())
        .unwrap_or(class.0)
}

/// [`argmin_member`] over the members `banned` does not name. Returns `None`
/// when every candidate is banned, which is what makes the seed's cycle-repair
/// loop terminate.
pub fn argmin_member_excluding(
    graph: &EGraph,
    lb: &[Picoseconds],
    launches: &[u32],
    class: ClassId,
    caps: &Caps,
    banned: &rustc_hash::FxHashSet<Id>,
) -> Option<Id> {
    crate::realize::selectable(graph, class, caps)
        .into_iter()
        .filter(|m| !banned.contains(m))
        .min_by_key(|m| (lb[m.index()], launches[m.index()], *m))
}

/// The launch-count analogue of [`lower_bound_scoped`]: per node, the fewest
/// launches any realization of that node's chain can dispatch — every
/// non-leaf node is one launch plus its distinct child chains, sharing free,
/// `min` over members. Same Kleene iteration, same closure requirement on the
/// mask. Consumed by [`argmin_member`] as the tie-break only.
pub fn launch_bound_scoped(graph: &EGraph, mask: &fixedbitset::FixedBitSet) -> Vec<u32> {
    let ids: Vec<Id> = mask.ones().map(|i| Id(i as u32)).collect();
    launch_bound_over(graph, &ids)
}

/// [`launch_bound_scoped`] over the whole graph.
pub fn launch_bound(graph: &EGraph) -> Vec<u32> {
    let ids: Vec<Id> = (0..graph.len()).map(|i| Id(i as u32)).collect();
    launch_bound_over(graph, &ids)
}

fn launch_bound_over(graph: &EGraph, ids: &[Id]) -> Vec<u32> {
    let mut l = vec![0u32; graph.len()];
    if ids.is_empty() {
        return l;
    }
    let order = postorder(graph, ids);
    for _ in 0..MAX_PASSES {
        let mut changed = false;
        for id in &order {
            let next = launch_combine(graph, *id, &l);
            if next != l[id.index()] {
                l[id.index()] = next;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    l
}

fn launch_combine(graph: &EGraph, id: Id, l: &[u32]) -> u32 {
    let node = graph.node(id);
    match &node.op {
        Op::Union(a, b) => l[a.index()].min(l[b.index()]),
        Op::Logical(Logical::Leaf(_)) => 0,
        _ => {
            let mut seen: SmallVec<[ClassId; 4]> = SmallVec::new();
            let mut total = 1u32;
            for child in node.children.iter() {
                let class = graph.class_of(*child);
                if seen.contains(&class) {
                    continue;
                }
                seen.push(class);
                total = total.saturating_add(l[class.0.index()]);
            }
            total
        }
    }
}

fn combine(graph: &EGraph, id: Id, math: &[Picoseconds], lb: &[Picoseconds]) -> Picoseconds {
    let node = graph.node(id);
    match &node.op {
        Op::Union(a, b) => lb[a.index()].min(lb[b.index()]),
        Op::Logical(Logical::Leaf(_)) => Picoseconds(0),
        _ => {
            // Deduplicate children by class: a node reading the same class
            // twice contributes once, which is what makes sharing free.
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

fn node_math_table(graph: &EGraph, cost: &dyn CostModel, ids: &[Id]) -> Vec<Picoseconds> {
    let n = graph.len();
    let mut out = vec![Picoseconds(0); n];
    // Identical nodes at identical operand facts share a scan.
    let mut memo: FxHashMap<u64, Picoseconds> = FxHashMap::default();
    let mut budget = math_call_budget(ids.len());
    for id in ids {
        let id = *id;
        let node = graph.node(id);
        if matches!(node.op, Op::Union(..) | Op::Logical(Logical::Leaf(_))) {
            continue;
        }
        let slot = &mut out[id.index()];
        // Hashing operand facts costs about two `node_math` calls, so only a
        // domain wide enough to pay for it gets a memo entry.
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
    let node = graph.node(id);
    let ins: SmallVec<[ValueFacts; 4]> = node
        .children
        .iter()
        .map(|c| graph.facts(*c).clone())
        .collect();
    let out = graph.facts(id);

    let domain = match &node.op {
        Op::Launch(l1) => l1.schedule(),
        _ => None,
    };
    match domain {
        None | Some(ScheduleDomain::Point) => {
            if *budget == 0 {
                return Picoseconds(0);
            }
            *budget -= 1;
            cost.node_math(node, &ins, out, None)
        }
        Some(domain) => {
            // `node_math` depends on the point only through the MAC unit and
            // the padded tile, so a domain is scanned once per *math-distinct*
            // point, not once per point.
            let mut seen: SmallVec<[(u8, u32, u32); 12]> = SmallVec::new();
            let mut best: Option<Picoseconds> = None;
            for theta in domain.iter() {
                let key = match theta {
                    fusor2_ir::ir::launch::SchedPoint::Coop { geom, .. } => {
                        (1u8, geom.bm, geom.bn)
                    }
                    fusor2_ir::ir::launch::SchedPoint::Sgemm(p) => (2u8, p.bm, p.bn),
                    _ => (0u8, 0, 0),
                };
                if seen.contains(&key) {
                    continue;
                }
                seen.push(key);
                if *budget == 0 {
                    // An unfinished scan is still a lower bound only if we
                    // drop the term entirely; a partial min could exceed the
                    // true minimum and break admissibility.
                    return Picoseconds(0);
                }
                *budget -= 1;
                let v = cost.node_math(node, &ins, out, Some(theta));
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
        assert_eq!(
            argmin_member(&graph, &lb, &launch_bound(&graph), class, &test_caps()),
            cheap
        );
    }

    /// An `Logical` node never wins selection over its own lowering, however the
    /// bound compares.
    ///
    /// A floor lowering is worse in *schedule*, not in arithmetic, so the two
    /// members' bounds are at best a tie and can favour the `Logical` node
    /// outright. The `Logical` node is not runnable, so only
    /// [`crate::realize::selectable`] members are candidates.
    #[test]
    fn an_l0_node_never_wins_selection_over_its_own_lowering() {
        use crate::realize::testkit::{N, buffer, kmap, new_graph};
        use fusor2_ir::ir::Level;

        let mut graph = new_graph();
        let shape = [N];
        let leaf = buffer(&mut graph, 0, &shape);
        // An `Logical::Map` and an `Launch::Map` with the identical body: same work,
        // same bound, and the Logical node has the smaller id.
        let l0 = graph
            .add(Op::Logical(Logical::Map {
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
        // The premise: on cost alone the Logical node wins — it is no dearer, and
        // it holds the smaller id, so it also wins every tie-break.
        assert!(lb[l0.index()] <= lb[l1.index()]);
        assert!(l0 < l1);

        let picked = argmin_member(&graph, &lb, &launch_bound(&graph), class, &test_caps());
        assert_eq!(picked, l1, "selection must skip the un-lowered Logical node");
        assert_eq!(graph.level(picked), Level::Launch);
        assert!(crate::realize::is_runnable(&graph, picked));
        assert!(!crate::realize::is_runnable(&graph, l0));
    }

    /// Equal picosecond bounds break toward the member whose chain realizes
    /// fewer launches — the fused spelling a rule minted — never toward the
    /// smaller id.
    #[test]
    fn equal_bounds_break_toward_fewer_launches() {
        use crate::realize::testkit::{N, buffer, kmap};

        let mut g = new_graph();
        let shape = [N];
        let leaf = buffer(&mut g, 0, &shape);
        let copy = kmap(&mut g, leaf, &shape, 1);
        // The unfused spelling reads the copy; the fused one — the composed
        // body, minted later, so the larger id — reads the leaf directly.
        let unfused = kmap(&mut g, copy, &shape, 1);
        let fused = kmap(&mut g, leaf, &shape, 2);
        let union = g.union(unfused, fused).unwrap();
        let class = g.class_of(union);
        assert!(unfused < fused);

        let launches = launch_bound(&g);
        assert!(
            launches[fused.index()] < launches[unfused.index()],
            "reading the leaf directly is one launch; reading the copy is two"
        );
        // Bounds pinned equal, so only the tie-break decides.
        let tied = vec![Picoseconds(0); g.len()];
        assert_eq!(
            argmin_member(&g, &tied, &launches, class, &test_caps()),
            fused
        );
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
        assert_eq!(
            argmin_member(&graph, &lb, &launch_bound(&graph), class, &test_caps()),
            leaf
        );
    }
}
