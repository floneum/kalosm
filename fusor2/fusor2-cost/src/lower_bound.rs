//! The admissible lower bound.
//!
//! `lb(c) = min over n in c of ( math_ps(n) + sum over *distinct* child chains
//! lb(child) )` — zero traffic, free sharing, min over the schedule domain.
//!
//! It is a genuine relaxation in **both** regimes: an inlined node's true cost
//! pays math `k` times where `lb` pays once; a materialized node's true cost
//! pays math plus traffic where `lb` pays math. So it is a valid seed *and* a
//! valid branch-and-bound prune.
//!
//! The alternative seed — assume everything shared is materialized — is not a
//! lower bound at all: it maximizes launch count and pays a write plus a read
//! for every edge the optimal fused cut deletes, which is precisely the
//! conv-epilogue shape the trainer is made of.
//!
//! Owned by W7.

use fusor2_ir::cost::{CostModel, Picoseconds};
use fusor2_ir::device::Caps;
use fusor2_ir::egraph::{ClassId, EGraph, Id};
use fusor2_ir::facts::ValueFacts;
use fusor2_ir::ir::Op;
use fusor2_ir::ir::level0::L0;
use fusor2_ir::ir::level1::ScheduleDomain;
use rustc_hash::{FxHashMap, FxHasher};
use smallvec::SmallVec;
use std::hash::{Hash, Hasher};

/// `union(a, b)` allocates an id *above* both operands, so a consumer built
/// before its producer gained an alternative sees `root_of(child) > own id`.
/// A single forward sweep would then read an unwritten slot. Kleene iteration
/// from `0` fixes that without giving up admissibility: the operator is
/// monotone (`min` and `+` over non-negative picoseconds), so iterating from
/// the bottom converges upward onto the least fixpoint — the exact class
/// bound wherever the class graph is acyclic, and a safe underestimate where
/// a class cycle exists.
const MAX_PASSES: u32 = 8;

/// Ceiling on `node_math` evaluations spent scanning schedule domains. Past
/// it a node's math term degrades to zero, which is still a *lower* bound and
/// therefore still admissible. Without a ceiling one `CoopDomain` of ~8,300
/// points per contraction would put the bound alone over the 2 ms budget.
const MATH_CALL_BUDGET: usize = 200_000;

/// The budget for one graph. The flat 200k ceiling was sized against suite
/// graphs (hundreds to a few thousand nodes); a transformer decode graph is
/// 100k+ nodes and every non-leaf node consumes at least one call, so a flat
/// ceiling zeroes the math term for *everything past the exhaustion point*.
/// Zeroed terms tie every class's lower bound at 0 and `argmin_member`'s
/// smallest-id tie-break then structurally prefers the construction-time defn
/// expansion — which is how an 8B model's quantized matmuls all seeded as
/// fold-over-materialized-dequant (~27 GB of f32 launch roots) while the same
/// op in a 300-node bench seeded as the in-place `KContract`. Scaling with
/// the node count keeps every suite graph's behavior bit-identical (they are
/// far below 100k nodes) and keeps the bound meaningful on model graphs; the
/// memo already collapses the repeated-layer domain scans that made a flat
/// ceiling necessary.
fn math_call_budget(nodes: usize) -> usize {
    // x16, not the x2 that first replaced the flat ceiling: an 8B decode
    // graph is ~46k nodes whose contraction classes hold ~34 members with
    // coop domains ~5,700 points wide, so a couple of classes exhausted
    // 2 x nodes before the memo could amortize anything and every bound past
    // them was zero. Zero-tied classes then select by tie-break alone, which
    // is how all 33 m = 1 projection gemvs seeded as padded Coop tiles (16x
    // the useful MACs) in one measured round. Suite graphs sit far below the
    // 200k floor either way, so their plans are bit-identical; only model
    // graphs buy the extra scan, ~1.5M cheap table calls once per resolve.
    MATH_CALL_BUDGET.max(nodes.saturating_mul(16))
}

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
    let ids: Vec<Id> = (0..graph.len()).map(|i| Id(i as u32)).collect();
    lower_bound_over(graph, cost, &ids)
}

/// [`lower_bound`] over the masked slots only. The vector is still indexed by
/// node id — unmasked slots stay `0`, and the extractor never reads them: the
/// mask comes from [`crate::realize::reachable`], which is closed under both
/// class membership and children, so every id selection or move generation
/// can index is masked. On a long-lived session graph this is the difference
/// between a resolve pricing what it asked for and pricing every node the
/// graph has ever held.
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
/// # Postorder, not ascending id order
///
/// The sweep used to walk `ids` ascending. On suite graphs that converges in
/// two passes, but `union(a, b)` allocates an id *above* both operands, so on
/// a deep model graph a consumer routinely reads a child class whose union
/// root has the **larger** id — one pass of staleness per such inversion,
/// and a 32-layer decode chain crosses thousands of them against
/// [`MAX_PASSES`]` = 8`. The bound never converged there, and the error was
/// not uniform: two members of one class read their children at different
/// sweep positions, so the earlier-created member summed values one pass
/// staler — systematically *smaller* — than its later-minted sibling. That
/// bias is larger than the real difference between a fused and an unfused
/// spelling (measured 913 ps on an equal-math pair whose child chains tie),
/// so seeding compared noise, not cost. A DFS postorder over the class DAG
/// puts every child value before its consumers; one pass then converges the
/// whole acyclic graph exactly, members of one class read identical child
/// values, and the remaining passes only chase class cycles, where the
/// capped iteration keeps today's safe underestimate.
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
/// wherever the class graph is acyclic. Iterative (a model chain is tens of
/// thousands of nodes deep), deterministic (roots ascending, edges in operand
/// order), and restricted to `ids` so a scoped bound stays scoped.
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
/// Selectable, not just cheapest: the floor lowerings tie with the `L0` node
/// they replace on math, so an unrestricted `min_by_key` would return the
/// un-lowered original every time. See [`crate::realize::selectable`].
///
/// # Why the launch bound is the tie-break
///
/// For a single-reader elementwise producer, the fused spelling a rule minted
/// (`MAP_INTO_MAP`, `ABSORB`, the sink rules' folded-view operands) and the
/// unfused chain have **exactly equal** picosecond bounds: the relaxation is
/// zero traffic, free sharing and math-once, which erases precisely the
/// launch and traffic the fusion deletes. Under `(lb, id)` the smaller id —
/// always the unfused original, minted first — won every such tie, so a
/// decode-step graph seeded 1,823 `KMap` launches of which 1,088 were pure
/// identity copies of views, and at model scale the move budget
/// (`max_move_work / nodes` ≈ 1 move at 45k nodes) can never repair a single
/// one of them. Ranking equal-picosecond members by how few launches their
/// chain can realize adopts the merged spellings the rules already minted,
/// exactly where doing so is free by the picosecond bound.
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
    // TEMPORARY PROBE — delete before finishing. `FUSOR2_SEED_DEBUG=<id>`
    // prints every selectable member's seed key for that class.
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

/// [`argmin_member`] over the members `banned` does not name.
///
/// The one caller is the seed's cycle repair: a class on a cyclic selection
/// re-picks with the member that closed the loop struck out. Returning `None`
/// when every candidate is banned is what makes that loop terminate — each
/// repair strikes one member off a finite pool, and a class with nothing left
/// is a genuine failure the caller reports rather than one it can plan around.
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
/// mask, same admissibility argument (`min` and saturating `+` over `u32` are
/// monotone). Consumed by [`argmin_member`] as the tie-break only; the
/// picosecond bound stays the primary key, so a strictly cheaper member is
/// never displaced.
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
        Op::L0(L0::Leaf(_)) => 0,
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

// ---------------------------------------------------------------------------

fn combine(graph: &EGraph, id: Id, math: &[Picoseconds], lb: &[Picoseconds]) -> Picoseconds {
    let node = graph.node(id);
    match &node.op {
        Op::Union(a, b) => lb[a.index()].min(lb[b.index()]),
        Op::L0(L0::Leaf(_)) => Picoseconds(0),
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
    // Identical nodes at identical operand facts share a scan. Repeated
    // transformer layers and the trainer's three conv stages collapse to one
    // domain walk each.
    let mut memo: FxHashMap<u64, Picoseconds> = FxHashMap::default();
    // Sized on the scope, not the ambient graph: the ceiling exists to bound
    // this table's work, and this table only visits `ids`.
    let mut budget = math_call_budget(ids.len());
    for id in ids {
        let id = *id;
        let node = graph.node(id);
        if matches!(node.op, Op::Union(..) | Op::L0(L0::Leaf(_))) {
            continue;
        }
        let slot = &mut out[id.index()];
        // Hashing a node's operand facts costs about as much as two
        // `node_math` calls, so only a domain wide enough to pay for it gets
        // a memo entry.
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
        Op::L1(l1) => l1.schedule(),
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
            // `node_math` depends on the point only through the MAC unit
            // and the padded tile (`Roofline::node_math`: Coop pads to the
            // geometry's `(bm, bn)` at the coop rate, Sgemm to its
            // `(bm, bn)` at Fma, every other family prices as un-padded
            // Fma), so a domain is scanned once per *math-distinct* point,
            // not once per point. A coop domain is ~5,700 points wide but
            // holds only a handful of distinct `(bm, bn)` pairs — split,
            // subgroup and staging axes never move this term — and without
            // the dedupe those scans exhausted `math_call_budget` on an 8B
            // decode graph, zeroing every later node's math term. Zero-tied
            // classes then seeded on the launch-count tie-break alone,
            // which is how 65 m = n = 1 rmsnorm dots seeded as serial
            // 573 us Coop tiles that no race gate could ever reprice.
            let mut seen: SmallVec<[(u8, u32, u32); 12]> = SmallVec::new();
            let mut best: Option<Picoseconds> = None;
            for theta in domain.iter() {
                let key = match theta {
                    fusor2_ir::ir::level1::SchedPoint::Coop { geom, .. } => {
                        (1u8, geom.bm, geom.bn)
                    }
                    fusor2_ir::ir::level1::SchedPoint::Sgemm(p) => (2u8, p.bm, p.bn),
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

    /// An `L0` node never wins selection over its own lowering, however the
    /// bound compares.
    ///
    /// A floor lowering is worse in *schedule*, not in arithmetic, so the two
    /// members' bounds are at best a tie and can favour the `L0` node
    /// outright. Either way the `L0` node is not runnable, so cost must not
    /// get a vote: only [`crate::realize::selectable`] members are candidates.
    ///
    /// The regression this pins: `argmin_member` ranked every member and broke
    /// ties by smaller `Id` — always the un-lowered original — so `verify_plan`
    /// rejected every plan with "selected %N is at l0 but only L1 nodes are
    /// runnable", for every op, on both backends.
    #[test]
    fn an_l0_node_never_wins_selection_over_its_own_lowering() {
        use crate::realize::testkit::{N, buffer, kmap, new_graph};
        use fusor2_ir::ir::Level;

        let mut graph = new_graph();
        let shape = [N];
        let leaf = buffer(&mut graph, 0, &shape);
        // An `L0::Map` and an `L1::KMap` with the identical body: same work,
        // same bound, and the L0 node has the smaller id.
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
        // The premise: on cost alone the L0 node wins — it is no dearer, and
        // it holds the smaller id, so it also wins every tie-break.
        assert!(lb[l0.index()] <= lb[l1.index()]);
        assert!(l0 < l1);

        let picked = argmin_member(&graph, &lb, &launch_bound(&graph), class, &test_caps());
        assert_eq!(picked, l1, "selection must skip the un-lowered L0 node");
        assert_eq!(graph.level(picked), Level::L1);
        assert!(crate::realize::is_runnable(&graph, picked));
        assert!(!crate::realize::is_runnable(&graph, l0));
    }

    /// Equal picosecond bounds break toward the member whose chain realizes
    /// fewer launches — the fused spelling a rule minted — never toward the
    /// smaller id. This is the decode-step regression in miniature: the
    /// relaxation erases exactly the launch and traffic a fusion deletes, so
    /// `(lb, id)` kept the unfused chain on every one of an 8B decode step's
    /// ~1,800 elementwise launches.
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
