//! The admissible lower bound.
//!
//! `lb(c) = min over n in c of ( math_ps(n) + sum over *distinct* child chains
//! lb(child) )` — zero traffic, free sharing, min over the schedule domain.
//! Admissible in both regimes, so it works as a seed and as a
//! branch-and-bound prune.

use fusor_ir::cost::{CostModel, Picoseconds};
use fusor_ir::device::Caps;
use fusor_ir::egraph::{ClassId, EGraph, Id};
use fusor_ir::facts::ValueFacts;
use fusor_ir::ir::Op;
use fusor_ir::ir::launch::{Launch, ScheduleDomain};
use fusor_ir::ir::logical::Logical;
use rustc_hash::{FxHashMap, FxHasher};
use smallvec::SmallVec;
use std::hash::{Hash, Hasher};

/// One postorder sweep from `0`. The operator is monotone, so the sweep is
/// exact wherever the class graph is acyclic and an underestimate through a
/// class cycle — still admissible, which is all the bound promises.
///
/// Not iterated to a fixpoint: through the identity-shaped cycles a large
/// graph carries (a value unioned with a copy of itself, a region reading
/// its own class) every further sweep compounds the cycle's cost into every
/// reader, ~35× per sweep on a 140k-node vision graph, until both bounds
/// saturate and every class downstream ties at infinity. The seed then
/// falls to the smallest id, which is the definitional fold of every
/// contraction.
const MAX_PASSES: u32 = 1;

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

/// Indexed by node id. One bottom-up sweep in dependency postorder.
pub(crate) fn lower_bound(graph: &EGraph, cost: &dyn CostModel) -> Vec<Picoseconds> {
    let ids: Vec<Id> = (0..graph.len()).map(|i| Id(i as u32)).collect();
    lower_bound_over(graph, cost, &ids)
}

/// [`lower_bound`] over the masked slots only. The vector is still indexed by
/// node id — unmasked slots stay `0`. The mask must come from
/// [`crate::realize::reachable`], which is closed under both class membership
/// and children, so every id the extractor can index is masked.
pub(crate) fn lower_bound_scoped(
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

    let debug = std::env::var_os("FUSOR_SEED_DEBUG").is_some();
    for pass in 0..MAX_PASSES {
        let mut changed = 0usize;
        for id in &order {
            let next = combine(graph, *id, &math, &lb, cost.facts().launch_ps);
            if next != lb[id.index()] {
                lb[id.index()] = next;
                changed += 1;
            }
        }
        if debug {
            let max = lb.iter().map(|p| p.0).max().unwrap_or(0);
            eprintln!("[lb] pass {pass}: {changed} changed, max {max}");
        }
        if changed == 0 {
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
            let debug = std::env::var_os("FUSOR_SEED_DEBUG").is_some();
            let push = |next: Id, stack: &mut Vec<(Id, bool)>| {
                if state[next.index()] == 0 && masked.contains(next.index()) {
                    stack.push((next, false));
                } else if debug && state[next.index()] == 1 {
                    // A back edge: the class graph has a cycle through here,
                    // and both bounds will climb until they saturate.
                    let show = |i: Id| {
                        let s = format!("{:?}", graph.node(i).op);
                        s.chars().take(160).collect::<String>()
                    };
                    eprintln!(
                        "[lb] class cycle edge {id:?} -> {next:?}\n      {id:?} = {}\n      {next:?} = {}",
                        show(id),
                        show(next)
                    );
                }
            };
            match &node.op {
                Op::Union(a, b) => {
                    push(*a, &mut stack);
                    push(*b, &mut stack);
                }
                // A composite names its members by id, and its last member
                // shares its class: through the class that edge is a cycle
                // and the member's bound would be read before it is made.
                Op::Launch(Launch::Slab { members, .. } | Launch::Group { members, .. }) => {
                    for m in members.iter() {
                        push(*m, &mut stack);
                        // A member's class too: a group prices each member
                        // against its class's best.
                        push(graph.class_of(*m).0, &mut stack);
                    }
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
/// The relaxation erases exactly the launch and traffic a fusion deletes,
/// so fused and unfused spellings tie on picoseconds; each launch the chain
/// keeps is priced at the device's dispatch cost, which is what a fusion
/// saves.
pub(crate) fn argmin_member(
    graph: &EGraph,
    lb: &[Picoseconds],
    launches: &[u32],
    class: ClassId,
    caps: &Caps,
    launch_ps: u64,
) -> Id {
    if crate::realize::is_singleton(graph, class) {
        return class.0;
    }
    // `FUSOR_SEED_DEBUG=<id>` prints every selectable member's seed key for
    // that class.
    if let Ok(want) = std::env::var("FUSOR_SEED_DEBUG")
        && want == class.0.index().to_string()
    {
        for m in crate::realize::selectable(graph, class, caps) {
            let show: String = format!("{:?}", graph.node(m).op)
                .replace("ScalarExpr(ScalarNode { kind: ", "")
                .chars()
                .take(220)
                .collect();
            let excess: Vec<String> = match &graph.node(m).op {
                Op::Launch(Launch::Group { members, .. }) => members
                    .iter()
                    .map(|x| {
                        let c = graph.class_of(*x);
                        format!(
                            "{x}:c{}:+{}us:best={:?}",
                            c.0.index(),
                            lb[x.index()].0.saturating_sub(lb[c.0.index()].0) / 1_000_000,
                            argmin_member_excluding(graph, lb, launches, c, caps, launch_ps, &Default::default())
                        )
                    })
                    .collect(),
                _ => Vec::new(),
            };
            eprintln!(
                "[seed] class {} member {m:?} lb={} launches={} excess={excess:?} op={show}",
                class.0.index(),
                lb[m.index()].0,
                launches[m.index()],
            );
        }
    }
    let chosen =
        argmin_member_excluding(graph, lb, launches, class, caps, launch_ps, &Default::default())
            .unwrap_or(class.0);
    if let Ok(want) = std::env::var("FUSOR_SEED_DEBUG")
        && want == class.0.index().to_string()
    {
        eprintln!("[seed] class {} chose {chosen:?}", class.0.index());
    }
    chosen
}

/// [`argmin_member`] over the members `banned` does not name. Returns `None`
/// when every candidate is banned, which is what makes the seed's cycle-repair
/// loop terminate.
pub(crate) fn argmin_member_excluding(
    graph: &EGraph,
    lb: &[Picoseconds],
    launches: &[u32],
    class: ClassId,
    caps: &Caps,
    launch_ps: u64,
    banned: &rustc_hash::FxHashSet<Id>,
) -> Option<Id> {
    crate::realize::selectable(graph, class, caps)
        .into_iter()
        .filter(|m| !banned.contains(m))
        .min_by_key(|m| {
            let _ = launch_ps;
            (lb[m.index()], launches[m.index()], *m)
        })
}

/// The launch-count analogue of [`lower_bound_scoped`]: per node, the fewest
/// launches any realization of that node's chain can dispatch — every
/// non-leaf node is one launch plus its distinct child chains, sharing free,
/// `min` over members. Same Kleene iteration, same closure requirement on the
/// mask. Consumed by [`argmin_member`] as the tie-break only.
pub(crate) fn launch_bound_scoped(graph: &EGraph, mask: &fixedbitset::FixedBitSet) -> Vec<u32> {
    let ids: Vec<Id> = mask.ones().map(|i| Id(i as u32)).collect();
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
        // One dispatch for every stage, plus whatever feeds the stages from
        // outside.
        Op::Launch(Launch::Slab { members, .. }) => {
            let mut total = 1u32;
            for class in slab_inputs(graph, members) {
                total = total.saturating_add(l[class.0.index()]);
            }
            total
        }
        // The last member's chain, less the dispatches the other members
        // would have been.
        Op::Launch(Launch::Group { members, .. }) => {
            let last = members.last().copied().unwrap_or(id);
            let mut total = l[last.index()].saturating_sub(members.len().saturating_sub(1) as u32);
            for m in &members[..members.len().saturating_sub(1)] {
                let excess = l[m.index()].saturating_sub(l[graph.class_of(*m).0.index()]);
                total = total.saturating_add(excess);
            }
            total
        }
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

/// A launch node's bound is its math, its own dispatch, and its children's
/// bounds; a slab's is one dispatch, its stages' math, and its inputs'
/// bounds. The dispatch is in the bound itself so a spelling that fuses
/// launches away is cheaper by that much in one number — a separate launch
/// count, relaxed on its own, credits a fold over split partials with the
/// unsplit contraction's single dispatch.
fn combine(
    graph: &EGraph,
    id: Id,
    math: &[Picoseconds],
    lb: &[Picoseconds],
    launch_ps: u64,
) -> Picoseconds {
    let node = graph.node(id);
    match &node.op {
        Op::Union(a, b) => lb[a.index()].min(lb[b.index()]),
        Op::Logical(Logical::Leaf(_)) => Picoseconds(0),
        // The stages' own math, once each, plus the bound of what feeds them
        // from outside. Summing the members' bounds would count every
        // stage's producers once per stage that reads them.
        Op::Launch(Launch::Slab { members, .. }) => {
            let mut total = math[id.index()] + Picoseconds(launch_ps);
            for m in members {
                total += math[m.index()];
            }
            for class in slab_inputs(graph, members) {
                total += lb[class.0.index()];
            }
            // A middle member that is a root of the graph — an optimizer
            // state, say — would be its own dispatch otherwise; the slab
            // computes it on the way. The bound is for the plan, and the
            // plan pays that dispatch nowhere else.
            let roots: SmallVec<[ClassId; 8]> = graph.roots().iter().map(|r| graph.class_of(*r)).collect();
            let saved = members[..members.len().saturating_sub(1)]
                .iter()
                .filter(|m| roots.contains(&graph.class_of(**m)))
                .count() as u64;
            Picoseconds(total.0.saturating_sub(saved.saturating_mul(launch_ps)))
        }
        // Every member but the last is computed anyway — each is a root or
        // another launch's input — so the group ties its last member's own
        // spelling and wins on the launch count. Crediting the dispatches
        // here would flow into every reader's bound.
        // A member spelled worse than its class's best costs the group the
        // difference: groups minted before the fused spellings existed
        // carry the plain ones.
        // Each other member is computed anyway — a root, or another
        // launch's input — and the group is its dispatch cheaper. The
        // credit reaches a root's readers as a uniform shift, which moves
        // no choice of theirs.
        Op::Launch(Launch::Group { members, .. }) => {
            let last = members.last().copied().unwrap_or(id);
            let mut total = lb[last.index()];
            for m in &members[..members.len().saturating_sub(1)] {
                let excess = lb[m.index()].0.saturating_sub(lb[graph.class_of(*m).0.index()].0);
                total = Picoseconds(total.0.saturating_add(excess));
            }
            let saved = members.len().saturating_sub(1) as u64;
            Picoseconds(total.0.saturating_sub(saved.saturating_mul(launch_ps)))
        }
        _ => {
            // Deduplicate children by class: a node reading the same class
            // twice contributes once, which is what makes sharing free.
            let mut seen: SmallVec<[ClassId; 4]> = SmallVec::new();
            // A logical node is computed by some launch too; without its
            // dispatch every class's bound would run through its logical
            // spelling and no fusion would ever look cheaper than none.
            let dispatch = launch_ps;
            let mut total = math[id.index()] + Picoseconds(dispatch);
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

/// The classes a slab reads that none of its members produce, each once.
fn slab_inputs(graph: &EGraph, members: &[Id]) -> Vec<ClassId> {
    let own: SmallVec<[ClassId; 8]> = members.iter().map(|m| graph.class_of(*m)).collect();
    let mut out: Vec<ClassId> = Vec::new();
    for m in members {
        for child in graph.node(*m).children.iter() {
            let class = graph.class_of(*child);
            if !own.contains(&class) && !out.contains(&class) {
                out.push(class);
            }
        }
    }
    out
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
            if slot.0 >= u64::MAX / 4 && std::env::var_os("FUSOR_SEED_DEBUG").is_some() {
                let show: String = format!("{:?}", node.op).chars().take(200).collect();
                eprintln!("[lb] math saturated at {id:?}: {show}");
            }
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
    if let Ok(want) = std::env::var("FUSOR_SEED_DEBUG") {
        for id in ids {
            if want == graph.class_of(*id).0.index().to_string() {
                let show: String = format!("{:?}", graph.node(*id).op).chars().take(120).collect();
                eprintln!("[math] class {want} node {id} math={} budget_left={budget} {show}", out[id.index()].0);
            }
        }
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
                // A promoted fold inherits its pre-promotion domain, most of
                // which its carrier's footprint rules out; the floor is over
                // the points that can lower.
                if !crate::realize::point_is_legal(graph, id, theta, &cost.facts().caps) {
                    continue;
                }
                let key = match theta {
                    // The k-step floor moves with `bk` and the split count,
                    // so those are part of the key.
                    fusor_ir::ir::launch::SchedPoint::Coop { geom, splits, .. } => {
                        (1u8, geom.bm * 1024 + geom.bn, geom.bk * 1024 + splits)
                    }
                    fusor_ir::ir::launch::SchedPoint::Sgemm(p) => (2u8, p.bm * 1024 + p.bn, p.bk),
                    // A fold's floor moves with its lane group.
                    fusor_ir::ir::launch::SchedPoint::Fold(s) => {
                        (3u8, s.lane_group(cost.facts().caps.subgroup_width()), 0)
                    }
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
