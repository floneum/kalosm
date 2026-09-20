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

pub(crate) struct Bounds {
    pub costs: Vec<Picoseconds>,
    pub launches: Vec<u32>,
}

pub(crate) fn lower_bound(graph: &EGraph, cost: &dyn CostModel) -> Vec<Picoseconds> {
    let ids: Vec<Id> = (0..graph.len()).map(|i| Id(i as u32)).collect();
    bounds_over(graph, Some(cost), &ids).costs
}

/// Both bounds use the same dependency postorder. Unmasked slots stay zero.
pub(crate) fn bounds_scoped(
    graph: &EGraph,
    cost: Option<&dyn CostModel>,
    mask: &fixedbitset::FixedBitSet,
) -> Bounds {
    let ids: Vec<Id> = mask.ones().map(|i| Id(i as u32)).collect();
    bounds_over(graph, cost, &ids)
}

fn bounds_over(graph: &EGraph, cost: Option<&dyn CostModel>, ids: &[Id]) -> Bounds {
    let mut bounds = Bounds {
        costs: vec![Picoseconds(0); graph.len()],
        launches: vec![0; graph.len()],
    };
    let math = cost.map_or_else(
        || vec![Picoseconds(0); graph.len()],
        |cost| node_math_table(graph, cost, ids),
    );
    let launch_ps = cost.map_or(0, |cost| cost.facts().launch_ps);
    // One sweep underestimates class cycles. Iterating compounds their cost
    // into their readers until every downstream member ties at saturation.
    for id in postorder(graph, ids) {
        let (time, launches) = combine(graph, id, &math, &bounds, launch_ps);
        bounds.costs[id.index()] = time;
        bounds.launches[id.index()] = launches;
    }
    bounds
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
                            argmin_member_excluding(
                                graph,
                                lb,
                                launches,
                                c,
                                caps,
                                &Default::default()
                            )
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
    let chosen = argmin_member_excluding(graph, lb, launches, class, caps, &Default::default())
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
    banned: &rustc_hash::FxHashSet<Id>,
) -> Option<Id> {
    crate::realize::selectable(graph, class, caps)
        .into_iter()
        .filter(|m| !banned.contains(m))
        .min_by_key(|m| (lb[m.index()], launches[m.index()], *m))
}

fn combine(
    graph: &EGraph,
    id: Id,
    math: &[Picoseconds],
    bounds: &Bounds,
    launch_ps: u64,
) -> (Picoseconds, u32) {
    let (lb, launches) = (&bounds.costs, &bounds.launches);
    let node = graph.node(id);
    match &node.op {
        Op::Union(a, b) => (
            lb[a.index()].min(lb[b.index()]),
            launches[a.index()].min(launches[b.index()]),
        ),
        Op::Logical(Logical::Leaf(_)) => (Picoseconds(0), 0),
        Op::Launch(Launch::Slab { members, .. }) => {
            let mut time = math[id.index()] + Picoseconds(launch_ps);
            let mut count = 1u32;
            for m in members {
                time += math[m.index()];
            }
            for class in slab_inputs(graph, members) {
                time += lb[class.0.index()];
                count = count.saturating_add(launches[class.0.index()]);
            }
            let roots: SmallVec<[ClassId; 8]> =
                graph.roots().iter().map(|r| graph.class_of(*r)).collect();
            let saved = members[..members.len().saturating_sub(1)]
                .iter()
                .filter(|m| roots.contains(&graph.class_of(**m)))
                .count() as u64;
            (
                Picoseconds(time.0.saturating_sub(saved.saturating_mul(launch_ps))),
                count,
            )
        }
        Op::Launch(Launch::Group { members, .. }) => {
            let last = members.last().copied().unwrap_or(id);
            let saved = members.len().saturating_sub(1);
            let mut time = lb[last.index()];
            let mut count = launches[last.index()].saturating_sub(saved as u32);
            for m in &members[..saved] {
                let class = graph.class_of(*m).0.index();
                time = Picoseconds(
                    time.0
                        .saturating_add(lb[m.index()].0.saturating_sub(lb[class].0)),
                );
                count = count.saturating_add(launches[m.index()].saturating_sub(launches[class]));
            }
            (
                Picoseconds(
                    time.0
                        .saturating_sub((saved as u64).saturating_mul(launch_ps)),
                ),
                count,
            )
        }
        _ => {
            let mut seen: SmallVec<[ClassId; 4]> = SmallVec::new();
            let mut time = math[id.index()] + Picoseconds(launch_ps);
            let mut count = 1u32;
            for child in node.children.iter() {
                let class = graph.class_of(*child);
                if !seen.contains(&class) {
                    seen.push(class);
                    time += lb[class.0.index()];
                    count = count.saturating_add(launches[class.0.index()]);
                }
            }
            (time, count)
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
                let show: String = format!("{:?}", graph.node(*id).op)
                    .chars()
                    .take(120)
                    .collect();
                eprintln!(
                    "[math] class {want} node {id} math={} budget_left={budget} {show}",
                    out[id.index()].0
                );
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
                let key = match theta {
                    // The k-step floor moves with `bk` and the split count,
                    // so those are part of the key.
                    fusor_ir::ir::launch::SchedPoint::Coop { geom, .. } => {
                        (1u8, geom.bm * 1024 + geom.bn, geom.bk)
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
