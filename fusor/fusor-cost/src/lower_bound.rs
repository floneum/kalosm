//! Per-node arithmetic estimates for initial selection and candidate ordering.
//! Complete selected DAGs are compared through the realized cost model.

use fusor_ir::cost::{CostModel, Picoseconds};
use fusor_ir::device::Caps;
use fusor_ir::egraph::{ClassId, EGraph, Id};
use fusor_ir::facts::ValueFacts;
use fusor_ir::ir::Op;
use fusor_ir::ir::launch::{Launch, ScheduleDomain};
use fusor_ir::ir::logical::Logical;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

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
    /// The node itself needs a dispatch unless it is a leaf or a union.
    pub launches: Vec<u32>,
}

pub(crate) fn lower_bound(graph: &EGraph, cost: &dyn CostModel) -> Vec<Picoseconds> {
    let ids: Vec<Id> = (0..graph.len()).map(|i| Id(i as u32)).collect();
    bounds_over(graph, Some(cost), &ids).costs
}

/// Unmasked slots stay zero; all selected candidate plans are priced separately.
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
    for &id in ids {
        let (time, launches) = match &graph.node(id).op {
            Op::Union(a, b) => (
                bounds.costs[a.index()].min(bounds.costs[b.index()]),
                bounds.launches[a.index()].min(bounds.launches[b.index()]),
            ),
            Op::Logical(Logical::Leaf(_)) => (Picoseconds(0), 0),
            Op::Launch(Launch::Slab { members, .. } | Launch::Group { members, .. }) => {
                let time = members.iter().fold(Picoseconds(launch_ps), |sum, m| {
                    sum + Picoseconds(bounds.costs[m.index()].0.saturating_sub(launch_ps))
                });
                (time, 1)
            }
            _ => (math[id.index()] + Picoseconds(launch_ps), 1),
        };
        bounds.costs[id.index()] = time;
        bounds.launches[id.index()] = launches;
    }
    bounds
}

/// The cheapest **selectable** member of `class`, picosecond ties broken by
/// own dispatch count, then by smaller [`Id`]. Used for cycle repair and
/// classes without an ordinary lowering.
///
/// Selectable, not just cheapest: the floor lowerings tie with the `Logical` node
/// they replace on math, so an unrestricted `min_by_key` would return the
/// un-lowered original every time. See [`crate::realize::selectable`].
///
/// Dependencies are deliberately absent from this ordering estimate. Their
/// complete shared DAG is priced when the candidate selection is realized.
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

fn node_math_table(graph: &EGraph, cost: &dyn CostModel, ids: &[Id]) -> Vec<Picoseconds> {
    let n = graph.len();
    let mut out = vec![Picoseconds(0); n];
    // Identical nodes at identical operand facts share a scan.
    let mut memo: FxHashMap<ShapeKey, Picoseconds> = FxHashMap::default();
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
            *slot = best_math(graph, cost, id);
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
                let v = best_math(graph, cost, id);
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
                    "[math] class {want} node {id} math={} {show}",
                    out[id.index()].0
                );
            }
        }
    }
    out
}

/// `argmin over sched.iter()` of `node_math`; `ScheduleDomain::Point` passes
/// `None`, as does any node without a domain.
fn best_math(graph: &EGraph, cost: &dyn CostModel, id: Id) -> Picoseconds {
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
        None | Some(ScheduleDomain::Point) => cost.node_math(node, &ins, out, None),
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
                    fusor_ir::ir::launch::SchedPoint::Sgemv(p) => {
                        let caps = &cost.facts().caps;
                        let width = caps.subgroup_width();
                        let lanes = if p.cols > 1 {
                            width
                        } else {
                            (p.subgroups.max(1) * width)
                                .min(caps.limits.max_compute_invocations_per_workgroup)
                                .max(1)
                        };
                        (4u8, lanes, 0)
                    }
                    _ => (0u8, 0, 0),
                };
                if seen.contains(&key) {
                    continue;
                }
                seen.push(key);
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

pub(crate) type ShapeKey = (Op, SmallVec<[ValueFacts; 4]>, ValueFacts);

pub(crate) fn shape_key(graph: &EGraph, id: Id) -> ShapeKey {
    struct ArithmeticOperands;
    impl fusor_ir::ir::visit::VisitMut for ArithmeticOperands {
        fn operand(&mut self, operand: &mut fusor_ir::ir::launch::Operand) {
            operand.src = Id(0);
        }
    }
    let node = graph.node(id);
    let mut op = node.op.clone();
    // Arithmetic depends on operand facts and access maps, not their arena ids.
    op.visit_mut(&mut ArithmeticOperands);
    (
        op,
        node.children
            .iter()
            .map(|c| graph.facts(*c).clone())
            .collect(),
        graph.facts(id).clone(),
    )
}
