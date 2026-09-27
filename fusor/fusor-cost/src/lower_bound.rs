//! Per-node arithmetic estimates for initial selection and candidate ordering.
//! Complete selected DAGs are compared through the realized cost model.

use crate::nodes::{composite_members, domain_of, sgemv_lanes};
use fusor_ir::cost::{CostModel, Picoseconds};
use fusor_ir::device::Caps;
use fusor_ir::egraph::{ClassId, EGraph, Id};
use fusor_ir::facts::ValueFacts;
use fusor_ir::ir::launch::{SchedPoint, ScheduleDomain};
use fusor_ir::ir::logical::Logical;
use fusor_ir::ir::{Node, Op};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

/// Domain size past which a node earns a memo entry.
const MEMO_THRESHOLD: usize = 8;

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
            _ => match composite_members(graph, id) {
                Some(members) => {
                    let time = members.iter().fold(Picoseconds(launch_ps), |sum, m| {
                        sum + Picoseconds(bounds.costs[m.index()].0.saturating_sub(launch_ps))
                    });
                    (time, 1)
                }
                None => (math[id.index()] + Picoseconds(launch_ps), 1),
            },
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
    crate::debug::seed_members(graph, lb, launches, class, caps);
    let chosen = argmin_member_excluding(graph, lb, launches, class, caps, &Default::default())
        .unwrap_or(class.0);
    crate::debug::seed_chosen(class, chosen);
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
        if matches!(
            graph.node(id).op,
            Op::Union(..) | Op::Logical(Logical::Leaf(_))
        ) {
            continue;
        }
        let slot = &mut out[id.index()];
        // Hashing operand facts costs about two `node_math` calls, so only a
        // domain wide enough to pay for it gets a memo entry.
        if domain_of(graph, id).map_or(1, |d| d.len()) <= MEMO_THRESHOLD {
            *slot = best_math(graph, cost, id);
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
    crate::debug::math_table(graph, ids, &out);
    out
}

/// `argmin over sched.iter()` of `node_math`; `ScheduleDomain::Point` passes
/// `None`, as does any node without a domain.
fn best_math(graph: &EGraph, cost: &dyn CostModel, id: Id) -> Picoseconds {
    let node = graph.node(id);
    let (ins, out) = node_facts(graph, id);
    match domain_of(graph, id) {
        None | Some(ScheduleDomain::Point) => cost.node_math(node, &ins, out, None),
        Some(domain) => {
            let mut best: Option<Picoseconds> = None;
            priced_points(node, &ins, out, domain, cost, |_, v| {
                best = Some(best.map_or(v, |b| b.min(v)));
            });
            best.unwrap_or(Picoseconds(0))
        }
    }
}

/// The first of `domain`'s points with the least `node_math`.
pub(crate) fn cheapest_point(
    node: &Node,
    ins: &[ValueFacts],
    out: &ValueFacts,
    domain: &ScheduleDomain,
    cost: &dyn CostModel,
) -> Option<SchedPoint> {
    let mut best: Option<(SchedPoint, Picoseconds)> = None;
    priced_points(node, ins, out, domain, cost, |theta, v| {
        if best.is_none_or(|(_, b)| v < b) {
            best = Some((theta, v));
        }
    });
    best.map(|(theta, _)| theta)
}

/// `domain`'s points, cheapest `node_math` first, ties by domain index.
pub(crate) fn ranked_points(
    node: &Node,
    ins: &[ValueFacts],
    out: &ValueFacts,
    domain: &ScheduleDomain,
    cost: &dyn CostModel,
) -> Vec<SchedPoint> {
    let mut points: Vec<(Picoseconds, usize, SchedPoint)> = Vec::with_capacity(domain.len());
    priced_points(node, ins, out, domain, cost, |theta, v| {
        points.push((v, points.len(), theta));
    });
    points.sort_by_key(|(v, i, _)| (*v, *i));
    points.into_iter().map(|(_, _, theta)| theta).collect()
}

/// Every point of `domain` in order with its `node_math`. `node_math`
/// depends on the point only through the MAC unit, the padded tile, the
/// k-step floor and a fold's lane group, so it runs once per math-distinct
/// point.
fn priced_points(
    node: &Node,
    ins: &[ValueFacts],
    out: &ValueFacts,
    domain: &ScheduleDomain,
    cost: &dyn CostModel,
    mut visit: impl FnMut(SchedPoint, Picoseconds),
) {
    let caps = &cost.facts().caps;
    let mut seen: SmallVec<[(MathKey, Picoseconds); 12]> = SmallVec::new();
    for theta in domain.iter() {
        let key = math_key(theta, caps);
        let v = match seen.iter().find(|(k, _)| *k == key) {
            Some((_, v)) => *v,
            None => {
                let v = cost.node_math(node, ins, out, Some(theta));
                seen.push((key, v));
                v
            }
        };
        visit(theta, v);
    }
}

/// What `node_math` reads off a point.
type MathKey = (u8, u32, u32, u32);

fn math_key(theta: SchedPoint, caps: &Caps) -> MathKey {
    match theta {
        // The k-step floor moves with `bk`, so it is part of the key.
        SchedPoint::Coop { geom, .. } => (1, geom.bm, geom.bn, geom.bk),
        SchedPoint::Sgemm(p) => (2, p.bm, p.bn, p.bk),
        // A fold's floor moves with its lane group.
        SchedPoint::Fold(s) => (3, s.lane_group(caps.subgroup_width()), 0, 0),
        SchedPoint::Sgemv(p) => (4, sgemv_lanes(p, caps), 0, 0),
        _ => (0, 0, 0, 0),
    }
}

/// The facts of `id`'s operands, in child order, and of its value.
pub(crate) fn node_facts(graph: &EGraph, id: Id) -> (SmallVec<[ValueFacts; 4]>, &ValueFacts) {
    let ins = graph
        .node(id)
        .children
        .iter()
        .map(|c| graph.facts(*c).clone())
        .collect();
    (ins, graph.facts(id))
}

pub(crate) type ShapeKey = (Op, SmallVec<[ValueFacts; 4]>, ValueFacts);

pub(crate) fn shape_key(graph: &EGraph, id: Id) -> ShapeKey {
    struct ArithmeticOperands;
    impl fusor_ir::ir::visit::VisitMut for ArithmeticOperands {
        fn operand(&mut self, operand: &mut fusor_ir::ir::launch::Operand) {
            operand.src = Id(0);
        }
    }
    let mut op = graph.node(id).op.clone();
    // Arithmetic depends on operand facts and access maps, not their arena ids.
    op.visit_mut(&mut ArithmeticOperands);
    let (ins, out) = node_facts(graph, id);
    (op, ins, out.clone())
}
