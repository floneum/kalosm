//! The three local-search moves and their incremental deltas.
//!
//! Each move's delta is recomputed over only the affected launches, via a
//! union-find over the realized cut. The accept test is always the exact
//! global cost; the schedule score only orders the `RESCHEDULE` frontier,
//! never gates it.
//!
//! `FLIP` is refused when the node is pinned: inlining an `Effect::InPlace`
//! node (an atomic scatter) into two consumers doubles the write. Purity is
//! a precondition of the materialization move.

use crate::realize::{self, Realized};
use fusor_ir::cost::{CostModel, Picoseconds};
use fusor_ir::egraph::{ClassId, EGraph, Id};
use fusor_ir::extract::{ExtractBudget, Extraction, Move};
use fusor_ir::facts::ValueFacts;
use fusor_ir::ir::Op;
use fusor_ir::ir::launch::{Effect, Launch, SchedPoint, ScheduleDomain};
use fusor_ir::shape::Layout;
use rustc_hash::{FxHashMap, FxHasher};
use smallvec::SmallVec;
use std::hash::{Hash, Hasher};

/// One concrete state change a [`Move`] can produce. A `Move` names the
/// dimension; a `Candidate` names the value.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Candidate {
    Select { class: ClassId, node: Id },
    Materialize { node: Id, on: bool },
    Schedule { node: Id, theta: SchedPoint },
}

/// Enough state to revert one move exactly.
#[derive(Clone, Debug)]
pub(crate) enum Undo {
    /// `node` and `node_was_materialized` are the *new* member's own prior
    /// state: a reselect carries `M` across with the selection, so reverting
    /// has to put the new member's bit back as well as the old selection.
    Reselect {
        class: ClassId,
        was: Id,
        node: Id,
        node_was_materialized: bool,
    },
    Flip {
        node: Id,
        was_materialized: bool,
    },
    Reschedule {
        node: Id,
        was: Option<SchedPoint>,
    },
}

/// Memo for the `RESCHEDULE` frontier: the per-point score and the sorted
/// order, both keyed on `(node, context_hash)`.
#[derive(Default)]
pub(crate) struct SchedCache {
    order: FxHashMap<(Id, u64), Vec<SchedPoint>>,
    score: FxHashMap<(Id, SchedPoint, u64), Picoseconds>,
}

impl SchedCache {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Points of `id`'s domain, cheapest `node_math` first. The full domain
    /// is always returned; ordering never gates.
    pub(crate) fn ordered(
        &mut self,
        graph: &EGraph,
        id: Id,
        context: u64,
        cost: &dyn CostModel,
    ) -> &[SchedPoint] {
        if !self.order.contains_key(&(id, context)) {
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
            let mut points: Vec<(Picoseconds, usize, SchedPoint)> = match domain {
                None | Some(ScheduleDomain::Point) => Vec::new(),
                Some(d) => d
                    .iter()
                    .enumerate()
                    .map(|(i, theta)| {
                        let s = cost.node_math(node, &ins, out, Some(theta));
                        self.score.insert((id, theta, context), s);
                        (s, i, theta)
                    })
                    .collect(),
            };
            // Ties break by domain index, so the order is total and stable.
            points.sort_by_key(|(s, i, _)| (*s, *i));
            self.order.insert(
                (id, context),
                points.into_iter().map(|(_, _, t)| t).collect(),
            );
        }
        &self.order[&(id, context)]
    }
}

/// Every move worth offering at this state, in a deterministic order:
/// classes ascending, then nodes ascending.
pub(crate) fn frontier(
    graph: &EGraph,
    extraction: &Extraction,
    classes: &[ClassId],
    budget: ExtractBudget,
) -> Vec<Move> {
    let _ = budget;
    let mut out = Vec::new();
    for class in classes {
        if !realize::is_singleton(graph, *class) {
            out.push(Move::Reselect(*class));
        }
    }
    let mut selected: Vec<Id> = extraction.sigma.values().copied().collect();
    selected.sort_unstable();
    selected.dedup();
    for id in selected {
        out.push(Move::Flip(id));
        if let Some(d) = domain(graph, id)
            && d.len() > 1
        {
            out.push(Move::Reschedule(id));
        }
    }
    out
}

/// The concrete states `mv` can move to, best first, excluding the state the
/// extraction is already in.
pub(crate) fn candidates(
    graph: &EGraph,
    extraction: &Extraction,
    realized: &Realized,
    mv: Move,
    lb: &[Picoseconds],
    cache: &mut SchedCache,
    cost: &dyn CostModel,
) -> SmallVec<[Candidate; 8]> {
    let mut out: SmallVec<[Candidate; 8]> = SmallVec::new();
    match mv {
        Move::Reselect(class) => {
            let current = extraction.sigma.get(&class).copied();
            // Only runnable members: the verifier rejects the un-lowered
            // `Logical` node.
            let mut members = realize::selectable(graph, class, &cost.facts().caps);
            // lb-ascending, ties by smaller id.
            members.sort_by_key(|m| (lb[m.index()], *m));
            for m in members {
                if Some(m) != current {
                    out.push(Candidate::Select { class, node: m });
                }
            }
        }
        Move::Flip(node) => {
            let on = !extraction.is_materialized(node);
            // Only leaving `M` needs a guard: a node cut from a consumer by
            // structure has to land in a buffer, or the consumer's launch
            // reads a value nothing ever wrote.
            let blocked = !on
                && (is_pinned(graph, &realized.roots, node)
                    || at_structural_boundary(graph, realized, node));
            if !blocked {
                out.push(Candidate::Materialize { node, on });
            }
        }
        Move::Reschedule(node) => {
            let current = extraction.theta.get(&node).copied();
            let context = context_hash(graph, realized, node);
            for theta in cache.ordered(graph, node, context, cost) {
                // A point whose footprint is over the device cap hard-asserts
                // in lowering. `has_legal_point` passes as soon as one point
                // fits, so each point must be checked here as well.
                if !realize::point_is_legal(graph, node, *theta, &cost.facts().caps) {
                    continue;
                }
                if Some(*theta) != current {
                    out.push(Candidate::Schedule {
                        node,
                        theta: *theta,
                    });
                }
            }
        }
    }
    out
}

/// Apply a candidate in place, returning the previous state.
pub(crate) fn apply(graph: &EGraph, extraction: &mut Extraction, c: Candidate) -> Option<Undo> {
    match c {
        Candidate::Select { class, node } => {
            let was = *extraction.sigma.get(&class)?;
            if was == node {
                return None;
            }
            let node_was_materialized = extraction.is_materialized(node);
            extraction.sigma.insert(class, node);
            // `M` is keyed by node, but the decision it records belongs to the
            // class: a value that had to land in a buffer still has to,
            // whichever member computes it.
            if extraction.is_materialized(was)
                && realize::leaf_role(graph, node) == realize::LeafRole::NotLeaf
            {
                set_materialized(extraction, node, true);
            }
            Some(Undo::Reselect {
                class,
                was,
                node,
                node_was_materialized,
            })
        }
        Candidate::Materialize { node, on } => {
            let was = extraction.is_materialized(node);
            if was == on {
                return None;
            }
            if on && realize::leaf_role(graph, node) != realize::LeafRole::NotLeaf {
                // A leaf already lives in a buffer (or is a literal); there
                // is no write for `M` to pay for.
                return None;
            }
            set_materialized(extraction, node, on);
            Some(Undo::Flip {
                node,
                was_materialized: was,
            })
        }
        Candidate::Schedule { node, theta } => {
            let was = extraction.theta.insert(node, theta);
            if was == Some(theta) {
                return None;
            }
            Some(Undo::Reschedule { node, was })
        }
    }
}

pub(crate) fn undo(extraction: &mut Extraction, undo: Undo) {
    match undo {
        Undo::Reselect {
            class,
            was,
            node,
            node_was_materialized,
        } => {
            set_materialized(extraction, node, node_was_materialized);
            extraction.sigma.insert(class, was);
        }
        Undo::Flip {
            node,
            was_materialized,
        } => {
            set_materialized(extraction, node, was_materialized);
        }
        Undo::Reschedule { node, was } => match was {
            Some(t) => {
                extraction.theta.insert(node, t);
            }
            None => {
                extraction.theta.remove(&node);
            }
        },
    }
}

/// Set one node's `M` bit, growing the set when the id is past its end.
fn set_materialized(extraction: &mut Extraction, node: Id, on: bool) {
    if extraction.m.len() <= node.index() {
        extraction.m.grow(node.index() + 1);
    }
    extraction.m.set(node.index(), on);
}

/// True when some realized consumer of `id` [`realize::needs_own_buffer`],
/// so `id` cannot leave `M` without breaking that consumer's launch.
pub(crate) fn at_structural_boundary(graph: &EGraph, realized: &Realized, id: Id) -> bool {
    realized
        .consumer_nodes
        .get(id)
        .map(|c| c.as_slice())
        .unwrap_or(&[])
        .iter()
        .any(|c| realize::needs_own_buffer(graph, id, *c))
}

/// True when `id` may not leave the materialized set: an `Effect::InPlace`
/// node, a root, or a leaf (which has no write to elide).
pub(crate) fn is_pinned(graph: &EGraph, roots: &[Id], id: Id) -> bool {
    if roots.contains(&id) {
        return true;
    }
    if realize::leaf_role(graph, id) != realize::LeafRole::NotLeaf {
        return true;
    }
    graph.semantics().effect(&graph.node(id).op) != Effect::Pure
}

/// Everything a schedule score depends on besides the point itself: the
/// epilogue signature, operand layouts and the consumer demand set. Two
/// occurrences of the same node in different surroundings therefore do not
/// share a memo entry.
pub(crate) fn context_hash(graph: &EGraph, realized: &Realized, node: Id) -> u64 {
    let mut h = FxHasher::default();
    let n = graph.node(node);

    match &n.op {
        Op::Launch(Launch::Contract { a, b, post, .. }) => {
            h.write_u64(a.pre.structural_hash());
            h.write_u64(b.pre.structural_hash());
            h.write_u64(post.structural_hash());
        }
        Op::Launch(Launch::Fold { carrier, post, .. }) => {
            for l in &carrier.lift {
                h.write_u64(l.structural_hash());
            }
            for m in &carrier.merge {
                h.write_u64(m.structural_hash());
            }
            for p in post {
                h.write_u64(p.structural_hash());
            }
        }
        Op::Launch(Launch::Map { body, .. }) => h.write_u64(body.structural_hash()),
        _ => h.write_u64(0),
    }

    for layout in operand_layouts(&n.op) {
        layout.hash(&mut h);
    }

    let mut demand: Vec<u32> = realized
        .consumer_nodes
        .get(node)
        .map(|c| c.as_slice())
        .unwrap_or(&[])
        .iter()
        .map(|consumer| graph.class_of(*consumer).0.0)
        .collect();
    demand.sort_unstable();
    demand.dedup();
    for c in demand {
        h.write_u32(c);
    }
    h.finish()
}

fn operand_layouts(op: &Op) -> SmallVec<[Layout; 4]> {
    let mut out: SmallVec<[Layout; 4]> = SmallVec::new();
    if let Op::Launch(l1) = op {
        match l1 {
            Launch::Map { ops, .. }
            | Launch::Fold { ops, .. }
            | Launch::Gather { ops, .. }
            | Launch::Scatter { ops, .. }
            | Launch::Ext { ops, .. } => out.extend(ops.iter().map(|o| o.layout.clone())),
            Launch::Contract { a, b, .. } => {
                out.extend(a.ops.iter().chain(b.ops.iter()).map(|o| o.layout.clone()))
            }
            Launch::Region { .. } => {}
        }
    }
    out
}

fn domain(graph: &EGraph, id: Id) -> Option<&ScheduleDomain> {
    match &graph.node(id).op {
        Op::Launch(l1) => l1.schedule(),
        _ => None,
    }
}
