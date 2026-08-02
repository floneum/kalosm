//! The three local-search moves and their incremental deltas.
//!
//! Each move's delta is recomputed over only the affected launches, via a
//! union-find over the realized cut. **The accept test is always the exact
//! global cost** — the schedule score orders the `RESCHEDULE` frontier, it
//! never gates candidates, so the full ~8,300-point domain stays reachable.
//!
//! `FLIP` is refused when the node is pinned: an `Effect::InPlace` node is
//! pinned in `M`, because inlining an atomic scatter into two consumers
//! doubles the embedding gradient. Purity is a *precondition* of the
//! materialization move, not an afterthought.
//!
//! Owned by W7.

use crate::realize::{self, Realized};
use fusor2_ir::Result;
use fusor2_ir::cost::{CostModel, Picoseconds};
use fusor2_ir::egraph::{ClassId, EGraph, Id};
use fusor2_ir::extract::{ExtractBudget, Extraction, Move};
use fusor2_ir::facts::ValueFacts;
use fusor2_ir::ir::Op;
use fusor2_ir::ir::level1::{Effect, L1, SchedPoint, ScheduleDomain};
use fusor2_ir::ir::level2::ArenaPlanner;
use fusor2_ir::shape::Layout;
use rustc_hash::{FxHashMap, FxHasher};
use smallvec::SmallVec;
use std::hash::{Hash, Hasher};

/// One concrete state change a [`Move`] can produce. A `Move` names the
/// dimension; a `Candidate` names the value.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Candidate {
    Select { class: ClassId, node: Id },
    Materialize { node: Id, on: bool },
    Schedule { node: Id, theta: SchedPoint },
}

/// Enough state to revert one move exactly.
#[derive(Clone, Debug)]
pub enum Undo {
    /// `node` and `node_was_materialized` are the *new* member's own prior
    /// state: a reselect carries `M` across with the selection, so reverting
    /// has to put the new member's bit back as well as the old selection.
    Reselect {
        class: ClassId,
        was: Id,
        node: Id,
        node_was_materialized: bool,
    },
    Flip { node: Id, was_materialized: bool },
    Reschedule { node: Id, was: Option<SchedPoint> },
}

/// Memo for the `RESCHEDULE` frontier: the per-point score and the sorted
/// order, both keyed on `(node, context_hash)`. This is what makes an
/// ~8,300-point `CoopDomain` affordable — without it every sweep would
/// rescore the whole domain of every contraction.
#[derive(Default)]
pub struct SchedCache {
    order: FxHashMap<(Id, u64), Vec<SchedPoint>>,
    score: FxHashMap<(Id, SchedPoint, u64), Picoseconds>,
}

impl SchedCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.score.len()
    }

    pub fn is_empty(&self) -> bool {
        self.score.is_empty()
    }

    /// Points of `id`'s domain, cheapest `node_math` first. The full domain
    /// is always returned; ordering never gates.
    pub fn ordered(
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
                Op::L1(l1) => l1.schedule(),
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

    pub fn score_of(&self, id: Id, theta: SchedPoint, context: u64) -> Option<Picoseconds> {
        self.score.get(&(id, theta, context)).copied()
    }
}

/// Every move worth offering at this state, in a deterministic order:
/// classes ascending, then nodes ascending.
pub fn frontier(
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
pub fn candidates(
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
            // Only runnable members: proposing the un-lowered `L0` node would
            // be a move the objective cannot distinguish and the verifier
            // rejects outright.
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
            // Only *leaving* `M` needs a guard. A node cut from a consumer by
            // structure has to land in a buffer whatever it costs, or the
            // consumer's launch reads a value nothing ever wrote.
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
                // A point whose footprint is over the device cap is
                // unselectable, not merely slow: §4.2 makes a lowering refusal
                // a hard assert, so offering one lets the climb move onto a
                // state that mints a crash. `has_legal_point` gates the node
                // and passes as soon as *one* point fits, so the per-point
                // test has to be here as well.
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
pub fn apply(graph: &EGraph, extraction: &mut Extraction, c: Candidate) -> Option<Undo> {
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
            // whichever member computes it. Without this, reselecting a root's
            // class silently un-materializes the root and nothing lands
            // anywhere.
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

pub fn undo(extraction: &mut Extraction, undo: Undo) {
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
pub fn at_structural_boundary(graph: &EGraph, realized: &Realized, id: Id) -> bool {
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
pub fn is_pinned(graph: &EGraph, roots: &[Id], id: Id) -> bool {
    if roots.contains(&id) {
        return true;
    }
    if realize::leaf_role(graph, id) != realize::LeafRole::NotLeaf {
        return true;
    }
    graph.semantics().effect(&graph.node(id).op) != Effect::Pure
}

/// Exact realized cost of the current state.
pub fn evaluate(
    graph: &EGraph,
    extraction: &Extraction,
    roots: &[Id],
    cost: &dyn CostModel,
    arena: &dyn ArenaPlanner,
) -> Result<Picoseconds> {
    let realized = realize::realize(graph, roots, extraction, cost, arena)?;
    Ok(realize::exact_cost(&realized, extraction, cost))
}

/// The launches a change at `node` can move: the launch holding it plus the
/// launch of every realized consumer. Ascending, deduplicated.
pub fn affected_launches(realized: &Realized, node: Id) -> SmallVec<[u32; 4]> {
    let mut out: SmallVec<[u32; 4]> = SmallVec::new();
    let mut push = |l: u32| {
        if !out.contains(&l) {
            out.push(l);
        }
    };
    if let Some(l) = realized.launch_of.get(node) {
        push(*l);
    }
    for consumer in realized
        .consumer_nodes
        .get(node)
        .map(|c| c.as_slice())
        .unwrap_or(&[])
    {
        if let Some(l) = realized.launch_of.get(*consumer) {
            push(*l);
        }
    }
    out.sort_unstable();
    out
}

/// Everything a schedule score depends on besides the point itself: merged
/// segment count, epilogue signature, operand layouts and the consumer
/// demand set. Two occurrences of the same node in different surroundings
/// therefore do not share a memo entry.
pub fn context_hash(graph: &EGraph, realized: &Realized, node: Id) -> u64 {
    let mut h = FxHasher::default();
    let n = graph.node(node);

    let segments = match &n.op {
        Op::L1(L1::KMerged(m)) => m.segments().len() as u64,
        _ => 0,
    };
    h.write_u64(segments);

    match &n.op {
        Op::L1(L1::KContract {
            pre_a, pre_b, post, ..
        }) => {
            h.write_u64(pre_a.structural_hash());
            h.write_u64(pre_b.structural_hash());
            h.write_u64(post.structural_hash());
        }
        Op::L1(L1::KQContract { post, .. }) => h.write_u64(post.structural_hash()),
        Op::L1(L1::KFold { carrier, post, .. }) => {
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
        Op::L1(L1::KMap { body, .. }) => h.write_u64(body.structural_hash()),
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
    if let Op::L1(l1) = op {
        match l1 {
            L1::KMap { ops, .. }
            | L1::KFold { ops, .. }
            | L1::KGather { ops, .. }
            | L1::KScatter { ops, .. }
            | L1::Ext { ops, .. } => out.extend(ops.iter().map(|o| o.layout.clone())),
            L1::KContract { a, b, .. } | L1::KQContract { a, b, .. } => {
                out.push(a.layout.clone());
                out.push(b.layout.clone());
            }
            L1::KRegion { .. } | L1::KMerged(_) => {}
        }
    }
    out
}

fn domain(graph: &EGraph, id: Id) -> Option<&ScheduleDomain> {
    match &graph.node(id).op {
        Op::L1(l1) => l1.schedule(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::realize::testkit::{
        N, TestCost, TestPlanner, buffer, chain_graph, fork_graph, kscatter, new_graph, seed_for,
    };

    #[test]
    fn flip_is_refused_on_an_inplace_node() {
        let mut g = new_graph();
        let shape = [N];
        let base = buffer(&mut g, 0, &shape);
        let idx = buffer(&mut g, 1, &shape);
        let upd = buffer(&mut g, 2, &shape);
        let sc = kscatter(&mut g, base, idx, upd, &shape);
        let a = crate::realize::testkit::kmap(&mut g, sc, &shape, 1);
        let b = crate::realize::testkit::kmap(&mut g, sc, &shape, 2);
        g.add_root(a);
        g.add_root(b);
        let roots = g.roots().to_vec();
        let ex = seed_for(&g, &roots);
        let cost = TestCost::default();
        let arena = TestPlanner;
        let realized = crate::realize::realize(&g, &roots, &ex, &cost, &arena).unwrap();

        assert!(is_pinned(&g, &realized.roots, sc));
        let mut cache = SchedCache::new();
        let lb = crate::lower_bound::lower_bound(&g, &cost);
        let c = candidates(&g, &ex, &realized, Move::Flip(sc), &lb, &mut cache, &cost);
        assert!(c.is_empty(), "an atomic scatter may not leave M");
        assert!(ex.is_materialized(sc));
    }

    #[test]
    fn flip_round_trips() {
        let (g, roots) = chain_graph(3);
        let mut ex = seed_for(&g, &roots);
        // The middle map is inlinable.
        let target = *ex
            .sigma
            .values()
            .filter(|id| !roots.contains(id) && !is_pinned(&g, &roots, **id))
            .min()
            .unwrap();
        let before = ex.is_materialized(target);
        let u = apply(
            &g,
            &mut ex,
            Candidate::Materialize {
                node: target,
                on: !before,
            },
        )
        .unwrap();
        assert_eq!(ex.is_materialized(target), !before);
        undo(&mut ex, u);
        assert_eq!(ex.is_materialized(target), before);
    }

    /// `M` is keyed by node but the decision belongs to the class.
    ///
    /// The regression: `RESELECT` swapped `sigma` and left `M` alone, so the
    /// incoming member arrived unmaterialized. On a root's class that means
    /// nothing lands in a buffer, `verify_plan`'s clause 6 rejects the
    /// winner, and step 4 of extraction — the whole local search — turns
    /// every valid plan it touches into an error.
    #[test]
    fn reselect_carries_the_materialized_bit_across_the_swap() {
        let (g, roots, cheap, dear, class) = crate::realize::testkit::seeded_graph();
        let mut ex = seed_for(&g, &roots);
        assert_eq!(ex.selected(class), Some(cheap));
        assert!(ex.is_materialized(cheap), "a root lands in a buffer");
        assert!(!ex.is_materialized(dear));

        let u = apply(&g, &mut ex, Candidate::Select { class, node: dear }).unwrap();
        assert_eq!(ex.selected(class), Some(dear));
        assert!(
            ex.is_materialized(dear),
            "the incoming member inherits the obligation"
        );

        undo(&mut ex, u);
        assert_eq!(ex.selected(class), Some(cheap));
        assert!(ex.is_materialized(cheap));
        assert!(!ex.is_materialized(dear), "undo restores the incoming bit");
    }

    /// Every state one `RESELECT` can reach still derives a plan that
    /// verifies. This is the assertion the desync above actually broke.
    #[test]
    fn every_reselect_state_still_verifies() {
        let (g, roots, _cheap, _dear, class) = crate::realize::testkit::seeded_graph();
        let cost = TestCost::default();
        let arena = TestPlanner;
        let ex0 = seed_for(&g, &roots);
        let realized = crate::realize::realize(&g, &roots, &ex0, &cost, &arena).unwrap();
        let lb = crate::lower_bound::lower_bound(&g, &cost);
        let mut cache = SchedCache::new();
        let options = candidates(
            &g,
            &ex0,
            &realized,
            Move::Reselect(class),
            &lb,
            &mut cache,
            &cost,
        );
        assert!(!options.is_empty(), "a two-member class offers a swap");
        for c in options {
            let mut ex = ex0.clone();
            let Some(_) = apply(&g, &mut ex, c) else {
                continue;
            };
            let r = crate::realize::realize(&g, &roots, &ex, &cost, &arena).unwrap();
            let plan = crate::plan::derive_plan(
                &g,
                &ex,
                &r,
                cost.facts(),
                crate::realize::exact_cost(&r, &ex, &cost),
            )
            .unwrap();
            crate::verify_plan::verify_plan(&g, &plan)
                .unwrap_or_else(|e| panic!("{c:?} produced an unverifiable plan: {e}"));
        }
    }

    #[test]
    fn affected_set_is_the_node_plus_its_consumers() {
        let (g, roots, shared) = fork_graph();
        let ex = seed_for(&g, &roots);
        let cost = TestCost::default();
        let arena = TestPlanner;
        let r = crate::realize::realize(&g, &roots, &ex, &cost, &arena).unwrap();
        let a = affected_launches(&r, shared);
        assert!(!a.is_empty());
        assert!(a.windows(2).all(|w| w[0] < w[1]), "ascending and unique");
    }
}
