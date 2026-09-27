//! Reversible selection and schedule moves. Schedule estimates only order
//! candidates; exact realized cost decides whether to keep them.

use crate::nodes::{composite_members, domain_of};
use crate::realize;
use fusor_ir::cost::{CostModel, Picoseconds};
use fusor_ir::egraph::{ClassId, EGraph, Id};
use fusor_ir::extract::{Extraction, Move};
use fusor_ir::ir::launch::{SchedPoint, ScheduleDomain};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

/// One concrete state change a [`Move`] can produce.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Candidate {
    Select { class: ClassId, node: Id },
    Schedule { node: Id, theta: SchedPoint },
}

/// Trial edits and their construction obligations share one rollback log.
#[derive(Default)]
pub(crate) struct Trail(SmallVec<[Change; 16]>);

enum Change {
    Buffers(fixedbitset::FixedBitSet),
    Select(ClassId, Option<Id>),
    Schedule(Id, Option<SchedPoint>),
}

impl Trail {
    pub(crate) fn mark(&self) -> usize {
        self.0.len()
    }

    pub(crate) fn selected_since(&self, mark: usize) -> bool {
        self.0[mark..]
            .iter()
            .any(|c| matches!(c, Change::Select(..)))
    }

    pub(crate) fn buffers(&mut self, ex: &mut Extraction, buffers: fixedbitset::FixedBitSet) {
        self.0
            .push(Change::Buffers(std::mem::replace(&mut ex.m, buffers)));
    }

    pub(crate) fn select(&mut self, ex: &mut Extraction, class: ClassId, node: Id) {
        let was = ex.sigma.insert(class, node);
        if was != Some(node) {
            crate::debug::sigma(class, node, "select");
            self.0.push(Change::Select(class, was));
        }
    }

    pub(crate) fn schedule(&mut self, ex: &mut Extraction, node: Id, theta: SchedPoint) {
        let was = ex.theta.insert(node, theta);
        if was != Some(theta) {
            self.0.push(Change::Schedule(node, was));
        }
    }

    pub(crate) fn rollback(&mut self, ex: &mut Extraction, mark: usize) {
        for change in self.0.drain(mark..).rev() {
            match change {
                Change::Buffers(was) => ex.m = was,
                Change::Select(class, Some(was)) => {
                    ex.sigma.insert(class, was);
                }
                Change::Select(class, None) => {
                    ex.sigma.remove(&class);
                }
                Change::Schedule(node, Some(was)) => {
                    ex.theta.insert(node, was);
                }
                Change::Schedule(node, None) => {
                    ex.theta.remove(&node);
                }
            }
        }
    }

    pub(crate) fn apply(&mut self, ex: &mut Extraction, c: Candidate) -> bool {
        let mark = self.mark();
        match c {
            Candidate::Select { class, node } => {
                let Some(was) = ex.selected(class) else {
                    return false;
                };
                if was == node {
                    return false;
                }
                self.select(ex, class, node);
            }
            Candidate::Schedule { node, theta } => self.schedule(ex, node, theta),
        }
        self.mark() != mark
    }
}

/// Schedule ordering for one search with a fixed graph and cost model.
#[derive(Default)]
pub(crate) struct SchedCache {
    order: FxHashMap<Id, Vec<SchedPoint>>,
}

impl SchedCache {
    /// Every point of `id`'s domain, cheapest `node_math` first.
    pub(crate) fn ordered(
        &mut self,
        graph: &EGraph,
        id: Id,
        cost: &dyn CostModel,
    ) -> &[SchedPoint] {
        self.order
            .entry(id)
            .or_insert_with(|| match domain_of(graph, id) {
                None | Some(ScheduleDomain::Point) => Vec::new(),
                Some(domain) => {
                    let (ins, out) = crate::lower_bound::node_facts(graph, id);
                    crate::lower_bound::ranked_points(graph.node(id), &ins, out, domain, cost)
                }
            })
    }
}

/// Every move worth offering, classes then nodes ascending.
pub(crate) fn frontier(graph: &EGraph, selected: &[Id]) -> Vec<Move> {
    let mut out = Vec::new();
    let mut classes: Vec<_> = selected.iter().map(|id| graph.class_of(*id)).collect();
    classes.sort_unstable();
    classes.dedup();
    for class in classes {
        if !realize::is_singleton(graph, class) {
            out.push(Move::Reselect(class));
        }
    }
    let mut selected = selected.to_vec();
    selected.sort_unstable();
    selected.dedup();
    for id in selected {
        if let Some(d) = domain_of(graph, id)
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
    selected: &[Id],
    mv: Move,
    lb: &[Picoseconds],
    cache: &mut SchedCache,
    cost: &dyn CostModel,
) -> SmallVec<[Candidate; 8]> {
    let mut out: SmallVec<[Candidate; 8]> = SmallVec::new();
    match mv {
        Move::Reselect(class) => {
            // A member of a selected slab is that slab's to select.
            if !selected.iter().any(|id| graph.class_of(*id) == class)
                || slab_pinned(graph, selected, class)
            {
                return out;
            }
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
        Move::Reschedule(node) => {
            if !selected.contains(&node) {
                return out;
            }
            let current = extraction.theta.get(&node).copied();
            for theta in cache.ordered(graph, node, cost) {
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

/// Whether `class` is a middle member's class of a live composite.
fn slab_pinned(graph: &EGraph, selected: &[Id], class: ClassId) -> bool {
    selected.iter().any(|sel| {
        composite_members(graph, *sel).is_some_and(|members| {
            members[..members.len().saturating_sub(1)]
                .iter()
                .any(|m| graph.class_of(*m) == class)
        })
    })
}
