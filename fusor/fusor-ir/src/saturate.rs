//! The saturation driver: a creation-order worklist over a `(RuleId, Id)`
//! bitset, bounded by [`SaturationBudget`]. On exhaustion only
//! [`RuleTag::StrictlyLowering`] rules run, so the plan degrades but stays valid.

use crate::device::Caps;
use crate::egraph::{EGraph, Id, Rule, RuleTag, Saturate, SaturationBudget, SaturationReport};
use crate::error::Result;
use crate::ir::{Level, Op, OpTag};
use crate::rules::RuleId;
use fixedbitset::FixedBitSet;
use smallvec::SmallVec;
use std::collections::VecDeque;
use web_time::Instant;

/// The shipped driver. Targets contribute rules, never a driver.
#[derive(Default, Debug, Clone, Copy)]
pub struct CoreSaturate;

/// The name `lib.rs` re-exports.
pub type Driver = CoreSaturate;

impl CoreSaturate {
    pub const fn new() -> Self {
        Self
    }
}

/// `by_head[tag as usize]`: positions into the caller's `rules` slice.
type HeadTable = [SmallVec<[RuleId; 8]>; OpTag::Union as usize + 1];

fn head_table(rules: &[Rule]) -> HeadTable {
    let mut table: HeadTable = std::array::from_fn(|_| SmallVec::new());
    for (i, r) in rules.iter().enumerate() {
        for &head in r.heads {
            table[head as usize].push(RuleId(i as u16));
        }
    }
    table
}

impl Saturate for CoreSaturate {
    fn saturate(
        &self,
        graph: &mut EGraph,
        caps: &Caps,
        rules: &[Rule],
        budget: SaturationBudget,
    ) -> Result<SaturationReport> {
        let start = Instant::now();
        let initial = graph.len();

        let by_head = head_table(rules);
        let mut fired_counts = vec![0u32; rules.len()];
        let mut truncated: Vec<Id> = Vec::new();
        let mut saturated = true;
        let mut rounds = 0u32;
        let mut applications = 0u32;

        // Creation order is topological. Only root-reachable nodes not
        // covered by an earlier bounded search are offered, so the arena's
        // old terms never grow alternatives without bound.
        let reachable = graph.reachable_from_roots();
        let mut work: VecDeque<Id> = reachable
            .ones()
            .map(|i| Id(i as u32))
            .filter(|id| !graph.is_offered(*id))
            .collect();
        let new_nodes = work.len();
        let max_nodes = (initial - new_nodes)
            .saturating_add((budget.node_slope as usize).saturating_mul(new_nodes))
            .saturating_add(budget.node_slack as usize);
        let max_applications = budget.max_applications.max(
            budget
                .application_slope
                .saturating_mul(new_nodes.min(u32::MAX as usize) as u32),
        );
        // One rule fires at most once per node; the stride is fixed per call.
        let stride = max_nodes.max(initial).saturating_add(4096).max(64);
        let mut fired = FixedBitSet::with_capacity(rules.len().saturating_mul(64));

        let mut next: Vec<Id> = Vec::new();

        'rounds: while rounds < budget.max_rounds && !work.is_empty() {
            rounds += 1;
            let mut fired_this_round = 0u32;
            while let Some(id) = work.pop_front() {
                if id.index() >= graph.len() {
                    continue;
                }
                let candidates = &by_head[graph.node(id).op.tag() as usize];
                if candidates.is_empty() {
                    continue;
                }
                let node = graph.node(id).clone();
                let facts = graph.facts_view(id, caps);
                for &rid in candidates.iter() {
                    if graph.len() >= max_nodes || applications >= max_applications {
                        saturated = false;
                        let class = graph.class_of(id).0;
                        if !truncated.contains(&class) {
                            truncated.push(class);
                        }
                        break 'rounds;
                    }
                    let bit = rid.0 as usize * stride + id.index();
                    if bit >= stride * rules.len() {
                        continue;
                    }
                    if fired.contains(bit) {
                        continue;
                    }
                    fired.grow_and_insert(bit);
                    let before = graph.len();
                    let mut builder = graph.builder(caps);
                    applications += 1;
                    let applied = (rules[rid.0 as usize].apply)(&mut builder, id, &node, &facts);
                    if applied.is_some() {
                        fired_counts[rid.0 as usize] += 1;
                        fired_this_round += 1;
                        for i in before..graph.len() {
                            next.push(Id(i as u32));
                        }
                    }
                }
            }
            work.extend(next.drain(..));
            if fired_this_round == 0 {
                break;
            }
        }
        if !work.is_empty() || !next.is_empty() {
            // The round budget ran out with work still queued.
            saturated = false;
        }

        // The degraded pass, when a budget was hit or a chain has no Launch
        // member. Lowering is idempotent by hash-consing, so it ignores the
        // fired set and the node ceiling.
        if !saturated || missing_l1(graph, &reachable) {
            applications +=
                lower_everything(graph, caps, rules, &by_head, &mut fired_counts, &reachable);
        }
        // Mark this region searched; unrelated nodes stay eligible.
        for i in reachable.ones().chain(initial..graph.len()) {
            graph.mark_offered(Id(i as u32));
        }

        let fired_report: Vec<(&'static str, u32)> = rules
            .iter()
            .zip(fired_counts.iter())
            .filter(|&(_, &c)| c > 0)
            .map(|(r, &c)| (r.name, c))
            .collect();

        Ok(SaturationReport {
            initial_nodes: initial,
            final_nodes: graph.len(),
            rounds,
            micros: start.elapsed().as_micros() as u64,
            applications,
            saturated,
            truncated,
            fired: fired_report,
        })
    }
}

/// Whether any non-leaf Logical value still has no Launch spelling, the
/// extractor's only contract with saturation.
fn missing_l1(graph: &EGraph, reachable: &FixedBitSet) -> bool {
    // Nodes minted during this pass are reachable by construction.
    let minted = reachable.len()..graph.len();
    reachable.ones().chain(minted).any(|i| {
        let id = Id(i as u32);
        let node = graph.node(id);
        if node.level != Level::Logical
            || matches!(node.op, Op::Logical(crate::ir::logical::Logical::Leaf(_)))
        {
            return false;
        }
        !graph
            .members(graph.class_of(id))
            .iter()
            .any(|&m| graph.level(m) == Level::Launch)
    })
}

fn lower_everything(
    graph: &mut EGraph,
    caps: &Caps,
    rules: &[Rule],
    by_head: &HeadTable,
    fired_counts: &mut [u32],
    reachable: &FixedBitSet,
) -> u32 {
    let mut applications = 0u32;
    // The reachable set, then every id minted past it as the pass runs.
    let bound = reachable.len();
    let mut pending: Vec<Id> = reachable.ones().map(|i| Id(i as u32)).collect();
    pending.reverse();
    let mut minted = bound;
    loop {
        let id = match pending.pop() {
            Some(id) => id,
            None if minted < graph.len() => {
                let id = Id(minted as u32);
                minted += 1;
                id
            }
            None => break,
        };
        if graph.node(id).level != Level::Logical {
            continue;
        }
        let candidates = &by_head[graph.node(id).op.tag() as usize];
        if candidates.is_empty() {
            continue;
        }
        let node = graph.node(id).clone();
        let facts = graph.facts_view(id, caps);
        for &rid in candidates.iter() {
            let rule = &rules[rid.0 as usize];
            if rule.tag != RuleTag::StrictlyLowering {
                continue;
            }
            let before = graph.len();
            let mut builder = graph.builder(caps);
            applications += 1;
            let applied = (rule.apply)(&mut builder, id, &node, &facts);
            // A memo hit is not a firing.
            if applied.is_some() && graph.len() > before {
                fired_counts[rid.0 as usize] += 1;
            }
        }
    }
    applications
}
