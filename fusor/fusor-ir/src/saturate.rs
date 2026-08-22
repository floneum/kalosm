//! The saturation driver: a worklist in creation order over a
//! `(RuleId, Id)` bitset, bounded by [`SaturationBudget`]. On exhaustion it
//! offers only [`RuleTag::StrictlyLowering`] rules, guaranteeing every chain
//! provably reaches a runnable Launch form — budget exhaustion yields a
//! degraded-but-valid plan, never a hard error. Truncation is never silent.

use crate::device::Caps;
use crate::egraph::{EGraph, Id, Rule, RuleTag, Saturate, SaturationBudget, SaturationReport};
use crate::error::Result;
use crate::ir::{Level, Op, OpTag};
use crate::rules::RuleId;
use fixedbitset::FixedBitSet;
use smallvec::SmallVec;
use std::collections::VecDeque;
use std::time::Instant;

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

/// Dense index of an [`OpTag`], for the O(1) head-dispatch table.
const TAG_COUNT: usize = 19;

const fn tag_index(tag: OpTag) -> usize {
    match tag {
        OpTag::Leaf => 0,
        OpTag::Map => 1,
        OpTag::Fold => 2,
        OpTag::Contract => 3,
        OpTag::Restride => 4,
        OpTag::Window => 5,
        OpTag::Gather => 6,
        OpTag::Scatter => 7,
        OpTag::Dequant => 8,
        OpTag::Project => 9,
        OpTag::LaunchMap => 10,
        OpTag::LaunchFold => 11,
        OpTag::LaunchContract => 12,
        OpTag::LaunchGather => 13,
        OpTag::LaunchScatter => 14,
        OpTag::LaunchRegion => 15,
        OpTag::Ext => 16,
        OpTag::Union => 17,
    }
}

/// `by_head[tag_index(rule.head)]` — built once per call, positions into the
/// `rules` slice, so `RuleId` is positional within whatever slice the caller
/// concatenated.
type HeadTable = [SmallVec<[RuleId; 8]>; TAG_COUNT];

fn head_table(rules: &[Rule]) -> HeadTable {
    let mut table: HeadTable = std::array::from_fn(|_| SmallVec::new());
    for (i, r) in rules.iter().enumerate() {
        table[tag_index(r.head)].push(RuleId(i as u16));
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
        let max_nodes = budget.node_slope as usize * initial + budget.node_slack as usize;

        let by_head = head_table(rules);
        let mut fired_counts = vec![0u32; rules.len()];
        let mut truncated: Vec<Id> = Vec::new();
        let mut saturated = true;
        let mut rounds = 0u32;
        let mut applications = 0u32;
        // Where a hard budget (nodes / applications) stopped the walk, if it
        // did. Everything *before* this creation index was offered.
        let mut budget_break: Option<usize> = None;

        // One rule fires at most once per node. The stride is fixed for the
        // whole call so a bit's index never moves; the set itself grows with
        // the graph.
        let stride = max_nodes.max(initial).saturating_add(4096).max(64);
        let mut fired = FixedBitSet::with_capacity(rules.len().saturating_mul(64));

        // Creation order is already a topological order: children are
        // strictly smaller ids. Nodes below the frontier were fully offered
        // in an earlier pass over this same graph; a rule's applicability
        // depends only on the node and its (immutable) child facts, and
        // re-offering rules that mint fresh ids re-expands an already-
        // saturated region without bound.
        let from = graph.saturation_frontier.min(initial);
        let mut work: VecDeque<Id> = (from..initial).map(|i| Id(i as u32)).collect();
        let mut next: Vec<Id> = Vec::new();

        'rounds: while rounds < budget.max_rounds && !work.is_empty() {
            rounds += 1;
            let mut fired_this_round = 0u32;
            while let Some(id) = work.pop_front() {
                if id.index() >= graph.len() {
                    continue;
                }
                let candidates = &by_head[tag_index(graph.node(id).op.tag())];
                if candidates.is_empty() {
                    continue;
                }
                let node = graph.node(id).clone();
                let facts = graph.facts_view(id, caps);
                for &rid in candidates.iter() {
                    if graph.len() >= max_nodes || applications >= budget.max_applications {
                        saturated = false;
                        budget_break = Some(id.index());
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

        // The degraded pass. Runs when a budget was hit, and unconditionally
        // as a final sweep whenever some chain has no Launch member. A
        // `StrictlyLowering` rule is idempotent by hash-consing, so
        // re-offering one is a memo hit; that is what lets this ignore the
        // fired set and the node ceiling entirely.
        if !saturated || missing_l1(graph) {
            applications += lower_everything(graph, caps, rules, &by_head, &mut fired_counts);
        }

        // Advance the frontier: nodes below it have been offered every rule.
        // A full drain or round exhaustion covers the whole graph; a hard
        // budget break covers exactly the prefix walked. Without this, every
        // resolve of a long-lived graph re-offers every historical node and
        // the id-minting rules re-mint their results each time.
        graph.saturation_frontier = budget_break
            .unwrap_or_else(|| graph.len())
            .max(graph.saturation_frontier);

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

/// Whether any non-leaf Logical value still has no Launch spelling. This is the
/// extractor's only contract with saturation, so it is checked rather than
/// assumed.
fn missing_l1(graph: &EGraph) -> bool {
    (0..graph.len()).any(|i| {
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
) -> u32 {
    let mut applications = 0u32;
    let mut i = 0usize;
    // New ids appear as the pass runs; walking to the current length keeps
    // the floor total without a second sweep.
    while i < graph.len() {
        let id = Id(i as u32);
        i += 1;
        if graph.node(id).level != Level::Logical {
            continue;
        }
        let candidates = &by_head[tag_index(graph.node(id).op.tag())];
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
            // Only a pass that actually grew the graph counts as a firing;
            // a memo hit on an already-lowered node is not news.
            if applied.is_some() && graph.len() > before {
                fired_counts[rid.0 as usize] += 1;
            }
        }
    }
    applications
}
