//! The saturation driver: a worklist in creation order over a `(RuleId, Id)`
//! bitset, bounded by [`SaturationBudget`]. On exhaustion it offers only
//! [`RuleTag::StrictlyLowering`] rules, so every chain reaches a runnable L1
//! form: budget exhaustion yields a degraded-but-valid plan, reported in
//! [`SaturationReport::truncated`], never an error.

use crate::device::Caps;
use crate::egraph::{
    EGraph, Id, Rule, RuleTag, Saturate, SaturationBudget, SaturationReport,
};
use crate::error::Result;
use crate::ir::{Level, Op, OpTag};
use crate::rules::RuleId;
use fixedbitset::FixedBitSet;
use smallvec::SmallVec;
use std::collections::VecDeque;
use std::time::Instant;

/// The shipped driver. Targets contribute rules, not a driver.
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
const TAG_COUNT: usize = 20;

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
        OpTag::KMap => 10,
        OpTag::KFold => 11,
        OpTag::KContract => 12,
        OpTag::KGather => 13,
        OpTag::KScatter => 14,
        OpTag::KRegion => 15,
        OpTag::KMerged => 16,
        OpTag::Ext => 17,
        OpTag::Union => 18,
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

        // One rule fires at most once per node. The stride is fixed for the
        // whole call so a bit's index never moves; the set itself grows with
        // the graph.
        let stride = max_nodes.max(initial).saturating_add(4096).max(64);
        let mut fired = FixedBitSet::with_capacity(rules.len().saturating_mul(64));

        // Creation order is already a topological order, because children are
        // strictly smaller ids.
        let mut work: VecDeque<Id> = (0..initial).map(|i| Id(i as u32)).collect();
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
                // A refcount bump, not a copy: the arena entry is immutable,
                // so the pin stays valid across the `&mut Builder` below.
                let node = graph.node_arc(id);
                let facts = graph.facts_view(id, caps);
                for &rid in candidates.iter() {
                    if graph.len() >= max_nodes || applications >= budget.max_applications {
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

        // The degraded pass runs when a budget was hit and as a final sweep
        // whenever some chain has no L1 member. A `StrictlyLowering` rule is
        // idempotent by hash-consing, so re-offering one is a memo hit and this
        // pass ignores the fired set and the node ceiling.
        if !saturated || missing_l1(graph) {
            applications += lower_everything(graph, caps, rules, &by_head, &mut fired_counts);
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

/// Whether any non-leaf L0 value has no L1 spelling — the extractor's only
/// contract with saturation.
fn missing_l1(graph: &EGraph) -> bool {
    // Two linear sweeps over a bitset keyed by class root, instead of
    // enumerating each node's whole class: first mark every class holding an
    // L1 member, then look for an L0 value whose class was never marked.
    let mut has_l1 = FixedBitSet::with_capacity(graph.len());
    for i in 0..graph.len() {
        let id = Id(i as u32);
        let node = graph.node(id);
        if node.level == Level::L1 && !matches!(node.op, Op::Union(..)) {
            has_l1.insert(graph.root_of(id).index());
        }
    }
    (0..graph.len()).any(|i| {
        let id = Id(i as u32);
        let node = graph.node(id);
        if node.level != Level::L0 || matches!(node.op, Op::L0(crate::ir::level0::L0::Leaf(_))) {
            return false;
        }
        !has_l1.contains(graph.root_of(id).index())
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
        if graph.node(id).level != Level::L0 {
            continue;
        }
        let candidates = &by_head[tag_index(graph.node(id).op.tag())];
        if candidates.is_empty() {
            continue;
        }
        let node = graph.node_arc(id);
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
            // Only a pass that grew the graph counts as a firing; a memo hit on
            // an already-lowered node does not.
            if applied.is_some() && graph.len() > before {
                fired_counts[rid.0 as usize] += 1;
            }
        }
    }
    applications
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::Dtype;
    use crate::ir::level0::L0;
    use crate::ir::level1::{AccessPlan, L1};
    use crate::rules::test_support as ts;
    use crate::rules::{CORE_RULES, alias_operand_of, ident_expr};
    use crate::scalar::{BinOp, ScalarExpr, UnOp};
    use crate::shape::Dim;
    use rustc_hash::FxHashSet;

    /// A small forward chain: buffer -> map -> fold, plus a matmul-shaped
    /// product fold.
    fn small_graph() -> (EGraph, Vec<Id>) {
        let mut g = ts::graph();
        let shape = [Dim::Const(8), Dim::Const(16)];
        let x = ts::buffer(&mut g, Dtype::F32, &shape);
        let y = ts::buffer(&mut g, Dtype::F32, &shape);
        let prod = ts::map(
            &mut g,
            ScalarExpr::bin(
                BinOp::Mul,
                ScalarExpr::arg(0, Dtype::F32),
                ScalarExpr::arg(1, Dtype::F32),
            ),
            &[x, y],
        );
        let f = ts::fold(&mut g, ts::binop_carrier(BinOp::Add, Dtype::F32), 1, Dtype::F32, prod);
        g.add_root(f);
        (g, vec![prod, f])
    }

    /// The shipped budget. Every term in it is a count, not a clock.
    fn untimed() -> SaturationBudget {
        SaturationBudget::default()
    }

    /// A fingerprint of a node's whole subterm, with ids erased.
    ///
    /// Raw `(op, children)` keys cannot be compared across two rule orders: a
    /// rewrite that names a freshly minted node (`KRegion::members`,
    /// `KMerged::segments`) records the id it was given, and ids are a
    /// creation-order artifact.
    fn fingerprints(g: &EGraph) -> FxHashSet<u64> {
        use std::hash::{Hash, Hasher};
        let mut fp: Vec<u64> = Vec::with_capacity(g.len());
        let mut out = FxHashSet::default();
        for i in 0..g.len() {
            let id = Id(i as u32);
            let node = g.node(id);
            let mut h = rustc_hash::FxHasher::default();
            scrub_ids(&format!("{:?}", node.op)).hash(&mut h);
            for c in &node.children {
                fp[c.index()].hash(&mut h);
            }
            let v = h.finish();
            fp.push(v);
            if !matches!(node.op, Op::Union(..)) {
                out.insert(v);
            }
        }
        out
    }

    /// Replace every `Id(<digits>)` with `Id(_)`.
    fn scrub_ids(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        let bytes = s.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if s[i..].starts_with("Id(") {
                let rest = &s[i + 3..];
                let digits = rest.chars().take_while(char::is_ascii_digit).count();
                if digits > 0 && rest[digits..].starts_with(')') {
                    out.push_str("Id(_)");
                    i += 3 + digits + 1;
                    continue;
                }
            }
            out.push(bytes[i] as char);
            i += 1;
        }
        out
    }

    /// The fixed rule order carries no semantics.
    #[test]
    fn rule_order_is_semantically_inert() {
        let caps = ts::caps();
        let (mut a, _) = small_graph();
        let ra = CoreSaturate
            .saturate(&mut a, &caps, CORE_RULES, untimed())
            .unwrap();

        let reversed: Vec<Rule> = CORE_RULES.iter().rev().copied().collect();
        let (mut b, _) = small_graph();
        let rb = CoreSaturate
            .saturate(&mut b, &caps, &reversed, untimed())
            .unwrap();

        assert_eq!(ra.final_nodes, rb.final_nodes);
        assert_eq!(fingerprints(&a), fingerprints(&b));
        assert!(ra.saturated && rb.saturated);
        // The same rules fired the same number of times, in either order.
        let mut fa: Vec<_> = ra.fired.clone();
        let mut fb: Vec<_> = rb.fired.clone();
        fa.sort_unstable();
        fb.sort_unstable();
        assert_eq!(fa, fb);
    }

    /// Every budget path returns `Ok`, reports the truncation, and leaves every
    /// original L0 root with an L1 spelling.
    #[test]
    fn budget_exhaustion_degrades_not_errors() {
        let caps = ts::caps();
        let (mut g, roots) = small_graph();
        let initial = g.len();
        let report = CoreSaturate
            .saturate(
                &mut g,
                &caps,
                CORE_RULES,
                SaturationBudget {
                    node_slope: 0,
                    node_slack: 4,
                    max_rounds: 1,
                    max_applications: 0,
                },
            )
            .unwrap();
        assert!(!report.saturated);
        assert!(!report.truncated.is_empty());
        assert_eq!(report.initial_nodes, initial);
        for root in roots {
            let members = g.chain(root);
            assert!(
                members.iter().any(|&m| g.level(m) == Level::L1),
                "chain of {root} has no L1 member"
            );
        }
    }

    /// Saturation is a pure function of `(graph, caps, rules, budget)`. Every
    /// budget term is a count, so two runs reporting different `micros` report
    /// the same of everything else.
    #[test]
    fn saturation_is_deterministic_under_any_wall_time() {
        let caps = ts::caps();
        let run = |budget: SaturationBudget| {
            let (mut g, _) = small_graph();
            let r = CoreSaturate
                .saturate(&mut g, &caps, CORE_RULES, budget)
                .unwrap();
            (r, fingerprints(&g))
        };
        let (first, fp) = run(SaturationBudget::default());
        for _ in 0..8 {
            let (again, fp_again) = run(SaturationBudget::default());
            assert_eq!(again.final_nodes, first.final_nodes);
            assert_eq!(again.applications, first.applications);
            assert_eq!(again.rounds, first.rounds);
            assert_eq!(again.saturated, first.saturated);
            assert_eq!(again.truncated, first.truncated);
            assert_eq!(fp_again, fp);
        }

        // A budget that stops the sweep part way stops it at the same place
        // every time.
        let tight = SaturationBudget {
            max_applications: first.applications / 2,
            ..SaturationBudget::default()
        };
        let (cut, cut_fp) = run(tight);
        assert!(!cut.saturated, "the tight budget has to bind: {cut:?}");
        assert!(cut.final_nodes < first.final_nodes);
        for _ in 0..8 {
            let (again, again_fp) = run(tight);
            assert_eq!(again.final_nodes, cut.final_nodes);
            assert_eq!(again.applications, cut.applications);
            assert_eq!(again.truncated, cut.truncated);
            assert_eq!(again_fp, cut_fp);
        }
    }

    /// Acyclicity does not hold structurally. `form_kregion` mints
    /// `KRegion { members: [producer, fused] }` acyclically and `map_into_fold`
    /// then unions that hash-consed `fused` into the class, retroactively
    /// making the region name its own class. The invariant is enforced at
    /// selection: `fusor2_cost::realize::selectable` drops a self-referential
    /// member. `KRegion` is the only op allowed to name its own class.
    #[test]
    fn the_only_self_referential_members_left_are_regions() {
        let caps = ts::caps();
        let (mut g, _) = small_graph();
        CoreSaturate
            .saturate(&mut g, &caps, CORE_RULES, untimed())
            .unwrap();
        for i in 0..g.len() {
            let id = Id(i as u32);
            // A `Union` names both its operands' classes by construction and
            // is never a selectable member.
            if matches!(g.node(id).op, Op::Union(..)) {
                continue;
            }
            let class = g.class_of(id);
            let self_ref = g
                .node(id)
                .children
                .iter()
                .any(|c| g.class_of(*c) == class);
            if self_ref {
                assert!(
                    matches!(g.node(id).op, Op::L1(L1::KRegion { .. })),
                    "{id} ({:?}) names its own class; only KRegion may",
                    g.node(id).op.tag()
                );
            }
        }
        // No `KMerged` of one survives.
        for i in 0..g.len() {
            if let Op::L1(L1::KMerged(w)) = &g.node(Id(i as u32)).op {
                assert!(w.segments().len() >= 2, "a wave of one at %{i}");
            }
        }
    }

    /// Hash-consing shares isomorphic layers outright.
    #[test]
    fn hash_consing_shares_isomorphic_layers() {
        fn layer(g: &mut EGraph, input: Id, width: usize) -> Id {
            let shape = [Dim::Const(8), Dim::Const(16)];
            let mut cur = input;
            for _ in 0..width {
                cur = ts::map(
                    g,
                    ScalarExpr::un(UnOp::Tanh, ScalarExpr::arg(0, Dtype::F32)),
                    &[cur],
                );
            }
            let _ = shape;
            cur
        }

        let mut g = ts::graph();
        let x = ts::buffer(&mut g, Dtype::F32, &[Dim::Const(8), Dim::Const(16)]);
        let first = layer(&mut g, x, 40);
        let after_first = g.len();
        // A second, structurally identical layer over the same input adds
        // nothing: every subterm hash-conses.
        let second = layer(&mut g, x, 40);
        assert_eq!(second, first);
        assert_eq!(g.len(), after_first);

        // Saturating the shared graph costs what one layer costs.
        let caps = ts::caps();
        let r = CoreSaturate
            .saturate(&mut g, &caps, CORE_RULES, untimed())
            .unwrap();
        assert!(r.saturated);
        let one_layer_nodes = r.final_nodes;

        // A distinct second layer (different input) adds its own nodes.
        let mut h = ts::graph();
        let hx = ts::buffer(&mut h, Dtype::F32, &[Dim::Const(8), Dim::Const(16)]);
        let hy = ts::buffer(&mut h, Dtype::F32, &[Dim::Const(8), Dim::Const(16)]);
        layer(&mut h, hx, 40);
        layer(&mut h, hy, 40);
        let r2 = CoreSaturate
            .saturate(&mut h, &caps, CORE_RULES, untimed())
            .unwrap();
        assert!(r2.saturated);
        assert!(r2.final_nodes > one_layer_nodes);
    }

    /// A synthetic forward+backward graph of trainer size stays inside the
    /// shipped budget.
    #[test]
    fn saturation_stays_in_budget_on_a_trainer_sized_graph() {
        let mut g = ts::graph();
        let shape = [Dim::Const(128), Dim::Const(64)];
        let mut layers: Vec<Id> = Vec::new();
        let mut cur = ts::buffer(&mut g, Dtype::F32, &shape);
        // An elementwise stack, then the same again standing in for the
        // adjoint.
        for step in 0..950u32 {
            cur = ts::map(
                &mut g,
                ScalarExpr::bin(
                    BinOp::Add,
                    ScalarExpr::un(UnOp::Tanh, ScalarExpr::arg(0, Dtype::F32)),
                    ScalarExpr::lit(crate::dtype::Splat::F32(step as f32)),
                ),
                &[cur],
            );
            layers.push(cur);
        }
        let mut back = ts::buffer(&mut g, Dtype::F32, &shape);
        for step in 0..950u32 {
            back = ts::map(
                &mut g,
                ScalarExpr::bin(
                    BinOp::Mul,
                    ScalarExpr::un(UnOp::Exp, ScalarExpr::arg(0, Dtype::F32)),
                    ScalarExpr::arg(1, Dtype::F32),
                ),
                &[back, layers[step as usize]],
            );
        }
        g.add_root(back);
        let initial = g.len();
        assert!((1_800..2_100).contains(&initial), "{initial}");

        let caps = ts::caps();
        let report = CoreSaturate
            .saturate(&mut g, &caps, CORE_RULES, untimed())
            .unwrap();
        assert!(report.saturated, "{report:?}");
        assert!(
            report.final_nodes <= 8 * 1900 + 4096,
            "{}",
            report.final_nodes
        );
        assert!(report.rounds <= 6, "{}", report.rounds);
    }

    #[test]
    fn head_dispatch_reaches_every_tag_a_core_rule_names() {
        let table = head_table(CORE_RULES);
        let total: usize = table.iter().map(|v| v.len()).sum();
        assert_eq!(total, CORE_RULES.len());
        assert!(table[tag_index(OpTag::Union)].is_empty());
        assert!(table[tag_index(OpTag::Leaf)].is_empty());
    }

    #[test]
    fn saturation_lowers_a_map_fold_chain_end_to_end() {
        let caps = ts::caps();
        let (mut g, roots) = small_graph();
        let report = CoreSaturate
            .saturate(&mut g, &caps, CORE_RULES, untimed())
            .unwrap();
        assert!(report.saturated);
        assert!(report.truncated.is_empty());
        assert!(report.fired.iter().any(|(n, _)| *n == "LOWER_MAP"));
        assert!(report.fired.iter().any(|(n, _)| *n == "LOWER_FOLD"));
        assert!(report.fired.iter().any(|(n, _)| *n == "RECOGNIZE_CONTRACT"));
        // The fold's class holds the composed fold, the recognized
        // contraction, and at least one nest spelling of each.
        let members = g.chain(roots[1]);
        assert!(
            members
                .iter()
                .any(|&m| matches!(g.node(m).op, Op::L0(L0::Contract { .. })))
        );
        assert!(
            members
                .iter()
                .any(|&m| matches!(g.node(m).op, Op::L1(L1::KFold { .. })))
        );
        // The fused nest reads the two buffers directly.
        let fused = members
            .iter()
            .copied()
            .find(|&m| match &g.node(m).op {
                Op::L1(L1::KFold { ops, .. }) => {
                    ops.len() == 2 && ops.iter().all(|o| matches!(o.access, AccessPlan::Alias))
                }
                _ => false,
            })
            .expect("map_into_fold should have inlined the product");
        let _ = fused;
        let _ = alias_operand_of;
        let _ = ident_expr;
    }
}
