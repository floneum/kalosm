//! Delta application: write extraction's non-identity selections back onto
//! the execution graph.
//!
//! Reuses `commit_recognized` — the destructive optimizer's own commit
//! surgery — so the two invariants every rewrite maintains hold here by
//! construction: execution-graph edges match the new payload's `inputs`, and
//! `add_physical_dependencies` records persistent inner-graph edges (firing
//! the flush-replay recording hook identically). Killed producers fall out
//! through `remove_node_if_dead`, mirroring extraction's kill cascade.

use egg::Language;

use super::super::Resolver;
use super::EGraphDriver;
use super::extract::Extraction;
use super::interner::{rebind_variant_dependencies, variant_dependencies};
use crate::compute_graph::ComputeGraphInner;

impl Resolver {
    pub(super) fn apply_egraph_deltas(
        &mut self,
        graph: &mut ComputeGraphInner,
        driver: &EGraphDriver,
        extraction: &Extraction,
    ) -> usize {
        let mut applied = 0;
        for (prov, enode) in extraction.deltas() {
            let facts = driver.egraph.analysis.facts_of(prov);
            let exec_idx = facts
                .exec
                .expect("deltas only select alternatives for execution nodes");
            // An earlier delta's commit can kill this delta's target: when
            // one recognized cluster's root is another cluster's
            // intermediate (semantic identity lets both carry deltas), the
            // outer commit rewires past the inner root and its kill cascade
            // removes it. The removed node is unconsumed and not a target,
            // so its rewrite is vacuous — both application orders converge
            // to the same final graph.
            if !self.execution_graph.contains_node(exec_idx) {
                continue;
            }
            let payload = enode
                .payload()
                .expect("non-identity selections carry a payload");
            let mut variant = driver.egraph.analysis.payloads.get(payload).clone();
            // The payload may have been interned by a different
            // structurally-identical instance; its concrete inputs belong to
            // that instance. Rebind them to this e-node's actual children,
            // resolved through the same class-representative mapping
            // extraction used for its read/kill accounting, so the graph
            // edges agree with what extraction kept alive.
            let child_inners: Vec<crate::compute_graph::NodeIndex> = enode
                .children()
                .iter()
                .map(|&child| {
                    let child_prov = driver.prov_of_class(child, &extraction.needed);
                    driver.egraph.analysis.facts_of(child_prov).inner
                })
                .collect();
            rebind_variant_dependencies(&mut variant, &child_inners);
            let dependencies = variant_dependencies(&variant);
            debug_assert_eq!(
                dependencies, child_inners,
                "rebinding must place every child in a dependency slot"
            );
            self.commit_recognized(graph, exec_idx, &dependencies, variant);
            applied += 1;
        }
        applied
    }
}
