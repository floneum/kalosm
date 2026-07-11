//! Delta application: write extraction's non-identity selections back onto
//! the execution graph.
//!
//! Reuses `commit_recognized` — the destructive optimizer's own commit
//! surgery — so the two invariants every rewrite maintains hold here by
//! construction: execution-graph edges match the new payload's `inputs`, and
//! `add_physical_dependencies` records persistent inner-graph edges (firing
//! the flush-replay recording hook identically). Killed producers fall out
//! through `remove_node_if_dead`, mirroring extraction's kill cascade.

use super::super::Resolver;
use super::EGraphDriver;
use super::extract::Extraction;
use super::interner::variant_dependencies;
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
            let payload = enode
                .payload()
                .expect("non-identity selections carry a payload");
            let variant = driver.egraph.analysis.payloads.get(payload).clone();
            let dependencies = variant_dependencies(&variant);
            self.commit_recognized(graph, exec_idx, &dependencies, variant);
            applied += 1;
        }
        applied
    }
}
