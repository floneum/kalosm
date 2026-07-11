//! Per-class analysis data and the driver context rules read through the
//! e-graph.
//!
//! Under provenance salting every e-class denotes exactly one execution
//! node, so the only per-class datum is that node's [`Prov`]; `merge`
//! asserts the salting invariant (unions may only join alternatives of one
//! node). Everything else rules need — per-node facts snapshotted at
//! ingestion, the payload table, the device — lives on the analysis struct
//! itself, reachable from appliers via `egraph.analysis`.

use egg::{Analysis, DidMerge, EGraph, Id};

use super::interner::PayloadTable;
use super::lang::{FusorLang, Prov};
use crate::compute_graph::NodeIndex;

/// Facts about one execution node (or cached-boundary leaf), snapshotted at
/// ingestion. Indexed by `Prov`.
#[derive(Debug, Clone)]
pub(super) struct NodeFacts {
    pub(super) inner: NodeIndex,
    /// The execution-graph node, `None` for cached-boundary leaves (which
    /// are excluded from the execution graph).
    pub(super) exec: Option<super::super::ExecutionNodeIndex>,
    /// `reference_count > 0` at ingestion: a user handle exists. Blocks
    /// recognition cluster claims; does NOT block nary fusion (matching the
    /// destructive optimizer's gates).
    pub(super) externally_live: bool,
    /// A resolve target: must materialize, may never be killed.
    pub(super) is_target: bool,
    /// Execution-graph consumer count, counted per edge occurrence
    /// (parallel edges from repeated reads count separately, mirroring
    /// `build_execution_graph`'s one-edge-per-dependency-occurrence).
    pub(super) consumer_count: u32,
}

pub(super) struct FusorAnalysis {
    /// Indexed by `Prov`.
    pub(super) facts: Vec<NodeFacts>,
    pub(super) payloads: PayloadTable,
}

impl FusorAnalysis {
    pub(super) fn facts_of(&self, prov: Prov) -> &NodeFacts {
        &self.facts[prov.0 as usize]
    }
}

#[derive(Debug)]
pub(super) struct ClassData {
    pub(super) prov: Prov,
}

impl Analysis<FusorLang> for FusorAnalysis {
    type Data = ClassData;

    fn make(_egraph: &EGraph<FusorLang, Self>, enode: &FusorLang) -> Self::Data {
        ClassData { prov: enode.prov() }
    }

    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        // The salting invariant: a union may only join alternatives of one
        // execution node. A mismatch here is a rule bug.
        debug_assert_eq!(
            a.prov, b.prov,
            "e-graph union across distinct execution nodes"
        );
        DidMerge(false, false)
    }

    fn modify(_egraph: &mut EGraph<FusorLang, Self>, _id: Id) {}
}
