//! Per-class analysis data and the driver context rules read through the
//! e-graph.
//!
//! Per-observation facts live in [`FusorAnalysis::facts`]. E-class membership
//! is maintained by the driver because hash-consing may attach several graph
//! observations to one class without invoking `Analysis::merge`.

use egg::{Analysis, DidMerge, EGraph, Id};
use rustc_hash::FxHashMap;

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
}

#[derive(Default)]
pub(super) struct FusorAnalysis {
    /// Indexed by `Prov`.
    pub(super) facts: Vec<NodeFacts>,
    pub(super) payloads: PayloadTable,
    /// Inner node -> the e-class assigned during ingestion. Ids may become
    /// non-canonical after unions; callers canonicalize with `EGraph::find`.
    pub(super) class_of_inner: FxHashMap<NodeIndex, Id>,
}

impl FusorAnalysis {
    pub(super) fn facts_of(&self, prov: Prov) -> &NodeFacts {
        &self.facts[prov.0 as usize]
    }
}

impl Analysis<FusorLang> for FusorAnalysis {
    /// No per-class data: every fact this optimizer needs is per observation.
    type Data = ();

    fn make(_egraph: &mut EGraph<FusorLang, Self>, _enode: &FusorLang, _id: Id) -> Self::Data {}

    fn merge(&mut self, _a: &mut Self::Data, _b: Self::Data) -> DidMerge {
        DidMerge(false, false)
    }
}
