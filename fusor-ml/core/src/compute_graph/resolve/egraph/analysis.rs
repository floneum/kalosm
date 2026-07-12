//! Per-class analysis data and the driver context rules read through the
//! e-graph.
//!
//! Per-observation facts live in [`FusorAnalysis::facts`]. E-class membership
//! is maintained by the driver because hash-consing may attach several graph
//! observations to one class without invoking `Analysis::merge`.

use std::sync::Arc;

use egg::{Analysis, DidMerge, EGraph, Id};
use rustc_hash::FxHashMap;

use super::interner::PayloadTable;
use super::lang::{FusorLang, Prov};
use crate::compute_graph::{ComputeGraphInner, ComputeGraphNodeVariant, NodeIndex};

/// Immutable resolver inputs needed by programmatic egg searchers/appliers.
///
/// `egg::Rewrite` values are owned by the runner, so they cannot borrow the
/// live compute graph held behind the resolver lock.  Snapshot only the
/// structural data recognition consults; concrete allocation identity
/// remains in the e-graph leaves themselves.
pub(super) struct PlannerSnapshot {
    pub(super) device: crate::Device,
    dequantize: FxHashMap<NodeIndex, crate::dequantize::DequantizeOperation>,
    views: FxHashMap<NodeIndex, crate::view::ViewOperation>,
}

impl PlannerSnapshot {
    pub(super) fn new(
        graph: &ComputeGraphInner,
        nodes: impl IntoIterator<Item = NodeIndex>,
    ) -> Self {
        let mut dequantize = FxHashMap::default();
        let mut views = FxHashMap::default();
        for node in nodes {
            let Some(data) = graph.nodes.nodes.node_weight(node) else {
                continue;
            };
            match &data.variant {
                ComputeGraphNodeVariant::QMatrix(operation) => {
                    dequantize.insert(node, operation.clone());
                }
                ComputeGraphNodeVariant::View(operation) => {
                    views.insert(node, operation.clone());
                }
                _ => {}
            }
        }
        Self {
            device: graph.device(),
            dequantize,
            views,
        }
    }

    pub(super) fn dequantize(
        &self,
        node: NodeIndex,
    ) -> Option<crate::dequantize::DequantizeOperation> {
        self.dequantize.get(&node).cloned()
    }

    pub(super) fn view(&self, node: NodeIndex) -> Option<&crate::view::ViewOperation> {
        self.views.get(&node)
    }
}

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
    /// Planner-wide immutable context used by native egg rules.
    pub(super) planner: Option<Arc<PlannerSnapshot>>,
    /// Inner node -> the e-class assigned during ingestion. Ids may become
    /// non-canonical after unions; callers canonicalize with `EGraph::find`.
    pub(super) class_of_inner: FxHashMap<NodeIndex, Id>,
}

impl FusorAnalysis {
    pub(super) fn facts_of(&self, prov: Prov) -> &NodeFacts {
        &self.facts[prov.0 as usize]
    }
}

#[derive(Debug)]
pub(super) struct ClassData;

impl Analysis<FusorLang> for FusorAnalysis {
    type Data = ClassData;

    fn make(_egraph: &EGraph<FusorLang, Self>, _enode: &FusorLang) -> Self::Data {
        ClassData
    }

    fn merge(&mut self, a: &mut Self::Data, b: Self::Data) -> DidMerge {
        let _ = (a, b);
        DidMerge(false, false)
    }

    fn modify(_egraph: &mut EGraph<FusorLang, Self>, _id: Id) {}
}
