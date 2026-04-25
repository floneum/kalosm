use egg::{EGraph, Id, Rewrite};

use crate::analysis::TensorAnalysis;
use crate::applier::SimpleEclassSearcher;
use crate::language::{DispatchNode, TensorIr, add_list, extract_list};

pub(super) fn build() -> Rewrite<TensorIr, TensorAnalysis> {
    Rewrite::new(
        "pipeline-fusion",
        SimpleEclassSearcher::new(|egraph, eclass| {
            egraph[eclass]
                .iter()
                .any(|node| pipeline_candidate_dispatches(egraph, node).is_some())
        }),
        crate::applier::AdaptedApplier(PipelineFusionApplier),
    )
    .unwrap()
}

fn pipeline_candidate_dispatches(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    node: &TensorIr,
) -> Option<Vec<Id>> {
    let TensorIr::Dispatch(DispatchNode::Seq(dispatches)) = node else {
        return None;
    };
    let dispatches = extract_list(egraph, *dispatches);
    if dispatches.len() < 2 {
        return None;
    }
    dispatches
        .iter()
        .all(|id| {
            egraph[*id]
                .iter()
                .any(|n| matches!(n, TensorIr::Dispatch(DispatchNode::Dispatch { .. })))
        })
        .then_some(dispatches)
}

struct PipelineFusionApplier;

impl crate::applier::TypedApplier for PipelineFusionApplier {
    fn apply(&self, egraph: &mut EGraph<TensorIr, TensorAnalysis>, eclass: Id) -> Vec<Id> {
        if egraph[eclass]
            .iter()
            .any(|node| matches!(node, TensorIr::Dispatch(DispatchNode::Pipeline(_))))
        {
            return vec![];
        }

        let Some(dispatches) = egraph[eclass]
            .iter()
            .find_map(|node| pipeline_candidate_dispatches(egraph, node))
        else {
            return vec![];
        };

        let stages = add_list(egraph, &dispatches);
        let pipeline = egraph.add(TensorIr::Dispatch(DispatchNode::Pipeline(stages)));
        egraph.union(eclass, pipeline);
        vec![pipeline]
    }
}
