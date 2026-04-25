use egg::{BackoffScheduler, EGraph, Id, Rewrite, Runner};

use crate::analysis::TensorAnalysis;
use crate::applier::SimpleEclassSearcher;
use crate::language::{DispatchNode, TensorIr};
use crate::skeleton::{dispatch_node_is_state_threaded, rewrite_dispatch_node};
use crate::types::{DeviceProfile, LoweringOptions};

fn build(device: DeviceProfile, lowering: LoweringOptions) -> Rewrite<TensorIr, TensorAnalysis> {
    Rewrite::new(
        "state-threaded-dispatch",
        SimpleEclassSearcher::new(|egraph, eclass| {
            egraph[eclass].iter().any(|node| {
                matches!(node, TensorIr::Dispatch(DispatchNode::Dispatch { .. }))
                    && !dispatch_node_is_state_threaded(egraph, node)
            })
        }),
        crate::applier::AdaptedApplier(StateThreadedDispatchApplier { device, lowering }),
    )
    .unwrap()
}

#[must_use]
pub fn state_threading_rules(
    device: DeviceProfile,
    lowering: LoweringOptions,
) -> Vec<Rewrite<TensorIr, TensorAnalysis>> {
    vec![build(device, lowering)]
}

struct StateThreadedDispatchApplier {
    device: DeviceProfile,
    lowering: LoweringOptions,
}

impl crate::applier::TypedApplier for StateThreadedDispatchApplier {
    fn apply(&self, egraph: &mut EGraph<TensorIr, TensorAnalysis>, eclass: Id) -> Vec<Id> {
        let dispatches: Vec<_> = egraph[eclass]
            .iter()
            .filter(|node| matches!(node, TensorIr::Dispatch(DispatchNode::Dispatch { .. })))
            .cloned()
            .collect();

        let mut results = Vec::new();
        for dispatch in dispatches {
            if dispatch_node_is_state_threaded(egraph, &dispatch) {
                continue;
            }

            let Some(new_dispatch) =
                rewrite_dispatch_node(egraph, &dispatch, &self.device, &self.lowering)
            else {
                continue;
            };

            egraph.union(eclass, new_dispatch);
            results.push(new_dispatch);
        }

        results
    }
}

/// Run the dispatch-layer state-threading rewrite after optimizer phases have
/// produced candidate `Dispatch` nodes.
#[must_use]
pub fn saturate_state_threading(
    egraph: EGraph<TensorIr, TensorAnalysis>,
    device: DeviceProfile,
    lowering: LoweringOptions,
    iter_limit: usize,
    node_limit: usize,
    time_limit_secs: u64,
) -> EGraph<TensorIr, TensorAnalysis> {
    let rules = state_threading_rules(device, lowering);
    let runner = Runner::default()
        .with_egraph(egraph)
        .with_iter_limit(iter_limit)
        .with_node_limit(node_limit)
        .with_time_limit(std::time::Duration::from_secs(time_limit_secs))
        .with_scheduler(BackoffScheduler::default())
        .run(&rules);

    runner.egraph
}
