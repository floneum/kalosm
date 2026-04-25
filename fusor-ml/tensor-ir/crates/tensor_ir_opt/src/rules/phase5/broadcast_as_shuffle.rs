use egg::{EGraph, Id, Rewrite};

use crate::analysis::TensorAnalysis;
use crate::applier::SimpleEclassSearcher;
use crate::language::{SimdNode, TensorIr};
use crate::types::{MemTier, ScalarValue};

pub(super) fn build() -> Rewrite<TensorIr, TensorAnalysis> {
    Rewrite::new(
        "broadcast-as-shuffle",
        SimpleEclassSearcher::new(|egraph, eclass| {
            let eclass_data = &egraph[eclass];
            eclass_data.iter().any(|node| {
                let TensorIr::Simd(SimdNode::Load {
                    tier: MemTier::Threadgroup(_),
                    children,
                }) = node
                else {
                    return false;
                };
                !egraph[children[0]].data.dep.contains_lane()
                    && !eclass_data
                        .iter()
                        .any(|n| matches!(n, TensorIr::Simd(SimdNode::Shuffle(_))))
            })
        }),
        crate::applier::AdaptedApplier(BroadcastApplier),
    )
    .unwrap()
}

struct BroadcastApplier;

impl crate::applier::TypedApplier for BroadcastApplier {
    fn apply(&self, egraph: &mut EGraph<TensorIr, TensorAnalysis>, eclass: Id) -> Vec<Id> {
        let node = egraph[eclass]
            .iter()
            .find(|n| {
                matches!(
                    n,
                    TensorIr::Simd(SimdNode::Load {
                        tier: MemTier::Threadgroup(_),
                        ..
                    })
                )
            })
            .cloned();

        let Some(TensorIr::Simd(SimdNode::Load { tier, children })) = node else {
            return vec![];
        };
        let addr = children[0];
        let state = children[1];

        let load_node = egraph.add(TensorIr::Simd(SimdNode::Load {
            tier,
            children: [addr, state],
        }));
        let lane_zero = egraph.add(TensorIr::Const(ScalarValue::U32(0)));
        let broadcast = egraph.add(TensorIr::Simd(SimdNode::Shuffle([load_node, lane_zero])));

        egraph.union(eclass, broadcast);
        vec![broadcast]
    }
}
