//! Phase 5: Shuffle-Based Data Sharing (Rules 13-14)

mod broadcast_as_shuffle;
mod simdgroup_broadcast;

use egg::{EGraph, Id, Rewrite};

use crate::analysis::TensorAnalysis;
use crate::language::{SimdNode, TensorIr};
use crate::rules::RunnerConfig;
use crate::types::{DeviceProfile, MemTier, ScalarValue};

#[must_use]
pub fn rules(config: &RunnerConfig) -> Vec<Rewrite<TensorIr, TensorAnalysis>> {
    vec![
        broadcast_as_shuffle::build(),
        simdgroup_broadcast::build(config),
    ]
}

pub fn threadgroup_to_shuffle_reuse(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    load_eclass: Id,
    num_unique_values: u32,
    device: &DeviceProfile,
) -> Option<Vec<Id>> {
    let node = egraph[load_eclass]
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
        .cloned()?;

    let TensorIr::Simd(SimdNode::Load { tier, children }) = node else {
        return None;
    };
    let addr = children[0];
    let state = children[1];

    if num_unique_values >= device.simd_width {
        return None;
    }

    let mut results = Vec::new();
    let raw_load = egraph.add(TensorIr::Simd(SimdNode::Load {
        tier,
        children: [addr, state],
    }));

    for lane_src in 0..num_unique_values {
        let src_lit = egraph.add(TensorIr::Const(ScalarValue::U32(lane_src)));
        let shuffled = egraph.add(TensorIr::Simd(SimdNode::Shuffle([raw_load, src_lit])));
        results.push(shuffled);
    }

    Some(results)
}
