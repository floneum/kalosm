//! Simdgroup broadcast rule.
//!
//! For a TG `Load` whose address depends on `lane` only through patterns that
//! partition the simdgroup into buckets (e.g. `lane / N * stride + k`), the
//! address has fewer than `simd_width` unique values per simdgroup. Every
//! bucket's lanes read the same value, so the load can be rewritten as
//! `Shuffle(load, representative_lane)` where the representative lane source
//! is computed from `lane` itself — `(lane / N) * N` picks the bucket's first
//! lane. Downstream codegen turns this into a `simd_broadcast`/`simd_shuffle`
//! intrinsic that loads once per bucket instead of once per lane.
//!
//! The fully-invariant case (N = simd_width → 1 unique value across the
//! whole simdgroup) is handled by `broadcast_as_shuffle`; this rule only
//! fires for partial partitions where `1 < unique_values < simd_width`.

use std::collections::{HashMap, HashSet};

use egg::{EGraph, Id, Rewrite};

use crate::analysis::TensorAnalysis;
use crate::applier::SimpleEclassSearcher;
use crate::language::{SimdNode, TensorIr};
use crate::rules::RunnerConfig;
use crate::types::{BinaryOp, BinderKind, IndexLevel, MemTier, ScalarValue, VarRef, slots};

pub(super) fn build(config: &RunnerConfig) -> Rewrite<TensorIr, TensorAnalysis> {
    let simd_width = config.device.simd_width;
    Rewrite::new(
        "simdgroup-broadcast",
        SimpleEclassSearcher::new(move |egraph, eclass| {
            let eclass_data = &egraph[eclass];
            if eclass_data
                .iter()
                .any(|n| matches!(n, TensorIr::Simd(SimdNode::Shuffle(_))))
            {
                return false;
            }
            eclass_data.iter().any(|node| {
                let TensorIr::Simd(SimdNode::Load {
                    tier: MemTier::Threadgroup(_),
                    children,
                }) = node
                else {
                    return false;
                };
                let addr_dep = egraph[children[0]].data.dep;
                if !addr_dep.contains_lane() {
                    return false;
                }
                matches!(
                    infer_lane_range(egraph, children[0], simd_width),
                    Some(range) if range > 1 && range < simd_width
                )
            })
        }),
        crate::applier::AdaptedApplier(SimdgroupBroadcastApplier { simd_width }),
    )
    .unwrap()
}

struct SimdgroupBroadcastApplier {
    simd_width: u32,
}

impl crate::applier::TypedApplier for SimdgroupBroadcastApplier {
    fn apply(&self, egraph: &mut EGraph<TensorIr, TensorAnalysis>, eclass: Id) -> Vec<Id> {
        if egraph[eclass]
            .iter()
            .any(|n| matches!(n, TensorIr::Simd(SimdNode::Shuffle(_))))
        {
            return vec![];
        }

        let load_node = egraph[eclass]
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
        let Some(TensorIr::Simd(SimdNode::Load { tier, children })) = load_node else {
            return vec![];
        };
        let addr = children[0];

        let range = match infer_lane_range(egraph, addr, self.simd_width) {
            Some(r) if r > 1 && r < self.simd_width => r,
            _ => return vec![],
        };
        // bucket_size is the number of consecutive lanes that share one value.
        // simd_width/range is exact because both `range` and `simd_width` are
        // powers of two in every supported device profile; round up to be safe.
        let bucket_size = self.simd_width / range.max(1);
        if bucket_size <= 1 {
            return vec![];
        }

        let lane = egraph.add(TensorIr::Simd(SimdNode::Var(VarRef::thread(
            IndexLevel::Lane,
        ))));
        let bucket_lit = egraph.add(TensorIr::Const(ScalarValue::U32(bucket_size)));
        let lane_div = egraph.add(TensorIr::BinOp(BinaryOp::Div, [lane, bucket_lit]));
        let rep_lane = egraph.add(TensorIr::BinOp(BinaryOp::Mul, [lane_div, bucket_lit]));

        let reload = egraph.add(TensorIr::Simd(SimdNode::Load { tier, children }));
        let shuffled = egraph.add(TensorIr::Simd(SimdNode::Shuffle([reload, rep_lane])));
        tracing::debug!(
            bucket_size,
            unique_values = range,
            "promoting TG load to simdgroup broadcast shuffle"
        );
        egraph.union(eclass, shuffled);
        vec![shuffled]
    }
}

/// Bound on the number of distinct values `id` takes across one simdgroup.
/// Returns `Some(simd_width)` for the lane var itself, divides by constant
/// divisors, passes through constant-scale `Add`/`Sub`/`Mul` into the
/// lane-dependent operand, and returns `None` on unrecognized shapes.
///
/// Duplicated from `phase3/tiled_load_promotion::infer_local_axis_range` for
/// now — promoting both copies to a shared `addr_analysis` module is a
/// follow-up. Keep the two implementations in lock-step.
fn infer_lane_range(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    id: Id,
    simd_width: u32,
) -> Option<u32> {
    fn rec(
        egraph: &EGraph<TensorIr, TensorAnalysis>,
        id: Id,
        simd_width: u32,
        memo: &mut HashMap<Id, Option<u32>>,
        visiting: &mut HashSet<Id>,
    ) -> Option<u32> {
        let canonical = egraph.find(id);
        if let Some(&cached) = memo.get(&canonical) {
            return cached;
        }
        if !visiting.insert(canonical) {
            return None;
        }

        let mut best = None;
        for node in egraph[canonical].iter() {
            let candidate = match node {
                TensorIr::Simd(SimdNode::Var(VarRef::Bound {
                    kind: BinderKind::Dispatch,
                    slot: slots::DISPATCH_LANE,
                    depth: 0,
                })) => Some(simd_width),
                TensorIr::BinOp(BinaryOp::Mod, args) => {
                    if let Some(ScalarValue::U32(v)) = &egraph[args[1]].data.constant {
                        Some(*v)
                    } else {
                        None
                    }
                }
                TensorIr::BinOp(BinaryOp::Div, args) => {
                    if let Some(ScalarValue::U32(v)) = &egraph[args[1]].data.constant {
                        rec(egraph, args[0], simd_width, memo, visiting)
                            .map(|range| range.div_ceil(*v))
                    } else {
                        None
                    }
                }
                TensorIr::BinOp(BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul, args) => {
                    let left_dep = egraph[args[0]].data.dep;
                    let right_dep = egraph[args[1]].data.dep;
                    if left_dep.contains_lane() && !right_dep.contains_lane() {
                        rec(egraph, args[0], simd_width, memo, visiting)
                    } else if right_dep.contains_lane() && !left_dep.contains_lane() {
                        rec(egraph, args[1], simd_width, memo, visiting)
                    } else {
                        None
                    }
                }
                _ => None,
            };
            if let Some(range) = candidate.filter(|range| *range > 0) {
                best = Some(best.map_or(range, |current: u32| current.min(range)));
            }
        }

        visiting.remove(&canonical);
        memo.insert(canonical, best);
        best
    }

    let mut memo = HashMap::new();
    let mut visiting = HashSet::new();
    rec(egraph, id, simd_width, &mut memo, &mut visiting)
}
