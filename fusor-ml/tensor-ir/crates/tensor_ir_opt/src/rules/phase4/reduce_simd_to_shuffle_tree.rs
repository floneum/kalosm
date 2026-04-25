use egg::{EGraph, Id, Rewrite};

use crate::analysis::TensorAnalysis;
use crate::applier::SimpleEclassSearcher;
use crate::language::{SimdNode, TensorIr};
use crate::rules::RunnerConfig;
use crate::types::{BinaryOp, IndexLevel, ScalarValue, VarRef};

pub(super) fn build(config: &RunnerConfig) -> Rewrite<TensorIr, TensorAnalysis> {
    Rewrite::new(
        "reduce-simd-to-shuffle-tree",
        SimpleEclassSearcher::new(|egraph, eclass| {
            let eclass_data = &egraph[eclass];
            eclass_data
                .iter()
                .any(|n| matches!(n, TensorIr::Simd(SimdNode::ReduceSimd { .. })))
                && !eclass_data.iter().any(|n| matches!(n, TensorIr::BinOp(..)))
        }),
        crate::applier::AdaptedApplier(ReduceSimdToShuffleApplier {
            simd_width: config.device.simd_width,
        }),
    )
    .unwrap()
}

struct ReduceSimdToShuffleApplier {
    simd_width: u32,
}

impl crate::applier::TypedApplier for ReduceSimdToShuffleApplier {
    fn apply(&self, egraph: &mut EGraph<TensorIr, TensorAnalysis>, eclass: Id) -> Vec<Id> {
        let node = egraph[eclass]
            .iter()
            .find(|n| matches!(n, TensorIr::Simd(SimdNode::ReduceSimd { .. })))
            .cloned();

        let Some(TensorIr::Simd(SimdNode::ReduceSimd { op, src })) = node else {
            return vec![];
        };

        // Butterfly reduction runs log2(simd_width) XOR-shuffle steps; the
        // power-of-2 assumption is what makes the XOR pattern collapse to a
        // tree. Bail out rather than emit an incorrect half-tree on exotic
        // widths.
        if !self.simd_width.is_power_of_two() || self.simd_width < 2 {
            return vec![];
        }
        let steps = self.simd_width.trailing_zeros();

        let lane = egraph.add(TensorIr::Simd(SimdNode::Var(VarRef::thread(
            IndexLevel::Lane,
        ))));
        let op_name = op.bin_op();
        let mut current = src;
        for i in 0..steps {
            let offset = 1u32 << i;
            let offset_lit = egraph.add(TensorIr::Const(ScalarValue::U32(offset)));
            let xor_lane = egraph.add(TensorIr::BinOp(BinaryOp::Xor, [lane, offset_lit]));
            let shuffled = egraph.add(TensorIr::Simd(SimdNode::Shuffle([current, xor_lane])));
            current = egraph.add(TensorIr::BinOp(op_name, [current, shuffled]));
        }

        egraph.union(eclass, current);
        vec![current]
    }
}
