use egg::{EGraph, Id, Rewrite};

use crate::analysis::TensorAnalysis;
use crate::applier::SimpleEclassSearcher;
use crate::language::{DispatchNode, HighLevelNode, TensorIr, add_list, extract_list};
use crate::rules::RunnerConfig;
use crate::types::{IndexLevel, VarRef};

use super::{
    compute_flat_addr, decompose_flat_index, find_underlying_input, lower_scalar_body_strided,
};

pub(super) fn build(config: &RunnerConfig) -> Rewrite<TensorIr, TensorAnalysis> {
    let simd_width = config.device.simd_width;
    Rewrite::new(
        "elementwise-to-dispatch",
        SimpleEclassSearcher::new(move |egraph, eclass| {
            let Some(shape) = &egraph[eclass].data.shape else {
                return false;
            };
            shape
                .static_numel()
                .is_some_and(|n| n >= simd_width && n % simd_width == 0)
                && egraph[eclass]
                    .iter()
                    .any(|n| matches!(n, TensorIr::HighLevel(HighLevelNode::Elementwise { .. })))
        }),
        crate::applier::AdaptedApplier(ElementwiseApplier { simd_width }),
    )
    .unwrap()
}

struct ElementwiseApplier {
    simd_width: u32,
}

impl crate::applier::TypedApplier for ElementwiseApplier {
    fn apply(&self, egraph: &mut EGraph<TensorIr, TensorAnalysis>, eclass: Id) -> Vec<Id> {
        let node = egraph[eclass]
            .iter()
            .find(|n| matches!(n, TensorIr::HighLevel(HighLevelNode::Elementwise { .. })))
            .cloned();

        let Some(TensorIr::HighLevel(HighLevelNode::Elementwise {
            index_space,
            num_inputs,
            children_list,
        })) = node
        else {
            return vec![];
        };

        let Some(output_elements) = index_space.static_numel() else {
            return vec![];
        };
        if output_elements < self.simd_width || !output_elements.is_multiple_of(self.simd_width) {
            return vec![];
        }

        let children = extract_list(egraph, children_list);
        let input_count = num_inputs as usize;
        if children.len() <= input_count {
            return vec![];
        }
        let ewise_inputs = &children[..input_count];
        let body = children[input_count];

        let workgroups = output_elements / self.simd_width;
        let wg = egraph.add(TensorIr::Simd(crate::language::SimdNode::Var(
            VarRef::thread(IndexLevel::Workgroup),
        )));
        let lane = egraph.add(TensorIr::Simd(crate::language::SimdNode::Var(
            VarRef::thread(IndexLevel::Lane),
        )));
        let sw = egraph.add(TensorIr::Const(crate::types::ScalarValue::U32(
            self.simd_width,
        )));
        let wg_offset = egraph.add(TensorIr::BinOp(crate::types::BinaryOp::Mul, [wg, sw]));
        let out_flat = egraph.add(TensorIr::BinOp(
            crate::types::BinaryOp::Add,
            [wg_offset, lane],
        ));
        let out_indices = decompose_flat_index(egraph, out_flat, &index_space);
        let value = lower_scalar_body_strided(egraph, body, ewise_inputs, &out_indices);
        let out_addr = compute_flat_addr(egraph, &out_indices, &index_space);

        let mut dispatch_children: Vec<Id> = ewise_inputs
            .iter()
            .map(|input| find_underlying_input(egraph, *input))
            .collect();
        dispatch_children.push(value);
        dispatch_children.push(out_addr);
        let children_list = add_list(egraph, &dispatch_children);
        let dispatch = egraph.add(TensorIr::Dispatch(DispatchNode::Dispatch {
            workgroups,
            num_inputs,
            children_list,
        }));
        egraph.union(eclass, dispatch);
        vec![dispatch]
    }
}
