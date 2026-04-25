use egg::{EGraph, Id, Rewrite};

use crate::analysis::TensorAnalysis;
use crate::applier::SimpleEclassSearcher;
use crate::language::{DispatchNode, HighLevelNode, SimdNode, TensorIr, add_list};
use crate::rules::RunnerConfig;
use crate::types::{BinaryOp, BufferRef, IndexLevel, MemTier, ScalarValue, TernaryOp, VarRef};

use super::{
    compute_flat_addr, compute_strided_addr, decompose_flat_index, find_underlying_input,
    get_restride_layout,
};

pub(super) fn build(config: &RunnerConfig) -> Rewrite<TensorIr, TensorAnalysis> {
    let simd_width = config.device.simd_width;
    Rewrite::new(
        "slice-assign-to-dispatch",
        SimpleEclassSearcher::new(move |egraph, eclass| {
            let Some(shape) = &egraph[eclass].data.shape else {
                return false;
            };
            shape
                .static_numel()
                .is_some_and(|n| n >= simd_width && n % simd_width == 0)
                && egraph[eclass]
                    .iter()
                    .any(|n| matches!(n, TensorIr::HighLevel(HighLevelNode::SliceAssign { .. })))
        }),
        crate::applier::AdaptedApplier(SliceAssignApplier { simd_width }),
    )
    .unwrap()
}

struct SliceAssignApplier {
    simd_width: u32,
}

fn u32_lit(egraph: &mut EGraph<TensorIr, TensorAnalysis>, value: u32) -> Id {
    egraph.add(TensorIr::Const(ScalarValue::U32(value)))
}

fn load_slot(egraph: &mut EGraph<TensorIr, TensorAnalysis>, slot: u32, addr: Id) -> Id {
    let state = egraph.add(TensorIr::Dispatch(DispatchNode::Token));
    egraph.add(TensorIr::Simd(SimdNode::Load {
        tier: MemTier::Device(BufferRef::Input(slot)),
        children: [addr, state],
    }))
}

fn add_offset(egraph: &mut EGraph<TensorIr, TensorAnalysis>, addr: Id, offset: i64) -> Id {
    if offset == 0 {
        return addr;
    }
    let offset = u32::try_from(offset).expect("slice assign offset fits in u32");
    let offset = u32_lit(egraph, offset);
    egraph.add(TensorIr::BinOp(BinaryOp::Add, [addr, offset]))
}

fn build_slice_assign_output(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    output_shape: &crate::types::Shape,
    slices: &[(u32, u32)],
    input: Id,
    value: Id,
    out_flat: Id,
) -> (Id, Id) {
    let out_indices = decompose_flat_index(egraph, out_flat, output_shape);

    let input_addr = {
        let (strides, offset) = get_restride_layout(egraph, input);
        let addr = compute_strided_addr(egraph, &out_indices, &strides);
        add_offset(egraph, addr, offset)
    };
    let input_value = load_slot(egraph, 0, input_addr);

    let mut in_slice = egraph.add(TensorIr::Const(ScalarValue::Bool(true)));
    let mut relative_indices = Vec::with_capacity(out_indices.len());
    for (idx, (start, end)) in out_indices.iter().zip(slices.iter()) {
        let start_lit = u32_lit(egraph, *start);
        let end_lit = u32_lit(egraph, *end);
        let ge_start = egraph.add(TensorIr::BinOp(BinaryOp::Ge, [*idx, start_lit]));
        let lt_end = egraph.add(TensorIr::BinOp(BinaryOp::Lt, [*idx, end_lit]));
        let axis_in_slice = egraph.add(TensorIr::BinOp(BinaryOp::And, [ge_start, lt_end]));
        in_slice = egraph.add(TensorIr::BinOp(BinaryOp::And, [in_slice, axis_in_slice]));

        let relative = if *start == 0 {
            *idx
        } else {
            egraph.add(TensorIr::BinOp(BinaryOp::Sub, [*idx, start_lit]))
        };
        relative_indices.push(relative);
    }

    let zero = u32_lit(egraph, 0);
    let safe_value_indices = relative_indices
        .into_iter()
        .map(|idx| egraph.add(TensorIr::TernOp(TernaryOp::Select, [in_slice, idx, zero])))
        .collect::<Vec<_>>();
    let value_addr = {
        let (strides, offset) = get_restride_layout(egraph, value);
        let addr = compute_strided_addr(egraph, &safe_value_indices, &strides);
        add_offset(egraph, addr, offset)
    };
    let replacement_value = load_slot(egraph, 1, value_addr);
    let output_value = egraph.add(TensorIr::TernOp(
        TernaryOp::Select,
        [in_slice, replacement_value, input_value],
    ));
    let output_addr = compute_flat_addr(egraph, &out_indices, output_shape);
    (output_value, output_addr)
}

impl crate::applier::TypedApplier for SliceAssignApplier {
    fn apply(&self, egraph: &mut EGraph<TensorIr, TensorAnalysis>, eclass: Id) -> Vec<Id> {
        let node = egraph[eclass]
            .iter()
            .find(|n| matches!(n, TensorIr::HighLevel(HighLevelNode::SliceAssign { .. })))
            .cloned();

        let Some(TensorIr::HighLevel(HighLevelNode::SliceAssign {
            output_shape,
            slices,
            children: [input, value],
        })) = node
        else {
            return vec![];
        };

        let Some(output_elements) = output_shape.static_numel() else {
            return vec![];
        };
        if output_elements < self.simd_width || !output_elements.is_multiple_of(self.simd_width) {
            return vec![];
        }

        let outputs_per_workgroup = if output_elements.is_multiple_of(self.simd_width * 2) {
            2
        } else {
            1
        };
        let workgroups = output_elements / (self.simd_width * outputs_per_workgroup);
        let wg = egraph.add(TensorIr::Simd(SimdNode::Var(VarRef::thread(
            IndexLevel::Workgroup,
        ))));
        let lane = egraph.add(TensorIr::Simd(SimdNode::Var(VarRef::thread(
            IndexLevel::Lane,
        ))));
        let stride = u32_lit(egraph, self.simd_width * outputs_per_workgroup);
        let wg_offset = egraph.add(TensorIr::BinOp(BinaryOp::Mul, [wg, stride]));

        let mut dispatch_children = vec![
            find_underlying_input(egraph, input),
            find_underlying_input(egraph, value),
        ];
        for output_index in 0..outputs_per_workgroup {
            let flat_offset = if output_index == 0 {
                lane
            } else {
                let output_offset = u32_lit(egraph, output_index * self.simd_width);
                egraph.add(TensorIr::BinOp(BinaryOp::Add, [lane, output_offset]))
            };
            let out_flat = egraph.add(TensorIr::BinOp(BinaryOp::Add, [wg_offset, flat_offset]));
            let (output_value, output_addr) =
                build_slice_assign_output(egraph, &output_shape, &slices, input, value, out_flat);
            dispatch_children.push(output_value);
            dispatch_children.push(output_addr);
        }
        let children_list = add_list(egraph, &dispatch_children);
        let dispatch = egraph.add(TensorIr::Dispatch(DispatchNode::Dispatch {
            workgroups,
            num_inputs: 2,
            children_list,
        }));
        egraph.union(eclass, dispatch);
        vec![dispatch]
    }
}
