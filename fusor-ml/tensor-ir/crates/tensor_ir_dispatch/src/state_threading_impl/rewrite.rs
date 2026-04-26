//! Dispatch-node rewriting (called from phase-7 state threading).

use std::collections::HashMap;

use egg::{EGraph, Id};

use crate::analysis::TensorAnalysis;
use crate::language::{
    DispatchNode, SimdNode, TensorIr, extract_list, try_add_value_addr_dispatch,
};
use crate::types::{DeviceProfile, LoweringOptions, VarRef};

use super::*;
use super::{
    StateThreadedOutput, collect_threadgroup_tile_info_from_egraph, dispatch_node_is_state_threaded,
};

pub(crate) fn rewrite_dispatch_node(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    dispatch: &TensorIr,
    device: &DeviceProfile,
    lowering: &LoweringOptions,
) -> Option<Id> {
    if dispatch_node_is_state_threaded(egraph, dispatch) {
        return None;
    }

    let layout = extract_dispatch_rewrite_layout(egraph, dispatch)?;
    if let Some(dispatch_id) = try_rewrite_tiled_dispatch(egraph, &layout, device, lowering) {
        return Some(dispatch_id);
    }
    if collect_threadgroup_tile_info_from_egraph(layout.body_addr_pairs[0].0, egraph).is_empty() {
        return None;
    }
    try_rewrite_recovered_tiled_dispatch(egraph, &layout, device, lowering)
}

pub(in crate::state_threading_impl) struct DispatchRewriteLayout {
    workgroups: u32,
    num_inputs: usize,
    input_ids: Vec<Id>,
    body_addr_pairs: Vec<(Id, Id)>,
}

impl DispatchRewriteLayout {
    fn num_outputs_u32(&self) -> u32 {
        u32::try_from(self.body_addr_pairs.len()).expect("output count fits in u32")
    }
}

pub(super) fn extract_dispatch_rewrite_layout(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    dispatch: &TensorIr,
) -> Option<DispatchRewriteLayout> {
    let TensorIr::Dispatch(DispatchNode::Dispatch {
        workgroups,
        num_inputs,
        children_list,
    }) = dispatch
    else {
        return None;
    };
    let num_inputs = *num_inputs as usize;
    let children = extract_list(egraph, *children_list);
    let body_len = children.len().saturating_sub(num_inputs);
    if !body_len.is_multiple_of(2) {
        return None;
    }
    let num_outputs = body_len / 2;

    let body_addr_pairs = (0..num_outputs)
        .map(|index| {
            let pair_start = num_inputs + index * 2;
            (children[pair_start], children[pair_start + 1])
        })
        .collect();
    Some(DispatchRewriteLayout {
        workgroups: workgroups.as_const()?,
        num_inputs,
        input_ids: children[..num_inputs].to_vec(),
        body_addr_pairs,
    })
}

pub(super) fn collect_egraph_tiled_outputs(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    body_addr_pairs: &[(Id, Id)],
    outer_k: u32,
    tile_k: u32,
) -> Option<Vec<TiledOutput>> {
    fn embedded_state_layout(
        egraph: &EGraph<TensorIr, TensorAnalysis>,
        init_id: Id,
        update_id: Id,
        output_index: usize,
    ) -> (usize, Option<usize>) {
        let init_arity = egraph[egraph.find(init_id)].iter().find_map(|node| {
            if let TensorIr::Dispatch(DispatchNode::Pack { children_list }) = node {
                Some(extract_list(egraph, *children_list).len())
            } else {
                None
            }
        });
        let update_arity = egraph[egraph.find(update_id)].iter().find_map(|node| {
            if let TensorIr::Dispatch(DispatchNode::Pack { children_list }) = node {
                Some(extract_list(egraph, *children_list).len())
            } else {
                None
            }
        });
        if output_index == 0 && init_arity == Some(4) && update_arity == Some(4) {
            (2, Some(3))
        } else {
            (output_index, None)
        }
    }

    let mut tiled_outputs = Vec::with_capacity(body_addr_pairs.len());
    for (output_index, (pair_body_id, pair_addr)) in body_addr_pairs.iter().enumerate() {
        let (pair_init_id, pair_outer_count_id, pair_inner_id) =
            egraph[egraph.find(*pair_body_id)].iter().find_map(|node| {
                let TensorIr::Simd(SimdNode::Theta {
                    children: [init_id, outer_count_id, inner_id],
                    ..
                }) = node
                else {
                    return None;
                };
                Some((*init_id, *outer_count_id, *inner_id))
            })?;
        let (pair_inner_count_id, pair_update) =
            egraph[egraph.find(pair_inner_id)].iter().find_map(|node| {
                let TensorIr::Simd(SimdNode::Theta {
                    children: [_inner_init, inner_count_id, update],
                    ..
                }) = node
                else {
                    return None;
                };
                Some((*inner_count_id, *update))
            })?;

        if get_u32_lit_from_egraph(egraph, pair_outer_count_id) != Some(outer_k)
            || get_u32_lit_from_egraph(egraph, pair_inner_count_id) != Some(tile_k)
        {
            return None;
        }

        // All outputs share the canonical acc(0). The output's identity
        // is the pair's position in `body_addr_pairs`, not a named
        // blocked accumulator ref.
        let acc_name = VarRef::acc(0);
        let (result_slot, state_slot) =
            embedded_state_layout(egraph, pair_init_id, pair_update, output_index);
        tiled_outputs.push(TiledOutput {
            acc_name,
            value_id: pair_inner_id,
            update_id: Some(pair_update),
            init_id: pair_init_id,
            addr_id: *pair_addr,
            result_slot,
            state_slot,
        });
    }
    Some(tiled_outputs)
}

pub(super) fn rebuild_dispatch_with_outputs(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    layout: &DispatchRewriteLayout,
    outputs: Vec<StateThreadedOutput>,
) -> Option<Id> {
    let mut children = layout.input_ids.clone();
    for output in outputs {
        children.push(output.value_id);
        children.push(output.addr_id);
    }
    try_add_value_addr_dispatch(
        egraph,
        layout.workgroups,
        u32::try_from(layout.num_inputs).expect("dispatch input count fits in u32"),
        &children,
    )
}

pub(super) fn try_rewrite_tiled_dispatch(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    layout: &DispatchRewriteLayout,
    device: &DeviceProfile,
    lowering: &LoweringOptions,
) -> Option<Id> {
    let mut chosen = HashMap::new();
    let body_id = layout.body_addr_pairs[0].0;
    let body_theta = egraph[egraph.find(body_id)].iter().find_map(|node| {
        let TensorIr::Simd(SimdNode::Theta {
            children: [_init_id, outer_count_id, inner_id],
            ..
        }) = node
        else {
            return None;
        };
        let inner_theta = egraph[egraph.find(*inner_id)]
            .iter()
            .find_map(|inner_node| {
                let TensorIr::Simd(SimdNode::Theta {
                    children: [_inner_init, inner_count_id, _inner_update],
                    ..
                }) = inner_node
                else {
                    return None;
                };
                Some((*outer_count_id, *inner_count_id))
            })?;
        Some(inner_theta)
    })?;
    let outer_k = get_u32_lit_from_egraph(egraph, body_theta.0)?;
    let tile_k = get_u32_lit_from_egraph(egraph, body_theta.1)?;
    let tiled_outputs =
        collect_egraph_tiled_outputs(egraph, &layout.body_addr_pairs, outer_k, tile_k)?;
    let tg_buffers = collect_threadgroup_tile_info_with_device_addrs(
        &tiled_outputs
            .iter()
            .map(|output| output.value_id)
            .collect::<Vec<_>>(),
        egraph,
        &mut chosen,
        tile_k,
        device,
    )?;
    if !selected_subtree_has_coherent_k_tile_stride(
        egraph,
        &mut chosen,
        tiled_outputs[0].value_id,
        tiled_outputs[0].value_id,
        tile_k,
    ) {
        return None;
    }

    let num_simdgroups = compute_simdgroups_for_tiled(
        &tg_buffers,
        layout.workgroups,
        layout.num_outputs_u32(),
        device,
    );
    let (outputs, _) = build_tiled_outputs(
        outer_k,
        Some(tile_k),
        &tiled_outputs,
        &tg_buffers,
        num_simdgroups,
        egraph,
        &mut chosen,
        device,
        lowering,
    );
    rebuild_dispatch_with_outputs(egraph, layout, outputs)
}

pub(super) fn try_rewrite_recovered_tiled_dispatch(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    layout: &DispatchRewriteLayout,
    device: &DeviceProfile,
    lowering: &LoweringOptions,
) -> Option<Id> {
    let mut chosen = HashMap::new();
    let body_id = layout.body_addr_pairs[0].0;
    let output_addr = layout.body_addr_pairs[0].1;
    let (outer_k, inner_theta_id, init_id, tg_bufs) =
        find_nested_theta_in_egraph(body_id, egraph, &mut chosen, device)?;
    let tg_sg = compute_simdgroups_for_tiled(
        &tg_bufs,
        layout.workgroups,
        layout.num_outputs_u32(),
        device,
    );
    let tiled_outputs = [TiledOutput {
        acc_name: VarRef::acc(0),
        value_id: inner_theta_id,
        update_id: None,
        init_id,
        addr_id: output_addr,
        result_slot: 0,
        state_slot: None,
    }];
    let (outputs, _) = build_tiled_outputs(
        outer_k,
        None,
        &tiled_outputs,
        &tg_bufs,
        tg_sg,
        egraph,
        &mut chosen,
        device,
        lowering,
    );
    rebuild_dispatch_with_outputs(egraph, layout, outputs)
}
