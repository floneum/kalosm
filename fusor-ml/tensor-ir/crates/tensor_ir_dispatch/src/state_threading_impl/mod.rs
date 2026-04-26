use std::collections::{HashMap, HashSet};

use egg::{EGraph, Id, Language};

use crate::analysis::TensorAnalysis;
use crate::language::{DispatchNode, HighLevelNode, SimdNode, TensorIr, add_list, extract_list};
use crate::types::{BinaryOp, BufferRef, DType, LoweringOptions, MemTier, ScalarValue};
pub use tensor_ir_egraph::add_and_choose;

/// Resolve the per-element byte size for a device buffer (`Input(N)`) by
/// looking up the matching `Input` node in the e-graph. Falls back to 4 bytes
/// if the buffer is an output ref or the matching input is missing.
pub(super) fn dtype_bytes_for_device_buffer(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    dev_name: BufferRef,
) -> u32 {
    let Some(target_id) = dev_name.input_index() else {
        return DType::F32.byte_size();
    };
    for class in egraph.classes() {
        for node in class.iter() {
            if let TensorIr::HighLevel(HighLevelNode::Input { id, dtype, .. }) = node
                && *id == target_id
            {
                return dtype.byte_size();
            }
        }
    }
    DType::F32.byte_size()
}

/// A (value, address) pair for one output element of a dispatch.
#[derive(Debug, Clone)]
pub struct StateThreadedOutput {
    /// The e-graph Id representing the output value (reduction result).
    pub value_id: Id,
    /// The e-graph Id for the output buffer address (flat index).
    pub addr_id: Id,
}

/// Metadata about a threadgroup buffer used in a dispatch.
#[derive(Debug, Clone)]
pub struct ThreadgroupTileInfo {
    /// Threadgroup buffer (typed reference to the underlying input/output).
    pub tg_name: BufferRef,
    /// Corresponding device buffer (same `BufferRef` — `tg_name` and
    /// `device_name` always refer to the same slot, just at different tiers).
    pub device_name: BufferRef,
    /// Total tile size in elements (`tile_rows` * `tile_cols`).
    pub size: u32,
    /// Storage bytes per element of this buffer. Resolved from the device
    /// load / source eclass's `TensorData::dtype_bytes` at construction; falls
    /// back to 4 bytes if dtype is unknown.
    pub dtype_bytes: u32,
    /// Number of columns per tile row (inner stride of the tg layout).
    pub tile_cols: u32,
    /// Number of rows in the tile.
    pub tile_rows: u32,
    /// E-graph Id for the row base offset in device memory.
    /// For `input_0` this is `wg_row * 16`, for `input_1` it's `k_outer * 16`.
    pub device_row_base: Option<Id>,
    /// E-graph Id for the column base offset in device memory.
    /// For `input_0` this is `k_outer * 16`, for `input_1` it's `wg_col * 16`.
    pub device_col_base: Option<Id>,
    /// Row stride in device memory (e.g., 64 for a 64-column matrix).
    pub device_row_stride: u32,
}

/// Split a dispatch node's children into input ids and state-threaded output
/// value/address pairs.
pub(super) fn dispatch_body_pairs(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    node: &TensorIr,
) -> Option<(usize, Vec<(Id, Id)>)> {
    let TensorIr::Dispatch(DispatchNode::Dispatch {
        num_inputs,
        children_list: children,
        ..
    }) = node
    else {
        return None;
    };

    let num_inputs = *num_inputs as usize;
    let pairs_start = num_inputs;
    let children = extract_list(egraph, *children);
    // Body layout is `[inputs (num_inputs), (value, addr) pairs ...]`.
    // Output arity is derived structurally from the children count.
    let body_len = children.len().saturating_sub(pairs_start);
    if !body_len.is_multiple_of(2) {
        return None;
    }
    let num_outputs = body_len / 2;

    let pairs = (0..num_outputs)
        .map(|i| {
            (
                children[pairs_start + i * 2],
                children[pairs_start + i * 2 + 1],
            )
        })
        .collect();
    Some((num_inputs, pairs))
}

pub(crate) fn dispatch_node_is_state_threaded(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    node: &TensorIr,
) -> bool {
    fn has_state_threading(
        egraph: &EGraph<TensorIr, TensorAnalysis>,
        id: Id,
        seen: &mut HashSet<Id>,
    ) -> bool {
        let canonical = egraph.find(id);
        if !seen.insert(canonical) {
            return false;
        }

        egraph[canonical].iter().any(|node| {
            matches!(
                node,
                TensorIr::Simd(
                    SimdNode::Store { .. } | SimdNode::StoreIf { .. } | SimdNode::Barrier { .. }
                ) | TensorIr::Dispatch(
                    DispatchNode::Pack { children_list: _ } | DispatchNode::Extract { .. }
                )
            ) || node
                .children()
                .iter()
                .any(|child| has_state_threading(egraph, *child, seen))
        })
    }

    let Some((_, pairs)) = dispatch_body_pairs(egraph, node) else {
        return false;
    };

    pairs
        .iter()
        .all(|(body_id, _)| has_state_threading(egraph, *body_id, &mut HashSet::new()))
}

/// Maximum inner-K iterations to unroll into straight-line stores when
/// `LoweringOptions::unroll` is enabled. When unrolling is disabled the
/// threshold is treated as 0 so the inner-K loop is always emitted as a
/// `Theta`, keeping the IR compact at a small perf cost.
const FUSED_INNER_UNROLL_THRESHOLD: u32 = 32;

#[inline]
pub(super) const fn fused_inner_unroll_threshold(lowering: &LoweringOptions) -> u32 {
    if lowering.unroll {
        FUSED_INNER_UNROLL_THRESHOLD
    } else {
        0
    }
}

/// Compute the device address for a cooperative load given a flat threadgroup index.
pub(super) fn compute_device_addr(
    flat_tg_idx: Id,
    buf_info: &ThreadgroupTileInfo,
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
) -> Id {
    if let (Some(row_base), Some(col_base)) = (buf_info.device_row_base, buf_info.device_col_base) {
        let tile_cols = buf_info.tile_cols;
        let device_stride = buf_info.device_row_stride;

        let tile_cols_lit =
            add_and_choose(egraph, chosen, TensorIr::Const(ScalarValue::U32(tile_cols)));
        let device_stride_lit = add_and_choose(
            egraph,
            chosen,
            TensorIr::Const(ScalarValue::U32(device_stride)),
        );

        let row_in_tile = add_and_choose(
            egraph,
            chosen,
            TensorIr::BinOp(BinaryOp::Div, [flat_tg_idx, tile_cols_lit]),
        );
        let col_in_tile = add_and_choose(
            egraph,
            chosen,
            TensorIr::BinOp(BinaryOp::Mod, [flat_tg_idx, tile_cols_lit]),
        );

        let device_row = add_and_choose(
            egraph,
            chosen,
            TensorIr::BinOp(BinaryOp::Add, [row_base, row_in_tile]),
        );
        let device_col = add_and_choose(
            egraph,
            chosen,
            TensorIr::BinOp(BinaryOp::Add, [col_base, col_in_tile]),
        );

        let row_times_stride = add_and_choose(
            egraph,
            chosen,
            TensorIr::BinOp(BinaryOp::Mul, [device_row, device_stride_lit]),
        );
        add_and_choose(
            egraph,
            chosen,
            TensorIr::BinOp(BinaryOp::Add, [row_times_stride, device_col]),
        )
    } else if let Some(row_base) = buf_info.device_row_base {
        add_and_choose(
            egraph,
            chosen,
            TensorIr::BinOp(BinaryOp::Add, [row_base, flat_tg_idx]),
        )
    } else {
        flat_tg_idx
    }
}

/// Collect threadgroup buffer info reachable from an e-graph root.
/// Returns `(tg_name, device_name, region_size)` triples.
#[must_use]
pub fn collect_threadgroup_tile_info_from_egraph(
    root: Id,
    egraph: &EGraph<TensorIr, TensorAnalysis>,
) -> Vec<(BufferRef, BufferRef, u32)> {
    // Walk the e-graph from root, find Load(Threadgroup(name), _) nodes.
    // For each, determine the corresponding device buffer and region size.
    let mut visited = HashSet::new();
    let mut results = Vec::new();
    collect_tg_loads_rec(root, egraph, &mut visited, &mut results);
    results
}

pub(super) fn collect_tg_loads_rec(
    id: Id,
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    visited: &mut HashSet<Id>,
    results: &mut Vec<(BufferRef, BufferRef, u32)>,
) {
    let canonical = egraph.find(id);
    if !visited.insert(canonical) {
        return;
    }

    for node in egraph[canonical].iter() {
        if let TensorIr::Simd(SimdNode::Load {
            tier: MemTier::Threadgroup(tg_name),
            children,
        }) = node
        {
            // Device-tier counterpart of a threadgroup buffer is the same
            // `BufferRef` — `MemTier` encodes the tier, so the underlying
            // buffer slot is shared.
            let dev_name = *tg_name;

            // Estimate region size from address expression constants.
            let region_size = estimate_region_size(children[0], egraph);

            // Avoid duplicates
            if !results.iter().any(|(tg, _, _)| *tg == *tg_name) {
                results.push((*tg_name, dev_name, region_size));
            }
        }

        // Recurse into children
        for child in node.children() {
            collect_tg_loads_rec(*child, egraph, visited, results);
        }
    }
}

/// Estimate the region size for a threadgroup buffer by walking the address
/// expression and finding multiplication constants that indicate the range.
#[must_use]
pub fn estimate_region_size(addr: Id, egraph: &EGraph<TensorIr, TensorAnalysis>) -> u32 {
    let mut visited = HashSet::new();
    let mut max_const: u32 = 1;
    estimate_region_rec(addr, egraph, &mut visited, &mut max_const);
    // Use the largest constant as a conservative region size estimate.
    // In practice this captures tile dimensions like 16, 32, etc.
    max_const
}

pub(super) fn estimate_region_rec(
    id: Id,
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    visited: &mut HashSet<Id>,
    max_const: &mut u32,
) {
    let canonical = egraph.find(id);
    if !visited.insert(canonical) {
        return;
    }

    for node in egraph[canonical].iter() {
        if let TensorIr::Const(ScalarValue::U32(v)) = node
            && *v > *max_const
        {
            *max_const = *v;
        }
        for child in node.children() {
            estimate_region_rec(*child, egraph, visited, max_const);
        }
    }
}

pub(super) fn get_u32_lit_from_egraph(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    id: Id,
) -> Option<u32> {
    let canonical = egraph.find(id);
    egraph[canonical].iter().find_map(|node| {
        if let TensorIr::Const(ScalarValue::U32(v)) = node {
            Some(*v)
        } else {
            None
        }
    })
}

mod cooperative;
mod decompose;
mod rewrite;
mod substitute;

pub(super) use cooperative::*;
use decompose::*;
pub(crate) use rewrite::rewrite_dispatch_node;
use substitute::*;
