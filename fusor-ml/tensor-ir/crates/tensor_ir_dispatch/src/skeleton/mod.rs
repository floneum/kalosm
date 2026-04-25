use std::collections::{HashMap, HashSet};
use std::fmt;

use egg::{CostFunction, EGraph, Extractor, Id, Language, RecExpr};

use crate::extractor::{BeamConfig, beam_extract_candidates};

use crate::analysis::TensorAnalysis;
use crate::language::{
    DispatchNode, HighLevelNode, SimdNode, TensorIr, add_list, extract_list, extract_recexpr_list,
};
use crate::types::{
    BinaryOp, BufferRef, DType, DeviceProfile, LoweringOptions, MemTier, ScalarValue, VarRef,
};
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

/// Same as [`dtype_bytes_for_device_buffer`], but scans a `RecExpr` instead of
/// an e-graph (used inside post-extraction passes that operate on extracted
/// nodes only).
pub(super) fn dtype_bytes_for_device_buffer_in_recexpr(
    nodes: &[TensorIr],
    dev_name: BufferRef,
) -> u32 {
    let Some(target_id) = dev_name.input_index() else {
        return DType::F32.byte_size();
    };
    for node in nodes {
        if let TensorIr::HighLevel(HighLevelNode::Input { id, dtype, .. }) = node
            && *id == target_id
        {
            return dtype.byte_size();
        }
    }
    DType::F32.byte_size()
}

/// A (value, address) pair for one output element of a dispatch.
#[derive(Debug, Clone)]
pub struct OutputElement {
    /// The e-graph Id representing the output value (reduction result).
    pub value_id: Id,
    /// The e-graph Id for the output buffer address (flat index).
    pub addr_id: Id,
}

/// Information about a single GPU dispatch (kernel launch).
#[derive(Debug, Clone)]
pub struct DispatchInfo {
    /// Device buffer inputs to this dispatch.
    pub inputs: Vec<Id>,
    /// Number of virtual workgroups (one per simdgroup) to launch.
    /// Physical workgroups = workgroups / simdgroups.
    pub workgroups: u32,
    /// Number of simdgroups per physical workgroup.
    /// For non-tiled dispatches this is 1 (32 threads per workgroup).
    /// For tiled dispatches this can be up to `MAX_SIMDGROUPS` (8),
    /// giving up to 256 threads per workgroup for cooperative loading.
    pub simdgroups: u32,
    /// Output elements. For scalar (non-register-blocked) dispatches,
    /// this contains exactly one element. For register-blocked dispatches,
    /// this contains `reg_m` × `reg_n` elements.
    pub outputs: Vec<OutputElement>,
    /// Threadgroup buffers used by this dispatch, with their sizes.
    pub tg_buffers: Vec<TgBufferInfo>,
    /// Canonical e-class of this dispatch's semantic output — i.e. the class
    /// that the phase-1 rules unioned with the `Dispatch` node (the original
    /// `Reduce`/`Elementwise`). Downstream dispatches reference their inputs
    /// through this class (via `find_underlying_input`), not through the
    /// internal `outputs[0].value_id` (which is the `Theta`/`ReduceSimd`
    /// node). Runtime uses it to wire dispatch-to-dispatch buffer flow.
    pub semantic_output_id: Id,
    /// Index of the enclosing logical pipeline, if this dispatch came from a
    /// `DispatchNode::Pipeline` root. `None` means the dispatch is standalone
    /// or part of an unfused `Seq`.
    pub pipeline_index: Option<usize>,
    /// Zero-based stage number within `pipeline_index`.
    pub pipeline_stage: usize,
}

impl DispatchInfo {
    /// Sum of `tg_buffers[i].size` (in *elements*) for this dispatch.
    #[must_use]
    pub fn threadgroup_elements(&self) -> u64 {
        self.tg_buffers
            .iter()
            .map(|b| u64::from(b.size))
            .sum::<u64>()
    }

    /// Threadgroup memory consumption in bytes, summing each buffer's
    /// per-element dtype size from [`TgBufferInfo::dtype_bytes`].
    #[must_use]
    pub fn threadgroup_bytes(&self) -> u64 {
        self.tg_buffers.iter().map(TgBufferInfo::total_bytes).sum()
    }
}

impl DispatchProgram {
    /// Maximum threadgroup memory consumed by any single dispatch in this
    /// program, in bytes. Used by the staged pipeline to enforce
    /// `DeviceProfile::max_threadgroup_bytes`.
    #[must_use]
    pub fn peak_threadgroup_bytes(&self) -> u64 {
        self.dispatches
            .iter()
            .map(DispatchInfo::threadgroup_bytes)
            .max()
            .unwrap_or(0)
    }
}

/// A complete GPU program extracted from the e-graph.
///
/// Contains the e-graph (for looking up pure expressions)
/// and an ordered sequence of dispatches with their skeletons.
#[derive(Debug)]
pub struct DispatchProgram {
    /// The e-graph containing all pure expressions.
    pub egraph: EGraph<TensorIr, TensorAnalysis>,
    /// Ordered sequence of GPU dispatches.
    pub dispatches: Vec<DispatchInfo>,
    /// Logical fused pipelines recovered from the extracted program.
    ///
    /// Compatible linear plain pipelines may already collapse into one
    /// executable dispatch at skeleton-build time; more complex pipelines are
    /// preserved here as ordered stage groupings.
    pub pipelines: Vec<PipelineInfo>,
    /// Root e-graph Ids representing program outputs.
    pub outputs: Vec<Id>,
    /// Canonical e-graph Id → the specific extracted node.
    ///
    /// When multiple equivalent nodes occupy the same e-class (e.g., a
    /// `Load(Device)` and a `Load(Threadgroup)` are unified), this map
    /// records which variant the extractor chose so codegen uses the
    /// correct one.
    pub chosen_nodes: HashMap<Id, TensorIr>,
    /// Device profile this program was lowered for. Codegen and runtime read
    /// `device.simd_width` instead of a global constant so backends can target
    /// hardware with non-32 lane widths.
    pub device: DeviceProfile,
}

/// Ordered stages that originated from one extracted `DispatchNode::Pipeline`.
#[derive(Debug, Clone)]
pub struct PipelineInfo {
    /// Indices into `DispatchProgram::dispatches` in execution order.
    pub dispatch_indices: Vec<usize>,
    /// Semantic output ids exposed by the final stage(s) of this pipeline.
    pub outputs: Vec<Id>,
}

/// Metadata about a threadgroup buffer used in a dispatch.
#[derive(Debug, Clone)]
pub struct TgBufferInfo {
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
    /// Per-simdgroup elements in this tg buffer (for read offset computation).
    /// When >0, codegen adds `Index(Simdgroup) * sg_read_stride` to every
    /// `Load(Threadgroup(tg_name))` address. This is for scaled buffers where
    /// different simdgroups read different slices.
    pub sg_read_stride: u32,
}

impl TgBufferInfo {
    /// Total bytes this buffer occupies in threadgroup memory.
    #[must_use]
    pub const fn total_bytes(&self) -> u64 {
        (self.size as u64) * (self.dtype_bytes as u64)
    }
}

/// Insert a cooperative load for a threadgroup memory region.
///
/// This is the post-extraction pass for Rule 8: when extraction picks
/// a `Load(Threadgroup(...))` but no Store to that region exists above it,
/// this function inserts a cooperative load loop + Barrier.
///
/// When `buf_info` is provided with device address mapping (row/col bases,
/// stride), the cooperative load computes correct device addresses for
/// non-contiguous tile elements. Without mapping info, falls back to a
/// simple flat copy (only correct when tile elements are contiguous).
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

pub(super) fn compute_simdgroups_for_independent_subgroups(
    outputs: &[OutputElement],
    workgroups: u32,
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen: &HashMap<Id, TensorIr>,
    device: &DeviceProfile,
) -> u32 {
    if outputs.is_empty() || workgroups <= 1 {
        return 1;
    }

    for output in outputs {
        let mut visited = HashSet::new();
        let Ok(()) =
            analyze_independent_subgroup_subtree(output.value_id, egraph, chosen, &mut visited)
        else {
            return 1;
        };
    }

    let mut sg = device.max_simdgroups.min(workgroups);
    while sg > 1 && !workgroups.is_multiple_of(sg) {
        sg -= 1;
    }
    sg.max(1)
}

pub(super) fn analyze_independent_subgroup_subtree(
    id: Id,
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen: &HashMap<Id, TensorIr>,
    visited: &mut HashSet<Id>,
) -> Result<(), ()> {
    let canonical = egraph.find(id);
    if !visited.insert(canonical) {
        return Ok(());
    }

    let Some(node) = select_substitution_node(egraph, chosen, canonical) else {
        return Err(());
    };

    if let TensorIr::Simd(
        SimdNode::Load {
            tier: MemTier::Threadgroup(_),
            ..
        }
        | SimdNode::Store {
            tier: MemTier::Threadgroup(_),
            ..
        }
        | SimdNode::StoreIf {
            tier: MemTier::Threadgroup(_),
            ..
        }
        | SimdNode::Barrier { .. },
    ) = &node
    {
        return Err(());
    }

    for child in node.children() {
        analyze_independent_subgroup_subtree(*child, egraph, chosen, visited)?;
    }

    Ok(())
}

/// Compute the device address for a cooperative load given a flat threadgroup index.
pub(super) fn compute_device_addr(
    flat_tg_idx: Id,
    buf_info: &TgBufferInfo,
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
pub fn collect_tg_buffer_info_from_egraph(
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

// ═══════════════════════════════════════════════════════════════
// RecExpr → DispatchProgram pipeline
// ═══════════════════════════════════════════════════════════════

/// Build a `DispatchProgram` from an extracted `RecExpr`.
///
/// Walks the extracted expression to find Dispatch nodes. For dispatches
/// whose body is a nested Theta with threadgroup loads (the K-tiled pattern),
/// builds a skeleton with cooperative loading (Store + Barrier per K-tile).
#[must_use]
pub fn build_dispatch_program_from_extracted(
    extracted: &RecExpr<TensorIr>,
    egraph: EGraph<TensorIr, TensorAnalysis>,
    device: &DeviceProfile,
    lowering: &LoweringOptions,
) -> DispatchProgram {
    let nodes = extracted.as_ref();
    let mut egraph = egraph;
    let mut dispatches = Vec::new();
    let mut pipelines = Vec::new();
    let mut outputs = Vec::new();
    let mut chosen = HashMap::new();

    // Find the root node (last in RecExpr)
    let root_idx = nodes.len() - 1;
    if let Some(root_node) = nodes.get(root_idx) {
        match root_node {
            TensorIr::Dispatch(DispatchNode::Dispatch { .. }) => {
                let dispatch_order = collect_dispatch_order(nodes, root_idx);
                for idx in dispatch_order {
                    if let Some(dispatch) = build_single_dispatch(
                        nodes,
                        idx,
                        &mut egraph,
                        &mut chosen,
                        device,
                        lowering,
                    ) {
                        if idx == root_idx {
                            outputs.push(dispatch.outputs[0].value_id);
                        }
                        dispatches.push(dispatch);
                    }
                }
            }
            TensorIr::Dispatch(DispatchNode::Seq(list_id)) => {
                // Multiple dispatches
                for child_id in extract_recexpr_list(nodes, *list_id) {
                    let child_idx = usize::from(child_id);
                    if matches!(
                        &nodes[child_idx],
                        TensorIr::Dispatch(DispatchNode::Dispatch { .. })
                    ) && let Some(dispatch) = build_single_dispatch(
                        nodes,
                        child_idx,
                        &mut egraph,
                        &mut chosen,
                        device,
                        lowering,
                    ) {
                        outputs.push(dispatch.outputs[0].value_id);
                        dispatches.push(dispatch);
                    }
                }
            }
            TensorIr::Dispatch(DispatchNode::Pipeline(list_id)) => {
                let stage_ids = extract_recexpr_list(nodes, *list_id);
                let direct_stage_dispatches = stage_ids
                    .iter()
                    .map(|child_id| {
                        let child_idx = usize::from(*child_id);
                        if !matches!(
                            &nodes[child_idx],
                            TensorIr::Dispatch(DispatchNode::Dispatch { .. })
                        ) {
                            return None;
                        }
                        build_single_dispatch(
                            nodes,
                            child_idx,
                            &mut egraph,
                            &mut chosen,
                            device,
                            lowering,
                        )
                    })
                    .collect::<Option<Vec<_>>>();

                if let Some(mut fused_dispatch) =
                    direct_stage_dispatches.as_deref().and_then(|stages| {
                        try_fuse_linear_pipeline_dispatches(
                            &mut egraph,
                            &mut chosen,
                            stages,
                            device,
                        )
                    })
                {
                    let output_value = fused_dispatch.outputs[0].value_id;
                    fused_dispatch.pipeline_index = Some(0);
                    fused_dispatch.pipeline_stage = 0;
                    outputs.push(output_value);
                    dispatches.push(fused_dispatch);
                    pipelines.push(PipelineInfo {
                        dispatch_indices: vec![0],
                        outputs: outputs.clone(),
                    });
                } else {
                    let mut produced = HashSet::new();
                    let mut stage_dispatch_indices = Vec::new();
                    for (stage_index, child_id) in stage_ids.iter().enumerate() {
                        let child_idx = usize::from(*child_id);
                        if !matches!(
                            &nodes[child_idx],
                            TensorIr::Dispatch(DispatchNode::Dispatch { .. })
                        ) {
                            continue;
                        }
                        for idx in collect_dispatch_order(nodes, child_idx) {
                            if let Some(mut dispatch) = build_single_dispatch(
                                nodes,
                                idx,
                                &mut egraph,
                                &mut chosen,
                                device,
                                lowering,
                            ) {
                                let sem = egraph.find(dispatch.semantic_output_id);
                                if !produced.insert(sem) {
                                    continue;
                                }
                                dispatch.pipeline_index = Some(0);
                                dispatch.pipeline_stage = stage_dispatch_indices.len();
                                if idx == child_idx && stage_index + 1 == stage_ids.len() {
                                    outputs.push(dispatch.outputs[0].value_id);
                                }
                                stage_dispatch_indices.push(dispatches.len());
                                dispatches.push(dispatch);
                            }
                        }
                    }
                    if !stage_dispatch_indices.is_empty() {
                        pipelines.push(PipelineInfo {
                            dispatch_indices: stage_dispatch_indices,
                            outputs: outputs.clone(),
                        });
                    }
                }
            }
            _ => {
                // Non-dispatch root: wrap in a trivial dispatch
                let root_id = egraph.add(root_node.clone());
                outputs.push(root_id);
            }
        }
    }

    // Phase-1 lowering emits TG Load tile-local addresses in terms of
    // `wg_in_tile = Mod(Workgroup, wgs_per_tile)` — the virtual-wg-in-tile
    // index, which pre-promotion was 1:1 with simdgroups. After
    // simdgroup promotion (num_simdgroups > 1) those addresses stop
    // tracking the per-physical-wg buffer layout the cooperative Store
    // establishes: each physical workgroup has its own TG buffer, and the
    // intra-physical variation of `Workgroup` is exactly the `Simdgroup`
    // binder. Rewrite every TG Load's `tg_addr` by substituting
    // `Workgroup → Simdgroup` so the Load indexes the buffer the Store
    // actually wrote into.
    for dispatch in dispatches.iter_mut() {
        if dispatch.simdgroups > 1 {
            for output in dispatch.outputs.iter_mut() {
                output.value_id = rewrite_tg_load_workgroup_to_simdgroup(
                    output.value_id,
                    &mut egraph,
                    &mut chosen,
                );
            }
        }
    }

    // Reject candidates whose TG Loads can *provably* address past their
    // declared buffer under any choice of thread vars within the
    // dispatch's workgroup range. `collect_tg_buffer_info_for_load` sizes
    // the buffer for the worst-case simdgroup promotion; after the
    // Workgroup→Simdgroup rewrite above, a correctly-built Load yields
    // `interval.hi < buf.size`. If it can still be proven OOB, the rule
    // emitted an expression outside the lowering contract — skip the
    // candidate and let the extractor try another.
    if !dispatches.is_empty()
        && dispatches
            .iter()
            .any(|d| !all_tg_loads_in_bounds(&egraph, &chosen, d, device))
    {
        dispatches.clear();
        pipelines.clear();
        outputs.clear();
    }

    // Materialize any intermediate dispatches that the beam extractor left
    // out. Softmax-style fusion can produce a RecExpr where the final
    // dispatch's input is the eclass of an intermediate `Elementwise` whose
    // buffer was never produced — the consumer has no produced buffer to read.
    // Extract a dispatch-form representative from the e-graph for each such
    // class so the program chains end-to-end.
    materialize_missing_dispatches(&mut egraph, &mut dispatches, &mut chosen, device, lowering);
    if !dispatches.is_empty()
        && dispatches
            .iter()
            .any(|d| !all_tg_loads_in_bounds(&egraph, &chosen, d, device))
    {
        dispatches.clear();
        pipelines.clear();
        outputs.clear();
    }

    // Late-canonicalize: further `add_recexpr_subtree` calls during later
    // dispatches may trigger unions that change the canonical of IDs we
    // recorded earlier. Rebuild once and re-project every ID that
    // cross-dispatch lookups will key on.
    egraph.rebuild();
    for dispatch in dispatches.iter_mut() {
        dispatch.semantic_output_id = egraph.find(dispatch.semantic_output_id);
        for input in dispatch.inputs.iter_mut() {
            *input = egraph.find(*input);
        }
    }
    for output in outputs.iter_mut() {
        *output = egraph.find(*output);
    }
    for pipeline in pipelines.iter_mut() {
        for output in pipeline.outputs.iter_mut() {
            *output = egraph.find(*output);
        }
    }
    // Reject candidates where any dispatch's input doesn't resolve against
    // either an external `HighLevel::Input`, an equivalent `Load(Device(..))`,
    // or an earlier dispatch's `semantic_output_id`. Otherwise codegen or the
    // runtime would panic downstream; the extractor will try the next
    // candidate in the beam.
    if !dispatches.is_empty() && !all_inputs_resolvable(&egraph, &dispatches) {
        dispatches.clear();
        pipelines.clear();
        outputs.clear();
    }

    if pipelines.is_empty()
        && dispatches.len() > 1
        && let Some(mut fused_dispatch) =
            try_fuse_linear_pipeline_dispatches(&mut egraph, &mut chosen, &dispatches, device)
    {
        let output_value = fused_dispatch.outputs[0].value_id;
        fused_dispatch.pipeline_index = Some(0);
        fused_dispatch.pipeline_stage = 0;
        dispatches = vec![fused_dispatch];
        outputs = vec![output_value];
        pipelines = vec![PipelineInfo {
            dispatch_indices: vec![0],
            outputs: outputs.clone(),
        }];
    }

    // Reject candidates whose output stores can provably address past the
    // buffer the runtime allocates for that dispatch. Such programs are not
    // runnable even if their input chain and TG accesses are otherwise valid.
    if !dispatches.is_empty()
        && dispatches
            .iter()
            .any(|d| !all_output_addrs_in_bounds(&egraph, &chosen, d, device))
    {
        dispatches.clear();
        pipelines.clear();
        outputs.clear();
    }
    DispatchProgram {
        egraph,
        dispatches,
        pipelines,
        outputs,
        chosen_nodes: chosen,
        device: *device,
    }
}

/// Build a `DispatchInfo` from a Dispatch node in a `RecExpr`. The
/// output arity is the length of `body_addr_pairs` — no separate
/// `reg_m` / `reg_n` tag.
pub(crate) struct RecexprDispatchLayout {
    workgroups: u32,
    input_ids: Vec<Id>,
    body_addr_pairs: Vec<(usize, Id)>,
}

impl RecexprDispatchLayout {
    /// Number of output pairs (= per-lane register block size).
    pub(crate) fn num_outputs_u32(&self) -> u32 {
        u32::try_from(self.body_addr_pairs.len()).expect("num outputs fits in u32")
    }
}

pub(super) fn extract_recexpr_dispatch_layout(
    nodes: &[TensorIr],
    dispatch_idx: usize,
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
) -> Option<RecexprDispatchLayout> {
    let TensorIr::Dispatch(DispatchNode::Dispatch {
        workgroups,
        num_inputs,
        children_list: children,
    }) = &nodes[dispatch_idx]
    else {
        return None;
    };

    let num_inputs = *num_inputs as usize;
    let workgroups = *workgroups;
    let children = extract_recexpr_list(nodes, *children);
    // Output arity = (children.len() - num_inputs) / 2. Structural, no
    // separate reg_m/reg_n fields to consult.
    let body_len = children.len().saturating_sub(num_inputs);
    if !body_len.is_multiple_of(2) {
        return None;
    }
    let num_outputs = body_len / 2;

    // Re-add input nodes to the e-graph
    let input_ids: Vec<Id> = children[..num_inputs]
        .iter()
        .map(|child_id| add_recexpr_subtree(nodes, usize::from(*child_id), egraph, chosen))
        .collect();

    let pairs_start = num_inputs;
    let mut body_addr_pairs: Vec<(usize, Id)> = Vec::with_capacity(num_outputs);
    for i in 0..num_outputs {
        let body_idx = usize::from(children[pairs_start + i * 2]);
        let addr_idx = usize::from(children[pairs_start + i * 2 + 1]);
        let output_addr = add_recexpr_subtree(nodes, addr_idx, egraph, chosen);
        body_addr_pairs.push((body_idx, output_addr));
    }

    Some(RecexprDispatchLayout {
        workgroups,
        input_ids,
        body_addr_pairs,
    })
}

pub(super) fn nested_recexpr_tiled_counts(
    nodes: &[TensorIr],
    body_idx: usize,
) -> Option<(u32, u32)> {
    let TensorIr::Simd(SimdNode::Theta {
        children: [_init_id, outer_count_id, inner_id],
        ..
    }) = &nodes[body_idx]
    else {
        return None;
    };
    let inner_idx = usize::from(*inner_id);
    let TensorIr::Simd(SimdNode::Theta {
        children: [_inner_init, inner_count_id, _inner_update],
        ..
    }) = &nodes[inner_idx]
    else {
        return None;
    };
    if !subtree_has_tg_loads(nodes, inner_idx) {
        return None;
    }
    Some((
        get_u32_lit(nodes, usize::from(*outer_count_id))?,
        get_u32_lit(nodes, usize::from(*inner_count_id))?,
    ))
}

pub(super) fn collect_recexpr_tiled_outputs(
    nodes: &[TensorIr],
    body_addr_pairs: &[(usize, Id)],
    outer_k: u32,
    tile_k: u32,
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
) -> Option<Vec<TiledOutput>> {
    fn embedded_state_layout(
        nodes: &[TensorIr],
        init_id: Id,
        update_id: Id,
        output_index: usize,
    ) -> (usize, Option<usize>) {
        let init_arity = match &nodes[usize::from(init_id)] {
            TensorIr::Dispatch(DispatchNode::Pack { children_list }) => {
                Some(extract_recexpr_list(nodes, *children_list).len())
            }
            _ => None,
        };
        let update_arity = match &nodes[usize::from(update_id)] {
            TensorIr::Dispatch(DispatchNode::Pack { children_list }) => {
                Some(extract_recexpr_list(nodes, *children_list).len())
            }
            _ => None,
        };
        if output_index == 0 && init_arity == Some(4) && update_arity == Some(4) {
            (2, Some(3))
        } else {
            (output_index, None)
        }
    }

    let mut tiled_outputs = Vec::with_capacity(body_addr_pairs.len());
    for (output_index, &(pair_body_idx, pair_addr)) in body_addr_pairs.iter().enumerate() {
        let TensorIr::Simd(SimdNode::Theta {
            children: [pair_init_id, pair_outer_count_id, pair_inner_id],
            ..
        }) = &nodes[pair_body_idx]
        else {
            return None;
        };
        let pair_inner_idx = usize::from(*pair_inner_id);
        let TensorIr::Simd(SimdNode::Theta {
            children: [_pair_inner_init, pair_inner_count_id, pair_update],
            ..
        }) = &nodes[pair_inner_idx]
        else {
            return None;
        };
        if get_u32_lit(nodes, usize::from(*pair_outer_count_id)) != Some(outer_k)
            || get_u32_lit(nodes, usize::from(*pair_inner_count_id)) != Some(tile_k)
        {
            return None;
        }

        // All outputs share the canonical `acc(0)` accumulator name.
        // Per-output distinction is encoded by which `(value, addr)`
        // pair this output came from — its position in the children
        // list — not by a separately-named accumulator ref.
        let acc_name = VarRef::acc(0);
        let (result_slot, state_slot) =
            embedded_state_layout(nodes, *pair_init_id, *pair_update, output_index);
        tiled_outputs.push(TiledOutput {
            acc_name,
            value_id: add_recexpr_subtree(nodes, pair_inner_idx, egraph, chosen),
            update_id: Some(add_recexpr_subtree(
                nodes,
                usize::from(*pair_update),
                egraph,
                chosen,
            )),
            init_id: add_recexpr_subtree(nodes, usize::from(*pair_init_id), egraph, chosen),
            addr_id: pair_addr,
            result_slot,
            state_slot,
        });
    }
    Some(tiled_outputs)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_tiled_dispatch_info(
    layout: &RecexprDispatchLayout,
    outer_count: u32,
    tile_k: Option<u32>,
    tiled_outputs: &[TiledOutput],
    tg_buffers: &[TgBufferInfo],
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    device: &DeviceProfile,
    lowering: &LoweringOptions,
) -> DispatchInfo {
    let simdgroups = compute_simdgroups_for_tiled(
        tg_buffers,
        layout.workgroups,
        layout.num_outputs_u32(),
        device,
    );
    let (outputs, scaled_tg) = build_tiled_outputs(
        outer_count,
        tile_k,
        tiled_outputs,
        tg_buffers,
        simdgroups,
        egraph,
        chosen,
        device,
        lowering,
    );
    DispatchInfo {
        inputs: layout.input_ids.clone(),
        workgroups: layout.workgroups,
        simdgroups,
        outputs,
        tg_buffers: scaled_tg,
        semantic_output_id: Id::from(0),
        pipeline_index: None,
        pipeline_stage: 0,
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_recovered_tiled_dispatch(
    layout: &RecexprDispatchLayout,
    source_id: Id,
    output_addr: Id,
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    device: &DeviceProfile,
    lowering: &LoweringOptions,
) -> Option<DispatchInfo> {
    let mut recovered_chosen = HashMap::new();
    let (outer_k, inner_theta_id, init_id, tg_bufs) =
        find_nested_theta_in_egraph(source_id, egraph, &mut recovered_chosen, device)?;
    chosen.extend(recovered_chosen);
    let tiled_outputs = [TiledOutput {
        acc_name: VarRef::acc(0),
        value_id: inner_theta_id,
        update_id: None,
        init_id,
        addr_id: output_addr,
        result_slot: 0,
        state_slot: None,
    }];
    Some(build_tiled_dispatch_info(
        layout,
        outer_k,
        None,
        &tiled_outputs,
        &tg_bufs,
        egraph,
        chosen,
        device,
        lowering,
    ))
}

pub(super) fn try_build_recexpr_tiled_dispatch(
    nodes: &[TensorIr],
    layout: &RecexprDispatchLayout,
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    device: &DeviceProfile,
    lowering: &LoweringOptions,
) -> Option<DispatchInfo> {
    let (body_idx, output_addr) = layout.body_addr_pairs[0];
    let (outer_k, tile_k) = nested_recexpr_tiled_counts(nodes, body_idx)?;
    let outer_theta_id = add_recexpr_subtree(nodes, body_idx, egraph, chosen);
    let tiled_outputs = collect_recexpr_tiled_outputs(
        nodes,
        &layout.body_addr_pairs,
        outer_k,
        tile_k,
        egraph,
        chosen,
    )?;
    let tg_buffers = collect_tg_buffer_info_with_device_addrs(
        &tiled_outputs
            .iter()
            .map(|output| output.value_id)
            .collect::<Vec<_>>(),
        egraph,
        chosen,
        tile_k,
        device,
    )?;
    let inner_theta_id = tiled_outputs[0].value_id;
    let coherent = selected_subtree_has_coherent_k_tile_stride(
        egraph,
        chosen,
        inner_theta_id,
        inner_theta_id,
        tile_k,
    );
    if coherent {
        return Some(build_tiled_dispatch_info(
            layout,
            outer_k,
            Some(tile_k),
            &tiled_outputs,
            &tg_buffers,
            egraph,
            chosen,
            device,
            lowering,
        ));
    }
    build_recovered_tiled_dispatch(
        layout,
        outer_theta_id,
        output_addr,
        egraph,
        chosen,
        device,
        lowering,
    )
}

pub(super) fn build_plain_dispatch_from_recexpr(
    nodes: &[TensorIr],
    layout: &RecexprDispatchLayout,
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    device: &DeviceProfile,
) -> DispatchInfo {
    let outputs = layout
        .body_addr_pairs
        .iter()
        .map(|(pair_body_idx, pair_addr)| OutputElement {
            value_id: add_recexpr_subtree(nodes, *pair_body_idx, egraph, chosen),
            addr_id: *pair_addr,
        })
        .collect::<Vec<_>>();
    let simdgroups = compute_simdgroups_for_independent_subgroups(
        &outputs,
        layout.workgroups,
        egraph,
        chosen,
        device,
    );
    DispatchInfo {
        inputs: layout.input_ids.clone(),
        workgroups: layout.workgroups,
        simdgroups,
        outputs,
        tg_buffers: Vec::new(),
        semantic_output_id: Id::from(0),
        pipeline_index: None,
        pipeline_stage: 0,
    }
}

#[derive(Clone, Copy)]
enum FusedPipelineInputBinding {
    External { slot: u32 },
    Intermediate { producer_stage: usize },
}

fn try_fuse_linear_pipeline_dispatches(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    stage_dispatches: &[DispatchInfo],
    device: &DeviceProfile,
) -> Option<DispatchInfo> {
    if stage_dispatches.len() < 2 {
        return None;
    }

    let workgroups = stage_dispatches[0].workgroups;
    if stage_dispatches.iter().any(|dispatch| {
        dispatch.workgroups != workgroups
            || dispatch.outputs.len() != 1
            || !dispatch.tg_buffers.is_empty()
            || dispatch.simdgroups != 1
    }) {
        return None;
    }

    let mut stage_output_to_index = HashMap::new();
    let mut external_inputs = Vec::new();
    let mut external_slot_by_id = HashMap::new();
    let mut stage_bindings = Vec::with_capacity(stage_dispatches.len());
    let mut saw_intermediate = false;

    for (stage_index, dispatch) in stage_dispatches.iter().enumerate() {
        let mut bindings = Vec::with_capacity(dispatch.inputs.len());
        let mut stage_intermediate = 0usize;
        for input in &dispatch.inputs {
            let canonical = egraph.find(*input);
            if let Some(&producer_stage) = stage_output_to_index.get(&canonical) {
                if producer_stage + 1 != stage_index {
                    return None;
                }
                stage_intermediate += 1;
                bindings.push(FusedPipelineInputBinding::Intermediate { producer_stage });
                continue;
            }

            let slot = if let Some(slot) = external_slot_by_id.get(&canonical) {
                *slot
            } else {
                let slot =
                    u32::try_from(external_inputs.len()).expect("pipeline input count fits in u32");
                external_inputs.push(canonical);
                external_slot_by_id.insert(canonical, slot);
                slot
            };
            bindings.push(FusedPipelineInputBinding::External { slot });
        }
        if stage_index > 0 && stage_intermediate != 1 {
            return None;
        }
        saw_intermediate |= stage_intermediate > 0;
        stage_bindings.push(bindings);
        stage_output_to_index.insert(egraph.find(dispatch.semantic_output_id), stage_index);
    }

    if !saw_intermediate {
        return None;
    }

    let final_stage = stage_dispatches.len() - 1;
    let mut memo = HashMap::new();
    let final_output = stage_dispatches[final_stage]
        .outputs
        .first()
        .expect("plain pipeline stage should have one output");
    let value_id = clone_fused_pipeline_subtree(
        egraph,
        chosen,
        stage_dispatches,
        &stage_bindings,
        final_stage,
        final_output.value_id,
        &mut memo,
    )?;
    let addr_id = clone_fused_pipeline_subtree(
        egraph,
        chosen,
        stage_dispatches,
        &stage_bindings,
        final_stage,
        final_output.addr_id,
        &mut memo,
    )?;

    egraph.rebuild();
    let outputs = vec![OutputElement { value_id, addr_id }];
    let simdgroups =
        compute_simdgroups_for_independent_subgroups(&outputs, workgroups, egraph, chosen, device);

    Some(DispatchInfo {
        inputs: external_inputs,
        workgroups,
        simdgroups,
        outputs,
        tg_buffers: Vec::new(),
        semantic_output_id: egraph.find(stage_dispatches[final_stage].semantic_output_id),
        pipeline_index: Some(0),
        pipeline_stage: 0,
    })
}

fn clone_fused_pipeline_subtree(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    stage_dispatches: &[DispatchInfo],
    stage_bindings: &[Vec<FusedPipelineInputBinding>],
    stage_index: usize,
    id: Id,
    memo: &mut HashMap<(usize, Id), Id>,
) -> Option<Id> {
    let canonical = egraph.find(id);
    if let Some(existing) = memo.get(&(stage_index, canonical)) {
        return Some(*existing);
    }

    let node = chosen
        .get(&canonical)
        .cloned()
        .or_else(|| egraph[canonical].iter().next().cloned())?;

    let result = match node {
        TensorIr::Simd(SimdNode::Load {
            tier: MemTier::Device(BufferRef::Input(slot)),
            children: [addr, state],
        }) => {
            let binding = *stage_bindings
                .get(stage_index)?
                .get(slot as usize)
                .expect("dispatch-local input slot should be in range");
            match binding {
                FusedPipelineInputBinding::External { slot } => {
                    let new_addr = clone_fused_pipeline_subtree(
                        egraph,
                        chosen,
                        stage_dispatches,
                        stage_bindings,
                        stage_index,
                        addr,
                        memo,
                    )?;
                    let new_state = clone_fused_pipeline_subtree(
                        egraph,
                        chosen,
                        stage_dispatches,
                        stage_bindings,
                        stage_index,
                        state,
                        memo,
                    )?;
                    let new_node = TensorIr::Simd(SimdNode::Load {
                        tier: MemTier::Device(BufferRef::Input(slot)),
                        children: [new_addr, new_state],
                    });
                    let added = egraph.add(new_node.clone());
                    let canonical_added = egraph.find(added);
                    chosen.insert(canonical_added, new_node);
                    added
                }
                FusedPipelineInputBinding::Intermediate { producer_stage } => {
                    let producer = stage_dispatches.get(producer_stage)?;
                    let producer_output = producer.outputs.first()?;
                    if egraph.find(addr) != egraph.find(producer_output.addr_id) {
                        return None;
                    }
                    let inlined = clone_fused_pipeline_subtree(
                        egraph,
                        chosen,
                        stage_dispatches,
                        stage_bindings,
                        producer_stage,
                        producer_output.value_id,
                        memo,
                    )?;
                    memo.insert((stage_index, canonical), inlined);
                    return Some(inlined);
                }
            }
        }
        TensorIr::Simd(SimdNode::Load {
            tier: MemTier::Threadgroup(_),
            ..
        })
        | TensorIr::Dispatch(DispatchNode::Dispatch { .. })
        | TensorIr::Dispatch(DispatchNode::Seq(_))
        | TensorIr::Dispatch(DispatchNode::Pipeline(_))
        | TensorIr::HighLevel(_) => {
            return None;
        }
        mut other => {
            for child in other.children_mut() {
                *child = clone_fused_pipeline_subtree(
                    egraph,
                    chosen,
                    stage_dispatches,
                    stage_bindings,
                    stage_index,
                    *child,
                    memo,
                )?;
            }
            let added = egraph.add(other.clone());
            let canonical_added = egraph.find(added);
            chosen.insert(canonical_added, other);
            added
        }
    };

    memo.insert((stage_index, canonical), result);
    Some(result)
}

/// Topologically order every `Dispatch` node reachable from `root_idx`, with
/// predecessors before successors. A predecessor is any `Dispatch` that
/// appears inside another dispatch's input subtree (phase-1 lowering inlines
/// predecessor `Dispatch` nodes into the consumer's input slot rather than
/// wiring a `Seq`, so we recover the execution order here).
fn collect_dispatch_order(nodes: &[TensorIr], root_idx: usize) -> Vec<usize> {
    let mut order = Vec::new();
    let mut visited = HashSet::new();
    visit_dispatches(nodes, root_idx, &mut visited, &mut order);
    order
}

fn visit_dispatches(
    nodes: &[TensorIr],
    idx: usize,
    visited: &mut HashSet<usize>,
    order: &mut Vec<usize>,
) {
    if !visited.insert(idx) {
        return;
    }
    match &nodes[idx] {
        TensorIr::Dispatch(DispatchNode::Dispatch {
            num_inputs,
            children_list,
            ..
        }) => {
            let children = extract_recexpr_list(nodes, *children_list);
            for child in children.iter().take(*num_inputs as usize) {
                collect_predecessor_dispatches(nodes, usize::from(*child), visited, order);
            }
            order.push(idx);
        }
        _ => {
            for child in nodes[idx].children() {
                visit_dispatches(nodes, usize::from(*child), visited, order);
            }
        }
    }
}

/// Walk a dispatch's input subtree looking for nested `Dispatch` nodes, which
/// are the predecessor dispatches in the execution graph. Non-dispatch nodes
/// (loads, restrides, inputs) are skipped since they don't carry their own
/// dispatch metadata.
fn collect_predecessor_dispatches(
    nodes: &[TensorIr],
    idx: usize,
    visited: &mut HashSet<usize>,
    order: &mut Vec<usize>,
) {
    if visited.contains(&idx) {
        return;
    }
    match &nodes[idx] {
        TensorIr::Dispatch(DispatchNode::Dispatch { .. }) => {
            visit_dispatches(nodes, idx, visited, order);
        }
        _ => {
            for child in nodes[idx].children() {
                collect_predecessor_dispatches(nodes, usize::from(*child), visited, order);
            }
        }
    }
}

/// Ensure each dispatch carries its semantic output class — the e-class the
/// phase-1 lowering rules union with the `Dispatch` node. Multi-dispatch
/// consumers reference intermediate buffers through this class (via
/// `find_underlying_input`), so the runtime keys produced-buffer lookup on
/// it rather than on the `Theta`/`ReduceSimd` value id inside `outputs[0]`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CandidateValidationReport {
    pub requested_limit: usize,
    pub raw_candidate_limit: usize,
    pub raw_candidates: usize,
    pub invalid_var_scopes: usize,
    pub empty_dispatch_programs: usize,
    pub verification_failures: usize,
    pub accepted_before_limit: usize,
    pub returned: usize,
}

/// Extract beam-search candidates that all produce well-formed
/// `DispatchProgram`s. Candidate generation is generic: we over-sample the
/// raw beam, add greedy dispatch- and composite-dispatch-biased extracts,
/// then validate each candidate by actually building the dispatch program.
/// Selection is therefore driven by e-graph analysis, dispatch shape, cost,
/// and executable lowering success rather than by workload-specific root
/// reconstruction.
#[must_use]
pub fn beam_extract_valid_candidates(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    root: Id,
    config: &BeamConfig,
    device: &DeviceProfile,
    lowering: &LoweringOptions,
    limit: usize,
) -> Vec<(f64, RecExpr<TensorIr>)> {
    beam_extract_valid_candidates_with_report(egraph, root, config, device, lowering, limit).0
}

/// Extract valid beam-search candidates and return candidate-filtering
/// diagnostics alongside the selected candidates.
#[must_use]
pub fn beam_extract_valid_candidates_with_report(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    root: Id,
    config: &BeamConfig,
    device: &DeviceProfile,
    lowering: &LoweringOptions,
    limit: usize,
) -> (Vec<(f64, RecExpr<TensorIr>)>, CandidateValidationReport) {
    if limit == 0 {
        return (
            Vec::new(),
            CandidateValidationReport {
                requested_limit: limit,
                ..CandidateValidationReport::default()
            },
        );
    }

    let mut report = CandidateValidationReport {
        requested_limit: limit,
        ..CandidateValidationReport::default()
    };

    // Over-sample from the raw beam so we have headroom when filtering drops
    // structurally-broken candidates. The cap keeps the per-call cost bounded
    // when the beam would otherwise flood us with near-duplicates.
    const OVER_SAMPLE: usize = 8;
    const MAX_RAW: usize = 256;
    let raw_limit = limit.saturating_mul(OVER_SAMPLE).min(MAX_RAW).max(limit);
    let mut raw = beam_extract_candidates(egraph, root, config, raw_limit);
    let greedy = greedy_candidate_pool(egraph, root);
    for candidate in greedy {
        push_unique_candidate(&mut raw, candidate);
    }
    report.raw_candidate_limit = raw_limit;
    report.raw_candidates += raw.len();
    let selected = validate_candidate_exprs(raw, egraph, device, lowering, limit, &mut report);
    (selected, report)
}

fn greedy_candidate_pool(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    root: Id,
) -> Vec<(f64, RecExpr<TensorIr>)> {
    let mut raw = Vec::new();
    let greedy_dispatch = {
        let extractor = Extractor::new(egraph, DispatchPreferredCost);
        extractor.find_best(root)
    };
    push_unique_candidate(&mut raw, greedy_dispatch);
    let greedy_tiled_dispatch = {
        let extractor = Extractor::new(egraph, TiledDispatchPreferredCost);
        extractor.find_best(root)
    };
    push_unique_candidate(&mut raw, greedy_tiled_dispatch);
    let greedy_blocked_dispatch = {
        let extractor = Extractor::new(egraph, BlockedDispatchPreferredCost);
        extractor.find_best(root)
    };
    push_unique_candidate(&mut raw, greedy_blocked_dispatch);
    let greedy_single_dispatch = {
        let extractor = Extractor::new(egraph, SingleDispatchPreferredCost);
        extractor.find_best(root)
    };
    push_unique_candidate(&mut raw, greedy_single_dispatch);
    let greedy_composite = {
        let extractor = Extractor::new(egraph, CompositePreferredCost);
        extractor.find_best(root)
    };
    push_unique_candidate(&mut raw, greedy_composite);
    // Dispatch with bias toward `ReduceSimd` / short inner Thetas. Gives
    // `theta_inner_cooperative` rewrites a shot at the top of the list when
    // their Dispatch-level cost is otherwise a wash against the non-coop
    // form (same #dispatches, same ops, just a wrapper node extra).
    let greedy_reduce_simd = {
        let extractor = Extractor::new(egraph, ReduceSimdPreferredCost);
        extractor.find_best(root)
    };
    push_unique_candidate(&mut raw, greedy_reduce_simd);
    for forced in forced_root_dispatch_candidates(egraph, root) {
        push_unique_candidate(&mut raw, forced);
    }
    raw
}

fn push_unique_candidate(
    raw: &mut Vec<(f64, RecExpr<TensorIr>)>,
    candidate: (f64, RecExpr<TensorIr>),
) {
    if !raw.iter().any(|(_, expr)| expr == &candidate.1) {
        raw.push(candidate);
    }
}

fn validate_candidate_exprs(
    raw: Vec<(f64, RecExpr<TensorIr>)>,
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    device: &DeviceProfile,
    lowering: &LoweringOptions,
    limit: usize,
    report: &mut CandidateValidationReport,
) -> Vec<(f64, RecExpr<TensorIr>)> {
    let mut valid = Vec::with_capacity(limit);
    for (cost, expr) in raw {
        if !recexpr_has_valid_var_scopes(&expr) {
            report.invalid_var_scopes += 1;
            continue;
        }
        let program =
            build_dispatch_program_from_extracted(&expr, egraph.clone(), device, lowering);
        if program.dispatches.is_empty() {
            report.empty_dispatch_programs += 1;
            continue;
        }
        if crate::verify::verify(&program).is_err() {
            report.verification_failures += 1;
            continue;
        }
        let empty_tg_dispatches = program
            .dispatches
            .iter()
            .filter(|dispatch| dispatch.tg_buffers.is_empty())
            .count();
        let peak_tg_bytes = program.peak_threadgroup_bytes();
        let pack_backed_outputs = program.dispatches.iter().any(|dispatch| {
            dispatch
                .outputs
                .iter()
                .any(|output| program.egraph[output.value_id].data.contains_pack)
        });
        let pack_rank = usize::from(!pack_backed_outputs);
        let tuple_output_count = if pack_backed_outputs {
            program
                .dispatches
                .iter()
                .map(|dispatch| dispatch.outputs.len())
                .sum::<usize>()
        } else {
            0
        };
        let execution_score = execution_weighted_expr_cost(&expr, device)
            + 250.0 * program.dispatches.len() as f64
            + 500.0 * empty_tg_dispatches as f64;
        valid.push((
            program.dispatches.len(),
            program.pipelines.len(),
            empty_tg_dispatches,
            peak_tg_bytes,
            pack_rank,
            tuple_output_count,
            execution_score,
            cost,
            expr,
        ));
    }

    valid.sort_by(|lhs, rhs| {
        lhs.0
            .cmp(&rhs.0)
            .then_with(|| lhs.1.cmp(&rhs.1))
            .then_with(|| lhs.2.cmp(&rhs.2))
            .then_with(|| rhs.3.cmp(&lhs.3))
            .then_with(|| lhs.4.cmp(&rhs.4))
            .then_with(|| lhs.5.cmp(&rhs.5))
            .then_with(|| {
                lhs.6
                    .partial_cmp(&rhs.6)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| {
                lhs.7
                    .partial_cmp(&rhs.7)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    report.accepted_before_limit = valid.len();
    let selected = valid
        .into_iter()
        .take(limit)
        .map(|(_, _, _, _, _, _, _, cost, expr)| (cost, expr))
        .collect::<Vec<_>>();
    report.returned = selected.len();
    selected
}

fn forced_root_dispatch_candidates(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    root: Id,
) -> Vec<(f64, RecExpr<TensorIr>)> {
    let root = egraph.find(root);
    let mut out = Vec::new();
    for node in egraph[root].iter() {
        if !matches!(node, TensorIr::Dispatch(DispatchNode::Dispatch { .. })) {
            continue;
        }
        if let Some(candidate) =
            forced_root_dispatch_candidate_with(egraph, node, CompositePreferredCost)
        {
            out.push(candidate);
        }
        if let Some(candidate) =
            forced_root_dispatch_candidate_with(egraph, node, ReduceSimdPreferredCost)
        {
            out.push(candidate);
        }
    }
    out
}

fn forced_root_dispatch_candidate_with<C>(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    node: &TensorIr,
    cost_fn: C,
) -> Option<(f64, RecExpr<TensorIr>)>
where
    C: CostFunction<TensorIr, Cost = f64>,
{
    let extractor = Extractor::new(egraph, cost_fn);
    let mut expr = RecExpr::default();
    let mut forced = node.clone();
    let mut total_cost = 0.0;
    for child in forced.children_mut() {
        let (cost, child_expr) = extractor.find_best(*child);
        total_cost += cost;
        *child = append_recexpr(&mut expr, &child_expr);
    }
    expr.add(forced);
    if recexpr_has_valid_var_scopes(&expr) {
        Some((total_cost, expr))
    } else {
        None
    }
}

fn append_recexpr(dst: &mut RecExpr<TensorIr>, src: &RecExpr<TensorIr>) -> Id {
    let offset = dst.as_ref().len();
    for node in src.as_ref() {
        let mut node = node.clone();
        for child in node.children_mut() {
            *child = Id::from(usize::from(*child) + offset);
        }
        dst.add(node);
    }
    Id::from(dst.as_ref().len() - 1)
}

fn execution_weighted_expr_cost(expr: &RecExpr<TensorIr>, device: &DeviceProfile) -> f64 {
    const DISPATCH_LAUNCH_COST: f64 = 150.0;
    const DEVICE_ACCESS_COST: f64 = 100.0;
    const THREADGROUP_ACCESS_COST: f64 = 5.0;
    const ARITH_COST: f64 = 1.0;
    const REDUCE_SIMD_COST: f64 = 10.0;
    const SHUFFLE_COST: f64 = 2.0;
    const BARRIER_COST: f64 = 50.0;
    const THETA_OVERHEAD_COST: f64 = 0.5;
    const MAX_EXEC_MULTIPLIER: f64 = 1_000_000_000.0;

    fn const_u32(nodes: &[TensorIr], id: Id) -> Option<u32> {
        match &nodes[usize::from(id)] {
            TensorIr::Const(ScalarValue::U32(v)) => Some(*v),
            TensorIr::Const(ScalarValue::I32(v)) if *v > 0 => Some((*v).cast_unsigned()),
            _ => None,
        }
    }

    fn child_score(
        nodes: &[TensorIr],
        id: Id,
        multiplier: f64,
        device: &DeviceProfile,
        seen: &mut HashSet<(usize, u64)>,
    ) -> f64 {
        score_node(
            nodes,
            usize::from(id),
            multiplier.min(MAX_EXEC_MULTIPLIER),
            device,
            seen,
        )
    }

    fn score_node(
        nodes: &[TensorIr],
        idx: usize,
        multiplier: f64,
        device: &DeviceProfile,
        seen: &mut HashSet<(usize, u64)>,
    ) -> f64 {
        let multiplier = multiplier.min(MAX_EXEC_MULTIPLIER);
        if !seen.insert((idx, multiplier.to_bits())) {
            return 0.0;
        }

        let node = &nodes[idx];
        match node {
            TensorIr::Dispatch(DispatchNode::Dispatch {
                workgroups,
                num_inputs,
                children_list,
            }) => {
                let children = extract_recexpr_list(nodes, *children_list);
                let num_inputs =
                    usize::try_from(*num_inputs).expect("dispatch input count fits in usize");
                // Output arity is derived from the children layout
                // `[inputs (num_inputs), (value, addr) pairs ...]`.
                let outputs = children.len().saturating_sub(num_inputs) / 2;
                let launch_multiplier = multiplier.min(MAX_EXEC_MULTIPLIER);
                let body_multiplier =
                    (launch_multiplier * f64::from(*workgroups)).min(MAX_EXEC_MULTIPLIER);

                let mut score = DISPATCH_LAUNCH_COST * launch_multiplier;
                for child in &children[..num_inputs.min(children.len())] {
                    score += child_score(nodes, *child, launch_multiplier, device, seen);
                }
                let pairs_start = num_inputs;
                for output_idx in 0..outputs {
                    let base = pairs_start + output_idx * 2;
                    if base + 1 >= children.len() {
                        break;
                    }
                    score += child_score(nodes, children[base], body_multiplier, device, seen);
                    score += child_score(nodes, children[base + 1], body_multiplier, device, seen);
                }
                score
            }
            TensorIr::Dispatch(DispatchNode::Seq(list_id) | DispatchNode::Pipeline(list_id)) => {
                extract_recexpr_list(nodes, *list_id)
                    .into_iter()
                    .map(|child| child_score(nodes, child, multiplier, device, seen))
                    .sum()
            }
            TensorIr::Dispatch(DispatchNode::Pack { children_list }) => {
                extract_recexpr_list(nodes, *children_list)
                    .into_iter()
                    .map(|child| child_score(nodes, child, multiplier, device, seen))
                    .sum()
            }
            TensorIr::Dispatch(DispatchNode::Extract { tuple, .. }) => {
                child_score(nodes, *tuple, multiplier, device, seen)
            }
            TensorIr::Simd(SimdNode::Theta {
                children: [init, count, update],
                ..
            }) => {
                let loop_count = const_u32(nodes, *count)
                    .map(f64::from)
                    .unwrap_or(f64::from(device.simd_width.max(1)));
                THETA_OVERHEAD_COST * multiplier
                    + child_score(nodes, *init, multiplier, device, seen)
                    + child_score(nodes, *count, multiplier, device, seen)
                    + child_score(nodes, *update, multiplier * loop_count, device, seen)
            }
            TensorIr::Simd(SimdNode::Load { tier, children }) => {
                let local = match tier {
                    MemTier::Device(_) => DEVICE_ACCESS_COST,
                    MemTier::Threadgroup(_) => THREADGROUP_ACCESS_COST,
                };
                local * multiplier
                    + child_score(nodes, children[0], multiplier, device, seen)
                    + child_score(nodes, children[1], multiplier, device, seen)
            }
            TensorIr::Simd(SimdNode::Store { tier, children }) => {
                let local = match tier {
                    MemTier::Device(_) => DEVICE_ACCESS_COST,
                    MemTier::Threadgroup(_) => THREADGROUP_ACCESS_COST,
                };
                local * multiplier
                    + child_score(nodes, children[0], multiplier, device, seen)
                    + child_score(nodes, children[1], multiplier, device, seen)
                    + child_score(nodes, children[2], multiplier, device, seen)
            }
            TensorIr::Simd(SimdNode::StoreIf { tier, children }) => {
                let local = match tier {
                    MemTier::Device(_) => DEVICE_ACCESS_COST,
                    MemTier::Threadgroup(_) => THREADGROUP_ACCESS_COST,
                };
                local * multiplier
                    + child_score(nodes, children[0], multiplier, device, seen)
                    + child_score(nodes, children[1], multiplier, device, seen)
                    + child_score(nodes, children[2], multiplier, device, seen)
                    + child_score(nodes, children[3], multiplier, device, seen)
            }
            TensorIr::Simd(SimdNode::Barrier { state, .. }) => {
                BARRIER_COST * multiplier + child_score(nodes, *state, multiplier, device, seen)
            }
            TensorIr::Simd(SimdNode::ReduceSimd { src, .. }) => {
                REDUCE_SIMD_COST * multiplier + child_score(nodes, *src, multiplier, device, seen)
            }
            TensorIr::Simd(SimdNode::Shuffle(children)) => {
                SHUFFLE_COST * multiplier
                    + child_score(nodes, children[0], multiplier, device, seen)
                    + child_score(nodes, children[1], multiplier, device, seen)
            }
            TensorIr::BinOp(_, children) => {
                ARITH_COST * multiplier
                    + child_score(nodes, children[0], multiplier, device, seen)
                    + child_score(nodes, children[1], multiplier, device, seen)
            }
            TensorIr::UnOp(_, child) => {
                ARITH_COST * multiplier + child_score(nodes, *child, multiplier, device, seen)
            }
            TensorIr::TernOp(_, children) => {
                ARITH_COST * multiplier
                    + child_score(nodes, children[0], multiplier, device, seen)
                    + child_score(nodes, children[1], multiplier, device, seen)
                    + child_score(nodes, children[2], multiplier, device, seen)
            }
            TensorIr::HighLevel(HighLevelNode::Restride { expr, .. })
            | TensorIr::HighLevel(HighLevelNode::Reduce { expr, .. }) => {
                child_score(nodes, *expr, multiplier, device, seen)
            }
            TensorIr::HighLevel(HighLevelNode::Elementwise { children_list, .. }) => {
                extract_recexpr_list(nodes, *children_list)
                    .into_iter()
                    .map(|child| child_score(nodes, child, multiplier, device, seen))
                    .sum()
            }
            TensorIr::HighLevel(HighLevelNode::IndexedParam { children_list, .. }) => {
                extract_recexpr_list(nodes, *children_list)
                    .into_iter()
                    .map(|child| child_score(nodes, child, multiplier, device, seen))
                    .sum()
            }
            TensorIr::Nil
            | TensorIr::Cons(_)
            | TensorIr::Const(_)
            | TensorIr::HighLevel(
                HighLevelNode::Input { .. } | HighLevelNode::Param(_) | HighLevelNode::Index(_),
            )
            | TensorIr::Dispatch(DispatchNode::Token)
            | TensorIr::Simd(SimdNode::Var(_)) => 0.0,
        }
    }

    let nodes = expr.as_ref();
    if nodes.is_empty() {
        return 0.0;
    }
    let mut seen = HashSet::new();
    score_node(nodes, nodes.len() - 1, 1.0, device, &mut seen)
}

fn recexpr_has_valid_var_scopes(expr: &RecExpr<TensorIr>) -> bool {
    let nodes = expr.as_ref();
    if nodes.is_empty() {
        return true;
    }

    for (idx, node) in nodes.iter().enumerate() {
        if node
            .children()
            .iter()
            .any(|child| usize::from(*child) >= idx)
        {
            return false;
        }
    }

    let mut stack: Vec<(usize, u32)> = vec![(nodes.len() - 1, 0)];
    while let Some((idx, binder_depth)) = stack.pop() {
        match &nodes[idx] {
            TensorIr::Simd(SimdNode::Var(crate::types::VarRef::Bound {
                kind: crate::types::BinderKind::Theta,
                depth,
                ..
            })) => {
                if *depth >= binder_depth {
                    return false;
                }
            }
            TensorIr::Simd(SimdNode::Theta {
                children: [init, count, update],
                ..
            }) => {
                stack.push((usize::from(*init), binder_depth));
                stack.push((usize::from(*count), binder_depth));
                stack.push((usize::from(*update), binder_depth + 1));
            }
            node => {
                for child in node.children() {
                    stack.push((usize::from(*child), binder_depth));
                }
            }
        }
    }

    true
}

/// Cost function for materializing a missing intermediate dispatch. It
/// strongly prefers lowered `Dispatch` nodes over `HighLevel` nodes so the
/// extracted `RecExpr` uses the dispatch-form representative of each eclass
/// (this is the same preference baked into `SyntheticCostModel`, reused here
/// for the simpler greedy `Extractor`).
struct DispatchPreferredCost;

impl CostFunction<TensorIr> for DispatchPreferredCost {
    type Cost = f64;

    fn cost<C>(&mut self, enode: &TensorIr, mut costs: C) -> f64
    where
        C: FnMut(Id) -> f64,
    {
        // Prefer any Dispatch form over residual HighLevel nodes. Do not
        // prefer threadgroup-tier Loads here: a missing intermediate may be
        // materialized as a plain dispatch, and plain skeletons cannot declare
        // threadgroup buffers for arbitrary nested TG-load representatives.
        let base: f64 = match enode {
            TensorIr::HighLevel(HighLevelNode::Elementwise { .. })
            | TensorIr::HighLevel(HighLevelNode::Reduce { .. }) => 1_000_000.0,
            TensorIr::HighLevel(HighLevelNode::Restride { .. }) => 100.0,
            TensorIr::Simd(SimdNode::Load {
                tier: MemTier::Threadgroup(_),
                ..
            }) => 10_000.0,
            TensorIr::Simd(SimdNode::Load { .. }) => 1.0,
            TensorIr::Dispatch(DispatchNode::Dispatch { .. }) => 0.1,
            _ => 1.0,
        };
        enode.fold(base, |sum, child| sum + costs(child))
    }
}

struct TiledDispatchPreferredCost;

impl CostFunction<TensorIr> for TiledDispatchPreferredCost {
    type Cost = f64;

    fn cost<C>(&mut self, enode: &TensorIr, mut costs: C) -> f64
    where
        C: FnMut(Id) -> f64,
    {
        let base: f64 = match enode {
            TensorIr::HighLevel(_) => 10_000.0,
            TensorIr::Dispatch(DispatchNode::Dispatch { .. }) => 0.1,
            _ => 1.0,
        };
        enode.fold(base, |sum, child| sum + costs(child))
    }
}

/// Prefer Dispatches whose bodies contain `ReduceSimd` (shuffle-reduced
/// Thetas produced by `theta_split_cooperative` / `theta_inner_cooperative`)
/// and short inner Theta counts. Costs stay non-negative: egg extraction can
/// revisit cyclic e-classes indefinitely when a preference creates a negative
/// cycle through equivalent nodes.
struct ReduceSimdPreferredCost;

impl CostFunction<TensorIr> for ReduceSimdPreferredCost {
    type Cost = f64;

    fn cost<C>(&mut self, enode: &TensorIr, mut costs: C) -> f64
    where
        C: FnMut(Id) -> f64,
    {
        // Keep HighLevel disqualifyingly expensive even when ReduceSimd nodes
        // are strongly preferred by their tiny positive cost.
        const HIGHLEVEL_PENALTY: f64 = 1.0e15;
        const REDUCE_SIMD_COST: f64 = 0.001;
        // Threadgroup loads are cheap per access but pulling one into the
        // body drags in a whole tile setup the plain skeleton can't emit —
        // `build_single_dispatch_inner` rejects has_tg_loads bodies that
        // don't match the tiled pattern. Price TG loads the same as device
        // loads so extraction doesn't swap a valid Device form for a TG
        // form that later fails skeleton build.
        let base: f64 = match enode {
            TensorIr::HighLevel(HighLevelNode::Elementwise { .. })
            | TensorIr::HighLevel(HighLevelNode::Reduce { .. }) => HIGHLEVEL_PENALTY,
            TensorIr::HighLevel(HighLevelNode::Restride { .. }) => 100.0,
            TensorIr::Simd(SimdNode::Load { .. }) => 1.0,
            TensorIr::Dispatch(DispatchNode::Dispatch { .. }) => 0.1,
            TensorIr::Simd(SimdNode::ReduceSimd { .. }) => REDUCE_SIMD_COST,
            _ => 1.0,
        };
        enode.fold(base, |sum, child| sum + costs(child))
    }
}

/// Surface register-blocked composite dispatches into the valid-candidate pool.
/// The normal beam cost intentionally prices every output value, but kernels
/// like online attention share a tuple Theta across all output stores; this
/// greedy pass gives those shared-body forms a chance to be validated and then
/// ranked by the executable cost model.
struct BlockedDispatchPreferredCost;

impl CostFunction<TensorIr> for BlockedDispatchPreferredCost {
    type Cost = f64;

    fn cost<C>(&mut self, enode: &TensorIr, mut costs: C) -> f64
    where
        C: FnMut(Id) -> f64,
    {
        let base: f64 = match enode {
            TensorIr::HighLevel(HighLevelNode::Elementwise { .. })
            | TensorIr::HighLevel(HighLevelNode::Reduce { .. }) => 1.0e30,
            TensorIr::HighLevel(HighLevelNode::Restride { .. }) => 100.0,
            TensorIr::Simd(SimdNode::Load {
                tier: MemTier::Threadgroup(_),
                ..
            }) => 10_000.0,
            TensorIr::Simd(SimdNode::Load { .. }) => 1.0,
            TensorIr::Dispatch(DispatchNode::Dispatch { workgroups, .. }) => {
                1.0e6 / f64::from((*workgroups).max(1))
            }
            TensorIr::Dispatch(DispatchNode::Seq(_) | DispatchNode::Pipeline(_)) => 1.0e9,
            _ => 1.0,
        };
        enode.fold(base, |sum, child| sum + costs(child))
    }
}

/// Prefer extraction candidates with fewer dispatch nodes, then with more
/// external inputs on the dispatch. This keeps fully inlined composite kernels
/// in the valid-candidate pool even when their scalar body is more expensive
/// than materializing an intermediate dispatch and consuming it later.
struct SingleDispatchPreferredCost;

impl CostFunction<TensorIr> for SingleDispatchPreferredCost {
    type Cost = f64;

    fn cost<C>(&mut self, enode: &TensorIr, mut costs: C) -> f64
    where
        C: FnMut(Id) -> f64,
    {
        let base: f64 = match enode {
            TensorIr::HighLevel(HighLevelNode::Elementwise { .. })
            | TensorIr::HighLevel(HighLevelNode::Reduce { .. }) => 1.0e12,
            TensorIr::HighLevel(HighLevelNode::Restride { .. }) => 100.0,
            TensorIr::Simd(SimdNode::Load { .. }) => 1.0,
            TensorIr::Dispatch(DispatchNode::Dispatch { num_inputs, .. }) => {
                1.0e15 - 1.0e9 * f64::from(*num_inputs)
            }
            TensorIr::Dispatch(DispatchNode::Seq(_) | DispatchNode::Pipeline(_)) => 1.0e12,
            _ => 1.0,
        };
        enode.fold(base, |sum, child| sum + costs(child))
    }
}

/// Prefer any Dispatch form over un-lowered HighLevel, and prefer
/// threadgroup loads over device loads. The old "composite vs plain"
/// body-kind tag is gone.
struct CompositePreferredCost;

impl CostFunction<TensorIr> for CompositePreferredCost {
    type Cost = f64;

    fn cost<C>(&mut self, enode: &TensorIr, mut costs: C) -> f64
    where
        C: FnMut(Id) -> f64,
    {
        let base: f64 = match enode {
            TensorIr::HighLevel(HighLevelNode::Elementwise { .. })
            | TensorIr::HighLevel(HighLevelNode::Reduce { .. }) => 1_000_000.0,
            TensorIr::HighLevel(HighLevelNode::Restride { .. }) => 100.0,
            TensorIr::Simd(SimdNode::Load {
                tier: MemTier::Threadgroup(_),
                ..
            }) => 1_000.0,
            TensorIr::Simd(SimdNode::Load { .. }) => 1.0,
            TensorIr::Dispatch(DispatchNode::Dispatch { .. }) => 1.0,
            TensorIr::Dispatch(DispatchNode::Seq(_) | DispatchNode::Pipeline(_)) => 1_000_000.0,
            _ => 1.0,
        };
        enode.fold(base, |sum, child| sum + costs(child))
    }
}

/// Materialize any dispatch input whose eclass contains a `Dispatch` node
/// that wasn't already built. Returns the list of newly-built `DispatchInfo`
/// records in topological order (predecessors before consumers). We keep
/// extracting until every input resolves to either an external buffer or a
/// dispatch we've produced.
fn materialize_missing_dispatches(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    dispatches: &mut Vec<DispatchInfo>,
    chosen: &mut HashMap<Id, TensorIr>,
    device: &DeviceProfile,
    lowering: &LoweringOptions,
) {
    use crate::language::{HighLevelNode, SimdNode};
    use crate::types::{BufferRef, MemTier};

    // Cap the materialization loop — every iteration adds at least one new
    // dispatch or terminates because no progress is possible, but the bound
    // guards against degenerate e-graph shapes.
    for _ in 0..64 {
        egraph.rebuild();
        let mut produced: HashSet<Id> = dispatches
            .iter()
            .map(|d| egraph.find(d.semantic_output_id))
            .collect();

        let mut missing: Option<Id> = None;
        'outer: for dispatch in dispatches.iter() {
            for input in &dispatch.inputs {
                let canonical = egraph.find(*input);
                if produced.contains(&canonical) {
                    continue;
                }
                let external = egraph[canonical].iter().any(|node| {
                    matches!(
                        node,
                        TensorIr::HighLevel(HighLevelNode::Input { .. })
                            | TensorIr::Simd(SimdNode::Load {
                                tier: MemTier::Device(BufferRef::Input(_)),
                                ..
                            })
                    )
                });
                if external {
                    continue;
                }
                let has_dispatch = egraph[canonical]
                    .iter()
                    .any(|node| matches!(node, TensorIr::Dispatch(DispatchNode::Dispatch { .. })));
                if !has_dispatch {
                    continue;
                }
                missing = Some(canonical);
                break 'outer;
            }
        }

        let Some(canonical) = missing else {
            return;
        };

        let sub_expr = {
            let extractor = Extractor::new(egraph, DispatchPreferredCost);
            extractor.find_best(canonical).1
        };
        let sub_nodes = sub_expr.as_ref();
        let root_idx = sub_nodes.len() - 1;
        if !matches!(
            &sub_nodes[root_idx],
            TensorIr::Dispatch(DispatchNode::Dispatch { .. })
        ) {
            // Couldn't extract a dispatch root for this class; give up so the
            // outer `all_inputs_resolvable` check rejects the candidate.
            return;
        }

        // Pull in every `Dispatch` node in topological order so nested
        // intermediates (e.g. softmax's max-reduce consumed by shifted) also
        // get built before their consumer.
        let order = collect_dispatch_order(sub_nodes, root_idx);
        let mut added = false;
        for idx in order {
            if let Some(dispatch) =
                build_single_dispatch(sub_nodes, idx, egraph, chosen, device, lowering)
            {
                let sem = egraph.find(dispatch.semantic_output_id);
                if !produced.insert(sem) {
                    continue;
                }
                dispatches.push(dispatch);
                added = true;
            }
        }

        if !added {
            return;
        }

        // Keep overall order topological: re-sort by dependency. This is a
        // small list, so a linear pass with a swap-based topological bubble
        // is sufficient.
        topological_sort_dispatches(egraph, dispatches);
    }
}

fn topological_sort_dispatches(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    dispatches: &mut Vec<DispatchInfo>,
) {
    let n = dispatches.len();
    let semantic: Vec<Id> = dispatches
        .iter()
        .map(|d| egraph.find(d.semantic_output_id))
        .collect();
    let inputs: Vec<Vec<Id>> = dispatches
        .iter()
        .map(|d| d.inputs.iter().map(|id| egraph.find(*id)).collect())
        .collect();

    let mut order = Vec::with_capacity(n);
    let mut placed = vec![false; n];
    for _ in 0..n {
        let mut picked = None;
        for (i, _) in dispatches.iter().enumerate() {
            if placed[i] {
                continue;
            }
            let deps_met = inputs[i].iter().all(|canonical| {
                semantic
                    .iter()
                    .enumerate()
                    .all(|(j, sem)| placed[j] || sem != canonical || i == j)
            });
            if deps_met {
                picked = Some(i);
                break;
            }
        }
        // If no dispatch has all deps placed (cycle / external-only inputs
        // already count as met above), fall back to the first remaining.
        let next = picked.unwrap_or_else(|| placed.iter().position(|p| !p).unwrap());
        placed[next] = true;
        order.push(next);
    }

    let original = std::mem::take(dispatches);
    let mut by_index: Vec<Option<DispatchInfo>> = original.into_iter().map(Some).collect();
    for idx in order {
        dispatches.push(by_index[idx].take().expect("each dispatch placed once"));
    }
}

fn all_inputs_resolvable(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    dispatches: &[DispatchInfo],
) -> bool {
    use crate::language::{HighLevelNode, SimdNode};
    use crate::types::{BufferRef, MemTier};

    let mut produced: HashSet<Id> = HashSet::new();
    for dispatch in dispatches {
        for input in &dispatch.inputs {
            let canonical = egraph.find(*input);
            if produced.contains(&canonical) {
                continue;
            }
            let external = egraph[canonical].iter().any(|node| {
                matches!(
                    node,
                    TensorIr::HighLevel(HighLevelNode::Input { .. })
                        | TensorIr::Simd(SimdNode::Load {
                            tier: MemTier::Device(BufferRef::Input(_)),
                            ..
                        })
                )
            });
            if !external {
                return false;
            }
        }
        produced.insert(egraph.find(dispatch.semantic_output_id));
    }
    true
}

fn all_output_addrs_in_bounds(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen: &HashMap<Id, TensorIr>,
    dispatch: &DispatchInfo,
    device: &DeviceProfile,
) -> bool {
    use crate::skeleton::substitute::compute_addr_interval_bounded;

    let output_count = u32::try_from(dispatch.outputs.len()).expect("output count fits in u32");
    let output_len = dispatch
        .workgroups
        .saturating_mul(device.simd_width)
        .saturating_mul(output_count);
    if output_len == 0 {
        return false;
    }

    let workgroup_bound = dispatch.workgroups.saturating_sub(1);
    let simdgroup_bound = dispatch.simdgroups.saturating_sub(1);
    let var_bound = output_len.max(1);
    dispatch.outputs.iter().all(|output| {
        let interval = compute_addr_interval_bounded(
            egraph,
            chosen,
            output.addr_id,
            var_bound,
            device,
            workgroup_bound,
            simdgroup_bound,
        );
        interval.hi == u32::MAX || interval.hi < output_len
    })
}

/// Validate that every threadgroup `Load` in every dispatch addresses only
/// positions inside its declared `tg_buffer.size`, when the `Workgroup` and
/// `Simdgroup` vars range over their post-promotion intra-physical-wg
/// extents. A TG Load whose address can reach past the buffer was produced
/// by a rule whose tile-local formula doesn't match the cooperative Store
/// it's paired with (typical symptom: the Store writes
/// `A[row_base + sg, k_col]` into `tg[sg * tile_cols + lane]`, but the
/// Load indexes by `wg_in_tile * tile_cols + k_inner` where `wg_in_tile`
/// varies across physical workgroups). Reading out of bounds is silently
/// zero on Metal and gives subtly wrong results, so we reject the whole
/// candidate here — the extractor retries with the next beam entry.
fn all_tg_loads_in_bounds(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen: &HashMap<Id, TensorIr>,
    dispatch: &DispatchInfo,
    device: &DeviceProfile,
) -> bool {
    // Check against the *full* dispatch workgroup range: each physical
    // workgroup has its own TG buffer, but its Load must index only the
    // intra-physical portion of that buffer. If the address grows with
    // the absolute `Workgroup` var (e.g. via `Mod(wg, wgs_per_tile)`
    // where `wgs_per_tile > num_simdgroups`), then different physical
    // workgroups within the same output tile would access out of bounds.
    //
    // `workgroup_bound = min(dispatch.workgroups, wgs_per_tile) - 1`
    // captures the range: the `Workgroup` var varies across the tile
    // (either `dispatch.workgroups` total, when there's a single tile, or
    // `wgs_per_tile` consecutive values within one tile otherwise), and
    // the Load's address must fit the buffer for every such value.
    let workgroup_bound = dispatch.workgroups.saturating_sub(1);
    let simdgroup_bound = dispatch.simdgroups.saturating_sub(1);

    for output in &dispatch.outputs {
        if !load_subtree_in_bounds(
            egraph,
            chosen,
            output.value_id,
            &dispatch.tg_buffers,
            device,
            workgroup_bound,
            simdgroup_bound,
            &mut HashSet::new(),
        ) {
            return false;
        }
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn load_subtree_in_bounds(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen: &HashMap<Id, TensorIr>,
    id: Id,
    tg_buffers: &[TgBufferInfo],
    device: &DeviceProfile,
    workgroup_bound: u32,
    simdgroup_bound: u32,
    visited: &mut HashSet<Id>,
) -> bool {
    use crate::language::SimdNode;
    use crate::skeleton::substitute::compute_addr_interval_bounded;
    use crate::types::MemTier;

    let canonical = egraph.find(id);
    if !visited.insert(canonical) {
        return true;
    }
    // Only inspect the node variant the extractor actually chose. A single
    // e-class often carries both a Device Load and the corresponding
    // Threadgroup Load (unioned by the phase-1 lowering), and we must not
    // reject the candidate for the un-picked sibling's address shape.
    let Some(node) = chosen
        .get(&canonical)
        .cloned()
        .or_else(|| egraph[canonical].iter().next().cloned())
    else {
        return true;
    };

    if let TensorIr::Simd(SimdNode::Load {
        tier: MemTier::Threadgroup(name),
        children,
    }) = &node
    {
        let Some(buf) = tg_buffers.iter().find(|b| b.tg_name == *name) else {
            // No matching buffer info — the skeleton never declared a tile
            // layout for this load, so codegen would panic.
            return false;
        };

        // Layout-consistency check: the buffer's declared tile dimensions
        // (`tile_rows * tile_cols`) must cover its `size` — otherwise the
        // cooperative Store, which indexes the buffer via
        // `(row_in_tile * tile_cols + col_in_tile)`, only populates the
        // first `tile_rows * tile_cols` slots even though the Load's
        // address can reach further. This happens when `decompose` fails
        // to extract a 2D row/col split (so `tile_rows == 1` is used as a
        // 1D-broadcast fallback) while the Load actually wants 2D
        // access — the phase-1 rule emitted an address the skeleton
        // can't pair with a matching Store. Reject.
        let covered = buf.tile_rows.saturating_mul(buf.tile_cols);
        if covered < buf.size {
            return false;
        }

        // `tile_cols` is the contiguous stride; reserve room for the inner-K
        // sweep (codegen assumes `tile_cols == tile_k`-aligned slabs).
        let tile_k = buf.tile_cols.max(1);
        let interval = compute_addr_interval_bounded(
            egraph,
            chosen,
            children[0],
            tile_k,
            device,
            workgroup_bound,
            simdgroup_bound,
        );
        // Reject only on *provable* OOB: if the tightest upper bound we
        // can derive already exceeds the buffer size, no runtime choice of
        // `lane`/`workgroup`/`simdgroup`/`iter` can rescue it. The
        // `unknown()` interval (u32::MAX) is filtered via the explicit
        // check against buf.size with saturation — a proven unbounded
        // address is itself a rejection signal because a correctly built
        // Load always yields a tight interval.
        if interval.hi >= buf.size && interval.hi != u32::MAX {
            return false;
        }
    }

    for child in node.children() {
        if !load_subtree_in_bounds(
            egraph,
            chosen,
            *child,
            tg_buffers,
            device,
            workgroup_bound,
            simdgroup_bound,
            visited,
        ) {
            return false;
        }
    }
    true
}

fn dispatch_semantic_output_id(
    nodes: &[TensorIr],
    dispatch_idx: usize,
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
) -> Id {
    let id = add_recexpr_subtree(nodes, dispatch_idx, egraph, chosen);
    egraph.find(id)
}

pub(super) fn build_single_dispatch(
    nodes: &[TensorIr],
    dispatch_idx: usize,
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    device: &DeviceProfile,
    lowering: &LoweringOptions,
) -> Option<DispatchInfo> {
    let mut dispatch =
        build_single_dispatch_inner(nodes, dispatch_idx, egraph, chosen, device, lowering)?;
    // Rebuild so analysis-driven unions from any nodes added by
    // `extract_recexpr_dispatch_layout` / `build_*` are reflected in the
    // canonical IDs we read below (downstream resolution keys on these).
    egraph.rebuild();
    dispatch.semantic_output_id = dispatch_semantic_output_id(nodes, dispatch_idx, egraph, chosen);
    // Canonicalize recorded input IDs under the post-rebuild view so they
    // match lookups performed later against the same e-graph.
    for input in dispatch.inputs.iter_mut() {
        *input = egraph.find(*input);
    }
    Some(dispatch)
}

fn build_single_dispatch_inner(
    nodes: &[TensorIr],
    dispatch_idx: usize,
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    device: &DeviceProfile,
    lowering: &LoweringOptions,
) -> Option<DispatchInfo> {
    let layout = extract_recexpr_dispatch_layout(nodes, dispatch_idx, egraph, chosen)?;
    if let Some(dispatch) =
        try_build_recexpr_tiled_dispatch(nodes, &layout, egraph, chosen, device, lowering)
    {
        return Some(dispatch);
    }

    let (body_idx, output_addr) = layout.body_addr_pairs[0];
    let output_id = add_recexpr_subtree(nodes, body_idx, egraph, chosen);
    let has_tg_loads = subtree_has_tg_loads(nodes, body_idx);
    if has_tg_loads
        && let Some(dispatch) = build_recovered_tiled_dispatch(
            &layout,
            output_id,
            output_addr,
            egraph,
            chosen,
            device,
            lowering,
        )
    {
        return Some(dispatch);
    }

    // If the extracted body references threadgroup buffers but neither the
    // tiled nor the recovered-tiled skeleton matched, the plain skeleton
    // can't declare the expected `Threadgroup` buffers and naga codegen
    // would panic. Reject this candidate so the extractor can try another.
    if has_tg_loads {
        return None;
    }

    Some(build_plain_dispatch_from_recexpr(
        nodes, &layout, egraph, chosen, device,
    ))
}

mod cooperative;
mod decompose;
mod rewrite;
mod substitute;

pub(super) use cooperative::*;
use decompose::*;
pub(crate) use rewrite::rewrite_dispatch_node;
pub use substitute::collect_tg_buffer_info;
pub(super) use substitute::*;
