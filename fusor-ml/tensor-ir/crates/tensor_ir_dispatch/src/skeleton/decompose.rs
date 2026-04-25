//! Address & nested-Theta decomposition helpers.

use std::collections::{HashMap, HashSet};

use egg::{EGraph, Id, Language};

use crate::analysis::TensorAnalysis;
use crate::language::{SimdNode, TensorIr};
use crate::types::{BinaryOp, BufferRef, DeviceProfile, IndexLevel, MemTier, ScalarValue, VarRef};

use super::*;
use super::{TgBufferInfo, add_and_choose, dtype_bytes_for_device_buffer};

#[derive(Debug, Clone, Copy)]
pub(super) struct DeviceAddressLayout {
    tile_rows: u32,
    tile_cols: u32,
    device_row_base: Option<Id>,
    device_col_base: Option<Id>,
    device_row_stride: u32,
}

/// Search the e-graph for a nested Theta (outer → inner) equivalent
/// to the given `output_id`. Returns (`outer_k`, `inner_theta_id`, `init_id`, `tg_buffers`)
/// if found.
///
/// This handles the case where the extractor picks a flat Theta but tg loads
/// from a tiled variant leaked into its e-classes. We recover the tiled
/// structure from the e-graph to build proper cooperative loads.
pub(super) fn find_nested_theta_in_egraph(
    output_id: Id,
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    device: &DeviceProfile,
) -> Option<(u32, Id, Id, Vec<TgBufferInfo>)> {
    let canonical = egraph.find(output_id);
    // Clone outer nodes to avoid borrow conflicts.
    let outer_nodes: Vec<TensorIr> = egraph[canonical].iter().cloned().collect();
    for node in &outer_nodes {
        if let TensorIr::Simd(SimdNode::Theta {
            children: [init, outer_count, inner],
            ..
        }) = node
        {
            let outer_k = match &egraph[*outer_count].data.constant {
                Some(ScalarValue::U32(v)) => *v,
                _ => continue,
            };
            let inner_canonical = egraph.find(*inner);
            let inner_nodes: Vec<TensorIr> = egraph[inner_canonical].iter().cloned().collect();
            for inner_node in &inner_nodes {
                if let TensorIr::Simd(SimdNode::Theta {
                    children: [_inner_init, inner_count, inner_update],
                    ..
                }) = inner_node
                {
                    let tile_k = match &egraph[*inner_count].data.constant {
                        Some(ScalarValue::U32(v)) => *v,
                        _ => continue,
                    };
                    let has_tg =
                        egraph_subtree_has_tg_loads(egraph, *inner_update, inner_canonical);
                    if has_tg {
                        let tg_bufs = collect_tg_buffer_info_with_device_addrs(
                            &[inner_canonical],
                            egraph,
                            chosen,
                            tile_k,
                            device,
                        )?;
                        if !selected_subtree_has_coherent_k_tile_stride(
                            egraph,
                            chosen,
                            inner_canonical,
                            inner_canonical,
                            tile_k,
                        ) {
                            continue;
                        }
                        return Some((outer_k, inner_canonical, *init, tg_bufs));
                    }
                }
            }
        }
    }
    None
}

/// Collect `TgBufferInfo` from the e-graph, including device address decomposition.
///
/// For each Load(Threadgroup(name), `tg_addr`) reachable from the inner Theta,
/// finds the sibling Load(Device(name), `dev_addr`) in the same e-class. The
/// device address is decomposed into row/col components to populate the
/// `TgBufferInfo` with device address mapping for cooperative loads.
pub(super) fn collect_tg_buffer_info_with_device_addrs(
    inner_theta_ids: &[Id],
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    tile_k: u32,
    device: &DeviceProfile,
) -> Option<Vec<TgBufferInfo>> {
    let mut merged: Vec<TgBufferInfo> = Vec::new();
    for &inner_theta_id in inner_theta_ids {
        let mut results = Vec::new();
        let mut visited = HashSet::new();
        collect_tg_with_device_rec(
            inner_theta_id,
            egraph,
            chosen,
            inner_theta_id,
            tile_k,
            device,
            &mut results,
            &mut visited,
        )?;

        for buf in results {
            if let Some(existing) = merged
                .iter_mut()
                .find(|existing| existing.tg_name == buf.tg_name)
            {
                merge_tg_buffer_info(egraph, existing, &buf);
            } else {
                merged.push(buf);
            }
        }
    }
    Some(merged)
}

pub(super) fn merge_tg_buffer_info(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    dst: &mut TgBufferInfo,
    src: &TgBufferInfo,
) {
    if dst.device_name != src.device_name || dst.device_row_stride != src.device_row_stride {
        return;
    }

    dst.size = dst.size.max(src.size);

    if dst.tile_cols == src.tile_cols {
        dst.tile_rows = dst
            .tile_rows
            .max(src.tile_rows)
            .max(dst.size.div_ceil(dst.tile_cols.max(1)));
    } else if dst.tile_rows == src.tile_rows {
        dst.tile_cols = dst
            .tile_cols
            .max(src.tile_cols)
            .max(dst.size.div_ceil(dst.tile_rows.max(1)));
    } else {
        let row_has_k_outer = dst.device_row_base.is_some_and(|id| {
            egraph[egraph.find(id)]
                .data
                .var_dep
                .contains(&VarRef::iter(1))
        });
        let col_has_k_outer = dst.device_col_base.is_some_and(|id| {
            egraph[egraph.find(id)]
                .data
                .var_dep
                .contains(&VarRef::iter(1))
        });

        if row_has_k_outer && !col_has_k_outer {
            dst.tile_cols = dst
                .tile_cols
                .max(src.tile_cols)
                .max(dst.size.div_ceil(dst.tile_rows.max(1)));
        } else if col_has_k_outer && !row_has_k_outer {
            dst.tile_rows = dst
                .tile_rows
                .max(src.tile_rows)
                .max(dst.size.div_ceil(dst.tile_cols.max(1)));
        } else {
            dst.tile_rows = dst.tile_rows.max(src.tile_rows);
            dst.tile_cols = dst.tile_cols.max(src.tile_cols);
            let packed_size = dst.tile_rows.saturating_mul(dst.tile_cols);
            if packed_size < dst.size {
                dst.tile_rows = dst.size.div_ceil(dst.tile_cols.max(1));
            }
        }
    }

    dst.size = dst.size.max(dst.tile_rows.saturating_mul(dst.tile_cols));
}

pub(super) fn collect_tg_buffer_info_for_load(
    canonical: Id,
    tg_name: BufferRef,
    tg_addr: Id,
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    tile_k: u32,
    device: &DeviceProfile,
) -> Option<TgBufferInfo> {
    // Device-tier counterpart of a threadgroup buffer is the same `BufferRef`.
    let dev_name = tg_name;
    // Size the buffer against the *worst-case* simdgroup promotion. The
    // rule-emitted Load's address depends on the `Workgroup` and
    // `Simdgroup` binders; downstream `scale_tg_buffers_for_simdgroups`
    // may pick any `num_simdgroups` in `[1, device.max_simdgroups]` later,
    // and we must guarantee the buffer fits the widest address range the
    // Load can reach under that choice. Using `max_simdgroups - 1` as the
    // conservative intra-physical-workgroup bound keeps the buffer large
    // enough for every valid promotion. Silent OOB reads occur when the
    // post-promotion Load reaches past a buffer sized for `workgroup_bound = 0`.
    let sg_bound = device.max_simdgroups.saturating_sub(1);
    let interval =
        compute_addr_interval_bounded(egraph, chosen, tg_addr, tile_k, device, sg_bound, sg_bound);
    let expected_size = interval.hi.saturating_add(1);
    let dev_addrs: Vec<Id> = egraph[canonical]
        .iter()
        .filter_map(|sibling| {
            if let TensorIr::Simd(SimdNode::Load {
                tier: MemTier::Device(buf),
                children,
            }) = sibling
            {
                (*buf == dev_name).then_some(children[0])
            } else {
                None
            }
        })
        .collect();

    let mut decompose_result = None;
    let mut best_size_delta = u32::MAX;
    for dev_addr_id in &dev_addrs {
        let result =
            decompose_device_address(egraph, chosen, *dev_addr_id, tg_addr, tile_k, device);
        let size = result.tile_rows.saturating_mul(result.tile_cols);
        let is_valid_2d = result.device_row_base.is_some() && result.device_col_base.is_some();
        let is_valid_1d = result.device_row_base.is_some()
            && result.device_col_base.is_none()
            && result.tile_cols == 1;
        if !(is_valid_2d || is_valid_1d) {
            continue;
        }
        let row_k_ok = result.device_row_base.is_none_or(|base| {
            !egraph[egraph.find(base)]
                .data
                .var_dep
                .contains(&VarRef::iter(1))
                || matches_k_outer_stride(egraph, chosen, base, tile_k)
        });
        let col_k_ok = result.device_col_base.is_none_or(|base| {
            !egraph[egraph.find(base)]
                .data
                .var_dep
                .contains(&VarRef::iter(1))
                || matches_k_outer_stride(egraph, chosen, base, tile_k)
        });
        if !(row_k_ok && col_k_ok) {
            continue;
        }
        let size_delta = size.abs_diff(expected_size);
        if decompose_result.is_none() || size_delta < best_size_delta {
            decompose_result = Some(result);
            best_size_delta = size_delta;
        }
        if size_delta == 0 {
            break;
        }
    }

    let DeviceAddressLayout {
        mut tile_rows,
        mut tile_cols,
        device_row_base,
        device_col_base,
        device_row_stride,
    } = decompose_result?;
    let k_var = VarRef::iter(0);
    let row_base = device_row_base.map(|id| substitute_var_with_zero(egraph, chosen, id, k_var));
    let col_base = device_col_base.map(|id| substitute_var_with_zero(egraph, chosen, id, k_var));

    if let Some(s) = extract_tg_addr_stride(egraph, chosen, tg_addr) {
        if s < tile_cols {
            return None;
        }
        if s > tile_cols {
            tile_cols = s;
            tile_rows = tile_rows.max(expected_size.div_ceil(tile_cols.max(1)));
        }
    }

    if std::env::var("TENSOR_IR_DEBUG_TG").is_ok() {
        eprintln!(
            "collect_tg_buffer_info_for_load: tg={tg_name:?} expected_size={expected_size} tile_rows={tile_rows} tile_cols={tile_cols}"
        );
    }
    Some(TgBufferInfo {
        tg_name,
        device_name: dev_name,
        size: expected_size,
        dtype_bytes: dtype_bytes_for_device_buffer(egraph, dev_name),
        tile_cols,
        tile_rows,
        device_row_base: row_base,
        device_col_base: col_base,
        device_row_stride,
        sg_read_stride: 0,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn collect_tg_with_device_rec(
    id: Id,
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    theta_class: Id,
    tile_k: u32,
    device: &DeviceProfile,
    results: &mut Vec<TgBufferInfo>,
    visited: &mut HashSet<Id>,
) -> Option<()> {
    let canonical = egraph.find(id);
    if canonical == egraph.find(theta_class) && !visited.is_empty() {
        return Some(()); // skip self-reference (but process on first call)
    }
    if !visited.insert(canonical) {
        return Some(());
    }

    let node = select_substitution_node(egraph, chosen, canonical)?;

    if let TensorIr::Simd(SimdNode::Load {
        tier: MemTier::Threadgroup(tg_name),
        children,
    }) = &node
    {
        let tg_addr = children[0];
        if !results.iter().any(|b| b.tg_name == *tg_name) {
            results.push(collect_tg_buffer_info_for_load(
                canonical, *tg_name, tg_addr, egraph, chosen, tile_k, device,
            )?);
        }
    }

    for child in node.children() {
        collect_tg_with_device_rec(
            *child,
            egraph,
            chosen,
            theta_class,
            tile_k,
            device,
            results,
            visited,
        )?;
    }
    Some(())
}

/// Decompose a device address into tile layout and offset components.
///
/// Given:
///   `device_addr` = `add(mul(row_expr`, stride), `col_expr`)
///   `tg_addr`     = `add(mul(thread_local`, `tile_cols`), `k_local`)
///
/// `row_base` is the workgroup/k_outer-dependent part of `row_expr`;
/// subtracting it from `row_expr` yields the lane-dependent part.
/// `col_base` is the k_outer-dependent part of `col_expr`;
/// subtracting it from `col_expr` yields the lane-dependent part.
pub(super) fn decompose_device_address(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    dev_addr: Id,
    tg_addr: Id,
    tile_k: u32,
    device: &DeviceProfile,
) -> DeviceAddressLayout {
    let mut visited = HashSet::new();
    if let Some(result) = decompose_device_address_rec(
        egraph,
        chosen,
        dev_addr,
        tg_addr,
        tile_k,
        device,
        &mut visited,
    ) {
        return result;
    }

    let size =
        compute_max_addr_from_egraph(egraph, chosen, tg_addr, tile_k, device).saturating_add(1);
    DeviceAddressLayout {
        tile_rows: size,
        tile_cols: 1,
        device_row_base: None,
        device_col_base: None,
        device_row_stride: 0,
    }
}

fn decompose_device_address_rec(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    dev_addr: Id,
    tg_addr: Id,
    tile_k: u32,
    device: &DeviceProfile,
    visited: &mut HashSet<Id>,
) -> Option<DeviceAddressLayout> {
    let dev_canonical = egraph.find(dev_addr);
    if !visited.insert(dev_canonical) {
        return None;
    }

    let k_var = VarRef::iter(0);
    let k_outer_var = VarRef::iter(1);
    if selected_subtree_is_k_only(egraph, chosen, dev_addr, k_var, k_outer_var) {
        // Extract k_outer base: the part of the address depending on _k_outer
        let k_base = extract_additive_base(egraph, chosen, dev_addr, k_outer_var);
        // 1D broadcast: tile_k elements, no column axis
        return Some(DeviceAddressLayout {
            tile_rows: tile_k,
            tile_cols: 1,
            device_row_base: k_base,
            device_col_base: None,
            device_row_stride: 1,
        });
    }

    // Collect all nodes to avoid borrow conflict with &mut egraph.
    let nodes: Vec<_> = egraph[dev_canonical].iter().cloned().collect();

    // Look for add(mul(row, stride), col) pattern in any node of the e-class.
    // Identity rules may merge e-classes so the add node might not be first.
    for node in &nodes {
        if let Some(inner_addr) = divmod_recomposed_value(egraph, node)
            && egraph.find(inner_addr) != dev_canonical
            && let Some(result) = decompose_device_address_rec(
                egraph, chosen, inner_addr, tg_addr, tile_k, device, visited,
            )
        {
            return Some(result);
        }

        if let TensorIr::BinOp(name, args) = node
            && matches!(name, BinaryOp::Add)
            && args.len() == 2
        {
            let left = args[0];
            let right = args[1];

            // Try left = mul(row, stride), right = col
            if let Some(result) =
                try_decompose_axes(egraph, chosen, left, right, tg_addr, tile_k, device)
            {
                return Some(result);
            }
            // Try right = mul(row, stride), left = col
            if let Some(result) =
                try_decompose_axes(egraph, chosen, right, left, tg_addr, tile_k, device)
            {
                return Some(result);
            }
        }
    }

    None
}

fn divmod_recomposed_value(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    node: &TensorIr,
) -> Option<Id> {
    let TensorIr::BinOp(BinaryOp::Add, args) = node else {
        return None;
    };
    for (div_scaled, modulo) in [(args[0], args[1]), (args[1], args[0])] {
        let Some((value, divisor)) = scaled_division(egraph, div_scaled) else {
            continue;
        };
        if modulo_matches(egraph, modulo, value, divisor) {
            return Some(value);
        }
    }
    None
}

fn scaled_division(egraph: &EGraph<TensorIr, TensorAnalysis>, id: Id) -> Option<(Id, u32)> {
    for node in egraph[egraph.find(id)].iter() {
        let TensorIr::BinOp(BinaryOp::Mul, args) = node else {
            continue;
        };
        for (div_side, const_side) in [(args[0], args[1]), (args[1], args[0])] {
            let Some(divisor) = const_u32(egraph, const_side).filter(|v| *v > 0) else {
                continue;
            };
            let Some(value) = division_by(egraph, div_side, divisor) else {
                continue;
            };
            return Some((value, divisor));
        }
    }
    None
}

fn division_by(egraph: &EGraph<TensorIr, TensorAnalysis>, id: Id, divisor: u32) -> Option<Id> {
    for node in egraph[egraph.find(id)].iter() {
        let TensorIr::BinOp(BinaryOp::Div, args) = node else {
            continue;
        };
        if const_u32(egraph, args[1]) == Some(divisor) {
            return Some(args[0]);
        }
    }
    None
}

fn modulo_matches(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    id: Id,
    value: Id,
    divisor: u32,
) -> bool {
    egraph[egraph.find(id)].iter().any(|node| {
        let TensorIr::BinOp(BinaryOp::Mod, args) = node else {
            return false;
        };
        egraph.find(args[0]) == egraph.find(value) && const_u32(egraph, args[1]) == Some(divisor)
    })
}

fn const_u32(egraph: &EGraph<TensorIr, TensorAnalysis>, id: Id) -> Option<u32> {
    match &egraph[egraph.find(id)].data.constant {
        Some(ScalarValue::U32(v)) => Some(*v),
        _ => None,
    }
}

pub(super) fn match_mul_row_expr(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    args: &[Id],
) -> Option<(Id, u32)> {
    if let Some(ScalarValue::U32(v)) = &egraph[args[1]].data.constant
        && *v > 1
    {
        return Some((args[0], *v));
    }
    if let Some(ScalarValue::U32(v)) = &egraph[args[0]].data.constant
        && *v > 1
    {
        return Some((args[1], *v));
    }
    None
}

pub(super) fn find_row_expr_and_stride(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    row_side: Id,
) -> Option<(Id, u32)> {
    let row_canonical = egraph.find(row_side);
    for node in egraph[row_canonical].iter() {
        if let TensorIr::BinOp(name, args) = node
            && matches!(name, BinaryOp::Mul)
            && args.len() == 2
            && let Some((row_expr, stride)) = match_mul_row_expr(egraph, args)
        {
            chosen.insert(row_canonical, node.clone());
            return Some((row_expr, stride));
        }
    }
    None
}

pub(super) fn k_base_matches_tile_stride(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    k_base: Option<Id>,
    tile_k: u32,
) -> bool {
    let Some(k_base) = k_base else {
        return true;
    };
    let kb_can = egraph.find(k_base);
    for node in egraph[kb_can].iter() {
        if let TensorIr::BinOp(name, args) = node
            && matches!(name, BinaryOp::Mul)
            && args.len() == 2
        {
            let left_matches =
                matches!(&egraph[args[0]].data.constant, Some(ScalarValue::U32(v)) if *v == tile_k);
            let right_matches =
                matches!(&egraph[args[1]].data.constant, Some(ScalarValue::U32(v)) if *v == tile_k);
            if left_matches || right_matches {
                return true;
            }
        }
        if matches!(node, TensorIr::Simd(SimdNode::Var(_))) {
            return true;
        }
    }
    false
}

pub(super) fn try_decompose_axes(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    row_side: Id,
    col_side: Id,
    _tg_addr: Id,
    tile_k: u32,
    device: &DeviceProfile,
) -> Option<DeviceAddressLayout> {
    let (row_expr, stride) = find_row_expr_and_stride(egraph, chosen, row_side)?;

    // Determine which axis varies with _k and which with lane.
    let k_var = VarRef::iter(0);
    let k_outer_var = VarRef::iter(1);

    let row_var_dep = &egraph[row_expr].data.var_dep;
    let col_var_dep = &egraph[col_side].data.var_dep;

    let row_has_k = row_var_dep.contains(&k_var) || row_var_dep.contains(&k_outer_var);
    let col_has_k = col_var_dep.contains(&k_var) || col_var_dep.contains(&k_outer_var);

    // One axis should have _k/_k_outer, the other should have lane
    if !row_has_k && !col_has_k {
        return None;
    }

    // Decompose each axis into base + local parts.
    // The base is the part depending on workgroup/k_outer.
    // The local is the part depending on lane/_k only.
    let (k_expr, thread_expr, k_is_row) = if col_has_k && !row_has_k {
        (col_side, row_expr, false)
    } else if row_has_k && !col_has_k {
        (row_expr, col_side, true)
    } else {
        // Both have _k — ambiguous
        return None;
    };

    // Extract the base offset from the k-axis expression.
    // The k-axis is typically: add(_k_outer * tile_k, _k)
    // The base is: _k_outer * tile_k
    let k_base = extract_additive_base(egraph, chosen, k_expr, k_outer_var);
    if !k_base_matches_tile_stride(egraph, k_base, tile_k) {
        return None;
    }

    // Extract the base offset from the thread-axis expression.
    // The thread-axis may be: add(wg_block_offset, div(add(wg_thread_offset, lane), C))
    // The base is the FULL workgroup contribution: wg_block_offset + div(wg_thread_offset, C).
    // We compute this by substituting the lane index with 0 in the expression.
    let thread_base = substitute_index_with_zero(egraph, chosen, thread_expr, IndexLevel::Lane);
    let thread_base = if thread_base == thread_expr {
        // If lane substitution didn't change anything, try extracting top-level wg base
        extract_workgroup_additive_base(egraph, chosen, thread_expr)
    } else {
        Some(thread_base)
    };

    // Compute tile dimensions from the local parts (after stripping bases).
    let k_range = tile_k;
    let thread_local_expr = thread_base.map(|base| {
        add_and_choose(
            egraph,
            chosen,
            TensorIr::BinOp(BinaryOp::Sub, [thread_expr, base]),
        )
    });
    let thread_range = thread_local_expr.map_or_else(
        || {
            if egraph[thread_expr].data.dep.contains_lane() {
                compute_thread_range_from_dep(egraph, chosen, thread_expr, device)
            } else {
                1
            }
        },
        |local_expr| compute_thread_range_from_dep(egraph, chosen, local_expr, device),
    );

    if k_is_row {
        // row = k-axis, col = thread-axis
        // tile layout: k_range rows, thread_range cols
        Some(DeviceAddressLayout {
            tile_rows: k_range,
            tile_cols: thread_range,
            device_row_base: k_base,
            device_col_base: thread_base,
            device_row_stride: stride,
        })
    } else {
        // row = thread-axis, col = k-axis
        // tile layout: thread_range rows, k_range cols
        Some(DeviceAddressLayout {
            tile_rows: thread_range,
            tile_cols: k_range,
            device_row_base: thread_base,
            device_col_base: k_base,
            device_row_stride: stride,
        })
    }
}

/// Extract the additive base from an expression that depends on `var_name`.
/// For `add(base, local)` where base contains `var_name`, returns base.
///
/// Also pins the matched `add` node and its base child in `chosen` so that
/// the codegen renders the correct representation after identity merges.
pub(super) fn extract_additive_base(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    expr: Id,
    var: VarRef,
) -> Option<Id> {
    let canonical = egraph.find(expr);
    let candidates: Vec<TensorIr> = chosen.get(&canonical).cloned().map_or_else(
        || egraph[canonical].iter().cloned().collect(),
        |node| vec![node],
    );
    for node in &candidates {
        if let TensorIr::BinOp(name, args) = node
            && matches!(name, BinaryOp::Add)
            && args.len() == 2
        {
            let left_has_var = egraph[args[0]].data.var_dep.contains(&var);
            let right_has_var = egraph[args[1]].data.var_dep.contains(&var);
            if left_has_var && !right_has_var {
                // Pin the add node itself and recurse to pin the base subtree
                chosen.insert(canonical, node.clone());
                pin_subtree_chosen(egraph, chosen, args[0]);
                return Some(args[0]);
            }
            if right_has_var && !left_has_var {
                chosen.insert(canonical, node.clone());
                pin_subtree_chosen(egraph, chosen, args[1]);
                return Some(args[1]);
            }
        }
    }
    // Check if the whole expression is the var-dependent part
    if egraph[expr].data.var_dep.contains(&var) {
        // Entire expression depends on var — it IS the base (local = 0)
        pin_subtree_chosen(egraph, chosen, expr);
        Some(expr)
    } else {
        None
    }
}

/// Extract the workgroup-dependent additive base from an expression.
/// For `add(wg_offset, lane_expr)`, returns `wg_offset`.
///
/// Also pins the matched `add` node and its base child in `chosen`.
pub(super) fn extract_workgroup_additive_base(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    expr: Id,
) -> Option<Id> {
    let canonical = egraph.find(expr);
    let candidates: Vec<TensorIr> = chosen.get(&canonical).cloned().map_or_else(
        || egraph[canonical].iter().cloned().collect(),
        |node| vec![node],
    );
    for node in &candidates {
        if let TensorIr::BinOp(name, args) = node
            && matches!(name, BinaryOp::Add)
            && args.len() == 2
        {
            let left_dep = egraph[args[0]].data.dep;
            let right_dep = egraph[args[1]].data.dep;

            let left_is_wg_only = left_dep.contains_workgroup()
                && !left_dep.contains_lane()
                && !left_dep.contains_simdgroup();
            let right_is_wg_only = right_dep.contains_workgroup()
                && !right_dep.contains_lane()
                && !right_dep.contains_simdgroup();

            if left_is_wg_only && !right_is_wg_only {
                chosen.insert(canonical, node.clone());
                pin_subtree_chosen(egraph, chosen, args[0]);
                return Some(args[0]);
            }
            if right_is_wg_only && !left_is_wg_only {
                chosen.insert(canonical, node.clone());
                pin_subtree_chosen(egraph, chosen, args[1]);
                return Some(args[1]);
            }
            // Both sides are workgroup-only (and neither pulls in lane or
            // simdgroup) — the entire `Add` is the workgroup-additive base.
            // This happens when the thread-axis expression, evaluated at
            // `lane = 0`, collapses into something like `row_base +
            // wg_in_tile` where *both* `row_base` (tile_row * tile_m) and
            // `wg_in_tile` (Mod(workgroup, wgs_per_tile)) are
            // workgroup-dependent. Both contributions must be included.
            if left_is_wg_only && right_is_wg_only {
                chosen.insert(canonical, node.clone());
                pin_subtree_chosen(egraph, chosen, canonical);
                return Some(canonical);
            }
        }
    }
    None
}

/// Recursively pin chosen nodes for an e-graph subtree.
///
/// For each e-class visited, if there's no existing `chosen_nodes` entry,
/// picks the "best" node representation. For `LowOp` nodes with constant
/// children (like mul(_`k_outer`, 16)), prefers the node whose constant
/// child has the largest value (to avoid identity-merged variants like
/// mul(_`k_outer`, 0) or mul(_`k_outer`, 1)).
pub(super) fn pin_subtree_chosen(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    id: Id,
) {
    let mut visited = HashSet::new();
    pin_subtree_chosen_rec(egraph, chosen, id, &mut visited);
}

pub(super) fn pin_subtree_chosen_rec(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    id: Id,
    visited: &mut HashSet<Id>,
) {
    let canonical = egraph.find(id);
    if !visited.insert(canonical) {
        return;
    }
    // If already pinned, just recurse into children of the pinned node
    if let Some(node) = chosen.get(&canonical).cloned() {
        for child in node.children() {
            pin_subtree_chosen_rec(egraph, chosen, *child, visited);
        }
        return;
    }

    // Pick the best node: prefer nodes whose constant children have
    // the largest values (avoids identity artifacts like mul(x, 0)).
    let nodes: Vec<TensorIr> = egraph[canonical].iter().cloned().collect();
    let mut best_node = None;
    let mut best_score = 0u64;

    for node in &nodes {
        let mut score = 1u64; // base score
        if let TensorIr::BinOp(name, args) = node
            && matches!(
                name,
                BinaryOp::Mul | BinaryOp::Add | BinaryOp::Div | BinaryOp::Mod
            )
        {
            for arg in args {
                if let Some(ScalarValue::U32(v)) = &egraph[*arg].data.constant {
                    score = score.saturating_add(u64::from(*v));
                }
            }
        }
        if score > best_score || best_node.is_none() {
            best_score = score;
            best_node = Some(node.clone());
        }
    }

    if let Some(node) = best_node {
        chosen.insert(canonical, node.clone());
        for child in node.children() {
            pin_subtree_chosen_rec(egraph, chosen, *child, visited);
        }
    }
}
