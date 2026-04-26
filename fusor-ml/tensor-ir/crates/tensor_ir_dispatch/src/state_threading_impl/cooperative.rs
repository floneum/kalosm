//! Cooperative-load building & tiled output construction.

use std::collections::{HashMap, HashSet};

use egg::{EGraph, Id, Language};

use crate::analysis::TensorAnalysis;
use crate::binding;
use crate::language::{DispatchNode, SimdNode, TensorIr, extract_list};
use crate::types::{
    BinaryOp, BinderKind, BufferRef, DeviceProfile, IndexLevel, LoweringOptions, MemTier,
    ScalarValue, VarRef, slots,
};
use crate::unroll::{k_step_lit, unroll_fold_substituted};

use super::*;
use super::{StateThreadedOutput, ThreadgroupTileInfo, add_and_choose};

/// Compute the number of simdgroups per physical workgroup for a tiled dispatch.
///
/// With multi-simdgroup workgroups, tg buffers are scaled to full tile size
/// and all simdgroups cooperatively load them. Each simdgroup reads its own
/// portion via an offset.
///
/// The number of simdgroups is chosen so that cooperative loads need ~1-2
/// iterations: simdgroups = `min(device.max_simdgroups, max_tg_buf_size / SIMD_WIDTH)`.
/// Must also divide workgroups evenly.
pub(super) fn compute_simdgroups_for_tiled(
    tg_buffers: &[ThreadgroupTileInfo],
    workgroups: u32,
    _num_outputs: u32,
    device: &DeviceProfile,
) -> u32 {
    fn scaling_fits_u32(buf: &ThreadgroupTileInfo, num_simdgroups: u32) -> bool {
        let tile_rows_needed = u64::from(buf.tile_rows.max(1));
        let num_simdgroups = u64::from(num_simdgroups.max(1));
        let per_sg_rows = tile_rows_needed.div_ceil(num_simdgroups).max(1);
        let full_rows = per_sg_rows * num_simdgroups;
        let tile_cols = u64::from(buf.tile_cols);
        full_rows <= u64::from(u32::MAX)
            && full_rows
                .checked_mul(tile_cols)
                .is_some_and(|size| size <= u64::from(u32::MAX))
            && per_sg_rows
                .checked_mul(tile_cols)
                .is_some_and(|stride| stride <= u64::from(u32::MAX))
    }

    if tg_buffers.is_empty() {
        return 1;
    }

    // Find the largest per-simdgroup buffer. The full-tile version of this
    // buffer is `max_buf * simdgroups` elements. We want enough threads
    // so that each thread loads ~1-2 elements from the largest buffer.
    let max_per_sg: u32 = tg_buffers.iter().map(|b| b.size).max().unwrap_or(0);
    if max_per_sg <= device.simd_width {
        // Already fits in 1 iteration with 1 simdgroup — no benefit.
        return 1;
    }

    // Target: full_tile / (sg * SIMD_WIDTH) ≈ 1..2 iterations.
    // full_tile = max_per_sg * sg
    // iterations = max_per_sg * sg / (sg * SIMD_WIDTH) = max_per_sg / SIMD_WIDTH
    // This is independent of sg! The cooperative load iterations per buffer
    // are always max_per_sg/SIMD_WIDTH regardless of simdgroups.
    //
    // BUT the total iterations across ALL buffers is sum(buf_size)/SIMD_WIDTH.
    // With multi-simdgroup and FULL-TILE buffers shared across simdgroups,
    // the total load is sum(full_size) / (sg * SIMD_WIDTH).
    // Where full_size = per_sg_size * sg for dimension-varying buffers,
    // and full_size = per_sg_size for shared buffers.
    //
    // For matmul: buf_A varies (scaled by sg), buf_B is shared.
    // total_load = (per_sg_A * sg + per_sg_B) / (sg * SIMD_WIDTH)
    //            ≈ per_sg_A/SIMD_WIDTH + per_sg_B/(sg*SIMD_WIDTH)
    // As sg increases, the B term gets smaller (amortized across simdgroups).
    //
    // Simple heuristic: use max(1, max_per_sg / device.simd_width) simdgroups.
    let max_useful = max_per_sg / device.simd_width;
    if max_useful <= 1 {
        return 1;
    }

    let mut sg = max_useful.min(device.max_simdgroups);
    // simdgroups must divide workgroups evenly (no cross-tile boundary)
    while sg > 1
        && (!workgroups.is_multiple_of(sg)
            || !tg_buffers.iter().all(|buf| scaling_fits_u32(buf, sg)))
    {
        sg -= 1;
    }
    sg
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct TiledOutput {
    /// Variable that holds the running accumulator inside this output's
    /// reduction body. For output 0 it is the depth-0 acc binder
    /// (`VarRef::acc(0)`); register-blocked outputs are represented by
    /// `VarRef::BlockedAcc { rm, rn }`.
    pub(super) acc_name: VarRef,
    pub(super) value_id: Id,
    pub(super) update_id: Option<Id>,
    pub(super) init_id: Id,
    pub(super) addr_id: Id,
    pub(super) result_slot: usize,
    pub(super) state_slot: Option<usize>,
}

/// Build a state-threaded body for a tiled dispatch with cooperative loading.
///
/// Produces:
/// ```text
/// Let { _acc = Var(_acc) }             // self-ref marker (makes _acc a local var)
/// Loop(k_outer < outer_count) {
///   Let { _k_outer = Var(_k_outer) }   // self-ref marker
///   // cooperative loads
///   Barrier([tg_buf_0, tg_buf_1, ...])
///   Let { _acc = inner_theta }          // updates accumulator each iteration
/// }
/// ```
///
/// Check whether an expression tree (rooted at `id`) contains `Index(Workgroup)`.
/// This indicates the expression varies per simdgroup in multi-simdgroup mode.
pub(super) fn expr_depends_on_workgroup(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen: &HashMap<Id, TensorIr>,
    id: Id,
) -> bool {
    let mut visited = HashSet::new();
    expr_depends_on_workgroup_rec(egraph, chosen, id, &mut visited)
}

pub(super) fn expr_depends_on_workgroup_rec(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen: &HashMap<Id, TensorIr>,
    id: Id,
    visited: &mut HashSet<Id>,
) -> bool {
    let canonical = egraph.find(id);
    if !visited.insert(canonical) {
        return false;
    }
    let node = select_substitution_node(egraph, chosen, canonical)
        .unwrap_or_else(|| egraph[canonical].iter().next().unwrap().clone());
    match &node {
        TensorIr::Simd(SimdNode::Var(VarRef::Bound {
            kind: BinderKind::Dispatch,
            slot: slots::DISPATCH_WORKGROUP | slots::DISPATCH_SIMDGROUP,
            depth: 0,
        })) => true,
        _ => node
            .children()
            .iter()
            .any(|c| expr_depends_on_workgroup_rec(egraph, chosen, *c, visited)),
    }
}

pub(super) fn make_pack(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    values: &[Id],
) -> Id {
    let values = add_list(egraph, values);
    add_and_choose(
        egraph,
        chosen,
        TensorIr::Dispatch(DispatchNode::Pack {
            children_list: values,
        }),
    )
}

pub(super) fn make_extract(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    tuple: Id,
    index: usize,
) -> Id {
    add_and_choose(
        egraph,
        chosen,
        TensorIr::Dispatch(DispatchNode::Extract {
            index: u32::try_from(index).expect("extract index fits in u32"),
            tuple,
        }),
    )
}

fn tuple_arity(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen: &HashMap<Id, TensorIr>,
    id: Id,
) -> Option<usize> {
    let canonical = egraph.find(id);
    let node = select_substitution_node(egraph, chosen, canonical)
        .unwrap_or_else(|| egraph[canonical].iter().next().unwrap().clone());
    if let TensorIr::Dispatch(DispatchNode::Pack { children_list }) = node {
        return Some(extract_list(egraph, children_list).len());
    }
    None
}

fn replace_tuple_slot(
    tuple: Id,
    arity: usize,
    index: usize,
    replacement: Id,
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
) -> Id {
    let values: Vec<Id> = (0..arity)
        .map(|slot| {
            if slot == index {
                replacement
            } else {
                make_extract(egraph, chosen, tuple, slot)
            }
        })
        .collect();
    make_pack(egraph, chosen, &values)
}

pub(super) fn make_store(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    tier: MemTier,
    addr: Id,
    value: Id,
    state: Id,
) -> Id {
    add_and_choose(
        egraph,
        chosen,
        TensorIr::Simd(SimdNode::Store {
            tier,
            children: [addr, value, state],
        }),
    )
}

pub(super) fn make_store_if(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    tier: MemTier,
    cond: Id,
    addr: Id,
    value: Id,
    state: Id,
) -> Id {
    add_and_choose(
        egraph,
        chosen,
        TensorIr::Simd(SimdNode::StoreIf {
            tier,
            children: [cond, addr, value, state],
        }),
    )
}

pub(super) fn make_barrier(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    regions: Vec<BufferRef>,
    state: Id,
) -> Id {
    add_and_choose(
        egraph,
        chosen,
        TensorIr::Simd(SimdNode::Barrier { regions, state }),
    )
}

pub(super) fn remap_threadgroup_load_state(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    root: Id,
    state: Id,
) -> Id {
    fn rec(
        egraph: &mut EGraph<TensorIr, TensorAnalysis>,
        chosen: &mut HashMap<Id, TensorIr>,
        id: Id,
        state: Id,
        memo: &mut HashMap<Id, Id>,
    ) -> Id {
        let canonical = egraph.find(id);
        if let Some(&cached) = memo.get(&canonical) {
            return cached;
        }

        let Some(node) = select_substitution_node(egraph, chosen, canonical) else {
            return id;
        };

        let rebuilt = match node {
            TensorIr::Simd(SimdNode::Load {
                tier: MemTier::Threadgroup(tier_name),
                children,
            }) => {
                let addr = rec(egraph, chosen, children[0], state, memo);
                TensorIr::Simd(SimdNode::Load {
                    tier: MemTier::Threadgroup(tier_name),
                    children: [addr, state],
                })
            }
            mut other => {
                for child in other.children_mut() {
                    let child_canonical = egraph.find(*child);
                    if child_canonical == canonical {
                        continue;
                    }
                    *child = rec(egraph, chosen, *child, state, memo);
                }
                other
            }
        };

        let new_id = add_and_choose(egraph, chosen, rebuilt);
        memo.insert(canonical, new_id);
        new_id
    }

    let mut memo = HashMap::new();
    rec(egraph, chosen, root, state, &mut memo)
}

pub(super) fn build_cooperative_load_state(
    state_in: Id,
    buf_info: &ThreadgroupTileInfo,
    num_simdgroups: u32,
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    device: &DeviceProfile,
) -> Id {
    let total_threads = num_simdgroups * device.simd_width;
    let region_size = buf_info.size;
    let device_buf = buf_info.device_name;
    let tg_name = buf_info.tg_name;

    let sg_idx = add_and_choose(
        egraph,
        chosen,
        TensorIr::Simd(SimdNode::Var(VarRef::thread(IndexLevel::Simdgroup))),
    );
    let lane_idx = add_and_choose(
        egraph,
        chosen,
        TensorIr::Simd(SimdNode::Var(VarRef::thread(IndexLevel::Lane))),
    );
    let sw_lit = add_and_choose(
        egraph,
        chosen,
        TensorIr::Const(ScalarValue::U32(device.simd_width)),
    );
    let sg_times_sw = add_and_choose(
        egraph,
        chosen,
        TensorIr::BinOp(BinaryOp::Mul, [sg_idx, sw_lit]),
    );
    let tid = add_and_choose(
        egraph,
        chosen,
        TensorIr::BinOp(BinaryOp::Add, [sg_times_sw, lane_idx]),
    );

    let iterations = region_size.div_ceil(total_threads);
    let region_size_lit = add_and_choose(
        egraph,
        chosen,
        TensorIr::Const(ScalarValue::U32(region_size)),
    );

    let mut state = state_in;
    for i in 0..iterations {
        let flat_tg_idx_i = if i == 0 {
            tid
        } else {
            let offset_lit = add_and_choose(
                egraph,
                chosen,
                TensorIr::Const(ScalarValue::U32(i * total_threads)),
            );
            add_and_choose(
                egraph,
                chosen,
                TensorIr::BinOp(BinaryOp::Add, [offset_lit, tid]),
            )
        };

        let device_addr_i = compute_device_addr(flat_tg_idx_i, buf_info, egraph, chosen);
        let device_addr_i =
            substitute_var_with_zero(egraph, chosen, device_addr_i, VarRef::iter(0));
        let dev_load_i = add_and_choose(
            egraph,
            chosen,
            TensorIr::Simd(SimdNode::Load {
                tier: MemTier::Device(device_buf),
                children: [device_addr_i, state],
            }),
        );

        let needs_guard = i > 0 || !region_size.is_multiple_of(total_threads);
        if needs_guard && (i + 1) * total_threads > region_size {
            let cond = add_and_choose(
                egraph,
                chosen,
                TensorIr::BinOp(BinaryOp::Lt, [flat_tg_idx_i, region_size_lit]),
            );
            state = make_store_if(
                egraph,
                chosen,
                MemTier::Threadgroup(tg_name),
                cond,
                flat_tg_idx_i,
                dev_load_i,
                state,
            );
        } else {
            state = make_store(
                egraph,
                chosen,
                MemTier::Threadgroup(tg_name),
                flat_tg_idx_i,
                dev_load_i,
                state,
            );
        }
    }

    state
}

pub(super) fn build_tiled_inner_theta(
    output: &TiledOutput,
    current_acc: Id,
    state: Id,
    tile_k: u32,
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
) -> Id {
    let update_id = output
        .update_id
        .expect("non-unrolled tiled kernels require update expressions");
    // We're about to wrap the body in a new `KInner` Theta whose own `Acc(0)`
    // binder takes over the running-accumulator role. Single-output kernels
    // already encode the running acc as `Bound { Acc, 0 }` in the body — those
    // refs *correctly* resolve to the new binder once wrapped, so no
    // substitution is needed. Register-blocked kernels, however, name their
    // accumulators with kernel-scope `BlockedAcc { rm, rn }` (one per output)
    // and need each one threaded to its current carry value.
    let mut update_expr = if matches!(output.acc_name, VarRef::Bound { .. }) {
        update_id
    } else {
        substitute_var_with_id(egraph, chosen, update_id, output.acc_name, current_acc)
    };
    // The state token threaded into threadgroup loads is built at the outer
    // scope; shift its free `Bound` refs up by one so they keep pointing at
    // the outer binder once the body sits under the new `KInner` binder.
    let body_state = binding::shift_in_egraph(egraph, chosen, state, BinderKind::Theta, 0, 1);
    update_expr = remap_threadgroup_load_state(egraph, chosen, update_expr, body_state);
    let tile_k_lit = add_and_choose(egraph, chosen, TensorIr::Const(ScalarValue::U32(tile_k)));
    add_and_choose(
        egraph,
        chosen,
        TensorIr::Simd(SimdNode::Theta {
            children: [current_acc, tile_k_lit, update_expr],
        }),
    )
}

pub(super) fn scale_tg_buffers_for_simdgroups(
    tg_buffers: &[ThreadgroupTileInfo],
    num_simdgroups: u32,
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    _device: &DeviceProfile,
) -> Vec<ThreadgroupTileInfo> {
    if num_simdgroups <= 1 {
        return tg_buffers.to_vec();
    }

    let sg_idx = add_and_choose(
        egraph,
        chosen,
        TensorIr::Simd(SimdNode::Var(VarRef::thread(IndexLevel::Simdgroup))),
    );

    tg_buffers
        .iter()
        .map(|buf| {
            let depends_on_wg = buf
                .device_row_base
                .is_some_and(|rb| expr_depends_on_workgroup(egraph, chosen, rb));
            if !depends_on_wg {
                return buf.clone();
            }

            // Target: the buffer holds `buf.tile_rows` rows total when
            // `num_simdgroups` simdgroups cooperate. Each simdgroup owns
            // `per_sg_rows = tile_rows / num_simdgroups` consecutive rows;
            // the cooperative Store's `device_row_base` subtracts
            // `sg_idx * per_sg_rows` so every simdgroup's flat-tg index
            // lands in its own row shard. We pick the smallest
            // `num_simdgroups`-divisible `per_sg_rows` that still covers
            // `buf.tile_rows` — that way the Load's tile-local address
            // (which reaches up to `buf.tile_rows * buf.tile_cols - 1`)
            // fits in `full_size`.
            let tile_rows_needed = buf.tile_rows.max(1);
            let per_sg_rows = tile_rows_needed.div_ceil(num_simdgroups).max(1);
            let full_rows = per_sg_rows * num_simdgroups;
            let full_size = full_rows * buf.tile_cols;
            let device_row_base = buf.device_row_base.map(|rb| {
                let per_sg_rows_lit = add_and_choose(
                    egraph,
                    chosen,
                    TensorIr::Const(ScalarValue::U32(per_sg_rows)),
                );
                let sg_row_offset = add_and_choose(
                    egraph,
                    chosen,
                    TensorIr::BinOp(BinaryOp::Mul, [sg_idx, per_sg_rows_lit]),
                );
                add_and_choose(
                    egraph,
                    chosen,
                    TensorIr::BinOp(BinaryOp::Sub, [rb, sg_row_offset]),
                )
            });

            ThreadgroupTileInfo {
                tg_name: buf.tg_name,
                device_name: buf.device_name,
                size: full_size,
                dtype_bytes: buf.dtype_bytes,
                tile_cols: buf.tile_cols,
                tile_rows: full_rows,
                device_row_base,
                device_col_base: buf.device_col_base,
                device_row_stride: buf.device_row_stride,
            }
        })
        .collect()
}

pub(super) fn init_tiled_carry_state(
    tiled_outputs: &[TiledOutput],
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
) -> (Id, Vec<Id>, Id) {
    let init_state = add_and_choose(egraph, chosen, TensorIr::Dispatch(DispatchNode::Token));
    let init_pack_values: Vec<Id> = tiled_outputs
        .iter()
        .map(|output| output.init_id)
        .chain(std::iter::once(init_state))
        .collect();
    let init_pack = make_pack(egraph, chosen, &init_pack_values);
    let acc_pack_var = add_and_choose(
        egraph,
        chosen,
        TensorIr::Simd(SimdNode::Var(VarRef::acc(0))),
    );
    let current_accs = (0..tiled_outputs.len())
        .map(|index| make_extract(egraph, chosen, acc_pack_var, index))
        .collect();
    let state = make_extract(egraph, chosen, acc_pack_var, tiled_outputs.len());
    (init_pack, current_accs, state)
}

pub(super) fn apply_tiled_load_prologue(
    mut state: Id,
    scaled_tg_buffers: &[ThreadgroupTileInfo],
    num_simdgroups: u32,
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    device: &DeviceProfile,
) -> Id {
    let barrier_regions: Vec<BufferRef> = scaled_tg_buffers.iter().map(|buf| buf.tg_name).collect();
    for buf in scaled_tg_buffers {
        state = build_cooperative_load_state(state, buf, num_simdgroups, egraph, chosen, device);
    }
    if barrier_regions.is_empty() {
        state
    } else {
        make_barrier(egraph, chosen, barrier_regions, state)
    }
}

pub(super) fn build_tiled_update_expr(
    output: &TiledOutput,
    current_acc: Id,
    state: Id,
    k_lit: Option<Id>,
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
) -> Id {
    let source = output.update_id.unwrap_or(output.value_id);
    let mut expr = substitute_var_with_id(egraph, chosen, source, output.acc_name, current_acc);
    if let Some(k_lit) = k_lit {
        // Unrolling: substitute the inner Theta's iter(0) binder with a
        // constant. The accompanying `shift(cutoff=1, delta=-1)` that
        // removes the now-consumed binder happens ONCE at the top of
        // `build_tiled_outputs`, after all unroll iterations have finished —
        // doing it here would conflict with re-entering this helper on the
        // next unroll step (the surviving `iter(1)` refs would be mistakenly
        // collapsed too early).
        expr = substitute_var_with_id(egraph, chosen, expr, VarRef::iter(0), k_lit);
    }
    remap_threadgroup_load_state(egraph, chosen, expr, state)
}

pub(super) fn unroll_multi_output_tiled_accs(
    tile_k: u32,
    tiled_outputs: &[TiledOutput],
    current_accs: &mut [Id],
    state: Id,
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
) {
    for k_step in 0..tile_k {
        let k_lit = k_step_lit(egraph, chosen, k_step);
        for (index, output) in tiled_outputs.iter().enumerate() {
            current_accs[index] = build_tiled_update_expr(
                output,
                current_accs[index],
                state,
                Some(k_lit),
                egraph,
                chosen,
            );
        }
    }
}

pub(super) fn update_multi_output_tiled_accs(
    tile_k: u32,
    tiled_outputs: &[TiledOutput],
    current_accs: &mut [Id],
    state: Id,
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    lowering: &LoweringOptions,
) {
    if tile_k <= fused_inner_unroll_threshold(lowering) {
        unroll_multi_output_tiled_accs(tile_k, tiled_outputs, current_accs, state, egraph, chosen);
        return;
    }

    for (index, output) in tiled_outputs.iter().enumerate() {
        current_accs[index] =
            build_tiled_inner_theta(output, current_accs[index], state, tile_k, egraph, chosen);
    }
}

pub(super) fn update_single_tiled_acc(
    output: &TiledOutput,
    current_acc: Id,
    state: Id,
    tile_k: Option<u32>,
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    lowering: &LoweringOptions,
) -> Id {
    let Some(update_id) = output.update_id else {
        return build_tiled_update_expr(output, current_acc, state, None, egraph, chosen);
    };
    let Some(tile_k) = tile_k else {
        return build_tiled_update_expr(output, current_acc, state, None, egraph, chosen);
    };
    if tile_k > fused_inner_unroll_threshold(lowering) {
        return build_tiled_inner_theta(output, current_acc, state, tile_k, egraph, chosen);
    }

    // The outer-binder-collapse shift (iter(1) → iter(0) etc.) happens once
    // over the entire `update_pack` in `build_tiled_outputs`, not here.
    unroll_fold_substituted(
        egraph,
        chosen,
        tile_k,
        current_acc,
        |egraph, chosen, k_step, acc| {
            let k_lit = k_step_lit(egraph, chosen, k_step);
            let expr = substitute_var_with_id(egraph, chosen, update_id, output.acc_name, acc);
            let expr = substitute_var_with_id(egraph, chosen, expr, VarRef::iter(0), k_lit);
            remap_threadgroup_load_state(egraph, chosen, expr, state)
        },
    )
}

pub(super) fn build_tiled_output_elements(
    outer_theta: Id,
    tiled_outputs: &[TiledOutput],
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
) -> Vec<StateThreadedOutput> {
    tiled_outputs
        .iter()
        .map(|output| StateThreadedOutput {
            value_id: make_extract(egraph, chosen, outer_theta, output.result_slot),
            addr_id: output.addr_id,
        })
        .collect()
}

/// Build pure output expressions for a tiled dispatch. The outer theta carries
/// all accumulator values plus an explicit state token; cooperative loads and
/// barriers become pure state-transforming nodes inside the update term.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_tiled_outputs(
    outer_count: u32,
    tile_k: Option<u32>,
    tiled_outputs: &[TiledOutput],
    tg_buffers: &[ThreadgroupTileInfo],
    num_simdgroups: u32,
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    device: &DeviceProfile,
    lowering: &LoweringOptions,
) -> (Vec<StateThreadedOutput>, Vec<ThreadgroupTileInfo>) {
    let scaled_tg_buffers =
        scale_tg_buffers_for_simdgroups(tg_buffers, num_simdgroups, egraph, chosen, device);

    if tiled_outputs.len() == 1 && tiled_outputs[0].state_slot.is_some() {
        let output = &tiled_outputs[0];
        let state_slot = output
            .state_slot
            .expect("stateful tiled output must record its state slot");
        let arity = tuple_arity(egraph, chosen, output.init_id)
            .unwrap_or_else(|| state_slot.saturating_add(1));
        let outer_count_lit = add_and_choose(
            egraph,
            chosen,
            TensorIr::Const(ScalarValue::U32(outer_count)),
        );
        let acc_var = add_and_choose(
            egraph,
            chosen,
            TensorIr::Simd(SimdNode::Var(VarRef::acc(0))),
        );
        let state = make_extract(egraph, chosen, acc_var, state_slot);
        let state = apply_tiled_load_prologue(
            state,
            &scaled_tg_buffers,
            num_simdgroups,
            egraph,
            chosen,
            device,
        );
        let current_acc = replace_tuple_slot(acc_var, arity, state_slot, state, egraph, chosen);
        let mut updated_acc =
            update_single_tiled_acc(output, current_acc, state, tile_k, egraph, chosen, lowering);

        if !scaled_tg_buffers.is_empty() && num_simdgroups > 1 {
            let end_regions = scaled_tg_buffers.iter().map(|buf| buf.tg_name).collect();
            let end_state = make_barrier(egraph, chosen, end_regions, state);
            updated_acc =
                replace_tuple_slot(updated_acc, arity, state_slot, end_state, egraph, chosen);
        }

        let updated_acc =
            binding::shift_in_egraph(egraph, chosen, updated_acc, BinderKind::Theta, 1, -1);
        let outer_theta = add_and_choose(
            egraph,
            chosen,
            TensorIr::Simd(SimdNode::Theta {
                children: [output.init_id, outer_count_lit, updated_acc],
            }),
        );
        return (
            vec![StateThreadedOutput {
                value_id: make_extract(egraph, chosen, outer_theta, output.result_slot),
                addr_id: output.addr_id,
            }],
            scaled_tg_buffers,
        );
    }

    // Post-promotion Load address fixup: the phase-1 rule emits TG Load
    // tile-local addresses in terms of `wg_in_tile = Mod(Workgroup,
    // wgs_per_tile)` because before simdgroup promotion each virtual
    // workgroup maps to one simdgroup. After promotion with
    // `num_simdgroups > 1`, the *intra-physical-workgroup* variation of
    // `Workgroup` is exactly `Simdgroup` (they enumerate the same 0..sg
    // range), and `wg_in_tile` / `wgs_per_tile`'s division is a
    // phys-wg-constant that's already captured by each physical
    // workgroup's private TG buffer. Substituting `Workgroup → Simdgroup`
    // inside every TG Load's `tg_addr` rewrites the Load to index the
    // buffer the Store wrote into — without it, the higher-physical
    // workgroups silently read past their buffer (OOB → zeros on Metal).
    let owned_tiled_outputs: Vec<TiledOutput> = tiled_outputs.to_vec();
    let tiled_outputs = owned_tiled_outputs.as_slice();
    let outer_count_lit = add_and_choose(
        egraph,
        chosen,
        TensorIr::Const(ScalarValue::U32(outer_count)),
    );
    let (init_pack, mut current_accs, mut state) =
        init_tiled_carry_state(tiled_outputs, egraph, chosen);
    state = apply_tiled_load_prologue(
        state,
        &scaled_tg_buffers,
        num_simdgroups,
        egraph,
        chosen,
        device,
    );

    if tiled_outputs.len() > 1 {
        let tile_k = tile_k.expect("register-blocked tiled kernels require tile_k");
        update_multi_output_tiled_accs(
            tile_k,
            tiled_outputs,
            &mut current_accs,
            state,
            egraph,
            chosen,
            lowering,
        );
    } else {
        current_accs[0] = update_single_tiled_acc(
            &tiled_outputs[0],
            current_accs[0],
            state,
            tile_k,
            egraph,
            chosen,
            lowering,
        );
    }

    if !scaled_tg_buffers.is_empty() && num_simdgroups > 1 {
        let end_regions = scaled_tg_buffers.iter().map(|buf| buf.tg_name).collect();
        state = make_barrier(egraph, chosen, end_regions, state);
    }

    let update_pack_values: Vec<Id> = current_accs
        .into_iter()
        .chain(std::iter::once(state))
        .collect();
    let update_pack = make_pack(egraph, chosen, &update_pack_values);
    // Inner-K unrolling substitutes the inner Theta's `iter(0)` with
    // constants and splices `iter(1)` (outer-K) references from the fused-
    // reduce rewrite. This leaves a single binder — the outer Theta — but
    // refs to outer-K are still written as `iter(1)`. Shift every free
    // `Bound { depth >= 1 }` down by one so the outer Theta's `iter(0)`
    // slot resolves them. Cooperative-load prologue addresses (which also
    // reference `iter(1)` for the outer-K stripe) are inside `update_pack`
    // too and get corrected in the same pass.
    let update_pack =
        binding::shift_in_egraph(egraph, chosen, update_pack, BinderKind::Theta, 1, -1);
    let outer_theta = add_and_choose(
        egraph,
        chosen,
        TensorIr::Simd(SimdNode::Theta {
            children: [init_pack, outer_count_lit, update_pack],
        }),
    );
    let outputs = build_tiled_output_elements(outer_theta, tiled_outputs, egraph, chosen);
    (outputs, scaled_tg_buffers)
}
