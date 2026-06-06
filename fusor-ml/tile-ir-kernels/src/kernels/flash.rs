use fusor_tile_ir::{
    tile::{range, Mask, PrivateLocal, Storage, Tile, TileBlock, WorkgroupTile},
    ElementType, ScalarElement, SubgroupToken, TileLiteral, WorkgroupAxis,
};

use super::helpers::{index_n, reduce_workgroup, supports_float, u32_tile, zero_fill, NEG_MAX_F32};
use super::softmax::{softmax_partial_scale, workgroup_softmax_block};
use super::types::{FlashAttentionDims, FlashAttentionMeta, FlashDecodeSmallMeta};

const FLASH_BLOCK: u32 = 256;
const TILED_OUTS_PER_SUBGROUP: u32 = 4;

/// Runtime tensor bindings consumed by the streaming flash-attention kernels.
pub struct FlashAttentionTensors<B> {
    pub q: fusor_tile_ir::KernelTensorRef<B>,
    pub k: fusor_tile_ir::KernelTensorRef<B>,
    pub v: fusor_tile_ir::KernelTensorRef<B>,
    pub mask: Option<fusor_tile_ir::KernelTensorRef<B>>,
    pub output: fusor_tile_ir::KernelTensorRef<B>,
}

/// Storage-typed `0.0` literal for the kernel's `element` (F32 or F16).
fn f32_tile(value: f32) -> Tile {
    Tile::literal(TileLiteral::f32(value))
}

/// Number of output dimensions a single workgroup writes per KV-axis pass,
/// given the device's hardware subgroup size. Each subgroup handles one
/// output dim; the workgroup contains `FLASH_BLOCK / subgroup_size` subgroups.
pub const fn flash_outputs_per_workgroup(subgroup_size: u32) -> u32 {
    FLASH_BLOCK / subgroup_size
}

/// Number of output dimensions the tiled prefill kernel writes per workgroup.
///
/// It keeps the physical workgroup at `FLASH_BLOCK` lanes, but each subgroup
/// serially accumulates a small run of output dimensions while reusing the
/// same QK score.
pub const fn flash_tiled_outputs_per_workgroup(subgroup_size: u32) -> u32 {
    flash_outputs_per_workgroup(subgroup_size) * TILED_OUTS_PER_SUBGROUP
}

/// Build a streaming flash-attention kernel for F32 or F16 tensors.
///
/// `element` is the runtime storage element (F32 or F16); the accumulator math
/// is always F32 (ARBOR_DESIGN.md §2 — element is data, not a marker type).
///
/// `subgroup_size` must match the runtime hardware subgroup width on the
/// target device — the kernel layout assigns one subgroup per output dim and
/// uses `subgroup_reduce_*` to fold a `subgroup_size`-wide chunk of KV. Pick
/// the size by reading the device's subgroup caps; pinning is not exposed by
/// wgpu, so this must come from a fixed `(min == max)` adapter range.
///
/// The metadata supplies tensor strides, offsets, dimensions, scale, and the
/// dispatch grid. Returns `None` if the tensor ranks or optional mask binding
/// are inconsistent with the metadata.
pub fn flash_attention<B>(
    kb: &mut fusor_tile_ir::KernelBuilder<B>,
    element: ElementType,
    tensors: FlashAttentionTensors<B>,
    meta: FlashAttentionMeta,
    subgroup: SubgroupToken,
    subgroup_size: u32,
) -> Option<()> {
    let FlashAttentionTensors {
        q,
        k,
        v,
        mask,
        output,
    } = tensors;
    if !supports_float(element) {
        return None;
    }
    if subgroup_size == 0 || !FLASH_BLOCK.is_multiple_of(subgroup_size) {
        return None;
    }
    let outputs_per_workgroup = flash_outputs_per_workgroup(subgroup_size);
    let q_strides: [u32; 4] = meta.q_meta.strides.as_slice().try_into().ok()?;
    let k_strides: [u32; 4] = meta.k_meta.strides.as_slice().try_into().ok()?;
    let v_strides: [u32; 4] = meta.v_meta.strides.as_slice().try_into().ok()?;
    let output_strides: [u32; 4] = meta.output_meta.strides.as_slice().try_into().ok()?;
    let mask_strides: Option<[u32; 2]> = if let Some(mask) = meta.mask_meta.as_ref() {
        Some(mask.strides.as_slice().try_into().ok()?)
    } else {
        None
    };
    if meta.dims.batch == 0
        || meta.dims.num_heads == 0
        || meta.dims.num_kv_heads == 0
        || meta.dims.q_seq_len == 0
        || meta.dims.kv_seq_len == 0
        || meta.dims.head_dim == 0
    {
        return None;
    }
    if meta.mask_meta.is_some() != mask.is_some() {
        return None;
    }
    if meta.causal && meta.mask_meta.is_some() {
        // Causal mode is mutually exclusive with an explicit additive mask:
        // the kernel skips out-of-bound kv positions and emits no mask add.
        return None;
    }
    let groups = meta.dims.num_heads.checked_div(meta.dims.num_kv_heads)?;
    if groups == 0 {
        return None;
    }
    let elem_fill = zero_fill(element);
    let q = kb.read(element, q);
    let k = kb.read(element, k);
    let v = kb.read(element, v);
    let mask = mask.map(|m| kb.read(element, m));
    let output = kb.write(element, output);
    let phase = kb.program();
    {
        phase.program_grid(FLASH_BLOCK, meta.dispatch_size, |program| {
            let lane = program.lane();
            let workgroup_x = program.program_id(WorkgroupAxis::X);
            let row = program.program_id(WorkgroupAxis::Y);
            let q_idx = program.bind(row.clone() % meta.dims.q_seq_len);
            let row_over_q = row.clone() / meta.dims.q_seq_len;
            let head_idx = program.bind(row_over_q.clone() % meta.dims.num_heads);
            let batch_idx = program.bind(row / (meta.dims.q_seq_len * meta.dims.num_heads));
            let kv_head_idx = program.bind(head_idx.clone() / u32_tile(groups));
            let kv_lane = lane.clone() % subgroup_size;
            let out_dim =
                program.bind(workgroup_x * outputs_per_workgroup + (lane.clone() / subgroup_size));
            let out_valid = program.bind(out_dim.clone().lt(u32_tile(meta.dims.head_dim)));
            // Per-iteration scratch locals — used to bridge values across
            // `if_then` branches inside the body. Not loop-carried.
            let score_local = program.private(ElementType::F32);
            let weighted_local = program.private(ElementType::F32);

            let kv_chunks = meta.dims.kv_seq_len.div_ceil(subgroup_size);
            let causal = meta.causal;
            let [_final_m, final_s, final_o] = program.fold(
                range(kv_chunks),
                [f32_tile(NEG_MAX_F32), f32_tile(0.0), f32_tile(0.0)],
                |program, chunk_idx, [m_state, s_state, o_state]| {
                    let chunk = chunk_idx;
                    let kv_idx =
                        program.bind(chunk.clone() * u32_tile(subgroup_size) + kv_lane.clone());
                    let bound_valid =
                        program.bind(kv_idx.clone().lt(u32_tile(meta.dims.kv_seq_len)));
                    // For causal attention we additionally restrict to kv <= q.
                    // We can also skip the per-dim Q·K work for an entire
                    // chunk when `chunk_start > q_idx`: in that case every
                    // lane's kv_idx > q_idx and the chunk's contribution is
                    // zero. We still need to fold (so the post-loop
                    // accumulator state is correct), but we gate the heavy
                    // load+dot under `chunk_in_range`.
                    let (kv_valid, chunk_in_range) = if causal {
                        let chunk_start = program.bind(chunk.clone() * u32_tile(subgroup_size));
                        let chunk_in_range = program.bind(chunk_start.le(q_idx.clone()));
                        let kv_le_q = kv_idx.clone().le(q_idx.clone());
                        let kv_valid = program.bind(bound_valid.clone().and(kv_le_q));
                        (kv_valid, Some(chunk_in_range))
                    } else {
                        (bound_valid.clone(), None)
                    };
                    program.store_local(&score_local, f32_tile(NEG_MAX_F32));
                    let compute_guard = match &chunk_in_range {
                        Some(in_range) => kv_valid.clone().and(in_range.clone()),
                        None => kv_valid.clone(),
                    };
                    program.if_then(compute_guard, |program| {
                        let mut products = Vec::with_capacity(meta.dims.head_dim as usize);
                        for dim in 0..meta.dims.head_dim {
                            let q_index = index_n(
                                meta.q_meta.offset,
                                q_strides,
                                (batch_idx.clone(), head_idx.clone(), q_idx.clone(), dim),
                            );
                            let k_index = index_n(
                                meta.k_meta.offset,
                                k_strides,
                                (batch_idx.clone(), kv_head_idx.clone(), kv_idx.clone(), dim),
                            );
                            let q_value = program
                                .load(q.at(q_index), Mask::all(), elem_fill)
                                .cast(ElementType::F32);
                            let k_value = program
                                .load(k.at(k_index), Mask::all(), elem_fill)
                                .cast(ElementType::F32);
                            products.push(q_value * k_value);
                        }
                        let mut score =
                            program.sum(products, ElementType::F32) * f32_tile(meta.scale.get());
                        if let (Some(mask), Some(mask_meta), Some(mask_strides)) =
                            (&mask, meta.mask_meta.as_ref(), mask_strides)
                        {
                            let mask_index = index_n(
                                mask_meta.offset,
                                mask_strides,
                                (q_idx.clone(), kv_idx.clone()),
                            );
                            let mask_value = program
                                .load(mask.at(mask_index), Mask::all(), elem_fill)
                                .cast(ElementType::F32);
                            score = score + mask_value;
                        }
                        program.store_local(&score_local, score);
                    });

                    let score = program.bind(program.load_local(&score_local));
                    let block_max =
                        program.bind(subgroup.subgroup_reduce_max(program, score.clone()));
                    let old_m = program.bind(m_state);
                    let new_m = program.bind(old_m.clone().max(block_max.clone()));
                    let raw_exp = (score.clone() - new_m.clone()).exp();
                    let exp_score =
                        program.bind(Tile::select(kv_valid.clone(), raw_exp, f32_tile(0.0)));
                    let block_sum =
                        program.bind(subgroup.subgroup_reduce_sum(program, exp_score.clone()));

                    program.store_local(&weighted_local, f32_tile(0.0));
                    let valid_value = kv_valid.clone().and(out_valid.clone());
                    program.if_then(valid_value, |program| {
                        let v_index = index_n(
                            meta.v_meta.offset,
                            v_strides,
                            (
                                batch_idx.clone(),
                                kv_head_idx.clone(),
                                kv_idx.clone(),
                                out_dim.clone(),
                            ),
                        );
                        let v_value = program
                            .load(v.at(v_index), Mask::all(), elem_fill)
                            .cast(ElementType::F32);
                        program.store_local(&weighted_local, exp_score.clone() * v_value);
                    });
                    let weighted = program.load_local(&weighted_local);
                    let block_out = program.bind(subgroup.subgroup_reduce_sum(program, weighted));

                    let old_m_scale = program.bind((old_m.clone() - new_m.clone()).exp());
                    let new_s = s_state * old_m_scale.clone() + block_sum;
                    let new_o = o_state * old_m_scale + block_out;
                    [new_m, new_s, new_o]
                },
            );

            let store_valid = kv_lane.eq(u32_tile(0)).and(out_valid.clone());
            // Bind the fold results so the divide and the `if_then` body share
            // the same SSA values rather than re-emitting the loop materialize.
            let final_o_bound = program.bind(final_o);
            let final_s_bound = program.bind(final_s);
            // Evaluate `final_m` once so the loop fires even though we don't
            // need its value in the post-loop stage.
            program.if_then(store_valid, |program| {
                let output_value = (final_o_bound.clone() / final_s_bound.clone()).cast(element);
                let output_index = index_n(
                    meta.output_meta.offset,
                    output_strides,
                    (
                        batch_idx.clone(),
                        head_idx.clone(),
                        q_idx.clone(),
                        out_dim.clone(),
                    ),
                );
                program.store(output.at(output_index), output_value, Mask::all());
            });
        });
    }
    Some(())
}

/// Q-batched streaming flash attention. Same online-softmax algorithm as
/// [`flash_attention`], but each workgroup handles `q_block` contiguous q_idx
/// values, caching Q, K, and V slices in workgroup memory so the big
/// `(kv_seq_len * head_dim)` K traffic and the per-chunk Q reads are reused
/// across `q_block` queries instead of being re-fetched per query.
///
/// Layout:
/// - One workgroup per (batch, head, q_block).
/// - `FLASH_BLOCK` lanes split into subgroups of `subgroup_size` lanes each.
///   Each subgroup is pinned to one kv position per lane and accumulates a
///   short run of output dims serially, reusing the same QK score.
/// - Once per workgroup: load `Q[q_block, head_dim]` into workgroup memory.
/// - Per chunk: cooperatively load `K[chunk_kv_positions, head_dim]` and
///   `V[chunk_kv_positions, workgroup_out_dims]` into workgroup memory once,
///   then loop over `q_block` queries reusing those loads.
pub fn flash_attention_tiled<B>(
    kb: &mut fusor_tile_ir::KernelBuilder<B>,
    element: ElementType,
    tensors: FlashAttentionTensors<B>,
    meta: FlashAttentionMeta,
    subgroup: SubgroupToken,
    subgroup_size: u32,
    q_block: u32,
) -> Option<()> {
    let FlashAttentionTensors {
        q,
        k,
        v,
        mask,
        output,
    } = tensors;
    if !supports_float(element) {
        return None;
    }
    if subgroup_size == 0 || !FLASH_BLOCK.is_multiple_of(subgroup_size) || q_block == 0 {
        return None;
    }
    let outputs_per_workgroup = flash_tiled_outputs_per_workgroup(subgroup_size);
    let q_strides: [u32; 4] = meta.q_meta.strides.as_slice().try_into().ok()?;
    let k_strides: [u32; 4] = meta.k_meta.strides.as_slice().try_into().ok()?;
    let v_strides: [u32; 4] = meta.v_meta.strides.as_slice().try_into().ok()?;
    let output_strides: [u32; 4] = meta.output_meta.strides.as_slice().try_into().ok()?;
    let mask_strides: Option<[u32; 2]> = if let Some(mask) = meta.mask_meta.as_ref() {
        Some(mask.strides.as_slice().try_into().ok()?)
    } else {
        None
    };
    if meta.dims.batch == 0
        || meta.dims.num_heads == 0
        || meta.dims.num_kv_heads == 0
        || meta.dims.q_seq_len == 0
        || meta.dims.kv_seq_len == 0
        || meta.dims.head_dim == 0
    {
        return None;
    }
    if meta.mask_meta.is_some() != mask.is_some() {
        return None;
    }
    if meta.causal && meta.mask_meta.is_some() {
        return None;
    }
    let groups = meta.dims.num_heads.checked_div(meta.dims.num_kv_heads)?;
    if groups == 0 {
        return None;
    }
    // Workgroup-memory budgets — keep totals well under the 32KB Apple cap.
    // K cache: subgroup_size rows × head_dim cols.
    let k_cache_elems = subgroup_size.checked_mul(meta.dims.head_dim)?;
    if k_cache_elems > 4096 {
        return None;
    }
    // V cache: subgroup_size rows × outputs_per_workgroup cols.
    let v_cache_elems = subgroup_size.checked_mul(outputs_per_workgroup)?;
    if v_cache_elems > 4096 {
        return None;
    }
    // Q cache: q_block rows × head_dim cols.
    let q_cache_elems = q_block.checked_mul(meta.dims.head_dim)?;
    if q_cache_elems > 4096 {
        return None;
    }
    let elem_fill = zero_fill(element);
    let q = kb.read(element, q);
    let k = kb.read(element, k);
    let v = kb.read(element, v);
    let mask = mask.map(|m| kb.read(element, m));
    let output = kb.write(element, output);
    let phase = kb.program();
    {
        // Allocate workgroup caches once.
        let k_cache = phase.alloc_workgroup_array(ScalarElement::F32, k_cache_elems);
        let v_cache = phase.alloc_workgroup_array(ScalarElement::F32, v_cache_elems);
        let q_cache = phase.alloc_workgroup_array(ScalarElement::F32, q_cache_elems);
        let q_blocks = meta.dims.q_seq_len.div_ceil(q_block);
        let dispatch_size = [
            meta.dims.head_dim.div_ceil(outputs_per_workgroup),
            meta.dims
                .batch
                .checked_mul(meta.dims.num_heads)?
                .checked_mul(q_blocks)?,
            1,
        ];

        phase.program_grid(FLASH_BLOCK, dispatch_size, |program| {
            let lane = program.lane();
            let workgroup_x = program.program_id(WorkgroupAxis::X);
            let row = program.program_id(WorkgroupAxis::Y);
            let q_block_idx = program.bind(row.clone() % q_blocks);
            let row_over_q = row.clone() / q_blocks;
            let head_idx = program.bind(row_over_q.clone() % meta.dims.num_heads);
            let batch_idx = program.bind(row_over_q / meta.dims.num_heads);
            let kv_head_idx = program.bind(head_idx.clone() / u32_tile(groups));
            let subgroup_idx = program.bind(lane.clone() / subgroup_size);
            let kv_lane = program.bind(lane.clone() % subgroup_size);
            let out_dim_base = program.bind(
                workgroup_x * outputs_per_workgroup
                    + subgroup_idx.clone() * TILED_OUTS_PER_SUBGROUP,
            );
            let mut out_dims = Vec::with_capacity(TILED_OUTS_PER_SUBGROUP as usize);
            let mut out_valids = Vec::with_capacity(TILED_OUTS_PER_SUBGROUP as usize);
            for out_offset in 0..TILED_OUTS_PER_SUBGROUP {
                let out_dim = program.bind(out_dim_base.clone() + u32_tile(out_offset));
                let out_valid = program.bind(out_dim.clone().lt(u32_tile(meta.dims.head_dim)));
                out_dims.push(out_dim);
                out_valids.push(out_valid);
            }

            // Per-query accumulator state, stored in private locals indexed by
            // q-offset within the workgroup's Q-block. Initialised below.
            let mut m_locals: Vec<PrivateLocal> = Vec::with_capacity(q_block as usize);
            let mut s_locals: Vec<PrivateLocal> = Vec::with_capacity(q_block as usize);
            let mut o_locals: Vec<PrivateLocal> =
                Vec::with_capacity((q_block * TILED_OUTS_PER_SUBGROUP) as usize);
            for _ in 0..q_block {
                m_locals.push(program.private(ElementType::F32));
                s_locals.push(program.private(ElementType::F32));
                for _ in 0..TILED_OUTS_PER_SUBGROUP {
                    o_locals.push(program.private(ElementType::F32));
                }
            }
            for q_offset in 0..q_block {
                program.store_local(&m_locals[q_offset as usize], f32_tile(NEG_MAX_F32));
                program.store_local(&s_locals[q_offset as usize], f32_tile(0.0));
                for out_offset in 0..TILED_OUTS_PER_SUBGROUP {
                    let o_idx = (q_offset * TILED_OUTS_PER_SUBGROUP + out_offset) as usize;
                    program.store_local(&o_locals[o_idx], f32_tile(0.0));
                }
            }

            let kv_chunks = meta.dims.kv_seq_len.div_ceil(subgroup_size);
            let causal = meta.causal;

            // Q-block base index in the q dimension.
            let q_block_base = program.bind(q_block_idx.clone() * u32_tile(q_block));

            // -----------------------------------------------------------
            // One-shot Q cache load.
            //
            // Layout: q_cache[q_offset * head_dim + dim].
            // Loaded once per workgroup, indexed by lane in strided passes
            // of FLASH_BLOCK lanes each. Total = q_block * head_dim ≤ 4096.
            //
            // The original streaming kernel re-loaded Q from global per
            // (kv_chunk, q_idx, dim, lane) — 256× duplicate reads per
            // (q_idx, dim) within a workgroup. Caching Q removes that.
            // -----------------------------------------------------------
            let total_q_loads = q_block * meta.dims.head_dim;
            let q_passes = total_q_loads.div_ceil(FLASH_BLOCK);
            for pass in 0..q_passes {
                let pass_base = pass * FLASH_BLOCK;
                let idx = program.bind(lane.clone() + u32_tile(pass_base));
                let q_offset_local = program.bind(idx.clone() / meta.dims.head_dim);
                let dim_local = program.bind(idx.clone() % meta.dims.head_dim);
                let q_pos = program.bind(q_block_base.clone() + q_offset_local.clone());
                let in_bounds = idx
                    .clone()
                    .lt(u32_tile(total_q_loads))
                    .and(q_pos.clone().lt(u32_tile(meta.dims.q_seq_len)));
                let q_index = index_n(
                    meta.q_meta.offset,
                    q_strides,
                    (
                        batch_idx.clone(),
                        head_idx.clone(),
                        q_pos.clone(),
                        dim_local.clone(),
                    ),
                );
                let q_val = program
                    .load(q.at(q_index), in_bounds.clone(), elem_fill)
                    .cast(ElementType::F32);
                let store_in_bounds = idx.clone().lt(u32_tile(total_q_loads));
                program.if_then(store_in_bounds, |program| {
                    program.store_workgroup(&q_cache, idx.clone(), q_val);
                });
            }
            program.workgroup_barrier();

            // Per-iteration scratch local used to bridge values across if_then.
            let score_local = program.private(ElementType::F32);

            // Runtime chunk counter; collapses the kv-chunk dimension into a
            // single IR loop rather than unrolling `kv_chunks` copies of the
            // body (which blew up code size for big prefills).
            let chunk_local = program.private(ElementType::U32);
            program.store_local(&chunk_local, u32_tile(0));
            program.loop_forever(|program| {
                let chunk_tile = program.bind(program.load_local(&chunk_local));
                program.break_if(chunk_tile.clone().ge(u32_tile(kv_chunks)));
                let chunk_start = program.bind(chunk_tile.clone() * u32_tile(subgroup_size));
                if causal {
                    let q_block_last = program.bind(
                        (q_block_base.clone() + u32_tile(q_block.saturating_sub(1)))
                            .min(u32_tile(meta.dims.q_seq_len - 1)),
                    );
                    program.break_if(chunk_start.clone().gt(q_block_last));
                }
                let kv_idx = program.bind(chunk_start.clone() + kv_lane.clone());
                let bound_valid = program.bind(kv_idx.clone().lt(u32_tile(meta.dims.kv_seq_len)));

                // -------------------------------------------------------
                // Cooperative load of K[chunk kv rows, all head_dim] into
                // workgroup memory. Indexed by `kv_local * head_dim + dim`.
                // Strided across FLASH_BLOCK lanes. The k_cache_elems check
                // above guarantees total ≤ 4096, and we use mask-load + an
                // if_then-guarded store to handle the last partial pass
                // (only fires when total_k_loads % FLASH_BLOCK != 0).
                // -------------------------------------------------------
                let total_k_loads = subgroup_size * meta.dims.head_dim;
                let k_passes = total_k_loads.div_ceil(FLASH_BLOCK);
                let k_aligned = total_k_loads.is_multiple_of(FLASH_BLOCK);
                for pass in 0..k_passes {
                    let pass_base = pass * FLASH_BLOCK;
                    let idx = program.bind(lane.clone() + u32_tile(pass_base));
                    let kv_local = program.bind(idx.clone() / meta.dims.head_dim);
                    let dim_local = program.bind(idx.clone() % meta.dims.head_dim);
                    let kv_pos = program.bind(chunk_start.clone() + kv_local.clone());
                    let in_bounds = if k_aligned {
                        kv_pos.clone().lt(u32_tile(meta.dims.kv_seq_len))
                    } else {
                        idx.clone()
                            .lt(u32_tile(total_k_loads))
                            .and(kv_pos.clone().lt(u32_tile(meta.dims.kv_seq_len)))
                    };
                    let k_index = index_n(
                        meta.k_meta.offset,
                        k_strides,
                        (
                            batch_idx.clone(),
                            kv_head_idx.clone(),
                            kv_pos.clone(),
                            dim_local.clone(),
                        ),
                    );
                    let k_val = program
                        .load(k.at(k_index), in_bounds.clone(), elem_fill)
                        .cast(ElementType::F32);
                    if k_aligned {
                        program.store_workgroup(&k_cache, idx.clone(), k_val);
                    } else {
                        let store_in_bounds = idx.clone().lt(u32_tile(total_k_loads));
                        program.if_then(store_in_bounds, |program| {
                            program.store_workgroup(&k_cache, idx.clone(), k_val.clone());
                        });
                    }
                }

                // Cooperative load of V[chunk kv rows, out_dims handled by
                // this workgroup] into workgroup memory. Each subgroup loads
                // a short run of output dims for the same kv lane.
                for out_offset in 0..TILED_OUTS_PER_SUBGROUP {
                    let out_idx = out_offset as usize;
                    let v_cell_idx = program.bind(
                        kv_lane.clone() * outputs_per_workgroup
                            + subgroup_idx.clone() * TILED_OUTS_PER_SUBGROUP
                            + u32_tile(out_offset),
                    );
                    let v_in_bounds = bound_valid.clone().and(out_valids[out_idx].clone());
                    let v_index = index_n(
                        meta.v_meta.offset,
                        v_strides,
                        (
                            batch_idx.clone(),
                            kv_head_idx.clone(),
                            kv_idx.clone(),
                            out_dims[out_idx].clone(),
                        ),
                    );
                    let v_val = program
                        .load(v.at(v_index), v_in_bounds.clone(), elem_fill)
                        .cast(ElementType::F32);
                    program.store_workgroup(&v_cache, v_cell_idx.clone(), v_val);
                }

                program.workgroup_barrier();

                // -------------------------------------------------------
                // For each query in the Q block, compute score, update the
                // online-softmax stats, accumulate weighted V. Q and K both
                // come from workgroup memory; only the optional mask still
                // comes from global.
                // -------------------------------------------------------
                for q_offset in 0..q_block {
                    let q_idx = program.bind(q_block_base.clone() + u32_tile(q_offset));
                    let q_in_range = program.bind(q_idx.clone().lt(u32_tile(meta.dims.q_seq_len)));

                    let (kv_valid, chunk_in_range) = if causal {
                        let chunk_in_range = program.bind(chunk_start.clone().le(q_idx.clone()));
                        let kv_le_q = kv_idx.clone().le(q_idx.clone());
                        let kv_valid =
                            program.bind(bound_valid.clone().and(kv_le_q).and(q_in_range.clone()));
                        (kv_valid, Some(chunk_in_range))
                    } else {
                        let kv_valid = program.bind(bound_valid.clone().and(q_in_range.clone()));
                        (kv_valid, None)
                    };

                    program.store_local(&score_local, f32_tile(NEG_MAX_F32));
                    let compute_guard = match &chunk_in_range {
                        Some(in_range) => kv_valid.clone().and(in_range.clone()),
                        None => kv_valid.clone(),
                    };
                    program.if_then(compute_guard, |program| {
                        let mut products = Vec::with_capacity(meta.dims.head_dim as usize);
                        for dim in 0..meta.dims.head_dim {
                            // Q from workgroup cache: q_cache[q_offset*head_dim + dim].
                            let q_cache_idx = u32_tile(q_offset * meta.dims.head_dim + dim);
                            let q_value = program.load_workgroup(&q_cache, q_cache_idx);
                            // K from workgroup cache: k_cache[kv_lane*head_dim + dim].
                            let k_cache_idx = kv_lane.clone() * meta.dims.head_dim + u32_tile(dim);
                            let k_value = program.load_workgroup(&k_cache, k_cache_idx);
                            products.push(q_value * k_value);
                        }
                        let mut score =
                            program.sum(products, ElementType::F32) * f32_tile(meta.scale.get());
                        if let (Some(mask), Some(mask_meta), Some(mask_strides)) =
                            (&mask, meta.mask_meta.as_ref(), mask_strides)
                        {
                            let mask_index = index_n(
                                mask_meta.offset,
                                mask_strides,
                                (q_idx.clone(), kv_idx.clone()),
                            );
                            let mask_value = program
                                .load(mask.at(mask_index), Mask::all(), elem_fill)
                                .cast(ElementType::F32);
                            score = score + mask_value;
                        }
                        program.store_local(&score_local, score);
                    });

                    let score = program.bind(program.load_local(&score_local));
                    let block_max =
                        program.bind(subgroup.subgroup_reduce_max(program, score.clone()));
                    let old_m = program.bind(program.load_local(&m_locals[q_offset as usize]));
                    let new_m = program.bind(old_m.clone().max(block_max.clone()));
                    let raw_exp = (score.clone() - new_m.clone()).exp();
                    let exp_score =
                        program.bind(Tile::select(kv_valid.clone(), raw_exp, f32_tile(0.0)));
                    let block_sum =
                        program.bind(subgroup.subgroup_reduce_sum(program, exp_score.clone()));

                    let old_m_scale = program.bind((old_m.clone() - new_m.clone()).exp());
                    let old_s = program.load_local(&s_locals[q_offset as usize]);
                    let new_s = old_s * old_m_scale.clone() + block_sum;
                    for out_offset in 0..TILED_OUTS_PER_SUBGROUP {
                        let out_idx = out_offset as usize;
                        let v_cache_idx = kv_lane.clone() * outputs_per_workgroup
                            + subgroup_idx.clone() * TILED_OUTS_PER_SUBGROUP
                            + u32_tile(out_offset);
                        let v_cached = program.bind(program.load_workgroup(&v_cache, v_cache_idx));
                        let weighted = Tile::select(
                            kv_valid.clone().and(out_valids[out_idx].clone()),
                            exp_score.clone() * v_cached,
                            f32_tile(0.0),
                        );
                        let block_out =
                            program.bind(subgroup.subgroup_reduce_sum(program, weighted));
                        let o_idx = (q_offset * TILED_OUTS_PER_SUBGROUP + out_offset) as usize;
                        let old_o = program.load_local(&o_locals[o_idx]);
                        let new_o = old_o * old_m_scale.clone() + block_out;
                        program.store_local(&o_locals[o_idx], new_o);
                    }
                    program.store_local(&m_locals[q_offset as usize], new_m);
                    program.store_local(&s_locals[q_offset as usize], new_s);
                }

                // Barrier before next chunk's K/V load overwrites the cache.
                program.workgroup_barrier();

                program.store_local(&chunk_local, chunk_tile + u32_tile(1));
            });

            // Write each query's accumulated output.
            for q_offset in 0..q_block {
                let q_idx = program.bind(q_block_base.clone() + u32_tile(q_offset));
                let q_in_range = program.bind(q_idx.clone().lt(u32_tile(meta.dims.q_seq_len)));
                let final_s = program.bind(program.load_local(&s_locals[q_offset as usize]));
                for out_offset in 0..TILED_OUTS_PER_SUBGROUP {
                    let out_idx = out_offset as usize;
                    let store_valid = kv_lane
                        .clone()
                        .eq(u32_tile(0))
                        .and(out_valids[out_idx].clone())
                        .and(q_in_range.clone());
                    let o_idx = (q_offset * TILED_OUTS_PER_SUBGROUP + out_offset) as usize;
                    let final_o = program.bind(program.load_local(&o_locals[o_idx]));
                    program.if_then(store_valid, |program| {
                        let output_value = (final_o.clone() / final_s.clone()).cast(element);
                        let output_index = index_n(
                            meta.output_meta.offset,
                            output_strides,
                            (
                                batch_idx.clone(),
                                head_idx.clone(),
                                q_idx.clone(),
                                out_dims[out_idx].clone(),
                            ),
                        );
                        program.store(output.at(output_index), output_value, Mask::all());
                    });
                }
            }
        });
    }
    Some(())
}

/// Dispatch grid for the tiled (Q-batched) flash-attention kernel.
pub fn flash_tiled_dispatch_size(
    dims: FlashAttentionDims,
    outputs_per_workgroup: u32,
    q_block: u32,
) -> [u32; 3] {
    [
        dims.head_dim.div_ceil(outputs_per_workgroup),
        dims.batch
            .checked_mul(dims.num_heads)
            .and_then(|value| value.checked_mul(dims.q_seq_len.div_ceil(q_block)))
            .expect("flash attention tiled dispatch overflow"),
        1,
    ]
}

struct DecodeScoreForKv<'a> {
    q: &'a Storage,
    k: &'a Storage,
    meta: FlashDecodeSmallMeta,
    batch_idx: Tile,
    head_idx: Tile,
    kv_head_idx: Tile,
    kv: Tile,
    score_acc: &'a PrivateLocal,
    dim_local: &'a PrivateLocal,
}

fn decode_score_for_kv(program: &mut TileBlock<'_>, request: DecodeScoreForKv<'_>) -> Tile {
    let DecodeScoreForKv {
        q,
        k,
        meta,
        batch_idx,
        head_idx,
        kv_head_idx,
        kv,
        score_acc,
        dim_local,
    } = request;
    program.store_local(score_acc, f32_tile(0.0));
    program.store_local(dim_local, u32_tile(0));
    program.loop_forever(|program| {
        let dim = program.load_local(dim_local);
        program.break_if(dim.clone().ge(u32_tile(meta.dims.head_dim)));
        let q_index = index_n(
            meta.q_offset,
            meta.q_strides,
            (batch_idx.clone(), head_idx.clone(), 0, dim.clone()),
        );
        let k_index = index_n(
            meta.k_offset,
            meta.k_strides,
            (
                batch_idx.clone(),
                kv_head_idx.clone(),
                kv.clone(),
                dim.clone(),
            ),
        );
        let q_value = program.load(q.at(q_index), Mask::all(), TileLiteral::f32(0.0));
        let k_value = program.load(k.at(k_index), Mask::all(), TileLiteral::f32(0.0));
        let acc = program.load_local(score_acc);
        program.store_local(score_acc, acc + q_value * k_value);
        program.store_local(dim_local, dim + u32_tile(1));
    });
    program.load_local(score_acc) * f32_tile(meta.scale.get())
}

struct DecodeOutputLoop<'a> {
    v: &'a Storage,
    output: &'a Storage,
    probs: &'a WorkgroupTile,
    meta: FlashDecodeSmallMeta,
    batch_idx: Tile,
    head_idx: Tile,
    kv_head_idx: Tile,
    out_dim: Tile,
    active_kv_len: Tile,
    acc: &'a PrivateLocal,
    kv_local: &'a PrivateLocal,
}

struct FlashDecodeSmallBlock<B> {
    q: fusor_tile_ir::KernelTensorRef<B>,
    k: fusor_tile_ir::KernelTensorRef<B>,
    v: fusor_tile_ir::KernelTensorRef<B>,
    output: fusor_tile_ir::KernelTensorRef<B>,
    params: fusor_tile_ir::KernelTensorRef<B>,
    meta: FlashDecodeSmallMeta,
}

struct FlashDecodeSplitPartialsBlock<B> {
    q: fusor_tile_ir::KernelTensorRef<B>,
    k: fusor_tile_ir::KernelTensorRef<B>,
    v: fusor_tile_ir::KernelTensorRef<B>,
    scratch: fusor_tile_ir::KernelTensorRef<B>,
    params: fusor_tile_ir::KernelTensorRef<B>,
    meta: FlashDecodeSmallMeta,
}

fn append_decode_output_loop(program: &mut TileBlock<'_>, request: DecodeOutputLoop<'_>) {
    let DecodeOutputLoop {
        v,
        output,
        probs,
        meta,
        batch_idx,
        head_idx,
        kv_head_idx,
        out_dim,
        active_kv_len,
        acc,
        kv_local,
    } = request;
    program.loop_forever(|program| {
        let kv = program.load_local(kv_local);
        program.break_if(kv.clone().ge(active_kv_len.clone()));
        let prob = program.load_workgroup(probs, kv.clone());
        let v_index = index_n(
            meta.v_offset,
            meta.v_strides,
            (
                batch_idx.clone(),
                kv_head_idx.clone(),
                kv.clone(),
                out_dim.clone(),
            ),
        );
        let v_value = program.load(v.at(v_index), Mask::all(), TileLiteral::f32(0.0));
        let current = program.load_local(acc);
        program.store_local(acc, current + prob * v_value);
        program.store_local(kv_local, kv + u32_tile(1));
    });

    let output_value = program.load_local(acc);
    let output_index = index_n(
        meta.output_offset,
        meta.output_strides,
        (batch_idx, head_idx, 0, out_dim),
    );
    program.store(output.at(output_index), output_value, Mask::all());
}

fn flash_decode_small_block<B>(
    kb: &mut fusor_tile_ir::KernelBuilder<B>,
    block: u32,
    args: FlashDecodeSmallBlock<B>,
) {
    let FlashDecodeSmallBlock {
        q,
        k,
        v,
        output,
        params,
        meta,
    } = args;
    let q = kb.read(ElementType::F32, q);
    let k = kb.read(ElementType::F32, k);
    let v = kb.read(ElementType::F32, v);
    let output = kb.write(ElementType::F32, output);
    let params = kb.read(ElementType::U32, params);
    let phase = kb.program();
    let probs = phase.alloc_workgroup_array(ScalarElement::F32, block);
    let reduce = phase.alloc_workgroup_array(ScalarElement::F32, block);

    phase.program_grid(
        block,
        [meta.dims.batch * meta.dims.num_heads, 1, 1],
        |program| {
            let lane = program.lane();
            let row = program.program_id(WorkgroupAxis::X);
            let active_kv_len = program.load(
                params.at(0),
                Mask::all(),
                TileLiteral::U32(meta.active_kv_len),
            );
            let head_idx = row.clone() % meta.dims.num_heads;
            let batch_idx = row / meta.dims.num_heads;
            let kv_head_idx = head_idx.clone() / u32_tile(meta.groups);
            let lane_value = lane.clone();
            let acc = program.private(ElementType::F32);
            let kv_local = program.private(ElementType::U32);
            let item = program.private(ElementType::U32);
            let dim = program.private(ElementType::U32);
            let score_acc = program.private(ElementType::F32);
            let score_local = program.private(ElementType::F32);
            let max_score_local = program.private(ElementType::F32);

            if meta.tiled {
                program.store_workgroup(&reduce, lane.clone(), f32_tile(NEG_MAX_F32));
                program.store_local(&kv_local, u32_tile(0));
                program.loop_forever(|program| {
                    let tile_base = program.load_local(&kv_local);
                    program.break_if(tile_base.clone().ge(active_kv_len.clone()));
                    let kv = tile_base.clone() + lane_value.clone();
                    let kv_valid = kv.clone().lt(active_kv_len.clone());
                    program.if_then(kv_valid, |program| {
                        let score = decode_score_for_kv(
                            program,
                            DecodeScoreForKv {
                                q: &q,
                                k: &k,
                                meta,
                                batch_idx: batch_idx.clone(),
                                head_idx: head_idx.clone(),
                                kv_head_idx: kv_head_idx.clone(),
                                kv: kv.clone(),
                                score_acc: &score_acc,
                                dim_local: &dim,
                            },
                        );
                        let current = program.load_workgroup(&reduce, lane.clone());
                        program.store_workgroup(&reduce, lane.clone(), current.max(score));
                    });
                    program.store_local(&kv_local, tile_base + u32_tile(block));
                });
                program.workgroup_barrier();
                reduce_workgroup(program, &reduce, lane.clone(), |lhs, rhs| lhs.max(rhs));
                let max_score = program.load_workgroup(&reduce, 0u32);
                program.store_local(&max_score_local, max_score);
                let max_score = program.load_local(&max_score_local);

                // All lanes load `reduce[0]` (the max) above. Before any lane
                // overwrites `reduce[lane]` for the denominator accumulator we
                // need a barrier — without it, lane 0's store to slot 0 races
                // with other lanes still loading slot 0, which surfaces as
                // intermittent wrong values for individual heads.
                program.workgroup_barrier();
                program.store_workgroup(&reduce, lane.clone(), f32_tile(0.0));
                program.store_local(&kv_local, u32_tile(0));
                program.loop_forever(|program| {
                    let tile_base = program.load_local(&kv_local);
                    program.break_if(tile_base.clone().ge(active_kv_len.clone()));
                    let kv = tile_base.clone() + lane_value.clone();
                    let kv_valid = kv.clone().lt(active_kv_len.clone());
                    program.if_then(kv_valid, |program| {
                        let score = decode_score_for_kv(
                            program,
                            DecodeScoreForKv {
                                q: &q,
                                k: &k,
                                meta,
                                batch_idx: batch_idx.clone(),
                                head_idx: head_idx.clone(),
                                kv_head_idx: kv_head_idx.clone(),
                                kv: kv.clone(),
                                score_acc: &score_acc,
                                dim_local: &dim,
                            },
                        );
                        let prob = (score - max_score.clone()).exp();
                        let current = program.load_workgroup(&reduce, lane.clone());
                        program.store_workgroup(&reduce, lane.clone(), current + prob);
                    });
                    program.store_local(&kv_local, tile_base + u32_tile(block));
                });
                program.workgroup_barrier();
                reduce_workgroup(program, &reduce, lane.clone(), |lhs, rhs| lhs + rhs);
                let denom = program.load_workgroup(&reduce, 0u32);

                program.store_local(&acc, f32_tile(0.0));
                program.store_local(&kv_local, u32_tile(0));
                program.loop_forever(|program| {
                    let tile_base = program.load_local(&kv_local);
                    program.break_if(tile_base.clone().ge(active_kv_len.clone()));
                    let kv = tile_base.clone() + lane_value.clone();
                    let kv_valid = kv.clone().lt(active_kv_len.clone());
                    program.if_else(
                        kv_valid,
                        |program| {
                            let score = decode_score_for_kv(
                                program,
                                DecodeScoreForKv {
                                    q: &q,
                                    k: &k,
                                    meta,
                                    batch_idx: batch_idx.clone(),
                                    head_idx: head_idx.clone(),
                                    kv_head_idx: kv_head_idx.clone(),
                                    kv: kv.clone(),
                                    score_acc: &score_acc,
                                    dim_local: &dim,
                                },
                            );
                            let prob = (score - max_score.clone()).exp() / denom.clone();
                            program.store_workgroup(&probs, lane.clone(), prob);
                        },
                        |program| {
                            program.store_workgroup(&probs, lane.clone(), f32_tile(0.0));
                        },
                    );
                    program.workgroup_barrier();
                    program.store_local(&item, u32_tile(0));
                    let out_condition = lane_value.clone().lt(u32_tile(meta.dims.head_dim));
                    program.if_then(out_condition, |program| {
                        program.loop_forever(|program| {
                            let item_value = program.load_local(&item);
                            let block_done = item_value.clone().ge(u32_tile(block));
                            let kv = tile_base.clone() + item_value.clone();
                            let kv_done = kv.clone().ge(active_kv_len.clone());
                            program.break_if(block_done.or(kv_done));
                            let prob = program.load_workgroup(&probs, item_value.clone());
                            let v_index = index_n(
                                meta.v_offset,
                                meta.v_strides,
                                (
                                    batch_idx.clone(),
                                    kv_head_idx.clone(),
                                    kv,
                                    lane_value.clone(),
                                ),
                            );
                            let v_value =
                                program.load(v.at(v_index), Mask::all(), TileLiteral::f32(0.0));
                            let current = program.load_local(&acc);
                            program.store_local(&acc, current + prob * v_value);
                            program.store_local(&item, item_value + u32_tile(1));
                        });
                    });
                    program.workgroup_barrier();
                    program.store_local(&kv_local, tile_base + u32_tile(block));
                });
                let out_condition = lane_value.clone().lt(u32_tile(meta.dims.head_dim));
                program.if_then(out_condition, |program| {
                    let output_value = program.load_local(&acc);
                    let output_index = index_n(
                        meta.output_offset,
                        meta.output_strides,
                        (batch_idx.clone(), head_idx.clone(), 0, lane_value.clone()),
                    );
                    program.store(output.at(output_index), output_value, Mask::all());
                });
                return;
            }

            let kv_valid = lane_value.clone().lt(active_kv_len.clone());
            program.store_local(&score_local, f32_tile(NEG_MAX_F32));
            program.if_then(kv_valid.clone(), |program| {
                let score = decode_score_for_kv(
                    program,
                    DecodeScoreForKv {
                        q: &q,
                        k: &k,
                        meta,
                        batch_idx: batch_idx.clone(),
                        head_idx: head_idx.clone(),
                        kv_head_idx: kv_head_idx.clone(),
                        kv: lane_value.clone(),
                        score_acc: &score_acc,
                        dim_local: &dim,
                    },
                );
                program.store_local(&score_local, score);
            });
            let stats = workgroup_softmax_block(
                program,
                lane.clone(),
                program.load_local(&score_local),
                kv_valid.clone(),
                &reduce,
                Some(&probs),
            );
            program.if_then(kv_valid, |program| {
                program.store_workgroup(
                    &probs,
                    lane.clone(),
                    stats.prob.clone() / stats.denom.clone(),
                );
            });
            program.workgroup_barrier();

            let out_condition = lane_value.clone().lt(u32_tile(meta.dims.head_dim));
            program.if_then(out_condition, |program| {
                program.store_local(&acc, f32_tile(0.0));
                program.store_local(&kv_local, u32_tile(0));
                append_decode_output_loop(
                    program,
                    DecodeOutputLoop {
                        v: &v,
                        output: &output,
                        probs: &probs,
                        meta,
                        batch_idx: batch_idx.clone(),
                        head_idx: head_idx.clone(),
                        kv_head_idx: kv_head_idx.clone(),
                        out_dim: lane_value.clone(),
                        active_kv_len: active_kv_len.clone(),
                        acc: &acc,
                        kv_local: &kv_local,
                    },
                );
            });
        },
    );
}

fn flash_decode_split_partials_block<B>(
    kb: &mut fusor_tile_ir::KernelBuilder<B>,
    block: u32,
    args: FlashDecodeSplitPartialsBlock<B>,
) {
    let FlashDecodeSplitPartialsBlock {
        q,
        k,
        v,
        scratch,
        params,
        meta,
    } = args;
    let q = kb.read(ElementType::F32, q);
    let k = kb.read(ElementType::F32, k);
    let v = kb.read(ElementType::F32, v);
    let scratch = kb.write(ElementType::F32, scratch);
    let params = kb.read(ElementType::U32, params);
    let phase = kb.program();
    let probs = phase.alloc_workgroup_array(ScalarElement::F32, block);
    let reduce = phase.alloc_workgroup_array(ScalarElement::F32, block);

    phase.program_grid(
        block,
        [
            meta.dims.batch * meta.dims.num_heads * meta.split_blocks,
            1,
            1,
        ],
        |program| {
            let lane = program.lane();
            let tile = program.program_id(WorkgroupAxis::X);
            let rows = u32_tile(meta.dims.batch * meta.dims.num_heads);
            let row = program.bind(tile.clone() % rows.clone());
            let split_block = program.bind(tile / rows);
            let active_kv_len = program.load(
                params.at(0),
                Mask::all(),
                TileLiteral::U32(meta.active_kv_len),
            );
            let head_idx = row.clone() % meta.dims.num_heads;
            let batch_idx = row.clone() / meta.dims.num_heads;
            let kv_head_idx = head_idx.clone() / u32_tile(meta.groups);
            let lane_value = lane.clone();
            let tile_base = program.bind(split_block.clone() * u32_tile(meta.decode_block));
            let tile_end = program
                .bind((tile_base.clone() + u32_tile(meta.decode_block)).min(active_kv_len.clone()));
            let kv = program.bind(tile_base.clone() + lane_value.clone());
            let kv_valid = kv.clone().lt(tile_end.clone());
            let item = program.private(ElementType::U32);
            let acc = program.private(ElementType::F32);
            let dim = program.private(ElementType::U32);
            let score_acc = program.private(ElementType::F32);
            let score_local = program.private(ElementType::F32);

            program.store_local(&score_local, f32_tile(NEG_MAX_F32));
            program.if_then(kv_valid.clone(), |program| {
                let score = decode_score_for_kv(
                    program,
                    DecodeScoreForKv {
                        q: &q,
                        k: &k,
                        meta,
                        batch_idx: batch_idx.clone(),
                        head_idx: head_idx.clone(),
                        kv_head_idx: kv_head_idx.clone(),
                        kv: kv.clone(),
                        score_acc: &score_acc,
                        dim_local: &dim,
                    },
                );
                program.store_local(&score_local, score);
            });
            let stats = workgroup_softmax_block(
                program,
                lane.clone(),
                program.load_local(&score_local),
                kv_valid.clone(),
                &reduce,
                Some(&probs),
            );

            let scratch_base = program.bind(
                (row.clone() * u32_tile(meta.split_blocks) + split_block.clone())
                    * u32_tile(meta.dims.head_dim + 2),
            );
            program.if_then(lane_value.clone().eq(u32_tile(0)), |program| {
                program.store(
                    scratch.at(scratch_base.clone() + meta.dims.head_dim),
                    stats.denom.clone(),
                    Mask::all(),
                );
                program.store(
                    scratch.at(scratch_base.clone() + meta.dims.head_dim + 1),
                    stats.max.clone(),
                    Mask::all(),
                );
            });

            let out_condition = lane_value.clone().lt(u32_tile(meta.dims.head_dim));
            program.if_then(out_condition, |program| {
                program.store_local(&acc, f32_tile(0.0));
                program.store_local(&item, u32_tile(0));
                program.loop_forever(|program| {
                    let item_value = program.load_local(&item);
                    let block_done = item_value.clone().ge(u32_tile(block));
                    let kv = tile_base.clone() + item_value.clone();
                    let kv_done = kv.clone().ge(tile_end.clone());
                    program.break_if(block_done.or(kv_done));
                    let prob = program.load_workgroup(&probs, item_value.clone());
                    let v_index = index_n(
                        meta.v_offset,
                        meta.v_strides,
                        (
                            batch_idx.clone(),
                            kv_head_idx.clone(),
                            kv,
                            lane_value.clone(),
                        ),
                    );
                    let v_value = program.load(v.at(v_index), Mask::all(), TileLiteral::f32(0.0));
                    let current = program.load_local(&acc);
                    program.store_local(&acc, current + prob * v_value);
                    program.store_local(&item, item_value + u32_tile(1));
                });
                program.store(
                    scratch.at(scratch_base.clone() + lane_value.clone()),
                    program.load_local(&acc),
                    Mask::all(),
                );
            });
        },
    );
}

pub fn flash_decode_split_partials<B>(
    kb: &mut fusor_tile_ir::KernelBuilder<B>,
    q: fusor_tile_ir::KernelTensorRef<B>,
    k: fusor_tile_ir::KernelTensorRef<B>,
    v: fusor_tile_ir::KernelTensorRef<B>,
    scratch: fusor_tile_ir::KernelTensorRef<B>,
    params: fusor_tile_ir::KernelTensorRef<B>,
    meta: FlashDecodeSmallMeta,
) -> Option<()> {
    if meta.dims.head_dim == 0
        || meta.dims.head_dim > meta.decode_block
        || meta.decode_block == 0
        || meta.groups == 0
        || meta.split_blocks < 2
    {
        return None;
    }
    if !matches!(meta.decode_block, 128 | 256 | 512 | 1024) {
        return None;
    }
    flash_decode_split_partials_block(
        kb,
        meta.decode_block,
        FlashDecodeSplitPartialsBlock {
            q,
            k,
            v,
            scratch,
            params,
            meta,
        },
    );
    Some(())
}

pub fn flash_decode_split_reduce<B>(
    kb: &mut fusor_tile_ir::KernelBuilder<B>,
    scratch: fusor_tile_ir::KernelTensorRef<B>,
    output: fusor_tile_ir::KernelTensorRef<B>,
    meta: FlashDecodeSmallMeta,
) -> Option<()> {
    if meta.dims.head_dim == 0 || meta.groups == 0 || meta.split_blocks < 2 {
        return None;
    }
    let output_strides: [u32; 4] = meta.output_strides;
    let scratch = kb.read(ElementType::F32, scratch);
    let output = kb.write(ElementType::F32, output);
    let phase = kb.program();
    phase.program_grid(
        meta.dims.head_dim,
        [meta.dims.batch * meta.dims.num_heads, 1, 1],
        |program| {
            let lane = program.lane();
            let row = program.program_id(WorkgroupAxis::X);
            let out_dim = lane.clone();
            let head_idx = row.clone() % meta.dims.num_heads;
            let batch_idx = row.clone() / meta.dims.num_heads;
            let scratch_stride = meta.dims.head_dim + 2;
            let row_base = program.bind(row.clone() * u32_tile(meta.split_blocks * scratch_stride));
            let mut max_score = f32_tile(NEG_MAX_F32);
            for split_block in 0..meta.split_blocks {
                let block_base = row_base.clone() + split_block * scratch_stride;
                let block_max = program.load(
                    scratch.at(block_base + meta.dims.head_dim + 1),
                    Mask::all(),
                    TileLiteral::f32(NEG_MAX_F32),
                );
                max_score = max_score.max(block_max);
            }
            let max_score = program.bind(max_score);

            let mut denom = f32_tile(0.0);
            let mut acc = f32_tile(0.0);
            for split_block in 0..meta.split_blocks {
                let block_base = row_base.clone() + split_block * scratch_stride;
                let block_denom = program.load(
                    scratch.at(block_base.clone() + meta.dims.head_dim),
                    Mask::all(),
                    TileLiteral::f32(0.0),
                );
                let block_max = program.load(
                    scratch.at(block_base.clone() + meta.dims.head_dim + 1),
                    Mask::all(),
                    TileLiteral::f32(NEG_MAX_F32),
                );
                let scale = softmax_partial_scale(block_max, max_score.clone());
                denom = denom + block_denom * scale.clone();
                let partial = program.load(
                    scratch.at(block_base + out_dim.clone()),
                    Mask::all(),
                    TileLiteral::f32(0.0),
                );
                acc = acc + partial * scale;
            }
            let output_index = index_n(
                meta.output_offset,
                output_strides,
                (batch_idx, head_idx, 0, out_dim),
            );
            program.store(output.at(output_index), acc / denom, Mask::all());
        },
    );
    Some(())
}

/// Build the small F32 decode-attention kernel.
///
/// Supports head dimensions no larger than the decode block size and the decode block sizes accepted by
/// [`FlashDecodeSmallMeta::decode_block`](crate::FlashDecodeSmallMeta::decode_block).
pub fn flash_decode_small<B>(
    kb: &mut fusor_tile_ir::KernelBuilder<B>,
    q: fusor_tile_ir::KernelTensorRef<B>,
    k: fusor_tile_ir::KernelTensorRef<B>,
    v: fusor_tile_ir::KernelTensorRef<B>,
    output: fusor_tile_ir::KernelTensorRef<B>,
    params: fusor_tile_ir::KernelTensorRef<B>,
    meta: FlashDecodeSmallMeta,
) -> Option<()> {
    if meta.dims.head_dim == 0
        || meta.dims.head_dim > meta.decode_block
        || meta.decode_block == 0
        || meta.groups == 0
    {
        return None;
    }
    if !matches!(meta.decode_block, 128 | 256 | 512 | 1024) {
        return None;
    }
    flash_decode_small_block(
        kb,
        meta.decode_block,
        FlashDecodeSmallBlock {
            q,
            k,
            v,
            output,
            params,
            meta,
        },
    );
    Some(())
}
