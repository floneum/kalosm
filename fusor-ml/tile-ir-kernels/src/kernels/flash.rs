use fusor_tile_ir::{
    tile::{self, range, Mask, Tile, TileBlock, Workgroup},
    ElementType, Numeric, TileLiteral, WorkgroupAxis, F32, U32,
};

use super::helpers::{index_n, reduce_workgroup, NEG_MAX_F32};
use super::types::{FlashAttentionMeta, FlashDecodeSmallMeta};

const FLASH_BLOCK: usize = 256;
const DECODE_HEAD_DIM: u32 = 128;

fn zero_fill<E: Numeric>() -> TileLiteral {
    match E::ELEMENT {
        ElementType::F32 => TileLiteral::f32(0.0),
        ElementType::F16 => TileLiteral::F16(0),
        _ => panic!("flash_attention only supports F32 and F16 element types"),
    }
}

/// Number of output dimensions a single workgroup writes per KV-axis pass,
/// given the device's hardware subgroup size. Each subgroup handles one
/// output dim; the workgroup contains `FLASH_BLOCK / subgroup_size` subgroups.
pub const fn flash_outputs_per_workgroup(subgroup_size: u32) -> u32 {
    FLASH_BLOCK as u32 / subgroup_size
}

/// Build a streaming flash-attention kernel for F32 or F16 tensors.
///
/// `SUBGROUP_SIZE` must match the runtime hardware subgroup width on the
/// target device — the kernel layout assigns one subgroup per output dim and
/// uses `subgroup_reduce_*` to fold a `SUBGROUP_SIZE`-wide chunk of KV. Pick
/// the size by reading the device's subgroup caps; pinning is not exposed by
/// wgpu, so this must come from a fixed `(min == max)` adapter range.
///
/// The metadata supplies tensor strides, offsets, dimensions, scale, and the
/// dispatch grid. Returns `None` if the tensor ranks or optional mask binding
/// are inconsistent with the metadata.
pub fn flash_attention<E: Numeric, const SUBGROUP_SIZE: u32, B>(
    kb: &mut fusor_tile_ir::KernelBuilder<B>,
    q: fusor_tile_ir::KernelTensorRef<B>,
    k: fusor_tile_ir::KernelTensorRef<B>,
    v: fusor_tile_ir::KernelTensorRef<B>,
    mask: Option<fusor_tile_ir::KernelTensorRef<B>>,
    output: fusor_tile_ir::KernelTensorRef<B>,
    meta: FlashAttentionMeta,
) -> Option<()> {
    if SUBGROUP_SIZE == 0 || !(FLASH_BLOCK as u32).is_multiple_of(SUBGROUP_SIZE) {
        return None;
    }
    let outputs_per_workgroup = flash_outputs_per_workgroup(SUBGROUP_SIZE);
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
    let groups = meta.dims.num_heads.checked_div(meta.dims.num_kv_heads)?;
    if groups == 0 {
        return None;
    }
    let elem_fill = zero_fill::<E>();
    let q = kb.read::<E, 1>(q);
    let k = kb.read::<E, 1>(k);
    let v = kb.read::<E, 1>(v);
    let mask = mask.map(|m| kb.read::<E, 1>(m));
    let output = kb.write::<E, 1>(output);
    let phase = kb.program();
    {
        phase.program_grid::<FLASH_BLOCK>(meta.dispatch_size, |program| {
            let lane = program.lane();
            let workgroup_x = program.program_id(WorkgroupAxis::X);
            let row = program.program_id(WorkgroupAxis::Y);
            let q_idx = program.bind(program.index(row.clone() % meta.dims.q_seq_len));
            let row_over_q = row.clone() / meta.dims.q_seq_len;
            let head_idx = program.bind(program.index(row_over_q.clone() % meta.dims.num_heads));
            let batch_idx =
                program.bind(program.index(row / (meta.dims.q_seq_len * meta.dims.num_heads)));
            let kv_head_idx =
                program.bind(head_idx.clone() / Tile::literal(TileLiteral::U32(groups)));
            let kv_lane = program.index(lane.clone() % SUBGROUP_SIZE);
            let out_dim = program.bind(
                program.index(workgroup_x * outputs_per_workgroup + (lane.clone() / SUBGROUP_SIZE)),
            );
            let out_valid = program.bind(
                out_dim
                    .clone()
                    .lt(Tile::literal(TileLiteral::U32(meta.dims.head_dim))),
            );
            // Per-iteration scratch locals — used to bridge values across
            // `if_then` branches inside the body. Not loop-carried.
            let score_local = program.private::<F32>();
            let weighted_local = program.private::<F32>();

            let kv_chunks = meta.dims.kv_seq_len.div_ceil(SUBGROUP_SIZE);
            let [_final_m, final_s, final_o] = program.fold(
                range(Tile::literal(TileLiteral::U32(kv_chunks))),
                [
                    Tile::literal(TileLiteral::f32(NEG_MAX_F32)),
                    Tile::literal(TileLiteral::f32(0.0)),
                    Tile::literal(TileLiteral::f32(0.0)),
                ],
                |program, chunk_idx, [m_state, s_state, o_state]| {
                    let chunk = Tile::from_index(chunk_idx);
                    let kv_idx = program.bind(
                        chunk * Tile::literal(TileLiteral::U32(SUBGROUP_SIZE)) + kv_lane.clone(),
                    );
                    let kv_valid = program.bind(
                        kv_idx
                            .clone()
                            .lt(Tile::literal(TileLiteral::U32(meta.dims.kv_seq_len))),
                    );
                    program.store_local(&score_local, Tile::literal(TileLiteral::f32(NEG_MAX_F32)));
                    program.if_then(kv_valid.clone(), |program| {
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
                                .cast::<F32>();
                            let k_value = program
                                .load(k.at(k_index), Mask::all(), elem_fill)
                                .cast::<F32>();
                            products.push(q_value * k_value);
                        }
                        let mut score = program.sum(products)
                            * Tile::literal(TileLiteral::f32(meta.scale.get()));
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
                                .cast::<F32>();
                            score = score + mask_value;
                        }
                        program.store_local(&score_local, score);
                    });

                    let score = program.bind(program.load_local(&score_local));
                    let block_max = program.bind(program.subgroup_reduce_max(score.clone()));
                    let old_m = program.bind(m_state);
                    let new_m = program.bind(old_m.clone().max(block_max.clone()));
                    let raw_exp = (score.clone() - new_m.clone()).exp();
                    let exp_score = program.bind(Tile::select(
                        kv_valid.clone(),
                        raw_exp,
                        Tile::literal(TileLiteral::f32(0.0)),
                    ));
                    let block_sum = program.bind(program.subgroup_reduce_sum(exp_score.clone()));

                    program.store_local(&weighted_local, Tile::literal(TileLiteral::f32(0.0)));
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
                            .cast::<F32>();
                        program.store_local(&weighted_local, exp_score.clone() * v_value);
                    });
                    let weighted = program.load_local(&weighted_local);
                    let block_out = program.bind(program.subgroup_reduce_sum(weighted));

                    let old_m_scale = program.bind((old_m.clone() - new_m.clone()).exp());
                    let new_s = s_state * old_m_scale.clone() + block_sum;
                    let new_o = o_state * old_m_scale + block_out;
                    [new_m, new_s, new_o]
                },
            );

            let store_valid = kv_lane
                .eq(Tile::literal(TileLiteral::U32(0)))
                .and(out_valid.clone());
            // Bind the fold results so the divide and the `if_then` body share
            // the same SSA values rather than re-emitting the loop materialize.
            let final_o_bound = program.bind(final_o);
            let final_s_bound = program.bind(final_s);
            // Evaluate `final_m` once so the loop fires even though we don't
            // need its value in the post-loop stage.
            program.if_then(store_valid, |program| {
                let output_value = (final_o_bound.clone() / final_s_bound.clone()).cast::<E>();
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

struct DecodeScoreForKv<'a> {
    q: &'a tile::Storage<F32, 1>,
    k: &'a tile::Storage<F32, 1>,
    meta: FlashDecodeSmallMeta,
    batch_idx: Tile<U32>,
    head_idx: Tile<U32>,
    kv_head_idx: Tile<U32>,
    kv: Tile<U32>,
    score_acc: &'a tile::Local<F32>,
    dim_local: &'a tile::Local<U32>,
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
    program.store_local(score_acc, Tile::literal(TileLiteral::f32(0.0)));
    program.store_local(dim_local, Tile::literal(TileLiteral::U32(0)));
    program.loop_forever(|program| {
        let dim = program.load_local(dim_local);
        program.break_if(
            dim.clone()
                .ge(Tile::literal(TileLiteral::U32(DECODE_HEAD_DIM))),
        );
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
        program.store_local(dim_local, dim + Tile::literal(TileLiteral::U32(1)));
    });
    program.load_local(score_acc) * Tile::literal(TileLiteral::f32(meta.scale.get()))
}

struct DecodeOutputLoop<'a> {
    v: &'a tile::Storage<F32, 1>,
    output: &'a tile::Storage<F32, 1>,
    probs: Workgroup<F32>,
    meta: FlashDecodeSmallMeta,
    batch_idx: Tile<U32>,
    head_idx: Tile<U32>,
    kv_head_idx: Tile<U32>,
    out_dim: Tile<U32>,
    active_kv_len: Tile<U32>,
    acc: &'a tile::Local<F32>,
    kv_local: &'a tile::Local<U32>,
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
        program.store_local(kv_local, kv + Tile::literal(TileLiteral::U32(1)));
    });

    let output_value = program.load_local(acc);
    let output_index = index_n(
        meta.output_offset,
        meta.output_strides,
        (batch_idx, head_idx, 0, out_dim),
    );
    program.store(output.at(output_index), output_value, Mask::all());
}

fn flash_decode_small_block<const BLOCK: usize, B>(
    kb: &mut fusor_tile_ir::KernelBuilder<B>,
    q: fusor_tile_ir::KernelTensorRef<B>,
    k: fusor_tile_ir::KernelTensorRef<B>,
    v: fusor_tile_ir::KernelTensorRef<B>,
    output: fusor_tile_ir::KernelTensorRef<B>,
    params: fusor_tile_ir::KernelTensorRef<B>,
    meta: FlashDecodeSmallMeta,
) {
    let q = kb.read::<F32, 1>(q);
    let k = kb.read::<F32, 1>(k);
    let v = kb.read::<F32, 1>(v);
    let output = kb.write::<F32, 1>(output);
    let params = kb.read::<U32, 1>(params);
    let phase = kb.program();
    let scores = phase.alloc_workgroup_array::<F32>(BLOCK as u32);
    let probs = phase.alloc_workgroup_array::<F32>(BLOCK as u32);
    let reduce = phase.alloc_workgroup_array::<F32>(BLOCK as u32);

    phase.program_grid::<BLOCK>([meta.dims.batch * meta.dims.num_heads, 1, 1], |program| {
        let lane = program.lane();
        let row = program.program_id(WorkgroupAxis::X);
        let active_kv_len = program.load(
            params.at(0),
            Mask::all(),
            TileLiteral::U32(meta.active_kv_len),
        );
        let head_idx = program.index(row.clone() % meta.dims.num_heads);
        let batch_idx = program.index(row / meta.dims.num_heads);
        let kv_head_idx = head_idx.clone() / Tile::literal(TileLiteral::U32(meta.groups));
        let lane_value = program.index(lane.clone());
        let acc = program.private::<F32>();
        let kv_local = program.private::<U32>();
        let item = program.private::<U32>();
        let dim = program.private::<U32>();
        let score_acc = program.private::<F32>();
        let max_score_local = program.private::<F32>();

        if meta.tiled {
            program.store_workgroup(
                reduce,
                lane.clone(),
                Tile::literal(TileLiteral::f32(NEG_MAX_F32)),
            );
            program.store_local(&kv_local, lane_value.clone());
            program.loop_forever(|program| {
                let kv = program.load_local(&kv_local);
                program.break_if(kv.clone().ge(active_kv_len.clone()));
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
                let current = program.load_workgroup(reduce, lane.clone());
                program.store_workgroup(reduce, lane.clone(), current.max(score));
                program.store_local(
                    &kv_local,
                    kv + Tile::literal(TileLiteral::U32(BLOCK as u32)),
                );
            });
            program.workgroup_barrier();
            reduce_workgroup(program, reduce, lane.clone(), |lhs, rhs| lhs.max(rhs));
            let max_score = program.load_workgroup(reduce, 0);
            program.store_local(&max_score_local, max_score);
            let max_score = program.load_local(&max_score_local);

            program.store_workgroup(reduce, lane.clone(), Tile::literal(TileLiteral::f32(0.0)));
            program.store_local(&kv_local, lane_value.clone());
            program.loop_forever(|program| {
                let kv = program.load_local(&kv_local);
                program.break_if(kv.clone().ge(active_kv_len.clone()));
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
                let current = program.load_workgroup(reduce, lane.clone());
                program.store_workgroup(reduce, lane.clone(), current + prob);
                program.store_local(
                    &kv_local,
                    kv + Tile::literal(TileLiteral::U32(BLOCK as u32)),
                );
            });
            program.workgroup_barrier();
            reduce_workgroup(program, reduce, lane.clone(), |lhs, rhs| lhs + rhs);
            let denom = program.load_workgroup(reduce, 0);

            program.store_local(&acc, Tile::literal(TileLiteral::f32(0.0)));
            program.store_local(&kv_local, Tile::literal(TileLiteral::U32(0)));
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
                        program.store_workgroup(probs, lane.clone(), prob);
                    },
                    |program| {
                        program.store_workgroup(
                            probs,
                            lane.clone(),
                            Tile::literal(TileLiteral::f32(0.0)),
                        );
                    },
                );
                program.workgroup_barrier();
                program.store_local(&item, Tile::literal(TileLiteral::U32(0)));
                let out_condition = lane_value
                    .clone()
                    .lt(Tile::literal(TileLiteral::U32(DECODE_HEAD_DIM)));
                program.if_then(out_condition, |program| {
                    program.loop_forever(|program| {
                        let item_value = program.load_local(&item);
                        let block_done = item_value
                            .clone()
                            .ge(Tile::literal(TileLiteral::U32(BLOCK as u32)));
                        let kv = tile_base.clone() + item_value.clone();
                        let kv_done = kv.clone().ge(active_kv_len.clone());
                        program.break_if(block_done.or(kv_done));
                        let prob = program.load_workgroup(probs, item_value.clone());
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
                        program.store_local(&item, item_value + Tile::literal(TileLiteral::U32(1)));
                    });
                });
                program.workgroup_barrier();
                program.store_local(
                    &kv_local,
                    tile_base + Tile::literal(TileLiteral::U32(BLOCK as u32)),
                );
            });
            let out_condition = lane_value
                .clone()
                .lt(Tile::literal(TileLiteral::U32(DECODE_HEAD_DIM)));
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
        program.store_workgroup(
            scores,
            lane.clone(),
            Tile::literal(TileLiteral::f32(NEG_MAX_F32)),
        );
        program.store_workgroup(
            reduce,
            lane.clone(),
            Tile::literal(TileLiteral::f32(NEG_MAX_F32)),
        );
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
            program.store_workgroup(scores, lane.clone(), score.clone());
            program.store_workgroup(reduce, lane.clone(), score);
        });
        program.workgroup_barrier();
        reduce_workgroup(program, reduce, lane.clone(), |lhs, rhs| lhs.max(rhs));
        let max_score = program.load_workgroup(reduce, 0);
        let score_value = program.load_workgroup(scores, lane.clone());
        let raw_prob = (score_value - max_score).exp();
        let prob = Tile::select(
            kv_valid.clone(),
            raw_prob,
            Tile::literal(TileLiteral::f32(0.0)),
        );
        program.store_workgroup(probs, lane.clone(), prob.clone());
        program.store_workgroup(reduce, lane.clone(), prob);
        program.workgroup_barrier();
        reduce_workgroup(program, reduce, lane.clone(), |lhs, rhs| lhs + rhs);
        let denom = program.load_workgroup(reduce, 0);
        program.if_then(kv_valid, |program| {
            let prob = program.load_workgroup(probs, lane.clone()) / denom.clone();
            program.store_workgroup(probs, lane.clone(), prob);
        });
        program.workgroup_barrier();

        let out_condition = lane_value
            .clone()
            .lt(Tile::literal(TileLiteral::U32(DECODE_HEAD_DIM)));
        program.if_then(out_condition, |program| {
            program.store_local(&acc, Tile::literal(TileLiteral::f32(0.0)));
            program.store_local(&kv_local, Tile::literal(TileLiteral::U32(0)));
            append_decode_output_loop(
                program,
                DecodeOutputLoop {
                    v: &v,
                    output: &output,
                    probs,
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
    });
}

/// Build the small F32 decode-attention kernel.
///
/// Supports fixed head dimension 128 and the decode block sizes accepted by
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
    if meta.dims.head_dim != DECODE_HEAD_DIM || meta.decode_block == 0 || meta.groups == 0 {
        return None;
    }
    match meta.decode_block {
        128 => flash_decode_small_block::<128, B>(kb, q, k, v, output, params, meta),
        512 => flash_decode_small_block::<512, B>(kb, q, k, v, output, params, meta),
        1024 => flash_decode_small_block::<1024, B>(kb, q, k, v, output, params, meta),
        _ => return None,
    }
    Some(())
}
