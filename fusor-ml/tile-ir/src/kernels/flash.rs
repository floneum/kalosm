use crate::{
    tile::{self, range, Tile, TileBlock},
    ElementType, F32Bits, Numeric, TileLiteral, WorkgroupAxis, F32, U32,
};

use super::helpers::{
    all, f32_tile, index2, index4, index4_const_last, reduce_workgroup, u32_tile, NEG_MAX_F32,
};
use super::types::{FlashAttentionMeta, FlashDecodeSmallMeta};

const FLASH_BLOCK: usize = 256;
const FLASH_SIMD_WIDTH: u32 = 32;
const FLASH_OUTPUTS_PER_WORKGROUP: u32 = FLASH_BLOCK as u32 / FLASH_SIMD_WIDTH;
const DECODE_HEAD_DIM: u32 = 128;

fn zero_fill<E: Numeric>() -> TileLiteral {
    match E::ELEMENT {
        ElementType::F32 => TileLiteral::F32(F32Bits::new(0.0)),
        ElementType::F16 => TileLiteral::F16(0),
        _ => panic!("flash_attention only supports F32 and F16 element types"),
    }
}

pub fn flash_attention<E: Numeric, B>(
    kb: &mut crate::kernel_builder::KernelBuilder<B>,
    q: crate::kernel_builder::KernelTensorRef<B>,
    k: crate::kernel_builder::KernelTensorRef<B>,
    v: crate::kernel_builder::KernelTensorRef<B>,
    mask: Option<crate::kernel_builder::KernelTensorRef<B>>,
    output: crate::kernel_builder::KernelTensorRef<B>,
    meta: FlashAttentionMeta,
) -> Option<()> {
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
            let lane = program.arange();
            let workgroup_x = program.program_id(WorkgroupAxis::X);
            let row = program.program_id(WorkgroupAxis::Y);
            let q_idx = program.bind(program.index(row.clone() % meta.dims.q_seq_len));
            let row_over_q = row.clone() / meta.dims.q_seq_len;
            let head_idx = program.bind(program.index(row_over_q.clone() % meta.dims.num_heads));
            let batch_idx =
                program.bind(program.index(row / (meta.dims.q_seq_len * meta.dims.num_heads)));
            let kv_head_idx = program.bind(head_idx.get() / u32_tile(groups));
            let kv_lane = program.index(lane.clone() % FLASH_SIMD_WIDTH);
            let out_dim = program.bind(program.index(
                workgroup_x * FLASH_OUTPUTS_PER_WORKGROUP + (lane.clone() / FLASH_SIMD_WIDTH),
            ));
            let out_valid = program.bind(out_dim.get().lt(u32_tile(meta.dims.head_dim)));
            // Per-iteration scratch locals — used to bridge values across
            // `if_then` branches inside the body. Not loop-carried.
            let score_local = program.private::<F32>();
            let weighted_local = program.private::<F32>();


            let kv_chunks = meta.dims.kv_seq_len.div_ceil(FLASH_SIMD_WIDTH);
            let [_final_m, final_s, final_o] = program.fold(
                range::<FLASH_BLOCK>(u32_tile::<FLASH_BLOCK>(kv_chunks)),
                [f32_tile(NEG_MAX_F32), f32_tile(0.0), f32_tile(0.0)],
                |program, chunk_idx, [m_state, s_state, o_state]| {
                    let chunk = Tile::from_index(chunk_idx);
                    let kv_idx =
                        program.bind(chunk * u32_tile(FLASH_SIMD_WIDTH) + kv_lane.clone());
                    let kv_valid = program.bind(kv_idx.get().lt(u32_tile(meta.dims.kv_seq_len)));
                    program.store_local(&score_local, f32_tile(NEG_MAX_F32));
                    program.if_then(kv_valid.get(), |program| {
                        let mut products = Vec::with_capacity(meta.dims.head_dim as usize);
                        for dim in 0..meta.dims.head_dim {
                            let q_index = index4_const_last(
                                meta.q_meta.offset,
                                q_strides,
                                batch_idx.get(),
                                head_idx.get(),
                                q_idx.get(),
                                dim,
                            );
                            let k_index = index4_const_last(
                                meta.k_meta.offset,
                                k_strides,
                                batch_idx.get(),
                                kv_head_idx.get(),
                                kv_idx.get(),
                                dim,
                            );
                            let q_value = program
                                .load_linear(q.at(q_index), all(), elem_fill)
                                .cast(ElementType::F32);
                            let k_value = program
                                .load_linear(k.at(k_index), all(), elem_fill)
                                .cast(ElementType::F32);
                            products.push(q_value * k_value);
                        }
                        let mut score = program.sum(products) * f32_tile(meta.scale.get());
                        if let (Some(mask), Some(mask_meta), Some(mask_strides)) =
                            (&mask, meta.mask_meta.as_ref(), mask_strides)
                        {
                            let mask_index =
                                index2(mask_meta.offset, mask_strides, q_idx.get(), kv_idx.get());
                            let mask_value = program
                                .load_linear(mask.at(mask_index), all(), elem_fill)
                                .cast(ElementType::F32);
                            score = score + mask_value;
                        }
                        program.store_local(&score_local, score);
                    });

                    let score = program.bind(program.load_local(&score_local));
                    let block_max = program.bind(program.subgroup_reduce_max(score.get()));
                    let old_m = program.bind(m_state);
                    let new_m = program.bind(old_m.get().max(block_max.get()));
                    let raw_exp = (score.get() - new_m.get()).exp();
                    let exp_score =
                        program.bind(Tile::select(kv_valid.get(), raw_exp, f32_tile(0.0)));
                    let block_sum = program.bind(program.subgroup_reduce_sum(exp_score.get()));

                    program.store_local(&weighted_local, f32_tile(0.0));
                    let valid_value = kv_valid.get().and(out_valid.get());
                    program.if_then(valid_value, |program| {
                        let v_index = index4(
                            meta.v_meta.offset,
                            v_strides,
                            batch_idx.get(),
                            kv_head_idx.get(),
                            kv_idx.get(),
                            out_dim.get(),
                        );
                        let v_value = program
                            .load_linear(v.at(v_index), all(), elem_fill)
                            .cast(ElementType::F32);
                        program.store_local(&weighted_local, exp_score.get() * v_value);
                    });
                    let weighted = program.load_local(&weighted_local);
                    let block_out = program.bind(program.subgroup_reduce_sum(weighted));

                    let old_m_scale = program.bind((old_m.get() - new_m.get()).exp());
                    let new_s = s_state * old_m_scale.get() + block_sum.get();
                    let new_o = o_state * old_m_scale.get() + block_out.get();
                    [new_m.get(), new_s, new_o]
                },
            );

            let store_valid = kv_lane.eq(u32_tile(0)).and(out_valid.get());
            // Bind the fold results so the divide and the `if_then` body share
            // the same SSA values rather than re-emitting the loop materialize.
            let final_o_bound = program.bind(final_o);
            let final_s_bound = program.bind(final_s);
            // Evaluate `final_m` once so the loop fires even though we don't
            // need its value in the post-loop stage.
            program.if_then(store_valid, |program| {
                let output_value =
                    (final_o_bound.get() / final_s_bound.get()).cast(E::ELEMENT);
                let output_index = index4(
                    meta.output_meta.offset,
                    output_strides,
                    batch_idx.get(),
                    head_idx.get(),
                    q_idx.get(),
                    out_dim.get(),
                );
                program.store_linear(output.at(output_index), output_value, all());
            });
        });
    }
    Some(())
}

fn decode_score_for_kv<const BLOCK: usize>(
    program: &mut TileBlock<'_, BLOCK>,
    q: &tile::Storage<F32, 1>,
    k: &tile::Storage<F32, 1>,
    meta: FlashDecodeSmallMeta,
    batch_idx: Tile<BLOCK>,
    head_idx: Tile<BLOCK>,
    kv_head_idx: Tile<BLOCK>,
    kv: Tile<BLOCK>,
    score_acc: &tile::Local<F32, BLOCK>,
    dim_local: &tile::Local<U32, BLOCK>,
) -> Tile<BLOCK> {
    program.store_local(score_acc, f32_tile(0.0));
    program.store_local(dim_local, u32_tile(0));
    program.loop_forever(|program| {
        let dim = program.load_local(dim_local);
        program.break_if(dim.clone().ge(u32_tile(DECODE_HEAD_DIM)));
        let q_index = index4(
            meta.q_offset,
            meta.q_strides,
            batch_idx.clone(),
            head_idx.clone(),
            u32_tile(0),
            dim.clone(),
        );
        let k_index = index4(
            meta.k_offset,
            meta.k_strides,
            batch_idx.clone(),
            kv_head_idx.clone(),
            kv.clone(),
            dim.clone(),
        );
        let q_value =
            program.load_linear(q.at(q_index), all(), TileLiteral::F32(F32Bits::new(0.0)));
        let k_value =
            program.load_linear(k.at(k_index), all(), TileLiteral::F32(F32Bits::new(0.0)));
        let acc = program.load_local(score_acc);
        program.store_local(score_acc, acc + q_value * k_value);
        program.store_local(dim_local, dim + u32_tile(1));
    });
    program.load_local(score_acc) * f32_tile(meta.scale.get())
}

#[allow(clippy::too_many_arguments)]
fn append_decode_output_loop<const BLOCK: usize>(
    program: &mut TileBlock<'_, BLOCK>,
    v: &tile::Storage<F32, 1>,
    output: &tile::Storage<F32, 1>,
    probs: crate::TileRef,
    meta: FlashDecodeSmallMeta,
    batch_idx: Tile<BLOCK>,
    head_idx: Tile<BLOCK>,
    kv_head_idx: Tile<BLOCK>,
    out_dim: Tile<BLOCK>,
    active_kv_len: Tile<BLOCK>,
    acc: &tile::Local<F32, BLOCK>,
    kv_local: &tile::Local<U32, BLOCK>,
) {
    program.loop_forever(|program| {
        let kv = program.load_local(kv_local);
        program.break_if(kv.clone().ge(active_kv_len.clone()));
        let prob = program.load_workgroup(probs, kv.clone());
        let v_index = index4(
            meta.v_offset,
            meta.v_strides,
            batch_idx.clone(),
            kv_head_idx.clone(),
            kv.clone(),
            out_dim.clone(),
        );
        let v_value =
            program.load_linear(v.at(v_index), all(), TileLiteral::F32(F32Bits::new(0.0)));
        let current = program.load_local(acc);
        program.store_local(acc, current + prob * v_value);
        program.store_local(kv_local, kv + u32_tile(1));
    });

    let output_value = program.load_local(acc);
    let output_index = index4(
        meta.output_offset,
        meta.output_strides,
        batch_idx,
        head_idx,
        u32_tile(0),
        out_dim,
    );
    program.store_linear(output.at(output_index), output_value, all());
}

fn flash_decode_small_block<const BLOCK: usize, B>(
    kb: &mut crate::kernel_builder::KernelBuilder<B>,
    q: crate::kernel_builder::KernelTensorRef<B>,
    k: crate::kernel_builder::KernelTensorRef<B>,
    v: crate::kernel_builder::KernelTensorRef<B>,
    output: crate::kernel_builder::KernelTensorRef<B>,
    params: crate::kernel_builder::KernelTensorRef<B>,
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
            let lane = program.arange();
            let row = program.program_id(WorkgroupAxis::X);
            let active_kv_len =
                program.load_linear(params.at(0), all(), TileLiteral::U32(meta.active_kv_len));
            let head_idx = program.index(row.clone() % meta.dims.num_heads);
            let batch_idx = program.index(row / meta.dims.num_heads);
            let kv_head_idx = head_idx.clone() / u32_tile(meta.groups);
            let lane_value = program.index(lane.clone());
            let acc = program.private::<F32>();
            let kv_local = program.private::<U32>();
            let item = program.private::<U32>();
            let dim = program.private::<U32>();
            let score_acc = program.private::<F32>();
            let max_score_local = program.private::<F32>();

            if meta.tiled {
                program.store_workgroup(reduce, lane.clone(), f32_tile(NEG_MAX_F32));
                program.store_local(&kv_local, lane_value.clone());
                program.loop_forever(|program| {
                    let kv = program.load_local(&kv_local);
                    program.break_if(kv.clone().ge(active_kv_len.clone()));
                    let score = decode_score_for_kv(
                        program,
                        &q,
                        &k,
                        meta,
                        batch_idx.clone(),
                        head_idx.clone(),
                        kv_head_idx.clone(),
                        kv.clone(),
                        &score_acc,
                        &dim,
                    );
                    let current = program.load_workgroup(reduce, lane.clone());
                    program.store_workgroup(reduce, lane.clone(), current.max(score));
                    program.store_local(&kv_local, kv + u32_tile(BLOCK as u32));
                });
                program.workgroup_barrier();
                reduce_workgroup(program, reduce, lane.clone(), |lhs, rhs| lhs.max(rhs));
                let max_score = program.load_workgroup(reduce, 0);
                program.store_local(&max_score_local, max_score);
                let max_score = program.load_local(&max_score_local);

                program.store_workgroup(reduce, lane.clone(), f32_tile(0.0));
                program.store_local(&kv_local, lane_value.clone());
                program.loop_forever(|program| {
                    let kv = program.load_local(&kv_local);
                    program.break_if(kv.clone().ge(active_kv_len.clone()));
                    let score = decode_score_for_kv(
                        program,
                        &q,
                        &k,
                        meta,
                        batch_idx.clone(),
                        head_idx.clone(),
                        kv_head_idx.clone(),
                        kv.clone(),
                        &score_acc,
                        &dim,
                    );
                    let prob = (score - max_score.clone()).exp();
                    let current = program.load_workgroup(reduce, lane.clone());
                    program.store_workgroup(reduce, lane.clone(), current + prob);
                    program.store_local(&kv_local, kv + u32_tile(BLOCK as u32));
                });
                program.workgroup_barrier();
                reduce_workgroup(program, reduce, lane.clone(), |lhs, rhs| lhs + rhs);
                let denom = program.load_workgroup(reduce, 0);

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
                                &q,
                                &k,
                                meta,
                                batch_idx.clone(),
                                head_idx.clone(),
                                kv_head_idx.clone(),
                                kv.clone(),
                                &score_acc,
                                &dim,
                            );
                            let prob = (score - max_score.clone()).exp() / denom.clone();
                            program.store_workgroup(probs, lane.clone(), prob);
                        },
                        |program| {
                            program.store_workgroup(probs, lane.clone(), f32_tile(0.0));
                        },
                    );
                    program.workgroup_barrier();
                    program.store_local(&item, u32_tile(0));
                    let out_condition = lane_value.clone().lt(u32_tile(DECODE_HEAD_DIM));
                    program.if_then(out_condition, |program| {
                        program.loop_forever(|program| {
                            let item_value = program.load_local(&item);
                            let block_done = item_value.clone().ge(u32_tile(BLOCK as u32));
                            let kv = tile_base.clone() + item_value.clone();
                            let kv_done = kv.clone().ge(active_kv_len.clone());
                            program.break_if(block_done.or(kv_done));
                            let prob = program.load_workgroup(probs, item_value.clone());
                            let v_index = index4(
                                meta.v_offset,
                                meta.v_strides,
                                batch_idx.clone(),
                                kv_head_idx.clone(),
                                kv,
                                lane_value.clone(),
                            );
                            let v_value = program.load_linear(
                                v.at(v_index),
                                all(),
                                TileLiteral::F32(F32Bits::new(0.0)),
                            );
                            let current = program.load_local(&acc);
                            program.store_local(&acc, current + prob * v_value);
                            program.store_local(&item, item_value + u32_tile(1));
                        });
                    });
                    program.workgroup_barrier();
                    program.store_local(&kv_local, tile_base + u32_tile(BLOCK as u32));
                });
                let out_condition = lane_value.clone().lt(u32_tile(DECODE_HEAD_DIM));
                program.if_then(out_condition, |program| {
                    let output_value = program.load_local(&acc);
                    let output_index = index4(
                        meta.output_offset,
                        meta.output_strides,
                        batch_idx.clone(),
                        head_idx.clone(),
                        u32_tile(0),
                        lane_value.clone(),
                    );
                    program.store_linear(output.at(output_index), output_value, all());
                });
                return;
            }

            let kv_valid = lane_value.clone().lt(active_kv_len.clone());
            program.store_workgroup(scores, lane.clone(), f32_tile(NEG_MAX_F32));
            program.store_workgroup(reduce, lane.clone(), f32_tile(NEG_MAX_F32));
            program.if_then(kv_valid.clone(), |program| {
                let score = decode_score_for_kv(
                    program,
                    &q,
                    &k,
                    meta,
                    batch_idx.clone(),
                    head_idx.clone(),
                    kv_head_idx.clone(),
                    lane_value.clone(),
                    &score_acc,
                    &dim,
                );
                program.store_workgroup(scores, lane.clone(), score.clone());
                program.store_workgroup(reduce, lane.clone(), score);
            });
            program.workgroup_barrier();
            reduce_workgroup(program, reduce, lane.clone(), |lhs, rhs| lhs.max(rhs));
            let max_score = program.load_workgroup(reduce, 0);
            let score_value = program.load_workgroup(scores, lane.clone());
            let raw_prob = (score_value - max_score).exp();
            let prob = Tile::select(kv_valid.clone(), raw_prob, f32_tile(0.0));
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

            let out_condition = lane_value.clone().lt(u32_tile(DECODE_HEAD_DIM));
            program.if_then(out_condition, |program| {
                program.store_local(&acc, f32_tile(0.0));
                program.store_local(&kv_local, u32_tile(0));
                append_decode_output_loop(
                    program,
                    &v,
                    &output,
                    probs,
                    meta,
                    batch_idx.clone(),
                    head_idx.clone(),
                    kv_head_idx.clone(),
                    lane_value.clone(),
                    active_kv_len.clone(),
                    &acc,
                    &kv_local,
                );
            });
        });
}

pub fn flash_decode_small<B>(
    kb: &mut crate::kernel_builder::KernelBuilder<B>,
    q: crate::kernel_builder::KernelTensorRef<B>,
    k: crate::kernel_builder::KernelTensorRef<B>,
    v: crate::kernel_builder::KernelTensorRef<B>,
    output: crate::kernel_builder::KernelTensorRef<B>,
    params: crate::kernel_builder::KernelTensorRef<B>,
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
