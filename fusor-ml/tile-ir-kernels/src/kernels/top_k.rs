use fusor_tile_ir::{
    tile::{self, Tile, TileBlock},
    Bool, ElementType, TileLiteral, TileUnaryOp, WorkgroupAxis, F32, U32,
};

use super::helpers::{
    all, f32_tile, lane_zero, reduce_workgroup, u32_tile, MAX_F32, NEG_MAX_F32, TOP_K_BLOCK,
};
use super::types::{MergeTopKMeta, TopKChunkMeta, TopKExactnessMeta};

const TOP_K_CHUNK: u32 = TOP_K_BLOCK as u32;

pub(super) fn is_finite<const BLOCK: usize>(value: Tile<BLOCK>) -> Tile<BLOCK> {
    let self_equal = value.clone().eq(value.clone());
    let finite_magnitude = value.unary(TileUnaryOp::Abs).le(f32_tile(MAX_F32));
    self_equal.and(finite_magnitude)
}

pub(super) fn better_candidate<const BLOCK: usize>(
    value: Tile<BLOCK>,
    id: Tile<BLOCK>,
    best_value: Tile<BLOCK>,
    best_id: Tile<BLOCK>,
) -> Tile<BLOCK> {
    let value_greater = value.clone().gt(best_value.clone());
    let value_equal = value.eq(best_value);
    let id_greater = id.gt(best_id);
    value_greater.or(value_equal.and(id_greater))
}

pub(super) fn index1<const BLOCK: usize>(
    index: Tile<BLOCK>,
    offset: u32,
    stride: u32,
) -> Tile<BLOCK> {
    super::helpers::add_scaled_index(u32_tile(offset), index, stride)
}

fn load_processor_param_f32(
    program: &TileBlock<'_, TOP_K_BLOCK>,
    params: &tile::Storage<U32, 1>,
    index: u32,
) -> Tile<TOP_K_BLOCK> {
    program
        .load_linear(params.at(index), all(), TileLiteral::U32(0))
        .bitcast(ElementType::F32)
}

pub fn top_k_chunk<B>(
    kb: &mut fusor_tile_ir::kernel_builder::KernelBuilder<B>,
    input: fusor_tile_ir::kernel_builder::KernelTensorRef<B>,
    output_ids: fusor_tile_ir::kernel_builder::KernelTensorRef<B>,
    output_values: fusor_tile_ir::kernel_builder::KernelTensorRef<B>,
    processors: Option<(
        fusor_tile_ir::kernel_builder::KernelTensorRef<B>,
        fusor_tile_ir::kernel_builder::KernelTensorRef<B>,
    )>,
    meta: TopKChunkMeta,
) -> Option<()> {
    if meta.input_len == 0 || meta.output_per_chunk == 0 {
        return None;
    }
    if meta.processors != processors.is_some() {
        return None;
    }

    let input = kb.read::<F32, 1>(input);
    let output_ids = kb.write::<U32, 1>(output_ids);
    let output_values = kb.write::<F32, 1>(output_values);
    let processors = processors.map(|(prev, params)| {
        (
            kb.read::<U32, 1>(prev),
            kb.read::<U32, 1>(params),
        )
    });
    let phase = kb.program();
    let scratch_values = phase.alloc_workgroup_array::<F32>(TOP_K_BLOCK as u32);
    let scratch_ids = phase.alloc_workgroup_array::<U32>(TOP_K_BLOCK as u32);
    let chunks = meta.input_len.div_ceil(TOP_K_CHUNK);

    phase.program_grid::<TOP_K_BLOCK>([chunks, 1, 1], |program| {
            let lane = program.arange();
            let chunk = program.program_id(WorkgroupAxis::X);
            let current_value = program.private::<F32>();
            let current_id = program.private::<U32>();
            let previous_index = program.private::<U32>();
            let previous_len_local = program.private::<U32>();
            let repeated = program.private::<Bool>();
            let sort_current_value = program.private::<F32>();
            let sort_current_id = program.private::<U32>();
            let sort_partner_value = program.private::<F32>();
            let sort_partner_id = program.private::<U32>();

            program.store_local(&current_value, f32_tile(NEG_MAX_F32));
            program.store_local(&current_id, u32_tile(u32::MAX));

            let token_id = program.index(chunk.clone() * TOP_K_CHUNK + lane.clone());
            let token_valid = token_id.clone().lt(u32_tile(meta.input_len));
            program.if_then(token_valid, |program| {
                let input_index = index1(token_id.clone(), meta.input_offset, meta.input_stride);
                let raw = program.load_linear(
                    input.at(input_index),
                    all(),
                    TileLiteral::f32(NEG_MAX_F32),
                );
                let raw_finite = is_finite(raw.clone());
                program.if_then(raw_finite, |program| {
                    let mut value = raw;
                    if let Some((previous_tokens, processor_params)) = &processors {
                        program.store_local(&current_value, value);
                        program.store_local(&previous_index, u32_tile(0));
                        program.store_local(&repeated, program.bool(false));
                        let previous_len =
                            program.load_linear(processor_params.at(2), all(), TileLiteral::U32(0));
                        program.store_local(&previous_len_local, previous_len);
                        program.loop_forever(|program| {
                            let previous_index_value = program.load_local(&previous_index);
                            let previous_len = program.load_local(&previous_len_local);
                            program.if_then(
                                previous_index_value.clone().ge(previous_len),
                                |program| {
                                    program.break_loop();
                                },
                            );
                            let previous_index_value = program.load_local(&previous_index);
                            let previous_token = program.load_linear(
                                previous_tokens.at(previous_index_value),
                                all(),
                                TileLiteral::U32(0),
                            );
                            let is_repeated = previous_token.eq(token_id.clone());
                            program.if_then(is_repeated, |program| {
                                program.store_local(&repeated, program.bool(true));
                                program.break_loop();
                            });
                            let previous_index_value = program.load_local(&previous_index);
                            program
                                .store_local(&previous_index, previous_index_value + u32_tile(1));
                        });

                        let repetition_penalty =
                            load_processor_param_f32(program, processor_params, 1);
                        let penalty_gt_one = repetition_penalty.clone().gt(f32_tile(1.0));
                        let should_apply_penalty =
                            program.load_local(&repeated).and(penalty_gt_one);
                        program.if_then(should_apply_penalty, |program| {
                            let current = program.load_local(&current_value);
                            let non_positive = current.clone().le(f32_tile(0.0));
                            program.if_else(
                                non_positive,
                                |program| {
                                    let current = program.load_local(&current_value);
                                    program.store_local(
                                        &current_value,
                                        current * repetition_penalty.clone(),
                                    );
                                },
                                |program| {
                                    let current = program.load_local(&current_value);
                                    program.store_local(
                                        &current_value,
                                        current / repetition_penalty.clone(),
                                    );
                                },
                            );
                        });

                        let temperature = load_processor_param_f32(program, processor_params, 0);
                        let temp_nonzero = temperature.clone().ne(f32_tile(0.0));
                        program.if_then(temp_nonzero, |program| {
                            let current = program.load_local(&current_value);
                            program.store_local(&current_value, current / temperature.clone());
                        });
                        value = program.load_local(&current_value);
                    }
                    let finite = is_finite(value.clone());
                    program.if_then(finite, |program| {
                        program.store_local(&current_value, value);
                        program.store_local(&current_id, token_id.clone());
                    });
                });
            });

            let value = program.load_local(&current_value);
            let id = program.load_local(&current_id);
            program.store_workgroup(scratch_values, lane.clone(), value);
            program.store_workgroup(scratch_ids, lane.clone(), id);
            program.workgroup_barrier();

            let mut size = 2;
            while size <= TOP_K_BLOCK as u32 {
                let mut stride = size / 2;
                while stride > 0 {
                    let partner = lane.clone() ^ stride;
                    let lower_lane = program.index(lane.clone() & stride).eq(u32_tile(0));
                    program.if_then(lower_lane, |program| {
                        let current_value = program.load_workgroup(scratch_values, lane.clone());
                        let current_id = program.load_workgroup(scratch_ids, lane.clone());
                        let partner_value = program.load_workgroup(scratch_values, partner.clone());
                        let partner_id = program.load_workgroup(scratch_ids, partner.clone());
                        program.store_local(&sort_current_value, current_value);
                        program.store_local(&sort_current_id, current_id);
                        program.store_local(&sort_partner_value, partner_value);
                        program.store_local(&sort_partner_id, partner_id);

                        let current_value = program.load_local(&sort_current_value);
                        let current_id = program.load_local(&sort_current_id);
                        let partner_value = program.load_local(&sort_partner_value);
                        let partner_id = program.load_local(&sort_partner_id);
                        let descending = program.index(lane.clone() & size).eq(u32_tile(0));
                        let partner_better = better_candidate(
                            partner_value.clone(),
                            partner_id.clone(),
                            current_value.clone(),
                            current_id.clone(),
                        );
                        let current_better = better_candidate(
                            current_value.clone(),
                            current_id.clone(),
                            partner_value.clone(),
                            partner_id.clone(),
                        );
                        let ascending = descending.clone().eq(program.bool(false));
                        let should_swap = descending
                            .and(partner_better)
                            .or(ascending.and(current_better));
                        program.if_then(should_swap, |program| {
                            let current_value = program.load_local(&sort_current_value);
                            let current_id = program.load_local(&sort_current_id);
                            let partner_value = program.load_local(&sort_partner_value);
                            let partner_id = program.load_local(&sort_partner_id);
                            program.store_workgroup(
                                scratch_values,
                                lane.clone(),
                                partner_value.clone(),
                            );
                            program.store_workgroup(scratch_ids, lane.clone(), partner_id.clone());
                            program.store_workgroup(
                                scratch_values,
                                partner.clone(),
                                current_value.clone(),
                            );
                            program.store_workgroup(
                                scratch_ids,
                                partner.clone(),
                                current_id.clone(),
                            );
                        });
                    });
                    program.workgroup_barrier();
                    stride /= 2;
                }
                size *= 2;
            }

            let writes_output = lane.lt(meta.output_per_chunk);
            let output_index = program.index(chunk * meta.output_per_chunk + lane.clone());
            let selected_value = program.load_workgroup(scratch_values, lane.clone());
            let selected_id = program.load_workgroup(scratch_ids, lane.clone());
            program.store_linear(
                output_values.at(output_index.clone()),
                selected_value,
                writes_output.clone(),
            );
            program.store_linear(output_ids.at(output_index), selected_id, writes_output);
        });
    Some(())
}

pub fn top_k_exactness<B>(
    kb: &mut fusor_tile_ir::kernel_builder::KernelBuilder<B>,
    top_values: fusor_tile_ir::kernel_builder::KernelTensorRef<B>,
    chunk_values: fusor_tile_ir::kernel_builder::KernelTensorRef<B>,
    flag: fusor_tile_ir::kernel_builder::KernelTensorRef<B>,
    meta: TopKExactnessMeta,
) -> Option<()> {
    if meta.top_k == 0 || meta.candidate_count >= meta.output_per_chunk {
        return None;
    }

    let top_values = kb.read::<F32, 1>(top_values);
    let chunk_values = kb.read::<F32, 1>(chunk_values);
    let flag = kb.write::<U32, 1>(flag);
    let phase = kb.program();
    let scratch = phase.alloc_workgroup_array::<U32>(TOP_K_BLOCK as u32);

    phase.program_grid::<TOP_K_BLOCK>([1, 1, 1], |program| {
            let lane = program.arange();
            let chunk = program.private::<U32>();
            let inexact = program.private::<U32>();
            let threshold_local = program.private::<F32>();
            let threshold_finite_local = program.private::<Bool>();

            let threshold_rank = u32_tile(meta.top_k - 1);
            let threshold_index = index1(
                threshold_rank,
                meta.top_values_offset,
                meta.top_values_stride,
            );
            let threshold = program.load_linear(
                top_values.at(threshold_index),
                all(),
                TileLiteral::f32(NEG_MAX_F32),
            );
            let threshold_finite = is_finite(threshold.clone());
            program.store_local(&threshold_local, threshold);
            program.store_local(&threshold_finite_local, threshold_finite);
            program.store_local(&inexact, u32_tile(0));
            program.store_local(&chunk, program.index(lane.clone()));

            program.loop_forever(|program| {
                let chunk_value = program.load_local(&chunk);
                program.break_if(chunk_value.clone().ge(u32_tile(meta.chunks)));
                let bound_rank = chunk_value.clone() * u32_tile(meta.output_per_chunk)
                    + u32_tile(meta.candidate_count);
                let bound_index = index1(
                    bound_rank,
                    meta.chunk_values_offset,
                    meta.chunk_values_stride,
                );
                let bound = program.load_linear(
                    chunk_values.at(bound_index),
                    all(),
                    TileLiteral::f32(NEG_MAX_F32),
                );
                let bound_finite = is_finite(bound.clone());
                let threshold = program.load_local(&threshold_local);
                let threshold_finite = program.load_local(&threshold_finite_local);
                let finite_inexact = threshold_finite
                    .clone()
                    .and(bound_finite.clone().and(bound.clone().ge(threshold)));
                let nonfinite_inexact = threshold_finite.eq(program.bool(false)).and(bound_finite);
                let is_inexact = finite_inexact.or(nonfinite_inexact);
                program.if_then(is_inexact, |program| {
                    program.store_local(&inexact, u32_tile(1));
                });
                let chunk_value = program.load_local(&chunk);
                program.store_local(&chunk, chunk_value + u32_tile(TOP_K_BLOCK as u32));
            });

            let inexact_value = program.load_local(&inexact);
            program.store_workgroup(scratch, lane.clone(), inexact_value);
            program.workgroup_barrier();

            reduce_workgroup(program, scratch, lane.clone(), |lhs, rhs| lhs.bit_or(rhs));

            let lane_zero = lane_zero(program, &lane);
            program.if_then(lane_zero, |program| {
                let root = program.load_workgroup(scratch, 0);
                let exact = root.eq(u32_tile(0));
                program.if_else(
                    exact,
                    |program| program.store_linear(flag.at(0), u32_tile(1), all()),
                    |program| program.store_linear(flag.at(0), u32_tile(0), all()),
                );
            });
        });
    Some(())
}

pub fn top_k_merge<B>(
    kb: &mut fusor_tile_ir::kernel_builder::KernelBuilder<B>,
    input_ids: fusor_tile_ir::kernel_builder::KernelTensorRef<B>,
    input_values: fusor_tile_ir::kernel_builder::KernelTensorRef<B>,
    output_ids: fusor_tile_ir::kernel_builder::KernelTensorRef<B>,
    output_values: fusor_tile_ir::kernel_builder::KernelTensorRef<B>,
    meta: MergeTopKMeta,
) -> Option<()> {
    if meta.chunks == 0 || meta.chunk_len == 0 || meta.k == 0 {
        return None;
    }

    let input_ids = kb.read::<U32, 1>(input_ids);
    let input_values = kb.read::<F32, 1>(input_values);
    let output_ids = kb.write::<U32, 1>(output_ids);
    let output_values = kb.write::<F32, 1>(output_values);
    let phase = kb.program();
    let chunk_positions = phase.alloc_workgroup_array::<U32>(meta.chunks);
    let scratch_values = phase.alloc_workgroup_array::<F32>(TOP_K_BLOCK as u32);
    let scratch_ids = phase.alloc_workgroup_array::<U32>(TOP_K_BLOCK as u32);
    let scratch_chunks = phase.alloc_workgroup_array::<U32>(TOP_K_BLOCK as u32);

    phase.program_grid::<TOP_K_BLOCK>([1, 1, 1], |program| {
            let lane = program.arange();
            let rank = program.private::<U32>();
            let scan_chunk = program.private::<U32>();
            let local_best_value = program.private::<F32>();
            let local_best_id = program.private::<U32>();
            let local_best_chunk = program.private::<U32>();
            let reduce_step = program.private::<U32>();
            let selected_value_local = program.private::<F32>();
            let selected_id_local = program.private::<U32>();
            let selected_chunk_local = program.private::<U32>();

            program.store_local(&scan_chunk, program.index(lane.clone()));
            program.loop_forever(|program| {
                let chunk = program.load_local(&scan_chunk);
                program.break_if(chunk.clone().ge(u32_tile(meta.chunks)));
                program.store_workgroup(chunk_positions, chunk.clone(), u32_tile(0));
                program.store_local(&scan_chunk, chunk + u32_tile(TOP_K_BLOCK as u32));
            });
            program.workgroup_barrier();

            program.store_local(&rank, u32_tile(0));
            program.loop_forever(|program| {
                let rank_value = program.load_local(&rank);
                program.break_if(rank_value.clone().ge(u32_tile(meta.k)));
                program.store_local(&local_best_value, f32_tile(NEG_MAX_F32));
                program.store_local(&local_best_id, u32_tile(u32::MAX));
                program.store_local(&local_best_chunk, u32_tile(u32::MAX));
                program.store_local(&scan_chunk, program.index(lane.clone()));

                program.loop_forever(|program| {
                    let chunk = program.load_local(&scan_chunk);
                    program.break_if(chunk.clone().ge(u32_tile(meta.chunks)));
                    let position = program.load_workgroup(chunk_positions, chunk.clone());
                    let in_chunk = position.clone().lt(u32_tile(meta.chunk_len));
                    program.if_then(in_chunk, |program| {
                        let index = chunk.clone() * u32_tile(meta.chunk_stride) + position.clone();
                        let id = program.load_linear(
                            input_ids.at(index.clone()),
                            all(),
                            TileLiteral::U32(u32::MAX),
                        );
                        let value = program.load_linear(
                            input_values.at(index),
                            all(),
                            TileLiteral::f32(NEG_MAX_F32),
                        );
                        let valid = id
                            .clone()
                            .lt(u32_tile(meta.input_len))
                            .and(is_finite(value.clone()));
                        let best_value = program.load_local(&local_best_value);
                        let best_id = program.load_local(&local_best_id);
                        let better =
                            better_candidate(value.clone(), id.clone(), best_value, best_id);
                        program.if_then(valid.and(better), |program| {
                            program.store_local(&local_best_value, value);
                            program.store_local(&local_best_id, id);
                            program.store_local(&local_best_chunk, chunk.clone());
                        });
                    });
                    let chunk = program.load_local(&scan_chunk);
                    program.store_local(&scan_chunk, chunk + u32_tile(TOP_K_BLOCK as u32));
                });

                let best_value = program.load_local(&local_best_value);
                let best_id = program.load_local(&local_best_id);
                let best_chunk = program.load_local(&local_best_chunk);
                program.store_workgroup(scratch_values, lane.clone(), best_value);
                program.store_workgroup(scratch_ids, lane.clone(), best_id);
                program.store_workgroup(scratch_chunks, lane.clone(), best_chunk);
                program.workgroup_barrier();

                program.store_local(&reduce_step, u32_tile(TOP_K_BLOCK as u32 / 2));
                program.loop_forever(|program| {
                    let step = program.load_local(&reduce_step);
                    program.break_if(step.clone().eq(u32_tile(0)));
                    let participates = program.index(lane.clone()).lt(step.clone());
                    program.if_then(participates, |program| {
                        let other_index = program.index(lane.clone()) + step.clone();
                        let other_value =
                            program.load_workgroup(scratch_values, other_index.clone());
                        let other_id = program.load_workgroup(scratch_ids, other_index.clone());
                        let other_chunk = program.load_workgroup(scratch_chunks, other_index);
                        let current_value = program.load_workgroup(scratch_values, lane.clone());
                        let current_id = program.load_workgroup(scratch_ids, lane.clone());
                        let better = better_candidate(
                            other_value.clone(),
                            other_id.clone(),
                            current_value,
                            current_id,
                        );
                        program.if_then(better, |program| {
                            program.store_workgroup(
                                scratch_values,
                                lane.clone(),
                                other_value.clone(),
                            );
                            program.store_workgroup(scratch_ids, lane.clone(), other_id.clone());
                            program.store_workgroup(
                                scratch_chunks,
                                lane.clone(),
                                other_chunk.clone(),
                            );
                        });
                    });
                    program.workgroup_barrier();
                    let step = program.load_local(&reduce_step);
                    program.store_local(&reduce_step, step / u32_tile(2));
                });

                let lane_zero = lane_zero(program, &lane);
                program.if_then(lane_zero, |program| {
                    let selected_value = program.load_workgroup(scratch_values, 0);
                    let selected_id = program.load_workgroup(scratch_ids, 0);
                    let selected_chunk = program.load_workgroup(scratch_chunks, 0);
                    program.store_local(&selected_value_local, selected_value);
                    program.store_local(&selected_id_local, selected_id);
                    program.store_local(&selected_chunk_local, selected_chunk);

                    let selected_value = program.load_local(&selected_value_local);
                    let selected_id = program.load_local(&selected_id_local);
                    let selected_chunk = program.load_local(&selected_chunk_local);
                    let rank_value = program.load_local(&rank);
                    program.store_linear(
                        output_values.at(rank_value.clone()),
                        selected_value,
                        all(),
                    );
                    program.store_linear(output_ids.at(rank_value), selected_id, all());
                    let valid_chunk = selected_chunk.clone().lt(u32_tile(meta.chunks));
                    program.if_then(valid_chunk, |program| {
                        let position =
                            program.load_workgroup(chunk_positions, selected_chunk.clone());
                        program.store_workgroup(
                            chunk_positions,
                            selected_chunk,
                            position + u32_tile(1),
                        );
                    });
                });
                program.workgroup_barrier();

                let rank_value = program.load_local(&rank);
                program.store_local(&rank, rank_value + u32_tile(1));
            });
        });
    Some(())
}
