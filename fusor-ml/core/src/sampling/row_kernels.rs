use fusor_tile_ir::{
    ElementType, KernelBuilder, KernelTensorRef, ScalarElement, TileLiteral, TileUnaryOp,
    tile::{Mask, Storage, Tile, TileBlock},
};

use crate::{
    row_dispatch::{RowDispatchSpec, emit_row_grid},
    sampling::{
        GPU_SAMPLE_STATUS_INVALID, GPU_SAMPLE_STATUS_RETRY_NEEDED, GPU_SAMPLE_STATUS_SAMPLED,
        TOP_K_BLOCK,
    },
};

// Naga's WGSL writer prints `f32::MAX` as a decimal literal that the WGSL
// parser rejects on WebGPU. Keep shader sentinels just below that edge.
const MAX_F32: f32 = 3.40282e38;
const NEG_MAX_F32: f32 = -MAX_F32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TopKChunkMeta {
    pub(crate) input_len: u32,
    pub(crate) output_per_chunk: u32,
    pub(crate) input_offset: u32,
    pub(crate) input_stride: u32,
    pub(crate) processors: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TopKExactnessMeta {
    pub(crate) chunks: u32,
    pub(crate) candidate_count: u32,
    pub(crate) output_per_chunk: u32,
    pub(crate) top_k: u32,
    pub(crate) top_values_offset: u32,
    pub(crate) top_values_stride: u32,
    pub(crate) chunk_values_offset: u32,
    pub(crate) chunk_values_stride: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MergeTopKMeta {
    pub(crate) chunks: u32,
    pub(crate) chunk_len: u32,
    pub(crate) chunk_stride: u32,
    pub(crate) input_len: u32,
    pub(crate) k: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SamplerMeta {
    pub(crate) top_k: u32,
    pub(crate) ids_offset: u32,
    pub(crate) ids_stride: u32,
    pub(crate) values_offset: u32,
    pub(crate) values_stride: u32,
    pub(crate) has_exactness_flag: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CategoricalSamplerMeta {
    pub(crate) input_len: u32,
    pub(crate) input_offset: u32,
    pub(crate) input_stride: u32,
    pub(crate) block: u32,
}

fn u32t(value: u32) -> Tile {
    Tile::u32(value)
}

fn f32t(value: f32) -> Tile {
    Tile::f32(value)
}

fn index1(offset: u32, stride: u32, index: Tile) -> Tile {
    match stride {
        0 => u32t(offset),
        1 => u32t(offset) + index,
        _ => u32t(offset) + index * u32t(stride),
    }
}

fn first_lane(lane: &Tile) -> Mask {
    lane.eq(u32t(0))
}

fn is_finite(value: Tile) -> Mask {
    let self_equal = value.clone().eq(value.clone());
    let finite_magnitude = value.unary(TileUnaryOp::Abs).le(f32t(MAX_F32));
    self_equal & finite_magnitude
}

fn better_candidate(value: Tile, id: Tile, best_value: Tile, best_id: Tile) -> Mask {
    value.clone().gt(best_value.clone()) | (value.eq(best_value) & id.gt(best_id))
}

fn load_processor_param_f32(program: &TileBlock<'_>, params: &Storage, index: u32) -> Tile {
    program
        .load(params.at(index), Mask::all(), TileLiteral::U32(0))
        .bitcast(ElementType::F32)
}

pub(crate) fn top_k_chunk<B>(
    kb: &mut KernelBuilder<B>,
    input: KernelTensorRef<B>,
    output_ids: KernelTensorRef<B>,
    output_values: KernelTensorRef<B>,
    processors: Option<(KernelTensorRef<B>, KernelTensorRef<B>)>,
    meta: TopKChunkMeta,
) -> Option<()> {
    if meta.input_len == 0 || meta.output_per_chunk == 0 {
        return None;
    }
    if meta.processors != processors.is_some() {
        return None;
    }

    let input = kb.read(ElementType::F32, input);
    let output_ids = kb.write(ElementType::U32, output_ids);
    let output_values = kb.write(ElementType::F32, output_values);
    let processors = processors.map(|(prev, params)| {
        (
            kb.read(ElementType::U32, prev),
            kb.read(ElementType::U32, params),
        )
    });
    let phase = kb.program();
    let scratch_values = phase.alloc_workgroup_array(ScalarElement::F32, TOP_K_BLOCK);
    let scratch_ids = phase.alloc_workgroup_array(ScalarElement::U32, TOP_K_BLOCK);
    let chunks = meta.input_len.div_ceil(TOP_K_BLOCK);

    emit_row_grid(
        phase,
        RowDispatchSpec::explicit(chunks, TOP_K_BLOCK, [chunks, 1, 1]),
        |program, ctx| {
            let lane = ctx.lane;
            let chunk = ctx.row;
            let current_value = program.private(ElementType::F32);
            let current_id = program.private(ElementType::U32);
            let repeated = program.private(ElementType::Bool);
            let sort_current_value = program.private(ElementType::F32);
            let sort_current_id = program.private(ElementType::U32);
            let sort_partner_value = program.private(ElementType::F32);
            let sort_partner_id = program.private(ElementType::U32);

            program.store_local(&current_value, f32t(NEG_MAX_F32));
            program.store_local(&current_id, u32t(u32::MAX));

            let token_id = chunk.clone() * TOP_K_BLOCK + lane.clone();
            let token_valid = token_id.clone().lt(u32t(meta.input_len));
            program.if_then(token_valid, |program| {
                let input_index = index1(meta.input_offset, meta.input_stride, token_id.clone());
                let raw = program.load(
                    input.at(input_index),
                    Mask::all(),
                    TileLiteral::f32(NEG_MAX_F32),
                );
                let raw_finite = is_finite(raw.clone());
                program.if_then(raw_finite, |program| {
                    let mut value = raw;
                    if let Some((previous_tokens, processor_params)) = &processors {
                        program.store_local(&current_value, value);
                        program.store_local(&repeated, Tile::bool(false));
                        let previous_len = program.bind(program.load(
                            processor_params.at(2),
                            Mask::all(),
                            TileLiteral::U32(0),
                        ));
                        program.fold_state(
                            u32t(0),
                            |_, previous_index| previous_index.ge(previous_len.clone()),
                            |program, previous_index| {
                                let previous_token = program.load(
                                    previous_tokens.at(previous_index.clone()),
                                    Mask::all(),
                                    TileLiteral::U32(0),
                                );
                                let is_repeated = previous_token.eq(token_id.clone());
                                program.if_then(is_repeated, |program| {
                                    program.store_local(&repeated, Tile::bool(true));
                                    program.break_loop();
                                });
                                previous_index + u32t(1)
                            },
                        );

                        let repetition_penalty =
                            load_processor_param_f32(program, processor_params, 1);
                        let penalty_gt_one = repetition_penalty.clone().gt(f32t(1.0));
                        let should_apply_penalty = program.load_local(&repeated) & penalty_gt_one;
                        program.if_then(should_apply_penalty, |program| {
                            let current = program.load_local(&current_value);
                            let penalized = Tile::select(
                                current.clone().le(f32t(0.0)),
                                current.clone() * repetition_penalty.clone(),
                                current / repetition_penalty.clone(),
                            );
                            program.store_local(&current_value, penalized);
                        });

                        let temperature = load_processor_param_f32(program, processor_params, 0);
                        let temp_nonzero = temperature.clone().ne(f32t(0.0));
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
            program.store_workgroup(&scratch_values, lane.clone(), value);
            program.store_workgroup(&scratch_ids, lane.clone(), id);
            program.workgroup_barrier();

            let mut size = 2;
            while size <= TOP_K_BLOCK {
                let mut stride = size / 2;
                while stride > 0 {
                    let partner = lane.clone() ^ stride;
                    let lower_lane = (lane.clone() & stride).eq(u32t(0));
                    program.if_then(lower_lane, |program| {
                        let current_value = program.load_workgroup(&scratch_values, lane.clone());
                        let current_id = program.load_workgroup(&scratch_ids, lane.clone());
                        let partner_value =
                            program.load_workgroup(&scratch_values, partner.clone());
                        let partner_id = program.load_workgroup(&scratch_ids, partner.clone());
                        program.store_local(&sort_current_value, current_value);
                        program.store_local(&sort_current_id, current_id);
                        program.store_local(&sort_partner_value, partner_value);
                        program.store_local(&sort_partner_id, partner_id);

                        let current_value = program.load_local(&sort_current_value);
                        let current_id = program.load_local(&sort_current_id);
                        let partner_value = program.load_local(&sort_partner_value);
                        let partner_id = program.load_local(&sort_partner_id);
                        let descending = (lane.clone() & size).eq(u32t(0));
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
                        let ascending = descending.clone().eq(Tile::bool(false));
                        let should_swap =
                            (descending & partner_better) | (ascending & current_better);
                        program.if_then(should_swap, |program| {
                            let current_value = program.load_local(&sort_current_value);
                            let current_id = program.load_local(&sort_current_id);
                            let partner_value = program.load_local(&sort_partner_value);
                            let partner_id = program.load_local(&sort_partner_id);
                            program.store_workgroup(
                                &scratch_values,
                                lane.clone(),
                                partner_value.clone(),
                            );
                            program.store_workgroup(&scratch_ids, lane.clone(), partner_id.clone());
                            program.store_workgroup(
                                &scratch_values,
                                partner.clone(),
                                current_value.clone(),
                            );
                            program.store_workgroup(&scratch_ids, partner.clone(), current_id);
                        });
                    });
                    program.workgroup_barrier();
                    stride /= 2;
                }
                size *= 2;
            }

            let writes_output = lane.lt(meta.output_per_chunk);
            let output_index = chunk * meta.output_per_chunk + lane.clone();
            let selected_value = program.load_workgroup(&scratch_values, lane.clone());
            let selected_id = program.load_workgroup(&scratch_ids, lane.clone());
            program.store(
                output_values.at(output_index.clone()),
                selected_value,
                writes_output.clone(),
            );
            program.store(output_ids.at(output_index), selected_id, writes_output);
        },
    );
    Some(())
}

pub(crate) fn top_k_exactness<B>(
    kb: &mut KernelBuilder<B>,
    top_values: KernelTensorRef<B>,
    chunk_values: KernelTensorRef<B>,
    flag: KernelTensorRef<B>,
    meta: TopKExactnessMeta,
) -> Option<()> {
    if meta.top_k == 0 || meta.candidate_count >= meta.output_per_chunk {
        return None;
    }

    let top_values = kb.read(ElementType::F32, top_values);
    let chunk_values = kb.read(ElementType::F32, chunk_values);
    let flag = kb.write(ElementType::U32, flag);
    let phase = kb.program();

    emit_row_grid(
        phase,
        RowDispatchSpec::single(TOP_K_BLOCK),
        |program, ctx| {
            let lane = ctx.lane;
            let inexact = program.private(ElementType::U32);

            let threshold_rank = u32t(meta.top_k - 1);
            let threshold_index = index1(
                meta.top_values_offset,
                meta.top_values_stride,
                threshold_rank,
            );
            let threshold = program.load(
                top_values.at(threshold_index),
                Mask::all(),
                TileLiteral::f32(NEG_MAX_F32),
            );
            let threshold = program.bind(threshold);
            let threshold_finite = program.bind(is_finite(threshold.clone()));
            program.store_local(&inexact, u32t(0));

            program.fold_state(
                lane.clone(),
                |_, chunk_value| chunk_value.ge(u32t(meta.chunks)),
                |program, chunk_value| {
                    let bound_rank = chunk_value.clone() * u32t(meta.output_per_chunk)
                        + u32t(meta.candidate_count);
                    let bound_index = index1(
                        meta.chunk_values_offset,
                        meta.chunk_values_stride,
                        bound_rank,
                    );
                    let bound = program.load(
                        chunk_values.at(bound_index),
                        Mask::all(),
                        TileLiteral::f32(NEG_MAX_F32),
                    );
                    let bound_finite = is_finite(bound.clone());
                    let finite_inexact = threshold_finite.clone()
                        & (bound_finite.clone() & bound.clone().ge(threshold.clone()));
                    let nonfinite_inexact = threshold_finite.eq(Tile::bool(false)) & bound_finite;
                    let is_inexact = finite_inexact | nonfinite_inexact;
                    program.if_then(is_inexact, |program| {
                        program.store_local(&inexact, u32t(1));
                    });
                    chunk_value + u32t(TOP_K_BLOCK)
                },
            );

            let inexact_value = program.load_local(&inexact);
            let any_inexact = program.reduce_max(inexact_value);
            let any_inexact = program.bind(any_inexact);

            program.if_then(first_lane(&lane), |program| {
                let exact = any_inexact.eq(u32t(0));
                let value = Tile::select(exact, u32t(1), u32t(0));
                program.store(flag.at(0), value, Mask::all());
            });
        },
    );
    Some(())
}

pub(crate) fn top_k_merge<B>(
    kb: &mut KernelBuilder<B>,
    input_ids: KernelTensorRef<B>,
    input_values: KernelTensorRef<B>,
    output_ids: KernelTensorRef<B>,
    output_values: KernelTensorRef<B>,
    meta: MergeTopKMeta,
) -> Option<()> {
    if meta.chunks == 0 || meta.chunk_len == 0 || meta.k == 0 {
        return None;
    }

    let input_ids = kb.read(ElementType::U32, input_ids);
    let input_values = kb.read(ElementType::F32, input_values);
    let output_ids = kb.write(ElementType::U32, output_ids);
    let output_values = kb.write(ElementType::F32, output_values);
    let phase = kb.program();
    let chunk_positions = phase.alloc_workgroup_array(ScalarElement::U32, meta.chunks);

    emit_row_grid(
        phase,
        RowDispatchSpec::single(TOP_K_BLOCK),
        |program, ctx| {
            let lane = ctx.lane;
            let local_best_value = program.private(ElementType::F32);
            let local_best_id = program.private(ElementType::U32);
            let local_best_chunk = program.private(ElementType::U32);

            program.fold_state(
                lane.clone(),
                |_, chunk| chunk.ge(u32t(meta.chunks)),
                |program, chunk| {
                    program.store_workgroup(&chunk_positions, chunk.clone(), u32t(0));
                    chunk + u32t(TOP_K_BLOCK)
                },
            );
            program.workgroup_barrier();

            let first_lane = first_lane(&lane);
            program.if_then(first_lane, |program| {
                program.fold_state(
                    u32t(0),
                    |_, rank| rank.ge(u32t(meta.k)),
                    |program, rank| {
                        program.store_local(&local_best_value, f32t(NEG_MAX_F32));
                        program.store_local(&local_best_id, u32t(u32::MAX));
                        program.store_local(&local_best_chunk, u32t(u32::MAX));

                        program.fold_state(
                            u32t(0),
                            |_, chunk| chunk.ge(u32t(meta.chunks)),
                            |program, chunk| {
                                let position =
                                    program.load_workgroup(&chunk_positions, chunk.clone());
                                let in_chunk = position.clone().lt(u32t(meta.chunk_len));
                                program.if_then(in_chunk, |program| {
                                    let index =
                                        chunk.clone() * u32t(meta.chunk_stride) + position.clone();
                                    let id = program.load(
                                        input_ids.at(index.clone()),
                                        Mask::all(),
                                        TileLiteral::U32(u32::MAX),
                                    );
                                    let value = program.load(
                                        input_values.at(index),
                                        Mask::all(),
                                        TileLiteral::f32(NEG_MAX_F32),
                                    );
                                    let valid = id.clone().lt(u32t(meta.input_len))
                                        & is_finite(value.clone());
                                    let best_value = program.load_local(&local_best_value);
                                    let best_id = program.load_local(&local_best_id);
                                    let better = better_candidate(
                                        value.clone(),
                                        id.clone(),
                                        best_value,
                                        best_id,
                                    );
                                    program.if_then(valid & better, |program| {
                                        program.store_local(&local_best_value, value);
                                        program.store_local(&local_best_id, id);
                                        program.store_local(&local_best_chunk, chunk.clone());
                                    });
                                });
                                chunk + u32t(1)
                            },
                        );

                        let selected_value = program.load_local(&local_best_value);
                        let selected_id = program.load_local(&local_best_id);
                        let selected_chunk = program.load_local(&local_best_chunk);
                        program.store(output_values.at(rank.clone()), selected_value, Mask::all());
                        program.store(output_ids.at(rank.clone()), selected_id, Mask::all());
                        let valid_chunk = selected_chunk.clone().lt(u32t(meta.chunks));
                        program.if_then(valid_chunk, |program| {
                            let position =
                                program.load_workgroup(&chunk_positions, selected_chunk.clone());
                            program.store_workgroup(
                                &chunk_positions,
                                selected_chunk,
                                position + u32t(1),
                            );
                        });
                        rank + u32t(1)
                    },
                );
            });
        },
    );
    Some(())
}

fn sampler_top_value(
    program: &TileBlock<'_>,
    values: &Storage,
    meta: SamplerMeta,
    index: Tile,
) -> Tile {
    let index = index1(meta.values_offset, meta.values_stride, index);
    program.load(values.at(index), Mask::all(), TileLiteral::f32(NEG_MAX_F32))
}

fn sampler_top_id(program: &TileBlock<'_>, ids: &Storage, meta: SamplerMeta, index: Tile) -> Tile {
    let index = index1(meta.ids_offset, meta.ids_stride, index);
    program.load(ids.at(index), Mask::all(), TileLiteral::U32(u32::MAX))
}

fn sampler_top_weight(
    program: &TileBlock<'_>,
    values: &Storage,
    meta: SamplerMeta,
    max_value: Tile,
    index: Tile,
) -> Tile {
    (sampler_top_value(program, values, meta, index) - max_value).exp()
}

fn load_param_f32(program: &TileBlock<'_>, params: &Storage, index: u32) -> Tile {
    program.load(params.at(index), Mask::all(), TileLiteral::f32(0.0))
}

fn store_sample_result(program: &mut TileBlock<'_>, output: &Storage, status: u32, token: Tile) {
    program.store(output.at(0), u32t(status), Mask::all());
    program.store(output.at(1), token, Mask::all());
}

pub(crate) struct CategoricalSampler<B> {
    pub(crate) logits: KernelTensorRef<B>,
    pub(crate) params: KernelTensorRef<B>,
    pub(crate) output: KernelTensorRef<B>,
    pub(crate) meta: CategoricalSamplerMeta,
}

/// Sample directly from one complete, unfiltered logits row. This avoids the
/// separate chunk-sort, merge, and sampler dispatches used by the general
/// top-k path. The in-workgroup sort preserves that path's sampling order:
/// processed logit descending, with token id descending for ties.
pub(crate) fn categorical_sampler<B>(
    kb: &mut KernelBuilder<B>,
    spec: CategoricalSampler<B>,
) -> Option<()> {
    let CategoricalSampler {
        logits,
        params,
        output,
        meta,
    } = spec;
    if meta.input_len == 0
        || meta.block == 0
        || !meta.block.is_power_of_two()
        || meta.input_len > meta.block
        || meta.block > TOP_K_BLOCK
    {
        return None;
    }

    let logits = kb.read(ElementType::F32, logits);
    let params = kb.read(ElementType::F32, params);
    let output = kb.write(ElementType::U32, output);
    let phase = kb.program();
    let scratch_values = phase.alloc_workgroup_array(ScalarElement::F32, meta.block);
    let scratch_ids = phase.alloc_workgroup_array(ScalarElement::U32, meta.block);
    let weights = phase.alloc_workgroup_array(ScalarElement::F32, meta.block);

    emit_row_grid(
        phase,
        RowDispatchSpec::single(meta.block),
        |program, ctx| {
            let lane = ctx.lane;
            let sort_current_value = program.private(ElementType::F32);
            let sort_current_id = program.private(ElementType::U32);
            let sort_partner_value = program.private(ElementType::F32);
            let sort_partner_id = program.private(ElementType::U32);
            let active = lane.clone().lt(u32t(meta.input_len));
            let input_index = index1(meta.input_offset, meta.input_stride, lane.clone());
            let raw = program.load(
                logits.at(input_index),
                active.clone(),
                TileLiteral::f32(NEG_MAX_F32),
            );
            let scaled = program.private(ElementType::F32);
            program.store_local(&scaled, raw.clone());
            let temperature = load_param_f32(program, &params, 1);
            program.if_then(temperature.clone().ne(f32t(0.0)), |program| {
                program.store_local(&scaled, raw.clone() / temperature.clone());
            });
            let scaled = program.load_local(&scaled);
            let valid = active & is_finite(raw) & is_finite(scaled.clone());
            let value = Tile::select(valid.clone(), scaled, f32t(NEG_MAX_F32));
            let id = Tile::select(valid, lane.clone(), u32t(u32::MAX));
            program.store_workgroup(&scratch_values, lane.clone(), value);
            program.store_workgroup(&scratch_ids, lane.clone(), id);
            program.workgroup_barrier();

            let mut size = 2;
            while size <= meta.block {
                let mut stride = size / 2;
                while stride > 0 {
                    let partner = lane.clone() ^ stride;
                    let lower_lane = (lane.clone() & stride).eq(u32t(0));
                    program.if_then(lower_lane, |program| {
                        let current_value = program.load_workgroup(&scratch_values, lane.clone());
                        let current_id = program.load_workgroup(&scratch_ids, lane.clone());
                        let partner_value =
                            program.load_workgroup(&scratch_values, partner.clone());
                        let partner_id = program.load_workgroup(&scratch_ids, partner.clone());
                        program.store_local(&sort_current_value, current_value);
                        program.store_local(&sort_current_id, current_id);
                        program.store_local(&sort_partner_value, partner_value);
                        program.store_local(&sort_partner_id, partner_id);

                        let current_value = program.load_local(&sort_current_value);
                        let current_id = program.load_local(&sort_current_id);
                        let partner_value = program.load_local(&sort_partner_value);
                        let partner_id = program.load_local(&sort_partner_id);
                        let descending = (lane.clone() & size).eq(u32t(0));
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
                        let ascending = descending.clone().eq(Tile::bool(false));
                        let should_swap =
                            (descending & partner_better) | (ascending & current_better);
                        program.if_then(should_swap, |program| {
                            let current_value = program.load_local(&sort_current_value);
                            let current_id = program.load_local(&sort_current_id);
                            let partner_value = program.load_local(&sort_partner_value);
                            let partner_id = program.load_local(&sort_partner_id);
                            program.store_workgroup(
                                &scratch_values,
                                lane.clone(),
                                partner_value.clone(),
                            );
                            program.store_workgroup(&scratch_ids, lane.clone(), partner_id.clone());
                            program.store_workgroup(
                                &scratch_values,
                                partner.clone(),
                                current_value.clone(),
                            );
                            program.store_workgroup(&scratch_ids, partner, current_id);
                        });
                    });
                    program.workgroup_barrier();
                    stride /= 2;
                }
                size *= 2;
            }

            let max_value = program.load_workgroup(&scratch_values, u32t(0));
            let sorted_value = program.load_workgroup(&scratch_values, lane.clone());
            let sorted_id = program.load_workgroup(&scratch_ids, lane.clone());
            let valid = sorted_id.ne(u32t(u32::MAX));
            let weight = Tile::select(
                valid,
                (sorted_value - max_value).unary(TileUnaryOp::Exp),
                f32t(0.0),
            );
            program.store_workgroup(&weights, lane.clone(), weight.clone());
            program.workgroup_barrier();
            let total = program.reduce_sum(weight);
            let total = program.bind(total);

            program.if_then(first_lane(&lane), |program| {
                let first_id = program.load_workgroup(&scratch_ids, u32t(0));
                let total_invalid = is_finite(total.clone()).eq(Tile::bool(false))
                    | total.clone().le(f32t(0.0))
                    | first_id.clone().eq(u32t(u32::MAX));
                program.if_then(total_invalid, |program| {
                    store_sample_result(program, &output, GPU_SAMPLE_STATUS_INVALID, u32t(0));
                    program.return_();
                });

                let random = load_param_f32(program, &params, 0);
                let cutoff = program.private(ElementType::U32);
                let cutoff_sum = program.private(ElementType::F32);
                program.store_local(&cutoff, u32t(meta.input_len));
                program.store_local(&cutoff_sum, f32t(0.0));
                program.fold_state(
                    u32t(0),
                    |_, index| index.ge(u32t(meta.input_len)),
                    |program, index| {
                        let weight = program.load_workgroup(&weights, index.clone());
                        let next = program.load_local(&cutoff_sum) + weight;
                        program.store_local(&cutoff_sum, next.clone());
                        program.if_then(next.ge(total.clone()), |program| {
                            program.store_local(&cutoff, index.clone() + u32t(1));
                            program.break_loop();
                        });
                        index + u32t(1)
                    },
                );
                let cutoff_sum = program.load_local(&cutoff_sum).max(f32t(1.0e-20));
                let threshold = random * cutoff_sum;
                let cumulative = program.private(ElementType::F32);
                let selected = program.private(ElementType::U32);
                program.store_local(&cumulative, f32t(0.0));
                program.store_local(&selected, first_id);
                program.fold_state(
                    u32t(0),
                    |program, index| index.ge(program.load_local(&cutoff)),
                    |program, index| {
                        let weight = program.load_workgroup(&weights, index.clone());
                        let next = program.load_local(&cumulative) + weight.clone();
                        program.if_then(next.clone().ge(threshold.clone()), |program| {
                            let id = program.load_workgroup(&scratch_ids, index.clone());
                            program.store_local(&selected, id);
                            program.break_loop();
                        });
                        program.store_local(&cumulative, next);
                        index + u32t(1)
                    },
                );
                let selected = program.load_local(&selected);
                store_sample_result(program, &output, GPU_SAMPLE_STATUS_SAMPLED, selected);
            });
        },
    );
    Some(())
}

fn emit_sampler_guards(
    program: &mut TileBlock<'_>,
    lane: &Tile,
    exactness_flag: Option<&Storage>,
    ids: &Storage,
    output: &Storage,
    meta: SamplerMeta,
) {
    if let Some(exactness_flag) = exactness_flag {
        let flag = program.load(exactness_flag.at(0), Mask::all(), TileLiteral::U32(0));
        let retry = flag.eq(u32t(0));
        program.if_then(retry, |program| {
            program.if_then(first_lane(lane), |program| {
                store_sample_result(program, output, GPU_SAMPLE_STATUS_RETRY_NEEDED, u32t(0));
            });
            program.return_();
        });
    }

    let top_id = sampler_top_id(program, ids, meta, u32t(0));
    let invalid = top_id.eq(u32t(u32::MAX));
    program.if_then(invalid, |program| {
        program.if_then(first_lane(lane), |program| {
            store_sample_result(program, output, GPU_SAMPLE_STATUS_INVALID, u32t(0));
        });
        program.return_();
    });
}

fn sum_top_weights(
    program: &mut TileBlock<'_>,
    lane: Tile,
    values: &Storage,
    meta: SamplerMeta,
    max_value: Tile,
    min_weight: Option<Tile>,
) -> Tile {
    let local_sum = program.private(ElementType::F32);
    program.store_local(&local_sum, f32t(0.0));
    program.fold_state(
        lane,
        |_, index| index.ge(u32t(meta.top_k)),
        |program, index| {
            let weight =
                sampler_top_weight(program, values, meta, max_value.clone(), index.clone());
            let passes = match &min_weight {
                Some(min_weight) => weight.ge(min_weight.clone()),
                None => Tile::bool(true),
            };
            program.if_then(passes, |program| {
                let current = program.load_local(&local_sum);
                program.store_local(&local_sum, current + weight);
            });
            index + u32t(TOP_K_BLOCK)
        },
    );

    let total = program.reduce_sum(program.load_local(&local_sum));
    program.bind(total)
}

fn sum_prefix_weights(
    program: &mut TileBlock<'_>,
    values: &Storage,
    meta: SamplerMeta,
    max_value: Tile,
    cutoff: Tile,
) -> Tile {
    let sum = program.private(ElementType::F32);
    program.store_local(&sum, f32t(0.0));
    program.fold_state(
        u32t(0),
        |_, index| index.ge(cutoff),
        |program, index| {
            let weight =
                sampler_top_weight(program, values, meta, max_value.clone(), index.clone());
            let current = program.load_local(&sum);
            program.store_local(&sum, current + weight);
            index + u32t(1)
        },
    );
    program.load_local(&sum)
}

struct WeightedPickInputs<'a> {
    ids: &'a Storage,
    values: &'a Storage,
    meta: SamplerMeta,
    max_value: Tile,
}

fn weighted_pick(
    program: &mut TileBlock<'_>,
    inputs: WeightedPickInputs<'_>,
    cutoff: Tile,
    total: Tile,
    random: Tile,
) -> (Tile, Tile) {
    let cumulative = program.private(ElementType::F32);
    let selected = program.private(ElementType::U32);
    let selected_probability = program.private(ElementType::F32);
    let threshold = random * total.clone();

    let first_token = sampler_top_id(program, inputs.ids, inputs.meta, u32t(0));
    let first_weight = sampler_top_weight(
        program,
        inputs.values,
        inputs.meta,
        inputs.max_value.clone(),
        u32t(0),
    );
    program.store_local(&selected, first_token);
    program.store_local(&selected_probability, first_weight / total.clone());
    program.store_local(&cumulative, f32t(0.0));
    program.fold_state(
        u32t(0),
        |_, index| index.ge(cutoff),
        |program, index| {
            let weight = sampler_top_weight(
                program,
                inputs.values,
                inputs.meta,
                inputs.max_value.clone(),
                index.clone(),
            );
            let cumulative_value = program.load_local(&cumulative) + weight.clone();
            let picked = cumulative_value.clone().ge(threshold.clone());
            program.if_then(picked, |program| {
                let token = sampler_top_id(program, inputs.ids, inputs.meta, index.clone());
                program.store_local(&selected, token);
                program.store_local(&selected_probability, weight / total.clone());
                program.break_loop();
            });
            program.store_local(&cumulative, cumulative_value);
            index + u32t(1)
        },
    );

    (
        program.load_local(&selected),
        program.load_local(&selected_probability),
    )
}

pub(crate) struct Mirostat2<B> {
    pub(crate) ids: KernelTensorRef<B>,
    pub(crate) values: KernelTensorRef<B>,
    pub(crate) state: KernelTensorRef<B>,
    pub(crate) params: KernelTensorRef<B>,
    pub(crate) output: KernelTensorRef<B>,
    pub(crate) exactness_flag: Option<KernelTensorRef<B>>,
    pub(crate) meta: SamplerMeta,
}

pub(crate) fn mirostat2<B>(kb: &mut KernelBuilder<B>, spec: Mirostat2<B>) -> Option<()> {
    let Mirostat2 {
        ids,
        values,
        state,
        params,
        output,
        exactness_flag,
        meta,
    } = spec;
    if meta.top_k == 0 {
        return None;
    }
    if meta.has_exactness_flag != exactness_flag.is_some() {
        return None;
    }

    let ids = kb.read(ElementType::U32, ids);
    let values = kb.read(ElementType::F32, values);
    let state = kb.write(ElementType::F32, state);
    let params = kb.read(ElementType::F32, params);
    let output = kb.write(ElementType::U32, output);
    let exactness_flag = exactness_flag.map(|tensor| kb.read(ElementType::U32, tensor));
    let phase = kb.program();

    emit_row_grid(
        phase,
        RowDispatchSpec::single(TOP_K_BLOCK),
        |program, ctx| {
            let lane = ctx.lane;
            let cutoff = program.private(ElementType::U32);

            emit_sampler_guards(program, &lane, exactness_flag.as_ref(), &ids, &output, meta);

            let max_value = sampler_top_value(program, &values, meta, u32t(0));
            let total_sum = sum_top_weights(
                program,
                lane.clone(),
                &values,
                meta,
                max_value.clone(),
                None,
            );

            let first_lane = first_lane(&lane);
            program.if_else(
                first_lane,
                |program| {
                    let epsilon = f32t(1.0e-20);
                    let total = total_sum.max(epsilon.clone());
                    let mu = program.load(state.at(0), Mask::all(), TileLiteral::f32(0.0));
                    program.store_local(&cutoff, u32t(meta.top_k));
                    program.fold_state(
                        u32t(0),
                        |_, scan| scan.ge(u32t(meta.top_k)),
                        |program, scan| {
                            let weight = sampler_top_weight(
                                program,
                                &values,
                                meta,
                                max_value.clone(),
                                scan.clone(),
                            );
                            let probability = (weight / total.clone()).max(epsilon.clone());
                            let surprise = probability.unary(TileUnaryOp::Log2) * f32t(-1.0);
                            let too_surprising = surprise.gt(mu.clone());
                            program.if_then(too_surprising, |program| {
                                let cutoff_value =
                                    Tile::select(scan.clone().gt(u32t(1)), scan.clone(), u32t(1));
                                program.store_local(&cutoff, cutoff_value);
                                program.break_loop();
                            });
                            scan + u32t(1)
                        },
                    );

                    let cutoff_value = program.load_local(&cutoff);
                    let cutoff_sum_value = sum_prefix_weights(
                        program,
                        &values,
                        meta,
                        max_value.clone(),
                        cutoff_value.clone(),
                    )
                    .max(epsilon.clone());
                    let random = load_param_f32(program, &params, 2);
                    let (token, probability) = weighted_pick(
                        program,
                        WeightedPickInputs {
                            ids: &ids,
                            values: &values,
                            meta,
                            max_value: max_value.clone(),
                        },
                        cutoff_value,
                        cutoff_sum_value,
                        random,
                    );

                    let selected_probability_value = probability.max(epsilon);
                    let surprise = selected_probability_value.unary(TileUnaryOp::Log2) * f32t(-1.0);
                    let tau = load_param_f32(program, &params, 0);
                    let eta = load_param_f32(program, &params, 1);
                    let error = surprise - tau;
                    let correction = eta * error;
                    let next_mu = mu - correction;
                    program.store(state.at(0), next_mu, Mask::all());
                    store_sample_result(program, &output, GPU_SAMPLE_STATUS_SAMPLED, token);
                },
                |program| {
                    program.return_();
                },
            );
        },
    );
    Some(())
}

pub(crate) struct StandardSampler<B> {
    pub(crate) ids: KernelTensorRef<B>,
    pub(crate) values: KernelTensorRef<B>,
    pub(crate) params: KernelTensorRef<B>,
    pub(crate) output: KernelTensorRef<B>,
    pub(crate) exactness_flag: Option<KernelTensorRef<B>>,
    pub(crate) meta: SamplerMeta,
}

pub(crate) fn standard_sampler<B>(
    kb: &mut KernelBuilder<B>,
    spec: StandardSampler<B>,
) -> Option<()> {
    let StandardSampler {
        ids,
        values,
        params,
        output,
        exactness_flag,
        meta,
    } = spec;
    if meta.top_k == 0 {
        return None;
    }
    if meta.has_exactness_flag != exactness_flag.is_some() {
        return None;
    }

    let ids = kb.read(ElementType::U32, ids);
    let values = kb.read(ElementType::F32, values);
    let params = kb.read(ElementType::F32, params);
    let output = kb.write(ElementType::U32, output);
    let exactness_flag = exactness_flag.map(|tensor| kb.read(ElementType::U32, tensor));
    let phase = kb.program();

    emit_row_grid(
        phase,
        RowDispatchSpec::single(TOP_K_BLOCK),
        |program, ctx| {
            let lane = ctx.lane;
            let cutoff = program.private(ElementType::U32);
            let cutoff_sum = program.private(ElementType::F32);

            emit_sampler_guards(program, &lane, exactness_flag.as_ref(), &ids, &output, meta);

            let max_value = sampler_top_value(program, &values, meta, u32t(0));
            let min_p = load_param_f32(program, &params, 2);
            let filtered_sum = sum_top_weights(
                program,
                lane.clone(),
                &values,
                meta,
                max_value.clone(),
                Some(min_p.clone()),
            );

            let first_lane = first_lane(&lane);
            program.if_else(
                first_lane,
                |program| {
                    let epsilon = f32t(1.0e-20);
                    let filtered_total = filtered_sum.max(epsilon.clone());
                    let top_p = load_param_f32(program, &params, 1);
                    let target = filtered_total.clone() * top_p;
                    program.store_local(&cutoff, u32t(meta.top_k));
                    program.store_local(&cutoff_sum, f32t(0.0));
                    program.fold_state(
                        u32t(0),
                        |_, scan| scan.ge(u32t(meta.top_k)),
                        |program, scan| {
                            let weight = sampler_top_weight(
                                program,
                                &values,
                                meta,
                                max_value.clone(),
                                scan.clone(),
                            );
                            let passes_min_p = weight.clone().ge(min_p.clone());
                            program.if_else(
                                passes_min_p,
                                |program| {
                                    let current = program.load_local(&cutoff_sum);
                                    let next = current + weight;
                                    program.store_local(&cutoff_sum, next.clone());
                                    let picked = next.ge(target.clone());
                                    program.if_then(picked, |program| {
                                        program.store_local(&cutoff, scan.clone() + u32t(1));
                                        program.break_loop();
                                    });
                                },
                                |program| {
                                    program.store_local(&cutoff, scan.clone());
                                    program.break_loop();
                                },
                            );
                            scan + u32t(1)
                        },
                    );

                    let cutoff_value = program.load_local(&cutoff);
                    let no_candidates = cutoff_value.eq(u32t(0));
                    let first_weight =
                        sampler_top_weight(program, &values, meta, max_value.clone(), u32t(0));
                    let cutoff_value = Tile::select(no_candidates.clone(), u32t(1), cutoff_value);
                    let cutoff_sum =
                        Tile::select(no_candidates, first_weight, program.load_local(&cutoff_sum));
                    let cutoff_sum_value = cutoff_sum.max(epsilon.clone());
                    let random = load_param_f32(program, &params, 0);
                    let (token, _) = weighted_pick(
                        program,
                        WeightedPickInputs {
                            ids: &ids,
                            values: &values,
                            meta,
                            max_value: max_value.clone(),
                        },
                        cutoff_value,
                        cutoff_sum_value,
                        random,
                    );
                    store_sample_result(program, &output, GPU_SAMPLE_STATUS_SAMPLED, token);
                },
                |program| {
                    program.return_();
                },
            );
        },
    );
    Some(())
}
