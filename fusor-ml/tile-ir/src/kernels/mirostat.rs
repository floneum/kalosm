use crate::{
    tile::{self, Tile, TileBlock},
    F32Bits, TileLiteral, TileUnaryOp, F32, U32,
};

use super::helpers::{all, f32_tile, u32_tile, NEG_MAX_F32, TOP_K_BLOCK};
use super::top_k::index1;
use super::types::Mirostat2Meta;

const GPU_SAMPLE_STATUS_RETRY_NEEDED: u32 = 0;
const GPU_SAMPLE_STATUS_SAMPLED: u32 = 1;
const GPU_SAMPLE_STATUS_INVALID: u32 = 2;

fn mirostat_top_value(
    program: &TileBlock<'_, TOP_K_BLOCK>,
    values: &tile::Storage<F32, 1>,
    meta: Mirostat2Meta,
    index: Tile<TOP_K_BLOCK>,
) -> Tile<TOP_K_BLOCK> {
    let index = index1(index, meta.values_offset, meta.values_stride);
    program.load_linear(
        values.at(index),
        all(),
        TileLiteral::F32(F32Bits::new(NEG_MAX_F32)),
    )
}

fn mirostat_top_id(
    program: &TileBlock<'_, TOP_K_BLOCK>,
    ids: &tile::Storage<U32, 1>,
    meta: Mirostat2Meta,
    index: Tile<TOP_K_BLOCK>,
) -> Tile<TOP_K_BLOCK> {
    let index = index1(index, meta.ids_offset, meta.ids_stride);
    program.load_linear(ids.at(index), all(), TileLiteral::U32(u32::MAX))
}

fn mirostat_top_weight(
    program: &TileBlock<'_, TOP_K_BLOCK>,
    values: &tile::Storage<F32, 1>,
    meta: Mirostat2Meta,
    max_value: Tile<TOP_K_BLOCK>,
    index: Tile<TOP_K_BLOCK>,
) -> Tile<TOP_K_BLOCK> {
    (mirostat_top_value(program, values, meta, index) - max_value).exp()
}

fn load_param_f32(
    program: &TileBlock<'_, TOP_K_BLOCK>,
    params: &tile::Storage<F32, 1>,
    index: u32,
) -> Tile<TOP_K_BLOCK> {
    program.load_linear(params.at(index), all(), TileLiteral::F32(F32Bits::new(0.0)))
}

fn store_sample_result(
    program: &mut TileBlock<'_, TOP_K_BLOCK>,
    output: &tile::Storage<U32, 1>,
    status: u32,
    token: Tile<TOP_K_BLOCK>,
) {
    program.store_linear(output.at(0), u32_tile(status), all());
    program.store_linear(output.at(1), token, all());
}

pub fn mirostat2<B>(
    kb: &mut crate::kernel_builder::KernelBuilder<B>,
    ids: crate::kernel_builder::KernelTensorRef<B>,
    values: crate::kernel_builder::KernelTensorRef<B>,
    state: crate::kernel_builder::KernelTensorRef<B>,
    params: crate::kernel_builder::KernelTensorRef<B>,
    output: crate::kernel_builder::KernelTensorRef<B>,
    exactness_flag: Option<crate::kernel_builder::KernelTensorRef<B>>,
    meta: Mirostat2Meta,
) -> Option<()> {
    if meta.top_k == 0 {
        return None;
    }
    if meta.has_exactness_flag != exactness_flag.is_some() {
        return None;
    }

    let ids = kb.read::<U32, 1>(ids);
    let values = kb.read::<F32, 1>(values);
    let state = kb.write::<F32, 1>(state);
    let params = kb.read::<F32, 1>(params);
    let output = kb.write::<U32, 1>(output);
    let exactness_flag = exactness_flag.map(|tensor| kb.read::<U32, 1>(tensor));
    let phase = kb.program();
    let scratch = phase.alloc_workgroup_array::<F32>(TOP_K_BLOCK as u32);

    phase.program_grid::<TOP_K_BLOCK>([1, 1, 1], |program| {
            let lane = program.arange();
            let index = program.private::<U32>();
            let local_sum = program.private::<F32>();
            let reduce_step = program.private::<U32>();
            let cutoff = program.private::<U32>();
            let scan = program.private::<U32>();
            let cutoff_sum = program.private::<F32>();
            let cumulative = program.private::<F32>();
            let selected = program.private::<U32>();
            let selected_probability = program.private::<F32>();

            if let Some(exactness_flag) = &exactness_flag {
                let flag = program.load_linear(exactness_flag.at(0), all(), TileLiteral::U32(0));
                let retry = flag.eq(u32_tile(0));
                program.if_then(retry, |program| {
                    let lane_zero = program.index(lane.clone()).eq(u32_tile(0));
                    program.if_then(lane_zero, |program| {
                        store_sample_result(
                            program,
                            &output,
                            GPU_SAMPLE_STATUS_RETRY_NEEDED,
                            u32_tile(0),
                        );
                    });
                    program.return_();
                });
            }

            let top_id = mirostat_top_id(program, &ids, meta, u32_tile(0));
            let invalid = top_id.eq(u32_tile(u32::MAX));
            program.if_then(invalid, |program| {
                let lane_zero = program.index(lane.clone()).eq(u32_tile(0));
                program.if_then(lane_zero, |program| {
                    store_sample_result(program, &output, GPU_SAMPLE_STATUS_INVALID, u32_tile(0));
                });
                program.return_();
            });

            let max_value = mirostat_top_value(program, &values, meta, u32_tile(0));
            program.store_local(&local_sum, f32_tile(0.0));
            program.store_local(&index, program.index(lane.clone()));
            program.loop_forever(|program| {
                let index_value = program.load_local(&index);
                program.if_then(index_value.clone().ge(u32_tile(meta.top_k)), |program| {
                    program.break_loop();
                });
                let weight =
                    mirostat_top_weight(program, &values, meta, max_value.clone(), index_value);
                let current = program.load_local(&local_sum);
                program.store_local(&local_sum, current + weight);
                let index_value = program.load_local(&index);
                program.store_local(&index, index_value + u32_tile(TOP_K_BLOCK as u32));
            });

            let local_sum_value = program.load_local(&local_sum);
            program.store_workgroup(scratch, lane.clone(), local_sum_value);
            program.workgroup_barrier();

            program.store_local(&reduce_step, u32_tile(TOP_K_BLOCK as u32 / 2));
            program.loop_forever(|program| {
                let step = program.load_local(&reduce_step);
                program.if_then(step.clone().eq(u32_tile(0)), |program| {
                    program.break_loop();
                });
                let participates = program.index(lane.clone()).lt(step.clone());
                program.if_then(participates, |program| {
                    let rhs_index = program.index(lane.clone()) + step.clone();
                    let lhs = program.load_workgroup(scratch, lane.clone());
                    let rhs = program.load_workgroup(scratch, rhs_index);
                    program.store_workgroup(scratch, lane.clone(), lhs + rhs);
                });
                program.workgroup_barrier();
                let step = program.load_local(&reduce_step);
                program.store_local(&reduce_step, step / u32_tile(2));
            });

            let lane_zero = program.index(lane.clone()).eq(u32_tile(0));
            program.if_else(
                lane_zero,
                |program| {
                    let epsilon = f32_tile(1.0e-20);
                    let total = program.load_workgroup(scratch, 0).max(epsilon.clone());
                    let mu = program.load_linear(
                        state.at(0),
                        all(),
                        TileLiteral::F32(F32Bits::new(0.0)),
                    );
                    program.store_local(&cutoff, u32_tile(0));
                    program.store_local(&scan, u32_tile(0));
                    program.loop_forever(|program| {
                        let scan_value = program.load_local(&scan);
                        let done = scan_value.clone().ge(u32_tile(meta.top_k));
                        program.if_then(done, |program| {
                            program.store_local(&cutoff, u32_tile(1));
                            program.break_loop();
                        });
                        let scan_value = program.load_local(&scan);
                        let weight = mirostat_top_weight(
                            program,
                            &values,
                            meta,
                            max_value.clone(),
                            scan_value.clone(),
                        );
                        let probability = (weight / total.clone()).max(epsilon.clone());
                        let surprise = probability.unary(TileUnaryOp::Log2) * f32_tile(-1.0);
                        let too_surprising = surprise.gt(mu.clone());
                        program.if_then(too_surprising, |program| {
                            let scan_value = program.load_local(&scan);
                            let scan_gt_one = scan_value.clone().gt(u32_tile(1));
                            program.if_else(
                                scan_gt_one,
                                |program| {
                                    let scan_value = program.load_local(&scan);
                                    program.store_local(&cutoff, scan_value);
                                },
                                |program| {
                                    program.store_local(&cutoff, u32_tile(1));
                                },
                            );
                            program.break_loop();
                        });
                        let scan_value = program.load_local(&scan);
                        program.store_local(&scan, scan_value + u32_tile(1));
                    });

                    program.store_local(&cutoff_sum, f32_tile(0.0));
                    program.store_local(&scan, u32_tile(0));
                    program.loop_forever(|program| {
                        let scan_value = program.load_local(&scan);
                        let cutoff_value = program.load_local(&cutoff);
                        program.if_then(scan_value.clone().ge(cutoff_value), |program| {
                            program.break_loop();
                        });
                        let scan_value = program.load_local(&scan);
                        let weight = mirostat_top_weight(
                            program,
                            &values,
                            meta,
                            max_value.clone(),
                            scan_value,
                        );
                        let current = program.load_local(&cutoff_sum);
                        program.store_local(&cutoff_sum, current + weight);
                        let scan_value = program.load_local(&scan);
                        program.store_local(&scan, scan_value + u32_tile(1));
                    });
                    let cutoff_sum_value = program.load_local(&cutoff_sum).max(epsilon.clone());
                    program.store_local(&cutoff_sum, cutoff_sum_value.clone());

                    let random = load_param_f32(program, &params, 2);
                    let threshold = random * cutoff_sum_value.clone();
                    program.store_local(&cumulative, f32_tile(0.0));
                    let selected_token = mirostat_top_id(program, &ids, meta, u32_tile(0));
                    program.store_local(&selected, selected_token);
                    let selected_weight =
                        mirostat_top_weight(program, &values, meta, max_value.clone(), u32_tile(0));
                    let selected_prob = selected_weight / cutoff_sum_value.clone();
                    program.store_local(&selected_probability, selected_prob);
                    program.store_local(&scan, u32_tile(0));
                    program.loop_forever(|program| {
                        let scan_value = program.load_local(&scan);
                        let cutoff_value = program.load_local(&cutoff);
                        program.if_then(scan_value.clone().ge(cutoff_value), |program| {
                            program.break_loop();
                        });
                        let scan_value = program.load_local(&scan);
                        let weight = mirostat_top_weight(
                            program,
                            &values,
                            meta,
                            max_value.clone(),
                            scan_value.clone(),
                        );
                        let cumulative_value = program.load_local(&cumulative) + weight.clone();
                        let picked = cumulative_value.clone().ge(threshold.clone());
                        program.if_then(picked, |program| {
                            let scan_value = program.load_local(&scan);
                            let token = mirostat_top_id(program, &ids, meta, scan_value);
                            program.store_local(&selected, token);
                            program.store_local(
                                &selected_probability,
                                weight / cutoff_sum_value.clone(),
                            );
                            program.break_loop();
                        });
                        program.store_local(&cumulative, cumulative_value);
                        let scan_value = program.load_local(&scan);
                        program.store_local(&scan, scan_value + u32_tile(1));
                    });

                    let selected_probability_value =
                        program.load_local(&selected_probability).max(epsilon);
                    let surprise =
                        selected_probability_value.unary(TileUnaryOp::Log2) * f32_tile(-1.0);
                    let tau = load_param_f32(program, &params, 0);
                    let eta = load_param_f32(program, &params, 1);
                    let error = surprise - tau;
                    let correction = eta * error;
                    let next_mu = mu - correction;
                    program.store_linear(state.at(0), next_mu, all());
                    let selected_token = program.load_local(&selected);
                    store_sample_result(
                        program,
                        &output,
                        GPU_SAMPLE_STATUS_SAMPLED,
                        selected_token,
                    );
                },
                |program| {
                    program.return_();
                },
            );
        });
    Some(())
}
