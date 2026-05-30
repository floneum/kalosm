use fusor_tile_ir::{
    tile::{self, Mask, Tile, TileBlock},
    TileLiteral, TileUnaryOp, F32, U32,
};

use super::helpers::{index_n, NEG_MAX_F32, TOP_K_BLOCK};
use super::types::Mirostat2Meta;

const GPU_SAMPLE_STATUS_RETRY_NEEDED: u32 = 0;
const GPU_SAMPLE_STATUS_SAMPLED: u32 = 1;
const GPU_SAMPLE_STATUS_INVALID: u32 = 2;

fn mirostat_top_value(
    program: &TileBlock<'_>,
    values: &tile::Storage<F32, 1>,
    meta: Mirostat2Meta,
    index: Tile<U32>,
) -> Tile {
    let index = index_n(meta.values_offset, [meta.values_stride], index);
    program.load(values.at(index), Mask::all(), TileLiteral::f32(NEG_MAX_F32))
}

fn mirostat_top_id(
    program: &TileBlock<'_>,
    ids: &tile::Storage<U32, 1>,
    meta: Mirostat2Meta,
    index: Tile<U32>,
) -> Tile<U32> {
    let index = index_n(meta.ids_offset, [meta.ids_stride], index);
    program.load(ids.at(index), Mask::all(), TileLiteral::U32(u32::MAX))
}

fn mirostat_top_weight(
    program: &TileBlock<'_>,
    values: &tile::Storage<F32, 1>,
    meta: Mirostat2Meta,
    max_value: Tile,
    index: Tile<U32>,
) -> Tile {
    (mirostat_top_value(program, values, meta, index) - max_value).exp()
}

fn load_param_f32(program: &TileBlock<'_>, params: &tile::Storage<F32, 1>, index: u32) -> Tile {
    program.load(params.at(index), Mask::all(), TileLiteral::f32(0.0))
}

fn store_sample_result(
    program: &mut TileBlock<'_>,
    output: &tile::Storage<U32, 1>,
    status: u32,
    token: Tile<U32>,
) {
    program.store(
        output.at(0),
        Tile::literal(TileLiteral::U32(status)),
        Mask::all(),
    );
    program.store(output.at(1), token, Mask::all());
}

/// Tensor bindings and metadata for [`mirostat2`].
///
/// `ids` and `values` must reference a sorted top-k list. `state` stores the
/// mutable mirostat `mu` value, `params` stores tau/eta/random, and `output`
/// receives `[status, token]`.
///
/// ```no_run
/// # use fusor_tile_ir::{KernelBuilder, KernelTensorRef};
/// # use fusor_tile_ir_kernels::{linear_storage_layout, mirostat2, Mirostat2, Mirostat2Meta};
/// let layout = linear_storage_layout();
/// let mut kb = KernelBuilder::<()>::new();
/// mirostat2(
///     &mut kb,
///     Mirostat2 {
///         ids: KernelTensorRef::new((), layout.clone()),
///         values: KernelTensorRef::new((), layout.clone()),
///         state: KernelTensorRef::new((), layout.clone()),
///         params: KernelTensorRef::new((), layout.clone()),
///         output: KernelTensorRef::new((), layout),
///         exactness_flag: None,
///         meta: Mirostat2Meta {
///             top_k: 32,
///             ids_offset: 0,
///             ids_stride: 1,
///             values_offset: 0,
///             values_stride: 1,
///             has_exactness_flag: false,
///         },
///     },
/// );
/// ```
pub struct Mirostat2<B> {
    /// Sorted top-k token ids.
    pub ids: fusor_tile_ir::KernelTensorRef<B>,
    /// Sorted top-k logits/probability scores.
    pub values: fusor_tile_ir::KernelTensorRef<B>,
    /// Mutable sampler state containing `mu`.
    pub state: fusor_tile_ir::KernelTensorRef<B>,
    /// Sampler params: tau, eta, and random threshold.
    pub params: fusor_tile_ir::KernelTensorRef<B>,
    /// Output `[status, token]`.
    pub output: fusor_tile_ir::KernelTensorRef<B>,
    /// Optional flag proving whether the top-k set is exact.
    pub exactness_flag: Option<fusor_tile_ir::KernelTensorRef<B>>,
    /// Top-k offsets and exactness configuration.
    pub meta: Mirostat2Meta,
}

/// Build a Mirostat v2 sampling kernel over a sorted top-k candidate list.
///
/// Returns `None` for empty top-k metadata or when the optional exactness flag
/// binding does not match [`Mirostat2Meta::has_exactness_flag`].
pub fn mirostat2<B>(kb: &mut fusor_tile_ir::KernelBuilder<B>, spec: Mirostat2<B>) -> Option<()> {
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

    let ids = kb.read::<U32, 1>(ids);
    let values = kb.read::<F32, 1>(values);
    let state = kb.write::<F32, 1>(state);
    let params = kb.read::<F32, 1>(params);
    let output = kb.write::<U32, 1>(output);
    let exactness_flag = exactness_flag.map(|tensor| kb.read::<U32, 1>(tensor));
    let phase = kb.program();
    let scratch = phase.alloc_workgroup_array::<F32>(TOP_K_BLOCK as u32);

    phase.program_grid::<TOP_K_BLOCK>([1, 1, 1], |program| {
        let lane = program.lane();
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
            let flag = program.load(exactness_flag.at(0), Mask::all(), TileLiteral::U32(0));
            let retry = flag.eq(Tile::literal(TileLiteral::U32(0)));
            program.if_then(retry, |program| {
                let first_lane = program
                    .index(lane.clone())
                    .eq(Tile::literal(TileLiteral::U32(0)));
                program.if_then(first_lane, |program| {
                    store_sample_result(
                        program,
                        &output,
                        GPU_SAMPLE_STATUS_RETRY_NEEDED,
                        Tile::literal(TileLiteral::U32(0)),
                    );
                });
                program.return_();
            });
        }

        let top_id = mirostat_top_id(program, &ids, meta, Tile::literal(TileLiteral::U32(0)));
        let invalid = top_id.eq(Tile::literal(TileLiteral::U32(u32::MAX)));
        program.if_then(invalid, |program| {
            let first_lane = program
                .index(lane.clone())
                .eq(Tile::literal(TileLiteral::U32(0)));
            program.if_then(first_lane, |program| {
                store_sample_result(
                    program,
                    &output,
                    GPU_SAMPLE_STATUS_INVALID,
                    Tile::literal(TileLiteral::U32(0)),
                );
            });
            program.return_();
        });

        let max_value =
            mirostat_top_value(program, &values, meta, Tile::literal(TileLiteral::U32(0)));
        program.store_local(&local_sum, Tile::literal(TileLiteral::f32(0.0)));
        program.store_local(&index, program.index(lane.clone()));
        program.loop_forever(|program| {
            let index_value = program.load_local(&index);
            program.break_if(
                index_value
                    .clone()
                    .ge(Tile::literal(TileLiteral::U32(meta.top_k))),
            );
            let weight =
                mirostat_top_weight(program, &values, meta, max_value.clone(), index_value);
            let current = program.load_local(&local_sum);
            program.store_local(&local_sum, current + weight);
            let index_value = program.load_local(&index);
            program.store_local(
                &index,
                index_value + Tile::literal(TileLiteral::U32(TOP_K_BLOCK as u32)),
            );
        });

        let local_sum_value = program.load_local(&local_sum);
        program.store_workgroup(scratch, lane.clone(), local_sum_value);
        program.workgroup_barrier();

        program.store_local(
            &reduce_step,
            Tile::literal(TileLiteral::U32(TOP_K_BLOCK as u32 / 2)),
        );
        program.loop_forever(|program| {
            let step = program.load_local(&reduce_step);
            program.break_if(step.clone().eq(Tile::literal(TileLiteral::U32(0))));
            let participates = program.index(lane.clone()).lt(step.clone());
            program.if_then(participates, |program| {
                let rhs_index = program.index(lane.clone()) + step.clone();
                let lhs = program.load_workgroup(scratch, lane.clone());
                let rhs = program.load_workgroup(scratch, rhs_index);
                program.store_workgroup(scratch, lane.clone(), lhs + rhs);
            });
            program.workgroup_barrier();
            let step = program.load_local(&reduce_step);
            program.store_local(&reduce_step, step / Tile::literal(TileLiteral::U32(2)));
        });

        let first_lane = program
            .index(lane.clone())
            .eq(Tile::literal(TileLiteral::U32(0)));
        program.if_else(
            first_lane,
            |program| {
                let epsilon = Tile::literal(TileLiteral::f32(1.0e-20));
                let total = program.load_workgroup(scratch, 0).max(epsilon.clone());
                let mu = program.load(state.at(0), Mask::all(), TileLiteral::f32(0.0));
                program.store_local(&cutoff, Tile::literal(TileLiteral::U32(0)));
                program.store_local(&scan, Tile::literal(TileLiteral::U32(0)));
                program.loop_forever(|program| {
                    let scan_value = program.load_local(&scan);
                    let done = scan_value
                        .clone()
                        .ge(Tile::literal(TileLiteral::U32(meta.top_k)));
                    program.if_then(done, |program| {
                        program.store_local(&cutoff, Tile::literal(TileLiteral::U32(1)));
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
                    let surprise = probability.unary(TileUnaryOp::Log2)
                        * Tile::literal(TileLiteral::f32(-1.0));
                    let too_surprising = surprise.gt(mu.clone());
                    program.if_then(too_surprising, |program| {
                        let scan_value = program.load_local(&scan);
                        let scan_gt_one = scan_value.clone().gt(Tile::literal(TileLiteral::U32(1)));
                        program.if_else(
                            scan_gt_one,
                            |program| {
                                let scan_value = program.load_local(&scan);
                                program.store_local(&cutoff, scan_value);
                            },
                            |program| {
                                program.store_local(&cutoff, Tile::literal(TileLiteral::U32(1)));
                            },
                        );
                        program.break_loop();
                    });
                    let scan_value = program.load_local(&scan);
                    program.store_local(&scan, scan_value + Tile::literal(TileLiteral::U32(1)));
                });

                program.store_local(&cutoff_sum, Tile::literal(TileLiteral::f32(0.0)));
                program.store_local(&scan, Tile::literal(TileLiteral::U32(0)));
                program.loop_forever(|program| {
                    let scan_value = program.load_local(&scan);
                    let cutoff_value = program.load_local(&cutoff);
                    program.break_if(scan_value.clone().ge(cutoff_value));
                    let scan_value = program.load_local(&scan);
                    let weight =
                        mirostat_top_weight(program, &values, meta, max_value.clone(), scan_value);
                    let current = program.load_local(&cutoff_sum);
                    program.store_local(&cutoff_sum, current + weight);
                    let scan_value = program.load_local(&scan);
                    program.store_local(&scan, scan_value + Tile::literal(TileLiteral::U32(1)));
                });
                let cutoff_sum_value = program.load_local(&cutoff_sum).max(epsilon.clone());
                program.store_local(&cutoff_sum, cutoff_sum_value.clone());

                let random = load_param_f32(program, &params, 2);
                let threshold = random * cutoff_sum_value.clone();
                program.store_local(&cumulative, Tile::literal(TileLiteral::f32(0.0)));
                let selected_token =
                    mirostat_top_id(program, &ids, meta, Tile::literal(TileLiteral::U32(0)));
                program.store_local(&selected, selected_token);
                let selected_weight = mirostat_top_weight(
                    program,
                    &values,
                    meta,
                    max_value.clone(),
                    Tile::literal(TileLiteral::U32(0)),
                );
                let selected_prob = selected_weight / cutoff_sum_value.clone();
                program.store_local(&selected_probability, selected_prob);
                program.store_local(&scan, Tile::literal(TileLiteral::U32(0)));
                program.loop_forever(|program| {
                    let scan_value = program.load_local(&scan);
                    let cutoff_value = program.load_local(&cutoff);
                    program.break_if(scan_value.clone().ge(cutoff_value));
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
                        program
                            .store_local(&selected_probability, weight / cutoff_sum_value.clone());
                        program.break_loop();
                    });
                    program.store_local(&cumulative, cumulative_value);
                    let scan_value = program.load_local(&scan);
                    program.store_local(&scan, scan_value + Tile::literal(TileLiteral::U32(1)));
                });

                let selected_probability_value =
                    program.load_local(&selected_probability).max(epsilon);
                let surprise = selected_probability_value.unary(TileUnaryOp::Log2)
                    * Tile::literal(TileLiteral::f32(-1.0));
                let tau = load_param_f32(program, &params, 0);
                let eta = load_param_f32(program, &params, 1);
                let error = surprise - tau;
                let correction = eta * error;
                let next_mu = mu - correction;
                program.store(state.at(0), next_mu, Mask::all());
                let selected_token = program.load_local(&selected);
                store_sample_result(program, &output, GPU_SAMPLE_STATUS_SAMPLED, selected_token);
            },
            |program| {
                program.return_();
            },
        );
    });
    Some(())
}
