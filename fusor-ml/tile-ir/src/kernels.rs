use crate::{
    tile::{self, Mask, Tile, TileBlock},
    Bool, ElementType, F32Bits, KernelIr, Layout, MemoryLevel, Shape, Strides, TileLiteral,
    TileReduceOp, TileUnaryOp, WorkgroupAxis, F32, U32,
};

const RMS_NORM_VEC4_BLOCK: usize = 128;
const TOP_K_BLOCK: usize = 256;
const TOP_K_CHUNK: u32 = TOP_K_BLOCK as u32;
const FLASH_BLOCK: usize = 256;
const FLASH_SIMD_WIDTH: u32 = 32;
const FLASH_OUTPUTS_PER_WORKGROUP: u32 = FLASH_BLOCK as u32 / FLASH_SIMD_WIDTH;
const DECODE_HEAD_DIM: u32 = 128;
const MAX_F32: f32 = f32::MAX;
const NEG_MAX_F32: f32 = -f32::MAX;
const GPU_SAMPLE_STATUS_RETRY_NEEDED: u32 = 0;
const GPU_SAMPLE_STATUS_SAMPLED: u32 = 1;
const GPU_SAMPLE_STATUS_INVALID: u32 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlashAttentionDims {
    pub batch: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub q_seq_len: u32,
    pub kv_seq_len: u32,
    pub head_dim: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorMeta {
    pub strides: Vec<u32>,
    pub offset: u32,
}

impl TensorMeta {
    pub fn new(strides: Vec<u32>, offset: u32) -> Self {
        Self { strides, offset }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlashAttentionMeta {
    pub dims: FlashAttentionDims,
    pub scale: F32Bits,
    pub q_meta: TensorMeta,
    pub k_meta: TensorMeta,
    pub v_meta: TensorMeta,
    pub mask_meta: Option<TensorMeta>,
    pub output_meta: TensorMeta,
    pub dispatch_size: [u32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlashDecodeSmallMeta {
    pub dims: FlashAttentionDims,
    pub scale: F32Bits,
    pub active_kv_len: u32,
    pub decode_block: u32,
    pub tiled: bool,
    pub groups: u32,
    pub q_offset: u32,
    pub k_offset: u32,
    pub v_offset: u32,
    pub output_offset: u32,
    pub q_strides: [u32; 4],
    pub k_strides: [u32; 4],
    pub v_strides: [u32; 4],
    pub output_strides: [u32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RmsNormVec4Meta {
    pub cols: u32,
    pub cols_vec: u32,
    pub eps: F32Bits,
    pub input_offset_vec: u32,
    pub input_row_stride_vec: u32,
    pub residual_offset_vec: Option<u32>,
    pub residual_row_stride_vec: u32,
    pub weight_offset_vec: u32,
    pub bias_offset_vec: Option<u32>,
    pub output_offset_vec: u32,
    pub output_row_stride_vec: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TopKChunkMeta {
    pub input_len: u32,
    pub output_per_chunk: u32,
    pub input_offset: u32,
    pub input_stride: u32,
    pub processors: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TopKExactnessMeta {
    pub chunks: u32,
    pub candidate_count: u32,
    pub output_per_chunk: u32,
    pub top_k: u32,
    pub top_values_offset: u32,
    pub top_values_stride: u32,
    pub chunk_values_offset: u32,
    pub chunk_values_stride: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MergeTopKMeta {
    pub chunks: u32,
    pub chunk_len: u32,
    pub chunk_stride: u32,
    pub input_len: u32,
    pub k: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mirostat2Meta {
    pub top_k: u32,
    pub ids_offset: u32,
    pub ids_stride: u32,
    pub values_offset: u32,
    pub values_stride: u32,
    pub has_exactness_flag: bool,
}

pub fn rms_norm_vec4(meta: RmsNormVec4Meta, rows: u32) -> Option<KernelIr> {
    if rows == 0 || meta.cols == 0 || meta.cols_vec == 0 {
        return None;
    }

    let vec_layout = Layout::strided(MemoryLevel::Storage, Shape::new([1]), Strides::new([1]));
    let chunks = meta.cols_vec.div_ceil(RMS_NORM_VEC4_BLOCK as u32);
    let eps = meta.eps.get();

    Some(tile::build(move |phase| {
        let input = phase.storage_read_with_layout_offset::<crate::F32Vec4, 1>(
            vec_layout.clone(),
            meta.input_offset_vec,
        );
        let residual = meta.residual_offset_vec.map(|offset| {
            phase.storage_read_with_layout_offset::<crate::F32Vec4, 1>(vec_layout.clone(), offset)
        });
        let weight = phase.storage_read_with_layout_offset::<crate::F32Vec4, 1>(
            vec_layout.clone(),
            meta.weight_offset_vec,
        );
        let bias = meta.bias_offset_vec.map(|offset| {
            phase.storage_read_with_layout_offset::<crate::F32Vec4, 1>(vec_layout.clone(), offset)
        });
        let output = phase.storage_write_with_layout_offset::<crate::F32Vec4, 1>(
            vec_layout,
            meta.output_offset_vec,
        );

        phase.program_grid::<RMS_NORM_VEC4_BLOCK>([rows, 1, 1], |program| {
            let row = program.program_id(WorkgroupAxis::X);
            let lane = program.arange();
            let reduce_col = program.loop_index() * RMS_NORM_VEC4_BLOCK as u32 + lane.clone();
            let reduce_mask = reduce_col.lt(meta.cols_vec);
            let input_index = row.clone() * meta.input_row_stride_vec + reduce_col.clone();
            let mut value = program.load_vec4(input.at(input_index), reduce_mask.clone(), 0.0);
            if let Some(residual) = &residual {
                let residual_index =
                    row.clone() * meta.residual_row_stride_vec + reduce_col.clone();
                value = value + program.load_vec4(residual.at(residual_index), reduce_mask, 0.0);
            }
            let dot = program.vec4_dot(value.clone(), value);
            let partial_sum = program.loop_fold(
                TileReduceOp::Sum,
                chunks,
                dot,
                TileLiteral::F32(F32Bits::new(0.0)),
            );
            let total_sum = program.group_reduce_sum::<RMS_NORM_VEC4_BLOCK>(partial_sum);
            let mean = total_sum
                / Tile::<RMS_NORM_VEC4_BLOCK>::literal(TileLiteral::F32(F32Bits::new(
                    meta.cols as f32,
                )));
            let scale = (mean
                + Tile::<RMS_NORM_VEC4_BLOCK>::literal(TileLiteral::F32(F32Bits::new(eps))))
            .inverse_sqrt();
            let scale = program.pin(scale);

            for chunk in 0..chunks {
                let col = lane.clone() + chunk * RMS_NORM_VEC4_BLOCK as u32;
                let mask = col.lt(meta.cols_vec);
                let input_index = row.clone() * meta.input_row_stride_vec + col.clone();
                let mut value = program.load_vec4(input.at(input_index), mask.clone(), 0.0);
                if let Some(residual) = &residual {
                    let residual_index = row.clone() * meta.residual_row_stride_vec + col.clone();
                    value =
                        value + program.load_vec4(residual.at(residual_index), mask.clone(), 0.0);
                }
                let scale = program.vec4_splat(scale.get());
                let weight = program.load_vec4(weight.at(col.clone()), mask.clone(), 0.0);
                let mut normalized = value * scale * weight;
                if let Some(bias) = &bias {
                    normalized =
                        normalized + program.load_vec4(bias.at(col.clone()), mask.clone(), 0.0);
                }
                let output_index = row.clone() * meta.output_row_stride_vec + col;
                program.store_vec4(output.at(output_index), normalized, mask);
            }
        });
    }))
}

fn linear_storage_layout() -> Layout {
    Layout::strided(MemoryLevel::Storage, Shape::new([1]), Strides::new([1]))
}

fn all<const BLOCK: usize>() -> Mask<BLOCK> {
    Mask::all()
}

fn f32_tile<const BLOCK: usize>(value: f32) -> Tile<BLOCK> {
    Tile::literal(TileLiteral::F32(F32Bits::new(value)))
}

fn u32_tile<const BLOCK: usize>(value: u32) -> Tile<BLOCK> {
    Tile::literal(TileLiteral::U32(value))
}

fn index1<const BLOCK: usize>(index: Tile<BLOCK>, offset: u32, stride: u32) -> Tile<BLOCK> {
    let scaled = if stride == 1 {
        index
    } else {
        index * u32_tile(stride)
    };
    scaled + u32_tile(offset)
}

fn is_finite<const BLOCK: usize>(value: Tile<BLOCK>) -> Tile<BLOCK> {
    let self_equal = value.clone().eq(value.clone());
    let finite_magnitude = value.unary(TileUnaryOp::Abs).le(f32_tile(MAX_F32));
    self_equal.and(finite_magnitude)
}

fn better_candidate<const BLOCK: usize>(
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

fn load_processor_param_f32(
    program: &TileBlock<'_, TOP_K_BLOCK>,
    params: &tile::Storage<U32, 1>,
    index: u32,
) -> Tile<TOP_K_BLOCK> {
    program
        .load_linear(params.at(index), all(), TileLiteral::U32(0))
        .bitcast(ElementType::F32)
}

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

fn add_scaled_index<const BLOCK: usize>(
    index: Tile<BLOCK>,
    component: Tile<BLOCK>,
    stride: u32,
) -> Tile<BLOCK> {
    if stride == 0 {
        index
    } else {
        index + component * u32_tile(stride)
    }
}

fn index2<const BLOCK: usize>(
    offset: u32,
    strides: [u32; 2],
    i0: Tile<BLOCK>,
    i1: Tile<BLOCK>,
) -> Tile<BLOCK> {
    let index = add_scaled_index(u32_tile(offset), i0, strides[0]);
    add_scaled_index(index, i1, strides[1])
}

fn index3_with_base<const BLOCK: usize>(
    base: u32,
    strides: [u32; 3],
    i0: Tile<BLOCK>,
    i1: Tile<BLOCK>,
    i2: Tile<BLOCK>,
) -> Tile<BLOCK> {
    let index = add_scaled_index(u32_tile(base), i0, strides[0]);
    let index = add_scaled_index(index, i1, strides[1]);
    add_scaled_index(index, i2, strides[2])
}

fn index4<const BLOCK: usize>(
    offset: u32,
    strides: [u32; 4],
    i0: Tile<BLOCK>,
    i1: Tile<BLOCK>,
    i2: Tile<BLOCK>,
    i3: Tile<BLOCK>,
) -> Tile<BLOCK> {
    let index = index3_with_base(offset, [strides[0], strides[1], strides[2]], i0, i1, i2);
    add_scaled_index(index, i3, strides[3])
}

fn index4_const_last<const BLOCK: usize>(
    offset: u32,
    strides: [u32; 4],
    i0: Tile<BLOCK>,
    i1: Tile<BLOCK>,
    i2: Tile<BLOCK>,
    i3: u32,
) -> Tile<BLOCK> {
    let base = offset + i3 * strides[3];
    index3_with_base(base, [strides[0], strides[1], strides[2]], i0, i1, i2)
}

fn reduce_workgroup_f32<const BLOCK: usize>(
    program: &mut TileBlock<'_, BLOCK>,
    scratch: crate::TileRef,
    lane: tile::Range<BLOCK>,
    sum: bool,
) {
    let mut stride = BLOCK as u32 / 2;
    while stride > 0 {
        let participates = program.index(lane.clone()).lt(u32_tile(stride));
        program.if_then(participates, |program| {
            let left = program.load_workgroup(scratch, lane.clone());
            let rhs_index = lane.clone() + stride;
            let right = program.load_workgroup(scratch, rhs_index);
            let reduced = if sum { left + right } else { left.max(right) };
            program.store_workgroup(scratch, lane.clone(), reduced);
        });
        program.workgroup_barrier();
        stride /= 2;
    }
}

pub fn flash_attention(meta: FlashAttentionMeta) -> Option<KernelIr> {
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
    let groups = meta.dims.num_heads.checked_div(meta.dims.num_kv_heads)?;
    if groups == 0 {
        return None;
    }
    Some(tile::build(move |phase| {
        let layout = linear_storage_layout();
        let q = phase.storage_read_with_layout::<F32, 1>(layout.clone());
        let k = phase.storage_read_with_layout::<F32, 1>(layout.clone());
        let v = phase.storage_read_with_layout::<F32, 1>(layout.clone());
        let mask = meta
            .mask_meta
            .as_ref()
            .map(|_| phase.storage_read_with_layout::<F32, 1>(layout.clone()));
        let output = phase.storage_write_with_layout::<F32, 1>(layout);

        phase.program_grid::<FLASH_BLOCK>(meta.dispatch_size, |program| {
            let lane = program.arange();
            let workgroup_x = program.program_id(WorkgroupAxis::X);
            let row = program.program_id(WorkgroupAxis::Y);
            let q_idx = program.pin(program.index(row.clone() % meta.dims.q_seq_len));
            let row_over_q = row.clone() / meta.dims.q_seq_len;
            let head_idx = program.pin(program.index(row_over_q.clone() % meta.dims.num_heads));
            let batch_idx =
                program.pin(program.index(row / (meta.dims.q_seq_len * meta.dims.num_heads)));
            let kv_head_idx = program.pin(head_idx.get() / u32_tile(groups));
            let kv_lane = program.index(lane.clone() % FLASH_SIMD_WIDTH);
            let out_dim = program.pin(program.index(
                workgroup_x * FLASH_OUTPUTS_PER_WORKGROUP + (lane.clone() / FLASH_SIMD_WIDTH),
            ));
            let out_valid = program.pin(out_dim.get().lt(u32_tile(meta.dims.head_dim)));
            let loop_idx = program.private::<U32>();
            let score_local = program.private::<F32>();
            let weighted_local = program.private::<F32>();
            let m_local = program.private::<F32>();
            let s_local = program.private::<F32>();
            let o_local = program.private::<F32>();

            program.emit(q_idx.get());
            program.emit(head_idx.get());
            program.emit(batch_idx.get());
            program.emit(kv_head_idx.get());
            program.emit(out_dim.get());
            program.emit(out_valid.get());

            program.store_local(&m_local, f32_tile(NEG_MAX_F32));
            program.store_local(&s_local, f32_tile(0.0));
            program.store_local(&o_local, f32_tile(0.0));
            program.store_local(&loop_idx, u32_tile(0));

            let kv_chunks = meta.dims.kv_seq_len.div_ceil(FLASH_SIMD_WIDTH);
            program.loop_forever(|program| {
                let chunk = program.load_local(&loop_idx);
                program.if_then(chunk.clone().ge(u32_tile(kv_chunks)), |program| {
                    program.break_loop();
                });
                let kv_idx =
                    program.pin(chunk.clone() * u32_tile(FLASH_SIMD_WIDTH) + kv_lane.clone());
                let kv_valid = program.pin(kv_idx.get().lt(u32_tile(meta.dims.kv_seq_len)));
                program.emit(kv_idx.get());
                program.emit(kv_valid.get());
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
                        let q_value = program.load_linear(
                            q.at(q_index),
                            all(),
                            TileLiteral::F32(F32Bits::new(0.0)),
                        );
                        let k_value = program.load_linear(
                            k.at(k_index),
                            all(),
                            TileLiteral::F32(F32Bits::new(0.0)),
                        );
                        products.push(q_value * k_value);
                    }
                    let mut score = program.sum(products) * f32_tile(meta.scale.get());
                    if let (Some(mask), Some(mask_meta), Some(mask_strides)) =
                        (&mask, meta.mask_meta.as_ref(), mask_strides)
                    {
                        let mask_index =
                            index2(mask_meta.offset, mask_strides, q_idx.get(), kv_idx.get());
                        let mask_value = program.load_linear(
                            mask.at(mask_index),
                            all(),
                            TileLiteral::F32(F32Bits::new(0.0)),
                        );
                        score = score + mask_value;
                    }
                    program.store_local(&score_local, score);
                });

                let score = program.pin(program.load_local(&score_local));
                program.emit(score.get());
                let block_max = program.pin(program.subgroup_reduce_max(score.get()));
                program.emit(block_max.get());
                let old_m = program.pin(program.load_local(&m_local));
                program.emit(old_m.get());
                let new_m = program.pin(old_m.get().max(block_max.get()));
                program.emit(new_m.get());
                let raw_exp = (score.get() - new_m.get()).exp();
                let exp_score = program.pin(Tile::select(kv_valid.get(), raw_exp, f32_tile(0.0)));
                program.emit(exp_score.get());
                let block_sum = program.pin(program.subgroup_reduce_sum(exp_score.get()));
                program.emit(block_sum.get());

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
                    let v_value = program.load_linear(
                        v.at(v_index),
                        all(),
                        TileLiteral::F32(F32Bits::new(0.0)),
                    );
                    program.store_local(&weighted_local, exp_score.get() * v_value);
                });
                let weighted = program.load_local(&weighted_local);
                let block_out = program.pin(program.subgroup_reduce_sum(weighted));
                program.emit(block_out.get());

                let old_m_scale = program.pin((old_m.get() - new_m.get()).exp());
                program.emit(old_m_scale.get());
                let new_s = program.load_local(&s_local) * old_m_scale.get() + block_sum.get();
                let new_o = program.load_local(&o_local) * old_m_scale.get() + block_out.get();
                program.store_local(&s_local, new_s);
                program.store_local(&o_local, new_o);
                program.store_local(&m_local, new_m.get());
                program.store_local(&loop_idx, chunk + u32_tile(1));
            });

            let store_valid = kv_lane.eq(u32_tile(0)).and(out_valid.get());
            program.if_then(store_valid, |program| {
                let output_value = program.load_local(&o_local) / program.load_local(&s_local);
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
    }))
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
        program.if_then(dim.clone().ge(u32_tile(DECODE_HEAD_DIM)), |program| {
            program.break_loop();
        });
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
        program.if_then(kv.clone().ge(active_kv_len.clone()), |program| {
            program.break_loop();
        });
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

fn flash_decode_small_block<const BLOCK: usize>(meta: FlashDecodeSmallMeta) -> KernelIr {
    tile::build(move |phase| {
        let layout = linear_storage_layout();
        let q = phase.storage_read_with_layout::<F32, 1>(layout.clone());
        let k = phase.storage_read_with_layout::<F32, 1>(layout.clone());
        let v = phase.storage_read_with_layout::<F32, 1>(layout.clone());
        let output = phase.storage_write_with_layout::<F32, 1>(layout.clone());
        let params = phase.storage_read_with_layout::<U32, 1>(layout);
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
                    program.if_then(kv.clone().ge(active_kv_len.clone()), |program| {
                        program.break_loop();
                    });
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
                reduce_workgroup_f32(program, reduce, lane.clone(), false);
                let max_score = program.load_workgroup(reduce, 0);
                program.store_local(&max_score_local, max_score);
                let max_score = program.load_local(&max_score_local);

                program.store_workgroup(reduce, lane.clone(), f32_tile(0.0));
                program.store_local(&kv_local, lane_value.clone());
                program.loop_forever(|program| {
                    let kv = program.load_local(&kv_local);
                    program.if_then(kv.clone().ge(active_kv_len.clone()), |program| {
                        program.break_loop();
                    });
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
                reduce_workgroup_f32(program, reduce, lane.clone(), true);
                let denom = program.load_workgroup(reduce, 0);

                program.store_local(&acc, f32_tile(0.0));
                program.store_local(&kv_local, u32_tile(0));
                program.loop_forever(|program| {
                    let tile_base = program.load_local(&kv_local);
                    program.if_then(tile_base.clone().ge(active_kv_len.clone()), |program| {
                        program.break_loop();
                    });
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
                            program.if_then(block_done.or(kv_done), |program| {
                                program.break_loop();
                            });
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
            reduce_workgroup_f32(program, reduce, lane.clone(), false);
            let max_score = program.load_workgroup(reduce, 0);
            let score_value = program.load_workgroup(scores, lane.clone());
            let raw_prob = (score_value - max_score).exp();
            let prob = Tile::select(kv_valid.clone(), raw_prob, f32_tile(0.0));
            program.store_workgroup(probs, lane.clone(), prob.clone());
            program.store_workgroup(reduce, lane.clone(), prob);
            program.workgroup_barrier();
            reduce_workgroup_f32(program, reduce, lane.clone(), true);
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
    })
}

pub fn flash_decode_small(meta: FlashDecodeSmallMeta) -> Option<KernelIr> {
    if meta.dims.head_dim != DECODE_HEAD_DIM || meta.decode_block == 0 || meta.groups == 0 {
        return None;
    }
    match meta.decode_block {
        128 => Some(flash_decode_small_block::<128>(meta)),
        512 => Some(flash_decode_small_block::<512>(meta)),
        1024 => Some(flash_decode_small_block::<1024>(meta)),
        _ => None,
    }
}

pub fn top_k_chunk(meta: TopKChunkMeta) -> Option<KernelIr> {
    if meta.input_len == 0 || meta.output_per_chunk == 0 {
        return None;
    }

    Some(tile::build(move |phase| {
        let layout = linear_storage_layout();
        let input = phase.storage_read_with_layout::<F32, 1>(layout.clone());
        let output_ids = phase.storage_write_with_layout::<U32, 1>(layout.clone());
        let output_values = phase.storage_write_with_layout::<F32, 1>(layout.clone());
        let processors = meta.processors.then(|| {
            (
                phase.storage_read_with_layout::<U32, 1>(layout.clone()),
                phase.storage_read_with_layout::<U32, 1>(layout),
            )
        });
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
                    TileLiteral::F32(F32Bits::new(NEG_MAX_F32)),
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
    }))
}

pub fn top_k_exactness(meta: TopKExactnessMeta) -> Option<KernelIr> {
    if meta.top_k == 0 || meta.candidate_count >= meta.output_per_chunk {
        return None;
    }

    Some(tile::build(move |phase| {
        let layout = linear_storage_layout();
        let top_values = phase.storage_read_with_layout::<F32, 1>(layout.clone());
        let chunk_values = phase.storage_read_with_layout::<F32, 1>(layout.clone());
        let flag = phase.storage_write_with_layout::<U32, 1>(layout);
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
                TileLiteral::F32(F32Bits::new(NEG_MAX_F32)),
            );
            let threshold_finite = is_finite(threshold.clone());
            program.store_local(&threshold_local, threshold);
            program.store_local(&threshold_finite_local, threshold_finite);
            program.store_local(&inexact, u32_tile(0));
            program.store_local(&chunk, program.index(lane.clone()));

            program.loop_forever(|program| {
                let chunk_value = program.load_local(&chunk);
                program.if_then(chunk_value.clone().ge(u32_tile(meta.chunks)), |program| {
                    program.break_loop();
                });
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
                    TileLiteral::F32(F32Bits::new(NEG_MAX_F32)),
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

            let mut stride = TOP_K_BLOCK as u32 / 2;
            while stride > 0 {
                let condition = program.index(lane.clone()).lt(u32_tile(stride));
                program.if_then(condition, |program| {
                    let rhs_index = lane.clone() + stride;
                    let lhs = program.load_workgroup(scratch, lane.clone());
                    let rhs = program.load_workgroup(scratch, rhs_index);
                    program.store_workgroup(scratch, lane.clone(), lhs.bit_or(rhs));
                });
                program.workgroup_barrier();
                stride /= 2;
            }

            let lane_zero = program.index(lane.clone()).eq(u32_tile(0));
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
    }))
}

pub fn top_k_merge(meta: MergeTopKMeta) -> Option<KernelIr> {
    if meta.chunks == 0 || meta.chunk_len == 0 || meta.k == 0 {
        return None;
    }

    Some(tile::build(move |phase| {
        let layout = linear_storage_layout();
        let input_ids = phase.storage_read_with_layout::<U32, 1>(layout.clone());
        let input_values = phase.storage_read_with_layout::<F32, 1>(layout.clone());
        let output_ids = phase.storage_write_with_layout::<U32, 1>(layout.clone());
        let output_values = phase.storage_write_with_layout::<F32, 1>(layout);
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
                program.if_then(chunk.clone().ge(u32_tile(meta.chunks)), |program| {
                    program.break_loop();
                });
                program.store_workgroup(chunk_positions, chunk.clone(), u32_tile(0));
                program.store_local(&scan_chunk, chunk + u32_tile(TOP_K_BLOCK as u32));
            });
            program.workgroup_barrier();

            program.store_local(&rank, u32_tile(0));
            program.loop_forever(|program| {
                let rank_value = program.load_local(&rank);
                program.if_then(rank_value.clone().ge(u32_tile(meta.k)), |program| {
                    program.break_loop();
                });
                program.store_local(&local_best_value, f32_tile(NEG_MAX_F32));
                program.store_local(&local_best_id, u32_tile(u32::MAX));
                program.store_local(&local_best_chunk, u32_tile(u32::MAX));
                program.store_local(&scan_chunk, program.index(lane.clone()));

                program.loop_forever(|program| {
                    let chunk = program.load_local(&scan_chunk);
                    program.if_then(chunk.clone().ge(u32_tile(meta.chunks)), |program| {
                        program.break_loop();
                    });
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
                            TileLiteral::F32(F32Bits::new(NEG_MAX_F32)),
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
                    program.if_then(step.clone().eq(u32_tile(0)), |program| {
                        program.break_loop();
                    });
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

                let lane_zero = program.index(lane.clone()).eq(u32_tile(0));
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
    }))
}

pub fn mirostat2(meta: Mirostat2Meta) -> Option<KernelIr> {
    if meta.top_k == 0 {
        return None;
    }

    Some(tile::build(move |phase| {
        let layout = linear_storage_layout();
        let ids = phase.storage_read_with_layout::<U32, 1>(layout.clone());
        let values = phase.storage_read_with_layout::<F32, 1>(layout.clone());
        let state = phase.storage_write_with_layout::<F32, 1>(layout.clone());
        let params = phase.storage_read_with_layout::<F32, 1>(layout.clone());
        let output = phase.storage_write_with_layout::<U32, 1>(layout.clone());
        let exactness_flag = meta
            .has_exactness_flag
            .then(|| phase.storage_read_with_layout::<U32, 1>(layout));
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
    }))
}
