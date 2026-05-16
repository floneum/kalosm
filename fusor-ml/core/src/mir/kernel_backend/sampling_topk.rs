use std::hash::Hash;

use fusor_tile_ir_kernels as tile_ir_kernels;

use crate::{
    kernel_selection::{Axis, KernelDeviceCaps, KernelShape, ShapeRule, ShapeSelector, eq, range},
    mir::kernel_backend,
    sampling::{
        TOP_K_BLOCK, TOP_K_CHUNK,
        processors::{fixed_previous_tokens_data, processor_params_data},
    },
    tensor::{DataTypeEnum, TensorData},
};
use wgpu::CommandEncoder;

const TOPK_A: Axis<0> = Axis;
const TOPK_B: Axis<1> = Axis;
const TOPK_C: Axis<2> = Axis;
const TOPK_D: Axis<3> = Axis;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TopKChunkVariant {
    Empty,
    Kernel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TopKMergeVariant {
    Empty,
    Kernel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TopKExactnessVariant {
    Ineligible,
    Kernel,
}

struct ProveTopKExactKernelVariant;
struct ChunkTopKPairsKernelVariant;
struct MergeSortedChunkTopKPairsKernelVariant;

fn topk_empty_caps() -> KernelDeviceCaps {
    KernelDeviceCaps {
        subgroups_supported: false,
        cooperative_matrix_supported: false,
        min_subgroup_size: 0,
        max_subgroup_size: 0,
        max_compute_invocations_per_workgroup: 0,
        max_compute_workgroup_storage_size: 0,
        max_compute_workgroup_size_x: 0,
        max_compute_workgroups_per_dimension: 0,
    }
}

fn topk_nonempty_selector<Variant: Copy>(
    empty: Variant,
    kernel: Variant,
) -> ShapeSelector<3, (), Variant> {
    ShapeSelector::new()
        .rule(empty, ShapeRule::new().axis(TOPK_A, eq(0)))
        .rule(empty, ShapeRule::new().axis(TOPK_B, eq(0)))
        .rule(empty, ShapeRule::new().axis(TOPK_C, eq(0)))
        .rule(kernel, ShapeRule::new())
}

fn topk_chunk_selector() -> ShapeSelector<3, (), TopKChunkVariant> {
    topk_nonempty_selector(TopKChunkVariant::Empty, TopKChunkVariant::Kernel)
}

fn select_topk_chunk_variant(
    input_len: usize,
    candidate_count: usize,
    output_per_chunk: usize,
) -> TopKChunkVariant {
    topk_chunk_selector()
        .select(
            KernelShape::new([input_len, candidate_count, output_per_chunk]),
            &(),
            topk_empty_caps(),
        )
        .expect("top-k chunk selector has a catch-all rule")
}

fn topk_merge_selector() -> ShapeSelector<3, (), TopKMergeVariant> {
    topk_nonempty_selector(TopKMergeVariant::Empty, TopKMergeVariant::Kernel)
}

fn select_topk_merge_variant(
    chunks: usize,
    chunk_len: usize,
    output_len: usize,
) -> TopKMergeVariant {
    topk_merge_selector()
        .select(
            KernelShape::new([chunks, chunk_len, output_len]),
            &(),
            topk_empty_caps(),
        )
        .expect("top-k merge selector has a catch-all rule")
}

fn topk_exactness_selector() -> ShapeSelector<4, (), TopKExactnessVariant> {
    ShapeSelector::new()
        .rule(
            TopKExactnessVariant::Ineligible,
            ShapeRule::new().axis(TOPK_D, eq(0)),
        )
        .rule(
            TopKExactnessVariant::Ineligible,
            ShapeRule::new().when(|shape: KernelShape<4>, _ctx: &(), _caps| {
                let top_values_len = shape[TOPK_A];
                let candidate_count = shape[TOPK_B];
                let output_per_chunk = shape[TOPK_C];
                let top_k = shape[TOPK_D];
                top_k == 0 || top_values_len < top_k || candidate_count >= output_per_chunk
            }),
        )
        .rule(
            TopKExactnessVariant::Kernel,
            ShapeRule::new()
                .axis(TOPK_A, range(1024..=8192))
                .axis(TOPK_B, range(1..=64))
                .axis(TOPK_C, range(65..=128))
                .axis(TOPK_D, range(1..=512)),
        )
        .rule(TopKExactnessVariant::Kernel, ShapeRule::new())
}

fn select_topk_exactness_variant(
    top_values_len: usize,
    candidate_count: usize,
    output_per_chunk: usize,
    top_k: usize,
) -> TopKExactnessVariant {
    topk_exactness_selector()
        .select(
            KernelShape::new([top_values_len, candidate_count, output_per_chunk, top_k]),
            &(),
            topk_empty_caps(),
        )
        .expect("top-k exactness selector has a catch-all rule")
}

pub(crate) fn top_k_exactness_flag_data_with_encoder(
    top_values: &TensorData,
    chunk_values: &TensorData,
    chunks: usize,
    candidate_count: usize,
    output_per_chunk: usize,
    top_k: usize,
    encoder: Option<&mut CommandEncoder>,
) -> Option<TensorData> {
    if top_values.datatype() != DataTypeEnum::F32
        || chunk_values.datatype() != DataTypeEnum::F32
        || top_values.layout().rank() != 1
        || chunk_values.layout().rank() != 1
        || !top_values.device().is_same_device(chunk_values.device())
    {
        return None;
    }
    if select_topk_exactness_variant(
        top_values.layout().shape()[0],
        candidate_count,
        output_per_chunk,
        top_k,
    ) == TopKExactnessVariant::Ineligible
    {
        return None;
    }

    let device = top_values.device();
    let flag = TensorData::new_for_shape(device, &[1], DataTypeEnum::U32);
    let meta = tile_ir_kernels::TopKExactnessMeta {
        chunks: chunks.try_into().ok()?,
        candidate_count: candidate_count.try_into().ok()?,
        output_per_chunk: output_per_chunk.try_into().ok()?,
        top_k: top_k.try_into().ok()?,
        top_values_offset: top_values.layout().offset().try_into().ok()?,
        top_values_stride: top_values.layout().strides()[0].try_into().ok()?,
        chunk_values_offset: chunk_values.layout().offset().try_into().ok()?,
        chunk_values_stride: chunk_values.layout().strides()[0].try_into().ok()?,
    };
    let cache_key = kernel_backend::module_key_from(|state| {
        kernel_backend::KernelVariantKey::of::<ProveTopKExactKernelVariant>().hash(state);
        TOP_K_BLOCK.hash(state);
        chunks.hash(state);
        candidate_count.hash(state);
        output_per_chunk.hash(state);
        top_k.hash(state);
        top_values.layout().offset().hash(state);
        top_values.layout().shape().hash(state);
        top_values.layout().strides().hash(state);
        chunk_values.layout().offset().hash(state);
        chunk_values.layout().shape().hash(state);
        chunk_values.layout().strides().hash(state);
    });
    let kernel = kernel_backend::run_kernel(
        device,
        "prove_top_k_exact_f32",
        cache_key,
        [1, 1, 1],
        |kb| {
            tile_ir_kernels::top_k_exactness(
                kb,
                kernel_backend::linear_tensor_ref(top_values),
                kernel_backend::linear_tensor_ref(chunk_values),
                kernel_backend::linear_tensor_ref(&flag),
                meta,
            )
        },
    )?;

    kernel_backend::run_direct_kernel(device, "prove_top_k_exact_f32 encoder", &kernel, encoder);

    Some(flag)
}

pub(crate) fn chunk_top_k_pair_data_with_encoder(
    input: &TensorData,
    candidate_count: usize,
    output_per_chunk: usize,
    encoder: Option<&mut CommandEncoder>,
) -> Option<(TensorData, TensorData)> {
    chunk_top_k_pair_data_inner_with_encoder(
        input,
        candidate_count,
        output_per_chunk,
        None,
        encoder,
    )
}

pub(crate) fn chunk_top_k_pair_data_with_processors_with_encoder(
    input: &TensorData,
    previous_tokens: &[u32],
    temperature: f32,
    repetition_penalty: f32,
    candidate_count: usize,
    output_per_chunk: usize,
    encoder: Option<&mut CommandEncoder>,
) -> Option<(TensorData, TensorData)> {
    let device = input.device();
    let (previous_tokens, previous_len) = fixed_previous_tokens_data(device, previous_tokens);
    let params = processor_params_data(device, temperature, repetition_penalty, previous_len);
    chunk_top_k_pair_data_inner_with_encoder(
        input,
        candidate_count,
        output_per_chunk,
        Some((&previous_tokens, &params)),
        encoder,
    )
}

fn chunk_top_k_pair_data_inner_with_encoder(
    input: &TensorData,
    candidate_count: usize,
    output_per_chunk: usize,
    processors: Option<(&TensorData, &TensorData)>,
    encoder: Option<&mut CommandEncoder>,
) -> Option<(TensorData, TensorData)> {
    if input.datatype() != DataTypeEnum::F32 || input.layout().rank() != 1 {
        return None;
    }

    let input_len = input.layout().shape()[0];
    let chunks = input_len.div_ceil(TOP_K_CHUNK);
    let output_len = chunks.checked_mul(output_per_chunk)?;
    let device = input.device();
    let ids = TensorData::new_for_shape(device, &[output_len], DataTypeEnum::U32);
    let values = TensorData::new_for_shape(device, &[output_len], DataTypeEnum::F32);
    if select_topk_chunk_variant(input_len, candidate_count, output_per_chunk)
        == TopKChunkVariant::Empty
    {
        return Some((ids, values));
    }

    let input_offset = input.layout().offset();
    let input_stride = input.layout().strides()[0];
    let has_processors = processors.is_some();
    let cache_key = kernel_backend::module_key_from(|state| {
        kernel_backend::KernelVariantKey::of::<ChunkTopKPairsKernelVariant>().hash(state);
        TOP_K_BLOCK.hash(state);
        TOP_K_CHUNK.hash(state);
        input_len.hash(state);
        candidate_count.hash(state);
        output_per_chunk.hash(state);
        input_offset.hash(state);
        input_stride.hash(state);
        has_processors.hash(state);
    });

    let kernel = kernel_backend::run_kernel(
        device,
        "chunk_top_k_pairs_f32",
        cache_key,
        [chunks.try_into().ok()?, 1, 1],
        |kb| {
            tile_ir_kernels::top_k_chunk(
                kb,
                kernel_backend::linear_tensor_ref(input),
                kernel_backend::linear_tensor_ref(&ids),
                kernel_backend::linear_tensor_ref(&values),
                processors.map(|(previous_tokens, params)| {
                    (
                        kernel_backend::linear_tensor_ref(previous_tokens),
                        kernel_backend::linear_tensor_ref(params),
                    )
                }),
                tile_ir_kernels::TopKChunkMeta {
                    input_len: input_len.try_into().ok()?,
                    output_per_chunk: output_per_chunk.try_into().ok()?,
                    input_offset: input_offset.try_into().ok()?,
                    input_stride: input_stride.try_into().ok()?,
                    processors: has_processors,
                },
            )
        },
    )?;

    kernel_backend::run_direct_kernel(device, "chunk_top_k_pairs_f32 encoder", &kernel, encoder);

    Some((ids, values))
}

pub(crate) fn merge_sorted_chunk_top_k_pair_data_with_encoder(
    input_ids: &TensorData,
    input_values: &TensorData,
    chunks: usize,
    chunk_len: usize,
    chunk_stride: usize,
    input_len: usize,
    k: usize,
    encoder: Option<&mut CommandEncoder>,
) -> Option<(TensorData, TensorData)> {
    if input_ids.datatype() != DataTypeEnum::U32 || input_values.datatype() != DataTypeEnum::F32 {
        return None;
    }
    if input_ids.layout().rank() != 1 || input_values.layout().rank() != 1 {
        return None;
    }
    let input_ids_len = input_ids.layout().shape()[0];
    let input_values_len = input_values.layout().shape()[0];
    let expected_len = if chunks == 0 {
        0
    } else {
        (chunks - 1)
            .checked_mul(chunk_stride)?
            .checked_add(chunk_len)?
    };
    if input_ids_len < expected_len || input_values_len < expected_len {
        return None;
    }

    let device = input_values.device();
    let output_len = k.min(input_len);
    let ids = TensorData::new_for_shape(device, &[output_len], DataTypeEnum::U32);
    let values = TensorData::new_for_shape(device, &[output_len], DataTypeEnum::F32);
    if select_topk_merge_variant(chunks, chunk_len, output_len) == TopKMergeVariant::Empty {
        return Some((ids, values));
    }

    let cache_key = kernel_backend::module_key_from(|state| {
        kernel_backend::KernelVariantKey::of::<MergeSortedChunkTopKPairsKernelVariant>()
            .hash(state);
        TOP_K_BLOCK.hash(state);
        chunks.hash(state);
        chunk_len.hash(state);
        chunk_stride.hash(state);
        input_len.hash(state);
        output_len.hash(state);
        input_ids.layout().offset().hash(state);
        input_ids.layout().shape().hash(state);
        input_ids.layout().strides().hash(state);
        input_values.layout().offset().hash(state);
        input_values.layout().shape().hash(state);
        input_values.layout().strides().hash(state);
    });
    let kernel = kernel_backend::run_kernel(
        device,
        "merge_sorted_chunk_top_k_pairs_f32",
        cache_key,
        [1, 1, 1],
        |kb| {
            tile_ir_kernels::top_k_merge(
                kb,
                kernel_backend::linear_tensor_ref(input_ids),
                kernel_backend::linear_tensor_ref(input_values),
                kernel_backend::linear_tensor_ref(&ids),
                kernel_backend::linear_tensor_ref(&values),
                tile_ir_kernels::MergeTopKMeta {
                    chunks: chunks.try_into().ok()?,
                    chunk_len: chunk_len.try_into().ok()?,
                    chunk_stride: chunk_stride.try_into().ok()?,
                    input_len: input_len.try_into().ok()?,
                    k: output_len.try_into().ok()?,
                },
            )
        },
    )?;

    kernel_backend::run_direct_kernel(
        device,
        "merge_sorted_chunk_top_k_pairs_f32 encoder",
        &kernel,
        encoder,
    );

    Some((ids, values))
}

#[cfg(test)]
mod selection_tests {
    use super::*;
    use crate::kernel_selection::assert_selector_generates;

    #[test]
    fn topk_chunk_selector_generates_each_variant() {
        let selector = topk_chunk_selector();
        assert_selector_generates(
            &selector,
            [TopKChunkVariant::Empty, TopKChunkVariant::Kernel]
                .map(|variant| (variant, (), topk_empty_caps())),
        );
    }

    #[test]
    fn topk_merge_selector_generates_each_variant() {
        let selector = topk_merge_selector();
        assert_selector_generates(
            &selector,
            [TopKMergeVariant::Empty, TopKMergeVariant::Kernel]
                .map(|variant| (variant, (), topk_empty_caps())),
        );
    }

    #[test]
    fn topk_exactness_selector_generates_each_variant() {
        let selector = topk_exactness_selector();
        assert_selector_generates(
            &selector,
            [
                TopKExactnessVariant::Ineligible,
                TopKExactnessVariant::Kernel,
            ]
            .map(|variant| (variant, (), topk_empty_caps())),
        );
    }
}
