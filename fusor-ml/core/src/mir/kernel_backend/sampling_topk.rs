use std::hash::Hash;

use fusor_tile_ir_kernels as tile_ir_kernels;

use crate::{
    mir::kernel_backend,
    sampling::{
        TOP_K_BLOCK, TOP_K_CHUNK,
        processors::{fixed_previous_tokens_data, processor_params_data},
    },
    tensor::{DataTypeEnum, TensorData},
};
use wgpu::CommandEncoder;

struct ProveTopKExactKernelVariant;
struct ChunkTopKPairsKernelVariant;
struct MergeSortedChunkTopKPairsKernelVariant;

/// True when any dimension of the top-k working set is zero; in that case
/// every top-k kernel short-circuits.
fn top_k_dims_empty(dims: &[usize]) -> bool {
    dims.iter().any(|&d| d == 0)
}

/// True when the inputs to `top_k_exactness` can't be sharpened any further
/// by running the exactness kernel (no candidates per output, or no top_k).
fn top_k_exactness_ineligible(
    top_values_len: usize,
    candidate_count: usize,
    output_per_chunk: usize,
    top_k: usize,
) -> bool {
    top_k == 0 || top_values_len < top_k || candidate_count >= output_per_chunk
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
    if top_k_exactness_ineligible(
        top_values.layout().shape()[0],
        candidate_count,
        output_per_chunk,
        top_k,
    ) {
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
    let cache_key = kernel_backend::KernelCacheKey::from_hash_inputs(|state| {
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
        device.kernel_cache(),
        "prove_top_k_exact_f32",
        cache_key,
        [1, 1, 1],
        |kb| {
            tile_ir_kernels::top_k_exactness(
                kb,
                top_values.as_kernel_tensor_ref(),
                chunk_values.as_kernel_tensor_ref(),
                flag.as_kernel_tensor_ref(),
                meta,
            )
        },
    )?;

    kernel_backend::run_direct_kernel(
        device.kernel_cache(),
        device.wgpu_queue(),
        "prove_top_k_exact_f32 encoder",
        &kernel,
        encoder,
    );

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
    if top_k_dims_empty(&[input_len, candidate_count, output_per_chunk]) {
        return Some((ids, values));
    }

    let input_offset = input.layout().offset();
    let input_stride = input.layout().strides()[0];
    let has_processors = processors.is_some();
    let cache_key = kernel_backend::KernelCacheKey::from_hash_inputs(|state| {
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
        device.kernel_cache(),
        "chunk_top_k_pairs_f32",
        cache_key,
        [chunks.try_into().ok()?, 1, 1],
        |kb| {
            tile_ir_kernels::top_k_chunk(
                kb,
                input.as_kernel_tensor_ref(),
                ids.as_kernel_tensor_ref(),
                values.as_kernel_tensor_ref(),
                processors.map(|(previous_tokens, params)| {
                    (
                        previous_tokens.as_kernel_tensor_ref(),
                        params.as_kernel_tensor_ref(),
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

    kernel_backend::run_direct_kernel(
        device.kernel_cache(),
        device.wgpu_queue(),
        "chunk_top_k_pairs_f32 encoder",
        &kernel,
        encoder,
    );

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
    if top_k_dims_empty(&[chunks, chunk_len, output_len]) {
        return Some((ids, values));
    }

    let cache_key = kernel_backend::KernelCacheKey::from_hash_inputs(|state| {
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
        device.kernel_cache(),
        "merge_sorted_chunk_top_k_pairs_f32",
        cache_key,
        [1, 1, 1],
        |kb| {
            tile_ir_kernels::top_k_merge(
                kb,
                input_ids.as_kernel_tensor_ref(),
                input_values.as_kernel_tensor_ref(),
                ids.as_kernel_tensor_ref(),
                values.as_kernel_tensor_ref(),
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
        device.kernel_cache(),
        device.wgpu_queue(),
        "merge_sorted_chunk_top_k_pairs_f32 encoder",
        &kernel,
        encoder,
    );

    Some((ids, values))
}

#[cfg(test)]
mod selection_tests {
    use super::*;

    #[test]
    fn empty_dims_short_circuit() {
        assert!(top_k_dims_empty(&[0, 4, 4]));
        assert!(top_k_dims_empty(&[4, 0, 4]));
        assert!(top_k_dims_empty(&[4, 4, 0]));
        assert!(!top_k_dims_empty(&[1, 1, 1]));
    }

    #[test]
    fn exactness_ineligible_matches_old_selector() {
        // top_k == 0 → ineligible.
        assert!(top_k_exactness_ineligible(1024, 4, 64, 0));
        // top_values_len < top_k → ineligible.
        assert!(top_k_exactness_ineligible(100, 4, 64, 200));
        // candidate_count >= output_per_chunk → ineligible.
        assert!(top_k_exactness_ineligible(1024, 64, 64, 16));
        // Sized values within the eligible window.
        assert!(!top_k_exactness_ineligible(2048, 4, 128, 32));
    }
}
