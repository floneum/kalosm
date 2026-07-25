use std::hash::Hash;
use std::sync::Arc;

use fusor_tile_ir as tile_ir;

use crate::{
    Device,
    mir::{kernel_backend, operation::hash_layout},
    sampling::{
        GPU_SAMPLE_RESULT_WORDS, GpuSamplerRequest, TOP_K_BLOCK, TOP_K_CHUNK,
        processors::{
            fixed_previous_tokens_data, fixed_previous_tokens_data_with_gpu_tail,
            processor_params_data,
        },
        row_kernels,
    },
    tensor::{DataTypeEnum, TensorData},
};
use wgpu::CommandEncoder;

struct ProveTopKExactKernelVariant;
struct ChunkTopKPairsKernelVariant;
struct MergeSortedChunkTopKPairsKernelVariant;
struct Mirostat2SortedTopKKernelVariant;
struct StandardSamplerSortedTopKKernelVariant;
struct UnfilteredCategoricalSamplerKernelVariant;

/// Build (or reuse) one sampling kernel and record it. Every stage of the
/// sampling tail binds raw buffers it already holds — no compute-graph
/// access — so they all launch through here: with an encoder the dispatch
/// joins the resolver's submission, without one it takes its own.
fn launch(
    device: &Device,
    name: &'static str,
    cache_key: kernel_backend::KernelCacheKey,
    dispatch_size: [u32; 3],
    encoder: Option<&mut CommandEncoder>,
    body: impl FnOnce(&mut tile_ir::KernelBuilder<Arc<wgpu::Buffer>>) -> Option<()>,
) -> Option<()> {
    let kernel =
        kernel_backend::run_kernel(device.kernel_cache(), name, cache_key, dispatch_size, body)?;
    kernel_backend::run_direct_kernel(
        device.kernel_cache(),
        device.wgpu_queue(),
        &format!("{name} encoder"),
        &kernel,
        encoder,
    );
    Some(())
}

/// The four-word parameter buffer every sampler kernel binds.
fn sampler_params_data(device: &Device, words: [f32; 4]) -> TensorData {
    let buffer = device.create_buffer_init(
        bytemuck::bytes_of(&words),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    );
    TensorData::new_from_buffer(device, buffer, &[4], DataTypeEnum::F32)
}

/// True when any dimension of the top-k working set is zero; in that case
/// every top-k kernel short-circuits.
fn top_k_dims_empty(dims: &[usize]) -> bool {
    dims.contains(&0)
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
    let meta = row_kernels::TopKExactnessMeta {
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
        hash_layout(state, top_values.layout());
        hash_layout(state, chunk_values.layout());
    });
    launch(
        device,
        "prove_top_k_exact_f32",
        cache_key,
        [1, 1, 1],
        encoder,
        |kb| {
            row_kernels::top_k_exactness(
                kb,
                top_values.as_kernel_tensor_ref(),
                chunk_values.as_kernel_tensor_ref(),
                flag.as_kernel_tensor_ref(),
                meta,
            )
        },
    )?;

    Some(flag)
}

/// Logit-processor settings (temperature scaling and repetition penalty)
/// applied before the top-k reduction.
#[derive(Clone, Copy)]
pub(crate) struct ProcessorSettings {
    pub temperature: f32,
    pub repetition_penalty: f32,
}

/// The repetition window the processors score against: the host-side token
/// history, optionally extended by a token still living on the GPU (the
/// previous pending sample, copied into the window on `encoder`).
pub(crate) struct ChunkProcessors<'a> {
    pub previous_tokens: &'a [u32],
    pub gpu_tail: Option<&'a TensorData>,
    pub settings: ProcessorSettings,
}

pub(crate) fn chunk_top_k_pair_data_with_encoder(
    input: &TensorData,
    processors: Option<ChunkProcessors<'_>>,
    candidate_count: usize,
    output_per_chunk: usize,
    mut encoder: Option<&mut CommandEncoder>,
) -> Option<(TensorData, TensorData)> {
    let device = input.device();
    let processors = match processors {
        None => None,
        Some(processors) => {
            let previous_len;
            let previous_tokens;
            match processors.gpu_tail {
                None => {
                    (previous_tokens, previous_len) =
                        fixed_previous_tokens_data(device, processors.previous_tokens);
                }
                Some(gpu_tail) => {
                    if gpu_tail.datatype() != DataTypeEnum::U32
                        || gpu_tail.layout().rank() != 1
                        || gpu_tail
                            .layout()
                            .shape()
                            .first()
                            .copied()
                            .unwrap_or_default()
                            == 0
                        || !device.is_same_device(gpu_tail.device())
                    {
                        return None;
                    }
                    let encoder = encoder.as_deref_mut()?;
                    (previous_tokens, previous_len) = fixed_previous_tokens_data_with_gpu_tail(
                        device,
                        processors.previous_tokens,
                        gpu_tail,
                        encoder,
                    );
                }
            }
            let params = processor_params_data(
                device,
                processors.settings.temperature,
                processors.settings.repetition_penalty,
                previous_len,
            );
            Some((previous_tokens, params))
        }
    };

    if input.datatype() != DataTypeEnum::F32 || input.layout().rank() != 1 {
        return None;
    }

    let input_len = input.layout().shape()[0];
    let chunks = input_len.div_ceil(TOP_K_CHUNK);
    let output_len = chunks.checked_mul(output_per_chunk)?;
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

    launch(
        device,
        "chunk_top_k_pairs_f32",
        cache_key,
        [chunks.try_into().ok()?, 1, 1],
        encoder,
        |kb| {
            row_kernels::top_k_chunk(
                kb,
                input.as_kernel_tensor_ref(),
                ids.as_kernel_tensor_ref(),
                values.as_kernel_tensor_ref(),
                processors.as_ref().map(|(previous_tokens, params)| {
                    (
                        previous_tokens.as_kernel_tensor_ref(),
                        params.as_kernel_tensor_ref(),
                    )
                }),
                row_kernels::TopKChunkMeta {
                    input_len: input_len.try_into().ok()?,
                    output_per_chunk: output_per_chunk.try_into().ok()?,
                    input_offset: input_offset.try_into().ok()?,
                    input_stride: input_stride.try_into().ok()?,
                    processors: has_processors,
                },
            )
        },
    )?;

    Some((ids, values))
}

pub(crate) struct MergeSortedChunkTopKParams {
    pub chunks: usize,
    pub chunk_len: usize,
    pub chunk_stride: usize,
    pub input_len: usize,
    pub k: usize,
}

pub(crate) fn merge_sorted_chunk_top_k_pair_data_with_encoder(
    input_ids: &TensorData,
    input_values: &TensorData,
    params: MergeSortedChunkTopKParams,
    encoder: Option<&mut CommandEncoder>,
) -> Option<(TensorData, TensorData)> {
    let MergeSortedChunkTopKParams {
        chunks,
        chunk_len,
        chunk_stride,
        input_len,
        k,
    } = params;
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
        hash_layout(state, input_ids.layout());
        hash_layout(state, input_values.layout());
    });
    launch(
        device,
        "merge_sorted_chunk_top_k_pairs_f32",
        cache_key,
        [1, 1, 1],
        encoder,
        |kb| {
            row_kernels::top_k_merge(
                kb,
                input_ids.as_kernel_tensor_ref(),
                input_values.as_kernel_tensor_ref(),
                ids.as_kernel_tensor_ref(),
                values.as_kernel_tensor_ref(),
                row_kernels::MergeTopKMeta {
                    chunks: chunks.try_into().ok()?,
                    chunk_len: chunk_len.try_into().ok()?,
                    chunk_stride: chunk_stride.try_into().ok()?,
                    input_len: input_len.try_into().ok()?,
                    k: output_len.try_into().ok()?,
                },
            )
        },
    )?;

    Some((ids, values))
}

fn normalized_probability(value: f32, default: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        default
    }
}

pub(crate) fn supports_unfiltered_categorical(
    input_len: usize,
    params: crate::sampling::GpuStandardSamplerParams,
) -> bool {
    input_len > 0
        && input_len <= TOP_K_BLOCK as usize
        && params.top_k >= input_len
        && normalized_probability(params.top_p, 1.0) == 1.0
        && normalized_probability(params.min_p, 0.0) == 0.0
        && params.temperature.is_finite()
        && params.random.is_finite()
}

pub(crate) fn sample_categorical_logits_data_with_encoder(
    logits: &TensorData,
    params: crate::sampling::GpuStandardSamplerParams,
    encoder: Option<&mut CommandEncoder>,
) -> Option<TensorData> {
    if logits.datatype() != DataTypeEnum::F32 || logits.layout().rank() != 1 {
        return None;
    }
    let input_len = logits.layout().shape()[0];
    if !supports_unfiltered_categorical(input_len, params) {
        return None;
    }

    let device = logits.device();
    let params_data = sampler_params_data(
        device,
        [
            params.random.clamp(0.0, 0.999_999_94),
            params.temperature,
            0.0,
            0.0,
        ],
    );
    let output = TensorData::new_for_shape(device, &[GPU_SAMPLE_RESULT_WORDS], DataTypeEnum::U32);
    let block = u32::try_from(input_len.next_power_of_two()).ok()?;
    let meta = row_kernels::CategoricalSamplerMeta {
        input_len: input_len.try_into().ok()?,
        input_offset: logits.layout().offset().try_into().ok()?,
        input_stride: logits.layout().strides()[0].try_into().ok()?,
        block,
    };
    let cache_key = kernel_backend::KernelCacheKey::from_hash_inputs(|state| {
        kernel_backend::KernelVariantKey::of::<UnfilteredCategoricalSamplerKernelVariant>()
            .hash(state);
        input_len.hash(state);
        block.hash(state);
        hash_layout(state, logits.layout());
    });
    launch(
        device,
        "sample_categorical_logits_f32",
        cache_key,
        [1, 1, 1],
        encoder,
        |kb| {
            row_kernels::categorical_sampler(
                kb,
                row_kernels::CategoricalSampler {
                    logits: logits.as_kernel_tensor_ref(),
                    params: params_data.as_kernel_tensor_ref(),
                    output: output.as_kernel_tensor_ref(),
                    meta,
                },
            )
        },
    )?;
    Some(output)
}

/// Terminate the top-k tail with the sampler `request` selects. Both kernels
/// read the same sorted candidate pairs and the same optional exactness flag;
/// only the parameter buffer and the selection rule differ.
pub(crate) fn sample_from_sorted_top_k_data_with_encoder(
    ids: &TensorData,
    values: &TensorData,
    request: &mut GpuSamplerRequest<'_>,
    exactness_flag: Option<&TensorData>,
    encoder: Option<&mut CommandEncoder>,
) -> Option<TensorData> {
    if ids.datatype() != DataTypeEnum::U32 || values.datatype() != DataTypeEnum::F32 {
        return None;
    }
    if ids.layout().rank() != 1 || values.layout().rank() != 1 {
        return None;
    }
    if let Some(flag) = exactness_flag
        && (flag.datatype() != DataTypeEnum::U32
            || flag.layout().rank() != 1
            || flag.layout().shape()[0] == 0
            || !values.device().is_same_device(flag.device()))
    {
        return None;
    }

    let top_k = request
        .top_k()
        .min(ids.layout().shape()[0])
        .min(values.layout().shape()[0]);
    if top_k == 0 {
        return None;
    }
    let device = values.device();
    let has_exactness_flag = exactness_flag.is_some();
    let output = TensorData::new_for_shape(device, &[GPU_SAMPLE_RESULT_WORDS], DataTypeEnum::U32);
    let meta = row_kernels::SamplerMeta {
        top_k: top_k.try_into().ok()?,
        ids_offset: ids.layout().offset().try_into().ok()?,
        ids_stride: ids.layout().strides()[0].try_into().ok()?,
        values_offset: values.layout().offset().try_into().ok()?,
        values_stride: values.layout().strides()[0].try_into().ok()?,
        has_exactness_flag,
    };
    let sorted_top_k_cache_key = |variant: kernel_backend::KernelVariantKey| {
        kernel_backend::KernelCacheKey::from_hash_inputs(|state| {
            variant.hash(state);
            TOP_K_BLOCK.hash(state);
            top_k.hash(state);
            hash_layout(state, ids.layout());
            hash_layout(state, values.layout());
            has_exactness_flag.hash(state);
        })
    };

    match request {
        GpuSamplerRequest::Mirostat2 { sampler, params } => {
            let params_data = sampler_params_data(
                device,
                [
                    params.tau,
                    params.eta,
                    params.random.clamp(0.0, 0.999_999_94),
                    0.0,
                ],
            );
            let variant =
                kernel_backend::KernelVariantKey::of::<Mirostat2SortedTopKKernelVariant>();
            launch(
                device,
                "sample_mirostat2_sorted_top_k_f32",
                sorted_top_k_cache_key(variant),
                [1, 1, 1],
                encoder,
                |kb| {
                    row_kernels::mirostat2(
                        kb,
                        row_kernels::Mirostat2 {
                            ids: ids.as_kernel_tensor_ref(),
                            values: values.as_kernel_tensor_ref(),
                            state: sampler.state.as_kernel_tensor_ref(),
                            params: params_data.as_kernel_tensor_ref(),
                            output: output.as_kernel_tensor_ref(),
                            exactness_flag: exactness_flag.map(|t| t.as_kernel_tensor_ref()),
                            meta,
                        },
                    )
                },
            )?;
        }
        GpuSamplerRequest::Standard { params } => {
            let params_data = sampler_params_data(
                device,
                [
                    params.random.clamp(0.0, 0.999_999_94),
                    normalized_probability(params.top_p, 1.0),
                    normalized_probability(params.min_p, 0.0),
                    0.0,
                ],
            );
            let variant =
                kernel_backend::KernelVariantKey::of::<StandardSamplerSortedTopKKernelVariant>();
            launch(
                device,
                "sample_standard_sorted_top_k_f32",
                sorted_top_k_cache_key(variant),
                [1, 1, 1],
                encoder,
                |kb| {
                    row_kernels::standard_sampler(
                        kb,
                        row_kernels::StandardSampler {
                            ids: ids.as_kernel_tensor_ref(),
                            values: values.as_kernel_tensor_ref(),
                            params: params_data.as_kernel_tensor_ref(),
                            output: output.as_kernel_tensor_ref(),
                            exactness_flag: exactness_flag.map(|t| t.as_kernel_tensor_ref()),
                            meta,
                        },
                    )
                },
            )?;
        }
    }

    Some(output)
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
