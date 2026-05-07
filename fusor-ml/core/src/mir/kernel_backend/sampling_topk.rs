mod chunk;
mod exactness;
mod merge;

use crate::{
    mir::{direct_kernel::DirectKernelBinding, kernel_backend},
    sampling::{
        TOP_K_BLOCK, TOP_K_CHUNK,
        processors::{fixed_previous_tokens_data, processor_params_data},
    },
    tensor::{DataTypeEnum, TensorData},
};
use wgpu::{
    CommandEncoder,
    naga::{GlobalVariable, Handle, LocalVariable},
};

#[derive(Clone, Copy)]
pub(crate) struct TopKExactnessMeta {
    chunks: u32,
    candidate_count: u32,
    output_per_chunk: u32,
    top_k: u32,
    top_values_offset: u32,
    top_values_stride: u32,
    chunk_values_offset: u32,
    chunk_values_stride: u32,
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
        || top_k == 0
        || top_values.layout().shape()[0] < top_k
        || candidate_count >= output_per_chunk
    {
        return None;
    }

    let device = top_values.device();
    let flag = TensorData::new_for_shape(device, &[1], DataTypeEnum::U32);
    let meta = TopKExactnessMeta {
        chunks: chunks.try_into().ok()?,
        candidate_count: candidate_count.try_into().ok()?,
        output_per_chunk: output_per_chunk.try_into().ok()?,
        top_k: top_k.try_into().ok()?,
        top_values_offset: top_values.layout().offset().try_into().ok()?,
        top_values_stride: top_values.layout().strides()[0].try_into().ok()?,
        chunk_values_offset: chunk_values.layout().offset().try_into().ok()?,
        chunk_values_stride: chunk_values.layout().strides()[0].try_into().ok()?,
    };
    let cache_key = format!(
        "prove_top_k_exact_f32:block={TOP_K_BLOCK}:chunks={chunks}:candidate_count={candidate_count}:output_per_chunk={output_per_chunk}:top_k={top_k}:top={:?}:chunk={:?}",
        top_values.layout(),
        chunk_values.layout()
    );
    let kernel = kernel_backend::dynamic_kernel_from_backend_naga_module(
        device,
        "prove_top_k_exact_f32",
        cache_key,
        || TopKExactnessModuleBuilder::new(meta).build(),
        vec![
            DirectKernelBinding::Storage {
                binding: 0,
                buffer: top_values.buffer().clone(),
                read_only: true,
            },
            DirectKernelBinding::Storage {
                binding: 1,
                buffer: chunk_values.buffer().clone(),
                read_only: true,
            },
            DirectKernelBinding::Storage {
                binding: 2,
                buffer: flag.buffer().clone(),
                read_only: false,
            },
        ],
        [1, 1, 1],
    )?;

    if let Some(encoder) = encoder {
        kernel.run(device, encoder);
    } else {
        let mut encoder =
            device
                .wgpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("prove_top_k_exact_f32 encoder"),
                });
        kernel.run(device, &mut encoder);
        device.wgpu_queue().submit(Some(encoder.finish()));
    }

    Some(flag)
}

pub(crate) fn chunk_top_k_pair_data(
    input: &TensorData,
    candidate_count: usize,
    output_per_chunk: usize,
) -> Option<(TensorData, TensorData)> {
    chunk_top_k_pair_data_with_encoder(input, candidate_count, output_per_chunk, None)
}

fn chunk_top_k_pair_data_with_encoder(
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
    if candidate_count == 0 || output_per_chunk == 0 || input_len == 0 {
        return Some((ids, values));
    }

    let input_offset = input.layout().offset();
    let input_stride = input.layout().strides()[0];
    let has_processors = processors.is_some();
    let cache_key = format!(
        "chunk_top_k_pairs_f32:block={TOP_K_BLOCK}:chunk={TOP_K_CHUNK}:len={input_len}:candidate_count={candidate_count}:output_per_chunk={output_per_chunk}:offset={input_offset}:stride={input_stride}:processors={has_processors}"
    );
    let build_module = || {
        TopKModuleBuilder::new(
            input_len.try_into().ok()?,
            output_per_chunk.try_into().ok()?,
            input_offset.try_into().ok()?,
            input_stride.try_into().ok()?,
            has_processors,
        )
        .build()
    };

    let kernel = if let Some((previous_tokens, params)) = processors {
        let bindings = vec![
            DirectKernelBinding::Storage {
                binding: 0,
                buffer: input.buffer().clone(),
                read_only: true,
            },
            DirectKernelBinding::Storage {
                binding: 1,
                buffer: ids.buffer().clone(),
                read_only: false,
            },
            DirectKernelBinding::Storage {
                binding: 2,
                buffer: values.buffer().clone(),
                read_only: false,
            },
            DirectKernelBinding::Storage {
                binding: 3,
                buffer: previous_tokens.buffer().clone(),
                read_only: true,
            },
            DirectKernelBinding::Storage {
                binding: 4,
                buffer: params.buffer().clone(),
                read_only: true,
            },
        ];
        kernel_backend::dynamic_kernel_from_backend_naga_module(
            device,
            "chunk_top_k_pairs_f32",
            cache_key,
            build_module,
            bindings,
            [chunks.try_into().ok()?, 1, 1],
        )?
    } else {
        kernel_backend::dynamic_kernel_from_backend_naga_module(
            device,
            "chunk_top_k_pairs_f32",
            cache_key,
            build_module,
            vec![
                DirectKernelBinding::Storage {
                    binding: 0,
                    buffer: input.buffer().clone(),
                    read_only: true,
                },
                DirectKernelBinding::Storage {
                    binding: 1,
                    buffer: ids.buffer().clone(),
                    read_only: false,
                },
                DirectKernelBinding::Storage {
                    binding: 2,
                    buffer: values.buffer().clone(),
                    read_only: false,
                },
            ],
            [chunks.try_into().ok()?, 1, 1],
        )?
    };

    if let Some(encoder) = encoder {
        kernel.run(device, encoder);
    } else {
        let mut encoder =
            device
                .wgpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("chunk_top_k_pairs_f32 encoder"),
                });
        kernel.run(device, &mut encoder);
        device.wgpu_queue().submit(Some(encoder.finish()));
    }

    Some((ids, values))
}

pub(crate) fn merge_sorted_chunk_top_k_pair_data(
    input_ids: &TensorData,
    input_values: &TensorData,
    chunks: usize,
    chunk_len: usize,
    chunk_stride: usize,
    input_len: usize,
    k: usize,
) -> Option<(TensorData, TensorData)> {
    merge_sorted_chunk_top_k_pair_data_with_encoder(
        input_ids,
        input_values,
        chunks,
        chunk_len,
        chunk_stride,
        input_len,
        k,
        None,
    )
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
    if chunks == 0 || chunk_len == 0 || output_len == 0 {
        return Some((ids, values));
    }

    let cache_key = format!(
        "merge_sorted_chunk_top_k_pairs_f32:block={TOP_K_BLOCK}:chunks={chunks}:chunk_len={chunk_len}:chunk_stride={chunk_stride}:input_len={input_len}:k={output_len}:ids={:?}:values={:?}",
        input_ids.layout(),
        input_values.layout()
    );
    let kernel = kernel_backend::dynamic_kernel_from_backend_naga_module(
        device,
        "merge_sorted_chunk_top_k_pairs_f32",
        cache_key,
        || {
            MergeTopKModuleBuilder::new(
                chunks.try_into().ok()?,
                chunk_len.try_into().ok()?,
                chunk_stride.try_into().ok()?,
                input_len.try_into().ok()?,
                output_len.try_into().ok()?,
            )
            .build()
        },
        vec![
            DirectKernelBinding::Storage {
                binding: 0,
                buffer: input_ids.buffer().clone(),
                read_only: true,
            },
            DirectKernelBinding::Storage {
                binding: 1,
                buffer: input_values.buffer().clone(),
                read_only: true,
            },
            DirectKernelBinding::Storage {
                binding: 2,
                buffer: ids.buffer().clone(),
                read_only: false,
            },
            DirectKernelBinding::Storage {
                binding: 3,
                buffer: values.buffer().clone(),
                read_only: false,
            },
        ],
        [1, 1, 1],
    )?;

    if let Some(encoder) = encoder {
        kernel.run(device, encoder);
    } else {
        let mut encoder =
            device
                .wgpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("merge_sorted_chunk_top_k_pairs_f32 encoder"),
                });
        kernel.run(device, &mut encoder);
        device.wgpu_queue().submit(Some(encoder.finish()));
    }

    Some((ids, values))
}

struct TopKModuleBuilder {
    input_len: u32,
    output_per_chunk: u32,
    input_offset: u32,
    input_stride: u32,
    processors: bool,
}

struct TopKGlobals {
    input: Handle<GlobalVariable>,
    output_ids: Handle<GlobalVariable>,
    output_values: Handle<GlobalVariable>,
    previous_tokens: Option<Handle<GlobalVariable>>,
    processor_params: Option<Handle<GlobalVariable>>,
    scratch_values: Handle<GlobalVariable>,
    scratch_ids: Handle<GlobalVariable>,
}

struct TopKLocals {
    current_value: Handle<LocalVariable>,
    current_id: Handle<LocalVariable>,
    previous_index: Handle<LocalVariable>,
    repeated: Handle<LocalVariable>,
}

struct TopKExactnessModuleBuilder {
    meta: TopKExactnessMeta,
}

#[derive(Clone, Copy)]
struct TopKExactnessGlobals {
    top_values: Handle<GlobalVariable>,
    chunk_values: Handle<GlobalVariable>,
    flag: Handle<GlobalVariable>,
    scratch: Handle<GlobalVariable>,
}

#[derive(Clone, Copy)]
struct TopKExactnessLocals {
    chunk: Handle<LocalVariable>,
    inexact: Handle<LocalVariable>,
}

struct MergeTopKModuleBuilder {
    chunks: u32,
    chunk_len: u32,
    chunk_stride: u32,
    input_len: u32,
    k: u32,
}

struct MergeTopKGlobals {
    input_ids: Handle<GlobalVariable>,
    input_values: Handle<GlobalVariable>,
    output_ids: Handle<GlobalVariable>,
    output_values: Handle<GlobalVariable>,
    chunk_positions: Handle<GlobalVariable>,
    scratch_values: Handle<GlobalVariable>,
    scratch_ids: Handle<GlobalVariable>,
    scratch_chunks: Handle<GlobalVariable>,
}

struct MergeTopKLocals {
    rank: Handle<LocalVariable>,
    scan_chunk: Handle<LocalVariable>,
    local_best_value: Handle<LocalVariable>,
    local_best_id: Handle<LocalVariable>,
    local_best_chunk: Handle<LocalVariable>,
    reduce_step: Handle<LocalVariable>,
}
