use fusor_tile_ir as tile_ir;

use crate::{
    kernel_selection::{Axis, KernelDeviceCaps, KernelShape, ShapeRule, ShapeSelector, eq, range},
    mir::{direct_kernel::DirectKernelBinding, kernel_backend},
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

fn topk_chunk_selector() -> ShapeSelector<3, (), TopKChunkVariant> {
    ShapeSelector::new()
        .rule(
            TopKChunkVariant::Empty,
            ShapeRule::new().axis(TOPK_A, eq(0)),
        )
        .rule(
            TopKChunkVariant::Empty,
            ShapeRule::new().axis(TOPK_B, eq(0)),
        )
        .rule(
            TopKChunkVariant::Empty,
            ShapeRule::new().axis(TOPK_C, eq(0)),
        )
        .rule(TopKChunkVariant::Kernel, ShapeRule::new())
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
    ShapeSelector::new()
        .rule(
            TopKMergeVariant::Empty,
            ShapeRule::new().axis(TOPK_A, eq(0)),
        )
        .rule(
            TopKMergeVariant::Empty,
            ShapeRule::new().axis(TOPK_B, eq(0)),
        )
        .rule(
            TopKMergeVariant::Empty,
            ShapeRule::new().axis(TOPK_C, eq(0)),
        )
        .rule(TopKMergeVariant::Kernel, ShapeRule::new())
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
    let kernel = kernel_backend::dynamic_kernel_from_ir(
        device,
        "prove_top_k_exact_f32",
        cache_key,
        || {
            tile_ir::kernels::top_k_exactness(tile_ir::TopKExactnessMeta {
                chunks: meta.chunks,
                candidate_count: meta.candidate_count,
                output_per_chunk: meta.output_per_chunk,
                top_k: meta.top_k,
                top_values_offset: meta.top_values_offset,
                top_values_stride: meta.top_values_stride,
                chunk_values_offset: meta.chunk_values_offset,
                chunk_values_stride: meta.chunk_values_stride,
            })
        },
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
    let cache_key = format!(
        "chunk_top_k_pairs_f32:block={TOP_K_BLOCK}:chunk={TOP_K_CHUNK}:len={input_len}:candidate_count={candidate_count}:output_per_chunk={output_per_chunk}:offset={input_offset}:stride={input_stride}:processors={has_processors}"
    );
    let build_ir = || {
        tile_ir::kernels::top_k_chunk(tile_ir::TopKChunkMeta {
            input_len: input_len.try_into().ok()?,
            output_per_chunk: output_per_chunk.try_into().ok()?,
            input_offset: input_offset.try_into().ok()?,
            input_stride: input_stride.try_into().ok()?,
            processors: has_processors,
        })
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
        kernel_backend::dynamic_kernel_from_ir(
            device,
            "chunk_top_k_pairs_f32",
            cache_key,
            build_ir,
            bindings,
            [chunks.try_into().ok()?, 1, 1],
        )?
    } else {
        kernel_backend::dynamic_kernel_from_ir(
            device,
            "chunk_top_k_pairs_f32",
            cache_key,
            build_ir,
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

    let cache_key = format!(
        "merge_sorted_chunk_top_k_pairs_f32:block={TOP_K_BLOCK}:chunks={chunks}:chunk_len={chunk_len}:chunk_stride={chunk_stride}:input_len={input_len}:k={output_len}:ids={:?}:values={:?}",
        input_ids.layout(),
        input_values.layout()
    );
    let kernel = kernel_backend::dynamic_kernel_from_ir(
        device,
        "merge_sorted_chunk_top_k_pairs_f32",
        cache_key,
        || {
            tile_ir::kernels::top_k_merge(tile_ir::MergeTopKMeta {
                chunks: chunks.try_into().ok()?,
                chunk_len: chunk_len.try_into().ok()?,
                chunk_stride: chunk_stride.try_into().ok()?,
                input_len: input_len.try_into().ok()?,
                k: output_len.try_into().ok()?,
            })
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

#[cfg(test)]
mod selection_tests {
    use super::*;
    use crate::kernel_selection::DeterministicShapeRng;

    #[test]
    fn topk_chunk_selector_generates_each_variant() {
        let selector = topk_chunk_selector();
        let mut rng = DeterministicShapeRng::default();

        for variant in [TopKChunkVariant::Empty, TopKChunkVariant::Kernel] {
            let shape = selector
                .generate_for(variant, &(), topk_empty_caps(), &mut rng)
                .expect("variant should generate");
            assert_eq!(
                selector.select(shape, &(), topk_empty_caps()),
                Some(variant)
            );
        }
    }

    #[test]
    fn topk_merge_selector_generates_each_variant() {
        let selector = topk_merge_selector();
        let mut rng = DeterministicShapeRng::default();

        for variant in [TopKMergeVariant::Empty, TopKMergeVariant::Kernel] {
            let shape = selector
                .generate_for(variant, &(), topk_empty_caps(), &mut rng)
                .expect("variant should generate");
            assert_eq!(
                selector.select(shape, &(), topk_empty_caps()),
                Some(variant)
            );
        }
    }

    #[test]
    fn topk_exactness_selector_generates_each_variant() {
        let selector = topk_exactness_selector();
        let mut rng = DeterministicShapeRng::default();

        for variant in [
            TopKExactnessVariant::Ineligible,
            TopKExactnessVariant::Kernel,
        ] {
            let shape = selector
                .generate_for(variant, &(), topk_empty_caps(), &mut rng)
                .expect("variant should generate");
            assert_eq!(
                selector.select(shape, &(), topk_empty_caps()),
                Some(variant)
            );
        }
    }
}
