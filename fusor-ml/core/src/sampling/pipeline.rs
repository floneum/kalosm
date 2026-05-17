use crate::{
    quantized::QMatrix,
    tensor::{DataTypeEnum, TensorData},
};
use wgpu::CommandEncoder;

use super::{
    GPU_SAMPLE_RESULT_WORDS, GPU_SAMPLE_STATUS_INVALID, GPU_SAMPLE_STATUS_RETRY_NEEDED,
    GPU_SAMPLE_STATUS_SAMPLED, GpuMirostat2Sampler, GpuMirostat2SamplerParams, TOP_K_CHUNK,
    mirostat::sample_from_sorted_top_k_data_with_encoder,
    qmat_topk::{
        initial_sampler_candidate_count, next_sampler_candidate_count,
        qmat_logits_data_with_encoder, sampler_output_per_chunk,
    },
    topk::{
        chunk_top_k_pair_data_with_processors_with_encoder,
        merge_sorted_chunk_top_k_pair_data_with_encoder, top_k_exactness_flag_data_with_encoder,
    },
};

pub(crate) async fn mirostat2_sample_token_to_host(
    input: &TensorData,
    sampler: &mut GpuMirostat2Sampler,
    previous_tokens: &[u32],
    params: GpuMirostat2SamplerParams,
) -> Result<Option<u32>, wgpu::BufferAsyncError> {
    sample_processed_logits_to_host(
        input,
        sampler,
        previous_tokens,
        params,
        None,
        "mirostat2 sampled token download",
    )
    .await
}

pub(crate) async fn qmat_mirostat2_sample_token_to_host(
    hidden: &TensorData,
    matrix: &QMatrix,
    sampler: &mut GpuMirostat2Sampler,
    previous_tokens: &[u32],
    params: GpuMirostat2SamplerParams,
) -> Result<Option<u32>, wgpu::BufferAsyncError> {
    if hidden.datatype() != DataTypeEnum::F32 || hidden.layout().rank() != 1 {
        return Ok(None);
    }
    let hidden_len = hidden.layout().shape()[0];
    let [vocab_len, hidden_matrix_len] = matrix.shape() else {
        return Ok(None);
    };
    if hidden_len != *hidden_matrix_len || *vocab_len == 0 {
        return Ok(None);
    }
    if !hidden.device().is_same_device(matrix.device()) {
        return Ok(None);
    }

    let device = hidden.device();
    let mut encoder =
        device
            .wgpu_device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("qmat_mirostat2_sample_token_to_host encoder"),
            });

    let Some(logits) = qmat_logits_data_with_encoder(hidden, matrix, &mut encoder) else {
        return Ok(None);
    };
    sample_processed_logits_to_host(
        &logits,
        sampler,
        previous_tokens,
        params,
        Some(encoder),
        "qmat mirostat2 sampled token download",
    )
    .await
}

async fn sample_processed_logits_to_host(
    input: &TensorData,
    sampler: &mut GpuMirostat2Sampler,
    previous_tokens: &[u32],
    params: GpuMirostat2SamplerParams,
    mut initial_encoder: Option<CommandEncoder>,
    download_label: &'static str,
) -> Result<Option<u32>, wgpu::BufferAsyncError> {
    if input.datatype() != DataTypeEnum::F32 || input.layout().rank() != 1 {
        return Ok(None);
    }

    let input_len = input.layout().shape()[0];
    let top_k = params.top_k.min(input_len);
    if top_k == 0 {
        return Ok(None);
    }

    let chunks = input_len.div_ceil(TOP_K_CHUNK);
    let mut candidate_count = initial_sampler_candidate_count(top_k, chunks);
    loop {
        let device = input.device();
        let mut encoder = initial_encoder.take().unwrap_or_else(|| {
            device
                .wgpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("mirostat2_sample_token_to_host encoder"),
                })
        });

        let output_per_chunk = sampler_output_per_chunk(candidate_count);
        let Some((chunk_ids, chunk_values)) = chunk_top_k_pair_data_with_processors_with_encoder(
            input,
            previous_tokens,
            params.temperature,
            params.repetition_penalty,
            candidate_count,
            output_per_chunk,
            Some(&mut encoder),
        ) else {
            return Ok(None);
        };
        let Some((ids, values)) = merge_sorted_chunk_top_k_pair_data_with_encoder(
            &chunk_ids,
            &chunk_values,
            crate::sampling::topk::MergeSortedChunkTopKParams {
                chunks,
                chunk_len: candidate_count,
                chunk_stride: output_per_chunk,
                input_len,
                k: top_k,
            },
            Some(&mut encoder),
        ) else {
            return Ok(None);
        };
        let exactness_flag = if candidate_count < top_k && candidate_count < TOP_K_CHUNK {
            let Some(flag) = top_k_exactness_flag_data_with_encoder(
                &values,
                &chunk_values,
                chunks,
                candidate_count,
                output_per_chunk,
                top_k,
                Some(&mut encoder),
            ) else {
                return Ok(None);
            };
            Some(flag)
        } else {
            None
        };
        let Some(output) = sample_from_sorted_top_k_data_with_encoder(
            &ids,
            &values,
            sampler,
            params,
            exactness_flag.as_ref(),
            Some(&mut encoder),
        ) else {
            return Ok(None);
        };

        let download_size = (std::mem::size_of::<u32>() * GPU_SAMPLE_RESULT_WORDS) as u64;
        let download = device.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            size: download_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
            label: Some(download_label),
        });
        encoder.copy_buffer_to_buffer(output.buffer(), 0, &download, 0, download_size);
        device.wgpu_queue().submit(Some(encoder.finish()));

        let (sender, receiver) = futures_channel::oneshot::channel();
        download
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                _ = sender.send(result);
            });
        #[cfg(not(target_arch = "wasm32"))]
        device.poll_wait();
        receiver.await.map_err(|_| wgpu::BufferAsyncError)??;

        let view = download.slice(..).get_mapped_range();
        let word_size = std::mem::size_of::<u32>();
        let status = view
            .get(..word_size)
            .map(bytemuck::from_bytes::<u32>)
            .copied()
            .unwrap_or(GPU_SAMPLE_STATUS_INVALID);
        let token = view
            .get(word_size..word_size * GPU_SAMPLE_RESULT_WORDS)
            .map(bytemuck::from_bytes::<u32>)
            .copied()
            .unwrap_or_default();
        drop(view);
        download.unmap();

        match status {
            GPU_SAMPLE_STATUS_SAMPLED => return Ok(Some(token)),
            GPU_SAMPLE_STATUS_RETRY_NEEDED => {}
            _ => return Ok(None),
        }

        let next = next_sampler_candidate_count(candidate_count, top_k);
        if next == candidate_count {
            return Ok(None);
        }
        candidate_count = next;
    }
}
