use crate::{
    Layout, Tensor,
    mir::kernel_backend::standard_sampler::{
        sample_categorical_logits_data_with_encoder, supports_unfiltered_categorical,
    },
    tensor::{DataTypeEnum, LazyTensorData, TensorData},
};
use web_time::Instant;

use wgpu::CommandEncoder;

use super::{
    GPU_SAMPLE_RESULT_WORDS, GPU_SAMPLE_STATUS_INVALID, GPU_SAMPLE_STATUS_RETRY_NEEDED,
    GPU_SAMPLE_STATUS_SAMPLED, GpuMirostat2Sampler, GpuMirostat2SamplerParams,
    GpuStandardSamplerParams, PendingGpuSampledToken, TOP_K_CHUNK, min_top_k_candidates_per_chunk,
    mirostat::sample_from_sorted_top_k_data_with_encoder,
    standard_sampler::sample_from_sorted_top_k_data_with_encoder as sample_standard_from_sorted_top_k_data_with_encoder,
    topk::{
        ProcessorSettings, chunk_top_k_pair_data_with_processors_and_gpu_tail_with_encoder,
        merge_sorted_chunk_top_k_pair_data_with_encoder, top_k_exactness_flag_data_with_encoder,
    },
};

/// Which sampler kernel terminates the top-k tail, along with its parameters.
/// The chunked top-k, merge, and exactness stages are shared between kinds.
pub(crate) enum GpuSamplerRequest<'a> {
    Mirostat2 {
        sampler: &'a mut GpuMirostat2Sampler,
        params: GpuMirostat2SamplerParams,
    },
    Standard {
        params: GpuStandardSamplerParams,
    },
}

impl GpuSamplerRequest<'_> {
    fn top_k(&self) -> usize {
        match self {
            Self::Mirostat2 { params, .. } => params.top_k,
            Self::Standard { params } => params.top_k,
        }
    }

    fn processor_settings(&self) -> ProcessorSettings {
        match self {
            Self::Mirostat2 { params, .. } => ProcessorSettings {
                temperature: params.temperature,
                repetition_penalty: params.repetition_penalty,
            },
            Self::Standard { params } => ProcessorSettings {
                temperature: params.temperature,
                repetition_penalty: params.repetition_penalty,
            },
        }
    }

    fn encode_sample(
        &mut self,
        ids: &TensorData,
        values: &TensorData,
        exactness_flag: Option<&TensorData>,
        encoder: &mut CommandEncoder,
    ) -> Option<TensorData> {
        match self {
            Self::Mirostat2 { sampler, params } => sample_from_sorted_top_k_data_with_encoder(
                ids,
                values,
                sampler,
                *params,
                exactness_flag,
                Some(encoder),
            ),
            Self::Standard { params } => sample_standard_from_sorted_top_k_data_with_encoder(
                ids,
                values,
                *params,
                exactness_flag,
                Some(encoder),
            ),
        }
    }
}

#[derive(Clone, Copy)]
struct SampleAttemptDims {
    top_k: usize,
    candidate_count: usize,
    chunks: usize,
    input_len: usize,
}

struct EncodedSampleAttempt {
    output: TensorData,
    debug_top_k: Option<(TensorData, TensorData)>,
}

fn initial_sampler_candidate_count(
    config: &fusor_tile_ir_runtime::FusorConfig,
    top_k: usize,
    chunks: usize,
) -> usize {
    top_k
        .div_ceil(chunks)
        .max(min_top_k_candidates_per_chunk(config))
        .min(top_k)
        .min(TOP_K_CHUNK)
}

fn sampler_output_per_chunk(candidate_count: usize) -> usize {
    if candidate_count >= TOP_K_CHUNK {
        TOP_K_CHUNK
    } else {
        candidate_count + 1
    }
}

fn next_sampler_candidate_count(candidate_count: usize, top_k: usize) -> usize {
    candidate_count
        .saturating_mul(2)
        .min(top_k)
        .min(TOP_K_CHUNK)
}

fn sampler_trace_enabled(config: &fusor_tile_ir_runtime::FusorConfig) -> bool {
    cfg!(target_arch = "wasm32") || config.trace_decode || config.trace_sampler
}

/// Encode one full sampling attempt into `encoder`. Complete, unfiltered
/// Standard requests over one workgroup take a direct categorical kernel;
/// every other request retains the processed chunk-top-k pipeline.
///
/// This runs inside the resolver tail while the graph lock is held, so it
/// must only touch raw buffers — no compute-graph access.
fn encode_sample_attempt(
    logits: &TensorData,
    previous_tokens: &[u32],
    previous_gpu_token: Option<&TensorData>,
    request: &mut GpuSamplerRequest<'_>,
    dims: SampleAttemptDims,
    encoder: &mut CommandEncoder,
) -> Option<EncodedSampleAttempt> {
    if previous_tokens.is_empty()
        && previous_gpu_token.is_none()
        && let GpuSamplerRequest::Standard { params } = request
        && supports_unfiltered_categorical(dims.input_len, *params)
        && let Some(output) =
            sample_categorical_logits_data_with_encoder(logits, *params, Some(encoder))
    {
        return Some(EncodedSampleAttempt {
            output,
            debug_top_k: None,
        });
    }

    let output_per_chunk = sampler_output_per_chunk(dims.candidate_count);
    let (chunk_ids, chunk_values) =
        chunk_top_k_pair_data_with_processors_and_gpu_tail_with_encoder(
            logits,
            previous_tokens,
            previous_gpu_token,
            request.processor_settings(),
            dims.candidate_count,
            output_per_chunk,
            Some(encoder),
        )?;

    let (ids, values) = merge_sorted_chunk_top_k_pair_data_with_encoder(
        &chunk_ids,
        &chunk_values,
        crate::sampling::topk::MergeSortedChunkTopKParams {
            chunks: dims.chunks,
            chunk_len: dims.candidate_count,
            chunk_stride: output_per_chunk,
            input_len: dims.input_len,
            k: dims.top_k,
        },
        Some(encoder),
    )?;

    let exactness_flag = if dims.candidate_count < dims.top_k && dims.candidate_count < TOP_K_CHUNK
    {
        Some(top_k_exactness_flag_data_with_encoder(
            &values,
            &chunk_values,
            dims.chunks,
            dims.candidate_count,
            output_per_chunk,
            dims.top_k,
            Some(encoder),
        )?)
    } else {
        None
    };

    let output = request.encode_sample(&ids, &values, exactness_flag.as_ref(), encoder)?;
    Some(EncodedSampleAttempt {
        output,
        debug_top_k: Some((ids, values)),
    })
}

fn encode_token_download(
    output: &TensorData,
    encoder: &mut CommandEncoder,
    label: &'static str,
) -> wgpu::Buffer {
    let device = output.device();
    let download_size = (std::mem::size_of::<u32>() * GPU_SAMPLE_RESULT_WORDS) as u64;
    let download = device.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
        size: download_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
        label: Some(label),
    });
    encoder.copy_buffer_to_buffer(output.buffer(), 0, &download, 0, download_size);
    download
}

async fn read_sample_result(
    device: &crate::Device,
    download: wgpu::Buffer,
    trace: bool,
) -> Result<(u32, u32), wgpu::BufferAsyncError> {
    #[cfg(target_arch = "wasm32")]
    let _ = device;
    let map_start = trace.then(Instant::now);
    let (sender, receiver) = futures_channel::oneshot::channel();
    download
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            _ = sender.send(result);
        });
    #[cfg(not(target_arch = "wasm32"))]
    device.poll_wait();
    receiver.await.map_err(|_| wgpu::BufferAsyncError)??;
    if let Some(start) = map_start {
        tracing::info!("sampler_trace map_wait elapsed={:?}", start.elapsed());
    }

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
    Ok((status, token))
}

/// Sample a token from a lazy 1-D logits tensor, downloading the result.
///
/// The first attempt's kernels ride the resolver's command encoder, so the
/// graph that produces the logits (including a recognized lm_head qgemv) and
/// the sampling tail land in a single submission. Retries with escalated
/// candidate counts re-encode over the already-materialized logits.
pub(crate) async fn sample_token_to_host(
    logits: &LazyTensorData,
    mut request: GpuSamplerRequest<'_>,
    previous_tokens: &[u32],
) -> Result<Option<u32>, wgpu::BufferAsyncError> {
    if logits.info.datatype() != DataTypeEnum::F32 || logits.info.rank() != 1 {
        return Ok(None);
    }
    let input_len = logits.info.shape()[0];
    let top_k = request.top_k().min(input_len);
    if top_k == 0 {
        return Ok(None);
    }

    let chunks = input_len.div_ceil(TOP_K_CHUNK);
    let config = logits.device.config();
    let mut candidate_count = initial_sampler_candidate_count(config, top_k, chunks);
    let trace = sampler_trace_enabled(config);
    let debug_dump = config.debug_sampler;

    let (logits_data, _, mut attempt) = logits.materialize_with_tail(|logits_data, encoder| {
        let attempt = encode_sample_attempt(
            logits_data,
            previous_tokens,
            None,
            &mut request,
            SampleAttemptDims {
                top_k,
                candidate_count,
                chunks,
                input_len,
            },
            encoder,
        )?;
        let download = encode_token_download(&attempt.output, encoder, "sampled token download");
        Some((attempt, download))
    });
    let device = logits_data.device().clone();

    let mut attempt_index = 0usize;
    loop {
        attempt_index += 1;
        let Some((attempt_data, download)) = attempt else {
            return Ok(None);
        };
        let debug_top_k = attempt_data.debug_top_k;
        let (status, token) = read_sample_result(&device, download, trace).await?;
        match status {
            GPU_SAMPLE_STATUS_SAMPLED => {
                if trace {
                    tracing::info!(
                        "sampler_trace sampled attempt={attempt_index} top_k={top_k} chunks={chunks} candidate_count={candidate_count} token={token}"
                    );
                }
                return Ok(Some(token));
            }
            GPU_SAMPLE_STATUS_RETRY_NEEDED => {
                if trace {
                    tracing::info!(
                        "sampler_trace retry attempt={attempt_index} top_k={top_k} chunks={chunks} candidate_count={candidate_count}"
                    );
                }
            }
            _ => {
                if trace {
                    tracing::warn!(
                        "sampler_trace invalid attempt={attempt_index} top_k={top_k} chunks={chunks} candidate_count={candidate_count} status={status}"
                    );
                }
                if debug_dump && let Some((ids, values)) = debug_top_k {
                    debug_dump_invalid_sample(&logits_data, &ids, &values, previous_tokens).await;
                }
                return Ok(None);
            }
        }

        let next = next_sampler_candidate_count(candidate_count, top_k);
        if next == candidate_count {
            return Ok(None);
        }
        candidate_count = next;

        let mut encoder =
            device
                .wgpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("sampler retry encoder"),
                });
        attempt = encode_sample_attempt(
            &logits_data,
            previous_tokens,
            None,
            &mut request,
            SampleAttemptDims {
                top_k,
                candidate_count,
                chunks,
                input_len,
            },
            &mut encoder,
        )
        .map(|attempt| {
            let download =
                encode_token_download(&attempt.output, &mut encoder, "sampled token download");
            (attempt, download)
        });
        device.wgpu_queue().submit(Some(encoder.finish()));
    }
}

/// Sample a token from a lazy 1-D logits tensor without waiting for the
/// result. The whole tail — including the optional copy of the previously
/// sampled GPU token into the repetition-penalty window — is appended to the
/// resolver's submission, and the token stays on the GPU as a 1-element
/// tensor for the next decode step.
pub(crate) fn sample_token_pending(
    logits: &LazyTensorData,
    mut request: GpuSamplerRequest<'_>,
    previous_tokens: &[u32],
    previous_gpu_token: Option<&Tensor>,
) -> Option<PendingGpuSampledToken> {
    if logits.info.datatype() != DataTypeEnum::F32 || logits.info.rank() != 1 {
        return None;
    }
    let input_len = logits.info.shape()[0];
    let top_k = request.top_k().min(input_len);
    if top_k == 0 || top_k > TOP_K_CHUNK {
        return None;
    }

    let previous_gpu_token = previous_gpu_token.and_then(materialize_gpu_previous_token);
    if previous_gpu_token
        .as_ref()
        .is_some_and(|token| !logits.device.is_same_device(token.device()))
    {
        return None;
    }

    let dims = SampleAttemptDims {
        top_k,
        candidate_count: top_k,
        chunks: input_len.div_ceil(TOP_K_CHUNK),
        input_len,
    };
    let (_, _, tail) = logits.materialize_with_tail(|logits_data, encoder| {
        let attempt = encode_sample_attempt(
            logits_data,
            previous_tokens,
            previous_gpu_token.as_ref(),
            &mut request,
            dims,
            encoder,
        )?;
        let download =
            encode_token_download(&attempt.output, encoder, "pending sampled token download");
        Some((attempt.output, download))
    });
    let (output, download) = tail?;

    let (sender, receiver) = futures_channel::oneshot::channel();
    download
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            _ = sender.send(result);
        });

    // The token lives at word 1 of the `[status, token]` output buffer.
    let token = Tensor::from(TensorData::new_from_parts(
        output.device(),
        output.buffer().clone(),
        Layout::from_parts(1, Box::new([1]), Box::new([1])),
        DataTypeEnum::U32,
    ));
    Some(PendingGpuSampledToken::new(token, download, receiver))
}

fn materialize_gpu_previous_token(token: &Tensor) -> Option<TensorData> {
    if token.datatype() != DataTypeEnum::U32
        || token.rank() != 1
        || token.shape().first().copied().unwrap_or_default() == 0
    {
        return None;
    }
    let (data, _) = token.data.materialize();
    Some(data)
}

async fn debug_dump_invalid_sample(
    logits: &TensorData,
    ids: &TensorData,
    values: &TensorData,
    previous_tokens: &[u32],
) {
    let Ok(ids_slice) = Tensor::as_slice_from_tensor_data::<1, u32>(ids).await else {
        return;
    };
    let Ok(values_slice) = Tensor::as_slice_from_tensor_data::<1, f32>(values).await else {
        return;
    };
    let Ok(logits_slice) = Tensor::as_slice_from_tensor_data::<1, f32>(logits).await else {
        return;
    };
    let logits_vec = logits_slice.as_slice();

    let mut nan_count = 0usize;
    let mut inf_pos = 0usize;
    let mut inf_neg = 0usize;
    let mut finite_count = 0usize;
    let mut min_f = f32::INFINITY;
    let mut max_f = f32::NEG_INFINITY;
    let mut argmax = 0usize;
    for (i, &v) in logits_vec.iter().enumerate() {
        if v.is_nan() {
            nan_count += 1;
        } else if v == f32::INFINITY {
            inf_pos += 1;
        } else if v == f32::NEG_INFINITY {
            inf_neg += 1;
        } else {
            finite_count += 1;
            if v < min_f {
                min_f = v;
            }
            if v > max_f {
                max_f = v;
                argmax = i;
            }
        }
    }
    tracing::warn!(
        "sampler_debug INVALID ids={:?} values={:?} logits_len={} nan={} +inf={} -inf={} finite={} min={} max={} argmax={} first8={:?} previous_tokens_last={:?}",
        ids_slice.as_slice(),
        values_slice.as_slice(),
        logits_vec.len(),
        nan_count,
        inf_pos,
        inf_neg,
        finite_count,
        min_f,
        max_f,
        argmax,
        logits_vec.iter().take(8).collect::<Vec<_>>(),
        previous_tokens.iter().rev().take(8).collect::<Vec<_>>()
    );
}
