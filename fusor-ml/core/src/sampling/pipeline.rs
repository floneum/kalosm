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

    let trace = std::env::var_os("FUSOR_TRACE_DECODE").is_some()
        || std::env::var_os("FUSOR_TRACE_SAMPLER").is_some();
    let qmat_start = trace.then(std::time::Instant::now);
    let Some(logits) = qmat_logits_data_with_encoder(hidden, matrix, &mut encoder) else {
        return Ok(None);
    };
    if let Some(start) = qmat_start {
        eprintln!(
            "sampler_trace qmat_logits_setup elapsed={:?}",
            start.elapsed()
        );
    }

    let hidden_dump_buffer = if std::env::var_os("FUSOR_DEBUG_SAMPLER").is_some() {
        let hidden_bytes = (std::mem::size_of::<f32>() * hidden_len) as u64;
        let hidden_dl = device.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            size: hidden_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
            label: Some("debug sampler hidden download"),
        });
        encoder.copy_buffer_to_buffer(hidden.buffer(), 0, &hidden_dl, 0, hidden_bytes);
        Some(hidden_dl)
    } else {
        None
    };

    let result = sample_processed_logits_to_host(
        &logits,
        sampler,
        previous_tokens,
        params,
        Some(encoder),
        "qmat mirostat2 sampled token download",
    )
    .await?;

    if result.is_none()
        && let Some(hidden_dl) = hidden_dump_buffer
    {
        let (tx, rx) = futures_channel::oneshot::channel();
        hidden_dl
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                _ = tx.send(r);
            });
        #[cfg(not(target_arch = "wasm32"))]
        device.poll_wait();
        let _ = rx.await;
        let view = hidden_dl.slice(..).get_mapped_range();
        let hidden_vec: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        hidden_dl.unmap();

        let mut nan = 0usize;
        let mut pos_inf = 0usize;
        let mut neg_inf = 0usize;
        let mut finite = 0usize;
        let mut min_h = f32::INFINITY;
        let mut max_h = f32::NEG_INFINITY;
        for &v in &hidden_vec {
            if v.is_nan() {
                nan += 1;
            } else if v == f32::INFINITY {
                pos_inf += 1;
            } else if v == f32::NEG_INFINITY {
                neg_inf += 1;
            } else {
                finite += 1;
                if v < min_h {
                    min_h = v;
                }
                if v > max_h {
                    max_h = v;
                }
            }
        }
        eprintln!(
            "sampler_debug HIDDEN len={} nan={} +inf={} -inf={} finite={} min={} max={} first8={:?}",
            hidden_vec.len(),
            nan,
            pos_inf,
            neg_inf,
            finite,
            min_h,
            max_h,
            hidden_vec.iter().take(8).collect::<Vec<_>>()
        );
    }

    Ok(result)
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
    let trace = std::env::var_os("FUSOR_TRACE_DECODE").is_some()
        || std::env::var_os("FUSOR_TRACE_SAMPLER").is_some();
    let debug_dump = std::env::var_os("FUSOR_DEBUG_SAMPLER").is_some();
    let mut attempt = 0usize;
    loop {
        attempt += 1;
        let device = input.device();
        let mut encoder = initial_encoder.take().unwrap_or_else(|| {
            device
                .wgpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("mirostat2_sample_token_to_host encoder"),
                })
        });

        let output_per_chunk = sampler_output_per_chunk(candidate_count);
        let topk_start = trace.then(std::time::Instant::now);
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
        if let Some(start) = topk_start {
            eprintln!("sampler_trace topk_setup elapsed={:?}", start.elapsed());
        }
        let merge_start = trace.then(std::time::Instant::now);
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
        if let Some(start) = merge_start {
            eprintln!("sampler_trace merge_setup elapsed={:?}", start.elapsed());
        }
        let exactness_start = trace.then(std::time::Instant::now);
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
        if let Some(start) = exactness_start {
            eprintln!(
                "sampler_trace exactness_setup elapsed={:?}",
                start.elapsed()
            );
        }
        let sample_start = trace.then(std::time::Instant::now);
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
        if let Some(start) = sample_start {
            eprintln!("sampler_trace sample_setup elapsed={:?}", start.elapsed());
        }

        let download_size = (std::mem::size_of::<u32>() * GPU_SAMPLE_RESULT_WORDS) as u64;
        let download = device.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            size: download_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
            label: Some(download_label),
        });
        encoder.copy_buffer_to_buffer(output.buffer(), 0, &download, 0, download_size);

        let debug_buffers = if debug_dump {
            let ids_bytes = (std::mem::size_of::<u32>() * top_k) as u64;
            let values_bytes = (std::mem::size_of::<f32>() * top_k) as u64;
            let logits_bytes = (std::mem::size_of::<f32>() * input_len) as u64;
            let ids_dl = device.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
                size: ids_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
                label: Some("debug sampler ids download"),
            });
            let values_dl = device.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
                size: values_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
                label: Some("debug sampler values download"),
            });
            let logits_dl = device.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
                size: logits_bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
                label: Some("debug sampler logits download"),
            });
            encoder.copy_buffer_to_buffer(ids.buffer(), 0, &ids_dl, 0, ids_bytes);
            encoder.copy_buffer_to_buffer(values.buffer(), 0, &values_dl, 0, values_bytes);
            encoder.copy_buffer_to_buffer(input.buffer(), 0, &logits_dl, 0, logits_bytes);
            Some((ids_dl, values_dl, logits_dl))
        } else {
            None
        };

        let submit_start = trace.then(std::time::Instant::now);
        device.wgpu_queue().submit(Some(encoder.finish()));
        if let Some(start) = submit_start {
            eprintln!("sampler_trace submit elapsed={:?}", start.elapsed());
        }

        let map_start = trace.then(std::time::Instant::now);
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
            eprintln!("sampler_trace map_wait elapsed={:?}", start.elapsed());
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

        match status {
            GPU_SAMPLE_STATUS_SAMPLED => {
                if trace {
                    eprintln!(
                        "sampler_trace sampled attempt={attempt} top_k={top_k} chunks={chunks} candidate_count={candidate_count} token={token}"
                    );
                }
                return Ok(Some(token));
            }
            GPU_SAMPLE_STATUS_RETRY_NEEDED => {
                if trace {
                    eprintln!(
                        "sampler_trace retry attempt={attempt} top_k={top_k} chunks={chunks} candidate_count={candidate_count}"
                    );
                }
            }
            _ => {
                if trace {
                    eprintln!(
                        "sampler_trace invalid attempt={attempt} top_k={top_k} chunks={chunks} candidate_count={candidate_count} status={status}"
                    );
                }
                if let Some((ids_dl, values_dl, logits_dl)) = debug_buffers {
                    let (id_tx, id_rx) = futures_channel::oneshot::channel();
                    ids_dl.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                        _ = id_tx.send(r);
                    });
                    let (val_tx, val_rx) = futures_channel::oneshot::channel();
                    values_dl
                        .slice(..)
                        .map_async(wgpu::MapMode::Read, move |r| {
                            _ = val_tx.send(r);
                        });
                    let (log_tx, log_rx) = futures_channel::oneshot::channel();
                    logits_dl
                        .slice(..)
                        .map_async(wgpu::MapMode::Read, move |r| {
                            _ = log_tx.send(r);
                        });
                    #[cfg(not(target_arch = "wasm32"))]
                    device.poll_wait();
                    let _ = id_rx.await;
                    let _ = val_rx.await;
                    let _ = log_rx.await;
                    let ids_view = ids_dl.slice(..).get_mapped_range();
                    let vals_view = values_dl.slice(..).get_mapped_range();
                    let logits_view = logits_dl.slice(..).get_mapped_range();
                    let ids_vec: Vec<u32> = bytemuck::cast_slice(&ids_view).to_vec();
                    let vals_vec: Vec<f32> = bytemuck::cast_slice(&vals_view).to_vec();
                    let logits_vec: Vec<f32> = bytemuck::cast_slice(&logits_view).to_vec();
                    drop(ids_view);
                    drop(vals_view);
                    drop(logits_view);
                    ids_dl.unmap();
                    values_dl.unmap();
                    logits_dl.unmap();

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
                    eprintln!(
                        "sampler_debug INVALID ids={:?} values={:?} logits_len={} nan={} +inf={} -inf={} finite={} min={} max={} argmax={} first8={:?} previous_tokens_last={:?}",
                        ids_vec,
                        vals_vec,
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
                return Ok(None);
            }
        }

        let next = next_sampler_candidate_count(candidate_count, top_k);
        if next == candidate_count {
            return Ok(None);
        }
        candidate_count = next;
    }
}
