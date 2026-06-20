use super::{DataTypeEnum, Tensor, TensorData};
use crate::top_k::GpuSamplerRequest;

impl Tensor {
    pub async fn sample_mirostat2_token(
        &self,
        sampler: &mut crate::top_k::GpuMirostat2Sampler,
        previous_tokens: &[u32],
        params: crate::top_k::GpuMirostat2SamplerParams,
    ) -> Result<u32, wgpu::BufferAsyncError> {
        self.assert_rank::<1>();
        self.assert_datatype::<f32>();
        if let Some(token) = crate::top_k::sample_token_to_host(
            &self.data,
            GpuSamplerRequest::Mirostat2 { sampler, params },
            previous_tokens,
        )
        .await?
        {
            return Ok(token);
        }

        let (ids, _) = self.top_k_pairs(params.top_k).await?;
        Ok(ids.first().copied().unwrap_or_default())
    }

    pub async fn sample_standard_token(
        &self,
        previous_tokens: &[u32],
        params: crate::top_k::GpuStandardSamplerParams,
    ) -> Result<u32, wgpu::BufferAsyncError> {
        self.assert_rank::<1>();
        self.assert_datatype::<f32>();
        if let Some(token) = crate::top_k::sample_token_to_host(
            &self.data,
            GpuSamplerRequest::Standard { params },
            previous_tokens,
        )
        .await?
        {
            return Ok(token);
        }

        let (ids, _) = self.top_k_pairs(params.top_k).await?;
        Ok(ids.first().copied().unwrap_or_default())
    }

    pub fn sample_mirostat2_token_pending(
        &self,
        sampler: &mut crate::top_k::GpuMirostat2Sampler,
        previous_tokens: &[u32],
        previous_gpu_token: Option<&Tensor>,
        params: crate::top_k::GpuMirostat2SamplerParams,
    ) -> Option<crate::top_k::PendingGpuSampledToken> {
        self.assert_rank::<1>();
        self.assert_datatype::<f32>();
        crate::top_k::sample_token_pending(
            &self.data,
            GpuSamplerRequest::Mirostat2 { sampler, params },
            previous_tokens,
            previous_gpu_token,
        )
    }

    pub fn sample_standard_token_pending(
        &self,
        previous_tokens: &[u32],
        previous_gpu_token: Option<&Tensor>,
        params: crate::top_k::GpuStandardSamplerParams,
    ) -> Option<crate::top_k::PendingGpuSampledToken> {
        self.assert_rank::<1>();
        self.assert_datatype::<f32>();
        crate::top_k::sample_token_pending(
            &self.data,
            GpuSamplerRequest::Standard { params },
            previous_tokens,
            previous_gpu_token,
        )
    }

    pub async fn top_k_pairs(
        &self,
        k: usize,
    ) -> Result<(Vec<u32>, Vec<f32>), wgpu::BufferAsyncError> {
        if k == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        if self.datatype() != DataTypeEnum::F32 || self.rank() != 1 {
            let (input, _) = self.data.materialize();
            return cpu_top_k_pairs_from_tensor_data(&input, k).await;
        }

        let input_len = self.shape()[0];
        let k = k.min(input_len);
        if k == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let chunks = input_len.div_ceil(crate::top_k::TOP_K_CHUNK);
        let mut candidate_count = k
            .div_ceil(chunks)
            .max(crate::top_k::min_top_k_candidates_per_chunk())
            .min(k)
            .min(crate::top_k::TOP_K_CHUNK);

        // The first attempt's chunk + merge kernels ride the resolver's
        // encoder, so the graph producing the logits and the top-k selection
        // land in one submission. Retries with escalated candidate counts
        // re-encode over the materialized input.
        let (input, _, mut attempt) = self.data.materialize_with_tail(|input, encoder| {
            encode_top_k_attempt(input, k, input_len, chunks, candidate_count, encoder)
        });

        loop {
            let Some(TopKAttempt {
                chunk_ids,
                chunk_values,
                merged_ids,
                merged_values,
                exhaustive,
            }) = attempt
            else {
                return cpu_top_k_pairs_from_tensor_data(&input, k).await;
            };
            if exhaustive {
                let ids = Tensor::as_slice_from_tensor_data::<1, u32>(&merged_ids).await?;
                let values = Tensor::as_slice_from_tensor_data::<1, f32>(&merged_values).await?;
                return Ok((ids.as_slice().to_vec(), values.as_slice().to_vec()));
            }
            let output_per_chunk = candidate_count + 1;
            let merged_ids = Tensor::as_slice_from_tensor_data::<1, u32>(&merged_ids).await?;
            let merged_values = Tensor::as_slice_from_tensor_data::<1, f32>(&merged_values).await?;
            let chunk_values = Tensor::as_slice_from_tensor_data::<1, f32>(&chunk_values).await?;
            let exact = top_k_chunk_bounds_prove_exact(
                merged_values.as_slice(),
                chunk_values.as_slice(),
                k,
                chunks,
                candidate_count,
                output_per_chunk,
            );
            if exact {
                return Ok((
                    merged_ids.as_slice().to_vec(),
                    merged_values.as_slice().to_vec(),
                ));
            }

            let ids = Tensor::as_slice_from_tensor_data::<1, u32>(&chunk_ids).await?;
            if let Some(top) = top_k_from_chunk_candidates(
                ids.as_slice(),
                chunk_values.as_slice(),
                k,
                input_len,
                chunks,
                candidate_count,
                output_per_chunk,
            ) {
                return Ok(top.into_iter().unzip());
            }

            if candidate_count >= crate::top_k::TOP_K_CHUNK {
                return cpu_top_k_pairs_from_tensor_data(&input, k).await;
            }
            candidate_count = (candidate_count * 2).min(crate::top_k::TOP_K_CHUNK);

            let mut encoder = input.device().wgpu_device().create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("top_k_pairs retry encoder"),
                },
            );
            attempt =
                encode_top_k_attempt(&input, k, input_len, chunks, candidate_count, &mut encoder);
            input.device().wgpu_queue().submit(Some(encoder.finish()));
        }
    }
}

struct TopKAttempt {
    chunk_ids: TensorData,
    chunk_values: TensorData,
    merged_ids: TensorData,
    merged_values: TensorData,
    /// Every chunk contributed all of its elements: the merge result is the
    /// exact top-k with no host-side exactness check needed.
    exhaustive: bool,
}

/// Encode one chunked top-k + merge attempt into `encoder`. Runs inside the
/// resolver tail while the graph lock is held, so it must only touch raw
/// buffers — no compute-graph access.
fn encode_top_k_attempt(
    input: &TensorData,
    k: usize,
    input_len: usize,
    chunks: usize,
    candidate_count: usize,
    encoder: &mut wgpu::CommandEncoder,
) -> Option<TopKAttempt> {
    let exhaustive = candidate_count >= crate::top_k::TOP_K_CHUNK;
    let output_per_chunk = if exhaustive {
        crate::top_k::TOP_K_CHUNK
    } else {
        candidate_count + 1
    };
    let (chunk_ids, chunk_values) = crate::top_k::chunk_top_k_pair_data_with_encoder(
        input,
        candidate_count,
        output_per_chunk,
        Some(encoder),
    )?;
    let (merged_ids, merged_values) =
        crate::top_k::merge_sorted_chunk_top_k_pair_data_with_encoder(
            &chunk_ids,
            &chunk_values,
            crate::top_k::MergeSortedChunkTopKParams {
                chunks,
                chunk_len: if exhaustive {
                    crate::top_k::TOP_K_CHUNK
                } else {
                    candidate_count
                },
                chunk_stride: output_per_chunk,
                input_len,
                k,
            },
            Some(encoder),
        )?;
    Some(TopKAttempt {
        chunk_ids,
        chunk_values,
        merged_ids,
        merged_values,
        exhaustive,
    })
}

fn top_k_chunk_bounds_prove_exact(
    top_values: &[f32],
    chunk_values: &[f32],
    k: usize,
    chunks: usize,
    candidate_count: usize,
    output_per_chunk: usize,
) -> bool {
    let Some(&threshold) = top_values.get(k.saturating_sub(1)) else {
        return !chunk_bounds(chunk_values, chunks, candidate_count, output_per_chunk)
            .any(|bound| bound.is_finite());
    };
    if !threshold.is_finite() {
        return !chunk_bounds(chunk_values, chunks, candidate_count, output_per_chunk)
            .any(|bound| bound.is_finite());
    }
    !chunk_bounds(chunk_values, chunks, candidate_count, output_per_chunk)
        .any(|bound| bound.is_finite() && bound >= threshold)
}

fn chunk_bounds(
    values: &[f32],
    chunks: usize,
    candidate_count: usize,
    output_per_chunk: usize,
) -> impl Iterator<Item = f32> + '_ {
    (0..chunks).filter_map(move |chunk| {
        let index = chunk
            .checked_mul(output_per_chunk)?
            .checked_add(candidate_count)?;
        values.get(index).copied()
    })
}

fn top_k_from_chunk_candidates(
    ids: &[u32],
    values: &[f32],
    k: usize,
    input_len: usize,
    chunks: usize,
    candidate_count: usize,
    output_per_chunk: usize,
) -> Option<Vec<(u32, f32)>> {
    let mut candidates = Vec::with_capacity(chunks * candidate_count);
    let mut bounds = Vec::with_capacity(chunks);

    for chunk in 0..chunks {
        let base = chunk * output_per_chunk;
        for rank in 0..candidate_count.min(output_per_chunk) {
            let index = base + rank;
            let logit = values[index];
            if logit.is_finite() && (ids[index] as usize) < input_len {
                candidates.push((ids[index], logit));
            }
        }
        if candidate_count < crate::top_k::TOP_K_CHUNK {
            let index = base + candidate_count;
            let valid = (ids[index] as usize) < input_len;
            bounds.push(valid.then_some(values[index]));
        }
    }

    candidates.sort_unstable_by_key(|(token_id, _)| *token_id);
    let top = fold_top_k_pairs(candidates, k);
    let Some((_, threshold)) = top.get(k.saturating_sub(1)).copied() else {
        if bounds.iter().flatten().any(|bound| bound.is_finite()) {
            return None;
        }
        return Some(top);
    };

    if candidate_count < crate::top_k::TOP_K_CHUNK
        && bounds
            .iter()
            .flatten()
            .any(|bound| bound.is_finite() && *bound >= threshold)
    {
        return None;
    }

    Some(top)
}

fn fold_top_k_pairs(candidates: impl IntoIterator<Item = (u32, f32)>, k: usize) -> Vec<(u32, f32)> {
    let mut top = Vec::<(u32, f32)>::with_capacity(k);
    for (token_id, logit) in candidates {
        if !logit.is_finite() {
            continue;
        }
        if top.len() == k {
            let Some((last_token_id, last_logit)) = top.last().copied() else {
                continue;
            };
            if logit > last_logit || (logit == last_logit && token_id > last_token_id) {
                top.truncate(k - 1);
            } else {
                continue;
            }
        }
        let insert = top.partition_point(|(existing_id, value)| {
            *value > logit || (*value == logit && *existing_id > token_id)
        });
        top.insert(insert, (token_id, logit));
    }
    top
}

async fn cpu_top_k_pairs_from_tensor_data(
    input: &TensorData,
    k: usize,
) -> Result<(Vec<u32>, Vec<f32>), wgpu::BufferAsyncError> {
    if k == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    let values = Tensor::as_slice_from_tensor_data::<1, f32>(input).await?;
    let top = fold_top_k_pairs(
        values
            .as_slice()
            .iter()
            .copied()
            .enumerate()
            .map(|(token_id, logit)| (token_id as u32, logit)),
        k,
    );
    Ok(top.into_iter().unzip())
}
