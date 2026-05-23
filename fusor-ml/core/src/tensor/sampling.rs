use super::{DataTypeEnum, Tensor, TensorData};
use crate::quantized::QMatrix;

impl Tensor<1, f32> {
    pub async fn try_sample_mirostat2_token_q_mat(
        &self,
        matrix: &QMatrix,
        sampler: &mut crate::top_k::GpuMirostat2Sampler,
        previous_tokens: &[u32],
        params: crate::top_k::GpuMirostat2SamplerParams,
    ) -> Result<Option<u32>, wgpu::BufferAsyncError> {
        let (input, _) = self.data.materialize();
        crate::top_k::qmat_mirostat2_sample_token_to_host(
            &input,
            matrix,
            sampler,
            previous_tokens,
            params,
        )
        .await
    }

    pub async fn sample_mirostat2_token(
        &self,
        sampler: &mut crate::top_k::GpuMirostat2Sampler,
        previous_tokens: &[u32],
        params: crate::top_k::GpuMirostat2SamplerParams,
    ) -> Result<u32, wgpu::BufferAsyncError> {
        let (input, _) = self.data.materialize();
        if let Some(token) =
            crate::top_k::mirostat2_sample_token_to_host(&input, sampler, previous_tokens, params)
                .await?
        {
            return Ok(token);
        }

        let (ids, _) = self.top_k_pairs(params.top_k).await?;
        Ok(ids.first().copied().unwrap_or_default())
    }

    pub async fn top_k_pairs(
        &self,
        k: usize,
    ) -> Result<(Vec<u32>, Vec<f32>), wgpu::BufferAsyncError> {
        if k == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let (input, _) = self.data.materialize();
        if input.datatype() != DataTypeEnum::F32 || input.layout().rank() != 1 {
            return cpu_top_k_pairs_from_tensor_data(&input, k).await;
        }

        let input_len = input.layout().shape()[0];
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

        loop {
            let output_per_chunk = if candidate_count >= crate::top_k::TOP_K_CHUNK {
                crate::top_k::TOP_K_CHUNK
            } else {
                candidate_count + 1
            };
            let mut encoder = input.device().wgpu_device().create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("top_k_pairs encoder"),
                },
            );
            let Some((ids, values)) = crate::top_k::chunk_top_k_pair_data_with_encoder(
                &input,
                candidate_count,
                output_per_chunk,
                Some(&mut encoder),
            ) else {
                return cpu_top_k_pairs_from_tensor_data(&input, k).await;
            };
            if candidate_count >= crate::top_k::TOP_K_CHUNK {
                let Some((ids, values)) =
                    crate::top_k::merge_sorted_chunk_top_k_pair_data_with_encoder(
                        &ids,
                        &values,
                        crate::top_k::MergeSortedChunkTopKParams {
                            chunks,
                            chunk_len: crate::top_k::TOP_K_CHUNK,
                            chunk_stride: crate::top_k::TOP_K_CHUNK,
                            input_len,
                            k,
                        },
                        Some(&mut encoder),
                    )
                else {
                    return cpu_top_k_pairs_from_tensor_data(&input, k).await;
                };
                input.device().wgpu_queue().submit(Some(encoder.finish()));
                let ids = Tensor::<1, u32>::as_slice_from_tensor_data(&ids).await?;
                let values = Tensor::<1, f32>::as_slice_from_tensor_data(&values).await?;
                return Ok((ids.as_slice().to_vec(), values.as_slice().to_vec()));
            }
            let Some((merged_ids, merged_values)) =
                crate::top_k::merge_sorted_chunk_top_k_pair_data_with_encoder(
                    &ids,
                    &values,
                    crate::top_k::MergeSortedChunkTopKParams {
                        chunks,
                        chunk_len: candidate_count,
                        chunk_stride: output_per_chunk,
                        input_len,
                        k,
                    },
                    Some(&mut encoder),
                )
            else {
                return cpu_top_k_pairs_from_tensor_data(&input, k).await;
            };
            input.device().wgpu_queue().submit(Some(encoder.finish()));
            let merged_ids = Tensor::<1, u32>::as_slice_from_tensor_data(&merged_ids).await?;
            let merged_values = Tensor::<1, f32>::as_slice_from_tensor_data(&merged_values).await?;
            let chunk_values = Tensor::<1, f32>::as_slice_from_tensor_data(&values).await?;
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

            let ids = Tensor::<1, u32>::as_slice_from_tensor_data(&ids).await?;
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
        }
    }
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

    let values = Tensor::<1, f32>::as_slice_from_tensor_data(input).await?;
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
