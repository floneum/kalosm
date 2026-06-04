use super::*;

impl<F: FloatDataType + SimdElement + Default + FloatOps + MatmulImpl> LlamaModel<F>
where
    F: CastTo<f32> + CastTensor<f32> + WasmNotSend + WasmNotSync + 'static,
    f32: CastTo<F> + CastTensor<F>,
    MulOp: SimdBinaryOp<F>,
    AddOp: SimdBinaryOp<F>,
    SumOp: SimdReduceOp<F>,
{
    fn prepare_forward_logits(
        ctx: ForwardInputs<'_, F>,
        fast_path: &'static str,
        fallback_path: &'static str,
    ) -> Result<PreparedForwardLogits, LlamaModelError> {
        let ForwardInputs {
            model,
            device,
            tokens,
            images,
            cache,
            tokenizer,
        } = ctx;
        #[cfg(not(debug_assertions))]
        let _ = tokenizer;
        if tokens.is_empty() {
            return Err(LlamaModelError::EmptyInput);
        }

        #[cfg(debug_assertions)]
        {
            tracing::trace!(
                "Running model with tokens: {:?}",
                tokenizer.decode(tokens, false)
            );
        }

        let trace_enabled = decode_trace_enabled();
        let decode_eligible = tokens.len() == 1
            && images.is_empty()
            && cache.as_ref().is_some_and(|cache| !cache.tokens.is_empty());
        let path = if decode_eligible {
            fast_path
        } else {
            fallback_path
        };
        let token_start = trace_enabled.then(Instant::now);
        let build_start = trace_enabled.then(Instant::now);
        let logits = model.forward(tokens, images, device, cache);
        if let Some(start) = build_start {
            eprintln!(
                "forward_graph_build path={path} decode_eligible={decode_eligible} elapsed={:?}",
                start.elapsed()
            );
        }
        let logits = logits.map_err(LlamaModelError::from)?;
        let logits = logits.squeeze(0);
        let logits: fusor::Tensor<1, f32> = logits.cast();
        let len = logits.shape()[0];
        let mut kernels = 0;
        if trace_enabled {
            if let Some(gpu_logits) = logits.as_gpu() {
                let resolve_start = Some(Instant::now());
                kernels = gpu_logits.count_kernels_to_resolve();
                if let Some(start) = resolve_start {
                    eprintln!(
                        "forward_resolve path={path} decode_eligible={decode_eligible} kernels={kernels} elapsed={:?}",
                        start.elapsed()
                    );
                }
            } else {
                eprintln!("forward_logits_on_cpu path={path} decode_eligible={decode_eligible}");
            }
        }

        Ok(PreparedForwardLogits {
            logits,
            len,
            trace: ForwardTrace {
                enabled: trace_enabled,
                decode_eligible,
                path,
                token_start,
                kernels,
            },
        })
    }

    #[cfg(feature = "structured")]
    pub(crate) fn forward(
        model: &Model<F>,
        device: &Device,
        tokens: &[u32],
        images: &[LlamaImage],
        cache: Option<&mut LlamaCache>,
        tokenizer: &LlamaTokenizer,
    ) -> Pin<
        Box<dyn kalosm_model_types::FutureWasmNotSend<Output = Result<Vec<f32>, LlamaModelError>>>,
    > {
        let prepared = match Self::prepare_forward_logits(
            ForwardInputs {
                model,
                device,
                tokens,
                images,
                cache,
                tokenizer,
            },
            "fast_decode_graph",
            "graph_fallback",
        ) {
            Ok(prepared) => prepared,
            Err(err) => return Box::pin(async move { Err(err) }),
        };
        let PreparedForwardLogits { logits, len, trace } = prepared;
        Box::pin(async move {
            let download_start = trace.step_start();
            let logits = logits.as_slice().await?;
            if let Some(start) = download_start {
                eprintln!(
                    "forward_download path={} decode_eligible={} elapsed={:?}",
                    trace.path,
                    trace.decode_eligible,
                    start.elapsed(),
                );
            }
            trace.record();
            let mut logits_vec = Vec::with_capacity(len);
            for i in 0..len {
                let logit = logits[[i]];
                logits_vec.push(logit);
            }

            Ok(logits_vec)
        })
    }

    pub(crate) fn forward_top_k(
        model: &Model<F>,
        device: &Device,
        tokens: &[u32],
        images: &[LlamaImage],
        cache: Option<&mut LlamaCache>,
        tokenizer: &LlamaTokenizer,
        top_k: usize,
    ) -> Pin<
        Box<
            dyn kalosm_model_types::FutureWasmNotSend<Output = Result<Vec<Logit>, LlamaModelError>>,
        >,
    > {
        let prepared = match Self::prepare_forward_logits(
            ForwardInputs {
                model,
                device,
                tokens,
                images,
                cache,
                tokenizer,
            },
            "fast_decode_graph_top_k",
            "graph_fallback_top_k",
        ) {
            Ok(prepared) => prepared,
            Err(err) => return Box::pin(async move { Err(err) }),
        };
        let PreparedForwardLogits { logits, len, trace } = prepared;
        Box::pin(async move {
            let download_start = trace.step_start();
            let top_logits = if use_full_logits_for_sampling(len) {
                let logits = logits.as_slice().await?;
                let mut logits_vec = Vec::with_capacity(len);
                for i in 0..len {
                    logits_vec.push(logits[[i]]);
                }
                top_k_logits_from_full(&logits_vec, top_k)
            } else {
                logits
                    .top_k_pairs(top_k)
                    .await?
                    .into_iter()
                    .map(|(token_id, logit)| Logit {
                        token_id,
                        logit,
                        prob: 0.0,
                    })
                    .collect()
            };
            if let Some(start) = download_start {
                eprintln!(
                    "forward_top_k_download path={} decode_eligible={} k={top_k} elapsed={:?}",
                    trace.path,
                    trace.decode_eligible,
                    start.elapsed(),
                );
            }
            trace.record();

            Ok(top_logits)
        })
    }

    pub(crate) fn forward_sample_token<'a>(
        ctx: ForwardInputs<'_, F>,
        sampler: &'a mut LlamaGpuSamplerState,
        previous_tokens: Vec<u32>,
        top_k: usize,
    ) -> Pin<
        Box<dyn kalosm_model_types::FutureWasmNotSend<Output = Result<u32, LlamaModelError>> + 'a>,
    > {
        let ForwardInputs {
            model,
            device,
            tokens,
            images,
            cache,
            tokenizer,
        } = ctx;
        if tokens.is_empty() {
            return Box::pin(async { Err(LlamaModelError::EmptyInput) });
        }

        #[cfg(debug_assertions)]
        {
            tracing::trace!(
                "Running model with tokens: {:?}",
                tokenizer.decode(tokens, false)
            );
        }

        if gpu_fused_logits_sampling_enabled() {
            return Self::forward_sample_token_fused_logits(
                ForwardInputs {
                    model,
                    device,
                    tokens,
                    images,
                    cache,
                    tokenizer,
                },
                sampler,
                previous_tokens,
                top_k,
            );
        }

        let prepared = match Self::prepare_forward_logits(
            ForwardInputs {
                model,
                device,
                tokens,
                images,
                cache,
                tokenizer,
            },
            "fast_decode_graph_sample_token",
            "graph_fallback_sample_token",
        ) {
            Ok(prepared) => prepared,
            Err(err) => return Box::pin(async move { Err(err) }),
        };
        let PreparedForwardLogits { logits, trace, .. } = prepared;
        Box::pin(async move {
            let download_start = trace.step_start();
            let token_id = match sampler.config.sampling_strategy {
                kalosm_language_model::SamplingStrategy::Mirostat2 => {
                    let params = sampler.mirostat_params(top_k);
                    let mirostat = sampler.mirostat.as_mut().ok_or_else(|| {
                        LlamaModelError::SamplerError("missing Mirostat GPU sampler".into())
                    })?;
                    logits
                        .sample_mirostat2_token(mirostat, &previous_tokens, params)
                        .await?
                }
                kalosm_language_model::SamplingStrategy::Standard => {
                    let params = sampler.standard_params(top_k);
                    logits
                        .sample_standard_token(&previous_tokens, params)
                        .await?
                }
            };
            if let Some(start) = download_start {
                eprintln!(
                    "forward_sample_token_download path={} decode_eligible={} k={} elapsed={:?}",
                    trace.path,
                    trace.decode_eligible,
                    top_k,
                    start.elapsed(),
                );
            }
            trace.record();

            Ok(token_id)
        })
    }

    pub(crate) fn forward_sample_token_pending(
        ctx: ForwardInputs<'_, F>,
        sampler: &mut LlamaGpuSamplerState,
        previous_tokens: Vec<u32>,
        top_k: usize,
    ) -> Result<Option<fusor::GpuSampledToken>, LlamaModelError> {
        let ForwardInputs {
            model,
            device,
            tokens,
            images,
            cache,
            tokenizer,
        } = ctx;
        #[cfg(not(debug_assertions))]
        let _ = tokenizer;
        if tokens.is_empty() {
            return Err(LlamaModelError::EmptyInput);
        }

        #[cfg(debug_assertions)]
        {
            tracing::trace!(
                "Running model with tokens: {:?}",
                tokenizer.decode(tokens, false)
            );
        }

        if !gpu_fused_logits_sampling_enabled() {
            return Ok(None);
        }

        Self::forward_sample_token_pending_from_hidden(
            model.forward_last_hidden_f32(tokens, images, device, cache)?,
            model,
            sampler,
            previous_tokens,
            None,
            top_k,
        )
    }

    pub(crate) fn forward_sample_token_from_gpu_token_pending(
        model: &Model<F>,
        device: &Device,
        token: &fusor::GpuSampledToken,
        cache: &mut LlamaCache,
        sampler: &mut LlamaGpuSamplerState,
        previous_tokens: Vec<u32>,
        top_k: usize,
    ) -> Result<Option<(fusor::GpuSampledToken, usize)>, LlamaModelError> {
        if !gpu_fused_logits_sampling_enabled() {
            return Ok(None);
        }

        let token_tensor = token.token_tensor();
        let (hidden, cache_slot) =
            model.forward_last_hidden_f32_gpu_token(&token_tensor, device, cache)?;
        let Some(sampled) = Self::forward_sample_token_pending_from_hidden(
            hidden,
            model,
            sampler,
            previous_tokens,
            Some(token),
            top_k,
        )?
        else {
            return Ok(None);
        };
        Ok(Some((sampled, cache_slot)))
    }

    fn forward_sample_token_pending_from_hidden(
        hidden: fusor::Tensor<2, f32>,
        model: &Model<F>,
        sampler: &mut LlamaGpuSamplerState,
        previous_tokens: Vec<u32>,
        previous_gpu_token: Option<&fusor::GpuSampledToken>,
        top_k: usize,
    ) -> Result<Option<fusor::GpuSampledToken>, LlamaModelError> {
        let hidden = hidden.squeeze(0).to_concrete();
        match sampler.config.sampling_strategy {
            kalosm_language_model::SamplingStrategy::Mirostat2 => {
                let params = sampler.mirostat_params(top_k);
                let mirostat = sampler.mirostat.as_mut().ok_or_else(|| {
                    LlamaModelError::SamplerError("missing Mirostat GPU sampler".into())
                })?;
                hidden
                    .try_sample_mirostat2_token_q_mat_pending(
                        model.output_matrix(),
                        mirostat,
                        &previous_tokens,
                        previous_gpu_token,
                        params,
                    )
                    .map_err(LlamaModelError::from)
            }
            kalosm_language_model::SamplingStrategy::Standard => {
                let params = sampler.standard_params(top_k);
                hidden
                    .try_sample_standard_token_q_mat_pending(
                        model.output_matrix(),
                        &previous_tokens,
                        previous_gpu_token,
                        params,
                    )
                    .map_err(LlamaModelError::from)
            }
        }
    }

    fn forward_sample_token_fused_logits<'a>(
        ctx: ForwardInputs<'_, F>,
        sampler: &'a mut LlamaGpuSamplerState,
        previous_tokens: Vec<u32>,
        top_k: usize,
    ) -> Pin<
        Box<dyn kalosm_model_types::FutureWasmNotSend<Output = Result<u32, LlamaModelError>> + 'a>,
    > {
        let ForwardInputs {
            model,
            device,
            tokens,
            images,
            cache,
            tokenizer: _,
        } = ctx;
        let trace = decode_trace_enabled();
        let decode_eligible = tokens.len() == 1
            && images.is_empty()
            && cache.as_ref().is_some_and(|cache| !cache.tokens.is_empty());
        let path = if decode_eligible {
            "fast_decode_graph_fused_sample_token"
        } else {
            "graph_fallback_fused_sample_token"
        };
        let token_start = trace.then(Instant::now);
        let build_start = trace.then(Instant::now);
        let hidden = model.forward_last_hidden_f32(tokens, images, device, cache);
        if let Some(start) = build_start {
            eprintln!(
                "forward_graph_build path={path} decode_eligible={decode_eligible} elapsed={:?}",
                start.elapsed()
            );
        }
        let hidden = match hidden {
            Ok(hidden) => hidden,
            Err(err) => return Box::pin(async move { Err(err.into()) }),
        };
        let hidden = hidden.squeeze(0).to_concrete();
        let output_matrix = model.output_matrix().clone();
        let mut kernels = 0;
        if trace {
            if let Some(gpu_hidden) = hidden.as_gpu() {
                let resolve_start = Some(Instant::now());
                kernels = gpu_hidden.count_kernels_to_resolve();
                if let Some(start) = resolve_start {
                    eprintln!(
                        "forward_resolve path={path} decode_eligible={decode_eligible} kernels={kernels} elapsed={:?}",
                        start.elapsed()
                    );
                }
            }
        }
        Box::pin(async move {
            let sample_start = trace.then(Instant::now);
            let token_id = match sampler.config.sampling_strategy {
                kalosm_language_model::SamplingStrategy::Mirostat2 => {
                    let params = sampler.mirostat_params(top_k);
                    let mirostat = sampler.mirostat.as_mut().ok_or_else(|| {
                        LlamaModelError::SamplerError("missing Mirostat GPU sampler".into())
                    })?;
                    hidden
                        .try_sample_mirostat2_token_q_mat(
                            &output_matrix,
                            mirostat,
                            &previous_tokens,
                            params,
                        )
                        .await?
                }
                kalosm_language_model::SamplingStrategy::Standard => {
                    let params = sampler.standard_params(top_k);
                    hidden
                        .try_sample_standard_token_q_mat(&output_matrix, &previous_tokens, params)
                        .await?
                }
            }
            .ok_or_else(|| {
                LlamaModelError::SamplerError("fused logits sampler refused slow fallback".into())
            })?;
            if let Some(start) = sample_start {
                eprintln!(
                    "forward_sample_token_download path={path} decode_eligible={decode_eligible} fused_logits=1 k={} elapsed={:?}",
                    top_k,
                    start.elapsed()
                );
            }
            if let Some(start) = token_start {
                record_decode_trace(path, decode_eligible, kernels, start.elapsed());
            }

            Ok(token_id)
        })
    }
}
