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
            mut cache,
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
        let token_start = trace_enabled.then(std::time::Instant::now);
        let build_start = trace_enabled.then(std::time::Instant::now);
        let logits = model.forward(tokens, images, device, cache.as_deref_mut());
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
        if let Some(logits_key) = logits.gpu_key() {
            let resolve_start = trace_enabled.then(std::time::Instant::now);
            kernels = device.resolve_batch(&[logits_key]);
            if let Some(start) = resolve_start {
                eprintln!(
                    "forward_resolve path={path} decode_eligible={decode_eligible} kernels={kernels} elapsed={:?}",
                    start.elapsed()
                );
            }
            if let Some(cache) = cache {
                cache.detach(device);
            }
        } else if trace_enabled {
            eprintln!("forward_logits_on_cpu path={path} decode_eligible={decode_eligible}");
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
        sampler: &'a mut fusor::Mirostat2Sampler,
        previous_tokens: Vec<u32>,
        params: fusor::Mirostat2SamplerParams,
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
                params,
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
            let token_id = logits
                .sample_mirostat2_token(sampler, &previous_tokens, params)
                .await?;
            if let Some(start) = download_start {
                eprintln!(
                    "forward_sample_token_download path={} decode_eligible={} k={} elapsed={:?}",
                    trace.path,
                    trace.decode_eligible,
                    params.top_k,
                    start.elapsed(),
                );
            }
            trace.record();

            Ok(token_id)
        })
    }

    fn forward_sample_token_fused_logits<'a>(
        ctx: ForwardInputs<'_, F>,
        sampler: &'a mut fusor::Mirostat2Sampler,
        previous_tokens: Vec<u32>,
        params: fusor::Mirostat2SamplerParams,
    ) -> Pin<
        Box<dyn kalosm_model_types::FutureWasmNotSend<Output = Result<u32, LlamaModelError>> + 'a>,
    > {
        let ForwardInputs {
            model,
            device,
            tokens,
            images,
            mut cache,
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
        let token_start = trace.then(std::time::Instant::now);
        let build_start = trace.then(std::time::Instant::now);
        let hidden = model.forward_last_hidden_f32(tokens, images, device, cache.as_deref_mut());
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
        if let Some(hidden_key) = hidden.gpu_key() {
            let resolve_start = trace.then(std::time::Instant::now);
            kernels = device.resolve_batch(&[hidden_key]);
            if let Some(start) = resolve_start {
                eprintln!(
                    "forward_resolve path={path} decode_eligible={decode_eligible} kernels={kernels} elapsed={:?}",
                    start.elapsed()
                );
            }
            if let Some(cache) = cache {
                cache.detach(device);
            }
        }
        Box::pin(async move {
            let sample_start = trace.then(std::time::Instant::now);
            let token_id = match hidden
                .try_sample_mirostat2_token_q_mat(&output_matrix, sampler, &previous_tokens, params)
                .await?
            {
                Some(token_id) => token_id,
                None => {
                    return Err(LlamaModelError::SamplerError(
                        "fused logits sampler refused slow fallback".into(),
                    ));
                }
            };
            if let Some(start) = sample_start {
                eprintln!(
                    "forward_sample_token_download path={path} decode_eligible={decode_eligible} fused_logits=1 k={} elapsed={:?}",
                    params.top_k,
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
