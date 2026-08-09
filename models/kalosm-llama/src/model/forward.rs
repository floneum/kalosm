use super::*;

impl LlamaModel {
    fn prepare_forward_logits(
        ctx: ForwardInputs<'_>,
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
        if !images.is_empty() {
            return Err(LlamaModelError::MediaUnsupported);
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
            && cache.as_ref().is_some_and(|cache| !cache.tokens.is_empty());
        let path = if decode_eligible {
            fast_path
        } else {
            fallback_path
        };
        let token_start = trace_enabled.then(Instant::now);
        let build_start = trace_enabled.then(Instant::now);
        let mut cache = cache;
        let logits = model.forward(tokens, device, cache.as_deref_mut());
        if let Some(start) = build_start {
            tracing::info!(
                "forward_graph_build path={path} decode_eligible={decode_eligible} elapsed={:?}",
                start.elapsed()
            );
        }
        let logits = logits.map_err(LlamaModelError::from)?;
        // The KV caches were resolved and committed inside `forward`: each
        // step's scatter output buffer was adopted by its cache leaf, so the
        // next step reuses the same graph — no detach, no host round trip.
        let _ = &cache;
        // The logits are `[1, vocab]`; row-major readback is the flat row.
        let len = logits
            .dim(logits.rank() - 1)
            .as_const()
            .expect("logits length is constant") as usize;

        Ok(PreparedForwardLogits {
            logits,
            len,
            trace: ForwardTrace {
                enabled: trace_enabled,
                decode_eligible,
                path,
                token_start,
                kernels: 0,
            },
        })
    }

    #[cfg(feature = "structured")]
    pub(crate) fn forward(
        model: &Model,
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
            let logits_vec = logits.to_vec_f32().map_err(LlamaModelError::from)?;
            debug_assert_eq!(logits_vec.len(), len);
            if let Some(start) = download_start {
                tracing::info!(
                    "forward_download path={} decode_eligible={} elapsed={:?}",
                    trace.path,
                    trace.decode_eligible,
                    start.elapsed(),
                );
            }
            trace.record();
            Ok(logits_vec)
        })
    }

    pub(crate) fn forward_top_k(
        model: &Model,
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
                let logits_vec = logits.to_vec_f32().map_err(LlamaModelError::from)?;
                top_k_logits_from_full(&logits_vec, top_k)
            } else {
                let k = top_k.clamp(1, len) as u32;
                let (values, ids) =
                    fusor2::sampling::top_k_pairs(&logits, k).map_err(LlamaModelError::from)?;
                let values = values.to_vec_f32().map_err(LlamaModelError::from)?;
                let ids = ids.to_vec_u32().map_err(LlamaModelError::from)?;
                ids.into_iter()
                    .zip(values)
                    .map(|(token_id, logit)| Logit {
                        token_id,
                        logit,
                        prob: 0.0,
                    })
                    .collect()
            };
            if let Some(start) = download_start {
                tracing::info!(
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

}
