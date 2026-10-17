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
        let input_len = tokens.len()
            + usize::from(
                cache
                    .as_ref()
                    .is_some_and(|cache| cache.pending_token.is_some()),
            );
        if input_len == 0 {
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
        let decode_eligible =
            input_len == 1 && cache.as_ref().is_some_and(|cache| !cache.tokens.is_empty());
        let path = if decode_eligible {
            fast_path
        } else {
            fallback_path
        };
        let token_start = trace_enabled.then(Instant::now);
        let build_start = trace_enabled.then(Instant::now);
        let mut cache = cache;
        #[cfg(feature = "kernel-visualize")]
        let host_stage = std::env::var_os("PROFILE_HOST_STAGES")
            .is_some()
            .then(Instant::now);
        let logits = model.forward(tokens, images, device, cache.as_deref_mut());
        #[cfg(feature = "kernel-visualize")]
        if let Some(start) = host_stage {
            eprintln!(
                "HOST_STAGE name=prepare_submit us={}",
                start.elapsed().as_micros()
            );
        }
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
        let len = logits.dim(1);

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
            let logits_vec = logits
                .to_vec_f32_async()
                .await
                .map_err(LlamaModelError::from)?;
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
        let profile = device.execution_profile();
        let PreparedForwardLogits { logits, len, trace } = prepared;
        Box::pin(async move {
            let download_start = trace.step_start();
            let top_logits = if use_full_logits_for_sampling(len) {
                #[cfg(feature = "kernel-visualize")]
                let host_stage = std::env::var_os("PROFILE_HOST_STAGES")
                    .is_some()
                    .then(Instant::now);
                #[cfg(feature = "kernel-visualize")]
                let sample_bytes = std::env::var_os("PROFILE_TOP_K_BYTES").is_some()
                    && logits.as_dyn().dtype() == fusor::Dtype::F32;
                #[cfg(not(feature = "kernel-visualize"))]
                let sample_bytes = false;
                // The readback already owns contiguous F32 bytes. Sampling
                // through a checked borrowed slice avoids a second full-vocab
                // allocation and copy. Other dtypes retain numeric widening.
                let raw = if sample_bytes {
                    Some(
                        logits
                            .as_dyn()
                            .to_bytes_async()
                            .await
                            .map_err(LlamaModelError::from)?,
                    )
                } else {
                    None
                };
                let logits_vec = if let Some(raw) = &raw {
                    f32_logits_from_bytes(raw)?
                } else {
                    std::borrow::Cow::Owned(
                        logits
                            .to_vec_f32_async()
                            .await
                            .map_err(LlamaModelError::from)?,
                    )
                };
                #[cfg(feature = "kernel-visualize")]
                if let Some(start) = host_stage {
                    eprintln!(
                        "HOST_STAGE name=readback_wait us={}",
                        start.elapsed().as_micros()
                    );
                }
                #[cfg(feature = "kernel-visualize")]
                if let Some(dir) = std::env::var_os("PROFILE_LOGITS_DIR") {
                    use std::sync::atomic::{AtomicUsize, Ordering};
                    static SEQUENCE: AtomicUsize = AtomicUsize::new(0);
                    let dir = std::path::PathBuf::from(dir);
                    std::fs::create_dir_all(&dir).map_err(|e| fusor::Error::Io(e.to_string()))?;
                    let index = SEQUENCE.fetch_add(1, Ordering::Relaxed);
                    let bytes: Vec<u8> = logits_vec.iter().flat_map(|v| v.to_le_bytes()).collect();
                    std::fs::write(dir.join(format!("{index:05}.f32")), bytes)
                        .map_err(|e| fusor::Error::Io(e.to_string()))?;
                }
                #[cfg(feature = "kernel-visualize")]
                let host_stage = std::env::var_os("PROFILE_HOST_STAGES")
                    .is_some()
                    .then(Instant::now);
                let result = top_k_logits_from_full(&logits_vec, top_k, profile);
                #[cfg(feature = "kernel-visualize")]
                if let Some(start) = host_stage {
                    eprintln!(
                        "HOST_STAGE name=cpu_top_k us={}",
                        start.elapsed().as_micros()
                    );
                }
                result
            } else {
                let k = top_k.clamp(1, len) as u32;
                // The row is `[1, vocab]` so the sampler can be a resolve
                // root; `top_k` reads it as the row it is.
                let (values, ids) = logits.reshape([len]).top_k(k);
                let values = values
                    .to_vec_f32_async()
                    .await
                    .map_err(LlamaModelError::from)?;
                let ids = ids.to_flat_async().await.map_err(LlamaModelError::from)?;
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
