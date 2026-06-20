use super::*;

impl<F: FloatDataType + SimdElement + Default + FloatOps + MatmulImpl> LlamaModel<F>
where
    F: CastTo<f32> + CastTensor<f32> + WasmNotSend + WasmNotSync + 'static,
    f32: CastTo<F> + CastTensor<F>,
    MulOp: SimdBinaryOp<F>,
    AddOp: SimdBinaryOp<F>,
    SumOp: SimdReduceOp<F>,
{
    pub(crate) async fn _infer(
        &mut self,
        settings: InferenceSettings<F>,
        mut on_token: crate::BoxedTokenCallback,
        finished: &futures_channel::oneshot::Sender<Result<(), LlamaModelError>>,
    ) -> Result<(), LlamaModelError> {
        let InferenceSettings {
            prompt,
            images,
            stop_on,
            sampler,
            session,
            max_tokens,
            seed,
        } = settings;

        let tokens = self
            .tokenizer
            .encode(&prompt, false)
            .map_err(LlamaModelError::Tokenizer)?;
        if std::env::var_os("KALOSM_TRACE_PROMPT").is_some() {
            let decoded = self
                .tokenizer
                .decode(&tokens, false)
                .unwrap_or_else(|err| format!("<decode error: {err}>"));
            eprintln!(
                "[prompt] len={} text={prompt:?} tokens={tokens:?} decoded={decoded:?}",
                tokens.len()
            );
        }
        let mut text_stream = TokenOutputStream::new(self.tokenizer.clone());
        for &token in &tokens {
            text_stream
                .next_token(token)
                .map_err(LlamaModelError::TokenOutputStreamError)?;
        }

        if mtp_speculative_enabled()
            && images.is_empty()
            && stop_on.is_none()
            && sampler.sampling_strategy == kalosm_language_model::SamplingStrategy::Standard
            && sampler.temperature <= 0.0
            && gpu_token_sampling_enabled()
            && self.mtp.is_some()
        {
            if let Some(gpu_sampler) = LlamaGpuSamplerState::new(&self.device, sampler, seed) {
                return self
                    .infer_mtp_speculative(
                        &tokens,
                        &images,
                        text_stream,
                        session,
                        max_tokens,
                        gpu_sampler,
                        on_token,
                        finished,
                    )
                    .await;
            }
        }

        if gpu_token_sampling_enabled() && stop_on.is_none() {
            if let Some(mut gpu_sampler) = LlamaGpuSamplerState::new(&self.device, sampler, seed) {
                let top_k = gpu_sample_top_k(&gpu_sampler.config);
                if gpu_run_ahead_enabled() && self.model.supports_gpu_token_run_ahead() {
                    let next_token = {
                        let previous_tokens = gpu_sampler.previous_tokens(&text_stream);
                        let mut session_lock = session
                            .cache
                            .write()
                            .map_err(|err| LlamaModelError::Session(err.to_string()))?;
                        Self::forward_sample_token_pending(
                            ForwardInputs {
                                model: &self.model,
                                device: &self.device,
                                tokens: &tokens,
                                images: &images,
                                cache: Some(&mut session_lock),
                                tokenizer: &self.tokenizer,
                            },
                            &mut gpu_sampler,
                            previous_tokens,
                            top_k,
                        )?
                    };

                    if let Some(mut next_token) = next_token {
                        let stop_tokens = &self.model.config.stop_tokens;
                        let mut tokens_generated = 0;
                        while !finished.is_canceled() && tokens_generated < max_tokens {
                            let scheduled_next = if tokens_generated + 1 < max_tokens {
                                let previous_tokens = gpu_sampler.previous_tokens(&text_stream);
                                let mut speculative_cache = session
                                    .cache
                                    .read()
                                    .map_err(|err| LlamaModelError::Session(err.to_string()))?
                                    .clone();
                                if speculative_cache.tokens.len() < self.model.config.context_length
                                {
                                    Self::forward_sample_token_from_gpu_token_pending(
                                        &self.model,
                                        &self.device,
                                        &next_token,
                                        &mut speculative_cache,
                                        &mut gpu_sampler,
                                        previous_tokens,
                                        top_k,
                                    )?
                                    .map(|(next, cache_slot)| (next, speculative_cache, cache_slot))
                                } else {
                                    None
                                }
                            } else {
                                None
                            };

                            let new_token = next_token
                                .read_token()
                                .await
                                .map_err(|err| LlamaModelError::Fusor(fusor::Error::Gpu(err)))?
                                .ok_or_else(|| {
                                    LlamaModelError::SamplerError(
                                        "pending GPU sampler refused slow fallback".into(),
                                    )
                                })?;
                            if std::env::var_os("KALOSM_TRACE_SAMPLED_TOKEN").is_some() {
                                let decoded = self
                                    .tokenizer
                                    .decode(&[new_token], false)
                                    .unwrap_or_else(|err| format!("<decode error: {err}>"));
                                eprintln!(
                                    "[sampled_token] index={tokens_generated} id={new_token} text={decoded:?} stop_tokens={stop_tokens:?}"
                                );
                            }
                            if stop_tokens.contains(&new_token) {
                                tracing::trace!("Stopping on stop token");
                                break;
                            }

                            tokens_generated += 1;
                            if let Some(new_text) = text_stream
                                .next_token(new_token)
                                .map_err(LlamaModelError::TokenOutputStreamError)?
                            {
                                on_token(new_text)?;
                            }

                            if let Some((scheduled_token, mut speculative_cache, cache_slot)) =
                                scheduled_next
                            {
                                if let Some(slot) = speculative_cache.tokens.get_mut(cache_slot) {
                                    *slot = new_token;
                                }
                                *session
                                    .cache
                                    .write()
                                    .map_err(|err| LlamaModelError::Session(err.to_string()))? =
                                    speculative_cache;
                                next_token = scheduled_token;
                            } else if !finished.is_canceled() && tokens_generated < max_tokens {
                                let previous_tokens = gpu_sampler.previous_tokens(&text_stream);
                                let mut session_lock = session
                                    .cache
                                    .write()
                                    .map_err(|err| LlamaModelError::Session(err.to_string()))?;
                                match Self::forward_sample_token_pending(
                                    ForwardInputs {
                                        model: &self.model,
                                        device: &self.device,
                                        tokens: &[new_token],
                                        images: &[],
                                        cache: Some(&mut session_lock),
                                        tokenizer: &self.tokenizer,
                                    },
                                    &mut gpu_sampler,
                                    previous_tokens,
                                    top_k,
                                )? {
                                    Some(sampled) => next_token = sampled,
                                    None => break,
                                }
                            } else {
                                break;
                            }

                            {
                                use std::sync::atomic::{AtomicBool, Ordering};
                                let yielded = AtomicBool::new(false);
                                std::future::poll_fn(|cx| {
                                    if yielded.load(Ordering::Relaxed) {
                                        std::task::Poll::Ready(())
                                    } else {
                                        yielded.store(true, Ordering::Relaxed);
                                        cx.waker().wake_by_ref();
                                        std::task::Poll::Pending
                                    }
                                })
                                .await;
                            }
                        }

                        return Ok(());
                    }
                }

                let mut next_token = {
                    let top_k = gpu_sample_top_k(&gpu_sampler.config);
                    let previous_tokens = gpu_sampler.previous_tokens(&text_stream);
                    let mut session_lock = session
                        .cache
                        .write()
                        .map_err(|err| LlamaModelError::Session(err.to_string()))?;
                    Self::forward_sample_token(
                        ForwardInputs {
                            model: &self.model,
                            device: &self.device,
                            tokens: &tokens,
                            images: &images,
                            cache: Some(&mut session_lock),
                            tokenizer: &self.tokenizer,
                        },
                        &mut gpu_sampler,
                        previous_tokens,
                        top_k,
                    )
                }
                .await?;

                let stop_tokens = &self.model.config.stop_tokens;
                let mut tokens_generated = 0;
                while !finished.is_canceled() && tokens_generated < max_tokens {
                    let new_token = next_token;
                    if std::env::var_os("KALOSM_TRACE_SAMPLED_TOKEN").is_some() {
                        let decoded = self
                            .tokenizer
                            .decode(&[new_token], false)
                            .unwrap_or_else(|err| format!("<decode error: {err}>"));
                        eprintln!(
                            "[sampled_token] index={tokens_generated} id={new_token} text={decoded:?} stop_tokens={stop_tokens:?}"
                        );
                    }
                    if stop_tokens.contains(&new_token) {
                        tracing::trace!("Stopping on stop token");
                        break;
                    }
                    tokens_generated += 1;
                    if let Some(new_text) = text_stream
                        .next_token(new_token)
                        .map_err(LlamaModelError::TokenOutputStreamError)?
                    {
                        on_token(new_text)?;
                    }

                    if finished.is_canceled() || tokens_generated >= max_tokens {
                        break;
                    }

                    next_token = {
                        let top_k = gpu_sample_top_k(&gpu_sampler.config);
                        let previous_tokens = gpu_sampler.previous_tokens(&text_stream);
                        let mut session_lock = session
                            .cache
                            .write()
                            .map_err(|err| LlamaModelError::Session(err.to_string()))?;
                        Self::forward_sample_token(
                            ForwardInputs {
                                model: &self.model,
                                device: &self.device,
                                tokens: &[new_token],
                                images: &[],
                                cache: Some(&mut session_lock),
                                tokenizer: &self.tokenizer,
                            },
                            &mut gpu_sampler,
                            previous_tokens,
                            top_k,
                        )
                    }
                    .await?;

                    {
                        use std::sync::atomic::{AtomicBool, Ordering};
                        let yielded = AtomicBool::new(false);
                        std::future::poll_fn(|cx| {
                            if yielded.load(Ordering::Relaxed) {
                                std::task::Poll::Ready(())
                            } else {
                                yielded.store(true, Ordering::Relaxed);
                                cx.waker().wake_by_ref();
                                std::task::Poll::Pending
                            }
                        })
                        .await;
                    }
                }

                return Ok(());
            }
        }

        let mut cpu_sampler = CpuSampler::new(sampler, seed);
        let sample_top_k = gpu_sample_top_k(&sampler);
        let logit_probs = {
            let mut session_lock = session
                .cache
                .write()
                .map_err(|err| LlamaModelError::Session(err.to_string()))?;
            Self::forward_top_k(
                &self.model,
                &self.device,
                &tokens,
                &images,
                Some(&mut session_lock),
                &self.tokenizer,
                sample_top_k,
            )
        }
        .await?;
        let mut logits = logits_from_sorted_top_k(logit_probs);
        // This stores a buffer of text that has been generated to check against the stop_on string. It should never be longer than the stop_on string.
        let mut queued_text_matching_stop_on = String::new();
        let stop_on_lowercase = stop_on.as_ref().map(|s| s.to_lowercase());
        let stop_on_lowercase = stop_on_lowercase.as_deref();
        let stop_tokens = &self.model.config.stop_tokens;
        let mut tokens_generated = 0;

        'generate: while !finished.is_canceled() && tokens_generated < max_tokens {
            let new_token = text_stream
                .sample_token(&mut cpu_sampler, logits, stop_on.as_deref(), sample_top_k)
                .map_err(LlamaModelError::TokenOutputStreamError)?;
            if std::env::var_os("KALOSM_TRACE_SAMPLED_TOKEN").is_some() {
                let decoded = self
                    .tokenizer
                    .decode(&[new_token], false)
                    .unwrap_or_else(|err| format!("<decode error: {err}>"));
                eprintln!(
                    "[sampled_token] index={tokens_generated} id={new_token} text={decoded:?} stop_tokens={stop_tokens:?}"
                );
            }
            if stop_tokens.contains(&new_token) {
                tracing::trace!("Stopping on stop token");
                break;
            }
            tokens_generated += 1;
            if let Some(mut new_text) = text_stream
                .next_token(new_token)
                .map_err(LlamaModelError::TokenOutputStreamError)?
            {
                if let Some(stop_on) = stop_on_lowercase {
                    let lowercase = new_text.to_lowercase();

                    // Check if the string ends with the start of the stop_on string
                    let mut before_stop_on = None;
                    let remaining_stop_on = stop_on
                        .strip_prefix(&queued_text_matching_stop_on)
                        .unwrap_or(stop_on);

                    // If the remaining stop_on string is empty, we have found a match
                    if remaining_stop_on.is_empty() {
                        break;
                    }

                    for (i, _) in lowercase.char_indices() {
                        let end_of_new_text = &lowercase[i..];
                        if end_of_new_text.is_empty() {
                            break;
                        }

                        // Check if we have matched all of the stop_on string
                        if end_of_new_text.starts_with(remaining_stop_on) {
                            queued_text_matching_stop_on += end_of_new_text;
                            break 'generate;
                        }

                        // Check if the string ends with the start of the stop_on string
                        if remaining_stop_on.starts_with(end_of_new_text) {
                            before_stop_on = Some(lowercase[..i].to_string());
                            queued_text_matching_stop_on += end_of_new_text;
                            break;
                        }
                    }

                    match before_stop_on {
                        Some(before_stop_on) => {
                            on_token(before_stop_on)?;
                        }
                        None => {
                            new_text =
                                std::mem::take(&mut queued_text_matching_stop_on) + &new_text;
                            on_token(new_text)?;
                        }
                    }
                } else {
                    on_token(new_text)?;
                }
            }

            if finished.is_canceled() || tokens_generated >= max_tokens {
                break;
            }

            let logit_probs = {
                let mut session_lock = session
                    .cache
                    .write()
                    .map_err(|err| LlamaModelError::Session(err.to_string()))?;
                Self::forward_top_k(
                    &self.model,
                    &self.device,
                    &[new_token],
                    &[],
                    Some(&mut session_lock),
                    &self.tokenizer,
                    sample_top_k,
                )
            }
            .await?;
            logits = logits_from_sorted_top_k(logit_probs);
            // Yield control to allow the stream to deliver tokens
            {
                use std::sync::atomic::{AtomicBool, Ordering};
                let yielded = AtomicBool::new(false);
                std::future::poll_fn(|cx| {
                    if yielded.load(Ordering::Relaxed) {
                        std::task::Poll::Ready(())
                    } else {
                        yielded.store(true, Ordering::Relaxed);
                        cx.waker().wake_by_ref();
                        std::task::Poll::Pending
                    }
                })
                .await;
            }
        }

        // Flush the queued text
        if let Some(stop_string) = stop_on_lowercase {
            if !queued_text_matching_stop_on.starts_with(stop_string) {
                on_token(queued_text_matching_stop_on)?;
            }
        }

        Ok(())
    }

    async fn infer_mtp_speculative(
        &self,
        prompt_tokens: &[u32],
        images: &[LlamaImage],
        mut text_stream: TokenOutputStream,
        session: crate::LlamaSession<F>,
        max_tokens: u32,
        mut gpu_sampler: LlamaGpuSamplerState,
        mut on_token: crate::BoxedTokenCallback,
        finished: &futures_channel::oneshot::Sender<Result<(), LlamaModelError>>,
    ) -> Result<(), LlamaModelError> {
        let mtp = self.mtp.as_ref().ok_or_else(|| {
            LlamaModelError::SamplerError("Gemma4 MTP assistant is not loaded".into())
        })?;
        let top_k = gpu_sample_top_k(&gpu_sampler.config);
        let target = {
            let mut session_lock = session
                .cache
                .write()
                .map_err(|err| LlamaModelError::Session(err.to_string()))?;
            self.model
                .forward_logits_and_nextn_f32(
                    prompt_tokens,
                    images,
                    &self.device,
                    Some(&mut session_lock),
                )
                .map_err(LlamaModelError::from)?
        };
        let last_row = target.logits.shape()[0].saturating_sub(1);
        let mut pending_h = Self::h_nextn_row(&target.h_nextn, last_row);
        let history = Self::mtp_previous_tokens(&gpu_sampler.config, text_stream.tokens(), &[]);
        let mut next_token = Self::sample_standard_logits_row(
            target.logits,
            last_row,
            &mut gpu_sampler,
            history,
            top_k,
        )
        .await?;

        let stop_tokens = &self.model.config.stop_tokens;
        let mut tokens_generated = 0usize;
        let mut drafted_total = 0usize;
        let mut accepted_total = 0usize;
        while !finished.is_canceled() && tokens_generated < max_tokens as usize {
            let new_token = next_token;
            if std::env::var_os("KALOSM_TRACE_SAMPLED_TOKEN").is_some() {
                let decoded = self
                    .tokenizer
                    .decode(&[new_token], false)
                    .unwrap_or_else(|err| format!("<decode error: {err}>"));
                eprintln!(
                    "[sampled_token] index={tokens_generated} id={new_token} text={decoded:?} stop_tokens={stop_tokens:?}"
                );
            }
            if stop_tokens.contains(&new_token) {
                tracing::trace!("Stopping on stop token");
                break;
            }
            tokens_generated += 1;
            if let Some(new_text) = text_stream
                .next_token(new_token)
                .map_err(LlamaModelError::TokenOutputStreamError)?
            {
                on_token(new_text)?;
            }
            if finished.is_canceled() || tokens_generated >= max_tokens as usize {
                break;
            }

            let remaining = max_tokens as usize - tokens_generated;
            let mut draft_limit = mtp_draft_limit(mtp.draft_n()).min(remaining);
            if mtp_auto_fallback_enabled() && drafted_total < mtp_fallback_probe_drafts() {
                draft_limit =
                    draft_limit.min(mtp_fallback_probe_drafts().saturating_sub(drafted_total));
            }
            let mut draft_tokens = Vec::with_capacity(draft_limit);
            let mut assistant_h = pending_h.clone();
            let mut assistant_token = new_token;
            let draft_position = {
                let session_lock = session
                    .cache
                    .read()
                    .map_err(|err| LlamaModelError::Session(err.to_string()))?;
                session_lock.tokens.len()
            };
            for _ in 0..draft_limit {
                let step = {
                    let session_lock = session
                        .cache
                        .read()
                        .map_err(|err| LlamaModelError::Session(err.to_string()))?;
                    mtp.draft_step(
                        &self.model,
                        assistant_token,
                        &assistant_h,
                        &session_lock,
                        &self.device,
                        draft_position,
                    )
                    .map_err(LlamaModelError::from)?
                };
                let history = Self::mtp_previous_tokens(
                    &gpu_sampler.config,
                    text_stream.tokens(),
                    &draft_tokens,
                );
                assistant_token =
                    Self::sample_standard_logits(step.logits, &mut gpu_sampler, history, top_k)
                        .await?;
                assistant_h = step.h_nextn;
                draft_tokens.push(assistant_token);
            }
            drafted_total += draft_tokens.len();

            let mut accepted = 0usize;
            let mut verify_input = new_token;
            let mut mismatched = false;
            for draft in draft_tokens.iter().copied() {
                let history =
                    Self::mtp_previous_tokens(&gpu_sampler.config, text_stream.tokens(), &[]);
                let (verified, h_nextn) = self
                    .mtp_target_step(verify_input, &session, &mut gpu_sampler, history, top_k)
                    .await?;
                pending_h = h_nextn;
                if verified != draft {
                    next_token = verified;
                    mismatched = true;
                    break;
                }

                accepted += 1;
                if stop_tokens.contains(&draft) {
                    tracing::trace!("Stopping on accepted MTP stop token");
                    if std::env::var_os("KALOSM_TRACE_MTP").is_some() {
                        tracing::info!(
                            "mtp_summary drafted={drafted_total} accepted={accepted_total}"
                        );
                    }
                    return Ok(());
                }

                tokens_generated += 1;
                if let Some(new_text) = text_stream
                    .next_token(draft)
                    .map_err(LlamaModelError::TokenOutputStreamError)?
                {
                    on_token(new_text)?;
                }

                verify_input = draft;
                if tokens_generated >= max_tokens as usize || finished.is_canceled() {
                    break;
                }
            }
            accepted_total += accepted;

            if !mismatched
                && accepted == draft_tokens.len()
                && tokens_generated < max_tokens as usize
                && !finished.is_canceled()
            {
                let history =
                    Self::mtp_previous_tokens(&gpu_sampler.config, text_stream.tokens(), &[]);
                let (bonus, h_nextn) = self
                    .mtp_target_step(verify_input, &session, &mut gpu_sampler, history, top_k)
                    .await?;
                pending_h = h_nextn;
                next_token = bonus;
            }

            if Self::should_mtp_fallback(drafted_total, accepted_total) {
                if std::env::var_os("KALOSM_TRACE_MTP").is_some() {
                    tracing::info!(
                        "mtp_auto_fallback drafted={drafted_total} accepted={accepted_total}"
                    );
                }
                return self
                    .infer_target_only_from_pending(
                        next_token,
                        text_stream,
                        session,
                        max_tokens,
                        tokens_generated,
                        gpu_sampler,
                        on_token,
                        finished,
                    )
                    .await;
            }

            {
                use std::sync::atomic::{AtomicBool, Ordering};
                let yielded = AtomicBool::new(false);
                std::future::poll_fn(|cx| {
                    if yielded.load(Ordering::Relaxed) {
                        std::task::Poll::Ready(())
                    } else {
                        yielded.store(true, Ordering::Relaxed);
                        cx.waker().wake_by_ref();
                        std::task::Poll::Pending
                    }
                })
                .await;
            }
        }
        if std::env::var_os("KALOSM_TRACE_MTP").is_some() {
            tracing::info!("mtp_summary drafted={drafted_total} accepted={accepted_total}");
        }
        Ok(())
    }

    async fn infer_target_only_from_pending(
        &self,
        mut next_token: u32,
        mut text_stream: TokenOutputStream,
        session: crate::LlamaSession<F>,
        max_tokens: u32,
        mut tokens_generated: usize,
        mut gpu_sampler: LlamaGpuSamplerState,
        mut on_token: crate::BoxedTokenCallback,
        finished: &futures_channel::oneshot::Sender<Result<(), LlamaModelError>>,
    ) -> Result<(), LlamaModelError> {
        let stop_tokens = &self.model.config.stop_tokens;
        let top_k = gpu_sample_top_k(&gpu_sampler.config);
        while !finished.is_canceled() && tokens_generated < max_tokens as usize {
            let new_token = next_token;
            if std::env::var_os("KALOSM_TRACE_SAMPLED_TOKEN").is_some() {
                let decoded = self
                    .tokenizer
                    .decode(&[new_token], false)
                    .unwrap_or_else(|err| format!("<decode error: {err}>"));
                eprintln!(
                    "[sampled_token] index={tokens_generated} id={new_token} text={decoded:?} stop_tokens={stop_tokens:?}"
                );
            }
            if stop_tokens.contains(&new_token) {
                tracing::trace!("Stopping on stop token");
                break;
            }
            tokens_generated += 1;
            if let Some(new_text) = text_stream
                .next_token(new_token)
                .map_err(LlamaModelError::TokenOutputStreamError)?
            {
                on_token(new_text)?;
            }
            if finished.is_canceled() || tokens_generated >= max_tokens as usize {
                break;
            }

            let previous_tokens = gpu_sampler.previous_tokens(&text_stream);
            let pending = {
                let mut session_lock = session
                    .cache
                    .write()
                    .map_err(|err| LlamaModelError::Session(err.to_string()))?;
                Self::forward_sample_token_pending(
                    ForwardInputs {
                        model: &self.model,
                        device: &self.device,
                        tokens: &[new_token],
                        images: &[],
                        cache: Some(&mut session_lock),
                        tokenizer: &self.tokenizer,
                    },
                    &mut gpu_sampler,
                    previous_tokens.clone(),
                    top_k,
                )?
            };
            if let Some(next_pending) = pending {
                return self
                    .infer_target_only_from_gpu_pending(
                        next_pending,
                        text_stream,
                        session,
                        max_tokens,
                        tokens_generated,
                        gpu_sampler,
                        on_token,
                        finished,
                    )
                    .await;
            }

            next_token = {
                let mut session_lock = session
                    .cache
                    .write()
                    .map_err(|err| LlamaModelError::Session(err.to_string()))?;
                Self::forward_sample_token(
                    ForwardInputs {
                        model: &self.model,
                        device: &self.device,
                        tokens: &[new_token],
                        images: &[],
                        cache: Some(&mut session_lock),
                        tokenizer: &self.tokenizer,
                    },
                    &mut gpu_sampler,
                    previous_tokens,
                    top_k,
                )
            }
            .await?;

            {
                use std::sync::atomic::{AtomicBool, Ordering};
                let yielded = AtomicBool::new(false);
                std::future::poll_fn(|cx| {
                    if yielded.load(Ordering::Relaxed) {
                        std::task::Poll::Ready(())
                    } else {
                        yielded.store(true, Ordering::Relaxed);
                        cx.waker().wake_by_ref();
                        std::task::Poll::Pending
                    }
                })
                .await;
            }
        }
        Ok(())
    }

    async fn infer_target_only_from_gpu_pending(
        &self,
        mut next_token: fusor::GpuSampledToken,
        mut text_stream: TokenOutputStream,
        session: crate::LlamaSession<F>,
        max_tokens: u32,
        mut tokens_generated: usize,
        mut gpu_sampler: LlamaGpuSamplerState,
        mut on_token: crate::BoxedTokenCallback,
        finished: &futures_channel::oneshot::Sender<Result<(), LlamaModelError>>,
    ) -> Result<(), LlamaModelError> {
        let stop_tokens = &self.model.config.stop_tokens;
        let top_k = gpu_sample_top_k(&gpu_sampler.config);
        while !finished.is_canceled() && tokens_generated < max_tokens as usize {
            let scheduled_next = if tokens_generated + 1 < max_tokens as usize {
                let previous_tokens = gpu_sampler.previous_tokens(&text_stream);
                let mut speculative_cache = session
                    .cache
                    .read()
                    .map_err(|err| LlamaModelError::Session(err.to_string()))?
                    .clone();
                if speculative_cache.tokens.len() < self.model.config.context_length {
                    Self::forward_sample_token_from_gpu_token_pending(
                        &self.model,
                        &self.device,
                        &next_token,
                        &mut speculative_cache,
                        &mut gpu_sampler,
                        previous_tokens,
                        top_k,
                    )?
                    .map(|(next, cache_slot)| (next, speculative_cache, cache_slot))
                } else {
                    None
                }
            } else {
                None
            };

            let new_token = next_token
                .read_token()
                .await
                .map_err(|err| LlamaModelError::Fusor(fusor::Error::Gpu(err)))?
                .ok_or_else(|| {
                    LlamaModelError::SamplerError(
                        "pending GPU sampler refused slow fallback".into(),
                    )
                })?;
            if std::env::var_os("KALOSM_TRACE_SAMPLED_TOKEN").is_some() {
                let decoded = self
                    .tokenizer
                    .decode(&[new_token], false)
                    .unwrap_or_else(|err| format!("<decode error: {err}>"));
                eprintln!(
                    "[sampled_token] index={tokens_generated} id={new_token} text={decoded:?} stop_tokens={stop_tokens:?}"
                );
            }
            if stop_tokens.contains(&new_token) {
                tracing::trace!("Stopping on stop token");
                break;
            }

            tokens_generated += 1;
            if let Some(new_text) = text_stream
                .next_token(new_token)
                .map_err(LlamaModelError::TokenOutputStreamError)?
            {
                on_token(new_text)?;
            }

            if let Some((scheduled_token, mut speculative_cache, cache_slot)) = scheduled_next {
                if let Some(slot) = speculative_cache.tokens.get_mut(cache_slot) {
                    *slot = new_token;
                }
                *session
                    .cache
                    .write()
                    .map_err(|err| LlamaModelError::Session(err.to_string()))? = speculative_cache;
                next_token = scheduled_token;
            } else if !finished.is_canceled() && tokens_generated < max_tokens as usize {
                let previous_tokens = gpu_sampler.previous_tokens(&text_stream);
                let pending = {
                    let mut session_lock = session
                        .cache
                        .write()
                        .map_err(|err| LlamaModelError::Session(err.to_string()))?;
                    Self::forward_sample_token_pending(
                        ForwardInputs {
                            model: &self.model,
                            device: &self.device,
                            tokens: &[new_token],
                            images: &[],
                            cache: Some(&mut session_lock),
                            tokenizer: &self.tokenizer,
                        },
                        &mut gpu_sampler,
                        previous_tokens,
                        top_k,
                    )?
                };
                match pending {
                    Some(sampled) => next_token = sampled,
                    None => break,
                }
            } else {
                break;
            }

            {
                use std::sync::atomic::{AtomicBool, Ordering};
                let yielded = AtomicBool::new(false);
                std::future::poll_fn(|cx| {
                    if yielded.load(Ordering::Relaxed) {
                        std::task::Poll::Ready(())
                    } else {
                        yielded.store(true, Ordering::Relaxed);
                        cx.waker().wake_by_ref();
                        std::task::Poll::Pending
                    }
                })
                .await;
            }
        }
        Ok(())
    }

    async fn mtp_target_step(
        &self,
        token: u32,
        session: &crate::LlamaSession<F>,
        sampler: &mut LlamaGpuSamplerState,
        previous_tokens: Vec<u32>,
        top_k: usize,
    ) -> Result<(u32, fusor::Tensor<2, f32>), LlamaModelError> {
        let target = {
            let mut session_lock = session
                .cache
                .write()
                .map_err(|err| LlamaModelError::Session(err.to_string()))?;
            self.model
                .forward_logits_and_nextn_f32(&[token], &[], &self.device, Some(&mut session_lock))
                .map_err(LlamaModelError::from)?
        };
        let h_nextn = Self::h_nextn_row(&target.h_nextn, 0);
        let token =
            Self::sample_standard_logits_row(target.logits, 0, sampler, previous_tokens, top_k)
                .await?;
        Ok((token, h_nextn))
    }

    fn should_mtp_fallback(drafted_total: usize, accepted_total: usize) -> bool {
        if !mtp_auto_fallback_enabled() {
            return false;
        }
        if drafted_total < mtp_fallback_probe_drafts() {
            return false;
        }
        accepted_total * 100 < drafted_total * mtp_fallback_min_accept_percent()
    }

    fn h_nextn_row(h_nextn: &fusor::Tensor<2, f32>, row: usize) -> fusor::Tensor<2, f32> {
        let row: fusor::Tensor<1, f32> = h_nextn.i((row, ..)).to_concrete();
        row.unsqueeze(0).to_concrete()
    }

    fn mtp_previous_tokens(
        config: &GpuSamplerConfig,
        base_tokens: &[u32],
        extra_tokens: &[u32],
    ) -> Vec<u32> {
        let range = config.repetition_penalty_range;
        if range == 0 {
            return Vec::new();
        }
        let total_len = base_tokens.len() + extra_tokens.len();
        let keep_from = total_len.saturating_sub(range);
        let mut result = Vec::with_capacity(range.min(total_len));
        if keep_from < base_tokens.len() {
            result.extend_from_slice(&base_tokens[keep_from..]);
            result.extend_from_slice(extra_tokens);
        } else {
            result.extend_from_slice(&extra_tokens[keep_from - base_tokens.len()..]);
        }
        result
    }

    async fn sample_standard_logits(
        logits: fusor::Tensor<1, f32>,
        sampler: &mut LlamaGpuSamplerState,
        previous_tokens: Vec<u32>,
        top_k: usize,
    ) -> Result<u32, LlamaModelError> {
        let params = sampler.standard_params(top_k);
        logits
            .sample_standard_token(&previous_tokens, params)
            .await
            .map_err(LlamaModelError::from)
    }

    async fn sample_standard_logits_row(
        logits: fusor::Tensor<2, f32>,
        row: usize,
        sampler: &mut LlamaGpuSamplerState,
        previous_tokens: Vec<u32>,
        top_k: usize,
    ) -> Result<u32, LlamaModelError> {
        let row_logits: fusor::Tensor<1, f32> = logits.i((row, ..)).to_concrete();
        Self::sample_standard_logits(row_logits, sampler, previous_tokens, top_k).await
    }
}
