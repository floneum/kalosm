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
        let mut text_stream = TokenOutputStream::new(self.tokenizer.clone());
        for &token in &tokens {
            text_stream
                .next_token(token)
                .map_err(LlamaModelError::TokenOutputStreamError)?;
        }

        if gpu_token_sampling_enabled() && stop_on.is_none() {
            if let Some(mut gpu_sampler) = LlamaGpuSamplerState::new(&self.device, sampler, seed) {
                let top_k = gpu_sample_top_k(&gpu_sampler.config);
                if gpu_run_ahead_enabled()
                    && images.is_empty()
                    && self.model.supports_gpu_token_run_ahead()
                {
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
                        let stop_token = self.model.config.stop_token;
                        let mut tokens_generated = 0;
                        while !finished.is_canceled() && tokens_generated < max_tokens {
                            let scheduled_next = if tokens_generated + 1 < max_tokens {
                                let previous_tokens = gpu_sampler.previous_tokens(&text_stream);
                                let mut speculative_cache = session
                                    .cache
                                    .read()
                                    .map_err(|err| LlamaModelError::Session(err.to_string()))?
                                    .clone();
                                if speculative_cache.tokens.len()
                                    < self.model.config.context_length
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
                            if new_token == stop_token {
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

                let stop_token = self.model.config.stop_token;
                let mut tokens_generated = 0;
                while !finished.is_canceled() && tokens_generated < max_tokens {
                    let new_token = next_token;
                    if new_token == stop_token {
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
        let stop_token = self.model.config.stop_token;
        let mut tokens_generated = 0;

        'generate: while !finished.is_canceled() && tokens_generated < max_tokens {
            let new_token = text_stream
                .sample_token(&mut cpu_sampler, logits, stop_on.as_deref(), sample_top_k)
                .map_err(LlamaModelError::TokenOutputStreamError)?;
            if new_token == stop_token {
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
}
