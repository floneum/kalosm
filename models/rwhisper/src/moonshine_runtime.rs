use std::{collections::HashMap, num::NonZeroUsize, ops::Range};

use futures_channel::mpsc::UnboundedSender;
use tokenizers::Tokenizer;

use crate::{
    moonshine_config::MoonshineStreamingConfig,
    quantized::{
        moonshine::{Moonshine, MoonshineDecoderCache, MoonshineEncoderStreamState},
        timestamps::extract_timestamps,
    },
    DecodingResult, Segment, TokenChunk, WhisperLanguage,
};
use fusor::{Device, Tensor, VarBuilder};

#[derive(Clone)]
struct CandidateDecode {
    tokens: Vec<u32>,
    token_timestamps: Vec<f32>,
    clip_end_s: f32,
    usable_samples: usize,
}

struct MoonshineStreamState {
    word_timestamps: bool,
    sender: UnboundedSender<Segment>,
    last_candidate: Option<CandidateDecode>,
    encoder_state: MoonshineEncoderStreamState,
    prepared_encoder_hidden_states: Option<Tensor<3, f32>>,
    decoder_prefix_cache: MoonshineDecoderCache,
    decoder_prefix_tokens: Vec<u32>,
    last_decoded_finalized_frames: usize,
    emitted_tokens: usize,
    last_emitted_end_s: f32,
}

pub(crate) struct MoonshineRuntime {
    device: Device,
    tokenizer: Tokenizer,
    model: Moonshine,
    start_token: u32,
    eos_token: u32,
    sample_rate: usize,
    frame_len: usize,
    max_tokens_per_second: f32,
    stream_decode_interval_frames: usize,
    stream_initial_holdback_tokens: usize,
    stream_stable_holdback_tokens: usize,
    alignment_heads: Option<&'static [[usize; 2]]>,
    streams: HashMap<u64, MoonshineStreamState>,
}

impl MoonshineRuntime {
    pub(crate) fn new(
        device: Device,
        weights: &[u8],
        tokenizer_bytes: &[u8],
        config: MoonshineStreamingConfig,
        alignment_heads: Option<&'static [[usize; 2]]>,
    ) -> Result<Self, crate::model::WhisperLoadingError> {
        let tokenizer = Tokenizer::from_bytes(tokenizer_bytes)
            .map_err(crate::model::WhisperLoadingError::LoadTokenizer)?;
        let start_token = config.decoder_start_token();
        let eos_token = config.eos_token_id;
        let sample_rate = config.encoder_config.sample_rate;
        let frame_len = config.encoder_config.frame_len();
        let max_tokens_per_second = std::env::var("RWHISPER_MOONSHINE_MAX_TOKENS_PER_SECOND")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(6.5);
        let stream_decode_interval_frames = std::env::var("RWHISPER_MOONSHINE_STREAM_DECODE_MS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .map(|milliseconds| (milliseconds / 20).max(1))
            .unwrap_or(10);
        let stream_stable_holdback_tokens =
            std::env::var("RWHISPER_MOONSHINE_STREAM_HOLDBACK_TOKENS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(2);
        let stream_initial_holdback_tokens =
            std::env::var("RWHISPER_MOONSHINE_STREAM_INITIAL_HOLDBACK_TOKENS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(4);

        let mut reader = std::io::Cursor::new(weights);
        let mut vb = VarBuilder::from_gguf(&mut reader).map_err(|err| {
            crate::model::WhisperLoadingError::LoadModel(fusor::Error::msg(err.to_string()))
        })?;
        let model = Moonshine::load(&device, &mut vb, config)
            .map_err(crate::model::WhisperLoadingError::LoadModel)?;

        Ok(Self {
            device,
            tokenizer,
            model,
            start_token,
            eos_token,
            sample_rate,
            frame_len,
            max_tokens_per_second,
            stream_decode_interval_frames,
            stream_initial_holdback_tokens,
            stream_stable_holdback_tokens,
            alignment_heads,
            streams: HashMap::new(),
        })
    }

    fn validate_language(
        &self,
        language: Option<WhisperLanguage>,
    ) -> Result<(), crate::model::WhisperLoadingError> {
        if let Some(language) = language {
            if !matches!(language, WhisperLanguage::English) {
                return Err(crate::model::WhisperLoadingError::UnsupportedLanguage(
                    language,
                ));
            }
        }
        Ok(())
    }

    fn usable_samples<'a>(&self, samples: &'a [f32]) -> &'a [f32] {
        let usable = samples.len() / self.frame_len * self.frame_len;
        &samples[..usable]
    }

    fn max_new_tokens(&self, sample_count: usize) -> usize {
        let seconds = sample_count as f32 / self.sample_rate as f32;
        (seconds * self.max_tokens_per_second).ceil() as usize + 8
    }

    fn encode_prepared_full(
        &mut self,
        samples: &[f32],
    ) -> Result<(Tensor<3, f32>, usize), crate::model::WhisperError> {
        let encoder_hidden_states = self
            .model
            .encoder
            .encode(&self.device, samples)
            .map_err(crate::model::WhisperError::Fusor)?;
        let total_frames = encoder_hidden_states.shape()[1];
        let adapted_encoder_hidden_states = self
            .model
            .decoder
            .prepare_encoder_hidden_states(&encoder_hidden_states)
            .map_err(crate::model::WhisperError::Fusor)?;
        Ok((adapted_encoder_hidden_states, total_frames))
    }

    async fn greedy_generate(
        &mut self,
        encoder_hidden_states: &Tensor<3, f32>,
        max_new_tokens: usize,
    ) -> Result<Vec<u32>, crate::model::WhisperError> {
        let mut decoder_cache = MoonshineDecoderCache::default();
        let mut decoder_inputs = vec![self.start_token];
        let mut generated = Vec::new();

        for _ in 0..max_new_tokens.max(1) {
            let logits = self
                .model
                .decoder
                .decode_cached(&decoder_inputs, &encoder_hidden_states, &mut decoder_cache)
                .map_err(crate::model::WhisperError::Fusor)?;
            let last_index = logits.shape()[1].saturating_sub(1);
            let last_logits = logits
                .narrow(1, last_index, 1)
                .squeeze(1)
                .squeeze(0)
                .to_concrete();
            let logits = last_logits.as_slice().await?;
            let mut best_token = 0u32;
            let mut best_logit = f32::NEG_INFINITY;
            for (token, value) in logits.as_slice().iter().copied().enumerate() {
                if value > best_logit {
                    best_logit = value;
                    best_token = token as u32;
                }
            }
            if best_token == self.eos_token {
                break;
            }
            generated.push(best_token);
            decoder_inputs.clear();
            decoder_inputs.push(best_token);
        }

        Ok(generated)
    }

    async fn greedy_generate_from_prefix(
        &mut self,
        prefix_tokens: &[u32],
        prefix_cache: &MoonshineDecoderCache,
        encoder_hidden_states: &Tensor<3, f32>,
        max_new_tokens: usize,
    ) -> Result<Vec<u32>, crate::model::WhisperError> {
        let mut decoder_cache = prefix_cache.clone();
        let mut generated = prefix_tokens.to_vec();
        if prefix_tokens.len() >= max_new_tokens {
            return Ok(generated);
        }

        let mut decoder_inputs = vec![prefix_tokens.last().copied().unwrap_or(self.start_token)];
        for _ in 0..max_new_tokens.saturating_sub(prefix_tokens.len()).max(1) {
            let logits = self
                .model
                .decoder
                .decode_cached(&decoder_inputs, &encoder_hidden_states, &mut decoder_cache)
                .map_err(crate::model::WhisperError::Fusor)?;
            let last_index = logits.shape()[1].saturating_sub(1);
            let last_logits = logits
                .narrow(1, last_index, 1)
                .squeeze(1)
                .squeeze(0)
                .to_concrete();
            let logits = last_logits.as_slice().await?;
            let mut best_token = 0u32;
            let mut best_logit = f32::NEG_INFINITY;
            for (token, value) in logits.as_slice().iter().copied().enumerate() {
                if value > best_logit {
                    best_logit = value;
                    best_token = token as u32;
                }
            }
            if best_token == self.eos_token {
                break;
            }
            generated.push(best_token);
            decoder_inputs.clear();
            decoder_inputs.push(best_token);
        }

        Ok(generated)
    }

    async fn token_timestamps(
        &mut self,
        generated: &[u32],
        encoder_hidden_states: &Tensor<3, f32>,
        seconds_per_frame: f32,
    ) -> Result<Vec<f32>, crate::model::WhisperError> {
        if generated.is_empty() {
            return Ok(Vec::new());
        }
        let mut decoder_inputs = Vec::with_capacity(generated.len());
        decoder_inputs.push(self.start_token);
        decoder_inputs.extend_from_slice(&generated[..generated.len().saturating_sub(1)]);

        let mut cross_attentions = Vec::new();
        let _ = self
            .model
            .decoder
            .decode_prepared(
                decoder_inputs.as_slice(),
                encoder_hidden_states,
                Some(&mut cross_attentions),
            )
            .map_err(crate::model::WhisperError::Fusor)?;

        let mask = vec![vec![true; generated.len()]];
        let timestamps = extract_timestamps(
            self.alignment_heads,
            &cross_attentions,
            const { NonZeroUsize::new(7).unwrap() },
            encoder_hidden_states.shape()[1],
            seconds_per_frame,
            mask,
        )
        .await
        .map_err(crate::model::WhisperError::Fusor)?;

        Ok(timestamps.into_iter().next().unwrap_or_default())
    }

    async fn decode_candidate_from_prepared(
        &mut self,
        adapted_encoder_hidden_states: &Tensor<3, f32>,
        clip_end_s: f32,
        usable_samples: usize,
        compute_timestamps: bool,
        reuse_prefix_tokens: &[u32],
        reuse_prefix_cache: Option<&MoonshineDecoderCache>,
    ) -> Result<Option<CandidateDecode>, crate::model::WhisperError> {
        let total_frames = adapted_encoder_hidden_states.shape()[1];
        if usable_samples == 0 || total_frames == 0 {
            return Ok(None);
        }

        let max_new_tokens = self.max_new_tokens(usable_samples);
        let generated = if let Some(reuse_prefix_cache) = reuse_prefix_cache {
            self.greedy_generate_from_prefix(
                reuse_prefix_tokens,
                reuse_prefix_cache,
                adapted_encoder_hidden_states,
                max_new_tokens,
            )
            .await?
        } else {
            self.greedy_generate(adapted_encoder_hidden_states, max_new_tokens)
                .await?
        };
        let seconds_per_frame = if total_frames == 0 {
            0.0
        } else {
            clip_end_s / total_frames as f32
        };
        let token_timestamps = if compute_timestamps {
            self.token_timestamps(&generated, adapted_encoder_hidden_states, seconds_per_frame)
                .await?
        } else {
            Vec::new()
        };

        Ok(Some(CandidateDecode {
            tokens: generated,
            token_timestamps,
            clip_end_s,
            usable_samples,
        }))
    }

    fn sync_stream_prefix_cache(
        &mut self,
        state: &mut MoonshineStreamState,
        prefix_tokens: &[u32],
        adapted_encoder_hidden_states: &Tensor<3, f32>,
    ) -> Result<(), crate::model::WhisperError> {
        if prefix_tokens.is_empty() {
            if !state.decoder_prefix_tokens.is_empty()
                || !state.decoder_prefix_cache.tokens.is_empty()
            {
                state.decoder_prefix_tokens.clear();
                state.decoder_prefix_cache = MoonshineDecoderCache::default();
            }
            return Ok(());
        }

        let target_cache_tokens = &prefix_tokens[..prefix_tokens.len().saturating_sub(1)];
        let common_prefix =
            self.common_prefix_len(&state.decoder_prefix_tokens, target_cache_tokens);
        if common_prefix < state.decoder_prefix_tokens.len() {
            state.decoder_prefix_tokens.clear();
            state.decoder_prefix_cache = MoonshineDecoderCache::default();
        }

        let mut tokens_to_append = Vec::with_capacity(
            target_cache_tokens
                .len()
                .saturating_sub(state.decoder_prefix_tokens.len())
                + 1,
        );
        if state.decoder_prefix_cache.tokens.is_empty() {
            tokens_to_append.push(self.start_token);
        }
        tokens_to_append
            .extend_from_slice(&target_cache_tokens[state.decoder_prefix_tokens.len()..]);

        if !tokens_to_append.is_empty() {
            let _ = self
                .model
                .decoder
                .decode_cached(
                    &tokens_to_append,
                    adapted_encoder_hidden_states,
                    &mut state.decoder_prefix_cache,
                )
                .map_err(crate::model::WhisperError::Fusor)?;
        }
        state.decoder_prefix_tokens = target_cache_tokens.to_vec();
        Ok(())
    }

    async fn populate_candidate_timestamps(
        &mut self,
        candidate: &mut CandidateDecode,
        adapted_encoder_hidden_states: &Tensor<3, f32>,
    ) -> Result<(), crate::model::WhisperError> {
        if !candidate.token_timestamps.is_empty() || candidate.tokens.is_empty() {
            return Ok(());
        }
        let total_frames = adapted_encoder_hidden_states.shape()[1];
        let seconds_per_frame = if total_frames == 0 {
            0.0
        } else {
            candidate.clip_end_s / total_frames as f32
        };
        candidate.token_timestamps = self
            .token_timestamps(
                &candidate.tokens,
                adapted_encoder_hidden_states,
                seconds_per_frame,
            )
            .await?;
        Ok(())
    }

    async fn update_stream_candidate(
        &mut self,
        state: &mut MoonshineStreamState,
        samples: &[f32],
        flush: bool,
    ) -> Result<Option<CandidateDecode>, crate::model::WhisperError> {
        let encoder_append = self
            .model
            .encoder
            .encode_stream(&self.device, &mut state.encoder_state, samples, flush)
            .map_err(crate::model::WhisperError::Fusor)?;

        let finalized_encoder_frames = state
            .prepared_encoder_hidden_states
            .as_ref()
            .map(|tensor| tensor.shape()[1])
            .unwrap_or(0);
        if encoder_append.hidden_states.is_none()
            && encoder_append.total_finalized_frames == finalized_encoder_frames
        {
            return Ok(state.last_candidate.clone());
        }

        if let Some(hidden_states) = encoder_append.hidden_states {
            let start_pos = state
                .prepared_encoder_hidden_states
                .as_ref()
                .map(|tensor| tensor.shape()[1])
                .unwrap_or(0);
            let prepared = self
                .model
                .decoder
                .prepare_encoder_hidden_states_range(&hidden_states, start_pos)
                .map_err(crate::model::WhisperError::Fusor)?;
            state.prepared_encoder_hidden_states =
                Some(match state.prepared_encoder_hidden_states.take() {
                    Some(existing) => {
                        Tensor::cat([existing, prepared], 1).to_materialized_blocking()
                    }
                    None => prepared,
                });
        }

        let Some(prepared_encoder_hidden_states) = state.prepared_encoder_hidden_states.as_ref()
        else {
            return Ok(None);
        };
        let prepared_encoder_hidden_states = prepared_encoder_hidden_states.clone();
        let newly_finalized_frames = encoder_append
            .total_finalized_frames
            .saturating_sub(state.last_decoded_finalized_frames);
        if !flush && newly_finalized_frames < self.stream_decode_interval_frames {
            return Ok(state.last_candidate.clone());
        }

        let clip_end_s = if encoder_append.total_seen_frames == 0 {
            0.0
        } else {
            encoder_append.usable_input_samples as f32 / self.sample_rate as f32
                * encoder_append.total_finalized_frames as f32
                / encoder_append.total_seen_frames as f32
        };
        let reuse_prefix_tokens = state
            .last_candidate
            .as_ref()
            .map(|candidate| {
                candidate.tokens[..state.emitted_tokens.min(candidate.tokens.len())].to_vec()
            })
            .unwrap_or_default();
        if !reuse_prefix_tokens.is_empty() {
            self.sync_stream_prefix_cache(
                state,
                &reuse_prefix_tokens,
                &prepared_encoder_hidden_states,
            )?;
        }
        let reuse_prefix_cache =
            (!reuse_prefix_tokens.is_empty()).then_some(state.decoder_prefix_cache.clone());
        let candidate = self
            .decode_candidate_from_prepared(
                &prepared_encoder_hidden_states,
                clip_end_s,
                encoder_append.usable_input_samples,
                false,
                &reuse_prefix_tokens,
                reuse_prefix_cache.as_ref(),
            )
            .await?;
        if candidate.is_some() {
            state.last_decoded_finalized_frames = encoder_append.total_finalized_frames;
        }
        Ok(candidate)
    }

    fn decoding_result_from_tokens(
        &self,
        tokens: &[u32],
        token_timestamps: &[f32],
        include_chunk_timestamps: bool,
        clip_end_s: f32,
    ) -> Result<DecodingResult, crate::model::WhisperError> {
        let mut processed_tokens = Vec::with_capacity(tokens.len());
        let mut timestamp_start: Option<f32> = None;
        let mut prev_text_len = 0usize;
        let mut chunks = Vec::new();
        let mut current_text = String::new();

        for (index, token) in tokens.iter().copied().enumerate() {
            processed_tokens.push(token);
            if timestamp_start.is_none() {
                timestamp_start = token_timestamps.get(index).copied();
            }
            let detokenized = self
                .tokenizer
                .decode(&processed_tokens, true)
                .map_err(crate::model::WhisperError::Tokenizer)?;
            if detokenized.len() > prev_text_len
                && detokenized.chars().last().unwrap_or_default().is_ascii()
            {
                let timestamp = if include_chunk_timestamps {
                    let start = timestamp_start.unwrap_or(0.0);
                    let end = token_timestamps.get(index).copied().unwrap_or(clip_end_s);
                    timestamp_start = Some(end);
                    Some(start..end)
                } else {
                    None
                };
                let text_range = current_text.len()..detokenized.len();
                current_text = detokenized;
                prev_text_len = current_text.len();
                chunks.push(TokenChunk {
                    text_range,
                    timestamp,
                });
            } else {
                prev_text_len = detokenized.len();
            }
        }

        if current_text.is_empty() && !tokens.is_empty() {
            current_text = self
                .tokenizer
                .decode(tokens, true)
                .map_err(crate::model::WhisperError::Tokenizer)?;
            if !current_text.is_empty() {
                chunks.push(TokenChunk {
                    text_range: 0..current_text.len(),
                    timestamp: include_chunk_timestamps.then_some(0.0..clip_end_s),
                });
            }
        }

        Ok(DecodingResult {
            text: current_text,
            avg_logprob: 0.0,
            no_speech_prob: 0.0,
            compression_ratio: 0.0,
            chunks,
        })
    }

    fn common_prefix_len(&self, left: &[u32], right: &[u32]) -> usize {
        left.iter()
            .zip(right.iter())
            .take_while(|(l, r)| l == r)
            .count()
    }

    fn stable_emit_len(&self, stable_len: usize, flush: bool) -> usize {
        if flush {
            stable_len
        } else {
            stable_len.saturating_sub(self.stream_stable_holdback_tokens)
        }
    }

    fn segment_from_token_range(
        &self,
        range: Range<usize>,
        candidate: &CandidateDecode,
        word_timestamps: bool,
        fallback_start_s: f32,
    ) -> Result<Option<Segment>, crate::model::WhisperError> {
        if range.is_empty() || range.start >= candidate.tokens.len() {
            return Ok(None);
        }

        let prefix_text = if range.start == 0 {
            String::new()
        } else {
            self.tokenizer
                .decode(&candidate.tokens[..range.start], true)
                .map_err(crate::model::WhisperError::Tokenizer)?
        };
        let token_timestamps = if candidate.token_timestamps.is_empty() {
            vec![]
        } else {
            candidate.token_timestamps[..range.end].to_vec()
        };
        let mut result = self.decoding_result_from_tokens(
            &candidate.tokens[..range.end],
            &token_timestamps,
            word_timestamps,
            candidate.clip_end_s,
        )?;
        if result.text.len() >= prefix_text.len() && result.text.starts_with(&prefix_text) {
            let prefix_len = prefix_text.len();
            result.text = result.text[prefix_len..].to_string();
            result.chunks = result
                .chunks
                .into_iter()
                .filter_map(|mut chunk| {
                    if chunk.text_range.end <= prefix_len {
                        return None;
                    }
                    chunk.text_range.start = chunk.text_range.start.saturating_sub(prefix_len);
                    chunk.text_range.end -= prefix_len;
                    Some(chunk)
                })
                .collect();
        }
        if result.text.is_empty() {
            return Ok(None);
        }

        let total_tokens = candidate.tokens.len().max(1) as f32;
        let mut start_s = token_timestamps
            .get(range.start)
            .copied()
            .unwrap_or_else(|| candidate.clip_end_s * range.start as f32 / total_tokens);
        let mut end_s = token_timestamps
            .get(range.end.saturating_sub(1))
            .copied()
            .unwrap_or_else(|| candidate.clip_end_s * range.end as f32 / total_tokens);
        start_s = start_s.max(fallback_start_s);
        if end_s <= start_s {
            end_s = candidate.clip_end_s.max(start_s);
        }
        start_s = start_s.max(0.0);
        let sample_start = (start_s * self.sample_rate as f32).round() as usize;
        let sample_end = (end_s * self.sample_rate as f32).round() as usize;
        let progress = if candidate.usable_samples == 0 {
            0.0
        } else {
            sample_end as f32 / candidate.usable_samples as f32
        };

        Ok(Some(Segment {
            sample_range: sample_start..sample_end.max(sample_start),
            start: start_s as f64,
            duration: (end_s - start_s).max(0.0) as f64,
            elapsed_time: None,
            remaining_time: None,
            progress: progress.clamp(0.0, 1.0),
            result,
        }))
    }

    pub(crate) async fn transcribe(
        &mut self,
        samples: Vec<f32>,
        language: Option<WhisperLanguage>,
        word_timestamps: bool,
        result: UnboundedSender<Segment>,
    ) -> Result<(), crate::model::WhisperError> {
        if let Err(err) = self.validate_language(language) {
            return Err(crate::model::WhisperError::Fusor(fusor::Error::msg(
                err.to_string(),
            )));
        }
        let samples = self.usable_samples(&samples);
        if samples.is_empty() {
            return Ok(());
        }
        let (adapted_encoder_hidden_states, total_frames) = self.encode_prepared_full(samples)?;
        let clip_end_s = samples.len() as f32 / self.sample_rate as f32;
        let Some(candidate) = self
            .decode_candidate_from_prepared(
                &adapted_encoder_hidden_states,
                if total_frames == 0 { 0.0 } else { clip_end_s },
                samples.len(),
                true,
                &[],
                None,
            )
            .await?
        else {
            return Ok(());
        };
        if let Some(segment) = self.segment_from_token_range(
            0..candidate.tokens.len(),
            &candidate,
            word_timestamps,
            0.0,
        )? {
            let mut result = result;
            let _ = result.start_send(segment);
        }
        Ok(())
    }

    pub(crate) fn start_stream(
        &mut self,
        session_id: u64,
        language: Option<WhisperLanguage>,
        word_timestamps: bool,
        sender: UnboundedSender<Segment>,
    ) -> Result<(), crate::model::WhisperLoadingError> {
        self.validate_language(language)?;
        self.streams.insert(
            session_id,
            MoonshineStreamState {
                word_timestamps,
                sender,
                last_candidate: None,
                encoder_state: self.model.encoder.new_stream_state(),
                prepared_encoder_hidden_states: None,
                decoder_prefix_cache: MoonshineDecoderCache::default(),
                decoder_prefix_tokens: Vec::new(),
                last_decoded_finalized_frames: 0,
                emitted_tokens: 0,
                last_emitted_end_s: 0.0,
            },
        );
        Ok(())
    }

    pub(crate) async fn push_stream_audio(
        &mut self,
        session_id: u64,
        samples: Vec<f32>,
    ) -> Result<(), crate::model::WhisperError> {
        let Some(mut state) = self.streams.remove(&session_id) else {
            return Ok(());
        };
        let Some(mut candidate) = self
            .update_stream_candidate(&mut state, &samples, false)
            .await?
        else {
            self.streams.insert(session_id, state);
            return Ok(());
        };
        let stable = if let Some(previous) = &state.last_candidate {
            self.stable_emit_len(
                self.common_prefix_len(&previous.tokens, &candidate.tokens),
                false,
            )
        } else {
            candidate
                .tokens
                .len()
                .saturating_sub(self.stream_initial_holdback_tokens)
        };
        if stable > state.emitted_tokens {
            if state.word_timestamps {
                if let Some(prepared_encoder_hidden_states) =
                    state.prepared_encoder_hidden_states.as_ref()
                {
                    let prepared_encoder_hidden_states = prepared_encoder_hidden_states.clone();
                    self.populate_candidate_timestamps(
                        &mut candidate,
                        &prepared_encoder_hidden_states,
                    )
                    .await?;
                }
            }
            if let Some(segment) = self.segment_from_token_range(
                state.emitted_tokens..stable,
                &candidate,
                state.word_timestamps,
                state.last_emitted_end_s,
            )? {
                state.last_emitted_end_s = segment.start as f32 + segment.duration as f32;
                let mut sender = state.sender.clone();
                let _ = sender.start_send(segment);
            }
            state.emitted_tokens = stable;
        }
        state.last_candidate = Some(candidate);
        self.streams.insert(session_id, state);
        Ok(())
    }

    pub(crate) async fn finish_stream(
        &mut self,
        session_id: u64,
    ) -> Result<(), crate::model::WhisperError> {
        let Some(state) = self.streams.remove(&session_id) else {
            return Ok(());
        };
        let mut state = state;
        let Some(mut candidate) = self.update_stream_candidate(&mut state, &[], true).await? else {
            return Ok(());
        };
        if state.word_timestamps {
            if let Some(prepared_encoder_hidden_states) =
                state.prepared_encoder_hidden_states.as_ref()
            {
                self.populate_candidate_timestamps(&mut candidate, prepared_encoder_hidden_states)
                    .await?;
            }
        }
        if let Some(segment) = self.segment_from_token_range(
            state.emitted_tokens..candidate.tokens.len(),
            &candidate,
            state.word_timestamps,
            state.last_emitted_end_s,
        )? {
            let mut sender = state.sender;
            let _ = sender.start_send(segment);
        }
        Ok(())
    }
}
