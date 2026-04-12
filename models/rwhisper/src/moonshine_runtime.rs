use std::{collections::HashMap, num::NonZeroUsize, ops::Range};

use futures_channel::mpsc::UnboundedSender;
use tokenizers::Tokenizer;

use crate::{
    moonshine_config::MoonshineStreamingConfig,
    quantized::{moonshine::{Moonshine, MoonshineDecoderCache}, timestamps::extract_timestamps},
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
    samples: Vec<f32>,
    word_timestamps: bool,
    sender: UnboundedSender<Segment>,
    last_candidate: Option<CandidateDecode>,
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
                return Err(crate::model::WhisperLoadingError::UnsupportedLanguage(language));
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
            let last_logits = logits.narrow(1, last_index, 1).squeeze(1).squeeze(0).to_concrete();
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
            .decode_prepared(decoder_inputs.as_slice(), encoder_hidden_states, Some(&mut cross_attentions))
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

    async fn decode_candidate(
        &mut self,
        samples: &[f32],
    ) -> Result<Option<CandidateDecode>, crate::model::WhisperError> {
        let samples = self.usable_samples(samples);
        if samples.is_empty() {
            return Ok(None);
        }

        let encoder_hidden_states = self
            .model
            .encoder
            .encode(&self.device, samples)
            .map_err(crate::model::WhisperError::Fusor)?;
        let adapted_encoder_hidden_states = self
            .model
            .decoder
            .prepare_encoder_hidden_states(&encoder_hidden_states)
            .map_err(crate::model::WhisperError::Fusor)?;
        let max_new_tokens = self.max_new_tokens(samples.len());
        let generated = self
            .greedy_generate(&adapted_encoder_hidden_states, max_new_tokens)
            .await?;
        let clip_end_s = samples.len() as f32 / self.sample_rate as f32;
        let seconds_per_frame = if encoder_hidden_states.shape()[1] == 0 {
            0.0
        } else {
            clip_end_s / encoder_hidden_states.shape()[1] as f32
        };
        let token_timestamps = self
            .token_timestamps(&generated, &adapted_encoder_hidden_states, seconds_per_frame)
            .await?;

        Ok(Some(CandidateDecode {
            tokens: generated,
            token_timestamps,
            clip_end_s,
            usable_samples: samples.len(),
        }))
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

        let tokens = &candidate.tokens[range.clone()];
        let token_timestamps = if candidate.token_timestamps.is_empty() {
            vec![]
        } else {
            candidate.token_timestamps[range.clone()].to_vec()
        };
        let result = self.decoding_result_from_tokens(
            tokens,
            &token_timestamps,
            word_timestamps,
            candidate.clip_end_s,
        )?;
        if result.text.is_empty() {
            return Ok(None);
        }

        let mut start_s = token_timestamps.first().copied().unwrap_or(fallback_start_s);
        let mut end_s = token_timestamps.last().copied().unwrap_or(candidate.clip_end_s);
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
        let Some(candidate) = self.decode_candidate(&samples).await? else {
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
                samples: Vec::new(),
                word_timestamps,
                sender,
                last_candidate: None,
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
        state.samples.extend(samples);
        let Some(candidate) = self.decode_candidate(&state.samples).await? else {
            self.streams.insert(session_id, state);
            return Ok(());
        };
        if let Some(previous) = &state.last_candidate {
            let stable = self.common_prefix_len(&previous.tokens, &candidate.tokens);
            if stable > state.emitted_tokens {
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
        let Some(candidate) = self.decode_candidate(&state.samples).await? else {
            return Ok(());
        };
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
