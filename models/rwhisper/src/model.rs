use flate2::{write::ZlibEncoder, Compression};
use fusor::{cache::TensorCache, Device, Error, Tensor};
use futures_channel::mpsc::UnboundedSender;
use rand::{
    distr::{weighted::WeightedIndex, Distribution},
    SeedableRng,
};
use std::{
    io::Write,
    num::NonZeroUsize,
    ops::RangeInclusive,
    time::{Duration, Instant},
};
use tokenizers::Tokenizer;

use super::{DecodingResult, Segment};
use crate::{
    audio, config::*, quantized::TextDecoderCache, Task, TaskType, TokenChunk, WhisperBuilder,
    WhisperLanguage,
};
use kalosm_common::CacheError;

enum ModelType {
    Quantized(crate::quantized::Whisper),
}

impl ModelType {
    fn load(weights: &[u8], device: &Device, config: Config) -> fusor::Result<Self> {
        let mut reader = std::io::Cursor::new(weights);
        let mut vb = fusor::VarBuilder::from_gguf(&mut reader)?;
        Ok(Self::Quantized(crate::quantized::Whisper::load(
            device, &mut vb, config,
        )?))
    }

    fn config(&self) -> &Config {
        match self {
            Self::Quantized(model) => &model.config,
        }
    }
}

/// An error that can occur when loading a [`Whisper`](crate::Whisper) model.
#[derive(Debug, thiserror::Error)]
pub enum WhisperLoadingError {
    /// An error that can occur when trying to load a [`Whisper`](crate::Whisper) model from huggingface or a local file.
    #[error("Failed to load model from huggingface or local file: {0}")]
    DownloadingError(#[from] CacheError),
    /// An error that can occur when trying to load a [`Whisper`](crate::Whisper) model.
    #[error("Failed to load model into device: {0}")]
    LoadModel(#[from] fusor::Error),
    /// An error that can occur when trying to load the whisper tokenizer.
    #[error("Failed to load tokenizer: {0}")]
    LoadTokenizer(tokenizers::Error),
    /// An error that can occur when trying to load the whisper config.
    #[error("Failed to load config: {0}")]
    LoadConfig(serde_json::Error),
    /// Unsupported mel filter length
    #[error("Unsupported mel filter length: {0}; only 80 and 128 are supported")]
    UnsupportedMelFilterLength(usize),
    /// Language not supported
    #[error("Language not supported: {0}")]
    UnsupportedLanguage(WhisperLanguage),
}

/// An error that can occur when running a [`Whisper`] model.
#[derive(Debug, thiserror::Error)]
pub enum WhisperError {
    /// An error that can occur when trying to run a [`Whisper`] model.
    #[error("Fusor error: {0}")]
    Fusor(#[from] fusor::Error),
    /// An error that can occur when encoding or decoding for a [`Whisper`] model.
    #[error("Tokenizer error: {0}")]
    Tokenizer(tokenizers::Error),
    /// An error that can occur when compressing the text the model generates to determine the compression ratio.
    #[error("Compression error: {0}")]
    Compression(std::io::Error),
}

pub(crate) struct WhisperInner {
    mel_filters: Vec<f32>,
    device: Device,
    decoder: Decoder,
    config: Config,
}

impl WhisperInner {
    pub(crate) async fn new(
        settings: WhisperBuilder,
        weights: &[u8],
        tokenizer: &[u8],
        config: &[u8],
    ) -> Result<Self, WhisperLoadingError> {
        // Set FUSOR_USE_GPU=1 to use GPU, otherwise CPU
        let use_gpu = std::env::var("FUSOR_USE_GPU")
            .map(|v| v == "1")
            .unwrap_or(false);
        let device = if use_gpu {
            Device::new().await?
        } else {
            Device::cpu()
        };

        let tokenizer =
            Tokenizer::from_bytes(tokenizer).map_err(WhisperLoadingError::LoadTokenizer)?;
        let config: Config =
            serde_json::from_slice(config).map_err(WhisperLoadingError::LoadConfig)?;

        let mel_bytes = match config.num_mel_bins {
            80 => include_bytes!("melfilters.bytes").as_slice(),
            128 => include_bytes!("melfilters128.bytes").as_slice(),
            nmel => return Err(WhisperLoadingError::UnsupportedMelFilterLength(nmel)),
        };
        let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
        <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(
            mel_bytes,
            &mut mel_filters,
        );
        let attention_heads = settings.model.heads;

        let model = ModelType::load(weights, &device, config.clone())?;
        let language_token = if settings.model.multilingual {
            let language = settings.language.unwrap_or(WhisperLanguage::English);
            match token_id(&tokenizer, &format!("<|{language}|>")) {
                Ok(token_id) => Some(token_id),
                Err(_) => return Err(WhisperLoadingError::UnsupportedLanguage(language)),
            }
        } else {
            None
        };
        let decoder = Decoder::new(
            model,
            tokenizer,
            0,
            &device,
            language_token,
            attention_heads,
        )?;

        Ok(Self {
            mel_filters,
            device,
            decoder,
            config,
        })
    }

    pub(crate) async fn transcribe(
        &mut self,
        pcm_data: Vec<f32>,
        word_level_time_stamps: bool,
        language: Option<WhisperLanguage>,
        result: UnboundedSender<Segment>,
    ) {
        let mel = audio::pcm_to_mel(&self.config, &pcm_data, &self.mel_filters);
        let mel_len = mel.len();
        let mel = Tensor::new(&self.device, &mel)
            .reshape([self.config.num_mel_bins, mel_len / self.config.num_mel_bins])
            .to_concrete()
            .cast();

        if let Some(language) = language {
            if let Err(err) = self.decoder.set_language_token(language) {
                // Log error or send error message to result channel
                // Continue with default language
                tracing::error!("Error updating language token: {err}");
            }
        }

        if let Err(err) = self
            .decoder
            .run(
                &mel,
                pcm_data.len(),
                Task {
                    task_type: TaskType::Unset,
                    word_level_time_stamps,
                    without_timestamps: true,
                },
                result,
            )
            .await
        {
            tracing::error!("Error transcribing audio: {err}");
        }
    }
}

struct Decoder {
    model: ModelType,
    rng: rand::rngs::StdRng,
    tokenizer: Tokenizer,
    suppress_tokens: Tensor<1, crate::WhisperDType>,
    sot_token: u32,
    transcribe_token: u32,
    translate_token: u32,
    eot_token: u32,
    no_speech_token: u32,
    no_timestamps_token: u32,
    language_token: Option<u32>,
    timestamp_token_range: RangeInclusive<u32>,
    attention_heads: Option<&'static [[usize; 2]]>,
}

impl Decoder {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: ModelType,
        tokenizer: Tokenizer,
        seed: u64,
        device: &Device,
        language_token: Option<u32>,
        attention_heads: Option<&'static [[usize; 2]]>,
    ) -> fusor::Result<Self> {
        let no_timestamps_token = token_id(&tokenizer, NO_TIMESTAMPS_TOKEN)?;
        // Suppress the notimestamps token when in timestamps mode.
        // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L452
        let suppress_tokens: Vec<crate::WhisperDType> = (0..model.config().vocab_size as u32)
            .map(|i| {
                if model.config().suppress_tokens.contains(&i) {
                    crate::WhisperDType::NEG_INFINITY
                } else {
                    crate::WhisperDType::from(0.0)
                }
            })
            .collect();
        let suppress_tokens = Tensor::new(device, suppress_tokens.as_slice());
        let sot_token = token_id(&tokenizer, SOT_TOKEN)?;
        let transcribe_token = token_id(&tokenizer, TRANSCRIBE_TOKEN)?;
        let translate_token = token_id(&tokenizer, TRANSLATE_TOKEN)?;
        let eot_token = token_id(&tokenizer, EOT_TOKEN)?;
        let no_speech_token = NO_SPEECH_TOKENS
            .iter()
            .find_map(|token| token_id(&tokenizer, token).ok())
            .ok_or_else(|| fusor::Error::msg("no_speech_token not found"))?;
        let timestamp_tokens = (0..=1500)
            .map(|i| token_id(&tokenizer, &format!("<|{:.2}|>", i as f32 * 0.02)))
            .collect::<fusor::Result<Vec<_>>>()?;
        let timestamp_token_range =
            *timestamp_tokens.first().unwrap()..=*timestamp_tokens.last().unwrap();
        debug_assert!(timestamp_tokens
            .iter()
            .all(|t| timestamp_token_range.contains(t)));

        Ok(Self {
            model,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            tokenizer,
            suppress_tokens,
            sot_token,
            transcribe_token,
            translate_token,
            eot_token,
            no_speech_token,
            language_token,
            no_timestamps_token,
            timestamp_token_range,
            attention_heads,
        })
    }

    pub(crate) fn is_special(&self, token: u32) -> bool {
        self.timestamp_token_range.contains(&token)
            || token == self.sot_token
            || token == self.transcribe_token
            || token == self.translate_token
            || token == self.no_timestamps_token
            || token == self.eot_token
            || Some(token) == self.language_token
            || token == self.no_speech_token
    }

    fn set_language_token(&mut self, language: WhisperLanguage) -> Result<(), WhisperLoadingError> {
        match token_id(&self.tokenizer, &format!("<|{language}|>")) {
            Ok(token_id) => self.language_token = Some(token_id),
            Err(_) => return Err(WhisperLoadingError::UnsupportedLanguage(language)),
        }
        Ok(())
    }

    fn is_timestamp_or_eot(&self, token: u32) -> bool {
        self.timestamp_token_range.contains(&token) || token == self.eot_token
    }

    fn special_tokens(&self) -> impl Iterator<Item = u32> + '_ {
        self.timestamp_token_range
            .clone()
            .chain(std::iter::once(self.eot_token))
    }

    fn encode(
        &mut self,
        mel: &Tensor<3, crate::WhisperDType>,
    ) -> fusor::Result<Tensor<3, crate::WhisperDType>> {
        let tensor = match &mut self.model {
            ModelType::Quantized(model) => model.encoder.forward(mel)?,
        };

        Ok(tensor)
    }

    async fn decode(
        &mut self,
        audio_features: &Tensor<2, crate::WhisperDType>,
        temperature: f64,
        task: Task,
        previous_tokens: &[u32],
        n_frames: usize,
    ) -> Result<DecodingResult, WhisperError> {
        let sample_len = self.model.config().max_target_positions / 2;
        let mut sum_logprob = 0f64;
        let mut no_speech_prob = f64::NAN;
        let mut tokens = vec![self.sot_token];
        if let Some(language_token) = self.language_token {
            tokens.push(language_token);
        }
        match task.task_type {
            TaskType::Transcribe => tokens.push(self.transcribe_token),
            TaskType::Translate => tokens.push(self.translate_token),
            TaskType::Unset => {}
        }
        if task.without_timestamps {
            tokens.push(self.no_timestamps_token);
        } else {
            tokens.push(*self.timestamp_token_range.start());
        }
        tokens.extend(previous_tokens);
        let mut token_mask = vec![false; tokens.len()];
        // The tokens that are queued for decoding
        let mut queued_tokens = tokens.clone();
        let mut cache = TextDecoderCache::new();
        let mut attention_output = None;
        for i in 0..sample_len {
            let ys = match &mut self.model {
                ModelType::Quantized(model) => {
                    if task.word_level_time_stamps && i == 0 {
                        attention_output = Some({
                            let mut outputs = Vec::new();
                            for _ in 0..model.decoder.block_count() {
                                outputs.push(TensorCache::new(2, usize::MAX));
                            }
                            outputs
                        });
                    }
                    if let Some(last_mut) = queued_tokens.last_mut() {
                        if last_mut == &self.eot_token {
                            // When configured to output word-level timestamps, the OpenAI inference
                            // implementation passes a timestamp token with the nearest second in the
                            // last pass. While the predicted token from this pass is not included in the
                            // output transcript, it impacts the word/token-level timestamps.
                            let nearest_second =
                                n_frames as f32 * HOP_LENGTH as f32 / SAMPLE_RATE as f32;
                            let nearest_second_02 = nearest_second / 0.02;
                            let nearest_second_02 = nearest_second_02 as usize;
                            let timestamp_token =
                                *self.timestamp_token_range.start() + nearest_second_02 as u32;

                            *last_mut = timestamp_token;
                        }
                    }
                    let result = model.decoder.forward(
                        &queued_tokens,
                        audio_features,
                        &mut cache,
                        attention_output.as_deref_mut(),
                    )?;

                    // The quantized model caches tokens so we can remove any old tokens
                    queued_tokens.clear();
                    result
                }
            };

            // Extract the no speech probability on the first iteration by looking at the first
            // token logits and the probability for the according token.
            if i == 0 {
                let logits = match &mut self.model {
                    ModelType::Quantized(model) => {
                        let ys_slice = ys.narrow(0, 0, 1).to_concrete();
                        model.decoder.final_linear(&ys_slice)?
                    }
                };
                let logits_2d = logits.i((.., 0, ..));
                let logits_1d = logits_2d.narrow(0, 0, 1).squeeze(0).to_concrete();
                let softmax_result = logits_1d.softmax(0);
                let token_prob = softmax_result
                    .narrow(0, self.no_speech_token as usize, 1)
                    .squeeze(0)
                    .to_concrete();
                no_speech_prob = token_prob
                    .to_scalar()
                    .await
                    .map_err(|e: fusor::Error| WhisperError::Fusor(e))?
                    .into();
            }

            let [_, seq_len, _] = ys.shape();
            let logits = match &mut self.model {
                ModelType::Quantized(model) => {
                    let ys_slice = ys.narrow(0, 0, 1).narrow(1, seq_len - 1, 1).to_concrete();
                    model.decoder.final_linear(&ys_slice)?
                }
            };
            let logits_2d = logits.i((.., 0, ..));
            let logits_2d_narrowed = logits_2d.narrow(0, 0, 1);
            let logits_1d = logits_2d_narrowed.squeeze(0);

            // TODO: Besides suppress tokens, we should apply the heuristics from
            // ApplyTimestampRules, i.e.:
            // - Timestamps come in pairs, except before EOT.
            // - Timestamps should be non-decreasing.
            // - If the sum of the probabilities of timestamps is higher than any other tokens,
            //   only consider timestamps when sampling.
            // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L439
            let logits = (&logits_1d + &self.suppress_tokens).to_concrete();
            let next_token = if temperature > 0f64 {
                let prs = logits
                    .clone()
                    .div_scalar(crate::WhisperDType::from(temperature as f32));
                let prs = prs.softmax(0);
                let logits_v = self
                    .apply_timestamp_rules(prs, &tokens, task.without_timestamps)
                    .await?;
                // Weights may be NaN if decoding fails
                let distr = WeightedIndex::new(&logits_v)
                    .map_err(|_| fusor::Error::msg("Weights were invalid distribution"))?;
                distr.sample(&mut self.rng) as u32
            } else {
                let logits_sm = logits.softmax(0);
                let logits_v = self
                    .apply_timestamp_rules(logits_sm, &tokens, task.without_timestamps)
                    .await?;
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };
            token_mask.push(!self.is_special(next_token));
            // Debug: print top 5 tokens and their probabilities
            if i < 5 {
                let logits_sm = logits.softmax(0);
                let logits_slice = logits_sm.as_slice().await.unwrap();
                let logits_data = logits_slice.as_slice();

                let mut indexed: Vec<(usize, f32)> = logits_data
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (i, *v))
                    .collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            }
            // After the final pass if word level timestamps are requested, we stop decoding
            if task.word_level_time_stamps && tokens.last() == Some(&self.eot_token) {
                break;
            }
            tokens.push(next_token);
            queued_tokens.push(next_token);
            let prob_softmax = logits.softmax_last_dim();
            let prob_narrowed = prob_softmax.narrow(0, next_token as usize, 1);
            let prob_tensor = prob_narrowed.squeeze(0);
            let prob: f64 = prob_tensor
                .to_scalar()
                .await
                .map_err(WhisperError::Fusor)?
                .into();
            // If we have read the maximum number of tokens, stop regardless of the eot token
            // Or if word level timestamps are disabled, stop as soon was we reach the eot token
            if tokens.len() > self.model.config().max_target_positions
                || (!task.word_level_time_stamps && next_token == self.eot_token)
            {
                break;
            }
            sum_logprob += prob.ln();
        }
        let mut token_timestamps = None;
        if let Some(attention_output) = attention_output.as_mut() {
            let result = crate::quantized::Whisper::dtw_timestamps(
                self.attention_heads,
                const { NonZeroUsize::new(7).unwrap() },
                n_frames,
                vec![token_mask],
                attention_output,
            )
            .await?;
            if let [timestamps] = result.as_slice() {
                token_timestamps = Some(timestamps.clone());
            }
        }

        let (text, chunks) = {
            let mut remaining_tokens: Vec<_> = tokens
                .iter()
                .copied()
                .filter(|t| !self.is_special(*t))
                .enumerate()
                .collect();
            remaining_tokens.reverse();
            let mut queued_tokens = Vec::new();
            let mut timestamp_start: Option<f32> = None;
            let mut prev_text_len = 0;
            let mut chunks = Vec::new();
            let mut current_text = String::new();
            while let Some((index, token)) = remaining_tokens.pop() {
                queued_tokens.push(token);
                if let Some(timestamps) = &token_timestamps {
                    if timestamp_start.is_none() {
                        timestamp_start = Some(timestamps[index]);
                    }
                }
                let detokenized = self
                    .tokenizer
                    .decode(&queued_tokens, true)
                    .map_err(WhisperError::Tokenizer)?;
                if detokenized.len() > prev_text_len
                    && detokenized.chars().last().unwrap().is_ascii()
                {
                    let timestamp = token_timestamps.as_ref().map(|timestamps| {
                        let start = timestamp_start.unwrap();
                        let end = timestamps.get(index).copied().unwrap_or_else(|| {
                            n_frames as f32 * HOP_LENGTH as f32 / SAMPLE_RATE as f32
                        });
                        timestamp_start = Some(end);
                        start..end
                    });
                    let text_range = current_text.len()..current_text.len() + detokenized.len();
                    current_text += &detokenized;
                    queued_tokens.clear();
                    prev_text_len = 0;
                    let token = TokenChunk {
                        text_range,
                        timestamp,
                    };
                    chunks.push(token);
                } else {
                    prev_text_len = detokenized.len();
                }
            }

            if !queued_tokens.is_empty() {
                let detokenized = self
                    .tokenizer
                    .decode(&queued_tokens, true)
                    .map_err(WhisperError::Tokenizer)?;
                let timestamp = token_timestamps.as_ref().map(|timestamps| {
                    let start = timestamp_start.unwrap();
                    let end = *timestamps.last().unwrap();
                    start..end
                });
                let text_range = current_text.len()..current_text.len() + detokenized.len();
                current_text += &detokenized;
                let token = TokenChunk {
                    text_range,
                    timestamp,
                };
                chunks.push(token);
            }

            (current_text, chunks)
        };
        let avg_logprob = sum_logprob / tokens.len() as f64;

        let compression_ratio = {
            let mut e = ZlibEncoder::new(Vec::new(), Compression::default());
            e.write_all(text.as_bytes())
                .map_err(WhisperError::Compression)?;
            let compressed_bytes = e.finish();

            compressed_bytes
                .map(|buf| text.len() as f64 / buf.len() as f64)
                .unwrap_or_default()
        };

        Ok(DecodingResult {
            text,
            avg_logprob,
            no_speech_prob,
            compression_ratio,
            chunks,
        })
    }

    async fn decode_with_fallback(
        &mut self,
        audio_features: &Tensor<2, crate::WhisperDType>,
        task: Task,
        previous_tokens: &[u32],
        n_frames: usize,
    ) -> Result<DecodingResult, WhisperError> {
        for (i, &t) in TEMPERATURES.iter().enumerate() {
            let dr: Result<DecodingResult, WhisperError> = self
                .decode(audio_features, t, task, previous_tokens, n_frames)
                .await;
            if i == TEMPERATURES.len() - 1 {
                return dr;
            }
            // On errors, we try again with a different temperature.
            match dr {
                Ok(dr) => {
                    let needs_fallback = dr.compression_ratio > COMPRESSION_RATIO_THRESHOLD
                        || dr.avg_logprob < LOGPROB_THRESHOLD;
                    if !needs_fallback && dr.no_speech_prob < NO_SPEECH_THRESHOLD {
                        return Ok(dr);
                    }
                }
                Err(err) => {
                    tracing::trace!("Error running at {t}: {err}")
                }
            }
        }
        unreachable!()
    }

    async fn run(
        &mut self,
        mel: &Tensor<2, crate::WhisperDType>,
        audio_frames: usize,
        task: Task,
        mut result: UnboundedSender<Segment>,
    ) -> Result<(), WhisperError> {
        // TODO: This should be dynamic based on how much memory the model uses and how much memory is available
        const MAX_CHUNKS: usize = 1;

        let [_, content_frames] = mel.shape();
        let mut seek = 0;
        let start_time = cfg!(not(target_arch = "wasm32")).then(Instant::now);
        let mut chunk_indices = Vec::new();
        let mut chunked = Vec::new();
        // Keep looping until we have all the chunks we need
        while seek < content_frames {
            // Take a chunk up to the maximum size
            chunk_indices.clear();
            chunked.clear();
            while chunk_indices.len() < MAX_CHUNKS && seek < content_frames {
                let remaining_frames = content_frames - seek;
                let segment_size = usize::min(remaining_frames, N_FRAMES);
                // If the new frame doesn't fit into a perfect chunk, just include it in the next chunk
                if remaining_frames < N_FRAMES && !chunk_indices.is_empty() {
                    break;
                }
                chunk_indices.push(seek..seek + segment_size);
                let mel_segment = mel.narrow(1, seek, segment_size);
                chunked.push(mel_segment);
                seek += segment_size;
            }

            // Encode all of the chunks
            let batched_mel_segment: Tensor<3, crate::WhisperDType> =
                Tensor::stack(chunked.iter().cloned(), 0);

            let batched_audio_features = self.encode(&batched_mel_segment)?;

            let split = batched_audio_features.chunk(chunk_indices.len(), 0);

            // Tokens that are remaining in the last chunk's sentence fragment
            let mut tokens_in_sentence_fragment = Vec::new();

            for (audio_features, range) in split.iter().zip(chunk_indices.iter()) {
                let segment_size = range.end - range.start;
                let start = range.start;
                let end = range.end;
                let start_time_offset = (start * HOP_LENGTH) as f64 / SAMPLE_RATE as f64;
                let time_offset = (end * HOP_LENGTH) as f64 / SAMPLE_RATE as f64;

                let segment_duration = (segment_size * HOP_LENGTH) as f64 / SAMPLE_RATE as f64;

                // Squeeze the batch dimension since decode_with_fallback expects 2D tensor
                let audio_features_2d = audio_features.squeeze(0).to_concrete();

                let mut dr = self
                    .decode_with_fallback(
                        &audio_features_2d,
                        task,
                        &tokens_in_sentence_fragment,
                        segment_size,
                    )
                    .await?;

                for chunk in dr.chunks.iter_mut() {
                    // Change to iter_mut() to allow mutable access
                    if let Some(timestamp) = &mut chunk.timestamp {
                        timestamp.start += start_time_offset as f32;
                        timestamp.end += start_time_offset as f32;
                    }
                }
                tokens_in_sentence_fragment.clear();
                if dr.no_speech_prob > NO_SPEECH_THRESHOLD && dr.avg_logprob < LOGPROB_THRESHOLD {
                    tracing::trace!("no speech detected, skipping {end} {dr:?}");
                    continue;
                }

                // Grab any text that was in the previous sentence fragment
                if let Some(index) = dr.text.char_indices().rev().find_map(|(idx, c)| {
                    if c == '.' || c == '?' || c == '!' {
                        Some(idx)
                    } else {
                        None
                    }
                }) {
                    let text_after_last_sentence = &dr.text[index + 1..];
                    let tokens = self
                        .tokenizer
                        .encode(text_after_last_sentence, false)
                        .map_err(WhisperError::Tokenizer)?;
                    tokens_in_sentence_fragment.extend(tokens.get_ids());
                };

                let elapsed = start_time.map(|start| start.elapsed());
                let remaining = elapsed.map(|elapsed| {
                    Duration::from_millis(
                        ((elapsed.as_millis() as usize / seek) * (content_frames - seek)) as u64,
                    )
                });
                let progress = end as f32 / content_frames as f32;
                let segment = Segment {
                    sample_range: (range.start * HOP_LENGTH)
                        ..audio_frames.min(range.end * HOP_LENGTH),
                    start: time_offset,
                    duration: segment_duration,
                    remaining_time: remaining,
                    elapsed_time: elapsed,
                    progress,
                    result: dr,
                };

                if let Err(err) = result.start_send(segment) {
                    tracing::error!("Error sending segment: {err}");
                    break;
                }
            }
        }

        Ok(())
    }
}

pub fn token_id(tokenizer: &Tokenizer, token: &str) -> fusor::Result<u32> {
    match tokenizer.token_to_id(token) {
        None => Err(Error::msg(format!("no token-id for {token}"))),
        Some(id) => Ok(id),
    }
}

impl Decoder {
    async fn apply_timestamp_rules(
        &self,
        logits: Tensor<1, crate::WhisperDType>,
        tokens: &[u32],
        no_timestamps: bool,
    ) -> fusor::Result<Vec<crate::WhisperDType>> {
        let logits_slice = logits.as_slice().await?;
        let len = logits_slice.shape()[0];
        let mut logits = (0..len).map(|i| logits_slice[[i]]).collect::<Vec<_>>();

        logits[self.no_timestamps_token as usize] = crate::WhisperDType::from(0.0);
        logits[self.sot_token as usize] = crate::WhisperDType::from(0.0);
        logits[self.transcribe_token as usize] = crate::WhisperDType::from(0.0);
        logits[self.translate_token as usize] = crate::WhisperDType::from(0.0);

        if no_timestamps {
            for i in self.timestamp_token_range.clone() {
                logits[i as usize] = crate::WhisperDType::from(0.0);
            }
            return Ok(logits);
        }

        let mut iter_rev = tokens.iter().rev();
        let last_was_timestamp = iter_rev
            .next()
            .map(|t| self.is_special(*t))
            .unwrap_or(false);
        let penultimate_was_timestamp =
            iter_rev.next().map(|t| self.is_special(*t)).unwrap_or(true);

        match (penultimate_was_timestamp, last_was_timestamp) {
            // If the last two tokens were timestamps, then the new token cannot be a timestamp
            (true, true) => {
                for i in self.special_tokens() {
                    logits[i as usize] = crate::WhisperDType::from(0.0);
                }
            }
            // If the last token was a timestamp and the penultimate token was not, then the new token must be a timestamp
            (false, true) => {
                for (i, logit) in logits.iter_mut().enumerate() {
                    if !self.is_timestamp_or_eot(i as u32) {
                        *logit = crate::WhisperDType::from(0.0);
                    }
                }
            }
            _ => {}
        }

        // Make sure timestamps don't decrease
        let timestamp_last = tokens
            .iter()
            .rev()
            .find(|t| self.timestamp_token_range.contains(t))
            .copied()
            .unwrap_or(0);
        let timestamp_last = if last_was_timestamp && !penultimate_was_timestamp {
            timestamp_last
        } else {
            timestamp_last + 1
        };

        for (i, logit) in logits.iter_mut().enumerate() {
            if self.timestamp_token_range.contains(&(i as u32)) && i < timestamp_last as usize {
                *logit = crate::WhisperDType::from(0.0);
            }
        }

        // If the sum of the probability over timestamps is more than any other individual token, sample a timestamp
        let mut timestamp_sum_prob = crate::WhisperDType::from(0.0);
        let mut max_text_token_prob = crate::WhisperDType::from(0.0);
        for (i, logit) in logits.iter().enumerate() {
            if self.is_timestamp_or_eot(i as u32) {
                timestamp_sum_prob += logit;
            } else if *logit > max_text_token_prob {
                max_text_token_prob = *logit;
            }
        }

        if timestamp_sum_prob > max_text_token_prob {
            for (i, logit) in logits.iter_mut().enumerate() {
                if !self.is_timestamp_or_eot(i as u32) {
                    *logit = crate::WhisperDType::from(0.0);
                }
            }
        }

        Ok(logits)
    }
}
