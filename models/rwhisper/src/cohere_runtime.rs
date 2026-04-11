use futures_channel::mpsc::UnboundedSender;
use half::bf16;
use tokenizers::Tokenizer;

use crate::{
    cohere_audio::pcm_to_features,
    cohere_config::CohereConfig,
    quantized::{cohere::Cohere, timestamps::extract_timestamps},
    DecodingResult, Segment, TokenChunk, WhisperLanguage,
};
use fusor::{Device, Tensor, VarBuilder};
use std::num::NonZeroUsize;
use std::ops::Range;
use std::time::Instant;

pub(crate) struct CohereRuntime {
    device: Device,
    tokenizer: Tokenizer,
    model: Cohere,
    filterbank: Vec<f32>,
    eos_token: u32,
    max_new_tokens: usize,
}

impl CohereRuntime {
    pub(crate) fn new(
        device: Device,
        weights: &[u8],
        tokenizer_bytes: &[u8],
        config: CohereConfig,
    ) -> Result<Self, crate::model::WhisperLoadingError> {
        let tokenizer = Tokenizer::from_bytes(tokenizer_bytes)
            .map_err(crate::model::WhisperLoadingError::LoadTokenizer)?;
        let eos_token = tokenizer
            .token_to_id("<|endoftext|>")
            .ok_or_else(|| fusor::Error::msg("missing <|endoftext|> token"))?;
        let max_new_tokens = std::env::var("RWHISPER_COHERE_MAX_NEW_TOKENS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(256);
        let filter_bytes = include_bytes!("cohere_melfilters128.bytes").as_slice();
        let mut filterbank = vec![0.0f32; filter_bytes.len() / 4];
        <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(
            filter_bytes,
            &mut filterbank,
        );
        for value in &mut filterbank {
            *value = bf16::from_f32(*value).to_f32();
        }

        let mut reader = std::io::Cursor::new(weights);
        let mut vb = VarBuilder::from_gguf(&mut reader).map_err(|err| {
            crate::model::WhisperLoadingError::LoadModel(fusor::Error::msg(err.to_string()))
        })?;
        let model = Cohere::load(&device, &mut vb, config)?;

        Ok(Self {
            device,
            tokenizer,
            model,
            filterbank,
            eos_token,
            max_new_tokens,
        })
    }

    fn prompt_ids(&self, language: WhisperLanguage) -> Result<Vec<u32>, fusor::Error> {
        let language = language.to_string();
        let prompt = format!(
            "<|startofcontext|><|startoftranscript|><|emo:undefined|><|{language}|><|{language}|><|pnc|><|noitn|><|notimestamp|><|nodiarize|>"
        );
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|err| fusor::Error::msg(err.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    async fn transcribe_clip(
        &self,
        samples: &[f32],
        language: WhisperLanguage,
        with_timestamps: bool,
    ) -> Result<DecodingResult, crate::model::WhisperError> {
        let profile = std::env::var("RWHISPER_COHERE_PROFILE").ok().as_deref() == Some("1");
        let total_start = Instant::now();
        let prompt_ids = self.prompt_ids(language)?;
        if profile {
            eprintln!("cohere prompt ids: {:.3}s", total_start.elapsed().as_secs_f32());
        }
        let (features, total_frames, valid_frames) =
            pcm_to_features(&self.model.config, samples, &self.filterbank);
        if profile {
            eprintln!(
                "cohere features: {:.3}s total_frames={} valid_frames={}",
                total_start.elapsed().as_secs_f32(),
                total_frames,
                valid_frames
            );
        }
        let input_features = Tensor::from_slice(
            &self.device,
            [1, self.model.config.preprocessor.features, total_frames],
            &features,
        );
        if profile {
            eprintln!("cohere input tensor: {:.3}s", total_start.elapsed().as_secs_f32());
        }
        let (generated, token_timestamps) = if with_timestamps {
            let (generated, cross_attentions, encoder_length) = self
                .model
                .generate_greedy_with_attention(
                    &input_features,
                    valid_frames,
                    &prompt_ids,
                    self.eos_token,
                    self.max_new_tokens,
                )
                .await?;
            let duration_s = samples.len() as f32 / self.model.config.sample_rate as f32;
            let seconds_per_frame = if encoder_length == 0 {
                0.0
            } else {
                duration_s / encoder_length as f32
            };
            let token_mask = vec![vec![true; generated.len()]];
            let token_timestamps = extract_timestamps(
                None,
                &cross_attentions,
                const { NonZeroUsize::new(7).unwrap() },
                encoder_length,
                seconds_per_frame,
                token_mask,
            )
            .await?
            .into_iter()
            .next();
            (generated, token_timestamps)
        } else {
            if profile {
                eprintln!("cohere generate start: {:.3}s", total_start.elapsed().as_secs_f32());
            }
            (
                self.model
                    .generate_greedy(
                        &input_features,
                        valid_frames,
                        &prompt_ids,
                        self.eos_token,
                        self.max_new_tokens,
                    )
                    .await?,
                None,
            )
        };
        if profile {
            eprintln!(
                "cohere generate done: {:.3}s generated_tokens={}",
                total_start.elapsed().as_secs_f32(),
                generated.len()
            );
        }

        let mut remaining_tokens: Vec<_> = generated.iter().copied().enumerate().collect();
        remaining_tokens.reverse();
        let mut processed_tokens = Vec::new();
        let mut timestamp_start: Option<f32> = None;
        let mut prev_text_len = 0;
        let mut chunks = Vec::new();
        let mut current_text = String::new();
        let clip_end = samples.len() as f32 / self.model.config.sample_rate as f32;

        while let Some((index, token)) = remaining_tokens.pop() {
            processed_tokens.push(token);
            if let Some(timestamps) = &token_timestamps {
                if timestamp_start.is_none() {
                    timestamp_start = timestamps.get(index).copied();
                }
            }
            let detokenized = self
                .tokenizer
                .decode(&processed_tokens, true)
                .map_err(crate::model::WhisperError::Tokenizer)?;
            if detokenized.len() > prev_text_len
                && detokenized.chars().last().unwrap_or_default().is_ascii()
            {
                let timestamp = token_timestamps.as_ref().and_then(|timestamps| {
                    let start = timestamp_start?;
                    let end = timestamps.get(index).copied().unwrap_or(clip_end);
                    timestamp_start = Some(end);
                    Some(start..end)
                });
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

        Ok(DecodingResult {
            text: current_text,
            avg_logprob: 0.0,
            no_speech_prob: 0.0,
            compression_ratio: 0.0,
            chunks,
        })
    }

    pub(crate) async fn transcribe(
        &self,
        samples: Vec<f32>,
        language: Option<WhisperLanguage>,
        with_timestamps: bool,
        result: UnboundedSender<Segment>,
    ) -> Result<(), crate::model::WhisperError> {
        let language = language.unwrap_or(WhisperLanguage::English);
        let mut result = result;
        let chunk_ranges = split_audio_chunks_energy(&self.model.config, &samples);
        let total_samples = samples.len();
        let start_time = cfg!(not(target_arch = "wasm32")).then(Instant::now);

        for range in chunk_ranges {
            let mut decoding = self
                .transcribe_clip(&samples[range.clone()], language, with_timestamps)
                .await?;
            let start_seconds = range.start as f32 / self.model.config.sample_rate as f32;
            for chunk in &mut decoding.chunks {
                if let Some(timestamp) = &mut chunk.timestamp {
                    timestamp.start += start_seconds;
                    timestamp.end += start_seconds;
                }
            }
            let elapsed_time = start_time.map(|start| start.elapsed());
            let remaining_time = elapsed_time.map(|elapsed| {
                let processed = range.end.max(1);
                std::time::Duration::from_millis(
                    ((elapsed.as_millis() as usize / processed) * (total_samples.saturating_sub(range.end)))
                        as u64,
                )
            });
            let segment = Segment {
                sample_range: range.clone(),
                start: start_seconds as f64,
                duration: (range.end - range.start) as f64 / self.model.config.sample_rate as f64,
                elapsed_time,
                remaining_time,
                progress: range.end as f32 / total_samples.max(1) as f32,
                result: decoding,
            };
            let _ = result.start_send(segment);
        }
        Ok(())
    }
}

fn split_audio_chunks_energy(config: &CohereConfig, waveform: &[f32]) -> Vec<Range<usize>> {
    let sample_rate = config.sample_rate;
    let chunk_size = (config.max_audio_clip_s * sample_rate).max(1);
    let boundary_context_size = (config.overlap_chunk_second * sample_rate).max(1);
    let fast_path_threshold = chunk_size.saturating_sub(boundary_context_size);

    if waveform.len() <= fast_path_threshold {
        return vec![0..waveform.len()];
    }

    let mut chunks = Vec::new();
    let mut idx = 0usize;
    while idx < waveform.len() {
        if idx + chunk_size >= waveform.len() {
            chunks.push(idx..waveform.len());
            break;
        }

        let search_start = (idx + chunk_size).saturating_sub(boundary_context_size).max(idx);
        let search_end = (idx + chunk_size).min(waveform.len());
        let split_point = if search_end <= search_start {
            idx + chunk_size
        } else {
            find_split_point_energy(
                waveform,
                search_start,
                search_end,
                config.min_energy_window_samples,
            )
        };
        let split_point = split_point.max(idx + 1).min(waveform.len());
        chunks.push(idx..split_point);
        idx = split_point;
    }

    chunks
}

fn find_split_point_energy(
    waveform: &[f32],
    start_idx: usize,
    end_idx: usize,
    min_energy_window_samples: usize,
) -> usize {
    let segment = &waveform[start_idx..end_idx];
    if segment.len() <= min_energy_window_samples {
        return (start_idx + end_idx) / 2;
    }

    let mut min_energy = f32::INFINITY;
    let mut quietest_idx = start_idx;
    let upper = segment.len().saturating_sub(min_energy_window_samples);
    let step = min_energy_window_samples.max(1);
    let mut i = 0usize;
    while i < upper {
        let window = &segment[i..(i + min_energy_window_samples)];
        let energy = (window.iter().map(|sample| sample * sample).sum::<f32>() / window.len() as f32)
            .sqrt();
        if energy < min_energy {
            min_energy = energy;
            quietest_idx = start_idx + i;
        }
        i += step;
    }
    quietest_idx
}
