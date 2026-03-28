use futures_channel::mpsc::UnboundedSender;
use half::bf16;
use tokenizers::Tokenizer;

use crate::{
    cohere_audio::pcm_to_features, cohere_config::CohereConfig, quantized::cohere::Cohere,
    DecodingResult, Segment, TokenChunk, WhisperLanguage,
};
use fusor::{Device, Tensor, VarBuilder};

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
        let tokenizer =
            Tokenizer::from_bytes(tokenizer_bytes).map_err(crate::model::WhisperLoadingError::LoadTokenizer)?;
        let eos_token = tokenizer
            .token_to_id("<|endoftext|>")
            .ok_or_else(|| fusor::Error::msg("missing <|endoftext|> token"))?;
        let max_new_tokens = std::env::var("RWHISPER_COHERE_MAX_NEW_TOKENS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(256);
        let filter_bytes = include_bytes!("cohere_melfilters128.bytes").as_slice();
        let mut filterbank = vec![0.0f32; filter_bytes.len() / 4];
        <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(filter_bytes, &mut filterbank);
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
    ) -> Result<String, crate::model::WhisperError> {
        let prompt_ids = self.prompt_ids(language)?;
        let (features, total_frames, valid_frames) =
            pcm_to_features(&self.model.config, samples, &self.filterbank);
        let input_features = Tensor::from_slice(
            &self.device,
            [1, self.model.config.preprocessor.features, total_frames],
            &features,
        );
        let generated = self
            .model
            .generate_greedy(
                &input_features,
                valid_frames,
                &prompt_ids,
                self.eos_token,
                self.max_new_tokens,
            )
            .await?;
        let text = self
            .tokenizer
            .decode(&generated, true)
            .map_err(crate::model::WhisperError::Tokenizer)?;
        Ok(text.trim().to_owned())
    }

    pub(crate) async fn transcribe(
        &self,
        samples: Vec<f32>,
        language: Option<WhisperLanguage>,
        result: UnboundedSender<Segment>,
    ) -> Result<(), crate::model::WhisperError> {
        let language = language.unwrap_or(WhisperLanguage::English);
        let text = self.transcribe_clip(&samples, language).await?;
        let segment = Segment {
            sample_range: 0..samples.len(),
            start: 0.0,
            duration: samples.len() as f64 / self.model.config.sample_rate as f64,
            elapsed_time: None,
            remaining_time: None,
            progress: 1.0,
            result: DecodingResult {
                text: text.clone(),
                avg_logprob: 0.0,
                no_speech_prob: 0.0,
                compression_ratio: 0.0,
                chunks: vec![TokenChunk {
                    text_range: 0..text.len(),
                    timestamp: None,
                }],
            },
        };
        let mut result = result;
        let _ = result.start_send(segment);
        Ok(())
    }
}
