use crate::gguf_tokenizer::get_pre_tokenizer;
use crate::raw::cache::LlamaCache;
use crate::raw::Model;
use crate::token_stream::TokenOutputStream;
use crate::token_stream::TokenOutputStreamError;
use crate::LlamaConfigJson;
use fusor::AddOp;
use fusor::CastTensor;
use fusor::CastTo;
use fusor::Device;
use fusor::FloatDataType;
use fusor::FloatOps;
use fusor::MatmulImpl;
use fusor::MulOp;
use fusor::ShardedVarBuilder;
use fusor::SimdBinaryOp;
use fusor::SimdElement;
use fusor::SimdReduceOp;
use fusor::SumOp;
use fusor::{WasmNotSend, WasmNotSync};
use fusor_gguf::GgufMetadata;
use fusor_gguf::GgufValue;
use kalosm_language_model::ImageFetchError;
use kalosm_language_model::MediaHints;
use kalosm_model_types::ModelLoadingProgress;
use llm_samplers::types::Logits;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use tokenizers::Tokenizer;

use crate::{InferenceSettings, LlamaSourceError};

/// An error that can occur when running a [`LlamaModel`].
#[derive(Debug, thiserror::Error)]
pub enum LlamaModelError {
    /// An error from candle while running the model.
    #[error("Candle error: {0}")]
    Candle(#[from] fusor::Error),

    /// An error from tokenizers while running the model.
    #[error("Tokenizer error: {0}")]
    Tokenizer(tokenizers::Error),

    /// An error while sampling tokens.
    #[error("Sampler error: {0}")]
    SamplerError(Box<dyn std::error::Error + Send + Sync>),

    /// A streaming detokenization error.
    #[error("Token output stream error: {0}")]
    TokenOutputStreamError(TokenOutputStreamError),

    /// An error while writing to the session cache.
    #[error("Session cache error: {0}")]
    Session(String),

    /// No valid tokens were sampled during structured generation
    #[error("No valid tokens were sampled")]
    NoValidTokens,

    /// The model has already stopped.
    #[error("Model stopped")]
    ModelStopped,

    /// No chat template was provided
    #[error("No chat template was provided")]
    NoChatTemplate,

    /// Error running the chat template
    #[error("Error running the chat template: {0}")]
    ChatTemplateError(#[from] minijinja::Error),

    /// Cannot run the model on an empty input
    #[error("Cannot run the model on an empty input")]
    EmptyInput,

    /// Failed to load images
    #[error("Failed to load images: {0}")]
    ImageLoadingError(#[from] ImageFetchError),
}

impl From<image::ImageError> for LlamaModelError {
    fn from(err: image::ImageError) -> Self {
        LlamaModelError::ImageLoadingError(err.into())
    }
}

/// The inner, synchronous Llama model.
pub(crate) struct LlamaModel<F: FloatDataType + SimdElement = f32> {
    pub(crate) model: Model<F>,
    pub(crate) device: Device,
    pub(crate) tokenizer: Arc<Tokenizer>,
}

impl<F: FloatDataType + SimdElement + Default + FloatOps + MatmulImpl> LlamaModel<F>
where
    F: CastTo<f32> + CastTensor<f32> + WasmNotSend + WasmNotSync + 'static,
    f32: CastTo<F> + CastTensor<F>,
    MulOp: SimdBinaryOp<F>,
    AddOp: SimdBinaryOp<F>,
    SumOp: SimdReduceOp<F>,
{
    pub(crate) fn forward(
        model: &Model<F>,
        device: &Device,
        tokens: &[u32],
        images: &[(image::DynamicImage, MediaHints)],
        cache: Option<&mut LlamaCache>,
        #[allow(unused)] tokenizer: &Tokenizer,
    ) -> impl kalosm_model_types::FutureWasmNotSend<Output = Result<Vec<f32>, LlamaModelError>>
    {
        if tokens.is_empty() {
            return Box::pin(async { Err(LlamaModelError::EmptyInput) })
                as Pin<
                    Box<
                        dyn kalosm_model_types::FutureWasmNotSend<
                            Output = Result<Vec<f32>, LlamaModelError>,
                        >,
                    >,
                >;
        }

        #[cfg(debug_assertions)]
        {
            tracing::trace!(
                "Running model with tokens: {:?}",
                tokenizer.decode(tokens, false)
            );
        }

        let logits = model.forward(tokens, images, device, cache);
        Box::pin(async move {
            let logits = logits?;
            let logits = logits.squeeze(0);
            // Cast logits back to f32 for sampling
            let logits: fusor::Tensor<1, f32> = logits.cast();
            let len = logits.shape()[0];
            let logits = logits.as_slice().await?;
            let mut logits_vec = Vec::with_capacity(len);
            for i in 0..len {
                let logit = logits[[i]];
                logits_vec.push(logit);
            }

            Ok(logits_vec)
        })
    }

    /// Create a new sync Llama model from a builder.
    pub(crate) async fn from_builder(
        builder: crate::LlamaBuilder<F>,
        mut handler: impl FnMut(ModelLoadingProgress) + WasmNotSend + WasmNotSync + 'static,
    ) -> Result<Self, LlamaSourceError> {
        let device = builder.get_device().await;

        // Download the model and tokenizer. These are relatively cheap operations that can be run in the async runtime
        let tokenizer_source = match &builder.source.tokenizer {
            Some(tokenizer) => {
                let tokenizer_source = format!("Tokenizer ({tokenizer})");
                let mut create_progress =
                    ModelLoadingProgress::downloading_progress(tokenizer_source);
                let tokenizer_source = builder
                    .source
                    .cache
                    .get_bytes(tokenizer, |progress| handler(create_progress(progress)))
                    .await?;
                Some(tokenizer_source)
            }
            None => None,
        };

        // Download the config file if it exists
        let config_bytes = match &builder.source.config {
            Some(config) => {
                let config_source = format!("Config ({config})");
                let mut create_progress = ModelLoadingProgress::downloading_progress(config_source);
                let config_bytes = builder
                    .source
                    .cache
                    .get_bytes(config, |progress| handler(create_progress(progress)))
                    .await?;
                Some(config_bytes)
            }
            None => None,
        };

        let vision_model_bytes = match &builder.source.vision_model {
            Some(vision_model) => {
                let vision_model_source = format!("Vision Model ({vision_model})");
                let mut create_progress =
                    ModelLoadingProgress::downloading_progress(vision_model_source);
                let vision_model_bytes = builder
                    .source
                    .cache
                    .get_bytes(vision_model, |progress| handler(create_progress(progress)))
                    .await?;
                Some(vision_model_bytes)
            }
            None => None,
        };

        let source = format!("Model ({})", builder.source.model[0]);
        let mut create_progress = ModelLoadingProgress::downloading_progress(source);
        let model_bytes = builder
            .source
            .model(|progress| handler(create_progress(progress)))
            .await?;

        // Then actually load the model and tokenizer. This is expensive, so we do it in a blocking task
        let load_model = {
            let device = device.clone();
            move || -> Result<(Model<F>, Tokenizer), LlamaSourceError> {
                let tokenizer = match tokenizer_source {
                    Some(tokenizer_source) => {
                        let tokenizer = Tokenizer::from_bytes(tokenizer_source)
                            .map_err(LlamaSourceError::Tokenizer)?;
                        Some(tokenizer)
                    }
                    None => None,
                };

                let config = match config_bytes {
                    Some(config_bytes) => {
                        let config: LlamaConfigJson = serde_json::from_slice(&config_bytes)
                            .map_err(LlamaSourceError::Config)?;
                        config.rope_scaling
                    }
                    None => None,
                };

                let override_stop_token_string = builder.source.override_stop_token_string;
                let override_chat_template = builder.source.override_chat_template;

                if model_bytes.is_empty() {
                    return Err(LlamaSourceError::InvalidGguf);
                }

                // Read metadata from all model files
                let mut files_with_metadata = Vec::new();
                for bytes in &model_bytes {
                    let mut cursor = std::io::Cursor::new(bytes);
                    let metadata = GgufMetadata::read(&mut cursor)?;
                    files_with_metadata.push((metadata, cursor));
                }

                let mut source = ShardedVarBuilder::new(files_with_metadata);

                let (vision_ct, vision_bytes) = match vision_model_bytes {
                    Some(bytes) => {
                        let mut cursor = std::io::Cursor::new(&bytes);
                        let metadata = GgufMetadata::read(&mut cursor)?;
                        (Some(metadata), Some(bytes))
                    }
                    None => (None, None),
                };

                let tokenizer = match tokenizer {
                    Some(tokenizer) => tokenizer,
                    None => {
                        let tokenizer_model: Box<str> = source
                            .get("tokenizer.ggml.model")
                            .map_err(|_| LlamaSourceError::NoTokenizer)?
                            .clone()
                            .try_into()
                            .map_err(|_| LlamaSourceError::NoTokenizer)?;
                        if &*tokenizer_model != "gpt2" {
                            return Err(LlamaSourceError::NoTokenizer);
                        }
                        let pre: Box<str> = source
                            .get("tokenizer.ggml.pre")
                            .map_err(|_| LlamaSourceError::NoTokenizer)?
                            .clone()
                            .try_into()
                            .map_err(|_| LlamaSourceError::NoTokenizer)?;
                        let add_bos_token = source
                            .get("tokenizer.ggml.add_bos_token")
                            .ok()
                            .cloned()
                            .and_then(|v| v.try_into().ok());
                        let config = get_pre_tokenizer(&pre, add_bos_token);

                        let token_values: Box<[GgufValue]> = source
                            .get("tokenizer.ggml.tokens")
                            .map_err(|_| LlamaSourceError::NoTokenizer)?
                            .clone()
                            .try_into()
                            .map_err(|_| LlamaSourceError::NoTokenizer)?;
                        let tokens: Result<Vec<_>, _> =
                            token_values.iter().map(|v| v.clone().try_into()).collect();
                        let tokens: Vec<Box<str>> =
                            tokens.map_err(|_| LlamaSourceError::NoTokenizer)?;
                        let token_type_values: Box<[GgufValue]> = source
                            .get("tokenizer.ggml.token_type")
                            .map_err(|_| LlamaSourceError::NoTokenizer)?
                            .clone()
                            .try_into()
                            .map_err(|_| LlamaSourceError::NoTokenizer)?;
                        let types: Result<Vec<_>, _> = token_type_values
                            .iter()
                            .map(|v| v.to_u8().map_err(|_| LlamaSourceError::NoTokenizer))
                            .collect();
                        let types = types.map_err(|_| LlamaSourceError::NoTokenizer)?;
                        let vocab: HashMap<_, _> = tokens
                            .iter()
                            .enumerate()
                            .map(|(id, v)| (v.to_string(), id as u32))
                            .collect();
                        let merges: Box<[GgufValue]> = source
                            .get("tokenizer.ggml.merges")
                            .map_err(|_| LlamaSourceError::NoTokenizer)?
                            .clone()
                            .try_into()
                            .map_err(|_| LlamaSourceError::NoTokenizer)?;
                        let merges: Result<Vec<_>, _> = merges
                            .iter()
                            .map(|v| {
                                let as_str: Box<str> = v
                                    .clone()
                                    .try_into()
                                    .map_err(|_| LlamaSourceError::NoTokenizer)?;
                                as_str
                                    .split_once(' ')
                                    .ok_or(LlamaSourceError::NoTokenizer)
                                    .map(|(a, b)| (a.to_string(), b.to_string()))
                            })
                            .collect();
                        let merges = merges.map_err(|_| LlamaSourceError::NoTokenizer)?;

                        let eos = source
                            .get("tokenizer.ggml.eos_token_id")
                            .map_err(|_| LlamaSourceError::NoTokenizer)?;
                        let eos: u32 = eos.try_into().map_err(|_| LlamaSourceError::NoTokenizer)?;
                        let eos = &tokens[eos as usize];

                        let bos = source
                            .get("tokenizer.ggml.bos_token_id")
                            .map_err(|_| LlamaSourceError::NoTokenizer)?;
                        let bos: u32 = bos.try_into().map_err(|_| LlamaSourceError::NoTokenizer)?;
                        let bos = &tokens[bos as usize];

                        config
                            .build(vocab, types, merges, bos, eos)
                            .map_err(LlamaSourceError::Tokenizer)?
                    }
                };
                let model = Model::from_gguf(
                    &mut source,
                    vision_ct,
                    vision_bytes,
                    &device,
                    override_stop_token_string,
                    override_chat_template,
                    config,
                )?;
                Ok((model, tokenizer))
            }
        };

        let (model, tokenizer) = load_model()?;

        Ok(Self {
            model,
            tokenizer: Arc::new(tokenizer),
            device,
        })
    }

    pub(crate) async fn _infer(
        &mut self,
        settings: InferenceSettings<F>,
        mut on_token: crate::BoxedTokenCallback,
        finished: &futures::channel::oneshot::Sender<Result<(), LlamaModelError>>,
    ) -> Result<(), LlamaModelError> {
        let InferenceSettings {
            prompt,
            images,
            stop_on,
            mut sampler,
            session,
            max_tokens,
            seed,
        } = settings;

        let tokens = self
            .tokenizer
            .encode_fast(prompt, false)
            .map_err(LlamaModelError::Tokenizer)?;
        let tokens = tokens.get_ids();
        let mut text_stream = TokenOutputStream::new(self.tokenizer.clone());
        for &token in tokens {
            text_stream
                .next_token(token)
                .map_err(LlamaModelError::TokenOutputStreamError)?;
        }

        let logit_probs = {
            let mut session_lock = session
                .cache
                .write()
                .map_err(|err| LlamaModelError::Session(err.to_string()))?;
            Self::forward(
                &self.model,
                &self.device,
                tokens,
                &images,
                Some(&mut session_lock),
                &self.tokenizer,
            )
        }
        .await?;
        let mut logits = Logits::try_from_iter_top_k(logit_probs, 512)
            .expect("model output should be valid logits");
        // This stores a buffer of text that has been generated to check against the stop_on string. It should never be longer than the stop_on string.
        let mut queued_text_matching_stop_on = String::new();
        let stop_on_lowercase = stop_on.as_ref().map(|s| s.to_lowercase());
        let stop_on_lowercase = stop_on_lowercase.as_deref();
        let stop_token = self.model.config.stop_token;
        let mut tokens_generated = 0;

        'generate: while !finished.is_canceled() && tokens_generated < max_tokens {
            let new_token = text_stream
                .sample_token(&mut sampler, logits, stop_on.as_deref(), seed)
                .map_err(LlamaModelError::TokenOutputStreamError)?;
            let logit_probs = {
                let mut session_lock = session
                    .cache
                    .write()
                    .map_err(|err| LlamaModelError::Session(err.to_string()))?;
                Self::forward(
                    &self.model,
                    &self.device,
                    &[new_token],
                    &[],
                    Some(&mut session_lock),
                    &self.tokenizer,
                )
            }
            .await?;
            if new_token == stop_token {
                tracing::trace!("Stopping on stop token");
                break;
            }
            if let Some(mut new_text) = text_stream
                .next_token(new_token)
                .map_err(LlamaModelError::TokenOutputStreamError)?
            {
                tokens_generated += 1;
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
            logits = Logits::try_from_iter_top_k(logit_probs, 512)
                .expect("model output should be valid logits");
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
