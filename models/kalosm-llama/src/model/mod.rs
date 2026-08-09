use crate::gguf_tokenizer::get_pre_tokenizer;
use crate::raw::cache::LlamaCache;
use crate::raw::{LlamaVarSource, Model, RopeScalingConfig};
use crate::token_stream::TokenOutputStream;
use crate::token_stream::TokenOutputStreamError;
use crate::tokenizer::{LlamaTokenizer, LlamaTokenizerError};
#[cfg(feature = "hf-config-json")]
use crate::LlamaConfigJson;
use fusor2::device::Device;
use fusor2_gguf::{ShardedVarBuilder, VarBuilder};
#[cfg(feature = "vision")]
use kalosm_language_model::ImageFetchError;
use crate::WasmNotSync;
use kalosm_model_types::{ModelLoadingProgress, WasmNotSend};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::{Arc, Mutex, OnceLock};
use web_time::{Duration, Instant};

use crate::sampler::{CpuSampler, Logit, Logits};
use crate::{GpuSamplerConfig, InferenceSettings, LlamaImage, LlamaSourceError};

mod forward;
mod inference;

#[derive(Default)]
struct DecodeTraceStats {
    fast: Vec<Duration>,
    fallback: Vec<Duration>,
}

static DECODE_TRACE_STATS: OnceLock<Mutex<DecodeTraceStats>> = OnceLock::new();

fn record_decode_trace(path: &'static str, decode_eligible: bool, kernels: usize, total: Duration) {
    let stats = DECODE_TRACE_STATS.get_or_init(|| Mutex::new(DecodeTraceStats::default()));
    let mut stats = stats.lock().expect("decode trace mutex poisoned");
    let samples = if decode_eligible {
        &mut stats.fast
    } else {
        &mut stats.fallback
    };
    samples.push(total);
    let mut total_samples = samples.clone();
    total_samples.sort_unstable();
    let p50 = percentile_duration(&total_samples, 50);
    let p95 = percentile_duration(&total_samples, 95);
    tracing::info!(
        "decode_trace_summary samples={} path={path} decode_eligible={decode_eligible} kernels={kernels} total={total:?} p50={p50:?} p95={p95:?}",
        total_samples.len()
    );
}

fn percentile_duration(samples: &[Duration], percentile: usize) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }
    let index = ((samples.len() - 1) * percentile).div_ceil(100);
    samples[index]
}

fn logits_from_sorted_top_k(logits: Vec<Logit>) -> Logits {
    let mut result = Logits::default();
    result.extend(logits);
    result
}

fn use_full_logits_for_sampling(_vocab_len: usize) -> bool {
    // fusor2's device-resident sort is O(n^2) memory, which does not scale to
    // a real vocabulary; the host heap top-k over the full logits row is the
    // default. Set KALOSM_LLAMA_GPU_TOP_K=1 to opt in to the device path.
    let gpu_top_k_enabled = std::env::var_os("KALOSM_LLAMA_GPU_TOP_K")
        .map(|value| value == "1")
        .unwrap_or(false);
    !gpu_top_k_enabled
}

fn decode_trace_enabled() -> bool {
    std::env::var_os("KALOSM_TRACE_DECODE_TIMING").is_some()
        || std::env::var_os("FUSOR_TRACE_DECODE").is_some()
        || std::env::var_os("FUSOR_TRACE_RESOLVE").is_some()
}

fn gpu_sample_top_k(config: &GpuSamplerConfig) -> usize {
    let default_top_k = match config.sampling_strategy {
        kalosm_language_model::SamplingStrategy::Mirostat2 => 16,
        kalosm_language_model::SamplingStrategy::Standard => 64,
    };
    std::env::var("KALOSM_LLAMA_GPU_SAMPLE_TOP_K")
        .ok()
        .and_then(|value| value.parse().ok())
        .or(config.top_k)
        .unwrap_or(default_top_k)
        .max(1)
}

fn parse_external_tokenizer(
    tokenizer_source: Option<Vec<u8>>,
) -> Result<Option<LlamaTokenizer>, LlamaSourceError> {
    #[cfg(feature = "hf-tokenizer-json")]
    {
        match tokenizer_source {
            Some(tokenizer_source) => {
                let tokenizer = LlamaTokenizer::from_hf_bytes(tokenizer_source)
                    .map_err(|err| LlamaSourceError::Tokenizer(Box::new(err)))?;
                Ok(Some(tokenizer))
            }
            None => Ok(None),
        }
    }
    #[cfg(not(feature = "hf-tokenizer-json"))]
    {
        let _ = tokenizer_source;
        Ok(None)
    }
}

fn parse_external_config(
    config_bytes: Option<Vec<u8>>,
) -> Result<Option<RopeScalingConfig>, LlamaSourceError> {
    #[cfg(feature = "hf-config-json")]
    {
        match config_bytes {
            Some(config_bytes) => {
                let config: LlamaConfigJson =
                    serde_json::from_slice(&config_bytes).map_err(LlamaSourceError::Config)?;
                Ok(config.rope_scaling)
            }
            None => Ok(None),
        }
    }
    #[cfg(not(feature = "hf-config-json"))]
    {
        let _ = config_bytes;
        Ok(None)
    }
}

fn tokenizer_from_gguf_source<S: LlamaVarSource>(
    source: &S,
) -> Result<LlamaTokenizer, LlamaSourceError> {
    let tokenizer_model = source
        .get("tokenizer.ggml.model")
        .map_err(|_| LlamaSourceError::NoTokenizer)?
        .to_string_value()
        .map_err(|_| LlamaSourceError::NoTokenizer)?;
    if tokenizer_model != "gpt2" {
        return Err(LlamaSourceError::NoTokenizer);
    }
    let pre = source
        .get("tokenizer.ggml.pre")
        .map_err(|_| LlamaSourceError::NoTokenizer)?
        .to_string_value()
        .map_err(|_| LlamaSourceError::NoTokenizer)?;
    let add_bos_token = source
        .get("tokenizer.ggml.add_bos_token")
        .ok()
        .and_then(|v| v.to_bool().ok());
    let config = get_pre_tokenizer(pre, add_bos_token);

    let token_values = source
        .get("tokenizer.ggml.tokens")
        .map_err(|_| LlamaSourceError::NoTokenizer)?
        .to_array()
        .map_err(|_| LlamaSourceError::NoTokenizer)?;
    let tokens: Result<Vec<_>, _> = token_values
        .iter()
        .map(|v| {
            v.to_string_value()
                .map(|s| s.to_string().into_boxed_str())
                .map_err(|_| LlamaSourceError::NoTokenizer)
        })
        .collect();
    let tokens: Vec<Box<str>> = tokens?;
    let token_type_values = source
        .get("tokenizer.ggml.token_type")
        .map_err(|_| LlamaSourceError::NoTokenizer)?
        .to_array()
        .map_err(|_| LlamaSourceError::NoTokenizer)?;
    let types: Result<Vec<_>, _> = token_type_values
        .iter()
        .map(|v| v.to_u8().map_err(|_| LlamaSourceError::NoTokenizer))
        .collect();
    let types = types?;
    let vocab: HashMap<_, _> = tokens
        .iter()
        .enumerate()
        .map(|(id, v)| (v.to_string(), id as u32))
        .collect();
    let merges = source
        .get("tokenizer.ggml.merges")
        .map_err(|_| LlamaSourceError::NoTokenizer)?
        .to_array()
        .map_err(|_| LlamaSourceError::NoTokenizer)?;
    let merges: Result<Vec<_>, _> = merges
        .iter()
        .map(|v| {
            let as_str = v
                .to_string_value()
                .map_err(|_| LlamaSourceError::NoTokenizer)?;
            as_str
                .split_once(' ')
                .ok_or(LlamaSourceError::NoTokenizer)
                .map(|(a, b)| (a.to_string(), b.to_string()))
        })
        .collect();
    let merges = merges?;

    let eos = source
        .get("tokenizer.ggml.eos_token_id")
        .map_err(|_| LlamaSourceError::NoTokenizer)?;
    let eos: u32 = eos.to_u32().map_err(|_| LlamaSourceError::NoTokenizer)?;
    let eos = &tokens[eos as usize];

    // Some models (e.g. Qwen) don't use a BOS token and ship the GGUF
    // file without `tokenizer.ggml.bos_token_id`. Treat it as optional
    // rather than failing to load the embedded tokenizer entirely.
    let bos: Option<&str> = source
        .get("tokenizer.ggml.bos_token_id")
        .ok()
        .and_then(|v| {
            let id: u32 = v.to_u32().ok()?;
            Some(&*tokens[id as usize])
        });

    config
        .build(vocab, types, merges, bos, eos)
        .map(LlamaTokenizer::from_gguf)
        .map_err(|err| LlamaSourceError::Tokenizer(Box::new(err)))
}

struct ForwardTrace {
    enabled: bool,
    decode_eligible: bool,
    path: &'static str,
    token_start: Option<Instant>,
    kernels: usize,
}

impl ForwardTrace {
    fn step_start(&self) -> Option<Instant> {
        self.enabled.then(Instant::now)
    }

    fn record(&self) {
        if let Some(start) = self.token_start {
            record_decode_trace(
                self.path,
                self.decode_eligible,
                self.kernels,
                start.elapsed(),
            );
        }
    }
}

struct PreparedForwardLogits {
    logits: fusor2::Tensor,
    len: usize,
    trace: ForwardTrace,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct WorstFirstLogit {
    token_id: u32,
    logit: f32,
}

impl Eq for WorstFirstLogit {}

impl Ord for WorstFirstLogit {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.logit.total_cmp(&other.logit) {
            Ordering::Less => Ordering::Greater,
            Ordering::Greater => Ordering::Less,
            Ordering::Equal => other.token_id.cmp(&self.token_id),
        }
    }
}

impl PartialOrd for WorstFirstLogit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn top_k_logits_from_full(logits: &[f32], k: usize) -> Vec<Logit> {
    if k == 0 {
        return Vec::new();
    }

    let mut heap = BinaryHeap::with_capacity(k);
    for (token_id, logit) in logits.iter().copied().enumerate() {
        if !logit.is_finite() {
            continue;
        }
        let candidate = WorstFirstLogit {
            token_id: token_id as u32,
            logit,
        };
        if heap.len() < k {
            heap.push(candidate);
            continue;
        }
        let Some(worst) = heap.peek().copied() else {
            continue;
        };
        if logit > worst.logit || (logit == worst.logit && candidate.token_id > worst.token_id) {
            heap.pop();
            heap.push(candidate);
        }
    }

    let mut logits = heap
        .into_iter()
        .map(|candidate| Logit {
            token_id: candidate.token_id,
            logit: candidate.logit,
            prob: 0.0,
        })
        .collect::<Vec<_>>();
    logits.sort_unstable_by(|left, right| {
        right
            .logit
            .total_cmp(&left.logit)
            .then_with(|| right.token_id.cmp(&left.token_id))
    });
    logits
}

/// An error that can occur when running a [`LlamaModel`].
#[derive(Debug, thiserror::Error)]
pub enum LlamaModelError {
    /// An error from Fusor while running the model.
    #[error("Fusor error: {0}")]
    Fusor(#[from] fusor2::Error),

    /// An error from the tokenizer while running the model.
    #[error("Tokenizer error: {0}")]
    Tokenizer(#[from] LlamaTokenizerError),

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
    #[cfg(feature = "vision")]
    #[error("Failed to load images: {0}")]
    ImageLoadingError(#[from] ImageFetchError),

    /// The model was built without local vision support.
    #[error("Media inputs require the `vision` feature")]
    MediaUnsupported,
}

#[cfg(feature = "vision")]
impl From<image::ImageError> for LlamaModelError {
    fn from(err: image::ImageError) -> Self {
        LlamaModelError::ImageLoadingError(err.into())
    }
}

/// The inner, synchronous Llama model.
pub(crate) struct LlamaModel {
    pub(crate) model: Model,
    pub(crate) device: Device,
    pub(crate) tokenizer: Arc<LlamaTokenizer>,
}

pub(crate) struct ForwardInputs<'a> {
    pub(crate) model: &'a Model,
    pub(crate) device: &'a Device,
    pub(crate) tokens: &'a [u32],
    pub(crate) images: &'a [LlamaImage],
    pub(crate) cache: Option<&'a mut LlamaCache>,
    pub(crate) tokenizer: &'a LlamaTokenizer,
}

impl LlamaModel {
    pub(crate) async fn from_builder(
        builder: crate::LlamaBuilder,
        mut handler: impl FnMut(ModelLoadingProgress) + WasmNotSend + WasmNotSync + 'static,
    ) -> Result<Self, LlamaSourceError> {
        let device = builder.get_device().await;
        if decode_trace_enabled() {
            tracing::info!("llama_device={}", device.name());
        }

        // Download the model and tokenizer. These are relatively cheap operations that can be run in the async runtime
        #[cfg(feature = "hf-tokenizer-json")]
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
        #[cfg(not(feature = "hf-tokenizer-json"))]
        let tokenizer_source: Option<Vec<u8>> = {
            if builder.source.tokenizer.is_some() {
                return Err(LlamaSourceError::TokenizerJsonFeatureDisabled);
            }
            None
        };

        // Download the config file if it exists
        #[cfg(feature = "hf-config-json")]
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
        #[cfg(not(feature = "hf-config-json"))]
        let config_bytes: Option<Vec<u8>> = {
            if builder.source.config.is_some() {
                return Err(LlamaSourceError::ConfigJsonFeatureDisabled);
            }
            None
        };

        if builder.source.vision_model.is_some() {
            return Err(LlamaSourceError::VisionFeatureDisabled);
        }

        let source = format!("Model ({})", builder.source.model[0]);
        let mut create_progress = ModelLoadingProgress::downloading_progress(source);
        let model_bytes = builder
            .source
            .model(|progress| handler(create_progress(progress)))
            .await?;

        let override_stop_token_string = builder.source.override_stop_token_string.clone();
        let override_chat_template = builder.source.override_chat_template.clone();

        let (model, tokenizer) = {
            let device = device.clone();
            let load_model = move || -> Result<(Model, LlamaTokenizer), LlamaSourceError> {
                let tokenizer = parse_external_tokenizer(tokenizer_source)?;
                let config = parse_external_config(config_bytes)?;
                if model_bytes.is_empty() {
                    return Err(LlamaSourceError::InvalidGguf);
                }

                // Parse every shard from its in-memory bytes.
                let mut shards = Vec::new();
                for bytes in model_bytes {
                    let gguf = fusor2_gguf::Gguf::from_bytes(bytes)
                        .map_err(LlamaSourceError::Device)?;
                    shards.push(VarBuilder::new(std::sync::Arc::new(gguf)));
                }

                let mut source = ShardedVarBuilder::new(shards);

                let tokenizer = match tokenizer {
                    Some(tokenizer) => tokenizer,
                    None => tokenizer_from_gguf_source(&source)?,
                };
                let model = Model::from_gguf(
                    &mut source,
                    &device,
                    override_stop_token_string,
                    override_chat_template,
                    config,
                )?;
                Ok((model, tokenizer))
            };

            load_model()?
        };

        Ok(Self {
            model,
            tokenizer: Arc::new(tokenizer),
            device,
        })
    }
}
