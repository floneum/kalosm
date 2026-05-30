use crate::gguf_tokenizer::get_pre_tokenizer;
use crate::raw::cache::LlamaCache;
use crate::raw::Model;
use crate::token_stream::TokenOutputStream;
use crate::token_stream::TokenOutputStreamError;
use crate::tokenizer::{LlamaTokenizer, LlamaTokenizerError};
#[cfg(feature = "hf-config-json")]
use crate::LlamaConfigJson;
use fusor::{
    AddOp, CastTensor, CastTo, Device, FloatDataType, FloatOps, MatmulImpl, Mirostat2Sampler,
    Mirostat2SamplerParams, MulOp, ShardedVarBuilder, SimdBinaryOp, SimdElement, SimdReduceOp,
    SumOp, WasmNotSend, WasmNotSync,
};
use fusor_gguf::GgufMetadata;
use fusor_gguf::GgufValue;
#[cfg(feature = "vision")]
use kalosm_language_model::ImageFetchError;
use kalosm_model_types::ModelLoadingProgress;
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use crate::sampler::{CpuMirostat2Sampler, Logit, Logits};
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
    eprintln!(
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
    let gpu_top_k_enabled = std::env::var_os("KALOSM_LLAMA_GPU_TOP_K")
        .map(|value| value != "0")
        .unwrap_or(true);
    !gpu_top_k_enabled
}

fn gpu_token_sampling_enabled() -> bool {
    std::env::var_os("KALOSM_LLAMA_GPU_SAMPLE_TOKEN")
        .map(|value| value != "0")
        .unwrap_or(true)
}

fn gpu_fused_logits_sampling_enabled() -> bool {
    std::env::var_os("KALOSM_LLAMA_GPU_FUSED_LOGITS")
        .map(|value| value != "0")
        .unwrap_or(true)
}

fn gpu_fused_logits_prewarm_enabled() -> bool {
    std::env::var_os("KALOSM_LLAMA_GPU_FUSED_LOGITS_PREWARM")
        .map(|value| value != "0")
        .unwrap_or(true)
}

fn decode_trace_enabled() -> bool {
    std::env::var_os("KALOSM_TRACE_DECODE_TIMING").is_some()
        || std::env::var_os("FUSOR_TRACE_DECODE").is_some()
        || std::env::var_os("FUSOR_TRACE_RESOLVE").is_some()
}

fn gpu_sample_top_k(config: &GpuSamplerConfig) -> usize {
    std::env::var("KALOSM_LLAMA_GPU_SAMPLE_TOP_K")
        .ok()
        .and_then(|value| value.parse().ok())
        .or(config.top_k)
        .unwrap_or(16)
        .max(1)
}

fn unbounded_decode_reserve_tokens() -> usize {
    std::env::var("KALOSM_LLAMA_UNBOUNDED_DECODE_RESERVE")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(256)
}

struct LlamaGpuSamplerState {
    sampler: Mirostat2Sampler,
    config: GpuSamplerConfig,
    rng: rand::rngs::StdRng,
}

impl LlamaGpuSamplerState {
    fn new(device: &Device, config: GpuSamplerConfig, seed: Option<u64>) -> Option<Self> {
        let gpu_device = device.as_gpu()?;
        let rng = seed
            .map(rand::rngs::StdRng::seed_from_u64)
            .unwrap_or_else(rand::rngs::StdRng::from_os_rng);
        Some(Self {
            sampler: Mirostat2Sampler::new(gpu_device, config.mu),
            config,
            rng,
        })
    }

    fn params(&mut self, top_k: usize) -> Mirostat2SamplerParams {
        Mirostat2SamplerParams {
            top_k,
            temperature: self.config.temperature,
            repetition_penalty: self.config.repetition_penalty,
            tau: self.config.tau,
            eta: self.config.eta,
            random: self.rng.random::<f32>(),
        }
    }

    fn previous_tokens(&self, text_stream: &TokenOutputStream) -> Vec<u32> {
        let tokens = text_stream.tokens();
        let len = tokens.len().min(self.config.repetition_penalty_range);
        tokens[tokens.len().saturating_sub(len)..].to_vec()
    }
}

struct ForwardTrace {
    enabled: bool,
    decode_eligible: bool,
    path: &'static str,
    token_start: Option<std::time::Instant>,
    kernels: usize,
}

impl ForwardTrace {
    fn step_start(&self) -> Option<std::time::Instant> {
        self.enabled.then(std::time::Instant::now)
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
    logits: fusor::Tensor<1, f32>,
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
    /// An error from candle while running the model.
    #[error("Candle error: {0}")]
    Candle(#[from] fusor::Error),

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
pub(crate) struct LlamaModel<F: FloatDataType + SimdElement = f32> {
    pub(crate) model: Model<F>,
    pub(crate) device: Device,
    pub(crate) tokenizer: Arc<LlamaTokenizer>,
}

pub(crate) struct ForwardInputs<'a, F: FloatDataType + SimdElement> {
    pub(crate) model: &'a Model<F>,
    pub(crate) device: &'a Device,
    pub(crate) tokens: &'a [u32],
    pub(crate) images: &'a [LlamaImage],
    pub(crate) cache: Option<&'a mut LlamaCache>,
    pub(crate) tokenizer: &'a LlamaTokenizer,
}

impl<F: FloatDataType + SimdElement + Default + FloatOps + MatmulImpl> LlamaModel<F>
where
    F: CastTo<f32> + CastTensor<f32> + WasmNotSend + WasmNotSync + 'static,
    f32: CastTo<F> + CastTensor<F>,
    MulOp: SimdBinaryOp<F>,
    AddOp: SimdBinaryOp<F>,
    SumOp: SimdReduceOp<F>,
{
    async fn prewarm_fused_logits_sampling(model: &Model<F>, device: &Device) {
        if !gpu_fused_logits_sampling_enabled() || !gpu_fused_logits_prewarm_enabled() {
            return;
        }
        let Some(gpu_device) = device.as_gpu() else {
            return;
        };
        let shape = model.output_matrix().shape();
        if shape.len() != 2 || shape[1] == 0 {
            return;
        }

        let trace = decode_trace_enabled();
        let start = trace.then(std::time::Instant::now);
        let hidden_values = vec![0.0f32; shape[1]];
        let hidden: fusor::Tensor<1, f32> =
            fusor::Tensor::from_slice(device, [shape[1]], &hidden_values);
        let mut sampler = Mirostat2Sampler::new(gpu_device, 10.0);
        let params = Mirostat2SamplerParams {
            top_k: 16,
            temperature: 0.8,
            repetition_penalty: 1.3,
            tau: 5.0,
            eta: 0.1,
            random: 0.5,
        };

        let _ = hidden
            .try_sample_mirostat2_token_q_mat(model.output_matrix(), &mut sampler, &[], params)
            .await;
        if let Some(start) = start {
            eprintln!(
                "prewarm_fused_logits_sampling elapsed={:?}",
                start.elapsed()
            );
        }
    }

    pub(crate) async fn from_builder(
        builder: crate::LlamaBuilder<F>,
        mut handler: impl FnMut(ModelLoadingProgress) + WasmNotSend + WasmNotSync + 'static,
    ) -> Result<Self, LlamaSourceError> {
        let device = builder.get_device().await;
        if decode_trace_enabled() {
            match &device {
                Device::Cpu => eprintln!("llama_device=cpu"),
                Device::Gpu(gpu) => eprintln!(
                    "llama_device=gpu adapter={:?}",
                    gpu.wgpu_adapter().get_info(),
                ),
            }
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

        #[cfg(feature = "vision")]
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
        #[cfg(not(feature = "vision"))]
        let vision_model_bytes: Option<Vec<u8>> = {
            if builder.source.vision_model.is_some() {
                return Err(LlamaSourceError::VisionFeatureDisabled);
            }
            None
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
            move || -> Result<(Model<F>, LlamaTokenizer), LlamaSourceError> {
                #[cfg(feature = "hf-tokenizer-json")]
                let tokenizer = match tokenizer_source {
                    Some(tokenizer_source) => {
                        let tokenizer = LlamaTokenizer::from_hf_bytes(tokenizer_source)
                            .map_err(|err| LlamaSourceError::Tokenizer(Box::new(err)))?;
                        Some(tokenizer)
                    }
                    None => None,
                };
                #[cfg(not(feature = "hf-tokenizer-json"))]
                let tokenizer: Option<LlamaTokenizer> = {
                    let _ = tokenizer_source;
                    None
                };

                #[cfg(feature = "hf-config-json")]
                let config = match config_bytes {
                    Some(config_bytes) => {
                        let config: LlamaConfigJson = serde_json::from_slice(&config_bytes)
                            .map_err(LlamaSourceError::Config)?;
                        config.rope_scaling
                    }
                    None => None,
                };
                #[cfg(not(feature = "hf-config-json"))]
                let config = {
                    let _ = config_bytes;
                    None
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
                            .map(LlamaTokenizer::from_gguf)
                            .map_err(|err| LlamaSourceError::Tokenizer(Box::new(err)))?
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
        Self::prewarm_fused_logits_sampling(&model, &device).await;

        Ok(Self {
            model,
            tokenizer: Arc::new(tokenizer),
            device,
        })
    }
}
