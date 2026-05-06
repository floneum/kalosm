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
use llm_samplers::types::{Logit, Logits};
use rand::{Rng, SeedableRng};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use tokenizers::Tokenizer;

use crate::{GpuSamplerConfig, InferenceSettings, LlamaSourceError};

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
    result.set_sorted(true);
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

fn gpu_sample_top_k() -> usize {
    std::env::var("KALOSM_LLAMA_GPU_SAMPLE_TOP_K")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(512)
        .max(1)
}

fn unbounded_decode_reserve_tokens() -> usize {
    std::env::var("KALOSM_LLAMA_UNBOUNDED_DECODE_RESERVE")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(256)
}

struct LlamaGpuSamplerState {
    sampler: fusor::GpuMirostat2Sampler,
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
            sampler: fusor::GpuMirostat2Sampler::new(gpu_device, config.mu),
            config,
            rng,
        })
    }

    fn params(&mut self, top_k: usize) -> fusor::GpuMirostat2SamplerParams {
        fusor::GpuMirostat2SamplerParams {
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
        mut cache: Option<&mut LlamaCache>,
        #[allow(unused)] tokenizer: &Tokenizer,
    ) -> Pin<
        Box<dyn kalosm_model_types::FutureWasmNotSend<Output = Result<Vec<f32>, LlamaModelError>>>,
    > {
        if tokens.is_empty() {
            return Box::pin(async { Err(LlamaModelError::EmptyInput) });
        }

        #[cfg(debug_assertions)]
        {
            tracing::trace!(
                "Running model with tokens: {:?}",
                tokenizer.decode(tokens, false)
            );
        }

        let trace = std::env::var_os("KALOSM_TRACE_DECODE_TIMING").is_some()
            || std::env::var_os("FUSOR_TRACE_DECODE").is_some()
            || std::env::var_os("FUSOR_TRACE_RESOLVE").is_some();
        let fast_decode_enabled = std::env::var_os("KALOSM_LLAMA_FAST_DECODE")
            .map(|value| value != "0")
            .unwrap_or(true);
        let decode_eligible = fast_decode_enabled
            && tokens.len() == 1
            && images.is_empty()
            && cache.as_ref().is_some_and(|cache| !cache.tokens.is_empty());
        let path = if decode_eligible {
            "fast_decode_graph"
        } else {
            "graph_fallback"
        };
        let token_start = trace.then(std::time::Instant::now);
        let build_start = trace.then(std::time::Instant::now);
        let logits = model.forward(tokens, images, device, cache.as_deref_mut());
        if let Some(start) = build_start {
            eprintln!(
                "forward_graph_build path={path} decode_eligible={decode_eligible} elapsed={:?}",
                start.elapsed()
            );
        }
        let logits = match logits {
            Ok(logits) => logits,
            Err(err) => return Box::pin(async move { Err(err.into()) }),
        };
        let logits = logits.squeeze(0);
        // Cast logits back to f32 for sampling
        let logits: fusor::Tensor<1, f32> = logits.cast();
        let len = logits.shape()[0];
        let mut kernels = 0;
        if let Some(logits_key) = logits.gpu_key() {
            let resolve_start = trace.then(std::time::Instant::now);
            kernels = device.resolve_batch(&[logits_key]);
            if let Some(start) = resolve_start {
                eprintln!(
                    "forward_resolve path={path} decode_eligible={decode_eligible} kernels={kernels} elapsed={:?}",
                    start.elapsed()
                );
            }
            if let Some(cache) = cache.as_deref_mut() {
                cache.detach(device);
            }
        }
        Box::pin(async move {
            let download_start = trace.then(std::time::Instant::now);
            let logits = logits.as_slice().await?;
            if let Some(start) = download_start {
                eprintln!(
                    "forward_download path={path} decode_eligible={decode_eligible} elapsed={:?}",
                    start.elapsed()
                );
            }
            if let Some(start) = token_start {
                record_decode_trace(path, decode_eligible, kernels, start.elapsed());
            }
            let mut logits_vec = Vec::with_capacity(len);
            for i in 0..len {
                let logit = logits[[i]];
                logits_vec.push(logit);
            }

            Ok(logits_vec)
        })
    }

    pub(crate) fn forward_top_k(
        model: &Model<F>,
        device: &Device,
        tokens: &[u32],
        images: &[(image::DynamicImage, MediaHints)],
        mut cache: Option<&mut LlamaCache>,
        #[allow(unused)] tokenizer: &Tokenizer,
        top_k: usize,
    ) -> Pin<
        Box<
            dyn kalosm_model_types::FutureWasmNotSend<Output = Result<Vec<Logit>, LlamaModelError>>,
        >,
    > {
        if tokens.is_empty() {
            return Box::pin(async { Err(LlamaModelError::EmptyInput) });
        }

        #[cfg(debug_assertions)]
        {
            tracing::trace!(
                "Running model with tokens: {:?}",
                tokenizer.decode(tokens, false)
            );
        }

        let trace = std::env::var_os("KALOSM_TRACE_DECODE_TIMING").is_some()
            || std::env::var_os("FUSOR_TRACE_DECODE").is_some()
            || std::env::var_os("FUSOR_TRACE_RESOLVE").is_some();
        let fast_decode_enabled = std::env::var_os("KALOSM_LLAMA_FAST_DECODE")
            .map(|value| value != "0")
            .unwrap_or(true);
        let decode_eligible = fast_decode_enabled
            && tokens.len() == 1
            && images.is_empty()
            && cache.as_ref().is_some_and(|cache| !cache.tokens.is_empty());
        let path = if decode_eligible {
            "fast_decode_graph_top_k"
        } else {
            "graph_fallback_top_k"
        };
        let token_start = trace.then(std::time::Instant::now);
        let build_start = trace.then(std::time::Instant::now);
        let logits = model.forward(tokens, images, device, cache.as_deref_mut());
        if let Some(start) = build_start {
            eprintln!(
                "forward_graph_build path={path} decode_eligible={decode_eligible} elapsed={:?}",
                start.elapsed()
            );
        }
        let logits = match logits {
            Ok(logits) => logits,
            Err(err) => return Box::pin(async move { Err(err.into()) }),
        };
        let logits = logits.squeeze(0);
        let logits: fusor::Tensor<1, f32> = logits.cast();
        let len = logits.shape()[0];
        let mut kernels = 0;
        if let Some(logits_key) = logits.gpu_key() {
            let resolve_start = trace.then(std::time::Instant::now);
            kernels = device.resolve_batch(&[logits_key]);
            if let Some(start) = resolve_start {
                eprintln!(
                    "forward_resolve path={path} decode_eligible={decode_eligible} kernels={kernels} elapsed={:?}",
                    start.elapsed()
                );
            }
            if let Some(cache) = cache.as_deref_mut() {
                cache.detach(device);
            }
        }
        Box::pin(async move {
            let download_start = trace.then(std::time::Instant::now);
            let top_logits = if use_full_logits_for_sampling(len) {
                let logits = logits.as_slice().await?;
                let mut logits_vec = Vec::with_capacity(len);
                for i in 0..len {
                    logits_vec.push(logits[[i]]);
                }
                top_k_logits_from_full(&logits_vec, top_k)
            } else {
                logits
                    .top_k_pairs(top_k)
                    .await?
                    .into_iter()
                    .map(|(token_id, logit)| Logit {
                        token_id,
                        logit,
                        prob: 0.0,
                    })
                    .collect()
            };
            if let Some(start) = download_start {
                eprintln!(
                    "forward_top_k_download path={path} decode_eligible={decode_eligible} k={top_k} elapsed={:?}",
                    start.elapsed()
                );
            }
            if let Some(start) = token_start {
                record_decode_trace(path, decode_eligible, kernels, start.elapsed());
            }

            Ok(top_logits)
        })
    }

    pub(crate) fn forward_sample_token<'a>(
        model: &Model<F>,
        device: &Device,
        tokens: &[u32],
        images: &[(image::DynamicImage, MediaHints)],
        mut cache: Option<&mut LlamaCache>,
        #[allow(unused)] tokenizer: &Tokenizer,
        sampler: &'a mut fusor::GpuMirostat2Sampler,
        previous_tokens: Vec<u32>,
        params: fusor::GpuMirostat2SamplerParams,
    ) -> Pin<
        Box<dyn kalosm_model_types::FutureWasmNotSend<Output = Result<u32, LlamaModelError>> + 'a>,
    > {
        if tokens.is_empty() {
            return Box::pin(async { Err(LlamaModelError::EmptyInput) });
        }

        #[cfg(debug_assertions)]
        {
            tracing::trace!(
                "Running model with tokens: {:?}",
                tokenizer.decode(tokens, false)
            );
        }

        if gpu_fused_logits_sampling_enabled() {
            return Self::forward_sample_token_fused_logits(
                model,
                device,
                tokens,
                images,
                cache,
                tokenizer,
                sampler,
                previous_tokens,
                params,
            );
        }

        let trace = std::env::var_os("KALOSM_TRACE_DECODE_TIMING").is_some()
            || std::env::var_os("FUSOR_TRACE_DECODE").is_some()
            || std::env::var_os("FUSOR_TRACE_RESOLVE").is_some();
        let fast_decode_enabled = std::env::var_os("KALOSM_LLAMA_FAST_DECODE")
            .map(|value| value != "0")
            .unwrap_or(true);
        let decode_eligible = fast_decode_enabled
            && tokens.len() == 1
            && images.is_empty()
            && cache.as_ref().is_some_and(|cache| !cache.tokens.is_empty());
        let path = if decode_eligible {
            "fast_decode_graph_sample_token"
        } else {
            "graph_fallback_sample_token"
        };
        let token_start = trace.then(std::time::Instant::now);
        let build_start = trace.then(std::time::Instant::now);
        let logits = model.forward(tokens, images, device, cache.as_deref_mut());
        if let Some(start) = build_start {
            eprintln!(
                "forward_graph_build path={path} decode_eligible={decode_eligible} elapsed={:?}",
                start.elapsed()
            );
        }
        let logits = match logits {
            Ok(logits) => logits,
            Err(err) => return Box::pin(async move { Err(err.into()) }),
        };
        let logits = logits.squeeze(0);
        let logits: fusor::Tensor<1, f32> = logits.cast();
        let mut kernels = 0;
        if let Some(logits_key) = logits.gpu_key() {
            let resolve_start = trace.then(std::time::Instant::now);
            kernels = device.resolve_batch(&[logits_key]);
            if let Some(start) = resolve_start {
                eprintln!(
                    "forward_resolve path={path} decode_eligible={decode_eligible} kernels={kernels} elapsed={:?}",
                    start.elapsed()
                );
            }
            if let Some(cache) = cache.as_deref_mut() {
                cache.detach(device);
            }
        }
        Box::pin(async move {
            let download_start = trace.then(std::time::Instant::now);
            let token_id = logits
                .sample_mirostat2_token(sampler, &previous_tokens, params)
                .await?;
            if let Some(start) = download_start {
                eprintln!(
                    "forward_sample_token_download path={path} decode_eligible={decode_eligible} k={} elapsed={:?}",
                    params.top_k,
                    start.elapsed()
                );
            }
            if let Some(start) = token_start {
                record_decode_trace(path, decode_eligible, kernels, start.elapsed());
            }

            Ok(token_id)
        })
    }

    fn forward_sample_token_fused_logits<'a>(
        model: &Model<F>,
        device: &Device,
        tokens: &[u32],
        images: &[(image::DynamicImage, MediaHints)],
        mut cache: Option<&mut LlamaCache>,
        #[allow(unused)] tokenizer: &Tokenizer,
        sampler: &'a mut fusor::GpuMirostat2Sampler,
        previous_tokens: Vec<u32>,
        params: fusor::GpuMirostat2SamplerParams,
    ) -> Pin<
        Box<dyn kalosm_model_types::FutureWasmNotSend<Output = Result<u32, LlamaModelError>> + 'a>,
    > {
        let trace = std::env::var_os("KALOSM_TRACE_DECODE_TIMING").is_some()
            || std::env::var_os("FUSOR_TRACE_DECODE").is_some()
            || std::env::var_os("FUSOR_TRACE_RESOLVE").is_some();
        let fast_decode_enabled = std::env::var_os("KALOSM_LLAMA_FAST_DECODE")
            .map(|value| value != "0")
            .unwrap_or(true);
        let decode_eligible = fast_decode_enabled
            && tokens.len() == 1
            && images.is_empty()
            && cache.as_ref().is_some_and(|cache| !cache.tokens.is_empty());
        let path = if decode_eligible {
            "fast_decode_graph_fused_sample_token"
        } else {
            "graph_fallback_fused_sample_token"
        };
        let token_start = trace.then(std::time::Instant::now);
        let build_start = trace.then(std::time::Instant::now);
        let hidden = model.forward_last_hidden_f32(tokens, images, device, cache.as_deref_mut());
        if let Some(start) = build_start {
            eprintln!(
                "forward_graph_build path={path} decode_eligible={decode_eligible} elapsed={:?}",
                start.elapsed()
            );
        }
        let hidden = match hidden {
            Ok(hidden) => hidden,
            Err(err) => return Box::pin(async move { Err(err.into()) }),
        };
        let hidden = hidden.squeeze(0).to_concrete();
        let output_matrix = model.output_matrix().clone();
        let mut kernels = 0;
        if let Some(hidden_key) = hidden.gpu_key() {
            let resolve_start = trace.then(std::time::Instant::now);
            kernels = device.resolve_batch(&[hidden_key]);
            if let Some(start) = resolve_start {
                eprintln!(
                    "forward_resolve path={path} decode_eligible={decode_eligible} kernels={kernels} elapsed={:?}",
                    start.elapsed()
                );
            }
            if let Some(cache) = cache.as_deref_mut() {
                cache.detach(device);
            }
        }
        Box::pin(async move {
            let sample_start = trace.then(std::time::Instant::now);
            let token_id = match hidden
                .try_sample_mirostat2_token_q_mat(&output_matrix, sampler, &previous_tokens, params)
                .await?
            {
                Some(token_id) => token_id,
                None => {
                    return Err(LlamaModelError::SamplerError(
                        "fused logits sampler refused slow fallback".into(),
                    ));
                }
            };
            if let Some(start) = sample_start {
                eprintln!(
                    "forward_sample_token_download path={path} decode_eligible={decode_eligible} fused_logits=1 k={} elapsed={:?}",
                    params.top_k,
                    start.elapsed()
                );
            }
            if let Some(start) = token_start {
                record_decode_trace(path, decode_eligible, kernels, start.elapsed());
            }

            Ok(token_id)
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
            gpu_sampler,
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

        if gpu_token_sampling_enabled() && stop_on.is_none() {
            if let Some(gpu_sampler_config) = gpu_sampler {
                if let Some(mut gpu_sampler) =
                    LlamaGpuSamplerState::new(&self.device, gpu_sampler_config, seed)
                {
                    let mut next_token = {
                        let top_k = gpu_sample_top_k();
                        let previous_tokens = gpu_sampler.previous_tokens(&text_stream);
                        let params = gpu_sampler.params(top_k);
                        let mut session_lock = session
                            .cache
                            .write()
                            .map_err(|err| LlamaModelError::Session(err.to_string()))?;
                        Self::forward_sample_token(
                            &self.model,
                            &self.device,
                            tokens,
                            &images,
                            Some(&mut session_lock),
                            &self.tokenizer,
                            &mut gpu_sampler.sampler,
                            previous_tokens,
                            params,
                        )
                    }
                    .await?;
                    {
                        let mut session_lock = session
                            .cache
                            .write()
                            .map_err(|err| LlamaModelError::Session(err.to_string()))?;
                        let reserve_tokens = if max_tokens == u32::MAX {
                            unbounded_decode_reserve_tokens()
                        } else {
                            max_tokens as usize
                        };
                        session_lock.reserve_decode(&self.device, reserve_tokens);
                    }

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
                            let top_k = gpu_sample_top_k();
                            let previous_tokens = gpu_sampler.previous_tokens(&text_stream);
                            let params = gpu_sampler.params(top_k);
                            let mut session_lock = session
                                .cache
                                .write()
                                .map_err(|err| LlamaModelError::Session(err.to_string()))?;
                            if max_tokens == u32::MAX {
                                session_lock.reserve_decode(
                                    &self.device,
                                    unbounded_decode_reserve_tokens(),
                                );
                            }
                            Self::forward_sample_token(
                                &self.model,
                                &self.device,
                                &[new_token],
                                &[],
                                Some(&mut session_lock),
                                &self.tokenizer,
                                &mut gpu_sampler.sampler,
                                previous_tokens,
                                params,
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
        }

        let logit_probs = {
            let mut session_lock = session
                .cache
                .write()
                .map_err(|err| LlamaModelError::Session(err.to_string()))?;
            Self::forward_top_k(
                &self.model,
                &self.device,
                tokens,
                &images,
                Some(&mut session_lock),
                &self.tokenizer,
                512,
            )
        }
        .await?;
        {
            let mut session_lock = session
                .cache
                .write()
                .map_err(|err| LlamaModelError::Session(err.to_string()))?;
            let reserve_tokens = if max_tokens == u32::MAX {
                unbounded_decode_reserve_tokens()
            } else {
                max_tokens as usize
            };
            session_lock.reserve_decode(&self.device, reserve_tokens);
        }
        let mut logits = logits_from_sorted_top_k(logit_probs);
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
                if max_tokens == u32::MAX {
                    session_lock.reserve_decode(&self.device, unbounded_decode_reserve_tokens());
                }
                Self::forward_top_k(
                    &self.model,
                    &self.device,
                    &[new_token],
                    &[],
                    Some(&mut session_lock),
                    &self.tokenizer,
                    512,
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
