//! # RLlama
//!
//! RLlama is a Rust implementation of the quantized [Llama 7B](https://llama.ai/news/announcing-llama-7b/) language model.
//!
//! Llama 7B is a very small but performant language model that can be easily run on your local machine.
//!
//! This library uses Fusor to run Llama.
//!
//! ## Usage
//!
//! ```rust, no_run
//! use kalosm_llama::prelude::*;
//!
//! #[tokio::main]
//! async fn main() {
//!     let mut model = Llama::new().await.unwrap();
//!     let prompt = "The capital of France is ";
//!     let mut stream = model(prompt);
//!
//!     print!("{prompt}");
//!     while let Some(token) = stream.next().await {
//!         print!("{token}");
//!     }
//! }
//! ```

#![warn(missing_docs)]
#![recursion_limit = "256"]

mod chat;
mod chat_template;
mod gguf_tokenizer;
mod language_model;
mod model;
mod raw;
mod sampler;
mod session;
mod source;
#[cfg(feature = "structured")]
mod structured;
mod token_stream;
mod tokenizer;

pub use crate::chat::LlamaChatSession;
use crate::model::LlamaModel;
pub use crate::session::LlamaSession;
use futures_util::FutureExt;
use kalosm_language_model::{TextCompletionBuilder, TextCompletionModelExt};
pub use kalosm_model_types::FileSource;
use kalosm_model_types::FutureWasmNotSend;
use kalosm_model_types::ModelLoadingProgress;
use kalosm_model_types::WasmNotSend;

/// `Sync` on native targets, nothing on wasm — the marker the old fusor facade
/// exported under this name.
#[cfg(not(target_arch = "wasm32"))]
#[doc(hidden)]
pub trait WasmNotSync: Sync {}
#[cfg(not(target_arch = "wasm32"))]
impl<T: Sync> WasmNotSync for T {}
#[cfg(target_arch = "wasm32")]
#[doc(hidden)]
pub trait WasmNotSync {}
#[cfg(target_arch = "wasm32")]
impl<T> WasmNotSync for T {}
#[cfg(feature = "structured")]
use kalosm_sample::{LiteralParser, StopOn};
use model::LlamaModelError;
use raw::LlamaConfig;
pub use source::*;
use std::mem::MaybeUninit;
use std::ops::Deref;
use std::sync::Arc;
use std::sync::Mutex;
use std::task::{Context, Poll};
pub use tokenizer::{LlamaTokenizer, LlamaTokenizerError};

pub(crate) type LlamaImage = ();

/// Re-export half::f16 for users who want to use f16 activation types
pub use half::f16;

/// Re-export the fusor2 device handle used to run the model.
pub use fusor2::device::Device;

/// A prelude of commonly used items in kalosm-llama.
pub mod prelude {
    pub use crate::session::LlamaSession;
    pub use crate::{Llama, LlamaBuilder, LlamaSource};
    pub use kalosm_language_model::*;
    pub use kalosm_model_types::FileSource;
}

// On wasm32, callbacks don't need to be Send/Sync; the `WasmNot*` markers encode that.
pub(crate) trait TokenCallback:
    FnMut(String) -> Result<(), LlamaModelError> + WasmNotSend + WasmNotSync
{
}
impl<T: FnMut(String) -> Result<(), LlamaModelError> + WasmNotSend + WasmNotSync> TokenCallback
    for T
{
}
pub(crate) type BoxedTokenCallback = Box<dyn TokenCallback>;

use std::future::Future;
use std::pin::Pin;

#[cfg(feature = "structured")]
trait Runner:
    for<'a> FnOnce(&'a LlamaModel) -> Pin<Box<dyn FutureWasmNotSend<Output = ()> + 'a>> + WasmNotSend
{
}
#[cfg(feature = "structured")]
impl<
        T: for<'a> FnOnce(&'a LlamaModel) -> Pin<Box<dyn FutureWasmNotSend<Output = ()> + 'a>>
            + WasmNotSend,
    > Runner for T
{
}
#[cfg(feature = "structured")]
type BoxedRunner = Box<dyn Runner>;

enum Task {
    UnstructuredGeneration(UnstructuredGenerationTask),
    #[cfg(feature = "structured")]
    StructuredGeneration(StructuredGenerationTask),
}

#[allow(clippy::type_complexity)]
#[cfg(feature = "structured")]
struct StructuredGenerationTask {
    runner: BoxedRunner,
}

struct UnstructuredGenerationTask {
    settings: InferenceSettings,
    on_token: BoxedTokenCallback,
    finished: futures_channel::oneshot::Sender<Result<(), LlamaModelError>>,
}

struct LlamaTask {
    sender: futures_channel::mpsc::UnboundedSender<Task>,
    task: Mutex<Pin<Box<dyn FutureWasmNotSend<Output = ()> + 'static>>>,
}

/// A future that polls the background Llama task when awaited.
pub(crate) struct LlamaResultFuture<T> {
    llama: Llama,
    receiver: futures_channel::oneshot::Receiver<T>,
}

impl<T> Future for LlamaResultFuture<T> {
    type Output = Result<T, futures_channel::oneshot::Canceled>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let myself = self.get_mut();

        // Poll the background task to make progress
        {
            let mut task = myself.llama.inner.task.lock().unwrap();
            let _ = task.poll_unpin(cx);
        }

        // Poll the receiver for the result
        Pin::new(&mut myself.receiver).poll(cx)
    }
}

/// A quantized Llama language model with support for streaming generation.
pub struct Llama {
    config: Arc<LlamaConfig>,
    tokenizer: Arc<LlamaTokenizer>,
    inner: Arc<LlamaTask>,
}

impl Clone for Llama {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            tokenizer: self.tokenizer.clone(),
            inner: self.inner.clone(),
        }
    }
}

impl Llama {
    /// Create a default chat model.
    pub async fn new_chat() -> Result<Self, LlamaSourceError> {
        Self::builder()
            .with_source(LlamaSource::llama_3_1_8b_chat())
            .build()
            .await
    }

    /// Create a default phi-3 chat model.
    pub async fn phi_3() -> Result<Self, LlamaSourceError> {
        Self::builder()
            .with_source(LlamaSource::phi_3_5_mini_4k_instruct())
            .build()
            .await
    }

    /// Create a default text generation model.
    pub async fn new() -> Result<Self, LlamaSourceError> {
        Self::builder()
            .with_source(LlamaSource::llama_8b())
            .build()
            .await
    }

    /// Create a new builder for a Llama model.
    pub fn builder() -> LlamaBuilder {
        LlamaBuilder::default()
    }

    /// Get the tokenizer for the model.
    pub fn tokenizer(&self) -> &Arc<LlamaTokenizer> {
        &self.tokenizer
    }

    #[allow(clippy::too_many_arguments)]
    fn from_build(mut model: LlamaModel) -> Self {
        use futures_util::StreamExt;

        let (task_sender, mut task_receiver) = futures_channel::mpsc::unbounded();
        let config = model.model.config.clone();
        let tokenizer = model.tokenizer.clone();

        // Create a future that processes tasks when polled
        let task = Box::pin(async move {
            while let Some(task) = task_receiver.next().await {
                match task {
                    Task::UnstructuredGeneration(UnstructuredGenerationTask {
                        settings,
                        on_token,
                        finished,
                    }) => {
                        let result = model._infer(settings, on_token, &finished).await;
                        if let Err(err) = &result {
                            tracing::error!("Error running model: {err}");
                        }
                        _ = finished.send(result);
                    }
                    #[cfg(feature = "structured")]
                    Task::StructuredGeneration(StructuredGenerationTask { runner }) => {
                        runner(&model).await;
                    }
                }
            }
        });

        Self {
            config,
            tokenizer,
            inner: Arc::new(LlamaTask {
                sender: task_sender,
                task: Mutex::new(task),
            }),
        }
    }

    /// Get the default constraints for an assistant response. It parses any text until the end of the assistant's response.
    #[cfg(feature = "structured")]
    pub fn default_assistant_constraints(&self) -> StopOn<String> {
        let end_token = self.config.stop_token_string.clone();

        StopOn::from(end_token)
    }

    /// Get the constraints that end the assistant's response.
    #[cfg(feature = "structured")]
    pub fn end_assistant_marker_constraints(&self) -> LiteralParser {
        let end_token = self.config.stop_token_string.clone();

        LiteralParser::from(end_token)
    }
}

impl Deref for Llama {
    type Target = dyn Fn(&str) -> TextCompletionBuilder<Self>;

    fn deref(&self) -> &Self::Target {
        // https://github.com/dtolnay/case-studies/tree/master/callable-types

        // Create an empty allocation for Self.
        let uninit_callable = MaybeUninit::<Self>::uninit();
        // Move a closure that captures just self into the uninitialized memory. Closures create an anonymous type that implement
        // FnOnce. In this case, the layout of the type should just be Self because self is the only field in the closure type.
        let uninit_closure = move |text: &str| {
            TextCompletionModelExt::complete(unsafe { &*uninit_callable.as_ptr() }, text)
        };

        // Make sure the layout of the closure and Self is the same.
        let size_of_closure = std::alloc::Layout::for_value(&uninit_closure);
        assert_eq!(size_of_closure, std::alloc::Layout::new::<Self>());

        // Then cast the lifetime of the closure to the lifetime of &self.
        fn cast_lifetime<'a, T>(_a: &T, b: &'a T) -> &'a T {
            b
        }
        let reference_to_closure = cast_lifetime(
            {
                // The real closure that we will never use.
                &uninit_closure
            },
            #[allow(clippy::missing_transmute_annotations)]
            // We transmute self into a reference to the closure. This is safe because we know that the closure has the same memory layout as Self so &Closure == &Self.
            unsafe {
                std::mem::transmute(self)
            },
        );

        // Cast the closure to a trait object.
        reference_to_closure as &_
    }
}

/// A builder with configuration for a Llama model.
pub struct LlamaBuilder {
    source: source::LlamaSource,
    device: Option<Device>,
}

impl Default for LlamaBuilder {
    fn default() -> Self {
        Self {
            source: Default::default(),
            device: None,
        }
    }
}

impl LlamaBuilder {
    /// Create a new Llama builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the source for the model.
    pub fn with_source(mut self, source: source::LlamaSource) -> Self {
        self.source = source;
        self
    }

    /// Set the device to run the model with. (Defaults to an accelerator if available, otherwise the CPU)
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    /// Get the device or the default device if not set.
    pub(crate) async fn get_device(&self) -> Device {
        match self.device.clone() {
            Some(device) => device,
            None => match Device::gpu().await {
                Ok(device) => device,
                Err(err) => {
                    tracing::warn!("no gpu adapter, falling back to cpu: {err}");
                    Device::cpu()
                }
            },
        }
    }

    /// Build the model with a handler for progress as the download and loading progresses.
    pub async fn build_with_loading_handler(
        self,
        handler: impl FnMut(ModelLoadingProgress) + WasmNotSend + WasmNotSync + 'static,
    ) -> Result<Llama, LlamaSourceError> {
        let model = LlamaModel::from_builder(self, handler).await?;

        Ok(Llama::from_build(model))
    }

    /// Build the model (this will download the model if it is not already downloaded)
    pub async fn build(self) -> Result<Llama, LlamaSourceError> {
        self.build_with_loading_handler(ModelLoadingProgress::multi_bar_loading_indicator())
            .await
    }
}

pub(crate) struct InferenceSettings {
    /// The prompt to use.
    pub(crate) prompt: String,

    /// Images in the prompt
    pub(crate) images: Vec<LlamaImage>,

    /// The token to stop on.
    pub(crate) stop_on: Option<String>,

    /// The sampler to use.
    pub(crate) sampler: GpuSamplerConfig,

    /// The session to use.
    pub(crate) session: LlamaSession,

    /// The maximum number of tokens to generate.
    pub(crate) max_tokens: u32,

    /// The seed to use.
    pub(crate) seed: Option<u64>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GpuSamplerConfig {
    pub(crate) temperature: f32,
    pub(crate) tau: f32,
    pub(crate) eta: f32,
    pub(crate) mu: f32,
    pub(crate) sampling_strategy: kalosm_language_model::SamplingStrategy,
    pub(crate) top_p: Option<f32>,
    pub(crate) min_p: Option<f32>,
    pub(crate) repetition_penalty: f32,
    pub(crate) repetition_penalty_range: usize,
    pub(crate) top_k: Option<usize>,
}

impl GpuSamplerConfig {
    pub(crate) fn new(
        temperature: f32,
        tau: f32,
        eta: f32,
        mu: f32,
        repetition_penalty: f32,
        repetition_penalty_range: usize,
        top_k: Option<usize>,
    ) -> Self {
        Self {
            temperature,
            tau,
            eta,
            mu,
            sampling_strategy: kalosm_language_model::SamplingStrategy::Mirostat2,
            top_p: None,
            min_p: None,
            repetition_penalty,
            repetition_penalty_range,
            top_k,
        }
    }

    pub(crate) fn from_generation_parameters(
        sampler: &kalosm_language_model::GenerationParameters,
    ) -> Self {
        let mut config = Self::new(
            sampler.temperature(),
            sampler.tau(),
            sampler.eta(),
            sampler.mu(),
            sampler.repetition_penalty(),
            sampler.repetition_penalty_range() as usize,
            sampler.top_k().map(|top_k| top_k as usize),
        );
        config.sampling_strategy = sampler.sampling_strategy();
        config.top_p = sampler.top_p().map(|top_p| top_p as f32);
        config.min_p = sampler.min_p().map(|min_p| min_p as f32);
        config
    }
}
