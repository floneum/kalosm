//! # RLlama
//!
//! RLlama is a Rust implementation of the quantized [Llama 7B](https://llama.ai/news/announcing-llama-7b/) language model.
//!
//! Llama 7B is a very small but performant language model that can be easily run on your local machine.
//!
//! This library uses [Candle](https://github.com/huggingface/candle) to run Llama.
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

mod chat;
mod chat_template;
mod gguf_tokenizer;
mod language_model;
mod model;
mod raw;
mod session;
mod source;
mod structured;
mod token_stream;

pub use crate::chat::LlamaChatSession;
use crate::model::LlamaModel;
pub use crate::session::LlamaSession;
use fusor::{
    AddOp, CastTo, FloatOps, MatmulImpl, MulOp, SimdBinaryOp, SimdReduceOp, SumOp, WasmNotSend,
    WasmNotSync,
};
use futures::FutureExt;
use kalosm_language_model::{MediaHints, TextCompletionBuilder, TextCompletionModelExt};
pub use kalosm_model_types::FileSource;
use kalosm_model_types::FutureWasmNotSend;
use kalosm_model_types::ModelLoadingProgress;
use kalosm_sample::{LiteralParser, StopOn};
use model::LlamaModelError;
use raw::LlamaConfig;
pub use source::*;
use std::mem::MaybeUninit;
use std::ops::Deref;
use std::sync::Arc;
use std::sync::Mutex;
use std::task::{Context, Poll};
use tokenizers::Tokenizer;

/// Re-export half::f16 for users who want to use f16 activation types
pub use half::f16;

/// Re-export fusor types needed for the activation type generic
pub use fusor::{CastTensor, Device, FloatDataType, SimdElement};

/// A prelude of commonly used items in kalosm-llama.
pub mod prelude {
    pub use crate::session::LlamaSession;
    pub use crate::{Llama, LlamaBuilder, LlamaSource};
    pub use kalosm_language_model::*;
    pub use kalosm_model_types::FileSource;
}

// On wasm32, callbacks don't need to be Send/Sync
#[cfg(not(target_arch = "wasm32"))]
pub(crate) type BoxedTokenCallback =
    Box<dyn FnMut(String) -> Result<(), LlamaModelError> + Send + Sync>;
#[cfg(target_arch = "wasm32")]
pub(crate) type BoxedTokenCallback = Box<dyn FnMut(String) -> Result<(), LlamaModelError>>;

use std::future::Future;
use std::pin::Pin;

#[cfg(not(target_arch = "wasm32"))]
type BoxedRunner<F> = Box<
    dyn for<'a> FnOnce(&'a LlamaModel<F>) -> Pin<Box<dyn Future<Output = ()> + Send + 'a>> + Send,
>;
#[cfg(target_arch = "wasm32")]
type BoxedRunner<F> =
    Box<dyn for<'a> FnOnce(&'a LlamaModel<F>) -> Pin<Box<dyn Future<Output = ()> + 'a>>>;

enum Task<F: FloatDataType + SimdElement = f32> {
    UnstructuredGeneration(UnstructuredGenerationTask<F>),
    StructuredGeneration(StructuredGenerationTask<F>),
}

#[allow(clippy::type_complexity)]
struct StructuredGenerationTask<F: FloatDataType + SimdElement = f32> {
    runner: BoxedRunner<F>,
}

struct UnstructuredGenerationTask<F: FloatDataType + SimdElement = f32> {
    settings: InferenceSettings<F>,
    on_token: BoxedTokenCallback,
    finished: futures::channel::oneshot::Sender<Result<(), LlamaModelError>>,
}

struct LlamaTask<F: FloatDataType + SimdElement = f32> {
    sender: futures::channel::mpsc::UnboundedSender<Task<F>>,
    task: Mutex<Pin<Box<dyn FutureWasmNotSend<Output = ()> + 'static>>>,
}

/// A future that polls the background Llama task when awaited.
pub(crate) struct LlamaResultFuture<F: FloatDataType + SimdElement, T> {
    llama: Llama<F>,
    receiver: futures::channel::oneshot::Receiver<T>,
}

impl<F, T> Future for LlamaResultFuture<F, T>
where
    F: FloatDataType + SimdElement,
{
    type Output = Result<T, futures::channel::oneshot::Canceled>;

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
pub struct Llama<F: FloatDataType + SimdElement = f32> {
    config: Arc<LlamaConfig<F>>,
    tokenizer: Arc<Tokenizer>,
    inner: Arc<LlamaTask<F>>,
}

impl<F: FloatDataType + SimdElement> Clone for Llama<F> {
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
}

impl<F> Llama<F>
where
    F: FloatDataType
        + SimdElement
        + Default
        + CastTo<f32>
        + CastTensor<f32>
        + WasmNotSend
        + WasmNotSync
        + FloatOps
        + MatmulImpl
        + 'static,
    f32: CastTo<F> + CastTensor<F>,
    MulOp: SimdBinaryOp<F>,
    AddOp: SimdBinaryOp<F>,
    SumOp: SimdReduceOp<F>,
{
    /// Get the tokenizer for the model.
    pub fn tokenizer(&self) -> &Arc<Tokenizer> {
        &self.tokenizer
    }

    #[allow(clippy::too_many_arguments)]
    fn from_build(mut model: LlamaModel<F>) -> Self {
        use futures::StreamExt;

        let (task_sender, mut task_receiver) = futures::channel::mpsc::unbounded();
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
    pub fn default_assistant_constraints(&self) -> StopOn<String> {
        let end_token = self.config.stop_token_string.clone();

        StopOn::from(end_token)
    }

    /// Get the constraints that end the assistant's response.
    pub fn end_assistant_marker_constraints(&self) -> LiteralParser {
        let end_token = self.config.stop_token_string.clone();

        LiteralParser::from(end_token)
    }
}

impl<F> Deref for Llama<F>
where
    F: FloatDataType
        + SimdElement
        + Default
        + CastTo<f32>
        + CastTensor<f32>
        + WasmNotSend
        + WasmNotSync
        + FloatOps
        + MatmulImpl
        + 'static,
    f32: CastTo<F> + CastTensor<F>,
    MulOp: SimdBinaryOp<F>,
    AddOp: SimdBinaryOp<F>,
    SumOp: SimdReduceOp<F>,
{
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
pub struct LlamaBuilder<F: FloatDataType + SimdElement = f32> {
    source: source::LlamaSource,
    device: Option<Device>,
    flash_attn: bool,
    _marker: std::marker::PhantomData<F>,
}

impl<F: FloatDataType + SimdElement> Default for LlamaBuilder<F> {
    fn default() -> Self {
        Self {
            source: Default::default(),
            device: None,
            flash_attn: false,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<F: FloatDataType + SimdElement> LlamaBuilder<F> {
    /// Create a new Llama builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the source for the model.
    pub fn with_source(mut self, source: source::LlamaSource) -> Self {
        self.source = source;
        self
    }

    /// Set whether to use Flash Attention.
    pub fn with_flash_attn(mut self, use_flash_attn: bool) -> Self {
        self.flash_attn = use_flash_attn;
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
            None => Device::auto().await,
        }
    }
}

impl<F: FloatDataType + SimdElement + Default + FloatOps + MatmulImpl> LlamaBuilder<F>
where
    F: CastTo<f32> + CastTensor<f32> + WasmNotSend + WasmNotSync + 'static,
    f32: CastTo<F> + CastTensor<F>,
    MulOp: SimdBinaryOp<F>,
    AddOp: SimdBinaryOp<F>,
    SumOp: SimdReduceOp<F>,
{
    /// Build the model with a handler for progress as the download and loading progresses.
    ///
    /// ```rust, no_run
    /// use kalosm::language::*;
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), anyhow::Error> {
    /// // Create a new llama model with a loading handler
    /// let model = Llama::builder()
    ///     .build_with_loading_handler(|progress| match progress {
    ///         ModelLoadingProgress::Downloading { source, progress } => {
    ///             let progress_percent = (progress.progress * 100) as u32;
    ///             let elapsed = progress
    ///                 .start_time
    ///                 .expect("start time is available on native targets")
    ///                 .elapsed()
    ///                 .as_secs_f32();
    ///             println!("Downloading file {source} {progress_percent}% ({elapsed}s)");
    ///         }
    ///         ModelLoadingProgress::Loading { progress } => {
    ///             let progress = (progress * 100.0) as u32;
    ///             println!("Loading model {progress}%");
    ///         }
    ///     })
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn build_with_loading_handler(
        self,
        handler: impl FnMut(ModelLoadingProgress) + WasmNotSend + WasmNotSync + 'static,
    ) -> Result<Llama<F>, LlamaSourceError> {
        let model = LlamaModel::from_builder(self, handler).await?;

        Ok(Llama::from_build(model))
    }

    /// Build the model (this will download the model if it is not already downloaded)
    pub async fn build(self) -> Result<Llama<F>, LlamaSourceError> {
        self.build_with_loading_handler(ModelLoadingProgress::multi_bar_loading_indicator())
            .await
    }
}

pub(crate) struct InferenceSettings<F: FloatDataType + SimdElement = f32> {
    /// The prompt to use.
    prompt: String,

    /// Images in the prompt
    images: Vec<(image::DynamicImage, MediaHints)>,

    /// The token to stop on.
    stop_on: Option<String>,

    /// The sampler to use.
    sampler: std::sync::Arc<std::sync::Mutex<dyn llm_samplers::prelude::Sampler>>,

    /// The session to use.
    session: LlamaSession<F>,

    /// The maximum number of tokens to generate.
    max_tokens: u32,

    /// The seed to use.
    seed: Option<u64>,
}

impl<F: FloatDataType + SimdElement> InferenceSettings<F> {
    pub fn new(
        prompt: impl ToString,
        images: Vec<(image::DynamicImage, MediaHints)>,
        session: LlamaSession<F>,
        sampler: std::sync::Arc<std::sync::Mutex<dyn llm_samplers::prelude::Sampler>>,
        max_tokens: u32,
        stop_on: Option<String>,
        seed: Option<u64>,
    ) -> Self {
        let prompt = prompt.to_string();
        Self {
            prompt,
            images,
            stop_on,
            sampler,
            session,
            max_tokens,
            seed,
        }
    }
}
