use fusor::{
    AddOp, CastTensor, CastTo, FloatDataType, FloatOps, MatmulImpl, MulOp, SimdBinaryOp,
    SimdElement, SimdReduceOp, SumOp, WasmNotSync,
};
#[cfg(feature = "vision")]
use kalosm_language_model::ContentChunk;
use kalosm_language_model::{
    CreateTextCompletionSession, GenerationParameters, MessageContent, TextCompletionModel,
};
use kalosm_model_types::{ModelBuilder, ModelLoadingProgress, WasmNotSend};
#[cfg(feature = "structured")]
use kalosm_sample::{ArcParser, CreateParserState, Parse, Parser, ParserExt};
#[cfg(feature = "structured")]
use std::future::Future;

use crate::model::LlamaModelError;
#[cfg(feature = "structured")]
use crate::sampler::CpuMirostat2Sampler;
#[cfg(feature = "structured")]
use crate::structured::generate_structured;
pub use crate::Llama;
use crate::LlamaBuilder;
#[cfg(feature = "structured")]
use crate::StructuredGenerationTask;
use crate::{
    GpuSamplerConfig, InferenceSettings, LlamaResultFuture, LlamaSession, LlamaSourceError, Task,
    UnstructuredGenerationTask,
};

impl<F: FloatDataType + SimdElement + Default + FloatOps + MatmulImpl> ModelBuilder
    for LlamaBuilder<F>
where
    F: CastTo<f32> + CastTensor<f32> + WasmNotSend + WasmNotSync + 'static,
    f32: CastTo<F> + CastTensor<F>,
    MulOp: SimdBinaryOp<F>,
    AddOp: SimdBinaryOp<F>,
    SumOp: SimdReduceOp<F>,
{
    type Model = Llama<F>;
    type Error = LlamaSourceError;

    async fn start_with_loading_handler(
        self,
        handler: impl FnMut(ModelLoadingProgress) + WasmNotSend + WasmNotSync + 'static,
    ) -> Result<Self::Model, Self::Error> {
        self.build_with_loading_handler(handler).await
    }

    fn requires_download(&self) -> bool {
        let cache = &self.source.cache;
        let model_missing = !self.source.model.iter().all(|m| cache.exists(m));
        let tokenizer_missing = self
            .source
            .tokenizer
            .as_ref()
            .is_some_and(|tokenizer| !cache.exists(tokenizer));
        model_missing || tokenizer_missing
    }
}

impl<F: FloatDataType + SimdElement + Default + FloatOps + MatmulImpl> CreateTextCompletionSession
    for Llama<F>
where
    F: CastTo<f32> + CastTensor<f32> + WasmNotSend + WasmNotSync + 'static,
    f32: CastTo<F> + CastTensor<F>,
    MulOp: SimdBinaryOp<F>,
    AddOp: SimdBinaryOp<F>,
    SumOp: SimdReduceOp<F>,
{
    type Session = LlamaSession<F>;
    type Error = LlamaModelError;

    fn new_session(&self) -> Result<Self::Session, Self::Error> {
        Ok(LlamaSession::new(&self.config))
    }
}

impl<F: FloatDataType + SimdElement + Default + FloatOps + MatmulImpl>
    TextCompletionModel<GenerationParameters> for Llama<F>
where
    F: CastTo<f32> + CastTensor<f32> + WasmNotSend + WasmNotSync + 'static,
    f32: CastTo<F> + CastTensor<F>,
    MulOp: SimdBinaryOp<F>,
    AddOp: SimdBinaryOp<F>,
    SumOp: SimdReduceOp<F>,
{
    async fn stream_text_with_callback<'a>(
        &'a self,
        session: &'a mut Self::Session,
        msg: MessageContent,
        sampler: GenerationParameters,
        on_token: impl FnMut(String) -> Result<(), Self::Error> + WasmNotSend + WasmNotSync + 'static,
    ) -> Result<(), Self::Error> {
        let (tx, rx) = futures::channel::oneshot::channel();
        let max_tokens = sampler.max_length();
        let stop_on = sampler.stop_on().map(|s| s.to_string());
        let seed = sampler.seed();
        let sampler = GpuSamplerConfig::from_generation_parameters(&sampler);
        let on_token = Box::new(on_token);
        let text = msg.text();
        #[cfg(feature = "vision")]
        let images = {
            let msg = msg.resolve_media_sources().await?;
            let mut images = Vec::new();
            for chunk in msg.chunks() {
                if let ContentChunk::Media(media) = chunk {
                    if let Some(bytes) = &media.source().as_bytes() {
                        images.push((image::load_from_memory(bytes)?, media.hints().clone()))
                    }
                }
            }
            images
        };
        #[cfg(not(feature = "vision"))]
        let images = {
            if msg.has_media() {
                return Err(LlamaModelError::MediaUnsupported);
            }
            Vec::new()
        };
        self.inner
            .sender
            .unbounded_send(Task::UnstructuredGeneration(UnstructuredGenerationTask::<
                F,
            > {
                settings: InferenceSettings::<F> {
                    prompt: text,
                    images,
                    session: session.clone(),
                    sampler,
                    max_tokens,
                    stop_on,
                    seed,
                },
                on_token,
                finished: tx,
            }))
            .map_err(|_| LlamaModelError::ModelStopped)?;

        LlamaResultFuture {
            llama: self.clone(),
            receiver: rx,
        }
        .await
        .map_err(|_| LlamaModelError::ModelStopped)??;

        Ok(())
    }
}

#[cfg(feature = "structured")]
impl<F: FloatDataType + SimdElement + Default + FloatOps + MatmulImpl, T: Parse + 'static>
    kalosm_language_model::CreateDefaultChatConstraintsForType<T> for Llama<F>
where
    F: CastTo<f32> + CastTensor<f32> + WasmNotSend + WasmNotSync + 'static,
    f32: CastTo<F> + CastTensor<F>,
    MulOp: SimdBinaryOp<F>,
    AddOp: SimdBinaryOp<F>,
    SumOp: SimdReduceOp<F>,
{
    type DefaultConstraints = ArcParser<T>;

    fn create_default_constraints() -> Self::DefaultConstraints {
        T::new_parser().boxed()
    }
}

#[cfg(feature = "structured")]
impl<F: FloatDataType + SimdElement + Default + FloatOps + MatmulImpl, T: Parse + 'static>
    kalosm_language_model::CreateDefaultCompletionConstraintsForType<T> for Llama<F>
where
    F: CastTo<f32> + CastTensor<f32> + WasmNotSend + WasmNotSync + 'static,
    f32: CastTo<F> + CastTensor<F>,
    MulOp: SimdBinaryOp<F>,
    AddOp: SimdBinaryOp<F>,
    SumOp: SimdReduceOp<F>,
{
    type DefaultConstraints = ArcParser<T>;

    fn create_default_constraints() -> Self::DefaultConstraints {
        T::new_parser().boxed()
    }
}

#[cfg(feature = "structured")]
impl<F: FloatDataType + SimdElement + Default + FloatOps + MatmulImpl, Constraints>
    kalosm_language_model::StructuredTextCompletionModel<Constraints, GenerationParameters>
    for Llama<F>
where
    F: CastTo<f32> + CastTensor<f32> + WasmNotSend + WasmNotSync + 'static,
    f32: CastTo<F> + CastTensor<F>,
    MulOp: SimdBinaryOp<F>,
    AddOp: SimdBinaryOp<F>,
    SumOp: SimdReduceOp<F>,
    <Constraints as Parser>::Output: WasmNotSend,
    <Constraints as Parser>::PartialState: WasmNotSend,
    Constraints: CreateParserState + WasmNotSend + 'static,
{
    fn stream_text_with_callback_and_parser<'a>(
        &'a self,
        session: &'a mut Self::Session,
        text: MessageContent,
        sampler: GenerationParameters,
        parser: Constraints,
        on_token: impl FnMut(String) -> Result<(), Self::Error> + WasmNotSend + WasmNotSync + 'static,
    ) -> impl Future<Output = Result<Constraints::Output, Self::Error>> + WasmNotSend + 'a {
        let mut session = session.clone();
        async move {
            let (tx, rx) = futures::channel::oneshot::channel();
            let seed = sampler.seed();
            let sampler = CpuMirostat2Sampler::new(
                GpuSamplerConfig::from_generation_parameters(&sampler),
                seed,
            );
            let on_token = Box::new(on_token);
            #[cfg(feature = "vision")]
            let resolved_message = text.resolve_media_sources().await?;
            #[cfg(not(feature = "vision"))]
            let resolved_message = {
                if text.has_media() {
                    return Err(LlamaModelError::MediaUnsupported);
                }
                text
            };
            self.inner
                .sender
                .unbounded_send(Task::StructuredGeneration(StructuredGenerationTask {
                    runner: Box::new(move |model| {
                        Box::pin(async move {
                            let parser_state = parser.create_parser_state();
                            let result = generate_structured(
                                resolved_message,
                                model,
                                &mut session,
                                parser,
                                parser_state,
                                sampler,
                                on_token,
                                Some(64),
                            )
                            .await;
                            _ = tx.send(result);
                        })
                    }),
                }))
                .map_err(|_| LlamaModelError::ModelStopped)?;

            let result = LlamaResultFuture {
                llama: self.clone(),
                receiver: rx,
            }
            .await
            .map_err(|_| LlamaModelError::ModelStopped)??;

            Ok(result)
        }
    }
}
