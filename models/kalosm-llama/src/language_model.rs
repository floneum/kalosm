use fusor::{
    AddOp, CastTensor, CastTo, FloatDataType, FloatOps, MatmulImpl, MulOp, SimdBinaryOp,
    SimdElement, SimdReduceOp, SumOp, WasmNotSync,
};
use kalosm_language_model::{
    ContentChunk, CreateDefaultChatConstraintsForType, CreateDefaultCompletionConstraintsForType,
    CreateTextCompletionSession, GenerationParameters, MessageContent,
    StructuredTextCompletionModel, TextCompletionModel,
};
use kalosm_model_types::{ModelBuilder, ModelLoadingProgress, WasmNotSend};
use kalosm_sample::{ArcParser, CreateParserState, Parse, Parser, ParserExt};
use llm_samplers::types::Sampler;
use std::any::Any;
use std::future::Future;

use crate::model::LlamaModelError;
use crate::structured::generate_structured;
pub use crate::Llama;
use crate::LlamaBuilder;
use crate::{
    InferenceSettings, LlamaResultFuture, LlamaSession, LlamaSourceError, StructuredGenerationTask,
    Task, UnstructuredGenerationTask,
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
        !self.source.model.iter().all(|m| cache.exists(m))
            || self
                .source
                .tokenizer
                .as_ref()
                .filter(|t| cache.exists(t))
                .is_none()
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

impl<F: FloatDataType + SimdElement + Default + FloatOps + MatmulImpl, S: Sampler + 'static>
    TextCompletionModel<S> for Llama<F>
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
        sampler: S,
        on_token: impl FnMut(String) -> Result<(), Self::Error> + WasmNotSend + WasmNotSync + 'static,
    ) -> Result<(), Self::Error> {
        let (tx, rx) = futures::channel::oneshot::channel();
        let (max_tokens, stop_on, seed) =
            match (&sampler as &dyn Any).downcast_ref::<GenerationParameters>() {
                Some(sampler) => (
                    sampler.max_length(),
                    sampler.stop_on().map(|s| s.to_string()),
                    sampler.seed(),
                ),
                None => (u32::MAX, None, None),
            };
        let sampler = std::sync::Arc::new(std::sync::Mutex::new(sampler));
        let on_token = Box::new(on_token);
        let text = msg.text();
        let msg = msg.resolve_media_sources().await?;
        let mut images = Vec::new();
        for chunk in msg.chunks() {
            if let ContentChunk::Media(media) = chunk {
                if let Some(bytes) = &media.source().as_bytes() {
                    // Decode the image from the bytes
                    images.push((image::load_from_memory(bytes)?, media.hints().clone()))
                }
            }
        }
        self.inner
            .sender
            .unbounded_send(Task::UnstructuredGeneration(UnstructuredGenerationTask::<
                F,
            > {
                settings: InferenceSettings::<F>::new(
                    text,
                    images,
                    session.clone(),
                    sampler,
                    max_tokens,
                    stop_on,
                    seed,
                ),
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

impl<F: FloatDataType + SimdElement + Default + FloatOps + MatmulImpl, T: Parse + 'static>
    CreateDefaultChatConstraintsForType<T> for Llama<F>
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

impl<F: FloatDataType + SimdElement + Default + FloatOps + MatmulImpl, T: Parse + 'static>
    CreateDefaultCompletionConstraintsForType<T> for Llama<F>
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

impl<F: FloatDataType + SimdElement + Default + FloatOps + MatmulImpl, S, Constraints>
    StructuredTextCompletionModel<Constraints, S> for Llama<F>
where
    F: CastTo<f32> + CastTensor<f32> + WasmNotSend + WasmNotSync + 'static,
    f32: CastTo<F> + CastTensor<F>,
    MulOp: SimdBinaryOp<F>,
    AddOp: SimdBinaryOp<F>,
    SumOp: SimdReduceOp<F>,
    <Constraints as Parser>::Output: WasmNotSend,
    <Constraints as Parser>::PartialState: WasmNotSend,
    Constraints: CreateParserState + WasmNotSend + 'static,
    S: Sampler + 'static,
{
    fn stream_text_with_callback_and_parser<'a>(
        &'a self,
        session: &'a mut Self::Session,
        text: MessageContent,
        sampler: S,
        parser: Constraints,
        on_token: impl FnMut(String) -> Result<(), Self::Error> + WasmNotSend + WasmNotSync + 'static,
    ) -> impl Future<Output = Result<Constraints::Output, Self::Error>> + WasmNotSend + 'a {
        let mut session = session.clone();
        async move {
            let (tx, rx) = futures::channel::oneshot::channel();
            let seed = match (&sampler as &dyn Any).downcast_ref::<GenerationParameters>() {
                Some(sampler) => sampler.seed(),
                None => None,
            };
            let sampler = std::sync::Arc::new(std::sync::Mutex::new(sampler));
            let on_token = Box::new(on_token);
            let resolved_message = text.resolve_media_sources().await?;
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
                                seed,
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
