use std::{
    future::Future,
    sync::{Arc, RwLock},
};

use crate::{model::LlamaModelError, session::LlamaSessionLoadingError, Llama, LlamaSession};
use fusor::{
    AddOp, CastTensor, CastTo, FloatDataType, FloatOps, MatmulImpl, MulOp, SimdBinaryOp,
    SimdElement, SimdReduceOp, SumOp,
};
#[cfg(feature = "structured")]
use kalosm_language_model::StructuredTextCompletionModel;
use kalosm_language_model::{
    ChatMessage, ChatModel, ChatSession, ContentChunk, CreateChatSession,
    CreateTextCompletionSession, GenerationParameters, MessageContent, MessageType,
    TextCompletionModel,
};
use kalosm_model_types::{WasmNotSend, WasmNotSendSync};
#[cfg(feature = "structured")]
use kalosm_sample::{CreateParserState, Parser};
use minijinja::ErrorKind;

fn get_new_tokens<F: FloatDataType + SimdElement>(
    messages: &[ChatMessage],
    session: &mut LlamaChatSession<F>,
    model: &Llama<F>,
) -> Result<String, LlamaModelError> {
    let chat_template = model
        .config
        .chat_template
        .as_ref()
        .ok_or(LlamaModelError::NoChatTemplate)?;
    let bos_token = &model.config.start_token_string;
    let eos_token = &model.config.stop_token_string;
    let current_text = if session.history.is_empty() {
        String::new()
    } else {
        let old_formatted_text =
            chat_template.format(bos_token, eos_token, &session.history, true)?;
        // Some chat templates (like llama v3) always include the generation prompt even when we tell them not to. If they do, try to strip it off
        let (before_last_eos, _) = old_formatted_text
            .rsplit_once(eos_token)
            .unwrap_or((&old_formatted_text, ""));
        before_last_eos.to_string() + eos_token
    };
    session.history.extend_from_slice(messages);
    let updated_text = chat_template.format(bos_token, eos_token, &session.history, true)?;
    let new_text = updated_text.strip_prefix(&current_text).ok_or_else(|| {
        LlamaModelError::ChatTemplateError(minijinja::Error::new(
            ErrorKind::InvalidOperation,
            format!("Chat template should only add text to the end of the current text. Old text: {current_text}, new text: {updated_text}"),
        ))
    })?;

    Ok(new_text.to_string())
}

impl<F: FloatDataType + SimdElement + Default + FloatOps + MatmulImpl> CreateChatSession
    for Llama<F>
where
    F: CastTo<f32> + CastTensor<f32> + WasmNotSendSync + 'static,
    f32: CastTo<F> + CastTensor<F>,
    MulOp: SimdBinaryOp<F>,
    AddOp: SimdBinaryOp<F>,
    SumOp: SimdReduceOp<F>,
{
    type Error = LlamaModelError;
    type ChatSession = LlamaChatSession<F>;

    fn new_chat_session(&self) -> Result<Self::ChatSession, Self::Error> {
        Ok(LlamaChatSession::new(self.new_session()?))
    }
}

impl<F: FloatDataType + SimdElement + Default + FloatOps + MatmulImpl>
    ChatModel<GenerationParameters> for Llama<F>
where
    F: CastTo<f32> + CastTensor<f32> + WasmNotSendSync + 'static,
    f32: CastTo<F> + CastTensor<F>,
    MulOp: SimdBinaryOp<F>,
    AddOp: SimdBinaryOp<F>,
    SumOp: SimdReduceOp<F>,
{
    fn add_messages_with_callback<'a>(
        &'a self,
        mut session: Self::ChatSession,
        messages: &'a [ChatMessage],
        sampler: GenerationParameters,
        mut on_token: impl FnMut(String) -> Result<(), Self::Error> + WasmNotSendSync + 'static,
    ) -> impl Future<Output = Result<Self::ChatSession, Self::Error>> + WasmNotSend + 'a {
        let new_text = get_new_tokens(messages, &mut session, self);
        let mut content = MessageContent::new();
        for message in messages {
            for chunk in message.content().chunks() {
                if matches!(chunk, ContentChunk::Media(_)) {
                    content.push(chunk.clone());
                }
            }
        }
        async move {
            let new_text = new_text?;
            let model_response = Arc::new(RwLock::new(String::new()));
            let on_token = {
                let model_response = model_response.clone();
                move |token: String| {
                    let mut model_response = model_response.write().unwrap();
                    *model_response += &token;
                    on_token(token)
                }
            };
            content.push(new_text);

            self.stream_text_with_callback(&mut session.session, content, sampler, on_token)
                .await?;
            session.history.push(ChatMessage::new(
                MessageType::ModelAnswer,
                model_response.read().unwrap().clone(),
            ));
            Ok(session)
        }
    }
}

#[cfg(feature = "structured")]
impl<F: FloatDataType + SimdElement + Default + FloatOps + MatmulImpl, Constraints>
    kalosm_language_model::StructuredChatModel<Constraints, GenerationParameters> for Llama<F>
where
    F: CastTo<f32> + CastTensor<f32> + WasmNotSendSync + 'static,
    f32: CastTo<F> + CastTensor<F>,
    MulOp: SimdBinaryOp<F>,
    AddOp: SimdBinaryOp<F>,
    SumOp: SimdReduceOp<F>,
    <Constraints as Parser>::Output: WasmNotSend,
    <Constraints as Parser>::PartialState: WasmNotSend,
    Constraints: CreateParserState + WasmNotSend + 'static,
{
    fn add_message_with_callback_and_constraints<'a>(
        &'a self,
        mut session: Self::ChatSession,
        messages: &'a [ChatMessage],
        sampler: GenerationParameters,
        constraints: Constraints,
        mut on_token: impl FnMut(String) -> Result<(), Self::Error> + WasmNotSendSync + 'static,
    ) -> impl Future<
        Output = Result<
            (
                Self::ChatSession,
                <Constraints as kalosm_language_model::ModelConstraints>::Output,
            ),
            Self::Error,
        >,
    > + WasmNotSend
           + 'a
    where
        <Constraints as kalosm_language_model::ModelConstraints>::Output: 'a,
    {
        let mut content = MessageContent::new();
        for message in messages {
            for chunk in message.content().chunks() {
                if matches!(chunk, ContentChunk::Media(_)) {
                    content.push(chunk.clone());
                }
            }
        }
        let new_text = get_new_tokens(messages, &mut session, self);
        async move {
            let new_text = new_text?;
            let model_response = Arc::new(RwLock::new(String::new()));
            let on_token = {
                let model_response = model_response.clone();
                move |token: String| {
                    let mut model_response = model_response.write().unwrap();
                    *model_response += &token;
                    on_token(token)
                }
            };
            content.push(new_text);
            let result = self
                .stream_text_with_callback_and_parser(
                    &mut session.session,
                    content,
                    sampler,
                    constraints,
                    on_token,
                )
                .await?;
            session.history.push(ChatMessage::new(
                MessageType::ModelAnswer,
                model_response.read().unwrap().clone(),
            ));
            Ok((session, result))
        }
    }
}

/// A Llama chat session.
pub struct LlamaChatSession<F: FloatDataType + SimdElement = f32> {
    history: Vec<ChatMessage>,
    session: LlamaSession<F>,
}

impl<F: FloatDataType + SimdElement> Clone for LlamaChatSession<F> {
    fn clone(&self) -> Self {
        Self {
            history: self.history.clone(),
            session: self.session.clone(),
        }
    }
}

impl<F: FloatDataType + SimdElement> ChatSession for LlamaChatSession<F> {
    type Error = LlamaSessionLoadingError;

    fn history(&self) -> Vec<ChatMessage> {
        self.history.clone()
    }

    fn try_clone(&self) -> Result<Self, Self::Error>
    where
        Self: std::marker::Sized,
    {
        Ok(self.clone())
    }
}

impl<F: FloatDataType + SimdElement> LlamaChatSession<F> {
    #[allow(clippy::too_many_arguments)]
    /// Creates a new chat history.
    fn new(session: LlamaSession<F>) -> Self {
        Self {
            history: Vec::new(),
            session,
        }
    }
}
