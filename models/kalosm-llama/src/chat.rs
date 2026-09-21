use std::{
    future::Future,
    sync::{Arc, RwLock},
};

use crate::{model::LlamaModelError, session::LlamaSessionLoadingError, Llama, LlamaSession};
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

fn get_new_tokens(
    messages: &[ChatMessage],
    session: &mut LlamaChatSession,
    model: &Llama,
) -> Result<String, LlamaModelError> {
    let chat_template = model
        .config
        .chat_template
        .as_ref()
        .ok_or(LlamaModelError::NoChatTemplate)?;
    let bos_token = &model.config.start_token_string;
    let eos_token = &model.config.stop_token_string;
    let has_eos = {
        let cache = session
            .session
            .cache
            .read()
            .map_err(|err| LlamaModelError::Session(err.to_string()))?;
        cache.pending_token.or_else(|| cache.tokens.last().copied())
            == Some(model.config.stop_token)
    };
    format_new_tokens(
        messages,
        &mut session.history,
        chat_template,
        bos_token,
        eos_token,
        has_eos,
    )
}

fn format_new_tokens(
    messages: &[ChatMessage],
    history: &mut Vec<ChatMessage>,
    chat_template: &crate::chat_template::HuggingFaceChatTemplate,
    bos_token: &str,
    eos_token: &str,
    has_eos: bool,
) -> Result<String, LlamaModelError> {
    let current_text = if history.is_empty() {
        String::new()
    } else {
        let old_formatted_text = chat_template.format(bos_token, eos_token, history, true)?;
        // Leave the turn terminator in the new suffix unless it is already cached.
        let (before_last_eos, _) = old_formatted_text
            .rsplit_once(eos_token)
            .unwrap_or((&old_formatted_text, ""));
        if has_eos {
            before_last_eos.to_string() + eos_token
        } else {
            before_last_eos.to_string()
        }
    };
    history.extend_from_slice(messages);
    let updated_text = chat_template.format(bos_token, eos_token, history, true)?;
    let new_text = updated_text.strip_prefix(&current_text).ok_or_else(|| {
        LlamaModelError::ChatTemplateError(minijinja::Error::new(
            ErrorKind::InvalidOperation,
            format!("Chat template should only add text to the end of the current text. Old text: {current_text}, new text: {updated_text}"),
        ))
    })?;

    Ok(new_text.to_string())
}

impl CreateChatSession for Llama {
    type Error = LlamaModelError;
    type ChatSession = LlamaChatSession;

    fn new_chat_session(&self) -> Result<Self::ChatSession, Self::Error> {
        Ok(LlamaChatSession::new(self.new_session()?))
    }
}

impl ChatModel<GenerationParameters> for Llama {
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
impl<Constraints> kalosm_language_model::StructuredChatModel<Constraints, GenerationParameters>
    for Llama
where
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
pub struct LlamaChatSession {
    history: Vec<ChatMessage>,
    session: LlamaSession,
}

impl Clone for LlamaChatSession {
    fn clone(&self) -> Self {
        Self {
            history: self.history.clone(),
            session: self.session.clone(),
        }
    }
}

impl ChatSession for LlamaChatSession {
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

impl LlamaChatSession {
    #[allow(clippy::too_many_arguments)]
    /// Creates a new chat history.
    fn new(session: LlamaSession) -> Self {
        Self {
            history: Vec::new(),
            session,
        }
    }
}

#[test]
fn successive_chat_prompts_preserve_the_completed_turn() {
    let template = crate::chat_template::HuggingFaceChatTemplate::create(
        "{{ bos_token }}{% for message in messages %}{{ message['role'] }}:{{ message['content'] }}{{ eos_token }}{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}",
    )
    .unwrap();
    for (answer, eos_was_sampled) in [("answer", true), ("answer", false), ("", false)] {
        let mut history = Vec::new();
        let first = format_new_tokens(
            &[ChatMessage::new(MessageType::UserMessage, "first")],
            &mut history,
            &template,
            "<s>",
            "</s>",
            false,
        )
        .unwrap();
        history.push(ChatMessage::new(MessageType::ModelAnswer, answer));
        let next = format_new_tokens(
            &[ChatMessage::new(MessageType::UserMessage, "next")],
            &mut history,
            &template,
            "<s>",
            "</s>",
            eos_was_sampled,
        )
        .unwrap();
        let sampled_end = if eos_was_sampled { "</s>" } else { "" };
        assert_eq!(
            first + answer + sampled_end + &next,
            template.format("<s>", "</s>", &history, true).unwrap(),
            "sampled EOS: {eos_was_sampled}",
        );
    }
}
