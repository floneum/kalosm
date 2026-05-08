use serde::{Deserialize, Serialize};

use crate::{ChatMessage, ContentChunk, MessageType};

#[derive(Clone, Default, Serialize, Deserialize)]
pub(crate) struct ProviderChatSession {
    messages: Vec<ChatMessage>,
}

impl ProviderChatSession {
    pub(crate) fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }

    pub(crate) fn history(&self) -> Vec<ChatMessage> {
        self.messages.clone()
    }

    pub(crate) fn push_model_answer(&mut self, text: String) {
        self.messages
            .push(ChatMessage::new(MessageType::ModelAnswer, text));
    }
}

#[derive(Clone, Copy)]
pub(crate) enum ProviderMessageFormat {
    OpenAI,
    Anthropic,
}

pub(crate) fn extract_system_prompt(
    messages: &[ChatMessage],
) -> (Option<String>, Vec<&ChatMessage>) {
    let mut system_prompt = None;
    let filtered = messages
        .iter()
        .filter(|message| {
            if let MessageType::SystemPrompt = message.role() {
                system_prompt = message.content().as_str().map(ToString::to_string);
                false
            } else {
                true
            }
        })
        .collect();
    (system_prompt, filtered)
}

pub(crate) fn format_provider_messages<'a>(
    messages: impl IntoIterator<Item = &'a ChatMessage>,
    format: ProviderMessageFormat,
) -> serde_json::Value {
    messages
        .into_iter()
        .map(|message| {
            let content = message.content();
            let content: serde_json::Value = if let Some(string) = content.as_str() {
                string.into()
            } else {
                content
                    .chunks()
                    .iter()
                    .map(|chunk| format_content_chunk(chunk, format))
                    .collect::<Vec<_>>()
                    .into()
            };

            serde_json::json!({
                "role": message.role(),
                "content": content,
            })
        })
        .collect::<Vec<_>>()
        .into()
}

fn format_content_chunk(chunk: &ContentChunk, format: ProviderMessageFormat) -> serde_json::Value {
    match chunk {
        ContentChunk::Text(text) => {
            serde_json::json!({
                "type": "text",
                "text": text
            })
        }
        ContentChunk::Media(image) => match format {
            ProviderMessageFormat::OpenAI => {
                serde_json::json!({
                    "type": "image_url",
                    "image_url": {
                        "url": image.as_url()
                    }
                })
            }
            ProviderMessageFormat::Anthropic => {
                serde_json::json!({
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": image.as_url(),
                    }
                })
            }
        },
    }
}
