use super::{AnthropicCompatibleClient, NoAnthropicAPIKeyError};
use crate::{
    ChatMessage, ChatModel, ChatSession, ContentChunk, CreateChatSession,
    CreateDefaultChatConstraintsForType, GenerationParameters, SchemaParser, StructuredChatModel,
};
use futures_util::StreamExt;
use kalosm_model_types::{ModelBuilder, ModelLoadingProgress};
use kalosm_sample::Schema;
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{future::Future, sync::Arc};
use thiserror::Error;

#[derive(Debug)]
struct AnthropicCompatibleChatModelInner {
    model: String,
    max_tokens: u32,
    client: AnthropicCompatibleClient,
}

/// An chat model that uses Anthropic's API for the a remote chat model.
#[derive(Debug, Clone)]
pub struct AnthropicCompatibleChatModel {
    inner: Arc<AnthropicCompatibleChatModelInner>,
}

impl AnthropicCompatibleChatModel {
    /// Create a new builder for the Anthropic compatible chat model.
    pub fn builder() -> AnthropicCompatibleChatModelBuilder<false> {
        AnthropicCompatibleChatModelBuilder::new()
    }
}

/// A builder for an Anthropic compatible chat model.
#[derive(Debug, Default)]
pub struct AnthropicCompatibleChatModelBuilder<const WITH_NAME: bool> {
    model: Option<String>,
    max_tokens: u32,
    client: AnthropicCompatibleClient,
}

impl AnthropicCompatibleChatModelBuilder<false> {
    /// Creates a new builder
    pub fn new() -> Self {
        Self {
            model: None,
            max_tokens: 8192,
            client: Default::default(),
        }
    }
}

impl<const WITH_NAME: bool> AnthropicCompatibleChatModelBuilder<WITH_NAME> {
    /// Set the name of the model to use.
    pub fn with_model(self, model: impl ToString) -> AnthropicCompatibleChatModelBuilder<true> {
        AnthropicCompatibleChatModelBuilder {
            model: Some(model.to_string()),
            max_tokens: self.max_tokens,
            client: self.client,
        }
    }

    /// Set the default max tokens to use when generating text.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set the model to `claude-3-5-sonnet-20241022`
    pub fn with_claude_3_5_sonnet(self) -> AnthropicCompatibleChatModelBuilder<true> {
        self.with_model("claude-3-5-sonnet-20241022")
    }

    /// Set the model to `claude-3-5-haiku-20241022`
    pub fn with_claude_3_5_haiku(self) -> AnthropicCompatibleChatModelBuilder<true> {
        self.with_model("claude-3-5-haiku-20241022")
    }

    /// Set the model to `claude-3-opus-20240229`
    pub fn with_claude_3_opus(self) -> AnthropicCompatibleChatModelBuilder<true> {
        self.with_model("claude-3-opus-20240229")
            .with_max_tokens(4096)
    }

    /// Set the model to `claude-3-sonnet-20240229`
    pub fn with_claude_3_sonnet(self) -> AnthropicCompatibleChatModelBuilder<true> {
        self.with_model("claude-3-sonnet-20240229")
            .with_max_tokens(4096)
    }

    /// Set the model to `claude-3-haiku-20240307`
    pub fn with_claude_3_haiku(self) -> AnthropicCompatibleChatModelBuilder<true> {
        self.with_model("claude-3-haiku-20240307")
            .with_max_tokens(4096)
    }

    /// Set the model to `claude-opus-4-6`
    pub fn with_claude_opus_4_6(self) -> AnthropicCompatibleChatModelBuilder<true> {
        self.with_model("claude-opus-4-6").with_max_tokens(128_000)
    }

    /// Set the model to `claude-sonnet-4-6`
    pub fn with_claude_sonnet_4_6(self) -> AnthropicCompatibleChatModelBuilder<true> {
        self.with_model("claude-sonnet-4-6").with_max_tokens(64_000)
    }

    /// Set the model to `claude-haiku-4-5-20251001`
    pub fn with_claude_haiku_4_5(self) -> AnthropicCompatibleChatModelBuilder<true> {
        self.with_model("claude-haiku-4-5-20251001")
            .with_max_tokens(64_000)
    }

    /// Set the client used to make requests to the Anthropic API.
    pub fn with_client(mut self, client: AnthropicCompatibleClient) -> Self {
        self.client = client;
        self
    }
}

impl AnthropicCompatibleChatModelBuilder<true> {
    /// Build the model.
    pub fn build(self) -> AnthropicCompatibleChatModel {
        AnthropicCompatibleChatModel {
            inner: Arc::new(AnthropicCompatibleChatModelInner {
                model: self.model.unwrap(),
                max_tokens: self.max_tokens,
                client: self.client,
            }),
        }
    }
}

impl ModelBuilder for AnthropicCompatibleChatModelBuilder<true> {
    type Model = AnthropicCompatibleChatModel;
    type Error = std::convert::Infallible;

    async fn start_with_loading_handler(
        self,
        _: impl FnMut(ModelLoadingProgress) + Send + Sync + 'static,
    ) -> Result<Self::Model, Self::Error> {
        Ok(self.build())
    }

    fn requires_download(&self) -> bool {
        false
    }
}

/// An error that can occur when running a [`AnthropicCompatibleChatModel`].
#[derive(Error, Debug)]
pub enum AnthropicCompatibleChatModelError {
    /// An error occurred while resolving the API key.
    #[error("Error resolving API key: {0}")]
    APIKeyError(#[from] NoAnthropicAPIKeyError),
    /// An error occurred while making a request to the Anthropic API.
    #[error("Error making request: {0}")]
    ReqwestError(#[from] reqwest::Error),
    /// An error occurred while receiving server side events from the Anthropic API.
    #[error("Error receiving server side events: {0}")]
    EventSourceError(Box<reqwest_eventsource::Error>),
    /// Failed to deserialize Anthropic API response.
    #[error("Failed to deserialize Anthropic API response: {0}")]
    DeserializeError(#[from] serde_json::Error),
    /// An error occurred while streaming the response from the Anthropic API.
    #[error("Error streaming response from Anthropic API: {0}")]
    StreamError(#[from] AnthropicCompatibleChatResponseError),
}

impl From<reqwest_eventsource::Error> for AnthropicCompatibleChatModelError {
    fn from(e: reqwest_eventsource::Error) -> Self {
        Self::EventSourceError(Box::new(e))
    }
}

/// A chat session for the Anthropic compatible chat model.
#[derive(Serialize, Deserialize, Clone)]
pub struct AnthropicCompatibleChatSession {
    messages: Vec<crate::ChatMessage>,
}

impl AnthropicCompatibleChatSession {
    fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }
}

impl ChatSession for AnthropicCompatibleChatSession {
    type Error = serde_json::Error;

    fn history(&self) -> Vec<crate::ChatMessage> {
        self.messages.clone()
    }

    fn try_clone(&self) -> Result<Self, Self::Error>
    where
        Self: std::marker::Sized,
    {
        Ok(self.clone())
    }
}

impl CreateChatSession for AnthropicCompatibleChatModel {
    type ChatSession = AnthropicCompatibleChatSession;
    type Error = AnthropicCompatibleChatModelError;

    fn new_chat_session(&self) -> Result<Self::ChatSession, Self::Error> {
        Ok(AnthropicCompatibleChatSession::new())
    }
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
enum AnthropicCompatibleChatResponse {
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta(AnthropicCompatibleChatResponseContentBlockDelta),
    #[serde(rename = "content_block_stop")]
    ContentBlockStop,
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(rename = "error")]
    Error(AnthropicCompatibleChatResponseError),
    #[serde(other)]
    Unknown,
}

/// An error that can occur when receiving a stream from the Anthropic API.
#[derive(Serialize, Deserialize, Error, Debug)]
#[serde(tag = "type")]
#[non_exhaustive]
pub enum AnthropicCompatibleChatResponseError {
    /// The request was invalid.
    #[serde(rename = "invalid_request_error")]
    #[error("Invalid request error: {message}")]
    InvalidRequestError {
        /// The error message.
        message: String,
    },
    /// Authentication failed.
    #[serde(rename = "authentication_error")]
    #[error("Authentication error: {message}")]
    AuthenticationError {
        /// The error message.
        message: String,
    },
    /// An permission error occurred.
    #[serde(rename = "permission_error")]
    #[error("Permission error: {message}")]
    PermissionError {
        /// The error message.
        message: String,
    },
    /// The resource was not found.
    #[serde(rename = "not_found_error")]
    #[error("Not found error: {message}")]
    NotFoundError {
        /// The error message.
        message: String,
    },
    /// The request was too large.
    #[serde(rename = "request_too_large")]
    #[error("Request too large: {message}")]
    RequestTooLarge {
        /// The error message.
        message: String,
    },
    /// The rate limit was exceeded.
    #[serde(rename = "rate_limit_error")]
    #[error("Rate limit error: {message}")]
    RateLimitError {
        /// The error message.
        message: String,
    },
    /// An API error occurred.
    #[serde(rename = "api_error")]
    #[error("API error: {message}")]
    ApiError {
        /// The error message.
        message: String,
    },
    /// The server is overloaded.
    #[serde(rename = "overloaded_error")]
    #[error("Overloaded error: {message}")]
    OverloadedError {
        /// The error message.
        message: String,
    },
    /// An unknown error occurred.
    #[serde(other)]
    #[error("Unknown error")]
    Unknown,
}

#[derive(Serialize, Deserialize)]
struct AnthropicCompatibleChatResponseContentBlockDelta {
    index: u32,
    delta: AnthropicCompatibleChatResponseContentBlockDeltaMessage,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
enum AnthropicCompatibleChatResponseContentBlockDeltaMessage {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
    #[serde(other)]
    Unknown,
}

impl ChatModel<GenerationParameters> for AnthropicCompatibleChatModel {
    fn add_messages_with_callback<'a>(
        &'a self,
        session: &'a mut Self::ChatSession,
        messages: &[ChatMessage],
        sampler: GenerationParameters,
        mut on_token: impl FnMut(String) -> Result<(), Self::Error> + Send + Sync + 'static,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send + 'a {
        let (system_prompt, filtered) = extract_system_prompt(messages);
        let messages = format_messages(&filtered);
        let myself = &*self.inner;
        let json = build_anthropic_request_json(
            &myself.model,
            messages,
            &sampler,
            myself.max_tokens,
            system_prompt,
        );

        async move {
            let api_key = myself.client.resolve_api_key()?;
            let mut event_source = myself
                .client
                .reqwest_client
                .post(format!("{}/messages", myself.client.base_url()))
                .header("Content-Type", "application/json")
                .header("x-api-key", api_key)
                .header("anthropic-version", myself.client.version())
                .json(&json)
                .eventsource()
                .unwrap();

            let new_message_text =
                consume_anthropic_stream(&mut event_source, &mut on_token).await?;

            let new_message =
                crate::ChatMessage::new(crate::MessageType::ModelAnswer, new_message_text);

            session.messages.push(new_message);

            Ok(())
        }
    }
}

impl<T: Schema + DeserializeOwned> CreateDefaultChatConstraintsForType<T>
    for AnthropicCompatibleChatModel
{
    type DefaultConstraints = SchemaParser<T>;

    fn create_default_constraints() -> Self::DefaultConstraints {
        SchemaParser::new()
    }
}

impl<P> StructuredChatModel<SchemaParser<P>> for AnthropicCompatibleChatModel
where
    P: Schema + DeserializeOwned,
{
    fn add_message_with_callback_and_constraints<'a>(
        &'a self,
        session: &'a mut Self::ChatSession,
        messages: &[ChatMessage],
        sampler: GenerationParameters,
        _: SchemaParser<P>,
        mut on_token: impl FnMut(String) -> Result<(), Self::Error> + Send + Sync + 'static,
    ) -> impl Future<Output = Result<P, Self::Error>> + Send + 'a {
        let schema = P::schema();
        let mut schema: serde_json::Result<serde_json::Value> =
            serde_json::from_str(&schema.to_string());
        if let Ok(schema) = &mut schema {
            crate::remove_unsupported_schema_properties(schema);
        }

        let (system_prompt, filtered) = extract_system_prompt(messages);
        let messages = format_messages(&filtered);
        let myself = &*self.inner;

        let json = schema.map(|schema| {
            let mut json = build_anthropic_request_json(
                &myself.model,
                messages,
                &sampler,
                myself.max_tokens,
                system_prompt,
            );
            json["output_config"] = serde_json::json!({
                "format": {
                    "type": "json_schema",
                    "schema": schema
                }
            });
            json
        });

        async move {
            let json = json?;
            let api_key = myself.client.resolve_api_key()?;
            let mut event_source = myself
                .client
                .reqwest_client
                .post(format!("{}/messages", myself.client.base_url()))
                .header("Content-Type", "application/json")
                .header("x-api-key", &api_key)
                .header("anthropic-version", myself.client.version())
                .json(&json)
                .eventsource()
                .unwrap();

            let new_message_text =
                consume_anthropic_stream(&mut event_source, &mut on_token).await?;

            let result = serde_json::from_str::<P>(&new_message_text).map_err(|err| {
                tracing::error!(
                    "Failed to parse structured response: {new_message_text:?}\nerror: {err:?}"
                );
                AnthropicCompatibleChatModelError::DeserializeError(err)
            })?;

            let new_message =
                crate::ChatMessage::new(crate::MessageType::ModelAnswer, new_message_text);
            session.messages.push(new_message);

            Ok(result)
        }
    }
}

fn build_anthropic_request_json(
    model: &str,
    messages: serde_json::Value,
    sampler: &GenerationParameters,
    max_tokens: u32,
    system_prompt: Option<String>,
) -> serde_json::Value {
    let mut json = serde_json::json!({
        "model": model,
        "messages": messages,
        "stream": true,
        "temperature": sampler.temperature,
        "max_tokens": sampler.max_length.min(max_tokens),
    });
    if let Some(top_p) = sampler.top_p {
        json["top_p"] = top_p.into();
    }
    if let Some(top_k) = sampler.top_k {
        json["top_k"] = top_k.into();
    }
    if let Some(stop_on) = sampler.stop_on.as_ref() {
        json["stop"] = vec![stop_on.clone()].into();
    }
    if let Some(system) = system_prompt {
        json["system"] = system.into();
    }
    json
}

async fn consume_anthropic_stream(
    event_source: &mut reqwest_eventsource::EventSource,
    on_token: &mut impl FnMut(String) -> Result<(), AnthropicCompatibleChatModelError>,
) -> Result<String, AnthropicCompatibleChatModelError> {
    let mut new_message_text = String::new();

    while let Some(event) = event_source.next().await {
        match event? {
            Event::Open => {}
            Event::Message(message) => {
                let data = serde_json::from_str::<AnthropicCompatibleChatResponse>(&message.data)?;
                match data {
                    AnthropicCompatibleChatResponse::ContentBlockDelta(delta_block) => {
                        match delta_block.delta {
                            AnthropicCompatibleChatResponseContentBlockDeltaMessage::TextDelta {
                                text,
                            } => {
                                new_message_text += &text;
                                on_token(text)?;
                            }
                            AnthropicCompatibleChatResponseContentBlockDeltaMessage::InputJsonDelta {
                                partial_json,
                            } => {
                                new_message_text += &partial_json;
                                on_token(partial_json)?;
                            }
                            AnthropicCompatibleChatResponseContentBlockDeltaMessage::Unknown => {
                                tracing::trace!(
                                    "Unknown delta from Anthropic API: {:?}",
                                    message.data
                                );
                            }
                        }
                    }
                    AnthropicCompatibleChatResponse::ContentBlockStop => {}
                    AnthropicCompatibleChatResponse::MessageStop => {
                        break;
                    }
                    AnthropicCompatibleChatResponse::Error(err) => {
                        return Err(AnthropicCompatibleChatModelError::StreamError(err));
                    }
                    AnthropicCompatibleChatResponse::Unknown => {
                        tracing::trace!(
                            "Unknown response from Anthropic API: {:?}",
                            message.data
                        );
                    }
                }
            }
        }
    }

    Ok(new_message_text)
}

fn extract_system_prompt(messages: &[ChatMessage]) -> (Option<String>, Vec<&ChatMessage>) {
    let mut system_prompt = None;
    let filtered: Vec<_> = messages
        .iter()
        .filter(|message| {
            if let crate::MessageType::SystemPrompt = message.role() {
                system_prompt = message.content().as_str().map(ToString::to_string);
                false
            } else {
                true
            }
        })
        .collect();
    (system_prompt, filtered)
}

fn format_messages(messages: &[&crate::ChatMessage]) -> serde_json::Value {
    messages
        .iter()
        .map(|m| {
            let content = m.content();
            let content: serde_json::Value = if let Some(string) = content.as_str() {
                string.into()
            } else {
                content
                    .chunks()
                    .iter()
                    .map(|chunk| match chunk {
                        ContentChunk::Text(text) => {
                            serde_json::json!({
                                "type": "text",
                                "text": text
                            })
                        }
                        ContentChunk::Media(image) => {
                            serde_json::json!({
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": image.as_url(),
                                }
                            })
                        }
                    })
                    .collect::<Vec<_>>()
                    .into()
            };

            serde_json::json!({
                "role": m.role(),
                "content": content,
            })
        })
        .collect::<Vec<_>>()
        .into()
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, RwLock};

    use serde::Deserialize;

    use super::{
        AnthropicCompatibleChatModelBuilder, AnthropicCompatibleChatResponse,
        AnthropicCompatibleChatResponseContentBlockDeltaMessage, ChatModel, CreateChatSession,
        GenerationParameters, StructuredChatModel,
    };

    #[tokio::test]
    async fn test_claude_4_6_haiku() {
        let model = AnthropicCompatibleChatModelBuilder::new()
            .with_claude_haiku_4_5()
            .build();

        let mut session = model.new_chat_session().unwrap();

        let messages = vec![
            crate::ChatMessage::new(
                crate::MessageType::SystemPrompt,
                "Respond like a pirate.".to_string(),
            ),
            crate::ChatMessage::new(
                crate::MessageType::UserMessage,
                (
                    crate::MediaSource::url(
                        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    ),
                    "Describe this image".to_string(),
                ),
            ),
        ];
        let all_text = Arc::new(RwLock::new(String::new()));
        model
            .add_messages_with_callback(&mut session, &messages, GenerationParameters::default(), {
                let all_text = all_text.clone();
                move |token| {
                    let mut all_text = all_text.write().unwrap();
                    all_text.push_str(&token);
                    print!("{token}");
                    std::io::Write::flush(&mut std::io::stdout()).unwrap();
                    Ok(())
                }
            })
            .await
            .unwrap();

        let all_text = all_text.read().unwrap();
        println!("{all_text}");

        assert!(!all_text.is_empty());
    }

    #[test]
    fn test_remove_unsupported_properties() {
        let mut schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 100,
                    "pattern": "^[a-z]+$"
                },
                "age": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 150
                },
                "tags": {
                    "type": "array",
                    "items": { "type": "string", "format": "uri" },
                    "minItems": 1,
                    "maxItems": 10,
                    "uniqueItems": true
                }
            },
            "required": ["name", "age"]
        });

        crate::remove_unsupported_schema_properties(&mut schema);

        let name_props = &schema["properties"]["name"];
        assert_eq!(name_props["type"], "string");
        assert!(name_props.get("minLength").is_none());
        assert!(name_props.get("maxLength").is_none());
        assert!(name_props.get("pattern").is_none());

        let age_props = &schema["properties"]["age"];
        assert_eq!(age_props["type"], "integer");
        assert!(age_props.get("minimum").is_none());
        assert!(age_props.get("maximum").is_none());

        let tags_props = &schema["properties"]["tags"];
        assert_eq!(tags_props["type"], "array");
        assert!(tags_props.get("minItems").is_none());
        assert!(tags_props.get("maxItems").is_none());
        assert!(tags_props.get("uniqueItems").is_none());
        assert!(tags_props["items"].get("format").is_none());

        assert!(schema["required"].is_array());
    }

    #[test]
    fn test_deserialize_input_json_delta() {
        let json = r#"{"type":"input_json_delta","partial_json":"{\"name\":"}"#;
        let delta: AnthropicCompatibleChatResponseContentBlockDeltaMessage =
            serde_json::from_str(json).unwrap();
        match delta {
            AnthropicCompatibleChatResponseContentBlockDeltaMessage::InputJsonDelta {
                partial_json,
            } => {
                assert_eq!(partial_json, r#"{"name":"#);
            }
            _ => panic!("Expected InputJsonDelta variant"),
        }
    }

    #[test]
    fn test_deserialize_text_delta() {
        let json = r#"{"type":"text_delta","text":"Hello world"}"#;
        let delta: AnthropicCompatibleChatResponseContentBlockDeltaMessage =
            serde_json::from_str(json).unwrap();
        match delta {
            AnthropicCompatibleChatResponseContentBlockDeltaMessage::TextDelta { text } => {
                assert_eq!(text, "Hello world");
            }
            _ => panic!("Expected TextDelta variant"),
        }
    }

    #[test]
    fn test_deserialize_content_block_delta_event() {
        let json = r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"key\":\"val\"}"}}"#;
        let event: AnthropicCompatibleChatResponse = serde_json::from_str(json).unwrap();
        match event {
            AnthropicCompatibleChatResponse::ContentBlockDelta(block) => {
                assert_eq!(block.index, 0);
                match block.delta {
                    AnthropicCompatibleChatResponseContentBlockDeltaMessage::InputJsonDelta {
                        partial_json,
                    } => {
                        assert_eq!(partial_json, r#"{"key":"val"}"#);
                    }
                    _ => panic!("Expected InputJsonDelta"),
                }
            }
            _ => panic!("Expected ContentBlockDelta"),
        }
    }

    #[test]
    fn test_deserialize_message_stop_event() {
        let json = r#"{"type":"message_stop"}"#;
        let event: AnthropicCompatibleChatResponse = serde_json::from_str(json).unwrap();
        assert!(matches!(
            event,
            AnthropicCompatibleChatResponse::MessageStop
        ));
    }

    #[test]
    fn test_deserialize_content_block_stop_event() {
        let json = r#"{"type":"content_block_stop","index":0}"#;
        let event: AnthropicCompatibleChatResponse = serde_json::from_str(json).unwrap();
        assert!(matches!(
            event,
            AnthropicCompatibleChatResponse::ContentBlockStop
        ));
    }

    #[test]
    fn test_unknown_event_types_deserialize_as_unknown() {
        let json = r#"{"type":"message_start","message":{"id":"msg_123"}}"#;
        let event: AnthropicCompatibleChatResponse = serde_json::from_str(json).unwrap();
        assert!(matches!(event, AnthropicCompatibleChatResponse::Unknown));

        let json = r#"{"type":"ping"}"#;
        let event: AnthropicCompatibleChatResponse = serde_json::from_str(json).unwrap();
        assert!(matches!(event, AnthropicCompatibleChatResponse::Unknown));
    }

    #[test]
    fn test_text_delta_used_for_structured_response() {
        let events = vec![
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"{\"name\":\"Alice\""}}"#,
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":",\"age\":30}"}}"#,
            r#"{"type":"content_block_stop","index":0}"#,
            r#"{"type":"message_stop"}"#,
        ];

        let mut accumulated = String::new();
        let mut stopped = false;

        for event_json in &events {
            let event: AnthropicCompatibleChatResponse = serde_json::from_str(event_json).unwrap();
            if let AnthropicCompatibleChatResponse::ContentBlockDelta(block) = event {
                if let AnthropicCompatibleChatResponseContentBlockDeltaMessage::TextDelta { text } =
                    block.delta
                {
                    accumulated += &text;
                }
            } else if let AnthropicCompatibleChatResponse::MessageStop = event {
                stopped = true;
            }
        }

        assert!(stopped);
        assert!(!accumulated.is_empty());
        let parsed: serde_json::Value = serde_json::from_str(&accumulated).unwrap();
        assert_eq!(parsed["name"], "Alice");
        assert_eq!(parsed["age"], 30);
    }

    #[tokio::test]
    async fn test_claude_haiku_4_5_constrained() {
        let model = AnthropicCompatibleChatModelBuilder::new()
            .with_claude_haiku_4_5()
            .build();

        let mut session = model.new_chat_session().unwrap();

        let messages = vec![crate::ChatMessage::new(
            crate::MessageType::UserMessage,
            "Give me a list of 5 primes.".to_string(),
        )];
        let all_text = Arc::new(RwLock::new(String::new()));

        #[derive(Debug, Clone, kalosm_sample::Parse, kalosm_sample::Schema, Deserialize)]
        struct Constraints {
            primes: Vec<u8>,
        }

        let response: Constraints = model
            .add_message_with_callback_and_constraints(
                &mut session,
                &messages,
                GenerationParameters::default(),
                crate::SchemaParser::new(),
                {
                    let all_text = all_text.clone();
                    move |token| {
                        let mut all_text = all_text.write().unwrap();
                        all_text.push_str(&token);
                        print!("{token}");
                        std::io::Write::flush(&mut std::io::stdout()).unwrap();
                        Ok(())
                    }
                },
            )
            .await
            .unwrap();
        println!("{response:?}");

        let all_text = all_text.read().unwrap();
        println!("{all_text}");

        assert!(!all_text.is_empty());
        assert!(!response.primes.is_empty());
    }

    #[tokio::test]
    async fn test_claude_haiku_4_5_constrained_with_enum() {
        let model = AnthropicCompatibleChatModelBuilder::new()
            .with_claude_haiku_4_5()
            .build();

        let mut session = model.new_chat_session().unwrap();

        #[derive(Debug, Clone, kalosm_sample::Parse, kalosm_sample::Schema, Deserialize)]
        struct Constraints {
            contains_name: ContainsName,
        }

        #[derive(Debug, Clone, kalosm_sample::Parse, kalosm_sample::Schema, Deserialize)]
        enum ContainsName {
            Yes,
            No,
        }

        {
            let all_text = Arc::new(RwLock::new(String::new()));
            let messages = vec![crate::ChatMessage::new(
                crate::MessageType::UserMessage,
                "Does this sentence contain a name: Evan is one of the developers of Kalosm"
                    .to_string(),
            )];

            let response: Constraints = model
                .add_message_with_callback_and_constraints(
                    &mut session,
                    &messages,
                    GenerationParameters::default(),
                    crate::SchemaParser::new(),
                    {
                        let all_text = all_text.clone();
                        move |token| {
                            let mut all_text = all_text.write().unwrap();
                            all_text.push_str(&token);
                            print!("{token}");
                            std::io::Write::flush(&mut std::io::stdout()).unwrap();
                            Ok(())
                        }
                    },
                )
                .await
                .unwrap();
            println!("{response:?}");

            let all_text = all_text.read().unwrap();
            println!("{all_text}");

            assert!(!all_text.is_empty());
            assert!(matches!(response.contains_name, ContainsName::Yes));
        }
        {
            let all_text = Arc::new(RwLock::new(String::new()));
            let messages = vec![crate::ChatMessage::new(
                crate::MessageType::UserMessage,
                "Does this sentence contain a name: The earth is round".to_string(),
            )];

            let response: Constraints = model
                .add_message_with_callback_and_constraints(
                    &mut session,
                    &messages,
                    GenerationParameters::default(),
                    crate::SchemaParser::new(),
                    {
                        let all_text = all_text.clone();
                        move |token| {
                            let mut all_text = all_text.write().unwrap();
                            all_text.push_str(&token);
                            print!("{token}");
                            std::io::Write::flush(&mut std::io::stdout()).unwrap();
                            Ok(())
                        }
                    },
                )
                .await
                .unwrap();
            println!("{response:?}");

            let all_text = all_text.read().unwrap();
            println!("{all_text}");

            assert!(!all_text.is_empty());
            assert!(matches!(response.contains_name, ContainsName::No));
        }
    }
}
