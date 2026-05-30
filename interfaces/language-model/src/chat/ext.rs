use std::error::Error;

#[cfg(not(target_arch = "wasm32"))]
use super::BoxedChatModel;
#[cfg(not(target_arch = "wasm32"))]
use super::BoxedStructuredChatModel;
use super::Chat;
use super::ChatModel;
use super::ChatSession;
use super::CreateChatSession;
use super::CreateDefaultChatConstraintsForType;
use super::StructuredChatModel;
use super::Task;

/// An extension trait for chat models with helpers for handling chat sessions. This trait is implemented automatically for all [`crate::ChatModel`]s.
pub trait ChatModelExt: CreateChatSession {
    /// Create a new chat session with the model.
    #[doc = include_str!("../../docs/chat.md")]
    fn chat(&self) -> Chat<Self>
    where
        Self: Clone,
    {
        Chat::new(self.clone())
    }

    /// Create a new task with the model.
    #[doc = include_str!("../../docs/task.md")]
    fn task(&self, description: impl ToString) -> Task<Self>
    where
        Self: Clone,
    {
        Task::new(self.clone(), description)
    }

    /// Erase the type of the chat model. This can be used to make multiple implementations of
    /// [`ChatModel`] compatible with the same type.
    ///
    /// # Example
    ///
    /// ```rust, no_run
    /// # #![allow(unused)]
    /// # use kalosm::language::*;
    /// #
    /// # #[tokio::main]
    /// # async fn main() {
    /// let model = loop {
    ///     let input = prompt_input("Choose Model (gpt, claude, llama, or phi): ").unwrap();
    ///     match input.to_lowercase().as_str() {
    ///         "gpt" => {
    ///             break OpenAICompatibleChatModel::builder()
    ///                 .with_gpt_4o_mini()
    ///                 .build()
    ///                 .boxed_chat_model()
    ///         }
    ///         "claude" => {
    ///             break AnthropicCompatibleChatModel::builder()
    ///                 .with_claude_3_5_haiku()
    ///                 .build()
    ///                 .boxed_chat_model()
    ///         }
    ///         "llama" => {
    ///             break Llama::builder()
    ///                 .with_source(LlamaSource::llama_3_1_8b_chat())
    ///                 .build()
    ///                 .await
    ///                 .unwrap()
    ///                 .boxed_chat_model()
    ///         }
    ///         "phi" => {
    ///             break Llama::builder()
    ///                 .with_source(LlamaSource::phi_3_5_mini_4k_instruct())
    ///                 .build()
    ///                 .await
    ///                 .unwrap()
    ///                 .boxed_chat_model()
    ///         }
    ///         _ => {}
    ///     }
    /// };
    ///
    /// let mut chat = model
    ///     .chat()
    ///     .with_system_prompt("The assistant will act like a pirate");
    ///
    /// // Then chat with the session
    /// loop {
    ///     chat(&prompt_input("\n> ").unwrap())
    ///         .to_std_out()
    ///         .await
    ///         .unwrap();
    /// }
    /// # }
    /// ```
    #[cfg(not(target_arch = "wasm32"))]
    fn boxed_chat_model(self) -> BoxedChatModel
    where
        Self: ChatModel<
                Error: Send + Sync + std::error::Error + 'static,
                ChatSession: ChatSession<Error: std::error::Error + Send + Sync + 'static>
                                 + Clone
                                 + Send
                                 + Sync
                                 + 'static,
            > + Sized
            + Send
            + Sync
            + 'static,
    {
        BoxedChatModel::new(self)
    }

    /// Erase the type of the structured chat model. This can be used to make multiple implementations of
    /// [`StructuredChatModel`] compatible with the same type.
    ///
    /// # Example
    ///
    /// ```rust, no_run
    /// # #![allow(unused)]
    /// # use kalosm::language::*;
    /// # use serde::Deserialize;
    /// # #[tokio::main]
    /// # async fn main() {
    /// // You can derive an efficient parser for your struct with the `Parse` trait
    /// // OpenAI doesn't support root anyof schemas, so we need to wrap the constraints in a struct
    /// #[derive(Parse, Clone, Schema, Deserialize, Debug)]
    /// struct Response {
    ///     action: Action,
    /// }
    ///
    /// #[derive(Parse, Clone, Schema, Deserialize, Debug)]
    /// #[serde(tag = "type")]
    /// #[serde(content = "data")]
    /// pub enum Action {
    ///     Do(String),
    ///     Say(String),
    /// }
    ///
    /// let model: BoxedStructuredChatModel<Response> = loop {
    ///     let input = prompt_input("Choose Model (gpt, llama, or phi): ").unwrap();
    ///     match input.to_lowercase().as_str() {
    ///         "gpt" => {
    ///             break OpenAICompatibleChatModel::builder()
    ///                 .with_gpt_4o_mini()
    ///                 .build()
    ///                 .boxed_typed_chat_model()
    ///         }
    ///         "llama" => {
    ///             break Llama::builder()
    ///                 .with_source(LlamaSource::llama_3_1_8b_chat())
    ///                 .build()
    ///                 .await
    ///                 .unwrap()
    ///                 .boxed_typed_chat_model()
    ///         }
    ///         "phi" => {
    ///             break Llama::builder()
    ///                 .with_source(LlamaSource::phi_3_5_mini_4k_instruct())
    ///                 .build()
    ///                 .await
    ///                 .unwrap()
    ///                 .boxed_typed_chat_model()
    ///         }
    ///         _ => {}
    ///     }
    /// };
    ///
    /// let mut chat = model
    ///     .chat()
    ///     .with_system_prompt("The assistant will act like a pirate. You will respond with either something you do or something you say. Respond with JSON in the format { \"type\": \"Say\", \"data\": \"hello\" } or { \"type\": \"Do\", \"data\": \"run away\" }");
    ///
    /// // Then chat with the session
    /// loop {
    ///     let mut response = chat(&prompt_input("\n> ").unwrap()).typed::<Response>();
    ///     response.to_std_out().await.unwrap();
    ///     println!("{:?}", response.await);
    /// }
    /// # }
    /// ```
    #[cfg(not(target_arch = "wasm32"))]
    fn boxed_typed_chat_model<T>(self) -> BoxedStructuredChatModel<T>
    where
        Self: StructuredChatModel<
                Self::DefaultConstraints,
                Error: Send + Sync + Error + 'static,
                ChatSession: ChatSession<Error: Error + Send + Sync + 'static>
                                 + Clone
                                 + Send
                                 + Sync
                                 + 'static,
            > + CreateDefaultChatConstraintsForType<T>
            + Sized
            + Send
            + Sync
            + 'static,
        T: 'static,
    {
        BoxedStructuredChatModel::new(self)
    }
}

impl<M: CreateChatSession> ChatModelExt for M {}
