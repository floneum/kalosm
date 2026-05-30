use crate::GenerationParameters;
use crate::ModelConstraints;
use futures_util::Future;
use kalosm_model_types::{WasmNotSend, WasmNotSendSync};
use std::fmt::Arguments;
use std::fmt::Display;

mod ext;
pub use ext::*;
mod task;
pub use task::*;
mod chat_builder;
pub use chat_builder::*;
#[cfg(not(target_arch = "wasm32"))]
mod boxed;
#[cfg(not(target_arch = "wasm32"))]
pub use boxed::*;
mod content;
pub use content::*;

/// A trait for creating a chat session. While it the core trait
/// every chat session implementation implements, most methods to use models that implement
/// this trait are implemented in the [`ChatModelExt`] trait.
///
/// # Example
/// ```rust, no_run
/// use kalosm::language::*;
///
/// #[tokio::main]
/// async fn main() {
///     // Create a new model which implements the CreateChatSession trait
///     let llm = Llama::new_chat().await.unwrap();
///     // Create a new chat for the model
///     let mut chat = llm.chat();
///     // Add a message to the chat session
///     chat(&"Hello, world!").to_std_out().await.unwrap();
/// }
/// ```
pub trait CreateChatSession {
    /// The type of error the chat session may return during operations.
    type Error: Send + Sync + 'static;

    /// The type of chat session.
    type ChatSession: ChatSession;

    /// Create a new chat session for this model.
    ///
    /// # Example
    /// ```rust, no_run
    /// use kalosm::language::*;
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     // Create a new model which implements the CreateChatSession trait
    ///     let llm = Llama::new_chat().await.unwrap();
    ///     // Create a new chat session for the model
    ///     let mut chat_session = llm.new_chat_session().unwrap();
    /// }
    /// ```
    fn new_chat_session(&self) -> Result<Self::ChatSession, Self::Error>;
}

/// A trait for unstructured chat models. This trait is required for any chat models
/// even if they do not support structured generation. While this trait is implemented for
/// all chat models, most methods to use models that implement this trait are implemented
/// in the [`ChatModelExt`] trait.
///
/// # Example
/// ```rust, no_run
/// use kalosm::language::*;
///
/// #[tokio::main]
/// async fn main() {
///     // Create a new model which implements the CreateChatSession trait
///     let llm = Llama::new_chat().await.unwrap();
///     // Create a new chat session for the model
///     let chat_session = llm.new_chat_session().unwrap();
///     // Add a message to the chat session — the updated session is returned
///     let _chat_session = llm.add_messages_with_callback(
///         chat_session,
///         &[ChatMessage::new(MessageType::UserMessage, "Hello, world!")],
///         GenerationParameters::new(),
///         |token| {
///             println!("{token}");
///             Ok(())
///         },
///     )
///     .await
///     .unwrap();
/// }
/// ```
pub trait ChatModel<Sampler = GenerationParameters>: CreateChatSession {
    /// Add messages to the chat session with a callback that is called for each token.
    ///
    /// The session is taken by value and the updated session is returned in the
    /// future's output. On error the session is consumed and not returned —
    /// callers should discard the chat.
    ///
    /// See [`Chat::add_message`] for nicer API with examples
    fn add_messages_with_callback<'a>(
        &'a self,
        session: Self::ChatSession,
        messages: &'a [ChatMessage],
        sampler: Sampler,
        on_token: impl FnMut(String) -> Result<(), Self::Error> + WasmNotSendSync + 'static,
    ) -> impl Future<Output = Result<Self::ChatSession, Self::Error>> + WasmNotSend + 'a;
}

/// A trait for unstructured chat models that support structured generation. While this trait is implemented for
/// all structured chat models, most methods to use models that implement this trait are implemented
/// in the [`ChatModelExt`] trait.
///
/// # Example
/// ```rust, no_run
/// use kalosm::language::*;
///
/// #[tokio::main]
/// async fn main() {
///     // Create a new model which implements the CreateChatSession trait
///     let llm = Llama::new_chat().await.unwrap();
///     // Create a new chat session for the model
///     let chat_session = llm.new_chat_session().unwrap();
///     // Create a parser for your data. Any type that implements the `Parse` trait has the `new_parser` method
///     let parser = i32::new_parser();
///     // Add a message with the given constraints; both the updated session and the parsed result are returned
///     let (_chat_session, result): (_, i32) = llm.add_message_with_callback_and_constraints(chat_session, &[ChatMessage::new(MessageType::UserMessage, "5 + 5")], GenerationParameters::new(), parser, |token| {
///         println!("{token}");
///         Ok(())
///     }).await.unwrap();
///     println!("{result}");
/// }
/// ```
pub trait StructuredChatModel<Constraints: ModelConstraints, Sampler = GenerationParameters>:
    ChatModel<Sampler>
{
    /// Add messages to the chat session with a callback that is called for each token and a constraints the response must follow.
    ///
    /// The session is taken by value and returned alongside the parsed output.
    /// On error the session is consumed and not returned.
    ///
    /// See [`ChatResponseBuilder::with_constraints`] for nicer API with examples
    fn add_message_with_callback_and_constraints<'a>(
        &'a self,
        session: Self::ChatSession,
        messages: &'a [ChatMessage],
        sampler: Sampler,
        constraints: Constraints,
        on_token: impl FnMut(String) -> Result<(), Self::Error> + WasmNotSendSync + 'static,
    ) -> impl Future<Output = Result<(Self::ChatSession, Constraints::Output), Self::Error>>
           + WasmNotSend
           + 'a
    where
        Constraints::Output: 'a;
}

/// A trait that defines the default constraints for a type with this chat model.
pub trait CreateDefaultChatConstraintsForType<T>:
    StructuredChatModel<Self::DefaultConstraints>
{
    /// The default constraints for this type that work with this chat model.
    type DefaultConstraints: ModelConstraints<Output = T>;

    /// Create [`Self::DefaultConstraints`] which parse the type `T` for this chat model.
    fn create_default_constraints() -> Self::DefaultConstraints;
}

#[doc = include_str!("../../docs/chat_session.md")]
pub trait ChatSession {
    /// The type of error the chat session may return during operations.
    type Error: Send + Sync + 'static;

    /// # Session History
    ///
    /// Get the history of the session. The history is a list of messages that have been sent to the model.
    ///
    /// ## Example
    /// ```rust, no_run
    /// # use kalosm::language::*;
    /// # #[tokio::main]
    /// # async fn main() {
    /// let mut llm = Llama::new_chat().await.unwrap();
    /// let mut chat = llm.chat();
    /// // Add a message to the session
    /// chat(&"Hello, world!");
    /// // Get the history of the session
    /// let history = chat.session().unwrap().history();
    /// assert_eq!(history.len(), 1);
    /// assert_eq!(history[0].role(), MessageType::UserMessage);
    /// assert_eq!(history[0].content(), "Hello, world!");
    /// # }
    /// ```
    fn history(&self) -> Vec<ChatMessage>;

    /// # Cloning Sessions
    ///
    /// Not all chat models support cloning sessions, but if a model does support
    /// cloning sessions, you can clone a session using the [`ChatSession::try_clone`] method
    /// to clone a session state while retaining the original session.
    ///
    /// ```rust, no_run
    /// use kalosm::language::*;
    /// use std::io::Write;
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let mut llm = Llama::new_chat().await.unwrap();
    ///     let mut chat = llm.chat();
    ///
    ///     // Feed some text into the session
    ///     chat(&"What is the capital of France?").await.unwrap();
    ///     let mut session = chat.session().unwrap();
    ///
    ///     // Clone the session
    ///     let cloned_session = session.try_clone().unwrap();
    ///
    ///     // Feed some more text into the cloned session
    ///     let mut chat = llm.chat().with_session(cloned_session);
    ///     chat(&"What was my first question?").await.unwrap();
    /// }
    /// ```
    fn try_clone(&self) -> Result<Self, Self::Error>
    where
        Self: std::marker::Sized;
}

/// A simple helper function for prompting the user for input.
pub fn prompt_input(prompt: impl Display) -> Result<String, std::io::Error> {
    use std::io::Write;
    print!("{prompt}");
    std::io::stdout().flush()?;
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    input.pop();
    Ok(input)
}

/// The type of a chat message
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MessageType {
    /// A system prompt message. System prompts should always be the first message in a chat session.
    #[cfg_attr(feature = "serde", serde(rename = "developer"))]
    SystemPrompt,
    /// A user message.
    #[cfg_attr(feature = "serde", serde(rename = "user"))]
    UserMessage,
    /// A model answer.
    #[cfg_attr(feature = "serde", serde(rename = "assistant"))]
    ModelAnswer,
}

/// A single item in the chat history.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ChatMessage {
    role: MessageType,
    content: MessageContent,
}

impl ChatMessage {
    /// Creates a new chat history item.
    ///
    /// # Example
    /// ```rust, no_run
    /// # use kalosm::language::*;
    /// # #[tokio::main]
    /// # async fn main() {
    /// let llm = Llama::new_chat().await.unwrap();
    /// let mut chat = llm.chat();
    /// chat.add_message(ChatMessage::new(MessageType::UserMessage, "Hello, world!"));
    /// # }
    /// ```
    pub fn new(role: MessageType, contents: impl Into<MessageContent>) -> Self {
        Self {
            role,
            content: contents.into(),
        }
    }

    /// Returns the type of the chat message.
    ///
    /// # Example
    /// ```rust, no_run
    /// # use kalosm::language::*;
    /// # #[tokio::main]
    /// # async fn main() {
    /// let message = ChatMessage::new(MessageType::UserMessage, "Hello, world!");
    /// assert_eq!(message.role(), MessageType::UserMessage);
    /// # }
    /// ```
    pub fn role(&self) -> MessageType {
        self.role
    }

    /// Returns the content of the item.
    ///
    /// # Example
    /// ```rust, no_run
    /// # use kalosm::language::*;
    /// # #[tokio::main]
    /// # async fn main() {
    /// let message = ChatMessage::new(MessageType::UserMessage, "Hello, world!");
    /// assert_eq!(message.content(), "Hello, world!");
    /// # }
    /// ```
    pub fn content(&self) -> &MessageContent {
        &self.content
    }
}

/// A trait for types that can be converted into a chat message.
///
/// # Example
/// ```rust, no_run
/// # use kalosm::language::*;
/// # #[tokio::main]
/// # async fn main() {
/// // Displayable types are converted into a user chat message
/// let user_message = "Hello, world!";
/// let chat_message = user_message.into_chat_message();
/// assert_eq!(chat_message.role(), MessageType::UserMessage);
///
/// // Or you can create a chat message manually
/// let chat_message = ChatMessage::new(MessageType::ModelAnswer, "Hello, world!".to_string());
/// assert_eq!(chat_message.role(), MessageType::ModelAnswer);
/// # }
/// ```
pub trait IntoChatMessage {
    /// Convert the type into a chat message.
    fn into_chat_message(self) -> ChatMessage;
}

impl IntoChatMessage for &str {
    fn into_chat_message(self) -> ChatMessage {
        ChatMessage::new(MessageType::UserMessage, self.to_string())
    }
}

impl IntoChatMessage for String {
    fn into_chat_message(self) -> ChatMessage {
        ChatMessage::new(MessageType::UserMessage, self)
    }
}

impl IntoChatMessage for Arguments<'_> {
    fn into_chat_message(self) -> ChatMessage {
        ChatMessage::new(MessageType::UserMessage, self.to_string())
    }
}

impl IntoChatMessage for ChatMessage {
    fn into_chat_message(self) -> ChatMessage {
        self.clone()
    }
}

macro_rules! impl_to_chat_message_tuple {
    ($($name:ident),+) => {
        #[allow(non_snake_case)]
        impl<$($name: Into<ContentChunk> + Clone),+> IntoChatMessage for ($($name,)+) {
            fn into_chat_message(self) -> ChatMessage {
                let ($($name,)+) = self;
                ChatMessage::new(MessageType::UserMessage, ($($name.clone(),)+))
            }
        }
    };
}

impl_to_chat_message_tuple!(A);
impl_to_chat_message_tuple!(A, B);
impl_to_chat_message_tuple!(A, B, C);
impl_to_chat_message_tuple!(A, B, C, D);
impl_to_chat_message_tuple!(A, B, C, D, E);
impl_to_chat_message_tuple!(A, B, C, D, E, F);
impl_to_chat_message_tuple!(A, B, C, D, E, F, G);
impl_to_chat_message_tuple!(A, B, C, D, E, F, G, H);
impl_to_chat_message_tuple!(A, B, C, D, E, F, G, H, I);
impl_to_chat_message_tuple!(A, B, C, D, E, F, G, H, I, J);
impl_to_chat_message_tuple!(A, B, C, D, E, F, G, H, I, J, K);
impl_to_chat_message_tuple!(A, B, C, D, E, F, G, H, I, J, K, L);
impl_to_chat_message_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M);
impl_to_chat_message_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N);

/// A trait for any type that can be converted from a reference to a [`ChatMessage`]. This trait is implemented for any types
/// that implement [`Clone`] + [`IntoChatMessage`].
pub trait ToChatMessage {
    /// Convert the type from a reference to a [`ChatMessage`].
    fn to_chat_message(&self) -> ChatMessage;
}

impl<T: IntoChatMessage + Clone> ToChatMessage for T {
    fn to_chat_message(&self) -> ChatMessage {
        <T as Clone>::clone(self).into_chat_message()
    }
}
