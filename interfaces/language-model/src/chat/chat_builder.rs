use crate::GenerationParameters;
use crate::ModelConstraints;
use crate::NoConstraints;
use crate::ToChatMessage;
use async_lock::Mutex as AsyncMutex;
use futures_channel::mpsc::UnboundedReceiver;
use futures_channel::oneshot::Receiver;
use futures_util::Future;
use futures_util::FutureExt;
use futures_util::Stream;
use futures_util::StreamExt;
use kalosm_model_types::{WasmNotSend, WasmNotSendSync};
use std::any::Any;
use std::fmt::Debug;
use std::future::IntoFuture;
use std::mem::MaybeUninit;
use std::ops::Deref;
use std::ops::DerefMut;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::RwLock;
use std::task::Poll;

// On wasm32, futures don't need to be Send, and Box<dyn Any> doesn't need Send
#[cfg(not(target_arch = "wasm32"))]
type BoxedTaskFuture = Pin<Box<dyn Future<Output = ()> + Send>>;
#[cfg(target_arch = "wasm32")]
type BoxedTaskFuture = Pin<Box<dyn Future<Output = ()>>>;

#[cfg(not(target_arch = "wasm32"))]
type BoxedAny = Box<dyn Any + Send>;
#[cfg(target_arch = "wasm32")]
type BoxedAny = Box<dyn Any>;

#[cfg(not(target_arch = "wasm32"))]
type BoxedIntoFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
#[cfg(target_arch = "wasm32")]
type BoxedIntoFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

use super::ChatMessage;
use super::ChatModel;
use super::ChatSession;
use super::ContentChunk;
use super::CreateChatSession;
use super::CreateDefaultChatConstraintsForType;
use super::IntoChatMessage;
use super::MessageType;
use super::StructuredChatModel;

/// [`Chat`] is a chat interface that builds on top of [`crate::ChatModel`] and [`crate::StructuredChatModel`]. It makes it easy to create a chat session with streaming responses, and constraints.
#[doc = include_str!("../../docs/chat.md")]
pub struct Chat<M: CreateChatSession> {
    model: Arc<M>,
    #[allow(clippy::type_complexity)]
    session: OnceLock<Result<Arc<AsyncMutex<M::ChatSession>>, M::Error>>,
    queued_messages: Vec<ChatMessage>,
}

impl<M: CreateChatSession + Debug> Debug for Chat<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Chat")
            .field("model", &self.model)
            .field("queued_messages", &self.queued_messages)
            .finish()
    }
}

impl<M: CreateChatSession> Clone for Chat<M> {
    fn clone(&self) -> Self {
        let model = self.model.clone();
        let mut queued_messages = self.queued_messages.clone();
        let session = OnceLock::new();
        if let Some(Ok(old_session)) = self.session.get() {
            #[cfg(target_arch = "wasm32")]
            let old_session = old_session.try_lock().unwrap();
            #[cfg(not(target_arch = "wasm32"))]
            let old_session = old_session.lock_blocking();
            if let Ok(old_session) = old_session.try_clone() {
                session
                    .set(Ok(Arc::new(AsyncMutex::new(old_session))))
                    .unwrap_or_else(|_| panic!("Chat session should be empty initially"));
            } else {
                queued_messages.extend_from_slice(&old_session.history());
            }
        }

        Self {
            session,
            model,
            queued_messages,
        }
    }
}

impl<M: CreateChatSession> Chat<M> {
    /// Create a new chat session with the default settings.
    ///
    /// # Example
    /// ```rust, no_run
    /// # use kalosm::language::*;
    /// # #[tokio::main]
    /// # async fn main() {
    /// // Before you create a chat session, you need to create a model. Llama::new_chat will create a good default chat model.
    /// let model = Llama::new_chat().await.unwrap();
    /// // If you don't need to customize the chat session, you can use the `new` method to create a chat session with the default settings
    /// let mut chat = Chat::new(model);
    /// # }
    /// ```
    pub fn new(model: M) -> Chat<M> {
        Self {
            model: Arc::new(model),
            session: OnceLock::new(),
            queued_messages: Vec::new(),
        }
    }

    /// Adds a system prompt to the chat. The system prompt guides the model to respond in a certain way.
    /// If no system prompt is added, the model will use a default system prompt that instructs the model to respond in a way that is safe and respectful.
    ///
    /// # Example
    /// ```rust, no_run
    /// # use kalosm::language::*;
    /// # #[tokio::main]
    /// # async fn main() {
    /// let model = Llama::new_chat().await.unwrap();
    /// let mut chat = model
    ///     .chat()
    ///     .with_system_prompt("The assistant will act like a pirate.");
    /// # }
    /// ```
    pub fn with_system_prompt(mut self, system_prompt: impl ToString) -> Self {
        #[cfg(debug_assertions)]
        if let Some(Ok(session)) = self.session.get() {
            if let Some(session) = session.try_lock() {
                let mut existing_history = session.history();
                existing_history.extend_from_slice(&self.queued_messages);
                if !existing_history.is_empty() {
                    tracing::error!("System prompt should be the first message in the history. System prompt was added to the end of the history: {:?}", existing_history);
                }
            }
        }

        // Add the system prompt to the queue
        self.queued_messages.push(ChatMessage::new(
            MessageType::SystemPrompt,
            system_prompt.to_string(),
        ));

        self
    }

    /// Starts the chat instance with the given model session. This can be useful for resuming a chat session with a long context that has already been processed.
    ///
    /// # Example
    /// ```rust, no_run
    /// # use kalosm::language::*;
    /// # #[tokio::main]
    /// # async fn main() {
    /// let model = Llama::new_chat().await.unwrap();
    /// // Get a session from an existing chat
    /// let mut old_chat = model.chat();
    /// old_chat(&"Hello!").to_std_out().await.unwrap();
    /// let session = old_chat.session().unwrap().try_clone().unwrap();
    /// // Start a new chat session with the cloned session
    /// let mut chat = model.chat().with_session(session);
    /// # }
    /// ```
    pub fn with_session(mut self, session: M::ChatSession) -> Self {
        // Copy the history from the current chat session into the queue for the new chat session
        let mut existing_history = session.history();
        existing_history.extend_from_slice(&self.queued_messages);
        self.queued_messages = existing_history;

        // Set the new chat session
        self.session
            .set(Ok(Arc::new(AsyncMutex::new(session))))
            .unwrap_or_else(|_| panic!("Chat session already set"));

        self
    }

    /// Adds a user message to the chat session and streams the bot response.
    ///
    /// # Example
    /// ```rust, no_run
    /// # use kalosm::language::*;
    /// # #[tokio::main]
    /// # async fn main() {
    /// let model = Llama::new_chat().await.unwrap();
    /// let mut chat = model.chat();
    /// let prompt = prompt_input("\n> ").unwrap();
    ///
    /// // You can add the user message to the chat session with the `add_message` method
    /// let mut response_stream = chat.add_message(prompt);
    /// // And then stream the result to std out
    /// response_stream.to_std_out().await.unwrap();
    /// # }
    /// ```
    pub fn add_message<Msg: IntoChatMessage>(
        &mut self,
        message: Msg,
    ) -> ChatResponseBuilder<'_, M> {
        // First push the message to the queue
        self.queued_messages.push(message.into_chat_message());

        // Then create the builder that will respond to the message if it is awaited
        ChatResponseBuilder {
            chat_session: MaybeOwnedSession::Borrowed(self),
            constraints: None,
            sampler: Some(GenerationParameters::default()),
            task: OnceLock::new(),
            queued_tokens: None,
            result: None,
        }
    }

    /// Adds a user message to the chat session and streams the bot response while consuming the chat session.
    ///
    /// # Example
    /// ```rust, no_run
    /// # use kalosm::language::*;
    /// # #[tokio::main]
    /// # async fn main() {
    /// let model = Llama::new_chat().await.unwrap();
    /// let mut chat = model.chat();
    /// let prompt = prompt_input("\n> ").unwrap();
    ///
    /// // You can add the user message to the chat session with the `add_message` method
    /// let mut response_stream = chat.into_add_message(prompt);
    /// // And then stream the result to std out
    /// response_stream.to_std_out().await.unwrap();
    /// # }
    /// ```
    pub fn into_add_message<Msg: IntoChatMessage>(
        mut self,
        message: Msg,
    ) -> ChatResponseBuilder<'static, M> {
        // First push the message to the queue
        self.queued_messages.push(message.into_chat_message());

        // Then create the builder that will respond to the message if it is awaited
        ChatResponseBuilder {
            chat_session: MaybeOwnedSession::Owned(self),
            constraints: None,
            sampler: Some(GenerationParameters::default()),
            task: OnceLock::new(),
            queued_tokens: None,
            result: None,
        }
    }

    fn session_clone(&mut self) -> Result<Arc<AsyncMutex<M::ChatSession>>, M::Error> {
        let session = self.session.get_or_init(|| {
            self.model
                .new_chat_session()
                .map(|session| Arc::new(AsyncMutex::new(session)))
        });

        match session {
            Ok(session) => Ok(session.clone()),
            Err(_) => {
                let session_owned = self.session.take().unwrap();
                match session_owned {
                    Ok(_) => unreachable!(),
                    Err(session_err) => Err(session_err),
                }
            }
        }
    }

    /// Get a reference to the chat session or an error if the session failed to load.
    ///
    /// You can use the session to get the chat history:
    /// ```rust, no_run
    /// # use kalosm::language::*;
    /// # #[tokio::main]
    /// # async fn main() {
    /// let model = Llama::new_chat().await.unwrap();
    /// let mut chat = model.chat();
    /// // Add a message to the chat history
    /// chat(&"Hello, world!").to_std_out().await.unwrap();
    /// // Get the chat session
    /// let session = chat.session().unwrap();
    /// // Get the chat history
    /// let history = session.history();
    /// println!("{:?}", history);
    /// # }
    /// ```
    pub fn session(&self) -> Result<impl Deref<Target = M::ChatSession> + use<'_, M>, &M::Error> {
        match self
            .session
            .get_or_init(|| {
                self.model
                    .new_chat_session()
                    .map(|session| Arc::new(AsyncMutex::new(session)))
            })
            .as_ref()
        {
            #[cfg(target_arch = "wasm32")]
            Ok(session) => Ok(session.try_lock().unwrap()),
            #[cfg(not(target_arch = "wasm32"))]
            Ok(session) => Ok(session.lock_blocking()),
            Err(err) => Err(err),
        }
    }
}

impl<M: CreateChatSession + Clone + 'static> Deref for Chat<M> {
    type Target = dyn FnMut(&dyn ToChatMessage) -> ChatResponseBuilder<'static, M>;

    fn deref(&self) -> &Self::Target {
        // https://github.com/dtolnay/case-studies/tree/master/callable-types

        // Create an empty allocation for Self.
        let uninit_callable = MaybeUninit::<Self>::uninit();
        // Move a closure that captures just self into the uninitialized memory. Closures create an anonymous type that implement
        // FnOnce. In this case, the layout of the type should just be Self because self is the only field in the closure type.
        let uninit_closure = move |_: &dyn ToChatMessage| {
            #[allow(clippy::diverging_sub_expression)]
            let _unreachable: ChatResponseBuilder<'static, M> = unreachable!(
                "FnMut cannot be called from a reference. Called from pointer {:p}",
                uninit_callable.as_ptr()
            );
            #[allow(unreachable_code)]
            _unreachable
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

impl<M: CreateChatSession + Clone + 'static> DerefMut for Chat<M> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // https://github.com/dtolnay/case-studies/tree/master/callable-types

        // Create an empty allocation for Self.
        let mut uninit_callable = MaybeUninit::<Self>::uninit();
        // Move a closure that captures just self into the uninitialized memory. Closures create an anonymous type that implement
        // FnOnce. In this case, the layout of the type should just be Self because self is the only field in the closure type.
        let mut uninit_closure = move |message: &dyn ToChatMessage| {
            Self::add_message(
                unsafe { &mut *uninit_callable.as_mut_ptr() },
                message.to_chat_message(),
            )
        };

        // Make sure the layout of the closure and Self is the same.
        let size_of_closure = std::alloc::Layout::for_value(&uninit_closure);
        assert_eq!(size_of_closure, std::alloc::Layout::new::<Self>());

        // Then cast the lifetime of the closure to the lifetime of &self.
        fn cast_lifetime<'a, T>(_a: &mut T, b: &'a mut T) -> &'a mut T {
            b
        }
        let reference_to_closure = cast_lifetime(
            {
                // The real closure that we will never use.
                &mut uninit_closure
            },
            #[allow(clippy::missing_transmute_annotations)]
            // We transmute self into a reference to the closure. This is safe because we know that the closure has the same memory layout as Self so &Closure == &Self.
            unsafe {
                std::mem::transmute(self)
            },
        );

        // Cast the closure to a trait object.
        reference_to_closure as &mut _
    }
}

enum MaybeOwnedSession<'a, M: CreateChatSession> {
    Owned(Chat<M>),
    Borrowed(&'a mut Chat<M>),
}

impl<M: CreateChatSession> Deref for MaybeOwnedSession<'_, M> {
    type Target = Chat<M>;

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Owned(session) => session,
            Self::Borrowed(session) => session,
        }
    }
}

impl<M: CreateChatSession> DerefMut for MaybeOwnedSession<'_, M> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Owned(session) => session,
            Self::Borrowed(session) => session,
        }
    }
}

/// A builder for a chat response. This is returned by [`Chat::add_message`] and can be modified until you start awaiting the response.
///
/// # Example
/// ```rust, no_run
/// # use kalosm::language::*;
/// # #[tokio::main]
/// # async fn main() {
/// let model = Llama::new_chat().await.unwrap();
/// let mut chat = model
///     .chat()
///     .with_system_prompt("The first prime number larger than 4");
///
/// // Add a message to the chat session with the given message
/// let mut response = chat(&prompt_input("\n> ").unwrap());
/// // Before you start streaming the response, you can add constraints to the response
/// let mut response = response.with_constraints(i32::new_parser());
/// // Once you start streaming the response, the generation starts
/// response.to_std_out().await.unwrap();
/// // The response can be awaited to get the final (typed) result
/// let all_text: i32 = response.await.unwrap();
/// println!("{all_text}");
/// # }
/// ```
pub struct ChatResponseBuilder<
    'a,
    M: CreateChatSession,
    Constraints = NoConstraints,
    Sampler = GenerationParameters,
> {
    chat_session: MaybeOwnedSession<'a, M>,
    constraints: Option<Constraints>,
    sampler: Option<Sampler>,
    task: OnceLock<RwLock<BoxedTaskFuture>>,
    #[allow(clippy::type_complexity)]
    result: Option<Receiver<Result<BoxedAny, M::Error>>>,
    queued_tokens: Option<UnboundedReceiver<String>>,
}

impl<'a, M: CreateChatSession, Constraints, Sampler>
    ChatResponseBuilder<'a, M, Constraints, Sampler>
{
    /// Constrains the model's response to the given parser. This can be used to make the model start with a certain phrase, or to make the model respond in a certain way.
    ///
    /// # Example
    /// ```rust, no_run
    /// # use kalosm::language::*;
    /// # #[tokio::main]
    /// # async fn main() {
    /// let model = Llama::new_chat().await.unwrap();
    /// // Create constraints that parses Yes! and then stops on the end of the assistant's response
    /// let constraints = LiteralParser::new("Yes!").then(model.default_assistant_constraints());
    /// // Create a chat session with the model
    /// let mut chat = model.chat();
    ///
    /// // Chat with the user
    /// loop {
    ///     // Set the constraints for the response
    ///     let mut output_stream =
    ///         chat(&prompt_input("\n> ").unwrap()).with_constraints(constraints.clone());
    ///     output_stream.to_std_out().await.unwrap();
    /// }
    /// # }
    /// ```
    pub fn with_constraints<NewConstraints>(
        self,
        constraints: NewConstraints,
    ) -> ChatResponseBuilder<'a, M, NewConstraints, Sampler> {
        ChatResponseBuilder {
            chat_session: self.chat_session,
            constraints: Some(constraints),
            sampler: self.sampler,
            queued_tokens: None,
            result: None,
            task: OnceLock::new(),
        }
    }

    /// Constrains the model's response to the the default parser for the given type. This can be used to make the model return a specific type.
    ///
    /// # Example
    /// ```rust, no_run
    /// # use kalosm::language::*;
    /// # #[tokio::main]
    /// # async fn main() {
    /// #[derive(Schema, Parse, Clone, Debug)]
    /// struct Pet {
    ///     name: String,
    ///     age: u32,
    ///     description: String,
    /// }
    ///
    /// // First create a model
    /// let model = Llama::new_chat().await.unwrap();
    /// // Create a chat session
    /// let mut chat = model.chat().with_system_prompt(format!(
    ///     "The assistant turns descriptions of pets into JSON in this format {}",
    ///     Pet::schema()
    /// ));
    /// // Finally, add a message and make it typed to get the parsed response
    /// let pet: Pet = chat(&"JSON for an adorable dog named ruffles")
    ///     .typed()
    ///     .await
    ///     .unwrap();
    /// println!("{pet:?}");
    /// # }
    /// ```
    pub fn typed<T>(
        self,
    ) -> ChatResponseBuilder<
        'a,
        M,
        <M as CreateDefaultChatConstraintsForType<T>>::DefaultConstraints,
        Sampler,
    >
    where
        M: CreateDefaultChatConstraintsForType<T>,
    {
        self.with_constraints(M::create_default_constraints())
    }

    /// Sets the sampler to use for generating responses. The sampler determines how tokens are choosen from the probability distribution
    /// the model generates. They can be used to make the model more or less predictable and prevent repetition.
    ///
    /// # Example
    /// ```rust, no_run
    /// # use kalosm::language::*;
    /// # #[tokio::main]
    /// # async fn main() {
    /// let model = Llama::new_chat().await.unwrap();
    /// // Create the sampler to use for the chat session
    /// let sampler = GenerationParameters::default();
    ///
    /// // Create a chat session with the model
    /// let mut chat = model.chat();
    ///
    /// // Chat with the user
    /// loop {
    ///     // Stream a response with the sampler
    ///     let mut output_stream = chat(&prompt_input("\n> ").unwrap()).with_sampler(sampler.clone());
    ///     output_stream.to_std_out().await.unwrap();
    /// }
    /// # }
    /// ```
    pub fn with_sampler<NewSampler>(
        self,
        sampler: NewSampler,
    ) -> ChatResponseBuilder<'a, M, Constraints, NewSampler> {
        ChatResponseBuilder {
            chat_session: self.chat_session,
            constraints: self.constraints,
            sampler: Some(sampler),
            queued_tokens: None,
            result: None,
            task: OnceLock::new(),
        }
    }

    /// Add a new chunk to the current message
    ///
    /// # Example
    /// ```rust, no_run
    /// # use kalosm::language::*;
    /// # #[tokio::main]
    /// # async fn main() {
    /// let model = Llama::new_chat().await.unwrap();
    /// let mut chat = model.chat();
    /// let prompt = prompt_input("\n> ").unwrap();
    ///
    /// // You can add the user message to the chat session by calling the chat function
    /// let mut response_stream = chat(&prompt)
    ///     // Then add a new chunk to the current message
    ///     .then(" and then some more");
    /// // And then stream the result to std out
    /// response_stream.to_std_out().await.unwrap();
    /// # }
    /// ```
    pub fn then<Msg: Into<ContentChunk>>(
        self,
        message: Msg,
    ) -> ChatResponseBuilder<'a, M, Constraints, Sampler> {
        let mut myself = self;
        myself
            .chat_session
            .queued_messages
            .last_mut()
            .unwrap()
            .content
            .push(message.into());
        myself
    }
}

impl<M, Sampler> ChatResponseBuilder<'_, M, NoConstraints, Sampler>
where
    Sampler: WasmNotSend + Unpin + 'static,
    M: ChatModel<Sampler> + WasmNotSendSync + Clone + Unpin + 'static,
    M::ChatSession: WasmNotSendSync + Unpin + 'static,
{
    fn ensure_unstructured_task_started(&mut self) {
        if self.task.get().is_none() {
            let messages = std::mem::take(&mut self.chat_session.queued_messages);
            let sampler = self
                .sampler
                .take()
                .expect("ChatResponseBuilder cannot be turned into a future twice");
            let (mut tx, rx) = futures_channel::mpsc::unbounded();
            let (result_tx, result_rx) = futures_channel::oneshot::channel();
            self.queued_tokens = Some(rx);
            self.result = Some(result_rx);
            let all_text = Arc::new(Mutex::new(String::new()));
            let on_token = {
                let all_text = all_text.clone();
                move |tok: String| {
                    all_text.lock().unwrap().push_str(&tok);
                    _ = tx.start_send(tok);
                    Ok(())
                }
            };
            let session = self.chat_session.session_clone();
            let model = self.chat_session.model.clone();
            let future = async move {
                let session = session?;
                let mut session = session.lock().await;
                model
                    .add_messages_with_callback(&mut session, &messages, sampler, on_token)
                    .await?;
                let mut all_text = all_text.lock().unwrap();
                let all_text = std::mem::take(&mut *all_text);
                Ok(Box::new(all_text) as BoxedAny)
            };
            let wrapped = async move {
                let result: Result<BoxedAny, M::Error> = future.await;
                _ = result_tx.send(result);
            };
            let task = Box::pin(wrapped);
            self.task
                .set(RwLock::new(task))
                .unwrap_or_else(|_| panic!("Task already set"));
        }
    }
}

impl<M, Sampler> Stream for ChatResponseBuilder<'_, M, NoConstraints, Sampler>
where
    Sampler: WasmNotSend + Unpin + 'static,
    M: ChatModel<Sampler> + WasmNotSendSync + Clone + Unpin + 'static,
    M::ChatSession: WasmNotSendSync + Unpin + 'static,
    M::Error: WasmNotSendSync + Unpin,
{
    type Item = String;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let myself = Pin::get_mut(self);
        myself.ensure_unstructured_task_started();
        {
            if let Some(token) = &mut myself.queued_tokens {
                if let Poll::Ready(Some(token)) = token.poll_next_unpin(cx) {
                    return Poll::Ready(Some(token));
                }
            }
        }
        let mut task = myself.task.get().unwrap().write().unwrap();
        match task.poll_unpin(cx) {
            Poll::Ready(_) => {
                *task = Box::pin(async move {});
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<'a, M, Sampler> IntoFuture for ChatResponseBuilder<'a, M, NoConstraints, Sampler>
where
    Sampler: WasmNotSend + Unpin + 'static,
    M: ChatModel<Sampler> + WasmNotSendSync + Unpin + Clone + 'static,
    M::ChatSession: Clone + WasmNotSendSync + Unpin + 'static,
{
    type Output = Result<String, M::Error>;
    type IntoFuture = BoxedIntoFuture<'a, Self::Output>;

    fn into_future(mut self) -> Self::IntoFuture {
        self.ensure_unstructured_task_started();

        Box::pin(async move {
            self.task.into_inner().unwrap().into_inner().unwrap().await;
            let result = self.result.take().unwrap().await.unwrap();
            result.map(|boxed| *boxed.downcast::<String>().unwrap())
        })
    }
}

impl<M, Constraints, Sampler> ChatResponseBuilder<'_, M, Constraints, Sampler>
where
    Constraints: ModelConstraints + WasmNotSendSync + Unpin + 'static,
    Sampler: WasmNotSend + Unpin + 'static,
    M: StructuredChatModel<Constraints, Sampler> + WasmNotSendSync + Clone + Unpin + 'static,
    M::ChatSession: Clone + WasmNotSendSync + Unpin + 'static,
    Constraints::Output: WasmNotSend + 'static,
{
    fn ensure_structured_task_started(&mut self) {
        if self.task.get().is_none() {
            let messages = std::mem::take(&mut self.chat_session.queued_messages);
            let sampler = self
                .sampler
                .take()
                .expect("ChatResponseBuilder cannot be turned into a future twice");
            let constraints = self
                .constraints
                .take()
                .expect("ChatResponseBuilder cannot be turned into a future twice");
            let (mut tx, rx) = futures_channel::mpsc::unbounded();
            let (result_tx, result_rx) = futures_channel::oneshot::channel();
            self.queued_tokens = Some(rx);
            self.result = Some(result_rx);
            let on_token = move |tok: String| {
                _ = tx.start_send(tok);
                Ok(())
            };
            let session = self.chat_session.session_clone();
            let model = self.chat_session.model.clone();
            let future = async move {
                let session = session?;
                let mut session = session.lock().await;
                model
                    .add_message_with_callback_and_constraints(
                        &mut session,
                        &messages,
                        sampler,
                        constraints,
                        on_token,
                    )
                    .await
                    .map(|value| Box::new(value) as BoxedAny)
            };
            let wrapped = async move {
                let result: Result<BoxedAny, M::Error> = future.await;
                _ = result_tx.send(result);
            };
            let task = Box::pin(wrapped);
            self.task
                .set(RwLock::new(task))
                .unwrap_or_else(|_| panic!("Task already set"));
        }
    }
}

impl<M, Constraints, Sampler> Stream for ChatResponseBuilder<'_, M, Constraints, Sampler>
where
    Constraints: ModelConstraints + WasmNotSendSync + Unpin + 'static,
    Sampler: WasmNotSend + Unpin + 'static,
    M: StructuredChatModel<Constraints, Sampler> + WasmNotSendSync + Clone + Unpin + 'static,
    M::ChatSession: Clone + WasmNotSendSync + Unpin + 'static,
    M::Error: WasmNotSendSync + Unpin,
    Constraints::Output: WasmNotSend + 'static,
{
    type Item = String;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let myself = Pin::get_mut(self);
        myself.ensure_structured_task_started();
        {
            if let Some(token) = &mut myself.queued_tokens {
                if let Poll::Ready(Some(token)) = token.poll_next_unpin(cx) {
                    return Poll::Ready(Some(token));
                }
            }
        }
        let mut task = myself.task.get().unwrap().write().unwrap();
        match task.poll_unpin(cx) {
            Poll::Ready(_) => {
                *task = Box::pin(async move {});
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<'a, M, Constraints, Sampler> IntoFuture for ChatResponseBuilder<'a, M, Constraints, Sampler>
where
    Constraints: ModelConstraints + WasmNotSendSync + Unpin + 'static,
    Sampler: WasmNotSend + Unpin + 'static,
    M: StructuredChatModel<Constraints, Sampler> + WasmNotSendSync + Clone + Unpin + 'static,
    M::ChatSession: Clone + WasmNotSendSync + Unpin + 'static,
    Constraints::Output: WasmNotSend + 'static,
{
    type Output = Result<Constraints::Output, M::Error>;
    type IntoFuture = BoxedIntoFuture<'a, Self::Output>;

    fn into_future(mut self) -> Self::IntoFuture {
        self.ensure_structured_task_started();

        Box::pin(async move {
            self.task.into_inner().unwrap().into_inner().unwrap().await;
            let result = self.result.take().unwrap().await.unwrap();
            result.map(|boxed| *boxed.downcast::<Constraints::Output>().unwrap())
        })
    }
}
