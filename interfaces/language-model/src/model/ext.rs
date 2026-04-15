use futures_channel::mpsc::UnboundedReceiver;
use futures_channel::oneshot::Receiver;
use futures_util::Future;
use futures_util::FutureExt;
use futures_util::Stream;
use futures_util::StreamExt;
use std::any::Any;
#[cfg(not(target_arch = "wasm32"))]
use std::error::Error;
use std::future::IntoFuture;
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
type BoxedIntoFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;
#[cfg(target_arch = "wasm32")]
type BoxedIntoFuture<T> = Pin<Box<dyn Future<Output = T>>>;

use crate::GenerationParameters;
use crate::MessageContent;
use crate::ModelConstraints;
use crate::NoConstraints;

#[cfg(not(target_arch = "wasm32"))]
use super::BoxedStructuredTextCompletionModel;
#[cfg(not(target_arch = "wasm32"))]
use super::BoxedTextCompletionModel;
use super::CreateDefaultCompletionConstraintsForType;
use super::CreateTextCompletionSession;
use super::StructuredTextCompletionModel;
use super::TextCompletionModel;
use super::TextCompletionSession;

#[doc = include_str!("../../docs/completion.md")]
pub trait TextCompletionModelExt: CreateTextCompletionSession {
    /// Create a new text completion builder for this model. See [`TextCompletionBuilder`] for more details.
    fn complete(&self, text: impl Into<MessageContent>) -> TextCompletionBuilder<Self>
    where
        Self: Clone,
    {
        // Then create the builder that will respond to the message if it is awaited
        TextCompletionBuilder {
            text: text.into(),
            model: Some(self.clone()),
            constraints: None,
            sampler: Some(GenerationParameters::default()),
            task: OnceLock::new(),
            queued_tokens: None,
            result: None,
        }
    }

    /// Erase the type of the text completion model. This can be used to make multiple implementations of
    /// [`TextCompletionModel`] compatible with the same type.
    #[cfg(not(target_arch = "wasm32"))]
    fn boxed_completion_model(self) -> BoxedTextCompletionModel
    where
        Self: TextCompletionModel<
                Error: Send + Sync + std::error::Error + 'static,
                Session: TextCompletionSession<Error: std::error::Error + Send + Sync + 'static>
                             + Clone
                             + Send
                             + Sync
                             + 'static,
            > + Sized
            + Send
            + Sync
            + 'static,
    {
        BoxedTextCompletionModel::new(self)
    }

    /// Erase the type of the structured text completion model. This can be used to make multiple implementations of
    /// [`StructuredTextCompletionModel`] compatible with the same type.
    #[cfg(not(target_arch = "wasm32"))]
    fn boxed_typed_completion_model<T>(self) -> BoxedStructuredTextCompletionModel<T>
    where
        Self: StructuredTextCompletionModel<
                Self::DefaultConstraints,
                Error: Send + Sync + Error + 'static,
                Session: TextCompletionSession<Error: Error + Send + Sync + 'static>
                             + Clone
                             + Send
                             + Sync
                             + 'static,
            > + CreateDefaultCompletionConstraintsForType<T>
            + Sized
            + Send
            + Sync
            + 'static,
        T: 'static,
    {
        BoxedStructuredTextCompletionModel::new(self)
    }
}

impl<M: CreateTextCompletionSession> TextCompletionModelExt for M {}

/// A builder for a text completion response. This is returned by [`TextCompletionModelExt::complete`]
/// and can be modified with [`TextCompletionBuilder::with_sampler`] and [`TextCompletionBuilder::with_constraints`]
/// until you start awaiting the response.
///
///
/// Once you are done setting up the response, you can call `.await` for the full text completion, or [`ModelConstraints::Output`]:
/// ```rust, no_run
/// use kalosm::language::*;
///
/// #[tokio::main]
/// async fn main() {
///     let mut llm = Llama::new().await.unwrap();
///     let prompt = "The following is a 300 word essay about why the capital of France is Paris:";
///     print!("{prompt}");
///     let mut completion = llm.complete(prompt).await.unwrap();
///     println!("{completion}");
/// }
/// ```
///
/// Or use the response as a [`Stream`]:
///
/// ```rust, no_run
/// use kalosm::language::*;
/// use std::io::Write;
///
/// #[tokio::main]
/// async fn main() {
///     let mut llm = Llama::new().await.unwrap();
///     let prompt = "The following is a 300 word essay about why the capital of France is Paris:";
///     print!("{prompt}");
///     let mut completion = llm.complete(prompt);
///     while let Some(token) = completion.next().await {
///         print!("{token}");
///         std::io::stdout().flush().unwrap();
///     }
/// }
/// ```
pub struct TextCompletionBuilder<
    M: CreateTextCompletionSession,
    Constraints = NoConstraints,
    Sampler = GenerationParameters,
> {
    text: MessageContent,
    model: Option<M>,
    constraints: Option<Constraints>,
    sampler: Option<Sampler>,
    task: OnceLock<RwLock<BoxedTaskFuture>>,
    #[allow(clippy::type_complexity)]
    result: Option<Receiver<Result<BoxedAny, M::Error>>>,
    queued_tokens: Option<UnboundedReceiver<String>>,
}

impl<M: CreateTextCompletionSession, Constraints, Sampler>
    TextCompletionBuilder<M, Constraints, Sampler>
{
    /// Constrains the model's response to the given parser. This can be used to make the model start with a certain phrase, or to make the model respond in a certain way.
    ///
    /// # Example
    /// ```rust, no_run
    /// # use kalosm::language::*;
    /// # #[tokio::main]
    /// # async fn main() {
    /// #[derive(Parse, Clone, Debug)]
    /// struct Pet {
    ///     name: String,
    ///     age: u32,
    ///     description: String,
    /// }
    ///
    /// // First create a model
    /// let model = Llama::new().await.unwrap();
    /// // Then create a parser for your data. Any type that implements the `Parse` trait has the `new_parser` method
    /// let parser = Pet::new_parser();
    /// // Create a text completion stream with the constraints
    /// let description = model.complete("JSON for an adorable dog named ruffles: ")
    ///     .with_constraints(parser);
    /// // Finally, await the stream to get the parsed response
    /// let pet: Pet = description.await.unwrap();
    /// println!("{pet:?}");
    /// # }
    /// ```
    pub fn with_constraints<NewConstraints: ModelConstraints>(
        self,
        constraints: NewConstraints,
    ) -> TextCompletionBuilder<M, NewConstraints, Sampler> {
        TextCompletionBuilder {
            text: self.text,
            model: self.model,
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
    /// #[derive(Parse, Clone, Debug)]
    /// struct Pet {
    ///     name: String,
    ///     age: u32,
    ///     description: String,
    /// }
    ///
    /// // First create a model
    /// let model = Llama::new().await.unwrap();
    /// // Create a text completion stream with the typed response
    /// let description = model
    ///     .complete("JSON for an adorable dog named ruffles: ")
    ///     .typed();
    /// // Finally, await the stream to get the parsed response
    /// let pet: Pet = description.await.unwrap();
    /// println!("{pet:?}");
    /// # }
    /// ```
    pub fn typed<T>(
        self,
    ) -> TextCompletionBuilder<
        M,
        <M as CreateDefaultCompletionConstraintsForType<T>>::DefaultConstraints,
        Sampler,
    >
    where
        M: CreateDefaultCompletionConstraintsForType<T>,
    {
        self.with_constraints(M::create_default_constraints())
    }

    /// Sets the sampler to use for generating responses. The sampler determines how tokens are chosen from the probability distribution
    /// the model generates. They can be used to make the model more or less predictable and prevent repetition.
    ///
    /// # Example
    /// ```rust, no_run
    /// # use kalosm::language::*;
    /// # #[tokio::main]
    /// # async fn main() {
    /// let model = Llama::new().await.unwrap();
    /// // Create the sampler to use for the text completion
    /// let sampler = GenerationParameters::default().sampler();
    /// // Create a completion request with the sampler
    /// let mut stream = model
    ///     .complete("Here is a list of 5 primes: ")
    ///     .with_sampler(sampler);
    /// stream.to_std_out().await.unwrap();
    /// # }
    /// ```
    pub fn with_sampler<NewSampler>(
        self,
        sampler: NewSampler,
    ) -> TextCompletionBuilder<M, Constraints, NewSampler> {
        TextCompletionBuilder {
            text: self.text,
            model: self.model,
            constraints: self.constraints,
            sampler: Some(sampler),
            queued_tokens: None,
            result: None,
            task: OnceLock::new(),
        }
    }
}

impl<M, Sampler> TextCompletionBuilder<M, NoConstraints, Sampler>
where
    Sampler: Send + Unpin + 'static,
    M: TextCompletionModel<Sampler> + Send + Sync + Unpin + 'static,
    M::Session: Send + Sync + Unpin + 'static,
{
    fn ensure_unstructured_task_started(&mut self) {
        if self.task.get().is_none() {
            let text = std::mem::take(&mut self.text);
            let model = self
                .model
                .take()
                .expect("TextCompletionBuilder cannot be turned into a future twice");
            let sampler = self
                .sampler
                .take()
                .expect("TextCompletionBuilder cannot be turned into a future twice");
            let (tx, rx) = futures_channel::mpsc::unbounded();
            let (result_tx, result_rx) = futures_channel::oneshot::channel();
            self.queued_tokens = Some(rx);
            self.result = Some(result_rx);
            let all_text = Arc::new(Mutex::new(String::new()));
            let on_token = {
                let all_text = all_text.clone();
                move |tok: String| {
                    all_text.lock().unwrap().push_str(&tok);
                    _ = tx.unbounded_send(tok);
                    Ok(())
                }
            };
            let future = async move {
                let mut session = model.new_session()?;
                model
                    .stream_text_with_callback(&mut session, text, sampler, on_token)
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

impl<M, Sampler> Stream for TextCompletionBuilder<M, NoConstraints, Sampler>
where
    Sampler: Send + Unpin + 'static,
    M: TextCompletionModel<Sampler> + Send + Sync + Unpin + 'static,
    M::Session: Send + Sync + Unpin + 'static,
{
    type Item = String;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let myself = Pin::get_mut(self);
        myself.ensure_unstructured_task_started();
        if let Some(token) = &mut myself.queued_tokens {
            match token.poll_next_unpin(cx) {
                Poll::Ready(Some(token)) => {
                    return Poll::Ready(Some(token));
                }
                Poll::Ready(None) => {}
                Poll::Pending => {}
            }
        }
        let mut task = myself.task.get().unwrap().write().unwrap();
        task.poll_unpin(cx).map(|_| None)
    }
}

impl<M, Sampler> IntoFuture for TextCompletionBuilder<M, NoConstraints, Sampler>
where
    Sampler: Send + Unpin + 'static,
    M: TextCompletionModel<Sampler> + Send + Sync + Unpin + 'static,
    M::Session: Clone + Send + Sync + Unpin + 'static,
{
    type Output = Result<String, M::Error>;
    type IntoFuture = BoxedIntoFuture<Self::Output>;

    fn into_future(mut self) -> Self::IntoFuture {
        self.ensure_unstructured_task_started();

        Box::pin(async move {
            if self.result.is_none() {
                self.task.into_inner().unwrap().into_inner().unwrap().await;
            }
            let result = self.result.take().unwrap().await.unwrap();
            result.map(|boxed| *boxed.downcast::<String>().unwrap())
        })
    }
}

impl<M, Constraints, Sampler> TextCompletionBuilder<M, Constraints, Sampler>
where
    Constraints: ModelConstraints + Send + Sync + Unpin + 'static,
    Sampler: Send + Unpin + 'static,
    M: StructuredTextCompletionModel<Constraints, Sampler> + Send + Sync + Unpin + 'static,
    M::Session: Clone + Send + Sync + Unpin + 'static,
    Constraints::Output: Send + 'static,
{
    fn ensure_structured_task_started(&mut self) {
        if self.task.get().is_none() {
            let text = std::mem::take(&mut self.text);
            let model = self
                .model
                .take()
                .expect("TextCompletionBuilder cannot be turned into a future twice");
            let sampler = self
                .sampler
                .take()
                .expect("TextCompletionBuilder cannot be turned into a future twice");
            let constraints = self
                .constraints
                .take()
                .expect("TextCompletionBuilder cannot be turned into a future twice");
            let (tx, rx) = futures_channel::mpsc::unbounded();
            let (result_tx, result_rx) = futures_channel::oneshot::channel();
            self.queued_tokens = Some(rx);
            self.result = Some(result_rx);
            let on_token = move |tok: String| {
                _ = tx.unbounded_send(tok);
                Ok(())
            };
            let future = async move {
                let mut session = model.new_session()?;
                model
                    .stream_text_with_callback_and_parser(
                        &mut session,
                        text,
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

impl<M, Constraints, Sampler> Stream for TextCompletionBuilder<M, Constraints, Sampler>
where
    Constraints: ModelConstraints + Send + Sync + Unpin + 'static,
    Sampler: Send + Unpin + 'static,
    M: StructuredTextCompletionModel<Constraints, Sampler> + Send + Sync + Unpin + 'static,
    M::Session: Clone + Send + Sync + Unpin + 'static,
    Constraints::Output: Send + 'static,
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
        task.poll_unpin(cx).map(|_| None)
    }
}

impl<M, Constraints, Sampler> IntoFuture for TextCompletionBuilder<M, Constraints, Sampler>
where
    Constraints: ModelConstraints + Send + Sync + Unpin + 'static,
    Sampler: Send + Unpin + 'static,
    M: StructuredTextCompletionModel<Constraints, Sampler> + Send + Sync + Unpin + 'static,
    M::Session: Clone + Send + Sync + Unpin + 'static,
    Constraints::Output: Send + 'static,
{
    type Output = Result<Constraints::Output, M::Error>;
    type IntoFuture = BoxedIntoFuture<Self::Output>;

    fn into_future(mut self) -> Self::IntoFuture {
        self.ensure_structured_task_started();

        Box::pin(async move {
            if self.result.is_none() {
                self.task.into_inner().unwrap().into_inner().unwrap().await;
            }
            let result = self.result.take().unwrap().await.unwrap();
            result.map(|boxed| *boxed.downcast::<Constraints::Output>().unwrap())
        })
    }
}
