use kalosm_model_types::{WasmNotSend, WasmNotSendSync};
use kalosm_sample::Parser;
use std::{convert::Infallible, future::Future};

mod generation_parameters;
pub use generation_parameters::*;
mod ext;
pub use ext::*;
#[cfg(not(target_arch = "wasm32"))]
mod boxed;
#[cfg(not(target_arch = "wasm32"))]
pub use boxed::*;

use crate::MessageContent;

/// A parser for any type that implements the [`Schema`](kalosm_sample::Schema) trait and [`Deserialize`](serde::Deserialize).
///
/// Used by remote chat model adapters (e.g. OpenAI, Anthropic) to enforce structured
/// JSON schema generation at the API level.
#[derive(Debug, Clone, Copy)]
pub struct SchemaParser<P> {
    phantom: std::marker::PhantomData<P>,
}

impl<P> Default for SchemaParser<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P> SchemaParser<P> {
    /// Create a new parser for the given schema.
    pub const fn new() -> Self {
        Self {
            phantom: std::marker::PhantomData,
        }
    }
}

impl<P> ModelConstraints for SchemaParser<P> {
    type Output = P;
}

#[doc = include_str!("../../docs/completion_session.md")]
pub trait TextCompletionSession {
    /// The type of error the session may return during operations.
    type Error: Send + Sync + 'static;

    /// # Cloning Sessions
    ///
    /// Not all models support cloning sessions, but if a model does support cloning sessions, you can clone a session using the [`TextCompletionSession::try_clone`] method to clone a session state while retaining the original session.
    ///
    /// ```rust, no_run
    /// use kalosm::language::*;
    /// use std::io::Write;
    ///
    /// #[tokio::main]
    /// async fn main() {
    ///     let mut llm = Llama::new().await.unwrap();
    ///     let mut session = llm.new_session().unwrap();
    ///
    ///     // Feed some text into the session
    ///     llm.stream_text_with_callback(
    ///         &mut session,
    ///         "The capital of France is ".into(),
    ///         GenerationParameters::new().with_max_length(0),
    ///         |_| Ok(()),
    ///     )
    ///     .await
    ///     .unwrap();
    ///
    ///     // Clone the session
    ///     let cloned_session = session.try_clone().unwrap();
    ///
    ///     // Feed some more text into the cloned session
    ///     llm.stream_text_with_callback(
    ///         &mut session,
    ///         "The capital of France is ".into(),
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
    fn try_clone(&self) -> Result<Self, Self::Error>
    where
        Self: std::marker::Sized;
}

impl TextCompletionSession for () {
    type Error = Infallible;

    fn try_clone(&self) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// A marker type that indicates that no constraints were supplied.
#[derive(Debug, Clone, Copy)]
pub struct NoConstraints;

/// A type that can constrain the output of a model into a specific output type.
pub trait ModelConstraints {
    /// The type of the output of the constraints.
    type Output;
}

impl<P: Parser> ModelConstraints for P {
    type Output = <P as Parser>::Output;
}

/// A trait for creating a text completion session for a model. While it the core trait
/// every text completion model must implement, most methods to use models that implement
/// this trait are implemented in the [`TextCompletionModelExt`] trait.
///
/// # Example
///
/// ```rust, no_run
/// use kalosm::language::*;
///
/// #[tokio::main]
/// async fn main() {
///     // Create a new model which implements the CreateTextCompletionSession trait
///     let mut llm = Llama::new().await.unwrap();
///     // Create a new session for the model
///     let mut session = llm.new_session().unwrap();
/// }
/// ```
pub trait CreateTextCompletionSession {
    /// The type of error this model may return during operations.
    type Error: Send + Sync + 'static;

    /// The type of the session that this model uses.
    type Session: TextCompletionSession;

    /// Create a new session for this model.
    ///
    /// # Example
    /// ```rust, no_run
    /// # use kalosm::language::*;
    /// # #[tokio::main]
    /// # async fn main() {
    /// // Create a new model which implements the CreateTextCompletionSession trait
    /// let llm = Llama::new().await.unwrap();
    /// // Create a new session for the model
    /// let session = llm.new_session().unwrap();
    /// # }
    /// ```
    fn new_session(&self) -> Result<Self::Session, Self::Error>;
}

/// A trait that defines the default constraints for a type with this model.
pub trait CreateDefaultCompletionConstraintsForType<T>:
    StructuredTextCompletionModel<Self::DefaultConstraints>
{
    /// The default constraints for this type that work with this model.
    type DefaultConstraints: ModelConstraints<Output = T>;

    /// Create [`Self::DefaultConstraints`] which parse the type `T` for this model.
    fn create_default_constraints() -> Self::DefaultConstraints;
}

/// A trait for unstructured text completion models. This trait is required for any text completion models
/// even if they do not support structured generation. While this trait is implemented for all text completion models,
/// most methods to use models that implement this trait are implemented in the [`TextCompletionModelExt`] trait.
///
/// # Example
///
/// ```rust, no_run
/// use kalosm::language::*;
///
/// #[tokio::main]
/// async fn main() {
///     // Create a new model which implements the CreateTextCompletionSession trait
///     let mut llm = Llama::new().await.unwrap();
///     // Create a new session for the model
///     let mut session = llm.new_session().unwrap();
///     // Feed some text into the session using the raw text completion api that accepts a session, prompt, sampler, and on token callback
///     llm.stream_text_with_callback(&mut session, "The capital of France is ".into(), GenerationParameters::new(), |token| {println!("{token}"); Ok(())}).await.unwrap();
/// }
/// ```
pub trait TextCompletionModel<Sampler = GenerationParameters>: CreateTextCompletionSession {
    /// Generate text with the given prompt.
    ///
    /// See [`TextCompletionModelExt::complete`] for nicer API with an example.
    fn stream_text_with_callback<'a>(
        &'a self,
        session: &'a mut Self::Session,
        text: MessageContent,
        sampler: Sampler,
        on_token: impl FnMut(String) -> Result<(), Self::Error> + WasmNotSendSync + 'static,
    ) -> impl Future<Output = Result<(), Self::Error>> + WasmNotSend + 'a;
}

/// A trait for text completion models that support structured generation. While this trait is implemented for
/// all structured text completion models, most methods to use models that implement this trait are implemented
/// in the [`TextCompletionModelExt`] trait.
///
/// # Example
///
/// ```rust, no_run
/// use kalosm::language::*;
///
/// #[tokio::main]
/// async fn main() {
///     // Create a new model which implements the CreateTextCompletionSession trait
///     let mut llm = Llama::new().await.unwrap();
///     // Create a new session for the model
///     let mut session = llm.new_session().unwrap();
///     // Create a parser for your data. Different models accept different types of parsers. The Llama model accepts
///     // any parsers that implements the `Parse` trait.
///     let parser = i32::new_parser();
///     // Feed some text into the session using the raw structured text completion api that accepts a session, prompt, sampler, and on token callback
///     llm.stream_text_with_callback_and_parser(&mut session, "5 * 5 = ".into(), GenerationParameters::new(), parser, |token| {println!("{token}"); Ok(())}).await.unwrap();
/// }
/// ```
pub trait StructuredTextCompletionModel<
    Constraints: ModelConstraints,
    Sampler = GenerationParameters,
>: TextCompletionModel<Sampler>
{
    /// Generate text with the given prompt and the given constraints.
    ///
    /// See [`TextCompletionModelExt::complete`] for nicer API with an example.
    fn stream_text_with_callback_and_parser<'a>(
        &'a self,
        session: &'a mut Self::Session,
        text: MessageContent,
        sampler: Sampler,
        parser: Constraints,
        on_token: impl FnMut(String) -> Result<(), Self::Error> + WasmNotSendSync + 'static,
    ) -> impl Future<Output = Result<Constraints::Output, Self::Error>> + WasmNotSend + 'a;
}

/// Remove JSON Schema properties that are not supported by structured output APIs
/// (e.g. Anthropic, OpenAI). This strips validation-only keywords like `minLength`,
/// `pattern`, `minimum`, `maxItems`, etc. that would cause API errors.
#[cfg(any(feature = "anthropic", feature = "openai"))]
pub(crate) fn remove_unsupported_schema_properties(schema: &mut serde_json::Value) {
    match schema {
        serde_json::Value::Null
        | serde_json::Value::Bool(_)
        | serde_json::Value::Number(_)
        | serde_json::Value::String(_) => {}
        serde_json::Value::Array(array) => {
            for item in array {
                remove_unsupported_schema_properties(item);
            }
        }
        serde_json::Value::Object(map) => {
            map.retain(|key, value| {
                const UNSUPPORTED_PROPERTIES: [&str; 19] = [
                    "minLength",
                    "maxLength",
                    "pattern",
                    "format",
                    "minimum",
                    "maximum",
                    "multipleOf",
                    "patternProperties",
                    "unevaluatedProperties",
                    "propertyNames",
                    "minProperties",
                    "maxProperties",
                    "unevaluatedItems",
                    "contains",
                    "minContains",
                    "maxContains",
                    "minItems",
                    "maxItems",
                    "uniqueItems",
                ];
                if UNSUPPORTED_PROPERTIES.contains(&key.as_str()) {
                    return false;
                }
                remove_unsupported_schema_properties(value);
                true
            });
        }
    }
}
