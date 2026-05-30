use crate::MessageContent;
use kalosm_model_types::{WasmNotSend, WasmNotSendSync};

use super::{
    CreateDefaultCompletionConstraintsForType, CreateTextCompletionSession, ModelConstraints,
    StructuredTextCompletionModel, TextCompletionModel, TextCompletionSession,
};
use std::{error::Error, future::Future, pin::Pin, sync::Arc};

/// A boxed [`TextCompletionModel`].
#[derive(Clone)]
pub struct BoxedTextCompletionModel {
    model: Arc<dyn DynTextCompletionModel + Send + Sync>,
}

impl BoxedTextCompletionModel {
    pub(crate) fn new(
        model: impl TextCompletionModel<
                Error: Send + Sync + Error + 'static,
                Session: TextCompletionSession<Error: Error + Send + Sync + 'static>
                             + Clone
                             + Send
                             + Sync
                             + 'static,
            > + Send
            + Sync
            + 'static,
    ) -> Self {
        Self {
            model: Arc::new(model),
        }
    }
}

impl CreateTextCompletionSession for BoxedTextCompletionModel {
    type Session = BoxedTextCompletionSession;
    type Error = Box<dyn std::error::Error + Send + Sync + 'static>;

    fn new_session(&self) -> Result<Self::Session, Self::Error> {
        self.model.new_session_boxed()
    }
}

impl TextCompletionModel for BoxedTextCompletionModel {
    fn stream_text_with_callback<'a>(
        &'a self,
        session: &'a mut Self::Session,
        text: MessageContent,
        sampler: super::GenerationParameters,
        on_token: impl FnMut(String) -> Result<(), Self::Error> + WasmNotSendSync + 'static,
    ) -> impl Future<Output = Result<(), Self::Error>> + WasmNotSend + 'a {
        self.model
            .add_messages_with_callback_boxed(session, text, sampler, Box::new(on_token))
    }
}

/// A boxed [`BoxedStructuredTextCompletionModel`].
#[derive(Clone)]
pub struct BoxedStructuredTextCompletionModel<T> {
    model: Arc<dyn DynStructuredTextCompletionModel<T> + Send + Sync>,
}

impl<T> BoxedStructuredTextCompletionModel<T> {
    pub(crate) fn new<S>(model: S) -> Self
    where
        S: StructuredTextCompletionModel<
                S::DefaultConstraints,
                Error: Send + Sync + Error + 'static,
                Session: TextCompletionSession<Error: Error + Send + Sync + 'static>
                             + Clone
                             + Send
                             + Sync
                             + 'static,
            > + CreateDefaultCompletionConstraintsForType<T>
            + Send
            + Sync
            + 'static,
        T: 'static,
    {
        Self {
            model: Arc::new(model),
        }
    }
}

impl<T> CreateTextCompletionSession for BoxedStructuredTextCompletionModel<T> {
    type Session = BoxedTextCompletionSession;
    type Error = Box<dyn std::error::Error + Send + Sync + 'static>;

    fn new_session(&self) -> Result<Self::Session, Self::Error> {
        self.model.new_session_boxed()
    }
}

impl<T> TextCompletionModel for BoxedStructuredTextCompletionModel<T> {
    fn stream_text_with_callback<'a>(
        &'a self,
        session: &'a mut Self::Session,
        text: MessageContent,
        sampler: super::GenerationParameters,
        on_token: impl FnMut(String) -> Result<(), Self::Error> + WasmNotSendSync + 'static,
    ) -> impl Future<Output = Result<(), Self::Error>> + WasmNotSend + 'a {
        self.model
            .add_messages_with_callback_boxed(session, text, sampler, Box::new(on_token))
    }
}

impl<T> StructuredTextCompletionModel<BoxedCompletionConstraintsForType<T>>
    for BoxedStructuredTextCompletionModel<T>
{
    fn stream_text_with_callback_and_parser<'a>(
        &'a self,
        session: &'a mut Self::Session,
        text: MessageContent,
        sampler: super::GenerationParameters,
        parser: BoxedCompletionConstraintsForType<T>,
        on_token: impl FnMut(String) -> Result<(), Self::Error> + WasmNotSendSync + 'static,
    ) -> impl Future<Output = Result<T, Self::Error>> + WasmNotSend + 'a {
        self.model.add_messages_with_callback_and_parser_boxed(
            session,
            text,
            sampler,
            parser,
            Box::new(on_token),
        )
    }
}

/// A boxed [`TextCompletionSession`].
pub struct BoxedTextCompletionSession {
    session: Box<dyn DynTextCompletionSession + Send + Sync>,
}

impl Clone for BoxedTextCompletionSession {
    fn clone(&self) -> Self {
        DynTextCompletionSession::clone_(&*self.session)
    }
}

impl TextCompletionSession for BoxedTextCompletionSession {
    type Error = Box<dyn std::error::Error + Send + Sync + 'static>;

    fn try_clone(&self) -> Result<Self, Self::Error>
    where
        Self: std::marker::Sized,
    {
        self.session.try_clone_boxed()
    }
}

#[derive(Debug)]
struct MismatchedSessionType;

impl std::error::Error for MismatchedSessionType {}

impl std::fmt::Display for MismatchedSessionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MismatchedSessionType")
    }
}

trait DynCreateTextCompletionSession {
    fn new_session_boxed(
        &self,
    ) -> Result<BoxedTextCompletionSession, Box<dyn std::error::Error + Send + Sync>>;
}

impl<S> DynCreateTextCompletionSession for S
where
    S: CreateTextCompletionSession<
        Error: Send + Sync + Error,
        Session: TextCompletionSession<Error: Error> + Clone + Send + Sync + 'static,
    >,
{
    fn new_session_boxed(
        &self,
    ) -> Result<BoxedTextCompletionSession, Box<dyn std::error::Error + Send + Sync>> {
        let session = self
            .new_session()
            .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)?;
        let session = Box::new(session) as Box<dyn DynTextCompletionSession + Send + Sync>;
        Ok(BoxedTextCompletionSession { session })
    }
}

trait DynTextCompletionSession {
    fn try_clone_boxed(
        &self,
    ) -> Result<BoxedTextCompletionSession, Box<dyn std::error::Error + Send + Sync + 'static>>;

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;

    fn clone_(&self) -> BoxedTextCompletionSession;
}

impl<S: TextCompletionSession<Error: Error> + Clone + Send + Sync + 'static>
    DynTextCompletionSession for S
{
    fn try_clone_boxed(
        &self,
    ) -> Result<BoxedTextCompletionSession, Box<dyn std::error::Error + Send + Sync + 'static>>
    {
        let session = self
            .try_clone()
            .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)?;
        let session = Box::new(session) as Box<dyn DynTextCompletionSession + Send + Sync>;
        Ok(BoxedTextCompletionSession { session })
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn clone_(&self) -> BoxedTextCompletionSession {
        BoxedTextCompletionSession {
            session: Box::new(Clone::clone(self)),
        }
    }
}

pub(crate) type BoxedMaybeFuture<'a, T = ()> = Pin<
    Box<
        dyn Future<Output = Result<T, Box<dyn std::error::Error + Send + Sync + 'static>>>
            + Send
            + 'a,
    >,
>;
pub(crate) type BoxedTokenClosure = Box<
    dyn FnMut(String) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>>
        + Send
        + Sync
        + 'static,
>;

trait DynTextCompletionModel: DynCreateTextCompletionSession {
    fn add_messages_with_callback_boxed<'a>(
        &'a self,
        session: &'a mut BoxedTextCompletionSession,
        text: MessageContent,
        sampler: crate::GenerationParameters,
        on_token: BoxedTokenClosure,
    ) -> BoxedMaybeFuture<'a>;
}

impl<S> DynTextCompletionModel for S
where
    S: TextCompletionModel<
        Error: Send + Sync + Error + 'static,
        Session: TextCompletionSession<Error: Error + Send + Sync + 'static>
                     + Clone
                     + Send
                     + Sync
                     + 'static,
    >,
{
    fn add_messages_with_callback_boxed<'a>(
        &'a self,
        session: &'a mut BoxedTextCompletionSession,
        text: MessageContent,
        sampler: crate::GenerationParameters,
        mut on_token: BoxedTokenClosure,
    ) -> BoxedMaybeFuture<'a> {
        let session = session.session.as_any_mut();

        let Some(session) = session.downcast_mut::<S::Session>() else {
            return Box::pin(async move {
                Err(Box::new(MismatchedSessionType) as Box<dyn Error + Send + Sync>)
            });
        };
        let on_token = move |token: String| {
            if let Err(err) = on_token(token) {
                tracing::error!("Error running on_token callback: {}", err);
            }
            Ok(())
        };
        let future = self.stream_text_with_callback(session, text, sampler, on_token);
        // Double box prevents a rust compiler error with lifetimes. See https://github.com/rust-lang/rust/issues/102211
        let boxed: Pin<Box<dyn Future<Output = Result<(), _>> + Send>> = Box::pin(future);
        Box::pin(async move {
            boxed
                .await
                .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync + 'static>)
        })
    }
}

/// A constraints for [`CreateDefaultCompletionConstraintsForType`] that work with boxed [`TextCompletionModel`]s.
pub struct BoxedCompletionConstraintsForType<T> {
    phantom: std::marker::PhantomData<T>,
}

impl<T> ModelConstraints for BoxedCompletionConstraintsForType<T> {
    type Output = T;
}

trait DynStructuredTextCompletionModel<T>: DynTextCompletionModel {
    fn add_messages_with_callback_and_parser_boxed<'a>(
        &'a self,
        session: &'a mut BoxedTextCompletionSession,
        text: MessageContent,
        sampler: crate::GenerationParameters,
        constraints: BoxedCompletionConstraintsForType<T>,
        on_token: BoxedTokenClosure,
    ) -> BoxedMaybeFuture<'a, T>;
}

impl<S, T> DynStructuredTextCompletionModel<T> for S
where
    S: StructuredTextCompletionModel<
            S::DefaultConstraints,
            Error: Send + Sync + Error + 'static,
            Session: TextCompletionSession<Error: Error + Send + Sync + 'static>
                         + Clone
                         + Send
                         + Sync
                         + 'static,
        > + CreateDefaultCompletionConstraintsForType<T>,
    T: 'static,
{
    fn add_messages_with_callback_and_parser_boxed<'a>(
        &'a self,
        session: &'a mut BoxedTextCompletionSession,
        text: MessageContent,
        sampler: crate::GenerationParameters,
        _: BoxedCompletionConstraintsForType<T>,
        mut on_token: BoxedTokenClosure,
    ) -> BoxedMaybeFuture<'a, T> {
        let constraints =
            <S as CreateDefaultCompletionConstraintsForType<T>>::create_default_constraints();
        let session = session.session.as_any_mut();

        let Some(session) = session.downcast_mut::<S::Session>() else {
            return Box::pin(async move {
                Err(Box::new(MismatchedSessionType) as Box<dyn Error + Send + Sync>)
            });
        };
        let on_token = move |token: String| {
            if let Err(err) = on_token(token) {
                tracing::error!("Error running on_token callback: {}", err);
            }
            Ok(())
        };
        let future = self.stream_text_with_callback_and_parser(
            session,
            text,
            sampler,
            constraints,
            on_token,
        );
        // Double box prevents a rust compiler error with lifetimes. See https://github.com/rust-lang/rust/issues/102211
        let boxed: Pin<Box<dyn Future<Output = Result<T, _>> + Send>> = Box::pin(future);
        Box::pin(async move {
            boxed
                .await
                .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync + 'static>)
        })
    }
}
