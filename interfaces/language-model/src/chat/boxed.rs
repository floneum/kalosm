use crate::{BoxedMaybeFuture, BoxedTokenClosure, ModelConstraints};
use kalosm_model_types::{WasmNotSend, WasmNotSendSync};

use super::{
    ChatMessage, ChatModel, ChatSession, CreateChatSession, CreateDefaultChatConstraintsForType,
    StructuredChatModel,
};
use std::{error::Error, future::Future, pin::Pin, sync::Arc};

/// A boxed [`ChatModel`].
#[derive(Clone)]
pub struct BoxedChatModel {
    model: Arc<dyn DynChatModel + Send + Sync>,
}

impl BoxedChatModel {
    pub(crate) fn new(
        model: impl ChatModel<
                Error: Send + Sync + Error + 'static,
                ChatSession: ChatSession<Error: Error + Send + Sync + 'static>
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

impl CreateChatSession for BoxedChatModel {
    type ChatSession = BoxedChatSession;
    type Error = Box<dyn std::error::Error + Send + Sync + 'static>;

    fn new_chat_session(&self) -> Result<Self::ChatSession, Self::Error> {
        self.model.new_chat_session_boxed()
    }
}

impl ChatModel for BoxedChatModel {
    fn add_messages_with_callback<'a>(
        &'a self,
        session: Self::ChatSession,
        messages: &'a [ChatMessage],
        sampler: crate::GenerationParameters,
        on_token: impl FnMut(String) -> Result<(), Self::Error> + WasmNotSendSync + 'static,
    ) -> impl Future<Output = Result<Self::ChatSession, Self::Error>> + WasmNotSend + 'a {
        self.model
            .add_messages_with_callback_boxed(session, messages, sampler, Box::new(on_token))
    }
}

/// A boxed [`StructuredChatModel`].
#[derive(Clone)]
pub struct BoxedStructuredChatModel<T> {
    model: Arc<dyn DynStructuredChatModel<T> + Send + Sync>,
}

impl<T> BoxedStructuredChatModel<T> {
    pub(crate) fn new<S>(model: S) -> Self
    where
        S: StructuredChatModel<
                S::DefaultConstraints,
                Error: Send + Sync + Error + 'static,
                ChatSession: ChatSession<Error: Error + Send + Sync + 'static>
                                 + Clone
                                 + Send
                                 + Sync
                                 + 'static,
            > + CreateDefaultChatConstraintsForType<T>
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

impl<T> CreateChatSession for BoxedStructuredChatModel<T> {
    type ChatSession = BoxedChatSession;
    type Error = Box<dyn std::error::Error + Send + Sync + 'static>;

    fn new_chat_session(&self) -> Result<Self::ChatSession, Self::Error> {
        self.model.new_chat_session_boxed()
    }
}

impl<T> ChatModel for BoxedStructuredChatModel<T> {
    fn add_messages_with_callback<'a>(
        &'a self,
        session: Self::ChatSession,
        messages: &'a [ChatMessage],
        sampler: crate::GenerationParameters,
        on_token: impl FnMut(String) -> Result<(), Self::Error> + WasmNotSendSync + 'static,
    ) -> impl Future<Output = Result<Self::ChatSession, Self::Error>> + WasmNotSend + 'a {
        self.model
            .add_messages_with_callback_boxed(session, messages, sampler, Box::new(on_token))
    }
}

impl<T: 'static> StructuredChatModel<BoxedChatConstraintsForType<T>>
    for BoxedStructuredChatModel<T>
{
    fn add_message_with_callback_and_constraints<'a>(
        &'a self,
        session: Self::ChatSession,
        messages: &'a [ChatMessage],
        sampler: crate::GenerationParameters,
        constraints: BoxedChatConstraintsForType<T>,
        on_token: impl FnMut(String) -> Result<(), Self::Error> + WasmNotSendSync + 'static,
    ) -> impl Future<Output = Result<(Self::ChatSession, T), Self::Error>> + WasmNotSend + 'a
    where
        T: 'a,
    {
        self.model.add_message_with_callback_and_constraints_boxed(
            session,
            messages,
            sampler,
            constraints,
            Box::new(on_token),
        )
    }
}

impl<T> CreateDefaultChatConstraintsForType<T> for BoxedStructuredChatModel<T>
where
    T: 'static,
{
    type DefaultConstraints = BoxedChatConstraintsForType<T>;

    fn create_default_constraints() -> Self::DefaultConstraints {
        BoxedChatConstraintsForType {
            phantom: std::marker::PhantomData,
        }
    }
}

/// A boxed [`ChatSession`].
pub struct BoxedChatSession {
    session: Box<dyn DynChatSession + Send + Sync>,
}

impl Clone for BoxedChatSession {
    fn clone(&self) -> Self {
        DynChatSession::clone_(&*self.session)
    }
}

impl ChatSession for BoxedChatSession {
    type Error = Box<dyn std::error::Error + Send + Sync + 'static>;

    fn history(&self) -> Vec<super::ChatMessage> {
        self.session.history_boxed()
    }

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

trait DynCreateChatSession {
    fn new_chat_session_boxed(
        &self,
    ) -> Result<BoxedChatSession, Box<dyn std::error::Error + Send + Sync>>;
}

impl<S> DynCreateChatSession for S
where
    S: CreateChatSession<
        Error: Send + Sync + Error,
        ChatSession: ChatSession<Error: Error> + Clone + Send + Sync + 'static,
    >,
{
    fn new_chat_session_boxed(
        &self,
    ) -> Result<BoxedChatSession, Box<dyn std::error::Error + Send + Sync>> {
        let session = self
            .new_chat_session()
            .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)?;
        let session = Box::new(session) as Box<dyn DynChatSession + Send + Sync>;
        Ok(BoxedChatSession { session })
    }
}

trait DynChatSession {
    fn history_boxed(&self) -> Vec<super::ChatMessage>;

    fn try_clone_boxed(
        &self,
    ) -> Result<BoxedChatSession, Box<dyn std::error::Error + Send + Sync + 'static>>;

    fn into_any_box(self: Box<Self>) -> Box<dyn std::any::Any + Send + Sync>;

    fn clone_(&self) -> BoxedChatSession;
}

impl<S: ChatSession<Error: Error> + Send + Sync + Clone + 'static> DynChatSession for S {
    fn history_boxed(&self) -> Vec<super::ChatMessage> {
        self.history()
    }

    fn try_clone_boxed(
        &self,
    ) -> Result<BoxedChatSession, Box<dyn std::error::Error + Send + Sync + 'static>> {
        let session = self
            .try_clone()
            .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)?;
        let session = Box::new(session) as Box<dyn DynChatSession + Send + Sync>;
        Ok(BoxedChatSession { session })
    }

    fn into_any_box(self: Box<Self>) -> Box<dyn std::any::Any + Send + Sync> {
        self
    }

    fn clone_(&self) -> BoxedChatSession {
        BoxedChatSession {
            session: Box::new(Clone::clone(self)),
        }
    }
}

trait DynChatModel: DynCreateChatSession {
    fn add_messages_with_callback_boxed<'a>(
        &'a self,
        session: BoxedChatSession,
        messages: &'a [super::ChatMessage],
        sampler: crate::GenerationParameters,
        on_token: BoxedTokenClosure,
    ) -> BoxedMaybeFuture<'a, BoxedChatSession>;
}

impl<S> DynChatModel for S
where
    S: ChatModel<
        Error: Send + Sync + Error + 'static,
        ChatSession: ChatSession<Error: Error + Send + Sync + 'static>
                         + Clone
                         + Send
                         + Sync
                         + 'static,
    >,
{
    fn add_messages_with_callback_boxed<'a>(
        &'a self,
        session: BoxedChatSession,
        messages: &'a [super::ChatMessage],
        sampler: crate::GenerationParameters,
        mut on_token: BoxedTokenClosure,
    ) -> BoxedMaybeFuture<'a, BoxedChatSession> {
        let any = session.session.into_any_box();
        let session = match any.downcast::<S::ChatSession>() {
            Ok(session) => *session,
            Err(_) => {
                return Box::pin(async move {
                    Err(Box::new(MismatchedSessionType) as Box<dyn Error + Send + Sync>)
                });
            }
        };
        let on_token = move |token: String| {
            if let Err(err) = on_token(token) {
                tracing::error!("Error running on_token callback: {}", err);
            }
            Ok(())
        };
        let future = self.add_messages_with_callback(session, messages, sampler, on_token);
        // Double box prevents a rust compiler error with lifetimes. See https://github.com/rust-lang/rust/issues/102211
        let future: Pin<Box<dyn Future<Output = Result<S::ChatSession, _>> + Send>> =
            Box::pin(future);
        Box::pin(async move {
            let updated = future
                .await
                .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync + 'static>)?;
            Ok(BoxedChatSession {
                session: Box::new(updated),
            })
        })
    }
}

/// A constraints for [`CreateDefaultChatConstraintsForType`] that work with boxed [`StructuredChatModel`]s.
pub struct BoxedChatConstraintsForType<T> {
    phantom: std::marker::PhantomData<T>,
}

impl<T> ModelConstraints for BoxedChatConstraintsForType<T> {
    type Output = T;
}

trait DynStructuredChatModel<T>: DynChatModel {
    fn add_message_with_callback_and_constraints_boxed<'a>(
        &'a self,
        session: BoxedChatSession,
        messages: &'a [ChatMessage],
        sampler: crate::GenerationParameters,
        constraints: BoxedChatConstraintsForType<T>,
        on_token: BoxedTokenClosure,
    ) -> BoxedMaybeFuture<'a, (BoxedChatSession, T)>;
}

impl<S, T> DynStructuredChatModel<T> for S
where
    S: StructuredChatModel<
            S::DefaultConstraints,
            Error: Send + Sync + Error + 'static,
            ChatSession: ChatSession<Error: Error + Send + Sync + 'static>
                             + Clone
                             + Send
                             + Sync
                             + 'static,
        > + CreateDefaultChatConstraintsForType<T>,
    T: 'static,
{
    fn add_message_with_callback_and_constraints_boxed<'a>(
        &'a self,
        session: BoxedChatSession,
        messages: &'a [ChatMessage],
        sampler: crate::GenerationParameters,
        _: BoxedChatConstraintsForType<T>,
        mut on_token: BoxedTokenClosure,
    ) -> BoxedMaybeFuture<'a, (BoxedChatSession, T)> {
        let constraints =
            <S as CreateDefaultChatConstraintsForType<T>>::create_default_constraints();
        let any = session.session.into_any_box();
        let session = match any.downcast::<S::ChatSession>() {
            Ok(session) => *session,
            Err(_) => {
                return Box::pin(async move {
                    Err(Box::new(MismatchedSessionType) as Box<dyn Error + Send + Sync>)
                });
            }
        };

        let on_token = move |token: String| {
            if let Err(err) = on_token(token) {
                tracing::error!("Error running on_token callback: {}", err);
            }
            Ok(())
        };

        let future = self.add_message_with_callback_and_constraints(
            session,
            messages,
            sampler,
            constraints,
            on_token,
        );
        // Double box prevents a rust compiler error with lifetimes. See https://github.com/rust-lang/rust/issues/102211
        let future: Pin<Box<dyn Future<Output = Result<(S::ChatSession, T), _>> + Send>> =
            Box::pin(future);
        Box::pin(async move {
            let (updated, value) = future
                .await
                .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync + 'static>)?;
            Ok((
                BoxedChatSession {
                    session: Box::new(updated),
                },
                value,
            ))
        })
    }
}
