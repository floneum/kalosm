use crate::raw::cache::LlamaCache;
use crate::raw::LlamaConfig;
use kalosm_language_model::TextCompletionSession;
use std::sync::{Arc, RwLock};

/// An error that can occur when saving or loading a [`LlamaSession`].
#[derive(Debug, thiserror::Error)]
pub enum LlamaSessionLoadingError {
    /// An error from Fusor while loading or saving a [`LlamaSession`].
    #[error("Fusor error: {0:?}")]
    Fusor(#[from] fusor2::Error),
    /// The chat messages deserialized from the session are invalid.
    #[error("Chat messages deserialized from the session are invalid")]
    InvalidChatMessages,
}

/// A Llama session with cached state for the current fed prompt
#[derive(Clone)]
pub struct LlamaSession {
    pub(crate) cache: Arc<RwLock<LlamaCache>>,
}

impl TextCompletionSession for LlamaSession {
    type Error = LlamaSessionLoadingError;

    fn try_clone(&self) -> Result<Self, Self::Error>
    where
        Self: std::marker::Sized,
    {
        Ok(self.clone())
    }
}

impl LlamaSession {
    /// Create a new session
    pub(crate) fn new(cache: &LlamaConfig) -> Self {
        Self {
            cache: Arc::new(RwLock::new(LlamaCache::new(cache))),
        }
    }
}
