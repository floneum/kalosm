use std::marker::PhantomData;

use crate::raw::cache::LlamaCache;
use crate::raw::LlamaConfig;
use fusor::FloatDataType;
use fusor::SimdElement;
use kalosm_language_model::TextCompletionSession;
use std::sync::{Arc, RwLock};

/// An error that can occur when saving or loading a [`LlamaSession`].
#[derive(Debug, thiserror::Error)]
pub enum LlamaSessionLoadingError {
    /// An error from safetensors while loading or saving a [`LlamaSession`].
    #[error("Safetensors error: {0}")]
    Safetensors(#[from] safetensors::SafeTensorError),
    /// An error from candle while loading or saving a [`LlamaSession`].
    #[error("Candle error: {0:?}")]
    Candle(#[from] fusor::Error),
    /// The chat messages deserialized from the session are invalid.
    #[error("Chat messages deserialized from the session are invalid")]
    InvalidChatMessages,
}

/// A Llama session with cached state for the current fed prompt
#[derive(Clone)]
pub struct LlamaSession<F: FloatDataType + SimdElement = f32> {
    pub(crate) cache: Arc<RwLock<LlamaCache>>,
    _marker: PhantomData<F>,
}

impl<F: FloatDataType + SimdElement> TextCompletionSession for LlamaSession<F> {
    type Error = LlamaSessionLoadingError;

    fn try_clone(&self) -> Result<Self, Self::Error>
    where
        Self: std::marker::Sized,
    {
        Ok(self.clone())
    }
}

impl<F: FloatDataType + SimdElement> LlamaSession<F> {
    /// Create a new session
    pub(crate) fn new(cache: &LlamaConfig<F>) -> Self {
        Self {
            cache: Arc::new(RwLock::new(LlamaCache::new(cache))),
            _marker: PhantomData,
        }
    }

    // /// Export the current cache tensor map.
    // pub fn get_tensor_map(&self, device: &Device) -> HashMap<String, Tensor> {
    //     let cache = self.cache.read().unwrap();
    //     cache.get_tensor_map(device)
    // }

    // /// Import a cache tensor map.
    // pub fn set_tensor_map(&mut self, map: HashMap<String, Tensor>) -> fusor::Result<()> {
    //     let mut cache = self.cache.write().unwrap();
    //     *cache = LlamaCache::from_tensor_map(map)?;
    //     Ok(())
    // }

    // /// Create a cache from a tensor map. This can be used to load a cache from disk.
    // pub fn from_tensor_map(map: HashMap<String, Tensor>) -> fusor::Result<Self> {
    //     Ok(Self {
    //         cache: Arc::new(RwLock::new(LlamaCache::from_tensor_map(map)?)),
    //     })
    // }
}
