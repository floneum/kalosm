use fusor2::cache::KvCache;

use super::LlamaConfig;

/// The dimension along which the attention cache is concatenated with attention for new tokens.
const CONCAT_DIMENSION: u32 = 2;

/// A cache for llama inference. This cache will speed up generation of sequential text significantly.
#[derive(Clone)]
pub struct LlamaCache {
    pub(crate) tokens: Vec<u32>,
    /// KV cache blocks, one per layer.
    pub(crate) blocks: Vec<KvCache>,
}

impl LlamaCache {
    /// Create a new cache for a model
    pub fn new(config: &LlamaConfig) -> Self {
        let mut blocks = Vec::with_capacity(config.n_layer);
        for _ in 0..config.n_layer {
            blocks.push(KvCache::new(CONCAT_DIMENSION))
        }
        Self {
            tokens: Vec::new(),
            blocks,
        }
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        for block in &mut self.blocks {
            block.reset()
        }
    }
}
