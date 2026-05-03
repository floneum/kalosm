use fusor::cache::KvCache;
use fusor::SimdElement;

use super::LlamaConfig;

/// The dimension along which the attention cache is concatenated with attention for new tokens.
const CONCAT_DIMENSION: usize = 2;

/// A cache for llama inference. This cache will speed up generation of sequential text significantly.
///
/// Note: The KV cache internally uses f32 for SIMD compatibility, regardless of the model's
/// storage type F. This enables f16 models to work with f32 computation.
#[derive(Clone)]
pub struct LlamaCache {
    pub(crate) start_time: u32,
    pub(crate) tokens: Vec<u32>,
    /// KV cache blocks - always f32 for SIMD operations
    pub(crate) blocks: Vec<KvCache<f32>>,
}

impl LlamaCache {
    /// Create a new cache for a model
    pub fn new<G: fusor::FloatDataType + SimdElement>(config: &LlamaConfig<G>) -> Self {
        let max_seq_len = config.context_length;
        let mut blocks = Vec::with_capacity(config.n_layer);
        for layer_idx in 0..config.n_layer {
            let max_seq_len = if let (Some(sliding_window_type), Some(sliding_window_size)) =
                (config.sliding_window_type, config.sliding_window_size)
            {
                let is_sliding = (layer_idx + 1) % sliding_window_type != 0;
                if is_sliding {
                    sliding_window_size
                } else {
                    max_seq_len
                }
            } else {
                max_seq_len
            };
            blocks.push(KvCache::new(CONCAT_DIMENSION, max_seq_len))
        }
        Self {
            start_time: 0,
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
