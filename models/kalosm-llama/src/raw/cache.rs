use fusor2::cache::KvCache;
use fusor2::Tensor;

use super::LlamaConfig;

/// The dimension along which the attention cache grows for new tokens.
const CONCAT_DIMENSION: u32 = 2;

/// Initial per-layer capacity (tokens) for an unwindowed cache. Grows by
/// doubling — a new shape family — when a generation exceeds it.
const INITIAL_CAPACITY: usize = 512;

/// A cache for llama inference. Fixed-capacity device buffers with a
/// `Dim::Sym` logical length: every decode step reuses one graph and one
/// plan, only the length binding and a few leaf bytes change.
#[derive(Clone)]
pub struct LlamaCache {
    pub(crate) tokens: Vec<u32>,
    /// KV cache blocks, one per layer.
    pub(crate) blocks: Vec<KvCache>,
    /// The logits root of the decode graph, once one has been built against
    /// *these* blocks' store leaves.
    ///
    /// Every decode step used to re-run the whole model builder, and every node
    /// it minted hash-consed onto one already in the e-graph — 46,978 nodes in,
    /// 46,978 nodes out, identical ids, for ~2.5 ms of pure rediscovery per
    /// token. The graph is built once and this holds its root; a step is then a
    /// leaf write, a `replay_append` per block, and the resolve.
    ///
    /// Held here and not on the `Model` because the graph names this cache's
    /// store leaves: two caches against one model are two graphs.
    pub(crate) decode_graph: Option<Tensor<2>>,
}

impl LlamaCache {
    /// Create a new cache for a model
    pub fn new(config: &LlamaConfig) -> Self {
        let mut blocks = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            let window = match (config.sliding_window_size, config.sliding_window_type) {
                (Some(size), Some(t)) if t != 0 && (i + 1) % t != 0 => Some(size),
                _ => None,
            };
            blocks.push(match window {
                Some(w) => KvCache::windowed(CONCAT_DIMENSION, w as u64),
                None => KvCache::with_capacity(
                    CONCAT_DIMENSION,
                    config.context_length.min(INITIAL_CAPACITY) as u64,
                ),
            });
        }
        Self {
            tokens: Vec::new(),
            blocks,
            decode_graph: None,
        }
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        for block in &mut self.blocks {
            block.reset()
        }
        // A reset block has no armed append, so the next step rebuilds anyway;
        // dropping the root here is what keeps that from being an invariant
        // spread across two files.
        self.decode_graph = None;
    }
}
