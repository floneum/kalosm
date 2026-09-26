use fusor::cache::KvCache;
use fusor::Tensor;

use super::LlamaConfig;

/// The dimension along which the attention cache grows for new tokens.
const CONCAT_DIMENSION: u32 = 2;

/// Initial per-layer capacity (tokens), doubled as needed.
const INITIAL_CAPACITY: usize = 512;

/// Device buffers with symbolic lengths and capacities. Decode replays one
/// graph while input bytes and dimension bindings change.
#[derive(Clone)]
pub struct LlamaCache {
    pub(crate) tokens: Vec<u32>,
    /// A sampled token that has not yet been forwarded into the KV cache.
    pub(crate) pending_token: Option<u32>,
    /// KV cache blocks, one per layer.
    pub(crate) blocks: Vec<KvCache>,
    /// Token-step logits for these cache stores. Replay updates input leaves
    /// and write indices without rebuilding the graph.
    pub(crate) decode_graph: Option<Tensor<2>>,
    /// The same memo for the embedding-row step an image prompt runs: its
    /// input is a `[1, 1, hidden]` embedding leaf and explicit rope rows,
    /// not a token id. One of the two memos is live at a time; the other
    /// rebuilds on its next use, since the caches' armed appends belong to
    /// whichever graph ran last.
    pub(crate) embed_graph: Option<Tensor<2>>,
    /// The rope position of the next token. Equal to `tokens.len()` for a
    /// text-only history; an image's tokens advance it by the larger side
    /// of their grid rather than by their count.
    pub(crate) rope_position: u32,
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
            pending_token: None,
            blocks,
            decode_graph: None,
            embed_graph: None,
            rope_position: 0,
        }
    }

    pub(crate) fn input_tokens<'a>(&self, tokens: &'a [u32]) -> std::borrow::Cow<'a, [u32]> {
        match self.pending_token {
            Some(token) => std::borrow::Cow::Owned(
                std::iter::once(token)
                    .chain(tokens.iter().copied())
                    .collect(),
            ),
            None => std::borrow::Cow::Borrowed(tokens),
        }
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        for block in &mut self.blocks {
            block.reset()
        }
        self.decode_graph = None;
        self.embed_graph = None;
        self.rope_position = 0;
    }
}
