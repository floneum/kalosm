//! ModernBERT encoder model.

use fusor::layers::{Embedding, LayerNorm};
use fusor::{Device, Result, RopeCache, Tensor, VarBuilder};

use super::config::ModernBertConfig;
use super::layer::ModernBertLayer;

/// A raw synchronous ModernBERT (Ettin) encoder model. This is a bidirectional
/// transformer with RoPE positional embeddings, pre-normalization, and GeGLU
/// feed-forward blocks.
pub struct ModernBertModel {
    token_embeddings: Embedding<f32>,
    /// Embedding norm applied after token embeddings, before first layer
    embedding_norm: LayerNorm<1, f32>,
    layers: Vec<ModernBertLayer>,
    final_norm: LayerNorm<1, f32>,
    rope_cache: RopeCache,
    pub(crate) device: Device,
    config: ModernBertConfig,
    span: tracing::Span,
}

impl ModernBertModel {
    /// Load ModernBERT from GGUF weights.
    pub fn load(device: &Device, vb: &mut VarBuilder) -> Result<Self> {
        let config = ModernBertConfig::from_gguf(vb)?;

        // Load token embeddings
        let token_embeddings = Embedding::load(device, &mut vb.pp("token_embd"))?;

        // Load embedding norm (applied before first layer)
        let embedding_norm = LayerNorm::load(device, &mut vb.pp("embd_norm"), config.norm_eps)?;

        // Create RoPE cache
        let rope_cache = RopeCache::new(
            config.head_dimension,
            config.context_length,
            config.rope_theta,
            device,
        )?;

        // Load transformer layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let layer = ModernBertLayer::load(
                device,
                &mut vb.pp(format!("blk.{i}")),
                config.num_heads,
                config.num_kv_heads,
                config.head_dimension,
                config.norm_eps,
                i,
            )?;
            layers.push(layer);
        }

        // Load final layer norm
        let final_norm = LayerNorm::load(device, &mut vb.pp("output_norm"), config.norm_eps)?;

        Ok(Self {
            token_embeddings,
            embedding_norm,
            layers,
            final_norm,
            rope_cache,
            device: device.clone(),
            config,
            span: tracing::span!(tracing::Level::TRACE, "modern-bert"),
        })
    }

    /// Forward pass through the model.
    ///
    /// Returns: [batch_size, seq_len, hidden_size]
    pub fn forward(
        &self,
        input_ids: &Tensor<2, u32>,
        attention_mask: Option<&Tensor<2, u32>>,
    ) -> Tensor<3, f32> {
        let _enter = self.span.enter();
        // Get token embeddings
        let hidden_states = self.token_embeddings.forward(input_ids);

        // Apply embedding norm (serves as pre-norm for layer 0)
        let mut hidden_states = self.embedding_norm.forward(&hidden_states);

        // Pass through transformer layers
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, &self.rope_cache, attention_mask);
        }

        // Apply final layer norm
        self.final_norm.forward(&hidden_states)
    }

    /// Get the maximum sequence length.
    pub fn max_seq_len(&self) -> usize {
        self.config.context_length
    }

    /// Get the embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.config.hidden_size
    }

    /// Get the device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Return the hidden state after each layer (for debugging / regression tests).
    #[doc(hidden)]
    pub fn debug_hidden_states(
        &self,
        input_ids: &Tensor<2, u32>,
        attention_mask: Option<&Tensor<2, u32>>,
    ) -> Vec<Tensor<3, f32>> {
        let _enter = self.span.enter();
        let mut states = Vec::with_capacity(self.layers.len() + 2);

        let hidden_states = self.token_embeddings.forward(input_ids);
        let mut hidden_states = self.embedding_norm.forward(&hidden_states);
        states.push(hidden_states.clone());

        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, &self.rope_cache, attention_mask);
            states.push(hidden_states.clone());
        }

        states.push(self.final_norm.forward(&hidden_states));
        states
    }
}
