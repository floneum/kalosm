//! ModernBERT encoder model.

use fusor::layers::{Embedding, LayerNorm};
use fusor::{Device, Result, RopeCache, Tensor, VarBuilder};

use super::super::utils::attention_mask_to_bias;
use super::config::ModernBertConfig;
use super::layer::ModernBertLayer;

/// A raw synchronous ModernBERT (Ettin) encoder model. This is a bidirectional
/// transformer with RoPE positional embeddings, pre-normalization, and GeGLU
/// feed-forward blocks. Each layer is a shared [`fusor::TransformerBlock`];
/// global/local attention routing is resolved per layer at load time.
pub struct ModernBertModel {
    token_embeddings: Embedding<f32>,
    /// Embedding norm applied after token embeddings, before first layer
    embedding_norm: LayerNorm<1, f32>,
    layers: Vec<ModernBertLayer>,
    final_norm: LayerNorm<1, f32>,
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

        // Create RoPE caches. Global and local layers may use different bases;
        // when the model has a single base the two caches are identical.
        let global_rope = RopeCache::new(
            config.head_dimension,
            config.context_length,
            config.rope_theta,
            device,
        )?;
        let local_rope = RopeCache::new(
            config.head_dimension,
            config.context_length,
            config.local_rope_theta,
            device,
        )?;
        let local_window = (config.local_attention > 0).then(|| config.local_attention / 2);

        // Load transformer layers, routing each to its global/local RoPE cache
        // and (for local layers) sliding-window. A layer is global when
        // `idx % global_attn_every_n_layers == 0`.
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let is_global = i % config.global_attn_every_n_layers == 0;
            let (rope, window) = if is_global {
                (&global_rope, None)
            } else {
                (&local_rope, local_window)
            };
            let layer = ModernBertLayer::load(
                device,
                &mut vb.pp(format!("blk.{i}")),
                &config,
                i,
                rope,
                window,
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

        // Convert the padding mask to an additive bias once, then reuse it for
        // every layer.
        let mask_bias = attention_mask.map(attention_mask_to_bias);

        // Pass through transformer layers (each routes itself global/local).
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, mask_bias.as_ref());
        }

        // Apply final layer norm
        self.final_norm.forward(&hidden_states)
    }
}
