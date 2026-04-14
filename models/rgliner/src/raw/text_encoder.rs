//! Text encoder wrapper for GLiNER.

use fusor::layers::Linear;
use fusor::{Device, Result, Tensor, VarBuilder};

use rbert::raw::ModernBertModel;

/// Text encoder for GLiNER (ModernBERT/Ettin).
pub struct TextEncoder {
    model: ModernBertModel,
    /// Optional output projection. Some bi-encoder variants (e.g.
    /// `gliner-bi-small-v2.0`) have a `token_rep_layer.projection` that maps
    /// the encoder's native hidden size down to the dim shared with the label
    /// encoder / downstream heads (e.g. 512 -> 384). Absent on `edge`.
    output_proj: Option<Linear<f32>>,
}

impl TextEncoder {
    /// Load text encoder from GGUF weights.
    pub fn load(device: &Device, vb: &mut VarBuilder) -> Result<Self> {
        // GLiNER GGUF uses "text." prefix for text encoder weights
        let mut text_vb = vb.pp("text");
        let model = ModernBertModel::load(device, &mut text_vb)?;
        // Optional output projection (small/base/large v2.0 variants).
        let output_proj = Linear::load(device, &mut text_vb.pp("output_proj")).ok();

        #[cfg(debug_assertions)]
        if let Some(ref p) = output_proj {
            eprintln!(
                "[DEBUG] TextEncoder output projection loaded: {} -> {}",
                p.in_features(),
                p.out_features()
            );
        }

        Ok(Self { model, output_proj })
    }

    /// Forward pass returning per-token embeddings.
    ///
    /// Returns: [batch_size, seq_len, hidden_size]  (hidden_size is the
    /// projected dim if `output_proj` is present)
    pub fn forward(
        &self,
        input_ids: &Tensor<2, u32>,
        attention_mask: Option<&Tensor<2, u32>>,
    ) -> Tensor<3, f32> {
        let hidden = self.model.forward(input_ids, attention_mask);
        if let Some(ref proj) = self.output_proj {
            proj.forward(&hidden)
        } else {
            hidden
        }
    }

    /// Get the maximum sequence length.
    pub fn max_seq_len(&self) -> usize {
        self.model.max_seq_len()
    }

    /// Get the embedding dimension seen by downstream layers (post-projection
    /// if this variant has one, otherwise the raw encoder hidden dim).
    pub fn embedding_dim(&self) -> usize {
        if let Some(ref proj) = self.output_proj {
            proj.out_features()
        } else {
            self.model.embedding_dim()
        }
    }

    /// Get the device.
    pub fn device(&self) -> &Device {
        self.model.device()
    }

    #[cfg(test)]
    pub fn debug_hidden_states(
        &self,
        input_ids: &Tensor<2, u32>,
        attention_mask: Option<&Tensor<2, u32>>,
    ) -> Vec<Tensor<3, f32>> {
        self.model.debug_hidden_states(input_ids, attention_mask)
    }
}
