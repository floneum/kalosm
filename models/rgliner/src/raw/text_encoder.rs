//! Text encoder wrapper for GLiNER.

use fusor::{Device, Result, Tensor, VarBuilder};

use super::modern_bert::ModernBertModel;

/// Text encoder for GLiNER (ModernBERT/Ettin).
pub struct TextEncoder {
    model: ModernBertModel,
}

impl TextEncoder {
    /// Load text encoder from GGUF weights.
    pub fn load(device: &Device, vb: &mut VarBuilder) -> Result<Self> {
        // GLiNER GGUF uses "text." prefix for text encoder weights
        let model = ModernBertModel::load(device, &mut vb.pp("text"))?;
        Ok(Self { model })
    }

    /// Forward pass returning per-token embeddings.
    ///
    /// Returns: [batch_size, seq_len, hidden_size]
    pub fn forward(
        &self,
        input_ids: &Tensor<2, u32>,
        attention_mask: Option<&Tensor<2, u32>>,
    ) -> Tensor<3, f32> {
        self.model.forward(input_ids, attention_mask)
    }

    /// Get the maximum sequence length.
    pub fn max_seq_len(&self) -> usize {
        self.model.max_seq_len()
    }

    /// Get the embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.model.embedding_dim()
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
