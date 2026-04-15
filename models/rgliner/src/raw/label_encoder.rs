//! Label encoder using sentence transformers.

use fusor::{Device, Result, Tensor, VarBuilder};
use kalosm_language_model::Embedding;
use rbert::{Bert, BertSource, Pooling};
use std::sync::Arc;

use crate::error::GlinerError;

/// Projection FFN for aligning label embeddings to text encoder dimension.
///
/// Architecture: Linear(hidden, hidden*4) -> ReLU -> Linear(hidden*4, hidden)
/// This matches the Python create_projection_layer() function in GLiNER.
pub struct ProjectionFFN {
    weight1: Tensor<2, f32>,
    bias1: Tensor<1, f32>,
    weight2: Tensor<2, f32>,
    bias2: Tensor<1, f32>,
}

impl ProjectionFFN {
    /// Load projection FFN from weights with proper transposition.
    pub fn load(device: &Device, vb: &mut VarBuilder<'_>) -> Result<Self> {
        // Try different naming conventions
        let (weight1, bias1) =
            Self::load_layer(device, vb, &["label_fnn.0", "label_ffn.0", "label_proj.0"])?;
        let (weight2, bias2) =
            Self::load_layer(device, vb, &["label_fnn.2", "label_ffn.2", "label_proj.2"])?;

        Ok(Self {
            weight1,
            bias1,
            weight2,
            bias2,
        })
    }

    fn load_layer(
        device: &Device,
        vb: &mut VarBuilder,
        prefixes: &[&str],
    ) -> Result<(Tensor<2, f32>, Tensor<1, f32>)> {
        for prefix in prefixes {
            let mut layer_vb = vb.pp(prefix);
            if let Ok(weight_q) = layer_vb.get("weight", device) {
                let weight: Tensor<2, f32> = weight_q.dequantize();

                // PyTorch nn.Linear stores weights as [out_features, in_features]
                // GGUF stores the same way. Fusor loads as-is.
                // For x @ W where x is [B, in], we need W to be [in, out]
                // So we transpose [out, in] -> [in, out]
                let weight_t = weight.t().to_concrete();

                if let Ok(bias_q) = layer_vb.get("bias", device) {
                    let bias: Tensor<1, f32> = bias_q.dequantize();
                    return Ok((weight_t, bias));
                }
            }
        }
        Err(fusor::Error::msg(format!(
            "Could not load projection layer with prefixes {:?}",
            prefixes
        )))
    }

    /// Get output dimension.
    pub fn out_features(&self) -> usize {
        // After transpose, weight2 is [in=1536, out=384], so output dim is shape[1]
        self.weight2.shape()[1]
    }

    /// Forward pass through projection.
    /// Computes: ReLU(x @ W1 + b1) @ W2 + b2
    pub fn forward(&self, x: &Tensor<2, f32>) -> Tensor<2, f32> {
        // Layer 1: x @ W1 + b1
        // x is [num_labels, 384], W1 is [384, 1536] after transpose
        // So x @ W1 = [num_labels, 384] @ [384, 1536] = [num_labels, 1536]
        let h1 = x.mat_mul(&self.weight1);

        let [num_labels, hidden_dim] = h1.shape();
        let bias1_broadcast: Tensor<2, f32> = self
            .bias1
            .unsqueeze(0)
            .to_concrete()
            .broadcast_as([num_labels, hidden_dim])
            .to_concrete();
        let h1_biased = (h1 + bias1_broadcast).to_concrete();

        // ReLU activation (Python GLiNER uses ReLU, not GELU)
        let h1_relu = h1_biased.relu();

        // Layer 2: h @ W2 + b2
        // h is [num_labels, 1536], W2 is [1536, 384] after transpose
        // So h @ W2 = [num_labels, 1536] @ [1536, 384] = [num_labels, 384]
        let out = h1_relu.mat_mul(&self.weight2);
        let [num_labels2, out_dim] = out.shape();
        let bias2_broadcast: Tensor<2, f32> = self
            .bias2
            .unsqueeze(0)
            .to_concrete()
            .broadcast_as([num_labels2, out_dim])
            .to_concrete();
        (out + bias2_broadcast).to_concrete()
    }
}

/// Label encoder: sentence transformer + projection FFN.
pub struct LabelEncoder {
    /// Sentence transformer model (reuses rbert).
    sentence_encoder: Arc<Bert>,
    /// Projection FFN to align dimensions.
    projection: ProjectionFFN,
    /// Output dimension.
    output_dim: usize,
    /// Device for creating tensors.
    device: Device,
}

impl LabelEncoder {
    /// Load label encoder from separate GGUF file.
    pub async fn load(
        device: &Device,
        projection_vb: &mut VarBuilder<'_>,
        sentence_encoder_source: BertSource,
    ) -> std::result::Result<Self, crate::error::GlinerLoadingError> {
        // Load sentence encoder from separate model
        let sentence_encoder = Bert::builder()
            .with_source(sentence_encoder_source)
            .with_device(device.clone())
            .build()
            .await?;

        let projection = ProjectionFFN::load(device, projection_vb)?;
        let output_dim = projection.out_features();

        Ok(Self {
            sentence_encoder: Arc::new(sentence_encoder),
            projection,
            output_dim,
            device: device.clone(),
        })
    }

    #[cfg(test)]
    pub async fn debug_sentence_embeddings(
        &self,
        labels: &[&str],
    ) -> std::result::Result<Tensor<2, f32>, GlinerError> {
        let embeddings = self
            .sentence_encoder
            .embed_batch_with_pooling_and_normalization(labels.to_vec(), Pooling::Mean, false)
            .await?;
        Ok(self.embeddings_to_tensor(&embeddings))
    }

    #[cfg(test)]
    pub fn debug_projection(&self, x: &Tensor<2, f32>) -> Tensor<2, f32> {
        self.projection.forward(x)
    }

    #[cfg(test)]
    pub fn debug_sentence_token_embeddings_and_mask(
        &self,
        labels: &[&str],
    ) -> std::result::Result<(Tensor<3, f32>, Tensor<2, u32>), GlinerError> {
        self.sentence_encoder
            .debug_batch_forward(labels.iter().map(|s| (*s).to_string()).collect())
            .map_err(Into::into)
    }

    #[cfg(test)]
    pub fn debug_sentence_mean_pool(
        &self,
        labels: &[&str],
    ) -> std::result::Result<Tensor<2, f32>, GlinerError> {
        self.sentence_encoder
            .debug_batch_mean_pool(labels.iter().map(|s| (*s).to_string()).collect(), false)
            .map_err(Into::into)
    }

    #[cfg(test)]
    pub fn debug_sentence_hidden_states(
        &self,
        labels: &[&str],
    ) -> std::result::Result<(Vec<Tensor<3, f32>>, Tensor<2, u32>), GlinerError> {
        self.sentence_encoder
            .debug_batch_hidden_states(labels.iter().map(|s| (*s).to_string()).collect())
            .map_err(Into::into)
    }

    #[cfg(test)]
    pub fn debug_sentence_first_layer(
        &self,
        labels: &[&str],
    ) -> std::result::Result<(Tensor<3, f32>, Tensor<3, f32>, Tensor<3, f32>), GlinerError> {
        self.sentence_encoder
            .debug_batch_first_layer(labels.iter().map(|s| (*s).to_string()).collect())
            .map_err(Into::into)
    }

    #[cfg(test)]
    pub fn debug_sentence_first_layer_attention(
        &self,
        labels: &[&str],
    ) -> std::result::Result<
        (
            Tensor<4, f32>,
            Tensor<4, f32>,
            Tensor<4, f32>,
            Tensor<3, f32>,
            Tensor<3, f32>,
        ),
        GlinerError,
    > {
        self.sentence_encoder
            .debug_batch_first_layer_attention(labels.iter().map(|s| (*s).to_string()).collect())
            .map_err(Into::into)
    }

    /// Encode labels to embeddings.
    ///
    /// # Arguments
    /// * `labels` - Label strings to encode
    ///
    /// # Returns
    /// Label embeddings [num_labels, output_dim]
    pub async fn encode_labels(
        &self,
        labels: &[&str],
    ) -> std::result::Result<Tensor<2, f32>, GlinerError> {
        if labels.is_empty() {
            return Ok(Tensor::zeros(&self.device, [0, self.output_dim]));
        }

        // Python GLiNER mean-pools label tokens without the L2 normalization that
        // rbert applies in its default embedding API.
        let embeddings = self
            .sentence_encoder
            .embed_batch_with_pooling_and_normalization(labels.to_vec(), Pooling::Mean, false)
            .await?;

        // Convert Embeddings to tensor
        let label_tensor = self.embeddings_to_tensor(&embeddings);

        // Project to text encoder dimension using the label_fnn
        let projected = self.projection.forward(&label_tensor);

        // Return projected embeddings without normalization
        // The model was trained end-to-end with this projection
        Ok(projected)
    }

    /// Convert Vec<Embedding> to Tensor<2, f32>
    fn embeddings_to_tensor(&self, embeddings: &[Embedding]) -> Tensor<2, f32> {
        if embeddings.is_empty() {
            return Tensor::zeros(&self.device, [0, self.output_dim]);
        }

        let num_labels = embeddings.len();
        let embed_dim = embeddings[0].vector().len();

        // Flatten all embeddings into a single Vec
        let mut data: Vec<f32> = Vec::with_capacity(num_labels * embed_dim);
        for emb in embeddings {
            data.extend_from_slice(emb.vector());
        }

        // Create tensor from flat data
        Tensor::new(&self.device, &data)
            .reshape([num_labels, embed_dim])
            .to_concrete()
    }
}

/// Cached label embeddings for efficient repeated inference.
pub struct CachedLabels {
    /// Original label strings.
    pub labels: Vec<String>,
    /// Precomputed label embeddings [num_labels, hidden_dim].
    pub embeddings: Tensor<2, f32>,
}

impl CachedLabels {
    /// Create cached labels from precomputed embeddings.
    pub fn new(labels: Vec<String>, embeddings: Tensor<2, f32>) -> Self {
        Self { labels, embeddings }
    }
}
