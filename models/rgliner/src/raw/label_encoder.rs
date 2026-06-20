//! Label encoder using sentence transformers.

use fusor::layers::Linear;
use fusor::{Device, Result, Tensor, VarBuilder};
use kalosm_language_model::Embedding;
use rbert::{Bert, BertSource, Pooling};
use std::sync::Arc;

use crate::error::GlinerError;

/// Projection FFN for aligning label embeddings to text encoder dimension.
///
/// Architecture: Linear(hidden, hidden*4) -> ReLU -> Linear(hidden*4, hidden).
/// This matches the Python `create_projection_layer()` function in GLiNER.
/// Both layers reuse [`fusor::layers::Linear`], which handles the `[out, in]`
/// weight layout, bias, and (de)quantization the same way as the model's other
/// heads — no manual transpose or whole-weight dequantization.
pub struct ProjectionFFN {
    fc1: Linear<f32>,
    fc2: Linear<f32>,
}

impl ProjectionFFN {
    /// Load projection FFN, trying the known weight-name conventions.
    pub fn load(device: &Device, vb: &mut VarBuilder<'_>) -> Result<Self> {
        let fc1 = Self::load_layer(device, vb, &["label_fnn.0", "label_ffn.0", "label_proj.0"])?;
        let fc2 = Self::load_layer(device, vb, &["label_fnn.2", "label_ffn.2", "label_proj.2"])?;
        Ok(Self { fc1, fc2 })
    }

    fn load_layer(device: &Device, vb: &mut VarBuilder, prefixes: &[&str]) -> Result<Linear<f32>> {
        for prefix in prefixes {
            if let Ok(linear) = Linear::load(device, &mut vb.pp(prefix)) {
                return Ok(linear);
            }
        }
        Err(fusor::Error::msg(format!(
            "Could not load projection layer with prefixes {prefixes:?}"
        )))
    }

    /// Get output dimension.
    pub fn out_features(&self) -> usize {
        self.fc2.out_features()
    }

    /// Forward pass through projection: `ReLU(x @ W1.T + b1) @ W2.T + b2`.
    /// (Python GLiNER uses ReLU, not GELU.)
    pub fn forward(&self, x: &Tensor<2, f32>) -> Tensor<2, f32> {
        let h1 = self.fc1.forward_2d(x).relu();
        self.fc2.forward_2d(&h1)
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
