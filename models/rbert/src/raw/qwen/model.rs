use fusor2::layers::{Embedding, RmsNorm};
use fusor2::{Device, Dim, Dtype, Error, QMatrix, Result, Tensor, VarBuilder};

use super::layer::QwenLayer;

/// Configuration for QwenEmbeddingModel loaded from GGUF metadata
#[derive(Debug, Clone)]
pub struct QwenConfig {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub head_dimension: usize,
    pub context_length: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
}

impl QwenConfig {
    /// Load configuration from GGUF metadata
    pub fn from_gguf(vb: &VarBuilder) -> Result<Self> {
        let num_heads = vb
            .get_metadata(".attention.head_count")
            .and_then(|v| v.to_u32().ok())
            .ok_or_else(|| {
                Error::Io("Missing required GGUF metadata: .attention.head_count".into())
            })? as usize;

        let num_kv_heads = vb
            .get_metadata(".attention.head_count_kv")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(num_heads as u32) as usize;

        let num_layers = vb
            .get_metadata(".block_count")
            .and_then(|v| v.to_u32().ok())
            .ok_or_else(|| Error::Io("Missing required GGUF metadata: .block_count".into()))?
            as usize;

        let hidden_size = vb
            .get_metadata(".embedding_length")
            .and_then(|v| v.to_u32().ok())
            .ok_or_else(|| Error::Io("Missing required GGUF metadata: .embedding_length".into()))?
            as usize;

        if !hidden_size.is_multiple_of(num_heads) {
            return Err(Error::Shape(format!(
                "hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )));
        }

        let context_length = vb
            .get_metadata(".context_length")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(32768) as usize;

        let rope_theta = vb
            .get_metadata(".rope.freq_base")
            .and_then(|v| v.to_f32().ok())
            .unwrap_or(1_000_000.0);

        let rms_norm_eps = vb
            .get_metadata(".attention.layer_norm_rms_epsilon")
            .and_then(|v| v.to_f32().ok())
            .unwrap_or(1e-6);

        // Use attention.key_length for head dimension (like kalosm-llama)
        // Fall back to hidden_size / num_heads if not present
        let head_dimension = vb
            .get_metadata(".attention.key_length")
            .and_then(|v| v.to_u32().ok())
            .map(|x| x as usize)
            .unwrap_or_else(|| hidden_size / num_heads);

        Ok(Self {
            num_heads,
            num_kv_heads,
            num_layers,
            hidden_size,
            head_dimension,
            context_length,
            rope_theta,
            rms_norm_eps,
        })
    }
}

/// A `[context, head_dim / 2]` cos/sin table consumed by
/// [`Tensor::rope_pair`].
pub(crate) struct QwenRope {
    pub(crate) cos: Tensor<2>,
    pub(crate) sin: Tensor<2>,
}

impl QwenRope {
    fn new(device: &Device, head_dim: usize, context_length: usize, theta: f32) -> Self {
        let inverse_frequency =
            fusor2::composite::rope::base_inverse_frequency(head_dim as u32, theta);
        let half = inverse_frequency.len();
        let mut sin = Vec::with_capacity(context_length * half);
        let mut cos = Vec::with_capacity(context_length * half);
        for pos in 0..context_length {
            for f in &inverse_frequency {
                // Accumulate the angle in f64: at large positions an f32
                // product has already lost the low bits.
                let angle = pos as f64 * *f as f64;
                sin.push(angle.sin() as f32);
                cos.push(angle.cos() as f32);
            }
        }
        let shape = [context_length, half];
        Self {
            sin: Tensor::from_slice(device, shape, &sin),
            cos: Tensor::from_slice(device, shape, &cos),
        }
    }
}

/// The token table, quantized (read in place through `rows_at`) or dense.
enum TokenEmbedding {
    Quantized(QMatrix),
    Dense(Embedding),
}

impl TokenEmbedding {
    fn load(vb: &VarBuilder, device: &Device) -> Result<Self> {
        let raw = vb.get_raw("token_embd.weight")?;
        if let Dtype::Q(fmt) = raw.fmt {
            let [rows, cols] = match raw.shape.as_slice() {
                [rows, cols] => [*rows, *cols],
                other => {
                    return Err(Error::Shape(format!(
                        "token_embd.weight has GGUF shape {other:?}; expected rank 2"
                    )));
                }
            };
            let q = QMatrix::from_raw_bytes(
                device.graph(),
                fmt,
                raw.layout,
                [Dim::Const(rows), Dim::Const(cols)],
                &raw.bytes,
            )?;
            Ok(Self::Quantized(q))
        } else {
            Ok(Self::Dense(Embedding::load(
                &vb.pp("token_embd"),
                device.graph().handle(),
            )?))
        }
    }

    /// `[batch, seq] -> [batch, seq, hidden]`.
    fn forward(&self, ids: &Tensor<2, u32>) -> Tensor<3> {
        match self {
            Self::Quantized(q) => {
                let [batch, seq] = ids.extents();
                let rows = q.rows_at(&ids.flatten_all());
                let hidden = rows.extent(1);
                rows.reshape_dims([batch, seq, hidden])
            }
            Self::Dense(embedding) => embedding.forward(ids),
        }
    }

    fn hidden_size(&self) -> Option<u64> {
        match self {
            Self::Quantized(q) => q.cols.as_const(),
            Self::Dense(e) => e.embedding_dim().as_const(),
        }
    }
}

/// Qwen embedding model (encoder-only for embeddings)
pub struct QwenEmbeddingModel {
    token_embeddings: TokenEmbedding,
    layers: Vec<QwenLayer>,
    final_norm: RmsNorm,
    rope: QwenRope,
    pub(crate) device: Device,
    config: QwenConfig,
}

impl QwenEmbeddingModel {
    /// Load QwenEmbeddingModel from GGUF weights
    pub fn load(device: &Device, vb: &VarBuilder) -> Result<Self> {
        let config = QwenConfig::from_gguf(vb)?;

        // Load token embeddings
        let token_embeddings = TokenEmbedding::load(vb, device)?;
        debug_assert_eq!(
            token_embeddings.hidden_size(),
            Some(config.hidden_size as u64)
        );

        // Create RoPE table
        let rope = QwenRope::new(
            device,
            config.head_dimension,
            config.context_length,
            config.rope_theta,
        );

        // Load transformer layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let layer = QwenLayer::load(
                device,
                &vb.pp(format!("blk.{i}")),
                config.num_heads,
                config.num_kv_heads,
                config.head_dimension,
                config.rms_norm_eps,
            )?;
            layers.push(layer);
        }

        // Load final layer norm
        let final_norm = RmsNorm::load(
            &vb.pp("output_norm"),
            device.graph().handle(),
            config.rms_norm_eps,
        )?;

        Ok(Self {
            token_embeddings,
            layers,
            final_norm,
            rope,
            device: device.clone(),
            config,
        })
    }

    /// Forward pass through the model
    ///
    /// Returns: [batch_size, seq_len, hidden_size]
    pub fn forward(
        &self,
        input_ids: &Tensor<2, u32>,
        attention_mask: Option<&Tensor<2, u32>>,
    ) -> Tensor<3> {
        // Get token embeddings
        let mut hidden_states = self.token_embeddings.forward(input_ids);

        // Pass through transformer layers
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, &self.rope, attention_mask);
        }

        // Apply final layer norm
        self.final_norm.forward(&hidden_states)
    }

    /// Get the maximum sequence length
    pub fn max_seq_len(&self) -> usize {
        self.config.context_length
    }

    /// Get the embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.config.hidden_size
    }
}
