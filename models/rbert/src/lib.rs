//! # rbert
//!
//! A Rust embedding model library supporting BERT and Qwen architectures.
//!
//! ## Usage
//!
//! ```rust, no_run
//! use kalosm_language_model::Embedder;
//! use rbert::*;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let mut bert = Bert::new().await?;
//!     let sentences = [
//!         "Cats are cool",
//!         "The geopolitical situation is dire",
//!         "Pets are great",
//!         "Napoleon was a tyrant",
//!         "Napoleon was a great general",
//!     ];
//!     let embeddings = bert.embed_batch(sentences).await?;
//!     println!("embeddings {:?}", embeddings);
//!
//!     // Find the cosine similarity between the first two sentences
//!     let mut similarities = vec![];
//!     let n_sentences = sentences.len();
//!     for (i, e_i) in embeddings.iter().enumerate() {
//!         for j in (i + 1)..n_sentences {
//!             let e_j = embeddings.get(j).unwrap();
//!             let cosine_similarity = e_j.cosine_similarity(e_i);
//!             similarities.push((cosine_similarity, i, j))
//!         }
//!     }
//!     similarities.sort_by(|u, v| v.0.total_cmp(&u.0));
//!     for &(score, i, j) in similarities.iter() {
//!         println!("score: {score:.2} '{}' '{}'", sentences[i], sentences[j])
//!     }
//!
//!     Ok(())
//! }
//! ```

#![warn(missing_docs)]

use std::sync::{Arc, RwLock};

use fusor2::tensor::Dyn as Tensor;
use fusor2::{Dim, Dtype, Result as FusorResult, VarBuilder};
use kalosm_common::*;
use kalosm_model_types::ModelLoadingProgress;
use tokenizers::{Encoding, PaddingDirection, PaddingParams, Tokenizer};

/// The device handle models run on, re-exported from fusor2.
pub use fusor2::device::Device;

mod language_model;
mod raw;
mod source;

pub use crate::language_model::*;
pub use crate::raw::{BertModel, Config, QwenEmbeddingModel};
pub use crate::source::*;

/// A builder for a [`Bert`] model
#[derive(Default)]
pub struct BertBuilder {
    source: BertSource,
    cache: kalosm_common::Cache,
    device: Option<Device>,
}

impl BertBuilder {
    /// Set the source of the model
    pub fn with_source(mut self, source: BertSource) -> Self {
        self.source = source;
        self
    }

    /// Build the model
    pub async fn build(self) -> Result<Bert, BertLoadingError> {
        self.build_with_loading_handler(ModelLoadingProgress::multi_bar_loading_indicator())
            .await
    }

    /// Set the device to use for the model. If not set, the best available device is selected automatically.
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    /// Set the cache location to use for the model (defaults DATA_DIR/kalosm/cache)
    pub fn with_cache(mut self, cache: kalosm_common::Cache) -> Self {
        self.cache = cache;

        self
    }

    /// Build the model with a loading handler
    ///
    /// ```rust, no_run
    /// use kalosm::language::*;
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), anyhow::Error> {
    /// // Create a new bert model with a loading handler
    /// let model = Bert::builder()
    ///     .build_with_loading_handler(|progress| match progress {
    ///         ModelLoadingProgress::Downloading { source, progress } => {
    ///             let progress_percent = (progress.progress * 100) as u32;
    ///             let elapsed = progress.start_time.unwrap().elapsed().as_secs_f32();
    ///             println!("Downloading file {source} {progress_percent}% ({elapsed}s)");
    ///         }
    ///         ModelLoadingProgress::Loading { progress } => {
    ///             let progress = (progress * 100.0) as u32;
    ///             println!("Loading model {progress}%");
    ///         }
    ///     })
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn build_with_loading_handler(
        self,
        loading_handler: impl FnMut(ModelLoadingProgress) + Send + 'static,
    ) -> Result<Bert, BertLoadingError> {
        Bert::from_builder(self, loading_handler).await
    }
}

/// An error that can occur when loading a Bert model.
#[derive(Debug, thiserror::Error)]
pub enum BertLoadingError {
    /// An error that can occur when trying to load a Bert model from huggingface or a local file.
    #[error("Failed to load model from huggingface or local file: {0}")]
    DownloadingError(#[from] CacheError),
    /// An error that can occur when trying to load a Bert model.
    #[error("Failed to load model into device: {0}")]
    LoadModel(#[from] fusor2::Error),
    /// An IO error that can occur when trying to load a bert model.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// An error that can occur when trying to load the bert tokenizer.
    #[error("Failed to load tokenizer: {0}")]
    LoadTokenizer(tokenizers::Error),
    /// An error that can occur when trying to load the bert config.
    #[error("Failed to load config: {0}")]
    LoadConfig(serde_json::Error),
    /// A config was not found
    #[error("Config not found")]
    ConfigNotFound,
}

/// An error that can occur when running a Bert model.
#[derive(Debug, thiserror::Error)]
pub enum BertError {
    /// An error that can occur when trying to run a Bert model.
    #[error("Failed to run model: {0}")]
    Fusor(#[from] fusor2::Error),
    /// An error that can occur when tokenizing or detokenizing text.
    #[error("Failed to tokenize: {0}")]
    TokenizerError(tokenizers::Error),
}

/// The pooling strategy to use when embedding text.
#[derive(Debug, Clone, Copy)]
pub enum Pooling {
    /// Take the mean embedding value for all tokens (except padding)
    Mean,
    /// Take the embedding of the CLS token for each sequence
    CLS,
    /// Take the embedding of the last token for each sequence (used by Qwen)
    Last,
}

/// An embedding model that can be either BERT or Qwen
pub enum EmbeddingModel {
    /// A BERT-style embedding model
    Bert(BertModel),
    /// A Qwen-style embedding model
    Qwen(QwenEmbeddingModel),
}

impl EmbeddingModel {
    /// Get the device the model is running on
    pub fn device(&self) -> &Device {
        match self {
            EmbeddingModel::Bert(model) => &model.device,
            EmbeddingModel::Qwen(model) => &model.device,
        }
    }

    /// Get the maximum sequence length
    pub fn max_seq_len(&self) -> usize {
        match self {
            EmbeddingModel::Bert(model) => model.max_seq_len(),
            EmbeddingModel::Qwen(model) => model.max_seq_len(),
        }
    }

    /// Get the embedding dimension
    pub fn embedding_dim(&self) -> usize {
        match self {
            EmbeddingModel::Bert(model) => model.embedding_dim(),
            EmbeddingModel::Qwen(model) => model.embedding_dim(),
        }
    }

    /// Get the default pooling strategy for this model
    pub fn default_pooling(&self) -> Pooling {
        match self {
            EmbeddingModel::Bert(_) => Pooling::CLS,
            EmbeddingModel::Qwen(_) => Pooling::Last,
        }
    }

    /// Forward pass through the model
    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> FusorResult<Tensor> {
        match self {
            EmbeddingModel::Bert(model) => {
                let token_type_ids = input_ids.zeros_like()?;
                model.forward(input_ids, &token_type_ids, attention_mask)
            }
            EmbeddingModel::Qwen(model) => model.forward(input_ids, attention_mask),
        }
    }
}

/// An embedding model supporting BERT and Qwen architectures.
/// The main interface for this model is [`EmbedderExt`].
///
/// # Example
/// ```rust, no_run
/// use kalosm_language_model::Embedder;
/// use rbert::*;
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     let mut bert = Bert::new().await?;
///     let sentences = [
///         "Cats are cool",
///         "The geopolitical situation is dire",
///         "Pets are great",
///         "Napoleon was a tyrant",
///         "Napoleon was a great general",
///     ];
///     let embeddings = bert.embed_batch(sentences).await?;
///     println!("embeddings {:?}", embeddings);
///
///     // Find the cosine similarity between the first two sentences
///     let mut similarities = vec![];
///     let n_sentences = sentences.len();
///     for (i, e_i) in embeddings.iter().enumerate() {
///         for j in (i + 1)..n_sentences {
///             let e_j = embeddings.get(j).unwrap();
///             let cosine_similarity = e_j.cosine_similarity(e_i);
///             similarities.push((cosine_similarity, i, j))
///         }
///     }
///     similarities.sort_by(|u, v| v.0.total_cmp(&u.0));
///     for &(score, i, j) in similarities.iter() {
///         println!("score: {score:.2} '{}' '{}'", sentences[i], sentences[j])
///     }
///
///     Ok(())
/// }
/// ```
#[derive(Clone)]
pub struct Bert {
    embedding_search_prefix: Arc<Option<String>>,
    model: Arc<EmbeddingModel>,
    tokenizer: Arc<RwLock<Tokenizer>>,
}

impl Bert {
    /// Create a new [`BertBuilder`]
    pub fn builder() -> BertBuilder {
        BertBuilder::default()
    }

    /// Create a new default bert model
    pub async fn new() -> Result<Self, BertLoadingError> {
        Self::builder().build().await
    }

    /// Create a new default bert model for search
    pub async fn new_for_search() -> Result<Self, BertLoadingError> {
        Self::builder()
            .with_source(BertSource::new_for_search())
            .build()
            .await
    }

    async fn from_builder(
        builder: BertBuilder,
        mut progress_handler: impl FnMut(ModelLoadingProgress) + Send + 'static,
    ) -> Result<Self, BertLoadingError> {
        let BertBuilder {
            source,
            cache,
            device,
        } = builder;
        let BertSource {
            config,
            tokenizer,
            model,
            search_embedding_prefix,
        } = source;

        let config_bytes = if let Some(config) = config {
            let source = format!("Config ({config})");
            let mut create_progress = ModelLoadingProgress::downloading_progress(source);
            Some(
                cache
                    .get_bytes(&config, |progress| {
                        progress_handler(create_progress(progress))
                    })
                    .await?,
            )
        } else {
            None
        };
        let tokenizer_source = format!("Tokenizer ({tokenizer})");
        let mut create_progress = ModelLoadingProgress::downloading_progress(tokenizer_source);
        let tokenizer_bytes = cache
            .get_bytes(&tokenizer, |progress| {
                progress_handler(create_progress(progress))
            })
            .await?;
        let model_source = format!("Model ({model})");
        let mut create_progress = ModelLoadingProgress::downloading_progress(model_source);
        let weights_bytes = cache
            .get_bytes(&model, |progress| {
                progress_handler(create_progress(progress))
            })
            .await?;

        let device = match device {
            Some(device) => device,
            None => match Device::gpu().await {
                Ok(device) => device,
                Err(_) => Device::cpu(),
            },
        };
        let gguf = fusor2_gguf::parse::Gguf::from_bytes(weights_bytes)
            .map_err(BertLoadingError::LoadModel)?;
        let vb = VarBuilder::new(Arc::new(gguf));

        // Detect architecture from GGUF metadata
        let architecture = vb.architecture().map(|a| a.to_string());
        let model = match architecture.as_deref() {
            Some("qwen3") | Some("qwen2") => {
                // Load Qwen embedding model
                let qwen_model = QwenEmbeddingModel::load(&device, &vb)?;
                EmbeddingModel::Qwen(qwen_model)
            }
            _ => {
                // Load BERT model (default)
                let config_bytes = config_bytes.ok_or(BertLoadingError::ConfigNotFound)?;
                let config: Config =
                    serde_json::from_slice(&config_bytes).map_err(BertLoadingError::LoadConfig)?;
                let bert_model = BertModel::load(&device, &vb, &config)?;
                EmbeddingModel::Bert(bert_model)
            }
        };

        let mut tokenizer =
            Tokenizer::from_bytes(&tokenizer_bytes).map_err(BertLoadingError::LoadTokenizer)?;
        tokenizer.with_padding(None);

        Ok(Bert {
            tokenizer: Arc::new(RwLock::new(tokenizer)),
            model: Arc::new(model),
            embedding_search_prefix: Arc::new(search_embedding_prefix),
        })
    }

    /// Embed a batch of sentences
    pub(crate) fn embed_batch_raw(
        &self,
        sentences: Vec<&str>,
        pooling: Pooling,
    ) -> Result<Vec<Tensor>, BertError> {
        let embedding_dim = self.model.embedding_dim();
        // Approximates the quadratic attention memory cost (seq_len^2).
        // Batches are split so that total work stays below this threshold.
        const MAX_BATCH_TOKENS_SQUARED: usize = 512 * 512;
        let limit = embedding_dim * MAX_BATCH_TOKENS_SQUARED * 2;

        // The sentences we are embedding may have a very different length. First we sort them so that similar length sentences are grouped together in the same batch to reduce the overhead of padding.
        let encodings = {
            let tokenizer_read = self.tokenizer.read().unwrap();
            tokenizer_read.encode_batch(sentences, true)
        }
        .map_err(BertError::TokenizerError)?;
        let mut encodings_with_indices = encodings.into_iter().enumerate().collect::<Vec<_>>();

        encodings_with_indices.sort_unstable_by_key(|(_, encoding)| encoding.len());

        let mut combined: Vec<Option<Tensor>> = vec![None; encodings_with_indices.len()];
        let mut chunks = Vec::new();
        let mut current_chunk_len = 0;
        let mut current_chunk_max_token_len = 0;
        let mut current_chunk_indices = Vec::new();
        let mut current_chunk_text: Vec<Encoding> = Vec::new();
        for (index, encoding) in encodings_with_indices {
            let len = encoding.get_ids().len();
            current_chunk_max_token_len = current_chunk_max_token_len.max(len);
            current_chunk_len += 1;
            let score = current_chunk_len
                * (embedding_dim * 8 + embedding_dim * current_chunk_max_token_len.pow(2));
            if score > limit {
                chunks.push((
                    std::mem::take(&mut current_chunk_indices),
                    std::mem::take(&mut current_chunk_text),
                ));
                current_chunk_max_token_len = len;
                current_chunk_len = 1;
            }
            current_chunk_indices.push(index);
            current_chunk_text.push(encoding);
        }
        // Add the last chunk even if the score isn't maxed out
        if !current_chunk_indices.is_empty() {
            chunks.push((
                std::mem::take(&mut current_chunk_indices),
                std::mem::take(&mut current_chunk_text),
            ));
        }

        for (indices, encodings) in chunks {
            let embeddings = self.embed_batch_raw_inner(encodings, pooling)?;
            for (i, embedding) in indices.iter().zip(embeddings) {
                combined[*i] = Some(embedding);
            }
        }
        Ok(combined.into_iter().map(|x| x.unwrap()).collect())
    }

    fn embed_batch_raw_inner(
        &self,
        mut tokens: Vec<Encoding>,
        pooling: Pooling,
    ) -> Result<Vec<Tensor>, BertError> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        let graph = self.model.device().graph().handle().clone();
        // Qwen models use last-token pooling and require left padding so the
        // last position always contains the final content token.
        let padding_direction = match pooling {
            Pooling::Last => PaddingDirection::Left,
            _ => PaddingDirection::Right,
        };
        let pp = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            direction: padding_direction,
            ..Default::default()
        };
        tokenizers::pad_encodings(&mut tokens, &pp).map_err(BertError::TokenizerError)?;

        let n_sentences = tokens.len();
        let max_seq_len = self.model.max_seq_len();
        let token_ids = tokens
            .iter()
            .map(|tokens| {
                let ids = tokens.get_ids();
                let ids = &ids[..max_seq_len.min(ids.len())];
                Tensor::from_elements(&graph, &[Dim::Const(ids.len() as u64)], ids)
            })
            .collect::<FusorResult<Vec<_>>>()?;
        let token_ids = Tensor::stack(&token_ids, 0)?;

        let attention_masks = tokens
            .iter()
            .map(|tokens| {
                let mask = tokens.get_attention_mask();
                let mask = &mask[..max_seq_len.min(mask.len())];
                Tensor::from_elements(&graph, &[Dim::Const(mask.len() as u64)], mask)
            })
            .collect::<FusorResult<Vec<_>>>()?;
        let attention_mask = Tensor::stack(&attention_masks, 0)?;

        let embeddings = self.model.forward(&token_ids, Some(&attention_mask))?;

        let n_tokens = embeddings
            .dim(1)
            .as_const()
            .expect("the sequence length is const") as usize;

        let pooled = match pooling {
            Pooling::Mean => {
                // Take the mean embedding value for all tokens (except padding)
                // Cast mask u32→f32, unsqueeze to [batch, seq_len, 1] for broadcasting
                let mask_f32 = attention_mask.cast(Dtype::F32)?;
                let mask_3d = mask_f32.unsqueeze(2)?;
                // Zero out padding positions, sum along seq dim → [batch, hidden_dim]
                let summed = embeddings.mul_(&mask_3d)?.sum(1)?;
                // Divide by valid token count [batch, 1] — broadcasts with [batch, hidden_dim]
                let valid_count = mask_f32.sum_keepdim(1)?;
                let embeddings = summed.div_(&valid_count)?;
                normalize_l2(&embeddings)?
            }
            Pooling::CLS => {
                // Index into the first token of each sentence which is the CLS token that contains the sentence embedding
                embeddings.narrow(1, 0, 1)?.squeeze(1)?
            }
            Pooling::Last => {
                // With left padding, the last token is always at the final position
                let last = embeddings.narrow(1, n_tokens - 1, 1)?.squeeze(1)?;
                normalize_l2(&last)?
            }
        };
        Ok(pooled.chunk(n_sentences, 0)?)
    }
}

fn normalize_l2(v: &Tensor) -> FusorResult<Tensor> {
    let sum_sq = v.sqr()?.sum_keepdim(1)?;
    v.div_(&sum_sq.add_scalar(1e-12f32)?.sqrt()?)
}
