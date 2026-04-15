//! Error types for rgliner.

use kalosm_common::CacheError;

/// An error that can occur when loading a GLiNER model.
#[derive(Debug, thiserror::Error)]
pub enum GlinerLoadingError {
    /// An error that can occur when trying to download model files.
    #[error("Failed to download model files: {0}")]
    DownloadingError(#[from] CacheError),
    /// An error that can occur when trying to load the model.
    #[error("Failed to load model: {0}")]
    LoadModel(#[from] fusor::Error),
    /// An IO error.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// An error that can occur when trying to load the tokenizer.
    #[error("Failed to load tokenizer: {0}")]
    LoadTokenizer(tokenizers::Error),
    /// An error that can occur when trying to load the config.
    #[error("Failed to load config: {0}")]
    LoadConfig(serde_json::Error),
    /// Config file not found.
    #[error("Config file not found")]
    ConfigNotFound,
    /// Label encoder loading error.
    #[error("Failed to load label encoder: {0}")]
    LabelEncoder(#[from] rbert::BertLoadingError),
}

/// An error that can occur when running GLiNER inference.
#[derive(Debug, thiserror::Error)]
pub enum GlinerError {
    /// An error that can occur when running tensor operations.
    #[error("Tensor operation error: {0}")]
    Fusor(#[from] fusor::Error),
    /// An error that can occur when tokenizing text.
    #[error("Tokenization error: {0}")]
    Tokenizer(tokenizers::Error),
    /// A tokenization error with a string message.
    #[error("Tokenization error: {0}")]
    TokenizationError(String),
    /// An error that can occur with the label encoder.
    #[error("Label encoder error: {0}")]
    LabelEncoder(#[from] rbert::BertError),
}
