//! GLiNER configuration parsing from gliner_config.json.

use serde::Deserialize;

/// GLiNER model configuration parsed from gliner_config.json.
///
/// This differs from standard HuggingFace config.json and contains
/// GLiNER-specific parameters.
#[derive(Debug, Clone, Deserialize)]
pub struct GlinerConfig {
    /// Text encoder model name (e.g., "jhu-clsp/ettin-encoder-32m")
    #[serde(default)]
    pub model_name: Option<String>,

    /// Label encoder model name (e.g., "sentence-transformers/all-MiniLM-L6-v2")
    /// If None, falls back to uni-encoder mode.
    #[serde(default)]
    pub labels_encoder: Option<String>,

    /// Maximum span width in words (default: 12)
    #[serde(default = "default_max_width")]
    pub max_width: usize,

    /// Hidden dimension for span and label FFNs
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,

    /// Dropout rate for FFN layers
    #[serde(default = "default_dropout")]
    pub dropout: f32,

    /// Subtoken pooling strategy: "first" or "mean"
    #[serde(default = "default_subtoken_pooling")]
    pub subtoken_pooling: String,

    /// Whether to enable cross-attention fusion (false for bi-encoder)
    #[serde(default)]
    pub fuse_layers: bool,

    /// Post-fusion schema (empty string for bi-encoder)
    #[serde(default)]
    pub post_fusion_schema: String,

    /// Span representation mode: "markerV0" for span-level
    #[serde(default = "default_span_mode")]
    pub span_mode: String,

    /// Index of the CLS token (typically 0, -1 means last token)
    #[serde(default, deserialize_with = "deserialize_token_index")]
    pub class_token_index: Option<usize>,

    /// Vocabulary size for output classes (-1 means use default)
    #[serde(default, deserialize_with = "deserialize_optional_size")]
    pub vocab_size: Option<usize>,
}

fn deserialize_token_index<'de, D>(deserializer: D) -> Result<Option<usize>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value: i64 = i64::deserialize(deserializer)?;
    if value < 0 {
        Ok(None) // -1 means last token or not applicable
    } else {
        Ok(Some(value as usize))
    }
}

fn deserialize_optional_size<'de, D>(deserializer: D) -> Result<Option<usize>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value: i64 = i64::deserialize(deserializer)?;
    if value < 0 {
        Ok(None) // -1 means use default or not applicable
    } else {
        Ok(Some(value as usize))
    }
}

fn default_max_width() -> usize {
    12
}

fn default_hidden_size() -> usize {
    768
}

fn default_dropout() -> f32 {
    0.4
}

fn default_subtoken_pooling() -> String {
    "first".to_string()
}

fn default_span_mode() -> String {
    "markerV0".to_string()
}

fn default_vocab_size() -> usize {
    2
}

impl GlinerConfig {
    /// Parse config from JSON bytes.
    pub fn from_json(json: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(json)
    }

    /// Check if this is a bi-encoder configuration.
    pub fn is_bi_encoder(&self) -> bool {
        self.labels_encoder.is_some() && self.post_fusion_schema.is_empty()
    }

    /// Check if span mode is markerV0.
    pub fn is_marker_v0(&self) -> bool {
        self.span_mode == "markerV0"
    }

    /// Check if subtoken pooling uses first token.
    pub fn uses_first_subtoken(&self) -> bool {
        self.subtoken_pooling == "first"
    }

    /// Whether the tokenizer should add [CLS]/[SEP] special tokens around text.
    ///
    /// Matches Python GLiNER's `_set_tokenizer_spec_tokens` behavior:
    /// ModernBERT/ettin-style encoders (which have `add_bos_token=False`
    /// semantics because they have no bos token) are fed raw text without
    /// [CLS]/[SEP] wrappers. DeBERTa/RoBERTa/XLM-R family keep them.
    pub fn should_add_special_tokens(&self) -> bool {
        match self.model_name.as_deref() {
            Some(name) => {
                let lower = name.to_ascii_lowercase();
                !(lower.contains("ettin")
                    || lower.contains("modernbert")
                    || lower.contains("modern-bert"))
            }
            None => true,
        }
    }
}

impl Default for GlinerConfig {
    fn default() -> Self {
        Self {
            model_name: None,
            labels_encoder: None,
            max_width: default_max_width(),
            hidden_size: default_hidden_size(),
            dropout: default_dropout(),
            subtoken_pooling: default_subtoken_pooling(),
            fuse_layers: false,
            post_fusion_schema: String::new(),
            span_mode: default_span_mode(),
            class_token_index: Some(0),
            vocab_size: Some(default_vocab_size()),
        }
    }
}
