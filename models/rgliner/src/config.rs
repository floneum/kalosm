//! GLiNER configuration parsing from gliner_config.json.

use serde::Deserialize;

/// GLiNER model configuration parsed from gliner_config.json.
///
/// This differs from standard HuggingFace config.json and contains
/// GLiNER-specific parameters. Unknown fields are ignored, so only the
/// values rgliner actually consumes are declared here.
#[derive(Debug, Clone, Deserialize)]
pub struct GlinerConfig {
    /// Text encoder model name (e.g., "jhu-clsp/ettin-encoder-32m")
    #[serde(default)]
    pub model_name: Option<String>,

    /// Maximum span width in words (default: 12)
    #[serde(default = "default_max_width")]
    pub max_width: usize,
}

fn default_max_width() -> usize {
    12
}

impl GlinerConfig {
    /// Parse config from JSON bytes.
    pub fn from_json(json: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(json)
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
