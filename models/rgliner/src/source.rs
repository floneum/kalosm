//! Model source configuration for GLiNER variants.

use kalosm_model_types::FileSource;
use std::path::{Path, PathBuf};

/// Source configuration for GLiNER models.
///
/// Specifies where to download the model files from.
pub struct GlinerSource {
    /// Main model GGUF file (text encoder + span layer weights)
    pub(crate) model: FileSource,
    /// Label encoder GGUF file (sentence transformer)
    pub(crate) label_encoder: FileSource,
    /// Label encoder config JSON file
    pub(crate) label_encoder_config: FileSource,
    /// Label encoder tokenizer JSON file
    pub(crate) label_encoder_tokenizer: FileSource,
    /// Tokenizer JSON file (for text encoder)
    pub(crate) tokenizer: FileSource,
    /// GLiNER config JSON file
    pub(crate) config: FileSource,
}

impl GlinerSource {
    fn huggingface_or_cached(model_id: &str, revision: &str, file: &str) -> FileSource {
        if let Some(path) = Self::find_cached_hf_file(model_id, revision, file) {
            FileSource::local(path)
        } else {
            FileSource::huggingface(model_id.to_string(), revision.to_string(), file.to_string())
        }
    }

    fn find_cached_hf_file(model_id: &str, revision: &str, file: &str) -> Option<PathBuf> {
        let snapshots_dir = Self::huggingface_cache_dir()?
            .join("hub")
            .join(format!("models--{}", model_id.replace('/', "--")))
            .join("snapshots");

        if !snapshots_dir.exists() {
            return None;
        }

        let file = Path::new(file);

        if revision != "main" {
            let candidate = snapshots_dir.join(revision).join(file);
            if candidate.exists() {
                return Some(candidate);
            }
        }

        let refs_path = snapshots_dir
            .parent()
            .map(|parent| parent.join("refs").join(revision));
        if let Some(refs_path) = refs_path {
            if let Ok(snapshot) = std::fs::read_to_string(&refs_path) {
                let candidate = snapshots_dir.join(snapshot.trim()).join(file);
                if candidate.exists() {
                    return Some(candidate);
                }
            }
        }

        let entries = std::fs::read_dir(&snapshots_dir).ok()?;
        for entry in entries.flatten() {
            let candidate = entry.path().join(file);
            if candidate.exists() {
                return Some(candidate);
            }
        }

        None
    }

    fn huggingface_cache_dir() -> Option<PathBuf> {
        if let Some(hf_home) = std::env::var_os("HF_HOME") {
            return Some(PathBuf::from(hf_home));
        }

        if let Some(xdg_cache) = std::env::var_os("XDG_CACHE_HOME") {
            return Some(PathBuf::from(xdg_cache).join("huggingface"));
        }

        std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache").join("huggingface"))
    }

    /// GLiNER bi-encoder v2.0 Edge variant (60M parameters).
    ///
    /// The smallest and fastest variant, using:
    /// - Text encoder: ettin-encoder-32m
    /// - Label encoder: all-MiniLM-L6-v2
    pub fn edge() -> Self {
        Self {
            model: FileSource::huggingface(
                "knowledgator/gliner-bi-edge-v2.0-gguf".to_string(),
                "main".to_string(),
                "gliner-bi-edge-v2.0-Q8_0.gguf".to_string(),
            ),
            label_encoder: FileSource::huggingface(
                "knowledgator/gliner-bi-edge-v2.0-gguf".to_string(),
                "main".to_string(),
                "label-encoder-Q8_0.gguf".to_string(),
            ),
            label_encoder_config: FileSource::huggingface(
                "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                "main".to_string(),
                "config.json".to_string(),
            ),
            label_encoder_tokenizer: FileSource::huggingface(
                "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                "main".to_string(),
                "tokenizer.json".to_string(),
            ),
            tokenizer: FileSource::huggingface(
                "knowledgator/gliner-bi-edge-v2.0".to_string(),
                "main".to_string(),
                "tokenizer.json".to_string(),
            ),
            config: FileSource::huggingface(
                "knowledgator/gliner-bi-edge-v2.0".to_string(),
                "main".to_string(),
                "gliner_config.json".to_string(),
            ),
        }
    }

    /// Demonthos GLiNER GGUF edge upload.
    ///
    /// Uses the GGUF weights and sidecar tokenizer/config files from
    /// `Demonthos/gliner-gguf`.
    pub fn demonthos_edge() -> Self {
        Self {
            model: Self::huggingface_or_cached(
                "Demonthos/gliner-gguf",
                "main",
                "gliner-edge.gguf",
            ),
            label_encoder: Self::huggingface_or_cached(
                "Demonthos/gliner-gguf",
                "main",
                "gliner-edge-label-encoder.gguf",
            ),
            label_encoder_config: Self::huggingface_or_cached(
                "Demonthos/gliner-gguf",
                "main",
                "label-encoder-config.json",
            ),
            label_encoder_tokenizer: Self::huggingface_or_cached(
                "Demonthos/gliner-gguf",
                "main",
                "label-encoder-tokenizer.json",
            ),
            tokenizer: Self::huggingface_or_cached(
                "Demonthos/gliner-gguf",
                "main",
                "text-tokenizer.json",
            ),
            config: Self::huggingface_or_cached(
                "Demonthos/gliner-gguf",
                "main",
                "text-gliner-config.json",
            ),
        }
    }

    /// GLiNER bi-encoder v2.0 Small variant (108M parameters).
    ///
    /// Good balance of speed and accuracy, using:
    /// - Text encoder: ettin-encoder-68m
    /// - Label encoder: all-MiniLM-L12-v2
    pub fn small() -> Self {
        Self {
            model: FileSource::huggingface(
                "knowledgator/gliner-bi-small-v2.0-gguf".to_string(),
                "main".to_string(),
                "gliner-bi-small-v2.0-Q8_0.gguf".to_string(),
            ),
            label_encoder: FileSource::huggingface(
                "knowledgator/gliner-bi-small-v2.0-gguf".to_string(),
                "main".to_string(),
                "label-encoder-Q8_0.gguf".to_string(),
            ),
            label_encoder_config: FileSource::huggingface(
                "sentence-transformers/all-MiniLM-L12-v2".to_string(),
                "main".to_string(),
                "config.json".to_string(),
            ),
            label_encoder_tokenizer: FileSource::huggingface(
                "sentence-transformers/all-MiniLM-L12-v2".to_string(),
                "main".to_string(),
                "tokenizer.json".to_string(),
            ),
            tokenizer: FileSource::huggingface(
                "knowledgator/gliner-bi-small-v2.0".to_string(),
                "main".to_string(),
                "tokenizer.json".to_string(),
            ),
            config: FileSource::huggingface(
                "knowledgator/gliner-bi-small-v2.0".to_string(),
                "main".to_string(),
                "gliner_config.json".to_string(),
            ),
        }
    }

    /// GLiNER bi-encoder v2.0 Base variant (194M parameters).
    ///
    /// Default variant with good accuracy, using:
    /// - Text encoder: ettin-encoder-150m
    /// - Label encoder: bge-small-en-v1.5
    pub fn base() -> Self {
        Self {
            model: FileSource::huggingface(
                "knowledgator/gliner-bi-base-v2.0-gguf".to_string(),
                "main".to_string(),
                "gliner-bi-base-v2.0-Q8_0.gguf".to_string(),
            ),
            label_encoder: FileSource::huggingface(
                "knowledgator/gliner-bi-base-v2.0-gguf".to_string(),
                "main".to_string(),
                "label-encoder-Q8_0.gguf".to_string(),
            ),
            label_encoder_config: FileSource::huggingface(
                "BAAI/bge-small-en-v1.5".to_string(),
                "main".to_string(),
                "config.json".to_string(),
            ),
            label_encoder_tokenizer: FileSource::huggingface(
                "BAAI/bge-small-en-v1.5".to_string(),
                "main".to_string(),
                "tokenizer.json".to_string(),
            ),
            tokenizer: FileSource::huggingface(
                "knowledgator/gliner-bi-base-v2.0".to_string(),
                "main".to_string(),
                "tokenizer.json".to_string(),
            ),
            config: FileSource::huggingface(
                "knowledgator/gliner-bi-base-v2.0".to_string(),
                "main".to_string(),
                "gliner_config.json".to_string(),
            ),
        }
    }

    /// GLiNER bi-encoder v2.0 Large variant (530M parameters).
    ///
    /// Highest accuracy variant, using:
    /// - Text encoder: ettin-encoder-400m
    /// - Label encoder: bge-base-en-v1.5
    pub fn large() -> Self {
        Self {
            model: FileSource::huggingface(
                "knowledgator/gliner-bi-large-v2.0-gguf".to_string(),
                "main".to_string(),
                "gliner-bi-large-v2.0-Q8_0.gguf".to_string(),
            ),
            label_encoder: FileSource::huggingface(
                "knowledgator/gliner-bi-large-v2.0-gguf".to_string(),
                "main".to_string(),
                "label-encoder-Q8_0.gguf".to_string(),
            ),
            label_encoder_config: FileSource::huggingface(
                "BAAI/bge-base-en-v1.5".to_string(),
                "main".to_string(),
                "config.json".to_string(),
            ),
            label_encoder_tokenizer: FileSource::huggingface(
                "BAAI/bge-base-en-v1.5".to_string(),
                "main".to_string(),
                "tokenizer.json".to_string(),
            ),
            tokenizer: FileSource::huggingface(
                "knowledgator/gliner-bi-large-v2.0".to_string(),
                "main".to_string(),
                "tokenizer.json".to_string(),
            ),
            config: FileSource::huggingface(
                "knowledgator/gliner-bi-large-v2.0".to_string(),
                "main".to_string(),
                "gliner_config.json".to_string(),
            ),
        }
    }

    /// Create a custom source with specific file locations.
    pub fn custom(
        model: FileSource,
        label_encoder: FileSource,
        label_encoder_config: FileSource,
        label_encoder_tokenizer: FileSource,
        tokenizer: FileSource,
        config: FileSource,
    ) -> Self {
        Self {
            model,
            label_encoder,
            label_encoder_config,
            label_encoder_tokenizer,
            tokenizer,
            config,
        }
    }
}

impl Default for GlinerSource {
    fn default() -> Self {
        Self::base()
    }
}

impl GlinerSource {
    /// Create a source from local GGUF files (for testing converted models).
    ///
    /// # Arguments
    /// * `model_path` - Path to main model GGUF (text encoder + span layer + projection)
    /// * `label_encoder_path` - Path to label encoder GGUF (BERT/MiniLM)
    pub fn local(
        model_path: impl Into<std::path::PathBuf>,
        label_encoder_path: impl Into<std::path::PathBuf>,
    ) -> Self {
        let model_path = model_path.into();
        let label_encoder_path = label_encoder_path.into();
        Self {
            model: FileSource::local(model_path),
            label_encoder: FileSource::local(label_encoder_path),
            label_encoder_config: Self::huggingface_or_cached(
                "sentence-transformers/all-MiniLM-L6-v2",
                "main",
                "config.json",
            ),
            label_encoder_tokenizer: Self::huggingface_or_cached(
                "sentence-transformers/all-MiniLM-L6-v2",
                "main",
                "tokenizer.json",
            ),
            tokenizer: Self::huggingface_or_cached(
                "knowledgator/gliner-bi-edge-v2.0",
                "main",
                "tokenizer.json",
            ),
            config: Self::huggingface_or_cached(
                "knowledgator/gliner-bi-edge-v2.0",
                "main",
                "gliner_config.json",
            ),
        }
    }
}
