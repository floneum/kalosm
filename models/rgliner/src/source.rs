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

        // Last resort: when the caller did not pin a revision (`main`), accept any
        // snapshot directory that contains the file. We deliberately skip this for
        // a pinned revision — returning an arbitrary snapshot in filesystem order
        // would silently serve the wrong revision's weights.
        if revision == "main" {
            let entries = std::fs::read_dir(&snapshots_dir).ok()?;
            for entry in entries.flatten() {
                let candidate = entry.path().join(file);
                if candidate.exists() {
                    return Some(candidate);
                }
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

    fn demonthos_gguf(file: &str) -> FileSource {
        Self::huggingface_or_cached("Demonthos/gliner-gguf", "main", file)
    }

    /// Build a bi-encoder v2.0 preset. `name` is the variant slug shared by the
    /// Demonthos GGUF filenames and the `knowledgator/gliner-bi-{name}-v2.0`
    /// repo; `label_encoder_model` is the sentence-transformer it pairs with.
    fn bi_encoder_v2(name: &str, label_encoder_model: &str) -> Self {
        let st = |file: &str| {
            FileSource::huggingface(label_encoder_model.to_string(), "main".to_string(), file.to_string())
        };
        let knowledgator = |file: &str| {
            FileSource::huggingface(
                format!("knowledgator/gliner-bi-{name}-v2.0"),
                "main".to_string(),
                file.to_string(),
            )
        };
        Self {
            model: Self::demonthos_gguf(&format!("gliner-bi-{name}-v2.0-Q4_K.gguf")),
            label_encoder: Self::demonthos_gguf(&format!(
                "gliner-bi-{name}-v2.0-Q4_K-label-encoder.gguf"
            )),
            label_encoder_config: st("config.json"),
            label_encoder_tokenizer: st("tokenizer.json"),
            tokenizer: knowledgator("tokenizer.json"),
            config: knowledgator("gliner_config.json"),
        }
    }

    /// GLiNER bi-encoder v2.0 Edge variant (60M parameters, Q4_K).
    ///
    /// The smallest and fastest variant, using:
    /// - Text encoder: ettin-encoder-32m
    /// - Label encoder: all-MiniLM-L6-v2
    pub fn edge() -> Self {
        Self::bi_encoder_v2("edge", "sentence-transformers/all-MiniLM-L6-v2")
    }

    /// GLiNER bi-encoder v2.0 Small variant (108M parameters, Q4_K).
    ///
    /// Good balance of speed and accuracy, using:
    /// - Text encoder: ettin-encoder-68m
    /// - Label encoder: all-MiniLM-L12-v2
    pub fn small() -> Self {
        Self::bi_encoder_v2("small", "sentence-transformers/all-MiniLM-L12-v2")
    }

    /// GLiNER bi-encoder v2.0 Base variant (194M parameters, Q4_K).
    ///
    /// Default variant with good accuracy, using:
    /// - Text encoder: ettin-encoder-150m
    /// - Label encoder: bge-small-en-v1.5
    pub fn base() -> Self {
        Self::bi_encoder_v2("base", "BAAI/bge-small-en-v1.5")
    }

    /// GLiNER bi-encoder v2.0 Large variant (530M parameters, Q4_K).
    ///
    /// Highest accuracy variant, using:
    /// - Text encoder: ettin-encoder-400m
    /// - Label encoder: bge-base-en-v1.5
    pub fn large() -> Self {
        Self::bi_encoder_v2("large", "BAAI/bge-base-en-v1.5")
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
