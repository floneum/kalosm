use kalosm_model_types::FileSource;

/// Predefined Whisper model sources
#[derive(Debug, Clone)]
pub struct WhisperSource {
    pub(crate) model: FileSource,
    pub(crate) tokenizer: FileSource,
    pub(crate) config: FileSource,
    pub(crate) multilingual: bool,
    pub(crate) heads: Option<&'static [[usize; 2]]>,
}

impl Default for WhisperSource {
    fn default() -> Self {
        Self::tiny_en()
    }
}

/// Build a `FileSource::huggingface` triplet from a (repo, file) pair, with
/// the `"main"` revision folded in. Every model constructor below references
/// three of these (model / tokenizer / config) — keeping them as a helper
/// trims 36 lines of `.to_owned()` repetition from this file.
fn hf(repo: &str, file: &str) -> FileSource {
    FileSource::huggingface(repo.to_owned(), "main".to_owned(), file.to_owned())
}

impl WhisperSource {
    /// Create a new WhisperSource
    pub fn new(
        model: FileSource,
        tokenizer: FileSource,
        config: FileSource,
        multilingual: bool,
        heads: Option<&'static [[usize; 2]]>,
    ) -> Self {
        Self {
            model,
            tokenizer,
            config,
            multilingual,
            heads,
        }
    }

    /// Build a `WhisperSource` from the per-asset (repo, file) parts. Each
    /// of the public constructors below differs only in these parts, so the
    /// constructor body is a single call into this helper.
    fn from_parts(
        model_repo: &str,
        model_file: &str,
        tokenizer_repo: &str,
        tokenizer_file: &str,
        config_repo: &str,
        config_file: &str,
        multilingual: bool,
        heads: Option<&'static [[usize; 2]]>,
    ) -> Self {
        WhisperSource::new(
            hf(model_repo, model_file),
            hf(tokenizer_repo, tokenizer_file),
            hf(config_repo, config_file),
            multilingual,
            heads,
        )
    }

    /// Tiny english model
    pub fn tiny_en() -> Self {
        Self::from_parts(
            "Demonthos/fusor-whisper-tiny-en",
            "whisper-tiny-en.gguf",
            "lmz/candle-whisper",
            "tokenizer-tiny-en.json",
            "lmz/candle-whisper",
            "config-tiny-en.json",
            false,
            Some(&[
                [1, 0],
                [2, 0],
                [2, 5],
                [3, 0],
                [3, 1],
                [3, 2],
                [3, 3],
                [3, 4],
            ]),
        )
    }

    /// Tiny model
    pub fn tiny() -> Self {
        Self::from_parts(
            "lmz/candle-whisper",
            "model-tiny-q80.gguf",
            "lmz/candle-whisper",
            "tokenizer-tiny.json",
            "lmz/candle-whisper",
            "config-tiny.json",
            true,
            Some(&[[2, 2], [3, 0], [3, 2], [3, 3], [3, 4], [3, 5]]),
        )
    }

    /// Base model
    pub fn base() -> Self {
        Self::from_parts(
            "Demonthos/fusor-whisper-base",
            "whisper-base.gguf",
            "openai/whisper-base",
            "tokenizer.json",
            "openai/whisper-base",
            "config.json",
            true,
            None,
        )
    }

    /// Base english model
    pub fn base_en() -> Self {
        Self::from_parts(
            "Demonthos/fusor-whisper-base-en",
            "whisper-base-en.gguf",
            "openai/whisper-base.en",
            "tokenizer.json",
            "openai/whisper-base.en",
            "config.json",
            false,
            None,
        )
    }

    /// Medium model
    pub fn medium() -> Self {
        Self::from_parts(
            "Demonthos/fusor-whisper-medium",
            "whisper-medium.gguf",
            "openai/whisper-medium",
            "tokenizer.json",
            "openai/whisper-medium",
            "config.json",
            true,
            None,
        )
    }

    /// Medium english model
    pub fn medium_en() -> Self {
        Self::from_parts(
            "Demonthos/fusor-whisper-medium-en",
            "whisper-medium-en.gguf",
            "openai/whisper-medium.en",
            "tokenizer.json",
            "openai/whisper-medium.en",
            "config.json",
            false,
            None,
        )
    }

    /// Large v3 model
    pub fn large_v3() -> Self {
        Self::from_parts(
            "Demonthos/fusor-whisper-large-v3",
            "whisper-large-v3.gguf",
            "openai/whisper-large-v3",
            "tokenizer.json",
            "openai/whisper-large-v3",
            "config.json",
            true,
            None,
        )
    }

    /// Distiled medium english model
    pub fn distil_medium_en() -> Self {
        Self::from_parts(
            "Demonthos/candle-quantized-whisper-medium-distil",
            "model.gguf",
            "Demonthos/candle-quantized-whisper-medium-distil",
            "tokenizer.json",
            "Demonthos/candle-quantized-whisper-medium-distil",
            "config.json",
            false,
            None,
        )
    }

    /// Distiled large v3.5 model
    pub fn distil_large_v3_5() -> Self {
        Self::from_parts(
            "Demonthos/fusor-distil-whisper-large-v3.5",
            "whisper-distil-large-3.5.gguf",
            "distil-whisper/distil-large-v3.5",
            "tokenizer.json",
            "distil-whisper/distil-large-v3.5",
            "config.json",
            true,
            Some(&[
                [1, 0],
                [1, 1],
                [1, 2],
                [1, 3],
                [1, 4],
                [1, 5],
                [1, 6],
                [1, 7],
                [1, 8],
                [1, 9],
                [1, 10],
                [1, 11],
                [1, 12],
                [1, 13],
                [1, 14],
                [1, 15],
                [1, 16],
                [1, 17],
                [1, 18],
                [1, 19],
            ]),
        )
    }

    /// Distiled large v3 model
    pub fn distil_large_v3() -> Self {
        Self::from_parts(
            "Demonthos/candle-quantized-whisper-distil-v3",
            "model.gguf",
            "Demonthos/candle-quantized-whisper-distil-v3",
            "tokenizer.json",
            "Demonthos/candle-quantized-whisper-distil-v3",
            "config.json",
            true,
            Some(&[
                [1, 0],
                [1, 1],
                [1, 2],
                [1, 3],
                [1, 4],
                [1, 5],
                [1, 6],
                [1, 7],
                [1, 8],
                [1, 9],
                [1, 10],
                [1, 11],
                [1, 12],
                [1, 13],
                [1, 14],
                [1, 15],
                [1, 16],
                [1, 17],
                [1, 18],
                [1, 19],
            ]),
        )
    }

    /// Large v3 turbo model
    pub fn large_v3_turbo() -> Self {
        Self::from_parts(
            "Demonthos/candle-quantized-whisper-large-v3-turbo",
            "model.gguf",
            "Demonthos/candle-quantized-whisper-large-v3-turbo",
            "tokenizer.json",
            "Demonthos/candle-quantized-whisper-large-v3-turbo",
            "config.json",
            true,
            Some(&[[2, 4], [2, 11], [3, 3], [3, 6], [3, 11], [3, 14]]),
        )
    }
}
