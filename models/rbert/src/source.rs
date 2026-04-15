use kalosm_model_types::FileSource;

const SNOWFLAKE_EMBEDDING_PREFIX: &str =
    "Represent this sentence for searching relevant passages: ";

const QWEN3_EMBEDDING_PREFIX: &str =
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ";

/// The source of a [`crate::Bert`] model
pub struct BertSource {
    pub(crate) search_embedding_prefix: Option<String>,
    pub(crate) config: Option<FileSource>,
    pub(crate) tokenizer: FileSource,
    pub(crate) model: FileSource,
}

impl BertSource {
    /// Create a new [`BertSource`] for embedding plain text
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new [`BertSource`] for embedding text for search
    pub fn new_for_search() -> Self {
        Self::snowflake_arctic_embed_extra_small()
    }

    /// Set the model to use, check out available models: <https://huggingface.co/models?library=sentence-transformers&sort=trending>
    pub fn with_model(mut self, model: FileSource) -> Self {
        self.model = model;
        self
    }

    /// Set the tokenizer to use
    pub fn with_tokenizer(mut self, tokenizer: FileSource) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    /// Set the config to use
    pub fn with_config(mut self, config: FileSource) -> Self {
        self.config = Some(config);
        self
    }

    /// Set the prefix to use when embedding search queries
    pub fn with_search_embedding_prefix(mut self, prefix: impl Into<Option<String>>) -> Self {
        self.search_embedding_prefix = prefix.into();
        self
    }

    /// Create a new [`BertSource`] with the BGE large english preset
    pub fn bge_large_en() -> Self {
        // https://huggingface.co/CompendiumLabs/bge-large-en-v1.5-gguf/blob/main/bge-large-en-v1.5-q4_k_m.gguf
        Self::default()
            .with_model(FileSource::huggingface(
                "CompendiumLabs/bge-large-en-v1.5-gguf".to_string(),
                "main".to_string(),
                "bge-large-en-v1.5-q4_k_m.gguf".to_string(),
            ))
            .with_tokenizer(FileSource::huggingface(
                "BAAI/bge-large-en-v1.5".to_string(),
                "refs/pr/5".to_string(),
                "tokenizer.json".to_string(),
            ))
            .with_config(FileSource::huggingface(
                "BAAI/bge-large-en-v1.5".to_string(),
                "refs/pr/5".to_string(),
                "config.json".to_string(),
            ))
    }

    /// Create a new [`BertSource`] with the BGE base english preset
    pub fn bge_base_en() -> Self {
        // https://huggingface.co/CompendiumLabs/bge-base-en-v1.5-gguf/blob/main/bge-base-en-v1.5-q4_k_m.gguf
        Self::default()
            .with_model(FileSource::huggingface(
                "BAAI/bge-base-en-v1.5".to_string(),
                "refs/pr/1".to_string(),
                "model.safetensors".to_string(),
            ))
            .with_tokenizer(FileSource::huggingface(
                "BAAI/bge-base-en-v1.5".to_string(),
                "refs/pr/1".to_string(),
                "tokenizer.json".to_string(),
            ))
            .with_config(FileSource::huggingface(
                "BAAI/bge-base-en-v1.5".to_string(),
                "refs/pr/1".to_string(),
                "config.json".to_string(),
            ))
    }

    /// Create a new [`BertSource`] with the BGE small english preset
    pub fn bge_small_en() -> Self {
        // https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/blob/main/bge-small-en-v1.5-q4_k_m.gguf
        Self {
            config: Some(FileSource::huggingface(
                "BAAI/bge-small-en-v1.5".to_string(),
                "main".to_string(),
                "config.json".to_string(),
            )),
            tokenizer: FileSource::huggingface(
                "BAAI/bge-small-en-v1.5".to_string(),
                "main".to_string(),
                "tokenizer.json".to_string(),
            ),
            model: FileSource::huggingface(
                "CompendiumLabs/bge-small-en-v1.5-gguf".to_string(),
                "main".to_string(),
                "bge-small-en-v1.5-q4_k_m.gguf".to_string(),
            ),
            search_embedding_prefix: None,
        }
    }

    /// Create a new [`BertSource`] with the MiniLM-L6-v2 preset
    pub fn mini_lm_l6_v2() -> Self {
        Self::default()
            .with_model(FileSource::huggingface(
                "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                "refs/pr/21".to_string(),
                "model.safetensors".to_string(),
            ))
            .with_tokenizer(FileSource::huggingface(
                "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                "refs/pr/21".to_string(),
                "tokenizer.json".to_string(),
            ))
            .with_config(FileSource::huggingface(
                "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                "refs/pr/21".to_string(),
                "config.json".to_string(),
            ))
    }

    /// Create a new [`BertSource`] with the [snowflake-arctic-embed-xs](https://huggingface.co/Snowflake/snowflake-arctic-embed-xs) model
    pub fn snowflake_arctic_embed_extra_small() -> Self {
        snowflake_source(
            "Snowflake/snowflake-arctic-embed-xs",
            "ChristianAzinn/snowflake-arctic-embed-xs-gguf",
            "snowflake-arctic-embed-xs--Q4_K_M.GGUF",
        )
    }

    /// Create a new [`BertSource`] with the [snowflake-arctic-embed-s](https://huggingface.co/Snowflake/snowflake-arctic-embed-s) model
    pub fn snowflake_arctic_embed_small() -> Self {
        snowflake_source(
            "Snowflake/snowflake-arctic-embed-s",
            "ChristianAzinn/snowflake-arctic-embed-s-gguf",
            "snowflake-arctic-embed-s--Q4_K_M.GGUF",
        )
    }

    /// Create a new [`BertSource`] with the [snowflake-arctic-embed-m](https://huggingface.co/Snowflake/snowflake-arctic-embed-m) model
    pub fn snowflake_arctic_embed_medium() -> Self {
        snowflake_source(
            "Snowflake/snowflake-arctic-embed-m",
            "ChristianAzinn/snowflake-arctic-embed-m-gguf",
            "snowflake-arctic-embed-m--Q4_K_M.GGUF",
        )
    }

    /// Create a new [`BertSource`] with the [snowflake-arctic-embed-m-long](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-long) model
    ///
    /// This model is slightly larger than [`Self::snowflake_arctic_embed_medium`] and supports longer contexts (up to 2048 tokens).
    pub fn snowflake_arctic_embed_medium_long() -> Self {
        snowflake_source(
            "Snowflake/snowflake-arctic-embed-m-long",
            "ChristianAzinn/snowflake-arctic-embed-m-long-GGUF",
            "snowflake-arctic-embed-m-long--Q4_K_M.GGUF",
        )
    }

    /// Create a new [`BertSource`] with the [snowflake-arctic-embed-l](https://huggingface.co/Snowflake/snowflake-arctic-embed-l) model
    pub fn snowflake_arctic_embed_large() -> Self {
        snowflake_source(
            "Snowflake/snowflake-arctic-embed-l",
            "ChristianAzinn/snowflake-arctic-embed-l-gguf",
            "snowflake-arctic-embed-l--Q4_K_M.GGUF",
        )
    }

    /// Create a new [`BertSource`] with the [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF) model
    ///
    /// This model uses the Qwen3 architecture and supports:
    /// - 100+ languages
    /// - 32K context length
    /// - 1024 embedding dimensions
    /// - Last-token pooling (automatic)
    pub fn qwen3_embedding_0_6b() -> Self {
        qwen3_source(
            "Qwen/Qwen3-Embedding-0.6B",
            "Qwen/Qwen3-Embedding-0.6B-GGUF",
            "Qwen3-Embedding-0.6B-Q8_0.gguf",
        )
    }

    /// Create a new [`BertSource`] with the [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B-GGUF) model
    ///
    /// This model uses the Qwen3 architecture and supports:
    /// - 100+ languages
    /// - 32K context length
    /// - 2560 embedding dimensions
    /// - Last-token pooling (automatic)
    ///
    /// This is a larger model than [`Self::qwen3_embedding_0_6b`] with higher quality embeddings.
    pub fn qwen3_embedding_4b() -> Self {
        qwen3_source(
            "Qwen/Qwen3-Embedding-4B",
            "Qwen/Qwen3-Embedding-4B-GGUF",
            "Qwen3-Embedding-4B-Q8_0.gguf",
        )
    }

    /// Create a new [`BertSource`] with the [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B-GGUF) model
    ///
    /// This model uses the Qwen3 architecture and supports:
    /// - 100+ languages
    /// - 32K context length
    /// - 4096 embedding dimensions
    /// - Last-token pooling (automatic)
    ///
    /// This is the largest Qwen3 embedding model with the highest quality embeddings.
    pub fn qwen3_embedding_8b() -> Self {
        qwen3_source(
            "Qwen/Qwen3-Embedding-8B",
            "Qwen/Qwen3-Embedding-8B-GGUF",
            "Qwen3-Embedding-8B-Q8_0.gguf",
        )
    }
}

fn snowflake_source(snowflake_repo: &str, gguf_repo: &str, gguf_file: &str) -> BertSource {
    BertSource::default()
        .with_config(FileSource::huggingface(
            snowflake_repo.to_string(),
            "main".to_string(),
            "config.json".to_string(),
        ))
        .with_tokenizer(FileSource::huggingface(
            snowflake_repo.to_string(),
            "main".to_string(),
            "tokenizer.json".to_string(),
        ))
        .with_model(FileSource::huggingface(
            gguf_repo.to_string(),
            "main".to_string(),
            gguf_file.to_string(),
        ))
        .with_search_embedding_prefix(SNOWFLAKE_EMBEDDING_PREFIX.to_string())
}

fn qwen3_source(model_repo: &str, gguf_repo: &str, gguf_file: &str) -> BertSource {
    BertSource {
        config: None,
        tokenizer: FileSource::huggingface(
            model_repo.to_string(),
            "main".to_string(),
            "tokenizer.json".to_string(),
        ),
        model: FileSource::huggingface(
            gguf_repo.to_string(),
            "main".to_string(),
            gguf_file.to_string(),
        ),
        search_embedding_prefix: Some(QWEN3_EMBEDDING_PREFIX.to_string()),
    }
}

impl Default for BertSource {
    fn default() -> Self {
        Self::bge_small_en()
    }
}
