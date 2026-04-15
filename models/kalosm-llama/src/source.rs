use fusor_gguf::GgufReadError;
use kalosm_common::CacheError;
use kalosm_model_types::{FileLoadingProgress, FileSource};

use crate::raw::RopeScalingConfig;

fn llama_tokenizer() -> FileSource {
    FileSource::huggingface(
        "hf-internal-testing/llama-tokenizer".to_string(),
        "main".to_string(),
        "tokenizer.json".to_string(),
    )
}

fn llama_v3_tokenizer() -> FileSource {
    FileSource::huggingface(
        "NousResearch/Meta-Llama-3-8B-Instruct".to_string(),
        "main".to_string(),
        "tokenizer.json".to_string(),
    )
}

fn mistral_tokenizer() -> FileSource {
    FileSource::huggingface(
        "mistralai/Mistral-7B-v0.1".to_string(),
        "main".to_string(),
        "tokenizer.json".to_string(),
    )
}

fn qwen_tokenizer() -> FileSource {
    FileSource::huggingface(
        "Qwen/Qwen2.5-0.5B".to_string(),
        "main".to_string(),
        "tokenizer.json".to_string(),
    )
}

fn qwen3_tokenizer() -> FileSource {
    FileSource::huggingface(
        "Qwen/Qwen3-0.6B".to_string(),
        "main".to_string(),
        "tokenizer.json".to_string(),
    )
}

#[derive(Debug, serde::Deserialize)]
pub(crate) struct LlamaConfigJson {
    pub(crate) rope_scaling: Option<RopeScalingConfig>,
}

/// A source for the Llama model.
#[derive(Clone, Debug)]
pub struct LlamaSource {
    pub(crate) model: Vec<FileSource>,
    pub(crate) vision_model: Option<FileSource>,
    pub(crate) tokenizer: Option<FileSource>,
    pub(crate) config: Option<FileSource>,
    pub(crate) group_query_attention: u8,
    pub(crate) cache: kalosm_common::Cache,
    pub(crate) override_stop_token_string: Option<String>,
    pub(crate) override_chat_template: Option<String>,
}

/// Errors that can occur when loading the Llama model.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum LlamaSourceError {
    /// An error occurred while loading the tokenizer.
    #[error("Failed to load the tokenizer: {0}")]
    Tokenizer(Box<dyn std::error::Error + Send + Sync>),
    /// An error occurred while loading the config.
    #[error("Failed to load the config: {0}")]
    Config(#[from] serde_json::Error),
    /// An error occurred while loading the model (from the cache or downloading it).
    #[error("Failed to load the model: {0}")]
    Model(#[from] CacheError),
    /// The model file is not a valid gguf file.
    #[error("The model file is not a valid gguf file")]
    InvalidGguf,
    /// Could not find a specific entry in the gguf file.
    #[error("Could not find {0} in the gguf file")]
    MissingGgufEntry(String),
    /// Failed to read the gguf file
    #[error("Failed to read the gguf file: {0}")]
    GgufRead(#[from] GgufReadError),
    /// An error occurred while loading the model onto the device.
    #[error("Failed to load the model onto the device: {0}")]
    Device(#[from] fusor::Error),
    /// No stop token was found.
    #[error("No stop token was found")]
    NoStopToken,
    /// An error loading the chat template.
    #[error("Failed to load chat template: {0}")]
    ChatTemplate(#[from] minijinja::Error),
    /// No tokenizer was provided and the ggml file format does not contain a tokenizer.
    #[error("No tokenizer was provided and the ggml file format does not contain a tokenizer")]
    NoTokenizer,
    /// The task loading the model panicked.
    #[error("The task loading the model panicked")]
    ModelLoadingPanic,
}

impl LlamaSource {
    /// Create a new source for the Llama model.
    pub fn new(model: FileSource) -> Self {
        Self {
            model: vec![model],
            tokenizer: None,
            config: None,
            group_query_attention: 1,
            cache: Default::default(),
            override_stop_token_string: None,
            override_chat_template: None,
            vision_model: None,
        }
    }

    /// Create a new source for the Llama model with multiple files.
    pub fn new_sharded(model: Vec<FileSource>) -> Self {
        Self {
            model,
            tokenizer: None,
            config: None,
            group_query_attention: 1,
            cache: Default::default(),
            override_stop_token_string: None,
            override_chat_template: None,
            vision_model: None,
        }
    }

    /// Set the model to use for the model
    pub fn with_model(mut self, model: impl Into<Vec<FileSource>>) -> Self {
        self.model = model.into();
        self
    }

    /// Set the tokenizer to use for the model. Kalosm will try to load the tokenizer from the gguf file if no tokenizer is provided.
    pub fn with_tokenizer(mut self, tokenizer: FileSource) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    /// Set the config to use for the model. Kalosm will try to load the config from the gguf file if no config is provided.
    pub fn with_config(mut self, config: FileSource) -> Self {
        self.config = Some(config);
        self
    }

    /// Set the cache location to use for the model (defaults DATA_DIR/kalosm/cache)
    pub fn with_cache(mut self, cache: kalosm_common::Cache) -> Self {
        self.cache = cache;

        self
    }

    /// Set the group query attention for the model
    /// For the llama family of models, this is typically 1
    /// For the mistral family of models, this is typically 8
    ///
    /// This is determined automatically for any gguf models
    pub fn with_group_query_attention(mut self, group_query_attention: u8) -> Self {
        self.group_query_attention = group_query_attention;

        self
    }

    /// Override the stop token string. This is useful for models that have the wrong default stop token string.
    pub fn with_override_stop_token_string(mut self, stop_token_string: impl ToString) -> Self {
        self.override_stop_token_string = Some(stop_token_string.to_string());

        self
    }

    /// Override the chat template. This is useful for models that have a missing or incorrect chat template.
    pub fn with_override_chat_template(mut self, chat_template: impl ToString) -> Self {
        self.override_chat_template = Some(chat_template.to_string());

        self
    }

    /// Set the clip model to use for vision encoding. This is used for Qwen-2.5 VL models to enable vision.
    pub fn with_vision_model(mut self, model: FileSource) -> Self {
        self.vision_model = Some(model);

        self
    }

    pub(crate) async fn model(
        &self,
        mut progress: impl FnMut(FileLoadingProgress),
    ) -> Result<Vec<Vec<u8>>, LlamaSourceError> {
        let mut model_bytes = Vec::new();
        for file in &self.model {
            model_bytes.push(self.cache.get_bytes(file, &mut progress).await?);
        }
        Ok(model_bytes)
    }

    /// A preset for Mistral7b
    pub fn mistral_7b() -> Self {
        Self::new(FileSource::huggingface(
            "TheBloke/Mistral-7B-v0.1-GGUF".to_string(),
            "main".to_string(),
            "mistral-7b-v0.1.Q4_K_M.gguf".to_string(),
        ))
        .with_tokenizer(mistral_tokenizer())
        .with_group_query_attention(8)
    }

    /// A preset for Mistral7bInstruct
    pub fn mistral_7b_instruct() -> Self {
        Self::new(FileSource::huggingface(
            "TheBloke/Mistral-7B-Instruct-v0.1-GGUF".to_string(),
            "main".to_string(),
            "mistral-7b-instruct-v0.1.Q4_K_M.gguf".to_string(),
        ))
        .with_tokenizer(mistral_tokenizer())
        .with_group_query_attention(8)
    }

    /// A preset for Mistral7bInstruct v0.2
    pub fn mistral_7b_instruct_2() -> Self {
        Self::new(FileSource::huggingface(
            "TheBloke/Mistral-7B-Instruct-v0.2-GGUF".to_string(),
            "main".to_string(),
            "mistral-7b-instruct-v0.2.Q4_K_M.gguf".to_string(),
        ))
        .with_tokenizer(mistral_tokenizer())
        .with_group_query_attention(8)
    }

    /// A preset for Codestral 22b
    pub fn codestral_22b() -> Self {
        Self::new(
            FileSource::HuggingFace {
                model_id: "bartowski/Codestral-22B-v0.1-GGUF".to_string(),
                revision: "main".to_string(),
                file: "Codestral-22B-v0.1-Q4_K_M.gguf".to_string(),
            },
        )
        .with_tokenizer(
            FileSource::HuggingFace {
                model_id: "mistralai/Codestral-22B-v0.1".to_string(),
                revision: "main".to_string(),
                file: "tokenizer.json".to_string(),
            },
        )
        .with_override_chat_template(r#"{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content'] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n\n{{- bos_token }}\n{%- for message in loop_messages %}\n    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}\n        {{- raise_exception('After the optional system message, conversation roles must alternate user/assistant/user/assistant/...') }}\n    {%- endif %}\n    {%- if message['role'] == 'user' %}\n        {%- if loop.last and system_message is defined %}\n            {{- '[INST] ' + system_message + '\\n\\n' + message['content'] + '[/INST]' }}\n        {%- else %}\n            {{- '[INST] ' + message['content'] + '[/INST]' }}\n        {%- endif %}\n    {%- elif message['role'] == 'assistant' %}\n        {{- ' ' + message['content'] + eos_token}}\n    {%- else %}\n        {{- raise_exception('Only user and assistant roles are supported, with the exception of an initial optional system message!') }}\n    {%- endif %}\n{%- endfor %}\n"#)
    }

    /// A preset for NeuralHermes-2.5-Mistral-7B-GGUF
    pub fn neural_hermes_2_5_mistral_7b() -> Self {
        Self::new(FileSource::huggingface(
            "TheBloke/NeuralHermes-2.5-Mistral-7B-GGUF".to_string(),
            "main".to_string(),
            "neuralhermes-2.5-mistral-7b.Q4_0.gguf".to_string(),
        ))
        .with_tokenizer(mistral_tokenizer())
        .with_group_query_attention(8)
    }

    /// A preset for Neural Chat v3.3
    pub fn neural_chat_7b_v3_3() -> Self {
        Self::new(FileSource::huggingface(
            "TheBloke/neural-chat-7B-v3-3-GGUF".to_string(),
            "main".to_string(),
            "neural-chat-7b-v3-3.Q4_0.gguf".to_string(),
        ))
        .with_tokenizer(FileSource::huggingface(
            "Intel/neural-chat-7b-v3-3".to_string(),
            "main".to_string(),
            "tokenizer.json".to_string(),
        ))
        .with_group_query_attention(8)
    }

    /// A preset for Zephyr7bAlpha
    pub fn zephyr_7b_alpha() -> Self {
        Self::new(FileSource::huggingface(
            "TheBloke/zephyr-7B-alpha-GGUF".to_string(),
            "main".to_string(),
            "zephyr-7b-alpha.Q4_K_M.gguf".to_string(),
        ))
        .with_tokenizer(mistral_tokenizer())
        .with_group_query_attention(8)
    }

    /// A preset for Zephyr7bBeta
    pub fn zephyr_7b_beta() -> Self {
        Self::new(FileSource::huggingface(
            "TheBloke/zephyr-7B-beta-GGUF".to_string(),
            "main".to_string(),
            "zephyr-7b-beta.Q4_K_M.gguf".to_string(),
        ))
        .with_tokenizer(mistral_tokenizer())
        .with_group_query_attention(8)
    }

    /// A preset for [Open chat 3.5 (0106)](https://huggingface.co/openchat/openchat-3.5-0106)
    pub fn open_chat_7b() -> Self {
        Self::new(FileSource::huggingface(
            "TheBloke/openchat-3.5-0106-GGUF".to_string(),
            "main".to_string(),
            "openchat-3.5-0106.Q4_K_M.gguf".to_string(),
        ))
        .with_tokenizer(FileSource::huggingface(
            "openchat/openchat-3.5-0106".to_string(),
            "main".to_string(),
            "tokenizer.json".to_string(),
        ))
        .with_group_query_attention(8)
    }

    /// A preset for Starling 7b Alpha
    pub fn starling_7b_alpha() -> Self {
        Self::new(FileSource::huggingface(
            "TheBloke/Starling-LM-7B-alpha-GGUF".to_string(),
            "main".to_string(),
            "starling-lm-7b-alpha.Q4_K_M.gguf".to_string(),
        ))
        .with_tokenizer(FileSource::huggingface(
            "berkeley-nest/Starling-LM-7B-alpha".to_string(),
            "main".to_string(),
            "tokenizer.json".to_string(),
        ))
        .with_group_query_attention(8)
    }

    /// A preset for Starling 7b Beta
    pub fn starling_7b_beta() -> Self {
        Self::new(FileSource::huggingface(
            "bartowski/Starling-LM-7B-beta-GGUF".to_string(),
            "main".to_string(),
            "Starling-LM-7B-beta-Q4_K_M.gguf".to_string(),
        ))
        .with_tokenizer(FileSource::huggingface(
            "Nexusflow/Starling-LM-7B-beta".to_string(),
            "main".to_string(),
            "tokenizer.json".to_string(),
        ))
        .with_group_query_attention(8)
    }

    /// A preset for WizardLM 2 7B
    pub fn wizard_lm_7b_v2() -> Self {
        Self::new(FileSource::huggingface(
            "bartowski/WizardLM-2-7B-GGUF".to_string(),
            "main".to_string(),
            "WizardLM-2-7B-Q4_K_M.gguf".to_string(),
        ))
        .with_tokenizer(mistral_tokenizer())
        .with_group_query_attention(8)
    }

    /// A preset for tiny llama 1.1b 1.0 Chat
    pub fn tiny_llama_1_1b_chat() -> Self {
        Self::new(FileSource::huggingface(
            "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF".to_string(),
            "main".to_string(),
            "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf".to_string(),
        ))
        .with_tokenizer(FileSource::huggingface(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string(),
            "main".to_string(),
            "tokenizer.json".to_string(),
        ))
        .with_group_query_attention(4)
    }

    /// A preset for tiny llama 1.1b 1.0
    pub fn tiny_llama_1_1b() -> Self {
        Self::new(FileSource::huggingface(
            "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF".to_string(),
            "main".to_string(),
            "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf".to_string(),
        ))
        .with_tokenizer(FileSource::huggingface(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0".to_string(),
            "main".to_string(),
            "tokenizer.json".to_string(),
        ))
        .with_group_query_attention(4)
    }

    /// A preset for Phi-3-mini-4k-instruct
    pub fn phi_3_mini_4k_instruct() -> Self {
        Self::new(FileSource::huggingface(
            "microsoft/Phi-3-mini-4k-instruct-gguf".to_string(),
            "5eef2ce24766d31909c0b269fe90c817a8f263fb".to_string(),
            "Phi-3-mini-4k-instruct-q4.gguf".to_string(),
        ))
        .with_tokenizer(FileSource::huggingface(
            "microsoft/Phi-3-mini-4k-instruct".to_string(),
            "main".to_string(),
            "tokenizer.json".to_string(),
        ))
        .with_group_query_attention(1)
        .with_override_stop_token_string("<|end|>".to_string())
    }

    /// A preset for Phi-3-mini-4k-instruct with the updated version of the model
    pub fn phi_3_1_mini_4k_instruct() -> Self {
        Self::new(FileSource::huggingface(
            "bartowski/Phi-3.1-mini-4k-instruct-GGUF".to_string(),
            "main".to_string(),
            "Phi-3.1-mini-4k-instruct-Q4_K_M.gguf".to_string(),
        ))
        .with_tokenizer(FileSource::huggingface(
            "microsoft/Phi-3-mini-4k-instruct".to_string(),
            "main".to_string(),
            "tokenizer.json".to_string(),
        ))
        .with_group_query_attention(1)
        .with_override_stop_token_string("<|end|>".to_string())
    }

    /// A preset for Phi-3.5-mini-4k-instruct with the updated version of the model
    pub fn phi_3_5_mini_4k_instruct() -> Self {
        Self::new(FileSource::huggingface(
            "bartowski/Phi-3.5-mini-instruct-GGUF".to_string(),
            "main".to_string(),
            "Phi-3.5-mini-instruct-Q4_K_M.gguf".to_string(),
        ))
        .with_tokenizer(FileSource::huggingface(
            "microsoft/Phi-3.5-mini-instruct".to_string(),
            "main".to_string(),
            "tokenizer.json".to_string(),
        ))
        .with_group_query_attention(1)
        .with_override_stop_token_string("<|end|>".to_string())
    }

    /// A preset for phi 4 (14b)
    pub fn phi_4() -> Self {
        Self::new(FileSource::huggingface(
            "microsoft/phi-4-gguf".to_string(),
            "main".to_string(),
            "phi-4-Q4_0.gguf".to_string(),
        ))
        .with_tokenizer(FileSource::huggingface(
            "microsoft/phi-4".to_string(),
            "main".to_string(),
            "tokenizer.json".to_string(),
        ))
        .with_override_stop_token_string("<|im_end|>".to_string())
    }

    /// A preset for Llama7b v2
    pub fn llama_7b() -> Self {
        Self::new(FileSource::huggingface(
            "TheBloke/Llama-2-7B-GGML".to_string(),
            "main".to_string(),
            "llama-2-7b.ggmlv3.q4_0.bin".to_string(),
        ))
        .with_tokenizer(llama_tokenizer())
        .with_group_query_attention(1)
    }

    /// A preset for Llama8b v3
    pub fn llama_8b() -> Self {
        Self::new(FileSource::huggingface(
            "NousResearch/Meta-Llama-3-8B-GGUF".to_string(),
            "main".to_string(),
            "Meta-Llama-3-8B-Q4_K_M.gguf".to_string(),
        ))
        .with_tokenizer(llama_v3_tokenizer())
        .with_group_query_attention(1)
    }

    /// A preset for Llama8b v3
    pub fn llama_8b_chat() -> Self {
        Self::new(FileSource::huggingface(
            "bartowski/Meta-Llama-3-8B-Instruct-GGUF".to_string(),
            "main".to_string(),
            "Meta-Llama-3-8B-Instruct-Q5_K_M.gguf".to_string(),
        ))
        .with_tokenizer(llama_v3_tokenizer())
        .with_group_query_attention(1)
    }

    /// A preset for Llama8b v3.1 Instruct
    pub fn llama_3_1_8b_chat() -> Self {
        Self::new(FileSource::huggingface(
            "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF".to_string(),
            "main".to_string(),
            "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf".to_string(),
        ))
        .with_tokenizer(llama_v3_tokenizer())
        // https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct/blob/main/config.json
        .with_config(FileSource::huggingface(
            "unsloth/Meta-Llama-3.1-8B-Instruct".to_string(),
            "main".to_string(),
            "config.json".to_string(),
        ))
        .with_group_query_attention(1)
    }

    /// A preset for Llama8b v3 at the Q8_0 quantization level. This file will be larger than [`llama_8b_chat`](Self::llama_8b_chat) but the model output will be more accurate.
    pub fn llama_8b_chat_q8() -> Self {
        Self::new(FileSource::huggingface(
            "bartowski/Meta-Llama-3-8B-Instruct-GGUF".to_string(),
            "main".to_string(),
            "Meta-Llama-3-8B-Instruct-Q8_0.gguf".to_string(),
        ))
        .with_tokenizer(llama_v3_tokenizer())
        .with_group_query_attention(1)
    }

    /// A preset for Llama8b SPPO Iter3
    pub fn llama_8b_sppo_iter3() -> Self {
        Self::new(FileSource::huggingface(
            "bartowski/Llama-3-Instruct-8B-SPPO-Iter3-GGUF".to_string(),
            "main".to_string(),
            "Llama-3-Instruct-8B-SPPO-Iter3-Q4_K_M.gguf".to_string(),
        ))
        .with_tokenizer(llama_v3_tokenizer())
        .with_group_query_attention(1)
    }

    /// A preset for Llama 2.3 1b
    pub fn llama_3_2_1b_chat() -> Self {
        Self::new(FileSource::huggingface(
            "lmstudio-community/Llama-3.2-1B-Instruct-GGUF".to_string(),
            "main".to_string(),
            "Llama-3.2-1B-Instruct-Q4_K_M.gguf".to_string(),
        ))
        .with_tokenizer(llama_v3_tokenizer())
        .with_config(FileSource::huggingface(
            "NousResearch/Llama-3.2-1B".to_string(),
            "main".to_string(),
            "config.json".to_string(),
        ))
        .with_group_query_attention(1)
    }

    /// A preset for Llama 2.3 3b
    pub fn llama_3_2_3b_chat() -> Self {
        Self::new(FileSource::huggingface(
            "lmstudio-community/Llama-3.2-3B-Instruct-GGUF".to_string(),
            "main".to_string(),
            "Llama-3.2-3B-Instruct-Q4_K_M.gguf".to_string(),
        ))
        .with_tokenizer(llama_v3_tokenizer())
        .with_config(FileSource::huggingface(
            "NousResearch/Hermes-3-Llama-3.2-3B".to_string(),
            "main".to_string(),
            "config.json".to_string(),
        ))
        .with_group_query_attention(1)
    }

    /// A preset for Llama13b
    pub fn llama_13b() -> Self {
        Self::new(FileSource::huggingface(
            "TheBloke/Llama-2-13B-GGML".to_string(),
            "main".to_string(),
            "llama-2-13b.ggmlv3.q4_0.bin".to_string(),
        ))
        .with_tokenizer(llama_tokenizer())
        .with_group_query_attention(1)
    }

    /// A preset for Llama70b
    pub fn llama_70b() -> Self {
        Self::new(FileSource::huggingface(
            "TheBloke/Llama-2-70B-GGML".to_string(),
            "main".to_string(),
            "llama-2-70b.ggmlv3.q4_0.bin".to_string(),
        ))
        .with_tokenizer(llama_tokenizer())
        .with_group_query_attention(8)
    }

    /// A preset for Llama7bChat
    pub fn llama_7b_chat() -> Self {
        Self::new(FileSource::huggingface(
            "TheBloke/Llama-2-7B-Chat-GGML".to_string(),
            "main".to_string(),
            "llama-2-7b-chat.ggmlv3.q4_0.bin".to_string(),
        ))
        .with_tokenizer(llama_tokenizer())
    }

    /// A preset for Llama13bChat
    pub fn llama_13b_chat() -> Self {
        Self::new(FileSource::huggingface(
            "TheBloke/Llama-2-13B-Chat-GGML".to_string(),
            "main".to_string(),
            "llama-2-13b-chat.ggmlv3.q4_0.bin".to_string(),
        ))
        .with_tokenizer(llama_tokenizer())
        .with_group_query_attention(1)
    }

    /// A preset for Llama70bChat
    pub fn llama_70b_chat() -> Self {
        Self::new(FileSource::huggingface(
            "TheBloke/Llama-2-70B-Chat-GGML".to_string(),
            "main".to_string(),
            "llama-2-70b-chat.ggmlv3.q4_0.bin".to_string(),
        ))
        .with_tokenizer(llama_tokenizer())
        .with_group_query_attention(8)
    }

    /// A preset for Llama7bCode
    pub fn llama_7b_code() -> Self {
        Self::new(FileSource::huggingface(
            "TheBloke/CodeLlama-7B-GGUF".to_string(),
            "main".to_string(),
            "codellama-7b.Q8_0.gguf".to_string(),
        ))
        .with_tokenizer(llama_tokenizer())
        .with_group_query_attention(1)
    }

    /// A preset for Llama13bCode
    pub fn llama_13b_code() -> Self {
        Self::new(FileSource::huggingface(
            "TheBloke/CodeLlama-13B-GGUF".to_string(),
            "main".to_string(),
            "codellama-13b.Q8_0.gguf".to_string(),
        ))
        .with_tokenizer(llama_tokenizer())
        .with_group_query_attention(1)
    }

    /// A preset for Llama34bCode
    pub fn llama_34b_code() -> Self {
        Self::new(FileSource::huggingface(
            "TheBloke/CodeLlama-34B-GGUF".to_string(),
            "main".to_string(),
            "codellama-34b.Q8_0.gguf".to_string(),
        ))
        .with_tokenizer(llama_tokenizer())
        .with_group_query_attention(1)
    }

    /// A preset for the SOLAR 10.7B model
    pub fn solar_10_7b() -> Self {
        Self::new(FileSource::huggingface(
            "TheBloke/SOLAR-10.7B-v1.0-GGUF".to_string(),
            "main".to_string(),
            "solar-10.7b-v1.0.Q4_K_M.gguf".to_string(),
        ))
        .with_tokenizer(FileSource::huggingface(
            "upstage/SOLAR-10.7B-v1.0".to_string(),
            "main".to_string(),
            "tokenizer.json".to_string(),
        ))
    }

    /// A preset for the SOLAR 10.7B Instruct model
    pub fn solar_10_7b_instruct() -> Self {
        Self::new(FileSource::huggingface(
            "TheBloke/SOLAR-10.7B-Instruct-v1.0-GGUF".to_string(),
            "main".to_string(),
            "solar-10.7b-instruct-v1.0.Q4_K_M.gguf".to_string(),
        ))
        .with_tokenizer(FileSource::huggingface(
            "upstage/SOLAR-10.7B-Instruct-v1.0".to_string(),
            "main".to_string(),
            "tokenizer.json".to_string(),
        ))
    }

    /// A preset for the Qwen2.5-0.5B Chat model
    pub fn qwen_2_5_0_5b_instruct() -> Self {
        Self::new(FileSource::huggingface(
            "Qwen/Qwen2.5-0.5B-Instruct-GGUF".to_string(),
            "main".to_string(),
            "qwen2.5-0.5b-instruct-q4_k_m.gguf".to_string(),
        ))
        .with_tokenizer(qwen_tokenizer())
        .with_group_query_attention(7)
    }

    /// A preset for the Qwen2.5-1.5B Chat model
    pub fn qwen_2_5_1_5b_instruct() -> Self {
        Self::new(FileSource::huggingface(
            "Qwen/Qwen2.5-1.5B-Instruct-GGUF".to_string(),
            "main".to_string(),
            "qwen2.5-1.5b-instruct-q4_k_m.gguf".to_string(),
        ))
        .with_tokenizer(qwen_tokenizer())
        .with_group_query_attention(7)
    }

    /// A preset for the Qwen2.5-3B Chat model
    pub fn qwen_2_5_3b_instruct() -> Self {
        Self::new(FileSource::huggingface(
            "Qwen/Qwen2.5-3B-Instruct-GGUF".to_string(),
            "main".to_string(),
            "qwen2.5-3b-instruct-q4_k_m.gguf".to_string(),
        ))
        .with_tokenizer(qwen_tokenizer())
        .with_group_query_attention(7)
    }

    /// A preset for the Qwen2.5-7B Chat model
    pub fn qwen_2_5_7b_instruct() -> Self {
        Self::new(FileSource::huggingface(
            "bartowski/Qwen2.5-7B-Instruct-GGUF".to_string(),
            "main".to_string(),
            "Qwen2.5-7B-Instruct-Q4_K_M.gguf".to_string(),
        ))
        .with_tokenizer(qwen_tokenizer())
        .with_group_query_attention(7)
    }

    /// A preset for the Qwen3-0.6B Chat model
    ///
    /// This model supports "thinking mode" which can be enabled via the `/think` and `/no_think`
    /// soft switches in the user message. The model wraps reasoning in `<think>...</think>` blocks.
    pub fn qwen_3_0_6b_instruct() -> Self {
        Self::new(FileSource::huggingface(
            "unsloth/Qwen3-0.6B-GGUF".to_string(),
            "main".to_string(),
            "Qwen3-0.6B-Q4_K_M.gguf".to_string(),
        ))
        .with_tokenizer(qwen3_tokenizer())
        .with_group_query_attention(2)
    }

    /// A preset for the Qwen3-1.7B Chat model
    ///
    /// This model supports "thinking mode" which can be enabled via the `/think` and `/no_think`
    /// soft switches in the user message. The model wraps reasoning in `<think>...</think>` blocks.
    pub fn qwen_3_1_7b_instruct() -> Self {
        Self::new(FileSource::huggingface(
            "unsloth/Qwen3-1.7B-GGUF".to_string(),
            "main".to_string(),
            "Qwen3-1.7B-Q4_K_M.gguf".to_string(),
        ))
        .with_tokenizer(qwen3_tokenizer())
        .with_group_query_attention(2)
    }

    /// A preset for the Qwen3-4B Chat model
    ///
    /// This model supports "thinking mode" which can be enabled via the `/think` and `/no_think`
    /// soft switches in the user message. The model wraps reasoning in `<think>...</think>` blocks.
    pub fn qwen_3_4b_instruct() -> Self {
        Self::new(FileSource::huggingface(
            "unsloth/Qwen3-4B-GGUF".to_string(),
            "main".to_string(),
            "Qwen3-4B-Q4_K_M.gguf".to_string(),
        ))
        .with_tokenizer(qwen3_tokenizer())
        .with_group_query_attention(4)
    }

    /// A preset for the Qwen3-8B Chat model
    ///
    /// This model supports "thinking mode" which can be enabled via the `/think` and `/no_think`
    /// soft switches in the user message. The model wraps reasoning in `<think>...</think>` blocks.
    pub fn qwen_3_8b_instruct() -> Self {
        Self::new(FileSource::huggingface(
            "unsloth/Qwen3-8B-GGUF".to_string(),
            "main".to_string(),
            "Qwen3-8B-Q4_K_M.gguf".to_string(),
        ))
        .with_tokenizer(qwen3_tokenizer())
        .with_group_query_attention(4)
    }

    /// A preset for the Qwen3-14B Chat model
    ///
    /// This model supports "thinking mode" which can be enabled via the `/think` and `/no_think`
    /// soft switches in the user message. The model wraps reasoning in `<think>...</think>` blocks.
    pub fn qwen_3_14b_instruct() -> Self {
        Self::new(FileSource::huggingface(
            "unsloth/Qwen3-14B-GGUF".to_string(),
            "main".to_string(),
            "Qwen3-14B-Q4_K_M.gguf".to_string(),
        ))
        .with_tokenizer(qwen3_tokenizer())
        .with_group_query_attention(4)
    }

    /// A preset for the Qwen3-32B Chat model
    ///
    /// This model supports "thinking mode" which can be enabled via the `/think` and `/no_think`
    /// soft switches in the user message. The model wraps reasoning in `<think>...</think>` blocks.
    pub fn qwen_3_32b_instruct() -> Self {
        Self::new(FileSource::huggingface(
            "unsloth/Qwen3-32B-GGUF".to_string(),
            "main".to_string(),
            "Qwen3-32B-Q4_K_M.gguf".to_string(),
        ))
        .with_tokenizer(qwen3_tokenizer())
        .with_group_query_attention(8)
    }

    /// A preset for the DeepSeek-R1 distill qwen 1.5b model
    pub fn deepseek_r1_distill_qwen_1_5b() -> Self {
        Self::new(FileSource::huggingface(
            "bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF".to_string(),
            "main".to_string(),
            "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf".to_string(),
        ))
    }

    /// A preset for the DeepSeek-R1 distill qwen 7b model
    pub fn deepseek_r1_distill_qwen_7b() -> Self {
        Self::new(FileSource::huggingface(
            "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF".to_string(),
            "main".to_string(),
            "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf".to_string(),
        ))
    }

    /// A preset for the DeepSeek-R1 distill qwen 14b model
    pub fn deepseek_r1_distill_qwen_14b() -> Self {
        Self::new(FileSource::huggingface(
            "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF",
            "main",
            "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
        ))
    }

    /// A preset for the DeepSeek-R1 distill llama 8b model
    pub fn deepseek_r1_distill_llama_8b() -> Self {
        Self::new(FileSource::huggingface(
            "bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF".to_string(),
            "main".to_string(),
            "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf".to_string(),
        ))
    }

    /// A preset for gemma 3 270m instruction fine tuned
    ///
    /// Note: The gemma model series does not support system prompts
    pub fn gemma_3_270m_chat() -> Self {
        Self::new(FileSource::huggingface(
            "unsloth/gemma-3-270m-it-GGUF".to_string(),
            "main".to_string(),
            "gemma-3-270m-it-Q4_K_M.gguf".to_string(),
        ))
        .with_tokenizer(FileSource::huggingface(
            "google/gemma-3-270m-it".to_string(),
            "main".to_string(),
            "tokenizer.json".to_string(),
        ))
        .with_override_stop_token_string("<end_of_turn>")
    }

    /// A preset for gemma 3 1b instruction fine tuned
    ///
    /// Note: The gemma model series does not support system prompts
    pub fn gemma_3_1b_chat() -> Self {
        Self::new(FileSource::huggingface(
            "google/gemma-3-1b-it-qat-q4_0-gguf".to_string(),
            "main".to_string(),
            "gemma-3-1b-it-q4_0.gguf".to_string(),
        ))
        .with_tokenizer(FileSource::huggingface(
            "google/gemma-3-1b-it".to_string(),
            "main".to_string(),
            "tokenizer.json".to_string(),
        ))
        .with_override_stop_token_string("<end_of_turn>")
    }

    /// A preset for gemma 3 4b instruction fine tuned
    ///
    /// Note: The gemma model series does not support system prompts
    pub fn gemma_3_4b_chat() -> Self {
        Self::new(FileSource::huggingface(
            "google/gemma-3-4b-it-qat-q4_0-gguf".to_string(),
            "main".to_string(),
            "gemma-3-4b-it-q4_0.gguf".to_string(),
        ))
        .with_tokenizer(FileSource::huggingface(
            "google/gemma-3-4b-it".to_string(),
            "main".to_string(),
            "tokenizer.json".to_string(),
        ))
        .with_override_stop_token_string("<end_of_turn>")
    }

    /// A preset for gemma 3 12b instruction fine tuned
    ///
    /// Note: The gemma model series does not support system prompts
    pub fn gemma_3_12b_chat() -> Self {
        Self::new(FileSource::huggingface(
            "google/gemma-3-12b-it-qat-q4_0-gguf".to_string(),
            "main".to_string(),
            "gemma-3-12b-it-q4_0.gguf".to_string(),
        ))
        .with_tokenizer(FileSource::huggingface(
            "google/gemma-3-12b-it".to_string(),
            "main".to_string(),
            "tokenizer.json".to_string(),
        ))
        .with_override_stop_token_string("<end_of_turn>")
    }

    /// A preset for gemma 3 27b instruction fine tuned
    ///
    /// Note: The gemma model series does not support system prompts
    pub fn gemma_3_27b_chat() -> Self {
        Self::new(FileSource::huggingface(
            "google/gemma-3-27b-it-qat-q4_0-gguf".to_string(),
            "main".to_string(),
            "gemma-3-27b-it-q4_0.gguf".to_string(),
        ))
        .with_tokenizer(FileSource::huggingface(
            "google/gemma-3-27b-it".to_string(),
            "main".to_string(),
            "tokenizer.json".to_string(),
        ))
        .with_override_stop_token_string("<end_of_turn>")
    }

    /// A preset for qwen 2.5 3b VL chat in f16 precision
    pub fn qwen_2_5_3b_vl_chat_f16() -> Self {
        Self::new(kalosm_model_types::FileSource::HuggingFace {
            model_id: "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF".into(),
            revision: "main".into(),
            file: "Qwen2.5-VL-3B-Instruct-f16.gguf".into(),
        })
        .with_vision_model(kalosm_model_types::FileSource::HuggingFace {
            model_id: "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF".into(),
            revision: "main".into(),
            file: "mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf".into(),
        })
        .with_tokenizer(kalosm_model_types::FileSource::HuggingFace {
            model_id: "Qwen/Qwen2.5-VL-3B-Instruct".into(),
            revision: "main".into(),
            file: "tokenizer.json".into(),
        })
    }

    /// A preset for qwen 2.5 3b VL chat in q4 precision
    pub fn qwen_2_5_3b_vl_chat_q4() -> Self {
        Self::new(kalosm_model_types::FileSource::HuggingFace {
            model_id: "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF".into(),
            revision: "main".into(),
            file: "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf".into(),
        })
        .with_vision_model(kalosm_model_types::FileSource::HuggingFace {
            model_id: "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF".into(),
            revision: "main".into(),
            file: "mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf".into(),
        })
        .with_tokenizer(kalosm_model_types::FileSource::HuggingFace {
            model_id: "Qwen/Qwen2.5-VL-3B-Instruct".into(),
            revision: "main".into(),
            file: "tokenizer.json".into(),
        })
    }

    /// A preset for qwen 2.5 3b VL chat in q8 precision
    pub fn qwen_2_5_3b_vl_chat_q8() -> Self {
        Self::new(kalosm_model_types::FileSource::HuggingFace {
            model_id: "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF".into(),
            revision: "main".into(),
            file: "Qwen2.5-VL-3B-Instruct-Q8_0.gguf".into(),
        })
        .with_vision_model(kalosm_model_types::FileSource::HuggingFace {
            model_id: "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF".into(),
            revision: "main".into(),
            file: "mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf".into(),
        })
        .with_tokenizer(kalosm_model_types::FileSource::HuggingFace {
            model_id: "Qwen/Qwen2.5-VL-3B-Instruct".into(),
            revision: "main".into(),
            file: "tokenizer.json".into(),
        })
    }

    /// A preset for qwen 2.5 7b VL chat in f16 precision
    pub fn qwen_2_5_7b_vl_chat_f16() -> Self {
        Self::new(kalosm_model_types::FileSource::HuggingFace {
            model_id: "ggml-org/Qwen2.5-VL-7B-Instruct-GGUF".into(),
            revision: "main".into(),
            file: "Qwen2.5-VL-7B-Instruct-f16.gguf".into(),
        })
        .with_vision_model(kalosm_model_types::FileSource::HuggingFace {
            model_id: "ggml-org/Qwen2.5-VL-7B-Instruct-GGUF".into(),
            revision: "main".into(),
            file: "mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf".into(),
        })
        .with_tokenizer(kalosm_model_types::FileSource::HuggingFace {
            model_id: "Qwen/Qwen2.5-VL-7B-Instruct".into(),
            revision: "main".into(),
            file: "tokenizer.json".into(),
        })
    }

    /// A preset for qwen 2.5 7b VL chat in q4 precision
    pub fn qwen_2_5_7b_vl_chat_q4() -> Self {
        Self::new(kalosm_model_types::FileSource::HuggingFace {
            model_id: "ggml-org/Qwen2.5-VL-7B-Instruct-GGUF".into(),
            revision: "main".into(),
            file: "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf".into(),
        })
        .with_vision_model(kalosm_model_types::FileSource::HuggingFace {
            model_id: "ggml-org/Qwen2.5-VL-7B-Instruct-GGUF".into(),
            revision: "main".into(),
            file: "mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf".into(),
        })
        .with_tokenizer(kalosm_model_types::FileSource::HuggingFace {
            model_id: "Qwen/Qwen2.5-VL-7B-Instruct".into(),
            revision: "main".into(),
            file: "tokenizer.json".into(),
        })
    }

    /// A preset for qwen 2.5 7b VL chat in q8 precision
    pub fn qwen_2_5_7b_vl_chat_q8() -> Self {
        Self::new(kalosm_model_types::FileSource::HuggingFace {
            model_id: "ggml-org/Qwen2.5-VL-7B-Instruct-GGUF".into(),
            revision: "main".into(),
            file: "Qwen2.5-VL-7B-Instruct-Q8_0.gguf".into(),
        })
        .with_vision_model(kalosm_model_types::FileSource::HuggingFace {
            model_id: "ggml-org/Qwen2.5-VL-7B-Instruct-GGUF".into(),
            revision: "main".into(),
            file: "mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf".into(),
        })
        .with_tokenizer(kalosm_model_types::FileSource::HuggingFace {
            model_id: "Qwen/Qwen2.5-VL-7B-Instruct".into(),
            revision: "main".into(),
            file: "tokenizer.json".into(),
        })
    }

    /// A preset for qwen 2.5 32b VL chat in f4 precision
    pub fn qwen_2_5_32b_vl_chat_q4() -> Self {
        Self::new(kalosm_model_types::FileSource::HuggingFace {
            model_id: "ggml-org/Qwen2.5-VL-32B-Instruct-GGUF".into(),
            revision: "main".into(),
            file: "Qwen2.5-VL-32B-Instruct-Q4_K_M.gguf".into(),
        })
        .with_vision_model(kalosm_model_types::FileSource::HuggingFace {
            model_id: "ggml-org/Qwen2.5-VL-32B-Instruct-GGUF".into(),
            revision: "main".into(),
            file: "mmproj-Qwen2.5-VL-32B-Instruct-f16.gguf".into(),
        })
        .with_tokenizer(kalosm_model_types::FileSource::HuggingFace {
            model_id: "Qwen/Qwen2.5-VL-32B-Instruct".into(),
            revision: "main".into(),
            file: "tokenizer.json".into(),
        })
    }

    /// A preset for qwen 2.5 32b VL chat in f8 precision
    pub fn qwen_2_5_32b_vl_chat_q8() -> Self {
        Self::new(kalosm_model_types::FileSource::HuggingFace {
            model_id: "ggml-org/Qwen2.5-VL-32B-Instruct-GGUF".into(),
            revision: "main".into(),
            file: "Qwen2.5-VL-32B-Instruct-Q8_0.gguf".into(),
        })
        .with_vision_model(kalosm_model_types::FileSource::HuggingFace {
            model_id: "ggml-org/Qwen2.5-VL-32B-Instruct-GGUF".into(),
            revision: "main".into(),
            file: "mmproj-Qwen2.5-VL-32B-Instruct-f16.gguf".into(),
        })
        .with_tokenizer(kalosm_model_types::FileSource::HuggingFace {
            model_id: "Qwen/Qwen2.5-VL-32B-Instruct".into(),
            revision: "main".into(),
            file: "tokenizer.json".into(),
        })
    }

    /// A preset for qwen 2.5 32b VL chat in f16 precision
    pub fn qwen_2_5_32b_vl_chat_f16() -> Self {
        Self::new_sharded(vec![
            kalosm_model_types::FileSource::HuggingFace {
                model_id: "ggml-org/Qwen2.5-VL-32B-Instruct-GGUF".into(),
                revision: "main".into(),
                file: "Qwen2.5-VL-32B-Instruct-f16-00001-of-00002.gguf".into(),
            },
            kalosm_model_types::FileSource::HuggingFace {
                model_id: "ggml-org/Qwen2.5-VL-32B-Instruct-GGUF".into(),
                revision: "main".into(),
                file: "Qwen2.5-VL-32B-Instruct-f16-00002-of-00002.gguf".into(),
            },
        ])
        .with_vision_model(kalosm_model_types::FileSource::HuggingFace {
            model_id: "ggml-org/Qwen2.5-VL-32B-Instruct-GGUF".into(),
            revision: "main".into(),
            file: "mmproj-Qwen2.5-VL-32B-Instruct-f16.gguf".into(),
        })
        .with_tokenizer(kalosm_model_types::FileSource::HuggingFace {
            model_id: "Qwen/Qwen2.5-VL-32B-Instruct".into(),
            revision: "main".into(),
            file: "tokenizer.json".into(),
        })
    }
}

impl Default for LlamaSource {
    fn default() -> Self {
        Self::llama_3_1_8b_chat()
    }
}
