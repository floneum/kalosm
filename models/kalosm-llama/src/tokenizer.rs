use crate::gguf_tokenizer::{GgufTokenizer, GgufTokenizerError};
use thiserror::Error;

/// A tokenizer used by Llama models.
#[derive(Clone)]
pub struct LlamaTokenizer {
    inner: LlamaTokenizerInner,
}

#[derive(Clone)]
enum LlamaTokenizerInner {
    Gguf(GgufTokenizer),
    #[cfg(feature = "hf-tokenizer-json")]
    HuggingFace(tokenizers::Tokenizer),
}

impl LlamaTokenizer {
    pub(crate) fn from_gguf(tokenizer: GgufTokenizer) -> Self {
        Self {
            inner: LlamaTokenizerInner::Gguf(tokenizer),
        }
    }

    #[cfg(feature = "hf-tokenizer-json")]
    pub(crate) fn from_hf_bytes(bytes: Vec<u8>) -> Result<Self, LlamaTokenizerError> {
        Ok(Self {
            inner: LlamaTokenizerInner::HuggingFace(tokenizers::Tokenizer::from_bytes(bytes)?),
        })
    }

    /// Encode text into token ids.
    pub fn encode(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Vec<u32>, LlamaTokenizerError> {
        match &self.inner {
            LlamaTokenizerInner::Gguf(tokenizer) => tokenizer
                .encode(text, add_special_tokens)
                .map_err(LlamaTokenizerError::from),
            #[cfg(feature = "hf-tokenizer-json")]
            LlamaTokenizerInner::HuggingFace(tokenizer) => Ok(tokenizer
                .encode_fast(text, add_special_tokens)?
                .get_ids()
                .to_vec()),
        }
    }

    /// Decode token ids into text.
    pub fn decode(
        &self,
        tokens: &[u32],
        skip_special_tokens: bool,
    ) -> Result<String, LlamaTokenizerError> {
        match &self.inner {
            LlamaTokenizerInner::Gguf(tokenizer) => {
                Ok(tokenizer.decode(tokens, skip_special_tokens))
            }
            #[cfg(feature = "hf-tokenizer-json")]
            LlamaTokenizerInner::HuggingFace(tokenizer) => {
                Ok(tokenizer.decode(tokens, skip_special_tokens)?)
            }
        }
    }

    #[cfg(feature = "structured")]
    pub(crate) fn is_added_or_special_token(&self, token: u32) -> bool {
        match &self.inner {
            LlamaTokenizerInner::Gguf(tokenizer) => tokenizer.is_special_token(token),
            #[cfg(feature = "hf-tokenizer-json")]
            LlamaTokenizerInner::HuggingFace(tokenizer) => {
                tokenizer.get_added_tokens_decoder().contains_key(&token)
            }
        }
    }
}

/// Errors from Llama tokenization.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LlamaTokenizerError {
    /// The embedded GGUF tokenizer failed.
    #[error("GGUF tokenizer error: {0}")]
    Gguf(String),

    /// The Hugging Face tokenizer failed.
    #[cfg(feature = "hf-tokenizer-json")]
    #[error("Hugging Face tokenizer error: {0}")]
    HuggingFace(#[from] tokenizers::Error),
}

impl From<GgufTokenizerError> for LlamaTokenizerError {
    fn from(err: GgufTokenizerError) -> Self {
        Self::Gguf(err.to_string())
    }
}
