use std::sync::Arc;

use crate::sampler::{CpuMirostat2Sampler, Logits};
use crate::tokenizer::{LlamaTokenizer, LlamaTokenizerError};
#[cfg(feature = "structured")]
use rayon::iter::{IntoParallelIterator, ParallelExtend, ParallelIterator};
use thiserror::Error;

/// An error that can occur when performing streaming detokenization.
#[derive(Debug, Error)]
pub enum TokenOutputStreamError {
    /// An error that can occur when tokenizing.
    #[error("Tokenization error: {0}")]
    TokenizationError(#[from] LlamaTokenizerError),

    /// The sampler did not sample any tokens.
    #[error("No token sampled")]
    NoTokenSampled,
}

/// This is a wrapper around a tokenizer to ensure that tokens can be returned to the user in a
/// streaming way rather than having to wait for the full decoding.
pub struct TokenOutputStream {
    tokenizer: Arc<LlamaTokenizer>,
    tokens: Vec<u32>,
    current_text: String,
    prev_index: usize,
    current_index: usize,
}

impl TokenOutputStream {
    /// Creates a new token output stream.
    pub fn new(tokenizer: Arc<LlamaTokenizer>) -> Self {
        Self {
            tokenizer,
            current_text: Default::default(),
            prev_index: 0,
            current_index: 0,
            tokens: Vec::new(),
        }
    }

    fn decode(&self, tokens: &[u32]) -> Result<String, TokenOutputStreamError> {
        self.tokenizer.decode(tokens, false).map_err(Into::into)
    }

    /// Samples a token from the logits.
    pub fn sample_token(
        &self,
        sampler: &mut CpuMirostat2Sampler,
        mut logits: Logits,
        stop_on: Option<&str>,
        top_k: usize,
    ) -> Result<u32, TokenOutputStreamError> {
        let tokenizer = &self.tokenizer;
        let previous_tokens = &self.tokens;

        let mut end_tokens = String::new();
        // grab as many characters as the stop_on string has from the end of the previous tokens
        if let Some(stop_on) = stop_on {
            let required_len = stop_on.len();
            let mut previous_token_iter = previous_tokens.iter().rev();
            while end_tokens.len() < required_len {
                match previous_token_iter.next() {
                    Some(token) => {
                        end_tokens = tokenizer
                            .decode(&[*token], true)
                            .map_err(TokenOutputStreamError::TokenizationError)?
                            .to_string()
                            + &end_tokens;
                    }
                    None => {
                        break;
                    }
                }
            }
        }
        for logit in logits.iter_mut() {
            let tid = logit.token_id;
            if let Some(stop_on) = stop_on {
                let token = tokenizer.decode(&[tid], false).unwrap();
                let combined = end_tokens.clone() + &token;
                if combined.contains(stop_on) && !combined.ends_with(stop_on) {
                    // If the token contains a stop string, but not at the end, remove it from sampling.
                    logit.logit = f32::NEG_INFINITY;
                }
            }
        }
        sampler
            .sample_token(&mut logits, previous_tokens, top_k)
            .ok_or(TokenOutputStreamError::NoTokenSampled)
    }

    /// Encode a string into a list of tokens after the current tokens.
    #[cfg(feature = "structured")]
    pub(crate) fn encode_after(
        &self,
        text: &str,
    ) -> Result<Option<Vec<u32>>, TokenOutputStreamError> {
        let all_text = self.current_text.clone() + text;
        let tokens_with_current_tokens = self
            .tokenizer
            .encode(&all_text, false)
            .map_err(TokenOutputStreamError::TokenizationError)?;

        let index_length = self.current_index - self.prev_index;

        // Some tokenizers may tokenize new text differently than the two tokens concatenated together. If they do, this function returns None
        if tokens_with_current_tokens.len() <= index_length {
            return Ok(None);
        }
        let (current_tokens, new_tokens) = tokens_with_current_tokens.split_at(index_length);

        if current_tokens != &self.tokens[self.prev_index..self.current_index] {
            return Ok(None);
        }

        Ok(Some(new_tokens.to_vec()))
    }

    /// Recalculate the current text
    pub(crate) fn recalculate_current_text(&mut self) -> Result<(), TokenOutputStreamError> {
        let current_text = if self.tokens.is_empty() {
            Default::default()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?.to_string()
        };

        self.current_text = current_text;

        Ok(())
    }

    /// Returns the next token.
    pub fn next_token(&mut self, token: u32) -> Result<Option<String>, TokenOutputStreamError> {
        let prev_text = &self.current_text;
        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() && text.chars().last().unwrap().is_ascii() {
            let text = text.split_at(prev_text.len());
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            self.recalculate_current_text()?;
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    /// Returns the next tokens
    #[cfg(feature = "structured")]
    pub fn next_tokens(
        &mut self,
        tokens: &[u32],
    ) -> Result<Option<String>, TokenOutputStreamError> {
        let prev_text = &self.current_text;
        self.tokens.extend(tokens.iter().copied());
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() && text.chars().last().unwrap().is_ascii() {
            let text = text.split_at(prev_text.len());
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            self.recalculate_current_text()?;
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    /// Peek encoding many next tokens (in sequence)
    #[cfg(feature = "structured")]
    pub fn peek_next_tokens(
        &self,
        tokens: impl IntoIterator<Item = u32>,
    ) -> Result<Option<String>, TokenOutputStreamError> {
        let mut current_tokens = self.tokens[self.prev_index..].to_vec();
        let prev_text = &self.current_text;
        current_tokens.extend(tokens);
        let text = self.decode(&current_tokens)?;
        if text.len() > prev_text.len() && text.chars().last().unwrap().is_ascii() {
            let text = text.split_at(prev_text.len());
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    /// Peek many possible next tokens in parallel
    #[cfg(feature = "structured")]
    pub fn peek_tokens(
        &self,
        tokens: impl IntoParallelIterator<Item = u32>,
        into: &mut impl ParallelExtend<Option<String>>,
    ) {
        let prev_text = &self.current_text;
        let prev_text_len = prev_text.len();
        into.par_extend(tokens.into_par_iter().map_init(
            || self.tokens[self.prev_index..].to_vec(),
            |tokens, token| {
                tokens.push(token);
                let text = self.decode(tokens).ok()?;
                tokens.pop();
                if text.len() > prev_text_len && text.chars().last().unwrap().is_ascii() {
                    let text = text.split_at(prev_text_len);
                    Some(text.1.to_string())
                } else {
                    None
                }
            },
        ));
    }

    /// Peek the next token.
    #[cfg(feature = "structured")]
    pub fn peek_token(&self, token: u32) -> Result<Option<String>, TokenOutputStreamError> {
        let prev_text = &self.current_text;
        let prev_text_len = prev_text.len();
        let mut tokens = self.tokens[self.prev_index..].to_vec();
        tokens.push(token);
        let text = self.decode(&tokens)?;
        tokens.pop();
        if text.len() > prev_text_len && text.chars().last().unwrap().is_ascii() {
            let text = text.split_at(prev_text_len);
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    /// Get the tokens
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
}
