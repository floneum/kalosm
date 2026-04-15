use fusor::{
    AddOp, CastTensor, CastTo, FloatDataType, FloatOps, MatmulImpl, MulOp, SimdBinaryOp,
    SimdElement, SimdReduceOp, SumOp, WasmNotSend, WasmNotSync,
};
use kalosm_language_model::{ContentChunk, MessageContent};
use kalosm_sample::CreateParserState;
use kalosm_sample::{LiteralParser, ParseStatus, Parser, ParserExt};
use llm_samplers::prelude::{Logit, Logits};
use llm_samplers::types::{HasSamplerResources, Sampler, SamplerError};
use rand::SeedableRng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::{
    fmt::{Debug, Formatter},
    sync::{Arc, Mutex},
};
use tokenizers::tokenizer::Tokenizer;

use crate::model::LlamaModelError;
use crate::token_stream::TokenOutputStream;
use crate::{LlamaModel, LlamaSession};

#[allow(clippy::too_many_arguments)]
pub(crate) async fn generate_structured<F, P: Parser>(
    prompt: MessageContent,
    llm: &LlamaModel<F>,
    session: &mut LlamaSession<F>,
    parser: P,
    parser_state: P::PartialState,
    mut sampler: Arc<Mutex<dyn Sampler>>,
    mut on_token: impl FnMut(String) -> Result<(), LlamaModelError>,
    top_k: Option<usize>,
    seed: Option<u64>,
) -> Result<P::Output, LlamaModelError>
where
    F: FloatDataType
        + SimdElement
        + Default
        + CastTo<f32>
        + CastTensor<f32>
        + WasmNotSend
        + WasmNotSync
        + FloatOps
        + MatmulImpl
        + 'static,
    f32: CastTo<F> + CastTensor<F>,
    MulOp: SimdBinaryOp<F>,
    AddOp: SimdBinaryOp<F>,
    SumOp: SimdReduceOp<F>,
{
    let eos_token = llm.model.config.stop_token_string.clone();
    let mut on_token = move |tok: String| {
        if tok == eos_token {
            return Ok(());
        }
        on_token(tok)
    };
    let tokenizer = &llm.tokenizer;

    let prompt_text = prompt.text();
    let images = prompt
        .chunks()
        .iter()
        .filter_map(|chunk| {
            if let ContentChunk::Media(media) = chunk {
                media.source().as_bytes().as_ref().map(|bytes| {
                    image::load_from_memory(bytes).map(|img| (img, media.hints().clone()))
                })
            } else {
                None
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let prompt_tokens = tokenizer
        .encode_fast(prompt_text, false)
        .map_err(LlamaModelError::Tokenizer)?;
    let mut prompt_tokens = prompt_tokens.get_ids();

    // Prompt healing
    // Trim the last token and add what it would decode to into the constraints
    let last_token = if let Some((last, tokens)) = prompt_tokens.split_last() {
        if tokenizer.get_added_tokens_decoder().contains_key(last) {
            None
        } else {
            prompt_tokens = tokens;
            Some(*last)
        }
    } else {
        None
    };

    let mut unprocessed_token_count = prompt_tokens.len();
    let mut token_stream = TokenOutputStream::new(tokenizer.clone());
    for token in prompt_tokens {
        token_stream
            .next_token(*token)
            .map_err(LlamaModelError::TokenOutputStreamError)?;
    }

    let remaining_prompt_text = last_token
        .map(|token| {
            token_stream
                .peek_token(token)
                .map_err(LlamaModelError::TokenOutputStreamError)
        })
        .transpose()?
        .flatten()
        .unwrap_or_default();

    let parser = LiteralParser::new(remaining_prompt_text.clone())
        .ignore_output_then(parser.with_initial_state(move || parser_state.clone()));
    {
        let mut parser_state = parser.create_parser_state();
        for c in remaining_prompt_text.chars() {
            let str = c.to_string();
            let bytes = str.as_bytes();
            let (parser_state_new, _) = parser
                .parse(&parser_state, bytes)
                .unwrap()
                .unwrap_incomplete();
            parser_state = parser_state_new;
        }
    }
    let mut parser_state = parser.create_parser_state();
    let mut strip_required_next = true;

    let mut rng = if let Some(seed) = seed {
        rand::rngs::StdRng::seed_from_u64(seed)
    } else {
        rand::rngs::StdRng::from_os_rng()
    };
    let mut state_map = vec![];
    let mut logits_indexed = Vec::new();
    let mut token_cache = DetokenizationCache::new();
    let mut logits = Logits::default();

    loop {
        let tokens = token_stream.tokens();
        let logit_probs = {
            let mut session_lock = session
                .cache
                .write()
                .map_err(|err| LlamaModelError::Session(err.to_string()))?;
            LlamaModel::forward(
                &llm.model,
                &llm.device,
                &tokens[tokens.len() - unprocessed_token_count..],
                &images,
                Some(&mut *session_lock),
                &llm.tokenizer,
            )
        }
        .await?;
        let resources = &mut SamplerResources {
            previous_tokens: tokens,
            rng: &mut rng,
        };

        // fill the state map with None for each token
        token_cache.clear(logit_probs.len());
        state_map.clear();
        logits_indexed.clear();
        logits.clear();
        for (id, prob) in logit_probs.iter().enumerate() {
            logits_indexed.push(Logit {
                token_id: id as u32,
                logit: *prob,
                prob: 0f32,
            });
            state_map.push(None);
        }

        let mut valid_tokens = false;

        // If we don't have a top k, then we can just cache the entire detokenization
        if top_k.is_none() {
            token_cache.expand(
                &(0..logit_probs.len() as u32).collect::<Vec<_>>(),
                &token_stream,
            );
        }

        const DETOKENIZATION_INITIAL_BATCH_SIZE: usize = 64;

        // Constraints tend to be either very difficult to satisfy or very easy to satisfy
        // We exponentially increase the batch size as a balance between the two
        // If the first half of the tokens are invalid, it is unlikely that the first 64 tokens of the second half will be valid
        let mut detokenization_batch_size = DETOKENIZATION_INITIAL_BATCH_SIZE;

        let mut partitioned_logits_index = top_k.map(|_| 0);

        for i in 0..logits_indexed.len() {
            // If we have top k enabled, and there are less than top k - committed logits sorted, we need to expand the partitioned logits
            if let (Some(top_k), Some(partitioned_index)) = (top_k, partitioned_logits_index) {
                // If the remaining logits are less than the top k, no need to partition
                let remaining_needed = top_k - logits.len();
                let remaining_possible = partitioned_index - i;
                if remaining_possible <= remaining_needed {
                    // We batch together updates to the cache by detokenization_batch_size
                    let logits_to_update = (remaining_needed.max(detokenization_batch_size))
                        .min(logits_indexed.len() - 1 - i);
                    let new_partitioned_index = i + logits_to_update;

                    // If we eliminated a logit, our partitioning of the logits is no longer valid
                    logits_indexed[i..].select_nth_unstable_by(logits_to_update, cmp_logits);
                    logits_indexed[i..=new_partitioned_index].sort_unstable_by(cmp_logits);
                    // Expand the cache to include the new logits
                    partitioned_logits_index = Some(new_partitioned_index);
                    token_cache.expand_with_logits(
                        &logits_indexed[i..=new_partitioned_index],
                        &token_stream,
                    );

                    // Double the batch size for next time
                    detokenization_batch_size = detokenization_batch_size.saturating_mul(4);
                }
            }

            let Logit {
                token_id, logit, ..
            } = logits_indexed[i];
            let Some(text) = token_cache.get(token_id as usize) else {
                continue;
            };
            if let Ok(result) = parser.parse(&parser_state, text.as_bytes()) {
                let parsed_bytes = match result {
                    ParseStatus::Finished { remaining, .. } => text.len() - remaining.len(),
                    ParseStatus::Incomplete { .. } => text.len(),
                };
                let result = result.without_remaining();
                state_map[token_id as usize] = Some((result, parsed_bytes));
                valid_tokens = true;
                logits.push(Logit {
                    token_id,
                    logit,
                    prob: 0f32,
                });
                // If we only need to keep the top k logits, then we can quit early once we have enough
                if let Some(top_k) = top_k {
                    if logits.len() >= top_k {
                        break;
                    }
                }
            }
        }

        // If there are no valid tokens, return an error
        if !valid_tokens {
            return Err(LlamaModelError::NoValidTokens);
        }
        let token_id = sampler
            .sample_token(resources, &mut logits)
            .map_err(|err| LlamaModelError::SamplerError(err.into()))?
            .ok_or(LlamaModelError::NoValidTokens)?;

        unprocessed_token_count = 1;
        let (result, parsed_bytes) = state_map
            .get_mut(token_id as usize)
            .unwrap()
            .take()
            .unwrap_or_else(|| panic!("Token {token_id} not found in state map"));
        let mut token = token_stream
            .next_token(token_id)
            .map_err(LlamaModelError::TokenOutputStreamError)?
            .unwrap();
        token.truncate(parsed_bytes);
        tracing::trace!("Adding token {} to parser", token);
        // If we are still loading the initial prompt, don't send that part of the text
        if strip_required_next {
            if let Some(stripped) = token.strip_prefix(&remaining_prompt_text) {
                token = stripped.to_string();
            }
            strip_required_next = false;
        }
        on_token(token)?;

        if let Some(result) = update_state(
            &parser,
            &mut parser_state,
            result,
            tokenizer,
            &mut token_stream,
            &mut on_token,
            &mut unprocessed_token_count,
        )? {
            return Ok(result);
        }
    }
}

fn cmp_logits(a: &Logit, b: &Logit) -> std::cmp::Ordering {
    f32::total_cmp(&b.logit, &a.logit)
}

#[allow(unused, clippy::all)]
fn update_state<P: Parser>(
    parser: &P,
    parser_state: &mut P::PartialState,
    result: ParseStatus<P::PartialState, P::Output>,
    tokenizer: &Tokenizer,
    token_stream: &mut TokenOutputStream,
    on_token: &mut impl FnMut(String) -> Result<(), LlamaModelError>,
    unprocessed_token_count: &mut usize,
) -> Result<Option<P::Output>, LlamaModelError> {
    match result {
        kalosm_sample::ParseStatus::Incomplete {
            new_state,
            required_next,
        } => {
            *parser_state = new_state;
            if required_next.is_empty() {
                Ok(None)
            } else {
                // The token may decode to a string that is a valid prefix of the required next token, but in a way that doesn't let us decode the required next tokens
                let Some(mut extra_tokens) = token_stream
                    .encode_after(&required_next)
                    .map_err(|err| LlamaModelError::TokenOutputStreamError(err))?
                else {
                    return Ok(None);
                };
                // Remove the last token to avoid influencing the next token
                extra_tokens.pop();
                // If there are no new tokens, continue generating tokens normally
                if extra_tokens.is_empty() {
                    return Ok(None);
                }

                let mut all_required_next = String::new();
                if let Some(next) = token_stream
                    .peek_next_tokens(extra_tokens.iter().copied())
                    .map_err(|err| LlamaModelError::TokenOutputStreamError(err))?
                {
                    all_required_next = next;
                }
                // The token may decode to a string that is a valid prefix of the required next token, but in a way that doesn't encode the same way.
                // Make sure the final text we are adding is actually valid
                if !required_next.starts_with(&all_required_next) {
                    return Ok(None);
                }
                token_stream
                    .next_tokens(&extra_tokens)
                    .map_err(|err| LlamaModelError::TokenOutputStreamError(err))?;
                *unprocessed_token_count += extra_tokens.len();
                on_token(all_required_next.clone())?;
                let mut result = parser
                    .parse(parser_state, all_required_next.as_bytes())
                    .unwrap_or_else(|_| {
                        unreachable!("Required next should always be valid attempted to add {:?} but got error", all_required_next)
                });
                update_state(
                    parser,
                    parser_state,
                    result,
                    tokenizer,
                    token_stream,
                    on_token,
                    unprocessed_token_count,
                )
            }
        }
        kalosm_sample::ParseStatus::Finished { result, .. } => Ok(Some(result)),
    }
}

struct SamplerResources<'a, 'b, R: rand::Rng> {
    rng: &'a mut R,
    previous_tokens: &'b [u32],
}

impl<R> Debug for SamplerResources<'_, '_, R>
where
    R: rand::Rng,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SamplerResources")
            .field("previous_tokens", &self.previous_tokens)
            .finish()
    }
}

impl<R> HasSamplerResources for SamplerResources<'_, '_, R>
where
    R: rand::Rng,
{
    fn with_rng_mut(
        &mut self,
        fun: &mut dyn FnMut(&mut dyn rand::RngCore),
    ) -> Result<(), SamplerError> {
        fun(self.rng);
        Ok(())
    }

    fn with_last_tokens(&self, fun: &mut dyn FnMut(&[u32])) -> Result<(), SamplerError> {
        fun(self.previous_tokens);
        Ok(())
    }
}

#[derive(Clone)]
enum TokenCacheStatus {
    Empty,
    Invalid,
    Valid(String),
}

struct DetokenizationCache {
    cache: Box<[TokenCacheStatus]>,
    vec: Vec<Option<String>>,
}

impl DetokenizationCache {
    fn new() -> Self {
        Self {
            cache: Box::new([]),
            vec: Vec::new(),
        }
    }

    fn get(&self, index: usize) -> Option<&str> {
        match &self.cache[index] {
            TokenCacheStatus::Empty => panic!("cache for token {index} is empty"),
            TokenCacheStatus::Invalid => None,
            TokenCacheStatus::Valid(token) => Some(token),
        }
    }

    fn expand_with_logits(&mut self, tokens: &[Logit], stream: &TokenOutputStream) {
        stream.peek_tokens(
            tokens.into_par_iter().map(|logit| logit.token_id),
            &mut self.vec,
        );

        for (logit, token) in tokens.iter().zip(self.vec.drain(..)) {
            self.cache[logit.token_id as usize] = match token {
                Some(token) => TokenCacheStatus::Valid(token),
                None => TokenCacheStatus::Invalid,
            };
        }
    }

    fn expand(&mut self, tokens: &[u32], stream: &TokenOutputStream) {
        stream.peek_tokens(tokens.into_par_iter().copied(), &mut self.vec);

        for (&i, token) in tokens.iter().zip(self.vec.drain(..)) {
            self.cache[i as usize] = match token {
                Some(token) => TokenCacheStatus::Valid(token),
                None => TokenCacheStatus::Invalid,
            };
        }
    }

    fn clear(&mut self, size: usize) {
        if self.cache.len() == size {
            for token in self.cache.iter_mut() {
                *token = TokenCacheStatus::Empty;
            }
        } else {
            self.cache = vec![TokenCacheStatus::Empty; size].into_boxed_slice();
        }
        self.vec.clear();
    }
}
