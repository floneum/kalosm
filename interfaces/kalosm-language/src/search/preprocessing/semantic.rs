/// Semantic chunks try to chunk together sentences with a similar meaning.
///
/// It starts by embedding the text and then merges chunks together with a score that incentivizes:
/// - Small chunks to merge with adjacent chunks. Small chunks often have very little meaning on their own. In "I am doing very well. What about you?", the sentence "What about you?" doesn't mean much on its own, but when you mere it with the previous sentence, it makes more sense.
/// - Similar chunks to merge. If two chunks are very similar, they are more likely to be merged together.
/// - Very large chunks to stay separate. If two chunks are very large, they are more likely to be kept separate. If we just merge anything that is similar, very large chunks tend to form. We want to keep some level of different chunks, so we don't always merge similar large chunks together.
///
/// The goals here are a bit difficult to define.
/// - We want chunks that have enough context to have meaning
/// - We also want to keep those chunks as small as possible while retaining that meaning to make them cheaper to feed to the LLM
/// - Ideally the chunks have unique embeddings so it easier to find them with a query vector
//
///   Potential ways we could improve the quality of the chunks:
/// - Word embeddings for first level pass? The sentence embeddings this implementation uses are very slow for large documents.
/// - Estimating merges by averaging the embeddings of the two chunks? Very similar chunks tend to have an embedding after merging that is very similar to the average of the embeddings of the two chunks (cos similarity of > 0.95)
/// - Find Keywords to detect what chunks refer to similar concepts? It is difficult to tell the meaning of some text that only refers to previous chunks. For example, "This further emphasizes the importance of the previous paragraph." means nothing on its own. It would be nice to know to chunk that with the previous paragraph.
/// - Try to optimize embedding chunks to be as different as possible from each other?
/// - Chunk parentheses, and quotes together? This seems fairly straightforward. Everything inside a short quote or parenthesis is likely to be similar.
use crate::prelude::*;
use kalosm_language_model::*;

#[derive(Debug, Clone)]
struct SemanticChunk {
    range: std::ops::Range<usize>,
    sentences: usize,
    embedding: Embedding,
    distance_to_next: Option<f32>,
}

/// A chunker that tries to create chunks of wroughly the same size while grouping together chunks with a similar meaning.
///
/// It starts by embedding the text and then merges chunks together while trying to create chunks with one coherent meaning without too many sentences.
pub struct SemanticChunker {
    /// The score we are trying to achieve when merging chunks together. Once we reach this score, we stop merging chunks together.
    target_score: f32,
    /// The maximum bonus for merging small chunks together. (default: 10.0)
    small_chunk_merge_bonus: f32,
    /// The exponent for characters in the bonus for merging a small chunk with an adjacent chunk. (default: -2.0)
    small_chunk_exponent: f32,
    /// The maximum penalty for merging large chunks together. (default: 200.0)
    large_chunk_penalty: f32,
    /// The exponent for sentences in the penalty for merging a large chunk with an adjacent token. (default: 1.5)
    large_chunk_exponent: f32,
}

impl Default for SemanticChunker {
    fn default() -> Self {
        Self::new()
    }
}

impl SemanticChunker {
    /// Create a new [`SemanticChunker`].
    pub const fn new() -> Self {
        Self {
            target_score: 0.65,
            small_chunk_merge_bonus: 10.0,
            small_chunk_exponent: -2.0,
            large_chunk_penalty: 200.0,
            large_chunk_exponent: 1.5,
        }
    }

    /// Set the target score for the chunker. Merging chunks will stop once this is the maximum score of merging two chunks together. A higher score will result in smaller chunks because merging chunks together will stop at a higher score.
    pub fn with_target_score(mut self, target_score: f32) -> Self {
        self.target_score = target_score;
        self
    }

    /// Set the maximum bonus for merging small chunks together. (default: 10.0)
    ///
    /// The bonus for merging two small chunks together is `small_chunk_merge_bonus * (min(bytes_in_first_chunk, bytes_in_second_chunk) ^ small_chunk_exponent)`.
    pub fn with_small_chunk_merge_bonus(mut self, small_chunk_merge_bonus: f32) -> Self {
        self.small_chunk_merge_bonus = small_chunk_merge_bonus;
        self
    }

    /// Set the exponent for characters in the bonus for merging a small chunk with an adjacent chunk. (default: -2.0)
    ///
    /// The bonus for merging two small chunks together is `small_chunk_merge_bonus * (min(bytes_in_first_chunk, bytes_in_second_chunk) ^ small_chunk_exponent)`.
    pub fn with_small_chunk_exponent(mut self, small_chunk_exponent: f32) -> Self {
        self.small_chunk_exponent = small_chunk_exponent;
        self
    }

    /// Set the maximum penalty for merging large chunks together. (default: 200.0)
    ///
    /// The penalty for merging two large chunks together is `-(max(sentences_in_first_chunk, sentences_in_second_chunk) ^ large_chunk_exponent)/large_chunk_penalty`.
    pub fn with_large_chunk_penalty(mut self, large_chunk_penalty: f32) -> Self {
        self.large_chunk_penalty = large_chunk_penalty;
        self
    }

    /// Set the exponent for sentences in the penalty for merging a large chunk with an adjacent token. (default: 1.5)
    ///
    /// The penalty for merging two large chunks together is `-(max(sentences_in_first_chunk, sentences_in_second_chunk) ^ large_chunk_exponent)/large_chunk_penalty`.
    pub fn with_large_chunk_exponent(mut self, large_chunk_exponent: f32) -> Self {
        self.large_chunk_exponent = large_chunk_exponent;
        self
    }

    fn score_merge(&self, first_chunk: &SemanticChunk, second_chunk: &SemanticChunk) -> f32 {
        // Score higher if one of the chunks is very short
        let short_chunk_merge_bonus = (self.small_chunk_merge_bonus
            / first_chunk.range.len().min(second_chunk.range.len()) as f32)
            .powf(self.small_chunk_exponent);
        // Score lower if the chunks are very long
        let large_chunk_penalty = (first_chunk.sentences.max(second_chunk.sentences) as f32)
            .powf(self.large_chunk_exponent)
            / -self.large_chunk_penalty;
        // Score higher if the similarity is high
        let similarity = first_chunk.distance_to_next.unwrap();
        similarity + short_chunk_merge_bonus + large_chunk_penalty
    }
}

impl Chunker for SemanticChunker {
    type Error<E: Send + Sync + 'static> = E;

    async fn chunk<E: Embedder + Send>(
        &self,
        document: &Document,
        embedder: &E,
    ) -> Result<Vec<Chunk>, E::Error> {
        let text = document.body();

        let mut current_chunks = Vec::new();

        // First chunk by sentences
        let chunker = ChunkStrategy::Sentence {
            sentence_count: 1,
            overlap: 0,
        };

        let mut initial_chunks = Vec::new();
        for chunk in chunker.chunk_str(text) {
            let trimmed = text[chunk.clone()].trim();
            if !trimmed.is_empty() {
                current_chunks.push(chunk);
                initial_chunks.push(trimmed.to_string());
            }
        }

        let embeddings = embedder.embed_vec(initial_chunks).await?;

        let mut chunks = Vec::new();

        // Find the chain of distances between sequential embeddings
        for (i, chunk) in current_chunks.iter().enumerate() {
            if i == current_chunks.len() - 1 {
                chunks.push(SemanticChunk {
                    range: chunk.clone(),
                    sentences: 1,
                    embedding: embeddings[i].clone(),
                    distance_to_next: None,
                });
                break;
            }
            let first = &embeddings[i];
            let second = &embeddings[i + 1];
            let distance_to_next = first.cosine_similarity(second);
            let chunk = SemanticChunk {
                range: chunk.clone(),
                sentences: 1,
                embedding: first.clone(),
                distance_to_next: Some(distance_to_next),
            };
            chunks.push(chunk);
        }

        // Now loop until we have the right number of chunks merging the two closest chunks
        // Find the lowest distance chunk
        while let Some((index, first_chunk)) = chunks
            .iter()
            .enumerate()
            .filter(|(_, c)| c.distance_to_next.is_some())
            .max_by(|(index, c1), (index2, c2)| {
                // Score higher if the similarity is high or if the text size of both is small
                let c1_score = self.score_merge(c1, &chunks[index + 1]);
                let c2_score = self.score_merge(c2, &chunks[index2 + 1]);

                f32::total_cmp(&c1_score, &c2_score)
            })
        {
            let second_chunk = &chunks[index + 1];

            let highest_similarity = self.score_merge(first_chunk, second_chunk);
            if highest_similarity < self.target_score {
                break;
            }

            // Merge the two chunks
            let range = first_chunk.range.start..second_chunk.range.end;
            let sentences = first_chunk.sentences + second_chunk.sentences;

            let new_text = text[range.clone()].trim();
            let embedding = embedder.embed(new_text).await?;

            // Calculate the distance to the next chunk
            let distance_to_next = chunks
                .get(index + 2)
                .map(|chunk_after_merge| embedding.cosine_similarity(&chunk_after_merge.embedding));

            // Recalculate the distance to the previous chunk
            if let Some(prev_chunk) = index.checked_sub(1).and_then(|index| chunks.get_mut(index)) {
                let distance_to_prev = prev_chunk.embedding.cosine_similarity(&embedding);
                prev_chunk.distance_to_next = Some(distance_to_prev);
            }

            let new_chunk = SemanticChunk {
                range,
                sentences,
                embedding,
                distance_to_next,
            };

            // Remove the last chunk
            chunks.remove(index + 1);

            // Add our merged chunk to the chunks
            chunks[index] = new_chunk;
        }

        let mut final_chunks = Vec::new();
        for chunk in chunks {
            let SemanticChunk {
                range, embedding, ..
            } = chunk;
            final_chunks.push(Chunk {
                byte_range: range,
                embeddings: vec![embedding],
            });
        }

        Ok(final_chunks)
    }
}
