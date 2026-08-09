use crate::prelude::{Chunk, Chunker, Document, Embedder};
use std::future::Future;

use super::{DefaultSentenceChunker, SentenceChunker};

impl Chunker for DefaultSentenceChunker {
    type Error<E: Send + Sync + 'static> = E;

    fn chunk<E: Embedder + Send>(
        &self,
        document: &Document,
        embedder: &E,
    ) -> impl Future<Output = Result<Vec<Chunk>, Self::Error<E::Error>>> + Send {
        let default = SentenceChunker::default();
        let mut initial_chunks = Vec::new();
        let body = document.body();
        let ranges = default.split_sentences(document.body());
        for chunk in &ranges {
            initial_chunks.push(body[chunk.clone()].to_string());
        }

        embed_chunk(embedder, initial_chunks, ranges)
    }
}

impl Chunker for SentenceChunker {
    type Error<E: Send + Sync + 'static> = E;

    fn chunk<E: Embedder + Send>(
        &self,
        document: &Document,
        embedder: &E,
    ) -> impl Future<Output = Result<Vec<Chunk>, Self::Error<E::Error>>> + Send {
        let mut initial_chunks = Vec::new();
        let body = document.body();
        let ranges = self.split_sentences(document.body());
        for chunk in &ranges {
            initial_chunks.push(body[chunk.clone()].to_string());
        }

        embed_chunk(embedder, initial_chunks, ranges)
    }
}

async fn embed_chunk<E: Embedder + Send>(
    embedder: &E,
    initial_chunks: Vec<String>,
    ranges: Vec<std::ops::Range<usize>>,
) -> Result<Vec<Chunk>, E::Error> {
    let embeddings = embedder.embed_vec(initial_chunks).await?;

    let mut chunks = Vec::new();
    for (embedding, chunk) in embeddings.into_iter().zip(ranges) {
        let chunk = Chunk {
            byte_range: chunk,
            embeddings: vec![embedding],
        };
        chunks.push(chunk);
    }

    Ok(chunks)
}
