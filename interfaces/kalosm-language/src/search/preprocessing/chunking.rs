use kalosm_language_model::Embedder;

use super::{ChunkStrategy, Chunker};
use crate::{prelude::Document, search::Chunk};

impl Chunker for ChunkStrategy {
    type Error<E: Send + Sync + 'static> = E;

    async fn chunk<E: Embedder + Send>(
        &self,
        document: &Document,
        embedder: &E,
    ) -> Result<Vec<Chunk>, E::Error> {
        let mut chunks = Vec::new();
        let body = document.body();
        let mut documents = Vec::new();
        let chunk_ranges = self.chunk_str(body);
        for byte_range in &chunk_ranges {
            documents.push(document.body()[byte_range.clone()].to_string());
        }
        let embeddings = embedder.embed_vec(documents).await?;
        for (byte_range, embedding) in chunk_ranges.into_iter().zip(embeddings) {
            chunks.push(Chunk {
                byte_range,
                embeddings: vec![embedding],
            });
        }
        Ok(chunks)
    }

    async fn chunk_batch<'a, I, E: Embedder + Send>(
        &self,
        documents: I,
        embedder: &E,
    ) -> Result<Vec<Vec<Chunk>>, E::Error>
    where
        I: IntoIterator<Item = &'a Document> + Send,
        I::IntoIter: Send,
    {
        let mut chunks = Vec::new();
        let mut chunk_strings = Vec::new();
        for document in documents {
            let body = document.body();
            let chunk = self.chunk_str(body);
            for byte_range in &chunk {
                chunk_strings.push(body[byte_range.clone()].to_string());
            }
            chunks.push(chunk);
        }

        let mut embeddings = embedder.embed_vec(chunk_strings).await?;
        let mut embeddings = embeddings.drain(..);
        let mut embedded_chunks = Vec::new();

        for chunk in chunks {
            let mut document_chunks = Vec::new();
            for byte_range in chunk {
                let embedding = embeddings.next().unwrap();
                document_chunks.push(Chunk {
                    byte_range,
                    embeddings: vec![embedding],
                });
            }
            embedded_chunks.push(document_chunks);
        }

        Ok(embedded_chunks)
    }
}
