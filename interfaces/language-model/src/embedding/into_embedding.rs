use futures_util::Future;
use kalosm_model_types::WasmNotSend;

use crate::{Embedder, EmbedderExt, Embedding};

/// Convert a type into an embedding with an embedding model.
pub trait IntoEmbedding {
    /// Convert the type into an embedding with the given embedding model.
    fn into_embedding<E: Embedder>(
        self,
        embedder: &E,
    ) -> impl Future<Output = Result<Embedding, E::Error>> + WasmNotSend;

    /// Convert the type into a query embedding with the given embedding model.
    fn into_query_embedding<E: Embedder>(
        self,
        embedder: &E,
    ) -> impl Future<Output = Result<Embedding, E::Error>> + WasmNotSend;
}

/// Convert any type that implements [`ToString`] into an embedding with an embedding model.
impl<S: ToString + Send> IntoEmbedding for S {
    async fn into_embedding<E: Embedder>(self, embedder: &E) -> Result<Embedding, E::Error> {
        embedder.embed(self).await
    }

    async fn into_query_embedding<E: Embedder>(self, embedder: &E) -> Result<Embedding, E::Error> {
        embedder.embed_query(self).await
    }
}

/// Convert an embedding of the same vector space into an embedding with an embedding model.
impl IntoEmbedding for Embedding {
    async fn into_embedding<E: Embedder>(self, _: &E) -> Result<Embedding, E::Error> {
        Ok(self)
    }

    async fn into_query_embedding<E: Embedder>(self, _: &E) -> Result<Embedding, E::Error> {
        Ok(self)
    }
}
