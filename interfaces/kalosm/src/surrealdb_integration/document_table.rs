use std::any::Any;
use std::any::TypeId;
use std::future::Future;
use std::future::IntoFuture;
use std::pin::Pin;

use super::EmbeddedIndexedTableError;

use super::IntoEmbeddingIndexedTableSearchFilter;
use super::{EmbeddingIndexedTable, EmbeddingIndexedTableSearchResult};
use crate::language::*;
use rbert::BertLoadingError;
use serde::de::DeserializeOwned;
use serde::Serialize;
use surrealdb::Connection;
use surrealdb::RecordIdKey;
use surrealdb::Surreal;

/// An error that can occur when adding items to a [`DocumentTable`].
#[derive(Debug, thiserror::Error)]
pub enum DocumentTableModifyError<E> {
    /// An error occurred while embedding the item to add.
    #[error("Failed to embed item: {0}")]
    EmbedItem(E),
    /// An error occurred in the database while adding the item.
    #[error("Failed to add item: {0}")]
    AddItem(#[from] EmbeddedIndexedTableError),
}

/// A table in a surreal database that is indexed by embeddings from a vector database.
///
/// # Example
/// ```rust, no_run
/// use kalosm::language::*;
/// use surrealdb::{engine::local::SurrealKv, Surreal};
///
/// #[tokio::main]
/// async fn main() {
///     let exists = std::path::Path::new("./db").exists();
///
///     // Create or open a database
///     let db = Surreal::new::<SurrealKv>("./db/temp.db").await.unwrap();
///
///     // Select a specific namespace / database
///     db.use_ns("rag").use_db("rag").await.unwrap();
///
///     // Create a chunker splits the document into chunks to be embedded
///     let chunker = SemanticChunker::new();
///
///     // Create a table in the surreal database to store the embeddings
///     let document_table = db
///         .document_table_builder("documents")
///         .with_chunker(chunker)
///         .at("./db/embeddings.db")
///         .build::<Document>()
///         .await
///         .unwrap();
///
///     // If the database is new, add documents to it
///     if !exists {
///         std::fs::create_dir_all("documents").unwrap();
///         let context = [
///             "https://floneum.com/kalosm/docs",
///             "https://floneum.com/kalosm/docs/guides/retrieval_augmented_generation",
///         ]
///         .iter()
///         .map(|url| Url::parse(url).unwrap());
///
///         document_table.add_context(context).await.unwrap();
///     }
///
///     // Search for data from the database
///     let user_question = prompt_input("Query: ").unwrap();
///
///     let nearest_5 = document_table
///         .search(&user_question)
///         .with_results(5)
///         .await
///         .unwrap();
///
///     println!("{:?}", nearest_5);
/// }
/// ```
pub struct DocumentTable<
    C: Connection,
    R = Document,
    M: Embedder = Bert,
    K: Chunker = SemanticChunker,
> {
    embedding_model: M,
    chunker: K,
    table: EmbeddingIndexedTable<C, R>,
}

impl<C: Connection, R, M: Embedder, K: Chunker> DocumentTable<C, R, M, K> {
    /// Create a new document table.
    pub fn new(embedding_model: M, table: EmbeddingIndexedTable<C, R>, chunker: K) -> Self {
        Self {
            embedding_model,
            table,
            chunker,
        }
    }

    /// Get the raw table.
    pub fn table(&self) -> &EmbeddingIndexedTable<C, R> {
        &self.table
    }

    /// Get the raw embedding model.
    pub fn embedding_model(&self) -> &M {
        &self.embedding_model
    }

    /// Get the raw chunker.
    pub fn chunker(&self) -> &K {
        &self.chunker
    }

    /// Delete the table from the database and clear the vector database. Returns the contents of the table.
    pub async fn delete_table(self) -> Result<Vec<(R, Vec<Chunk>)>, EmbeddedIndexedTableError>
    where
        R: DeserializeOwned,
    {
        self.table.delete_table().await
    }

    /// Insert a new record into the table with pre-computed chunks.
    pub async fn insert_with_chunks(
        &self,
        value: R,
        chunks: impl IntoIterator<Item = Chunk>,
    ) -> Result<RecordIdKey, EmbeddedIndexedTableError>
    where
        R: Serialize + DeserializeOwned + 'static,
    {
        self.table.insert(chunks, value).await
    }

    /// Insert a new record into the table and return the id of the record.
    pub async fn insert(
        &self,
        value: R,
    ) -> Result<RecordIdKey, DocumentTableModifyError<K::Error<M::Error>>>
    where
        R: AsRef<Document> + Serialize + DeserializeOwned + 'static,
    {
        let chunks = self
            .chunker
            .chunk(value.as_ref(), &self.embedding_model)
            .await
            .map_err(DocumentTableModifyError::EmbedItem)?;
        Ok(self.insert_with_chunks(value, chunks).await?)
    }

    /// Extend the table with a iterator of new records.
    pub async fn extend<T: IntoIterator<Item = R> + Send>(
        &self,
        iter: T,
    ) -> Result<Vec<RecordIdKey>, DocumentTableModifyError<K::Error<M::Error>>>
    where
        R: AsRef<Document> + Serialize + DeserializeOwned + 'static,
        K: Sync,
    {
        let entries = iter.into_iter().collect::<Vec<_>>();
        let documents = entries.iter().map(|v| v.as_ref()).collect::<Vec<_>>();
        let embeddings = self
            .chunker
            .chunk_batch(documents, &self.embedding_model)
            .await
            .map_err(DocumentTableModifyError::EmbedItem)?;
        let mut ids = Vec::new();
        for (value, embeddings) in entries.into_iter().zip(embeddings) {
            let id = self.table.insert(embeddings, value).await?;
            ids.push(id);
        }
        Ok(ids)
    }

    /// Update a record in the table with the given embedding id.
    pub async fn update(
        &self,
        id: impl Into<RecordIdKey>,
        value: R,
    ) -> Result<Option<R>, EmbeddedIndexedTableError>
    where
        R: Serialize + DeserializeOwned + 'static,
    {
        self.table.update(id, value).await
    }

    /// Select a record from the table with the given embedding id.
    pub async fn select(&self, id: impl Into<RecordIdKey>) -> Result<R, EmbeddedIndexedTableError>
    where
        R: Serialize + DeserializeOwned + 'static,
    {
        self.table.select(id).await
    }

    /// Delete a record from the table with the given embedding id.
    pub async fn delete(
        &self,
        id: impl Into<RecordIdKey>,
    ) -> Result<Option<R>, EmbeddedIndexedTableError>
    where
        R: Serialize + DeserializeOwned + 'static,
    {
        self.table.delete(id).await
    }

    /// Select all records from the table.
    pub async fn select_all(&self) -> Result<Vec<R>, EmbeddedIndexedTableError>
    where
        R: Serialize + DeserializeOwned + 'static,
    {
        self.table.select_all().await
    }

    /// Select the top k records nearest records to the given item.
    ///
    /// NOTE: If your embedding model has a different query embedding and you pass in a raw embedding, that embedding will perform best if it was created with [`EmbedderExt::embed_query`].
    pub fn search<E>(&self, embedding: E) -> DocumentTableSearchBuilder<'_, C, R, M, K, E>
    where
        E: IntoEmbedding,
        R: DeserializeOwned,
    {
        DocumentTableSearchBuilder {
            table: self,
            embedding,
            results: None,
            filter: None,
            phantom: std::marker::PhantomData,
        }
    }
}

/// An error that can occur while adding context to a [`DocumentTable`].
#[derive(Debug, thiserror::Error)]
pub enum DocumentTableAddContextError<D, M> {
    /// An error occurred while converting the item to a document.
    #[error("Failed to convert item to document: {0}")]
    ConvertItem(D),
    /// An error occurred while modifying the table.
    #[error("Failed to modify table: {0}")]
    ModifyTable(DocumentTableModifyError<M>),
}

impl<C: Connection, R, M: Embedder, K: Chunker> DocumentTable<C, R, M, K> {
    /// Extend the table from [`IntoDocuments`]
    pub async fn add_context<D: IntoDocuments>(
        &self,
        context: D,
    ) -> Result<Vec<RecordIdKey>, DocumentTableAddContextError<D::Error, K::Error<M::Error>>>
    where
        R: From<Document> + AsRef<Document> + Serialize + DeserializeOwned + 'static,
        K: Sync,
    {
        let documents = context
            .into_documents()
            .await
            .map_err(DocumentTableAddContextError::ConvertItem)?;
        let iter = documents.into_iter().map(|v| v.into());
        self.extend(iter)
            .await
            .map_err(DocumentTableAddContextError::ModifyTable)
    }
}

/// A builder for searching for embeddings in a vector database.
pub struct DocumentTableSearchBuilder<
    'a,
    Conn: Connection,
    Doc = Document,
    Model: Embedder = Bert,
    Chkr: Chunker = SemanticChunker,
    E = Embedding,
    F = Candidates,
    M = (),
> {
    table: &'a DocumentTable<Conn, Doc, Model, Chkr>,
    embedding: E,
    results: Option<usize>,
    filter: Option<F>,
    phantom: std::marker::PhantomData<M>,
}

/// An error that can occur while searching a [`DocumentTable`].
#[derive(Debug, thiserror::Error)]
pub enum DocumentTableSearchError<E> {
    /// An error occurred while embedding the search query.
    #[error("Failed to embed search query: {0}")]
    EmbedQuery(E),
    /// An error occurred while running the search on the underlying table.
    #[error("Failed to run search on table: {0}")]
    SearchTable(#[from] EmbeddedIndexedTableError),
}

impl<
        Conn: Connection,
        Doc: DeserializeOwned + Send + Sync,
        Model: Embedder,
        E: IntoEmbedding,
        F: IntoEmbeddingIndexedTableSearchFilter<Conn, Doc, M>,
        Chkr: Chunker,
        M,
    > DocumentTableSearchBuilder<'_, Conn, Doc, Model, Chkr, E, F, M>
{
    /// Set the number of results to return. Defaults to 10.
    pub fn with_results(mut self, results: usize) -> Self {
        self.results = Some(results);
        self
    }

    /// Run the search and return the results.
    pub async fn run(
        self,
    ) -> Result<Vec<EmbeddingIndexedTableSearchResult<Doc>>, DocumentTableSearchError<Model::Error>>
    {
        let embedding = self
            .embedding
            .into_embedding(&self.table.embedding_model)
            .await
            .map_err(DocumentTableSearchError::EmbedQuery)?;
        let mut query = self.table.table.search(&embedding);
        if let Some(results) = self.results {
            query = query.with_results(results);
        }
        if let Some(filter) = self.filter {
            let query = query.with_filter(filter);
            Ok(query.run().await?)
        } else {
            Ok(query.run().await?)
        }
    }
}

impl<
        'a,
        Conn: Connection + 'a,
        Doc: DeserializeOwned + Send + Sync + 'a,
        Model: Embedder + 'a,
        E: IntoEmbedding + Send + 'a,
        F: IntoEmbeddingIndexedTableSearchFilter<Conn, Doc, M> + Send + Sync + 'a,
        Chkr: Chunker + Send + Sync + 'a,
        M: Send + 'a,
    > IntoFuture for DocumentTableSearchBuilder<'a, Conn, Doc, Model, Chkr, E, F, M>
{
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output> + Send + 'a>>;
    type Output =
        Result<Vec<EmbeddingIndexedTableSearchResult<Doc>>, DocumentTableSearchError<Model::Error>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.run())
    }
}

impl<
        'a,
        Conn: Connection,
        Doc: DeserializeOwned,
        Model: Embedder,
        E: IntoEmbedding,
        F: IntoEmbeddingIndexedTableSearchFilter<Conn, Doc, M>,
        Chkr: Chunker,
        M,
    > DocumentTableSearchBuilder<'a, Conn, Doc, Model, Chkr, E, F, M>
{
    /// Set a filter to apply to the results. Only vectors that pass the filter will be returned.
    pub fn with_filter<Marker, F2>(
        self,
        filter: F2,
    ) -> DocumentTableSearchBuilder<'a, Conn, Doc, Model, Chkr, E, F2, Marker>
    where
        F2: IntoEmbeddingIndexedTableSearchFilter<Conn, Doc, Marker> + Send + Sync + 'static,
    {
        DocumentTableSearchBuilder {
            table: self.table,
            embedding: self.embedding,
            results: self.results,
            filter: Some(filter),
            phantom: std::marker::PhantomData,
        }
    }
}

/// A builder for creating a new document table.
pub struct DocumentTableBuilder<C: Connection, E = Bert, K: Chunker = SemanticChunker> {
    table: String,
    db: Surreal<C>,
    embedding_model: Option<E>,
    chunker: K,
    location: Option<std::path::PathBuf>,
}

impl<C: Connection> DocumentTableBuilder<C, Bert, ChunkStrategy> {
    /// Create a new document table builder.
    pub(crate) fn new(table: &str, db: Surreal<C>) -> Self {
        Self {
            table: table.to_string(),
            db,
            location: None,
            chunker: ChunkStrategy::Sentence {
                sentence_count: 1,
                overlap: 0,
            },
            embedding_model: None,
        }
    }
}

impl<C: Connection, E, K: Chunker> DocumentTableBuilder<C, E, K> {
    /// Set the location of the vector database.
    pub fn at(mut self, location: impl AsRef<std::path::Path>) -> Self {
        self.location = Some(location.as_ref().to_path_buf());
        self
    }

    /// Set the embedding model for the table.
    pub fn with_embedding_model<E2>(self, embedding_model: E2) -> DocumentTableBuilder<C, E2, K> {
        let Self {
            table,
            db,
            embedding_model: _,
            chunker,
            location,
        } = self;
        DocumentTableBuilder {
            table,
            db,
            embedding_model: Some(embedding_model),
            chunker,
            location,
        }
    }

    /// Set the chunking strategy for the table.
    pub fn with_chunker<K2: Chunker>(self, chunker: K2) -> DocumentTableBuilder<C, E, K2> {
        DocumentTableBuilder {
            chunker,
            table: self.table,
            db: self.db,
            location: self.location,
            embedding_model: self.embedding_model,
        }
    }

    /// Build the document table.
    pub async fn build<R: Serialize + DeserializeOwned>(
        self,
    ) -> Result<DocumentTable<C, R, E, K>, DocumentTableCreationError>
    where
        E: Embedder,
    {
        let vector_db = if let Some(location) = self.location {
            VectorDB::new_at(location)?
        } else {
            VectorDB::new()?
        };
        let table = EmbeddingIndexedTable {
            table: self.table.to_string(),
            db: self.db,
            vector_db,
            phantom: std::marker::PhantomData,
        };
        let embedding_model = match self.embedding_model {
            Some(embedding_model) => embedding_model,
            None => {
                if TypeId::of::<E>() == TypeId::of::<Bert>() {
                    let embedding_model = Bert::new_for_search().await?;
                    *(Box::new(embedding_model) as Box<dyn Any>)
                        .downcast::<E>()
                        .unwrap()
                } else {
                    return Err(DocumentTableCreationError::NoEmbeddingModel);
                }
            }
        };
        Ok(DocumentTable::new(embedding_model, table, self.chunker))
    }
}

/// An error that can occur while creating a [`DocumentTable`].
#[derive(Debug, thiserror::Error)]
pub enum DocumentTableCreationError {
    /// Creating the vector database failed.
    #[error("Failed to create vector database: {0}")]
    VectorDb(#[from] heed::Error),
    /// No embedding model was provided.
    #[error("No embedding model provided")]
    NoEmbeddingModel,
    /// The default embedding model failed to load.
    #[error("Failed to load default embedding model: {0}")]
    DefaultEmbeddingModel(#[from] BertLoadingError),
}

/// An extension trait for the surreal database to interact with document tables.
pub trait DocumentTableSurrealExt<C: Connection> {
    /// Create a new document table builder.
    fn document_table_builder(&self, table: &str) -> DocumentTableBuilder<C, Bert, ChunkStrategy>;
}

impl<C: Connection> DocumentTableSurrealExt<C> for Surreal<C> {
    fn document_table_builder(&self, table: &str) -> DocumentTableBuilder<C, Bert, ChunkStrategy> {
        DocumentTableBuilder::new(table, self.clone())
    }
}
