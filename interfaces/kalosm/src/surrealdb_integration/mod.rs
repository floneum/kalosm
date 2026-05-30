use crate::language::*;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::future::{Future, IntoFuture};
use std::ops::Range;
use std::pin::Pin;
use surrealdb::{Connection, RecordId, RecordIdKey, Surreal};

#[cfg(feature = "language")]
pub(crate) mod document_table;

#[cfg(feature = "language")]
pub(crate) mod hybrid_search;

/// An error that can occur when adding or searching for an embedding to the embedding indexed table.
#[derive(Debug, thiserror::Error)]
pub enum EmbeddedIndexedTableError {
    /// An error from the arroy crate.
    #[error("Arroy error: {0}")]
    Arroy(#[from] arroy::Error),
    /// An error from the SurrealDB crate.
    #[error("SurrealDB error: {0}")]
    Surreal(#[from] Box<surrealdb::Error>),
    /// An error from querying a record that does not exist.
    #[error("Record not found")]
    RecordNotFound,
    /// An error from querying an embedding id that does not exist.
    #[error("Embedding {0:?} not found")]
    EmbeddingNotFound(EmbeddingId),
}

impl From<heed::Error> for EmbeddedIndexedTableError {
    fn from(value: heed::Error) -> Self {
        Self::Arroy(value.into())
    }
}

impl From<VectorDbError> for EmbeddedIndexedTableError {
    fn from(value: VectorDbError) -> Self {
        match value {
            VectorDbError::Arroy(err) => Self::Arroy(err),
            VectorDbError::EmbeddingNotFound(id) => Self::EmbeddingNotFound(id),
        }
    }
}

impl From<surrealdb::Error> for EmbeddedIndexedTableError {
    fn from(value: surrealdb::Error) -> Self {
        Self::Surreal(Box::new(value))
    }
}

/// A link between a document and an embedding.
///
/// This type is stored in the [`EmbeddingIndexedTable::table_links`] table.
#[derive(Serialize, Deserialize)]
pub struct DocumentLink {
    document_id: RecordIdKey,
    byte_range: std::ops::Range<usize>,
    /// Optional materialized chunk text (only present when hybrid search is enabled)
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    chunk_text: Option<String>,
}

/// An object with associated embedding ids.
///
/// This type is stored in the [`EmbeddingIndexedTable::table`] table.
#[derive(Serialize, Deserialize)]
pub struct ObjectWithEmbeddingIds<T> {
    object: T,
    chunks: Vec<(Range<usize>, Vec<EmbeddingId>)>,
}

/// A table in a surreal database with a primary key tied to an embedding in a vector database.
pub struct EmbeddingIndexedTable<C: Connection, R> {
    table: String,
    db: Surreal<C>,
    vector_db: VectorDB,
    phantom: std::marker::PhantomData<R>,
}

impl<C: Connection, R> EmbeddingIndexedTable<C, R> {
    /// Get the name of the table.
    pub fn table(&self) -> &str {
        &self.table
    }

    /// Get the name of the table that links embedding ids to byte ranges in documents.
    pub fn table_links(&self) -> String {
        format!("{}-links", &self.table)
    }

    /// Get the raw vector database.
    pub fn vector_db(&self) -> &VectorDB {
        &self.vector_db
    }

    /// Get the raw surreal database.
    pub fn db(&self) -> &Surreal<C> {
        &self.db
    }

    /// Delete the table from the database and clear the vector database. Returns the contents of the table.
    pub async fn delete_table(self) -> Result<Vec<(R, Vec<Chunk>)>, EmbeddedIndexedTableError>
    where
        R: DeserializeOwned,
    {
        let _: Vec<DocumentLink> = self.db.delete(self.table_links()).await?;
        let embeddings: Vec<ObjectWithEmbeddingIds<R>> = self.db.delete(&self.table).await?;

        let mut documents = Vec::with_capacity(embeddings.len());

        for embedding in embeddings {
            let mut chunks = Vec::with_capacity(embedding.chunks.len());
            for (byte_range, embedding_ids) in embedding.chunks {
                let mut embeddings = Vec::with_capacity(embedding_ids.len());
                for embedding_id in embedding_ids {
                    let embedding = self.vector_db.get_embedding(embedding_id)?;
                    embeddings.push(embedding);
                }
                chunks.push(Chunk {
                    byte_range,
                    embeddings,
                });
            }
            documents.push((embedding.object, chunks));
        }
        self.vector_db.clear().await?;

        Ok(documents)
    }

    /// Insert a new record into the table with the given embedding.
    pub async fn insert(
        &self,
        chunks: impl IntoIterator<Item = Chunk>,
        value: R,
    ) -> Result<RecordIdKey, EmbeddedIndexedTableError>
    where
        R: Serialize + DeserializeOwned + 'static,
    {
        let id_uuid = surrealdb::sql::Uuid::new_v7().0;
        let id = RecordIdKey::from(id_uuid);

        let mut embedding_ids = Vec::new();
        let thing = RecordId::from_table_key(self.table.clone(), id.clone());

        for chunk in chunks {
            let chunk_embedding_ids = self.vector_db.add_embeddings(chunk.embeddings)?;
            for embedding_id in &chunk_embedding_ids {
                let byte_range = chunk.byte_range.clone();

                let link = RecordId::from_table_key(self.table_links(), embedding_id.0 as i64);
                self.db
                    .create::<Option<DocumentLink>>(link)
                    .content(DocumentLink {
                        document_id: id.clone(),
                        byte_range,
                        chunk_text: None,
                    })
                    .await?;
            }
            embedding_ids.push((chunk.byte_range.clone(), chunk_embedding_ids));
        }

        self.db
            .create::<Option<ObjectWithEmbeddingIds<R>>>(thing)
            .content(ObjectWithEmbeddingIds {
                object: value,
                chunks: embedding_ids,
            })
            .await?;

        Ok(id)
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
        let thing = RecordId::from_table_key(self.table.clone(), id);
        let old = self
            .db
            .update::<Option<ObjectWithEmbeddingIds<R>>>(thing)
            .merge(value)
            .await?;

        Ok(old.map(|v| v.object))
    }

    /// Select a record from the table with the given embedding id.
    pub async fn select(&self, id: impl Into<RecordIdKey>) -> Result<R, EmbeddedIndexedTableError>
    where
        R: DeserializeOwned,
    {
        let thing = RecordId::from_table_key(self.table.clone(), id);
        let record = self
            .db
            .select::<Option<ObjectWithEmbeddingIds<R>>>(thing)
            .await?;
        match record {
            Some(record) => Ok(record.object),
            None => Err(EmbeddedIndexedTableError::RecordNotFound),
        }
    }

    /// Delete a record from the table with the given embedding id.
    pub async fn delete(
        &self,
        id: impl Into<RecordIdKey>,
    ) -> Result<Option<R>, EmbeddedIndexedTableError>
    where
        R: Serialize + DeserializeOwned,
    {
        // First delete the record from the main table
        let thing = RecordId::from_table_key(self.table.clone(), id);
        let old = self
            .db
            .delete::<Option<ObjectWithEmbeddingIds<R>>>(thing)
            .await?;

        if let Some(old) = old {
            let ObjectWithEmbeddingIds {
                object,
                chunks: embedding_ids,
            } = old;
            // Then delete the links from the links table
            for id in embedding_ids
                .iter()
                .flat_map(|(_, ids)| ids.iter())
                .copied()
            {
                let link = RecordId::from_table_key(self.table_links(), id.0 as i64);
                self.db.delete::<Option<DocumentLink>>(link).await?;
                // Then delete the embedding from the vector db
                self.vector_db.remove_embedding(id)?;
            }

            Ok(Some(object))
        } else {
            Ok(None)
        }
    }

    /// Select all records from the table.
    pub async fn select_all(&self) -> Result<Vec<R>, EmbeddedIndexedTableError>
    where
        R: DeserializeOwned,
    {
        let records = self
            .db
            .select::<Vec<ObjectWithEmbeddingIds<R>>>(self.table.clone())
            .await?;
        Ok(records.into_iter().map(|v| v.object).collect())
    }

    /// Search for records that are close to the given embedding.
    pub fn search<'a>(
        &'a self,
        embedding: &'a Embedding,
    ) -> EmbeddingIndexedTableSearchBuilder<'a, C, R> {
        EmbeddingIndexedTableSearchBuilder {
            table: self,
            embedding,
            results: None,
            filter: None,
            phantom: std::marker::PhantomData,
        }
    }

    /// Insert with materialized chunk text (for hybrid search only)
    /// Only accessible within the crate, specifically for hybrid_search module
    pub(crate) async fn insert_with_chunk_text(
        &self,
        chunks: impl IntoIterator<Item = Chunk>,
        value: R,
        extract_text: impl Fn(&R, &Range<usize>) -> String,
    ) -> Result<RecordIdKey, EmbeddedIndexedTableError>
    where
        R: Serialize + DeserializeOwned + 'static,
    {
        let id_uuid = surrealdb::sql::Uuid::new_v7().0;
        let id = RecordIdKey::from(id_uuid);

        let mut embedding_ids = Vec::new();
        let thing = RecordId::from_table_key(self.table.clone(), id.clone());

        for chunk in chunks {
            let chunk_embedding_ids = self.vector_db.add_embeddings(chunk.embeddings)?;
            for embedding_id in &chunk_embedding_ids {
                let byte_range = chunk.byte_range.clone();
                let chunk_text = extract_text(&value, &byte_range);

                let link = RecordId::from_table_key(self.table_links(), embedding_id.0 as i64);

                // Create link WITH chunk_text
                self.db
                    .create::<Option<DocumentLink>>(link)
                    .content(DocumentLink {
                        document_id: id.clone(),
                        byte_range,
                        chunk_text: Some(chunk_text),
                    })
                    .await?;
            }
            embedding_ids.push((chunk.byte_range.clone(), chunk_embedding_ids));
        }

        self.db
            .create::<Option<ObjectWithEmbeddingIds<R>>>(thing)
            .content(ObjectWithEmbeddingIds {
                object: value,
                chunks: embedding_ids,
            })
            .await?;

        Ok(id)
    }
}

/// A trait for anything that can be used to filter the results of an embedded table search.
pub trait IntoEmbeddingIndexedTableSearchFilter<C: Connection, R, Marker> {
    /// Convert the filter into a set of candidates.
    fn into_embedding_indexed_table_search_filter(
        self,
        db: &EmbeddingIndexedTable<C, R>,
    ) -> impl std::future::Future<Output = Result<Candidates, EmbeddedIndexedTableError>> + Send;
}

impl<C: Connection, R: Send + Sync> IntoEmbeddingIndexedTableSearchFilter<C, R, ()> for Candidates {
    async fn into_embedding_indexed_table_search_filter(
        self,
        _: &EmbeddingIndexedTable<C, R>,
    ) -> Result<Candidates, EmbeddedIndexedTableError> {
        Ok(self)
    }
}

/// A marker type that allows kalosm to specialize the [`IntoEmbeddingIndexedTableSearchFilter`] trait for iterators.
pub struct IteratorMarker;

impl<C: Connection, R: DeserializeOwned + Send + Sync, I>
    IntoEmbeddingIndexedTableSearchFilter<C, R, IteratorMarker> for I
where
    I: IntoIterator<Item = RecordIdKey>,
    I::IntoIter: Send + Sync + 'static,
{
    fn into_embedding_indexed_table_search_filter(
        self,
        table: &EmbeddingIndexedTable<C, R>,
    ) -> impl Future<Output = Result<Candidates, EmbeddedIndexedTableError>> + Send {
        let ids = self.into_iter();
        async move {
            let mut candidates = Candidates::new();
            for id in ids {
                let thing = RecordId::from_table_key(table.table.clone(), id);
                let item: Option<ObjectWithEmbeddingIds<R>> = table.db.select(thing).await?;
                if let Some(item) = item {
                    for (_, embeddings) in item.chunks.iter() {
                        for embedding_id in embeddings.iter() {
                            candidates.insert(embedding_id.0);
                        }
                    }
                }
            }
            Ok(candidates)
        }
    }
}

/// A builder for searching for embeddings in a vector database.
pub struct EmbeddingIndexedTableSearchBuilder<'a, C: Connection, R, F = Candidates, M = ()> {
    table: &'a EmbeddingIndexedTable<C, R>,
    embedding: &'a Embedding,
    results: Option<usize>,
    filter: Option<F>,
    phantom: std::marker::PhantomData<M>,
}

impl<C: Connection, R: DeserializeOwned, F: IntoEmbeddingIndexedTableSearchFilter<C, R, M>, M>
    EmbeddingIndexedTableSearchBuilder<'_, C, R, F, M>
{
    /// Set the number of results to return. Defaults to 10.
    pub fn with_results(mut self, results: usize) -> Self {
        self.results = Some(results);
        self
    }

    /// Run the search and return the results.
    pub async fn run(
        self,
    ) -> Result<Vec<EmbeddingIndexedTableSearchResult<R>>, EmbeddedIndexedTableError> {
        let mut query = self.table.vector_db.search(self.embedding);
        if let Some(filter) = self.filter {
            query = query.with_filter(
                filter
                    .into_embedding_indexed_table_search_filter(self.table)
                    .await?,
            );
        }
        if let Some(results) = self.results {
            query = query.with_results(results);
        }
        let ids = query.run()?;
        let mut records = Vec::new();
        for id in ids {
            let main_table_id = self
                .table
                .db
                .select::<Option<DocumentLink>>(RecordId::from_table_key(
                    self.table.table_links(),
                    id.value.0 as i64,
                ))
                .await?
                .ok_or(EmbeddedIndexedTableError::RecordNotFound)?;
            let record = self.table.select(main_table_id.document_id.clone()).await?;
            records.push(EmbeddingIndexedTableSearchResult {
                distance: id.distance,
                id: id.value,
                record_id: main_table_id.document_id,
                byte_range: main_table_id.byte_range,
                record,
            });
        }
        Ok(records)
    }
}

impl<
        'a,
        C: Connection + 'a,
        R: DeserializeOwned + Send + Sync + 'a,
        F: IntoEmbeddingIndexedTableSearchFilter<C, R, M> + Send + 'a,
        M: Send + 'a,
    > IntoFuture for EmbeddingIndexedTableSearchBuilder<'a, C, R, F, M>
{
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output> + Send + 'a>>;
    type Output = Result<Vec<EmbeddingIndexedTableSearchResult<R>>, EmbeddedIndexedTableError>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.run())
    }
}

impl<'a, C: Connection, R: DeserializeOwned> EmbeddingIndexedTableSearchBuilder<'a, C, R> {
    /// Set a filter to apply to the results. Only vectors that pass the filter will be returned.
    pub fn with_filter<Marker, F>(
        self,
        filter: F,
    ) -> EmbeddingIndexedTableSearchBuilder<'a, C, R, F, Marker>
    where
        F: IntoEmbeddingIndexedTableSearchFilter<C, R, Marker>,
    {
        EmbeddingIndexedTableSearchBuilder {
            table: self.table,
            embedding: self.embedding,
            results: self.results,
            filter: Some(filter),
            phantom: std::marker::PhantomData,
        }
    }
}

/// The result of a search in an embedding indexed table.
#[derive(Debug, Clone)]
pub struct EmbeddingIndexedTableSearchResult<R> {
    /// The distance from the searched point.
    pub distance: f32,
    /// The embedding id of the record.
    pub id: EmbeddingId,
    /// The record id.
    pub record_id: RecordIdKey,
    /// The byte range of the record.
    pub byte_range: std::ops::Range<usize>,
    /// The record.
    pub record: R,
}

impl<R> EmbeddingIndexedTableSearchResult<R>
where
    R: DeserializeOwned,
{
    /// Get the text of the search result.
    pub fn text(&self) -> String
    where
        R: AsRef<Document>,
    {
        self.record.as_ref().body()[self.byte_range.clone()].to_string()
    }
}

/// A builder for creating a new document table.
pub struct EmbeddingIndexedTableBuilder<C: Connection> {
    table: String,
    db: Surreal<C>,
    location: Option<std::path::PathBuf>,
}

impl<C: Connection> EmbeddingIndexedTableBuilder<C> {
    /// Create a new document table builder.
    pub fn new(table: &str, db: Surreal<C>) -> Self {
        Self {
            table: table.to_string(),
            db,
            location: None,
        }
    }

    /// Set the location of the vector database.
    pub fn at(mut self, location: impl AsRef<std::path::Path>) -> Self {
        self.location = Some(location.as_ref().to_path_buf());
        self
    }

    /// Build the document table.
    pub fn build<R: Serialize + DeserializeOwned>(
        self,
    ) -> Result<EmbeddingIndexedTable<C, R>, EmbeddedIndexedTableError> {
        let vector_db = if let Some(location) = self.location {
            VectorDB::new_at(location)?
        } else {
            VectorDB::new()?
        };
        Ok(EmbeddingIndexedTable {
            table: self.table.to_string(),
            db: self.db,
            vector_db,
            phantom: std::marker::PhantomData,
        })
    }
}

/// An extension trait for the surreal database to interact with vector indexed tables.
pub trait VectorDbSurrealExt<C: Connection> {
    /// Create a new vector indexed table builder.
    fn vector_indexed_table_builder(&self, table: &str) -> EmbeddingIndexedTableBuilder<C>;
}

impl<C: Connection> VectorDbSurrealExt<C> for Surreal<C> {
    fn vector_indexed_table_builder(&self, table: &str) -> EmbeddingIndexedTableBuilder<C> {
        EmbeddingIndexedTableBuilder::new(table, self.clone())
    }
}
