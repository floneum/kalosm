use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use surrealdb::{Connection, RecordIdKey};

use kalosm_language::prelude::*;

use crate::language::{
    DocumentTable, DocumentTableBuilder, DocumentTableCreationError, DocumentTableSearchBuilder,
    DocumentTableSearchError,
};
use crate::{EmbeddedIndexedTableError, EmbeddingIndexedTable};

/// Errors that can occur during hybrid search operations
#[derive(Debug, thiserror::Error)]
pub enum HybridSearchError {
    /// Failed to set up hybrid search infrastructure
    #[error("Failed to enable hybrid search: {0}")]
    SetupError(String),

    /// Keyword search operation failed
    #[error("Keyword search failed: {0}")]
    KeywordSearchError(String),

    /// Database operation failed
    #[error("Database error: {0}")]
    DatabaseError(#[from] surrealdb::Error),

    /// Document table creation failed
    #[error("Document table creation failed: {0}")]
    TableCreationError(#[from] DocumentTableCreationError),

    /// No results found in search
    #[error("No results found")]
    NoResults,

    /// Semantic search operation failed
    #[error("Semantic search failed: {0}")]
    SemanticSearchError(#[from] EmbeddedIndexedTableError),

    /// Semantic query embedding failed
    #[error("Semantic query failed: {0}")]
    SemanticQueryError(String),

    /// Document table modification failed
    #[error("Document table modification failed: {0}")]
    ModifyError(String),

    /// Context addition failed
    #[error("Context addition failed: {0}")]
    ContextError(String),
}

impl<E> From<DocumentTableSearchError<E>> for HybridSearchError
where
    E: std::fmt::Debug + std::fmt::Display + 'static,
{
    fn from(error: DocumentTableSearchError<E>) -> Self {
        match error {
            DocumentTableSearchError::SearchTable(err) => Self::SemanticSearchError(err),
            DocumentTableSearchError::EmbedQuery(err) => {
                Self::SemanticQueryError(format!("Embedding failed: {}", err))
            }
        }
    }
}

/// Normalize scores to 0-1 range using max normalization
///
/// Returns `None` if all scores are zero or negative, as normalization
/// is not possible in that case.
///
/// # Arguments
/// * `scores` - Slice of scores to normalize
///
/// # Returns
/// * `Some(Vec<f32>)` - Normalized scores in 0-1 range
/// * `None` - If all scores are ≤ 0
pub fn normalize_scores(scores: &[f32]) -> Option<Vec<f32>> {
    if scores.is_empty() {
        return Some(vec![]);
    }

    let max_score = scores.iter().copied().fold(0.0f32, f32::max);

    if max_score <= 0.0 {
        return None;
    }

    Some(scores.iter().map(|&s| s / max_score).collect())
}

/// Calculate weighted combination of semantic and keyword scores
///
/// # Arguments
/// * `semantic_score` - Normalized semantic search score
/// * `keyword_score` - Normalized keyword search score
/// * `semantic_weight` - Weight for semantic score (typically 0.7)
/// * `keyword_weight` - Weight for keyword score (typically 0.3)
///
/// # Returns
/// Combined weighted score
pub fn calculate_weighted_score(
    semantic_score: f32,
    keyword_score: f32,
    semantic_weight: f32,
    keyword_weight: f32,
) -> f32 {
    semantic_score * semantic_weight + keyword_score * keyword_weight
}

/// Calculate Reciprocal Rank Fusion (RRF) score for a given rank
///
/// RRF is a method for combining rankings that is more robust than
/// score-based combination methods.
///
/// # Arguments
/// * `rank` - Zero-indexed rank position (0 = first place)
/// * `k` - RRF constant, typically 60. Lower values give more weight to top results
///
/// # Returns
/// RRF score for this rank
pub fn calculate_rrf_score(rank: usize, k: f32) -> f32 {
    1.0 / (k + rank as f32 + 1.0)
}

/// Extension trait to add `with_hybrid_search()` to DocumentTableBuilder
///
/// This trait enables the builder pattern for creating document tables
/// with hybrid search capabilities.
pub trait DocumentTableBuilderHybridExt<C: Connection, E, K: Chunker>: Sized {
    /// Enable hybrid search capability on this table
    ///
    /// The default field name is "body". Use `.with_search_field()` to change it.
    ///
    /// # Example
    /// ```rust,no_run
    /// use kalosm::language::*;
    /// use surrealdb::{engine::local::Db, Surreal};
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let db: Surreal<Db> = todo!();
    /// // Use default "body" field
    /// let table1 = db
    ///     .document_table_builder("documents")
    ///     .with_hybrid_search()
    ///     .with_chunker(SemanticChunker::new())
    ///     .at("./db/embeddings.db")
    ///     .build::<Document>()
    ///     .await?;
    ///
    /// // Or customize the field
    /// let table2 = db
    ///     .document_table_builder("articles")
    ///     .with_hybrid_search()
    ///     .with_search_field("content")
    ///     .with_chunker(SemanticChunker::new())
    ///     .at("./db/embeddings.db")
    ///     .build::<Document>()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    fn with_hybrid_search(self) -> HybridDocumentTableBuilder<C, E, K>;
}

impl<C: Connection, E, K: Chunker> DocumentTableBuilderHybridExt<C, E, K>
    for DocumentTableBuilder<C, E, K>
{
    fn with_hybrid_search(self) -> HybridDocumentTableBuilder<C, E, K> {
        HybridDocumentTableBuilder::new(self, "body")
    }
}

/// Builder for creating document tables with hybrid search enabled
///
/// This builder is created by calling `.with_hybrid_search()` on a `DocumentTableBuilder`.
/// It will automatically create the necessary full-text search index when `.build()` is called.
///
/// Hybrid search combines semantic (vector) search with keyword (full-text) search
/// to provide more relevant results than either approach alone.
pub struct HybridDocumentTableBuilder<C: Connection, E = Bert, K: Chunker = SemanticChunker> {
    inner: DocumentTableBuilder<C, E, K>,
    field_name: String,
}

impl<C: Connection, E, K: Chunker> HybridDocumentTableBuilder<C, E, K> {
    /// Create a new hybrid document table builder
    ///
    /// # Arguments
    /// * `inner` - The base document table builder
    /// * `field_name` - The field to index for full-text search (commonly "body", "content", "text")
    pub(crate) fn new(inner: DocumentTableBuilder<C, E, K>, field_name: impl Into<String>) -> Self {
        Self {
            inner,
            field_name: field_name.into(),
        }
    }

    /// Change which field to use for full-text search
    ///
    /// Default is "body" (set in `with_hybrid_search()`).
    ///
    /// # Arguments
    /// * `field` - Name of the field to index for keyword search
    pub fn with_search_field(mut self, field: impl Into<String>) -> Self {
        self.field_name = field.into();
        self
    }

    /// Set the location of the vector database
    ///
    /// # Arguments
    /// * `location` - Path to the vector database directory
    pub fn at(mut self, location: impl AsRef<std::path::Path>) -> Self {
        self.inner = self.inner.at(location);
        self
    }

    /// Set the embedding model
    ///
    /// # Arguments
    /// * `model` - The embedding model to use for semantic search
    pub fn with_embedding_model<E2>(self, model: E2) -> HybridDocumentTableBuilder<C, E2, K> {
        HybridDocumentTableBuilder {
            inner: self.inner.with_embedding_model(model),
            field_name: self.field_name,
        }
    }

    /// Set the chunking strategy
    ///
    /// # Arguments
    /// * `chunker` - The chunker to use for splitting documents
    pub fn with_chunker<K2: Chunker>(self, chunker: K2) -> HybridDocumentTableBuilder<C, E, K2> {
        HybridDocumentTableBuilder {
            inner: self.inner.with_chunker(chunker),
            field_name: self.field_name,
        }
    }

    /// Build the document table with hybrid search enabled
    ///
    /// This will:
    /// 1. Create the underlying document table
    /// 2. Define the chunk extraction function
    /// 3. Migrate any existing data to include chunk text
    /// 4. Create a full-text search index on the chunk text
    ///
    /// # Returns
    /// A `HybridDocumentTable` that supports both semantic and keyword search
    ///
    /// # Errors
    /// Returns `HybridSearchError` if table creation or index setup fails
    pub async fn build<R>(self) -> Result<HybridDocumentTable<C, R, E, K>, HybridSearchError>
    where
        E: Embedder,
        R: Serialize + DeserializeOwned + AsRef<Document>,
    {
        let table = self.inner.build().await?;
        let links_table = table.table().table_links();

        // Define the extraction function
        let func_query = format!(
            r#"
                DEFINE FUNCTION fn::extract_chunk($doc_id: record, $start: int, $end: int) {{
                    LET $doc = (SELECT object FROM {} WHERE meta::id(id) = $doc_id)[0];
                    RETURN string::slice($doc.object.{}, $start, $end);
                }};
            "#,
            table.table().table(),
            self.field_name
        );
        table.table().db().query(func_query).await?;

        // Add chunk_text field to existing links by updating them
        let update_links_query = r#"
                UPDATE type::table($table)
                SET chunk_text = fn::extract_chunk(document_id, byte_range.start, byte_range.end)
                WHERE chunk_text IS NONE;
            "#;
        table
            .table()
            .db()
            .query(update_links_query)
            .bind(("table", links_table.clone()))
            .await?;

        // Create the search index on the materialized field
        let index_query = format!(
            r#"
                DEFINE ANALYZER rag_analyzer
                TOKENIZERS blank,punct
                FILTERS lowercase;

                DEFINE INDEX chunk_text_idx ON TABLE `{}`
                FIELDS chunk_text
                SEARCH ANALYZER rag_analyzer BM25;
            "#,
            links_table
        );

        table.table().db().query(index_query).await?;

        Ok(HybridDocumentTable {
            inner: table,
            field_name: self.field_name,
        })
    }
}

/// A document table with hybrid search capabilities enabled
///
/// This type is returned when building a table with `.with_hybrid_search()`.
/// It provides both semantic search (via the inner `DocumentTable`) and
/// hybrid search that combines semantic and keyword matching.
///
/// Hybrid search is particularly effective because:
/// - Semantic search handles conceptual queries and synonyms
/// - Keyword search handles exact terms and specific phrases
/// - The combination provides better results than either alone
pub struct HybridDocumentTable<C: Connection, R, E: Embedder, K: Chunker> {
    inner: DocumentTable<C, R, E, K>,
    field_name: String,
}

impl<C: Connection, R, E: Embedder, K: Chunker> HybridDocumentTable<C, R, E, K> {
    /// Get a reference to the underlying document table
    pub fn inner(&self) -> &DocumentTable<C, R, E, K> {
        &self.inner
    }

    /// Get a mutable reference to the underlying document table
    pub fn inner_mut(&mut self) -> &mut DocumentTable<C, R, E, K> {
        &mut self.inner
    }

    /// Consume this hybrid table and return the underlying document table
    pub fn into_inner(self) -> DocumentTable<C, R, E, K> {
        self.inner
    }

    /// Get the field name used for keyword search
    pub fn field_name(&self) -> &str {
        &self.field_name
    }

    /// Perform hybrid search combining semantic and keyword search
    ///
    /// This is the primary search method for hybrid tables. It combines
    /// vector similarity search with full-text keyword search.
    ///
    /// # Arguments
    /// * `query` - The search query string
    ///
    /// # Returns
    /// A builder that allows configuring search parameters and fusion methods
    pub fn hybrid_search(&self, query: impl Into<String>) -> HybridSearchBuilder<'_, C, R, E, K> {
        HybridSearchBuilder::new(&self.inner, query)
    }

    /// Perform regular semantic search (without keyword matching)
    ///
    /// This bypasses hybrid search and uses only vector similarity.
    /// Use this when you want pure semantic matching without keyword influence.
    ///
    /// # Arguments
    /// * `embedding` - The embedding or text to search for
    pub fn search<Em>(&self, embedding: Em) -> DocumentTableSearchBuilder<'_, C, R, E, K, Em>
    where
        Em: IntoEmbedding,
        R: DeserializeOwned,
    {
        self.inner.search(embedding)
    }

    /// Get a reference to the underlying embedding indexed table
    pub fn table(&self) -> &EmbeddingIndexedTable<C, R> {
        self.inner.table()
    }

    /// Get a reference to the embedding model
    pub fn embedding_model(&self) -> &E {
        self.inner.embedding_model()
    }

    /// Get a reference to the chunker
    pub fn chunker(&self) -> &K {
        self.inner.chunker()
    }

    /// Insert a new record into the table
    ///
    /// This chunks the document, creates embeddings, and inserts it with
    /// materialized chunk text for keyword search.
    ///
    /// # Arguments
    /// * `value` - The document to insert
    ///
    /// # Returns
    /// The record ID of the inserted document
    ///
    /// # Errors
    /// Returns `HybridSearchError` if chunking or insertion fails
    pub async fn insert(&self, value: R) -> Result<RecordIdKey, HybridSearchError>
    where
        R: AsRef<Document> + Serialize + DeserializeOwned + 'static,
    {
        let chunks = self
            .inner
            .chunker()
            .chunk(value.as_ref(), self.inner.embedding_model())
            .await
            .map_err(|_| HybridSearchError::ModifyError("Chunking failed".to_string()))?;

        let record_key = self
            .inner
            .table()
            .insert_with_chunk_text(chunks, value, |v, range| {
                v.as_ref().body()[range.clone()].to_string()
            })
            .await?;

        Ok(record_key)
    }

    /// Rebuild the full-text search index
    ///
    /// Call this after bulk operations or if search results seem incomplete.
    /// This is automatically called by `extend()` and `add_context()`.
    ///
    /// # Errors
    /// Returns `HybridSearchError` if the rebuild operation fails
    pub async fn rebuild_search_index(&self) -> Result<(), HybridSearchError> {
        let links_table = self.table().table_links();
        let rebuild_query = format!(
            "REBUILD INDEX IF EXISTS chunk_text_idx ON TABLE `{}`;",
            links_table
        );

        self.table().db().query(rebuild_query).await?;

        Ok(())
    }

    /// Extend the table with an iterator of new records
    ///
    /// This is more efficient than calling `insert()` multiple times as it
    /// batches the chunking operations and rebuilds the search index once.
    ///
    /// # Arguments
    /// * `iter` - Iterator of documents to insert
    ///
    /// # Returns
    /// Vector of record IDs for the inserted documents
    ///
    /// # Errors
    /// Returns `HybridSearchError` if chunking or insertion fails
    pub async fn extend<T: IntoIterator<Item = R> + Send>(
        &self,
        iter: T,
    ) -> Result<Vec<RecordIdKey>, HybridSearchError>
    where
        R: AsRef<Document> + Serialize + DeserializeOwned + 'static,
        K: Sync,
    {
        let entries = iter.into_iter().collect::<Vec<_>>();
        let documents = entries.iter().map(|v| v.as_ref()).collect::<Vec<_>>();

        let embeddings = self
            .inner
            .chunker()
            .chunk_batch(documents, self.inner.embedding_model())
            .await
            .map_err(|_| HybridSearchError::ModifyError("Batch chunking failed".to_string()))?;

        let mut ids = Vec::new();
        for (value, chunks) in entries.into_iter().zip(embeddings) {
            let id = self
                .inner
                .table()
                .insert_with_chunk_text(chunks, value, |v, range| {
                    v.as_ref().body()[range.clone()].to_string()
                })
                .await?;
            ids.push(id);
        }

        if !ids.is_empty() {
            self.rebuild_search_index().await.ok();
        }

        Ok(ids)
    }

    /// Extend the table with context documents
    ///
    /// This is a convenience method for adding documents from various sources
    /// that implement `IntoDocuments`.
    ///
    /// # Arguments
    /// * `context` - The context source to add (e.g., files, URLs, text)
    ///
    /// # Returns
    /// Vector of record IDs for the inserted documents
    ///
    /// # Errors
    /// Returns `HybridSearchError` if conversion or insertion fails
    pub async fn add_context<D>(&self, context: D) -> Result<Vec<RecordIdKey>, HybridSearchError>
    where
        D: IntoDocuments,
        R: From<Document> + AsRef<Document> + Serialize + DeserializeOwned + 'static,
        K: Sync,
    {
        let documents = context.into_documents().await.map_err(|_| {
            HybridSearchError::ContextError("Context conversion failed".to_string())
        })?;
        let iter = documents.into_iter().map(|v| v.into());
        self.extend(iter).await
    }

    /// Select a record from the table by ID
    ///
    /// # Arguments
    /// * `id` - The record ID to retrieve
    ///
    /// # Returns
    /// The record if found
    ///
    /// # Errors
    /// Returns `HybridSearchError` if the record doesn't exist or can't be retrieved
    pub async fn select(&self, id: impl Into<RecordIdKey>) -> Result<R, HybridSearchError>
    where
        R: Serialize + DeserializeOwned + 'static,
    {
        Ok(self.inner.select(id).await?)
    }

    /// Delete a record from the table
    ///
    /// # Arguments
    /// * `id` - The record ID to delete
    ///
    /// # Returns
    /// The deleted record if it existed, or `None`
    ///
    /// # Errors
    /// Returns `HybridSearchError` if the deletion fails
    pub async fn delete(&self, id: impl Into<RecordIdKey>) -> Result<Option<R>, HybridSearchError>
    where
        R: Serialize + DeserializeOwned + 'static,
    {
        Ok(self.inner.delete(id).await?)
    }

    /// Select all records from the table
    ///
    /// # Returns
    /// Vector of all records in the table
    ///
    /// # Errors
    /// Returns `HybridSearchError` if the query fails
    pub async fn select_all(&self) -> Result<Vec<R>, HybridSearchError>
    where
        R: Serialize + DeserializeOwned + 'static,
    {
        Ok(self.inner.select_all().await?)
    }
}

/// Result from a hybrid search query
///
/// Contains both the document and scoring information from both
/// semantic and keyword search components.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSearchResult<Doc> {
    /// The retrieved document
    pub record: Doc,
    /// The record ID
    pub id: RecordIdKey,
    /// Combined score (fusion of semantic and keyword scores)
    pub score: f32,
    /// Semantic similarity score component
    pub semantic_score: f32,
    /// Keyword matching score component
    pub keyword_score: f32,
}

/// Builder for configuring and executing hybrid search queries
///
/// This builder allows you to configure search parameters like result count,
/// score weights, and fusion method before executing the search.
pub struct HybridSearchBuilder<'a, Conn: Connection, Doc, Model: Embedder, Chkr: Chunker> {
    table: &'a DocumentTable<Conn, Doc, Model, Chkr>,
    user_query: String,
    results: usize,
    semantic_weight: f32,
    keyword_weight: f32,
    rrf_k: f32,
}

/// Internal struct for deserializing keyword search results
#[derive(Deserialize, Debug)]
struct KeywordChunkResult {
    document_id: RecordIdKey,
    keyword_score: f32,
}

impl<'a, C: Connection, R, M: Embedder, K: Chunker> HybridSearchBuilder<'a, C, R, M, K> {
    /// Create a new hybrid search builder
    pub(crate) fn new(table: &'a DocumentTable<C, R, M, K>, query: impl Into<String>) -> Self {
        Self {
            table,
            user_query: query.into(),
            results: 10,
            semantic_weight: 0.7,
            keyword_weight: 0.3,
            rrf_k: 60.0,
        }
    }

    /// Set the number of results to return
    ///
    /// Default: 10
    ///
    /// # Arguments
    /// * `results` - Maximum number of results to return
    pub fn with_results(mut self, results: usize) -> Self {
        self.results = results;
        self
    }

    /// Set the weight of the semantic search
    ///
    /// The keyword weight will be automatically adjusted to `1.0 - weight`
    /// to ensure the weights sum to 1.0.
    ///
    /// Default: 0.7
    ///
    /// # Arguments
    /// * `weight` - Weight for semantic score (0.0 to 1.0)
    ///
    /// # Example
    /// ```rust,ignore
    /// // Setting semantic weight to 0.8 will set keyword weight to 0.2
    /// builder.with_semantic_weight(0.8)
    /// ```
    pub fn with_semantic_weight(mut self, weight: f32) -> Self {
        self.semantic_weight = weight.clamp(0.0, 1.0);
        self.keyword_weight = 1.0 - self.semantic_weight;
        self
    }

    /// Set the weight of the keyword search
    ///
    /// The semantic weight will be automatically adjusted to `1.0 - weight`
    /// to ensure the weights sum to 1.0.
    ///
    /// Default: 0.3
    ///
    /// # Arguments
    /// * `weight` - Weight for keyword score (0.0 to 1.0)
    ///
    /// # Example
    /// ```rust,ignore
    /// // Setting keyword weight to 0.4 will set semantic weight to 0.6
    /// builder.with_keyword_weight(0.4)
    /// ```
    pub fn with_keyword_weight(mut self, weight: f32) -> Self {
        self.keyword_weight = weight.clamp(0.0, 1.0);
        self.semantic_weight = 1.0 - self.keyword_weight;
        self
    }

    /// Set the RRF constant k
    ///
    /// Lower values give more weight to higher-ranked results.
    /// Typical values range from 1 to 100, with 60 being standard.
    ///
    /// Default: 60.0
    ///
    /// # Arguments
    /// * `k` - RRF constant k
    pub fn with_rrf_k(mut self, k: f32) -> Self {
        self.rrf_k = k.clamp(1.0, 100.0);
        self
    }

    /// Execute the hybrid search using weighted score combination
    ///
    /// This method normalizes both semantic and keyword scores to 0-1 range,
    /// then combines them using the configured weights.
    ///
    /// # Returns
    /// Vector of search results sorted by combined score
    ///
    /// # Errors
    /// Returns `HybridSearchError` if the search fails
    pub async fn run_weighted(self) -> Result<Vec<HybridSearchResult<R>>, HybridSearchError>
    where
        R: Serialize + DeserializeOwned + Clone + AsRef<Document> + Send + Sync,
        <M as Embedder>::Error: std::fmt::Debug + std::fmt::Display + 'static,
    {
        let search_limit = self.results * 3;

        let semantic_results = self
            .table
            .search(self.user_query.clone())
            .with_results(search_limit)
            .run()
            .await?;

        let links_table = self.table.table().table_links();
        let keyword_query = r#"
            SELECT
                document_id,
                search::score(1) as keyword_score
            FROM type::table($table)
            WHERE
                chunk_text @1@ $query
            ORDER BY
                keyword_score DESC
            LIMIT $limit;
        "#;

        let keyword_results: Vec<KeywordChunkResult> = self
            .table
            .table()
            .db()
            .query(keyword_query)
            .bind(("query", self.user_query.clone()))
            .bind(("table", links_table))
            .bind(("limit", search_limit))
            .await?
            .take(0)
            .map_err(|e| HybridSearchError::KeywordSearchError(e.to_string()))?;

        let semantic_inverse_scores: Vec<f32> = semantic_results
            .iter()
            .map(|r| {
                if r.distance > 0.0 {
                    1.0 / r.distance
                } else {
                    f32::MAX
                }
            })
            .collect();

        let normalized_semantic_vec = normalize_scores(&semantic_inverse_scores)
            .unwrap_or_else(|| vec![0.0; semantic_inverse_scores.len()]);

        let semantic_map: HashMap<String, (RecordIdKey, R, f32)> = semantic_results
            .into_iter()
            .zip(normalized_semantic_vec.iter())
            .map(|(result, &norm_score)| {
                (
                    result.record_id.to_string(),
                    (result.record_id, result.record, norm_score),
                )
            })
            .collect();

        let mut keyword_aggregated: HashMap<String, f32> = HashMap::new();
        for result in keyword_results {
            let key = result.document_id.to_string();
            keyword_aggregated
                .entry(key)
                .and_modify(|score| *score = score.max(result.keyword_score))
                .or_insert(result.keyword_score);
        }

        let keyword_keys: Vec<String> = keyword_aggregated.keys().cloned().collect();
        let keyword_scores: Vec<f32> = keyword_keys.iter().map(|k| keyword_aggregated[k]).collect();

        let normalized_keyword_vec =
            normalize_scores(&keyword_scores).unwrap_or_else(|| vec![0.0; keyword_scores.len()]);

        let keyword_map: HashMap<String, f32> = keyword_keys
            .into_iter()
            .zip(normalized_keyword_vec.iter())
            .map(|(k, &score)| (k, score))
            .collect();

        let mut combined_results = Vec::new();
        let all_keys: HashSet<String> = semantic_map
            .keys()
            .chain(keyword_map.keys())
            .cloned()
            .collect();

        for key in all_keys {
            let semantic_score = semantic_map.get(&key).map(|(_, _, s)| *s).unwrap_or(0.0);
            let keyword_score = keyword_map.get(&key).copied().unwrap_or(0.0);

            let combined_score = calculate_weighted_score(
                semantic_score,
                keyword_score,
                self.semantic_weight,
                self.keyword_weight,
            );

            let (record_id, record) = if let Some((id, rec, _)) = semantic_map.get(&key) {
                (id.clone(), rec.clone())
            } else {
                let id = RecordIdKey::from(key.as_str());
                match self.table.table().select(id.clone()).await {
                    Ok(rec) => (id, rec),
                    Err(_) => continue,
                }
            };

            combined_results.push(HybridSearchResult {
                record,
                id: record_id,
                score: combined_score,
                semantic_score,
                keyword_score,
            });
        }

        combined_results.sort_by(|a, b| f32::total_cmp(&b.score, &a.score));
        combined_results.truncate(self.results);

        Ok(combined_results)
    }

    /// Execute the hybrid search using Reciprocal Rank Fusion (RRF)
    ///
    /// RRF combines rankings from semantic and keyword search without
    /// requiring score normalization. It's often more robust than weighted
    /// combination when the score distributions differ significantly.
    ///
    /// # Returns
    /// Vector of search results sorted by RRF score
    ///
    /// # Errors
    /// Returns `HybridSearchError` if the search fails
    pub async fn run_rrf(self) -> Result<Vec<HybridSearchResult<R>>, HybridSearchError>
    where
        R: Serialize + DeserializeOwned + Clone + AsRef<Document> + Send + Sync,
        <M as Embedder>::Error: std::fmt::Debug + std::fmt::Display + 'static,
    {
        let k = self.rrf_k;
        let search_limit = self.results * 3;

        // Perform semantic search
        let semantic_results = self
            .table
            .search(self.user_query.clone())
            .with_results(search_limit)
            .run()
            .await?;

        // Perform keyword search on the links table
        let links_table = self.table.table().table_links();
        let keyword_query = r#"
            SELECT
                document_id,
                byte_range,
                chunk_text,
                search::score(1) as keyword_score
            FROM type::table($table)
            WHERE
                chunk_text @1@ $query
            ORDER BY
                keyword_score DESC
            LIMIT $limit;
        "#;

        let keyword_results: Vec<KeywordChunkResult> = self
            .table
            .table()
            .db()
            .query(keyword_query)
            .bind(("query", self.user_query.clone()))
            .bind(("table", links_table))
            .bind(("limit", search_limit))
            .await?
            .take(0)
            .map_err(|e| HybridSearchError::KeywordSearchError(e.to_string()))?;

        // Build semantic rank map
        let mut semantic_ranks: HashMap<String, (usize, RecordIdKey, R)> = HashMap::new();
        for (rank, result) in semantic_results.iter().enumerate() {
            let key = result.record_id.to_string();
            semantic_ranks.insert(key, (rank, result.record_id.clone(), result.record.clone()));
        }

        // Build keyword rank map (aggregate by document, keep best rank)
        let mut keyword_ranks: HashMap<String, (usize, RecordIdKey)> = HashMap::new();
        for (rank, result) in keyword_results.iter().enumerate() {
            let key = result.document_id.to_string();
            keyword_ranks
                .entry(key)
                .and_modify(|(best_rank, _)| {
                    if rank < *best_rank {
                        *best_rank = rank;
                    }
                })
                .or_insert((rank, result.document_id.clone()));
        }

        // Calculate RRF scores
        let mut rrf_results = Vec::new();
        let all_keys: HashSet<String> = semantic_ranks
            .keys()
            .chain(keyword_ranks.keys())
            .cloned()
            .collect();

        for key in all_keys {
            let semantic_entry = semantic_ranks.get(&key);
            let keyword_entry = keyword_ranks.get(&key);

            let semantic_rrf = semantic_entry
                .map(|(rank, _, _)| calculate_rrf_score(*rank, k))
                .unwrap_or(0.0);

            let keyword_rrf = keyword_entry
                .map(|(rank, _)| calculate_rrf_score(*rank, k))
                .unwrap_or(0.0);

            let rrf_score = semantic_rrf + keyword_rrf;

            // Get the record
            if let Some((_, record_id, record)) = semantic_entry {
                rrf_results.push(HybridSearchResult {
                    record: record.clone(),
                    id: record_id.clone(),
                    score: rrf_score,
                    semantic_score: semantic_rrf,
                    keyword_score: keyword_rrf,
                });
            } else if let Some((_, record_id)) = keyword_entry {
                if let Ok(record) = self.table.table().select(record_id.clone()).await {
                    rrf_results.push(HybridSearchResult {
                        record,
                        id: record_id.clone(),
                        score: rrf_score,
                        semantic_score: semantic_rrf,
                        keyword_score: keyword_rrf,
                    });
                }
            }
        }

        // Sort and truncate
        rrf_results.sort_by(|a, b| f32::total_cmp(&b.score, &a.score));
        rrf_results.truncate(self.results);

        Ok(rrf_results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_scores() {
        // Test Empty
        let scores = vec![];
        let normalized = normalize_scores(&scores);
        assert_eq!(normalized, Some(vec![]));

        // Test Single Score
        let scores = vec![5.0];
        let normalized = normalize_scores(&scores);
        assert_eq!(normalized, Some(vec![1.0]));

        // Test zeros
        let scores = vec![0.0, 0.0, 0.0];
        let normalized = normalize_scores(&scores);

        // Should return None - cannot normalize all zeros
        assert_eq!(normalized, None);

        // Test negatives
        let scores = vec![-1.0, -2.0, -3.0];
        let normalized = normalize_scores(&scores);

        // All negative scores should return None
        assert_eq!(normalized, None);
    }

    #[test]
    fn test_normalize_scores_basic() {
        let scores = vec![10.0, 5.0, 2.5];
        let normalized = normalize_scores(&scores).unwrap();

        // Max is 10.0, so: [10/10, 5/10, 2.5/10] = [1.0, 0.5, 0.25]
        assert!((normalized[0] - 1.0).abs() < 0.001);
        assert!((normalized[1] - 0.5).abs() < 0.001);
        assert!((normalized[2] - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_normalize_scores_range() {
        let scores = vec![100.0, 50.0, 25.0, 10.0];
        let normalized = normalize_scores(&scores).unwrap();

        // All should be in [0, 1]
        assert!(normalized.iter().all(|&n| (0.0..=1.0).contains(&n)));
        // Max should be 1.0
        assert!((normalized[0] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_weighted_score() {
        // Test pure semantic
        let score = calculate_weighted_score(0.9, 0.3, 1.0, 0.0);
        assert!((score - 0.9).abs() < 0.001);

        // Test pure keyword
        let score = calculate_weighted_score(0.9, 0.3, 0.0, 1.0);
        assert!((score - 0.3).abs() < 0.001);

        // Test balanced score
        let score = calculate_weighted_score(0.8, 0.4, 0.5, 0.5);
        // (0.8 + 0.4) / 2 = 0.6
        assert!((score - 0.6).abs() < 0.001);

        // Test zero weights - should return zero
        let score = calculate_weighted_score(0.0, 0.0, 0.7, 0.3);
        assert!((score - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_rrf_score() {
        let k = 60.0;

        // Test rank zero
        let score = calculate_rrf_score(0, k);
        // 1 / (60 + 0 + 1) = 1/61 ≈ 0.0164
        assert!((score - 0.0164).abs() < 0.001);

        // Test increasing rank
        let score0 = calculate_rrf_score(0, k);
        let score1 = calculate_rrf_score(1, k);
        let score2 = calculate_rrf_score(2, k);

        // Scores should decrease with rank
        assert!(score0 > score1);
        assert!(score1 > score2);

        // Test with different k params
        let rank = 0;
        let score_k10 = calculate_rrf_score(rank, 10.0);
        let score_k60 = calculate_rrf_score(rank, 60.0);
        let score_k100 = calculate_rrf_score(rank, 100.0);

        // Smaller k gives higher scores
        assert!(score_k10 > score_k60);
        assert!(score_k60 > score_k100);

        // Test high rank
        let score = calculate_rrf_score(1000, k);

        // Even at high ranks, score should be positive
        assert!(score > 0.0);
        // But very small
        assert!(score < 0.01);
    }
}
