//! A vector database that can be used to store embeddings and search for similar embeddings.

use arroy::distances::DotProduct;
use heed::{types::*, RwTxn};
use std::fmt::Debug;
use std::sync::atomic::AtomicUsize;

use arroy::{Database as ArroyDatabase, Reader, Writer};
use heed::types::SerdeJson;
use heed::{Database, EnvOpenOptions};
use kalosm_language_model::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// A set of candidates for a vector search.
pub type Candidates = roaring::RoaringBitmap;

/// An error that can occur when adding or searching for an embedding to the vector database.
#[derive(Debug, thiserror::Error)]
pub enum VectorDbError {
    /// An error from the arroy crate.
    #[error("Arroy error: {0}")]
    Arroy(#[from] arroy::Error),
    /// An error from querying an embedding id that does not exist.
    #[error("Embedding {0:?} not found")]
    EmbeddingNotFound(EmbeddingId),
}

impl From<heed::Error> for VectorDbError {
    fn from(value: heed::Error) -> Self {
        Self::Arroy(value.into())
    }
}

/// A vector database that can be used to store embeddings and search for similar embeddings.
///
/// It uses an in memory database with fast lookups for nearest neighbors and points within a certain distance.
///
/// # Example
///
/// ```rust, no_run
/// # use kalosm_language::prelude::*;
/// # use kalosm_language_model::*;
/// # use rbert::*;
/// # use std::collections::HashMap;
/// # #[tokio::main]
/// # async fn main() {
/// // Create a good default Bert model for search
/// let bert = Bert::new_for_search().await.unwrap();
/// let sentences = [
///     "Kalosm can be used to build local AI applications",
///     "With private LLMs data never leaves your computer",
///     "The quick brown fox jumps over the lazy dog",
/// ];
/// // Embed sentences into the vector space
/// let embeddings = bert.embed_batch(sentences).await.unwrap();
/// println!("embeddings {:?}", embeddings);
///
/// // Create a vector database from the embeddings along with a map between the embedding ids and the sentences
/// let db = VectorDB::new().unwrap();
/// let embeddings = db.add_embeddings(embeddings).unwrap();
/// let embedding_id_to_sentence: HashMap<EmbeddingId, &str> =
///     HashMap::from_iter(embeddings.into_iter().zip(sentences));
///
/// // Find the closest sentence to "What is Kalosm?"
/// let query = "What is Kalosm?";
/// // Embed the query into the vector space. We use `embed_query` instead of `embed` because some models embed queries differently than normal text.
/// let embedding = bert.embed_query(query).await.unwrap();
/// let closest = db.search(&embedding).run().unwrap();
/// if let [closest] = closest.as_slice() {
///     let distance = closest.distance;
///     let text = embedding_id_to_sentence.get(&closest.value).unwrap();
///     println!("distance: {distance}");
///     println!("closest:  {text}");
/// }
/// # }
/// ```
#[doc(alias = "VectorDatabase")]
#[doc(alias = "Vector Database")]
pub struct VectorDB {
    database: ArroyDatabase<DotProduct>,
    metadata: Database<Str, SerdeJson<Vec<u32>>>,
    env: heed::Env,
    dim: AtomicUsize,
}

impl Default for VectorDB {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

impl VectorDB {
    fn set_dim(&self, dim: usize) {
        if dim == 0 {
            panic!("Dimension cannot be 0");
        }
        self.dim.store(dim, std::sync::atomic::Ordering::Relaxed);
    }

    fn get_dim(&self) -> Result<usize, arroy::Error> {
        let mut dims = self.dim.load(std::sync::atomic::Ordering::Relaxed);
        if dims == 0 {
            let rtxn = self.env.read_txn()?;
            let reader = Reader::<DotProduct>::open(&rtxn, 0, self.database)?;
            dims = reader.dimensions();
            self.set_dim(dims);
        }
        Ok(dims)
    }

    /// Create a new temporary vector database.
    #[tracing::instrument]
    pub fn new() -> heed::Result<Self> {
        let dir = tempfile::tempdir()?;

        Self::new_at(dir.path())
    }

    /// Create a new vector database at the given path.
    pub fn new_at(path: impl AsRef<std::path::Path>) -> heed::Result<Self> {
        const TWENTY_HUNDRED_MIB: usize = 2 * 1024 * 1024 * 1024;

        std::fs::create_dir_all(&path)?;

        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(TWENTY_HUNDRED_MIB)
                .open(path)
        }?;

        let mut wtxn = env.write_txn()?;
        let db: ArroyDatabase<DotProduct> = env.create_database(&mut wtxn, None)?;
        let metadata: Database<Str, SerdeJson<Vec<u32>>> = env.create_database(&mut wtxn, None)?;
        wtxn.commit()?;

        Ok(Self {
            database: db,
            metadata,
            env,
            dim: AtomicUsize::new(0),
        })
    }

    fn take_id(&self, wtxn: &mut RwTxn) -> Result<EmbeddingId, heed::Error> {
        if let Some(mut free) = self.metadata.get(wtxn, "free")? {
            if let Some(id) = free.pop() {
                self.metadata.put(wtxn, "free", &free)?;
                return Ok(EmbeddingId(id));
            }
        }
        match self.metadata.get(wtxn, "max")? {
            Some(max) => {
                let id = max[0];
                self.metadata.put(wtxn, "max", &vec![id + 1])?;
                Ok(EmbeddingId(id))
            }
            None => {
                self.metadata.put(wtxn, "max", &vec![1])?;
                Ok(EmbeddingId(0))
            }
        }
    }

    fn recycle_id(&self, id: EmbeddingId, wtxn: &mut RwTxn) -> Result<(), heed::Error> {
        let mut free = self.metadata.get(wtxn, "free")?.unwrap_or_default();
        free.push(id.0);
        self.metadata.put(wtxn, "free", &free)?;

        Ok(())
    }

    /// Get the underlying database.
    pub fn raw(&self) -> (&ArroyDatabase<DotProduct>, &heed::Env) {
        (&self.database, &self.env)
    }

    /// Clear the vector database.
    pub async fn clear(&self) -> Result<(), arroy::Error> {
        let mut wtxn = self.env.write_txn()?;
        let dims = self.get_dim()?;
        let writer = Writer::<DotProduct>::new(self.database, 0, dims);
        writer.clear(&mut wtxn)?;

        // Reset the ids
        self.metadata.put(&mut wtxn, "max", &vec![0])?;
        self.metadata.put(&mut wtxn, "free", &vec![])?;
        wtxn.commit()?;

        Ok(())
    }

    /// Rebuild the database.
    pub fn rebuild(
        &self,
        writer: &mut Writer<DotProduct>,
        wtxn: &mut RwTxn,
    ) -> Result<(), arroy::Error> {
        let mut rng = StdRng::from_entropy();
        writer.builder(&mut rng).build(wtxn)?;

        Ok(())
    }

    /// Remove an embedding from the vector database.
    pub fn remove_embedding(&self, embedding_id: EmbeddingId) -> Result<(), arroy::Error> {
        let dims = self.get_dim()?;

        let mut wtxn = self.env.write_txn()?;

        let mut writer = Writer::<DotProduct>::new(self.database, 0, dims);

        writer.del_item(&mut wtxn, embedding_id.0)?;
        self.recycle_id(embedding_id, &mut wtxn)?;

        self.rebuild(&mut writer, &mut wtxn)?;

        wtxn.commit()?;

        Ok(())
    }

    /// Add a new embedding to the vector database.
    ///
    /// Note: Adding embeddings in a batch with [`VectorDB::add_embeddings`] will be faster.
    pub fn add_embedding(&self, embedding: Embedding) -> Result<EmbeddingId, VectorDbError> {
        let embedding = embedding.vector();

        self.set_dim(embedding.len());

        let mut wtxn = self.env.write_txn()?;

        let mut writer = Writer::<DotProduct>::new(self.database, 0, embedding.len());

        let id = self.take_id(&mut wtxn)?;

        writer.add_item(&mut wtxn, id.0, embedding)?;

        self.rebuild(&mut writer, &mut wtxn)?;

        wtxn.commit()?;

        Ok(id)
    }

    /// Add a new batch of embeddings to the vector database.
    pub fn add_embeddings(
        &self,
        embedding: impl IntoIterator<Item = Embedding>,
    ) -> Result<Vec<EmbeddingId>, VectorDbError> {
        let mut embeddings = embedding
            .into_iter()
            .map(|e| e.vector().to_vec().into_boxed_slice());
        let Some(first_embedding) = embeddings.next() else {
            return Ok(Vec::new());
        };
        self.set_dim(first_embedding.len());

        let mut wtxn = self.env.write_txn()?;
        let mut writer = Writer::<DotProduct>::new(self.database, 0, first_embedding.len());

        let mut ids: Vec<_> = Vec::with_capacity(embeddings.size_hint().0 + 1);

        {
            let first_id = self.take_id(&mut wtxn)?;
            writer.add_item(&mut wtxn, first_id.0, &first_embedding)?;
            ids.push(first_id);
        }

        for embedding in embeddings {
            let id = self.take_id(&mut wtxn)?;
            writer.add_item(&mut wtxn, id.0, &embedding)?;
            ids.push(id);
        }

        self.rebuild(&mut writer, &mut wtxn)?;

        wtxn.commit()?;

        Ok(ids)
    }

    /// Get the embedding for an embedding id.
    pub fn get_embedding(&self, embedding_id: EmbeddingId) -> Result<Embedding, VectorDbError> {
        let rtxn = self.env.read_txn()?;
        let reader = Reader::<DotProduct>::open(&rtxn, 0, self.database)?;

        let embedding = reader
            .item_vector(&rtxn, embedding_id.0)?
            .ok_or_else(|| VectorDbError::EmbeddingNotFound(embedding_id))?;

        Ok(Embedding::from(embedding))
    }

    /// Get the closest N embeddings to the given embedding.
    pub fn search<'a>(&'a self, embedding: &'a Embedding) -> VectorDBSearchBuilder<'a> {
        VectorDBSearchBuilder {
            db: self,
            embedding,
            results: None,
            filter: None,
        }
    }
}

/// A trait for anything that can be used to filter the results of a vector search.
pub trait IntoVectorDbSearchFilter<M> {
    /// Convert the filter into a set of candidates.
    fn into_vector_db_search_filter(self, db: &VectorDB) -> Candidates;
}

impl IntoVectorDbSearchFilter<()> for Candidates {
    fn into_vector_db_search_filter(self, _: &VectorDB) -> Candidates {
        self
    }
}

/// A marker type that allows kalosm to specialize the [`IntoVectorDbSearchFilter`] trait for iterators.
pub struct IteratorMarker;

impl<I> IntoVectorDbSearchFilter<IteratorMarker> for I
where
    I: IntoIterator<Item = EmbeddingId>,
{
    fn into_vector_db_search_filter(self, _: &VectorDB) -> Candidates {
        let mut candidates = Candidates::new();
        for id in self {
            candidates.insert(id.0);
        }
        candidates
    }
}

/// A marker type that allows kalosm to specialize the [`IntoVectorDbSearchFilter`] trait for closures.
pub struct ClosureMarker;

impl<I> IntoVectorDbSearchFilter<ClosureMarker> for I
where
    I: FnMut(Embedding) -> bool,
{
    fn into_vector_db_search_filter(mut self, db: &VectorDB) -> Candidates {
        let mut candidates = Candidates::new();
        let rtxn = match db.env.read_txn() {
            Ok(rtxn) => rtxn,
            Err(err) => {
                tracing::error!("Error opening read transaction: {:?}", err);
                return candidates;
            }
        };
        let reader = match Reader::<DotProduct>::open(&rtxn, 0, db.database) {
            Ok(reader) => reader,
            Err(err) => {
                tracing::error!("Error opening reader: {:?}", err);
                return candidates;
            }
        };
        for (key, tensor) in reader.iter(&rtxn).ok().into_iter().flatten().flatten() {
            let embedding = Embedding::from(tensor);
            if self(embedding) {
                candidates.insert(key);
            }
        }
        candidates
    }
}

/// A builder for searching for embeddings in a vector database.
pub struct VectorDBSearchBuilder<'a> {
    db: &'a VectorDB,
    embedding: &'a Embedding,
    results: Option<usize>,
    filter: Option<Candidates>,
}

impl VectorDBSearchBuilder<'_> {
    /// Set the number of results to return. Defaults to 10.
    pub fn with_results(mut self, results: usize) -> Self {
        self.results = Some(results);
        self
    }

    /// Set a filter to apply to the results. Only vectors that pass the filter will be returned.
    pub fn with_filter<Marker>(
        mut self,
        filter: impl IntoVectorDbSearchFilter<Marker> + Send + Sync + 'static,
    ) -> Self {
        self.filter = Some(filter.into_vector_db_search_filter(self.db));
        self
    }

    /// Run the search and return the results.
    pub fn run(self) -> Result<Vec<VectorDBSearchResult>, VectorDbError> {
        let rtxn = self.db.env.read_txn()?;
        let reader = Reader::<DotProduct>::open(&rtxn, 0, self.db.database)?;

        let vector = self.embedding.vector();
        let mut query = reader.nns(self.results.unwrap_or(10));
        if let Some(filter) = self.filter.as_ref() {
            query.candidates(filter);
        }
        let arroy_results = query.by_vector(&rtxn, vector)?;

        Ok(arroy_results
            .into_iter()
            .map(|(id, distance)| {
                let value = EmbeddingId(id);
                VectorDBSearchResult { distance, value }
            })
            .collect::<Vec<_>>())
    }
}

/// A resulting point from a search.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorDBSearchResult {
    /// The distance from the searched point.
    pub distance: f32,
    /// The value of the point.
    pub value: EmbeddingId,
}

/// A unique identifier for an embedding. If you delete an embedding, the id will be recycled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EmbeddingId(pub u32);

#[cfg(all(test, feature = "tokio"))]
#[tokio::test]
async fn test_vector_db_get_closest() {
    let db: VectorDB = VectorDB::new().unwrap();
    let first_vector = Embedding::from([1.0, 2.0, 3.0]);
    let second_embedding = Embedding::from([-1.0, 2.0, 3.0]);
    let id1 = db.add_embedding(first_vector.clone()).unwrap();
    let id2 = db.add_embedding(second_embedding.clone()).unwrap();
    assert_eq!(
        db.search(&first_vector)
            .with_results(1)
            .run()
            .unwrap()
            .iter()
            .map(|r| r.value)
            .collect::<Vec<_>>(),
        vec![id1]
    );
    assert_eq!(
        db.search(&second_embedding)
            .with_results(1)
            .run()
            .unwrap()
            .iter()
            .map(|r| r.value)
            .collect::<Vec<_>>(),
        vec![id2]
    );
    let third_embedding = Embedding::from([1.0, 0.0, 0.0]);
    assert_eq!(
        db.search(&third_embedding)
            .with_results(1)
            .run()
            .unwrap()
            .iter()
            .map(|r| r.value)
            .collect::<Vec<_>>(),
        vec![id1]
    );
    assert_eq!(
        db.search(&third_embedding)
            .with_filter(|vector: Embedding| vector.vector()[0] < 0.0)
            .run()
            .unwrap()
            .iter()
            .map(|r| r.value)
            .collect::<Vec<_>>(),
        vec![id2]
    );
}
