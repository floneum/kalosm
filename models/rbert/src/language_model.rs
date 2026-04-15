use std::future::Future;
use std::mem::MaybeUninit;
use std::ops::Deref;
use std::pin::Pin;

pub use crate::Bert;
use crate::BertBuilder;
use crate::BertError;
use crate::BertLoadingError;
use crate::Pooling;
use fusor::ToVec2;
pub use kalosm_language_model::{
    Embedder, EmbedderCacheExt, EmbedderExt, Embedding, EmbeddingInput, EmbeddingVariant,
};
use kalosm_model_types::{FutureWasmNotSend, ModelBuilder, ModelLoadingProgress, WasmNotSend};

impl ModelBuilder for BertBuilder {
    type Model = Bert;
    type Error = BertLoadingError;

    /// Start the model with a loading handler.
    async fn start_with_loading_handler(
        self,
        handler: impl FnMut(ModelLoadingProgress) + Send + Sync + 'static,
    ) -> Result<Self::Model, Self::Error> {
        self.build_with_loading_handler(handler).await
    }

    fn requires_download(&self) -> bool {
        true
    }
}

impl Bert {
    /// Convert a 2D tensor (containing a single embedding) into an Embedding.
    async fn tensor_to_embedding(
        &self,
        tensor: fusor::Tensor<2, f32>,
    ) -> Result<Embedding, BertError> {
        let slice = tensor.as_slice().await.map_err(BertError::Fusor)?;
        let slice_data = slice.to_vec2();
        Ok(Embedding::from(
            slice_data.into_iter().next().into_iter().next().unwrap(),
        ))
    }

    /// Embed a sentence with a specific pooling strategy.
    pub async fn embed_with_pooling(
        &self,
        input: &str,
        pooling: Pooling,
    ) -> Result<Embedding, BertError> {
        let mut tensors = self.embed_batch_raw(vec![input], pooling)?;
        self.tensor_to_embedding(tensors.pop().unwrap()).await
    }

    /// Embed a batch of sentences with a specific pooling strategy.
    pub async fn embed_batch_with_pooling(
        &self,
        inputs: Vec<&str>,
        pooling: Pooling,
    ) -> Result<Vec<Embedding>, BertError> {
        let tensors = self.embed_batch_raw(inputs, pooling)?;

        let mut embeddings = Vec::with_capacity(tensors.len());
        for tensor in tensors {
            embeddings.push(self.tensor_to_embedding(tensor).await?);
        }

        Ok(embeddings)
    }
}

impl Bert {
    /// Apply the search prefix to an embedding input if it's a query variant.
    fn apply_search_prefix(&self, input: EmbeddingInput) -> String {
        match (&*self.embedding_search_prefix, input.variant) {
            (Some(prefix), EmbeddingVariant::Query) => {
                let mut new_input = prefix.clone();
                new_input.push_str(&input.text);
                new_input
            }
            _ => input.text,
        }
    }
}

impl Embedder for Bert {
    type Error = BertError;

    fn embed_for(
        &self,
        input: EmbeddingInput,
    ) -> impl Future<Output = Result<Embedding, Self::Error>> + WasmNotSend {
        self.embed_string(self.apply_search_prefix(input))
    }

    fn embed_vec_for(
        &self,
        inputs: Vec<EmbeddingInput>,
    ) -> impl Future<Output = Result<Vec<Embedding>, Self::Error>> + WasmNotSend {
        let inputs = inputs
            .into_iter()
            .map(|input| self.apply_search_prefix(input))
            .collect::<Vec<_>>();
        self.embed_vec(inputs)
    }

    async fn embed_string(&self, input: String) -> Result<Embedding, Self::Error> {
        let pooling = self.model.default_pooling();
        self.embed_with_pooling(&input, pooling).await
    }

    async fn embed_vec(&self, inputs: Vec<String>) -> Result<Vec<Embedding>, Self::Error> {
        let inputs_borrowed = inputs.iter().map(|s| s.as_str()).collect::<Vec<_>>();
        let pooling = self.model.default_pooling();
        self.embed_batch_with_pooling(inputs_borrowed, pooling)
            .await
    }
}

impl Deref for Bert {
    type Target = dyn Fn(
        &str,
    ) -> Pin<
        Box<dyn FutureWasmNotSend<Output = Result<Embedding, BertError>> + 'static>,
    >;

    fn deref(&self) -> &Self::Target {
        // https://github.com/dtolnay/case-studies/tree/master/callable-types

        // Create an empty allocation for Self.
        let uninit_callable = MaybeUninit::<Self>::uninit();
        // Move a closure that captures just self into the uninitialized memory. Closures create an anonymous type that implement
        // FnOnce. In this case, the layout of the type should just be Self because self is the only field in the closure type.
        let uninit_closure = move |text: &str| {
            let myself = unsafe { &*uninit_callable.as_ptr() };
            let self_clone = myself.clone();
            let input = text.to_string();
            let pooling = self_clone.model.default_pooling();

            Box::pin(async move { self_clone.embed_with_pooling(&input, pooling).await })
                as Pin<Box<dyn FutureWasmNotSend<Output = Result<Embedding, BertError>> + 'static>>
        };

        // Make sure the layout of the closure and Self is the same.
        let size_of_closure = std::alloc::Layout::for_value(&uninit_closure);
        assert_eq!(size_of_closure, std::alloc::Layout::new::<Self>());

        // Then cast the lifetime of the closure to the lifetime of &self.
        fn cast_lifetime<'a, T>(_a: &T, b: &'a T) -> &'a T {
            b
        }
        let reference_to_closure = cast_lifetime(
            {
                // The real closure that we will never use.
                &uninit_closure
            },
            #[allow(clippy::missing_transmute_annotations)]
            // We transmute self into a reference to the closure. This is safe because we know that the closure has the same memory layout as Self so &Closure == &Self.
            unsafe {
                std::mem::transmute(self)
            },
        );

        // Cast the closure to a trait object.
        reference_to_closure as &_
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_bert() {
    use crate::BertSource;

    let bert = Bert::builder()
        .with_source(BertSource::snowflake_arctic_embed_extra_small())
        .build()
        .await
        .unwrap();
    let result = bert("The quick brown fox jumps over the lazy dog.")
        .await
        .unwrap();
    println!("{result:?}");
}

#[cfg(test)]
#[tokio::test]
async fn test_qwen3_embedding() {
    use crate::BertSource;

    let model = Bert::builder()
        .with_source(BertSource::qwen3_embedding_0_6b())
        .build()
        .await
        .unwrap();

    let result = model
        .embed_string("The quick brown fox jumps over the lazy dog.".to_string())
        .await
        .unwrap();

    let vec = result.vector();
    println!("Qwen3 embedding dimension: {}", vec.len());
    println!("First 5 values: {:?}", &vec[..5]);
    assert!(!vec.is_empty());
    assert_eq!(vec.len(), 1024); // Qwen3-Embedding-0.6B has 1024 dimensions
}

#[cfg(test)]
#[tokio::test]
async fn test_qwen3_batch_embedding() {
    use crate::{BertSource, Pooling};

    let model = Bert::builder()
        .with_source(BertSource::qwen3_embedding_0_6b())
        .build()
        .await
        .unwrap();

    // Two sentences of different lengths to exercise padding
    let sentences = vec![
        "Short sentence.",
        "This is a significantly longer sentence to test batch padding behavior.",
    ];

    let results = model
        .embed_batch_with_pooling(sentences, Pooling::Last)
        .await
        .unwrap();

    assert_eq!(results.len(), 2);
    for (i, emb) in results.iter().enumerate() {
        let vec = emb.vector();
        assert_eq!(vec.len(), 1024, "embedding {i} has wrong dimension");
        // Verify L2 normalization (norm should be ~1.0)
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "embedding {i} is not L2-normalized: norm = {norm}"
        );
    }
}
