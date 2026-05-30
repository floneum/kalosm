use super::{NoOpenAIAPIKeyError, OpenAICompatibleClient};
use crate::{Embedder, Embedding};
use kalosm_model_types::{ModelBuilder, ModelLoadingProgress};
use serde::Deserialize;
use std::future::Future;
use thiserror::Error;

/// An embedder that uses OpenAI's API for the a remote embedding model.
#[derive(Debug)]
pub struct OpenAICompatibleEmbeddingModel {
    model: String,
    client: OpenAICompatibleClient,
}

impl OpenAICompatibleEmbeddingModel {
    /// Create a new builder for [`OpenAICompatibleEmbeddingModel`]
    pub fn builder() -> OpenAICompatibleEmbeddingModelBuilder<false> {
        OpenAICompatibleEmbeddingModelBuilder::new()
    }
}

/// A builder for an openai compatible embedding model.
#[derive(Debug, Default)]
pub struct OpenAICompatibleEmbeddingModelBuilder<const WITH_NAME: bool> {
    model: Option<String>,
    client: OpenAICompatibleClient,
}

impl OpenAICompatibleEmbeddingModelBuilder<false> {
    /// Creates a new builder
    pub fn new() -> Self {
        Self {
            model: None,
            client: Default::default(),
        }
    }
}

impl<const WITH_NAME: bool> OpenAICompatibleEmbeddingModelBuilder<WITH_NAME> {
    /// Set the name of the model to use.
    pub fn with_model(self, model: impl ToString) -> OpenAICompatibleEmbeddingModelBuilder<true> {
        OpenAICompatibleEmbeddingModelBuilder {
            model: Some(model.to_string()),
            client: self.client,
        }
    }

    /// Set the model to text-embedding-3-small. This is the smallest model available with a score of 62.3% on mteb and a max sequence length of 8191
    pub fn with_text_embedding_3_small(self) -> OpenAICompatibleEmbeddingModelBuilder<true> {
        self.with_model("text-embedding-3-small")
    }

    /// Set the model to text-embedding-3-large. This is the smallest model available with a score of 64.6% on mteb and a max sequence length of 8191
    pub fn with_text_embedding_3_large(self) -> OpenAICompatibleEmbeddingModelBuilder<true> {
        self.with_model("text-embedding-3-large")
    }

    /// Set the client used to make requests to the OpenAI API.
    pub fn with_client(mut self, client: OpenAICompatibleClient) -> Self {
        self.client = client;
        self
    }
}

impl OpenAICompatibleEmbeddingModelBuilder<true> {
    /// Build the model.
    pub fn build(self) -> OpenAICompatibleEmbeddingModel {
        OpenAICompatibleEmbeddingModel {
            model: self.model.unwrap(),
            client: self.client,
        }
    }
}

impl ModelBuilder for OpenAICompatibleEmbeddingModelBuilder<true> {
    type Model = OpenAICompatibleEmbeddingModel;
    type Error = std::convert::Infallible;

    async fn start_with_loading_handler(
        self,
        _: impl FnMut(ModelLoadingProgress) + Send + Sync + 'static,
    ) -> Result<Self::Model, Self::Error> {
        Ok(self.build())
    }

    fn requires_download(&self) -> bool {
        false
    }
}

#[derive(Deserialize)]
struct CreateEmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    index: usize,
    embedding: Vec<f32>,
}

/// An error that can occur when running an [`OpenAICompatibleEmbeddingModel`].
#[derive(Error, Debug)]
pub enum OpenAICompatibleEmbeddingModelError {
    /// The API key was not set or was not valid.
    #[error("Error resolving API key: {0}")]
    APIKeyError(#[from] NoOpenAIAPIKeyError),
    /// An error occurred while making a request to the OpenAI API.
    #[error("Error making request: {0}")]
    ReqwestError(#[from] reqwest::Error),
    /// The response from the OpenAI API was not in the format kalosm expected.
    #[error("Invalid response from OpenAI API. The response returned did not contain embeddings for all input strings.")]
    InvalidResponse,
}

impl Embedder for OpenAICompatibleEmbeddingModel {
    type Error = OpenAICompatibleEmbeddingModelError;

    fn embed_for(
        &self,
        input: crate::EmbeddingInput,
    ) -> impl Future<Output = Result<Embedding, Self::Error>> + Send {
        self.embed_string(input.text)
    }

    fn embed_vec_for(
        &self,
        inputs: Vec<crate::EmbeddingInput>,
    ) -> impl Future<Output = Result<Vec<Embedding>, Self::Error>> + Send {
        let inputs = inputs
            .into_iter()
            .map(|input| input.text)
            .collect::<Vec<_>>();
        self.embed_vec(inputs)
    }

    /// Embed a single string.
    async fn embed_string(&self, input: String) -> Result<Embedding, Self::Error> {
        let api_key = self.client.resolve_api_key()?;
        let request = self
            .client
            .reqwest_client
            .post(format!("{}/embeddings", self.client.base_url()))
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&serde_json::json!({
                "model": self.model,
                "input": input,
                "encoding_format": "float"
            }))
            .send()
            .await?;
        let response = request.json::<CreateEmbeddingResponse>().await?;

        let embedding = Embedding::from(response.data[0].embedding.iter().copied());

        Ok(embedding)
    }

    /// Embed a single string.
    async fn embed_vec(&self, input: Vec<String>) -> Result<Vec<Embedding>, Self::Error> {
        let api_key = self.client.resolve_api_key()?;
        let request = self
            .client
            .reqwest_client
            .post(format!("{}/embeddings", self.client.base_url()))
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&serde_json::json!({
                "model": self.model,
                "input": input,
                "encoding_format": "float"
            }))
            .send()
            .await?;
        let mut response = request.json::<CreateEmbeddingResponse>().await?;

        // Verify that the response is valid
        response.data.sort_by_key(|data| data.index);
        #[cfg(debug_assertions)]
        {
            for (i, data) in response.data.iter().enumerate() {
                if data.index != i {
                    return Err(OpenAICompatibleEmbeddingModelError::InvalidResponse);
                }
            }
        }

        let embeddings = response
            .data
            .into_iter()
            .map(|data| Embedding::from(data.embedding))
            .collect();

        Ok(embeddings)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Embedder, EmbedderExt, OpenAICompatibleEmbeddingModelBuilder};

    #[tokio::test]
    async fn test_small_embedding_model() {
        let model = OpenAICompatibleEmbeddingModelBuilder::new()
            .with_client(
                crate::OpenAICompatibleClient::new().with_base_url("https://openrouter.ai/api/v1"),
            )
            .with_model("openai/text-embedding-3-small")
            .build();

        let embeddings = model
            .embed_vec(vec!["Hello, world!".to_string()])
            .await
            .unwrap();
        assert_eq!(embeddings.len(), 1);
        assert!(!embeddings[0].vector().is_empty());

        let embeddings = model.embed("Hello, world!").await.unwrap();
        assert!(!embeddings.vector().is_empty());
    }

    #[tokio::test]
    async fn test_large_embedding_model() {
        let model = OpenAICompatibleEmbeddingModelBuilder::new()
            .with_client(
                crate::OpenAICompatibleClient::new().with_base_url("https://openrouter.ai/api/v1"),
            )
            .with_model("openai/text-embedding-3-large")
            .build();

        let embeddings = model
            .embed_vec(vec!["Hello, world!".to_string()])
            .await
            .unwrap();
        assert_eq!(embeddings.len(), 1);
        assert!(!embeddings[0].vector().is_empty());

        let embeddings = model.embed("Hello, world!").await.unwrap();
        assert!(!embeddings.vector().is_empty());
    }
}
