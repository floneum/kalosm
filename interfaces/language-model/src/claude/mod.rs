use std::sync::OnceLock;

use thiserror::Error;

mod chat;
pub use chat::*;

/// A client for making requests to an OpenAI compatible API.
#[derive(Debug, Clone)]
pub struct AnthropicCompatibleClient {
    reqwest_client: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    resolved_api_key: OnceLock<String>,
    version: String,
    #[cfg(feature = "resilience")]
    pub(crate) resilience: Option<Resilience>,
}

/// Retry policy for overloaded responses, enabled with the `resilience` feature.
#[cfg(feature = "resilience")]
#[derive(Debug, Clone)]
pub(crate) struct Resilience {
    pub(crate) breaker: llm_resilience::CircuitBreaker,
    pub(crate) spend: llm_resilience::RetrySpend,
    pub(crate) call_usd: f64,
}

impl Default for AnthropicCompatibleClient {
    fn default() -> Self {
        Self::new()
    }
}

impl AnthropicCompatibleClient {
    /// Create a new client.
    pub fn new() -> Self {
        Self {
            reqwest_client: reqwest::Client::new(),
            base_url: "https://api.anthropic.com/v1/".to_string(),
            resolved_api_key: OnceLock::new(),
            api_key: None,
            version: "2023-06-01".to_string(),
            #[cfg(feature = "resilience")]
            resilience: None,
        }
    }

    /// Sets the API key for the builder. (defaults to the environment variable `ANTHROPIC_API_KEY`)
    ///
    /// The API key can be accessed from the Anthropic dashboard [here](https://console.anthropic.com/settings/keys).
    pub fn with_api_key(mut self, api_key: impl ToString) -> Self {
        self.api_key = Some(api_key.to_string());
        self
    }

    /// Set the base URL of the API. (defaults to `https://api.anthropic.com/v1/`)
    pub fn with_base_url(mut self, base_url: impl ToString) -> Self {
        self.base_url = base_url.to_string();
        self
    }

    /// Set the anthropic [version](https://docs.anthropic.com/en/api/versioning) you are using.
    ///
    /// Defaults to `2023-06-01``
    pub fn with_anthropic_version(mut self, version: impl ToString) -> Self {
        self.version = version.to_string();
        self
    }

    /// Set the reqwest client for the builder.
    pub fn with_reqwest_client(mut self, client: reqwest::Client) -> Self {
        self.reqwest_client = client;
        self
    }

    /// Retry 429, 529 and other transient statuses before any token is streamed.
    ///
    /// Each retry is booked against `spend` at `call_usd`, so retries are capped in
    /// dollars, and `breaker` fails calls fast while the API keeps failing.
    #[cfg(feature = "resilience")]
    pub fn with_resilience(
        mut self,
        breaker: llm_resilience::CircuitBreaker,
        spend: llm_resilience::RetrySpend,
        call_usd: f64,
    ) -> Self {
        self.resilience = Some(Resilience {
            breaker,
            spend,
            call_usd,
        });
        self
    }

    /// Resolve the anthropic API key from the environment variable `ANTHROPIC_API_KEY` or the provided api key.
    pub fn resolve_api_key(&self) -> Result<String, NoAnthropicAPIKeyError> {
        if let Some(api_key) = self.resolved_api_key.get() {
            return Ok(api_key.clone());
        }

        let anthropic_api_key = match self.api_key.clone() {
            Some(api_key) => api_key,
            None => std::env::var("ANTHROPIC_API_KEY").map_err(|_| NoAnthropicAPIKeyError)?,
        };

        self.resolved_api_key
            .set(anthropic_api_key.clone())
            .unwrap();

        Ok(anthropic_api_key)
    }

    /// Get the base URL for the Anthropic API.
    pub(crate) fn base_url(&self) -> &str {
        self.base_url.trim_end_matches('/')
    }

    /// Get the version of the Anthropic API.
    pub(crate) fn version(&self) -> &str {
        &self.version
    }
}

/// An error that can occur when building a remote Anthropic model without an API key.
#[derive(Debug, Error)]
#[error("No API key was provided in the [AnthropicCompatibleClient] builder or the environment variable `ANTHROPIC_API_KEY` was not set")]
pub struct NoAnthropicAPIKeyError;
