use std::future::Future;

use crate::ModelLoadingProgress;

/// A builder that can create a model asynchronously.
///
/// # Example
/// ```rust, no_run
/// use kalosm::language::*;
///
/// #[tokio::main]
/// async fn main() {
///     let model = Llama::builder().start().await.unwrap();
/// }
/// ```
pub trait ModelBuilder {
    /// The model that this trait creates.
    type Model;

    /// An error that can occur when creating the model.
    type Error: Send + Sync + 'static;

    /// Start the model.
    fn start(self) -> impl Future<Output = Result<Self::Model, Self::Error>>
    where
        Self: Sized,
    {
        async { self.start_with_loading_handler(|_| {}).await }
    }

    /// Start the model with a loading handler.
    fn start_with_loading_handler(
        self,
        handler: impl FnMut(ModelLoadingProgress) + Send + Sync + 'static,
    ) -> impl Future<Output = Result<Self::Model, Self::Error>>
    where
        Self: Sized;

    /// Check if the model will need to be downloaded before use (default: false)
    fn requires_download(&self) -> bool {
        false
    }
}
