use std::{convert::Infallible, future::Future};
#[cfg(feature = "document-extraction")]
use url::Url;
pub use whatlang::Lang;

/// A document is a piece of text with a title.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Document {
    title: String,
    body: String,
    summary: Option<String>,
    created_at: Option<chrono::DateTime<chrono::Utc>>,
    updated_at: Option<chrono::DateTime<chrono::Utc>>,
}

impl Document {
    /// Create a new document from a source.
    pub async fn new<T: IntoDocument>(source: T) -> Result<Self, T::Error> {
        source.into_document().await
    }

    /// Get the language of the document.
    pub fn language(&self) -> Option<whatlang::Lang> {
        whatlang::detect_lang(&self.body)
    }

    /// Create a new document from the raw parts.
    pub fn from_parts(title: impl Into<String>, body: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            body: body.into(),
            summary: None,
            created_at: None,
            updated_at: None,
        }
    }

    /// Set the summary of the document.
    pub fn set_summary(&mut self, summary: impl Into<String>) {
        self.summary = Some(summary.into());
    }

    /// Set the created at time of the document.
    pub fn set_created_at(&mut self, created_at: chrono::DateTime<chrono::Utc>) {
        self.created_at = Some(created_at);
    }

    /// Set the updated at time of the document.
    pub fn set_updated_at(&mut self, updated_at: chrono::DateTime<chrono::Utc>) {
        self.updated_at = Some(updated_at);
    }

    /// Get the title of the document.
    pub fn title(&self) -> &str {
        &self.title
    }

    /// Get the body of the document.
    pub fn body(&self) -> &str {
        &self.body
    }
}

impl From<String> for Document {
    fn from(value: String) -> Self {
        Self::from_parts("", value)
    }
}

impl From<&str> for Document {
    fn from(value: &str) -> Self {
        Self::from_parts("", value)
    }
}

impl AsRef<Document> for Document {
    fn as_ref(&self) -> &Document {
        self
    }
}

impl std::fmt::Display for Document {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}\n{}", self.title, self.body)
    }
}

/// A trait for types that can be converted into a document.
pub trait IntoDocument {
    /// The error type that can occur when converting the type into a [`Document`].
    type Error: Send + Sync + 'static;

    /// Convert the type into a document.
    fn into_document(self) -> impl Future<Output = Result<Document, Self::Error>> + Send;
}

impl IntoDocument for String {
    type Error = Infallible;

    async fn into_document(self) -> Result<Document, Self::Error> {
        Ok(Document::from_parts("", self))
    }
}

impl IntoDocument for &String {
    type Error = Infallible;

    async fn into_document(self) -> Result<Document, Self::Error> {
        Ok(Document::from_parts("", self.to_string()))
    }
}

impl IntoDocument for &str {
    type Error = Infallible;

    async fn into_document(self) -> Result<Document, Self::Error> {
        Ok(Document::from_parts("", self.to_string()))
    }
}

impl IntoDocument for Document {
    type Error = Infallible;

    async fn into_document(self) -> Result<Document, Self::Error> {
        Ok(self)
    }
}

/// A document that can be added to a search index.
pub trait IntoDocuments {
    /// The error type that can occur when converting the document into [`Document`]s.
    type Error: Send + Sync + 'static;

    /// Convert the document into a [`Document`]
    fn into_documents(self) -> impl Future<Output = Result<Vec<Document>, Self::Error>> + Send;
}

impl<T: IntoDocument + Send + Sync, I> IntoDocuments for I
where
    I: IntoIterator<Item = T> + Send + Sync,
    <I as IntoIterator>::IntoIter: Send + Sync,
{
    type Error = T::Error;

    async fn into_documents(self) -> Result<Vec<Document>, Self::Error> {
        let mut documents = Vec::new();
        for document in self {
            documents.push(document.into_document().await?);
        }
        Ok(documents)
    }
}

/// An error that can occur when extracting a document from a URL.
#[cfg(feature = "document-extraction")]
#[derive(Debug, thiserror::Error)]
pub enum ExtractDocumentError {
    /// An error occurred when fetching the HTML.
    #[cfg(feature = "web-documents")]
    #[error("Failed to fetch HTML: {0}")]
    FetchHtml(#[from] reqwest::Error),
    /// An error occurred when extracting the article.
    #[error("Failed to extract article: {0}")]
    ExtractArticle(#[from] readability::error::Error),
    /// Failed to parse the URL.
    #[error("Failed to parse URL: {0}")]
    ParseUrl(#[from] url::ParseError),
}

#[cfg(feature = "web-documents")]
pub(crate) async fn get_article(url: Url) -> Result<Document, ExtractDocumentError> {
    let html = reqwest::get(url.clone()).await?.text().await?;
    extract_article(&html)
}

#[cfg(feature = "document-extraction")]
pub(crate) fn extract_article(html: &str) -> Result<Document, ExtractDocumentError> {
    let cleaned =
        readability::extractor::extract(&mut html.as_bytes(), &Url::parse("https://example.com")?)
            .unwrap();
    Ok(Document::from_parts(cleaned.title, cleaned.text))
}

#[cfg(feature = "web-documents")]
impl IntoDocument for Url {
    type Error = ExtractDocumentError;

    async fn into_document(self) -> Result<Document, Self::Error> {
        get_article(self).await
    }
}
