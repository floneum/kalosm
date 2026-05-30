#![allow(missing_docs)]

use rand::seq::SliceRandom;
use url::Url;

use super::{
    document::{Document, IntoDocuments},
    get_article, ExtractDocumentError,
};

/// A search query that can be used to search for documents on the web.
///
/// # Example
/// ```rust, no_run
/// // You must have the SERPER_API_KEY environment variable set to run this example.
/// use kalosm_language::prelude::*;
///
/// #[tokio::main]
/// async fn main() {
///     let query = "What is the best way to learn a language?";
///     let api_key = std::env::var("SERPER_API_KEY").unwrap();
///     let search_query = SearchQuery::new(query, &api_key, 5);
///     let documents = search_query.into_documents().await.unwrap();
///     let mut text = String::new();
///     for document in documents {
///         for word in document.body().split(' ').take(300) {
///             text.push_str(word);
///             text.push(' ');
///         }
///         text.push('\n');
///     }
///     println!("{}", text);
/// }
/// ```
pub struct SearchQuery<'a> {
    query: &'a str,
    api_key: &'a str,
    top: usize,
}

impl<'a> SearchQuery<'a> {
    /// Create a new search query.
    pub fn new(query: &'a str, api_key: &'a str, top_n: usize) -> Self {
        Self {
            query,
            api_key,
            top: top_n,
        }
    }
}

impl IntoDocuments for SearchQuery<'_> {
    type Error = ExtractDocumentError;

    async fn into_documents(self) -> Result<Vec<Document>, Self::Error> {
        let mut search_results = search(self.api_key, self.query).await?;

        let mut documents = vec![];
        search_results.organic.shuffle(&mut rand::thread_rng());

        for result in search_results.organic.into_iter().take(self.top) {
            if let Some(link) = &result.link {
                documents.push(get_article(Url::parse(link)?).await?);
            }
        }

        Ok(documents)
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct SearchResult {
    pub knowledge_graph: Option<KnowledgeGraph>,
    #[serde(default)]
    pub organic: Vec<Organic>,
    #[serde(default)]
    pub people_also_ask: Vec<PeopleAlsoAsk>,
    #[serde(default)]
    pub related_searches: Vec<RelatedSearches>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct KnowledgeGraph {
    pub title: String,
    pub type_: String,
    pub website: String,
    pub image_url: String,
    pub description: String,
    pub description_source: String,
    pub description_link: String,
    #[serde(default)]
    pub attributes: Vec<Attributes>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct Attributes {
    pub key: String,
    pub value: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct Organic {
    pub title: Option<String>,
    pub link: Option<String>,
    #[serde(default)]
    pub snippet: String,
    #[serde(default)]
    pub sitelinks: Vec<Sitelinks>,
    pub position: u32,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct Sitelinks {
    pub title: String,
    pub link: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct PeopleAlsoAsk {
    pub question: String,
    pub snippet: String,
    pub title: String,
    pub link: String,
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub struct RelatedSearches {
    pub query: String,
}

pub async fn search(api_key: &str, query: &str) -> Result<SearchResult, reqwest::Error> {
    let url = Url::parse("https://google.serper.dev/search").unwrap();
    let client = reqwest::Client::new();
    let res = client
        .post(url)
        .header("X-API-KEY", api_key)
        .json(&serde_json::json!({
            "q": query
        }))
        .send()
        .await
        .unwrap();
    res.json().await
}

#[cfg(all(test, feature = "tokio"))]
#[tokio::test]
async fn search_result() {
    if let Some(key) = option_env!("SERPER_API_KEY") {
        let result = search(key, "apple inc").await.unwrap();
        println!("{result:#?}");
    }
}
