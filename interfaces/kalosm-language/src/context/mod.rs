//! Context for language models to consume.

mod document;
pub use document::*;
#[cfg(feature = "fs-documents")]
mod io;
#[cfg(feature = "fs-documents")]
pub use io::*;
#[cfg(feature = "scrape")]
mod page;
#[cfg(feature = "scrape")]
pub use page::*;
#[cfg(feature = "web-documents")]
mod rss;
#[cfg(feature = "web-documents")]
pub use self::rss::*;
#[cfg(feature = "web-documents")]
mod search;
#[cfg(feature = "web-documents")]
pub use search::*;

#[cfg(any(feature = "web-documents", feature = "scrape"))]
pub use url::Url;
