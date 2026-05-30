use std::{
    fmt::Arguments,
    ops::AddAssign,
    path::{Path, PathBuf},
    sync::Arc,
};

#[cfg(feature = "media")]
use image::ImageResult;
#[cfg(feature = "media")]
use thiserror::Error;

/// The contents of a chat message. The message can contain chunks of interleaved text and media.
///
/// ```rust, no_run
/// # use std::path::PathBuf;
/// # use kalosm::language::*;
/// // You can push text and media chunks to the message content.
/// let mut contents = MessageContent::new();
/// contents.push("Hello, world!");
/// contents.push(MediaSource::url("https://example.com/image.png"));
/// contents
///     .try_push(PathBuf::from("path/to/file.png"))
///     .unwrap();
///
/// // Or create a message content from a tuple of chunks.
/// let contents = MessageContent::from((
///     "Hello, world!",
///     MediaSource::url("https://example.com/image.png"),
/// ));
///
/// // Or use the `+=` operator to add chunks to the message content.
/// let mut contents = MessageContent::from("Hello, world!");
/// contents += MediaSource::url("https://example.com/image.png");
/// contents += MediaSource::try_from(PathBuf::from("path/to/file.png")).unwrap();
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MessageContent {
    chunks: Vec<ContentChunk>,
}

impl MessageContent {
    /// Create a new empty [`MessageContent`].
    pub fn new() -> Self {
        MessageContent { chunks: Vec::new() }
    }

    /// Push a chunk to the end of this message.
    pub fn push(&mut self, chunk: impl Into<ContentChunk>) {
        self.chunks.push(chunk.into());
    }

    /// Try to push a chunk to the end of this message.
    /// Returns an error if the chunk could not be converted to a [`ContentChunk`].
    pub fn try_push<T: TryInto<ContentChunk>>(&mut self, chunk: T) -> Result<(), T::Error> {
        let chunk = chunk.try_into()?;
        self.chunks.push(chunk);
        Ok(())
    }

    /// Get the chunks of this message.
    pub fn chunks(&self) -> &[ContentChunk] {
        &self.chunks
    }

    /// Try to get a reference to the message contents as a [`&str`]. This will only succeed if the message
    /// contains a single text chunk.
    pub fn as_str(&self) -> Option<&str> {
        match self.chunks.as_slice() {
            [ContentChunk::Text(text)] => Some(text),
            _ => None,
        }
    }

    /// Get just the text content of this message.
    pub fn text(&self) -> String {
        self.chunks.iter().fold(String::new(), |mut acc, chunk| {
            if let ContentChunk::Text(text) = chunk {
                acc.push_str(text);
            }
            acc
        })
    }

    /// Get just the media content of this message.
    pub fn media(&self) -> Vec<MediaChunk> {
        self.chunks
            .iter()
            .filter_map(|chunk| {
                if let ContentChunk::Media(media) = chunk {
                    Some(media.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Returns true if this message contains any media chunks.
    pub fn has_media(&self) -> bool {
        self.chunks
            .iter()
            .any(|chunk| matches!(chunk, ContentChunk::Media(_)))
    }

    /// Collect all images from the message content.
    #[cfg(feature = "media")]
    pub async fn images(&self) -> Result<Vec<image::DynamicImage>, ImageFetchError> {
        let mut images = Vec::new();
        for chunk in &self.chunks {
            if let ContentChunk::Media(media) = chunk {
                match &media.source.variant {
                    MediaSourceVariant::Url(url) => {
                        // Fetch the image from the URL
                        let response = reqwest::get(url).await?;
                        let bytes = response.bytes().await?;
                        let image = image::load_from_memory(&bytes)?;
                        images.push(image);
                    }
                    MediaSourceVariant::Bytes(bytes) => {
                        // Decode the image from the bytes
                        let image = image::load_from_memory(bytes.as_ref())?;
                        images.push(image);
                    }
                }
            }
        }
        Ok(images)
    }

    /// Collect all images that are loaded in memory. This will not return any images from [`MediaSourceVariant::Url`].
    #[cfg(feature = "media")]
    pub fn images_in_memory(&self) -> ImageResult<Vec<image::DynamicImage>> {
        self.chunks
            .iter()
            .filter_map(|chunk| {
                if let ContentChunk::Media(media) = chunk {
                    if let MediaSourceVariant::Bytes(bytes) = &media.source.variant {
                        // Decode the image from the bytes
                        Some(image::load_from_memory(bytes.as_ref()))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect()
    }

    /// Resolve all media sources and returned the message content with only [`MediaSourceVariant::Bytes`].
    #[cfg(feature = "media")]
    pub async fn resolve_media_sources(&self) -> Result<Self, ImageFetchError> {
        let mut resolved_chunks = Vec::new();
        for chunk in &self.chunks {
            if let ContentChunk::Media(media) = chunk {
                match &media.source.variant {
                    MediaSourceVariant::Url(url) => {
                        // Fetch the image from the URL
                        let response = reqwest::get(url).await?;
                        let bytes = response.bytes().await?;
                        resolved_chunks.push(ContentChunk::Media(MediaChunk {
                            media_type: media.media_type,
                            source: MediaSource::bytes(bytes.to_vec()),
                            hints: media.hints.clone(),
                        }));
                    }
                    MediaSourceVariant::Bytes(bytes) => {
                        resolved_chunks.push(ContentChunk::Media(MediaChunk {
                            media_type: media.media_type,
                            source: MediaSource::bytes(bytes.clone()),
                            hints: media.hints.clone(),
                        }));
                    }
                }
            } else {
                resolved_chunks.push(chunk.clone());
            }
        }
        Ok(MessageContent {
            chunks: resolved_chunks,
        })
    }
}

/// An error that can occur when fetching images from a URL.
#[cfg(feature = "media")]
#[derive(Error, Debug)]
pub enum ImageFetchError {
    /// An error that occurs when fetching a URL.
    #[error("Failed to fetch image from URL")]
    FetchError(#[from] reqwest::Error),
    /// An error that occurs when decoding an image.
    #[error("Failed to decode image")]
    DecodeError(#[from] image::ImageError),
}

impl PartialEq<str> for MessageContent {
    fn eq(&self, other: &str) -> bool {
        self.as_str().is_some_and(|text| text == other)
    }
}

impl PartialEq<String> for MessageContent {
    fn eq(&self, other: &String) -> bool {
        self.as_str().is_some_and(|text| text == *other)
    }
}

impl<T: Into<ContentChunk>> From<T> for MessageContent {
    fn from(chunk: T) -> Self {
        MessageContent {
            chunks: vec![chunk.into()],
        }
    }
}

macro_rules! impl_from_tuple {
    ($($name:ident),+) => {
        #[allow(non_snake_case)]
        impl<$($name: Into<ContentChunk>),+> From<($($name,)+)> for MessageContent {
            fn from(($($name,)+): ($($name,)+)) -> Self {
                let chunks = vec![
                    $(
                        $name.into(),
                    )+
                ];
                MessageContent { chunks }
            }
        }
    };
}

impl_from_tuple!(A);
impl_from_tuple!(A, B);
impl_from_tuple!(A, B, C);
impl_from_tuple!(A, B, C, D);
impl_from_tuple!(A, B, C, D, E);
impl_from_tuple!(A, B, C, D, E, F);
impl_from_tuple!(A, B, C, D, E, F, G);
impl_from_tuple!(A, B, C, D, E, F, G, H);
impl_from_tuple!(A, B, C, D, E, F, G, H, I);
impl_from_tuple!(A, B, C, D, E, F, G, H, I, J);
impl_from_tuple!(A, B, C, D, E, F, G, H, I, J, K);
impl_from_tuple!(A, B, C, D, E, F, G, H, I, J, K, L);
impl_from_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M);
impl_from_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N);

impl<T> AddAssign<T> for MessageContent
where
    T: Into<ContentChunk>,
{
    fn add_assign(&mut self, rhs: T) {
        self.chunks.push(rhs.into());
    }
}

/// A chunk of content in a chat message.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ContentChunk {
    /// A text chunk.
    Text(String),
    /// A media chunk.
    Media(MediaChunk),
}

impl From<String> for ContentChunk {
    fn from(text: String) -> Self {
        ContentChunk::Text(text)
    }
}

impl From<&String> for ContentChunk {
    fn from(text: &String) -> Self {
        ContentChunk::Text(text.clone())
    }
}

impl From<&str> for ContentChunk {
    fn from(text: &str) -> Self {
        ContentChunk::Text(text.to_string())
    }
}

impl From<Arguments<'_>> for ContentChunk {
    fn from(args: Arguments<'_>) -> Self {
        ContentChunk::Text(args.to_string())
    }
}

impl From<MediaSource> for ContentChunk {
    fn from(source: MediaSource) -> Self {
        ContentChunk::Media(MediaChunk {
            media_type: MediaType::Image,
            source,
            hints: MediaHints::default(),
        })
    }
}

impl From<MediaChunk> for ContentChunk {
    fn from(chunk: MediaChunk) -> Self {
        ContentChunk::Media(chunk)
    }
}

impl TryFrom<PathBuf> for ContentChunk {
    type Error = std::io::Error;

    fn try_from(path: PathBuf) -> Result<Self, Self::Error> {
        Ok(ContentChunk::Media(MediaChunk {
            media_type: MediaType::Image,
            source: MediaSource::try_from(path)?,
            hints: MediaHints::default(),
        }))
    }
}

/// A chunk of media content that can be used with a LLM.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MediaChunk {
    media_type: MediaType,
    source: MediaSource,
    hints: MediaHints,
}

impl MediaChunk {
    /// Create a new [`MediaChunk`] from a [`MediaSource`] and a [`MediaType`].
    pub fn new(source: MediaSource, media_type: MediaType) -> Self {
        MediaChunk {
            media_type,
            source,
            hints: MediaHints::default(),
        }
    }

    /// Get the media type of this chunk.
    pub fn media_type(&self) -> MediaType {
        self.media_type
    }

    /// Get the source of this chunk.
    pub fn source(&self) -> &MediaSource {
        &self.source
    }

    /// Get the hints of this chunk.
    pub fn hints(&self) -> &MediaHints {
        &self.hints
    }

    /// Set the hints of this chunk.
    pub fn with_hints(mut self, hints: MediaHints) -> Self {
        self.hints = hints;
        self
    }
}

/// Additional hints the LLM backend may use to process the media content.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MediaHints {
    min_pixels: Option<u32>,
    max_pixels: Option<u32>,
}

impl MediaHints {
    /// Create a new [`MediaHints`] with the default values.
    pub fn new() -> Self {
        MediaHints {
            min_pixels: None,
            max_pixels: None,
        }
    }

    /// Get the minimum number of tokens for this chunk.
    pub fn min_tokens(&self) -> Option<u32> {
        self.min_pixels
    }

    /// Get the maximum number of tokens for this chunk.
    pub fn max_tokens(&self) -> Option<u32> {
        self.max_pixels
    }

    /// Set the minimum number of tokens for this chunk.
    pub fn with_min_pixels(mut self, min_pixels: u32) -> Self {
        self.min_pixels = Some(min_pixels);
        self
    }

    /// Set the maximum number of tokens for this chunk.
    pub fn with_max_pixels(mut self, max_pixels: u32) -> Self {
        self.max_pixels = Some(max_pixels);
        self
    }
}

/// The type of a [`MediaChunk`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MediaType {
    /// An image media type (e.g. PNG, JPEG).
    Image,
    /// A video media type (e.g. MP4, AVI).
    Video,
}

/// The source of some media content that can be used with a LLM.
///
/// ```rust, no_run
/// # use std::path::PathBuf;
/// # use kalosm::language::*;
/// // You can create a `MediaSource` from a URL,
/// let source = MediaSource::url("https://example.com/image.png");
/// // byte array,
/// let source = MediaSource::bytes(vec![1, 2, 3]);
/// // or a file path.
/// let source = MediaSource::try_from(PathBuf::from("path/to/file.png")).unwrap();
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MediaSource {
    variant: MediaSourceVariant,
}

impl MediaSource {
    /// Create a new `MediaSource` from a URL.
    pub fn url(url: impl ToString) -> Self {
        MediaSource {
            variant: MediaSourceVariant::Url(url.to_string()),
        }
    }

    /// Create a new `MediaSource` from a byte array.
    pub fn bytes(bytes: impl Into<Arc<[u8]>>) -> Self {
        MediaSource {
            variant: MediaSourceVariant::Bytes(bytes.into()),
        }
    }

    /// Try to create a new `MediaSource` from a file path.
    /// Returns an error if the file could not be read.
    pub fn file(path: impl AsRef<Path>) -> Result<Self, std::io::Error> {
        let bytes = std::fs::read(path)?;
        Ok(MediaSource {
            variant: MediaSourceVariant::Bytes(bytes.into()),
        })
    }

    /// Try to get the contents as bytes
    pub fn as_bytes(&self) -> Option<&[u8]> {
        if let MediaSourceVariant::Bytes(bytes) = &self.variant {
            Some(bytes)
        } else {
            None
        }
    }

    /// Try to get the contents as a URL
    pub fn as_url(&self) -> Option<&str> {
        if let MediaSourceVariant::Url(url) = &self.variant {
            Some(url)
        } else {
            None
        }
    }
}

impl From<String> for MediaSource {
    fn from(url: String) -> Self {
        MediaSource {
            variant: MediaSourceVariant::Url(url),
        }
    }
}

impl From<&str> for MediaSource {
    fn from(url: &str) -> Self {
        MediaSource {
            variant: MediaSourceVariant::Url(url.to_string()),
        }
    }
}

impl TryFrom<PathBuf> for MediaSource {
    type Error = std::io::Error;

    fn try_from(path: PathBuf) -> Result<Self, Self::Error> {
        let bytes = std::fs::read(path)?;
        Ok(MediaSource {
            variant: MediaSourceVariant::Bytes(bytes.into()),
        })
    }
}

impl From<Vec<u8>> for MediaSource {
    fn from(bytes: Vec<u8>) -> Self {
        MediaSource {
            variant: MediaSourceVariant::Bytes(bytes.into()),
        }
    }
}

impl From<&[u8]> for MediaSource {
    fn from(bytes: &[u8]) -> Self {
        MediaSource {
            variant: MediaSourceVariant::Bytes(bytes.into()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
enum MediaSourceVariant {
    Url(String),
    #[cfg_attr(feature = "serde", serde(serialize_with = "serialize_bytes"))]
    #[cfg_attr(feature = "serde", serde(deserialize_with = "deserialize_bytes"))]
    Bytes(Arc<[u8]>),
}

#[cfg(feature = "serde")]
fn serialize_bytes<S>(bytes: &Arc<[u8]>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_bytes(&bytes[..])
}

#[cfg(feature = "serde")]
fn deserialize_bytes<'de, D>(deserializer: D) -> Result<Arc<[u8]>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let bytes = <Vec<u8> as serde::Deserialize>::deserialize(deserializer)?;
    Ok(bytes.into())
}

#[cfg(any(feature = "openai", feature = "anthropic"))]
impl MediaChunk {
    /// Create a url from the media chunk by either using the url or encoding the bytes as base64.
    pub fn as_url(&self) -> String {
        use base64::{prelude::BASE64_STANDARD_NO_PAD, Engine};

        match (&self.source.variant, &self.media_type) {
            (MediaSourceVariant::Url(url), _) => url.to_string(),
            (MediaSourceVariant::Bytes(bytes), MediaType::Image) => format!(
                "data:image/png;base64,{}",
                BASE64_STANDARD_NO_PAD.encode(bytes)
            ),
            (MediaSourceVariant::Bytes(bytes), MediaType::Video) => format!(
                "data:video/mp4;base64,{}",
                BASE64_STANDARD_NO_PAD.encode(bytes)
            ),
        }
    }
}
