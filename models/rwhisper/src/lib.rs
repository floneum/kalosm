//! # rwhisper
//! A Rust wrapper for [whisper](https://openai.com/research/whisper)
//!
//! ## Usage
//!
//! ```rust, no_run
//! use futures_util::StreamExt;
//! use kalosm::sound::*;
//! use tokio::time::{Duration, Instant};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), anyhow::Error> {
//!     // Create a new tiny whisper model.
//!     let model = WhisperBuilder::default()
//!         .with_source(WhisperSource::tiny())
//!         .build()
//!         .await?;
//!
//!     // Record audio from the microphone for 5 seconds.
//!     let audio = MicInput::default()
//!         .record_until(Instant::now() + Duration::from_secs(5))
//!         .await;
//!
//!     // Transcribe the audio.
//!     let mut text = model.transcribe(audio);
//!
//!     // As the model transcribes the audio, print the text to the console.
//!     text.to_std_out().await.unwrap();
//!
//!     Ok(())
//! }
//! ```

#![warn(missing_docs)]

type WhisperDType = f32;

use cpal::FromSample;
use futures_channel::mpsc::{UnboundedReceiver, UnboundedSender};
use kalosm_common::Cache;
use kalosm_model_types::FutureWasmNotSend;
pub use kalosm_model_types::{FileSource, ModelBuilder, ModelLoadingProgress};
use model::{WhisperInner, WhisperLoadingError};
use rodio::{source::UniformSourceIterator, Source};
use std::{
    fmt::Display,
    ops::Range,
    pin::Pin,
    str::FromStr,
    sync::{Arc, Mutex},
    time::Duration,
};

use futures_util::{FutureExt, Stream, StreamExt};

mod model;
mod source;
pub use source::*;

use crate::{config::SAMPLE_RATE, source::ModelFamily};
mod audio;
mod cohere_audio;
mod cohere_config;
mod cohere_runtime;
mod config;
mod quantized;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
struct DecodingResult {
    text: String,
    avg_logprob: f64,
    no_speech_prob: f64,
    compression_ratio: f64,
    chunks: Vec<TokenChunk>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
struct TokenChunk {
    text_range: Range<usize>,
    timestamp: Option<Range<f32>>,
}

/// A reference to a utf8 token chunk in a segment.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenChunkRef<'a> {
    chunk: &'a TokenChunk,
    text: &'a str,
}

impl<'a> TokenChunkRef<'a> {
    /// Get the byte range of the token chunk in the full segment text.
    pub fn text_range(&self) -> Range<usize> {
        self.chunk.text_range.clone()
    }

    /// Get the timestamp range of the token chunk in the full segment text if the transcription was created with word level timestamps.
    pub fn timestamp(&self) -> Option<Range<f32>> {
        self.chunk.timestamp.clone()
    }

    /// Get the text of the token chunk.
    pub fn text(&self) -> &'a str {
        &self.text[self.chunk.text_range.clone()]
    }
}

impl AsRef<str> for TokenChunkRef<'_> {
    fn as_ref(&self) -> &str {
        self.text()
    }
}

impl std::fmt::Display for TokenChunkRef<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.text())
    }
}

/// A transcribed segment of audio.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Segment {
    sample_range: Range<usize>,
    start: f64,
    duration: f64,
    elapsed_time: Option<Duration>,
    remaining_time: Option<Duration>,
    progress: f32,
    result: DecodingResult,
}

impl Segment {
    /// Get the range this segment covers in the original audio.
    pub fn sample_range(&self) -> Range<usize> {
        self.sample_range.clone()
    }

    /// Get the probability of no speech.
    pub fn probability_of_no_speech(&self) -> f64 {
        self.result.no_speech_prob
    }

    /// Get the text of the segment.
    pub fn text(&self) -> &str {
        &self.result.text
    }

    /// Get the utf8 token chunks of the segment.
    pub fn chunks(&self) -> impl Iterator<Item = TokenChunkRef<'_>> {
        self.result.chunks.iter().map(|chunk| TokenChunkRef {
            chunk,
            text: &self.result.text,
        })
    }

    /// Get the start timestamp of the segment.
    pub fn start(&self) -> f64 {
        self.start
    }

    /// Get the duration of the segment.
    pub fn duration(&self) -> f64 {
        self.duration
    }

    /// Get the elapsed time
    pub fn elapsed_time(&self) -> Option<Duration> {
        self.elapsed_time
    }

    /// Get the estimated time remaining to process the entire audio file
    pub fn remaining_time(&self) -> Option<Duration> {
        self.remaining_time
    }

    /// The progress of the transcription, from 0 to 1
    pub fn progress(&self) -> f32 {
        self.progress
    }

    /// Return the confidence of the transcription result (between 0 and 1)
    pub fn confidence(&self) -> f64 {
        self.result.avg_logprob.exp()
    }
}

impl AsRef<str> for Segment {
    fn as_ref(&self) -> &str {
        if self.probability_of_no_speech() < 0.10 {
            self.text()
        } else {
            ""
        }
    }
}

/// An extension trait to transcribe pre-chunked audio streams
pub trait TranscribeChunkedAudioStreamExt<S> {
    /// Transcribe each chunk of the audio stream with whisper and stream the result
    fn transcribe(self, model: Whisper) -> ChunkedTranscriptionTask<S>;
}

impl<S> TranscribeChunkedAudioStreamExt<S> for S
where
    S: Stream + std::marker::Unpin + 'static,
    <S as Stream>::Item: Source + 'static,
    <<S as Stream>::Item as Iterator>::Item: rodio::Sample,
    f32: FromSample<<<S as Stream>::Item as Iterator>::Item>,
{
    fn transcribe(self, model: Whisper) -> ChunkedTranscriptionTask<S> {
        ChunkedTranscriptionTask {
            word_level_time_stamps: false,
            stream: self,
            whisper: model,
            current_segment_task: None,
            language: Some(WhisperLanguage::English),
        }
    }
}

/// A chunked audio transcription task which can be streamed from a [`Whisper`] model.
pub struct ChunkedTranscriptionTask<S> {
    word_level_time_stamps: bool,
    stream: S,
    whisper: Whisper,
    current_segment_task: Option<TranscriptionTask>,
    language: Option<WhisperLanguage>,
}

impl<S> ChunkedTranscriptionTask<S> {
    /// Include word level timestamps in the transcription.
    pub fn timestamped(mut self) -> Self {
        self.word_level_time_stamps = true;
        self
    }

    /// Set the language to be used.
    pub fn with_language<L>(mut self, language: L) -> Self
    where
        L: Into<WhisperLanguage>,
    {
        self.language = Some(language.into());
        self
    }
}

impl<S> Stream for ChunkedTranscriptionTask<S>
where
    S: Stream + std::marker::Unpin + Send + 'static,
    <S as Stream>::Item: Source + Send + 'static,
    <<S as Stream>::Item as Iterator>::Item: rodio::Sample,
    f32: FromSample<<<S as Stream>::Item as Iterator>::Item>,
{
    type Item = Segment;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let myself = self.get_mut();

        loop {
            if let Some(task) = &mut myself.current_segment_task {
                match task.poll_next_unpin(cx) {
                    std::task::Poll::Ready(Some(ready)) => {
                        return std::task::Poll::Ready(Some(ready));
                    }
                    std::task::Poll::Ready(None) => {
                        myself.current_segment_task = None;
                    }
                    std::task::Poll::Pending => return std::task::Poll::Pending,
                }
            }

            match myself.stream.poll_next_unpin(cx) {
                std::task::Poll::Ready(Some(source)) => {
                    let mut task = myself.whisper.transcribe(source);
                    if myself.word_level_time_stamps {
                        task = task.timestamped();
                    }

                    if let Some(language) = myself.language {
                        task = task.with_language(language);
                    }

                    myself.current_segment_task = Some(task);
                }
                std::task::Poll::Ready(None) => return std::task::Poll::Ready(None),
                std::task::Poll::Pending => return std::task::Poll::Pending,
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Task {
    task_type: TaskType,
    word_level_time_stamps: bool,
    without_timestamps: bool,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
enum TaskType {
    Transcribe,
    Translate,
    Unset,
}

/// A builder with configuration for a Whisper model.
///
/// ```rust, no_run
/// use kalosm::sound::*;
/// # #[tokio::main]
/// # async fn main() -> Result<(), anyhow::Error> {
/// // Create a new whisper model with a loading handler
/// let model = Whisper::builder()
///     // You can set the model to use in the builder
///     .with_source(WhisperSource::distil_large_v3())
///     .build()
///     .await?;
/// # Ok(())
/// # }
#[derive(Debug)]
pub struct WhisperBuilder {
    /// The model to be used, can be tiny, small, medium.
    model: WhisperSource,

    /// Language.
    language: Option<WhisperLanguage>,

    /// The cache location to use for the model (defaults DATA_DIR/kalosm/cache)
    cache: kalosm_common::Cache,
}

impl Default for WhisperBuilder {
    fn default() -> Self {
        Self {
            model: WhisperSource::default(),
            language: Some(WhisperLanguage::English),
            cache: kalosm_common::Cache::default(),
        }
    }
}

impl ModelBuilder for WhisperBuilder {
    type Model = Whisper;
    type Error = WhisperLoadingError;

    async fn start_with_loading_handler(
        self,
        handler: impl FnMut(ModelLoadingProgress) + Send + Sync + 'static,
    ) -> Result<Self::Model, Self::Error> {
        self.build_with_loading_handler(handler).await
    }

    fn requires_download(&self) -> bool {
        let whisper = &self.model;
        let cache = Cache::default();
        !cache.exists(&whisper.model)
            || !cache.exists(&whisper.tokenizer)
            || !cache.exists(&whisper.config)
    }
}

impl WhisperBuilder {
    /// Build the model.
    pub async fn build(self) -> Result<Whisper, WhisperLoadingError> {
        self.build_with_loading_handler(ModelLoadingProgress::multi_bar_loading_indicator())
            .await
    }

    /// Build the model with a handler for progress as the download and loading progresses.
    ///
    /// ```rust, no_run
    /// use kalosm::sound::*;
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), anyhow::Error> {
    /// // Create a new whisper model with a loading handler
    /// let model = Whisper::builder()
    ///     .build_with_loading_handler(|progress| match progress {
    ///         ModelLoadingProgress::Downloading { source, progress } => {
    ///             let progress_percent = (progress.progress * 100) as u32;
    ///             let elapsed = progress.start_time.unwrap().elapsed().as_secs_f32();
    ///             println!("Downloading file {source} {progress_percent}% ({elapsed}s)");
    ///         }
    ///         ModelLoadingProgress::Loading { progress } => {
    ///             let progress = (progress * 100.0) as u32;
    ///             println!("Loading model {progress}%");
    ///         }
    ///     })
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn build_with_loading_handler(
        self,
        mut progress_handler: impl FnMut(ModelLoadingProgress) + 'static,
    ) -> Result<Whisper, WhisperLoadingError> {
        // Download section
        let whisper = self.model.clone();
        let tokenizer_source = &whisper.tokenizer;
        let model_source = &whisper.model;
        let config_source = &whisper.config;

        let display_tokenizer_source = format!("Tokenizer ({tokenizer_source})");
        let mut create_progress =
            ModelLoadingProgress::downloading_progress(display_tokenizer_source);
        let tokenizer = match tokenizer_source {
            FileSource::Local(path) => {
                let size = std::fs::metadata(path)
                    .map_err(kalosm_common::CacheError::from)?
                    .len();
                progress_handler(create_progress(kalosm_model_types::FileLoadingProgress {
                    start_time: None,
                    cached_size: 0,
                    size,
                    progress: size,
                }));
                std::fs::read(path).map_err(kalosm_common::CacheError::from)?
            }
            _ => {
                self.cache
                    .get_bytes(tokenizer_source, |progress| {
                        progress_handler(create_progress(progress))
                    })
                    .await?
            }
        };

        let display_model_source = format!("Model ({model_source})");
        let mut create_progress = ModelLoadingProgress::downloading_progress(display_model_source);
        let model = match model_source {
            FileSource::Local(path) => {
                let size = std::fs::metadata(path)
                    .map_err(kalosm_common::CacheError::from)?
                    .len();
                progress_handler(create_progress(kalosm_model_types::FileLoadingProgress {
                    start_time: None,
                    cached_size: 0,
                    size,
                    progress: size,
                }));
                std::fs::read(path).map_err(kalosm_common::CacheError::from)?
            }
            _ => {
                self.cache
                    .get_bytes(model_source, |progress| {
                        progress_handler(create_progress(progress))
                    })
                    .await?
            }
        };

        let display_config_source = format!("Config ({config_source})");
        let mut create_progress = ModelLoadingProgress::downloading_progress(display_config_source);
        let config = match config_source {
            FileSource::Local(path) => {
                let size = std::fs::metadata(path)
                    .map_err(kalosm_common::CacheError::from)?
                    .len();
                progress_handler(create_progress(kalosm_model_types::FileLoadingProgress {
                    start_time: None,
                    cached_size: 0,
                    size,
                    progress: size,
                }));
                std::fs::read(path).map_err(kalosm_common::CacheError::from)?
            }
            _ => {
                self.cache
                    .get_bytes(config_source, |progress| {
                        progress_handler(create_progress(progress))
                    })
                    .await?
            }
        };

        let (tx, rx) = futures_channel::mpsc::unbounded::<WhisperMessage>();
        let mut model = WhisperInner::new(self, &model, &tokenizer, &config).await?;

        let task = Box::pin(async move {
            let mut rx = rx;
            while let Some(message) = rx.next().await {
                model
                    .transcribe(
                        message.samples,
                        message.timestamps,
                        message.lang,
                        message.sender,
                    )
                    .await;
            }
        });

        Ok(Whisper {
            inner: Arc::new(WhisperTask {
                sender: tx,
                task: Mutex::new(task),
                apply_speech_filter: matches!(
                    whisper.family,
                    ModelFamily::Whisper {
                        apply_speech_filter: true,
                        ..
                    }
                ),
            }),
        })
    }

    /// Set the model to be used.
    pub fn with_source(mut self, model: WhisperSource) -> Self {
        self.model = model;
        self
    }

    /// Set the language to be used.
    pub fn with_language(mut self, language: Option<WhisperLanguage>) -> Self {
        self.language = language;
        self
    }

    /// Set the cache location to use for the model (defaults DATA_DIR/kalosm/cache)
    pub fn with_cache(mut self, cache: kalosm_common::Cache) -> Self {
        self.cache = cache;

        self
    }
}

/// A language whisper can use
#[allow(missing_docs)]
#[derive(Debug, Clone, Copy)]
pub enum WhisperLanguage {
    English,
    Chinese,
    German,
    Spanish,
    Russian,
    Korean,
    French,
    Japanese,
    Portuguese,
    Turkish,
    Polish,
    Catalan,
    Dutch,
    Arabic,
    Swedish,
    Italian,
    Indonesian,
    Hindi,
    Finnish,
    Vietnamese,
    Hebrew,
    Ukrainian,
    Greek,
    Malay,
    Czech,
    Romanian,
    Danish,
    Hungarian,
    Tamil,
    Norwegian,
    Thai,
    Urdu,
    Croatian,
    Bulgarian,
    Lithuanian,
    Latin,
    Maori,
    Malayalam,
    Welsh,
    Slovak,
    Telugu,
    Persian,
    Latvian,
    Bengali,
    Serbian,
    Azerbaijani,
    Slovenian,
    Kannada,
    Estonian,
    Macedonian,
    Breton,
    Basque,
    Icelandic,
    Armenian,
    Nepali,
    Mongolian,
    Bosnian,
    Kazakh,
    Albanian,
    Swahili,
    Galician,
    Marathi,
    Punjabi,
    Sinhala,
    Khmer,
    Shona,
    Yoruba,
    Somali,
    Afrikaans,
    Occitan,
    Georgian,
    Belarusian,
    Tajik,
    Sindhi,
    Gujarati,
    Amharic,
    Yiddish,
    Lao,
    Uzbek,
    Faroese,
    HaitianCreole,
    Pashto,
    Turkmen,
    Nynorsk,
    Maltese,
    Sanskrit,
    Luxembourgish,
    Myanmar,
    Tibetan,
    Tagalog,
    Malagasy,
    Assamese,
    Tatar,
    Hawaiian,
    Lingala,
    Hausa,
    Bashkir,
    Javanese,
    Sundanese,
}

/// Error that reports the unsupported value
#[derive(PartialEq, Eq)]
pub struct ParseWhisperLanguageError(String);

impl Display for ParseWhisperLanguageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Language {} not supported ", self.0)
    }
}

impl FromStr for WhisperLanguage {
    type Err = ParseWhisperLanguageError;

    fn from_str(s: &str) -> std::prelude::v1::Result<Self, Self::Err> {
        match s {
            "en" => Ok(WhisperLanguage::English),
            "zh" => Ok(WhisperLanguage::Chinese),
            "de" => Ok(WhisperLanguage::German),
            "es" => Ok(WhisperLanguage::Spanish),
            "ru" => Ok(WhisperLanguage::Russian),
            "ko" => Ok(WhisperLanguage::Korean),
            "fr" => Ok(WhisperLanguage::French),
            "ja" => Ok(WhisperLanguage::Japanese),
            "pt" => Ok(WhisperLanguage::Portuguese),
            "tr" => Ok(WhisperLanguage::Turkish),
            "pl" => Ok(WhisperLanguage::Polish),
            "ca" => Ok(WhisperLanguage::Catalan),
            "nl" => Ok(WhisperLanguage::Dutch),
            "ar" => Ok(WhisperLanguage::Arabic),
            "sv" => Ok(WhisperLanguage::Swedish),
            "it" => Ok(WhisperLanguage::Italian),
            "id" => Ok(WhisperLanguage::Indonesian),
            "hi" => Ok(WhisperLanguage::Hindi),
            "fi" => Ok(WhisperLanguage::Finnish),
            "vi" => Ok(WhisperLanguage::Vietnamese),
            "he" => Ok(WhisperLanguage::Hebrew),
            "uk" => Ok(WhisperLanguage::Ukrainian),
            "el" => Ok(WhisperLanguage::Greek),
            "ms" => Ok(WhisperLanguage::Malay),
            "cs" => Ok(WhisperLanguage::Czech),
            "ro" => Ok(WhisperLanguage::Romanian),
            "da" => Ok(WhisperLanguage::Danish),
            "hu" => Ok(WhisperLanguage::Hungarian),
            "ta" => Ok(WhisperLanguage::Tamil),
            "no" => Ok(WhisperLanguage::Norwegian),
            "th" => Ok(WhisperLanguage::Thai),
            "ur" => Ok(WhisperLanguage::Urdu),
            "hr" => Ok(WhisperLanguage::Croatian),
            "bg" => Ok(WhisperLanguage::Bulgarian),
            "lt" => Ok(WhisperLanguage::Lithuanian),
            "la" => Ok(WhisperLanguage::Latin),
            "mi" => Ok(WhisperLanguage::Maori),
            "ml" => Ok(WhisperLanguage::Malayalam),
            "cy" => Ok(WhisperLanguage::Welsh),
            "sk" => Ok(WhisperLanguage::Slovak),
            "te" => Ok(WhisperLanguage::Telugu),
            "fa" => Ok(WhisperLanguage::Persian),
            "lv" => Ok(WhisperLanguage::Latvian),
            "bn" => Ok(WhisperLanguage::Bengali),
            "sr" => Ok(WhisperLanguage::Serbian),
            "az" => Ok(WhisperLanguage::Azerbaijani),
            "sl" => Ok(WhisperLanguage::Slovenian),
            "kn" => Ok(WhisperLanguage::Kannada),
            "et" => Ok(WhisperLanguage::Estonian),
            "mk" => Ok(WhisperLanguage::Macedonian),
            "br" => Ok(WhisperLanguage::Breton),
            "eu" => Ok(WhisperLanguage::Basque),
            "is" => Ok(WhisperLanguage::Icelandic),
            "hy" => Ok(WhisperLanguage::Armenian),
            "ne" => Ok(WhisperLanguage::Nepali),
            "mn" => Ok(WhisperLanguage::Mongolian),
            "bs" => Ok(WhisperLanguage::Bosnian),
            "kk" => Ok(WhisperLanguage::Kazakh),
            "sq" => Ok(WhisperLanguage::Albanian),
            "sw" => Ok(WhisperLanguage::Swahili),
            "gl" => Ok(WhisperLanguage::Galician),
            "mr" => Ok(WhisperLanguage::Marathi),
            "pa" => Ok(WhisperLanguage::Punjabi),
            "si" => Ok(WhisperLanguage::Sinhala),
            "km" => Ok(WhisperLanguage::Khmer),
            "sn" => Ok(WhisperLanguage::Shona),
            "yo" => Ok(WhisperLanguage::Yoruba),
            "so" => Ok(WhisperLanguage::Somali),
            "af" => Ok(WhisperLanguage::Afrikaans),
            "oc" => Ok(WhisperLanguage::Occitan),
            "ka" => Ok(WhisperLanguage::Georgian),
            "be" => Ok(WhisperLanguage::Belarusian),
            "tg" => Ok(WhisperLanguage::Tajik),
            "sd" => Ok(WhisperLanguage::Sindhi),
            "gu" => Ok(WhisperLanguage::Gujarati),
            "am" => Ok(WhisperLanguage::Amharic),
            "yi" => Ok(WhisperLanguage::Yiddish),
            "lo" => Ok(WhisperLanguage::Lao),
            "uz" => Ok(WhisperLanguage::Uzbek),
            "fo" => Ok(WhisperLanguage::Faroese),
            "ht" => Ok(WhisperLanguage::HaitianCreole),
            "ps" => Ok(WhisperLanguage::Pashto),
            "tk" => Ok(WhisperLanguage::Turkmen),
            "nn" => Ok(WhisperLanguage::Nynorsk),
            "mt" => Ok(WhisperLanguage::Maltese),
            "sa" => Ok(WhisperLanguage::Sanskrit),
            "lb" => Ok(WhisperLanguage::Luxembourgish),
            "my" => Ok(WhisperLanguage::Myanmar),
            "bo" => Ok(WhisperLanguage::Tibetan),
            "tl" => Ok(WhisperLanguage::Tagalog),
            "mg" => Ok(WhisperLanguage::Malagasy),
            "as" => Ok(WhisperLanguage::Assamese),
            "tt" => Ok(WhisperLanguage::Tatar),
            "haw" => Ok(WhisperLanguage::Hawaiian),
            "ln" => Ok(WhisperLanguage::Lingala),
            "ha" => Ok(WhisperLanguage::Hausa),
            "ba" => Ok(WhisperLanguage::Bashkir),
            "jw" => Ok(WhisperLanguage::Javanese),
            "su" => Ok(WhisperLanguage::Sundanese),
            _ => Err(ParseWhisperLanguageError(s.to_owned())),
        }
    }
}

impl Display for WhisperLanguage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WhisperLanguage::English => write!(f, "en"),
            WhisperLanguage::Chinese => write!(f, "zh"),
            WhisperLanguage::German => write!(f, "de"),
            WhisperLanguage::Spanish => write!(f, "es"),
            WhisperLanguage::Russian => write!(f, "ru"),
            WhisperLanguage::Korean => write!(f, "ko"),
            WhisperLanguage::French => write!(f, "fr"),
            WhisperLanguage::Japanese => write!(f, "ja"),
            WhisperLanguage::Portuguese => write!(f, "pt"),
            WhisperLanguage::Turkish => write!(f, "tr"),
            WhisperLanguage::Polish => write!(f, "pl"),
            WhisperLanguage::Catalan => write!(f, "ca"),
            WhisperLanguage::Dutch => write!(f, "nl"),
            WhisperLanguage::Arabic => write!(f, "ar"),
            WhisperLanguage::Swedish => write!(f, "sv"),
            WhisperLanguage::Italian => write!(f, "it"),
            WhisperLanguage::Indonesian => write!(f, "id"),
            WhisperLanguage::Hindi => write!(f, "hi"),
            WhisperLanguage::Finnish => write!(f, "fi"),
            WhisperLanguage::Vietnamese => write!(f, "vi"),
            WhisperLanguage::Hebrew => write!(f, "he"),
            WhisperLanguage::Ukrainian => write!(f, "uk"),
            WhisperLanguage::Greek => write!(f, "el"),
            WhisperLanguage::Malay => write!(f, "ms"),
            WhisperLanguage::Czech => write!(f, "cs"),
            WhisperLanguage::Romanian => write!(f, "ro"),
            WhisperLanguage::Danish => write!(f, "da"),
            WhisperLanguage::Hungarian => write!(f, "hu"),
            WhisperLanguage::Tamil => write!(f, "ta"),
            WhisperLanguage::Norwegian => write!(f, "no"),
            WhisperLanguage::Thai => write!(f, "th"),
            WhisperLanguage::Urdu => write!(f, "ur"),
            WhisperLanguage::Croatian => write!(f, "hr"),
            WhisperLanguage::Bulgarian => write!(f, "bg"),
            WhisperLanguage::Lithuanian => write!(f, "lt"),
            WhisperLanguage::Latin => write!(f, "la"),
            WhisperLanguage::Maori => write!(f, "mi"),
            WhisperLanguage::Malayalam => write!(f, "ml"),
            WhisperLanguage::Welsh => write!(f, "cy"),
            WhisperLanguage::Slovak => write!(f, "sk"),
            WhisperLanguage::Telugu => write!(f, "te"),
            WhisperLanguage::Persian => write!(f, "fa"),
            WhisperLanguage::Latvian => write!(f, "lv"),
            WhisperLanguage::Bengali => write!(f, "bn"),
            WhisperLanguage::Serbian => write!(f, "sr"),
            WhisperLanguage::Azerbaijani => write!(f, "az"),
            WhisperLanguage::Slovenian => write!(f, "sl"),
            WhisperLanguage::Kannada => write!(f, "kn"),
            WhisperLanguage::Estonian => write!(f, "et"),
            WhisperLanguage::Macedonian => write!(f, "mk"),
            WhisperLanguage::Breton => write!(f, "br"),
            WhisperLanguage::Basque => write!(f, "eu"),
            WhisperLanguage::Icelandic => write!(f, "is"),
            WhisperLanguage::Armenian => write!(f, "hy"),
            WhisperLanguage::Nepali => write!(f, "ne"),
            WhisperLanguage::Mongolian => write!(f, "mn"),
            WhisperLanguage::Bosnian => write!(f, "bs"),
            WhisperLanguage::Kazakh => write!(f, "kk"),
            WhisperLanguage::Albanian => write!(f, "sq"),
            WhisperLanguage::Swahili => write!(f, "sw"),
            WhisperLanguage::Galician => write!(f, "gl"),
            WhisperLanguage::Marathi => write!(f, "mr"),
            WhisperLanguage::Punjabi => write!(f, "pa"),
            WhisperLanguage::Sinhala => write!(f, "si"),
            WhisperLanguage::Khmer => write!(f, "km"),
            WhisperLanguage::Shona => write!(f, "sn"),
            WhisperLanguage::Yoruba => write!(f, "yo"),
            WhisperLanguage::Somali => write!(f, "so"),
            WhisperLanguage::Afrikaans => write!(f, "af"),
            WhisperLanguage::Occitan => write!(f, "oc"),
            WhisperLanguage::Georgian => write!(f, "ka"),
            WhisperLanguage::Belarusian => write!(f, "be"),
            WhisperLanguage::Tajik => write!(f, "tg"),
            WhisperLanguage::Sindhi => write!(f, "sd"),
            WhisperLanguage::Gujarati => write!(f, "gu"),
            WhisperLanguage::Amharic => write!(f, "am"),
            WhisperLanguage::Yiddish => write!(f, "yi"),
            WhisperLanguage::Lao => write!(f, "lo"),
            WhisperLanguage::Uzbek => write!(f, "uz"),
            WhisperLanguage::Faroese => write!(f, "fo"),
            WhisperLanguage::HaitianCreole => write!(f, "ht"),
            WhisperLanguage::Pashto => write!(f, "ps"),
            WhisperLanguage::Turkmen => write!(f, "tk"),
            WhisperLanguage::Nynorsk => write!(f, "nn"),
            WhisperLanguage::Maltese => write!(f, "mt"),
            WhisperLanguage::Sanskrit => write!(f, "sa"),
            WhisperLanguage::Luxembourgish => write!(f, "lb"),
            WhisperLanguage::Myanmar => write!(f, "my"),
            WhisperLanguage::Tibetan => write!(f, "bo"),
            WhisperLanguage::Tagalog => write!(f, "tl"),
            WhisperLanguage::Malagasy => write!(f, "mg"),
            WhisperLanguage::Assamese => write!(f, "as"),
            WhisperLanguage::Tatar => write!(f, "tt"),
            WhisperLanguage::Hawaiian => write!(f, "haw"),
            WhisperLanguage::Lingala => write!(f, "ln"),
            WhisperLanguage::Hausa => write!(f, "ha"),
            WhisperLanguage::Bashkir => write!(f, "ba"),
            WhisperLanguage::Javanese => write!(f, "jw"),
            WhisperLanguage::Sundanese => write!(f, "su"),
        }
    }
}

struct WhisperTask {
    sender: UnboundedSender<WhisperMessage>,
    task: Mutex<Pin<Box<dyn FutureWasmNotSend<Output = ()> + 'static>>>,
    apply_speech_filter: bool,
}

#[derive(Clone)]
/// A quantized whisper audio transcription model.
pub struct Whisper {
    inner: Arc<WhisperTask>,
}

impl Whisper {
    /// Create a builder for a Whisper model.
    pub fn builder() -> WhisperBuilder {
        WhisperBuilder::default()
    }

    /// Create a new default whisper model.
    pub async fn new() -> Result<Self, WhisperLoadingError> {
        let model = Self::builder().build().await?;
        Ok(model)
    }

    /// Transcribe some audio into text.
    ///
    /// Dropping the returned channel will stop the transcription early.
    pub fn transcribe<S: Source>(&self, input: S) -> TranscriptionTask
    where
        <S as Iterator>::Item: rodio::Sample,
        f32: FromSample<<S as Iterator>::Item>,
    {
        let pcm_data: Vec<_> = normalize_audio(input, self.inner.apply_speech_filter);
        TranscriptionTask {
            word_level_time_stamps: false,
            audio: pcm_data,
            whisper: self.clone(),
            receiver: Default::default(),
            language: None,
        }
    }
}

/// A transcription task which can be streamed from a [`Whisper`] model.
pub struct TranscriptionTask {
    word_level_time_stamps: bool,
    audio: Vec<f32>,
    whisper: Whisper,
    receiver: Mutex<Option<UnboundedReceiver<Segment>>>,
    language: Option<WhisperLanguage>,
}

impl TranscriptionTask {
    /// Include word level timestamps in the transcription.
    pub fn timestamped(mut self) -> Self {
        self.word_level_time_stamps = true;
        self
    }

    /// Set the language whisper should output the transcription in. Multi-lingual whisper models can do
    /// translation to the target language.
    pub fn with_language<L>(mut self, language: L) -> Self
    where
        L: Into<WhisperLanguage>,
    {
        self.language = Some(language.into());
        self
    }
}

impl Stream for TranscriptionTask {
    type Item = Segment;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let myself = self.get_mut();
        let mut write = myself.receiver.lock().unwrap();
        if write.is_none() {
            let (sender, receiver) = futures_channel::mpsc::unbounded();
            let pcm_data = std::mem::take(&mut myself.audio);

            myself
                .whisper
                .inner
                .sender
                .unbounded_send(WhisperMessage {
                    samples: pcm_data,
                    timestamps: myself.word_level_time_stamps,
                    lang: myself.language,
                    sender,
                })
                .unwrap();

            *write = Some(receiver);
        }

        // Poll the background task to make progress on transcription
        let mut task = myself.whisper.inner.task.lock().unwrap();
        let _ = task.poll_unpin(cx);

        write.as_mut().unwrap().poll_next_unpin(cx)
    }
}

struct WhisperMessage {
    samples: Vec<f32>,
    timestamps: bool,
    lang: Option<WhisperLanguage>,
    sender: UnboundedSender<Segment>,
}

pub(crate) fn normalize_audio<S: Source>(input: S, apply_speech_filter: bool) -> Vec<f32>
where
    <S as Iterator>::Item: rodio::Sample,
    f32: FromSample<<S as Iterator>::Item>,
{
    let resample = UniformSourceIterator::new(input, 1, SAMPLE_RATE as u32);
    if apply_speech_filter {
        resample
            .low_pass(3000)
            .high_pass(200)
            .convert_samples()
            .collect::<Vec<f32>>()
    } else {
        resample.convert_samples().collect::<Vec<f32>>()
    }
}
