//! Common types for Kalosm models

use std::{fmt::Display, path::PathBuf};

mod builder;
pub use builder::ModelBuilder;
mod sync;
pub use sync::*;

/// The progress starting a model
#[derive(Clone, Debug)]
pub enum ModelLoadingProgress {
    /// The model is downloading
    Downloading {
        /// The source of the download. This is not a path or URL, but a description of the source
        source: String,
        progress: FileLoadingProgress,
    },
    /// The model is loading
    Loading {
        /// The progress of the loading, from 0 to 1
        progress: f32,
    },
}

/// The progress of a file download
#[derive(Clone, Debug)]
pub struct FileLoadingProgress {
    /// The time stamp the download started. This is None on wasm
    pub start_time: Option<std::time::Instant>,
    /// The size of the cached part of the download in bytes
    pub cached_size: u64,
    /// The size of the download in bytes
    pub size: u64,
    /// The progress of the download in bytes, from 0 to size
    pub progress: u64,
}

#[cfg(feature = "loading-progress-bar")]
#[derive(Default)]
struct LoadingIndicator {
    downloads: std::collections::HashMap<String, FileLoadingProgress>,
    last_render: Option<std::time::Instant>,
    rendered_line: bool,
    last_loading_percent: Option<u32>,
}

#[cfg(feature = "loading-progress-bar")]
impl LoadingIndicator {
    fn update(&mut self, progress: ModelLoadingProgress) {
        match progress {
            ModelLoadingProgress::Downloading { source, progress } => {
                let is_finished = progress.size > 0 && progress.progress >= progress.size;
                self.downloads.insert(source.clone(), progress.clone());

                if is_finished || self.should_render() {
                    self.render_download(&source, &progress);
                }
            }
            ModelLoadingProgress::Loading { progress } => self.render_loading(progress),
        }
    }

    fn should_render(&mut self) -> bool {
        const MIN_RENDER_INTERVAL: std::time::Duration = std::time::Duration::from_millis(100);

        let now = std::time::Instant::now();
        let should_render = match self.last_render {
            Some(last_render) => now.duration_since(last_render) >= MIN_RENDER_INTERVAL,
            None => true,
        };
        if should_render {
            self.last_render = Some(now);
        }
        should_render
    }

    fn render_download(&mut self, source: &str, progress: &FileLoadingProgress) {
        use std::io::Write;

        let percent = percent(progress.progress, progress.size);
        let speed = progress.start_time.and_then(|start_time| {
            bytes_per_second(progress.progress, progress.cached_size, start_time)
        });
        let eta = progress.start_time.and_then(|start_time| {
            estimate_time_remaining(
                progress.progress,
                progress.cached_size,
                progress.size,
                start_time,
            )
        });

        let mut stderr = std::io::stderr().lock();
        let _ = write!(
            stderr,
            "\r\x1b[KDownloading {source} {percent:>6.2}% ({}/{}, {}/s, ETA {})",
            format_bytes(progress.progress),
            format_bytes(progress.size),
            speed.map(format_bytes).unwrap_or_else(|| "-".to_string()),
            eta.map(format_duration).unwrap_or_else(|| "-".to_string()),
        );
        let _ = stderr.flush();
        self.rendered_line = true;
    }

    fn render_loading(&mut self, progress: f32) {
        use std::io::Write;

        if self.rendered_line {
            let _ = writeln!(std::io::stderr().lock());
            self.rendered_line = false;
        }

        for (source, progress) in self.downloads.drain() {
            let _ = writeln!(
                std::io::stderr().lock(),
                "Downloaded {source} ({})",
                format_bytes(progress.size)
            );
        }

        let percent = (progress.clamp(0.0, 1.0) * 100.0).round() as u32;
        if self.last_loading_percent == Some(percent) {
            return;
        }
        self.last_loading_percent = Some(percent);

        let _ = writeln!(std::io::stderr().lock(), "Loading {percent}%");
    }
}

#[cfg(feature = "loading-progress-bar")]
fn percent(progress: u64, size: u64) -> f64 {
    if size == 0 {
        0.0
    } else {
        progress as f64 / size as f64 * 100.0
    }
}

#[cfg(feature = "loading-progress-bar")]
fn bytes_per_second(
    progress: u64,
    cached_size: u64,
    start_time: std::time::Instant,
) -> Option<u64> {
    let elapsed = start_time.elapsed().as_secs_f64();
    if elapsed <= f64::EPSILON {
        None
    } else {
        Some(((progress.saturating_sub(cached_size) as f64) / elapsed) as u64)
    }
}

#[cfg(feature = "loading-progress-bar")]
fn estimate_time_remaining(
    progress: u64,
    cached_size: u64,
    size: u64,
    start_time: std::time::Instant,
) -> Option<std::time::Duration> {
    let bytes_per_second = bytes_per_second(progress, cached_size, start_time)?;
    if bytes_per_second == 0 || progress >= size {
        None
    } else {
        Some(std::time::Duration::from_secs(
            (size - progress) / bytes_per_second,
        ))
    }
}

#[cfg(feature = "loading-progress-bar")]
fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = UNITS[0];

    for next_unit in UNITS.iter().skip(1) {
        if value < 1024.0 {
            break;
        }
        value /= 1024.0;
        unit = next_unit;
    }

    if unit == "B" {
        format!("{bytes} {unit}")
    } else {
        format!("{value:.2} {unit}")
    }
}

#[cfg(feature = "loading-progress-bar")]
fn format_duration(duration: std::time::Duration) -> String {
    let seconds = duration.as_secs();
    let hours = seconds / 3600;
    let minutes = (seconds % 3600) / 60;
    let seconds = seconds % 60;

    if hours > 0 {
        format!("{hours}h {minutes}m {seconds}s")
    } else if minutes > 0 {
        format!("{minutes}m {seconds}s")
    } else {
        format!("{seconds}s")
    }
}

impl ModelLoadingProgress {
    /// Create a new downloading progress
    pub fn downloading(source: String, file_loading_progress: FileLoadingProgress) -> Self {
        Self::Downloading {
            source,
            progress: file_loading_progress,
        }
    }

    /// Create a new downloading progress
    pub fn downloading_progress(
        source: String,
    ) -> impl FnMut(FileLoadingProgress) -> Self + Send + Sync {
        move |progress| ModelLoadingProgress::downloading(source.clone(), progress)
    }

    /// Create a new loading progress
    pub fn loading(progress: f32) -> Self {
        Self::Loading { progress }
    }

    /// Return the percent complete
    pub fn progress(&self) -> f32 {
        match self {
            Self::Downloading {
                progress: FileLoadingProgress { progress, size, .. },
                ..
            } => *progress as f32 / *size as f32,
            Self::Loading { progress } => *progress,
        }
    }

    /// Try to estimate the time remaining for a download
    pub fn estimate_time_remaining(&self) -> Option<std::time::Duration> {
        match self {
            Self::Downloading {
                progress: FileLoadingProgress { start_time, .. },
                ..
            } => {
                let elapsed = start_time.as_ref()?.elapsed();
                let progress = self.progress();
                let remaining = (1. - progress) * elapsed.as_secs_f32();
                Some(std::time::Duration::from_secs_f32(remaining))
            }
            _ => None,
        }
    }

    #[cfg(feature = "loading-progress-bar")]
    /// A default loading progress bar
    pub fn multi_bar_loading_indicator() -> impl FnMut(ModelLoadingProgress) + Send + Sync + 'static
    {
        let mut indicator = LoadingIndicator::default();
        move |progress| indicator.update(progress)
    }
}

/// A source for a file, either from Hugging Face or a local path
#[derive(Clone, Debug)]
pub enum FileSource {
    /// A file from Hugging Face
    HuggingFace {
        /// The model id to use
        model_id: String,
        /// The revision to use
        revision: String,
        /// The file to use
        file: String,
    },
    /// A local file
    Local(PathBuf),
}

impl Display for FileSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FileSource::HuggingFace {
                model_id,
                revision,
                file,
            } => write!(f, "hf://{model_id}/{revision}/{file}"),
            FileSource::Local(path) => write!(f, "{}", path.display()),
        }
    }
}

impl FileSource {
    /// Create a new source for a file from Hugging Face
    pub fn huggingface(
        model_id: impl ToString,
        revision: impl ToString,
        file: impl ToString,
    ) -> Self {
        Self::HuggingFace {
            model_id: model_id.to_string(),
            revision: revision.to_string(),
            file: file.to_string(),
        }
    }

    /// Create a new source for a local file
    pub fn local(path: PathBuf) -> Self {
        Self::Local(path)
    }
}
