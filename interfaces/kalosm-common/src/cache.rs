use kalosm_model_types::{FileLoadingProgress, FileSource};
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CacheError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[cfg(target_arch = "wasm32")]
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[cfg(not(target_arch = "wasm32"))]
    #[error("HTTP error: {0}")]
    Http(#[from] Box<ureq::Error>),
    #[error("Unexpected status code: {0}")]
    UnexpectedStatusCode(u16),
    #[cfg(target_arch = "wasm32")]
    #[error("OPFS not available: {0}")]
    OpfsNotAvailable(String),
    #[cfg(target_arch = "wasm32")]
    #[error("OPFS operation failed: {0}")]
    OpfsError(String),
}

#[derive(Debug, Clone)]
pub struct Cache {
    location: PathBuf,
    /// The huggingface token to use (defaults to the token set with `huggingface-cli login`)
    huggingface_token: Option<String>,
}

impl Cache {
    /// Create a new cache with a specific location
    pub fn new(location: PathBuf) -> Self {
        Self {
            location,
            huggingface_token: None,
        }
    }

    /// Set the Hugging Face token to use for downloading (defaults to the token set with `huggingface-cli login`, and then the environment variable `HF_TOKEN`)
    pub fn with_huggingface_token(mut self, token: Option<String>) -> Self {
        self.huggingface_token = token;
        self
    }

    /// Check if the file exists locally (if it is a local file or if it has been downloaded)
    pub fn exists(&self, source: &FileSource) -> bool {
        match source {
            FileSource::HuggingFace {
                model_id,
                revision,
                file,
                ..
            } => {
                let path = self.location.join(model_id).join(revision);
                let complete_download = path.join(file);
                complete_download.exists()
            }
            FileSource::Local(path) => path.exists(),
        }
    }

    /// Get the bytes from the cache, downloading it if necessary
    pub async fn get_bytes(
        &self,
        source: &FileSource,
        #[cfg_attr(not(target_arch = "wasm32"), allow(unused_mut))] mut progress: impl FnMut(
            FileLoadingProgress,
        ),
    ) -> Result<Vec<u8>, CacheError> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let cache = self.clone();
            let source = source.clone();
            run_on_download_thread(
                move |progress| cache.get_bytes_blocking(&source, progress),
                progress,
            )
            .await
        }
        #[cfg(target_arch = "wasm32")]
        {
            use crate::opfs::is_opfs_available;

            // Try OPFS-backed caching first
            if is_opfs_available().await {
                match self.get_bytes_opfs(source, &mut progress).await {
                    Ok(bytes) => return Ok(bytes),
                    Err(e) => {
                        tracing::warn!("OPFS caching failed, falling back to in-memory: {}", e);
                    }
                }
            }

            // Fallback to in-memory streaming (no caching)
            self.get_bytes_memory(source, progress).await
        }
    }

    /// Get the file from the cache, downloading it if necessary
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn get(
        &self,
        source: &FileSource,
        progress: impl FnMut(FileLoadingProgress),
    ) -> Result<PathBuf, CacheError> {
        let cache = self.clone();
        let source = source.clone();
        run_on_download_thread(
            move |progress| cache.get_path_blocking(&source, progress),
            progress,
        )
        .await
    }

    /// WASM: Get bytes using OPFS for persistent caching
    ///
    /// Uses file size comparison with Content-Length to determine if a file is complete.
    /// No .partial files - writes directly to the final filename.
    #[cfg(target_arch = "wasm32")]
    async fn get_bytes_opfs(
        &self,
        source: &FileSource,
        progress: &mut impl FnMut(FileLoadingProgress),
    ) -> Result<Vec<u8>, CacheError> {
        use crate::opfs::{
            close_writable_stream, seek_writable_stream, write_chunk_to_stream, OpfsCache,
        };
        use futures_util::StreamExt;
        use reqwest::StatusCode;

        match source {
            FileSource::HuggingFace {
                model_id,
                revision,
                file,
            } => {
                let opfs = OpfsCache::new().await?;
                let cache_dir = opfs
                    .get_directory(&["kalosm", "cache", model_id, revision])
                    .await?;
                let safe_file = crate::opfs::sanitize_name(file);

                let token = self.huggingface_token.clone().or_else(huggingface_token);
                let url = format!("https://huggingface.co/{model_id}/resolve/{revision}/{file}");
                let client = reqwest::Client::new();

                // 1. HEAD request to get expected Content-Length
                let head_response = client
                    .head(&url)
                    .with_authorization_header(token.clone())
                    .send()
                    .await?;

                let expected_size = head_response
                    .headers()
                    .get(reqwest::header::CONTENT_LENGTH)
                    .and_then(|h| h.to_str().ok())
                    .and_then(|s| s.parse::<u64>().ok());

                // 2. Check local file size
                let local_size = opfs
                    .get_file_size(&cache_dir, &safe_file)
                    .await
                    .unwrap_or(0);

                // 3. Determine action based on size comparison
                if let Some(expected) = expected_size {
                    if local_size == expected {
                        // Cache hit - file is complete
                        let bytes = opfs.read_file(&cache_dir, &safe_file).await?;
                        progress(FileLoadingProgress {
                            progress: local_size,
                            cached_size: local_size,
                            size: local_size,
                            start_time: None,
                        });
                        return Ok(bytes);
                    } else if local_size > expected {
                        // File is corrupted (larger than expected), delete and start fresh
                        let _ = opfs.delete_file(&cache_dir, &safe_file).await;
                    }
                }

                // 4. Calculate start offset for resume
                let start_offset = if expected_size.map_or(false, |e| local_size > e) {
                    0 // We deleted the file above
                } else {
                    local_size
                };

                // 5. Resolve redirects (HuggingFace returns 302, Range headers get stripped)
                let final_url = head_response.url().clone();

                // 6. Send GET request with Range header if resuming
                let mut request = client.get(final_url.clone());
                if start_offset > 0 {
                    request =
                        request.header(reqwest::header::RANGE, format!("bytes={}-", start_offset));
                }

                let mut response = request.send().await?;
                let mut status = response.status();

                // Handle 416 Range Not Satisfiable
                if status == StatusCode::RANGE_NOT_SATISFIABLE {
                    if local_size > 0 {
                        let bytes = opfs.read_file(&cache_dir, &safe_file).await?;
                        progress(FileLoadingProgress {
                            progress: local_size,
                            cached_size: local_size,
                            size: local_size,
                            start_time: None,
                        });
                        return Ok(bytes);
                    }
                    // local_size is 0 but we got 416 - something is wrong, restart fresh
                    response = client
                        .get(final_url)
                        .with_authorization_header(token.clone())
                        .send()
                        .await?;
                    status = response.status();
                }

                // Note: In WASM, Content-Length from GET responses may not be accessible due to CORS
                // (servers must include it in Access-Control-Expose-Headers). We use expected_size
                // from the HEAD request as a fallback since it's more reliably available.
                let (total_size, resuming) = if status == StatusCode::PARTIAL_CONTENT {
                    let remaining = response
                        .headers()
                        .get(reqwest::header::CONTENT_LENGTH)
                        .and_then(|h| h.to_str().ok())
                        .and_then(|s| s.parse::<u64>().ok());
                    (remaining.map(|r| r + start_offset).or(expected_size), true)
                } else if status == StatusCode::OK {
                    let total = response
                        .headers()
                        .get(reqwest::header::CONTENT_LENGTH)
                        .and_then(|h| h.to_str().ok())
                        .and_then(|s| s.parse::<u64>().ok());
                    (total.or(expected_size), false)
                } else {
                    return Err(CacheError::UnexpectedStatusCode(status.as_u16()));
                };

                let actual_start = if resuming { start_offset } else { 0 };

                if let Some(size) = total_size {
                    progress(FileLoadingProgress {
                        progress: actual_start,
                        cached_size: actual_start,
                        size,
                        start_time: None,
                    });

                    // Already complete
                    if actual_start == size {
                        return opfs.read_file(&cache_dir, &safe_file).await;
                    }
                }

                // 7. If resuming, try to read existing data first
                let mut all_bytes = if resuming && actual_start > 0 {
                    match opfs.read_file(&cache_dir, &safe_file).await {
                        Ok(existing) => existing,
                        Err(e) => {
                            // Can't read existing file - delete it and return error
                            // The next call will start fresh
                            tracing::warn!(
                                "[OPFS] Can't read existing file for resume: {}, deleting",
                                e
                            );
                            let _ = opfs.delete_file(&cache_dir, &safe_file).await;
                            return Err(CacheError::OpfsError(format!(
                                "Failed to read partial download for resume: {}. File deleted, please retry.",
                                e
                            )));
                        }
                    }
                } else {
                    Vec::new()
                };

                // 8. Create writable stream and download
                let mut writable = opfs
                    .create_writable(&cache_dir, &safe_file, resuming)
                    .await?;

                if resuming && actual_start > 0 {
                    seek_writable_stream(&writable, actual_start).await?;
                }

                let mut current_progress = actual_start;
                let mut stream = response.bytes_stream();

                // Flush every 100MB
                const FLUSH_INTERVAL: u64 = 100 * 1024 * 1024;
                let mut bytes_since_flush: u64 = 0;

                while let Some(chunk_result) = stream.next().await {
                    let chunk = chunk_result?;
                    write_chunk_to_stream(&writable, &chunk).await?;
                    all_bytes.extend_from_slice(&chunk);

                    current_progress += chunk.len() as u64;
                    bytes_since_flush += chunk.len() as u64;

                    if let Some(size) = total_size {
                        progress(FileLoadingProgress {
                            progress: current_progress,
                            cached_size: actual_start,
                            size,
                            start_time: None,
                        });
                    }

                    // Periodic flush by closing and reopening
                    if bytes_since_flush >= FLUSH_INTERVAL {
                        close_writable_stream(&writable).await?;
                        writable = opfs.create_writable(&cache_dir, &safe_file, true).await?;
                        seek_writable_stream(&writable, current_progress).await?;
                        bytes_since_flush = 0;
                    }
                }

                close_writable_stream(&writable).await?;

                Ok(all_bytes)
            }
            FileSource::Local(_) => Err(CacheError::Io(std::io::Error::other(
                "Local file access not supported on WASM",
            ))),
        }
    }

    /// Fallback: Get bytes in-memory without caching (original WASM behavior)
    #[cfg(target_arch = "wasm32")]
    async fn get_bytes_memory(
        &self,
        source: &FileSource,
        mut progress: impl FnMut(FileLoadingProgress),
    ) -> Result<Vec<u8>, CacheError> {
        use reqwest::StatusCode;
        use std::str::FromStr;

        match source {
            FileSource::HuggingFace {
                model_id,
                revision,
                file,
            } => {
                let token = self.huggingface_token.clone().or_else(huggingface_token);
                let url = format!("https://huggingface.co/{model_id}/resolve/{revision}/{file}");
                let client = reqwest::Client::new();
                let head = client
                    .head(&url)
                    .with_authorization_header(token.clone())
                    .send()
                    .await
                    .map_err(CacheError::from)?;
                let length = head
                    .headers()
                    .get(reqwest::header::CONTENT_LENGTH)
                    .and_then(|length| length.to_str().ok().and_then(|s| u64::from_str(s).ok()));

                if let Some(length) = length {
                    progress(FileLoadingProgress {
                        progress: 0,
                        cached_size: 0,
                        size: length,
                        start_time: None,
                    });
                }

                let request = client.get(url).with_authorization_header(token);
                let response = request.send().await?;

                let status = response.status();
                if !(status == StatusCode::OK || status == StatusCode::PARTIAL_CONTENT) {
                    return Err(CacheError::UnexpectedStatusCode(status.as_u16()));
                }

                let mut current_progress = 0;
                let mut bytes = Vec::new();

                #[cfg(target_arch = "wasm32")]
                {
                    use futures_util::StreamExt;

                    let mut stream = response.bytes_stream();
                    while let Some(chunk) = stream.next().await {
                        let chunk = chunk?;
                        bytes.extend_from_slice(&chunk);
                        tracing::trace!("wrote chunk of size {}", chunk.len());
                        current_progress += chunk.len() as u64;
                        if let Some(length) = length {
                            progress(FileLoadingProgress {
                                progress: current_progress,
                                cached_size: 0,
                                size: length,
                                start_time: None,
                            });
                        }
                    }
                }

                Ok(bytes)
            }
            _ => Err(CacheError::Io(std::io::Error::other(
                "Local file access not supported on WASM",
            ))),
        }
    }

    /// Check if the file exists in the cache (async version for WASM)
    #[cfg(target_arch = "wasm32")]
    pub async fn exists_async(&self, source: &FileSource) -> bool {
        use crate::opfs::{is_opfs_available, sanitize_name, OpfsCache};

        match source {
            FileSource::HuggingFace {
                model_id,
                revision,
                file,
                ..
            } => {
                if !is_opfs_available().await {
                    return false;
                }

                let Ok(opfs) = OpfsCache::new().await else {
                    return false;
                };

                let Ok(cache_dir) = opfs
                    .get_directory(&["kalosm", "cache", model_id, revision])
                    .await
                else {
                    return false;
                };

                let safe_file = sanitize_name(file);

                // Check if file exists and has non-zero size
                opfs.get_file_size(&cache_dir, &safe_file)
                    .await
                    .map_or(false, |size| size > 0)
            }
            FileSource::Local(_) => false, // Local files not supported on WASM
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Cache {
    fn get_bytes_blocking(
        &self,
        source: &FileSource,
        progress: &mut dyn FnMut(FileLoadingProgress),
    ) -> Result<Vec<u8>, CacheError> {
        let path = self.get_path_blocking(source, progress)?;
        std::fs::read(path).map_err(CacheError::from)
    }

    fn get_path_blocking(
        &self,
        source: &FileSource,
        progress: &mut dyn FnMut(FileLoadingProgress),
    ) -> Result<PathBuf, CacheError> {
        match source {
            FileSource::Local(path) => Ok(path.clone()),
            FileSource::HuggingFace {
                model_id,
                revision,
                file,
            } => {
                let cache_dir = self.location.join(model_id).join(revision);
                let complete_download = cache_dir.join(file);
                if std::env::var_os("KALOSM_CACHE_OFFLINE").is_some() && complete_download.exists()
                {
                    return Ok(complete_download);
                }

                let token = self.huggingface_token.clone().or_else(huggingface_token);
                let url = huggingface_resolve_url(model_id, revision, file);
                let agent = ureq::Agent::new_with_defaults();
                let head = with_ureq_auth(agent.head(&url), token.clone())
                    .call()
                    .map_err(ureq_error);

                if is_file_current_blocking(&complete_download, head.as_ref()) {
                    return Ok(complete_download);
                }

                let head = head?;
                std::fs::create_dir_all(&cache_dir)?;
                if let Some(parent) = complete_download.parent() {
                    std::fs::create_dir_all(parent)?;
                }

                let lock_path = complete_download.with_file_name(format!(
                    "{}.lock",
                    complete_download
                        .file_name()
                        .map(|name| name.to_string_lossy())
                        .unwrap_or_else(|| "download".into())
                ));
                if let Some(parent) = lock_path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                let lock_file = std::fs::File::create(&lock_path)?;
                lock_file.lock()?;

                if is_file_current_blocking(&complete_download, Ok(&head)) {
                    drop(lock_file);
                    let _ = std::fs::remove_file(lock_path);
                    return Ok(complete_download);
                }

                let incomplete_download = partial_download_path(&complete_download);
                let download_result = download_into_blocking(
                    &agent,
                    &url,
                    &incomplete_download,
                    &head,
                    token,
                    progress,
                );

                match download_result {
                    Ok(()) => {
                        if let Some(parent) = complete_download.parent() {
                            std::fs::create_dir_all(parent)?;
                        }
                        std::fs::rename(&incomplete_download, &complete_download)?;
                        drop(lock_file);
                        let _ = std::fs::remove_file(&lock_path);
                        Ok(complete_download)
                    }
                    Err(err) => {
                        drop(lock_file);
                        let _ = std::fs::remove_file(&lock_path);
                        Err(err)
                    }
                }
            }
        }
    }
}

#[allow(clippy::derivable_impls)]
impl Default for Cache {
    fn default() -> Self {
        Self {
            location: {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    // Try various locations in order of preference
                    dirs::data_dir()
                        .or_else(dirs::cache_dir)
                        .or_else(|| std::env::var("HOME").ok().map(std::path::PathBuf::from))
                        .or_else(|| std::env::current_dir().ok())
                        .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
                        .join("kalosm")
                        .join("cache")
                }
                #[cfg(target_arch = "wasm32")]
                {
                    Default::default()
                }
            },
            huggingface_token: None,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
enum DownloadThreadEvent<T> {
    Progress(FileLoadingProgress),
    Done(Result<T, CacheError>),
}

#[cfg(not(target_arch = "wasm32"))]
async fn run_on_download_thread<T, F>(
    task: F,
    mut progress: impl FnMut(FileLoadingProgress),
) -> Result<T, CacheError>
where
    T: Send + 'static,
    F: FnOnce(&mut dyn FnMut(FileLoadingProgress)) -> Result<T, CacheError> + Send + 'static,
{
    use futures_channel::mpsc;
    use futures_util::StreamExt;

    let (events_tx, mut events_rx) = mpsc::unbounded();
    std::thread::spawn(move || {
        let progress_tx = events_tx.clone();
        let mut thread_progress = move |progress| {
            let _ = progress_tx.unbounded_send(DownloadThreadEvent::Progress(progress));
        };
        let result = task(&mut thread_progress);
        let _ = events_tx.unbounded_send(DownloadThreadEvent::Done(result));
    });

    while let Some(event) = events_rx.next().await {
        match event {
            DownloadThreadEvent::Progress(update) => progress(update),
            DownloadThreadEvent::Done(result) => return result,
        }
    }

    Err(CacheError::Io(std::io::Error::other(
        "download thread terminated without returning a result",
    )))
}

/// Check if the local file exists and is up-to-date compared to the server's Last-Modified header.
/// Returns true if the file can be used as-is, false if it needs to be downloaded.
#[cfg(not(target_arch = "wasm32"))]
fn is_file_current_blocking(
    path: &Path,
    response: Result<&ureq::http::Response<ureq::Body>, &CacheError>,
) -> bool {
    if !path.exists() {
        return false;
    }

    let Ok(metadata) = std::fs::metadata(path) else {
        return false;
    };

    let Ok(file_last_modified) = metadata.modified() else {
        return false;
    };

    // If the server says the file hasn't been modified since we downloaded it, we can use the local file.
    if let Some(last_updated) = response
        .ok()
        .and_then(|response| response.headers().get(ureq::http::header::LAST_MODIFIED))
        .and_then(|last_updated| last_updated.to_str().ok())
        .and_then(|s| httpdate::parse_http_date(s).ok())
    {
        last_updated <= file_last_modified
    } else {
        // If we're offline or the server doesn't provide Last-Modified, use the local file.
        true
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn download_into_blocking(
    agent: &ureq::Agent,
    url: &str,
    file: &Path,
    head: &ureq::http::Response<ureq::Body>,
    token: Option<String>,
    progress: &mut dyn FnMut(FileLoadingProgress),
) -> Result<(), CacheError> {
    use std::io::{Read, Write};
    use ureq::http::header::{CONTENT_LENGTH, RANGE};
    use ureq::http::StatusCode;

    let length = head
        .headers()
        .get(CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());

    if let Some(parent) = file.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut start = std::fs::metadata(file)
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    if length.is_some_and(|length| start > length) {
        let _ = std::fs::remove_file(file);
        start = 0;
    }

    let start_time = Some(std::time::Instant::now());
    if let Some(length) = length {
        progress(FileLoadingProgress {
            progress: start,
            cached_size: start,
            size: length,
            start_time,
        });
    }

    if Some(start) == length {
        tracing::trace!("File {} already downloaded", file.display());
        progress(FileLoadingProgress {
            progress: start,
            cached_size: start,
            size: length.unwrap_or(0),
            start_time,
        });
        return Ok(());
    }

    let mut request = with_ureq_auth(agent.get(url), token);
    if start > 0 {
        let range = if let Some(length) = length {
            format!("bytes={}-{}", start, length - 1)
        } else {
            format!("bytes={start}-")
        };
        tracing::trace!("Fetching range {range}");
        request = request.header(RANGE, range);
    }

    let mut response = request.call().map_err(ureq_error)?;
    let status = response.status();
    if !(status == StatusCode::OK || status == StatusCode::PARTIAL_CONTENT) {
        return Err(CacheError::UnexpectedStatusCode(status.as_u16()));
    }

    let mut output_file = if start > 0 && status == StatusCode::OK {
        start = 0;
        if let Some(length) = length {
            progress(FileLoadingProgress {
                progress: 0,
                cached_size: 0,
                size: length,
                start_time,
            });
        }
        std::fs::File::create(file)?
    } else {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(file)?
    };

    let mut current_progress = start;
    let cached_size = start;
    let mut buffer = [0; 64 * 1024];
    let mut reader = response.body_mut().as_reader();
    loop {
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }

        output_file.write_all(&buffer[..bytes_read])?;
        tracing::trace!("wrote chunk of size {bytes_read}");
        current_progress += bytes_read as u64;
        if let Some(length) = length {
            progress(FileLoadingProgress {
                progress: current_progress,
                cached_size,
                size: length,
                start_time,
            });
        }
    }

    tracing::trace!("Download of {} complete", file.display());

    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn partial_download_path(path: &Path) -> PathBuf {
    let mut partial = path.to_path_buf();
    let extension = path
        .extension()
        .map(|extension| {
            let mut extension = extension.to_os_string();
            extension.push(".partial");
            extension
        })
        .unwrap_or_else(|| "partial".into());
    partial.set_extension(extension);
    partial
}

#[cfg(not(target_arch = "wasm32"))]
fn with_ureq_auth(
    request: ureq::RequestBuilder<ureq::typestate::WithoutBody>,
    token: Option<String>,
) -> ureq::RequestBuilder<ureq::typestate::WithoutBody> {
    if let Some(token) = token {
        request.header(ureq::http::header::AUTHORIZATION, format!("Bearer {token}"))
    } else {
        request
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn ureq_error(err: ureq::Error) -> CacheError {
    match err {
        ureq::Error::StatusCode(status) => CacheError::UnexpectedStatusCode(status),
        err => CacheError::Http(Box::new(err)),
    }
}

#[cfg(target_arch = "wasm32")]
trait RequestBuilderExt {
    fn with_authorization_header(self, token: Option<String>) -> Self;
}

#[cfg(target_arch = "wasm32")]
impl RequestBuilderExt for reqwest::RequestBuilder {
    fn with_authorization_header(self, token: Option<String>) -> Self {
        if let Some(token) = token {
            self.header(reqwest::header::AUTHORIZATION, format!("Bearer {token}"))
        } else {
            self
        }
    }
}

fn huggingface_token() -> Option<String> {
    if cfg!(target_arch = "wasm32") {
        return None;
    }

    if let Ok(token) = std::env::var("HF_TOKEN") {
        let token = token.trim().to_string();
        if !token.is_empty() {
            return Some(token);
        }
    }

    let token_path = std::env::var_os("HF_HOME")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(".cache").join("huggingface")))
        .map(|path| path.join("token"))?;

    let token = std::fs::read_to_string(token_path).ok()?;
    let token = token.trim().to_string();
    (!token.is_empty()).then_some(token)
}

#[cfg(not(target_arch = "wasm32"))]
fn huggingface_resolve_url(model_id: &str, revision: &str, file: &str) -> String {
    format!("https://huggingface.co/{model_id}/resolve/{revision}/{file}")
}
