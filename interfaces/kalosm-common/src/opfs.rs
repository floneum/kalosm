//! OPFS (Origin Private File System) support for WASM targets
//!
//! This module provides persistent file caching in browser environments using the
//! Origin Private File System API.

use crate::CacheError;
use js_sys::Uint8Array;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    File, FileSystemCreateWritableOptions, FileSystemDirectoryHandle, FileSystemFileHandle,
    FileSystemGetDirectoryOptions, FileSystemGetFileOptions, FileSystemWritableFileStream,
};

/// Sanitize a name for OPFS compatibility
///
/// OPFS has restrictions on file/directory names:
/// - Cannot be "." or ".."
/// - Cannot contain "/" or "\" or null characters
/// - Some browsers may have additional restrictions
pub fn sanitize_name(name: &str) -> String {
    if name == "." || name == ".." {
        return format!("_{}_", name.len());
    }

    name.chars()
        .map(|c| match c {
            '/' | '\\' | '\0' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            _ => c,
        })
        .collect()
}

/// Check if OPFS is available in the current browser environment
pub async fn is_opfs_available() -> bool {
    let Some(window) = web_sys::window() else {
        return false;
    };

    let navigator = window.navigator();
    let storage = navigator.storage();

    match JsFuture::from(storage.get_directory()).await {
        Ok(_) => true,
        Err(_) => false,
    }
}

/// A wrapper around OPFS operations for caching files
pub struct OpfsCache {
    root: FileSystemDirectoryHandle,
}

impl OpfsCache {
    /// Initialize OPFS access by getting the root directory handle
    pub async fn new() -> Result<Self, CacheError> {
        let window = web_sys::window()
            .ok_or_else(|| CacheError::OpfsNotAvailable("No window object".to_string()))?;

        let navigator = window.navigator();
        let storage = navigator.storage();

        let root: FileSystemDirectoryHandle = JsFuture::from(storage.get_directory())
            .await
            .map_err(|e| CacheError::OpfsNotAvailable(format!("{:?}", e)))?
            .unchecked_into();

        Ok(Self { root })
    }

    /// Get or create a nested directory structure
    ///
    /// For example, `get_directory(&["kalosm", "cache", "model_id", "revision"])`
    /// will create/get the path `kalosm/cache/model_id/revision/`
    ///
    /// Note: Path segments containing `/` will be split into multiple directories,
    /// and other invalid characters will be sanitized.
    pub async fn get_directory(
        &self,
        path: &[&str],
    ) -> Result<FileSystemDirectoryHandle, CacheError> {
        let mut current = self.root.clone();

        for segment in path {
            // Split segments that contain '/' into multiple directories
            // e.g., "Qwen/Qwen2.5-0.5B" becomes ["Qwen", "Qwen2.5-0.5B"]
            let subsegments: Vec<&str> = segment.split('/').filter(|s| !s.is_empty()).collect();

            for subsegment in subsegments {
                // Sanitize the segment name for OPFS compatibility
                let sanitized = sanitize_name(subsegment);

                let options = FileSystemGetDirectoryOptions::new();
                options.set_create(true);

                current =
                    JsFuture::from(current.get_directory_handle_with_options(&sanitized, &options))
                        .await
                        .map_err(|e| {
                            CacheError::OpfsError(format!(
                                "Failed to get directory '{}': {:?}",
                                sanitized, e
                            ))
                        })?
                        .unchecked_into();
            }
        }

        Ok(current)
    }

    /// Check if a file exists in the given directory
    pub async fn file_exists(&self, dir: &FileSystemDirectoryHandle, name: &str) -> bool {
        let options = FileSystemGetFileOptions::new();
        options.set_create(false);

        JsFuture::from(dir.get_file_handle_with_options(name, &options))
            .await
            .is_ok()
    }

    /// Get the size of a file if it exists
    pub async fn get_file_size(&self, dir: &FileSystemDirectoryHandle, name: &str) -> Option<u64> {
        let options = FileSystemGetFileOptions::new();
        options.set_create(false);

        let file_handle: FileSystemFileHandle =
            JsFuture::from(dir.get_file_handle_with_options(name, &options))
                .await
                .ok()?
                .unchecked_into();

        let file: File = JsFuture::from(file_handle.get_file())
            .await
            .ok()?
            .unchecked_into();

        Some(file.size() as u64)
    }

    /// Read a file's contents as bytes
    pub async fn read_file(
        &self,
        dir: &FileSystemDirectoryHandle,
        name: &str,
    ) -> Result<Vec<u8>, CacheError> {
        let options = FileSystemGetFileOptions::new();
        options.set_create(false);

        let file_handle: FileSystemFileHandle =
            JsFuture::from(dir.get_file_handle_with_options(name, &options))
                .await
                .map_err(|e| {
                    CacheError::OpfsError(format!("Failed to get file handle '{}': {:?}", name, e))
                })?
                .unchecked_into();

        let file: File = JsFuture::from(file_handle.get_file())
            .await
            .map_err(|e| CacheError::OpfsError(format!("Failed to get file '{}': {:?}", name, e)))?
            .unchecked_into();

        let array_buffer = JsFuture::from(file.array_buffer()).await.map_err(|e| {
            CacheError::OpfsError(format!("Failed to read file '{}': {:?}", name, e))
        })?;

        let uint8_array = Uint8Array::new(&array_buffer);
        Ok(uint8_array.to_vec())
    }

    /// Create a writable stream for a file (for streaming writes)
    ///
    /// If `keep_existing` is true, existing file data is preserved and writes
    /// will need to seek to the appropriate position.
    pub async fn create_writable(
        &self,
        dir: &FileSystemDirectoryHandle,
        name: &str,
        keep_existing: bool,
    ) -> Result<FileSystemWritableFileStream, CacheError> {
        let options = FileSystemGetFileOptions::new();
        options.set_create(true);

        let file_handle: FileSystemFileHandle =
            JsFuture::from(dir.get_file_handle_with_options(name, &options))
                .await
                .map_err(|e| {
                    CacheError::OpfsError(format!("Failed to create file '{}': {:?}", name, e))
                })?
                .unchecked_into();

        let write_options = FileSystemCreateWritableOptions::new();
        write_options.set_keep_existing_data(keep_existing);

        let writable: FileSystemWritableFileStream =
            JsFuture::from(file_handle.create_writable_with_options(&write_options))
                .await
                .map_err(|e| {
                    CacheError::OpfsError(format!(
                        "Failed to create writable stream for '{}': {:?}",
                        name, e
                    ))
                })?
                .unchecked_into();

        Ok(writable)
    }

    /// Delete a file from the directory
    pub async fn delete_file(
        &self,
        dir: &FileSystemDirectoryHandle,
        name: &str,
    ) -> Result<(), CacheError> {
        JsFuture::from(dir.remove_entry(name)).await.map_err(|e| {
            CacheError::OpfsError(format!("Failed to delete file '{}': {:?}", name, e))
        })?;

        Ok(())
    }
}

/// Seek to a position in an OPFS writable stream
pub async fn seek_writable_stream(
    writable: &FileSystemWritableFileStream,
    position: u64,
) -> Result<(), CacheError> {
    JsFuture::from(
        writable
            .seek_with_f64(position as f64)
            .map_err(|e| CacheError::OpfsError(format!("Failed to seek: {:?}", e)))?,
    )
    .await
    .map_err(|e| CacheError::OpfsError(format!("Seek failed: {:?}", e)))?;

    Ok(())
}

/// Write a chunk of data to an OPFS writable stream
pub async fn write_chunk_to_stream(
    writable: &FileSystemWritableFileStream,
    chunk: &[u8],
) -> Result<(), CacheError> {
    JsFuture::from(
        writable
            .write_with_u8_array(chunk)
            .map_err(|e| CacheError::OpfsError(format!("Failed to write chunk: {:?}", e)))?,
    )
    .await
    .map_err(|e| CacheError::OpfsError(format!("Write failed: {:?}", e)))?;

    Ok(())
}

/// Close an OPFS writable stream
pub async fn close_writable_stream(
    writable: &FileSystemWritableFileStream,
) -> Result<(), CacheError> {
    JsFuture::from(writable.close())
        .await
        .map_err(|e| CacheError::OpfsError(format!("Failed to close stream: {:?}", e)))?;

    Ok(())
}
