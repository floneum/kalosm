//! VarBuilder for loading GGUF model weights to unified CPU/GPU tensors.
//!
//! This module provides a VarBuilder that wraps fusor-core's VarBuilder
//! and creates unified `fusor::QMatrix` tensors that can run on either CPU or GPU.

use std::{fmt::Debug, sync::Arc};

use crate::{Device, QMatrix};
pub use fusor_gguf::{GgufMetadata, GgufReadError, GgufValue};

trait ReadAndSeek: std::io::Read + std::io::Seek {}

impl<T: std::io::Read + std::io::Seek + ?Sized> ReadAndSeek for T {}

/// Calculate the byte size of tensor data for a given type and element count.
fn tensor_byte_size(ty: fusor_gguf::GgmlType, num_elements: usize) -> usize {
    let block_size = ty.block_size();
    let num_blocks = num_elements / block_size;
    num_blocks * ty.block_allocation_size()
}

/// A builder for loading tensors from GGUF files to either CPU or GPU.
///
/// This is the unified variant that creates `fusor::QMatrix` tensors
/// based on the target device.
pub struct VarBuilder<'a> {
    reader: &'a mut dyn ReadAndSeek,
    metadata: Arc<GgufMetadata>,
    path: Vec<String>,
}

impl<'a> Debug for VarBuilder<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VarBuilder")
            .field("path", &self.path)
            .field("metadata", &self.metadata)
            .finish()
    }
}

impl<'a> VarBuilder<'a> {
    /// Create a VarBuilder from a GGUF file reader.
    ///
    /// This reads the GGUF metadata and prepares for tensor loading.
    pub fn from_gguf<R: std::io::Read + std::io::Seek>(
        reader: &'a mut R,
    ) -> Result<Self, GgufReadError> {
        let metadata = GgufMetadata::read(&mut *reader)?.into();
        let path = Default::default();
        Ok(Self {
            reader,
            metadata,
            path,
        })
    }

    /// Create a sub-builder with an additional path prefix.
    ///
    /// This is used to navigate into nested tensor names like "model.encoder.layers.0".
    pub fn pp<'b, S: ToString>(&'b mut self, s: S) -> VarBuilder<'b> {
        let mut new_path = self.path.clone();
        new_path.push(s.to_string());
        VarBuilder {
            reader: &mut *self.reader,
            metadata: self.metadata.clone(),
            path: new_path,
        }
    }

    fn format_path(&self, name: &str) -> String {
        let mut full_path = self.path.join(".");
        if !full_path.is_empty() {
            full_path.push('.');
        }
        full_path.push_str(name);
        full_path
    }

    /// Get a quantized tensor by key, loading it to the specified device.
    ///
    /// The tensor is loaded as a `fusor::QMatrix` which can be either CPU or GPU
    /// depending on the device.
    pub fn get(&mut self, key: &str, device: &Device) -> crate::Result<QMatrix> {
        let full_path = self.format_path(key);
        let q_matrix_metadata = self.metadata.tensor_infos.get(&*full_path).ok_or_else(|| {
            crate::Error::VarBuilder(format!("Key '{}' not found in GGUF metadata", full_path))
        })?;

        // Read the raw bytes and create unified QMatrix
        let tensor_info = q_matrix_metadata;
        let offset = self.metadata.tensor_data_offset + tensor_info.offset;

        // Seek to tensor data
        self.reader
            .seek(std::io::SeekFrom::Start(offset))
            .map_err(|e| {
                crate::Error::VarBuilder(format!("Failed to seek to tensor data: {}", e))
            })?;

        // Calculate size and read bytes
        let ggml_type = tensor_info.ty;
        // Preserve the original shape
        let shape: Box<[usize]> = tensor_info
            .shape
            .iter()
            .map(|&d| d as usize)
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let num_elements: usize = shape.iter().product();
        let byte_size = tensor_byte_size(ggml_type, num_elements);

        let mut bytes = vec![0u8; byte_size];
        self.reader
            .read_exact(&mut bytes)
            .map_err(|e| crate::Error::VarBuilder(format!("Failed to read tensor data: {}", e)))?;

        // Use QMatrix::from_raw_bytes which dispatches to CPU or GPU
        QMatrix::from_raw_bytes(device, shape, &bytes, ggml_type)
            .map_err(|e| crate::Error::VarBuilder(format!("Failed to create QMatrix: {}", e)))
    }

    /// Check if a key exists in the GGUF metadata.
    pub fn contains_key(&self, key: &str) -> bool {
        let full_path = self.format_path(key);
        self.metadata.tensor_infos.contains_key(&*full_path)
    }

    /// List all tensor keys in the GGUF file.
    pub fn list_all_keys(&self) -> Vec<String> {
        self.metadata
            .tensor_infos
            .keys()
            .map(|k| k.to_string())
            .collect()
    }

    /// Get metadata from the GGUF file.
    pub fn metadata(&self) -> &GgufMetadata {
        &self.metadata
    }

    /// Get a metadata value by key.
    ///
    /// If the key starts with `.`, it will search for metadata keys that end with the given key
    /// and return the value for the shortest matching key. This allows architecture-prefixed
    /// keys like `qwen3.attention.head_count` to be found with `.attention.head_count`.
    pub fn get_metadata(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get_value(key)
    }

    /// Get the model architecture from GGUF metadata.
    ///
    /// Reads the "general.architecture" metadata key.
    pub fn architecture(&self) -> Option<String> {
        self.metadata.architecture()
    }
}

/// Sharded VarBuilder for loading from multiple GGUF files.
pub struct ShardedVarBuilder<R: std::io::Read + std::io::Seek> {
    contents: Vec<(GgufMetadata, R)>,
}

impl<R: std::io::Read + std::io::Seek> ShardedVarBuilder<R> {
    /// Create a new sharded VarBuilder from multiple GGUF files.
    pub fn new(contents: Vec<(GgufMetadata, R)>) -> Self {
        Self { contents }
    }

    /// Get a metadata value by key from any shard.
    pub fn get(&self, name: &str) -> crate::Result<&GgufValue> {
        for (content, _) in &self.contents {
            if let Some(value) = content.get_value(name) {
                return Ok(value);
            }
        }
        Err(crate::Error::VarBuilder(format!(
            "Key '{}' not found in GGUF metadata",
            name
        )))
    }

    /// Load a tensor from any shard to the specified device.
    pub fn tensor(&mut self, name: &str, device: &Device) -> crate::Result<QMatrix> {
        for (content, r) in &mut self.contents {
            if let Some(tensor_info) = content.tensor_infos.get(name) {
                let offset = content.tensor_data_offset + tensor_info.offset;

                // Seek to tensor data
                r.seek(std::io::SeekFrom::Start(offset))
                    .map_err(|e| crate::Error::VarBuilder(format!("Failed to seek: {}", e)))?;

                // Calculate size and read bytes
                let ggml_type = tensor_info.ty;
                // Preserve the original shape
                let shape: Box<[usize]> = tensor_info
                    .shape
                    .iter()
                    .map(|&d| d as usize)
                    .collect::<Vec<_>>()
                    .into_boxed_slice();

                let num_elements: usize = shape.iter().product();
                let byte_size = tensor_byte_size(ggml_type, num_elements);

                let mut bytes = vec![0u8; byte_size];
                r.read_exact(&mut bytes)
                    .map_err(|e| crate::Error::VarBuilder(format!("Failed to read: {}", e)))?;

                return QMatrix::from_raw_bytes(device, shape, &bytes, ggml_type).map_err(|e| {
                    crate::Error::VarBuilder(format!("Failed to create QMatrix: {}", e))
                });
            }
        }
        Err(crate::Error::VarBuilder(format!(
            "Key '{}' not found in GGUF metadata",
            name
        )))
    }
}
