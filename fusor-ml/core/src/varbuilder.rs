use std::{fmt::Debug, sync::Arc};

use crate::{Device, QMatrix};
use fusor_gguf::{GgufMetadata, GgufReadError, GgufValue};

trait ReadAndSeek: std::io::Read + std::io::Seek {}

impl<T: std::io::Read + std::io::Seek + ?Sized> ReadAndSeek for T {}

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

    pub fn get(&mut self, key: &str, device: &Device) -> crate::Result<QMatrix> {
        let full_path = self.format_path(key);
        let q_matrix_metadata = self.metadata.tensor_infos.get(&*full_path).ok_or_else(|| {
            crate::Error::VarBuilder(format!("Key '{}' not found in GGUF metadata", full_path))
        })?;

        let q_matrix = QMatrix::read(
            device,
            q_matrix_metadata,
            &mut self.reader,
            self.metadata.tensor_data_offset,
        )?;

        Ok(q_matrix)
    }

    pub fn contains_key(&self, key: &str) -> bool {
        let full_path = self.format_path(key);
        self.metadata.tensor_infos.contains_key(&*full_path)
    }

    pub fn list_all_keys(&self) -> Vec<String> {
        self.metadata
            .tensor_infos
            .keys()
            .map(|k| k.to_string())
            .collect()
    }

    /// Get a metadata value by key
    ///
    /// If the key starts with '.', it will try to find a key that ends with the given suffix
    /// and return the value with the shortest matching key.
    pub fn get_metadata(&self, name: &str) -> crate::Result<&GgufValue> {
        self.metadata.get_value(name).ok_or_else(|| {
            crate::Error::VarBuilder(format!("Key '{}' not found in GGUF metadata", name))
        })
    }

    /// Get the architecture from GGUF metadata
    pub fn architecture(&self) -> Option<String> {
        self.metadata.architecture()
    }

    /// List all metadata keys (for debugging)
    pub fn list_metadata_keys(&self) -> Vec<String> {
        self.metadata
            .metadata
            .keys()
            .map(|s| s.to_string())
            .collect()
    }
}

pub struct ShardedVarBuilder<R: std::io::Read + std::io::Seek> {
    contents: Vec<(GgufMetadata, R)>,
}

impl<R: std::io::Read + std::io::Seek> ShardedVarBuilder<R> {
    pub fn new(contents: Vec<(GgufMetadata, R)>) -> Self {
        Self { contents }
    }

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

    pub fn tensor(&mut self, name: &str, device: &Device) -> crate::Result<QMatrix> {
        for (content, r) in &mut self.contents {
            if let Some(tensor_info) = content.tensor_infos.get(name) {
                let q_matrix = QMatrix::read(device, tensor_info, r, content.tensor_data_offset)?;
                return Ok(q_matrix);
            }
        }
        Err(crate::Error::VarBuilder(format!(
            "Key '{}' not found in GGUF metadata",
            name
        )))
    }
}
