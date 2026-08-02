//! The synchronous `VarBuilder`: a prefixed view over a GGUF file's tensor
//! directory.
//!
//! `get` returns **bytes**, not a device tensor: this crate has no device and
//! no `Target`. The `fusor2` facade turns a [`RawTensorBytes`] into a
//! `LeafKind::Quantized` leaf and lets extraction decide whether to repack it.
//!
//! Owned by W11.

use fusor2_ir::Result;
use fusor2_ir::dtype::{Dtype, QLayout};
use fusor2_ir::error::Error;
use smallvec::SmallVec;
use std::sync::Arc;

use crate::parse::{Gguf, GgufMetadata, GgufTensor, GgufValue, ingest_qfmt};

/// One tensor as it sits in the file: raw bytes plus enough type information
/// to build a leaf.
///
/// `layout` is always [`QLayout::Native`] here — a GGUF file stores the raw
/// block bytes, and moving to `F32Scales` is the priced `qrepack` rewrite, not
/// something a loader decides.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawTensorBytes {
    pub name: Box<str>,
    pub fmt: Dtype,
    pub layout: QLayout,
    pub shape: SmallVec<[u64; 4]>,
    pub bytes: Box<[u8]>,
}

/// A prefixed cursor into one GGUF file.
#[derive(Clone)]
pub struct VarBuilder {
    file: Arc<Gguf>,
    prefix: String,
}

impl VarBuilder {
    pub fn new(file: Arc<Gguf>) -> Self {
        Self {
            file,
            prefix: String::new(),
        }
    }

    /// Open a file and scope to its root.
    pub fn from_gguf(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Ok(Self::new(Arc::new(Gguf::open(path)?)))
    }

    /// Push a path component: `pp("blk").pp(0)` scopes to `blk.0.`.
    pub fn pp<S: ToString>(&self, component: S) -> Self {
        let component = component.to_string();
        let mut prefix = self.prefix.clone();
        if !prefix.is_empty() {
            prefix.push('.');
        }
        prefix.push_str(&component);
        Self {
            file: Arc::clone(&self.file),
            prefix,
        }
    }

    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// The fully qualified name `name` resolves to under this scope.
    pub fn format_path(&self, name: &str) -> String {
        if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}.{name}", self.prefix)
        }
    }

    pub fn contains_key(&self, name: &str) -> bool {
        self.file.tensor(&self.format_path(name)).is_some()
    }

    /// Every tensor name visible under this scope, with the prefix stripped.
    pub fn list_all_keys(&self) -> Vec<String> {
        if self.prefix.is_empty() {
            return self.file.tensor_names().map(str::to_string).collect();
        }
        let head = format!("{}.", self.prefix);
        self.file
            .tensor_names()
            .filter_map(|n| n.strip_prefix(&head).map(str::to_string))
            .collect()
    }

    pub fn metadata(&self) -> &GgufMetadata {
        self.file.metadata()
    }

    /// Metadata lookup, sharing [`GgufMetadata::get_value`]'s suffix rule.
    pub fn get_metadata(&self, key: &str) -> Option<&GgufValue> {
        self.file.metadata().get_value(key)
    }

    pub fn architecture(&self) -> Option<&str> {
        self.file.metadata().architecture()
    }

    /// The directory entry plus its raw bytes, borrowed from the mapping.
    /// A missing key is an `Err`, never a panic.
    pub fn get(&self, name: &str) -> Result<(&GgufTensor, &[u8])> {
        let path = self.format_path(name);
        let tensor = self
            .file
            .tensor(&path)
            .ok_or_else(|| Error::Io(format!("no tensor named {path}")))?;
        Ok((tensor, self.file.tensor_bytes(&path)?))
    }

    /// The owned form the `fusor2` facade ingests.
    pub fn get_raw(&self, name: &str) -> Result<RawTensorBytes> {
        let (tensor, bytes) = self.get(name)?;
        Ok(RawTensorBytes {
            name: tensor.name.clone().into_boxed_str(),
            fmt: ingest_qfmt(tensor.ty)?,
            layout: QLayout::Native,
            shape: tensor.shape.clone(),
            bytes: bytes.to_vec().into_boxed_slice(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::{GgufVersion, fixture};
    use fusor2_ir::dtype::{GgmlType, QFmt};

    fn model() -> Arc<Gguf> {
        let file = fixture::build(
            GgufVersion::V3,
            &[("general.architecture", GgufValue::String("qwen3".into()))],
            &[
                (
                    "blk.0.attn_q.weight",
                    GgmlType::Q4_0,
                    &[1, 32],
                    (0..18u8).collect(),
                ),
                ("blk.0.attn_q.bias", GgmlType::F32, &[2], vec![1u8; 8]),
                ("output_norm.weight", GgmlType::F16, &[4], vec![2u8; 8]),
            ],
        );
        Arc::new(Gguf::from_bytes(file).unwrap())
    }

    #[test]
    fn varbuilder_prefix_scoping() {
        let vb = VarBuilder::new(model());
        assert_eq!(vb.architecture(), Some("qwen3"));

        let blk = vb.pp("blk").pp(0);
        assert_eq!(blk.prefix(), "blk.0");
        assert_eq!(blk.format_path("attn_q.weight"), "blk.0.attn_q.weight");
        assert!(blk.contains_key("attn_q.weight"));
        assert!(!blk.contains_key("attn_k.weight"));

        let (tensor, bytes) = blk.get("attn_q.weight").unwrap();
        assert_eq!(tensor.name, "blk.0.attn_q.weight");
        assert_eq!(bytes, (0..18u8).collect::<Vec<_>>().as_slice());

        let raw = blk.get_raw("attn_q.weight").unwrap();
        assert_eq!(&*raw.name, "blk.0.attn_q.weight");
        assert_eq!(raw.fmt, Dtype::Q(QFmt::Q4_0));
        assert_eq!(raw.layout, QLayout::Native);
        assert_eq!(raw.shape.as_slice(), &[1, 32]);
        assert_eq!(raw.bytes.len(), 18);

        let mut keys = blk.list_all_keys();
        keys.sort();
        assert_eq!(keys, vec!["attn_q.bias", "attn_q.weight"]);

        let mut all = vb.list_all_keys();
        all.sort();
        assert_eq!(
            all,
            vec![
                "blk.0.attn_q.bias",
                "blk.0.attn_q.weight",
                "output_norm.weight"
            ]
        );

        // A missing key is an error, never a panic.
        assert!(blk.get("nope").is_err());
        assert!(blk.get_raw("nope").is_err());
        assert!(vb.get("blk.1.attn_q.weight").is_err());
    }

    #[test]
    fn metadata_reads_through_the_builder() {
        let vb = VarBuilder::new(model());
        assert_eq!(
            vb.get_metadata("general.architecture")
                .and_then(|v| v.as_str()),
            Some("qwen3")
        );
        assert!(vb.get_metadata("missing").is_none());
        assert_eq!(vb.metadata().tensors.len(), 3);
    }
}
