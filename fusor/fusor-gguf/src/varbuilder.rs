//! The synchronous `VarBuilder`: a prefixed view over a GGUF file's tensor
//! directory.
//!
//! `get` returns bytes, not a device tensor: this crate has no device and
//! no `Target`. The `fusor` facade turns a [`RawTensorBytes`] into a
//! `LeafKind::Quantized` leaf and lets extraction decide whether to repack it.

use fusor_ir::Result;
use fusor_ir::dtype::{Dtype, QLayout};
use fusor_ir::error::Error;
use smallvec::SmallVec;
use std::sync::Arc;

use crate::parse::{Gguf, GgufMetadata, GgufTensor, GgufValue, ingest_qfmt};

/// One tensor as it sits in the file: raw bytes plus enough type information
/// to build a leaf.
///
/// `layout` is always [`QLayout::Native`] here — a GGUF file stores the raw
/// block bytes; repacking to `F32Scales` happens later via `qrepack`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawTensorBytes {
    pub name: Box<str>,
    pub fmt: Dtype,
    pub layout: QLayout,
    pub shape: SmallVec<[u64; 4]>,
    pub bytes: TensorBytes,
}

/// A contiguous, immutable range retaining its source file or owned bytes.
#[derive(Clone)]
pub struct TensorBytes {
    owner: Arc<dyn AsRef<[u8]> + Send + Sync>,
    range: std::ops::Range<usize>,
}
impl std::ops::Deref for TensorBytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.owner.as_ref().as_ref()[self.range.clone()]
    }
}
impl AsRef<[u8]> for TensorBytes {
    fn as_ref(&self) -> &[u8] {
        self
    }
}
impl std::fmt::Debug for TensorBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorBytes")
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}
impl PartialEq for TensorBytes {
    fn eq(&self, other: &Self) -> bool {
        self.as_ref() == other.as_ref()
    }
}
impl Eq for TensorBytes {}
impl From<Box<[u8]>> for TensorBytes {
    fn from(bytes: Box<[u8]>) -> Self {
        let range = 0..bytes.len();
        Self {
            owner: Arc::new(bytes),
            range,
        }
    }
}
impl TensorBytes {
    /// Keep this range alive in a device leaf without copying tensor data.
    pub fn shared_host_bytes(&self) -> Arc<fusor_ir::target::HostBytes> {
        Arc::new(
            fusor_ir::target::HostBytes::shared(Arc::new(self.clone()), 0..self.len())
                .expect("range spans this immutable tensor"),
        )
    }
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

    /// The owned form the `fusor` facade ingests.
    pub fn get_raw(&self, name: &str) -> Result<RawTensorBytes> {
        let (tensor, bytes) = self.get(name)?;
        Ok(RawTensorBytes {
            name: tensor.name.clone().into_boxed_str(),
            fmt: ingest_qfmt(tensor.ty)?,
            layout: QLayout::Native,
            shape: tensor.shape.clone(),
            bytes: {
                // get() already validated the tensor range. Derive the offset
                // from that exact borrowed slice, avoiding duplicate arithmetic.
                let start = bytes.as_ptr() as usize - self.file.bytes().as_ptr() as usize;
                TensorBytes {
                    owner: self.file.clone(),
                    range: start..start + bytes.len(),
                }
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::GgmlType;
    use crate::parse::{GgufVersion, fixture};
    use fusor_ir::dtype::QFmt;

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
    fn raw_tensor_and_upload_retain_their_own_shard_without_copying() {
        let first = model();
        let second = model();
        let first_ptr = first.tensor_bytes("blk.0.attn_q.weight").unwrap().as_ptr();
        let second_ptr = second.tensor_bytes("blk.0.attn_q.weight").unwrap().as_ptr();
        assert_ne!(first_ptr, second_ptr);
        let weak_first = Arc::downgrade(&first);
        let weak_second = Arc::downgrade(&second);
        let raw = VarBuilder::new(first)
            .get_raw("blk.0.attn_q.weight")
            .unwrap();
        let other = VarBuilder::new(second)
            .get_raw("blk.0.attn_q.weight")
            .unwrap();
        assert_eq!(raw.bytes.as_ptr(), first_ptr);
        assert_eq!(other.bytes.as_ptr(), second_ptr);
        let upload = raw.bytes.shared_host_bytes();
        drop(raw);
        drop(other);
        assert!(weak_second.upgrade().is_none());
        assert!(weak_first.upgrade().is_some());
        upload.with_slice(|bytes| {
            assert_eq!(bytes.as_ptr(), first_ptr);
            assert_eq!(bytes, (0..18u8).collect::<Vec<_>>());
        });
        drop(upload);
        assert!(weak_first.upgrade().is_none());
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
}
