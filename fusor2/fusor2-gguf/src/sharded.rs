//! Sharded VarBuilders: one logical model spread across several GGUF files.
//!
//! Lookup is first-match in shard order, which is stable across runs, for both
//! tensors and metadata.
//!
//! Owned by W11.

use fusor2_ir::Result;
use fusor2_ir::dtype::QLayout;
use fusor2_ir::error::Error;
use std::sync::Arc;

use crate::async_read::{AsyncReadRange, expect_len, read_metadata};
use crate::parse::{GgufMetadata, GgufTensor, GgufValue, ingest_qfmt};
use crate::varbuilder::{RawTensorBytes, VarBuilder};

/// Several files presented as one namespace.
#[derive(Clone, Default)]
pub struct ShardedVarBuilder {
    shards: Vec<VarBuilder>,
}

impl ShardedVarBuilder {
    pub fn new(shards: Vec<VarBuilder>) -> Self {
        Self { shards }
    }

    pub fn shards(&self) -> &[VarBuilder] {
        &self.shards
    }

    pub fn pp<S: ToString>(&self, component: S) -> Self {
        let component = component.to_string();
        Self {
            shards: self.shards.iter().map(|s| s.pp(&component)).collect(),
        }
    }

    pub fn contains_key(&self, name: &str) -> bool {
        self.shards.iter().any(|s| s.contains_key(name))
    }

    /// Every visible tensor name across every shard, prefix-stripped and
    /// deduplicated, in shard order.
    pub fn list_all_keys(&self) -> Vec<String> {
        let mut seen = rustc_hash::FxHashSet::default();
        let mut out = Vec::new();
        for shard in &self.shards {
            for key in shard.list_all_keys() {
                if seen.insert(key.clone()) {
                    out.push(key);
                }
            }
        }
        out
    }

    /// Metadata from the first shard that carries the key.
    pub fn get(&self, name: &str) -> Result<&GgufValue> {
        self.shards
            .iter()
            .find_map(|s| s.get_metadata(name))
            .ok_or_else(|| Error::Io(format!("no metadata key {name} in any shard")))
    }

    /// The directory entry plus raw bytes from the first shard that has it.
    pub fn get_tensor(&self, name: &str) -> Result<(&GgufTensor, &[u8])> {
        for shard in &self.shards {
            if shard.contains_key(name) {
                return shard.get(name);
            }
        }
        Err(Error::Io(format!("no tensor named {name} in any shard")))
    }

    /// The owned form the `fusor2` facade ingests.
    pub fn tensor(&self, name: &str) -> Result<RawTensorBytes> {
        for shard in &self.shards {
            if shard.contains_key(name) {
                return shard.get_raw(name);
            }
        }
        Err(Error::Io(format!("no tensor named {name} in any shard")))
    }

    /// The first shard that declares an architecture.
    pub fn architecture(&self) -> Option<&str> {
        self.shards.iter().find_map(|s| s.architecture())
    }
}

/// One shard of the streaming loader: a parsed header plus the byte-range
/// source its tensor data lives behind.
#[derive(Clone)]
pub struct AsyncShard {
    pub metadata: Arc<GgufMetadata>,
    pub source: Arc<dyn AsyncReadRange>,
}

/// The streaming variant: tensor bytes are fetched by range rather than
/// memory-mapped, so a model can be loaded straight from a remote store.
#[derive(Clone, Default)]
pub struct AsyncShardedVarBuilder {
    shards: Vec<AsyncShard>,
    prefix: String,
}

impl AsyncShardedVarBuilder {
    pub fn new(shards: Vec<AsyncShard>) -> Self {
        Self {
            shards,
            prefix: String::new(),
        }
    }

    /// Read each source's header, then scope to the root.
    pub async fn open(sources: Vec<Arc<dyn AsyncReadRange>>) -> Result<Self> {
        let mut shards = Vec::with_capacity(sources.len());
        for source in sources {
            let metadata = Arc::new(read_metadata(source.as_ref()).await?);
            shards.push(AsyncShard { metadata, source });
        }
        Ok(Self::new(shards))
    }

    pub fn pp<S: ToString>(&self, component: S) -> Self {
        let component = component.to_string();
        let mut prefix = self.prefix.clone();
        if !prefix.is_empty() {
            prefix.push('.');
        }
        prefix.push_str(&component);
        Self {
            shards: self.shards.clone(),
            prefix,
        }
    }

    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    pub fn shards(&self) -> &[AsyncShard] {
        &self.shards
    }

    pub fn sources(&self) -> Vec<Arc<dyn AsyncReadRange>> {
        self.shards.iter().map(|s| Arc::clone(&s.source)).collect()
    }

    pub fn format_path(&self, name: &str) -> String {
        if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}.{name}", self.prefix)
        }
    }

    pub fn contains_key(&self, name: &str) -> bool {
        let path = self.format_path(name);
        self.shards
            .iter()
            .any(|s| s.metadata.tensor(&path).is_some())
    }

    pub fn list_all_keys(&self) -> Vec<String> {
        let head = if self.prefix.is_empty() {
            String::new()
        } else {
            format!("{}.", self.prefix)
        };
        let mut seen = rustc_hash::FxHashSet::default();
        let mut out = Vec::new();
        for shard in &self.shards {
            for tensor in &shard.metadata.tensors {
                let Some(key) = tensor.name.strip_prefix(head.as_str()) else {
                    continue;
                };
                if seen.insert(key.to_string()) {
                    out.push(key.to_string());
                }
            }
        }
        out
    }

    /// Metadata from the first shard that carries the key.
    pub fn get(&self, name: &str) -> Result<&GgufValue> {
        self.shards
            .iter()
            .find_map(|s| s.metadata.get_value(name))
            .ok_or_else(|| Error::Io(format!("no metadata key {name} in any shard")))
    }

    pub fn architecture(&self) -> Option<&str> {
        self.shards.iter().find_map(|s| s.metadata.architecture())
    }

    /// Fetch one tensor's bytes with a single range request against the first
    /// shard that declares it.
    pub async fn tensor(&self, name: &str) -> Result<RawTensorBytes> {
        let path = self.format_path(name);
        for shard in &self.shards {
            let Some(tensor) = shard.metadata.tensor(&path) else {
                continue;
            };
            let start = shard.metadata.tensor_data_offset + tensor.offset;
            let want = tensor.bytes as usize;
            let bytes = shard.source.read_range(start, want).await?;
            let bytes = expect_len(bytes, want, &path)?;
            return Ok(RawTensorBytes {
                name: tensor.name.clone().into_boxed_str(),
                fmt: ingest_qfmt(tensor.ty)?,
                layout: QLayout::Native,
                shape: tensor.shape.clone(),
                bytes: bytes.into_boxed_slice(),
            });
        }
        Err(Error::Io(format!("no tensor named {path} in any shard")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::async_read::{BytesRange, block_on};
    use crate::parse::{Gguf, GgufVersion, fixture};
    use crate::parse::GgmlType;

    fn two_shards() -> (Vec<u8>, Vec<u8>) {
        let a = fixture::build(
            GgufVersion::V3,
            &[
                ("general.architecture", GgufValue::String("qwen3".into())),
                ("qwen3.block_count", GgufValue::U32(2)),
            ],
            &[(
                "blk.0.attn_q.weight",
                GgmlType::Q4_0,
                &[1, 32],
                (0..18u8).collect(),
            )],
        );
        let b = fixture::build(
            GgufVersion::V3,
            &[("qwen3.embedding_length", GgufValue::U32(64))],
            &[(
                "blk.1.attn_q.weight",
                GgmlType::Q4_0,
                &[1, 32],
                (100..118u8).collect(),
            )],
        );
        (a, b)
    }

    #[test]
    fn sharded_and_async_find_across_shards() {
        let (a, b) = two_shards();

        let sync = ShardedVarBuilder::new(vec![
            VarBuilder::new(Arc::new(Gguf::from_bytes(a.clone()).unwrap())),
            VarBuilder::new(Arc::new(Gguf::from_bytes(b.clone()).unwrap())),
        ]);
        assert_eq!(sync.architecture(), Some("qwen3"));
        assert_eq!(sync.get("qwen3.block_count").unwrap(), &GgufValue::U32(2));
        assert_eq!(
            sync.get("qwen3.embedding_length").unwrap(),
            &GgufValue::U32(64)
        );
        assert!(sync.get("missing").is_err());
        assert!(sync.contains_key("blk.0.attn_q.weight"));
        assert!(sync.contains_key("blk.1.attn_q.weight"));
        let mut keys = sync.list_all_keys();
        keys.sort();
        assert_eq!(keys, vec!["blk.0.attn_q.weight", "blk.1.attn_q.weight"]);

        let first = sync.tensor("blk.0.attn_q.weight").unwrap();
        let second = sync.tensor("blk.1.attn_q.weight").unwrap();
        assert_eq!(&*first.bytes, (0..18u8).collect::<Vec<_>>().as_slice());
        assert_eq!(&*second.bytes, (100..118u8).collect::<Vec<_>>().as_slice());
        assert!(sync.tensor("blk.9.attn_q.weight").is_err());

        // The prefixed sync view resolves the same tensor.
        let scoped = sync.pp("blk").pp(1);
        assert!(scoped.contains_key("attn_q.weight"));
        assert_eq!(scoped.tensor("attn_q.weight").unwrap(), second);

        // The async loader reads the same bytes through range requests.
        let sources: Vec<Arc<dyn AsyncReadRange>> = vec![
            Arc::new(BytesRange(Arc::from(a.into_boxed_slice()))),
            Arc::new(BytesRange(Arc::from(b.into_boxed_slice()))),
        ];
        let async_vb = block_on(AsyncShardedVarBuilder::open(sources)).unwrap();
        assert_eq!(async_vb.architecture(), Some("qwen3"));
        assert_eq!(
            async_vb.get("qwen3.embedding_length").unwrap(),
            &GgufValue::U32(64)
        );
        assert!(async_vb.get("missing").is_err());
        let mut keys = async_vb.list_all_keys();
        keys.sort();
        assert_eq!(keys, vec!["blk.0.attn_q.weight", "blk.1.attn_q.weight"]);

        let a0 = block_on(async_vb.tensor("blk.0.attn_q.weight")).unwrap();
        let a1 = block_on(async_vb.tensor("blk.1.attn_q.weight")).unwrap();
        assert_eq!(a0, first, "async shard 0 must be byte-identical to sync");
        assert_eq!(a1, second, "async shard 1 must be byte-identical to sync");
        assert!(block_on(async_vb.tensor("blk.9.attn_q.weight")).is_err());

        let scoped = async_vb.pp("blk").pp(0);
        assert_eq!(scoped.prefix(), "blk.0");
        assert!(scoped.contains_key("attn_q.weight"));
        assert_eq!(block_on(scoped.tensor("attn_q.weight")).unwrap(), first);
        assert_eq!(scoped.sources().len(), 2);
        assert_eq!(scoped.shards().len(), 2);
    }
}
