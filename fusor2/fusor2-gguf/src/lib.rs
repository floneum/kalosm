//! `fusor2-gguf` — quantized formats as data.
//!
//! Six ingestible GGUF block formats x two on-device storage layouts as table
//! rows, each carrying a `BlockProgram` that emits an L2 decode snippet rather
//! than a kernel; the repack between layouts; GGUF file parsing; and the
//! sync/sharded/async VarBuilders.
//!
//! Adding Q4_1 is a table row plus a block program — not a kernel and not a
//! selector arm.

pub mod async_read;
pub mod blocks;
pub mod decode;
pub mod decode_k;
pub mod parse;
pub mod repack;
pub mod sharded;
pub mod varbuilder;

pub use blocks::{BLOCK_SPECS, block_spec};
pub use parse::{Gguf, GgufMetadata, GgufTensor};
pub use repack::repack;
pub use sharded::{AsyncShardedVarBuilder, ShardedVarBuilder};
pub use varbuilder::VarBuilder;
