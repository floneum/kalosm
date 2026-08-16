//! `fusor2-gguf` — quantized formats as data.
//!
//! Six ingestible GGUF block formats x two on-device storage layouts as table
//! rows, each carrying a `BlockProgram` that emits an Kernel decode snippet rather
//! than a kernel; the repack between layouts; GGUF file parsing; and the
//! sync/sharded/async VarBuilders.
//!
//! Adding Q4_1 is a table row plus a block program — not a kernel and not a
//! selector arm.

#![warn(unreachable_pub)]

mod async_read;
pub mod blocks;
mod decode;
mod decode_k;
pub mod parse;
pub mod repack;
mod sharded;
mod varbuilder;

pub use blocks::{BLOCK_SPECS, block_spec};
pub use parse::{GgmlType, Gguf, GgufMetadata, GgufTensor, GgufValue};
pub use repack::repack;
pub use sharded::{AsyncShardedVarBuilder, ShardedVarBuilder};
pub use varbuilder::{RawTensorBytes, VarBuilder};
