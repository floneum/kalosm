//! `fusor-gpu` — the wgpu [`Target`](fusor_ir::target::Target) end to end:
//! baseline-limit probing, Launch lowering, naga emission with derived bind
//! groups, the pooled allocator and the encoder/submission model.

#![warn(unreachable_pub)]

mod bindings;
mod caps;
mod device;
mod emit;
mod flags;
pub mod launch;
mod lower;
pub mod pool;
pub mod reduction;
mod rules;
pub mod target;
mod uniforms;

pub use bindings::{BindingDesc, bindings_from_module};
pub use device::{GpuDevice, removed_reason};
pub use emit::emit;
pub use launch::Launcher;
pub use pool::BufferPool;
pub use rules::GPU_RULES;
pub use target::GpuTarget;

pub(crate) use flags::flags;
