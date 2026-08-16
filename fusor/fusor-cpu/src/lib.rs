//! `fusor-cpu` — the same `KernelIr` through a different emitter.
//!
//! A runtime-parameterized SIMD loop nest over `fearless_simd`'s
//! statically-known `f32x4`/`f32x8`/`f32x16`.
//!
//! `Barrier` splits the lane loop into two loops over the lane range; mapping
//! `Barrier` to a no-op miscompiles every kernel that stages through workgroup
//! memory.

#![warn(unreachable_pub)]

mod alloc;
mod caps;
mod emit;
mod launch;
mod lower;
mod pool;
mod rules;
mod target;

pub use alloc::AlignedBuf;
pub use caps::CpuCaps;
pub use emit::{CpuKernel, emit};
pub use pool::WorkerPool;
pub use rules::CPU_RULES;
pub use target::CpuTarget;
