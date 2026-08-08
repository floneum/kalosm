//! `fusor2-cpu` — the same `KernelIr` through a different emitter.
//!
//! A runtime-parameterized SIMD loop nest over `fearless_simd`'s
//! statically-known `f32x4`/`f32x8`/`f32x16`, so a 4x4 register accumulator
//! tile is expressible.
//!
//! `Barrier` splits the lane loop into two loops over the lane range: a block's
//! 256 lanes are an inner loop of 32 iterations at `W = 8`, and kernels that
//! stage through workgroup memory require the split for correctness.

pub mod alloc;
pub mod caps;
pub mod emit;
pub mod launch;
pub mod lower;
pub mod pool;
pub mod rules;
pub mod target;

pub use alloc::AlignedBuf;
pub use caps::CpuCaps;
pub use emit::{CpuKernel, emit};
pub use pool::WorkerPool;
pub use rules::CPU_RULES;
pub use target::CpuTarget;
