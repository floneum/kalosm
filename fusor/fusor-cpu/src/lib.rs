//! `fusor-cpu` — `KernelIr` through a CPU emitter: platform GEMM for dense
//! contractions, Cranelift for everything else. `Barrier` splits the lane loop.

#![warn(unreachable_pub)]

mod alloc;
mod caps;
mod emit;
mod gemm;
mod helpers;
#[cfg(not(target_arch = "wasm32"))]
#[cfg_attr(feature = "wasm-emit", allow(dead_code, unreachable_pub))]
mod jit;
pub mod kernel;
mod launch;
mod lower;
mod pool;
mod rows;
mod rules;
mod slices;
mod target;
mod wasm;

pub use alloc::AlignedBuf;
pub use caps::CpuCaps;
pub use emit::{CpuKernel, emit};
pub use launch::Bound;
pub use pool::WorkerPool;
pub use rules::CPU_RULES;
pub use target::CpuTarget;
