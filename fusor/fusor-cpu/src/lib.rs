//! `fusor-cpu` — `KernelIr` through a CPU emitter: platform GEMM for dense
//! contractions, Cranelift for everything else. `Barrier` splits the lane loop.

#![warn(unreachable_pub)]

mod alloc;
mod caps;
mod emit;
mod gemm;
mod jit;
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
