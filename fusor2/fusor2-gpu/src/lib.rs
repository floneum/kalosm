//! `fusor2-gpu` — the wgpu [`Target`](fusor2_ir::target::Target) end to end.
//!
//! Adapter/capability probing against **WebGPU baseline limits** with
//! per-kernel widening, L1+`SchedPoint` -> `KernelIr` lowering for every L1
//! node family, `KernelIr` -> naga `Module` emission with **derived** bind
//! groups and a storage `Uniforms` block at binding 0, the pooled allocator
//! with a platform memory ceiling, the plan cache (memory LRU + disk, salted
//! by exe identity and `DeviceFacts` fingerprint *including*
//! `max_compute_workgroup_storage_size`), and an encoder/submission model
//! whose only host syncs are readback, explicit wait and the allocator retry.

pub mod bindings;
pub mod caps;
pub mod device;
pub mod emit;
pub mod launch;
pub mod lower;
pub mod plan_cache;
pub mod pool;
pub mod rules;
pub mod target;
pub mod uniforms;

pub use bindings::{BindingDesc, bindings_from_module};
pub use device::GpuDevice;
pub use emit::emit;
pub use launch::Launcher;
pub use plan_cache::PlanCache;
pub use pool::BufferPool;
pub use rules::GPU_RULES;
pub use target::GpuTarget;
