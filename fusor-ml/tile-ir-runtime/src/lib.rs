//! Runtime caches and dispatch helpers that sit on top of `fusor-tile-ir`.
//!
//! Provides [`KernelCache`] (unified per-kernel naga / shader / pipeline +
//! bind-group caches + wgpu pipeline cache) and [`BufferPool`] (per-`(size,
//! usage)` buffer pool), together with the [`DirectKernel`] dispatch
//! helpers built on top of them.

mod buffer_pool;
mod cache;
mod config;
mod direct_kernel;
mod disk_cache;
mod dispatch;
mod plan_cache;

pub use buffer_pool::{BufferPool, BufferPoolCounters};
pub use cache::{
    CachedKernel, DirectDynamicBindGroupKey, KernelCache, KernelCacheKey, KernelVariantKey,
};
pub use config::FusorConfig;
pub use direct_kernel::{
    DirectKernel, DirectKernelBinding, DirectKernelTemplate, PreparedDirectDispatch,
};
pub use dispatch::{
    dynamic_kernel_from_ir, run_direct_kernel, run_kernel, three_buffer_pipeline_from_ir,
};
pub use plan_cache::KernelPlanCache;

/// Diagnostic: total shader-module / compute-pipeline compilations performed at
/// runtime. Each WGSL shader-module and pipeline creation bumps this counter.
/// It is logged when [`FusorConfig::trace_pipeline_compiles`] is set.
static COMPILES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(crate) fn note_compile(config: &FusorConfig, what: &str) {
    let n = COMPILES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    if config.trace_pipeline_compiles {
        tracing::info!("fusor_compile #{n} {what}");
    }
}
