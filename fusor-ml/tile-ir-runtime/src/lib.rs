//! Runtime caches and dispatch helpers that sit on top of `fusor-tile-ir`.
//!
//! Provides [`KernelCache`] (unified per-kernel naga / shader / pipeline +
//! bind-group caches + wgpu pipeline cache) and [`BufferPool`] (per-`(size,
//! usage)` buffer pool), together with the [`DirectKernel`] dispatch
//! helpers built on top of them.

mod buffer_pool;
mod cache;
mod direct_kernel;
mod dispatch;

pub use buffer_pool::BufferPool;
pub use cache::{
    CachedKernel, DirectDynamicBindGroupKey, KernelCache, KernelCacheKey, KernelVariantKey,
};
pub use direct_kernel::{
    DirectKernel, DirectKernelBinding, DirectKernelTemplate, PreparedDirectDispatch,
};
pub use dispatch::{
    dynamic_kernel_from_ir, run_direct_kernel, run_kernel, three_buffer_pipeline_from_ir,
};

/// Diagnostic: total shader-module / compute-pipeline compilations performed at
/// runtime. Each WGSL shader-module and pipeline creation bumps this counter.
/// It is logged when `FUSOR_TRACE_PIPELINE_COMPILES` is set.
static COMPILES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(crate) fn note_compile(what: &str) {
    let n = COMPILES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    if std::env::var_os("FUSOR_TRACE_PIPELINE_COMPILES").is_some() {
        tracing::info!("fusor_compile #{n} {what}");
    }
}
