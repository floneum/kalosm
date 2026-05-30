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
    CachedKernel, DirectDynamicBindGroupKey, DirectStorage3BindGroupKey, KernelCache,
    KernelCacheKey, KernelVariantKey, ModuleCache, module_cache,
};
pub use direct_kernel::{DirectKernel, DirectKernelBinding, PreparedDirectDispatch};
pub use dispatch::{
    cached_hashed_naga, dynamic_kernel_from_hashed_ir, dynamic_kernel_from_ir, run_direct_kernel,
    run_kernel, three_buffer_pipeline_from_cached_module, three_buffer_pipeline_from_ir,
};
