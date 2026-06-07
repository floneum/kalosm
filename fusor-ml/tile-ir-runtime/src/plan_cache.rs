use std::{
    num::NonZeroUsize,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use lru::LruCache;
use parking_lot::Mutex;
use rustc_hash::FxBuildHasher;

use crate::{DirectKernel, DirectKernelTemplate, KernelCacheKey};

#[cfg(not(target_arch = "wasm32"))]
const DIRECT_PLAN_CACHE_SIZE: usize = 4096;
#[cfg(target_arch = "wasm32")]
const DIRECT_PLAN_CACHE_SIZE: usize = 512;

/// Per-device cache for direct-kernel plans.
///
/// This stores bufferless direct-kernel templates and replays them with the
/// caller-provided binding buffers for the current dispatch. The cache never
/// infers binding provenance from pointer equality; callers must provide the
/// buffers in the exact order returned by [`DirectKernel::binding_buffers`].
pub struct DirectPlanCache {
    enabled: bool,
    plans: Mutex<LruCache<KernelCacheKey, Vec<CachedDirectKernelPlan>, FxBuildHasher>>,
    hits: AtomicU64,
    misses: AtomicU64,
}

struct CachedDirectKernelPlan {
    template: DirectKernelTemplate,
    binding_count: usize,
}

impl std::fmt::Debug for DirectPlanCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DirectPlanCache")
            .field("enabled", &self.enabled)
            .finish()
    }
}

impl Default for DirectPlanCache {
    fn default() -> Self {
        Self::new()
    }
}

impl DirectPlanCache {
    pub fn new() -> Self {
        Self {
            enabled: std::env::var_os("FUSOR_DISABLE_DECODE_PLAN_CACHE").is_none(),
            plans: Mutex::new(LruCache::with_hasher(
                NonZeroUsize::new(DIRECT_PLAN_CACHE_SIZE)
                    .expect("direct plan cache size must be non-zero"),
                Default::default(),
            )),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn try_get_or_insert<E>(
        &self,
        key: KernelCacheKey,
        binding_buffers: &[Arc<wgpu::Buffer>],
        build: impl FnOnce() -> Result<DirectKernel, E>,
    ) -> Result<DirectKernel, E> {
        let mut kernels = self.try_get_or_insert_many(key, &[binding_buffers], || {
            build().map(|kernel| vec![kernel])
        })?;
        Ok(kernels
            .pop()
            .expect("single direct plan cache result must contain one kernel"))
    }

    pub fn try_get_or_insert_many<E>(
        &self,
        key: KernelCacheKey,
        binding_buffers: &[&[Arc<wgpu::Buffer>]],
        build: impl FnOnce() -> Result<Vec<DirectKernel>, E>,
    ) -> Result<Vec<DirectKernel>, E> {
        if !self.enabled {
            return build();
        }

        {
            let mut plans = self.plans.lock();
            if let Some(plan) = plans.get(&key)
                && binding_shape_matches(plan, binding_buffers)
            {
                let hit_total = self.hits.fetch_add(1, Ordering::Relaxed) + 1;
                trace_cache_event(hit_total, self.misses.load(Ordering::Relaxed));
                return Ok(plan
                    .iter()
                    .zip(binding_buffers)
                    .map(|(plan, buffers)| plan.template.bind_buffers(buffers))
                    .collect());
            }
        }

        let miss_total = self.misses.fetch_add(1, Ordering::Relaxed) + 1;
        trace_cache_event(self.hits.load(Ordering::Relaxed), miss_total);
        let built = build()?;
        if binding_buffers_match(&built, binding_buffers) {
            self.plans.lock().put(key, record_plan(&built));
        }
        Ok(built)
    }
}

fn record_plan(kernels: &[DirectKernel]) -> Vec<CachedDirectKernelPlan> {
    kernels
        .iter()
        .map(|kernel| CachedDirectKernelPlan {
            template: kernel.to_template(),
            binding_count: kernel.binding_buffers().len(),
        })
        .collect()
}

fn binding_shape_matches(
    plan: &[CachedDirectKernelPlan],
    binding_buffers: &[&[Arc<wgpu::Buffer>]],
) -> bool {
    plan.len() == binding_buffers.len()
        && plan
            .iter()
            .zip(binding_buffers)
            .all(|(plan, buffers)| plan.binding_count == buffers.len())
}

fn binding_buffers_match(kernels: &[DirectKernel], expected: &[&[Arc<wgpu::Buffer>]]) -> bool {
    if kernels.len() != expected.len() {
        return false;
    }

    kernels.iter().zip(expected).all(|(kernel, expected)| {
        let actual = kernel.binding_buffers();
        actual.len() == expected.len()
            && actual
                .iter()
                .zip(*expected)
                .all(|(actual, expected)| Arc::ptr_eq(actual, expected))
    })
}

fn trace_cache_event(hits: u64, misses: u64) {
    if cfg!(target_arch = "wasm32") || std::env::var_os("FUSOR_TRACE_RESOLVE_HOST").is_some() {
        tracing::info!("direct_plan_cache hit={hits} miss={misses}");
    }
}
