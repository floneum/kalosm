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

use crate::cache::KernelCache;
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
    /// Persistent plan store, attached once the device capability
    /// fingerprint is known.
    disk: std::sync::OnceLock<Option<crate::disk_cache::DiskPlanCache>>,
    hits: AtomicU64,
    misses: AtomicU64,
    disk_hits: AtomicU64,
}

struct CachedDirectKernelPlan {
    template: DirectKernelTemplate,
    /// Caller-buffer index per kernel binding slot: kernels may bind the
    /// caller's buffers in any order (or bind one buffer several times), so
    /// rebinding routes `caller_buffers[permutation[slot]]` into each slot.
    permutation: Vec<usize>,
    /// For each caller-buffer position, the first position holding the same
    /// buffer at record time. The kernel body is only correct for callers
    /// with the *identical* aliasing pattern: a body built for distinct
    /// buffers binds an aliased pair twice (wrong and rejected by wgpu),
    /// and a body built over an alias (an in-place output) would clobber a
    /// caller whose buffers are distinct.
    alias_class: Vec<usize>,
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
            disk: std::sync::OnceLock::new(),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            disk_hits: AtomicU64::new(0),
        }
    }

    /// Attach the persistent plan store. Kernel codegen depends on device
    /// capabilities, so the store is salted by their fingerprint.
    pub fn attach_disk(&self, device_fingerprint: u64) {
        let _ = self
            .disk
            .set(crate::disk_cache::DiskPlanCache::open(device_fingerprint));
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn try_get_or_insert<E>(
        &self,
        cache: &KernelCache,
        key: KernelCacheKey,
        binding_buffers: &[Arc<wgpu::Buffer>],
        build: impl FnOnce() -> Result<DirectKernel, E>,
    ) -> Result<DirectKernel, E> {
        let mut kernels = self.try_get_or_insert_many(cache, key, &[binding_buffers], || {
            build().map(|kernel| vec![kernel])
        })?;
        Ok(kernels
            .pop()
            .expect("single direct plan cache result must contain one kernel"))
    }

    pub fn try_get_or_insert_many<E>(
        &self,
        cache: &KernelCache,
        key: KernelCacheKey,
        binding_buffers: &[&[Arc<wgpu::Buffer>]],
        build: impl FnOnce() -> Result<Vec<DirectKernel>, E>,
    ) -> Result<Vec<DirectKernel>, E> {
        if !self.enabled {
            return build();
        }
        if let Some(kernels) = self.get_many(cache, key, binding_buffers) {
            return Ok(kernels);
        }
        let built = build()?;
        self.insert_many(key, &built, binding_buffers);
        Ok(built)
    }

    /// A cached plan (memory first, then the persistent store) rebound to
    /// `binding_buffers`, or `None` on a miss.
    pub fn get_many(
        &self,
        cache: &KernelCache,
        key: KernelCacheKey,
        binding_buffers: &[&[Arc<wgpu::Buffer>]],
    ) -> Option<Vec<DirectKernel>> {
        if !self.enabled {
            return None;
        }
        {
            let mut plans = self.plans.lock();
            if let Some(plan) = plans.get(&key)
                && binding_shape_matches(plan, binding_buffers)
            {
                let hit_total = self.hits.fetch_add(1, Ordering::Relaxed) + 1;
                trace_cache_event(hit_total, self.misses.load(Ordering::Relaxed));
                return Some(bind_plan(plan, binding_buffers));
            }
        }

        if let Some(disk) = self.disk.get().and_then(Option::as_ref)
            && let Some(file) = disk.load(key)
            && let Some(plan) = plans_from_disk(file, cache)
            && binding_shape_matches(&plan, binding_buffers)
        {
            let bound = bind_plan(&plan, binding_buffers);
            self.plans.lock().put(key, plan);
            let disk_total = self.disk_hits.fetch_add(1, Ordering::Relaxed) + 1;
            tracing::debug!("direct_plan_disk_hit total={disk_total}");
            return Some(bound);
        }

        let miss_total = self.misses.fetch_add(1, Ordering::Relaxed) + 1;
        trace_cache_event(self.hits.load(Ordering::Relaxed), miss_total);
        None
    }

    /// Record a built plan when its true binding order matches the caller's
    /// buffer list; silently skips plans the positional rebind model cannot
    /// express (internal scratch allocations, deduplicated bindings).
    pub fn insert_many(
        &self,
        key: KernelCacheKey,
        kernels: &[DirectKernel],
        binding_buffers: &[&[Arc<wgpu::Buffer>]],
    ) {
        if !self.enabled {
            return;
        }
        let Some(plan) = record_plan(kernels, binding_buffers) else {
            return;
        };
        if let Some(disk) = self.disk.get().and_then(Option::as_ref)
            && let Some(file) = plans_to_disk(key, &plan)
        {
            disk.store(file);
        }
        self.plans.lock().put(key, plan);
    }
}

fn bind_plan(
    plan: &[CachedDirectKernelPlan],
    binding_buffers: &[&[Arc<wgpu::Buffer>]],
) -> Vec<DirectKernel> {
    plan.iter()
        .zip(binding_buffers)
        .map(|(plan, buffers)| {
            let routed: Vec<Arc<wgpu::Buffer>> = plan
                .permutation
                .iter()
                .map(|&index| buffers[index].clone())
                .collect();
            plan.template.bind_buffers(&routed)
        })
        .collect()
}

fn plans_from_disk(
    file: crate::disk_cache::DiskPlanFile,
    cache: &KernelCache,
) -> Option<Vec<CachedDirectKernelPlan>> {
    file.plans
        .into_iter()
        .map(|plan| {
            let template = DirectKernelTemplate::from_disk(plan.template, cache)?;
            let len = plan.alias_class.len();
            (plan.permutation.iter().all(|&index| index < len)
                && plan
                    .alias_class
                    .iter()
                    .enumerate()
                    .all(|(index, &class)| class <= index))
            .then_some(CachedDirectKernelPlan {
                template,
                permutation: plan.permutation,
                alias_class: plan.alias_class,
            })
        })
        .collect()
}

fn plans_to_disk(
    key: KernelCacheKey,
    plans: &[CachedDirectKernelPlan],
) -> Option<crate::disk_cache::DiskPlanFile> {
    let plans = plans
        .iter()
        .map(|plan| {
            Some(crate::disk_cache::DiskPlan {
                permutation: plan.permutation.clone(),
                alias_class: plan.alias_class.clone(),
                template: plan.template.to_disk()?,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    Some(crate::disk_cache::DiskPlanFile {
        format: crate::disk_cache::DISK_PLAN_FORMAT_VERSION,
        key: key.parts(),
        plans,
    })
}

/// Record templates plus the binding permutation against the caller's
/// buffers; `None` when a kernel binds a buffer the caller does not present
/// (an internal allocation the positional rebind model cannot express) or
/// when the caller's buffers alias each other. Aliasing makes the
/// permutation ambiguous: an in-place output recorded over its input would
/// permanently route the output binding into the input slot, corrupting any
/// later dispatch of the same structural key whose buffers do not alias.
fn record_plan(
    kernels: &[DirectKernel],
    binding_buffers: &[&[Arc<wgpu::Buffer>]],
) -> Option<Vec<CachedDirectKernelPlan>> {
    if kernels.len() != binding_buffers.len() {
        return None;
    }
    kernels
        .iter()
        .zip(binding_buffers)
        .map(|(kernel, expected)| {
            let permutation = kernel
                .binding_buffers()
                .iter()
                .map(|bound| expected.iter().position(|buffer| Arc::ptr_eq(buffer, bound)))
                .collect::<Option<Vec<usize>>>()?;
            Some(CachedDirectKernelPlan {
                template: kernel.to_template(),
                permutation,
                alias_class: alias_classes(expected),
            })
        })
        .collect()
}

/// For each position, the first position holding the same buffer.
fn alias_classes(buffers: &[Arc<wgpu::Buffer>]) -> Vec<usize> {
    buffers
        .iter()
        .enumerate()
        .map(|(index, buffer)| {
            buffers[..index]
                .iter()
                .position(|earlier| Arc::ptr_eq(earlier, buffer))
                .unwrap_or(index)
        })
        .collect()
}

/// Whether the caller's buffers reproduce the recorded aliasing pattern
/// exactly (same positions aliased, same positions distinct).
fn alias_pattern_matches(recorded: &[usize], buffers: &[Arc<wgpu::Buffer>]) -> bool {
    recorded.len() == buffers.len() && alias_classes(buffers) == recorded
}

fn binding_shape_matches(
    plan: &[CachedDirectKernelPlan],
    binding_buffers: &[&[Arc<wgpu::Buffer>]],
) -> bool {
    plan.len() == binding_buffers.len()
        && plan
            .iter()
            .zip(binding_buffers)
            .all(|(plan, buffers)| alias_pattern_matches(&plan.alias_class, buffers))
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
