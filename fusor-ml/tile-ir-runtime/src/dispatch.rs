use std::sync::Arc;

use fusor_tile_ir as tile_ir;
use wgpu::naga::{AddressSpace, StorageAccess};

use crate::cache::{CachedKernel, KernelCache, KernelCacheKey, ModuleCache};
use crate::direct_kernel::{DirectKernel, DirectKernelBinding};

/// Get the cached entry for `key`, or lower `build_ir` and insert it.
fn cached_kernel(
    cache: &KernelCache,
    key: KernelCacheKey,
    build_ir: impl FnOnce() -> Option<tile_ir::KernelIr>,
) -> Option<Arc<CachedKernel>> {
    if let Some(cached) = cache.kernels.write().get(&key) {
        return Some(cached.clone());
    }
    let naga = Arc::new(build_ir()?.lower_to_naga().ok()?.module().clone());
    Some(cache.get_or_insert_kernel(key, || naga))
}

/// Static-module-cache front for [`cached_kernel`]: hot kernel families
/// short-circuit through their per-family LRU before touching the device-wide
/// cache.
pub fn cached_hashed_naga(
    module_cache: &'static ModuleCache,
    key: KernelCacheKey,
    build_naga: impl FnOnce() -> Option<Arc<wgpu::naga::Module>>,
) -> Option<Arc<wgpu::naga::Module>> {
    if let Some(naga) = module_cache.write().get(&key) {
        return Some(naga.clone());
    }
    let naga = build_naga()?;
    Some(
        module_cache
            .write()
            .get_or_insert(key, || naga.clone())
            .clone(),
    )
}

/// Build a `DirectKernel` whose binding list is derived from the kernel's
/// own resource declarations.
///
/// `buffers` must list the storage buffers in the same order the tile-ir
/// kernel declared them (i.e. the order of `phase.storage_read*`/`storage_write*`
/// calls). The framework reads each binding's read/write access from the
/// lowered Naga module's `GlobalVariable` declarations.
pub fn dynamic_kernel_from_ir(
    cache: &KernelCache,
    name: impl Into<String>,
    cache_key: KernelCacheKey,
    build_ir: impl FnOnce() -> Option<tile_ir::KernelIr>,
    buffers: impl IntoIterator<Item = Arc<wgpu::Buffer>>,
    dispatch_size: [u32; 3],
) -> Option<DirectKernel> {
    let cached = cached_kernel(cache, cache_key, build_ir)?;
    let bindings = bindings_from_naga(&cached.naga, buffers)?;
    Some(DirectKernel::from_naga(
        name,
        cache_key,
        cached.naga.clone(),
        bindings,
        dispatch_size,
    ))
}

/// Two-tier variant of [`dynamic_kernel_from_ir`]: the static `module_cache`
/// short-circuits compilation; misses fall through to the device-wide cache
/// and finally to `build_ir`.
pub fn dynamic_kernel_from_hashed_ir(
    cache: &KernelCache,
    module_cache: &'static ModuleCache,
    label: &str,
    module_key: KernelCacheKey,
    buffers: impl IntoIterator<Item = Arc<wgpu::Buffer>>,
    dispatch_size: [u32; 3],
    build_ir: impl FnOnce() -> Option<tile_ir::KernelIr>,
) -> Option<DirectKernel> {
    let naga = cached_hashed_naga(module_cache, module_key, || {
        cached_kernel(cache, module_key, build_ir).map(|c| c.naga.clone())
    })?;
    let bindings = bindings_from_naga(&naga, buffers)?;
    Some(DirectKernel::from_naga(
        label,
        module_key,
        naga,
        bindings,
        dispatch_size,
    ))
}

/// Build a `DirectKernel` from a closure that builds the kernel's IR via
/// [`tile_ir::KernelBuilder`], pairing each storage declaration with the
/// matching runtime buffer so the two cannot drift.
pub fn run_kernel<F>(
    cache: &KernelCache,
    name: impl Into<String>,
    cache_key: KernelCacheKey,
    dispatch_size: [u32; 3],
    body: F,
) -> Option<DirectKernel>
where
    F: FnOnce(&mut tile_ir::KernelBuilder<Arc<wgpu::Buffer>>) -> Option<()>,
{
    let mut kb = tile_ir::KernelBuilder::<Arc<wgpu::Buffer>>::new();
    body(&mut kb)?;
    let (ir, buffers) = kb.finish();
    dynamic_kernel_from_ir(
        cache,
        name,
        cache_key,
        move || Some(ir),
        buffers,
        dispatch_size,
    )
}

pub fn run_direct_kernel(
    cache: &KernelCache,
    queue: &wgpu::Queue,
    label: &str,
    kernel: &DirectKernel,
    encoder: Option<&mut wgpu::CommandEncoder>,
) {
    if let Some(encoder) = encoder {
        kernel.run(cache, encoder);
    } else {
        let mut encoder = cache
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        kernel.run(cache, &mut encoder);
        queue.submit(Some(encoder.finish()));
    }
}

/// Build a compute pipeline using the singleton 3-buffer pipeline layout
/// for an already-cached kernel. The shader is shared with the dynamic path.
fn prepare_three_buffer_pipeline(
    cache: &KernelCache,
    name: &str,
    cached: &Arc<CachedKernel>,
) -> wgpu::ComputePipeline {
    let shader = cache.shader_for(cached);
    let pipeline_layout = cache.direct_three_buffer_pipeline_layout();
    cache
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name),
            layout: Some(&pipeline_layout),
            module: shader,
            entry_point: Some("main"),
            cache: cache.wgpu_cache.as_ref(),
            compilation_options: wgpu::PipelineCompilationOptions {
                zero_initialize_workgroup_memory: false,
                ..Default::default()
            },
        })
}

pub fn three_buffer_pipeline_from_ir(
    cache: &KernelCache,
    name: &str,
    cache_key: KernelCacheKey,
    build_ir: impl FnOnce() -> Option<tile_ir::KernelIr>,
) -> Option<wgpu::ComputePipeline> {
    let cached = cached_kernel(cache, cache_key, build_ir)?;
    Some(prepare_three_buffer_pipeline(cache, name, &cached))
}

pub fn three_buffer_pipeline_from_cached_module(
    cache: &KernelCache,
    name: &str,
    cache_key: KernelCacheKey,
) -> Option<wgpu::ComputePipeline> {
    let cached = cache.kernels.write().get(&cache_key).cloned()?;
    Some(prepare_three_buffer_pipeline(cache, name, &cached))
}

/// Read each storage `GlobalVariable` from the Naga module in `(group, binding)`
/// order and pair it with the supplied buffer at that position. The access
/// mode (read-only vs read-write) is taken from the IR-emitted `StorageAccess`
/// flags, so callers never specify it explicitly.
fn bindings_from_naga(
    module: &wgpu::naga::Module,
    buffers: impl IntoIterator<Item = Arc<wgpu::Buffer>>,
) -> Option<Vec<DirectKernelBinding>> {
    let mut storages: Vec<(u32, bool)> = module
        .global_variables
        .iter()
        .filter_map(|(_, gv)| match gv.space {
            AddressSpace::Storage { access } => {
                let binding = gv.binding.as_ref()?;
                let read_only = !access.contains(StorageAccess::STORE);
                Some((binding.binding, read_only))
            }
            _ => None,
        })
        .collect();
    storages.sort_unstable_by_key(|(binding, _)| *binding);

    let buffers: Vec<Arc<wgpu::Buffer>> = buffers.into_iter().collect();
    if buffers.len() != storages.len() {
        return None;
    }
    Some(
        storages
            .into_iter()
            .zip(buffers)
            .map(
                |((binding, read_only), buffer)| DirectKernelBinding::Storage {
                    binding,
                    buffer,
                    read_only,
                },
            )
            .collect(),
    )
}
