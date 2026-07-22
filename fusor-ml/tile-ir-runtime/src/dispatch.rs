use std::sync::Arc;

use fusor_tile_ir as tile_ir;
use wgpu::naga::{AddressSpace, StorageAccess};

use crate::cache::{CachedKernel, KernelCache, KernelCacheKey};
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
    let ir = build_ir()?;
    // The threadgroup footprint decides residency on Apple Silicon (two
    // workgroups per core at or below half the 32 KB budget); record it per
    // fresh build so selection work always has real numbers.
    tracing::debug!("kernel_built workgroup_bytes={}", ir.workgroup_bytes());
    let lowered = ir.lower_to_naga();
    if cache.config().trace_matmul_merge
        && let Err(error) = &lowered
    {
        eprintln!("tile_ir_lower_error: {error:?}");
    }
    let kernel = Arc::new(lowered.ok()?);
    Some(cache.get_or_insert_kernel(key, || kernel))
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
    let bindings = bindings_from_naga(
        cached.kernel.module(),
        buffers,
        cache.config().trace_matmul_merge,
    )?;
    Some(DirectKernel::from_cached(
        name,
        cached,
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
pub(crate) fn prepare_three_buffer_pipeline(
    cache: &KernelCache,
    name: &str,
    cached: &Arc<CachedKernel>,
) -> wgpu::ComputePipeline {
    let shader = cache.shader_for(cached);
    let pipeline_layout = cache.direct_three_buffer_pipeline_layout();
    cached
        .storage3_pipeline
        .get_or_init(|| {
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
        })
        .clone()
}

pub fn three_buffer_pipeline_from_ir(
    cache: &KernelCache,
    name: &str,
    cache_key: KernelCacheKey,
    build_ir: impl FnOnce() -> Option<tile_ir::KernelIr>,
) -> Option<(wgpu::ComputePipeline, Arc<CachedKernel>)> {
    let cached = cached_kernel(cache, cache_key, build_ir)?;
    let pipeline = prepare_three_buffer_pipeline(cache, name, &cached);
    Some((pipeline, cached))
}

/// Read each storage `GlobalVariable` from the Naga module in `(group, binding)`
/// order and pair it with the supplied buffer at that position. The access
/// mode (read-only vs read-write) is taken from the IR-emitted `StorageAccess`
/// flags, so callers never specify it explicitly.
fn bindings_from_naga(
    module: &wgpu::naga::Module,
    buffers: impl IntoIterator<Item = Arc<wgpu::Buffer>>,
    trace_mismatch: bool,
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
        if trace_mismatch {
            eprintln!(
                "bindings_from_naga mismatch: buffers={} storages={}",
                buffers.len(),
                storages.len()
            );
        }
        return None;
    }
    Some(
        storages
            .into_iter()
            .zip(buffers)
            .map(|((binding, read_only), buffer)| DirectKernelBinding {
                binding,
                buffer,
                read_only,
            })
            .collect(),
    )
}
