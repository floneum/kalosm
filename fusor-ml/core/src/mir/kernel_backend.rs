use std::{
    hash::{Hash, Hasher},
    num::NonZeroUsize,
    sync::Arc,
};

use fusor_tile_ir as tile_ir;
use fusor_tile_ir_kernels as tile_ir_kernels;
use lru::LruCache;
use parking_lot::RwLock;
use rustc_hash::{FxBuildHasher, FxHasher};
use wgpu::naga::{AddressSpace, StorageAccess};

use crate::{
    Device,
    mir::direct_kernel::{DirectKernel, DirectKernelBinding},
    tensor::TensorData,
};

pub(crate) mod flash_attention;
pub(crate) mod mirostat;
pub(crate) mod rms_norm;
pub(crate) mod sampling_topk;

#[derive(Clone, Debug)]
pub(crate) struct CompiledKernelModule {
    module: Arc<wgpu::naga::Module>,
}

pub(crate) type ModuleCache = RwLock<LruCache<[u64; 2], CompiledKernelModule, FxBuildHasher>>;

pub(crate) fn module_cache(capacity: usize) -> ModuleCache {
    RwLock::new(LruCache::with_hasher(
        NonZeroUsize::new(capacity).expect("module cache capacity must be non-zero"),
        Default::default(),
    ))
}

pub(crate) fn cached_hashed_kernel_module(
    cache: &'static ModuleCache,
    key: [u64; 2],
    build_module: impl FnOnce() -> Option<CompiledKernelModule>,
) -> Option<CompiledKernelModule> {
    if let Some(module) = cache.write().get(&key) {
        return Some(module.clone());
    }
    let module = build_module()?;
    Some(cache.write().get_or_insert(key, || module.clone()).clone())
}

pub(crate) fn hash_layout<H: Hasher>(state: &mut H, layout: &crate::Layout) {
    layout.offset().hash(state);
    layout.shape().hash(state);
    layout.strides().hash(state);
}

pub(crate) fn hash_strided_layout<H: Hasher>(state: &mut H, layout: &crate::Layout) {
    layout.offset().hash(state);
    layout.strides().hash(state);
}

fn compiled_module(module: wgpu::naga::Module) -> CompiledKernelModule {
    CompiledKernelModule {
        module: Arc::new(module),
    }
}

fn compile_ir(ir: tile_ir::KernelIr) -> Option<CompiledKernelModule> {
    Some(compiled_module(ir.lower_to_naga().ok()?.module().clone()))
}

fn cached_kernel_module(
    device: &Device,
    cache_key: impl Into<String>,
    build_module: impl FnOnce() -> Option<CompiledKernelModule>,
) -> Option<CompiledKernelModule> {
    let cache_key = cache_key.into();
    if let Some(module) = device.naga_module_cache().write().get(&cache_key) {
        return Some(CompiledKernelModule {
            module: Arc::new(module.clone()),
        });
    }

    let compiled = build_module()?;
    device
        .naga_module_cache()
        .write()
        .get_or_insert(cache_key, || compiled.module.as_ref().clone());
    Some(compiled)
}

pub(crate) fn cached_kernel_ir(
    device: &Device,
    cache_key: impl Into<String>,
    build_ir: impl FnOnce() -> Option<tile_ir::KernelIr>,
) -> Option<CompiledKernelModule> {
    cached_kernel_module(device, cache_key, || compile_ir(build_ir()?))
}

/// Build a `DirectKernel` whose binding list is derived from the kernel's
/// own resource declarations.
///
/// `buffers` must list the storage buffers in the same order the tile-ir
/// kernel declared them (i.e. the order of `phase.storage_read*`/`storage_write*`
/// calls). The framework reads each binding's read/write access from the
/// lowered Naga module's `GlobalVariable` declarations, so backends never
/// have to repeat that information.
pub(crate) fn dynamic_kernel_from_ir(
    device: &Device,
    name: impl Into<String>,
    cache_key: impl Into<String>,
    build_ir: impl FnOnce() -> Option<tile_ir::KernelIr>,
    buffers: impl IntoIterator<Item = Arc<wgpu::Buffer>>,
    dispatch_size: [u32; 3],
) -> Option<DirectKernel> {
    let cache_key = cache_key.into();
    let module = cached_kernel_ir(device, cache_key.clone(), build_ir)?;
    let bindings = bindings_from_module(&module, buffers)?;
    Some(dynamic_kernel_from_module(
        name,
        cache_key,
        module,
        bindings,
        dispatch_size,
    ))
}

pub(crate) fn dynamic_kernel_from_module(
    name: impl Into<String>,
    cache_key: impl Into<String>,
    module: CompiledKernelModule,
    bindings: Vec<DirectKernelBinding>,
    dispatch_size: [u32; 3],
) -> DirectKernel {
    DirectKernel::new_with_arc_module(name, cache_key, module.module, bindings, dispatch_size)
}

/// Read each storage `GlobalVariable` from the Naga module in `(group, binding)`
/// order and pair it with the supplied buffer at that position. The access
/// mode (read-only vs read-write) is taken from the IR-emitted `StorageAccess`
/// flags, so callers never specify it explicitly.
pub(crate) fn bindings_from_module(
    module: &CompiledKernelModule,
    buffers: impl IntoIterator<Item = Arc<wgpu::Buffer>>,
) -> Option<Vec<DirectKernelBinding>> {
    let mut storages: Vec<(u32, bool)> = module
        .module
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
            .map(|((binding, read_only), buffer)| DirectKernelBinding::Storage {
                binding,
                buffer,
                read_only,
            })
            .collect(),
    )
}

pub(crate) fn buffers_from_tensors<const N: usize>(
    tensors: [&TensorData; N],
) -> [Arc<wgpu::Buffer>; N] {
    tensors.map(|tensor| tensor.buffer().clone())
}

/// Build a `DirectKernel` whose IR and runtime bindings are produced together
/// by a single `body` closure: each [`tile_ir::KernelBuilder`] `read`/`write`
/// call inside the closure both declares an IR storage and appends the
/// matching runtime buffer to the dispatch's binding list, so the two can
/// never drift.
pub(crate) fn run_kernel<F>(
    device: &Device,
    name: impl Into<String>,
    cache_key: impl Into<String>,
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
        device,
        name,
        cache_key,
        move || Some(ir),
        buffers,
        dispatch_size,
    )
}

/// Same as [`run_kernel`] but uses the hashed two-tier module cache so
/// repeated dispatches with the same shape skip both the tile-ir build and
/// the Naga lowering.
pub(crate) fn run_kernel_with_hashed_cache<F>(
    device: &Device,
    cache: &'static ModuleCache,
    label: &str,
    module_key: [u64; 2],
    dispatch_size: [u32; 3],
    body: F,
) -> Option<DirectKernel>
where
    F: FnOnce(&mut tile_ir::KernelBuilder<Arc<wgpu::Buffer>>) -> Option<()>,
{
    let mut kb = tile_ir::KernelBuilder::<Arc<wgpu::Buffer>>::new();
    body(&mut kb)?;
    let (ir, buffers) = kb.finish();
    dynamic_kernel_from_hashed_ir(
        device,
        cache,
        label,
        module_key,
        buffers,
        dispatch_size,
        move || Some(ir),
    )
}

/// Convert a `&TensorData` directly into a tile-ir tensor ref using the
/// default rank-1 linear storage layout (i.e. the kernel's `Meta` struct
/// already encodes the offset/stride).
pub(crate) fn linear_tensor_ref(
    tensor: &TensorData,
) -> tile_ir::KernelTensorRef<Arc<wgpu::Buffer>> {
    tile_ir::KernelTensorRef::new(
        tensor.buffer().clone(),
        tile_ir_kernels::linear_storage_layout(),
    )
}

/// Hash a module key with the standard salt-pair pattern used by the backend
/// caches: returns `[u64; 2]` where each lane uses a distinct salt so callers
/// can derive both an LRU key and a cache string from the same input data.
pub(crate) fn module_key_from(hash_inputs: impl Fn(&mut FxHasher)) -> [u64; 2] {
    std::array::from_fn(|salt| {
        let mut hasher = FxHasher::default();
        (salt as u64).hash(&mut hasher);
        hash_inputs(&mut hasher);
        hasher.finish()
    })
}

/// Format the standard `"{label}:{hi:016x}{lo:016x}"` cache key used by the
/// hashed-module backend pattern.
pub(crate) fn hashed_cache_key(label: &str, module_key: [u64; 2]) -> String {
    format!("{label}:{:016x}{:016x}", module_key[0], module_key[1])
}

/// Build a `DirectKernel` using the hashed two-tier cache pattern: the
/// hashed module-key LRU short-circuits compilation; misses fall through to
/// the device-wide naga cache and finally to `build_ir`. Bindings are
/// derived from the lowered Naga module — see [`bindings_from_module`].
pub(crate) fn dynamic_kernel_from_hashed_ir(
    device: &Device,
    cache: &'static ModuleCache,
    label: &str,
    module_key: [u64; 2],
    buffers: impl IntoIterator<Item = Arc<wgpu::Buffer>>,
    dispatch_size: [u32; 3],
    build_ir: impl FnOnce() -> Option<tile_ir::KernelIr>,
) -> Option<DirectKernel> {
    let cache_key = hashed_cache_key(label, module_key);
    let module = cached_hashed_kernel_module(cache, module_key, || {
        cached_kernel_ir(device, cache_key.clone(), build_ir)
    })?;
    let bindings = bindings_from_module(&module, buffers)?;
    Some(dynamic_kernel_from_module(
        label,
        cache_key,
        module,
        bindings,
        dispatch_size,
    ))
}

pub(crate) fn run_direct_kernel(
    device: &Device,
    label: &str,
    kernel: &DirectKernel,
    encoder: Option<&mut wgpu::CommandEncoder>,
) {
    if let Some(encoder) = encoder {
        kernel.run(device, encoder);
    } else {
        let mut encoder = device
            .wgpu_device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        kernel.run(device, &mut encoder);
        device.wgpu_queue().submit(Some(encoder.finish()));
    }
}

pub(crate) fn storage3_kernel_with_prepared_pipeline(
    name: impl Into<String>,
    cache_key: impl Into<String>,
    pipeline: wgpu::ComputePipeline,
    input: Arc<wgpu::Buffer>,
    weight: Arc<wgpu::Buffer>,
    output: Arc<wgpu::Buffer>,
    dispatch_size: [u32; 3],
) -> DirectKernel {
    DirectKernel::new_storage3_with_prepared_pipeline(
        name,
        cache_key,
        pipeline,
        input,
        weight,
        output,
        dispatch_size,
    )
}

pub(crate) fn prepare_storage3_pipeline(
    device: &Device,
    name: &str,
    module: &CompiledKernelModule,
) -> wgpu::ComputePipeline {
    let shader = device.create_naga_shader_module(module.module.as_ref().clone());
    let pipeline_layout = device.direct_storage3_pipeline_layout();
    device
        .wgpu_device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            cache: device.wgpu_cache(),
            compilation_options: wgpu::PipelineCompilationOptions {
                zero_initialize_workgroup_memory: false,
                ..Default::default()
            },
        })
}

pub(crate) fn storage3_pipeline_from_ir(
    device: &Device,
    name: &str,
    cache_key: impl Into<String>,
    build_ir: impl FnOnce() -> Option<tile_ir::KernelIr>,
) -> Option<wgpu::ComputePipeline> {
    let module = cached_kernel_ir(device, cache_key, build_ir)?;
    Some(prepare_storage3_pipeline(device, name, &module))
}

pub(crate) fn storage3_pipeline_from_cached_module(
    device: &Device,
    name: &str,
    cache_key: &str,
) -> Option<wgpu::ComputePipeline> {
    let module = device
        .naga_module_cache()
        .write()
        .get(cache_key)
        .map(|module| CompiledKernelModule {
            module: Arc::new(module.clone()),
        })?;
    Some(prepare_storage3_pipeline(device, name, &module))
}
