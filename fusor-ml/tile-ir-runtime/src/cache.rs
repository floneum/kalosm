use std::{
    any::TypeId,
    borrow::Cow,
    hash::{Hash, Hasher},
    num::NonZeroUsize,
    path::PathBuf,
    sync::{Arc, OnceLock},
};

use fusor_tile_ir::NagaKernel;
use lru::LruCache;
use parking_lot::RwLock;
use rustc_hash::{FxBuildHasher, FxHasher};
use wgpu::{BindGroupLayout, PipelineLayout};

use crate::KernelPlanCache;

#[cfg(not(target_arch = "wasm32"))]
const KERNEL_CACHE_SIZE: usize = 4096;
#[cfg(target_arch = "wasm32")]
const KERNEL_CACHE_SIZE: usize = 512;
#[cfg(not(target_arch = "wasm32"))]
const DIRECT_DYNAMIC_BIND_GROUP_CACHE_SIZE: usize = 4096;
#[cfg(target_arch = "wasm32")]
const DIRECT_DYNAMIC_BIND_GROUP_CACHE_SIZE: usize = 512;

/// Content-addressed key used to dedupe compiled kernel modules, shader
/// modules, and pipelines across dispatches of the same kernel.
///
/// Built on the canonical two-lane hash (see [`crate::two_lane_salted`]);
/// trusted without exact verification, so the hashed inputs must cover every
/// fact that changes generated source or binding layout — a collision or an
/// omitted field both mean dispatching the wrong pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KernelCacheKey([u64; 2]);

impl KernelCacheKey {
    pub(crate) fn parts(&self) -> [u64; 2] {
        self.0
    }

    pub(crate) fn from_parts(parts: [u64; 2]) -> Self {
        Self(parts)
    }

    pub fn from_hash_inputs(hash_inputs: impl Fn(&mut FxHasher)) -> Self {
        Self(crate::two_lane_salted(hash_inputs))
    }
}

/// Key that pairs a Rust type id with a hashed payload, used for kernel
/// variant lookups (e.g. distinguishing two specializations of the same
/// generic kernel by their parameter struct).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KernelVariantKey {
    type_id: TypeId,
    payload: u64,
}

impl KernelVariantKey {
    pub fn of<T: 'static>() -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            payload: 0,
        }
    }

    pub fn with_payload<T: 'static>(hash_payload: impl Fn(&mut FxHasher)) -> Self {
        let mut hasher = FxHasher::default();
        hash_payload(&mut hasher);
        Self {
            type_id: TypeId::of::<T>(),
            payload: hasher.finish(),
        }
    }
}

/// A lowered kernel plus its lazily-built shader module and dynamic-path
/// compute pipeline. One entry per [`KernelCacheKey`] in [`KernelCache`].
#[derive(Debug)]
pub struct CachedKernel {
    pub(crate) kernel: Arc<NagaKernel>,
    pub(crate) shader: OnceLock<wgpu::ShaderModule>,
    pub(crate) dynamic_bind_group_layout: OnceLock<wgpu::BindGroupLayout>,
    pub(crate) dynamic_pipeline_layout: OnceLock<wgpu::PipelineLayout>,
    pub(crate) pipeline: OnceLock<wgpu::ComputePipeline>,
    pub(crate) storage3_pipeline: OnceLock<wgpu::ComputePipeline>,
}

impl CachedKernel {
    pub(crate) fn new(kernel: Arc<NagaKernel>) -> Self {
        Self {
            kernel,
            shader: OnceLock::new(),
            dynamic_bind_group_layout: OnceLock::new(),
            dynamic_pipeline_layout: OnceLock::new(),
            pipeline: OnceLock::new(),
            storage3_pipeline: OnceLock::new(),
        }
    }
}

#[derive(Debug)]
pub(crate) struct CachedDirectBindGroup {
    pub(crate) bind_group: wgpu::BindGroup,
    _buffers: Vec<Arc<wgpu::Buffer>>,
}

impl CachedDirectBindGroup {
    pub(crate) fn new(bind_group: wgpu::BindGroup, buffers: Vec<Arc<wgpu::Buffer>>) -> Self {
        Self {
            bind_group,
            _buffers: buffers,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DirectDynamicBindGroupKey {
    entries: Vec<DirectDynamicBindGroupEntryKey>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct DirectDynamicBindGroupEntryKey {
    binding: u32,
    read_only: bool,
    buffer: usize,
    size: u64,
}

impl DirectDynamicBindGroupKey {
    pub fn new(entries: impl IntoIterator<Item = (u32, bool, Arc<wgpu::Buffer>)>) -> Self {
        Self {
            entries: entries
                .into_iter()
                .map(
                    |(binding, read_only, buffer)| DirectDynamicBindGroupEntryKey {
                        binding,
                        read_only,
                        buffer: Arc::as_ptr(&buffer) as usize,
                        size: buffer.size(),
                    },
                )
                .collect(),
        }
    }
}

/// Per-device caches for everything needed to compile and dispatch a kernel:
/// bind-group layouts, pipeline layouts, the unified kernel cache (naga →
/// shader → pipeline), read-only dynamic bind groups, and the wgpu on-disk
/// pipeline cache.
pub struct KernelCache {
    pub(crate) device: Arc<wgpu::Device>,
    config: Arc<crate::FusorConfig>,
    pub(crate) wgpu_cache: Option<wgpu::PipelineCache>,
    cache_file: Option<PathBuf>,
    pub(crate) kernels: RwLock<LruCache<KernelCacheKey, Arc<CachedKernel>, FxBuildHasher>>,
    pub(crate) direct_dynamic_bind_group_cache:
        RwLock<LruCache<DirectDynamicBindGroupKey, CachedDirectBindGroup, FxBuildHasher>>,
    kernel_plan_cache: KernelPlanCache,
    direct_three_buffer_bind_group_layout: OnceLock<BindGroupLayout>,
    direct_three_buffer_pipeline_layout: OnceLock<PipelineLayout>,
}

impl std::fmt::Debug for KernelCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernelCache").finish_non_exhaustive()
    }
}

fn make_lru<K: Hash + Eq, V>(size: usize) -> RwLock<LruCache<K, V, FxBuildHasher>> {
    RwLock::new(LruCache::with_hasher(
        NonZeroUsize::new(size).expect("lru cache size must be non-zero"),
        Default::default(),
    ))
}

impl KernelCache {
    pub fn new(
        device: Arc<wgpu::Device>,
        adapter: &wgpu::Adapter,
        config: Arc<crate::FusorConfig>,
    ) -> Self {
        use wgpu::PipelineCacheDescriptor;
        // tile-ir cannot see the config (the dependency points this way);
        // push the liveness/arena trace flag down at device creation.
        fusor_tile_ir::set_liveness_trace(config.trace_arena);
        let filename = wgpu::util::pipeline_cache_key(&adapter.get_info());
        let (wgpu_cache, cache_file) = if let Some(filename) =
            filename.filter(|_| device.features().contains(wgpu::Features::PIPELINE_CACHE))
        {
            let cache_dir: PathBuf = PathBuf::from(".fusor").join("pipeline_cache");
            let cache_path = cache_dir.join(&filename);
            let cache_data = std::fs::read(&cache_path).ok();
            let pipeline_cache = unsafe {
                device.create_pipeline_cache(&PipelineCacheDescriptor {
                    data: cache_data.as_deref(),
                    label: Some("Fusor ML Pipeline Cache"),
                    fallback: true,
                })
            };
            (Some(pipeline_cache), Some(cache_path))
        } else {
            (None, None)
        };

        let kernel_plan_cache = KernelPlanCache::new(config.trace_resolve_host);
        kernel_plan_cache.attach_disk(
            device_capability_fingerprint(&device),
            config.kernel_cache_dir.clone(),
        );

        Self {
            device,
            config,
            wgpu_cache,
            cache_file,
            kernels: make_lru(KERNEL_CACHE_SIZE),
            direct_dynamic_bind_group_cache: make_lru(DIRECT_DYNAMIC_BIND_GROUP_CACHE_SIZE),
            kernel_plan_cache,
            direct_three_buffer_bind_group_layout: OnceLock::new(),
            direct_three_buffer_pipeline_layout: OnceLock::new(),
        }
    }

    pub fn wgpu_device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    /// The process configuration this cache was constructed with.
    pub fn config(&self) -> &crate::FusorConfig {
        &self.config
    }

    pub fn kernel_plan_cache(&self) -> &KernelPlanCache {
        &self.kernel_plan_cache
    }

    pub fn direct_three_buffer_bind_group_layout(&self) -> BindGroupLayout {
        self.direct_three_buffer_bind_group_layout
            .get_or_init(|| {
                self.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("direct storage3 bind group layout"),
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    })
            })
            .clone()
    }

    pub fn direct_three_buffer_pipeline_layout(&self) -> PipelineLayout {
        self.direct_three_buffer_pipeline_layout
            .get_or_init(|| {
                let bind_group_layout = self.direct_three_buffer_bind_group_layout();
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("direct storage3 pipeline layout"),
                        bind_group_layouts: &[Some(&bind_group_layout)],
                        immediate_size: 0,
                    })
            })
            .clone()
    }

    pub fn create_naga_shader_module(&self, kernel: &NagaKernel) -> wgpu::ShaderModule {
        crate::note_compile(&self.config, "shader");
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(dir) = &self.config.dump_shaders {
            dump_shader(kernel, dir);
        }
        // SAFETY: all kernels avoid out-of-bounds memory access and unbounded loops.
        unsafe {
            self.device.create_shader_module_trusted(
                wgpu::ShaderModuleDescriptor {
                    label: Some("Fusor ML Shader Module"),
                    source: shader_source(kernel),
                },
                wgpu::ShaderRuntimeChecks::unchecked(),
            )
        }
    }

    /// Get the cached kernel for `key`, or build it from `naga` and insert it.
    pub fn get_or_insert_kernel(
        &self,
        key: KernelCacheKey,
        kernel: impl FnOnce() -> Arc<NagaKernel>,
    ) -> Arc<CachedKernel> {
        self.kernels
            .write()
            .get_or_insert(key, || Arc::new(CachedKernel::new(kernel())))
            .clone()
    }

    pub(crate) fn shader_for<'a>(&self, cached: &'a Arc<CachedKernel>) -> &'a wgpu::ShaderModule {
        cached
            .shader
            .get_or_init(|| self.create_naga_shader_module(cached.kernel.as_ref()))
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn shader_source(kernel: &NagaKernel) -> wgpu::ShaderSource<'static> {
    wgpu::ShaderSource::Naga(Cow::Owned(kernel.module().clone()))
}

/// Debug aid: with `FUSOR_DUMP_SHADERS=<dir>`, every compiled kernel is also
/// serialized to WGSL in that directory, named by a running counter.
#[cfg(not(target_arch = "wasm32"))]
fn dump_shader(kernel: &NagaKernel, dir: &std::path::Path) {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let _ = std::fs::create_dir_all(dir);
    let _ = std::fs::write(
        dir.join(format!("shader_{n:03}.ir.txt")),
        format!("{:#?}", kernel.module()),
    );
}

#[cfg(target_arch = "wasm32")]
fn shader_source(kernel: &NagaKernel) -> wgpu::ShaderSource<'static> {
    let mut wgsl = String::from(kernel.wgsl_extension_prelude());
    let serialized = wgpu::naga::back::wgsl::write_string(
        kernel.module(),
        kernel.info(),
        wgpu::naga::back::wgsl::WriterFlags::empty(),
    )
    .expect("validated Naga kernel should serialize to WGSL");
    wgsl.push_str(&serialized);
    wgpu::ShaderSource::Wgsl(Cow::Owned(wgsl))
}

impl Drop for KernelCache {
    fn drop(&mut self) {
        if let (Some(pipeline_cache), Some(cache_file)) =
            (self.wgpu_cache.as_ref(), self.cache_file.as_ref())
            && let Some(data) = pipeline_cache.get_data()
        {
            let temp_file = cache_file.with_extension("temp");
            let _ = std::fs::write(&temp_file, &data);
            let _ = std::fs::rename(&temp_file, cache_file);
        }
    }
}

/// Everything device-side that steers kernel codegen: feature bits, the
/// limits the lowerer consults, and the codegen-altering environment
/// switches. Persistent plans are salted by this so a capability change can
/// never replay a mismatched kernel.
fn device_capability_fingerprint(device: &wgpu::Device) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = FxHasher::default();
    format!("{:?}", device.features()).hash(&mut hasher);
    let limits = device.limits();
    limits.max_compute_workgroup_size_x.hash(&mut hasher);
    limits.max_compute_workgroup_size_y.hash(&mut hasher);
    limits.max_compute_workgroup_size_z.hash(&mut hasher);
    limits
        .max_compute_invocations_per_workgroup
        .hash(&mut hasher);
    limits
        .max_compute_workgroups_per_dimension
        .hash(&mut hasher);
    limits
        .max_storage_buffers_per_shader_stage
        .hash(&mut hasher);
    hasher.finish()
}
