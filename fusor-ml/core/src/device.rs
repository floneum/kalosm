use std::{
    borrow::Cow,
    fmt::Debug,
    num::{NonZeroU64, NonZeroUsize},
    path::PathBuf,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use lru::LruCache;
use parking_lot::{Mutex, RwLock};
use rustc_hash::FxBuildHasher;
use wgpu::{
    BackendOptions, BindGroupLayout, BufferUsages, COPY_BUFFER_ALIGNMENT, Dx12BackendOptions,
    PipelineLayout, ShaderModule,
};

use crate::{compute_graph::ComputeGraph, mir::kernel_backend::KernelCacheKey};

#[derive(Debug)]
struct CachedBuffer {
    writen: bool,
    buffer: Arc<wgpu::Buffer>,
}

const MAX_FREE_BUFFERS_PER_BUCKET: usize = 4;
const BIND_GROUP_LAYOUT_CACHE_SIZE: usize = 256;
const PIPELINE_LAYOUT_CACHE_SIZE: usize = 256;
const NAGA_MODULE_CACHE_SIZE: usize = 128;
const SHADER_MODULE_CACHE_SIZE: usize = 128;
const COMPUTE_PIPELINE_CACHE_SIZE: usize = 128;
const DIRECT_STORAGE3_BIND_GROUP_CACHE_SIZE: usize = 4096;
const DIRECT_DYNAMIC_BIND_GROUP_CACHE_SIZE: usize = 4096;
const GPU_POLL_SPIN_BUDGET: Duration = Duration::from_millis(2);

fn padded_copy_size(size: u64) -> u64 {
    let align_mask = COPY_BUFFER_ALIGNMENT - 1;
    ((size + align_mask) & !align_mask).max(COPY_BUFFER_ALIGNMENT)
}

fn poll_until_queue_empty(device: &wgpu::Device) -> Result<wgpu::PollStatus, wgpu::PollError> {
    let start = Instant::now();
    loop {
        let status = device.poll(wgpu::PollType::Poll)?;
        if status.is_queue_empty() {
            return Ok(status);
        }
        if start.elapsed() >= GPU_POLL_SPIN_BUDGET {
            return device.poll(wgpu::PollType::wait_indefinitely());
        }
        std::thread::yield_now();
    }
}

async fn select_adapter(
    instance: &wgpu::Instance,
    backends: wgpu::Backends,
) -> Result<wgpu::Adapter, crate::Error> {
    let desired_adapter_name = std::env::var("WGPU_ADAPTER_NAME")
        .ok()
        .map(|name| name.to_ascii_lowercase());

    let mut adapters = instance.enumerate_adapters(backends).await;
    if let Some(desired_adapter_name) = desired_adapter_name {
        return adapters
            .into_iter()
            .find(|adapter| {
                adapter
                    .get_info()
                    .name
                    .to_ascii_lowercase()
                    .contains(&desired_adapter_name)
            })
            .ok_or_else(|| {
                crate::Error::msg(format!(
                    "WGPU_ADAPTER_NAME={desired_adapter_name:?} did not match any available adapter"
                ))
            });
    }

    if !adapters.is_empty() {
        adapters.sort_by_key(adapter_preference_rank);
        return Ok(adapters.remove(0));
    }

    let preferred = wgpu::PowerPreference::from_env().unwrap_or_default();
    let mut last_error = None;
    for power_preference in [
        preferred,
        wgpu::PowerPreference::HighPerformance,
        wgpu::PowerPreference::LowPower,
        wgpu::PowerPreference::None,
    ] {
        match instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
        {
            Ok(adapter) => return Ok(adapter),
            Err(error) => last_error = Some(error),
        }
    }

    let detail = last_error
        .map(|error| error.to_string())
        .unwrap_or_else(|| "no adapter returned".to_string());
    Err(crate::Error::msg(format!(
        "failed to find a suitable GPU adapter: {detail}"
    )))
}

fn adapter_preference_rank(adapter: &wgpu::Adapter) -> u8 {
    match adapter.get_info().device_type {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::VirtualGpu => 2,
        wgpu::DeviceType::Other => 3,
        wgpu::DeviceType::Cpu => 4,
    }
}

impl CachedBuffer {
    fn new(buffer: Arc<wgpu::Buffer>, writen: bool) -> Self {
        Self { writen, buffer }
    }

    fn initialized(&self) -> bool {
        self.writen
    }

    fn set_initialized(&mut self) {
        self.writen = true;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct DirectStorage3BindGroupKey {
    input: usize,
    weight: usize,
    output: usize,
}

impl DirectStorage3BindGroupKey {
    pub(crate) fn new(
        input: &Arc<wgpu::Buffer>,
        weight: &Arc<wgpu::Buffer>,
        output: &Arc<wgpu::Buffer>,
    ) -> Self {
        Self {
            input: Arc::as_ptr(input) as usize,
            weight: Arc::as_ptr(weight) as usize,
            output: Arc::as_ptr(output) as usize,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct DirectDynamicBindGroupKey {
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
    pub(crate) fn new(entries: impl IntoIterator<Item = (u32, bool, Arc<wgpu::Buffer>)>) -> Self {
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

struct DeviceInner {
    device: wgpu::Device,
    adapter: wgpu::Adapter,
    queue: wgpu::Queue,
    cache: Option<wgpu::PipelineCache>,
    cache_file: Option<PathBuf>,
    bind_group_layout_cache:
        RwLock<LruCache<Vec<wgpu::BindGroupLayoutEntry>, BindGroupLayout, FxBuildHasher>>,
    pipeline_layout_cache: RwLock<LruCache<BindGroupLayout, wgpu::PipelineLayout, FxBuildHasher>>,
    naga_module_cache: RwLock<LruCache<KernelCacheKey, wgpu::naga::Module, FxBuildHasher>>,
    shader_module_cache: RwLock<LruCache<KernelCacheKey, wgpu::ShaderModule, FxBuildHasher>>,
    compute_pipeline_cache:
        RwLock<LruCache<(PipelineLayout, ShaderModule), wgpu::ComputePipeline, FxBuildHasher>>,
    direct_dynamic_bind_group_cache:
        RwLock<LruCache<DirectDynamicBindGroupKey, wgpu::BindGroup, FxBuildHasher>>,
    direct_three_buffer_bind_group_cache:
        RwLock<LruCache<DirectStorage3BindGroupKey, wgpu::BindGroup, FxBuildHasher>>,
    direct_three_buffer_bind_group_layout: OnceLock<BindGroupLayout>,
    direct_three_buffer_pipeline_layout: OnceLock<PipelineLayout>,
    // Cache for buffer allocations, keyed by size in bytes
    buffer_allocation_cache:
        RwLock<LruCache<(u64, BufferUsages), Vec<CachedBuffer>, FxBuildHasher>>,
    initialized_buffers_dirty: AtomicBool,
    initialized_buffer_keys: Mutex<Vec<(u64, BufferUsages)>>,
    // Single compute graph shared by all tensors on this device
    compute_graph: OnceLock<ComputeGraph>,
}

impl Debug for DeviceInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceInner")
            .field("device", &self.device)
            .field("queue", &self.queue)
            .finish()
    }
}

impl Drop for DeviceInner {
    fn drop(&mut self) {
        // Flush pipeline cache to disk on shutdown
        if let (Some(pipeline_cache), Some(cache_file)) =
            (self.cache.as_ref(), self.cache_file.as_ref())
            && let Some(data) = pipeline_cache.get_data()
        {
            let temp_file = cache_file.with_extension("temp");
            let _ = std::fs::write(&temp_file, &data);
            let _ = std::fs::rename(&temp_file, cache_file);
        }
    }
}

/// A weak reference to a [`Device`] that does not prevent cleanup.
///
/// Used internally to break reference cycles (e.g., between Device and ComputeGraph).
#[derive(Clone, Debug)]
pub struct WeakDevice {
    inner: std::sync::Weak<DeviceInner>,
}

impl WeakDevice {
    /// Attempt to upgrade to a strong [`Device`] reference.
    /// Returns `None` if the device has already been dropped.
    pub fn upgrade(&self) -> Option<Device> {
        self.inner.upgrade().map(|inner| Device { inner })
    }
}

#[derive(Clone, Debug)]
pub struct Device {
    inner: Arc<DeviceInner>,
}

impl Device {
    pub async fn new() -> Result<Self, crate::Error> {
        let dx_compiler = wgpu::Dx12Compiler::from_env().unwrap_or(wgpu::Dx12Compiler::StaticDxc);
        let backends = wgpu::Backends::from_env().unwrap_or(wgpu::Backends::all());
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            backend_options: BackendOptions {
                dx12: Dx12BackendOptions {
                    shader_compiler: dx_compiler,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        });
        let adapter = select_adapter(&instance, backends).await?;
        let adapter_features = adapter.features();
        let mut required_features = wgpu::Features::empty();
        if adapter_features.contains(wgpu::Features::SUBGROUP) {
            required_features |= wgpu::Features::SUBGROUP;
        }
        if adapter_features.contains(wgpu::Features::SHADER_F16) {
            required_features |= wgpu::Features::SHADER_F16;
        }
        if std::env::var_os("FUSOR_TRACE_GPU_KERNELS").is_some() {
            if adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY) {
                required_features |= wgpu::Features::TIMESTAMP_QUERY;
                if adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES) {
                    required_features |= wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
                }
            } else {
                eprintln!(
                    "FUSOR_TRACE_GPU_KERNELS requested, but adapter does not support timestamp queries"
                );
            }
        }
        let mut experimental_features = wgpu::ExperimentalFeatures::default();
        if adapter_features.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
            required_features |= wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX;
            // SAFETY: cooperative matrix is an experimental feature that requires opting in
            experimental_features = unsafe { wgpu::ExperimentalFeatures::enabled() };
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Fusor ML Device"),
                required_features,
                required_limits: adapter.limits(),
                experimental_features,
                ..Default::default()
            })
            .await?;

        use wgpu::PipelineCacheDescriptor;
        let filename = wgpu::util::pipeline_cache_key(&adapter.get_info());
        let (cache, cache_file) = if let Some(filename) =
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

        let bind_group_layout_cache = RwLock::new(LruCache::with_hasher(
            NonZeroUsize::new(BIND_GROUP_LAYOUT_CACHE_SIZE).unwrap(),
            Default::default(),
        ));
        let pipeline_layout_cache = RwLock::new(LruCache::with_hasher(
            NonZeroUsize::new(PIPELINE_LAYOUT_CACHE_SIZE).unwrap(),
            Default::default(),
        ));
        let naga_module_cache = RwLock::new(LruCache::with_hasher(
            NonZeroUsize::new(NAGA_MODULE_CACHE_SIZE).unwrap(),
            Default::default(),
        ));
        let shader_module_cache = RwLock::new(LruCache::with_hasher(
            NonZeroUsize::new(SHADER_MODULE_CACHE_SIZE).unwrap(),
            Default::default(),
        ));
        let compute_pipeline_cache = RwLock::new(LruCache::with_hasher(
            NonZeroUsize::new(COMPUTE_PIPELINE_CACHE_SIZE).unwrap(),
            Default::default(),
        ));
        let direct_dynamic_bind_group_cache = RwLock::new(LruCache::with_hasher(
            NonZeroUsize::new(DIRECT_DYNAMIC_BIND_GROUP_CACHE_SIZE).unwrap(),
            Default::default(),
        ));
        let direct_three_buffer_bind_group_cache = RwLock::new(LruCache::with_hasher(
            NonZeroUsize::new(DIRECT_STORAGE3_BIND_GROUP_CACHE_SIZE).unwrap(),
            Default::default(),
        ));
        let buffer_allocation_cache = RwLock::new(LruCache::with_hasher(
            const { NonZeroUsize::new(128).unwrap() },
            Default::default(),
        ));
        let initialized_buffer_keys = Mutex::new(Vec::new());

        let inner = Arc::new(DeviceInner {
            device,
            adapter,
            queue,
            cache,
            cache_file,
            bind_group_layout_cache,
            pipeline_layout_cache,
            naga_module_cache,
            shader_module_cache,
            compute_pipeline_cache,
            direct_dynamic_bind_group_cache,
            direct_three_buffer_bind_group_cache,
            direct_three_buffer_bind_group_layout: OnceLock::new(),
            direct_three_buffer_pipeline_layout: OnceLock::new(),
            buffer_allocation_cache,
            initialized_buffers_dirty: AtomicBool::new(false),
            initialized_buffer_keys,
            compute_graph: OnceLock::new(),
        });

        let device = Device {
            inner: inner.clone(),
        };

        // Initialize the compute graph now that we have a valid device
        inner
            .compute_graph
            .set(ComputeGraph::new(&device))
            .ok()
            .expect("compute_graph should only be set once");

        let device = Device { inner };

        #[cfg(not(target_arch = "wasm32"))]
        std::thread::spawn({
            let weak_inner = Arc::downgrade(&device.inner);
            move || loop {
                let Some(inner) = weak_inner.upgrade() else {
                    break;
                };
                let result = poll_until_queue_empty(&inner.device);
                drop(inner);
                let Ok(status) = result else {
                    break;
                };
                if status == wgpu::PollStatus::QueueEmpty {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }
        });

        Ok(device)
    }

    /// Create a weak reference to this device that doesn't prevent cleanup.
    pub fn downgrade(&self) -> WeakDevice {
        WeakDevice {
            inner: Arc::downgrade(&self.inner),
        }
    }

    pub fn create_naga_shader_module(&self, module: wgpu::naga::Module) -> wgpu::ShaderModule {
        // SAFETY: all kernels avoid out-of-bounds memory access and unbounded loops.
        unsafe {
            self.inner.device.create_shader_module_trusted(
                wgpu::ShaderModuleDescriptor {
                    label: Some("Fusor ML Shader Module"),
                    source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
                },
                wgpu::ShaderRuntimeChecks::unchecked(),
            )
        }
    }

    pub fn limits(&self) -> wgpu::Limits {
        self.inner.adapter.limits()
    }

    pub fn features(&self) -> wgpu::Features {
        self.inner.device.features()
    }

    pub fn subgroups_supported(&self) -> bool {
        self.features().contains(wgpu::Features::SUBGROUP)
    }

    pub fn min_subgroup_size(&self) -> u32 {
        self.inner.adapter.get_info().subgroup_min_size
    }

    pub fn max_subgroup_size(&self) -> u32 {
        self.inner.adapter.get_info().subgroup_max_size
    }

    pub fn f16_supported(&self) -> bool {
        self.features().contains(wgpu::Features::SHADER_F16)
    }

    pub fn cooperative_matrix_supported(&self) -> bool {
        self.features()
            .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
    }

    pub fn wgpu_adapter(&self) -> &wgpu::Adapter {
        &self.inner.adapter
    }

    pub fn wgpu_device(&self) -> &wgpu::Device {
        &self.inner.device
    }

    pub fn wgpu_queue(&self) -> &wgpu::Queue {
        &self.inner.queue
    }

    pub(crate) fn is_same_device(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Block until all submitted GPU work has completed.
    pub fn poll_wait(&self) {
        poll_until_queue_empty(&self.inner.device).expect("Failed to poll GPU device");
    }

    pub(crate) fn wgpu_cache(&self) -> Option<&wgpu::PipelineCache> {
        self.inner.cache.as_ref()
    }

    pub(crate) fn bind_group_layout_cache(
        &self,
    ) -> &RwLock<LruCache<Vec<wgpu::BindGroupLayoutEntry>, BindGroupLayout, FxBuildHasher>> {
        &self.inner.bind_group_layout_cache
    }

    pub(crate) fn pipeline_layout_cache(
        &self,
    ) -> &RwLock<LruCache<BindGroupLayout, wgpu::PipelineLayout, FxBuildHasher>> {
        &self.inner.pipeline_layout_cache
    }

    pub(crate) fn direct_three_buffer_bind_group_layout(&self) -> BindGroupLayout {
        self.inner
            .direct_three_buffer_bind_group_layout
            .get_or_init(|| {
                self.wgpu_device()
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

    pub(crate) fn direct_three_buffer_bind_group_cache(
        &self,
    ) -> &RwLock<LruCache<DirectStorage3BindGroupKey, wgpu::BindGroup, FxBuildHasher>> {
        &self.inner.direct_three_buffer_bind_group_cache
    }

    pub(crate) fn direct_dynamic_bind_group_cache(
        &self,
    ) -> &RwLock<LruCache<DirectDynamicBindGroupKey, wgpu::BindGroup, FxBuildHasher>> {
        &self.inner.direct_dynamic_bind_group_cache
    }

    pub(crate) fn direct_three_buffer_pipeline_layout(&self) -> PipelineLayout {
        self.inner
            .direct_three_buffer_pipeline_layout
            .get_or_init(|| {
                let bind_group_layout = self.direct_three_buffer_bind_group_layout();
                self.wgpu_device()
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("direct storage3 pipeline layout"),
                        bind_group_layouts: &[Some(&bind_group_layout)],
                        immediate_size: 0,
                    })
            })
            .clone()
    }

    pub(crate) fn naga_module_cache(
        &self,
    ) -> &RwLock<LruCache<KernelCacheKey, wgpu::naga::Module, FxBuildHasher>> {
        &self.inner.naga_module_cache
    }

    pub(crate) fn shader_module_cache(
        &self,
    ) -> &RwLock<LruCache<KernelCacheKey, wgpu::ShaderModule, FxBuildHasher>> {
        &self.inner.shader_module_cache
    }

    pub(crate) fn compute_pipeline_cache(
        &self,
    ) -> &RwLock<LruCache<(PipelineLayout, ShaderModule), wgpu::ComputePipeline, FxBuildHasher>>
    {
        &self.inner.compute_pipeline_cache
    }

    /// Reset the initialized flag on all cached buffers.
    pub fn reset_initialized_buffers(&self) {
        if !self
            .inner
            .initialized_buffers_dirty
            .swap(false, Ordering::AcqRel)
        {
            return;
        }
        let keys = {
            let mut keys = self.inner.initialized_buffer_keys.lock();
            std::mem::take(&mut *keys)
        };
        let mut cache = self.inner.buffer_allocation_cache.write();
        for key in keys {
            if let Some(buffers) = cache.get_mut(&key) {
                for buffer in buffers.iter_mut() {
                    buffer.writen = false;
                }
                prune_cached_buffers(buffers);
            }
        }
    }

    /// Try to get a buffer from the allocation cache. Returns None if no buffer of the requested size is available.
    pub(crate) fn get_cached_buffer(
        &self,
        size: u64,
        usage: wgpu::BufferUsages,
        to_initilize: bool,
    ) -> Option<Arc<wgpu::Buffer>> {
        let mut cache = self.inner.buffer_allocation_cache.write();
        let items = cache.get_mut(&(size, usage))?;
        items.iter_mut().find_map(|a| {
            if Arc::strong_count(&a.buffer) == 1 {
                if to_initilize {
                    if a.initialized() {
                        return None;
                    }
                    a.set_initialized();
                }
                Some(a.buffer.clone())
            } else {
                None
            }
        })
    }

    /// Get or create a buffer of the specified size for a use
    fn create_buffer_inner(
        &self,
        size: u64,
        usage: wgpu::BufferUsages,
        to_initilize: bool,
    ) -> Arc<wgpu::Buffer> {
        if to_initilize {
            self.inner
                .initialized_buffers_dirty
                .store(true, Ordering::Release);
            self.inner
                .initialized_buffer_keys
                .lock()
                .push((size, usage));
        }
        // Try to get a buffer from the cache first
        self.get_cached_buffer(size, usage, to_initilize)
            .unwrap_or_else(|| {
                let new_buffer = self.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Tensor Buffer"),
                    size,
                    usage,
                    mapped_at_creation: false,
                });

                let buffer = Arc::new(new_buffer);
                self.inner
                    .buffer_allocation_cache
                    .write()
                    .get_or_insert_mut((size, usage), Vec::new)
                    .push(CachedBuffer::new(buffer.clone(), to_initilize));
                if let Some(buffers) = self
                    .inner
                    .buffer_allocation_cache
                    .write()
                    .get_mut(&(size, usage))
                {
                    prune_cached_buffers(buffers);
                }
                buffer
            })
    }

    /// Get or create a buffer of the specified size.
    pub fn create_buffer(&self, size: u64, usage: wgpu::BufferUsages) -> Arc<wgpu::Buffer> {
        self.create_buffer_inner(size, usage, false)
    }

    /// Get or create a buffer of the specified size.
    pub fn create_buffer_init(&self, data: &[u8], usage: wgpu::BufferUsages) -> Arc<wgpu::Buffer> {
        let padded_len = padded_copy_size(data.len() as u64);
        let buffer = self.create_buffer_inner(padded_len, usage, true);
        let mut write = self
            .wgpu_queue()
            .write_buffer_with(&buffer, 0, NonZeroU64::new(padded_len).unwrap())
            .expect("failed to map buffer for writing");
        write[..data.len()].copy_from_slice(data);
        write[data.len()..].fill(0);
        buffer
    }

    /// Get or create a buffer of the specified size.
    pub fn create_buffer_init_iter(
        &self,
        data: impl IntoIterator<Item = u8>,
        usage: wgpu::BufferUsages,
        len: u64,
    ) -> Arc<wgpu::Buffer> {
        let mut iter = data.into_iter();
        let padded_len = padded_copy_size(len);
        let buffer = self.create_buffer_inner(padded_len, usage, true);
        if let Some(len) = NonZeroU64::new(buffer.size()) {
            if let Some(mut write) = self.wgpu_queue().write_buffer_with(&buffer, 0, len) {
                for byte in write.iter_mut() {
                    *byte = iter.next().unwrap_or(0);
                }
            } else {
                panic!("Failed to map buffer for writing");
            }
        } else {
            panic!("Failed to map buffer for writing");
        }
        buffer
    }

    pub(crate) fn compute_graph(&self) -> &ComputeGraph {
        self.inner
            .compute_graph
            .get()
            .expect("compute_graph should be initialized")
    }

    /// Resolve multiple compute-graph nodes in a single pass. All targets share
    /// one execution graph so intermediate results can be freed as soon as every
    /// consumer within the batch has been computed. This keeps peak GPU memory
    /// much lower than resolving targets one-by-one.
    pub fn resolve_batch(&self, keys: &[crate::compute_graph::NodeIndex]) -> usize {
        self.compute_graph().resolve_batch(keys)
    }

    pub fn detach_cached(&self, keys: &[crate::compute_graph::NodeIndex]) {
        self.compute_graph().detach_cached(keys)
    }
}

fn prune_cached_buffers(buffers: &mut Vec<CachedBuffer>) {
    let mut kept_free_buffers = 0;
    buffers.retain(|cached| {
        let is_free = Arc::strong_count(&cached.buffer) == 1;
        if !is_free {
            return true;
        }

        if kept_free_buffers < MAX_FREE_BUFFERS_PER_BUCKET {
            kept_free_buffers += 1;
            true
        } else {
            false
        }
    });
}
