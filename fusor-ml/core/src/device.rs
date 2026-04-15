use std::{
    borrow::Cow,
    fmt::Debug,
    num::{NonZeroU64, NonZeroUsize},
    path::PathBuf,
    sync::{Arc, OnceLock},
};

use lru::LruCache;
use parking_lot::RwLock;
use rustc_hash::FxBuildHasher;
use wgpu::{
    BackendOptions, BindGroupLayout, BufferUsages, Dx12BackendOptions, PipelineLayout, ShaderModule,
};

use crate::compute_graph::ComputeGraph;

#[derive(Debug)]
struct CachedBuffer {
    writen: bool,
    buffer: Arc<wgpu::Buffer>,
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

struct DeviceInner {
    device: wgpu::Device,
    adapter: wgpu::Adapter,
    queue: wgpu::Queue,
    cache: Option<wgpu::PipelineCache>,
    cache_file: Option<PathBuf>,
    bind_group_layout_cache:
        RwLock<LruCache<Vec<wgpu::BindGroupLayoutEntry>, BindGroupLayout, FxBuildHasher>>,
    pipeline_layout_cache: RwLock<LruCache<BindGroupLayout, wgpu::PipelineLayout, FxBuildHasher>>,
    shader_module_cache: RwLock<LruCache<String, wgpu::ShaderModule, FxBuildHasher>>,
    compute_pipeline_cache:
        RwLock<LruCache<(PipelineLayout, ShaderModule), wgpu::ComputePipeline, FxBuildHasher>>,
    // Cache for buffer allocations, keyed by size in bytes
    buffer_allocation_cache:
        RwLock<LruCache<(u64, BufferUsages), Vec<CachedBuffer>, FxBuildHasher>>,
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
        let adapter = wgpu::util::initialize_adapter_from_env_or_default(&instance, None)
            .await
            .expect("Failed to find a suitable GPU adapter");
        let mut required_features = wgpu::Features::empty();
        if adapter.features().contains(wgpu::Features::SUBGROUP) {
            required_features |= wgpu::Features::SUBGROUP;
        }
        if adapter.features().contains(wgpu::Features::SHADER_F16) {
            required_features |= wgpu::Features::SHADER_F16;
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Fusor ML Device"),
                required_features,
                required_limits: adapter.limits(),
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

        let cache_size = const { NonZeroUsize::new(2048).unwrap() };
        let bind_group_layout_cache =
            RwLock::new(LruCache::with_hasher(cache_size, Default::default()));
        let pipeline_layout_cache =
            RwLock::new(LruCache::with_hasher(cache_size, Default::default()));
        let shader_module_cache =
            RwLock::new(LruCache::with_hasher(cache_size, Default::default()));
        let compute_pipeline_cache =
            RwLock::new(LruCache::with_hasher(cache_size, Default::default()));
        let buffer_allocation_cache = RwLock::new(LruCache::with_hasher(
            const { NonZeroUsize::new(128).unwrap() },
            Default::default(),
        ));

        let inner = Arc::new(DeviceInner {
            device,
            adapter,
            queue,
            cache,
            cache_file,
            bind_group_layout_cache,
            pipeline_layout_cache,
            shader_module_cache,
            compute_pipeline_cache,
            buffer_allocation_cache,
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
                let result = inner.device.poll(wgpu::PollType::wait_indefinitely());
                drop(inner);
                let Ok(status) = result else {
                    break;
                };
                if status == wgpu::PollStatus::QueueEmpty {
                    std::thread::sleep(std::time::Duration::from_nanos(10));
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

    pub(crate) fn create_shader_module<'a>(
        &self,
        source: impl Into<Cow<'a, str>>,
    ) -> wgpu::ShaderModule {
        // SAFTEY: All kernels don't access memory outside of bounds and don't have unbounded loops
        unsafe {
            self.inner.device.create_shader_module_trusted(
                wgpu::ShaderModuleDescriptor {
                    label: Some("Fusor ML Shader Module"),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
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

    pub fn wgpu_adapter(&self) -> &wgpu::Adapter {
        &self.inner.adapter
    }

    pub fn wgpu_device(&self) -> &wgpu::Device {
        &self.inner.device
    }

    pub(crate) fn wgpu_queue(&self) -> &wgpu::Queue {
        &self.inner.queue
    }

    pub(crate) fn wgpu_cache(&self) -> Option<&wgpu::PipelineCache> {
        self.inner.cache.as_ref()
    }

    pub(crate) fn wgpu_cache_file(&self) -> Option<&PathBuf> {
        self.inner.cache_file.as_ref()
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

    pub(crate) fn shader_module_cache(
        &self,
    ) -> &RwLock<LruCache<String, wgpu::ShaderModule, FxBuildHasher>> {
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
        let mut cache = self.inner.buffer_allocation_cache.write();
        for (_, buffers) in cache.iter_mut() {
            for buffer in buffers {
                buffer.writen = false;
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
                buffer
            })
    }

    /// Get or create a buffer of the specified size.
    pub fn create_buffer(&self, size: u64, usage: wgpu::BufferUsages) -> Arc<wgpu::Buffer> {
        self.create_buffer_inner(size, usage, false)
    }

    /// Get or create a buffer of the specified size.
    pub fn create_buffer_init(&self, data: &[u8], usage: wgpu::BufferUsages) -> Arc<wgpu::Buffer> {
        let buffer = self.create_buffer_inner(data.len() as u64, usage, true);
        self.wgpu_queue().write_buffer(&buffer, 0, data);
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
        let buffer = self.create_buffer_inner(len, usage, true);
        if let Some(len) = NonZeroU64::new(buffer.size()) {
            if let Some(mut write) = self.wgpu_queue().write_buffer_with(&buffer, 0, len) {
                write.iter_mut().zip(&mut iter).for_each(|(a, b)| *a = b);
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
}

#[cfg(test)]
impl Device {
    /// Get a shared device for tests. This reuses the same device across all tests
    /// to prevent resource exhaustion issues on some GPU drivers.
    ///
    /// Note: This must be called outside of an async context to avoid deadlocks.
    /// Use it at the start of tests before awaiting anything else.
    pub fn test_instance() -> Self {
        #[cfg(target_os = "macos")]
        {
            pollster::block_on(Device::new()).expect("Failed to create test device")
        }
        #[cfg(not(target_os = "macos"))]
        {
            /// Shared device for tests to avoid creating too many GPU devices.
            /// On some drivers (especially NVIDIA on Windows), creating many GPU devices
            /// in quick succession can cause crashes or resource exhaustion.
            static TEST_DEVICE: OnceLock<Device> = OnceLock::new();
            TEST_DEVICE
                .get_or_init(|| {
                    pollster::block_on(Device::new()).expect("Failed to create test device")
                })
                .clone()
        }
    }
}
