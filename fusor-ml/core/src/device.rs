use std::{
    fmt::Debug,
    sync::{Arc, OnceLock},
    time::{Duration, Instant},
};

use fusor_tile_ir_runtime::{BufferPool, KernelCache};
use wgpu::{BackendOptions, Dx12BackendOptions};

use crate::{compute_graph::ComputeGraph, kernel_selection::CooperativeMatrixCaps};

const GPU_POLL_SPIN_BUDGET: Duration = Duration::from_millis(2);

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

struct DeviceInner {
    device: Arc<wgpu::Device>,
    adapter: wgpu::Adapter,
    queue: Arc<wgpu::Queue>,
    kernel_cache: KernelCache,
    buffer_pool: BufferPool,
    cooperative_matrix_caps: CooperativeMatrixCaps,
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
        let dx_compiler = wgpu::Dx12Compiler::from_env().unwrap_or_default();
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
        let cooperative_matrix_properties =
            if required_features.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
                adapter.cooperative_matrix_properties()
            } else {
                Vec::new()
            };
        let cooperative_matrix_caps = CooperativeMatrixCaps::from_properties(
            required_features,
            &cooperative_matrix_properties,
        );
        if std::env::var_os("FUSOR_TRACE_GPU_KERNELS").is_some()
            && !cooperative_matrix_properties.is_empty()
        {
            eprintln!("Fusor cooperative matrix properties: {cooperative_matrix_properties:?}");
            eprintln!("Fusor cooperative matrix caps: {cooperative_matrix_caps:?}");
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

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let kernel_cache = KernelCache::new(device.clone(), &adapter);
        let buffer_pool = BufferPool::new(device.clone(), queue.clone());

        let inner = Arc::new(DeviceInner {
            device,
            adapter,
            queue,
            kernel_cache,
            buffer_pool,
            cooperative_matrix_caps,
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

    pub(crate) fn backend(&self) -> wgpu::Backend {
        self.inner.adapter.get_info().backend
    }

    pub fn fixed_width_subgroup_size(&self) -> Option<u32> {
        if !self.subgroups_supported() {
            return None;
        }

        let min = self.min_subgroup_size();
        let max = self.max_subgroup_size();
        if min == max && matches!(min, 4 | 8 | 16 | 32 | 64) {
            return Some(min);
        }

        // Apple GPUs execute subgroup operations with 32-wide SIMD groups even
        // though wgpu reports the broader Metal range.
        if self.backend() == wgpu::Backend::Metal && min <= 32 && max >= 32 {
            return Some(32);
        }

        None
    }

    pub fn f16_supported(&self) -> bool {
        self.features().contains(wgpu::Features::SHADER_F16)
    }

    pub(crate) fn cooperative_matrix_caps(&self) -> CooperativeMatrixCaps {
        self.inner.cooperative_matrix_caps
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

    pub(crate) fn kernel_cache(&self) -> &KernelCache {
        &self.inner.kernel_cache
    }

    /// Reset the initialized flag on all cached buffers.
    pub fn reset_initialized_buffers(&self) {
        self.inner.buffer_pool.reset_initialized_buffers();
    }

    /// Get or create a buffer of the specified size.
    pub fn create_buffer(&self, size: u64, usage: wgpu::BufferUsages) -> Arc<wgpu::Buffer> {
        self.inner.buffer_pool.create_buffer(size, usage)
    }

    /// Get or create a buffer of the specified size.
    pub fn create_buffer_init(&self, data: &[u8], usage: wgpu::BufferUsages) -> Arc<wgpu::Buffer> {
        self.inner.buffer_pool.create_buffer_init(data, usage)
    }

    /// Get or create a buffer of the specified size.
    pub fn create_buffer_init_iter(
        &self,
        data: impl IntoIterator<Item = u8>,
        usage: wgpu::BufferUsages,
        len: u64,
    ) -> Arc<wgpu::Buffer> {
        self.inner
            .buffer_pool
            .create_buffer_init_iter(data, usage, len)
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
