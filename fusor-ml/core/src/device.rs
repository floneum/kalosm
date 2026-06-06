use std::{
    fmt::Debug,
    sync::{Arc, OnceLock},
};

use fusor_tile_ir::{CoopMatrixToken, SubgroupToken};
use fusor_tile_ir_kernels::SubgroupConfig;
use fusor_tile_ir_runtime::{BufferPool, KernelCache};
use wgpu::{BackendOptions, Dx12BackendOptions};

use crate::{
    compute_graph::{ComputeGraph, DecodePlanCache},
    kernel_selection::{CooperativeMatrixCaps, CooperativeMatrixKind},
};

#[cfg(not(target_arch = "wasm32"))]
use web_time::{Duration, Instant};

#[cfg(not(target_arch = "wasm32"))]
const GPU_POLL_SPIN_BUDGET: Duration = Duration::from_millis(2);

#[cfg(not(target_arch = "wasm32"))]
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
        _ => 3,
    }
}

fn log_gpu_diagnostic(message: String) {
    #[cfg(target_arch = "wasm32")]
    web_sys::console::error_1(&message.into());
    #[cfg(not(target_arch = "wasm32"))]
    eprintln!("{message}");
}

fn install_device_diagnostics(device: &wgpu::Device) {
    device.set_device_lost_callback(|reason, message| {
        log_gpu_diagnostic(format!(
            "fusor: WebGPU device lost reason={reason:?} message={message:?}"
        ));
    });

    device.on_uncaptured_error(Arc::new(|error| {
        let message = match &error {
            wgpu::Error::OutOfMemory { source } => {
                format!("fusor: uncaptured WebGPU out-of-memory error: {source}")
            }
            wgpu::Error::Validation {
                description,
                source,
            } => {
                format!("fusor: uncaptured WebGPU validation error: {description}; source={source}")
            }
            wgpu::Error::Internal {
                description,
                source,
            } => {
                format!("fusor: uncaptured WebGPU internal error: {description}; source={source}")
            }
        };
        log_gpu_diagnostic(message);
    }));
}

struct DeviceInner {
    device: Arc<wgpu::Device>,
    adapter: wgpu::Adapter,
    /// Cached `adapter.get_info()` / `adapter.limits()`. These are constant for
    /// the device's lifetime; re-querying them per kernel build (every op, every
    /// token) is pure overhead — and on wasm each `get_info()` is a JS round-trip
    /// that allocates an `AdapterInfo`.
    adapter_info: wgpu::AdapterInfo,
    limits: wgpu::Limits,
    queue: Arc<wgpu::Queue>,
    kernel_cache: KernelCache,
    buffer_pool: BufferPool,
    /// Memoizes per-op `build_kernel` results across structurally-identical
    /// decode tokens (see [`DecodePlanCache`]). The dominant per-token host
    /// cost on the web build.
    decode_plan_cache: DecodePlanCache,
    cooperative_matrix_caps: CooperativeMatrixCaps,
    compute_graph: OnceLock<ComputeGraph>,
    /// When set, this device reports `subgroups_supported() == false` so kernel
    /// selection picks the no-subgroup fallbacks. A property of the device, so
    /// it survives the `WeakDevice` upgrade kernel selection goes through.
    disable_subgroups: bool,
    /// When set, kernel-output/scratch buffers allocated on this device are
    /// pre-filled with a poison pattern instead of left zeroed, reproducing the
    /// app's reused buffer pool. A property of the device for the same reason.
    poison_allocations: bool,
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
    /// Construct a sibling device that shares the same wgpu device, queue and
    /// adapter but carries its own flags, kernel cache, buffer pool and compute
    /// graph. Used to derive the no-subgroup / poisoned-allocation test devices
    /// without re-initializing the GPU.
    fn derive(&self, disable_subgroups: bool, poison_allocations: bool) -> Self {
        let src = &self.inner;
        let device = src.device.clone();
        let queue = src.queue.clone();
        let adapter = src.adapter.clone();
        let kernel_cache = KernelCache::new(device.clone(), &adapter);
        let buffer_pool = BufferPool::new(device.clone(), queue.clone());
        let inner = Arc::new(DeviceInner {
            device,
            adapter,
            adapter_info: src.adapter_info.clone(),
            limits: src.limits.clone(),
            queue,
            kernel_cache,
            buffer_pool,
            decode_plan_cache: DecodePlanCache::new(),
            cooperative_matrix_caps: src.cooperative_matrix_caps,
            compute_graph: OnceLock::new(),
            disable_subgroups,
            poison_allocations,
        });
        let device = Device {
            inner: inner.clone(),
        };
        inner
            .compute_graph
            .set(ComputeGraph::new(&device))
            .ok()
            .expect("compute_graph should only be set once");
        Device { inner }
    }

    /// Return a sibling device that reports no subgroup support, so the
    /// no-subgroup kernel fallbacks (the only path the web build takes) are
    /// exercised. Shares the underlying wgpu device with `self`.
    pub fn without_subgroups(&self) -> Self {
        self.derive(true, self.inner.poison_allocations)
    }

    /// Return a sibling device whose tensor allocations poison kernel-output
    /// buffers before use, reproducing the app's reused buffer pool. Shares the
    /// underlying wgpu device with `self`.
    pub fn with_poisoned_allocations(&self) -> Self {
        self.derive(self.inner.disable_subgroups, true)
    }

    /// Whether tensor allocations on this device are poisoned.
    pub fn poisons_allocations(&self) -> bool {
        self.inner.poison_allocations
    }

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
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = select_adapter(&instance, backends).await?;
        let adapter_features = adapter.features();
        #[cfg(target_arch = "wasm32")]
        {
            let info = adapter.get_info();
            web_sys::console::log_1(
                &format!(
                    "fusor: adapter subgroups={} subgroup_min={} subgroup_max={} shader_f16={} backend={:?} name={:?} (note: the wasm build never requests wgpu::Features::SUBGROUP, so subgroups_supported() stays false regardless of adapter support)",
                    adapter_features.contains(wgpu::Features::SUBGROUP),
                    info.subgroup_min_size,
                    info.subgroup_max_size,
                    adapter_features.contains(wgpu::Features::SHADER_F16),
                    info.backend,
                    info.name,
                )
                .into(),
            );
        }
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
        #[cfg(not(target_arch = "wasm32"))]
        let experimental_features = {
            let mut features = wgpu::ExperimentalFeatures::default();
            if adapter_features.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
                required_features |= wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX;
                // SAFETY: cooperative matrix is an experimental feature that requires opting in
                features = unsafe { wgpu::ExperimentalFeatures::enabled() };
            }
            features
        };
        #[cfg(target_arch = "wasm32")]
        let experimental_features = wgpu::ExperimentalFeatures::default();
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

        install_device_diagnostics(&device);

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let kernel_cache = KernelCache::new(device.clone(), &adapter);
        let buffer_pool = BufferPool::new(device.clone(), queue.clone());

        // `FUSOR_DISABLE_SUBGROUPS` / `FUSOR_DIRTY_BUFFERS` are construction-time
        // defaults for the device flags, so a plain `Device::gpu()` from a repro
        // binary reproduces the web path without code changes. Tests instead
        // derive `without_subgroups()` / `with_poisoned_allocations()` sibling
        // devices explicitly. `var_os` is always `None` on wasm.
        let disable_subgroups = std::env::var_os("FUSOR_DISABLE_SUBGROUPS").is_some();
        let poison_allocations = std::env::var_os("FUSOR_DIRTY_BUFFERS").is_some();

        let adapter_info = adapter.get_info();
        let limits = adapter.limits();
        let inner = Arc::new(DeviceInner {
            device,
            adapter,
            adapter_info,
            limits,
            queue,
            kernel_cache,
            buffer_pool,
            decode_plan_cache: DecodePlanCache::new(),
            cooperative_matrix_caps,
            compute_graph: OnceLock::new(),
            disable_subgroups,
            poison_allocations,
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
            let weak_device = Arc::downgrade(&device.inner.device);
            move || loop {
                let Some(device) = weak_device.upgrade() else {
                    break;
                };
                let result = device.poll(wgpu::PollType::Poll);
                drop(device);
                let Ok(status) = result else {
                    break;
                };
                if status.is_queue_empty() {
                    std::thread::sleep(Duration::from_millis(1));
                } else {
                    std::thread::yield_now();
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
        self.inner.limits.clone()
    }

    #[doc(hidden)]
    pub fn nary_direct_input_binding_budget(&self) -> usize {
        let limits = self.limits();
        let storage_bindings = limits.max_storage_buffers_per_shader_stage as usize;
        let bind_group_bindings = limits.max_bindings_per_bind_group as usize;

        // The direct n-ary kernel binds every unique input tensor plus one
        // output tensor, all as storage buffers in a single bind group.
        storage_bindings.min(bind_group_bindings).saturating_sub(1)
    }

    pub fn features(&self) -> wgpu::Features {
        self.inner.device.features()
    }

    pub fn subgroups_supported(&self) -> bool {
        // A device constructed via `without_subgroups()` (or built with
        // `FUSOR_DISABLE_SUBGROUPS` set) reports no subgroups, so kernel
        // selection picks the no-subgroup fallbacks. Browser builds also take
        // this path because they never request `wgpu::Features::SUBGROUP`.
        if self.inner.disable_subgroups {
            return false;
        }
        self.features().contains(wgpu::Features::SUBGROUP)
    }

    pub fn subgroup_token(&self) -> Option<SubgroupToken> {
        if !self.subgroups_supported() {
            return None;
        }

        Some(SubgroupToken::new_unchecked())
    }

    pub(crate) fn subgroup_config(&self) -> Option<SubgroupConfig> {
        Some(SubgroupConfig::new(
            self.subgroup_token()?,
            self.min_subgroup_size(),
            self.max_subgroup_size(),
        ))
    }

    /// Apple-silicon GPUs always execute 32-wide SIMD-groups, but wgpu/Metal
    /// advertises the conservative MSL range (`min` 4, `max` 64) because the
    /// exact width is only resolved at pipeline reflection time. Reporting that
    /// range makes the subgroup-size-aware kernels treat the device as having a
    /// variable subgroup width, which disables the qgemv ggml fast path
    /// (`supports_lanes_per_item`) and the cooperative-matrix tiles
    /// (`is_fixed`). Pinning the true fixed width of 32 keeps those fast routes
    /// available.
    fn apple_fixed_subgroup_size(&self) -> Option<u32> {
        let info = &self.inner.adapter_info;
        (info.backend == wgpu::Backend::Metal && info.name.starts_with("Apple")).then_some(32)
    }

    pub fn min_subgroup_size(&self) -> u32 {
        self.apple_fixed_subgroup_size()
            .unwrap_or(self.inner.adapter_info.subgroup_min_size)
    }

    pub fn max_subgroup_size(&self) -> u32 {
        self.apple_fixed_subgroup_size()
            .unwrap_or(self.inner.adapter_info.subgroup_max_size)
    }

    pub(crate) fn backend(&self) -> wgpu::Backend {
        self.inner.adapter_info.backend
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

        None
    }

    pub fn f16_supported(&self) -> bool {
        self.features().contains(wgpu::Features::SHADER_F16)
    }

    pub(crate) fn cooperative_matrix_caps(&self) -> CooperativeMatrixCaps {
        self.inner.cooperative_matrix_caps
    }

    pub(crate) fn coop_token(&self, kind: CooperativeMatrixKind) -> Option<CoopMatrixToken> {
        if !self.cooperative_matrix_caps().supports(kind) {
            return None;
        }

        Some(CoopMatrixToken::new_unchecked())
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
        #[cfg(target_arch = "wasm32")]
        {
            return;
        }
        #[cfg(not(target_arch = "wasm32"))]
        poll_until_queue_empty(&self.inner.device).expect("Failed to poll GPU device");
    }

    pub(crate) fn kernel_cache(&self) -> &KernelCache {
        &self.inner.kernel_cache
    }

    pub(crate) fn decode_plan_cache(&self) -> &DecodePlanCache {
        &self.inner.decode_plan_cache
    }

    /// Reset the initialized flag on all cached buffers.
    pub fn reset_initialized_buffers(&self) {
        self.inner.buffer_pool.reset_initialized_buffers();
    }

    /// Get or create a buffer of the specified size. Poisoned first when this
    /// handle was built with [`Device::with_poisoned_allocations`].
    pub fn create_buffer(&self, size: u64, usage: wgpu::BufferUsages) -> Arc<wgpu::Buffer> {
        self.inner
            .buffer_pool
            .create_buffer(size, usage, self.inner.poison_allocations)
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
}

#[cfg(test)]
mod dirty_buffer_tests {
    use super::*;

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn derived_device_keeps_native_poller_alive() {
        pollster::FutureExt::block_on(async {
            let Ok(derived) = Device::new().await.map(|device| device.without_subgroups()) else {
                return;
            };

            let buffer = derived
                .wgpu_device()
                .create_buffer(&wgpu::BufferDescriptor {
                    size: 4,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                    label: Some("derived poller readback"),
                });
            derived.wgpu_queue().write_buffer(&buffer, 0, &[1, 2, 3, 4]);
            let encoder = derived
                .wgpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            derived.wgpu_queue().submit(Some(encoder.finish()));

            let (sender, receiver) = std::sync::mpsc::channel();
            buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    let _ = sender.send(result);
                });

            let deadline = Instant::now() + Duration::from_secs(2);
            loop {
                match receiver.try_recv() {
                    Ok(result) => {
                        result.unwrap();
                        break;
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) if Instant::now() < deadline => {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => {
                        panic!("derived device map_async did not complete before timeout");
                    }
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        panic!("derived device map_async callback sender dropped");
                    }
                }
            }

            let view = buffer.slice(..).get_mapped_range();
            assert_eq!(&*view, &[1, 2, 3, 4]);
            drop(view);
            buffer.unmap();
        });
    }

    /// Positive control: a buffer handed out by a poisoned-allocation handle
    /// must read back as the poison byte, proving the poison actually lands on
    /// this backend (and is not silently zero-initialized away).
    #[test]
    fn dirty_mode_poisons_freshly_allocated_buffers() {
        pollster::FutureExt::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };

            let dirty_device = device.with_poisoned_allocations();
            let size = 256u64;
            let buffer = dirty_device.create_buffer(
                size,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            );

            let download = device.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
                label: None,
            });
            let mut encoder = device
                .wgpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            encoder.copy_buffer_to_buffer(&buffer, 0, &download, 0, size);
            device.wgpu_queue().submit(Some(encoder.finish()));

            let (sender, receiver) = futures_channel::oneshot::channel();
            download
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    let _ = sender.send(result);
                });
            device.poll_wait();
            receiver.await.unwrap().unwrap();
            let view = download.slice(..).get_mapped_range();
            let bytes: &[u8] = &view;
            let poison = bytes.iter().filter(|b| **b == 0xCD).count();
            assert_eq!(
                poison,
                bytes.len(),
                "expected all {} bytes to be poison 0xCD, got {poison}; first bytes={:?}",
                bytes.len(),
                &bytes[..bytes.len().min(16)]
            );
        });
    }
}
