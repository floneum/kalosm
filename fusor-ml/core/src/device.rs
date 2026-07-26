use std::{
    fmt::Debug,
    sync::{Arc, Mutex, OnceLock},
};

use fusor_tile_ir::{CoopMatrixToken, SubgroupToken};
use fusor_tile_ir_kernels::SubgroupConfig;
use fusor_tile_ir_runtime::{BufferPool, FusorConfig, KernelCache};
use wgpu::{BackendOptions, Dx12BackendOptions};

use crate::{
    compute_graph::ComputeGraph,
    kernel_selection::{CooperativeMatrixCaps, CooperativeMatrixKind},
};

#[cfg(not(target_arch = "wasm32"))]
use web_time::{Duration, Instant};

#[cfg(not(target_arch = "wasm32"))]
const GPU_POLL_SPIN_BUDGET: Duration = Duration::from_millis(2);

/// Apple-silicon cooperative-matmul rates (see [`Device::matmul_rates`]).
/// `dram_decibytes_per_ns` is the measured M2 Max bandwidth roof (379.5
/// GB/s). `mac_per_ns` is the measured simdgroup issue ceiling, 8.9 TF/s:
/// the 5.60 TF/s this used to carry was never a roof at all — the same
/// kernels sustain 7.57 TF/s on 16384x3072x1536 and 7.64 on the 4096-cube,
/// so the model was charging MMA issue above what the hardware bills and the
/// term crowded out every geometry difference it was supposed to arbitrate.
///
/// The other two have no roof to read and are fitted together against 13
/// contractions crossed with every legal tile — the eight merged microbench
/// cases (K from 64 to 16,384) plus five standalone `bench_coop_tiles`
/// contractions from 1024^3 to 16384x3072x1536, 65 measured spans. They are
/// what decides the geometry, so they are fitted for ranking, not for
/// absolute span: `workgroup_bytes_per_ns` sets what a wider tile buys and
/// `store_fs_per_element` what it costs, and the values below sit at the
/// centre of the box in which every one of the 13 picks the measured
/// fastest tile (`workgroup_bytes_per_ns` 650..800, `store_fs_per_element`
/// 2,000..4,200). Off the low side the K<=16 shapes take a 128-wide profile
/// and lose 8-22%; off the high side the K>=384 shapes take a 64-wide one
/// and lose 3-7%.
pub(crate) const APPLE_MATMUL_RATES: crate::occupancy::MatmulRates =
    crate::occupancy::MatmulRates {
        mac_per_ns: 4450,
        dram_decibytes_per_ns: 3795,
        workgroup_bytes_per_ns: 700,
        store_fs_per_element: 4_000,
        // Fitted against the direct measurement of the trade: the same
        // merged body staged from one pair instead of two, at identical
        // tile, split count and grid, runs 2048x64x64 -14.0%, 2048x64x256
        // -11.0%, 2048x256x64 +0.8% and 384x16384x1536 +7.2% (warm-resolve
        // span medians, 20 interleaved processes per arm, controls flat).
        // Those four shapes bracket the crossover at 4 < k_iterations < 16
        // against the residency credit in T3, which pins this to 103..108.
        single_buffered_traffic_pct: 105,
    };

/// Every other device class. The cost model is a ratio of these rates, so
/// what matters off Apple silicon is their proportion, not their scale: keep
/// the Apple ratios and scale the two absolute roofs down to a conservative
/// mid-range discrete part. Under-stating both roofs equally leaves every
/// comparison unchanged.
pub(crate) const OTHER_MATMUL_RATES: crate::occupancy::MatmulRates =
    crate::occupancy::MatmulRates {
        mac_per_ns: 2225,
        dram_decibytes_per_ns: 1900,
        workgroup_bytes_per_ns: 350,
        store_fs_per_element: 8_000,
        single_buffered_traffic_pct: 105,
    };

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
    desired_adapter_name: Option<&str>,
) -> Result<wgpu::Adapter, crate::Error> {
    let desired_adapter_name = desired_adapter_name.map(|name| name.to_ascii_lowercase());

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
                    "adapter name {desired_adapter_name:?} (WGPU_ADAPTER_NAME) did not match any available adapter"
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
    tracing::error!("{message}");
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

/// One timed dispatch aggregate (per category or per kernel name) from a
/// profiled resolve.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelProfileRow {
    pub name: String,
    pub count: usize,
    pub total_ms: f64,
    pub average_us: f64,
    pub max_us: f64,
}

/// GPU timestamp profile of one resolve, recorded when
/// [`FusorConfig::trace_gpu_kernels`] is set and drained with
/// [`Device::take_kernel_profiles`].
#[derive(Clone, Debug, PartialEq)]
pub struct KernelProfile {
    /// `"inside_pass"` when the adapter timestamps individual dispatches,
    /// otherwise `"pass_boundary"`.
    pub timestamp_mode: &'static str,
    pub kernels: usize,
    /// Dispatches the GPU did not sample. `accounted_ms` covers
    /// `kernels - unmeasured_kernels` dispatches, never all of them silently.
    pub unmeasured_kernels: usize,
    pub accounted_ms: f64,
    /// Wall span from the first sampled begin to the last sampled end, or `None`
    /// when any dispatch went unmeasured and the span would not cover the resolve.
    pub span_ms: Option<f64>,
    pub timestamp_period_ns: f64,
    /// Per-category aggregates, sorted by total time descending.
    pub categories: Vec<KernelProfileRow>,
    /// The most expensive kernel names, sorted by total time descending.
    pub top_names: Vec<KernelProfileRow>,
}

struct DeviceInner {
    device: Arc<wgpu::Device>,
    config: Arc<FusorConfig>,
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
    /// First-occurrence cache of recorded dense and quantized materialization
    /// plans, replayed by `flush_all_pending`. Lives here beside the kernel
    /// cache so it is reachable under the compute-graph write lock.
    flush_plan_cache: crate::compute_graph::FlushPlanCache,
    /// Structural fusion-plan decisions shared across resolves; templates are
    /// matrix-free so nothing here pins buffers or cycles back to this inner.
    fusion_plan_store: crate::compute_graph::FusionPlanStore,
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
    /// GPU kernel profiles recorded by profiled resolves
    /// (`FUSOR_TRACE_GPU_KERNELS`), drained by
    /// [`Device::take_kernel_profiles`].
    kernel_profiles: Mutex<Vec<KernelProfile>>,
    /// Memoized cooperative tile geometry per contraction shape. The scored
    /// selection enumerates every table entry against every legal split count
    /// (~1.5 us) and is asked five times per matmul per resolve, on a decode
    /// path that builds hundreds of dispatches per token. Same class as the
    /// cached adapter info above: a device-lifetime cache of a pure function
    /// of device state.
    coop_tile_memo: Mutex<rustc_hash::FxHashMap<CoopTileKey, Option<[u32; 5]>>>,
}

/// Everything [`crate::matmul::cost::plan_coop_tile`] reads that varies per
/// call; every other input is device state, which the memo's owner pins.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct CoopTileKey {
    pub(crate) m: u32,
    pub(crate) k: u32,
    pub(crate) n: u32,
    pub(crate) batch: u32,
    pub(crate) datatype: crate::DataTypeEnum,
    pub(crate) has_epilogues: bool,
    pub(crate) probe_group: u32,
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
        let config = src.config.clone();
        let kernel_cache = KernelCache::new(device.clone(), &adapter, config.clone());
        let buffer_pool = BufferPool::new(device.clone(), queue.clone(), &config);
        let inner = Arc::new(DeviceInner {
            device,
            config,
            adapter,
            adapter_info: src.adapter_info.clone(),
            limits: src.limits.clone(),
            queue,
            kernel_cache,
            buffer_pool,
            flush_plan_cache: Default::default(),
            fusion_plan_store: Default::default(),
            cooperative_matrix_caps: src.cooperative_matrix_caps,
            compute_graph: OnceLock::new(),
            disable_subgroups,
            poison_allocations,
            kernel_profiles: Default::default(),
            coop_tile_memo: Default::default(),
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
        Self::new_with_config(FusorConfig::from_env()).await
    }

    /// Construct a device with an explicit [`FusorConfig`] instead of reading
    /// the process environment.
    pub async fn new_with_config(config: FusorConfig) -> Result<Self, crate::Error> {
        let config = Arc::new(config);
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
        let adapter = select_adapter(&instance, backends, config.adapter_name.as_deref()).await?;
        let adapter_features = adapter.features();
        #[cfg(target_arch = "wasm32")]
        {
            let info = adapter.get_info();
            tracing::info!(
                "fusor: adapter subgroups={} subgroup_min={} subgroup_max={} shader_f16={} backend={:?} name={:?} (note: the wasm build never requests wgpu::Features::SUBGROUP, so subgroups_supported() stays false regardless of adapter support)",
                adapter_features.contains(wgpu::Features::SUBGROUP),
                info.subgroup_min_size,
                info.subgroup_max_size,
                adapter_features.contains(wgpu::Features::SHADER_F16),
                info.backend,
                info.name,
            );
        }
        let mut required_features = wgpu::Features::empty();
        if adapter_features.contains(wgpu::Features::SUBGROUP) {
            required_features |= wgpu::Features::SUBGROUP;
        }
        if adapter_features.contains(wgpu::Features::SHADER_F16) {
            required_features |= wgpu::Features::SHADER_F16;
        }
        if config.trace_gpu_kernels {
            if adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY) {
                required_features |= wgpu::Features::TIMESTAMP_QUERY;
                if adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES) {
                    required_features |= wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
                }
            } else {
                tracing::warn!(
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
            if adapter_features.contains(wgpu::Features::EXPERIMENTAL_WORKGROUP_MEMORY_ALIAS) {
                required_features |= wgpu::Features::EXPERIMENTAL_WORKGROUP_MEMORY_ALIAS;
                // SAFETY: same experimental opt-in as cooperative matrix.
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
        if config.trace_gpu_kernels && !cooperative_matrix_properties.is_empty() {
            tracing::info!(
                "Fusor cooperative matrix properties: {cooperative_matrix_properties:?}"
            );
            tracing::info!("Fusor cooperative matrix caps: {cooperative_matrix_caps:?}");
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

        let kernel_cache = KernelCache::new(device.clone(), &adapter, config.clone());
        let buffer_pool = BufferPool::new(device.clone(), queue.clone(), &config);

        // Capability simulation is explicit through derived test devices;
        // production always reflects the adapter's reported subgroup support.
        let disable_subgroups = false;
        let poison_allocations = false;

        let adapter_info = adapter.get_info();
        let limits = adapter.limits();
        let inner = Arc::new(DeviceInner {
            device,
            config,
            adapter,
            adapter_info,
            limits,
            queue,
            kernel_cache,
            buffer_pool,
            flush_plan_cache: Default::default(),
            fusion_plan_store: Default::default(),
            cooperative_matrix_caps,
            compute_graph: OnceLock::new(),
            disable_subgroups,
            poison_allocations,
            kernel_profiles: Default::default(),
            coop_tile_memo: Default::default(),
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

    /// A conservative estimate of the device's last-level cache. Data reused
    /// below this size is treated as cache-resident — re-reading it costs no
    /// bandwidth — so the reuse-driven tilings (which trade thread-level
    /// parallelism for explicit reuse) only engage above it. wgpu exposes no
    /// cache size, so this is a conservative floor per device class.
    pub(crate) fn last_level_cache_bytes(&self) -> u64 {
        let info = &self.inner.adapter_info;
        // Apple-silicon system-level cache starts at 8 MiB on the base M1
        // and only grows with tier; other GPU classes floor lower (older
        // discrete L2s are 2-4 MiB, integrated parts share CPU cache).
        if info.backend == wgpu::Backend::Metal && info.name.starts_with("Apple") {
            8 << 20
        } else {
            4 << 20
        }
    }

    /// Shader lanes that must be in flight before a dispatch policy may trade
    /// thread-level parallelism for per-thread work (register tiling, wider
    /// qgemv columns, skipping a fan-out split). wgpu exposes no core count,
    /// so this is a conservative per-class floor: base-tier GPUs keep on the
    /// order of 16K lanes resident and need ~4x oversubscription to hide
    /// memory latency. A conservative under-estimate only makes policies keep
    /// MORE parallelism, never less.
    pub(crate) fn saturation_lanes(&self) -> u32 {
        64 << 10
    }

    /// The memoized `[bm, bn, bk, row_groups, col_groups]` geometry for one
    /// contraction shape, `None` when the shape declines the coop family.
    pub(crate) fn coop_tile_memo(
        &self,
        key: CoopTileKey,
        plan: impl FnOnce() -> Option<[u32; 5]>,
    ) -> Option<[u32; 5]> {
        let mut memo = self
            .inner
            .coop_tile_memo
            .lock()
            .expect("coop tile memo poisoned");
        *memo.entry(key).or_insert_with(plan)
    }

    /// Physical rates the cooperative-matmul cost model prices its terms in.
    /// wgpu exposes no clock, no bandwidth and no core count, so these are
    /// per-class values in the same spirit as [`Self::saturation_lanes`] —
    /// but unlike a parallelism floor, a rate that is wrong in either
    /// direction moves the argmin, so they are anchored on measured roofs
    /// where a roof exists and on fitted spans where none does (see
    /// [`APPLE_MATMUL_RATES`]).
    pub(crate) fn matmul_rates(&self) -> crate::occupancy::MatmulRates {
        let info = &self.inner.adapter_info;
        if info.backend == wgpu::Backend::Metal && info.name.starts_with("Apple") {
            APPLE_MATMUL_RATES
        } else {
            OTHER_MATMUL_RATES
        }
    }

    /// The dispatch-sizing policy derived from this device's capabilities.
    /// Every "how many workgroups / how much work per thread" decision reads
    /// from this one place instead of local constants.
    pub(crate) fn dispatch_policy(&self) -> crate::occupancy::DispatchPolicy {
        crate::occupancy::DispatchPolicy::from_device(self)
    }

    pub fn features(&self) -> wgpu::Features {
        self.inner.device.features()
    }

    pub fn subgroups_supported(&self) -> bool {
        // A test device constructed via `without_subgroups()` reports no
        // subgroups, so selection picks the capability fallback. Browser
        // builds also take this path because they do not request the feature.
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

    /// Proof that the backend can alias workgroup tiles into one byte arena
    /// (the Metal workgroup-alias extension): mixed-stride tiles then pack
    /// at byte offsets instead of separate typed allocations.
    pub(crate) fn byte_arena_token(&self) -> Option<fusor_tile_ir::ByteArenaToken> {
        self.features()
            .contains(wgpu::Features::EXPERIMENTAL_WORKGROUP_MEMORY_ALIAS)
            .then(fusor_tile_ir::ByteArenaToken::new_unchecked)
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

    /// The process configuration this device was constructed with.
    pub fn config(&self) -> &FusorConfig {
        &self.inner.config
    }

    /// Every GPU kernel profile recorded since the last call, oldest first.
    /// One profile is recorded per profiled resolve when
    /// [`FusorConfig::trace_gpu_kernels`] is set; replayed resolves skip
    /// profiling.
    pub fn take_kernel_profiles(&self) -> Vec<KernelProfile> {
        std::mem::take(&mut self.inner.kernel_profiles.lock().unwrap())
    }

    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub(crate) fn record_kernel_profile(&self, profile: KernelProfile) {
        self.inner.kernel_profiles.lock().unwrap().push(profile);
    }

    pub(crate) fn flush_plan_cache(&self) -> &crate::compute_graph::FlushPlanCache {
        &self.inner.flush_plan_cache
    }

    pub(crate) fn fusion_plan_store(&self) -> &crate::compute_graph::FusionPlanStore {
        &self.inner.fusion_plan_store
    }

    /// Reset the initialized flag on all cached buffers.
    pub fn reset_initialized_buffers(&self) {
        self.inner.buffer_pool.reset_initialized_buffers();
    }

    /// Snapshot the cumulative buffer-pool allocation counters (buffers
    /// requested / buffers freshly created). Diff two snapshots to measure
    /// allocations over a window.
    pub fn buffer_pool_counters(&self) -> fusor_tile_ir_runtime::BufferPoolCounters {
        self.inner.buffer_pool.counters()
    }

    /// Whether the buffer pool holds its own tracked clone of `buffer` in the
    /// `(size, usage)` bucket. Used by liveness accounting to enumerate the
    /// pool as an expected strong-reference holder.
    pub(crate) fn buffer_pool_is_tracked(
        &self,
        size: u64,
        usage: wgpu::BufferUsages,
        buffer: &Arc<wgpu::Buffer>,
    ) -> bool {
        self.inner.buffer_pool.is_tracked(size, usage, buffer)
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

    /// Resolve every pending lazy tensor now, submitting the work to the GPU
    /// without waiting for it or downloading anything. Call at iteration
    /// boundaries in training-style loops: it keeps the pending graph (and
    /// per-resolve optimizer cost) bounded while leaving the GPU free to run
    /// ahead of the host.
    pub fn flush(&self) {
        if let Some(graph) = self.inner.compute_graph.get() {
            graph.flush();
        }
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

    /// The pool tracks every buffer it hands out (holding its own strong
    /// clone), reports it via `is_tracked` under the exact `(size, usage)`
    /// key only, and the allocation counters distinguish fresh creations
    /// from pool-cache hits.
    #[test]
    fn buffer_pool_tracking_and_counters() {
        pollster::FutureExt::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };

            let usage = wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST;
            let size = 512u64;

            let before = device.buffer_pool_counters();
            let buffer = device.create_buffer(size, usage);
            let after = device.buffer_pool_counters();
            assert_eq!(after.requested, before.requested + 1);
            assert_eq!(after.created, before.created + 1);

            // Tracked under its exact (size, usage) key, and the pool's own
            // clone means a freshly handed-out buffer has strong_count >= 2.
            assert!(device.buffer_pool_is_tracked(size, usage, &buffer));
            assert!(Arc::strong_count(&buffer) >= 2);
            // Not tracked under a different size or usage.
            assert!(!device.buffer_pool_is_tracked(size * 2, usage, &buffer));
            assert!(!device.buffer_pool_is_tracked(size, wgpu::BufferUsages::STORAGE, &buffer));
            // A foreign buffer (same size/usage, allocated outside the pool)
            // is not tracked.
            let foreign = Arc::new(device.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("foreign"),
                size,
                usage,
                mapped_at_creation: false,
            }));
            assert!(!device.buffer_pool_is_tracked(size, usage, &foreign));

            // Dropping the handle frees the pooled buffer; the next request
            // of the same shape is a cache hit, not a fresh creation.
            drop(buffer);
            let mid = device.buffer_pool_counters();
            let reused = device.create_buffer(size, usage);
            let end = device.buffer_pool_counters();
            assert_eq!(end.requested, mid.requested + 1);
            assert_eq!(end.created, mid.created);
            assert!(device.buffer_pool_is_tracked(size, usage, &reused));
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
