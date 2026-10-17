//! Adapter and device acquisition at WebGPU baseline limits, widened only
//! where a selected kernel proves it needs it, so plan legality is portable.

use fusor_ir::Result;
use fusor_ir::cost::DeviceFacts;
use fusor_ir::device::{Caps, DeviceKind};
use fusor_ir::error::Error;

use crate::caps;

/// Whether the wgpu device has been lost, and why. wgpu reports a loss only
/// through its callback; without this the first symptom is an unrelated
/// validation panic later on.
#[derive(Clone, Default)]
pub struct LostFlag(std::sync::Arc<parking_lot::Mutex<Option<String>>>);

impl LostFlag {
    /// The recorded reason, if the device is gone.
    pub fn reason(&self) -> Option<String> {
        self.0.lock().clone()
    }

    /// `Err` naming the loss when the device is gone.
    pub fn check(&self) -> Result<()> {
        match self.reason() {
            Some(reason) => Err(Error::Device(format!("the wgpu device was lost: {reason}"))),
            None => Ok(()),
        }
    }
}

/// A live wgpu device plus everything the compiler reads about it.
pub struct GpuDevice {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter: wgpu::Adapter,
    caps: Caps,
    facts: DeviceFacts,
    limits_used: wgpu::Limits,
    features: wgpu::Features,
    adapter_info: wgpu::AdapterInfo,
    lost: LostFlag,
}

impl GpuDevice {
    /// Probe an adapter, request a device at baseline limits widened by
    /// `extra`, then seed (or load cached) facts.
    pub async fn request(extra: Option<wgpu::Limits>) -> Result<Self> {
        Self::request_with_profile(extra, Default::default()).await
    }

    pub async fn request_with_profile(
        extra: Option<wgpu::Limits>,
        profile: fusor_ir::execution::ExecutionProfile,
    ) -> Result<Self> {
        request_device(extra, profile).await
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }
    pub fn adapter(&self) -> &wgpu::Adapter {
        &self.adapter
    }
    pub fn caps(&self) -> &Caps {
        &self.caps
    }
    pub fn facts(&self) -> &DeviceFacts {
        &self.facts
    }
    /// The limits actually requested from the adapter.
    pub fn limits_used(&self) -> &wgpu::Limits {
        &self.limits_used
    }
    /// The features actually granted. Each has a documented fallback, so a
    /// missing bit narrows the candidate set rather than failing a build.
    pub fn features(&self) -> wgpu::Features {
        self.features
    }
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }
    /// Set once the driver reports the device lost; see [`LostFlag`].
    pub fn lost(&self) -> &LostFlag {
        &self.lost
    }
}

// Select before requesting limits. A profile must never make an otherwise
// usable adapter fail device acquisition merely because it lacks an optimization.
fn select_execution_profile(
    requested_profile: fusor_ir::execution::ExecutionProfile,
    adapter_info: &wgpu::AdapterInfo,
    adapter_limits: &wgpu::Limits,
    features: wgpu::Features,
) -> fusor_ir::execution::ExecutionProfile {
    // wgpu's Metal backend reports vendor=0. Use the same adapter identity
    // as the existing fixed-subgroup capability probe, not a PCI vendor ID.
    if adapter_info.name.starts_with("Apple")
        && adapter_info.backend == wgpu::Backend::Metal
        && caps::subgroup_widths(
            features,
            adapter_info.backend,
            &adapter_info.name,
            adapter_info.subgroup_min_size,
            adapter_info.subgroup_max_size,
        )
        .is_some_and(|s| s.min == 32 && s.max == 32)
        && adapter_limits.max_compute_invocations_per_workgroup >= 1024
        && adapter_limits.max_compute_workgroup_size_x >= 1024
        && features.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
    {
        requested_profile
    } else {
        fusor_ir::execution::ExecutionProfile::General
    }
}

/// Pick an adapter, request its device at non-adapter limits, and probe
/// compiler capabilities.
async fn request_device(
    extra: Option<wgpu::Limits>,
    requested_profile: fusor_ir::execution::ExecutionProfile,
) -> Result<GpuDevice> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .map_err(|e| Error::Device(format!("request_adapter: {e}")))?;
    let adapter_info = adapter.get_info();

    let features = caps::requested_features(&adapter);
    let adapter_limits = adapter.limits();
    let mut limits = caps::widen_limits(caps::baseline_limits(), extra, &adapter_limits)?;
    let profile =
        select_execution_profile(requested_profile, &adapter_info, &adapter_limits, features);
    if let Ok(value) = profile.var("FUSOR_PROFILE_WORKGROUP_THREADS") {
        let threads = value
            .parse::<u32>()
            .map_err(|_| Error::Device("workgroup threads must be an integer".into()))?;
        if threads < limits.max_compute_invocations_per_workgroup
            || threads > adapter_limits.max_compute_invocations_per_workgroup
            || threads > adapter_limits.max_compute_workgroup_size_x
        {
            return Err(Error::Device(format!(
                "requested workgroup threads {threads} exceed adapter limits or reduce the baseline"
            )));
        }
        limits.max_compute_invocations_per_workgroup = threads;
        limits.max_compute_workgroup_size_x = limits.max_compute_workgroup_size_x.max(threads);
    }
    #[cfg(feature = "kernel-observer")]
    if let Ok(value) = std::env::var("FUSOR_PROFILE_WORKGROUP_STORAGE") {
        let bytes = value.parse::<u32>().map_err(|_| {
            Error::Device("FUSOR_PROFILE_WORKGROUP_STORAGE must be an integer".into())
        })?;
        if bytes < limits.max_compute_workgroup_storage_size
            || bytes > adapter_limits.max_compute_workgroup_storage_size
        {
            return Err(Error::Device(format!(
                "requested workgroup storage {bytes} must be between {} and adapter maximum {}",
                limits.max_compute_workgroup_storage_size,
                adapter_limits.max_compute_workgroup_storage_size
            )));
        }
        eprintln!(
            "PROFILE_DEVICE workgroup_storage baseline={} requested={bytes} adapter={}",
            limits.max_compute_workgroup_storage_size,
            adapter_limits.max_compute_workgroup_storage_size
        );
        limits.max_compute_workgroup_storage_size = bytes;
    }
    // Buffer-size ceilings are capacity, not legality: take the adapter's, since
    // 7B+ models have single weights past the baseline's 256 MiB.
    limits.max_buffer_size = limits.max_buffer_size.max(adapter_limits.max_buffer_size);
    // Bindings per stage cap a slab's stage count. One slot short of the
    // adapter's: wgpu's buffer-sizes table shares the argument table, and Metal
    // loses the device when a kernel fills all 31.
    limits.max_storage_buffers_per_shader_stage = limits.max_storage_buffers_per_shader_stage.max(
        adapter_limits
            .max_storage_buffers_per_shader_stage
            .saturating_sub(1),
    );
    limits.max_storage_buffer_binding_size = limits
        .max_storage_buffer_binding_size
        .max(adapter_limits.max_storage_buffer_binding_size);

    let descriptor = wgpu::DeviceDescriptor {
        label: Some("fusor"),
        required_features: features,
        required_limits: limits.clone(),
        // Cooperative matrices need the experimental token; every use is behind
        // `caps.coop_supported()`.
        experimental_features: if caps::needs_experimental(features) {
            // SAFETY: the only experimental bit is EXPERIMENTAL_COOPERATIVE_MATRIX,
            // used through naga's validated coop ops whose operands are range-checked.
            unsafe { wgpu::ExperimentalFeatures::enabled() }
        } else {
            wgpu::ExperimentalFeatures::disabled()
        },
        memory_hints: wgpu::MemoryHints::default(),
        trace: wgpu::Trace::Off,
    };

    let (device, queue) = adapter
        .request_device(&descriptor)
        .await
        .map_err(|e| Error::Device(format!("request_device: {e}")))?;
    let lost = LostFlag::default();
    {
        let lost = lost.clone();
        device.set_device_lost_callback(move |reason, message| {
            // `Destroyed` is the orderly teardown of a device nobody uses any
            // more; only a driver-side loss is worth recording.
            if reason == wgpu::DeviceLostReason::Destroyed {
                return;
            }
            let text = format!("{reason:?}: {message}");
            eprintln!("[fusor-gpu] wgpu device lost ({text})");
            *lost.0.lock() = Some(text);
        });
    }

    let granted = device.features();
    let coop_props = caps::coop_properties(&adapter);
    let mut caps = caps::build_caps(
        &adapter_info,
        granted,
        &limits,
        &coop_props,
        DeviceKind::Gpu,
    );
    caps.execution_profile = profile;
    #[cfg(target_arch = "wasm32")]
    let caps = probe_browser_matrices(&device, caps).await?;
    // Rates are calibrated or cached by fusor-cost; capabilities are always
    // re-probed so they cannot outlive a driver update.
    let facts = fusor_cost::facts::seed_facts(&caps);

    Ok(GpuDevice {
        device,
        queue,
        adapter,
        caps,
        facts,
        limits_used: limits,
        features: granted,
        adapter_info,
        lost,
    })
}

/// Probe the experimental matrix dialect once, through the same naga path as
/// real kernels, before Session admits matrix candidates.
#[cfg(target_arch = "wasm32")]
async fn probe_browser_matrices(device: &wgpu::Device, mut caps: Caps) -> Result<Caps> {
    if !caps.subgroups.is_some_and(|w| w.min == 32 && w.max == 32) {
        return Ok(caps);
    }
    let mut supported = smallvec::SmallVec::new();
    for kind in &caps.coop {
        let scalar = if kind.operand == fusor_ir::dtype::Dtype::F16 {
            "f16"
        } else {
            "f32"
        };
        let enable = if scalar == "f16" { "enable f16;" } else { "" };
        let source = format!(
            r#"
enable wgpu_cooperative_matrix;
{enable}
var<workgroup> tile: array<{scalar},64>;
@group(0) @binding(0) var<storage,read_write> output: array<{scalar}>;
@compute @workgroup_size(32)
fn main() {{
    let a = coopLoad<coop_mat8x8<{scalar},A>>(&tile[0],8u);
    let b = coopLoad<coop_mat8x8<{scalar},B>>(&tile[0],8u);
    let c = coopLoad<coop_mat8x8<{scalar},C>>(&tile[0],8u);
    let result = coopMultiplyAdd(a,b,c);
    coopStore(result,&output[0],8u);
}}
"#
        );
        let module = naga::front::wgsl::parse_str(&source)
            .map_err(|e| Error::Plan(e.emit_to_string(&source)))?;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fusor browser matrix capability probe"),
            source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(module)),
        });
        let _pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fusor browser matrix capability probe"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        if scope.pop().await.is_none() {
            supported.push(*kind);
        }
    }
    caps.coop = supported;
    Ok(caps)
}

// Explicit auto-trait impls; see the note on `GpuTarget`.
// SAFETY: `device_fields_are_send_sync` asserts every field is `Send + Sync`.
unsafe impl Send for GpuDevice {}
unsafe impl Sync for GpuDevice {}

#[allow(dead_code)]
fn device_fields_are_send_sync() {
    fn assert<T: Send + Sync>() {}
    assert::<wgpu::Device>();
    assert::<wgpu::Queue>();
    assert::<wgpu::Adapter>();
    assert::<Caps>();
    assert::<DeviceFacts>();
    assert::<wgpu::Limits>();
    assert::<wgpu::Features>();
    assert::<wgpu::AdapterInfo>();
    assert::<LostFlag>();
}

/// The D3D12 device-removed reason, if removed; `None` elsewhere. D3D12
/// fences complete instantly after removal, so this is the only signal.
pub fn removed_reason(device: &wgpu::Device) -> Option<String> {
    #[cfg(windows)]
    {
        // SAFETY: the hal device is only borrowed for the duration of one
        // query that does not touch any wgpu-owned state.
        let hal = unsafe { device.as_hal::<wgpu::hal::api::Dx12>() }?;
        unsafe { hal.raw_device().GetDeviceRemovedReason() }
            .err()
            .map(|e| e.to_string())
    }
    #[cfg(not(windows))]
    {
        let _ = device;
        None
    }
}

#[cfg(test)]
mod profile_tests {
    use super::*;
    use fusor_ir::execution::ExecutionProfile::{General, LowLatency};

    #[test]
    fn inference_profile_falls_back_before_requesting_unsupported_limits() {
        let mut info = wgpu::AdapterInfo {
            name: "Apple M2 Max".into(),
            vendor: 0,
            device: 0,
            device_type: wgpu::DeviceType::IntegratedGpu,
            device_pci_bus_id: String::new(),
            driver: String::new(),
            driver_info: String::new(),
            backend: wgpu::Backend::Metal,
            subgroup_min_size: 4,
            subgroup_max_size: 64,
            transient_saves_memory: false,
        };
        let features = wgpu::Features::SUBGROUP | wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX;
        let mut limits = wgpu::Limits {
            max_compute_invocations_per_workgroup: 1024,
            max_compute_workgroup_size_x: 1024,
            ..Default::default()
        };
        assert_eq!(
            select_execution_profile(LowLatency, &info, &limits, features),
            LowLatency
        );
        assert_eq!(
            select_execution_profile(General, &info, &limits, features),
            General
        );
        assert_eq!(
            select_execution_profile(LowLatency, &info, &limits, wgpu::Features::SUBGROUP),
            General
        );
        limits.max_compute_invocations_per_workgroup = 256;
        assert_eq!(
            select_execution_profile(LowLatency, &info, &limits, features),
            General
        );
        limits.max_compute_invocations_per_workgroup = 1024;
        limits.max_compute_workgroup_size_x = 256;
        assert_eq!(
            select_execution_profile(LowLatency, &info, &limits, features),
            General
        );
        limits.max_compute_workgroup_size_x = 1024;
        info.name = "Intel Iris".into();
        assert_eq!(
            select_execution_profile(LowLatency, &info, &limits, features),
            General
        );
    }
}
