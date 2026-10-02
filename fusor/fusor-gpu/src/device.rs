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
        request_device(extra).await
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

/// Pick an adapter, request its device at non-adapter limits, and probe
/// compiler capabilities.
async fn request_device(extra: Option<wgpu::Limits>) -> Result<GpuDevice> {
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
    let caps = caps::build_caps(
        &adapter_info,
        granted,
        &limits,
        &coop_props,
        DeviceKind::Gpu,
    );
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
