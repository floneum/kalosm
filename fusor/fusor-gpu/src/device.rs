//! Adapter and device acquisition. Requests WebGPU baseline limits and widens
//! only what a selected kernel's legality predicate proves it needs — so a
//! plan legal on one device is legal on another and the cost model's filters
//! mean the same thing everywhere.

use fusor_ir::Result;
use fusor_ir::cost::DeviceFacts;
use fusor_ir::device::{Caps, DeviceKind};
use fusor_ir::error::Error;

use crate::caps::{self, LimitWiden};

/// Whether the wgpu device has been lost, and why.
///
/// wgpu reports a lost device only through its device-lost callback: every
/// later `create_buffer` silently hands back an invalid handle, and every
/// error of type `DeviceLost` is dropped on the floor. Without recording the
/// reason here, the first visible symptom is an unrelated validation panic
/// several calls later ("buffer is invalid", "buffer is already mapped"), on
/// whichever call happens to touch the wreckage first.
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

/// How to acquire a device. `widen` carries the per-field ceilings a caller
/// has *proved* it needs; everything else stays at the WebGPU baseline.
#[derive(Clone, Debug, Default)]
pub(crate) struct DeviceOptions {
    pub widen: LimitWiden,
    /// Case-insensitive substring match against the adapter name, for
    /// reproducing a bug on a specific GPU. `None` takes the preferred
    /// adapter.
    pub adapter_name: Option<String>,
    pub power_preference: Option<wgpu::PowerPreference>,
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
        let mut opts = DeviceOptions::default();
        if let Some(extra) = extra {
            opts.widen = LimitWiden {
                max_compute_invocations_per_workgroup: Some(
                    extra.max_compute_invocations_per_workgroup,
                ),
                max_compute_workgroup_size_x: Some(extra.max_compute_workgroup_size_x),
                max_compute_workgroup_size_y: Some(extra.max_compute_workgroup_size_y),
                max_compute_workgroup_size_z: Some(extra.max_compute_workgroup_size_z),
                max_compute_workgroups_per_dimension: Some(
                    extra.max_compute_workgroups_per_dimension,
                ),
                max_compute_workgroup_storage_size: Some(extra.max_compute_workgroup_storage_size),
                max_storage_buffers_per_shader_stage: Some(
                    extra.max_storage_buffers_per_shader_stage,
                ),
                max_storage_buffer_binding_size: Some(extra.max_storage_buffer_binding_size),
                max_buffer_size: Some(extra.max_buffer_size),
            };
        }
        request_device(&opts).await
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
    /// The limits actually requested, which the plan-cache salt includes.
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

/// Pick an adapter, request a device at `baseline ∪ opts.widen`, probe caps.
///
/// `required_limits` is **never** `adapter.limits()`, so plan legality
/// means the same thing on every device.
pub(crate) async fn request_device(opts: &DeviceOptions) -> Result<GpuDevice> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pick_adapter(&instance, opts).await?;
    let adapter_info = adapter.get_info();

    let features = caps::requested_features(&adapter);
    let adapter_limits = adapter.limits();
    let mut limits = caps::widen_limits(caps::baseline_limits(), opts.widen, &adapter_limits)?;
    // The two buffer-size ceilings are memory *capacity*, not occupancy
    // legality, and 7B+ models have single weights past the WebGPU baseline's
    // 256 MiB. Take the adapter's capacity; the workgroup/occupancy limits
    // stay at the baseline so plan legality means the same thing everywhere.
    limits.max_buffer_size = limits.max_buffer_size.max(adapter_limits.max_buffer_size);
    limits.max_storage_buffer_binding_size = limits
        .max_storage_buffer_binding_size
        .max(adapter_limits.max_storage_buffer_binding_size);

    let descriptor = wgpu::DeviceDescriptor {
        label: Some("fusor"),
        required_features: features,
        required_limits: limits.clone(),
        // Cooperative matrices are an EXPERIMENTAL_ feature bit; wgpu refuses
        // to grant one without this token. Every use is behind
        // `caps.coop_supported()`, whose fallback is `Family::Sgemm`.
        experimental_features: if caps::needs_experimental(features) {
            // SAFETY: the only experimental bit requested is
            // EXPERIMENTAL_COOPERATIVE_MATRIX, used exclusively through
            // naga's validated CooperativeLoad/MultiplyAdd/Store, whose
            // operands the Kernel verifier and the emitter both range-check.
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
    // Rates are calibrated (or loaded from the on-disk cache) by fusor-cost;
    // capabilities are always re-probed, so a stale capability set cannot
    // outlive a driver update.
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

/// Rank adapters: discrete, then integrated, then
/// virtual, then CPU, then unknown.
fn adapter_preference_rank(kind: wgpu::DeviceType) -> u8 {
    match kind {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::VirtualGpu => 2,
        wgpu::DeviceType::Cpu => 3,
        wgpu::DeviceType::Other => 4,
    }
}

async fn pick_adapter(instance: &wgpu::Instance, opts: &DeviceOptions) -> Result<wgpu::Adapter> {
    if let Some(wanted) = &opts.adapter_name {
        let wanted = wanted.to_lowercase();
        let mut all = instance.enumerate_adapters(wgpu::Backends::all()).await;
        all.retain(|a| a.get_info().name.to_lowercase().contains(&wanted));
        all.sort_by_key(|a| adapter_preference_rank(a.get_info().device_type));
        return all
            .into_iter()
            .next()
            .ok_or_else(|| Error::Device(format!("no adapter matching {wanted:?}")));
    }
    instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: opts
                .power_preference
                .unwrap_or(wgpu::PowerPreference::HighPerformance),
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .map_err(|e| Error::Device(format!("request_adapter: {e}")))
}

// Explicit auto-trait impls; see the note on `GpuTarget`.
//
// SAFETY: `device_fields_are_send_sync` asserts `Send + Sync` for every
// field type, which is exactly what the auto impls would require.
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
