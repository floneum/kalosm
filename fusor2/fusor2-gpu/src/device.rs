//! Adapter and device acquisition. Requests WebGPU baseline limits and widens
//! only what a selected kernel's legality predicate proves it needs — so a
//! plan legal on one device is legal on another and the cost model's filters
//! mean the same thing everywhere.

use fusor2_ir::Result;
use fusor2_ir::cost::DeviceFacts;
use fusor2_ir::device::{Caps, DeviceKind};
use fusor2_ir::error::Error;

use crate::caps::{self, LimitWiden};

/// How to acquire a device. `widen` carries the per-field ceilings a caller
/// has *proved* it needs; everything else stays at the WebGPU baseline.
#[derive(Clone, Debug, Default)]
pub struct DeviceOptions {
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
}

/// Pick an adapter, request a device at `baseline ∪ opts.widen`, probe caps.
///
/// `required_limits` is **never** `adapter.limits()`, so plan legality
/// means the same thing on every device.
pub async fn request_device(opts: &DeviceOptions) -> Result<GpuDevice> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pick_adapter(&instance, opts).await?;
    let adapter_info = adapter.get_info();

    let features = caps::requested_features(&adapter);
    let adapter_limits = adapter.limits();
    let mut limits = caps::widen_limits(caps::baseline_limits(), opts.widen, &adapter_limits)?;
    // The two buffer-size ceilings are memory *capacity*, not occupancy
    // legality: no kernel changes shape because the buffer holding a weight is
    // bigger, and any 7B+ model has single weights past the WebGPU baseline's
    // 256 MiB (a Q6K vocab projection is ~430 MB). Take the adapter's
    // capacity; the workgroup/occupancy limits stay at the baseline so plan
    // legality still means the same thing on every device.
    limits.max_buffer_size = limits.max_buffer_size.max(adapter_limits.max_buffer_size);
    limits.max_storage_buffer_binding_size = limits
        .max_storage_buffer_binding_size
        .max(adapter_limits.max_storage_buffer_binding_size);

    let descriptor = wgpu::DeviceDescriptor {
        label: Some("fusor2"),
        required_features: features,
        required_limits: limits.clone(),
        // Cooperative matrices are an EXPERIMENTAL_ feature bit; wgpu refuses
        // to grant one without this token. Every use is behind
        // `caps.coop_supported()`, whose fallback is `Family::Sgemm`.
        experimental_features: if caps::needs_experimental(features) {
            // SAFETY: the only experimental bit requested is
            // EXPERIMENTAL_COOPERATIVE_MATRIX, used exclusively through
            // naga's validated CooperativeLoad/MultiplyAdd/Store, whose
            // operands the L2 verifier and the emitter both range-check.
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

    let granted = device.features();
    let coop_props = caps::coop_properties(&adapter);
    let caps = caps::build_caps(
        &adapter_info,
        granted,
        &limits,
        &coop_props,
        DeviceKind::Gpu,
    );
    // Rates are calibrated (or loaded from the on-disk cache) by fusor2-cost;
    // capabilities are always re-probed, so a stale capability set cannot
    // outlive a driver update.
    let facts = fusor2_cost::facts::seed_facts(&caps);

    Ok(GpuDevice {
        device,
        queue,
        adapter,
        caps,
        facts,
        limits_used: limits,
        features: granted,
        adapter_info,
    })
}

/// Blocking wrapper for callers outside an async runtime.
pub fn gpu_blocking(opts: &DeviceOptions) -> Result<GpuDevice> {
    pollster::block_on(request_device(opts))
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The descriptor's limits are the baseline even when the adapter reports
    /// far more. Run against a live adapter when one exists; skip cleanly
    /// otherwise, so the naga-only path stays testable on CI without a GPU.
    #[test]
    fn baseline_limits_are_requested_on_a_live_device() {
        let Ok(gpu) = gpu_blocking(&DeviceOptions::default()) else {
            eprintln!("no wgpu adapter; skipping");
            return;
        };
        let base = caps::baseline_limits();
        let used = gpu.limits_used();
        assert_eq!(
            used.max_compute_workgroup_storage_size,
            base.max_compute_workgroup_storage_size
        );
        assert_eq!(
            used.max_compute_invocations_per_workgroup,
            base.max_compute_invocations_per_workgroup
        );
        assert_eq!(
            used.max_compute_workgroups_per_dimension,
            base.max_compute_workgroups_per_dimension
        );
        // ... even though the adapter itself usually reports more.
        assert!(
            gpu.adapter().limits().max_compute_workgroup_storage_size
                >= base.max_compute_workgroup_storage_size
        );
    }

    #[test]
    fn widening_a_field_the_adapter_has_succeeds() {
        let Ok(base_gpu) = gpu_blocking(&DeviceOptions::default()) else {
            eprintln!("no wgpu adapter; skipping");
            return;
        };
        let ceiling = base_gpu
            .adapter()
            .limits()
            .max_compute_workgroup_storage_size;
        drop(base_gpu);
        let opts = DeviceOptions {
            widen: LimitWiden {
                max_compute_workgroup_storage_size: Some(ceiling),
                ..LimitWiden::NONE
            },
            ..DeviceOptions::default()
        };
        let gpu = gpu_blocking(&opts).expect("adapter supplies its own ceiling");
        assert_eq!(
            gpu.limits_used().max_compute_workgroup_storage_size,
            ceiling
        );
    }
}
