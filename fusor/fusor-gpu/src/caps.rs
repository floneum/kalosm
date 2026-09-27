//! Capability probing. Every performance feature has a working fallback
//! (SUBGROUP, SHADER_F16, cooperative matrix, TIMESTAMP_QUERY). Limits are the
//! WebGPU baseline, widened only where a selected kernel proves it needs it;
//! a widening the adapter cannot supply is an error.

use fusor_ir::device::{Caps, CoopKind, DeviceKind, Limits, SubgroupWidths};
use fusor_ir::dtype::Dtype;
use fusor_ir::error::{Error, Result};
use smallvec::SmallVec;

/// `true` only under `fork-metal`: workgroup-alias byte arenas and the
/// mixed-precision cooperative store. Their absence costs footprint only.
#[cfg(feature = "fork-metal")]
pub(crate) const FORK_METAL: bool = true;
/// See the `fork-metal` arm.
#[cfg(not(feature = "fork-metal"))]
pub(crate) const FORK_METAL: bool = false;

/// The WebGPU baseline, **not** `adapter.limits()`: wgpu's spec defaults with
/// the six limits the compiler reads taken from
/// [`fusor_ir::device::Limits::default()`].
pub(crate) fn baseline_limits() -> wgpu::Limits {
    let ir = Limits::default();
    wgpu::Limits {
        max_compute_invocations_per_workgroup: ir.max_compute_invocations_per_workgroup,
        max_compute_workgroup_size_x: ir.max_compute_workgroup_size[0],
        max_compute_workgroup_size_y: ir.max_compute_workgroup_size[1],
        max_compute_workgroup_size_z: ir.max_compute_workgroup_size[2],
        max_compute_workgroups_per_dimension: ir.max_compute_workgroups_per_dimension,
        max_compute_workgroup_storage_size: ir.max_compute_workgroup_storage_size,
        max_storage_buffers_per_shader_stage: ir.max_storage_buffers_per_shader_stage,
        max_storage_buffer_binding_size: ir.max_storage_buffer_binding_size,
        ..wgpu::Limits::default()
    }
}

/// Widen the compiler's limits while retaining the baseline as a floor.
/// Requests beyond the adapter's capability are rejected.
pub(crate) fn widen_limits(
    mut base: wgpu::Limits,
    extra: Option<wgpu::Limits>,
    adapter: &wgpu::Limits,
) -> Result<wgpu::Limits> {
    let Some(extra) = extra else { return Ok(base) };
    macro_rules! raise {
        ($field:ident) => {
            if extra.$field > adapter.$field {
                return Err(Error::Device(format!(
                    "adapter cannot supply {} = {} (reports {})",
                    stringify!($field),
                    extra.$field,
                    adapter.$field
                )));
            }
            base.$field = base.$field.max(extra.$field);
        };
    }
    raise!(max_compute_invocations_per_workgroup);
    raise!(max_compute_workgroup_size_x);
    raise!(max_compute_workgroup_size_y);
    raise!(max_compute_workgroup_size_z);
    raise!(max_compute_workgroups_per_dimension);
    raise!(max_compute_workgroup_storage_size);
    raise!(max_storage_buffers_per_shader_stage);
    raise!(max_storage_buffer_binding_size);
    raise!(max_buffer_size);
    Ok(base)
}

/// The wgpu features to request; each is optional and fallback-covered.
pub(crate) fn requested_features(adapter: &wgpu::Adapter) -> wgpu::Features {
    let available = adapter.features();
    let mut wanted = wgpu::Features::empty();
    let mut want = |f: wgpu::Features| {
        if available.contains(f) {
            wanted |= f;
        }
    };
    // Subgroups are optional on both native and browser adapters.
    want(wgpu::Features::SUBGROUP);
    want(wgpu::Features::SHADER_F16);
    want(wgpu::Features::PIPELINE_CACHE);
    // wasm32 never requests timestamps: the tuner reads its query set back
    // synchronously, which would deadlock a browser page.
    #[cfg(not(target_arch = "wasm32"))]
    if available.contains(wgpu::Features::TIMESTAMP_QUERY) {
        want(wgpu::Features::TIMESTAMP_QUERY);
        want(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES);
    }
    // Cooperative matrices are experimental; `device::request_device` supplies
    // the `ExperimentalFeatures` token.
    want(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX);
    // EXPERIMENTAL_WORKGROUP_MEMORY_ALIAS exists only on the wgpu fork; the
    // byte-arena emitter does not need it.
    wanted
}

/// True when this experimental-feature set needs the unsafe opt-in token.
pub(crate) fn needs_experimental(features: wgpu::Features) -> bool {
    features.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
}

/// Apple GPUs advertise a subgroup-size range but run 32-wide; a ranged
/// width would disable every cooperative tile and the qgemv fast path.
fn apple_fixed_subgroup_size(backend: wgpu::Backend, name: &str) -> Option<SubgroupWidths> {
    (backend == wgpu::Backend::Metal && name.starts_with("Apple"))
        .then_some(SubgroupWidths { min: 32, max: 32 })
}

/// Accept a cooperative-matrix property only in the shape the lowerer emits:
/// 8x8x8, non-saturating, F32/F32, or F16/F16 with `SHADER_F16`.
pub(crate) fn coop_kinds(
    features: wgpu::Features,
    props: &[wgpu::CooperativeMatrixProperties],
) -> SmallVec<[CoopKind; 4]> {
    let mut out: SmallVec<[CoopKind; 4]> = SmallVec::new();
    if !features.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
        return out;
    }
    let f16 = features.contains(wgpu::Features::SHADER_F16);
    for p in props {
        if p.m_size != 8 || p.n_size != 8 || p.k_size != 8 || p.saturating_accumulation {
            continue;
        }
        use wgpu::CooperativeScalarType as S;
        let kind = match (p.ab_type, p.cr_type) {
            (S::F32, S::F32) => CoopKind {
                operand: Dtype::F32,
                acc: Dtype::F32,
                m: 8,
                n: 8,
                k: 8,
            },
            (S::F16, S::F16) if f16 => CoopKind {
                operand: Dtype::F16,
                acc: Dtype::F16,
                m: 8,
                n: 8,
                k: 8,
            },
            // Mixed precision and integer fragments are refused outright.
            _ => continue,
        };
        if !out.contains(&kind) {
            out.push(kind);
        }
    }
    out
}

/// Mirror the six wgpu limits the compiler reads into the IR's model.
pub(crate) fn ir_limits(limits: &wgpu::Limits) -> Limits {
    Limits {
        max_compute_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
        max_compute_workgroup_size: [
            limits.max_compute_workgroup_size_x,
            limits.max_compute_workgroup_size_y,
            limits.max_compute_workgroup_size_z,
        ],
        max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
        max_compute_workgroup_storage_size: limits.max_compute_workgroup_storage_size,
        max_storage_buffers_per_shader_stage: limits.max_storage_buffers_per_shader_stage,
        max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
    }
}

/// What the adapter reports about subgroup widths, with the Apple override.
pub(crate) fn subgroup_widths(
    features: wgpu::Features,
    backend: wgpu::Backend,
    name: &str,
    min: u32,
    max: u32,
) -> Option<SubgroupWidths> {
    if !features.contains(wgpu::Features::SUBGROUP) {
        return None;
    }
    if let Some(fixed) = apple_fixed_subgroup_size(backend, name) {
        return Some(fixed);
    }
    (min > 0 && max >= min).then_some(SubgroupWidths { min, max })
}

/// Everything a legality predicate may read. Legality only: rates live in
/// [`fusor_ir::cost::DeviceFacts`].
pub(crate) fn build_caps(
    info: &wgpu::AdapterInfo,
    features: wgpu::Features,
    limits: &wgpu::Limits,
    coop_props: &[wgpu::CooperativeMatrixProperties],
    kind: DeviceKind,
) -> Caps {
    let subgroups = subgroup_widths(
        features,
        info.backend,
        &info.name,
        info.subgroup_min_size,
        info.subgroup_max_size,
    );
    let fork = FORK_METAL && info.backend == wgpu::Backend::Metal;
    Caps {
        kind,
        name: format!("{:?}/{}", info.backend, info.name),
        limits: ir_limits(limits),
        subgroups,
        f16: features.contains(wgpu::Features::SHADER_F16),
        // No wgpu backend exposes a bf16 shader type in 29; bf16 values are a
        // storage dtype widened to f32 for compute by the `widen-compute` rule.
        bf16: false,
        coop: coop_kinds(features, coop_props),
        // f32 `atomicAdd` is a bitcast compare-exchange loop every backend supports.
        atomic_f32: kind == DeviceKind::Gpu,
        workgroup_alias: fork,
        mixed_precision_coop_store: fork,
        pipeline_cache: features.contains(wgpu::Features::PIPELINE_CACHE),
        timestamp_query: features.contains(wgpu::Features::TIMESTAMP_QUERY),
        // GPU lanes are not SIMD lanes; the CPU target fills these in.
        simd_widths: SmallVec::new(),
        threads: 1,
    }
}

/// The adapter's cooperative-matrix configurations, or empty when the feature
/// is absent.
pub(crate) fn coop_properties(adapter: &wgpu::Adapter) -> Vec<wgpu::CooperativeMatrixProperties> {
    if !adapter
        .features()
        .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
    {
        return Vec::new();
    }
    adapter.cooperative_matrix_properties()
}
