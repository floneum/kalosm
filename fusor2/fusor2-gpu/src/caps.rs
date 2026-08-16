//! Capability probing. Every performance feature is probed with a working
//! fallback: SUBGROUP -> `WgTree` folds, SHADER_F16 -> f32,
//! EXPERIMENTAL_COOPERATIVE_MATRIX -> `Family::Sgemm`, PIPELINE_CACHE -> cold
//! compile, TIMESTAMP_QUERY -> no profiling.
//!
//! Limits are the **WebGPU baseline**, never `adapter.limits()`, ensuring plans
//! remain legal across all devices. A caller widens exactly the fields a selected
//! kernel proves it needs, and a widening the adapter cannot supply is an error
//! rather than a silent clamp downwards.

use fusor2_ir::device::{Caps, CoopKind, DeviceKind, Limits, SubgroupWidths};
use fusor2_ir::dtype::Dtype;
use fusor2_ir::error::{Error, Result};
use smallvec::SmallVec;

/// `true` only under the `fork-metal` cargo feature, which contributes
/// exactly two capabilities: workgroup-alias byte-arena packing and
/// mixed-precision cooperative store. Their absence costs `ArenaMode::Regions`
/// packing and a staging tile — footprint, never correctness.
#[cfg(feature = "fork-metal")]
pub(crate) const FORK_METAL: bool = true;
/// See the `fork-metal` arm.
#[cfg(not(feature = "fork-metal"))]
pub(crate) const FORK_METAL: bool = false;

/// The WebGPU baseline. **Not** `adapter.limits()`.
///
/// Starts from wgpu's own spec defaults and then overwrites, field for field,
/// the six limits the compiler actually reads from
/// [`fusor2_ir::device::Limits::default()`], so the wgpu request and the IR's
/// legality model cannot drift apart.
pub(crate) fn baseline_limits() -> wgpu::Limits {
    let ir = Limits::default();
    let mut limits = wgpu::Limits::default();
    limits.max_compute_invocations_per_workgroup = ir.max_compute_invocations_per_workgroup;
    limits.max_compute_workgroup_size_x = ir.max_compute_workgroup_size[0];
    limits.max_compute_workgroup_size_y = ir.max_compute_workgroup_size[1];
    limits.max_compute_workgroup_size_z = ir.max_compute_workgroup_size[2];
    limits.max_compute_workgroups_per_dimension = ir.max_compute_workgroups_per_dimension;
    limits.max_compute_workgroup_storage_size = ir.max_compute_workgroup_storage_size;
    limits.max_storage_buffers_per_shader_stage = ir.max_storage_buffers_per_shader_stage;
    limits.max_storage_buffer_binding_size = ir.max_storage_buffer_binding_size;
    limits
}

/// Per-field ceilings a caller *proves* it needs. Every field is optional;
/// `None` leaves the baseline alone.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct LimitWiden {
    pub max_compute_invocations_per_workgroup: Option<u32>,
    pub max_compute_workgroup_size_x: Option<u32>,
    pub max_compute_workgroup_size_y: Option<u32>,
    pub max_compute_workgroup_size_z: Option<u32>,
    pub max_compute_workgroups_per_dimension: Option<u32>,
    pub max_compute_workgroup_storage_size: Option<u32>,
    pub max_storage_buffers_per_shader_stage: Option<u32>,
    pub max_storage_buffer_binding_size: Option<u64>,
    pub max_buffer_size: Option<u64>,
}

/// Raise only the named fields of `base`, refusing a widening the adapter
/// cannot supply. A request *below* the baseline is a no-op: the baseline is a
/// floor, so a plan legal on one device stays legal on another.
pub(crate) fn widen_limits(
    base: wgpu::Limits,
    widen: LimitWiden,
    adapter: &wgpu::Limits,
) -> Result<wgpu::Limits> {
    let mut out = base;
    // (name, requested, current slot, adapter ceiling)
    macro_rules! raise {
        ($field:ident) => {
            if let Some(want) = widen.$field {
                if want > adapter.$field {
                    return Err(Error::Device(format!(
                        "adapter cannot supply {} = {} (reports {})",
                        stringify!($field),
                        want,
                        adapter.$field
                    )));
                }
                if want > out.$field {
                    out.$field = want;
                }
            }
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
    Ok(out)
}

/// The wgpu features to request, given what the adapter supports. Each is
/// optional and independently fallback-covered.
///
/// `TIMESTAMP_QUERY` is requested whenever available: profiling is read
/// through [`Caps::timestamp_query`], and a device that never resolves a query
/// set pays nothing for holding the feature bit.
pub(crate) fn requested_features(adapter: &wgpu::Adapter) -> wgpu::Features {
    let available = adapter.features();
    let mut wanted = wgpu::Features::empty();
    let mut want = |f: wgpu::Features| {
        if available.contains(f) {
            wanted |= f;
        }
    };
    // wasm32 never requests SUBGROUP: the browser's WebGPU surface does not
    // expose it, and the `WgTree` fold is the working fallback.
    #[cfg(not(target_arch = "wasm32"))]
    want(wgpu::Features::SUBGROUP);
    want(wgpu::Features::SHADER_F16);
    want(wgpu::Features::PIPELINE_CACHE);
    if available.contains(wgpu::Features::TIMESTAMP_QUERY) {
        want(wgpu::Features::TIMESTAMP_QUERY);
        want(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES);
    }
    // Cooperative matrices are experimental: requesting the bit additionally
    // needs `unsafe ExperimentalFeatures::enabled()` on the descriptor, which
    // `device::request_device` supplies. wasm32 requests neither.
    #[cfg(not(target_arch = "wasm32"))]
    want(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX);
    // The second experimental bit, EXPERIMENTAL_WORKGROUP_MEMORY_ALIAS, exists
    // only on the wgpu fork; released wgpu 29 does not define it. The
    // byte-arena emitter does not depend on it (see `emit::types`), so its
    // absence costs packing density, not correctness.
    wanted
}

/// True when this experimental-feature set needs the unsafe opt-in token.
pub(crate) fn needs_experimental(features: wgpu::Features) -> bool {
    features.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
}

/// Apple GPUs advertise a *range* of subgroup sizes even though every shipping
/// part runs 32-wide. A ranged width makes [`SubgroupWidths::is_fixed`] false,
/// which disables every cooperative tile and the qgemv fast path.
fn apple_fixed_subgroup_size(backend: wgpu::Backend, name: &str) -> Option<SubgroupWidths> {
    (backend == wgpu::Backend::Metal && name.starts_with("Apple"))
        .then_some(SubgroupWidths { min: 32, max: 32 })
}

/// Accept a cooperative-matrix property only in the one shape the lowerer can
/// emit: `m == n == k == 8 && !saturating_accumulation`, F32/F32 always and
/// F16/F16 only with `SHADER_F16`. Mixed F16-operand / F32-accumulator is
/// rejected even where the fork's MSL backend supports it, so a plan cannot
/// depend on a fork-only numeric behaviour.
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
/// [`fusor2_ir::cost::DeviceFacts`].
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
        // `atomicAdd` on f32 is emitted as a bitcast compare-exchange loop
        // (see `emit::stmt`), which every WebGPU backend supports. That loop is
        // what makes `ScatterMode::Atomic` a live candidate for the embedding
        // gradient.
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
