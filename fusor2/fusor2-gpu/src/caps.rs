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
pub const FORK_METAL: bool = true;
/// See the `fork-metal` arm.
#[cfg(not(feature = "fork-metal"))]
pub const FORK_METAL: bool = false;

/// The WebGPU baseline. **Not** `adapter.limits()`.
///
/// Starts from wgpu's own spec defaults and then overwrites, field for field,
/// the six limits the compiler actually reads from
/// [`fusor2_ir::device::Limits::default()`], so the wgpu request and the IR's
/// legality model cannot drift apart.
pub fn baseline_limits() -> wgpu::Limits {
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
pub struct LimitWiden {
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

impl LimitWiden {
    pub const NONE: Self = Self {
        max_compute_invocations_per_workgroup: None,
        max_compute_workgroup_size_x: None,
        max_compute_workgroup_size_y: None,
        max_compute_workgroup_size_z: None,
        max_compute_workgroups_per_dimension: None,
        max_compute_workgroup_storage_size: None,
        max_storage_buffers_per_shader_stage: None,
        max_storage_buffer_binding_size: None,
        max_buffer_size: None,
    };

    /// True when nothing is widened, i.e. the request is pure baseline.
    pub fn is_empty(&self) -> bool {
        *self == Self::NONE
    }
}

/// Raise only the named fields of `base`, refusing a widening the adapter
/// cannot supply. A request *below* the baseline is a no-op: the baseline is a
/// floor, so a plan legal on one device stays legal on another.
pub fn widen_limits(
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

/// Scaffold-compatible alias for [`widen_limits`], taking a whole `Limits` as
/// the request instead of a per-field option set. Every field of `needed` that
/// exceeds the baseline is treated as a proven requirement.
pub fn widen(base: wgpu::Limits, needed: &wgpu::Limits, adapter: &wgpu::Adapter) -> wgpu::Limits {
    let adapter_limits = adapter.limits();
    let widen = LimitWiden {
        max_compute_invocations_per_workgroup: Some(needed.max_compute_invocations_per_workgroup),
        max_compute_workgroup_size_x: Some(needed.max_compute_workgroup_size_x),
        max_compute_workgroup_size_y: Some(needed.max_compute_workgroup_size_y),
        max_compute_workgroup_size_z: Some(needed.max_compute_workgroup_size_z),
        max_compute_workgroups_per_dimension: Some(needed.max_compute_workgroups_per_dimension),
        max_compute_workgroup_storage_size: Some(needed.max_compute_workgroup_storage_size),
        max_storage_buffers_per_shader_stage: Some(needed.max_storage_buffers_per_shader_stage),
        max_storage_buffer_binding_size: Some(needed.max_storage_buffer_binding_size),
        max_buffer_size: Some(needed.max_buffer_size),
    };
    // A caller that hands over a whole `Limits` has already clamped it against
    // the adapter; anything it could not supply degrades to the baseline
    // rather than failing the device request.
    widen_limits(base.clone(), widen, &adapter_limits).unwrap_or(base)
}

/// The wgpu features to request, given what the adapter supports. Each is
/// optional and independently fallback-covered.
///
/// `TIMESTAMP_QUERY` is requested whenever available: profiling is read
/// through [`Caps::timestamp_query`], and a device that never resolves a query
/// set pays nothing for holding the feature bit.
pub fn requested_features(adapter: &wgpu::Adapter) -> wgpu::Features {
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
pub fn needs_experimental(features: wgpu::Features) -> bool {
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
pub fn coop_kinds(
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
pub fn ir_limits(limits: &wgpu::Limits) -> Limits {
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
pub fn subgroup_widths(
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
pub fn build_caps(
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

/// Scaffold-compatible convenience wrapper: probe a live adapter.
pub fn probe(adapter: &wgpu::Adapter, limits: &wgpu::Limits) -> Caps {
    let info = adapter.get_info();
    let features = requested_features(adapter);
    let props = coop_properties(adapter);
    build_caps(&info, features, limits, &props, DeviceKind::Gpu)
}

/// The adapter's cooperative-matrix configurations, or empty when the feature
/// is absent.
pub fn coop_properties(adapter: &wgpu::Adapter) -> Vec<wgpu::CooperativeMatrixProperties> {
    if !adapter
        .features()
        .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
    {
        return Vec::new();
    }
    adapter.cooperative_matrix_properties()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apple_info() -> wgpu::AdapterInfo {
        wgpu::AdapterInfo {
            name: "Apple M2 Max".into(),
            vendor: 0,
            device: 0,
            device_type: wgpu::DeviceType::IntegratedGpu,
            driver: String::new(),
            driver_info: String::new(),
            backend: wgpu::Backend::Metal,
            subgroup_min_size: 4,
            subgroup_max_size: 64,
            device_pci_bus_id: String::new(),
            transient_saves_memory: false,
        }
    }

    fn amd_info() -> wgpu::AdapterInfo {
        wgpu::AdapterInfo {
            name: "AMD Radeon Pro".into(),
            backend: wgpu::Backend::Vulkan,
            ..apple_info()
        }
    }

    fn prop(
        m: u32,
        n: u32,
        k: u32,
        ab: wgpu::CooperativeScalarType,
        cr: wgpu::CooperativeScalarType,
        sat: bool,
    ) -> wgpu::CooperativeMatrixProperties {
        wgpu::CooperativeMatrixProperties {
            m_size: m,
            n_size: n,
            k_size: k,
            ab_type: ab,
            cr_type: cr,
            saturating_accumulation: sat,
        }
    }

    fn coop_features() -> wgpu::Features {
        wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX | wgpu::Features::SHADER_F16
    }

    /// Test 1 — the descriptor's limits equal the IR baseline field for field,
    /// even when the adapter reports far more.
    #[test]
    fn baseline_limits_are_requested() {
        let base = baseline_limits();
        let ir = Limits::default();
        assert_eq!(
            base.max_compute_invocations_per_workgroup,
            ir.max_compute_invocations_per_workgroup
        );
        assert_eq!(
            [
                base.max_compute_workgroup_size_x,
                base.max_compute_workgroup_size_y,
                base.max_compute_workgroup_size_z
            ],
            ir.max_compute_workgroup_size
        );
        assert_eq!(
            base.max_compute_workgroups_per_dimension,
            ir.max_compute_workgroups_per_dimension
        );
        assert_eq!(base.max_compute_workgroup_storage_size, 16384);
        assert_eq!(base.max_storage_buffers_per_shader_stage, 8);
        assert_eq!(base.max_storage_buffer_binding_size, 128 << 20);
        // Round-tripping through the IR mirror is the identity.
        assert_eq!(ir_limits(&base), ir);
    }

    #[test]
    fn widen_raises_only_the_named_field() {
        let base = baseline_limits();
        let mut apple = base.clone();
        apple.max_compute_workgroup_storage_size = 32768;
        let widen = LimitWiden {
            max_compute_workgroup_storage_size: Some(32768),
            ..LimitWiden::NONE
        };
        let raised = widen_limits(base.clone(), widen, &apple).expect("apple supplies 32 KiB");
        assert_eq!(raised.max_compute_workgroup_storage_size, 32768);
        // Nothing else moved.
        assert_eq!(
            raised.max_compute_invocations_per_workgroup,
            base.max_compute_invocations_per_workgroup
        );
        assert_eq!(
            raised.max_compute_workgroups_per_dimension,
            base.max_compute_workgroups_per_dimension
        );

        // A mocked 16384 adapter cannot supply it.
        let small = base.clone();
        let err = widen_limits(base.clone(), widen, &small).unwrap_err();
        assert!(matches!(err, Error::Device(_)), "{err}");
    }

    /// Test 2 — the coop filter.
    #[test]
    fn coop_filter_rejects_non_8x8_and_saturating() {
        use wgpu::CooperativeScalarType as S;
        let f = coop_features();
        assert!(coop_kinds(f, &[prop(8, 8, 8, S::F32, S::F32, true)]).is_empty());
        assert!(coop_kinds(f, &[prop(16, 16, 16, S::F32, S::F32, false)]).is_empty());
        assert!(coop_kinds(f, &[prop(8, 8, 8, S::F16, S::F32, false)]).is_empty());
        assert!(coop_kinds(f, &[prop(8, 8, 8, S::I32, S::I32, false)]).is_empty());

        let good = coop_kinds(f, &[prop(8, 8, 8, S::F32, S::F32, false)]);
        assert_eq!(good.len(), 1);
        assert_eq!(good[0].operand, Dtype::F32);
        assert_eq!(good[0].acc, Dtype::F32);
        assert_eq!((good[0].m, good[0].n, good[0].k), (8, 8, 8));

        // F16/F16 needs SHADER_F16.
        let f16_only = coop_kinds(
            wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX,
            &[prop(8, 8, 8, S::F16, S::F16, false)],
        );
        assert!(f16_only.is_empty());
        let with_f16 = coop_kinds(f, &[prop(8, 8, 8, S::F16, S::F16, false)]);
        assert_eq!(with_f16.len(), 1);

        // Without the feature bit nothing is accepted at all.
        assert!(
            coop_kinds(
                wgpu::Features::empty(),
                &[prop(8, 8, 8, S::F32, S::F32, false)]
            )
            .is_empty()
        );
    }

    /// Test 3 — the Apple subgroup override, and what it unlocks.
    #[test]
    fn apple_subgroup_override() {
        use wgpu::CooperativeScalarType as S;
        let features = coop_features() | wgpu::Features::SUBGROUP;
        let props = [prop(8, 8, 8, S::F32, S::F32, false)];
        let limits = baseline_limits();

        let apple = build_caps(&apple_info(), features, &limits, &props, DeviceKind::Gpu);
        assert_eq!(apple.subgroups, Some(SubgroupWidths { min: 32, max: 32 }));
        assert!(apple.subgroups.unwrap().is_fixed());
        assert!(apple.coop_supported());

        let amd = build_caps(&amd_info(), features, &limits, &props, DeviceKind::Gpu);
        assert_eq!(amd.subgroups, Some(SubgroupWidths { min: 4, max: 64 }));
        assert!(!amd.subgroups.unwrap().is_fixed());
        assert!(!amd.coop_supported());
    }

    #[test]
    fn fallbacks_are_off_by_default() {
        let caps = build_caps(
            &apple_info(),
            wgpu::Features::empty(),
            &baseline_limits(),
            &[],
            DeviceKind::Gpu,
        );
        assert!(caps.subgroups.is_none(), "no SUBGROUP -> WgTree folds");
        assert!(!caps.f16, "no SHADER_F16 -> f32");
        assert!(caps.coop.is_empty(), "no coop -> Family::Sgemm");
        assert!(!caps.pipeline_cache, "no PIPELINE_CACHE -> cold compile");
        assert!(!caps.timestamp_query, "no TIMESTAMP_QUERY -> no profiling");
        assert_eq!(caps.workgroup_alias, FORK_METAL);
        assert_eq!(caps.mixed_precision_coop_store, FORK_METAL);
        assert!(caps.atomic_f32, "the CAS loop works on every GPU backend");
    }
}
