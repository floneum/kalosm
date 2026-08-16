//! ISA and cache detection. The detected `Level` is cached in a `OnceLock` and
//! dispatched once per kernel launch, not per row.

use fusor_ir::device::{Caps, DeviceKind, Limits, SubgroupWidths};
use smallvec::smallvec;
use std::sync::OnceLock;

pub(crate) use fearless_simd::Level;

static LEVEL: OnceLock<Level> = OnceLock::new();
static CAPS: OnceLock<Caps> = OnceLock::new();

/// The cached ISA level. Detection runs once per process; `dispatch!` runs
/// once per kernel launch, never per row.
pub(crate) fn level() -> Level {
    *LEVEL.get_or_init(Level::new)
}

/// A stable name for the detected level, used as `Caps::name`.
pub(crate) fn level_name(level: Level) -> &'static str {
    #[allow(unreachable_patterns)]
    match level {
        #[cfg(target_arch = "aarch64")]
        Level::Neon(_) => "cpu-neon",
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Level::Avx2(_) => "cpu-avx2",
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Level::WasmSimd128(_) => "cpu-wasm-simd128",
        _ => "cpu-fallback",
    }
}

/// The widest lane count the emitter will instantiate. `Reduce{Subgroup}` is
/// legal at this width and lowers to a horizontal reduce.
pub(crate) const MAX_WIDTH: u32 = 16;

/// Every width the emitter can instantiate, widest last.
pub(crate) const WIDTHS: [u32; 3] = [4, 8, 16];

/// CPU capability probe.
#[derive(Copy, Clone, Debug, Default)]
pub struct CpuCaps;

impl CpuCaps {
    /// Bytes of the last-level cache, feeding `DeviceFacts::llc_bytes`.
    pub fn llc_bytes() -> u64 {
        // No portable query exists. 8 MiB matches the Apple-silicon SLC slice a
        // single core sees and is a middling x86 L3-per-core figure.
        8 << 20
    }

    /// Available worker threads. 1 on wasm32.
    pub fn threads() -> u32 {
        #[cfg(target_arch = "wasm32")]
        {
            1
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            std::thread::available_parallelism().map_or(1, |n| n.get() as u32)
        }
    }
}

/// The CPU's [`Caps`]. Cached, because `Caps` owns a `String`.
pub(crate) fn cpu_caps() -> &'static Caps {
    CAPS.get_or_init(|| {
        let lvl = level();
        let w = MAX_WIDTH;
        Caps {
            kind: DeviceKind::Cpu,
            name: level_name(lvl).to_string(),
            limits: Limits {
                max_compute_invocations_per_workgroup: 1024,
                max_compute_workgroup_size: [1024, 1024, 64],
                max_compute_workgroups_per_dimension: u32::MAX,
                max_compute_workgroup_storage_size: 256 * 1024,
                max_storage_buffers_per_shader_stage: 64,
                max_storage_buffer_binding_size: u64::MAX,
            },
            // A "subgroup" on CPU is one SIMD register, so `Reduce{Subgroup}`
            // is legal and lowers to a horizontal reduce.
            subgroups: Some(SubgroupWidths { min: w, max: w }),
            f16: true,
            bf16: true,
            coop: smallvec![],
            // No f32 atomics: this forces `ScatterMode::SortSegment` and
            // makes `ScatterMode::Atomic` unreachable at mint time.
            atomic_f32: false,
            // Thread-local scratch aliases freely, so `ArenaMode::ByteArena` is
            // always available.
            workgroup_alias: true,
            mixed_precision_coop_store: false,
            pipeline_cache: false,
            timestamp_query: false,
            simd_widths: WIDTHS.iter().copied().collect(),
            threads: CpuCaps::threads(),
        }
    })
}
