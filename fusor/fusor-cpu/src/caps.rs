//! Static capabilities used to schedule Cranelift CPU kernels.

use fusor_ir::device::{Caps, DeviceKind, Limits, SubgroupWidths};
use smallvec::smallvec;
use std::sync::OnceLock;

static CAPS: OnceLock<Caps> = OnceLock::new();

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
        Caps {
            kind: DeviceKind::Cpu,
            name: "cpu-cranelift".to_string(),
            limits: Limits {
                // Keep GPU workgroup geometry out of the CPU schedule domain.
                // The native emitter owns its host loop chunk independently.
                max_compute_invocations_per_workgroup: 1,
                max_compute_workgroup_size: [1, 1, 1],
                max_compute_workgroups_per_dimension: u32::MAX,
                max_compute_workgroup_storage_size: 256 * 1024,
                max_storage_buffers_per_shader_stage: 64,
                max_storage_buffer_binding_size: u64::MAX,
            },
            // A "subgroup" on CPU is one SIMD register, so `Reduce{Subgroup}`
            // is legal and lowers to a horizontal reduce. Fixed width, which
            // is what every subgroup-size-aware kernel requires.
            subgroups: Some(SubgroupWidths { min: 1, max: 1 }),
            f16: true,
            bf16: true,
            coop: smallvec![],
            // No f32 atomics: this forces `ScatterMode::SortSegment` and
            // makes `ScatterMode::Atomic` unreachable at mint time. The nest
            // both lower to needs no atomic either way.
            atomic_f32: false,
            // Thread-local scratch aliases freely, so `ArenaMode::ByteArena` is
            // always available and the arena separation predicate is trivially
            // true.
            workgroup_alias: true,
            mixed_precision_coop_store: false,
            pipeline_cache: false,
            timestamp_query: false,
            simd_widths: WIDTHS.iter().copied().collect(),
            threads: CpuCaps::threads(),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caps_are_stable_and_coherent() {
        let c = cpu_caps();
        assert_eq!(c.kind, DeviceKind::Cpu);
        assert!(c.name.starts_with("cpu-"));
        assert!(
            !c.atomic_f32,
            "atomic_f32 must be false so Atomic scatter is unreachable"
        );
        assert!(c.workgroup_alias);
        assert!(
            c.coop.is_empty(),
            "Family::Coop must never be lowered on CPU"
        );
        assert!(c.subgroups.is_some_and(|s| s.is_fixed()));
        assert_eq!(c.simd_widths.as_slice(), &WIDTHS);
        assert!(c.threads >= 1);
    }
}
