//! The shipped per-class seed tables, used before (or instead of)
//! calibration.
//!
//! These exist only as a starting point: `score_fs`'s measured anchors ship
//! as the seed for the GPU class and a cache-derived table seeds the CPU
//! class. The seed is chosen by `Caps::kind`, never by the adapter name,
//! and calibration replaces it with measurement on the device that will
//! actually run.

use fusor2_ir::cost::{DeviceFacts, RateDtype};
use fusor2_ir::device::{Caps, DeviceKind};

/// Rows of [`DeviceFacts::mac_per_us`] in `MacUnit` order.
///
/// A GPU's half-precision FMA issues at twice the f32 rate, its integer
/// multiply at half, its cooperative unit at twice its scalar unit, and its
/// dp4a unit exists only for the two integer slots.
const fn gpu_mac_table(fma_f32: u64, dp4a: u64) -> [[u64; RateDtype::COUNT]; 3] {
    let half = fma_f32 * 2;
    let int = fma_f32 / 2;
    // F32, F16, BF16, U32, I32
    let fma = [fma_f32, half, half, int, int];
    let coop = [fma[0] * 2, fma[1] * 2, fma[2] * 2, fma[3] * 2, fma[4] * 2];
    // `1` rather than `0`: it prices a dp4a lowering of a float dtype out of
    // contention without relying on `mac_rate`'s clamp.
    let dp = [1, 1, 1, dp4a, dp4a];
    [fma, coop, dp]
}

/// A CPU core widens f16/bf16 into f32 registers and computes there, so the
/// half rows are the f32 rate rather than twice it. There is no cooperative
/// unit; the dp4a row prices the integer-dot lowering of a quantized dot at
/// four lanes per FMA slot.
const fn cpu_mac_table(fma_f32: u64) -> [[u64; RateDtype::COUNT]; 3] {
    let int = fma_f32 / 2;
    let fma = [fma_f32, fma_f32, fma_f32, int, int];
    let dp = [1, 1, 1, fma_f32 * 4, fma_f32 * 4];
    [fma, fma, dp]
}

/// The GPU seed, converted from the reference's `APPLE_MATMUL_RATES`
/// (`mac_per_ns: 4450`, `dram_decibytes_per_ns: 3795`,
/// `workgroup_bytes_per_ns: 700`, `store_fs_per_element: 4000`,
/// `single_buffered_traffic_pct: 105`) and `occupancy.rs`
/// (`saturation_lanes = 64 << 10`, `last_level_cache_bytes = 8 MiB`).
///
/// This is the only calibrated rate vector that exists, so it seeds *every*
/// GPU: an unmeasured device gets an honest starting point and calibration
/// overwrites it.
///
/// Not a `const fn`: [`DeviceFacts`] owns a [`Caps`], which owns a `String`.
pub fn seed_facts_gpu(caps: &Caps) -> DeviceFacts {
    DeviceFacts {
        launch_ps: 1_000_000,
        dram_bytes_per_us: 379_500,
        llc_bytes: 8 << 20,
        wg_bytes_per_us: 700_000,
        mac_per_us: gpu_mac_table(4_450_000, 17_800_000),
        trans_ps: 4,
        store_ps_per_element: 4,
        saturation_lanes: 65_536,
        single_buffered_traffic_pct: 105,
        compile_ps_per_kernel: 1_000_000_000,
        thread_wake_ps: 5_000_000,
        caps: caps.clone(),
    }
}

/// The CPU seed. Rates are per-core figures multiplied by `Caps::threads`,
/// so a 4-core laptop and a 64-core server get different facts without
/// either being named.
///
/// `launch_ps` is zero — a CPU kernel is a function call; the pool-wake cost
/// lives in `thread_wake_ps`, where the parallel-region decision reads it.
/// `wg_bytes_per_us` is Launch bandwidth, which is what a
/// workgroup tile maps onto (thread-local 64-byte-aligned scratch).
/// `store_ps_per_element` has no per-class CPU measurement; the GPU-class
/// value seeds it and `bench_epilogue_occupancy` overwrites it.
pub fn seed_facts_cpu(caps: &Caps) -> DeviceFacts {
    let threads = u64::from(caps.threads.max(1));
    DeviceFacts {
        launch_ps: 0,
        dram_bytes_per_us: 40_000,
        llc_bytes: 16 << 20,
        wg_bytes_per_us: 400_000,
        mac_per_us: cpu_mac_table(32_000 * threads),
        trans_ps: 20,
        store_ps_per_element: 4,
        saturation_lanes: caps.threads.max(1).saturating_mul(8),
        single_buffered_traffic_pct: 105,
        compile_ps_per_kernel: 50_000_000,
        thread_wake_ps: 5_000_000,
        caps: caps.clone(),
    }
}

/// The per-class seed, dispatched on `Caps::kind` and nothing else — never
/// on `Caps::name`. The answer to a wrong seed is calibration.
pub fn seed_facts(caps: &Caps) -> DeviceFacts {
    match caps.kind {
        DeviceKind::Gpu => seed_facts_gpu(caps),
        DeviceKind::Cpu => seed_facts_cpu(caps),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use fusor2_ir::cost::MacUnit;
    use fusor2_ir::device::{Limits, SubgroupWidths};
    use fusor2_ir::dtype::Dtype;

    /// Apple-class GPU caps: 32-wide fixed subgroups, 1024-lane workgroups,
    /// 32 KiB of threadgroup memory. The residency term in `drain_ps` is
    /// calibrated against that last number.
    pub(crate) fn gpu_caps(name: &str) -> Caps {
        Caps {
            kind: DeviceKind::Gpu,
            name: name.to_string(),
            limits: Limits {
                max_compute_workgroup_storage_size: 32 << 10,
                max_compute_invocations_per_workgroup: 1024,
                max_compute_workgroup_size: [1024, 1024, 64],
                ..Default::default()
            },
            subgroups: Some(SubgroupWidths { min: 32, max: 32 }),
            f16: true,
            bf16: false,
            coop: smallvec::smallvec![],
            atomic_f32: true,
            workgroup_alias: false,
            mixed_precision_coop_store: false,
            pipeline_cache: true,
            timestamp_query: false,
            simd_widths: smallvec::smallvec![],
            threads: 1,
        }
    }

    pub(crate) fn cpu_caps(name: &str, threads: u32) -> Caps {
        Caps {
            kind: DeviceKind::Cpu,
            name: name.to_string(),
            limits: Limits::default(),
            subgroups: None,
            f16: true,
            bf16: true,
            coop: smallvec::smallvec![],
            atomic_f32: true,
            workgroup_alias: false,
            mixed_precision_coop_store: false,
            pipeline_cache: false,
            timestamp_query: false,
            simd_widths: smallvec::smallvec![4, 8],
            threads,
        }
    }

    /// The seed is a function of `kind` alone. A `Caps` whose name says
    /// "Apple M2 Max" but whose kind says CPU gets the CPU table.
    #[test]
    fn seed_selected_by_kind_not_name() {
        let mislabelled = cpu_caps("Apple M2 Max", 8);
        let seed = seed_facts(&mislabelled);
        assert_eq!(seed.launch_ps, 0, "a CPU kernel is a function call");
        assert_eq!(seed.dram_bytes_per_us, 40_000);
        assert_eq!(seed.llc_bytes, 16 << 20);
        assert_eq!(seed.mac_rate(MacUnit::Fma, Dtype::F32), 32_000 * 8);
        assert_eq!(seed.saturation_lanes, 64);

        let apple = seed_facts(&gpu_caps("Apple M2 Max"));
        let other = seed_facts(&gpu_caps("NVIDIA GeForce RTX 4090"));
        assert_eq!(apple.launch_ps, other.launch_ps);
        assert_eq!(apple.dram_bytes_per_us, other.dram_bytes_per_us);
        assert_eq!(apple.llc_bytes, other.llc_bytes);
        assert_eq!(apple.wg_bytes_per_us, other.wg_bytes_per_us);
        assert_eq!(apple.mac_per_us, other.mac_per_us);
        assert_eq!(apple.trans_ps, other.trans_ps);
        assert_eq!(apple.store_ps_per_element, other.store_ps_per_element);
        assert_eq!(apple.saturation_lanes, other.saturation_lanes);
        assert_eq!(
            apple.single_buffered_traffic_pct,
            other.single_buffered_traffic_pct
        );
        assert_eq!(apple.compile_ps_per_kernel, other.compile_ps_per_kernel);
        assert_eq!(apple.thread_wake_ps, other.thread_wake_ps);
    }

    /// `DeviceFacts::fingerprint` hashes `Caps`, which carries
    /// `Limits::max_compute_workgroup_storage_size` — the coop legality
    /// filter reads that field, so the fingerprint must too.
    #[test]
    fn fingerprint_includes_workgroup_storage() {
        let mut a = gpu_caps("dev");
        let mut b = a.clone();
        a.limits.max_compute_workgroup_storage_size = 16 << 10;
        b.limits.max_compute_workgroup_storage_size = 32 << 10;
        assert_ne!(a, b);
        assert_ne!(
            seed_facts(&a).fingerprint(),
            seed_facts(&b).fingerprint(),
            "workgroup storage must be part of the facts fingerprint"
        );
        assert_ne!(a.fingerprint(), b.fingerprint());
    }

    /// The published anchors, spelled out so a typo in the table is a test
    /// failure rather than a silent 2x on every matmul decision.
    #[test]
    fn gpu_seed_matches_the_published_anchors() {
        let f = seed_facts(&gpu_caps("dev"));
        assert_eq!(f.mac_rate(MacUnit::Fma, Dtype::F32), 4_450_000);
        assert_eq!(f.mac_rate(MacUnit::Fma, Dtype::F16), 8_900_000);
        assert_eq!(f.mac_rate(MacUnit::Fma, Dtype::BF16), 8_900_000);
        assert_eq!(f.mac_rate(MacUnit::Fma, Dtype::U32), 2_225_000);
        assert_eq!(f.mac_rate(MacUnit::Fma, Dtype::I32), 2_225_000);
        assert_eq!(f.mac_rate(MacUnit::Coop, Dtype::F32), 8_900_000);
        assert_eq!(f.mac_rate(MacUnit::Coop, Dtype::F16), 17_800_000);
        assert_eq!(f.mac_rate(MacUnit::Dp4a, Dtype::I32), 17_800_000);
        assert_eq!(f.mac_rate(MacUnit::Dp4a, Dtype::U32), 17_800_000);
        assert_eq!(f.mac_rate(MacUnit::Dp4a, Dtype::F32), 1);
        assert_eq!(f.trans_ps, 4);
        assert_eq!(f.store_ps_per_element, 4);
        assert_eq!(f.single_buffered_traffic_pct, 105);
        assert_eq!(f.compile_ps_per_kernel, 1_000_000_000);
        assert_eq!(f.thread_wake_ps, 5_000_000);
    }

    /// Every field has a seed. A zero rate would divide by one and silently
    /// price a whole term at nothing.
    #[test]
    fn every_rate_is_positive_on_both_classes() {
        for f in [seed_facts(&gpu_caps("g")), seed_facts(&cpu_caps("c", 10))] {
            assert!(f.dram_bytes_per_us > 0);
            assert!(f.llc_bytes > 0);
            assert!(f.wg_bytes_per_us > 0);
            assert!(f.trans_ps > 0);
            assert!(f.store_ps_per_element > 0);
            assert!(f.saturation_lanes > 0);
            assert!(f.single_buffered_traffic_pct >= 100);
            assert!(f.compile_ps_per_kernel > 0);
            assert!(f.thread_wake_ps > 0);
            for row in f.mac_per_us {
                for rate in row {
                    assert!(rate > 0);
                }
            }
        }
    }
}
