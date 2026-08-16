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
