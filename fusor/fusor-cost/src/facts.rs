//! The shipped per-class seed facts, chosen by `Caps::kind` and replaced by
//! calibration.

use fusor_ir::cost::{DeviceFacts, RateDtype};
use fusor_ir::device::{Caps, DeviceKind};

/// Rows of [`DeviceFacts::mac_per_us`] in `MacUnit` order: half precision
/// at twice f32, integers at half, coop at twice scalar, dp4a integer-only.
const fn gpu_mac_table(fma_f32: u64, dp4a: u64) -> [[u64; RateDtype::COUNT]; 3] {
    let half = fma_f32 * 2;
    let int = fma_f32 / 2;
    // F32, F16, BF16, U32, I32
    let fma = [fma_f32, half, half, int, int];
    let coop = [fma[0] * 2, fma[1] * 2, fma[2] * 2, fma[3] * 2, fma[4] * 2];
    // `1`, not `0`, prices a float dp4a out without `mac_rate`'s clamp.
    let dp = [1, 1, 1, dp4a, dp4a];
    [fma, coop, dp]
}

/// A CPU computes halves in f32 registers, has no coop unit, and prices an
/// integer dot at four lanes per FMA slot.
const fn cpu_mac_table(fma_f32: u64) -> [[u64; RateDtype::COUNT]; 3] {
    let int = fma_f32 / 2;
    let fma = [fma_f32, fma_f32, fma_f32, int, int];
    let dp = [1, 1, 1, fma_f32 * 4, fma_f32 * 4];
    [fma, fma, dp]
}

/// The GPU seed: the one calibrated (Apple) rate vector, seeding every GPU.
pub(crate) fn seed_facts_gpu(caps: &Caps) -> DeviceFacts {
    DeviceFacts {
        // Measured on an unhidden k=1024 fragment chain.
        coop_step_ps: 450_000,
        lane_step_ps: 150_000,
        // The gap tiny kernels leave between GPU spans on Metal.
        launch_ps: 10_000_000,
        dram_bytes_per_us: 379_500,
        llc_bytes: 1 << 20,
        wg_bytes_per_us: 700_000,
        mac_per_us: gpu_mac_table(4_450_000, 17_800_000),
        trans_ps: 4,
        store_ps_per_element: 4,
        saturation_lanes: 65_536,
        single_buffered_traffic_pct: 105,
        thread_wake_ps: 5_000_000,
        caps: caps.clone(),
    }
}

/// The CPU seed: per-core rates times `Caps::threads`. A kernel is a call,
/// so `launch_ps` is zero; pool wakes are `thread_wake_ps`.
pub(crate) fn seed_facts_cpu(caps: &Caps) -> DeviceFacts {
    let threads = u64::from(caps.threads.max(1));
    DeviceFacts {
        coop_step_ps: 0,
        lane_step_ps: 0,
        launch_ps: 0,
        dram_bytes_per_us: 40_000,
        llc_bytes: 16 << 20,
        wg_bytes_per_us: 400_000,
        mac_per_us: cpu_mac_table(32_000 * threads),
        trans_ps: 20,
        store_ps_per_element: 4,
        saturation_lanes: caps.threads.max(1).saturating_mul(8),
        single_buffered_traffic_pct: 105,
        thread_wake_ps: 5_000_000,
        caps: caps.clone(),
    }
}

/// The per-class seed, by `Caps::kind` only.
pub fn seed_facts(caps: &Caps) -> DeviceFacts {
    match caps.kind {
        DeviceKind::Gpu => seed_facts_gpu(caps),
        DeviceKind::Cpu => seed_facts_cpu(caps),
    }
}
