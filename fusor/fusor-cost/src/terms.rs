//! The roofline terms, in `u128` integer arithmetic so the argmin is
//! bit-reproducible across platforms.

use fusor_ir::cost::{DeviceFacts, MacUnit, Picoseconds};
use fusor_ir::dtype::Dtype;
use fusor_ir::facts::Work;

/// Picoseconds per microsecond; every `DeviceFacts` rate is per microsecond.
const PS_PER_US: u128 = 1_000_000;

/// Floor of the `n`th root, by integer Newton iteration.
pub(crate) fn integer_root(value: u128, n: u32) -> u128 {
    if value < 2 {
        return value;
    }
    let mut x = 1u128 << (value.ilog2() / n + 1);
    loop {
        let next = ((u128::from(n) - 1) * x + value / x.pow(n - 1)) / u128::from(n);
        if next >= x {
            return x;
        }
        x = next;
    }
}

fn ps(value: u128) -> Picoseconds {
    Picoseconds(u64::try_from(value).unwrap_or(u64::MAX))
}

/// T1: MACs, transcendentals and index ops at their unit rates. No occupancy
/// or traffic: the admissible lower bound is built from it.
pub(crate) fn math_ps(facts: &DeviceFacts, work: Work, unit: MacUnit, dtype: Dtype) -> Picoseconds {
    let mac_rate = u128::from(facts.mac_rate(unit, dtype));
    let index_rate = u128::from(facts.mac_rate(MacUnit::Fma, Dtype::U32));
    let t = u128::from(work.macs) * PS_PER_US / mac_rate
        + u128::from(work.transcendentals) * u128::from(facts.trans_ps)
        + u128::from(work.index_ops) * PS_PER_US / index_rate;
    ps(t)
}

/// T2: workgroup-memory traffic; single staging pays
/// `single_buffered_traffic_pct`.
pub(crate) fn wg_ps(facts: &DeviceFacts, bytes: u64, staging: u8) -> Picoseconds {
    let pct = if staging == 1 {
        u128::from(facts.single_buffered_traffic_pct)
    } else {
        100
    };
    ps(u128::from(bytes) * PS_PER_US * pct / (u128::from(facts.wg_bytes_per_us.max(1)) * 100))
}

/// T3: the epilogue drain, per padded output element and per subgroup,
/// divided by the fourth root of co-resident workgroups (measured; a
/// reciprocal overstates the swing 4x).
pub(crate) fn drain_ps(
    facts: &DeviceFacts,
    padded_out_elems: u64,
    subgroups: u32,
    arena_bytes: u32,
    max_wg_storage: u32,
) -> Picoseconds {
    let core_slots = u128::from(max_wg_storage / arena_bytes.max(1)).max(1);
    let numerator = u128::from(padded_out_elems)
        * u128::from(facts.store_ps_per_element)
        * u128::from(subgroups.max(1))
        * 1_000;
    ps(numerator / integer_root(core_slots * 1_000_000_000_000, 4).max(1))
}

/// Effective bytes of one operand read `rereads` times: `bytes` while it
/// fits the LLC, rising continuously toward `bytes * rereads` past it.
pub(crate) fn effective_read_bytes(llc_bytes: u64, bytes: u64, rereads: u32) -> u128 {
    let bytes = u128::from(bytes);
    let rereads = u128::from(rereads.max(1));
    if bytes <= u128::from(llc_bytes) {
        return bytes;
    }
    let eff = bytes + (rereads - 1) * (bytes - u128::from(llc_bytes));
    eff.clamp(bytes, bytes * rereads)
}

/// T4: DRAM traffic; `reads` has one `(bytes, rereads)` per distinct operand.
pub(crate) fn dram_ps(
    facts: &DeviceFacts,
    reads: &[(u64, u32)],
    writes: u64,
    line_bytes: u64,
) -> Picoseconds {
    let mut total = u128::from(writes) + u128::from(line_bytes);
    for &(bytes, rereads) in reads {
        total += effective_read_bytes(facts.llc_bytes, bytes, rereads);
    }
    ps(total * PS_PER_US / u128::from(facts.dram_bytes_per_us.max(1)))
}

/// The occupancy shortfall as a rational `(num, den)`: the cube root of how
/// far the grid falls short of `saturation_lanes / 2` (measured on split-K).
pub(crate) fn occupancy_scale_num_den(facts: &DeviceFacts, resident_lanes: u64) -> (u128, u128) {
    let target = u128::from(facts.saturation_lanes / 2).max(1);
    let resident = u128::from(resident_lanes).max(1);
    if resident >= target {
        return (1, 1);
    }
    (
        integer_root(target * 1_000_000_000 / resident, 3).max(1),
        1_000,
    )
}

/// Apply an occupancy rational to a duration, saturating.
pub(crate) fn scaled(value: Picoseconds, num: u128, den: u128) -> Picoseconds {
    ps(u128::from(value.0) * num / den.max(1))
}

/// One workgroup's dependent chain at the device's step latencies.
pub(crate) fn serial_ps(facts: &DeviceFacts, coop: u64, lane: u64) -> Picoseconds {
    Picoseconds(
        coop.saturating_mul(facts.coop_step_ps)
            .saturating_add(lane.saturating_mul(facts.lane_step_ps)),
    )
}
