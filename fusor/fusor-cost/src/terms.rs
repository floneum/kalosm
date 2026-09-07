//! The individual roofline terms.
//!
//! Everything here is integer arithmetic on `u128`. Candidates tie exactly,
//! so the argmin has to be bit-reproducible across platforms.

use fusor_ir::cost::{DeviceFacts, MacUnit, Picoseconds};
use fusor_ir::dtype::Dtype;
use fusor_ir::facts::Work;

/// Picoseconds per microsecond. Every rate on `DeviceFacts` is per
/// microsecond, so every term is `quantity * PS_PER_US / rate`.
const PS_PER_US: u128 = 1_000_000;

/// Floor of the `n`th root, by Newton iteration on integers so the argmin
/// stays exactly reproducible across platforms.
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

/// T1 plus `index_ops`.
///
/// `macs / mac_rate(unit, dtype) + transcendentals * trans_ps + index_ops /
/// mac_rate(Fma, U32)`. The last addend is the view-fold-vs-gather term: an
/// aliased operand pays no index arithmetic, a gather pays one integer op
/// per element, and an unflattened conv-window operand pays one per divmod.
///
/// **No occupancy scaling and no traffic.** This is the term
/// `CostModel::node_math` returns and the admissible lower bound is built
/// from it; adding either would break admissibility.
pub(crate) fn math_ps(facts: &DeviceFacts, work: Work, unit: MacUnit, dtype: Dtype) -> Picoseconds {
    let mac_rate = u128::from(facts.mac_rate(unit, dtype));
    let index_rate = u128::from(facts.mac_rate(MacUnit::Fma, Dtype::U32));
    let t = u128::from(work.macs) * PS_PER_US / mac_rate
        + u128::from(work.transcendentals) * u128::from(facts.trans_ps)
        + u128::from(work.index_ops) * PS_PER_US / index_rate;
    ps(t)
}

/// T2: workgroup-memory traffic.
///
/// `bytes` is `fragment_bytes + stage_bytes` summed over the whole launch —
/// the caller supplies it from `Work::wg_bytes`, never from a re-estimate.
/// `staging == 1` loses the load/MMA overlap the rates were fitted on and
/// pays `single_buffered_traffic_pct`.
pub(crate) fn wg_ps(facts: &DeviceFacts, bytes: u64, staging: u8) -> Picoseconds {
    let pct = if staging == 1 {
        u128::from(facts.single_buffered_traffic_pct)
    } else {
        100
    };
    ps(u128::from(bytes) * PS_PER_US * pct / (u128::from(facts.wg_bytes_per_us.max(1)) * 100))
}

/// T3: accumulator zeroing, the cooperative store's fragment shuffles and
/// the store itself, over the padded output every workgroup emits.
///
/// Per element **and per subgroup** of the emitting workgroup: the epilogue
/// is a whole-workgroup drain behind one barrier, so a wider workgroup
/// serializes more of its output through the same threadgroup port. Divided
/// by the fourth root of how many workgroups of this arena footprint a core
/// holds at once, because a co-resident workgroup covers part of the drain.
///
/// The fourth root is load-bearing: a reciprocal predicts a 55% swing where
/// the measured pair (halving the footprint at fixed tile, splits and grid)
/// is 14%.
///
/// `arena_bytes` is the exact `ArenaPlan::total_bytes`, never a re-estimate.
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

/// Effective byte count of one operand read `rereads` times.
///
/// Continuous in `llc_bytes` with no discontinuity at the watermark: at
/// `bytes == llc_bytes` the interpolation is exactly `bytes`, and it rises
/// monotonically toward `bytes * rereads` as the working set outgrows the
/// cache. `bytes * (1 + (r - 1) * (bytes - llc) / bytes)` is exactly
/// `bytes + (r - 1) * (bytes - llc)`, so it is computed that way and stays
/// integral.
pub(crate) fn effective_read_bytes(llc_bytes: u64, bytes: u64, rereads: u32) -> u128 {
    let bytes = u128::from(bytes);
    let rereads = u128::from(rereads.max(1));
    if bytes <= u128::from(llc_bytes) {
        return bytes;
    }
    let eff = bytes + (rereads - 1) * (bytes - u128::from(llc_bytes));
    eff.clamp(bytes, bytes * rereads)
}

/// T4: DRAM traffic, reads and writes.
///
/// `reads` is one `(bytes, rereads)` pair per *distinct* operand, so a value
/// two consumers share is counted once and its reread factor carries the
/// sharing.
pub(crate) fn dram_ps(facts: &DeviceFacts, reads: &[(u64, u32)], writes: u64) -> Picoseconds {
    let mut total = u128::from(writes);
    for &(bytes, rereads) in reads {
        total += effective_read_bytes(facts.llc_bytes, bytes, rereads);
    }
    ps(total * PS_PER_US / u128::from(facts.dram_bytes_per_us.max(1)))
}

/// The occupancy shortfall as an exact rational `(num, den)`.
///
/// A grid short of the parallelism floor does not lose issue rate in
/// proportion to the lanes it is missing: the lanes it does have keep more
/// of the core's issue slots, its threadgroup port and its share of Kernel to
/// themselves. Measured on the split-K sweeps, a cube root reproduces the
/// curves.
///
/// The target is `saturation_lanes / 2`, a parallelism floor — never an
/// execution width and never a MAC-equivalent.
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

/// T5: the combine dispatch reads every partial slice and writes the output
/// behind its own barrier, so it **adds** rather than overlapping.
///
/// `padded_bytes` is one split's padded output, so `(splits + 1)` counts
/// reading `splits` partials and writing one result.
pub(crate) fn combine_ps(facts: &DeviceFacts, splits: u32, padded_bytes: u64) -> Picoseconds {
    if splits <= 1 {
        return Picoseconds(0);
    }
    ps(
        u128::from(splits + 1) * u128::from(padded_bytes) * PS_PER_US
            / u128::from(facts.dram_bytes_per_us.max(1)),
    )
}
