//! The individual roofline terms, `core/src/matmul/cost.rs::score_fs` ported
//! term for term.
//!
//! Everything here is integer arithmetic on `u128`. Candidates tie exactly —
//! the tile table is built from powers of two and several terms are
//! invariant to `n_passes` by construction — so the argmin has to be
//! bit-reproducible across platforms, which a floating `powf` is not.
//!
//! `score_fs` maps on one for one: T1 -> [`math_ps`], T2 -> [`wg_ps`],
//! T3 -> [`drain_ps`], T4 -> [`dram_ps`] under the `max`, T5 ->
//! [`combine_ps`], and the cube-root occupancy shortfall ->
//! [`occupancy_scale_num_den`].
//!
//! Two things the reference has are deliberately absent. There is no
//! `padded_macs * 4 > useful_macs * 5` routing guard: padded MACs enter
//! [`math_ps`] through `Work::macs` computed on the padded tile, so an
//! over-padded cooperative candidate simply prices above sgemv instead of
//! being routed around it. And the strict `<` LLC watermark is gone —
//! [`dram_ps`] interpolates, so one byte over the line cannot flip a tiling
//! plan.
//!
//! Owned by W6.

use fusor2_ir::cost::{DeviceFacts, MacUnit, Picoseconds};
use fusor2_ir::dtype::Dtype;
use fusor2_ir::facts::Work;

/// Picoseconds per microsecond. Every rate on `DeviceFacts` is per
/// microsecond, so every term is `quantity * PS_PER_US / rate`.
const PS_PER_US: u128 = 1_000_000;

/// Floor of the `n`th root, by Newton iteration on integers so the argmin
/// stays exactly reproducible across platforms. Ported verbatim from
/// `cost.rs:295`.
pub fn integer_root(value: u128, n: u32) -> u128 {
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
/// The reference has no equivalent, which is why its view folding is gated
/// on a hardcoded reread threshold instead of priced.
///
/// **No occupancy scaling and no traffic.** This is the term
/// `CostModel::node_math` returns and the admissible lower bound is built
/// from it; adding either would break admissibility.
pub fn math_ps(facts: &DeviceFacts, work: Work, unit: MacUnit, dtype: Dtype) -> Picoseconds {
    let mac_rate = u128::from(facts.mac_rate(unit, dtype));
    let index_rate = u128::from(facts.mac_rate(MacUnit::Fma, Dtype::U32));
    let t = u128::from(work.macs) * PS_PER_US / mac_rate
        + u128::from(work.transcendentals) * u128::from(facts.trans_ps)
        + u128::from(work.index_ops) * PS_PER_US / index_rate;
    ps(t)
}

/// T2: workgroup-memory traffic.
///
/// `bytes` is `fragment_bytes + stage_bytes` as `cost.rs:201-202` computes
/// them, summed over the whole launch — the caller supplies it from
/// `Work::wg_bytes`, never from a re-estimate. `staging == 1` loses the
/// load/MMA overlap the rates were fitted on and pays
/// `single_buffered_traffic_pct`.
pub fn wg_ps(facts: &DeviceFacts, bytes: u64, staging: u8) -> Picoseconds {
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
/// `arena_bytes` is the exact `ArenaPlan::total_bytes`. Never re-estimate
/// it — an estimator here is how the reference's packing result silently
/// changes tile selection on the next build.
pub fn drain_ps(
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
///
/// The reference's `DispatchPolicy::cache_resident` is a strict `<`, which
/// makes one byte over 8 MiB flip the tiling plan. That cliff is deleted.
pub fn effective_read_bytes(llc_bytes: u64, bytes: u64, rereads: u32) -> u128 {
    let bytes = u128::from(bytes);
    let rereads = u128::from(rereads.max(1));
    if bytes <= u128::from(llc_bytes) {
        return bytes;
    }
    let eff = bytes + (rereads - 1) * (bytes - u128::from(llc_bytes));
    eff.clamp(bytes, bytes * rereads)
}

/// T4's input: **reads and writes** — the reference's read-traffic spike
/// flag, shipped on by default. A write-only byte term makes producer
/// inlining look free and mis-prices every fusion in the graph.
///
/// `reads` is one `(bytes, rereads)` pair per *distinct* operand, so a value
/// two consumers share is counted once and its reread factor carries the
/// sharing.
pub fn dram_ps(facts: &DeviceFacts, reads: &[(u64, u32)], writes: u64) -> Picoseconds {
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
/// of the core's issue slots, its threadgroup port and its share of L2 to
/// themselves. Measured on the split-K sweeps, where the split count is
/// exactly a lane-count dial, a cube root reproduces the curves and a linear
/// law over-splits every one of them by 2-4x.
///
/// The target is `saturation_lanes / 2` — the reference's
/// `prefetched_saturation_lanes`, whose only role is as a parallelism floor,
/// never an execution width and never a MAC-equivalent.
pub fn occupancy_scale_num_den(facts: &DeviceFacts, resident_lanes: u64) -> (u128, u128) {
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

/// [`occupancy_scale_num_den`] as fixed point with a denominator of 1000,
/// for callers that want one number. `1000` means "saturated".
pub fn occupancy_scale(facts: &DeviceFacts, resident_lanes: u64) -> u64 {
    let (num, den) = occupancy_scale_num_den(facts, resident_lanes);
    u64::try_from(num * 1_000 / den).unwrap_or(u64::MAX)
}

/// Apply an occupancy rational to a duration, saturating.
pub fn scaled(value: Picoseconds, num: u128, den: u128) -> Picoseconds {
    ps(u128::from(value.0) * num / den.max(1))
}

/// T5: the combine dispatch reads every partial slice and writes the output
/// behind its own barrier, so it **adds** rather than overlapping.
///
/// `padded_bytes` is one split's padded output, so `(splits + 1)` counts
/// reading `splits` partials and writing one result.
pub fn combine_ps(facts: &DeviceFacts, splits: u32, padded_bytes: u64) -> Picoseconds {
    if splits <= 1 {
        return Picoseconds(0);
    }
    ps(u128::from(splits + 1) * u128::from(padded_bytes) * PS_PER_US
        / u128::from(facts.dram_bytes_per_us.max(1)))
}

/// Grid traversal swizzle: how many M-lines of the cooperative grid walk
/// together before advancing N.
///
/// The swizzle exists to share B column slabs across the resident wavefront,
/// and it pays exactly when B is too big to sit in the last-level cache.
/// This reads `llc_bytes` from the one device-fact source; the reference's
/// private `LLC_CLASS = 32 MiB` and `SMALL_B = 4 MiB` constants — unrelated
/// to its own `Device::last_level_cache_bytes()` — are deleted.
///
/// At Apple's 8 MiB this makes the 4096-cube f32 case (64 MiB of B) pick 4,
/// which is the reference's documented 2.7% miss.
pub fn swizzle_group_m(facts: &DeviceFacts, b_bytes: u64) -> u32 {
    if b_bytes <= facts.llc_bytes / 2 {
        1
    } else if b_bytes <= facts.llc_bytes * 8 {
        4
    } else {
        8
    }
}

// ---------------------------------------------------------------------------
// The four named per-kernel parameters
// ---------------------------------------------------------------------------
//
// The reference has one `work_per_thread = 4` governing elementwise register
// tiling, row-program lane-group narrowing, row-program register staging and
// the horizontal-merge element bound. One number tuned for elementwise
// register pressure silently controls three unrelated policies. Here each is
// its own function of `DeviceFacts` plus that kernel's own shape, and **none
// reads another's value**.

/// Live vector registers an elementwise body can hold staged outputs in:
/// ~32 architectural vector registers at ~8 per staged output.
const ELEMENTWISE_STAGED_OUTPUTS: u64 = 4;
/// Registers a row reduction can spend on staged reads before it spills its
/// accumulator. Wider than the elementwise class per value because the
/// reduction stages plain scalars, narrower in total because it also carries
/// per-phase state.
const ROW_STAGED_READS: u64 = 8;
/// Per-lane staging pressure of a merged wave's body. A merged segment
/// carries the union of its members' live values, so its budget is the
/// tightest of the four.
const MERGED_STAGED_VALUES: u64 = 4;

/// Outputs one thread computes when an elementwise register-reuse tiling
/// engages. Bounded above by the body's register budget and below by keeping
/// the post-tiling thread count at the parallelism floor.
pub fn elementwise_work_per_thread(facts: &DeviceFacts, elements: u64) -> u32 {
    let by_occupancy = elements / u64::from(facts.saturation_lanes).max(1);
    by_occupancy.clamp(1, ELEMENTWISE_STAGED_OUTPUTS) as u32
}

/// Narrowest lane group a row program may use for a reduction of extent `k`
/// over `rows` rows.
///
/// A group narrower than this leaves the dispatch short of the parallelism
/// floor; a group wider than `k` wastes lanes outright. Rounded up to a
/// power of two so the cross-lane tree stays balanced.
pub fn row_lane_group_floor(facts: &DeviceFacts, k: u64, rows: u64) -> u32 {
    let need = u64::from(facts.saturation_lanes)
        .max(1)
        .div_ceil(rows.max(1));
    let floor = need.clamp(1, k.max(1)).next_power_of_two();
    u32::try_from(floor).unwrap_or(u32::MAX)
}

/// Elements one lane may stage into registers before a row program falls
/// back to re-reading storage per phase. Zero means "do not stage".
pub fn row_staging_budget(facts: &DeviceFacts, k: u64, lanes_per_row: u32) -> u32 {
    // Deliberately independent of every other parameter here: the reference
    // shares one constant across this and three unrelated policies.
    let _ = facts;
    let per_lane = k.div_ceil(u64::from(lanes_per_row.max(1)));
    if per_lane > ROW_STAGED_READS {
        0
    } else {
        per_lane as u32
    }
}

/// Elementwise element-count ceiling for joining a merged wave.
///
/// A segment big enough to saturate the device on its own gains nothing from
/// sharing a dispatch, and a segment whose operands no longer fit the
/// last-level cache is bandwidth-bound rather than launch-bound. The bound
/// is the tighter of the two, so it degrades on a device with a small cache
/// instead of staying pinned at the reference's 262,144.
pub fn merge_elements_bound(facts: &DeviceFacts, elem_bytes: u32) -> u64 {
    let by_occupancy = u64::from(facts.saturation_lanes).max(1) * MERGED_STAGED_VALUES;
    let by_cache = facts.llc_bytes / u64::from(elem_bytes.max(1));
    by_occupancy.min(by_cache).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::facts::seed_facts;
    use crate::facts::tests::gpu_caps;

    /// Test 8.
    #[test]
    fn integer_root_is_exact() {
        for n in [3u32, 4] {
            let mut previous = 0u128;
            for x in 1u128..=4096 {
                let exact = integer_root(x.pow(n), n);
                assert_eq!(exact, x, "integer_root({x}^{n}, {n})");
                // One below a perfect power floors to x-1.
                assert_eq!(integer_root(x.pow(n) - 1, n), x - 1);
                assert!(exact >= previous, "integer_root must be monotone");
                previous = exact;
            }
            assert_eq!(integer_root(0, n), 0);
            assert_eq!(integer_root(1, n), 1);
        }
        // Monotone across arbitrary values, not just perfect powers.
        let mut last = 0;
        for v in (0u128..100_000).step_by(37) {
            let r = integer_root(v, 3);
            assert!(r >= last);
            last = r;
        }
    }

    /// Test 3.
    #[test]
    fn traffic_counts_reads_and_writes() {
        let f = seed_facts(&gpu_caps("dev"));
        let four_mib = 4u64 << 20;
        let got = dram_ps(&f, &[(four_mib, 1)], four_mib);
        let want = u128::from(8u64 << 20) * PS_PER_US / u128::from(f.dram_bytes_per_us);
        assert_eq!(u128::from(got.0), want);
        // Half of it is exactly what a write-only term would have said.
        let write_only = dram_ps(&f, &[], four_mib);
        assert_eq!(got.0, write_only.0 * 2);
    }

    /// Test 4. Continuity at the watermark, monotonicity in both arguments,
    /// and the large-working-set asymptote.
    #[test]
    fn llc_reread_is_continuous() {
        let f = seed_facts(&gpu_caps("dev"));
        let llc = f.llc_bytes;
        let at = |b: u64, r: u32| dram_ps(&f, &[(b, r)], 0).0 as f64;

        let below = at(llc - 1, 4);
        let above = at(llc + 1, 4);
        let on = at(llc, 4);
        assert!(
            (below - above).abs() / on < 0.001,
            "discontinuity at the watermark: {below} vs {above}"
        );

        // Monotone non-decreasing in bytes at fixed rereads.
        let mut previous = 0.0;
        for step in 0..64u64 {
            let bytes = (llc / 2) + step * (llc / 8);
            let now = at(bytes, 3);
            assert!(now >= previous, "traffic fell as bytes grew");
            previous = now;
        }
        // Monotone non-decreasing in rereads at fixed bytes.
        let mut previous = 0.0;
        for rereads in 1..32u32 {
            let now = at(4 * llc, rereads);
            assert!(now >= previous, "traffic fell as rereads grew");
            previous = now;
        }

        // Asymptote. `eff = bytes + (r-1)*(bytes - llc)` is 9.4% short of a
        // full `r * bytes` at 8x the cache and within 1% by ~75x. The W6
        // spec asserts 1% at 8x; that is the stated formula's arithmetic,
        // not its intent, so both points are pinned here and the deviation
        // is reported rather than papered over.
        let full = |b: u64, r: u32| {
            (u128::from(b) * u128::from(r) * PS_PER_US / u128::from(f.dram_bytes_per_us)) as f64
        };
        let ratio_8x = at(8 * llc, 4) / full(8 * llc, 4);
        assert!(
            (0.90..=1.0).contains(&ratio_8x),
            "8x the cache should count almost every reread, got {ratio_8x}"
        );
        let ratio_128x = at(128 * llc, 4) / full(128 * llc, 4);
        assert!(
            ratio_128x > 0.99,
            "far past the cache every reread must be counted, got {ratio_128x}"
        );
        // A working set inside the cache is free to reread.
        assert_eq!(at(llc / 2, 1), at(llc / 2, 16));
    }

    /// Test 6.
    #[test]
    fn swizzle_reads_llc_bytes() {
        let mut f = seed_facts(&gpu_caps("dev"));
        // 4096-cube f32: B is k*n*4 = 64 MiB.
        let b_bytes = 4096u64 * 4096 * 4;
        assert_eq!(f.llc_bytes, 8 << 20);
        assert_eq!(swizzle_group_m(&f, b_bytes), 4);

        f.llc_bytes = 512 << 20;
        assert_eq!(swizzle_group_m(&f, b_bytes), 1);

        f.llc_bytes = 8 << 20;
        assert_eq!(swizzle_group_m(&f, 256 << 20), 8);
    }

    /// The occupancy law is a cube root of the shortfall against half the
    /// saturation floor, and exactly 1 once saturated.
    #[test]
    fn occupancy_is_a_cube_root_of_the_shortfall() {
        let f = seed_facts(&gpu_caps("dev"));
        assert_eq!(occupancy_scale_num_den(&f, 32_768), (1, 1));
        assert_eq!(occupancy_scale_num_den(&f, 1 << 20), (1, 1));
        // 8x short of the target scales by 2.
        assert_eq!(occupancy_scale_num_den(&f, 32_768 / 8), (2_000, 1_000));
        // 20,480 lanes against a 32,768 target: 1.6^(1/3).
        let (num, den) = occupancy_scale_num_den(&f, 20_480);
        let scale = num as f64 / den as f64;
        assert!((scale - 1.6f64.cbrt()).abs() < 0.001, "{scale}");
    }

    /// The four per-kernel parameters are independent functions. Changing a
    /// fact only one of them reads must move only that one.
    #[test]
    fn per_kernel_parameters_are_independent() {
        let f = seed_facts(&gpu_caps("dev"));
        assert_eq!(elementwise_work_per_thread(&f, 1_000), 1);
        assert_eq!(elementwise_work_per_thread(&f, 4 * 65_536), 4);
        assert_eq!(elementwise_work_per_thread(&f, 1 << 30), 4);

        // A 64-row reduction needs 1024 lanes per row to reach the floor,
        // clamped by k.
        assert_eq!(row_lane_group_floor(&f, 4_096, 64), 1_024);
        assert_eq!(row_lane_group_floor(&f, 128, 64), 128);
        assert_eq!(row_lane_group_floor(&f, 4_096, 1 << 20), 1);

        assert_eq!(row_staging_budget(&f, 256, 64), 4);
        assert_eq!(row_staging_budget(&f, 512, 64), 8);
        assert_eq!(row_staging_budget(&f, 1_024, 64), 0);

        // Occupancy-bound on Apple's 8 MiB cache; cache-bound once the cache
        // shrinks, which is the whole point of not pinning a constant.
        assert_eq!(merge_elements_bound(&f, 4), 262_144);
        let mut small = f.clone();
        small.llc_bytes = 256 << 10;
        assert_eq!(merge_elements_bound(&small, 4), 65_536);
    }

    /// `math_ps` is a genuine lower bound: no traffic, no occupancy, and
    /// `index_ops` priced at the integer rate rather than the float one.
    #[test]
    fn math_term_prices_index_ops_at_the_integer_rate() {
        let f = seed_facts(&gpu_caps("dev"));
        let macs_only = math_ps(
            &f,
            Work {
                macs: 4_450_000,
                ..Default::default()
            },
            MacUnit::Fma,
            Dtype::F32,
        );
        assert_eq!(macs_only.0, 1_000_000, "one microsecond of f32 FMAs");

        let index_only = math_ps(
            &f,
            Work {
                index_ops: 2_225_000,
                ..Default::default()
            },
            MacUnit::Fma,
            Dtype::F32,
        );
        assert_eq!(index_only.0, 1_000_000, "index ops issue at the u32 rate");

        let trans_only = math_ps(
            &f,
            Work {
                transcendentals: 1_000,
                ..Default::default()
            },
            MacUnit::Fma,
            Dtype::F32,
        );
        assert_eq!(trans_only.0, 4_000);

        // The coop unit is twice the scalar unit at the same dtype.
        let coop = math_ps(
            &f,
            Work {
                macs: 4_450_000,
                ..Default::default()
            },
            MacUnit::Coop,
            Dtype::F32,
        );
        assert_eq!(coop.0, 500_000);
    }

    /// Single-buffered staging pays the fitted overlap penalty; anything
    /// else does not.
    #[test]
    fn wg_term_charges_single_buffering() {
        let f = seed_facts(&gpu_caps("dev"));
        assert_eq!(wg_ps(&f, 700_000, 2).0, 1_000_000);
        assert_eq!(wg_ps(&f, 700_000, 1).0, 1_050_000);
    }

    /// The drain's fourth root: halving the arena footprint doubles the
    /// slots and buys ~16%, not ~50%.
    #[test]
    fn drain_uses_a_fourth_root_of_residency() {
        let f = seed_facts(&gpu_caps("dev"));
        let max_wg = f.caps.limits.max_compute_workgroup_storage_size;
        let one = drain_ps(&f, 1 << 20, 8, max_wg, max_wg).0 as f64;
        let two = drain_ps(&f, 1 << 20, 8, max_wg / 2, max_wg).0 as f64;
        let ratio = two / one;
        assert!(
            (0.80..0.87).contains(&ratio),
            "the fourth root of 2 is 0.841, got {ratio}"
        );
    }

    /// T5 is zero without a split and linear in the split count with one.
    #[test]
    fn combine_only_exists_for_split_k() {
        let f = seed_facts(&gpu_caps("dev"));
        assert_eq!(combine_ps(&f, 1, 1 << 20).0, 0);
        assert_eq!(combine_ps(&f, 0, 1 << 20).0, 0);
        let two = combine_ps(&f, 2, 1 << 20).0;
        let four = combine_ps(&f, 4, 1 << 20).0;
        assert_eq!(four, two / 3 * 5);
    }
}
