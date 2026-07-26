//! Device-derived dispatch sizing.
//!
//! Every "how many workgroups / how wide a workgroup / how much work per
//! thread" decision in kernel dispatch reads from [`DispatchPolicy`] instead
//! of a local constant. The policy is derived from device capabilities plus
//! one calibrated parallelism floor ([`crate::Device::saturation_lanes`]),
//! so a policy value is never a shape rule: shapes enter only as the
//! arguments of the predicate being asked.

use crate::Device;

/// One full-width committed workgroup. WebGPU guarantees at least 256
/// invocations per workgroup, so this is exact on every conformant device;
/// [`DispatchPolicy::preferred_workgroup_lanes`] is its runtime clamp for
/// devices reporting less. Kept as a `const` because several kernel builders
/// take the block size as a const generic.
pub(crate) const FULL_WORKGROUP_LANES: u32 = 256;

/// Register-pressure class of a kernel body, used to pick how many outputs
/// one thread computes when a tiling trades thread-level parallelism for
/// register reuse.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RegPressure {
    /// A handful of values live across the reuse loop (elementwise bodies).
    ElementwiseFew,
}

/// Dispatch-sizing policy for one device. Cheap to construct (`Copy` data
/// gathered from cached device state); build it on demand via
/// [`crate::Device::dispatch_policy`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct DispatchPolicy {
    /// Lanes that must stay in flight before any policy trades parallelism
    /// for per-thread work. See [`crate::Device::saturation_lanes`].
    saturation_lanes: u32,
    /// Fixed subgroup width when known; 32 as a documented fallback — every
    /// policy derived from it is a floor, and 32 is the narrowest width on
    /// hardware fusor targets, so a wrong guess only keeps more parallelism.
    subgroup_width: u32,
    /// Hardware cap on one workgroup's invocations.
    max_workgroup_lanes: u32,
    last_level_cache_bytes: u64,
    /// Hardware cap on one workgroup's shared-memory bytes
    /// (`max_compute_workgroup_storage_size`: 16 KB WebGPU baseline, 32 KB
    /// on Apple silicon).
    max_workgroup_storage_bytes: u32,
    /// Physical rates the matmul cost model prices its terms in. See the
    /// accessors for units; all four are per-class floors derived in
    /// [`crate::Device::matmul_rates`].
    rates: MatmulRates,
}

/// The four device rates the cooperative-matmul cost model needs to express
/// MMA issue, threadgroup traffic, output stores and DRAM traffic in one
/// commensurable unit (femtoseconds). Integers so the argmin is exact and
/// bit-reproducible — candidate scores tie often, the tile table being built
/// from powers of two.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MatmulRates {
    /// Peak MAC issue in MACs per nanosecond.
    pub(crate) mac_per_ns: u64,
    /// Achievable DRAM bandwidth in bytes per nanosecond x10 (one decimal).
    pub(crate) dram_decibytes_per_ns: u64,
    /// Achievable threadgroup-memory bandwidth in bytes per nanosecond,
    /// covering both operand staging and cooperative fragment loads.
    pub(crate) workgroup_bytes_per_ns: u64,
    /// Accumulator zeroing, the cooperative store's fragment shuffles and the
    /// store itself, in femtoseconds per padded output element a workgroup
    /// emits *per subgroup in that workgroup* — the drain is barrier-gated
    /// across the whole workgroup, so its per-element cost tracks the
    /// workgroup's width. See the T3 term in `matmul::cost`.
    pub(crate) store_fs_per_element: u64,
    /// Threadgroup traffic of a single-buffered body as a percentage of the
    /// same traffic double-buffered. A single-buffered kernel cannot overlap
    /// the next K iteration's staging with the current iteration's MMAs, and
    /// the other three rates are all fitted on double-buffered bodies, so a
    /// body that stages from one tile pair pays this on its staging — every
    /// K iteration, which is why the penalty grows with the K loop while what
    /// it buys ([`Self::core_workgroup_slots`]) does not.
    pub(crate) single_buffered_traffic_pct: u64,
}

impl DispatchPolicy {
    pub(crate) fn from_device(device: &Device) -> Self {
        let limits = device.limits();
        let subgroup_width = if device.subgroups_supported() {
            device.max_subgroup_size().max(1)
        } else {
            32
        };
        Self::from_parts(
            device.saturation_lanes(),
            subgroup_width,
            limits
                .max_compute_workgroup_size_x
                .min(limits.max_compute_invocations_per_workgroup),
            device.last_level_cache_bytes(),
            limits.max_compute_workgroup_storage_size,
            device.matmul_rates(),
        )
    }

    pub(crate) fn from_parts(
        saturation_lanes: u32,
        subgroup_width: u32,
        max_workgroup_lanes: u32,
        last_level_cache_bytes: u64,
        max_workgroup_storage_bytes: u32,
        rates: MatmulRates,
    ) -> Self {
        Self {
            saturation_lanes: saturation_lanes.max(1),
            subgroup_width: subgroup_width.max(1),
            max_workgroup_lanes: max_workgroup_lanes.max(1),
            last_level_cache_bytes,
            max_workgroup_storage_bytes,
            rates,
        }
    }

    /// Physical rates for the cooperative-matmul cost model.
    pub(crate) fn matmul_rates(&self) -> MatmulRates {
        self.rates
    }

    /// The workgroup width for full-width dispatches.
    pub(crate) fn preferred_workgroup_lanes(&self) -> u32 {
        FULL_WORKGROUP_LANES.min(self.max_workgroup_lanes)
    }

    /// Hardware cap on one workgroup's invocations.
    pub(crate) fn max_workgroup_lanes(&self) -> u32 {
        self.max_workgroup_lanes
    }

    /// Hardware cap on one workgroup's shared-memory bytes.
    pub(crate) fn max_workgroup_storage_bytes(&self) -> u32 {
        self.max_workgroup_storage_bytes
    }

    /// The device-parallelism floor (see [`crate::Device::saturation_lanes`]).
    pub(crate) fn saturation_lanes(&self) -> u32 {
        self.saturation_lanes
    }

    /// The parallelism floor for a body that prefetches its own operands.
    /// [`Self::saturation_lanes`] carries a ~4x oversubscription factor
    /// because a latency-exposed kernel can only cover a DRAM round trip by
    /// having other warps ready to issue. A double-buffered cooperative-matmul
    /// body issues the next K tile's loads before running the current tile's
    /// MMAs, so it covers that round trip in software and reaches peak issue
    /// at half the residency.
    pub(crate) fn prefetched_saturation_lanes(&self) -> u32 {
        (self.saturation_lanes / 2).max(1)
    }

    /// How many workgroups of the given threadgroup-memory footprint a core
    /// can hold at once. Shared memory is carved from a per-core pool the
    /// same size as the per-workgroup cap, so this is a plain division —
    /// and it is the only place a kernel's footprint costs anything beyond
    /// legality. Co-resident workgroups cover each other's epilogue drain,
    /// which is why the T3 term in `matmul::cost` is divided by it.
    pub(crate) fn core_workgroup_slots(&self, workgroup_bytes: u64) -> u64 {
        (u64::from(self.max_workgroup_storage_bytes) / workgroup_bytes.max(1)).max(1)
    }

    /// Smallest workgroup worth a subgroup-accelerated whole-block
    /// reduction: two subgroups — one subgroup makes the cross-subgroup
    /// combine tree degenerate.
    pub(crate) fn min_reduction_lanes(&self) -> u32 {
        (2 * self.subgroup_width).min(self.preferred_workgroup_lanes())
    }

    /// A natural one-workgroup-per-row dispatch leaves the device idle, so a
    /// fan-out-plus-combine split pays for its combine kernel.
    pub(crate) fn should_split_for_occupancy(&self, natural_wgs: u32, wg_lanes: u32) -> bool {
        (natural_wgs as u64) * (wg_lanes as u64) < self.saturation_lanes as u64
    }

    /// Register tiling may trade threads for per-thread work only when the
    /// post-tiling thread count still saturates the device.
    pub(crate) fn tiling_leaves_saturated(&self, total_threads: u32) -> bool {
        total_threads >= self.saturation_lanes
    }

    /// Outputs one thread computes when a register-reuse tiling engages.
    /// Doubling past this halves thread count below the saturation floor at
    /// the engagement point; halving it buys too little load amortization.
    pub(crate) fn work_per_thread(&self, class: RegPressure) -> u32 {
        match class {
            RegPressure::ElementwiseFew => 4,
        }
    }

    /// Monomorphization buckets for dynamic-axis row kernels, smallest
    /// first: powers of two from half a full workgroup up to the hardware
    /// cap, at most four buckets to bound compile count.
    pub(crate) fn dynamic_block_buckets(&self) -> impl Iterator<Item = u32> + use<> {
        let start = self.preferred_workgroup_lanes() / 2;
        let max = self.max_workgroup_lanes;
        (0..4u32)
            .map(move |i| start << i)
            .filter(move |&b| b >= 1 && b <= max)
    }

    /// Data strictly below the cache watermark is treated as cache-resident:
    /// re-reads cost no bandwidth, so reuse-driven tilings should not
    /// engage. Strict comparison preserves the pre-policy gate exactly.
    pub(crate) fn cache_resident(&self, bytes: u64) -> bool {
        bytes < self.last_level_cache_bytes
    }

    /// Horizontal-merge ceiling for elementwise segments: the smallest op
    /// that could take the register-reuse tiled path must stay unmerged, so
    /// the bound is exactly the tiled path's engagement element count.
    pub(crate) fn merge_elements_bound(&self) -> usize {
        self.saturation_lanes as usize * self.work_per_thread(RegPressure::ElementwiseFew) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Apple-silicon Metal: 32-wide subgroups, 1024-lane workgroups.
    fn apple() -> DispatchPolicy {
        DispatchPolicy::from_parts(64 << 10, 32, 1024, 8 << 20, 32 << 10, crate::device::APPLE_MATMUL_RATES)
    }

    /// WebGPU baseline limits: 256-lane workgroups.
    fn webgpu_baseline() -> DispatchPolicy {
        DispatchPolicy::from_parts(64 << 10, 32, 256, 4 << 20, 16 << 10, crate::device::APPLE_MATMUL_RATES)
    }

    /// The derived values must reproduce the constants the kernels were
    /// tuned with before the policy existed. If one of these fails, a
    /// dispatch policy silently moved — decide deliberately, then update
    /// both the derivation and this pin.
    #[test]
    fn derived_values_match_legacy_constants() {
        let p = apple();
        assert_eq!(p.preferred_workgroup_lanes(), 256); // ex-BLOCK
        assert_eq!(p.min_reduction_lanes(), 64); // ex-MIN_STATIC_BLOCK
        assert_eq!(p.merge_elements_bound(), 262_144); // ex-MAX_MERGED_NARY_ELEMENTS
        assert_eq!(
            p.dynamic_block_buckets().collect::<Vec<_>>(),
            vec![128, 256, 512, 1024] // ex-ROW_DYNAMIC_BLOCKS
        );
        // ex-MIN_TILED_THREADS
        assert!(p.tiling_leaves_saturated(65_536));
        assert!(!p.tiling_leaves_saturated(65_535));
        // ex-SPLIT_ROWS_TARGET: rows < 256 at 256-lane blocks
        assert!(p.should_split_for_occupancy(255, 256));
        assert!(!p.should_split_for_occupancy(256, 256));
        // ex-NARY_TM
        assert_eq!(p.work_per_thread(RegPressure::ElementwiseFew), 4);
    }

    #[test]
    fn baseline_device_clamps() {
        let p = webgpu_baseline();
        assert_eq!(p.preferred_workgroup_lanes(), 256);
        assert_eq!(
            p.dynamic_block_buckets().collect::<Vec<_>>(),
            vec![128, 256]
        );
    }

    #[test]
    fn no_subgroup_device_floors() {
        // Subgroup width falls back to 32 → same reduction floor.
        let p = DispatchPolicy::from_parts(64 << 10, 32, 512, 4 << 20, 32 << 10, crate::device::APPLE_MATMUL_RATES);
        assert_eq!(p.min_reduction_lanes(), 64);
        assert_eq!(
            p.dynamic_block_buckets().collect::<Vec<_>>(),
            vec![128, 256, 512]
        );
    }
}
