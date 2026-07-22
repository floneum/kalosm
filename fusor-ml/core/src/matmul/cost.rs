//! General cooperative-tile selection.
//!
//! Selection enumerates the full kernel tile table and picks the argmin of a
//! lexicographic cost — no divisibility ladder, no per-shape rules. Alignment
//! enters only through what it does to the cost (edge tiles execute padded
//! MACs); the device enters only through legality and the occupancy policy.
//!
//! The key orders by, in priority:
//! 1. **Total padded MACs** (`tiles x bm*bn*k_pad`): padding is the dominant
//!    measured effect — a 33%-padded tile ran ~2x slower than its zero-pad
//!    neighbor at N=384, and a 6.7%-padded 128x512 ran ~2x slower than the
//!    least-padded choice on the vision shape.
//! 2. **Serial depth** (`waves x (per-lane tile MACs + overhead)`):
//!    workgroups within a wave run in parallel, waves serialize, and each
//!    tile pays a fixed prologue/epilogue (a quarter-saturation of per-lane
//!    MAC-equivalents). This prefers small tiles when the grid underfills
//!    the device (128x64 measured 3.64 TF/s vs 128x512's 2.15 at 1024^3 and
//!    2.6x at the 384x16384x1536 weight gradient) and large tiles when depth
//!    ties (fewer per-tile overheads at 4096^3, measured flat 5.12 = 5.12).
//! 3. **Larger tiles**: deterministic tie-break.
//!
//! Single-buffered profiles are excluded from automatic selection outright:
//! they cannot horizontally merge (`supports_horizontal_merge`), and their
//! best measured standalone advantage (+3% at 4096^3) never pays for
//! decomposing a merged wave into standalone dispatches (~15% on the vision
//! QKV family). They stay in the kernel table for forced experiments.
//!
//! Workgroup-memory footprint enters only as legality: entries whose
//! `CoopTileEntry::workgroup_bytes` for the contraction's stage element
//! exceed the device's workgroup-storage limit are unselectable (WebGPU's
//! 16 KB default would otherwise fail at pipeline creation). It stays out
//! of the score — see the residency notes on [`select_coop_tile`].
//!
//! Staging-traffic terms were evaluated and deliberately dropped: with the
//! verification tripwire in place, every earlier "large tiles stage less and
//! win" measurement turned out to be a stale-build or unverified-chain
//! artifact; verified numbers show zero-pad tiles flat except for occupancy.
//!
//! Measured anchors on M2 Max (min-of-many, verified chained bench):
//! 1024^3 — 128x64 3.64 / 64x64 2.69 / 128x512 2.15 TF/s; 384x16384x1536 —
//! 128x64 unsplit ~5.9 vs 128x512+split ~2.3; 4096^3 — 128x512 5.12 = 128x64
//! 5.12 (256x256 5.28, excluded as unmergeable); 16384x384x384 — 128x64 6.95
//! ~= 128x128 6.94, padded 128x256 3.25; 1944x1280x3840 — 64x128 ~= 128x64
//! ~2.1, padded 128x512 1.12; 1000x1024x1024 — 128x512 4.18 vs 16x64 3.85
//! (padding-first picks 16x64; accepted, least wasted work).

use std::cmp::Reverse;

use fusor_tile_ir::ScalarElement;
use fusor_tile_ir_kernels::coop_tile_entries;


use super::variants::CoopTile;
use crate::occupancy::DispatchPolicy;

type ScoreKey = (u128, u32, u128, Reverse<u64>, Reverse<u32>, Reverse<u32>);

/// Pick the cooperative-matrix tile for an `m x k @ k x n` contraction, or
/// `None` when no coop tile is worth it (degenerate contractions route to
/// the vector/generic families).
///
/// The score is a lexicographic hierarchy, each level validated against the
/// raw tile sweeps (`bench_coop_tiles`, both element types):
/// 1. `padded_macs` - total work including tile padding; waste dominates
///    everything else.
/// 2. `n_passes` - multi-pass profiles serialize barriered B-stages and lose
///    at every measured shape; reachable only when padding favors them.
/// 3. `depth` - wave-count latency model for starved grids.
/// 4. tile area / `bm` / `bn` - deterministic tie-breaks.
///
/// Threadgroup footprint is a hard legality bound and nothing more: entries
/// whose `workgroup_bytes` for the stage element exceed the device's
/// workgroup-storage limit are skipped. It is deliberately absent from the
/// score: the coop tiles are compute-bound and the sweeps show the 25.7 KB
/// double-buffered 128x64 profile beating every smaller-footprint profile,
/// so a residency term would only re-introduce the wave-model mistake of
/// rescuing wide tiles. Latency-bound kernel families that do trade on
/// residency read the real footprint from `KernelIr::workgroup_bytes` (also
/// traced per fresh build as `kernel_built workgroup_bytes=..`).
pub(super) fn select_coop_tile(
    m: u32,
    k: u32,
    n: u32,
    datatype: crate::DataTypeEnum,
    policy: &DispatchPolicy,
    max_subgroup_size: u32,
) -> Option<CoopTile> {
    if m == 0 || n == 0 || k == 0 {
        return None;
    }
    // The kernels stage operand tiles in the storage element (`staging`
    // stays off in production paths).
    let stage = match datatype {
        crate::DataTypeEnum::F16 => ScalarElement::F16,
        _ => ScalarElement::F32,
    };
    // Per-tile prologue/epilogue/launch overhead in per-lane MAC-equivalents.
    let tile_overhead = u128::from(policy.saturation_lanes() / 4);
    let mut best: Option<(ScoreKey, CoopTile, u128)> = None;
    for entry in coop_tile_entries() {
        let (bm, bn, bk) = (entry.tile.bm, entry.tile.bn, entry.tile.bk);
        if entry.single_buffered {
            continue;
        }
        if entry.workgroup_bytes(stage) > u64::from(policy.max_workgroup_storage_bytes()) {
            continue;
        }
        // The 16-wide profiles exist to avoid padding genuinely narrow
        // matrix sides. On a wide side they multiply workgroup count and
        // repeatedly stage the opposite operand; the serial-depth term can
        // otherwise prefer that artificial parallelism for long-K weight
        // gradients even though measured throughput collapses. Keep the
        // narrow profiles available through two 64-wide tiles (including the
        // 65-column vocabulary head), then use the regular tile family.
        const NARROW_PROFILE_LIMIT: u32 = 128;
        if (bm == 16 && m > NARROW_PROFILE_LIMIT) || (bn == 16 && n > NARROW_PROFILE_LIMIT) {
            continue;
        }
        let threads = entry.row_groups * entry.col_groups * max_subgroup_size;
        if threads == 0 || threads > policy.max_workgroup_lanes() {
            continue;
        }
        let tiles = u64::from(m.div_ceil(bm)) * u64::from(n.div_ceil(bn));
        let k_pad = u64::from(k.div_ceil(bk)) * u64::from(bk);
        let per_tile_macs = u128::from(bm) * u128::from(bn) * u128::from(k_pad);
        let padded_macs = u128::from(tiles) * per_tile_macs;
        let concurrent = u64::from((policy.saturation_lanes() / threads).max(1));
        let waves = u128::from(tiles.div_ceil(concurrent));
        let depth = waves * (per_tile_macs / u128::from(threads) + tile_overhead);
        // Multi-pass profiles serialize every K iteration on `n_passes`
        // barriered B-stage passes and re-read the whole A tile from
        // workgroup memory once per pass. The raw tile sweep (zero-pad,
        // saturated grids) measures them losing at every shape — no wave
        // structure trades against it: 16384x384x1536 — 128x64 (1 pass)
        // 7.40 TF/s vs 128x128 (2) 5.54, 128x256 (4) 5.53, 128x512 (8)
        // 5.34; 16384x1536x384 — 7.37 vs 3.32; 4096^3 — 6.32 vs 5.09;
        // 16384x3072x1536 — 5.56 vs 4.22. Pass count therefore orders
        // lexicographically after padding: multi-pass geometry is reachable
        // only when padding genuinely favors it.
        let key: ScoreKey = (
            padded_macs,
            entry.n_passes,
            depth,
            Reverse(u64::from(bm) * u64::from(bn)),
            Reverse(bm),
            Reverse(bn),
        );
        if best.as_ref().is_none_or(|(best_key, ..)| key < *best_key) {
            best = Some((key, CoopTile::new(bm, bn, bk), padded_macs));
        }
    }
    let (_, tile, padded_macs) = best?;
    // Even the best tile may waste more than a quarter of its work on
    // padding — degenerate (gemv-shaped) contractions with a tiny M or N.
    // Those belong to the vector/generic families; declining here is the
    // routing signal.
    let useful_macs =
        u128::from(m) * u128::from(n) * (u128::from(k.div_ceil(tile.bk)) * u128::from(tile.bk));
    if padded_macs * 4 > useful_macs * 5 {
        return None;
    }
    tracing::debug!(
        "coop_tile_select m={m} k={k} n={n} -> {}x{} bk={}",
        tile.bm,
        tile.bn,
        tile.bk
    );
    Some(tile)
}

/// Traversal-order parameter for the dense coop grid, decided per shape
/// from the raw swizzle sweeps (`bench_coop_tiles` `SWIZZLE=1,4,8,16`, both
/// element types, round-robin minima; winners with margins):
///
/// | shape (m,k,n)      | f32                | f16          |
/// |--------------------|--------------------|--------------|
/// | 16384,384,1536     | tie                | tie          |
/// | 16384,1536,384     | sw8 +4.5%          | sw1 +4.4%    |
/// | 384,16384,1536     | sw8 +4%            | sw8 +3.6%    |
/// | 16384,384,384      | tie                | tie          |
/// | 4096,4096,4096     | sw4 +2.7% (sw1 2nd)| sw1 +4.3%    |
/// | 16384,3072,1536    | sw1 +7%            | sw1 +2.2%    |
///
/// The mechanisms the winners trace: the swizzle exists to share B column
/// slabs across the resident wavefront, and it pays exactly when B is too
/// big to sit in the LLC; K-deep contractions are A-streaming-bound and
/// prefer plain row-major order (the swizzle's M-jumps break the A read
/// stream); halved f16 operands halve cache pressure and shift small-B
/// shapes to row-major too. The one point the rule leaves on the table is
/// 4096-cube f32 (sw4 beats the rule's sw8 by 2.7%) — an isolated winner a
/// dedicated branch would overfit.
pub(super) fn swizzle_group_m(
    m: usize,
    k: usize,
    n: usize,
    datatype: crate::DataTypeEnum,
) -> u32 {
    let _ = m;
    let element_size = datatype.element_size() as u128;
    let b_bytes = k as u128 * n as u128 * element_size;
    const LLC_CLASS: u128 = 32 << 20;
    const SMALL_B: u128 = 4 << 20;
    if k >= 2048 && b_bytes <= LLC_CLASS {
        // K-deep, cacheable B: streaming A wins (f32 +7%, f16 +2.2%;
        // 4096-cube f16 +4.3%).
        return 1;
    }
    if element_size == 2 && b_bytes <= SMALL_B {
        // f16 with a small B: row-major wins (+4.4% at 16384x1536x384).
        return 1;
    }
    // B-slab sharing pays when B outsizes the LLC (+4-4.5% at
    // 384x16384x1536, both dtypes) and measures as a tie on small grids.
    fusor_tile_ir_kernels::DEFAULT_SWIZZLE_GROUP_M
}

/// Split-K decision for a coop tile, pure in its inputs (see the gate and
/// divisor commentary at the call site in `kernel.rs`, which this preserves
/// verbatim). Returns the split count, or `None` to run unsplit.
#[allow(clippy::too_many_arguments)]
pub(super) fn split_k_plan(
    m: u32,
    k: u32,
    n: u32,
    batch: u32,
    tile: &CoopTile,
    policy: &DispatchPolicy,
    max_subgroup_size: u32,
    has_epilogues: bool,
) -> Option<u32> {
    if has_epilogues {
        return None;
    }
    let total_tiles = m
        .div_ceil(tile.bm)
        .checked_mul(n.div_ceil(tile.bn))?
        .checked_mul(batch)?;
    let threads = tile
        .subgroup_groups()
        .checked_mul(max_subgroup_size)
        .filter(|&threads| threads > 0)?;
    let k_iterations = k.div_ceil(tile.bk);
    let per_lane_macs = u64::from(tile.bm) * u64::from(tile.bn) * u64::from(k_iterations)
        * u64::from(tile.bk)
        / u64::from(threads);
    if !policy.split_amortizes_combine(total_tiles, threads)
        || k_iterations < 4
        || per_lane_macs < u64::from(policy.saturation_lanes() / 4)
    {
        return None;
    }
    let concurrent = (policy.saturation_lanes() / threads).max(1);
    let target = concurrent.div_ceil(total_tiles).clamp(2, k_iterations / 2);
    let splits = (2..=target)
        .rev()
        .find(|candidate| k_iterations.is_multiple_of(*candidate))
        .unwrap_or(target);
    Some(splits)
}

/// The dense matmul family's complete routing for one contraction, pure in
/// device-derived inputs: family variant, coop tile, split-K count, and
/// traversal group. One entry point makes the whole decision surface
/// golden-testable — new scoring terms change this function's output table
/// or they change nothing. Test-only: production composes the same pure
/// functions at its own call sites; the golden test locks them through this.
#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct DensePlan {
    pub variant: super::variants::DenseMatmulVariant,
    pub tile: Option<CoopTile>,
    pub splits: Option<u32>,
    pub swizzle_group_m: u32,
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn plan_dense_matmul(
    m: usize,
    k: usize,
    n: usize,
    batch: u32,
    datatype: crate::DataTypeEnum,
    policy: &DispatchPolicy,
    max_subgroup_size: u32,
    caps: crate::kernel_selection::KernelDeviceCaps,
) -> DensePlan {
    let ctx = super::variants::DenseMatmulCtx {
        coop_kinds: super::variants::dense_coop_kinds_from_datatype(datatype),
    };
    let shape = crate::kernel_selection::KernelShape::new([m, k, n]);
    let variant = super::variants::dense_matmul_selector()
        .select(shape, &ctx, caps)
        .expect("dense matmul selector has a catch-all rule");
    let tile = (variant == super::variants::DenseMatmulVariant::Coop)
        .then(|| {
            CoopTile::select(
                m as u32,
                k as u32,
                n as u32,
                datatype,
                policy,
                max_subgroup_size,
            )
        })
        .flatten();
    let splits = tile.as_ref().and_then(|tile| {
        split_k_plan(
            m as u32,
            k as u32,
            n as u32,
            batch,
            tile,
            policy,
            max_subgroup_size,
            false,
        )
    });
    DensePlan {
        variant,
        tile,
        splits,
        swizzle_group_m: swizzle_group_m(m, k, n, datatype),
    }
}
