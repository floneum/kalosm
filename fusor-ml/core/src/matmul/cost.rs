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

use fusor_tile_ir_kernels::coop_tile_entries;

use super::variants::CoopTile;
use crate::occupancy::DispatchPolicy;

type ScoreKey = (u128, u128, Reverse<u64>, Reverse<u32>, Reverse<u32>);

/// Pick the cooperative-matrix tile for an `m x k @ k x n` contraction, or
/// `None` when no coop tile is worth it (degenerate contractions route to
/// the vector/generic families).
pub(super) fn select_coop_tile(
    m: u32,
    k: u32,
    n: u32,
    policy: &DispatchPolicy,
    max_subgroup_size: u32,
) -> Option<CoopTile> {
    if m == 0 || n == 0 || k == 0 {
        return None;
    }
    // Per-tile prologue/epilogue/launch overhead in per-lane MAC-equivalents.
    let tile_overhead = u128::from(policy.saturation_lanes() / 4);
    let mut best: Option<(ScoreKey, CoopTile, u128)> = None;
    for entry in coop_tile_entries() {
        let (bm, bn, bk) = (entry.tile.bm, entry.tile.bn, entry.tile.bk);
        if entry.single_buffered {
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
        let key: ScoreKey = (
            padded_macs,
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
    let useful_macs = u128::from(m)
        * u128::from(n)
        * (u128::from(k.div_ceil(tile.bk)) * u128::from(tile.bk));
    if padded_macs * 4 > useful_macs * 5 {
        return None;
    }
    Some(tile)
}
