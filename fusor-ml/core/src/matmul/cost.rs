//! General cooperative-tile selection.
//!
//! Selection enumerates the full kernel tile table and picks the argmin of a
//! constant-free lexicographic cost — no divisibility ladder, no per-shape
//! rules, no tuned weights. Alignment enters only through what it does to
//! the cost (edge tiles execute padded MACs); the device enters only through
//! legality.
//!
//! The cost is `padded_macs + W x staging_elems`:
//! - **Padded MACs** (`tiles x bm*bn*k_pad`): edge tiles execute their pad
//!   region, and padding measured brutally real (a 33%-padded tile ran ~2x
//!   slower than its zero-pad neighbor at N=384; a 6.7%-padded 128x512 ran
//!   ~2x slower than the least-padded choice on the vision shape).
//! - **Staging traffic** (`tiles x k_pad*(bm+bn)` elements through
//!   workgroup memory): larger tiles re-stage less per MAC; this is what
//!   actually discriminates among zero-pad tiles (128x512 measured 2x
//!   faster than 128x64 at 1024^3, +9% at 4096^3).
//! - **W = 4** MAC-equivalents per staged element — the one calibrated
//!   constant, and it is bounded on both sides by measurement: W > 0.24 or
//!   16x64 would beat 128x512 at m=1000 (measured 3.85 vs 4.18 TF/s), and
//!   W < 7.96 or 128x512's 6.7% padding would beat 64x128 on the vision
//!   shape (measured 1.12 vs ~2.1 TF/s). The midpoint power of two is 4.
//!
//! Single-buffered profiles are excluded from automatic selection outright:
//! they cannot horizontally merge (`supports_horizontal_merge`), and their
//! best measured standalone advantage (+1.3% at 4096^3) never pays for
//! decomposing a merged wave into standalone dispatches (~15% on the vision
//! QKV family). They stay in the kernel table for forced experiments; if a
//! merge-aware selection context ever lands, revisit. Remaining ties break
//! toward larger tiles (fewer workgroups amortize prologue/epilogue).
//!
//! Occupancy terms were evaluated and deliberately dropped: across every
//! measured shape (16384x384x384, 384x1024x1024, 8192x1024x256, 1024^3,
//! 4096^3, 1944x1280x3840, 1000x1024x1024) workgroup-count differences never
//! showed outside noise once padding and staging were accounted for.
//!
//! Measured anchors on M2 Max (min-of-many, chained-dependency bench):
//! 16384x384x384 — 128x64 6.95 / 128x128 6.94 / 64x64 6.82 / padded 128x256
//! 3.25 TF/s; 4096^3 — 256x256 5.37 / 128x512 5.30 / 128x64 4.86; 1024^3 —
//! 128x512 1.60 vs 128x64 0.79 (the ladder chose 128x64 here); 1000x1024x1024
//! — 128x512 4.18 vs 16x64 3.85; 384x1024x1024 — all ~3.20; 8192x1024x256 —
//! all ~6.25; 1944x1280x3840 — 128x64 2.09 ~= 128x256 2.01, padded 128x512
//! 1.12.

use std::cmp::Reverse;

use fusor_tile_ir_kernels::coop_tile_entries;

use super::variants::CoopTile;
use crate::occupancy::DispatchPolicy;

/// MAC-equivalents one workgroup-memory staged element costs. See the
/// module docs for the measured bounds (0.24 < W < 7.96).
const STAGED_ELEMENT_MACS: u128 = 4;

type ScoreKey = (u128, Reverse<u64>, Reverse<u32>, Reverse<u32>);

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
    let mut best: Option<(ScoreKey, CoopTile, u128)> = None;
    for entry in coop_tile_entries() {
        let (bm, bn, bk) = (entry.tile.bm, entry.tile.bn, entry.tile.bk);
        if entry.single_buffered {
            continue;
        }
        let threads = entry.row_groups * entry.col_groups * max_subgroup_size;
        if threads == 0 || threads > policy.max_workgroup_lanes() {
            continue;
        }
        let tiles = u64::from(m.div_ceil(bm)) * u64::from(n.div_ceil(bn));
        let k_pad = u64::from(k.div_ceil(bk)) * u64::from(bk);
        let padded_macs =
            u128::from(tiles) * u128::from(bm) * u128::from(bn) * u128::from(k_pad);
        let staging = u128::from(tiles) * u128::from(k_pad) * u128::from(bm + bn);
        let key: ScoreKey = (
            padded_macs + STAGED_ELEMENT_MACS * staging,
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
