//! The independent all-pairs arena recheck: every byte-overlapping tile pair
//! must be separated by a *guaranteed uniform* barrier. Failing lowering beats
//! racing.
//!
//! "Independent" is load-bearing — this recomputes [`LivenessInfo`] from the
//! body rather than reusing the packer's conclusion, and it checks **all**
//! pairs where the packer's placement loop checked incrementally.

use fusor2_ir::Result;
use fusor2_ir::error::Error;
use fusor2_ir::ir::level2::{ArenaPlan, KernelIr, LowerError, Placement};

use crate::liveness::{LivenessInfo, tile_key};

/// True when two placements share any byte. In both arena modes a placement's
/// interval means "these bytes", so this one test covers Regions and
/// ByteArena alike.
pub fn bytes_overlap(a: &Placement, b: &Placement) -> bool {
    a.byte_offset < b.byte_offset + b.byte_len && b.byte_offset < a.byte_offset + a.byte_len
}

/// Recheck a finished plan against the kernel body.
pub fn verify_arena(ir: &KernelIr, plan: &ArenaPlan) -> Result<()> {
    let live = LivenessInfo::compute(ir);
    verify_placements(&live, &plan.placements)
}

/// The all-pairs core, split out so it can be driven from a hand-built
/// [`LivenessInfo`].
pub fn verify_placements(live: &LivenessInfo, placements: &[Placement]) -> Result<()> {
    for (index, a) in placements.iter().enumerate() {
        for b in &placements[index + 1..] {
            if !bytes_overlap(a, b) {
                continue;
            }
            let (Some(a_live), Some(b_live)) = (
                live.tiles.get(&tile_key(&a.tile)),
                live.tiles.get(&tile_key(&b.tile)),
            ) else {
                return Err(hazard(format!(
                    "placement names a tile the body never touches: {:?} / {:?}",
                    a.tile.name, b.tile.name
                )));
            };
            if live.can_follow_tiles(a_live, b_live) || live.can_follow_tiles(b_live, a_live) {
                continue;
            }
            return Err(hazard(format!(
                "tiles share bytes without a guaranteed separating barrier: \
                 {:?} x{} live ({},{}) at [{},{}) overlaps {:?} x{} live ({},{}) at [{},{})",
                a_live.element,
                a_live.elements,
                a_live.range.first,
                a_live.range.last,
                a.byte_offset,
                a.byte_offset + a.byte_len,
                b_live.element,
                b_live.elements,
                b_live.range.first,
                b_live.range.last,
                b.byte_offset,
                b.byte_offset + b.byte_len,
            )));
        }
    }
    Ok(())
}

fn hazard(msg: String) -> Error {
    Error::Lower(LowerError::BarrierHazard(msg))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liveness::{BarrierInfo, LiveRange, TileLiveness};
    use fusor2_ir::ir::level2::{
        MemoryLevel, ScalarElement, Tile, TileDecl, TileLayout,
    };
    use rustc_hash::FxHashMap;
    use std::sync::Arc;

    /// 16 f32 elements = 64 bytes per tile.
    fn synthetic(ranges: &[(u32, u32)], barriers: &[(u32, bool)]) -> (LivenessInfo, Vec<Tile>) {
        let tiles: Vec<Tile> = ranges
            .iter()
            .map(|_| {
                Arc::new(TileDecl::new(
                    ScalarElement::F32.element(),
                    TileLayout::contiguous(MemoryLevel::Workgroup, &[16]),
                    "t",
                ))
            })
            .collect();
        let mut map = FxHashMap::default();
        let mut order = Vec::new();
        for (tile, &(first, last)) in tiles.iter().zip(ranges) {
            let key = tile_key(tile);
            order.push(key);
            map.insert(
                key,
                TileLiveness {
                    tile: tile.clone(),
                    range: LiveRange { first, last },
                    element: ScalarElement::F32.element(),
                    elements: 16,
                    accesses: Vec::new(),
                    scoped: None,
                    coop: false,
                },
            );
        }
        let info = LivenessInfo {
            tiles: map,
            order,
            barriers: barriers
                .iter()
                .map(|&(position, guaranteed)| BarrierInfo {
                    position,
                    path: Vec::new(),
                    enclosing_loops: Vec::new(),
                    guaranteed,
                })
                .collect(),
            loops: Vec::new(),
        };
        (info, tiles)
    }

    /// Every tile in one region: same base, same length.
    fn one_region(tiles: &[Tile]) -> Vec<Placement> {
        tiles
            .iter()
            .map(|tile| Placement {
                tile: tile.clone(),
                byte_offset: 0,
                byte_len: 64,
            })
            .collect()
    }

    #[test]
    fn rejects_overlapping_ranges_in_one_region() {
        let (info, tiles) = synthetic(&[(1, 5), (3, 8)], &[(2, true)]);
        assert!(verify_placements(&info, &one_region(&tiles)).is_err());
    }

    #[test]
    fn rejects_disjoint_ranges_without_barrier() {
        let (info, tiles) = synthetic(&[(1, 3), (5, 8)], &[]);
        assert!(verify_placements(&info, &one_region(&tiles)).is_err());
    }

    #[test]
    fn rejects_disjoint_ranges_with_only_a_non_guaranteed_barrier() {
        let (info, tiles) = synthetic(&[(1, 3), (5, 8)], &[(4, false)]);
        assert!(verify_placements(&info, &one_region(&tiles)).is_err());
    }

    #[test]
    fn accepts_a_two_barrier_chain() {
        let (info, tiles) = synthetic(&[(1, 3), (5, 8), (10, 12)], &[(4, true), (9, true)]);
        verify_placements(&info, &one_region(&tiles)).unwrap();
    }

    #[test]
    fn byte_arena_overlap_requires_a_barrier() {
        let (info, tiles) = synthetic(&[(1, 3), (4, 8)], &[]);
        let placements = vec![
            Placement {
                tile: tiles[0].clone(),
                byte_offset: 0,
                byte_len: 64,
            },
            Placement {
                tile: tiles[1].clone(),
                byte_offset: 32,
                byte_len: 64,
            },
        ];
        assert!(verify_placements(&info, &placements).is_err());
    }

    #[test]
    fn byte_arena_disjoint_intervals_need_no_barrier() {
        let (info, tiles) = synthetic(&[(1, 3), (2, 8)], &[]);
        let placements = vec![
            Placement {
                tile: tiles[0].clone(),
                byte_offset: 0,
                byte_len: 64,
            },
            Placement {
                tile: tiles[1].clone(),
                byte_offset: 64,
                byte_len: 64,
            },
        ];
        verify_placements(&info, &placements).unwrap();
    }
}
