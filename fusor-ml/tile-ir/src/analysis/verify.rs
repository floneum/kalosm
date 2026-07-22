//! Barrier-hazard verification of a tile-arena placement.
//!
//! Independent recheck of every pair of tiles whose bytes overlap: the
//! packer's first-fit consults only each allocation's most recent occupant
//! (sound by barrier transitivity), so the verifier deliberately re-derives
//! legality for ALL pairs from the liveness facts alone. Any two tiles
//! sharing bytes must have disjoint expanded live ranges with a guaranteed
//! uniform barrier between them — otherwise a fast thread can touch one
//! tile while a slow thread still touches the other, through the same
//! memory. A failure is a lowering-time error, never a runtime NaN hunt.

use crate::lower::arena::{ArenaMode, Placement, TileArena};

use super::LivenessInfo;

pub(crate) fn verify_arena(info: &LivenessInfo, arena: &TileArena) -> Result<(), String> {
    // (byte interval, liveness) per placed tile; regions get disjoint
    // synthetic base offsets so interval overlap means "same bytes" in both
    // modes.
    let mut placed: Vec<(u64, u64, *const ())> = Vec::with_capacity(info.order.len());
    for &key in &info.order {
        let tile = &info.tiles[&key];
        let stride = tile
            .element
            .workgroup_array_stride()
            .map(u64::from)
            .unwrap_or_else(|| tile.element.byte_size());
        let extent = u64::from(tile.elements) * stride;
        let base = match arena.assignment.get(&key) {
            Some(Placement::Region { index }) => {
                debug_assert!(matches!(arena.mode, ArenaMode::Regions));
                // Regions cannot overlap each other; give each a base far
                // past any real allocation.
                (*index as u64) << 40
            }
            Some(Placement::Arena { byte_offset }) => u64::from(*byte_offset),
            None => continue,
        };
        placed.push((base, base + extent, key));
    }

    verify_overlaps(info, &placed)
}

fn verify_overlaps(
    info: &LivenessInfo,
    placed: &[(u64, u64, *const ())],
) -> Result<(), String> {
    for (index, &(a_start, a_end, a_key)) in placed.iter().enumerate() {
        for &(b_start, b_end, b_key) in &placed[index + 1..] {
            if a_start >= b_end || b_start >= a_end {
                continue;
            }
            let a = &info.tiles[&a_key];
            let b = &info.tiles[&b_key];
            let ordered =
                info.can_follow_tiles(a, b) || info.can_follow_tiles(b, a);
            if !ordered {
                return Err(format!(
                    "tiles share bytes without a guaranteed separating barrier: \
                     {:?} x{} live ({},{}) at [{},{}) overlaps {:?} x{} live ({},{}) at [{},{})",
                    a.element,
                    a.elements,
                    a.range.first,
                    a.range.last,
                    a_start,
                    a_end,
                    b.element,
                    b.elements,
                    b.range.first,
                    b.range.last,
                    b_start,
                    b_end,
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use rustc_hash::FxHashMap;

    use super::super::{BarrierInfo, LiveRange, LivenessInfo, TileLiveness};
    use super::*;
    use crate::ElementType;

    fn info_with(
        ranges: &[(u32, u32)],
        barriers: &[(u32, bool)],
    ) -> (LivenessInfo, Vec<*const ()>) {
        let keys: Vec<*const ()> = (1..=ranges.len()).map(|i| i as *const ()).collect();
        let mut tiles = FxHashMap::default();
        for (key, &(first, last)) in keys.iter().zip(ranges) {
            tiles.insert(
                *key,
                TileLiveness {
                    range: LiveRange { first, last },
                    element: ElementType::F32,
                    elements: 16,
                    accesses: Vec::new(),
                    scoped: None,
                    coop: false,
                },
            );
        }
        let info = LivenessInfo {
            tiles,
            order: keys.clone(),
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
        (info, keys)
    }

    fn same_region(keys: &[*const ()]) -> TileArena {
        let mut assignment = FxHashMap::default();
        for &key in keys {
            assignment.insert(key, Placement::Region { index: 0 });
        }
        TileArena {
            mode: ArenaMode::Regions,
            regions: vec![crate::lower::arena::Region {
                canonical: ElementType::F32,
                elements: 16,
            }],
            arena_bytes: 0,
            assignment,
        }
    }

    #[test]
    fn rejects_overlapping_ranges_in_one_region() {
        let (info, keys) = info_with(&[(1, 5), (3, 8)], &[(2, true)]);
        assert!(verify_arena(&info, &same_region(&keys)).is_err());
    }

    #[test]
    fn rejects_disjoint_ranges_without_barrier() {
        let (info, keys) = info_with(&[(1, 3), (5, 8)], &[]);
        assert!(verify_arena(&info, &same_region(&keys)).is_err());
    }

    #[test]
    fn rejects_disjoint_ranges_with_only_poisoned_barrier() {
        let (info, keys) = info_with(&[(1, 3), (5, 8)], &[(4, false)]);
        assert!(verify_arena(&info, &same_region(&keys)).is_err());
    }

    #[test]
    fn accepts_barrier_separated_chain() {
        let (info, keys) = info_with(&[(1, 3), (5, 8), (10, 12)], &[(4, true), (9, true)]);
        assert!(verify_arena(&info, &same_region(&keys)).is_ok());
    }

    #[test]
    fn byte_arena_overlap_requires_barrier() {
        let (info, keys) = info_with(&[(1, 3), (4, 8)], &[]);
        let mut assignment = FxHashMap::default();
        // 16 f32 elements = 64 bytes each; offsets 0 and 32 overlap.
        assignment.insert(keys[0], Placement::Arena { byte_offset: 0 });
        assignment.insert(keys[1], Placement::Arena { byte_offset: 32 });
        let arena = TileArena {
            mode: ArenaMode::ByteArena,
            regions: Vec::new(),
            arena_bytes: 96,
            assignment,
        };
        assert!(verify_arena(&info, &arena).is_err());
    }

    #[test]
    fn byte_arena_disjoint_intervals_need_no_barrier() {
        let (info, keys) = info_with(&[(1, 3), (2, 8)], &[]);
        let mut assignment = FxHashMap::default();
        assignment.insert(keys[0], Placement::Arena { byte_offset: 0 });
        assignment.insert(keys[1], Placement::Arena { byte_offset: 64 });
        let arena = TileArena {
            mode: ArenaMode::ByteArena,
            regions: Vec::new(),
            arena_bytes: 128,
            assignment,
        };
        assert!(verify_arena(&info, &arena).is_ok());
    }
}
