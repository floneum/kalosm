//! All-pairs arena recheck: every byte-overlapping tile pair must be separated
//! by a guaranteed uniform barrier. Recomputes [`LivenessInfo`] from the body
//! independently of the packer.

use fusor2_ir::Result;
use fusor2_ir::error::Error;
use fusor2_ir::ir::kernel::{ArenaPlan, KernelIr, LowerError, Placement};

use crate::liveness::{LivenessInfo, tile_key};

/// True when two placements share any byte, in either arena mode.
pub(crate) fn bytes_overlap(a: &Placement, b: &Placement) -> bool {
    a.byte_offset < b.byte_offset + b.byte_len && b.byte_offset < a.byte_offset + a.byte_len
}

/// Recheck a finished plan against the kernel body.
pub(crate) fn verify_arena(ir: &KernelIr, plan: &ArenaPlan) -> Result<()> {
    let live = LivenessInfo::compute(ir);
    verify_placements(&live, &plan.placements)
}

/// Checks all placement pairs against a [`LivenessInfo`].
pub(crate) fn verify_placements(live: &LivenessInfo, placements: &[Placement]) -> Result<()> {
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
