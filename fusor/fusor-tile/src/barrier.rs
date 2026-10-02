//! Barrier elision and insertion. Elision preserves separation structure: a
//! barrier is removable only when every hazard pair it separates (forward, or
//! across a loop back edge) is also separated by a surviving barrier.
//! Insertion places one uniform root barrier where it shrinks the arena.

use fusor_ir::Result;
use fusor_ir::error::Error;
use fusor_ir::ir::kernel::{BarrierSuggestion, KernelIr, Stmt};

use crate::arena;
use crate::liveness::{LivenessInfo, analyze};

/// Workgroup bytes the arena needs, the smaller of the two packings; only
/// used to order suggestions.
pub(crate) fn pack_bytes(live: &LivenessInfo) -> u32 {
    let regions = arena::regions(live).total_bytes;
    if arena::mixes_stride_widths(live)
        && let Some(packed) = arena::byte_arena(live)
    {
        return regions.min(packed.total_bytes);
    }
    regions
}

/// Root-level indices where one inserted barrier shrinks the arena, best
/// first. Root boundaries are uniform by construction.
pub(crate) fn barrier_suggestions(ir: &KernelIr) -> Vec<BarrierSuggestion> {
    let live = analyze(ir);
    suggestions(ir, &live)
}

/// [`barrier_suggestions`] against a liveness result the caller already has.
pub(crate) fn suggestions(ir: &KernelIr, live: &LivenessInfo) -> Vec<BarrierSuggestion> {
    let current = pack_bytes(live);
    let mut out = Vec::new();
    for index in 1..ir.body.len() {
        let mut candidate = ir.clone();
        candidate.body.insert(index, Stmt::Barrier);
        let candidate_live = analyze(&candidate);
        let saved = current.saturating_sub(pack_bytes(&candidate_live));
        if saved > 0 {
            out.push(BarrierSuggestion {
                index: index as u32,
                bytes_saved: saved,
            });
        }
    }
    out.sort_by_key(|suggestion| (std::cmp::Reverse(suggestion.bytes_saved), suggestion.index));
    out
}

/// Insert barriers at root-level indices of the original body, ascending.
pub(crate) fn insert(ir: &KernelIr, at: &[u32]) -> Result<KernelIr> {
    let mut indices: Vec<u32> = at.to_vec();
    indices.sort_unstable();
    let mut out = ir.clone();
    for (shift, index) in indices.iter().enumerate() {
        let position = *index as usize + shift;
        if position > out.body.len() {
            return Err(Error::Legality(format!(
                "barrier insertion index {index} is past the end of kernel {}",
                ir.name
            )));
        }
        out.body.insert(position, Stmt::Barrier);
    }
    Ok(out)
}
