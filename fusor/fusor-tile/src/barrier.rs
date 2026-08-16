//! Barrier elision and insertion.
//!
//! **Elision** never judges absolute correctness — lane-ownership discipline
//! is the kernel author's, and the analysis cannot see it. It preserves the
//! *separation structure* instead: a barrier is removable only when every
//! conservatively-hazardous access pair it currently separates is also
//! separated by a surviving barrier. Removal can then never introduce a race
//! the original ordering excluded.
//!
//! Hazard pairs are enumerated two ways:
//! - *Forward*: accesses `x < y` (any two tiles or one tile, at least one
//!   write). A barrier separates the pair when it sits in `(x, y]` and every
//!   loop between the barrier and the pair's innermost common loop is
//!   guaranteed to complete.
//! - *Back edge*: accesses `x`, `y` inside a loop `L` race from iteration
//!   `i`'s later access to iteration `i + 1`'s earlier one. A barrier inside
//!   `L` separates the wrap when it covers `(y, L.end) ∪ (L.start, x]`;
//!   `Break` does not invalidate it, because taking the back edge means the
//!   full body executed.
//!
//! **Insertion** is the other direction: one uniform barrier at a root
//! boundary can *shrink* the arena by separating two tiles' live ranges.
//! [`crate::planner::Planner::arena_plan`] computes and uses this delta.

use fusor_ir::Result;
use fusor_ir::error::Error;
use fusor_ir::ir::kernel::{BarrierSuggestion, KernelIr, Stmt};

use crate::arena;
use crate::liveness::{LivenessInfo, analyze};

/// Workgroup bytes the arena needs, taking the smaller of the two packings.
/// Mode availability is a device question the planner answers; a suggestion
/// only has to *order* candidates.
pub(crate) fn pack_bytes(live: &LivenessInfo) -> u32 {
    let regions = arena::regions(live).total_bytes;
    if arena::mixes_stride_widths(live)
        && let Some(packed) = arena::byte_arena(live)
    {
        return regions.min(packed.total_bytes);
    }
    regions
}

/// Candidate root-level statement indices at which inserting one uniform
/// barrier shrinks the arena, best `bytes_saved` first. Root boundaries are
/// uniform by construction — every thread reaches them.
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

/// Insert barriers at the given root-level indices. Indices name positions in
/// the *original* body; they are applied in ascending order.
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
