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

use fusor2_ir::Result;
use fusor2_ir::error::Error;
use fusor2_ir::ir::kernel::{BarrierSuggestion, KernelIr, Stmt};
use rustc_hash::FxHashSet;

use crate::arena;
use crate::liveness::{LivenessInfo, TileAccess, analyze};

/// Remove every removable barrier from `ir`. Returns how many were removed.
pub fn elide_barriers(ir: &mut KernelIr) -> usize {
    if !any_barrier(&ir.body) {
        return 0;
    }
    let info = analyze(ir);
    if info.is_empty() {
        // No workgroup tiles: control barriers still order storage traffic
        // paired with storage barriers; leave them alone.
        return 0;
    }
    let pairs = hazard_pairs(&info);
    let mut alive = vec![true; info.barriers.len()];
    // Greedy in index order, re-checking against the remaining set so two
    // barriers that mutually back each other up cannot both go.
    for candidate in 0..info.barriers.len() {
        alive[candidate] = false;
        let preserved = pairs.iter().all(|pair| {
            !separates(&info, candidate, pair)
                || (0..info.barriers.len())
                    .any(|other| alive[other] && separates(&info, other, pair))
        });
        if !preserved {
            alive[candidate] = true;
        }
    }

    let mut paths: Vec<Vec<u32>> = info
        .barriers
        .iter()
        .zip(&alive)
        .filter(|(_, alive)| !**alive)
        .map(|(barrier, _)| barrier.path.clone())
        .collect();
    if paths.is_empty() {
        return 0;
    }
    // Descending so earlier removals never shift later ones.
    paths.sort_unstable_by(|a, b| b.cmp(a));
    for path in &paths {
        remove_stmt(&mut ir.body, path);
    }
    paths.len()
}

fn any_barrier(stmts: &[Stmt]) -> bool {
    stmts.iter().any(|stmt| match stmt {
        Stmt::Barrier => true,
        Stmt::Loop { body, .. } => any_barrier(body),
        Stmt::If { accept, reject, .. } => any_barrier(accept) || any_barrier(reject),
        _ => false,
    })
}

/// One conservatively-hazardous access pair, at raw walk positions.
struct HazardPair {
    /// Earlier access position (forward), or the wrap target (back edge).
    x: u32,
    /// Later access position (forward), or the wrap source (back edge).
    y: u32,
    /// Innermost common loop for forward pairs; the wrapped loop for back
    /// edges.
    scope: Option<u32>,
    back_edge: bool,
}

fn hazard_pairs(info: &LivenessInfo) -> Vec<HazardPair> {
    // Pairs collapse to position signatures: tile identity is irrelevant to
    // separation, so deduping keeps the greedy pass cheap on barrier-heavy
    // kernels.
    let mut seen: FxHashSet<(u32, u32, bool)> = FxHashSet::default();
    let mut pairs = Vec::new();
    let accesses: Vec<TileAccess> = info
        .order
        .iter()
        .flat_map(|key| info.tiles[key].accesses.iter().copied())
        .collect();
    for (index, a) in accesses.iter().enumerate() {
        for b in &accesses[index + 1..] {
            if !a.kind.writes() && !b.kind.writes() {
                continue;
            }
            let (x, y) = if a.position <= b.position {
                (a.position, b.position)
            } else {
                (b.position, a.position)
            };
            let scope = info.innermost_common_loop(x, y);
            if x != y && seen.insert((x, y, false)) {
                pairs.push(HazardPair {
                    x,
                    y,
                    scope,
                    back_edge: false,
                });
            }
            // The wrap: both accesses inside a common loop race across the
            // back edge (y@i races x@i+1), including x == y across iterations.
            if scope.is_some() && seen.insert((x, y, true)) {
                pairs.push(HazardPair {
                    x,
                    y,
                    scope,
                    back_edge: true,
                });
            }
        }
    }
    pairs
}

/// Whether barrier `candidate` orders the pair. The barrier must execute on
/// every path between the two accesses.
fn separates(info: &LivenessInfo, candidate: usize, pair: &HazardPair) -> bool {
    let barrier = &info.barriers[candidate];
    if !guaranteed_within(info, candidate, pair.scope) {
        return false;
    }
    if pair.back_edge {
        let scope = info.loops[pair.scope.expect("back edges carry a scope") as usize].span;
        let position = barrier.position;
        let inside = scope.first < position && position < scope.last;
        inside && (position > pair.y || position <= pair.x)
    } else {
        barrier.position > pair.x && barrier.position <= pair.y
    }
}

/// Every loop enclosing the barrier strictly below `scope` completes every
/// pass, so the barrier executes whenever control flows from one end of the
/// scope to the other.
fn guaranteed_within(info: &LivenessInfo, candidate: usize, scope: Option<u32>) -> bool {
    let barrier = &info.barriers[candidate];
    match scope {
        None => barrier.guaranteed,
        // A barrier positionally between two same-scope accesses is inside the
        // scope by construction; `guaranteed_below` rejects the rest.
        Some(scope) => info.guaranteed_below(barrier, scope) || barrier.guaranteed,
    }
}

fn remove_stmt(body: &mut Vec<Stmt>, path: &[u32]) {
    let index = path[0] as usize;
    match path.len() {
        1 => {
            debug_assert!(matches!(body[index], Stmt::Barrier));
            body.remove(index);
        }
        _ => match &mut body[index] {
            Stmt::Loop { body: inner, .. } => remove_stmt(inner, &path[1..]),
            _ => unreachable!("barrier paths descend through loop bodies only"),
        },
    }
}

/// Workgroup bytes the arena needs, taking the smaller of the two packings.
/// Mode availability is a device question the planner answers; a suggestion
/// only has to *order* candidates.
pub fn pack_bytes(live: &LivenessInfo) -> u32 {
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
pub fn barrier_suggestions(ir: &KernelIr) -> Vec<BarrierSuggestion> {
    let live = analyze(ir);
    suggestions(ir, &live)
}

/// [`barrier_suggestions`] against a liveness result the caller already has.
pub fn suggestions(ir: &KernelIr, live: &LivenessInfo) -> Vec<BarrierSuggestion> {
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
pub fn insert(ir: &KernelIr, at: &[u32]) -> Result<KernelIr> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::TileBuilder;
    use crate::build::fixtures;

    #[test]
    fn two_redundant_barriers_lose_exactly_one() {
        let mut b = TileBuilder::new();
        let (a, c) = fixtures::two_f32_tiles(&mut b);
        let zero = b.lit_f32(0.0);
        let index = b.lit_u32(0);
        let write_a = b.store_tile(a, index.clone(), zero.clone());
        let write_c = b.store_tile(c, index, zero);
        b.set_body(vec![write_a, Stmt::Barrier, Stmt::Barrier, write_c]);
        let mut ir = b.finish([1, 1, 1], 64, "redundant");
        assert_eq!(elide_barriers(&mut ir), 1);
        assert_eq!(ir.body.len(), 3);
        // The surviving barrier still separates the pair.
        let live = analyze(&ir);
        let plan = arena::regions(&live);
        crate::verify_arena::verify_placements(&live, &plan.placements).unwrap();
    }

    #[test]
    fn two_mutually_backing_barriers_both_survive() {
        let mut b = TileBuilder::new();
        let (a, c) = fixtures::two_f32_tiles(&mut b);
        let zero = b.lit_f32(0.0);
        let index = b.lit_u32(0);
        let write_a = b.store_tile(a.clone(), index.clone(), zero.clone());
        let write_c = b.store_tile(c, index.clone(), zero.clone());
        let write_a2 = b.store_tile(a, index, zero);
        b.set_body(vec![
            write_a,
            Stmt::Barrier,
            write_c,
            Stmt::Barrier,
            write_a2,
        ]);
        let mut ir = b.finish([1, 1, 1], 64, "mutual");
        assert_eq!(elide_barriers(&mut ir), 0);
        let live = analyze(&ir);
        let plan = arena::regions(&live);
        crate::verify_arena::verify_placements(&live, &plan.placements).unwrap();
    }

    #[test]
    fn barrier_suggestion_bytes_saved_is_exact() {
        let mut b = TileBuilder::new();
        let ir = fixtures::pair_kernel(&mut b, Vec::new());
        let out = barrier_suggestions(&ir);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].index, 1);
        assert_eq!(out[0].bytes_saved, fixtures::UNSHARED - fixtures::SHARED);
    }

    #[test]
    fn insertion_indices_name_the_original_body() {
        let mut b = TileBuilder::new();
        let ir = fixtures::pair_kernel(&mut b, Vec::new());
        let widened = insert(&ir, &[1, 2]).unwrap();
        // Index 1 lands before the second write, index 2 after it.
        assert!(matches!(widened.body[1], Stmt::Barrier));
        assert!(matches!(widened.body[3], Stmt::Barrier));
        assert_eq!(widened.body.len(), ir.body.len() + 2);
    }
}
