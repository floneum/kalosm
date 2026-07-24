//! Barrier elision: remove workgroup barriers that separate nothing.
//!
//! Elision never judges absolute correctness — lane-ownership discipline is
//! the kernel author's (a tile whose rows are only ever touched by their
//! owning thread needs no barrier, and the analysis cannot see that). It
//! preserves the *separation structure* instead: a barrier is removable only
//! when every conservatively-hazardous access pair it currently separates is
//! also separated by another barrier. Under that rule removal can never
//! introduce a race that the original ordering excluded, and the packer
//! (which re-derives sharing legality from the post-elision body) can only
//! lose opportunities the removed barrier alone provided — which the rule
//! also forbids, because expanded-range endpoints are access positions.
//!
//! Hazard pairs are enumerated two ways:
//! - *Forward*: accesses `x < y` (any two tiles or one tile, at least one
//!   write). A barrier separates the pair when it sits in `(x, y]` and every
//!   loop between the barrier and the pair's innermost common loop is
//!   guaranteed to complete (a skippable inner loop can skip the barrier).
//! - *Back edge*: accesses `x`, `y` inside a loop `L` (at least one write)
//!   race from iteration `i`'s later access to iteration `i + 1`'s earlier
//!   one. A barrier inside `L` separates the wrap when it covers the
//!   circular interval `(y, L.end) ∪ (L.start, x]`; `Break` does not
//!   invalidate it (taking the back edge means the full body executed), so
//!   this holds even in break loops.

use crate::ir::{KernelIr, Stmt};

use super::LivenessInfo;

/// Remove every removable barrier from `ir`. Returns how many were removed.
pub(crate) fn elide_barriers(ir: &mut KernelIr) -> usize {
    let has_barrier = {
        fn any_barrier(stmts: &[Stmt]) -> bool {
            stmts.iter().any(|stmt| match stmt {
                Stmt::Barrier => true,
                Stmt::Loop { body, .. } => any_barrier(body),
                Stmt::If { accept, reject, .. } => any_barrier(accept) || any_barrier(reject),
                _ => false,
            })
        }
        any_barrier(&ir.body)
    };
    if !has_barrier {
        return 0;
    }

    let info = LivenessInfo::compute(ir);
    if info.tiles.is_empty() {
        // No workgroup tiles: control barriers still order storage traffic
        // paired with storage barriers; leave them alone.
        return 0;
    }
    let pairs = hazard_pairs(&info);
    let mut alive = vec![true; info.barriers.len()];
    // Greedy forward order, re-checking against the remaining set so two
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

    let removed: Vec<&[u32]> = info
        .barriers
        .iter()
        .zip(&alive)
        .filter(|(_, alive)| !**alive)
        .map(|(barrier, _)| barrier.path.as_slice())
        .collect();
    if removed.is_empty() {
        return 0;
    }
    if super::trace_enabled() {
        eprintln!("arena-elide removing {} barrier(s)", removed.len());
    }
    // Paths sorted descending so earlier removals never shift later ones.
    let mut paths: Vec<Vec<u32>> = removed.iter().map(|path| path.to_vec()).collect();
    paths.sort_unstable_by(|a, b| b.cmp(a));
    for path in &paths {
        remove_stmt(&mut ir.body, path);
    }
    paths.len()
}

/// One conservatively-hazardous access pair at raw walk positions.
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
    // separation, so dedupe keeps the greedy pass cheap on barrier-heavy
    // kernels (elision runs on every eager kernel build).
    let mut seen = rustc_hash::FxHashSet::default();
    let mut pairs = Vec::new();
    let accesses: Vec<_> = info
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
            // back edge (y@i races x@i+1), including x == y across
            // iterations.
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
/// every path between the two accesses: every loop enclosing the barrier
/// below the pair's scope must be guaranteed to complete.
fn separates(info: &LivenessInfo, candidate: usize, pair: &HazardPair) -> bool {
    let barrier = &info.barriers[candidate];
    if !guaranteed_within(info, candidate, pair.scope) {
        return false;
    }
    if pair.back_edge {
        let scope = &info.loops[pair.scope.expect("back edges carry a scope") as usize].span;
        // Inside the wrapped loop, covering (y, end) or (start, x].
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
        // A barrier positionally between two same-scope accesses is inside
        // the scope by construction; `guaranteed_below` rejects the rest.
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

/// A profitable barrier insertion: one uniform barrier at a top-level
/// boundary lets otherwise-blocked tiles share an allocation. Policy —
/// whether the saving crosses an occupancy class worth a barrier — lives
/// with the caller; tile-ir only reports the delta.
pub struct BarrierSuggestion {
    /// Root-level statement index to insert before.
    index: usize,
    /// Workgroup bytes saved by the insertion.
    pub bytes_saved: u64,
}

pub(crate) fn barrier_suggestions(ir: &KernelIr) -> Vec<BarrierSuggestion> {
    let current = crate::lower::workgroup_bytes(ir);
    let mut suggestions = Vec::new();
    // Root boundaries are uniform by construction (every thread reaches
    // them); simulate each insertion exactly rather than re-deriving the
    // packer's position arithmetic.
    for index in 1..ir.body.len() {
        let mut candidate = ir.clone();
        candidate.body.insert(index, Stmt::Barrier);
        let saved = current.saturating_sub(crate::lower::workgroup_bytes(&candidate));
        if saved > 0 {
            suggestions.push(BarrierSuggestion {
                index,
                bytes_saved: saved,
            });
        }
    }
    suggestions.sort_by_key(|suggestion| std::cmp::Reverse(suggestion.bytes_saved));
    suggestions
}

pub(crate) fn apply_barrier_suggestion(ir: &mut KernelIr, suggestion: &BarrierSuggestion) {
    ir.body.insert(suggestion.index, Stmt::Barrier);
}
