//! The elementwise schedule domain: one register-reuse tiling per eligible
//! dim, plus untiled. Every eligible dim is a candidate.

use fusor_ir::device::Caps;
use fusor_ir::ir::launch::{AccessPlan, IndexSpace, MapDomain, MapTiling};
use fusor_ir::shape::Dim;
use smallvec::SmallVec;

use crate::domains::{DomainCtx, map_order};

/// Per-thread output counts worth generating.
const TM_CHOICES: [u32; 3] = [2, 4, 8];

/// How many tilings survive. Bounds the move frontier.
pub const MAX_TILINGS: usize = 24;

/// Compatibility entry point kept for the scaffold's `domains::map_legal`
/// re-export.
pub fn legal(space: &IndexSpace, caps: &Caps) -> MapDomain {
    let cx = DomainCtx::new(caps, crate::domains::default_planner());
    map_domain(&space.dims, &[], &cx)
}

/// Candidate tilings over this index space. `vector` is the SIMD width on
/// the CPU backend and 1 on GPU.
///
/// The innermost dim is excluded: a thread-local run along it breaks
/// inter-thread store coalescing.
pub fn map_domain(shape: &[Dim], access: &[AccessPlan], cx: &DomainCtx<'_>) -> MapDomain {
    let widths: SmallVec<[u32; 3]> = if cx.caps.simd_widths.is_empty() {
        SmallVec::from_slice(&[1])
    } else {
        cx.caps.simd_widths.clone()
    };
    // A per-lane gather has no vector load to widen into, so a vectorized
    // tiling is unbuildable over one.
    let gathers = access.iter().any(|a| matches!(a, AccessPlan::Gather));

    let mut out: Vec<MapTiling> = Vec::new();
    for vector in widths.iter().copied() {
        if vector > 1 && gathers {
            continue;
        }
        out.push(MapTiling {
            dim: None,
            tm: 1,
            vector,
        });
        for dim in 0..shape.len().saturating_sub(1) {
            for tm in TM_CHOICES {
                if !shape[dim].at_least(u64::from(tm)) {
                    continue;
                }
                out.push(MapTiling {
                    dim: Some(dim as u32),
                    tm,
                    vector,
                });
            }
        }
    }

    out.sort_by_key(|t| (seed_rank(*t), map_order(t)));
    out.truncate(MAX_TILINGS);

    MapDomain {
        tilings: SmallVec::from_vec(out),
    }
}

/// Move-ordering seed: the untiled and `tm = 4` tilings lead the frontier,
/// everything else follows.
pub(crate) fn seed_rank(t: MapTiling) -> u8 {
    match (t.dim, t.tm) {
        (None, _) => 0,
        (Some(_), 4) => 0,
        (Some(_), _) => 1,
    }
}
