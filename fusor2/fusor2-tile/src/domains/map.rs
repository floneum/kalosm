//! The elementwise schedule domain: one register-reuse tiling per eligible
//! dim, plus untiled. Every eligible dim is a candidate.

use fusor2_ir::device::Caps;
use fusor2_ir::ir::level1::{AccessPlan, IndexSpace, MapDomain, MapTiling};
use fusor2_ir::shape::Dim;
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
/// inter-thread store coalescing, which makes the resulting kernel *wrong
/// in kind*, not merely slower. That is legality. Everything the reference
/// decided by watermark is a candidate.
pub fn map_domain(shape: &[Dim], access: &[AccessPlan], cx: &DomainCtx<'_>) -> MapDomain {
    let widths: SmallVec<[u32; 3]> = if cx.caps.simd_widths.is_empty() {
        SmallVec::from_slice(&[1])
    } else {
        cx.caps.simd_widths.clone()
    };
    // A per-lane gather has no vector load to widen into, so a vectorized
    // tiling is unbuildable over one. Legality, not preference.
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

/// Move-ordering seed. The reference shipped one tiling constant,
/// `work_per_thread(RegPressure::ElementwiseFew) = 4`, and an untiled
/// fallback; those two lead the frontier and everything else follows.
pub(crate) fn seed_rank(t: MapTiling) -> u8 {
    match (t.dim, t.tm) {
        (None, _) => 0,
        (Some(_), 4) => 0,
        (Some(_), _) => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domains::testing::{apple_caps, baseline_caps};
    use crate::domains::{DomainCtx, default_planner};

    fn dims(v: &[u64]) -> Vec<Dim> {
        v.iter().map(|d| Dim::Const(*d)).collect()
    }

    #[test]
    fn innermost_dim_never_tiled() {
        let caps = baseline_caps();
        let cx = DomainCtx::new(&caps, default_planner());
        let d = map_domain(&dims(&[128, 128, 128]), &[], &cx);
        assert!(d.tilings.iter().all(|t| t.dim != Some(2)));
        assert!(d.tilings.iter().any(|t| t.dim == Some(0)));
        assert!(d.tilings.iter().any(|t| t.dim == Some(1)));
    }

    /// A `[4, 1024]` f32 shape whose invariant operand is 16 KiB sits well
    /// under the reference's 8 MiB `cache_resident` watermark, so the
    /// reference emits no tiling at all. Here the candidate survives.
    #[test]
    fn cache_resident_does_not_prune() {
        let caps = baseline_caps();
        let cx = DomainCtx::new(&caps, default_planner());
        let d = map_domain(&dims(&[4, 1024]), &[], &cx);
        assert!(d.tilings.iter().any(|t| t.dim == Some(0)));
    }

    #[test]
    fn untiled_is_always_offered() {
        let caps = baseline_caps();
        let cx = DomainCtx::new(&caps, default_planner());
        for shape in [vec![], dims(&[7]), dims(&[3, 3]), dims(&[1024, 1024])] {
            let d = map_domain(&shape, &[], &cx);
            assert!(
                d.tilings.iter().any(|t| t.dim.is_none() && t.tm == 1),
                "{shape:?} lost its untiled candidate"
            );
        }
    }

    #[test]
    fn short_dims_are_not_tiled() {
        let caps = baseline_caps();
        let cx = DomainCtx::new(&caps, default_planner());
        let d = map_domain(&dims(&[3, 1024]), &[], &cx);
        // 3 < 2? no: tm = 2 fits, tm = 4 and 8 do not.
        let tms: Vec<u32> = d.tilings.iter().filter_map(|t| t.dim.map(|_| t.tm)).collect();
        assert_eq!(tms, vec![2]);
    }

    #[test]
    fn symbolic_dims_are_not_tiled() {
        use fusor2_ir::shape::SymId;
        let caps = baseline_caps();
        let cx = DomainCtx::new(&caps, default_planner());
        let d = map_domain(&[Dim::Sym(SymId(0)), Dim::Const(64)], &[], &cx);
        assert!(d.tilings.iter().all(|t| t.dim.is_none()));
    }

    #[test]
    fn simd_widths_multiply_the_domain() {
        let mut caps = apple_caps();
        caps.simd_widths = SmallVec::from_slice(&[4, 8]);
        let cx = DomainCtx::new(&caps, default_planner());
        let d = map_domain(&dims(&[256, 256]), &[], &cx);
        assert!(d.tilings.iter().any(|t| t.vector == 4));
        assert!(d.tilings.iter().any(|t| t.vector == 8));
    }

    #[test]
    fn a_gather_operand_forbids_vector_tilings() {
        let mut caps = apple_caps();
        caps.simd_widths = SmallVec::from_slice(&[1, 4]);
        let cx = DomainCtx::new(&caps, default_planner());
        let d = map_domain(&dims(&[256, 256]), &[AccessPlan::Gather], &cx);
        assert!(d.tilings.iter().all(|t| t.vector == 1));
        assert!(!d.tilings.is_empty());
    }

    #[test]
    fn domain_is_capped_and_round_trips() {
        use fusor2_ir::ir::level1::ScheduleDomain;
        let mut caps = apple_caps();
        caps.simd_widths = SmallVec::from_slice(&[1, 4, 8]);
        let cx = DomainCtx::new(&caps, default_planner());
        let d = ScheduleDomain::Map(map_domain(&dims(&[64, 64, 64, 64, 64]), &[], &cx));
        assert!(d.len() <= MAX_TILINGS);
        for i in 0..d.len() {
            assert!(d.point(i).is_some());
        }
    }
}
