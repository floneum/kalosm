//! The SGEMV schedule domain. The 21-arm measured bucket table is deleted;
//! its eleven distinct cells survive as move-ordering seeds only.
//!
//! The reference's `gemv_parameters(m, _: usize, k)` ignores `n` outright,
//! so two contractions with wildly different output widths and identical
//! `(m, k)` get identical vectorization. Here `n` participates in cost and
//! nowhere else — the domain is a pure function of the device.
//!
//! Owned by W4.

use fusor2_ir::device::Caps;
use fusor2_ir::dtype::Dtype;
use fusor2_ir::ir::level1::{SgemvDomain, SgemvParams};
use fusor2_ir::shape::Dim;
use smallvec::SmallVec;

use crate::domains::{DomainCtx, UNMEASURED, sgemv_order};

const CHUNK_CHOICES: [u32; 6] = [1, 2, 4, 8, 16, 32];
const VECTOR_CHOICES: [u32; 3] = [1, 2, 4];
const SUBGROUP_CHOICES: [u32; 6] = [1, 2, 4, 8, 16, 32];

/// The eleven distinct cells of the deleted bucket table. Ordering, never
/// gating.
pub static SEED_CELLS: &[SgemvParams] = &[
    v(16, 4, 16),
    v(2, 4, 1),
    v(8, 4, 2),
    v(8, 4, 8),
    v(16, 4, 1),
    v(8, 4, 16),
    v(8, 4, 32),
    v(32, 2, 8),
    v(8, 2, 1),
    v(32, 2, 16),
    v(32, 2, 32),
];

const fn v(chunk: u32, vector: u32, subgroups: u32) -> SgemvParams {
    SgemvParams {
        chunk,
        vector,
        subgroups,
    }
}

/// Compatibility entry point kept for the scaffold's `domains::sgemv_legal`
/// re-export. None of `m`, `n`, `k` or `dtype` filters the domain.
pub fn legal(m: Dim, n: Dim, k: Dim, dtype: Dtype, caps: &Caps) -> SgemvDomain {
    let _ = (m, n, k, dtype);
    let cx = DomainCtx::new(caps, crate::domains::default_planner());
    sgemv_domain(&cx)
}

/// Every legal `(chunk, vector, subgroups)` on this device, ordered by
/// `(seed_rank, chunk, vector, subgroups)`.
pub fn sgemv_domain(cx: &DomainCtx<'_>) -> SgemvDomain {
    let width = cx.caps.subgroup_width();
    let max_lanes = cx.caps.limits.max_compute_invocations_per_workgroup;

    let mut all: Vec<SgemvParams> = Vec::new();
    for chunk in CHUNK_CHOICES {
        for vector in VECTOR_CHOICES {
            for subgroups in SUBGROUP_CHOICES {
                // The launched block is `subgroups * subgroup_width`; a
                // block wider than the device's invocation limit cannot be
                // created at all.
                if subgroups.saturating_mul(width) > max_lanes {
                    continue;
                }
                all.push(v(chunk, vector, subgroups));
            }
        }
    }

    all.sort_by_key(|q| {
        let rank = if SEED_CELLS.contains(q) { 0 } else { UNMEASURED };
        (rank, sgemv_order(q))
    });

    SgemvDomain {
        params: SmallVec::from_vec(all),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domains::testing::{apple_caps, baseline_caps};
    use crate::domains::{DomainCtx, default_planner};

    #[test]
    fn all_21_measured_cells_are_generated() {
        let caps = apple_caps();
        let cx = DomainCtx::new(&caps, default_planner());
        let domain = sgemv_domain(&cx);
        for cell in SEED_CELLS {
            assert!(
                domain.params.contains(cell),
                "{cell:?} is a measured cell but is not generated"
            );
        }
    }

    #[test]
    fn n_is_not_an_input() {
        let caps = apple_caps();
        let a = legal(
            Dim::Const(1),
            Dim::Const(64),
            Dim::Const(4096),
            Dtype::F32,
            &caps,
        );
        let b = legal(
            Dim::Const(1),
            Dim::Const(65536),
            Dim::Const(4096),
            Dtype::F32,
            &caps,
        );
        assert_eq!(a, b);
    }

    #[test]
    fn baseline_lanes_bound_the_subgroup_count() {
        let caps = baseline_caps();
        let cx = DomainCtx::new(&caps, default_planner());
        let domain = sgemv_domain(&cx);
        assert!(!domain.params.is_empty());
        for q in &domain.params {
            assert!(q.subgroups * 32 <= 256, "{q:?}");
        }
    }

    #[test]
    fn every_point_round_trips() {
        use fusor2_ir::ir::level1::ScheduleDomain;
        let caps = apple_caps();
        let cx = DomainCtx::new(&caps, default_planner());
        let d = ScheduleDomain::Sgemv(sgemv_domain(&cx));
        for i in 0..d.len() {
            assert!(d.point(i).is_some());
        }
    }
}
