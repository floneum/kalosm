//! The SGEMM schedule domain. [`SgemmParams::legal`] holds the four structural
//! predicates a tiling must satisfy, and generates the candidate set.

use fusor2_ir::ir::level1::{SgemmDomain, SgemmParams};
use smallvec::SmallVec;

use crate::domains::{DomainCtx, UNMEASURED, sgemm_order};

const BM_CHOICES: [u32; 5] = [16, 32, 64, 128, 256];
const BN_CHOICES: [u32; 5] = [16, 32, 64, 128, 256];
const BK_CHOICES: [u32; 4] = [8, 16, 32, 64];
const T_CHOICES: [u32; 4] = [1, 2, 4, 8];

/// How many tilings survive into the domain, bounding the move frontier. The
/// cap keeps the [`SEED_LEAVES`] members first.
pub const MAX_PARAMS: usize = 64;

/// Measured-good tilings, used only to order the local search's move frontier.
/// Every entry must be reachable by [`generate_params`].
pub static SEED_LEAVES: &[SgemmParams] = &[
    p(false, 32, 32, 32, 2, 2),
    p(true, 16, 64, 32, 2, 2),
    p(true, 32, 32, 32, 2, 2),
    p(true, 64, 64, 32, 4, 4),
    p(false, 32, 32, 16, 2, 2),
    p(false, 32, 16, 32, 2, 2),
    p(false, 16, 32, 8, 2, 2),
    p(true, 128, 16, 8, 4, 4),
    p(true, 32, 32, 64, 2, 2),
    p(false, 64, 16, 32, 2, 2),
    p(false, 32, 16, 64, 2, 2),
    p(false, 16, 32, 32, 2, 2),
    p(true, 64, 16, 16, 2, 2),
    p(true, 32, 16, 32, 2, 2),
    p(false, 128, 16, 8, 4, 4),
    p(false, 64, 32, 8, 4, 4),
    p(true, 16, 32, 16, 2, 2),
    p(false, 16, 128, 8, 4, 4),
    p(false, 32, 32, 16, 4, 4),
    p(false, 32, 32, 8, 4, 4),
    p(false, 64, 32, 16, 4, 4),
    p(false, 32, 128, 8, 4, 4),
    p(false, 64, 16, 16, 2, 2),
];

const fn p(double_buffer: bool, bm: u32, bn: u32, bk: u32, tm: u32, tn: u32) -> SgemmParams {
    SgemmParams {
        double_buffer,
        bm,
        bn,
        bk,
        tm,
        tn,
    }
}

/// Every `(double_buffer, BM, BN, BK, TM, TN)` satisfying the four
/// structural predicates on this device, capped at [`MAX_PARAMS`] by
/// ascending `(seed_rank, bm, bn, bk, tm, tn, double_buffer)`.
pub fn sgemm_domain(elem_bytes: u32, cx: &DomainCtx<'_>) -> SgemmDomain {
    let key = (
        cx.caps.fingerprint(),
        elem_bytes.max(1),
        crate::domains::planner_id(cx.planner),
    );
    PARAM_MEMO.get_or_insert(&key, || generate_params(elem_bytes, cx))
}

/// `(caps fingerprint, element bytes, planner identity) -> tilings`. Nothing in
/// the key is shape-dependent, and the fingerprint is a `u64` digest, so a hit
/// is an integer compare.
static PARAM_MEMO: crate::domains::DomainMemo<(u64, u32, usize), SgemmDomain> =
    crate::domains::DomainMemo::new();

fn generate_params(elem_bytes: u32, cx: &DomainCtx<'_>) -> SgemmDomain {
    let max_storage = cx.caps.limits.max_compute_workgroup_storage_size;
    let max_lanes = cx.caps.limits.max_compute_invocations_per_workgroup;
    let elem_bytes = elem_bytes.max(1);

    let mut all: Vec<SgemmParams> = Vec::new();
    for double_buffer in [false, true] {
        for bm in BM_CHOICES {
            for bn in BN_CHOICES {
                for bk in BK_CHOICES {
                    for tm in T_CHOICES {
                        for tn in T_CHOICES {
                            let params = p(double_buffer, bm, bn, bk, tm, tn);
                            if params.legal(elem_bytes, max_storage, max_lanes) {
                                all.push(params);
                            }
                        }
                    }
                }
            }
        }
    }

    all.sort_by_key(|q| {
        let rank = if SEED_LEAVES.contains(q) { 0 } else { UNMEASURED };
        (rank, sgemm_order(q))
    });
    all.truncate(MAX_PARAMS);

    SgemmDomain {
        params: SmallVec::from_vec(all),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domains::testing::apple_caps;
    use crate::domains::{DomainCtx, default_planner};

    fn apple_domain() -> SgemmDomain {
        let caps = apple_caps();
        let cx = DomainCtx::new(&caps, default_planner());
        sgemm_domain(4, &cx)
    }

    /// Every emitted tiling passes all four predicates. The domain is a pure
    /// function of the device — no shape reaches [`sgemm_domain`] — so two
    /// independent calls must also agree.
    #[test]
    fn generated_params_are_all_legal() {
        let caps = apple_caps();
        let cx = DomainCtx::new(&caps, default_planner());
        let domain = sgemm_domain(4, &cx);
        assert_eq!(domain, sgemm_domain(4, &cx), "domain moved between calls");
        for q in &domain.params {
            assert!(q.bm.is_multiple_of(q.tm) && q.bn.is_multiple_of(q.tn), "{q:?}");
            let lanes = (q.bm / q.tm) * (q.bn / q.tn);
            assert!((32..=1024).contains(&lanes), "{q:?}: {lanes} lanes");
            let depth = if q.double_buffer { 2 } else { 1 };
            let bytes = u64::from((q.bm + q.bn) * q.bk * 4 * depth);
            assert!(bytes <= 32 * 1024, "{q:?}: {bytes}B");
        }
    }

    #[test]
    fn seeds_are_generated() {
        let domain = apple_domain();
        for seed in SEED_LEAVES {
            assert!(
                domain.params.contains(seed),
                "{seed:?} is a rank-0 seed but is not generated"
            );
        }
    }

    #[test]
    fn domain_is_capped_and_seed_ordered() {
        let domain = apple_domain();
        assert!(domain.params.len() <= MAX_PARAMS);
        // Every seed sorts ahead of every non-seed.
        let last_seed = domain
            .params
            .iter()
            .rposition(|q| SEED_LEAVES.contains(q))
            .expect("seeds survive the cap");
        assert!(
            domain.params[..=last_seed]
                .iter()
                .all(|q| SEED_LEAVES.contains(q))
        );
    }

    #[test]
    fn baseline_storage_shrinks_the_domain() {
        let caps = crate::domains::testing::baseline_caps();
        let cx = DomainCtx::new(&caps, default_planner());
        let baseline = sgemm_domain(4, &cx);
        for q in &baseline.params {
            let depth = if q.double_buffer { 2 } else { 1 };
            assert!((q.bm + q.bn) * q.bk * 4 * depth <= 16384, "{q:?}");
            let lanes = (q.bm / q.tm) * (q.bn / q.tn);
            assert!(lanes <= 256, "{q:?}: {lanes} lanes over the baseline limit");
        }
    }
}
