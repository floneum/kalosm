//! The SGEMM schedule domain. [`SgemmParams::legal`] holds exactly the four
//! predicates the reference asserts over its regression tree's leaves
//! (`core/src/matmul/mod.rs:600-623`); here they **generate** candidates
//! instead of validating one, and the 200-line tree is deleted.
//!
//! Owned by W4.

use fusor2_ir::device::Caps;
use fusor2_ir::dtype::Dtype;
use fusor2_ir::ir::level1::{SgemmDomain, SgemmParams};
use fusor2_ir::shape::Dim;
use smallvec::SmallVec;

use crate::domains::{DomainCtx, UNMEASURED, sgemm_order};

const BM_CHOICES: [u32; 5] = [16, 32, 64, 128, 256];
const BN_CHOICES: [u32; 5] = [16, 32, 64, 128, 256];
const BK_CHOICES: [u32; 4] = [8, 16, 32, 64];
const T_CHOICES: [u32; 4] = [1, 2, 4, 8];

/// How many tilings survive into the domain. Bounds the move frontier's
/// size; the cap keeps the [`SEED_LEAVES`] members first, so it never
/// removes a measured winner.
pub const MAX_PARAMS: usize = 64;

/// The distinct leaves of the deleted regression tree that this generator's
/// grid can reach, used **only** to order the local search's move frontier.
///
/// Four of the tree's leaves are outside the grid (`bn = 8`, `bm = 8`,
/// `bm = 48 / tm = 6`) and one (`bk = 4`) is below the smallest generated
/// depth; the tree also reaches `(true, 16, 64, 64, 2, 2)`, which needs
/// 40 KiB of staging and is illegal on every device fusor2 targets. None of
/// them is a seed, because a seed that cannot be generated would order a
/// frontier that does not contain it.
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

/// Compatibility entry point kept for the scaffold's `domains::sgemm_legal`
/// re-export. `m`, `n` and `k` price the domain; they never filter it.
pub fn legal(m: Dim, n: Dim, k: Dim, dtype: Dtype, caps: &Caps) -> SgemmDomain {
    let _ = (m, n, k);
    let cx = DomainCtx::new(caps, crate::domains::default_planner());
    sgemm_domain(dtype.byte_size() as u32, &cx)
}

/// Every `(double_buffer, BM, BN, BK, TM, TN)` satisfying the four
/// structural predicates on this device, capped at [`MAX_PARAMS`] by
/// ascending `(seed_rank, bm, bn, bk, tm, tn, double_buffer)`.
pub fn sgemm_domain(elem_bytes: u32, cx: &DomainCtx<'_>) -> SgemmDomain {
    let key = (
        cx.caps.clone(),
        elem_bytes.max(1),
        crate::domains::planner_id(cx.planner),
    );
    PARAM_MEMO.get_or_insert(&key, || generate_params(elem_bytes, cx))
}

/// `(caps, element bytes, planner identity) -> tilings`. 3,200 candidates
/// per call, none of them shape-dependent.
static PARAM_MEMO: crate::domains::DomainMemo<(Caps, u32, usize), SgemmDomain> =
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

    /// Every emitted tiling passes all four predicates at 4,000 random
    /// shapes drawn from the reference's own LCG. The domain does not vary
    /// with shape — that is the point — so this asserts the generator is
    /// shape-independent as well as legal.
    #[test]
    fn generated_params_are_all_legal() {
        let caps = apple_caps();
        let cx = DomainCtx::new(&caps, default_planner());
        let baseline = sgemm_domain(4, &cx);

        let mut lcg = 0x0fa1_1bac_c5u64;
        let mut next = |range: u32| {
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((lcg >> 33) as u32) % range + 1
        };
        for _ in 0..4000 {
            let (m, k, n) = (next(20_000), next(20_000), next(20_000));
            let domain = legal(
                Dim::Const(m.into()),
                Dim::Const(n.into()),
                Dim::Const(k.into()),
                Dtype::F32,
                &caps,
            );
            assert_eq!(domain, baseline, "m={m} n={n} k={k}: domain moved");
            for q in &domain.params {
                assert!(q.bm.is_multiple_of(q.tm) && q.bn.is_multiple_of(q.tn), "{q:?}");
                let lanes = (q.bm / q.tm) * (q.bn / q.tn);
                assert!((32..=1024).contains(&lanes), "{q:?}: {lanes} lanes");
                let depth = if q.double_buffer { 2 } else { 1 };
                let bytes = u64::from((q.bm + q.bn) * q.bk * 4 * depth);
                assert!(bytes <= 32 * 1024, "{q:?}: {bytes}B");
            }
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
