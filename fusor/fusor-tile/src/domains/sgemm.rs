//! The SGEMM schedule domain. [`SgemmParams::legal`] holds exactly the four
//! predicates for candidate generation.

use fusor_ir::device::Caps;
use fusor_ir::dtype::Dtype;
use fusor_ir::ir::launch::{SgemmDomain, SgemmParams};
use fusor_ir::shape::Dim;
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

/// Measured-winner tilings, used **only** to order the local search's move
/// frontier. Every seed must be reachable by the generator's grid: a seed
/// that cannot be generated would order a frontier that does not contain it.
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

/// Entry point for the scaffold's `domains::sgemm_legal` re-export.
/// `m`, `n` and `k` price the domain; they never filter it.
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
        let rank = if SEED_LEAVES.contains(q) {
            0
        } else {
            UNMEASURED
        };
        (rank, sgemm_order(q))
    });
    all.truncate(MAX_PARAMS);

    SgemmDomain {
        params: SmallVec::from_vec(all),
    }
}
