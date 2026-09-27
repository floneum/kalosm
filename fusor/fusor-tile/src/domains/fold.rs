//! The reduction schedule domain. Workgroup width, lane-group width and
//! staging depth are coupled, so they are enumerated and scored together.

use fusor_ir::ir::launch::{FoldDomain, FoldStrat};
use fusor_ir::shape::Dim;
use smallvec::SmallVec;

use crate::domains::{DomainCtx, fold_order};

/// Workgroup widths worth generating.
const BLOCK_CHOICES: [u32; 4] = [32, 64, 128, 256];

/// How many strategies survive; the cap keeps the lowest seed rank first.
pub const MAX_STRATEGIES: usize = 32;

/// The workgroup width emitters allocate scratch over and `verify_launch`
/// admits against.
pub use fusor_ir::ir::launch::{emitted_block, fold_scratch_bytes};

/// Every legal reduction strategy for an axis of extent `k`, single-lane f32
/// accumulator. See [`fold_domain_for`].
pub fn fold_domain(k: Dim, cx: &DomainCtx<'_>) -> FoldDomain {
    fold_domain_for(k, 1, 4, cx)
}

/// Every legal reduction strategy for `lanes` accumulators of `acc_bytes`.
/// [`FoldStrat::Subgroup`] needs a fixed subgroup width. Strategies whose
/// `lanes * block * acc_bytes` scratch exceeds workgroup storage are dropped.
pub fn fold_domain_for(k: Dim, lanes: u64, acc_bytes: u64, cx: &DomainCtx<'_>) -> FoldDomain {
    let caps = cx.caps;
    let max_block = caps
        .limits
        .max_compute_invocations_per_workgroup
        .min(caps.limits.max_compute_workgroup_size[0]);
    let max_storage = u64::from(caps.limits.max_compute_workgroup_storage_size);
    let fixed_width = caps.subgroups.filter(|s| s.is_fixed()).map(|s| s.assumed());

    // The same call `verify_launch` admits against.
    let fits = |lane_group: u32| -> bool {
        fold_scratch_bytes(
            &FoldStrat::WgTree { lane_group },
            lanes,
            acc_bytes,
            lane_group,
            caps,
        ) <= max_storage
    };

    fn push(s: FoldStrat, out: &mut Vec<FoldStrat>) {
        if !out.contains(&s) {
            out.push(s);
        }
    }

    let mut out: Vec<FoldStrat> = Vec::new();

    for block in BLOCK_CHOICES {
        if block > max_block {
            continue;
        }
        let mut lane_group = 1u32;
        while lane_group <= block {
            if block.is_multiple_of(lane_group) && fits(lane_group) {
                if fixed_width == Some(lane_group) {
                    push(FoldStrat::Subgroup, &mut out);
                }
                push(FoldStrat::WgTree { lane_group }, &mut out);
                if let Some(k) = k.as_const() {
                    let iterations = k.div_ceil(u64::from(lane_group));
                    // One iteration is a plain tree, no loop prologue.
                    if iterations >= 2 {
                        let iterations = u32::try_from(iterations).unwrap_or(u32::MAX);
                        push(
                            FoldStrat::LoopThenTree {
                                iterations,
                                lane_group,
                            },
                            &mut out,
                        );
                    }
                }
            }
            lane_group *= 2;
        }
    }

    out.sort_by_key(|s| (seed_rank(*s), fold_order(s)));
    out.truncate(MAX_STRATEGIES);

    FoldDomain {
        strategies: SmallVec::from_vec(out),
    }
}

/// Move-ordering seed: subgroup collective, then a full-width tree, then a
/// per-lane loop within the register budget.
pub(crate) fn seed_rank(s: FoldStrat) -> u8 {
    const STAGE_BUDGET: u32 = 4;
    match s {
        FoldStrat::Subgroup => 0,
        FoldStrat::WgTree { lane_group } => match lane_group {
            256 => 1,
            32 | 64 | 128 => 2,
            _ => 3,
        },
        FoldStrat::LoopThenTree { iterations, .. } => {
            if iterations <= STAGE_BUDGET {
                1
            } else {
                3
            }
        }
    }
}
