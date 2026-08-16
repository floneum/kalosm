//! The reduction schedule domain. Workgroup width, lane-group width and
//! staging depth are coupled, so they are enumerated and scored together.

use fusor2_ir::device::Caps;
use fusor2_ir::ir::launch::{FoldDomain, FoldStrat};
use fusor2_ir::shape::Dim;
use smallvec::SmallVec;

use crate::domains::{DomainCtx, fold_order};

/// Workgroup widths worth generating.
const BLOCK_CHOICES: [u32; 4] = [32, 64, 128, 256];

/// How many strategies survive; the cap keeps the lowest seed rank first.
pub const MAX_STRATEGIES: usize = 32;

/// How many strip-mine factors survive.
pub const MAX_BLOCKS: usize = 4;

/// The narrowest inner segment a split is generated for. Below this the outer
/// level costs a whole extra traversal, and the inner extent stops covering
/// one SIMD register on either backend.
const MIN_INNER: u64 = 8;

/// The widest split count generated.
const MAX_SPLIT: u64 = 64;

/// The workgroup width both emitters allocate scratch over; `verify_launch`
/// admits strategies against the same number.
pub use fusor2_ir::ir::launch::{emitted_block, fold_scratch_bytes};

/// Compatibility entry point for the scaffold's `domains::fold_legal`
/// re-export. `rows` prices the domain; it never filters it.
pub fn legal(axis_extent: Dim, rows: Dim, caps: &Caps) -> FoldDomain {
    let _ = rows;
    let cx = DomainCtx::new(caps, crate::domains::default_planner());
    fold_domain(axis_extent, &cx)
}

/// Every legal reduction strategy for an axis of extent `k` on this device,
/// for a single-lane f32 accumulator.
///
/// [`fold_domain_for`] is the general form.
pub fn fold_domain(k: Dim, cx: &DomainCtx<'_>) -> FoldDomain {
    fold_domain_for(k, 1, 4, cx)
}

/// Every legal reduction strategy for an axis of extent `k` carrying `lanes`
/// accumulator lanes of `acc_bytes` each.
///
/// [`FoldStrat::Subgroup`] appears only when the device reports a fixed
/// subgroup width — a ranged width makes a subgroup collective unusable.
///
/// A strategy whose cross-lane close needs
/// `lanes * emitted_block * acc_bytes` bytes of workgroup storage is dropped:
/// both emitters allocate one scratch tile of `block` elements per
/// accumulator lane, and `verify_launch` reads the same number from the same
/// arena function, so a strategy over the cap would assert, not merely run
/// slow. A wide enough carrier can empty the domain.
pub fn fold_domain_for(k: Dim, lanes: u64, acc_bytes: u64, cx: &DomainCtx<'_>) -> FoldDomain {
    let caps = cx.caps;
    let max_block = caps.limits.max_compute_invocations_per_workgroup;
    let max_storage = u64::from(caps.limits.max_compute_workgroup_storage_size);
    let fixed_width = caps
        .subgroups
        .filter(|s| s.is_fixed())
        .map(|s| s.assumed());

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
                    // One iteration is a plain tree; the loop prologue only
                    // exists when a lane strides the axis more than once.
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

/// The strip-mine factors of a reduction axis: every split count the law
/// `Fold{C,a}(x) == Fold{C.as_merge(),a}(Fold{C,a+1}(block(x,a,n)))` may be
/// instantiated at, `1` (unsplit) first.
///
/// TODO: this generator has no consumer yet — `FoldDomain` carries only
/// `strategies`; the factor becomes a schedule parameter once it also carries
/// `blocks: SmallVec<[u32; 4]>` and `SchedPoint::Fold` carries the chosen one.
///
/// Legality, not preference:
/// * the extent must be `Dim::Const` — `StrideSpec::multiplier` is a `u32`, so
///   the inner extent has to be spellable, and a `Dim::Sym` axis declines
///   rather than guessing;
/// * the count must divide the extent exactly (a ragged tail is the elide
///   clause's business, not the split's);
/// * the inner segment stays at or above [`MIN_INNER`].
pub fn fold_blocks(k: Dim, cx: &DomainCtx<'_>) -> SmallVec<[u32; 4]> {
    let _ = cx;
    let mut out: SmallVec<[u32; 4]> = SmallVec::new();
    out.push(1);
    let Some(extent) = k.as_const() else {
        return out;
    };

    let mut candidates: Vec<u32> = Vec::new();
    let mut blocks = 2u64;
    while blocks <= MAX_SPLIT {
        if extent.is_multiple_of(blocks) && extent / blocks >= MIN_INNER {
            candidates.push(blocks as u32);
        }
        blocks *= 2;
    }
    candidates.sort_by_key(|b| (block_seed_rank(extent, *b), *b));
    for b in candidates.into_iter().take(MAX_BLOCKS - 1) {
        out.push(b);
    }
    out
}

/// Move-ordering seed for one split count. Inner segments of 32..=128
/// elements lead the frontier. It orders moves; it never gates them.
fn block_seed_rank(extent: u64, blocks: u32) -> u8 {
    match extent / u64::from(blocks) {
        32..=128 => 0,
        16..=255 => 1,
        _ => 2,
    }
}

/// Move-ordering seed: a subgroup collective when one is available, then a
/// tree over a full-width workgroup, then a per-lane loop whose trip count
/// sits inside the register budget.
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
            if iterations <= STAGE_BUDGET { 1 } else { 3 }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domains::testing::{apple_caps, baseline_caps};
    use crate::domains::{DomainCtx, default_planner};

    #[test]
    fn k64_emits_subgroup_and_tree() {
        let caps = baseline_caps();
        let cx = DomainCtx::new(&caps, default_planner());
        let d = fold_domain(Dim::Const(64), &cx);
        assert!(d.strategies.contains(&FoldStrat::Subgroup));
        assert!(d.strategies.contains(&FoldStrat::WgTree { lane_group: 64 }));
        assert!(d.strategies.contains(&FoldStrat::LoopThenTree {
            iterations: 2,
            lane_group: 32,
        }));
        assert!(d.strategies.len() >= 3);
    }

    #[test]
    fn no_subgroup_without_fixed_width() {
        let mut caps = apple_caps();
        caps.subgroups = None;
        let cx = DomainCtx::new(&caps, default_planner());
        let d = fold_domain(Dim::Const(1024), &cx);
        assert!(!d.strategies.contains(&FoldStrat::Subgroup));
        assert!(!d.strategies.is_empty());
    }

    #[test]
    fn ranged_subgroups_are_not_fixed() {
        use fusor2_ir::device::SubgroupWidths;
        let mut caps = apple_caps();
        caps.subgroups = Some(SubgroupWidths { min: 16, max: 32 });
        let cx = DomainCtx::new(&caps, default_planner());
        let d = fold_domain(Dim::Const(1024), &cx);
        assert!(!d.strategies.contains(&FoldStrat::Subgroup));
    }

    #[test]
    fn symbolic_k_emits_no_loop_strategies() {
        use fusor2_ir::shape::SymId;
        let caps = apple_caps();
        let cx = DomainCtx::new(&caps, default_planner());
        let d = fold_domain(Dim::Sym(SymId(3)), &cx);
        assert!(
            d.strategies
                .iter()
                .all(|s| !matches!(s, FoldStrat::LoopThenTree { .. }))
        );
        assert!(d.strategies.contains(&FoldStrat::Subgroup));
    }

    #[test]
    fn domain_is_capped_and_round_trips() {
        use fusor2_ir::ir::launch::ScheduleDomain;
        let caps = apple_caps();
        let cx = DomainCtx::new(&caps, default_planner());
        let d = ScheduleDomain::Fold(fold_domain(Dim::Const(8192), &cx));
        assert!(d.len() <= MAX_STRATEGIES);
        for i in 0..d.len() {
            assert!(d.point(i).is_some());
        }
    }

    #[test]
    fn subgroup_leads_the_frontier() {
        let caps = apple_caps();
        let cx = DomainCtx::new(&caps, default_planner());
        let d = fold_domain(Dim::Const(256), &cx);
        assert_eq!(d.strategies.first(), Some(&FoldStrat::Subgroup));
    }

    /// Every extent the trainer and the conformance attention cases use
    /// generates at least one split.
    #[test]
    fn real_extents_generate_a_split() {
        let caps = apple_caps();
        let cx = DomainCtx::new(&caps, default_planner());
        for k in [512u64, 768, 1024, 2048] {
            let blocks = fold_blocks(Dim::Const(k), &cx);
            assert_eq!(blocks.first(), Some(&1), "k={k} lost the unsplit point");
            assert!(
                blocks.len() >= 2,
                "k={k} generated no split: {blocks:?} — this is the \
                 at_least(4096) gate, one level down"
            );
            for b in blocks.iter().copied() {
                assert!(k.is_multiple_of(u64::from(b)), "k={k} block={b} is ragged");
                assert!(k / u64::from(b) >= MIN_INNER, "k={k} block={b} too fine");
            }
        }
    }

    /// `Lk = 512` offers 8 blocks of 64, and every split it offers lands in
    /// the measured 32..=128 band.
    #[test]
    fn lk_512_offers_eight_blocks_of_sixty_four() {
        let caps = apple_caps();
        let cx = DomainCtx::new(&caps, default_planner());
        let blocks = fold_blocks(Dim::Const(512), &cx);
        assert!(blocks.contains(&8), "{blocks:?}");
        for b in blocks.iter().skip(1).copied() {
            let inner = 512 / u64::from(b);
            assert!((32..=128).contains(&inner), "block {b} -> inner {inner}");
        }
    }

    #[test]
    fn a_symbolic_extent_declines_to_the_unsplit_point() {
        use fusor2_ir::shape::SymId;
        let caps = apple_caps();
        let cx = DomainCtx::new(&caps, default_planner());
        let blocks = fold_blocks(Dim::Sym(SymId(2)), &cx);
        assert_eq!(&blocks[..], &[1], "a Sym extent cannot be spelled in a stride");
    }

    /// A prime extent has no exact factorization, so only the unsplit point
    /// survives. The ragged tail is the elide clause's business.
    #[test]
    fn an_indivisible_extent_keeps_only_the_unsplit_point() {
        let caps = apple_caps();
        let cx = DomainCtx::new(&caps, default_planner());
        assert_eq!(&fold_blocks(Dim::Const(521), &cx)[..], &[1]);
        // 64 = 8 x 8 is exactly at the floor; 32 = 4 x 8 too.
        assert!(fold_blocks(Dim::Const(64), &cx).contains(&8));
        assert_eq!(&fold_blocks(Dim::Const(16), &cx)[..], &[1, 2]);
        assert_eq!(&fold_blocks(Dim::Const(8), &cx)[..], &[1]);
    }

    #[test]
    fn the_block_set_is_capped() {
        let caps = apple_caps();
        let cx = DomainCtx::new(&caps, default_planner());
        assert!(fold_blocks(Dim::Const(4096), &cx).len() <= MAX_BLOCKS);
    }

    /// A single f32 lane fits every lane group, so the general form matches
    /// the single-lane domain.
    #[test]
    fn a_scalar_carrier_is_unfiltered() {
        let caps = apple_caps();
        let cx = DomainCtx::new(&caps, default_planner());
        assert_eq!(
            fold_domain_for(Dim::Const(1024), 1, 4, &cx),
            fold_domain(Dim::Const(1024), &cx)
        );
    }

    /// Flash's output accumulator: a `Vector(128)` f32 slot wants
    /// `128 * 256 * 4 = 128 KiB` of scratch against Apple's 32 KiB at every
    /// lane group that closes across lanes. Only the row-per-lane schedule
    /// (`WgTree { lane_group: 1 }`, zero scratch) survives; anything needing
    /// a cross-lane close would fail `verify_plan`.
    #[test]
    fn a_carrier_too_wide_for_workgroup_storage_keeps_only_the_row_per_lane_schedule() {
        let caps = apple_caps();
        let cx = DomainCtx::new(&caps, default_planner());
        let d = fold_domain_for(Dim::Const(1024), 128, 4, &cx);
        assert!(!d.strategies.is_empty(), "the row-per-lane schedule survives");
        for s in &d.strategies {
            assert_eq!(
                s.lane_group(caps.subgroup_width()),
                1,
                "a cross-lane close over a 128-lane carrier is unschedulable here: {s:?}"
            );
        }
    }

    /// A three-slot scalar carrier still fits, so Welford schedules
    /// everywhere an ordinary sum does.
    #[test]
    fn a_narrow_multi_slot_carrier_still_schedules() {
        let caps = apple_caps();
        let cx = DomainCtx::new(&caps, default_planner());
        let d = fold_domain_for(Dim::Const(1024), 3, 4, &cx);
        assert!(!d.strategies.is_empty());
        assert!(d.strategies.contains(&FoldStrat::Subgroup));
    }

    /// The filter is on the *emitted* footprint, so it moves with the cap.
    #[test]
    fn a_larger_storage_limit_admits_a_wider_carrier() {
        let mut caps = apple_caps();
        caps.limits.max_compute_workgroup_storage_size = 262_144;
        let cx = DomainCtx::new(&caps, default_planner());
        let d = fold_domain_for(Dim::Const(1024), 128, 4, &cx);
        assert!(!d.strategies.is_empty());
        for s in d.strategies.iter().copied() {
            let lg = match s {
                FoldStrat::Subgroup => 32,
                FoldStrat::WgTree { lane_group }
                | FoldStrat::LoopThenTree { lane_group, .. } => lane_group,
            };
            assert!(
                128 * u64::from(emitted_block(lg, &caps)) * 4
                    <= u64::from(caps.limits.max_compute_workgroup_storage_size),
                "{s:?} is over the cap"
            );
        }
    }
}
