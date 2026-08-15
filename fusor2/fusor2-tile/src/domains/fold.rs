//! The reduction schedule domain. Workgroup width, lane-group width and
//! staging depth are *coupled* — they are one enumeration scored together
//! rather than three formulas applied in sequence.

use fusor2_ir::device::Caps;
use fusor2_ir::ir::level1::{FoldDomain, FoldStrat};
use fusor2_ir::shape::Dim;
use smallvec::SmallVec;

use crate::domains::{DomainCtx, fold_order};

/// Workgroup widths worth generating, in the reference's own band.
const BLOCK_CHOICES: [u32; 4] = [32, 64, 128, 256];

/// How many strategies survive. Bounds the move frontier; the cap keeps the
/// lowest seed rank first.
pub const MAX_STRATEGIES: usize = 32;

/// How many strip-mine factors survive. Bounds the move frontier the same way
/// [`MAX_STRATEGIES`] does; `SmallVec<[u32; 4]>` is the shape the field these
/// feed is specified to take.
pub const MAX_BLOCKS: usize = 4;

/// The narrowest inner segment a split is generated for. Below this the outer
/// level costs a whole extra traversal to fold a handful of elements, and the
/// inner extent stops covering one SIMD register on either backend.
const MIN_INNER: u64 = 8;

/// The widest split count generated, matching the `StrideSpec::multiplier`
/// band the shipped blocking search already worked in.
const MAX_SPLIT: u64 = 64;

/// The workgroup width both emitters actually allocate scratch over — **the
/// single source of it**.
///
/// The definition moved to `fusor2-ir` beside [`FoldStrat`] itself, because
/// `verify_l1` must admit a strategy against the same number this domain
/// filters on and the emitters allocate. Re-exported here so every existing
/// call site — both lowerings and this generator — still names one function.
pub use fusor2_ir::ir::level1::{emitted_block, fold_scratch_bytes};

/// Compatibility entry point kept for the scaffold's `domains::fold_legal`
/// re-export. `rows` prices the domain; it never filters it.
pub fn legal(axis_extent: Dim, rows: Dim, caps: &Caps) -> FoldDomain {
    let _ = rows;
    let cx = DomainCtx::new(caps, crate::domains::default_planner());
    fold_domain(axis_extent, &cx)
}

/// Every legal reduction strategy for an axis of extent `k` on this device,
/// for a **single-lane** accumulator.
///
/// [`fold_domain_for`] is the general form; this is the `lanes = 1`,
/// f32-accumulator case every shipped call site is, kept so the emitted
/// domain of an ordinary `Fold{Add}` is unchanged to the point.
pub fn fold_domain(k: Dim, cx: &DomainCtx<'_>) -> FoldDomain {
    fold_domain_for(k, 1, 4, cx)
}

/// Every legal reduction strategy for an axis of extent `k` carrying `lanes`
/// accumulator lanes of `acc_bytes` each.
///
/// [`FoldStrat::Subgroup`] appears only when the device reports a *fixed*
/// subgroup width — a ranged width makes a subgroup collective unusable, so
/// that is legality, not preference.
///
/// **The footprint clause.** A strategy whose cross-lane close needs
/// `lanes * emitted_block * acc_bytes` bytes of workgroup storage is dropped:
/// both emitters allocate one scratch tile of `block` elements *per
/// accumulator lane*, and `verify_l1` reads the same number from the same
/// arena function, so a strategy over the cap is **unselectable**, not merely
/// slow. Section 4.2 makes a `verify_plan` failure a hard assert rather than a
/// fallback, so admitting one mints a crash instead of a slow plan.
///
/// A wide carrier can therefore empty the domain — a `Vector(128)` f32 slot
/// wants 128 KiB against a 32 KiB limit at every lane group, because the
/// emitted block is floored at 256 lanes regardless of the lane group. That
/// is reported honestly as an empty domain rather than papered over: closing
/// it is one strategy (a row per lane, no scratch and no cross-lane close),
/// which needs a `FoldStrat` variant this crate cannot add.
pub fn fold_domain_for(k: Dim, lanes: u64, acc_bytes: u64, cx: &DomainCtx<'_>) -> FoldDomain {
    let caps = cx.caps;
    let max_block = caps.limits.max_compute_invocations_per_workgroup;
    let max_storage = u64::from(caps.limits.max_compute_workgroup_storage_size);
    let fixed_width = caps
        .subgroups
        .filter(|s| s.is_fixed())
        .map(|s| s.assumed());

    // The same call `verify_l1` admits against, so a strategy this generator
    // keeps can never be one the verifier rejects.
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
/// This is a **schedule parameter**, not a constant the rule picks. The
/// shipped `fold_split` gates on `extent.at_least(4096)` and then takes the
/// widest power-of-two divisor up to 64, so at every `Lk` the trainer and the
/// conformance attention cases use — 512, 768, 1024, 2048 — nothing splits at
/// all and there is no block loop to strip-mine into. Every factor here is
/// generated from the extent alone; which one wins is priced against the exact
/// arena plan like any other point.
///
/// **This generator has no consumer yet, and the missing piece is one field.**
/// `FoldDomain` (`fusor2-ir/src/ir/level1.rs`) carries only `strategies`; the
/// factor becomes a schedule parameter the day it also carries
/// `blocks: SmallVec<[u32; 4]>` and `SchedPoint::Fold` carries the chosen one.
/// Until then `fusor2-ir/src/rules/algebra.rs::fold_split` mints its own
/// candidates as e-nodes and bounds them by
/// `extent > max_compute_invocations_per_workgroup`, which is a device fact
/// rather than the deleted `at_least(4096)` constant but is still a rewrite
/// picking a factor. Measured consequence: `STRIP` fires on a 4096-element
/// reduction and does not fire on a contraction's summed axis at
/// `k = 512..2048`, so split-K at every extent the trainer uses is still
/// unreachable.
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

/// Move-ordering seed for one split count. The measured band is an inner
/// segment of 32..=128 elements — the reference's own `KV_BLOCK` and split-K
/// chunk sizes both land there — so those lead the frontier. **It orders
/// moves; it never gates them.**
fn block_seed_rank(extent: u64, blocks: u32) -> u8 {
    match extent / u64::from(blocks) {
        32..=128 => 0,
        16..=255 => 1,
        _ => 2,
    }
}

/// Move-ordering seed. The reference's measured behaviour: a subgroup
/// collective when one is available, then a tree over a full-width
/// workgroup, then a per-lane loop whose trip count sits inside the
/// register budget the reference calls `work_per_thread`.
pub(crate) fn seed_rank(s: FoldStrat) -> u8 {
    /// The reference's `RegPressure::ElementwiseFew` budget. Here it orders
    /// a frontier instead of gating a formula.
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
        use fusor2_ir::ir::level1::ScheduleDomain;
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

    // -----------------------------------------------------------------
    // The strip-mine factor
    // -----------------------------------------------------------------

    /// Every extent the trainer and the conformance attention cases use.
    /// The shipped `extent.at_least(4096)` gate refuses all four, so today
    /// there is no block loop at any real attention or matmul shape.
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

    /// The named shape: `Lk = 512` offers 8 blocks of 64, which is the KV
    /// block loop the hand-written flash template hardcoded — and every split
    /// it does offer lands in the measured 32..=128 band, so the frontier
    /// leads with a real block size rather than a degenerate one.
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

    // -----------------------------------------------------------------
    // The carrier footprint clause
    // -----------------------------------------------------------------

    /// A single f32 lane fits every lane group, so the general form is the
    /// shipped domain to the point.
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
    /// lane group **that closes across lanes**, because the emitted block is
    /// floored at 256 lanes.
    ///
    /// It is nonetheless schedulable, and this test pins which schedules
    /// survive rather than that none do. A one-lane group stages nothing —
    /// every invocation owns a whole output row and reduces the axis into its
    /// own accumulator, so the merge is over a group of one — and that is the
    /// row-per-lane strategy. No new `FoldStrat` variant is needed: `WgTree {
    /// lane_group:
    /// 1 }` already spells it and both emitters lower it.
    /// `fold_scratch_bytes` reports 0 for it.
    ///
    /// The assertion that matters is that **nothing needing a cross-lane close
    /// survives**: admitting one of those would mint a `verify_plan` crash.
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

    /// A three-slot scalar carrier — `(n, mean, m2)`, or flash's `(m, l)`
    /// plus one vector position — still fits, so Welford schedules
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
