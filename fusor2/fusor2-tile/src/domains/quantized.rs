//! The quantized-contraction schedule domain.
//!
//! `KQContract` was the one contraction family whose geometry was decided by
//! the *rule* rather than by extraction: nine of the ten quantized rules
//! minted a schedule domain holding exactly one point, and the tenth minted
//! [`ScheduleDomain::Point`] outright. A geometry written into the node at
//! mint time is a decision the next decision cannot un-write, which is the
//! failure this compiler exists to remove.
//!
//! ## What a quantized body actually reads out of a schedule point
//!
//! Two numbers, and only two: how many lanes the workgroup launches
//! (`block`), and how many output columns one lane owns (`cols`). Everything
//! else a [`SchedPoint`](fusor2_ir::ir::level1::SchedPoint) can carry —
//! cooperative fragment grids, K-tile depths, split-K counts, staging depth —
//! is invisible to `fusor2-gpu::lower::quantized`, which stages no workgroup
//! tile, runs no cooperative MMA and never splits K. **A domain that varies
//! anything else is a search over byte-identical kernels**, so this generator
//! canonicalizes: one point per distinct [`QLanes`], and no point that names
//! hardware the body does not use.
//!
//! That is also why the tile family is spelled [`SgemmDomain`] and not
//! [`CoopDomain`](fusor2_ir::ir::level1::CoopDomain). A `CoopGeom` must factor
//! into whole `COOP_DIM`-multiple fragment grids, which forces every legal
//! `(bm, bn)` to `bm * bn >= max_lanes`; the emitter then clamps `block` to
//! the device limit, so *every* coop geometry collapses onto the same launch.
//! The shipped table declared `64x64` and `128x128` tiles — 4,096 and 16,384
//! lanes — on a 1,024-lane device, so the geometry `verify_l1` admitted and
//! the workgroup the emitter launched were never the same number. Under
//! `SgemmParams` the declared lane count `bm * bn` **is** the launched block,
//! and `SgemmParams::legal` — the exact predicate `verify_l1` applies to this
//! variant — is the filter.
//!
//! ## Mirrored predicates
//!
//! `fusor2-tile` cannot depend on a backend, so the two geometry facts this
//! module needs from `fusor2-gpu::lower::quantized::QGeom` are mirrored here
//! and pinned by tests, exactly as [`supports_q8_dp4a`] already is:
//! [`MAX_COLS_PER_LANE`] (`bn.min(4).max(1)`) and [`row_lanes`] (the row
//! shapes' block widths and the fallback order in `geom_for`).
//!
//! Owned by W4.

use fusor2_ir::device::Caps;
use fusor2_ir::dtype::{Dtype, QFmt};
use fusor2_ir::ir::level1::{SgemmDomain, SgemmParams, SgemvDomain, SgemvParams};
use fusor2_ir::shape::Dim;
use smallvec::SmallVec;

use crate::domains::DomainCtx;

/// Output columns one lane of a quantized body may own. Mirrors
/// `QGeom::cols_per_lane`, which is `bn.min(4).max(1)`, so a declared `bn`
/// above this is a number the emitter throws away.
pub const MAX_COLS_PER_LANE: u32 = 4;

/// The `(block, cols)` a quantized body reads out of a resolved schedule
/// point. Two points with equal [`QLanes`] emit the same kernel, so the
/// generators below keep exactly one of each.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct QLanes {
    /// Lanes the workgroup launches.
    pub block: u32,
    /// Output columns one lane owns.
    pub cols: u32,
}

impl QLanes {
    /// Output columns one *workgroup* covers. For the tile and workgroup
    /// shapes this bounds nothing — a workgroup walks the flattened output
    /// and edge lanes mask — but for the row shapes, which give one workgroup
    /// one output row, it is the number of columns the launch can reach at
    /// all.
    pub const fn covered_columns(self) -> u64 {
        (self.block as u64) * (self.cols as u64)
    }
}

/// Whether a format's block codes can feed a `Dot4I8Packed` against
/// int8-packed activations. Mirrors W11's `BlockSpec::activation` and
/// `QGeom::Q8Wide::legal`.
pub const fn supports_q8_dp4a(fmt: QFmt) -> bool {
    matches!(fmt, QFmt::Q8_0 | QFmt::Q6K)
}

/// The Q5 family, whose 22/24-byte native blocks straddle a word boundary and
/// get the narrowed row shape.
pub const fn is_q5_family(fmt: QFmt) -> bool {
    matches!(fmt, QFmt::Q5_0 | QFmt::Q5K)
}

// ---------------------------------------------------------------------------
// The tile family
// ---------------------------------------------------------------------------

/// The `(block, cols)` an [`SgemmParams`] resolves to in a quantized body:
/// `QGeom::Tile { bm, bn }` launches `bm * bn` lanes and gives each
/// `bn.min(4)` columns.
pub fn tile_lanes(p: &SgemmParams) -> QLanes {
    QLanes {
        block: p.bm.saturating_mul(p.bn),
        cols: p.bn.clamp(1, MAX_COLS_PER_LANE),
    }
}

/// Every legal `(block, cols)` for a tiled quantized contraction of this
/// shape on this device, as canonical [`SgemmParams`].
///
/// The generated predicates, all structural:
///
/// * `block` is a whole number of subgroups — a partial subgroup is a launch
///   the device does not have — and at most
///   `min(max_compute_invocations_per_workgroup, max_compute_workgroup_size[0])`.
/// * `block` does not exceed the output rounded up to a whole subgroup. A
///   workgroup wider than the matrix it tiles is not a tiling of it; rounding
///   up is what keeps the narrowest block alive at every shape, so the domain
///   is never empty on a device that can run anything at all.
/// * `cols` divides `block` and does not exceed `n` — a lane owning more
///   columns than the matrix has writes only masked lanes.
/// * [`SgemmParams::legal`], **the same function `verify_l1` calls**, at this
///   node's accumulator width: `tm | bm`, `tn | bn`, `32 <= bm*bn <= lanes`,
///   and the staged footprint within `max_compute_workgroup_storage_size`.
///
/// `bk`, `tm`, `tn` and `double_buffer` describe an SGEMM staging loop this
/// body does not have: it stages no workgroup tile and holds `cols`
/// accumulators in registers. They are declared at the values that make
/// `SgemmParams::legal`'s lane count `bm * bn` the block the emitter actually
/// launches, and its footprint the (register-only) truth.
pub fn qtile_domain(m: Dim, n: Dim, acc: Dtype, cx: &DomainCtx<'_>) -> SgemmDomain {
    let caps = cx.caps;
    let width = caps.subgroup_width().max(1);
    let max_lanes = caps
        .limits
        .max_compute_invocations_per_workgroup
        .min(caps.limits.max_compute_workgroup_size[0]);
    let max_storage = caps.limits.max_compute_workgroup_storage_size;
    let elem_bytes = acc.byte_size().max(1) as u32;

    let block_cap = output_block_cap(m, n, width).min(max_lanes);
    let col_cap = n
        .as_const()
        .map_or(MAX_COLS_PER_LANE, |n| {
            u32::try_from(n).unwrap_or(u32::MAX)
        })
        .clamp(1, MAX_COLS_PER_LANE);

    let mut params: Vec<SgemmParams> = Vec::new();
    let mut seen: Vec<QLanes> = Vec::new();
    let mut block = width;
    while block <= block_cap {
        let mut cols = 1;
        while cols <= col_cap {
            if block.is_multiple_of(cols) {
                let p = SgemmParams {
                    double_buffer: false,
                    bm: block / cols,
                    bn: cols,
                    bk: 1,
                    tm: 1,
                    tn: 1,
                };
                let lanes = tile_lanes(&p);
                if p.legal(elem_bytes, max_storage, max_lanes) && !seen.contains(&lanes) {
                    seen.push(lanes);
                    params.push(p);
                }
            }
            cols *= 2;
        }
        block = match block.checked_add(width) {
            Some(b) => b,
            None => break,
        };
    }

    params.sort_by_key(|p| {
        let l = tile_lanes(p);
        (l.block, l.cols, crate::domains::sgemm_order(p))
    });
    SgemmDomain {
        params: SmallVec::from_vec(params),
    }
}

/// [`qtile_domain`] narrowed to the points that tile this shape **exactly**:
/// `block` divides `m * n` and `cols` divides `n`, so no lane is masked and
/// no column is computed twice.
///
/// This is what the divisibility-guarded tile rules offer. It is a strict
/// subset of the masked domain, never a different geometry, which is the
/// property that keeps `n = 8191` next to `n = 8192` instead of several arms
/// down a first-match list.
pub fn qtile_exact_domain(m: Dim, n: Dim, acc: Dtype, cx: &DomainCtx<'_>) -> SgemmDomain {
    let (Some(m), Some(n)) = (m.as_const(), n.as_const()) else {
        return SgemmDomain::default();
    };
    let elems = m.saturating_mul(n);
    let full = qtile_domain(Dim::Const(m), Dim::Const(n), acc, cx);
    let params: Vec<SgemmParams> = full
        .params
        .into_iter()
        .filter(|p| {
            let l = tile_lanes(p);
            l.block != 0
                && l.cols != 0
                && elems.is_multiple_of(u64::from(l.block))
                && n.is_multiple_of(u64::from(l.cols))
        })
        .collect();
    SgemmDomain {
        params: SmallVec::from_vec(params),
    }
}

/// `m * n` rounded up to a whole subgroup, saturating on a symbolic extent
/// (which filters nothing — a runtime extent cannot bound a compile-time
/// block).
fn output_block_cap(m: Dim, n: Dim, width: u32) -> u32 {
    let (Some(m), Some(n)) = (m.as_const(), n.as_const()) else {
        return u32::MAX;
    };
    let elems = m.saturating_mul(n).max(1);
    let rounded = elems
        .div_ceil(u64::from(width))
        .saturating_mul(u64::from(width));
    u32::try_from(rounded).unwrap_or(u32::MAX)
}

// ---------------------------------------------------------------------------
// The single-row family
// ---------------------------------------------------------------------------

/// Columns per lane a row-family point asks for. `geom_for` reads only
/// `vector >= 4`, so these are the two distinguishable classes.
const ROW_VECTORS: [u32; 2] = [1, MAX_COLS_PER_LANE];

/// The `(block, cols)` a row-family [`SgemvParams`] resolves to, or `None`
/// when the point routes to `QGeom::Workgroup` instead — which covers the
/// whole output and therefore bounds nothing.
///
/// Mirrors `geom_for`'s fallback order: a `vector >= 4` point asks for
/// `Q8Wide` and falls through `Q5SmallSingleRow` to `SingleRow`; every row
/// shape needs subgroups, without which the point degrades to `Workgroup`.
pub fn row_lanes(fmt: QFmt, vector: u32, caps: &Caps) -> Option<QLanes> {
    // Every row shape needs subgroups; without them the point degrades to
    // `QGeom::Workgroup`, which walks the flattened output and bounds nothing.
    caps.subgroups?;
    let width = caps.subgroup_width().max(1);
    let cap = caps.limits.max_compute_invocations_per_workgroup;
    if vector >= MAX_COLS_PER_LANE {
        if supports_q8_dp4a(fmt) && fmt == QFmt::Q8_0 {
            // `QGeom::Q8Wide`: four columns per lane, two subgroups wide.
            return Some(QLanes {
                block: width.saturating_mul(2).min(cap),
                cols: MAX_COLS_PER_LANE,
            });
        }
        if is_q5_family(fmt) {
            // `QGeom::Q5SmallSingleRow`: half a subgroup.
            return Some(QLanes {
                block: (width / 2).max(1).min(cap),
                cols: 1,
            });
        }
    }
    Some(QLanes {
        block: width.min(cap),
        cols: 1,
    })
}

/// Every legal row-family point for this format and output width.
///
/// **The coverage predicate is the load-bearing one.** A row shape gives one
/// workgroup one output row and one lane `cols` columns, so it reaches
/// `block * cols` columns and no more; a point offered at a wider `n` names a
/// launch that never writes the remaining columns. It is a structural
/// legality fact about the launch geometry, not a cost judgement, so it
/// filters the domain rather than pricing it.
///
/// `subgroups` is 1 and `chunk` is 1 because the body launches exactly one
/// subgroup (two for `Q8Wide`, which `row_lanes` accounts for) and walks one
/// quantization block per iteration. Declaring more would name lanes and
/// chunks the kernel never has, and would put several points on the same
/// kernel.
///
/// `vector >= 4` is generated whenever the *format* can reach the packed dot,
/// not whenever the node currently carries it: `QACT_Q8_DP4A` rewrites `act`
/// on a node that keeps this domain, so the domain must already contain the
/// point that alternative needs.
pub fn qrow_domain(fmt: QFmt, n: Dim, cx: &DomainCtx<'_>) -> SgemvDomain {
    let caps = cx.caps;
    let width = caps.subgroup_width().max(1);
    let max_lanes = caps.limits.max_compute_invocations_per_workgroup;

    let mut params: Vec<SgemvParams> = Vec::new();
    let mut seen: Vec<Option<QLanes>> = Vec::new();
    for vector in ROW_VECTORS {
        let p = SgemvParams {
            chunk: 1,
            vector,
            subgroups: 1,
        };
        if p.subgroups.saturating_mul(width) > max_lanes {
            continue;
        }
        let lanes = row_lanes(fmt, vector, caps);
        if !covers(lanes, n) || seen.contains(&lanes) {
            continue;
        }
        seen.push(lanes);
        params.push(p);
    }
    SgemvDomain {
        params: SmallVec::from_vec(params),
    }
}

/// The Q5 narrowed row family: the `vector >= 4` points, which are the only
/// ones that reach `QGeom::Q5SmallSingleRow` at all.
///
/// A `vector < 4` point asks for `SingleRow`, which is legal for a Q5 weight,
/// so it never falls through to the narrowed shape. That is why this rule
/// minted `ScheduleDomain::Point` and got `QGeom::Workgroup`: the shape it
/// names was unreachable from the schedule it declared.
pub fn q5_row_domain(fmt: QFmt, n: Dim, cx: &DomainCtx<'_>) -> SgemvDomain {
    if !is_q5_family(fmt) {
        return SgemvDomain::default();
    }
    let caps = cx.caps;
    let width = caps.subgroup_width().max(1);
    let max_lanes = caps.limits.max_compute_invocations_per_workgroup;

    let mut params: Vec<SgemvParams> = Vec::new();
    let p = SgemvParams {
        chunk: 1,
        vector: MAX_COLS_PER_LANE,
        subgroups: 1,
    };
    if p.subgroups.saturating_mul(width) <= max_lanes && covers(row_lanes(fmt, p.vector, caps), n) {
        params.push(p);
    }
    SgemvDomain {
        params: SmallVec::from_vec(params),
    }
}

/// Whether a resolved row geometry reaches every column of an `n`-wide
/// output. `None` is the `QGeom::Workgroup` degradation, which reaches all of
/// them; a symbolic `n` is undecidable and declines.
fn covers(lanes: Option<QLanes>, n: Dim) -> bool {
    let Some(lanes) = lanes else { return true };
    match n.as_const() {
        Some(n) => lanes.covered_columns() >= n,
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domains::testing::{apple_caps, baseline_caps, no_coop_caps};
    use crate::domains::{DomainCtx, default_planner};
    use fusor2_ir::ir::level1::{L1, ScheduleDomain};
    use fusor2_ir::ir::level2::Tiles;
    use fusor2_ir::shape::SymId;

    fn ctx(caps: &Caps) -> DomainCtx<'_> {
        DomainCtx::new(caps, default_planner())
    }

    fn lanes_of(d: &SgemmDomain) -> Vec<QLanes> {
        d.params.iter().map(tile_lanes).collect()
    }

    /// The predicate mirrored from `QGeom::cols_per_lane`. If the emitter's
    /// cap moves, this is the assert that catches it. The expectation is
    /// spelled as the emitter spells it — `bn.min(4).max(1)`, not `clamp` —
    /// so the comparison is textual against the line being mirrored.
    #[allow(clippy::manual_clamp)]
    #[test]
    fn cols_per_lane_matches_the_emitter() {
        for bn in [1u32, 2, 3, 4, 8, 64, 128] {
            assert_eq!(
                tile_lanes(&SgemmParams {
                    double_buffer: false,
                    bm: 1,
                    bn,
                    bk: 1,
                    tm: 1,
                    tn: 1,
                })
                .cols,
                bn.min(4).max(1),
                "bn = {bn}"
            );
        }
    }

    /// The quantized body stages no workgroup tile: its per-workgroup storage
    /// is whatever the planner reports for an empty tile set, which is zero.
    /// That is why the footprint filter here is `SgemmParams::legal`'s own
    /// formula — the one `verify_l1` applies — rather than an arena query
    /// against tiles that do not exist.
    #[test]
    fn the_quantized_body_stages_nothing() {
        let caps = apple_caps();
        let bytes = default_planner()
            .workgroup_bytes(&Tiles::default(), &caps)
            .expect("an empty tile set has a footprint");
        assert_eq!(bytes, 0);
    }

    #[test]
    fn the_tile_domain_is_non_trivial_at_a_real_shape() {
        let caps = apple_caps();
        let d = qtile_domain(Dim::Const(128), Dim::Const(8192), Dtype::F32, &ctx(&caps));
        assert!(
            d.params.len() >= 32,
            "only {} points: {:?}",
            d.params.len(),
            lanes_of(&d)
        );
        let lanes = lanes_of(&d);
        // Both ends of the block range and every column class are present.
        assert!(lanes.contains(&QLanes { block: 32, cols: 1 }));
        assert!(lanes.contains(&QLanes {
            block: 1024,
            cols: 4
        }));
        for cols in [1, 2, 4] {
            assert!(
                lanes.iter().any(|l| l.cols == cols),
                "no point with cols = {cols}"
            );
        }
    }

    /// One point per emitted kernel. A domain with two points that lower to
    /// the same `(block, cols)` is a search over byte-identical shaders.
    #[test]
    fn no_two_points_emit_the_same_kernel() {
        for caps in [apple_caps(), baseline_caps(), no_coop_caps()] {
            let d = qtile_domain(Dim::Const(256), Dim::Const(4096), Dtype::F32, &ctx(&caps));
            let mut lanes = lanes_of(&d);
            let before = lanes.len();
            lanes.sort_unstable();
            lanes.dedup();
            assert_eq!(before, lanes.len(), "{} duplicates", before - lanes.len());
        }
    }

    /// Every declared point launches the block it declares. The shipped coop
    /// table declared 4,096- and 16,384-lane workgroups on a 1,024-lane
    /// device and let the emitter clamp; this is the assert that the two
    /// numbers are now the same one.
    #[test]
    fn the_declared_block_is_a_launchable_workgroup() {
        for caps in [apple_caps(), baseline_caps()] {
            let cap = caps
                .limits
                .max_compute_invocations_per_workgroup
                .min(caps.limits.max_compute_workgroup_size[0]);
            let d = qtile_domain(Dim::Const(512), Dim::Const(4096), Dtype::F32, &ctx(&caps));
            assert!(!d.params.is_empty());
            for p in &d.params {
                let l = tile_lanes(p);
                assert!(l.block <= cap, "{p:?} declares {} lanes", l.block);
                assert_eq!(
                    l.block % caps.subgroup_width(),
                    0,
                    "{p:?} is a partial subgroup"
                );
                assert_eq!(l.block, p.bm * p.bn, "{p:?}: declared lanes are clamped");
            }
        }
    }

    /// Every generated point passes the exact predicate `verify_l1` applies.
    #[test]
    fn every_point_survives_verify_l1() {
        let caps = apple_caps();
        let op = support::qcontract(ScheduleDomain::Sgemm(qtile_domain(
            Dim::Const(128),
            Dim::Const(4096),
            Dtype::F32,
            &ctx(&caps),
        )));
        fusor2_ir::verify_l1::check_schedule_domain(
            &op,
            op.schedule().unwrap(),
            &caps,
            default_planner(),
        )
        .expect("a generated tile domain must be admissible");

        let op = support::qcontract(ScheduleDomain::Sgemv(qrow_domain(
            QFmt::Q8_0,
            Dim::Const(64),
            &ctx(&caps),
        )));
        fusor2_ir::verify_l1::check_schedule_domain(
            &op,
            op.schedule().unwrap(),
            &caps,
            default_planner(),
        )
        .expect("a generated row domain must be admissible");
    }

    /// Different device caps generate different domains — the whole reason
    /// the domain is carried on the node rather than chosen at mint time.
    #[test]
    fn caps_change_the_domain() {
        let apple = apple_caps();
        let base = baseline_caps();
        let shape = (Dim::Const(256), Dim::Const(4096));
        let a = qtile_domain(shape.0, shape.1, Dtype::F32, &ctx(&apple));
        let b = qtile_domain(shape.0, shape.1, Dtype::F32, &ctx(&base));
        assert_ne!(a, b);
        // The 1,024-lane device reaches blocks the 256-lane device cannot.
        let widest = |d: &SgemmDomain| lanes_of(d).iter().map(|l| l.block).max().unwrap_or(0);
        assert_eq!(widest(&a), 1024);
        assert_eq!(widest(&b), 256);
        // And the widest point — the one a tie-broken resolution lands on
        // from either end of the domain — is a *different* `SchedPoint`.
        let last = |d: &SgemmDomain| {
            ScheduleDomain::Sgemm(d.clone())
                .point(d.params.len() - 1)
                .expect("a non-empty domain resolves its last point")
        };
        assert_ne!(last(&a), last(&b));
    }

    /// Different shapes generate different domains: a 6-element output does
    /// not get a 1,024-lane workgroup, and a 3-column output does not get a
    /// 4-column lane.
    #[test]
    fn shapes_change_the_domain() {
        let caps = apple_caps();
        let small = qtile_domain(Dim::Const(2), Dim::Const(3), Dtype::F32, &ctx(&caps));
        let big = qtile_domain(Dim::Const(128), Dim::Const(8192), Dtype::F32, &ctx(&caps));
        assert_ne!(small, big);
        assert!(!small.params.is_empty(), "the narrowest block must survive");
        let last = |d: &SgemmDomain| {
            ScheduleDomain::Sgemm(d.clone())
                .point(d.params.len() - 1)
                .expect("a non-empty domain resolves its last point")
        };
        assert_ne!(last(&small), last(&big));
        for p in &small.params {
            let l = tile_lanes(p);
            assert_eq!(l.block, 32, "{p:?} is wider than the whole output");
            assert!(l.cols <= 3, "{p:?} owns more columns than the matrix has");
        }
    }

    /// A symbolic extent bounds nothing at compile time and must not be
    /// guessed at.
    #[test]
    fn a_symbolic_shape_keeps_the_full_device_domain() {
        let caps = apple_caps();
        let sym = qtile_domain(Dim::Sym(SymId(3)), Dim::Sym(SymId(4)), Dtype::F32, &ctx(&caps));
        let dev = qtile_domain(
            Dim::Const(1 << 20),
            Dim::Const(1 << 20),
            Dtype::F32,
            &ctx(&caps),
        );
        assert_eq!(sym, dev);
        // ...and the exact-tiling narrowing declines rather than guessing.
        assert!(
            qtile_exact_domain(Dim::Sym(SymId(3)), Dim::Const(64), Dtype::F32, &ctx(&caps))
                .params
                .is_empty()
        );
    }

    /// The exact-tiling domain is always a subset of the masked one, and at a
    /// prime `n` it narrows to single-column points without the masked domain
    /// moving at all. That is the continuity property: one column short of a
    /// divisibility boundary costs the exact alternative, never the geometry.
    #[test]
    fn exact_tiling_is_a_subset_and_never_shifts_the_masked_domain() {
        let caps = apple_caps();
        let acc = Dtype::F32;
        let round = qtile_domain(Dim::Const(128), Dim::Const(8192), acc, &ctx(&caps));
        let prime = qtile_domain(Dim::Const(128), Dim::Const(8191), acc, &ctx(&caps));
        assert_eq!(
            lanes_of(&round),
            lanes_of(&prime),
            "the masked domain must not move at 8191"
        );

        let exact_round = qtile_exact_domain(Dim::Const(128), Dim::Const(8192), acc, &ctx(&caps));
        let exact_prime = qtile_exact_domain(Dim::Const(128), Dim::Const(8191), acc, &ctx(&caps));
        for d in [&exact_round, &exact_prime] {
            for p in &d.params {
                assert!(round.params.contains(p), "{p:?} is not in the masked domain");
            }
        }
        assert!(!exact_round.params.is_empty());
        // 8191 is prime, so only a one-column lane divides it.
        for p in &exact_prime.params {
            assert_eq!(tile_lanes(p).cols, 1, "{p:?} cannot tile 8191 columns");
        }
    }

    /// Exactness means what it says: no masked lane, no repeated column.
    #[test]
    fn every_exact_point_tiles_the_shape() {
        let caps = apple_caps();
        let (m, n) = (256u64, 4096u64);
        let d = qtile_exact_domain(Dim::Const(m), Dim::Const(n), Dtype::F32, &ctx(&caps));
        assert!(!d.params.is_empty());
        for p in &d.params {
            let l = tile_lanes(p);
            assert!((m * n).is_multiple_of(u64::from(l.block)), "{p:?}");
            assert!(n.is_multiple_of(u64::from(l.cols)), "{p:?}");
        }
    }

    // -- the row family -----------------------------------------------------

    /// The row shapes reach `block * cols` columns and no more, so a domain
    /// offered at a wider `n` names a launch that never writes the rest.
    #[test]
    fn the_row_family_declines_an_output_it_cannot_cover() {
        let caps = apple_caps();
        // `SingleRow` is one subgroup, one column per lane: 32 columns.
        let wide = qrow_domain(QFmt::Q4_0, Dim::Const(4096), &ctx(&caps));
        assert!(wide.params.is_empty(), "{wide:?} cannot cover 4096 columns");
        let fits = qrow_domain(QFmt::Q4_0, Dim::Const(32), &ctx(&caps));
        assert_eq!(fits.params.len(), 1);
        // Q8_0 reaches `Q8Wide`: two subgroups, four columns per lane.
        let q8 = qrow_domain(QFmt::Q8_0, Dim::Const(256), &ctx(&caps));
        assert_eq!(q8.params.len(), 1);
        assert_eq!(q8.params[0].vector, MAX_COLS_PER_LANE);
        assert!(qrow_domain(QFmt::Q8_0, Dim::Const(257), &ctx(&caps))
            .params
            .is_empty());
    }

    /// The packed-dot point is generated from the *format*, because
    /// `QACT_Q8_DP4A` rewrites `act` on a node that keeps this domain.
    #[test]
    fn the_packed_point_is_generated_for_the_format_not_the_current_act() {
        let caps = apple_caps();
        let d = qrow_domain(QFmt::Q8_0, Dim::Const(64), &ctx(&caps));
        assert!(d.params.iter().any(|p| p.vector >= MAX_COLS_PER_LANE));
    }

    /// The Q5 narrowed shape is half a subgroup wide, and only a `vector >= 4`
    /// point reaches it.
    #[test]
    fn the_q5_row_domain_is_the_narrowed_shape() {
        let caps = apple_caps();
        for fmt in [QFmt::Q5_0, QFmt::Q5K] {
            let d = q5_row_domain(fmt, Dim::Const(16), &ctx(&caps));
            assert_eq!(d.params.len(), 1, "{fmt:?}");
            assert_eq!(d.params[0].vector, MAX_COLS_PER_LANE);
            assert_eq!(
                row_lanes(fmt, MAX_COLS_PER_LANE, &caps),
                Some(QLanes { block: 16, cols: 1 })
            );
            // 17 columns is one past what half a subgroup reaches.
            assert!(q5_row_domain(fmt, Dim::Const(17), &ctx(&caps))
                .params
                .is_empty());
        }
        assert!(q5_row_domain(QFmt::Q4_0, Dim::Const(8), &ctx(&caps))
            .params
            .is_empty());
    }

    /// Without subgroups every row point degrades to `QGeom::Workgroup`,
    /// which covers the whole output — so the coverage predicate correctly
    /// stops binding.
    #[test]
    fn no_subgroups_makes_the_row_shapes_unreachable() {
        let mut caps = apple_caps();
        caps.subgroups = None;
        assert_eq!(row_lanes(QFmt::Q8_0, 4, &caps), None);
        assert!(!qrow_domain(QFmt::Q8_0, Dim::Const(4096), &ctx(&caps))
            .params
            .is_empty());
    }

    #[test]
    fn every_point_round_trips() {
        let caps = apple_caps();
        for d in [
            ScheduleDomain::Sgemm(qtile_domain(
                Dim::Const(128),
                Dim::Const(4096),
                Dtype::F32,
                &ctx(&caps),
            )),
            ScheduleDomain::Sgemv(qrow_domain(QFmt::Q8_0, Dim::Const(64), &ctx(&caps))),
        ] {
            assert!(!d.is_empty());
            for i in 0..d.len() {
                assert!(d.point(i).is_some(), "point {i} of {} is None", d.len());
            }
            assert!(d.point(d.len()).is_none());
        }
    }

    mod support {
        use super::*;
        use fusor2_ir::ir::level1::{AccessPlan, Operand};
        use fusor2_ir::scalar::ScalarExpr;
        use fusor2_ir::shape::Layout;

        pub fn qcontract(sched: ScheduleDomain) -> L1 {
            let shape = [Dim::Const(128), Dim::Const(4096)];
            let operand = || Operand {
                src: fusor2_ir::egraph::Id(0),
                layout: Layout::contiguous(&shape),
                access: AccessPlan::Alias,
            };
            L1::KQContract {
                fmt: QFmt::Q8_0,
                layout: fusor2_ir::dtype::QLayout::Native,
                act: fusor2_ir::dtype::QAct::F32,
                m: Dim::Const(128),
                n: Dim::Const(4096),
                k: Dim::Const(1024),
                acc: Dtype::F32,
                post: ScalarExpr::arg(0, Dtype::F32),
                a: operand(),
                b: operand(),
                sched,
            }
        }
    }
}
