//! The cooperative-matrix schedule domain: `geoms x splits x staging`, carried
//! whole on the node and resolved by extraction.
//!
//! The geometry list is generated: every `(bm, bn, bk, subgroups, n_passes)`
//! whose closed-form subgroup split exists, whose lanes fit, and whose exact
//! arena footprint fits is a candidate. Padded MACs enter the issue term, so
//! there is no padding routing guard.

use fusor2_ir::dtype::Dtype;
use fusor2_ir::ir::level1::{CoopDomain, CoopGeom};
use fusor2_ir::ir::level2::{
    ElementType, MemoryLevel, ScalarElement, TileDecl, TileLayout, Tiles,
};
use fusor2_ir::shape::Dim;
use smallvec::SmallVec;
use std::sync::Arc;

use crate::domains::{DomainCtx, MAX_SPLITS};

/// Block-M sides worth generating.
const BM_CHOICES: [u32; 6] = [16, 32, 64, 128, 256, 512];
/// Block-N sides worth generating.
const BN_CHOICES: [u32; 6] = [16, 32, 64, 128, 256, 512];
/// K-tile depths worth generating.
const BK_CHOICES: [u32; 3] = [8, 16, 32];
/// Subgroups per workgroup worth generating.
const SUBGROUP_CHOICES: [u32; 6] = [1, 2, 4, 8, 16, 32];
/// A cooperative fragment side. One `n_pass` covers at least this many
/// columns, which bounds `n_passes` at `bn / 16`.
const MIN_PASS_COLS: u32 = 16;

/// Every legal `(geom, splits, staging)` for this contraction on this device.
/// Empty when the device reports no usable cooperative configuration, which
/// makes the `Coop` alternative unselectable rather than an error.
pub fn coop_domain(
    m: Dim,
    n: Dim,
    k: Dim,
    batch: Dim,
    operand: Dtype,
    acc: Dtype,
    cx: &DomainCtx<'_>,
) -> CoopDomain {
    // `m`, `n` and `batch` price the domain; they do not filter it. Edge tiles
    // fill zero past the logical extents, so no shape is illegal for any
    // geometry.
    let _ = (m, n, batch);

    if cx.caps.coop_for(operand, acc).is_none() {
        return CoopDomain::default();
    }

    let geoms = candidate_geoms_for(operand, cx);
    if geoms.is_empty() {
        return CoopDomain::default();
    }

    // Splits are a domain-level list while `bk` is per-geometry, so the
    // candidate set is the union over the surviving depths. A pair whose spans
    // do not partition K exactly still runs: the split kernel bounds its K
    // span.
    let mut splits: SmallVec<[u32; 8]> = SmallVec::new();
    let mut depths: SmallVec<[u32; 3]> = SmallVec::new();
    for g in &geoms {
        if !depths.contains(&g.bk) {
            depths.push(g.bk);
        }
    }
    depths.sort_unstable();
    for bk in depths {
        for d in split_candidates(k, bk) {
            if !splits.contains(&d) {
                splits.push(d);
            }
        }
    }
    splits.sort_unstable();

    // Two staged pairs overlap the next K tile's fill with the current
    // tile's MMAs; one pair halves the footprint so a core holds more
    // workgroups. A split grid already exists to raise occupancy, so the
    // partials body is one pair outright.
    let staging: SmallVec<[u8; 2]> = if splits.as_slice() == [1] {
        SmallVec::from_slice(&[1, 2])
    } else {
        SmallVec::from_slice(&[1])
    };

    CoopDomain {
        geoms,
        splits,
        staging,
    }
}

/// `(caps fingerprint, staged element, planner identity) -> geometries`. The
/// grid is ~7,000 candidates each costing an exact arena query, and depends
/// only on the device. The fingerprint is the digest the arena-plan memo keys
/// on, so a hit is a `u64` compare.
static GEOM_MEMO: crate::domains::DomainMemo<
    (u64, ScalarElement, usize),
    SmallVec<[CoopGeom; 16]>,
> = crate::domains::DomainMemo::new();

/// # No cost term ranks these geometries
///
/// Every one of the ~5,700 points this domain carries prices **identically**
/// under the cost model. `math_ps` counts MACs, which a tiling does not
/// change; `dram_ps` reads a reread factor derived from the index space, which
/// a tiling does not change either. So the seed takes the argmin by domain
/// index — whatever this function emits first — and no `RESCHEDULE` can tell
/// the difference to move off it. A 2048-cube matmul runs at `bm=16, bn=16`.
///
/// The tile is worth real time. Measured by pinning one geometry at a time
/// through `FUSOR2_PIN_COOP` on an Apple M2 Max, f32, against this workspace's
/// `vs_fusor1` example (median ms):
///
/// | geom | `matmul` 2048-cube | attention `[1,8,1024,64]` |
/// |---|---|---|
/// | `16x16x8`   | **20.5** | 12.96 |
/// | `32x32x8`   | 21.2 | 17.13 |
/// | `64x64x8`   | 59.9 | 39.56 |
/// | `64x64x16`  | 102.7 | **7.91** |
/// | `128x64x8`  | 42.8 | **7.90** |
/// | `128x128x8` | 38.6 | 9.51 |
///
/// **The optimum is shape-dependent and the two orderings are inverted**: the
/// square matmul wants the narrowest tile in the set, attention one of the
/// widest. Analytical terms over these numbers either tie or regress matmul by
/// 2-3x: a blocked-GEMM reread count charged to `dram_ps` (matmul 20 -> 41 ms),
/// a per-tile roofline in `node_math` (20 -> 35 ms, and it moves
/// `argmin_member`'s *family* choice because the lower bound is built on
/// `node_math`), the same term in the exact launch cost (no change, the tile
/// never moves), and a device-wide `max(compute, load)` (20 -> 41 ms and Coop
/// abandoned entirely).
///
/// # Reordering this list is not a fix either
///
/// Putting a measured winner first **flips the family**: with `128x64x8`
/// first the seed adopts that point, the exact cost then prefers `Sgemv` over
/// the whole `Coop` node, and attention's `1024x1024x64` contraction lands on
/// a matrix-*vector* kernel. Part of the 12.96 -> 7.90 in the sweep above is
/// that family flip, not the tile.
///
/// Selection here is **a ranking problem across tiles and families at once**,
/// and the cost model separates neither. An analytical model that ranks GEMM
/// tiles well needs a two-level cache-hit-rate model with per-level bandwidths
/// and a wave/tail occupancy term; `DeviceFacts` carries one `llc_bytes` and
/// one `dram_bytes_per_us` and cannot express it. Ranking them by measurement
/// requires timing candidate *plans*, so family and geometry are ranked
/// together.
fn candidate_geoms_for(operand: Dtype, cx: &DomainCtx<'_>) -> SmallVec<[CoopGeom; 16]> {
    let key = (
        cx.caps.fingerprint(),
        stage_element(operand),
        crate::domains::planner_id(cx.planner),
    );
    GEOM_MEMO.get_or_insert(&key, || generate_geoms(operand, cx))
}

fn generate_geoms(operand: Dtype, cx: &DomainCtx<'_>) -> SmallVec<[CoopGeom; 16]> {
    let caps = cx.caps;
    let width = caps.subgroup_width();
    let max_lanes = caps.limits.max_compute_invocations_per_workgroup;
    let max_bytes = caps.limits.max_compute_workgroup_storage_size;
    let stage = stage_element(operand);

    // `FUSOR2_PIN_COOP="bm,bn,bk"` restricts the domain to one geometry, which
    // is how the table in this module's doc is measured. Ordinary runs never
    // set it.
    let pin: Option<(u32, u32, u32)> = std::env::var("FUSOR2_PIN_COOP").ok().and_then(|v| {
        let p: Vec<u32> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        (p.len() == 3).then(|| (p[0], p[1], p[2]))
    });
    let mut out: SmallVec<[CoopGeom; 16]> = SmallVec::new();
    for bm in BM_CHOICES {
        for bn in BN_CHOICES {
            for bk in BK_CHOICES {
                for subgroups in SUBGROUP_CHOICES {
                    let mut n_passes = 1;
                    while n_passes <= bn / MIN_PASS_COLS {
                        if let Some(geom) = geom_of(bm, bn, bk, n_passes, subgroups)
                            && geom.legal(width, max_lanes)
                            && cx
                                .planner
                                .workgroup_bytes(&coop_tiles(geom, stage), caps)
                                .is_ok_and(|bytes| bytes <= max_bytes)
                        {
                            if pin.is_none_or(|(pm, pn, pk)| {
                                geom.bm == pm && geom.bn == pn && geom.bk == pk
                            }) {
                                out.push(geom);
                            }
                        }
                        n_passes *= 2;
                    }
                }
            }
        }
    }
    out
}

/// One geometry, or `None` when no `(rg, cg)` factorization keeps both
/// fragment sides whole multiples of [`CoopGeom::COOP_DIM`]. There is no
/// fallback split: an unsplittable geometry is simply not a candidate, so it
/// never reaches a kernel whose divisibility asserts would catch it at build
/// time.
fn geom_of(bm: u32, bn: u32, bk: u32, n_passes: u32, subgroups: u32) -> Option<CoopGeom> {
    let (rg, cg) = CoopGeom::subgroup_split(bm, bn, n_passes, subgroups)?;
    Some(CoopGeom {
        bm,
        bn,
        bk,
        n_passes,
        subgroups,
        rg,
        cg,
    })
}

/// Workgroup element the operand stages through: f16 operands stage as f16,
/// everything else as f32.
pub const fn stage_element(operand: Dtype) -> ScalarElement {
    match operand {
        Dtype::F16 => ScalarElement::F16,
        _ => ScalarElement::F32,
    }
}

/// The workgroup tiles one staged operand pair declares: a `bm x (bk + 1)`
/// A tile and a `bk x (bn / n_passes + 1)` B tile, each less the pad after
/// its final row, which is never addressed. The `+1` is the shared-memory
/// bank-conflict pad.
///
/// **This is not the tile set the emitters lay out.** `verify_l1::coop_tiles`
/// is the single source — `check_schedule_domain` and `semantics::work` both
/// call it, and `fusor2-gpu`'s `lower_coop` declares exactly its shapes: an
/// unpadded `[bm, bk]` A tile and `[bk, bn_pass]` B tile, **replicated
/// `staging` times**, plus an f32 accumulator tile when the store element is
/// narrower. Two consequences, in opposite directions:
///
/// - at `staging == 1` this formula is the *stricter* of the two (it charges
///   `bm + bk - 2` elements of bank pad the emitter does not allocate), so
///   every geometry it admits does fit;
/// - at `staging == 2` it is the *looser* one by nearly 2x, so a geometry
///   admitted here can be one `check_schedule_domain` rejects and one whose
///   arena the emitter cannot pack.
///
/// Nothing exercises the second case: every coop point the conformance suite
/// resolves is `staging: 1` at `16x16x8`. Closing it means filtering the
/// geometry list at the deepest staging depth the domain offers, which moves
/// the admitted set for every contraction on the device.
pub fn coop_tiles(geom: CoopGeom, stage: ScalarElement) -> Tiles {
    let bn_pass = geom.bn / geom.n_passes.max(1);
    let a_elems = geom.bm * (geom.bk + 1) - 1;
    let b_elems = geom.bk * (bn_pass + 1) - 1;
    let element = ElementType::Scalar(stage);
    Tiles {
        decls: SmallVec::from_vec(vec![
            Arc::new(TileDecl::new(
                element,
                TileLayout::contiguous(MemoryLevel::Workgroup, &[a_elems]),
                "coop_a",
            )),
            Arc::new(TileDecl::new(
                element,
                TileLayout::contiguous(MemoryLevel::Workgroup, &[b_elems]),
                "coop_b",
            )),
        ]),
    }
}

/// Never-split, plus every divisor of the K loop leaving at least two
/// iterations per workgroup, capped at [`MAX_SPLITS`]. Epilogues do not gate
/// this — whether an epilogue survives a split is `unfuse_coop_epilogue`'s
/// business, not the split generator's.
///
/// A symbolic `k` cannot be divided at compile time, so it emits `[1]`.
pub fn split_candidates(k: Dim, bk: u32) -> Vec<u32> {
    let Some(k) = k.as_const() else {
        return vec![1];
    };
    let bk = u64::from(bk.max(1));
    let iterations = k.div_ceil(bk);
    let limit = (iterations / 2).min(u64::from(MAX_SPLITS)).max(1);
    (1..=limit)
        .filter(|d| *d == 1 || iterations % d == 0)
        .map(|d| d as u32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domains::testing::{apple_caps, baseline_caps, no_coop_caps};
    use crate::domains::{DomainCtx, default_planner};
    use fusor2_ir::device::Caps;
    use fusor2_ir::shape::SymId;

    fn ctx(caps: &Caps) -> DomainCtx<'_> {
        DomainCtx::new(caps, default_planner())
    }

    /// The subgroup-split objective, re-derived here so the expectation is the
    /// algorithm rather than a copied constant.
    fn reference_split(bm: u32, bn: u32, n_passes: u32, subgroups: u32) -> (u32, u32) {
        const COOP_DIM: u32 = 8;
        let bn_pass = bn / n_passes;
        let mut best_rg = 0;
        let mut best_loads = 0;
        let mut rg = 1;
        while rg <= subgroups {
            let cg = subgroups / rg;
            if subgroups % rg == 0 && bm % (COOP_DIM * rg) == 0 && bn_pass % (COOP_DIM * cg) == 0 {
                let loads = cg * bm + rg * bn_pass;
                if best_rg == 0 || loads < best_loads {
                    best_rg = rg;
                    best_loads = loads;
                }
            }
            rg += 1;
        }
        if best_rg == 0 {
            (1, subgroups)
        } else {
            (best_rg, subgroups / best_rg)
        }
    }

    /// Nine reference geometries, as `(bm, bn, bk, subgroups, n_passes)`.
    const REFERENCE_ROWS: [(u32, u32, u32, u32, u32); 9] = [
        (256, 256, 16, 8, 8),
        (128, 512, 16, 8, 8),
        (128, 256, 16, 8, 4),
        (128, 128, 16, 8, 2),
        (128, 64, 16, 8, 1),
        (64, 128, 16, 8, 2),
        (64, 64, 16, 4, 1),
        (64, 16, 16, 4, 1),
        (16, 64, 16, 4, 1),
    ];

    #[test]
    fn subgroup_split_matches_reference() {
        for (bm, bn, _bk, subgroups, n_passes) in REFERENCE_ROWS {
            let got = CoopGeom::subgroup_split(bm, bn, n_passes, subgroups)
                .unwrap_or_else(|| panic!("{bm}x{bn} n_passes={n_passes} has no legal split"));
            assert_eq!(
                got,
                reference_split(bm, bn, n_passes, subgroups),
                "{bm}x{bn} n_passes={n_passes} subgroups={subgroups}"
            );
        }
        // Two rows spelled out. `64x128 / n_passes 2` gives a 64-wide pass,
        // so `(2, 4)` minimizes `cg*64 + rg*64` at 384 against `(1, 8)`'s
        // 576. `128x64 / n_passes 1` lands on `(4, 2)` at
        // `2*128 + 4*64 = 512`, under `(2, 4)`'s 640 — the tie-break to the
        // smaller `rg` never runs here, which is why it has to be the
        // objective and not a table.
        assert_eq!(CoopGeom::subgroup_split(64, 128, 2, 8), Some((2, 4)));
        assert_eq!(CoopGeom::subgroup_split(128, 64, 1, 8), Some((4, 2)));
        // No factorization keeps both sides whole: not a candidate.
        assert_eq!(CoopGeom::subgroup_split(16, 16, 1, 8), None);
    }

    #[test]
    fn every_geom_fits_the_exact_arena() {
        let caps = baseline_caps();
        let cx = ctx(&caps);
        let geoms = candidate_geoms_for(Dtype::F32, &cx);
        assert!(!geoms.is_empty(), "baseline caps admit no coop geometry");
        for g in &geoms {
            let bytes = cx
                .planner
                .workgroup_bytes(&coop_tiles(*g, ScalarElement::F32), &caps)
                .expect("declared tiles have a footprint");
            assert!(
                bytes <= caps.limits.max_compute_workgroup_storage_size,
                "{g:?} needs {bytes}B"
            );
            assert!(g.legal(caps.subgroup_width(), 256), "{g:?} is not legal");
        }
    }

    #[test]
    fn domain_size_is_in_budget() {
        let caps = apple_caps();
        let cx = ctx(&caps);
        let d = coop_domain(
            Dim::Const(4096),
            Dim::Const(4096),
            Dim::Const(4096),
            Dim::Const(1),
            Dtype::F32,
            Dtype::F32,
            &cx,
        );
        assert!(d.geoms.len() >= 40, "only {} geoms", d.geoms.len());
        assert!(
            (6_000..=12_000).contains(&d.len()),
            "domain has {} points ({} geoms x {} splits x {} staging)",
            d.len(),
            d.geoms.len(),
            d.splits.len(),
            d.staging.len()
        );
    }

    #[test]
    fn every_point_round_trips() {
        let caps = apple_caps();
        let cx = ctx(&caps);
        let d = coop_domain(
            Dim::Const(1024),
            Dim::Const(1024),
            Dim::Const(1024),
            Dim::Const(1),
            Dtype::F32,
            Dtype::F32,
            &cx,
        );
        for i in 0..d.len() {
            assert!(d.point(i).is_some(), "point {i} of {} is None", d.len());
        }
        assert!(d.point(d.len()).is_none());
    }

    #[test]
    fn symbolic_k_emits_one_split() {
        let caps = apple_caps();
        let cx = ctx(&caps);
        let d = coop_domain(
            Dim::Const(4096),
            Dim::Const(4096),
            Dim::Sym(SymId(7)),
            Dim::Const(1),
            Dtype::F32,
            Dtype::F32,
            &cx,
        );
        assert_eq!(d.splits.as_slice(), [1]);
        assert_eq!(d.staging.as_slice(), [1, 2]);
    }

    #[test]
    fn padded_shapes_are_not_declined() {
        // At `1x4096x4096` Coop stays a live candidate: no padding gate
        // declines it, it either loses on cost or does not.
        let caps = apple_caps();
        let cx = ctx(&caps);
        let d = coop_domain(
            Dim::Const(1),
            Dim::Const(4096),
            Dim::Const(4096),
            Dim::Const(1),
            Dtype::F32,
            Dtype::F32,
            &cx,
        );
        assert!(!d.is_empty());
    }

    #[test]
    fn no_split_when_k_is_shallow() {
        assert_eq!(split_candidates(Dim::Const(16), 16), vec![1]);
        let caps = apple_caps();
        let cx = ctx(&caps);
        let d = coop_domain(
            Dim::Const(256),
            Dim::Const(256),
            Dim::Const(16),
            Dim::Const(1),
            Dtype::F32,
            Dtype::F32,
            &cx,
        );
        assert_eq!(d.splits.as_slice(), [1]);
    }

    #[test]
    fn split_candidates_match_the_reference_filter() {
        // k = 4096 at bk = 16 is 256 K iterations: 1 plus every divisor of
        // 256 up to 64.
        assert_eq!(
            split_candidates(Dim::Const(4096), 16),
            vec![1, 2, 4, 8, 16, 32, 64]
        );
        // 100 elements at bk = 32 is 4 iterations, limit 2.
        assert_eq!(split_candidates(Dim::Const(100), 32), vec![1, 2]);
    }

    #[test]
    fn no_coop_config_yields_an_empty_domain() {
        let caps = no_coop_caps();
        let cx = ctx(&caps);
        let d = coop_domain(
            Dim::Const(512),
            Dim::Const(512),
            Dim::Const(512),
            Dim::Const(1),
            Dtype::F32,
            Dtype::F32,
            &cx,
        );
        assert!(d.is_empty());
    }

    #[test]
    fn coop_tiles_match_the_reference_footprint() {
        // `stage_pair_elements` for the 128x64x16 row at one pass.
        let geom = geom_of(128, 64, 16, 1, 8).unwrap();
        let tiles = coop_tiles(geom, ScalarElement::F32);
        let elems: u64 = tiles.decls.iter().map(|t| t.layout.element_count()).sum();
        let expected = (128u64 * 17 - 1) + (16u64 * 65 - 1);
        assert_eq!(elems, expected);
    }

    #[test]
    fn f16_operands_stage_as_f16() {
        assert_eq!(stage_element(Dtype::F16), ScalarElement::F16);
        assert_eq!(stage_element(Dtype::F32), ScalarElement::F32);
        assert_eq!(stage_element(Dtype::BF16), ScalarElement::F32);
    }
}
