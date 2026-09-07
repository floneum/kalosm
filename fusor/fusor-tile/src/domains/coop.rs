//! The cooperative-matrix schedule domain: `geoms x splits x staging`,
//! carried whole on the node and resolved by extraction.
//!
//! Every `(bm, bn, bk, subgroups, n_passes)` whose closed-form subgroup
//! split exists, whose lanes fit, and whose exact arena footprint fits is
//! a candidate.

use fusor_ir::device::Caps;
use fusor_ir::dtype::Dtype;
use fusor_ir::ir::kernel::{ElementType, MemoryLevel, ScalarElement, TileDecl, TileLayout, Tiles};
use fusor_ir::ir::launch::{CoopDomain, CoopGeom};
use fusor_ir::shape::Dim;
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

/// Delegates to [`coop_domain`] with `batch = 1` and the crate-default
/// planner.
pub fn legal(m: Dim, n: Dim, k: Dim, operand: Dtype, acc: Dtype, caps: &Caps) -> CoopDomain {
    let cx = DomainCtx::new(caps, crate::domains::default_planner());
    coop_domain(m, n, k, Dim::Const(1), operand, acc, &cx)
}

/// Every legal `(geom, splits, staging)` for this contraction on this
/// device. Empty when the device reports no usable cooperative
/// configuration, which simply makes the `Coop` alternative unselectable —
/// never an error.
pub fn coop_domain(
    m: Dim,
    n: Dim,
    k: Dim,
    batch: Dim,
    operand: Dtype,
    acc: Dtype,
    cx: &DomainCtx<'_>,
) -> CoopDomain {
    // `m`, `n` and `batch` price the domain; they do not filter it by value.
    // Edge tiles fill zero past the logical extents, so no concrete shape is
    // illegal for any geometry.
    //
    // A symbolic `m` or `n` empties the domain: the whole-block cooperative
    // store requires an output padded to the geometry's tile, and a padding
    // of `Sym(s)` to a tile multiple is not expressible as a `Dim`. Symbolic
    // `k`/`batch` stay legal — they never enter the padded layout.
    let _ = batch;
    if m.as_const().is_none() || n.as_const().is_none() {
        return CoopDomain::default();
    }

    if cx.caps.coop_for(operand, acc).is_none() {
        return CoopDomain::default();
    }

    let geoms = candidate_geoms_for(operand, cx);
    if geoms.is_empty() {
        return CoopDomain::default();
    }

    // Splits are a domain-level list while `bk` is per-geometry, so the
    // candidate set is the union over the surviving depths: a split count
    // is a candidate when *some* surviving geometry admits it. A pair whose
    // spans do not partition K exactly still runs — the split kernel bounds
    // its K span — so the union is sound, and cost rejects the rest.
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

/// `(caps, staged element, planner identity) -> geometries`. The grid below
/// is ~7,000 candidates each costing an exact arena query, and none of it
/// depends on the contraction — only on the device.
static GEOM_MEMO: crate::domains::DomainMemo<
    (Caps, ScalarElement, usize),
    SmallVec<[CoopGeom; 16]>,
> = crate::domains::DomainMemo::new();

/// Every one of the ~5,700 points this domain carries is generated as a candidate.
/// The optimum is shape-dependent.
fn candidate_geoms_for(operand: Dtype, cx: &DomainCtx<'_>) -> SmallVec<[CoopGeom; 16]> {
    let key = (
        cx.caps.clone(),
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

    // `FUSOR_PIN_COOP="bm,bn,bk"` restricts the domain to one geometry, for
    // measurement. Ordinary runs never set it.
    let pin: Option<(u32, u32, u32)> = std::env::var("FUSOR_PIN_COOP").ok().and_then(|v| {
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
                            && pin.is_none_or(|(pm, pn, pk)| {
                                geom.bm == pm && geom.bn == pn && geom.bk == pk
                            })
                        {
                            out.push(geom);
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
/// fragment sides whole multiples of [`CoopGeom::COOP_DIM`]; an
/// unsplittable geometry is simply not a candidate.
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
/// iterations per workgroup, capped at [`MAX_SPLITS`].
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
