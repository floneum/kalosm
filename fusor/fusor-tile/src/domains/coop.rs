//! The cooperative-matrix schedule domain: supported `(geometry, staging)` pairs,
//! carried whole on the node and resolved by extraction.
//!
//! Every `(bm, bn, bk, subgroups, n_passes)` whose closed-form subgroup
//! split exists, whose lanes fit, and whose exact arena footprint fits is
//! a candidate.

use fusor_ir::device::Caps;
use fusor_ir::dtype::Dtype;
use fusor_ir::ir::kernel::ScalarElement;
use fusor_ir::ir::launch::{CoopDomain, CoopGeom, CoopSchedule};
use fusor_ir::shape::Dim;
use smallvec::SmallVec;

use crate::domains::DomainCtx;

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

/// Every supported `(geom, staging)` for this contraction on this
/// device. Empty when the device reports no usable cooperative configuration;
/// callers then decline to construct the `Coop` alternative.
pub fn coop_domain(
    m: Dim,
    n: Dim,
    _k: Dim,
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

    if !cx.caps.coop_supported()
        || !cx.caps.coop.iter().any(|c| {
            c.operand == operand
                && c.acc == acc
                && c.m == CoopGeom::COOP_DIM
                && c.n == CoopGeom::COOP_DIM
                && c.k == CoopGeom::COOP_DIM
        })
    {
        return CoopDomain::default();
    }

    CoopDomain {
        schedules: candidate_schedules_for(operand, cx),
    }
}

/// Geometry and staging depend only on the device and staged element type.
static GEOM_MEMO: crate::domains::DomainMemo<
    (Caps, ScalarElement, usize),
    SmallVec<[CoopSchedule; 16]>,
> = crate::domains::DomainMemo::new();

fn candidate_schedules_for(operand: Dtype, cx: &DomainCtx<'_>) -> SmallVec<[CoopSchedule; 16]> {
    let key = (
        cx.caps.clone(),
        stage_element(operand),
        crate::domains::planner_id(cx.planner),
    );
    GEOM_MEMO.get_or_insert(&key, || generate_schedules(operand, cx))
}

fn generate_schedules(operand: Dtype, cx: &DomainCtx<'_>) -> SmallVec<[CoopSchedule; 16]> {
    let caps = cx.caps;
    let width = caps.subgroup_width();
    let max_lanes = caps
        .limits
        .max_compute_invocations_per_workgroup
        .min(caps.limits.max_compute_workgroup_size[0]);
    let max_bytes = caps.limits.max_compute_workgroup_storage_size;
    let stage = stage_element(operand);
    let pin: Option<Vec<u32>> = std::env::var("FUSOR_PIN_COOP").ok().and_then(|v| {
        let p: Vec<u32> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        (p.len() >= 3).then_some(p)
    });
    let mut out = SmallVec::new();
    for bm in BM_CHOICES {
        for bn in BN_CHOICES {
            for bk in BK_CHOICES {
                for subgroups in SUBGROUP_CHOICES {
                    let mut n_passes = 1;
                    while n_passes <= bn / MIN_PASS_COLS {
                        if let Some(geom) = geom_of(bm, bn, bk, n_passes, subgroups)
                            && geom.legal(width, max_lanes)
                            && pin.as_ref().is_none_or(|p| {
                                geom.bm == p[0]
                                    && geom.bn == p[1]
                                    && geom.bk == p[2]
                                    && p.get(3).is_none_or(|sg| geom.subgroups == *sg)
                                    && p.get(4).is_none_or(|np| geom.n_passes == *np)
                            })
                        {
                            for staging in [1, 2] {
                                if cx
                                    .planner
                                    .workgroup_bytes(&coop_tiles(geom, stage, staging), caps)
                                    .is_ok_and(|bytes| bytes <= max_bytes)
                                {
                                    out.push(CoopSchedule { geom, staging });
                                }
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

/// Shared scratch declarations for cooperative schedules.
pub use fusor_ir::ir::launch::coop_tiles;
