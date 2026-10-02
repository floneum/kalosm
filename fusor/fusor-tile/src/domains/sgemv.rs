//! The SGEMV schedule domain. The domain is a pure function of the device.

use fusor_ir::device::Caps;
use fusor_ir::ir::launch::{SgemvDomain, SgemvParams};
use smallvec::SmallVec;

use crate::domains::{DomainCtx, UNMEASURED, sgemv_order};

/// Lane k-window widths. 32 and 64 serve quantized operands: one group-scale
/// decode per window, and both packed halves of every word in one lane.
const VECTOR_CHOICES: [u32; 7] = [1, 2, 4, 8, 16, 32, 64];
const SUBGROUP_CHOICES: [u32; 6] = [1, 2, 4, 8, 16, 32];
/// Columns per workgroup: `1` is whole-workgroup-per-element, the rest give
/// each subgroup `cols / subgroups` columns.
const COLS_CHOICES: [u32; 6] = [1, 2, 4, 8, 16, 32];
/// Accumulator-pressure bound on columns per subgroup.
const MAX_COLS_PER_SUBGROUP: u32 = 8;
/// Unroll-pressure bound on `vector * (cols / subgroups)`; past it the body
/// spills.
const MAX_UNROLL: u32 = 256;
/// Runs a split lane window is laid out as (`1` = consecutive). A split
/// window revisits packed words, whose loads hash-cons to one.
const PARTS_CHOICES: [u32; 2] = [2, 4];
/// K distances between a split window's runs.
const GAP_CHOICES: [u32; 3] = [16, 32, 64];

/// Measured cells, position = move-ordering rank. The race sees the front
/// five, so they must cover every (shape, dtype) winner.
pub static SEED_CELLS: &[SgemvParams] = &[
    // Measured on M2 Max over 8B-decode qgemv shapes, warm cache only.
    // 64-thread block: best on gateup q4k/q6k and down q6k.
    w(32, 2, 4, 4, 32),
    // One subgroup per workgroup: best on attn q4k/q6k and down q4k.
    w(32, 1, 2, 4, 32),
    // Near-universal 256-thread cell, kept as incumbent safety.
    w(32, 8, 16, 4, 32),
    // Attn-sized Q6K winner: the 16-window stays inside the unroll budget.
    w(16, 8, 32, 4, 32),
    // Unsplit multi-column, for operands a split window cannot hash-cons.
    v(16, 8, 16),
    // Runners-up, kept as explorer fodder past the race prefix.
    w(32, 8, 32, 4, 32),
    w(16, 8, 16, 4, 32),
    w(16, 8, 16, 2, 32),
    w(32, 8, 32, 2, 32),
    v(32, 8, 32),
    v(16, 4, 1),
    v(16, 4, 16),
    v(16, 8, 32),
    v(8, 8, 32),
    v(8, 8, 8),
    v(8, 4, 16),
    v(4, 16, 1),
    v(4, 1, 1),
    v(4, 2, 1),
    v(4, 8, 1),
    v(4, 32, 1),
    v(2, 8, 1),
    v(2, 1, 1),
    v(2, 16, 1),
    v(2, 32, 1),
];

const fn v(vector: u32, subgroups: u32, cols: u32) -> SgemvParams {
    w(vector, subgroups, cols, 1, 0)
}

const fn w(vector: u32, subgroups: u32, cols: u32, parts: u32, gap: u32) -> SgemvParams {
    SgemvParams {
        vector,
        subgroups,
        cols,
        parts,
        gap,
    }
}

/// Whether the subgroup-per-column structure tiles `p` exactly: fixed-width
/// subgroups each owning `cols / subgroups` columns, and a split window whose
/// runs, gap and width tile the pass. The lowering relies on this.
pub fn cols_structure_legal(p: &SgemvParams, caps: &Caps) -> bool {
    let width = caps.subgroup_width();
    let run = p.run();
    caps.subgroups.is_some_and(|s| s.is_fixed())
        && p.subgroups > 0
        && p.subgroups.saturating_mul(width) <= max_lanes(caps)
        && p.cols.is_multiple_of(p.subgroups)
        && (p.parts <= 1
            || (p.vector.is_multiple_of(p.parts)
                && run > 0
                && p.gap.is_multiple_of(run)
                && p.gap > run
                && (width * run).is_multiple_of(p.gap)))
}

fn max_lanes(caps: &Caps) -> u32 {
    caps.limits
        .max_compute_invocations_per_workgroup
        .min(caps.limits.max_compute_workgroup_size[0])
}

/// Every legal `(vector, subgroups, cols, parts, gap)` on this device,
/// ordered by `(seed_rank, sgemv_order)`.
pub fn sgemv_domain(cx: &DomainCtx<'_>) -> SgemvDomain {
    let width = cx.caps.subgroup_width();

    // `FUSOR_PIN_SGEMV="vector,subgroups,cols[,parts,gap]"` pins one cell.
    let pin = crate::flags::flags()
        .pin_sgemv
        .as_ref()
        .and_then(|p| match p.len() {
            3 => Some(v(p[0], p[1], p[2])),
            5 => Some(w(p[0], p[1], p[2], p[3], p[4])),
            _ => None,
        });

    let mut all: Vec<SgemvParams> = Vec::new();
    for vector in VECTOR_CHOICES {
        for subgroups in SUBGROUP_CHOICES {
            // The block must fit the device's invocation limit.
            if subgroups.saturating_mul(width) > max_lanes(cx.caps) {
                continue;
            }
            for cols in COLS_CHOICES {
                let cell = v(vector, subgroups, cols);
                if cols > 1
                    && (!cols_structure_legal(&cell, cx.caps)
                        || cols / subgroups > MAX_COLS_PER_SUBGROUP
                        || vector * (cols / subgroups) > MAX_UNROLL)
                {
                    continue;
                }
                if pin.is_none_or(|p| p == cell) {
                    all.push(cell);
                }
                // Split windows: multi-column only, and only where runs, gap and width
                // tile the subgroup's pass exactly.
                if cols <= 1 {
                    continue;
                }
                for parts in PARTS_CHOICES {
                    for gap in GAP_CHOICES {
                        let cell = w(vector, subgroups, cols, parts, gap);
                        if cols_structure_legal(&cell, cx.caps) && pin.is_none_or(|p| p == cell) {
                            all.push(cell);
                        }
                    }
                }
            }
        }
    }

    all.sort_by_key(|q| {
        // A seed's position is its rank, so the race prefix is in belief order.
        let rank = SEED_CELLS
            .iter()
            .position(|s| s == q)
            .map_or(UNMEASURED, |i| i as u8);
        (rank, sgemv_order(q))
    });

    SgemvDomain {
        params: SmallVec::from_vec(all),
    }
}
