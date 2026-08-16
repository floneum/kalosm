//! The SGEMV schedule domain. The domain is a pure function of the device.

use fusor2_ir::device::Caps;
use fusor2_ir::dtype::Dtype;
use fusor2_ir::ir::launch::{SgemvDomain, SgemvParams};
use fusor2_ir::shape::Dim;
use smallvec::SmallVec;

use crate::domains::{DomainCtx, UNMEASURED, sgemv_order};

/// Lane k-window widths. The wide entries (32, 64) exist for quantized
/// operands: a 32-element window amortizes one group-scale decode across the
/// whole group, and a 64-element window makes the lane consume both packed
/// halves of every data word it touches, so no word is ever loaded by two
/// lanes.
const VECTOR_CHOICES: [u32; 7] = [1, 2, 4, 8, 16, 32, 64];
const SUBGROUP_CHOICES: [u32; 6] = [1, 2, 4, 8, 16, 32];
/// Columns per workgroup. `1` is the whole-workgroup-per-element structure;
/// the rest hand each subgroup `cols / subgroups` columns (generated only
/// when that divides, on a fixed-subgroup device).
const COLS_CHOICES: [u32; 6] = [1, 2, 4, 8, 16, 32];
/// Accumulator-pressure bound: each lane carries `cols / subgroups`
/// accumulators plus its activation window, so a subgroup never owns more
/// than this many columns.
const MAX_COLS_PER_SUBGROUP: u32 = 8;
/// Unroll-pressure bound on the multi-column structure: the loop body is
/// `vector` activation loads plus `vector * (cols / subgroups)` FMAs, all
/// unrolled. 256 admits the widest window at 4 columns (64 x 4) and the
/// reference-shaped 32 x 8; past that the body is register spill, not math.
const MAX_UNROLL: u32 = 256;
/// Runs a split lane window is laid out as (`1` = consecutive, the only
/// structure the whole-workgroup path has). A split window revisits the same
/// packed word of a bit-packed operand at several k offsets, so its word
/// loads hash-cons to one evaluation.
const PARTS_CHOICES: [u32; 2] = [2, 4];
/// K distances between a split window's runs. The divisibility rules in the
/// generator make each one tile the pass exactly.
const GAP_CHOICES: [u32; 3] = [16, 32, 64];

/// Measured cells, position = move-ordering rank. `sample_points` offers the
/// front five of the domain to the race for every key, so the first five
/// entries must together cover every (shape, dtype) winner — the race picks
/// per key, the order only has to *reach* each winner, not rank one dtype
/// over another.
pub static SEED_CELLS: &[SgemvParams] = &[
    // Measured on M2 Max over six 8B-decode qgemv shapes; pin with
    // warm-cache runs only — a fresh tune cache re-fights the coop
    // candidates and contaminates the pin. This kernel has no
    // cross-subgroup sharing (no barrier, no workgroup scratch), so small
    // blocks at cps=2 beat 256-thread blocks; cps=1 halves activation
    // reuse and cps=4 at small blocks spills.
    //
    // 64-thread block: best on gateup q4k/q6k and down q6k.
    w(32, 2, 4, 4, 32),
    // One subgroup per workgroup: best on attn q4k/q6k and down q4k.
    w(32, 1, 2, 4, 32),
    // Near-universal 256-thread cell, kept as incumbent safety.
    w(32, 8, 16, 4, 32),
    // Attn-sized Q6K winner: the 16-window keeps the Q6K decode inside
    // the unroll budget where 32-windows spill.
    w(16, 8, 32, 4, 32),
    // Unsplit multi-column: the structure for operands where a split window
    // has nothing to hash-cons.
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

/// Compatibility entry point kept for the scaffold's `domains::sgemv_legal`
/// re-export. None of `m`, `n`, `k` or `dtype` filters the domain.
pub fn legal(m: Dim, n: Dim, k: Dim, dtype: Dtype, caps: &Caps) -> SgemvDomain {
    let _ = (m, n, k, dtype);
    let cx = DomainCtx::new(caps, crate::domains::default_planner());
    sgemv_domain(&cx)
}

/// Every legal `(vector, subgroups, cols, parts, gap)` on this device,
/// ordered by `(seed_rank, sgemv_order)`.
pub fn sgemv_domain(cx: &DomainCtx<'_>) -> SgemvDomain {
    let width = cx.caps.subgroup_width();
    let max_lanes = cx.caps.limits.max_compute_invocations_per_workgroup;
    // The subgroup-per-column structure indexes lanes by `subgroup_id` and
    // reduces within one subgroup, which is only a static schedule when the
    // device pins its subgroup width.
    let fixed_subgroup = cx.caps.subgroups.is_some_and(|s| s.is_fixed());

    // `FUSOR2_PIN_SGEMV="vector,subgroups,cols[,parts,gap]"` restricts the
    // domain to one cell for measuring per-shape kernel tables without the
    // adoption race in the loop. Ordinary runs never set it.
    let pin: Option<SgemvParams> = std::env::var("FUSOR2_PIN_SGEMV").ok().and_then(|s| {
        let p: Vec<u32> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
        match p.len() {
            3 => Some(v(p[0], p[1], p[2])),
            5 => Some(w(p[0], p[1], p[2], p[3], p[4])),
            _ => None,
        }
    });

    let mut all: Vec<SgemvParams> = Vec::new();
    for vector in VECTOR_CHOICES {
        for subgroups in SUBGROUP_CHOICES {
            // The launched block is `subgroups * subgroup_width`; a
            // block wider than the device's invocation limit cannot be
            // created at all.
            if subgroups.saturating_mul(width) > max_lanes {
                continue;
            }
            for cols in COLS_CHOICES {
                if cols > 1
                    && (!fixed_subgroup
                        || cols % subgroups != 0
                        || cols / subgroups > MAX_COLS_PER_SUBGROUP
                        || vector * (cols / subgroups) > MAX_UNROLL)
                {
                    continue;
                }
                let cell = v(vector, subgroups, cols);
                if pin.is_none_or(|p| p == cell) {
                    all.push(cell);
                }
                // Split lane windows: only on the multi-column structure,
                // and only where runs, gap and width tile the subgroup's
                // pass exactly (the same divisibility `verify_launch` holds
                // every domain to). The window still holds `vector`
                // elements, so the unroll bound above already applies.
                if cols <= 1 {
                    continue;
                }
                for parts in PARTS_CHOICES {
                    if vector % parts != 0 {
                        continue;
                    }
                    let run = vector / parts;
                    for gap in GAP_CHOICES {
                        if run == 0
                            || gap % run != 0
                            || gap <= run
                            || (width * run) % gap != 0
                        {
                            continue;
                        }
                        let cell = w(vector, subgroups, cols, parts, gap);
                        if pin.is_none_or(|p| p == cell) {
                            all.push(cell);
                        }
                    }
                }
            }
        }
    }

    all.sort_by_key(|q| {
        // A seed's position is its rank: the front of the domain is the
        // exact prefix `sample_points` hands the race, in belief order.
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
