//! Quantized matrix multiply program kernels.

use fusor_tile_ir::tile::{Program, Storage};
use fusor_tile_ir::{QuantizedMatrix, TileLiteral, TileReduceOp, WorkgroupAxis, F32};

use crate::{
    kernels::helpers::{
        coop_load_a_fragments, coop_load_b_fragments, coop_load_c_broadcast_fragments,
        coop_mma_grid, coop_set_c_grid, coop_store_acc_grid, load_qmatmul_extra,
        zero_coop_acc_grid,
    },
    types::{
        apply_qmatmul_post_epilogue, apply_qmatmul_pre_epilogue,
        cooperative_store_layout_supported, matrix_shape,
    },
};

/// Top-level quantized matrix multiply with optional activation/output
/// epilogues. Single-row inputs keep using qgemv; multi-row inputs use the
/// generalized qmatmul body. Callers with no epilogue pass
/// `&QmatmulEpilogues::empty()`.
///
/// ```
/// use fusor_tile_ir::{tile, GgmlQuantFormat, Shape, F32};
/// use fusor_tile_ir_kernels::{qmatmul_with_epilogue, quantized_matrix, QmatmulEpilogues};
///
/// let ir = tile::build(|program| {
///     let a = program.storage_read::<F32, 2>(Shape::new([8, 256]));
///     let b = quantized_matrix(program, GgmlQuantFormat::Q8_0, 256, 16);
///     let y = program.storage_write::<F32, 2>(Shape::new([8, 16]));
///     qmatmul_with_epilogue(program, &a, &b, &y, 4, &QmatmulEpilogues::empty(), 64, 64);
/// });
/// # let _ = ir;
/// ```
pub fn qmatmul_with_epilogue(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
    vector_width: u32,
    epilogues: &crate::types::QmatmulEpilogues<'_>,
    bm: u32,
    bn: u32,
) {
    assert!(bm > 0 && bn > 0, "qmatmul tile shape must be non-zero");
    assert!(vector_width > 0, "qmatmul vector width must be non-zero");
    let [m, k] = matrix_shape(&a.view().layout);
    let [y_m, y_n] = matrix_shape(&y.view().layout);
    assert_eq!(k, b.rows, "qmatmul K dimensions must match");
    assert_eq!(m, y_m, "qmatmul output row count must match A");
    assert_eq!(b.cols, y_n, "qmatmul output column count must match B");

    if m == 1 {
        super::qgemv::qgemv_with_epilogue(program, a, b, y, 1, epilogues);
    } else {
        qmatmul_tile_with_epilogue(program, a, b, y, epilogues, bm, bn);
    }
}

/// Scalar lane-mapped qmatmul body with optional pre/post epilogues. Public
/// so downstream crates can reproduce or replace the variant-selection layer
/// above (`qmatmul_options_with_epilogue` / `qmatmul_with_epilogue`).
///
/// The (bm, bn) argument only drives the cooperative fast-path selection.
/// If coop is unsupported or epilogues are non-empty, falls back to a fixed
/// 8x4x8 scalar tile that's small enough to always fit `LANES=256`.
pub(crate) fn qmatmul_tile_with_epilogue(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
    epilogues: &crate::types::QmatmulEpilogues<'_>,
    bm: u32,
    bn: u32,
) {
    const LANES: usize = 256;
    // Scalar fallback tile (8 * 4 * 8 == 256 == LANES).
    const SCALAR_BM: u32 = 8;
    const SCALAR_BN: u32 = 4;
    const SCALAR_BK: u32 = 8;
    assert!(bm > 0 && bn > 0, "qmatmul tile shape must be non-zero");
    let [m, k] = matrix_shape(&a.view().layout);

    if epilogues.pre.is_none()
        && epilogues.pre_with_extras.is_none()
        && epilogues.post.is_none()
        && epilogues.post_with_extras.is_none()
    {
        if let Some(acc_init) = epilogues.post_acc_init_col_vector {
            if qmatmul_try_coop_acc_init(program, a, b, acc_init, y, bm, bn) {
                return;
            }
        } else if qmatmul_try_coop(program, a, b, y, bm, bn) {
            return;
        }
    }

    let k_iterations = k.div_ceil(SCALAR_BK);
    program.program_grid::<LANES>(
        [b.cols.div_ceil(SCALAR_BN), m.div_ceil(SCALAR_BM), 1],
        |program| {
            let lane = program.lane();
            let k_lane = lane.clone() % SCALAR_BK;
            let output_lane = lane / SCALAR_BK;
            let row_lane = output_lane.clone() / SCALAR_BN;
            let col_lane = output_lane % SCALAR_BN;
            let row = program.program_id(WorkgroupAxis::Y) * SCALAR_BM + row_lane;
            let col = program.program_id(WorkgroupAxis::X) * SCALAR_BN + col_lane;
            let partial = program.loop_fold(
                TileReduceOp::Sum,
                k_iterations,
                TileLiteral::f32(0.0),
                |program, loop_index| {
                    let k_index = loop_index * SCALAR_BK + k_lane.clone();
                    let mask = row.lt(m).and(col.lt(b.cols)).and(k_index.lt(k));
                    let loaded = program.load(a.at((&row, &k_index)), mask.clone(), 0.0);
                    let pre_extras = epilogues
                        .pre_extra_inputs
                        .iter()
                        .map(|extra| load_qmatmul_extra(program, extra, &row, &k_index, k))
                        .collect::<Vec<_>>();
                    let a_value = apply_qmatmul_pre_epilogue(epilogues, loaded, pre_extras);
                    let b_value = program.load_quantized(b, &k_index, &col, mask.clone(), 0.0);
                    a_value * b_value
                },
            );
            let reduced = program.group_reduce_sum(SCALAR_BK, partial);
            let extras = epilogues
                .post_extra_inputs
                .iter()
                .map(|extra| load_qmatmul_extra(program, extra, &row, &col, b.cols))
                .collect::<Vec<_>>();
            let sum = apply_qmatmul_post_epilogue(epilogues, reduced, extras);
            let store_mask = k_lane.eq(0).and(row.lt(m)).and(col.lt(b.cols));
            program.store(y.at((row, col)), sum, store_mask);
        },
    );
}

/// Emit the cooperative-matrix qmatmul body when the requested tile shape
/// matches a supported fast tile geometry. All branches instantiate the same
/// generic body; only the tile dimensions differ.
/// `(bm, bn, bk, row_groups, col_groups, block)` for the supported coop-matrix
/// tile geometries. BK is pinned to 32 by the cooperative-matrix MMA shape
/// (8x8x8 along K, 4 lanes per subgroup).
const QMATMUL_COOP_TILE_TABLE: &[(u32, u32, u32, u32, u32, usize)] = &[
    (64, 32, 32, 2, 1, 64),
    (64, 64, 32, 2, 2, 128),
    (64, 128, 32, 2, 4, 256),
    (128, 64, 32, 4, 2, 256),
    (128, 128, 32, 4, 4, 512),
];

pub(crate) fn qmatmul_try_coop(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
    bm: u32,
    bn: u32,
) -> bool {
    let Some(&(_, _, bk, row_groups, col_groups, block)) = QMATMUL_COOP_TILE_TABLE
        .iter()
        .find(|&&(m, n, ..)| (m, n) == (bm, bn))
    else {
        return false;
    };
    let [m, k] = matrix_shape(&a.view().layout);
    if !m.is_multiple_of(bm)
        || !b.cols.is_multiple_of(bn)
        || !k.is_multiple_of(bk)
        || !cooperative_store_layout_supported(&y.view().layout)
    {
        return false;
    }
    match block {
        64 => qmatmul_perf::<64>(program, a, b, y, bm, bn, bk, row_groups, col_groups),
        128 => qmatmul_perf::<128>(program, a, b, y, bm, bn, bk, row_groups, col_groups),
        256 => qmatmul_perf::<256>(program, a, b, y, bm, bn, bk, row_groups, col_groups),
        512 => qmatmul_perf::<512>(program, a, b, y, bm, bn, bk, row_groups, col_groups),
        other => panic!("unsupported qmatmul coop BLOCK {other}"),
    }
    true
}

pub(crate) fn qmatmul_try_coop_acc_init(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    acc_init: &Storage<F32, 1>,
    y: &Storage<F32, 2>,
    bm: u32,
    bn: u32,
) -> bool {
    let Some(&(_, _, bk, row_groups, col_groups, block)) = QMATMUL_COOP_TILE_TABLE
        .iter()
        .find(|&&(m, n, ..)| (m, n) == (bm, bn))
    else {
        return false;
    };
    let [m, k] = matrix_shape(&a.view().layout);
    if !m.is_multiple_of(bm)
        || !b.cols.is_multiple_of(bn)
        || !k.is_multiple_of(bk)
        || !cooperative_store_layout_supported(&y.view().layout)
    {
        return false;
    }
    match block {
        64 => qmatmul_perf_acc_init::<64>(
            program, a, b, acc_init, y, bm, bn, bk, row_groups, col_groups,
        ),
        128 => qmatmul_perf_acc_init::<128>(
            program, a, b, acc_init, y, bm, bn, bk, row_groups, col_groups,
        ),
        256 => qmatmul_perf_acc_init::<256>(
            program, a, b, acc_init, y, bm, bn, bk, row_groups, col_groups,
        ),
        512 => qmatmul_perf_acc_init::<512>(
            program, a, b, acc_init, y, bm, bn, bk, row_groups, col_groups,
        ),
        other => panic!("unsupported qmatmul coop BLOCK {other}"),
    }
    true
}

/// Cooperative-matrix qmatmul body. Each workgroup produces one BMxBN output
/// tile via an interleaved `ROW_GROUPS x COL_GROUPS` grid of subgroups, each
/// holding `(32*32)/(8*8)` = 16 cooperative-matrix accumulators.
/// `BLOCK == ROW_GROUPS * COL_GROUPS * 32`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn qmatmul_perf<const BLOCK: usize>(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
    bm: u32,
    bn: u32,
    bk: u32,
    row_groups: u32,
    col_groups: u32,
) {
    const COOP_DIM: u32 = 8;
    const SUBGROUP_SIZE: u32 = 32;
    const SUBGROUP_ROWS: u32 = 32;
    const SUBGROUP_COLS: u32 = 32;
    debug_assert_eq!(row_groups * SUBGROUP_ROWS, bm);
    debug_assert_eq!(col_groups * SUBGROUP_COLS, bn);
    debug_assert_eq!(row_groups * col_groups * SUBGROUP_SIZE, BLOCK as u32);

    let [m, k] = matrix_shape(&a.view().layout);
    let n = b.cols;
    let n_grid_x = n / bn;
    let n_grid_y = m / bm;
    let k_iterations = k / bk;

    let a_tile = program.alloc_workgroup_tile_f32(bm, bk);
    let b_tile = program.alloc_workgroup_tile_f32(bk, bn);
    let b_clone = b.clone();
    let a_clone = a;
    let y_clone = y;

    const TILE_ROWS_PER_SG: u32 = SUBGROUP_ROWS / 8;
    const TILE_COLS_PER_SG: u32 = SUBGROUP_COLS / 8;

    program.program_grid::<BLOCK>([n_grid_x, n_grid_y, 1], |program| {
        let row_base = program.program_id(WorkgroupAxis::Y) * bm;
        let col_base = program.program_id(WorkgroupAxis::X) * bn;
        let subgroup_id = program.subgroup_id();
        let sg_row = subgroup_id.clone() / col_groups;
        let sg_col = subgroup_id % col_groups;
        let sg_row_base = sg_row * SUBGROUP_ROWS;
        let sg_col_base = sg_col * SUBGROUP_COLS;

        let accs = zero_coop_acc_grid(program, TILE_ROWS_PER_SG, TILE_COLS_PER_SG);

        program.while_true(k_iterations, |program, loop_index| {
            let k_base = loop_index * bk;
            program.copy_storage_to_tile(a_tile, a_clone, &row_base, &k_base);
            program.copy_quant_to_tile(b_tile, &b_clone, &k_base, &col_base);
            program.workgroup_barrier();

            let kk_steps = bk / COOP_DIM;
            for kk in 0..kk_steps {
                let a_frags =
                    coop_load_a_fragments(program, a_tile, &sg_row_base, kk, TILE_ROWS_PER_SG);
                let b_frags =
                    coop_load_b_fragments(program, b_tile, &sg_col_base, kk, TILE_COLS_PER_SG);
                coop_mma_grid(program, &accs, &a_frags, &b_frags);
            }
            program.workgroup_barrier();
        });

        coop_store_acc_grid(
            program,
            &accs,
            y_clone,
            None,
            &row_base,
            &col_base,
            &sg_row_base,
            &sg_col_base,
        );
    });
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn qmatmul_perf_acc_init<const BLOCK: usize>(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    acc_init: &Storage<F32, 1>,
    y: &Storage<F32, 2>,
    bm: u32,
    bn: u32,
    bk: u32,
    row_groups: u32,
    col_groups: u32,
) {
    const COOP_DIM: u32 = 8;
    const SUBGROUP_SIZE: u32 = 32;
    const SUBGROUP_ROWS: u32 = 32;
    const SUBGROUP_COLS: u32 = 32;
    debug_assert_eq!(row_groups * SUBGROUP_ROWS, bm);
    debug_assert_eq!(col_groups * SUBGROUP_COLS, bn);
    debug_assert_eq!(row_groups * col_groups * SUBGROUP_SIZE, BLOCK as u32);

    let [m, k] = matrix_shape(&a.view().layout);
    let n = b.cols;
    let n_grid_x = n / bn;
    let n_grid_y = m / bm;
    let k_iterations = k / bk;

    let a_tile = program.alloc_workgroup_tile_f32(bm, bk);
    let b_tile = program.alloc_workgroup_tile_f32(bk, bn);
    let b_clone = b.clone();
    let a_clone = a;
    let acc_init_clone = acc_init;
    let y_clone = y;

    const TILE_ROWS_PER_SG: u32 = SUBGROUP_ROWS / 8;
    const TILE_COLS_PER_SG: u32 = SUBGROUP_COLS / 8;

    program.program_grid::<BLOCK>([n_grid_x, n_grid_y, 1], |program| {
        let row_base = program.program_id(WorkgroupAxis::Y) * bm;
        let col_base = program.program_id(WorkgroupAxis::X) * bn;
        let subgroup_id = program.subgroup_id();
        let sg_row = subgroup_id.clone() / col_groups;
        let sg_col = subgroup_id % col_groups;
        let sg_row_base = sg_row * SUBGROUP_ROWS;
        let sg_col_base = sg_col * SUBGROUP_COLS;

        let accs = zero_coop_acc_grid(program, TILE_ROWS_PER_SG, TILE_COLS_PER_SG);
        let acc_init_col_base = col_base.clone() + sg_col_base.clone();
        let c_frags = coop_load_c_broadcast_fragments(
            program,
            acc_init_clone,
            &acc_init_col_base,
            TILE_COLS_PER_SG,
        );
        coop_set_c_grid(program, &accs, &c_frags);

        program.while_true(k_iterations, |program, loop_index| {
            let k_base = loop_index * bk;
            program.copy_storage_to_tile(a_tile, a_clone, &row_base, &k_base);
            program.copy_quant_to_tile(b_tile, &b_clone, &k_base, &col_base);
            program.workgroup_barrier();

            let kk_steps = bk / COOP_DIM;
            for kk in 0..kk_steps {
                let a_frags =
                    coop_load_a_fragments(program, a_tile, &sg_row_base, kk, TILE_ROWS_PER_SG);
                let b_frags =
                    coop_load_b_fragments(program, b_tile, &sg_col_base, kk, TILE_COLS_PER_SG);
                coop_mma_grid(program, &accs, &a_frags, &b_frags);
            }
            program.workgroup_barrier();
        });

        coop_store_acc_grid(
            program,
            &accs,
            y_clone,
            None,
            &row_base,
            &col_base,
            &sg_row_base,
            &sg_col_base,
        );
    });
}
