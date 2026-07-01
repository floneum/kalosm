//! Quantized matrix multiply program kernels.

use fusor_tile_ir::tile::{range, Program, Storage, Tile};
use fusor_tile_ir::{CoopMatrixToken, QuantizedMatrix, ScalarElement, WorkgroupAxis};

use crate::{
    dispatch::SubgroupConfig,
    kernels::helpers::{
        coop_acc_grid, coop_acc_grid_set_c, coop_load_a_fragments, coop_load_b_fragments,
        coop_mma_grid, coop_store_acc_grid, load_qmatmul_extra,
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
/// use fusor_tile_ir::{tile, ElementType, GgmlQuantFormat, Shape};
/// use fusor_tile_ir_kernels::{qmatmul_with_epilogue, quantized_matrix, QmatmulEpilogues};
///
/// let ir = tile::build(|program| {
///     let a = program.storage_read(ElementType::F32, Shape::new([8, 256]));
///     let b = quantized_matrix(program, GgmlQuantFormat::Q8_0, 256, 16);
///     let y = program.storage_write(ElementType::F32, Shape::new([8, 16]));
///     let subgroup = fusor_tile_ir::SubgroupToken::new_unchecked();
///     qmatmul_with_epilogue(
///         program,
///         &a,
///         &b,
///         &y,
///         &QmatmulEpilogues::empty(),
///         fusor_tile_ir::CoopMatrixToken::new_unchecked(),
///         fusor_tile_ir_kernels::SubgroupConfig::fixed(subgroup, 32),
///         64,
///         64,
///         32,
///     );
/// });
/// # let _ = ir;
/// ```
#[allow(clippy::too_many_arguments)]
pub fn qmatmul_with_epilogue(
    program: &mut Program,
    a: &Storage,
    b: &QuantizedMatrix,
    y: &Storage,
    epilogues: &crate::types::QmatmulEpilogues<'_>,
    coop: CoopMatrixToken,
    subgroups: SubgroupConfig,
    bm: u32,
    bn: u32,
    bk: u32,
) {
    assert!(
        bm > 0 && bn > 0 && bk > 0,
        "qmatmul tile shape must be non-zero"
    );
    let [m, k] = matrix_shape(a.layout());
    let [y_m, y_n] = matrix_shape(y.layout());
    assert_eq!(k, b.rows, "qmatmul K dimensions must match");
    assert_eq!(m, y_m, "qmatmul output row count must match A");
    assert_eq!(b.cols, y_n, "qmatmul output column count must match B");

    if m == 1 {
        super::qgemv::qgemv_with_epilogue(program, a, b, y, 1, subgroups, epilogues);
    } else {
        qmatmul_tile_with_epilogue(program, a, b, y, epilogues, coop, subgroups, bm, bn, bk);
    }
}

/// Scalar lane-mapped qmatmul body with optional pre/post epilogues. Public
/// so downstream crates can reproduce or replace the variant-selection layer
/// above (`qmatmul_options_with_epilogue` / `qmatmul_with_epilogue`).
///
/// The (bm, bn, bk) argument only drives the cooperative fast-path selection.
/// If coop is unsupported or epilogues are non-empty, falls back to a fixed
/// 8x4x8 scalar tile that's small enough to always fit `LANES=256`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn qmatmul_tile_with_epilogue(
    program: &mut Program,
    a: &Storage,
    b: &QuantizedMatrix,
    y: &Storage,
    epilogues: &crate::types::QmatmulEpilogues<'_>,
    coop: CoopMatrixToken,
    subgroups: SubgroupConfig,
    bm: u32,
    bn: u32,
    bk: u32,
) {
    const LANES: u32 = 256;
    // Scalar fallback tile (8 * 4 * 8 == 256 == LANES).
    const SCALAR_BM: u32 = 8;
    const SCALAR_BN: u32 = 4;
    const SCALAR_BK: u32 = 8;
    assert!(
        bm > 0 && bn > 0 && bk > 0,
        "qmatmul tile shape must be non-zero"
    );
    let [m, k] = matrix_shape(a.layout());

    if epilogues.pre.is_none()
        && epilogues.pre_with_extras.is_none()
        && epilogues.post.is_none()
        && epilogues.post_with_extras.is_none()
        && qmatmul_try_coop(
            program,
            a,
            b,
            epilogues.post_acc_init_col_vector,
            y,
            coop,
            subgroups,
            bm,
            bn,
            bk,
        )
    {
        return;
    }

    let k_iterations = k.div_ceil(SCALAR_BK);
    program.program_grid(
        LANES,
        [b.cols.div_ceil(SCALAR_BN), m.div_ceil(SCALAR_BM), 1],
        |program| {
            let lane = program.lane();
            let k_lane = lane.clone() % SCALAR_BK;
            let output_lane = lane / SCALAR_BK;
            let row_lane = output_lane.clone() / SCALAR_BN;
            let col_lane = output_lane % SCALAR_BN;
            let row = program.program_id(WorkgroupAxis::Y) * SCALAR_BM + row_lane;
            let col = program.program_id(WorkgroupAxis::X) * SCALAR_BN + col_lane;
            let [partial] = program.fold(
                range(k_iterations),
                [Tile::f32(0.0)],
                |program, loop_index, [acc]| {
                    let k_index = loop_index * SCALAR_BK + k_lane.clone();
                    let mask = row.lt(m) & col.lt(b.cols) & k_index.lt(k);
                    let loaded = program.load(a.at((&row, &k_index)), mask.clone(), 0.0);
                    let pre_extras = epilogues
                        .pre_extra_inputs
                        .iter()
                        .map(|extra| load_qmatmul_extra(program, extra, &row, &k_index, k))
                        .collect::<Vec<_>>();
                    let a_value = apply_qmatmul_pre_epilogue(epilogues, loaded, pre_extras);
                    let b_value = program.load_quantized(b, &k_index, &col, mask.clone(), 0.0);
                    [acc + a_value * b_value]
                },
            );
            let reduced = program.group_reduce_sum(SCALAR_BK, partial);
            let extras = epilogues
                .post_extra_inputs
                .iter()
                .map(|extra| load_qmatmul_extra(program, extra, &row, &col, b.cols))
                .collect::<Vec<_>>();
            let sum = apply_qmatmul_post_epilogue(epilogues, reduced, extras);
            let store_mask = k_lane.eq(0) & row.lt(m) & col.lt(b.cols);
            program.store(y.at((row, col)), sum, store_mask);
        },
    );
}

/// Emit the cooperative-matrix qmatmul body when the requested tile shape
/// matches a supported fast tile geometry. All branches instantiate the same
/// runtime body; only the tile dimensions differ.
/// `(bm, bn, bk, row_groups, col_groups)` for the supported coop-matrix
/// tile geometries. Current quantized fast tiles all use BK=32.
const QMATMUL_COOP_TILE_TABLE: &[(u32, u32, u32, u32, u32)] = &[
    (64, 32, 32, 2, 1),
    (64, 64, 32, 2, 2),
    (64, 128, 32, 2, 4),
    (128, 64, 32, 4, 2),
    (128, 128, 32, 4, 4),
];

/// Try the cooperative-matrix fast path for the requested `(bm, bn, bk)` tile.
/// `acc_init` is the optional rank-1 column vector seeding the accumulator
/// before the K-loop (the "preloaded C" path). The `block` workgroup size is a
/// runtime `u32` threaded straight into `qmatmul_coop` — no const-generic
/// monomorphization and no `match block` fan-out.
#[allow(clippy::too_many_arguments)]
pub(crate) fn qmatmul_try_coop(
    program: &mut Program,
    a: &Storage,
    b: &QuantizedMatrix,
    acc_init: Option<&Storage>,
    y: &Storage,
    coop: CoopMatrixToken,
    subgroups: SubgroupConfig,
    bm: u32,
    bn: u32,
    bk: u32,
) -> bool {
    if std::env::var_os("FUSOR_DIAG_DISABLE_COOP").is_some() {
        return false;
    }
    if b.format.is_q4k_family() || b.format.is_q6k_family() {
        return false;
    }
    if !subgroups.is_fixed() {
        return false;
    }
    let Some(&(_, _, table_bk, row_groups, col_groups)) = QMATMUL_COOP_TILE_TABLE
        .iter()
        .find(|&&(m, n, candidate_bk, ..)| (m, n, candidate_bk) == (bm, bn, bk))
    else {
        return false;
    };
    let [m, k] = matrix_shape(a.layout());
    if !m.is_multiple_of(bm)
        || !b.cols.is_multiple_of(bn)
        || !k.is_multiple_of(table_bk)
        || !cooperative_store_layout_supported(y.layout())
    {
        return false;
    }
    qmatmul_coop(
        program, a, b, acc_init, y, coop, bm, bn, table_bk, row_groups, col_groups, subgroups,
    );
    true
}

/// Cooperative-matrix qmatmul body. Each workgroup produces one BMxBN output
/// tile via an interleaved `ROW_GROUPS x COL_GROUPS` grid of subgroups, each
/// holding `(32*32)/(8*8)` = 16 cooperative-matrix accumulators.
///
/// When `acc_init` is `Some`, each accumulator is seeded with the broadcast
/// C-role fragment from the column vector instead of zero.
#[allow(clippy::too_many_arguments)]
pub(crate) fn qmatmul_coop(
    program: &mut Program,
    a: &Storage,
    b: &QuantizedMatrix,
    acc_init: Option<&Storage>,
    y: &Storage,
    coop: CoopMatrixToken,
    bm: u32,
    bn: u32,
    bk: u32,
    row_groups: u32,
    col_groups: u32,
    subgroups: SubgroupConfig,
) {
    const COOP_DIM: u32 = 8;
    const SUBGROUP_ROWS: u32 = 32;
    const SUBGROUP_COLS: u32 = 32;
    let subgroup = subgroups.token();
    let block = subgroups.block_for_subgroups(row_groups * col_groups);
    debug_assert_eq!(row_groups * SUBGROUP_ROWS, bm);
    debug_assert_eq!(col_groups * SUBGROUP_COLS, bn);

    let [m, k] = matrix_shape(a.layout());
    let n = b.cols;
    let n_grid_x = n / bn;
    let n_grid_y = m / bm;
    let k_iterations = k / bk;

    let a_tile = program.alloc_workgroup_tile(ScalarElement::F32, bm, bk);
    let b_tile = program.alloc_workgroup_tile(ScalarElement::F32, bk, bn);
    let b_clone = b.clone();

    const TILE_ROWS_PER_SG: u32 = SUBGROUP_ROWS / 8;
    const TILE_COLS_PER_SG: u32 = SUBGROUP_COLS / 8;

    program.program_grid(block, [n_grid_x, n_grid_y, 1], |program| {
        let row_base = program.program_id(WorkgroupAxis::Y) * bm;
        let col_base = program.program_id(WorkgroupAxis::X) * bn;
        let subgroup_id = subgroup.subgroup_id(program);
        let sg_row = subgroup_id.clone() / col_groups;
        let sg_col = subgroup_id % col_groups;
        let sg_row_base = sg_row * SUBGROUP_ROWS;
        let sg_col_base = sg_col * SUBGROUP_COLS;

        let accs = match acc_init {
            None => coop_acc_grid(
                program,
                coop,
                ScalarElement::F32,
                TILE_ROWS_PER_SG,
                TILE_COLS_PER_SG,
                |program, coop, _, _| {
                    coop.coop_zero(program, ScalarElement::F32, COOP_DIM, COOP_DIM)
                },
            ),
            Some(init) => {
                let acc_init_col_base = col_base.clone() + sg_col_base.clone();
                coop_acc_grid_set_c(
                    program,
                    coop,
                    init,
                    &acc_init_col_base,
                    ScalarElement::F32,
                    TILE_ROWS_PER_SG,
                    TILE_COLS_PER_SG,
                )
            }
        };

        program.loop_range(k_iterations, |program, loop_index| {
            let k_base = loop_index * bk;
            program.fill_tile(&a_tile, a, &row_base, &k_base);
            program.fill_tile_quantized(&b_tile, &b_clone, &k_base, &col_base);
            program.workgroup_barrier();

            let kk_steps = bk / COOP_DIM;
            for kk in 0..kk_steps {
                let a_frags = coop_load_a_fragments(
                    program,
                    coop,
                    &a_tile,
                    &sg_row_base,
                    kk,
                    TILE_ROWS_PER_SG,
                    ScalarElement::F32,
                );
                let b_frags = coop_load_b_fragments(
                    program,
                    coop,
                    &b_tile,
                    &sg_col_base,
                    kk,
                    TILE_COLS_PER_SG,
                    ScalarElement::F32,
                );
                coop_mma_grid(program, coop, &accs, &a_frags, &b_frags);
            }
            program.workgroup_barrier();
        });

        coop_store_acc_grid(
            program,
            coop,
            &accs,
            y,
            None,
            &row_base,
            &col_base,
            &sg_row_base,
            &sg_col_base,
        );
    });
}
