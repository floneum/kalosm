//! Workgroup-tiled quantized matmul / gemv kernels for GPUs that don't
//! expose `Features::SUBGROUP`.
//!
//! The subgroup-based qmatmul/qgemv paths in this crate partition lanes by
//! `subgroup_id` and reduce via `subgroup_reduce_*`, which `Mesa lavapipe`
//! (Linux CI's software Vulkan) and other adapters without the SUBGROUP
//! feature can't validate. The kernels below mirror the dense
//! `batched_matmul_with_epilogues` strategy — stage A and a dequantized B
//! into workgroup memory, then have each lane accumulate a `TM x TN`
//! register sub-tile — so they only use `program.lane()` and
//! `workgroup_barrier()`. They're cooperative across the workgroup, never
//! the subgroup.
//!
//! The bodies share three building blocks:
//! - [`stage_f32_tile_with_pre`] — cooperative per-lane staging of a dense
//!   f32 source into a workgroup tile, applying the optional pre-activation
//!   epilogue per element. Used for A in both kernels.
//! - `program.copy_quant_to_tile` (from `fusor_tile_ir`) — per-lane
//!   dequantize-into-workgroup-tile for B.
//! - [`accumulate_register_tile_from_workgroup`] — per-lane register
//!   accumulation reading both staged tiles. Generic over the register
//!   tile shape (`TM`, `TN`), so the matmul body uses 4x4 and the gemv
//!   body uses 1x1.

use fusor_tile_ir::tile::{Mask, Program, Storage, Tile, TileBlock, Workgroup};
use fusor_tile_ir::{F32, QuantizedMatrix, TileLiteral, TileReduceOp, U32, WorkgroupAxis};

use crate::kernels::helpers::{dispatch_grid_1d, load_qmatmul_extra};
use crate::types::{
    QmatmulEpilogues, apply_qmatmul_post_epilogue, apply_qmatmul_pre_epilogue, matrix_shape,
};

const QMATMUL_LANES: usize = 64;
const QGEMV_LANES: usize = 64;
const QMATMUL_TM: usize = 4;
const QMATMUL_TN: usize = 4;
const QGEMV_TN: usize = 1;

/// Stage `f32` source rows in `[row_base, row_base + ROWS)` and cols in
/// `[col_base, col_base + COLS)` into the workgroup tile `dst`, applying
/// `pre` per element. Cooperative across all `LANES` workgroup lanes. Pads
/// out-of-bound source positions with zero, and guards the workgroup-tile
/// store so lanes with `flat >= ROWS * COLS` don't write past the tile
/// (qgemv passes a 1xBK tile to a 64-lane workgroup; the unused lanes
/// would otherwise corrupt adjacent workgroup memory).
#[allow(clippy::too_many_arguments)]
fn stage_f32_tile_with_pre(
    program: &mut TileBlock<'_>,
    dst: Workgroup<F32>,
    src: &Storage<F32, 2>,
    row_base: &Tile<U32>,
    col_base: &Tile<U32>,
    tile_active: &Mask,
    src_rows: u32,
    src_cols: u32,
    epilogues: &QmatmulEpilogues<'_>,
    rows: u32,
    cols: u32,
    lanes: u32,
) {
    let tile_elements = rows * cols;
    let passes = (rows * cols).div_ceil(lanes);
    for pass in 0..passes {
        let flat = program.lane() + pass * lanes;
        let local_row = flat.clone() / cols;
        let local_col = flat.clone() % cols;
        let global_row = row_base.clone() + local_row.clone();
        let global_col = col_base.clone() + local_col.clone();
        let within_tile = flat.clone().lt(tile_elements);
        let in_bounds = tile_active
            .clone()
            .and(within_tile.clone())
            .and(global_row.clone().lt(src_rows))
            .and(global_col.clone().lt(src_cols));
        let loaded = program.load(
            src.at((global_row.clone(), &global_col)),
            in_bounds.clone(),
            0.0,
        );
        let pre_extras = epilogues
            .pre_extra_inputs
            .iter()
            .map(|extra| load_qmatmul_extra(program, extra, &global_row, &global_col, src_cols))
            .collect::<Vec<_>>();
        let value = Tile::select(
            in_bounds,
            apply_qmatmul_pre_epilogue(epilogues, loaded, pre_extras),
            Tile::literal(TileLiteral::f32(0.0)),
        );
        // Re-use the same flat index but only emit the store on lanes that
        // map to an actual tile slot.
        let flat_for_store = flat.clone();
        program.if_then(within_tile, |program| {
            program.store_workgroup(dst, flat_for_store, value);
        });
    }
}

/// Per-lane register accumulation `acc = A_tile @ B_tile` for a `TM x TN`
/// sub-tile rooted at `(lane_row * TM, lane_col * TN)` in the workgroup
/// tiles. Caller is responsible for the surrounding `workgroup_barrier()`s.
///
/// Layout: `A_tile` is row-major `BM x BK` (index = row*BK + k), `B_tile` is
/// row-major `BK x BN` (index = k*BN + col).
fn accumulate_register_tile_from_workgroup(
    program: &mut TileBlock<'_>,
    a_tile: Workgroup<F32>,
    b_tile: Workgroup<F32>,
    lane_row: &Tile<U32>,
    lane_col: &Tile<U32>,
    bn: u32,
    bk: u32,
    tm: u32,
    tn: u32,
) -> Vec<Tile> {
    (0..tm * tn)
        .map(|idx| {
            let r = idx / tn;
            let c = idx % tn;
            let local_row = lane_row.clone() * tm + r;
            let local_col = lane_col.clone() * tn + c;
            let mut sum = Tile::literal(TileLiteral::f32(0.0));
            for kk in 0..bk {
                let a_value = program.load_workgroup(a_tile, local_row.clone() * bk + kk);
                let b_value = program.load_workgroup(b_tile, local_col.clone() + kk * bn);
                sum = sum + a_value * b_value;
            }
            sum
        })
        .collect()
}

/// Workgroup-tiled quantized matmul. Each workgroup produces a `BM x BN`
/// output tile by staging A and a dequantized B into workgroup memory and
/// having every lane accumulate a `TM x TN` register sub-tile. No subgroup
/// ops — uses only `program.lane()` and `workgroup_barrier()`.
///
/// `BM` and `BN` must equal 32 (matches the `4x4` register tile across 64
/// lanes). `BK` is the K-axis staging chunk.
pub fn qmatmul_workgroup_with_epilogues(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
    epilogues: &QmatmulEpilogues<'_>,
    max_workgroups_per_dimension: u32,
) {
    // BM/BN are pinned to the 4x4-register-tile geometry across QMATMUL_LANES.
    // BK is the K-axis staging chunk per pass.
    const BM: u32 = 32;
    const BN: u32 = 32;
    const BK: u32 = 8;
    let bk = BK;

    let [m, k] = matrix_shape(&a.view().layout);
    let n = b.cols;
    assert_eq!(k, b.rows, "qmatmul K dimensions must match");
    let [y_m, y_n] = matrix_shape(&y.view().layout);
    assert_eq!(m, y_m, "qmatmul output row count must match A");
    assert_eq!(n, y_n, "qmatmul output column count must match B");

    let tiles_m = m.div_ceil(BM);
    let tiles_n = n.div_ceil(BN);
    let total_tiles = tiles_m * tiles_n;
    let k_tiles = k.div_ceil(bk);
    let grid = dispatch_grid_1d(total_tiles, max_workgroups_per_dimension);
    let a_tile = program.alloc_workgroup_tile_f32(BM, bk);
    let b_tile = program.alloc_workgroup_tile_f32(bk, BN);
    let b_clone = b.clone();

    program.program_grid::<QMATMUL_LANES>(grid, |program| {
        let tile_id = program.program_id(WorkgroupAxis::X)
            + program.program_id(WorkgroupAxis::Y) * grid[0]
            + program.program_id(WorkgroupAxis::Z) * grid[0] * grid[1];
        let tile_active = tile_id.clone().lt(total_tiles);
        let m_tile = tile_id.clone() / tiles_n;
        let n_tile = tile_id % tiles_n;

        let lane = program.lane();
        let lane_row = lane.clone() / (BN / QMATMUL_TN as u32);
        let lane_col = lane % (BN / QMATMUL_TN as u32);
        let m_tile_base = m_tile * BM;
        let n_tile_base = n_tile * BN;
        let row_base = m_tile_base.clone() + lane_row.clone() * QMATMUL_TM as u32;
        let col_base = n_tile_base.clone() + lane_col.clone() * QMATMUL_TN as u32;

        let sums: [Tile; QMATMUL_TM * QMATMUL_TN] = program
            .loop_fold_n::<{ QMATMUL_TM * QMATMUL_TN }, _, _>(
                TileReduceOp::Sum,
                k_tiles,
                [TileLiteral::f32(0.0); QMATMUL_TM * QMATMUL_TN],
                |program, k_tile| {
                    let k_base = k_tile * bk;
                    stage_f32_tile_with_pre(
                        program,
                        a_tile,
                        a,
                        &m_tile_base,
                        &k_base,
                        &tile_active,
                        m,
                        k,
                        epilogues,
                        BM,
                        bk,
                        QMATMUL_LANES as u32,
                    );
                    program.copy_quant_to_tile(b_tile, &b_clone, &k_base, &n_tile_base);
                    program.workgroup_barrier();

                    let chunk_vec = accumulate_register_tile_from_workgroup(
                        program,
                        a_tile,
                        b_tile,
                        &lane_row,
                        &lane_col,
                        BN,
                        bk,
                        QMATMUL_TM as u32,
                        QMATMUL_TN as u32,
                    );
                    let mut chunk_iter = chunk_vec.into_iter();
                    let chunk: [Tile; QMATMUL_TM * QMATMUL_TN] = std::array::from_fn(|_| {
                        program.bind(chunk_iter.next().expect("register tile size matches"))
                    });
                    program.workgroup_barrier();
                    chunk
                },
            );

        for (idx, sum) in sums.into_iter().enumerate() {
            let r = idx / QMATMUL_TN;
            let c = idx % QMATMUL_TN;
            let row = row_base.clone() + r as u32;
            let col = col_base.clone() + c as u32;
            let extras = epilogues
                .post_extra_inputs
                .iter()
                .map(|extra| load_qmatmul_extra(program, extra, &row, &col, n))
                .collect::<Vec<_>>();
            let value = apply_qmatmul_post_epilogue(epilogues, sum, extras);
            let mask = tile_active
                .clone()
                .and(row.clone().lt(m))
                .and(col.clone().lt(n));
            program.store(y.at((row, col)), value, mask);
        }
    });
}

/// Workgroup-tiled quantized GEMV (`m == 1`) for adapters without subgroups.
/// All `QGEMV_LANES` lanes fan out across the BN columns of one output tile.
/// Stages A's single row into workgroup memory and reuses
/// [`accumulate_register_tile_from_workgroup`] with `TM = 1`, `TN = 1`.
pub fn qgemv_workgroup_with_epilogue(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &QuantizedMatrix,
    y: &Storage<F32, 2>,
    epilogues: &QmatmulEpilogues<'_>,
    max_workgroups_per_dimension: u32,
) {
    // BN is pinned to QGEMV_LANES (one column per lane). BK is the K-axis
    // staging chunk per pass.
    const BN: u32 = QGEMV_LANES as u32;
    const BK: u32 = 8;
    let bk = BK;

    let [m, k] = matrix_shape(&a.view().layout);
    let n = b.cols;
    assert_eq!(m, 1, "qgemv_workgroup expects a single input row");
    assert_eq!(k, b.rows, "qgemv K dimensions must match");
    let [y_m, y_n] = matrix_shape(&y.view().layout);
    assert_eq!(y_m, 1, "qgemv output must be single-row");
    assert_eq!(n, y_n, "qgemv output column count must match B");

    let tiles_n = n.div_ceil(BN);
    let k_tiles = k.div_ceil(bk);
    let grid = dispatch_grid_1d(tiles_n, max_workgroups_per_dimension);
    // BM=1 logical row tile. Reuse the f32 stager with ROWS=1.
    let a_tile = program.alloc_workgroup_tile_f32(1, bk);
    let b_tile = program.alloc_workgroup_tile_f32(bk, BN);
    let b_clone = b.clone();

    program.program_grid::<QGEMV_LANES>(grid, |program| {
        let tile_id = program.program_id(WorkgroupAxis::X)
            + program.program_id(WorkgroupAxis::Y) * grid[0]
            + program.program_id(WorkgroupAxis::Z) * grid[0] * grid[1];
        let tile_active = tile_id.clone().lt(tiles_n);
        let n_tile = tile_id;
        let n_tile_base = n_tile * BN;

        let lane = program.lane();
        let lane_row = Tile::literal(TileLiteral::U32(0));
        let lane_col = lane;
        let row_base = Tile::literal(TileLiteral::U32(0));
        let col_base = n_tile_base.clone() + lane_col.clone() * QGEMV_TN as u32;

        let sums: [Tile; QGEMV_TN] = program.loop_fold_n::<QGEMV_TN, _, _>(
            TileReduceOp::Sum,
            k_tiles,
            [TileLiteral::f32(0.0); QGEMV_TN],
            |program, k_tile| {
                let k_base = k_tile * bk;
                stage_f32_tile_with_pre(
                    program,
                    a_tile,
                    a,
                    &row_base,
                    &k_base,
                    &tile_active,
                    1,
                    k,
                    epilogues,
                    1,
                    bk,
                    QGEMV_LANES as u32,
                );
                program.copy_quant_to_tile(b_tile, &b_clone, &k_base, &n_tile_base);
                program.workgroup_barrier();

                let chunk_vec = accumulate_register_tile_from_workgroup(
                    program,
                    a_tile,
                    b_tile,
                    &lane_row,
                    &lane_col,
                    BN,
                    bk,
                    1,
                    QGEMV_TN as u32,
                );
                let mut chunk_iter = chunk_vec.into_iter();
                let chunk: [Tile; QGEMV_TN] = std::array::from_fn(|_| {
                    program.bind(chunk_iter.next().expect("register tile size matches"))
                });
                program.workgroup_barrier();
                chunk
            },
        );

        for (idx, sum) in sums.into_iter().enumerate() {
            let row = Tile::literal(TileLiteral::U32(0));
            let col = col_base.clone() + idx as u32;
            let extras = epilogues
                .post_extra_inputs
                .iter()
                .map(|extra| load_qmatmul_extra(program, extra, &row, &col, n))
                .collect::<Vec<_>>();
            let value = apply_qmatmul_post_epilogue(epilogues, sum, extras);
            let mask = tile_active.clone().and(col.clone().lt(n));
            program.store(y.at((0u32, col)), value, mask);
        }
    });
}
