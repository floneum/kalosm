//! One register-tile template for the quantized matmul / gemv family that
//! runs where the subgroup paths can't: adapters without
//! `Features::SUBGROUP`, plus every f16-activation quantized matmul (the
//! selector routes those here on any adapter).
//!
//! The subgroup-based qmatmul/qgemv paths in this crate partition lanes by
//! `subgroup_id` and reduce via `subgroup_reduce_*`, which `Mesa lavapipe`
//! (Linux CI's software Vulkan) and other adapters without the SUBGROUP
//! feature can't validate. The template below stages A and a dequantized B
//! into workgroup memory and has each lane accumulate a `tm x tn` register
//! sub-tile, so it only uses `program.lane()` and `workgroup_barrier()`.
//! It's cooperative across the workgroup, never the subgroup.
//!
//! Two geometries instantiate it: [`MATMUL_SHAPE`] for the general case and
//! the single-row [`GEMV_SHAPE`] for `m == 1`. A single-row grid has no M
//! axis — every row index folds to the constant 0, all lanes fan across N,
//! and the accumulators fan across the post epilogue's matrix-column offsets.
//!
//! Storage and staging element types travel as [`ScalarElement`] data through
//! [`AccumCast`]. The body shares three building blocks:
//! - [`stage_storage_tile_with_pre`] — cooperative per-lane staging of a dense
//!   source into a workgroup tile, applying the optional pre-activation
//!   epilogue per element. Used for A.
//! - [`TileBlock::fill_tile_quantized`] — per-lane dequantize-into-workgroup-
//!   tile for B. This path is not coop-forcing, preserving the lavapipe
//!   invariant that this kernel requests neither `SUBGROUP` nor
//!   `COOPERATIVE_MATRIX`.
//! - [`accumulate_register_tile_from_workgroup`] — per-lane register
//!   accumulation reading both staged tiles.

use fusor_tile_ir::tile::{Mask, Program, Storage, Tile, TileBlock, WorkgroupTile};
use fusor_tile_ir::{QuantizedMatrix, ScalarElement, WorkgroupAxis};

use crate::kernels::helpers::{dispatch_grid_1d, load_qmatmul_extra, scalar_of, AccumCast};
use crate::types::{
    apply_qmatmul_post_epilogue_values, apply_qmatmul_pre_epilogue, matrix_shape, QmatmulEpilogues,
};

const LANES: u32 = 64;

/// One instantiation's geometry: a workgroup covers a `bm x bn` output tile,
/// staging `bk` K-elements of A and B per pass, and every lane accumulates a
/// `tm x tn` register sub-tile.
#[derive(Clone, Copy)]
struct WorkgroupTileShape {
    bm: u32,
    bn: u32,
    bk: u32,
    tm: u32,
    tn: u32,
}

/// `bm`/`bn` are pinned to the 4x4-register-tile geometry across [`LANES`].
const MATMUL_SHAPE: WorkgroupTileShape = WorkgroupTileShape {
    bm: 32,
    bn: 32,
    bk: 8,
    tm: 4,
    tn: 4,
};

/// Single output row, one output column per lane.
const GEMV_SHAPE: WorkgroupTileShape = WorkgroupTileShape {
    bm: 1,
    bn: LANES,
    bk: 8,
    tm: 1,
    tn: 1,
};

/// Stage `src` rows in `[row_base, row_base + rows)` and cols in
/// `[col_base, col_base + cols)` into the workgroup tile `dst`, applying
/// `pre` per element. Cooperative across all [`LANES`] workgroup lanes. Pads
/// out-of-bound source positions with zero, and guards the workgroup-tile
/// store so lanes with `flat >= rows * cols` don't write past the tile
/// (the gemv shape passes a 1xBK tile to a 64-lane workgroup; the unused
/// lanes would otherwise corrupt adjacent workgroup memory).
///
/// `stor_cast` promotes storage loads into the f32 accumulator; `staging_cast`
/// demotes the post-pre-epilogue f32 value back to the staged tile element.
#[allow(clippy::too_many_arguments)]
fn stage_storage_tile_with_pre(
    program: &mut TileBlock<'_>,
    dst: &WorkgroupTile,
    src: &Storage,
    stor_cast: &AccumCast,
    staging_cast: &AccumCast,
    row_base: &Tile,
    col_base: &Tile,
    tile_active: &Mask,
    src_rows: u32,
    src_cols: u32,
    epilogues: &QmatmulEpilogues<'_>,
    rows: u32,
    cols: u32,
) {
    let tile_elements = rows * cols;
    let passes = (rows * cols).div_ceil(LANES);
    for pass in 0..passes {
        let flat = program.lane() + pass * LANES;
        let local_row = flat.clone() / cols;
        let local_col = flat.clone() % cols;
        let global_row = row_base.clone() + local_row.clone();
        let global_col = col_base.clone() + local_col.clone();
        let within_tile = flat.clone().lt(tile_elements);
        let in_bounds = tile_active.clone()
            & within_tile.clone()
            & global_row.clone().lt(src_rows)
            & global_col.clone().lt(src_cols);
        let loaded = program.load(
            src.at((global_row.clone(), &global_col)),
            in_bounds.clone(),
            stor_cast.zero_storage(),
        );
        let loaded = stor_cast.into_accum(loaded);
        let pre_extras = epilogues
            .pre_extra_inputs
            .iter()
            .map(|extra| load_qmatmul_extra(program, extra, &global_row, &global_col, src_cols))
            .collect::<Vec<_>>();
        let value = Tile::select(
            in_bounds,
            apply_qmatmul_pre_epilogue(epilogues, loaded, pre_extras),
            Tile::f32(0.0),
        );
        let value = staging_cast.from_accum(value);
        // Re-use the same flat index but only emit the store on lanes that
        // map to an actual tile slot.
        let flat_for_store = flat.clone();
        program.if_then(within_tile, |program| {
            program.store_workgroup(dst, flat_for_store, value);
        });
    }
}

/// Per-lane register accumulation `acc = A_tile @ B_tile` for a `tm x tn`
/// sub-tile rooted at `(lane_row * tm, lane_col * tn)` in the workgroup
/// tiles. Caller is responsible for the surrounding `workgroup_barrier()`s.
///
/// Layout: `A_tile` is row-major `bm x bk` (index = row*bk + k), `B_tile` is
/// row-major `bk x bn` (index = k*bn + col). `staging_cast` promotes the
/// staged-tile loads to the f32 accumulator.
fn accumulate_register_tile_from_workgroup(
    program: &mut TileBlock<'_>,
    a_tile: &WorkgroupTile,
    b_tile: &WorkgroupTile,
    staging_cast: &AccumCast,
    lane_row: &Tile,
    lane_col: &Tile,
    shape: WorkgroupTileShape,
) -> Vec<Tile> {
    let WorkgroupTileShape { bn, bk, tm, tn, .. } = shape;
    (0..tm * tn)
        .map(|idx| {
            let r = idx / tn;
            let c = idx % tn;
            let local_row = lane_row.clone() * tm + r;
            let local_col = lane_col.clone() * tn + c;
            let mut sum = Tile::f32(0.0);
            for kk in 0..bk {
                let a_value = staging_cast
                    .into_accum(program.load_workgroup(a_tile, local_row.clone() * bk + kk));
                let b_value = staging_cast
                    .into_accum(program.load_workgroup(b_tile, local_col.clone() + kk * bn));
                sum = sum + a_value * b_value;
            }
            sum
        })
        .collect()
}

/// Workgroup-tiled quantized matmul staged through `staging_element` (f16
/// staging requires shader-f16 support; accumulation stays f32 and the output
/// keeps the storage element). Each workgroup produces one output tile by
/// staging A and a dequantized B into workgroup memory and having every lane
/// accumulate its register sub-tile. No subgroup ops — uses only
/// `program.lane()` and `workgroup_barrier()`.
pub fn qmatmul_workgroup_with_epilogues(
    program: &mut Program,
    a: &Storage,
    b: &QuantizedMatrix,
    y: &Storage,
    staging_element: ScalarElement,
    epilogues: &QmatmulEpilogues<'_>,
    max_workgroups_per_dimension: u32,
) {
    let [m, k] = matrix_shape(&a.view().layout);
    let shape = if m == 1 { GEMV_SHAPE } else { MATMUL_SHAPE };
    let WorkgroupTileShape { bm, bn, bk, tm, tn } = shape;
    let single_row = bm == 1;

    // F32 accumulation throughout; storage / staging elements are runtime data.
    let stor_cast = AccumCast::new(scalar_of(a.element()), ScalarElement::F32);
    let staging_cast = AccumCast::new(staging_element, ScalarElement::F32);

    let n = if single_row {
        epilogues.post_output_cols(b.cols)
    } else {
        b.cols
    };
    assert_eq!(k, b.rows, "qmatmul K dimensions must match");
    let [y_m, y_n] = matrix_shape(&y.view().layout);
    assert_eq!(m, y_m, "qmatmul output row count must match A");
    assert_eq!(n, y_n, "qmatmul output column count must match B");

    // The single-row shape accumulates one value per matrix-column offset of
    // the post epilogue (the default offset list is a single `0`); the tiled
    // shape holds one accumulator per output element at the tile's own base.
    let column_offsets: Vec<Option<u32>> = if single_row {
        epilogues
            .post_accumulator_offsets()
            .iter()
            .copied()
            .map(Some)
            .collect()
    } else {
        vec![None]
    };
    let value_count = column_offsets.len();

    let tiles_m = m.div_ceil(bm);
    let tiles_n = n.div_ceil(bn);
    let total_tiles = tiles_m * tiles_n;
    let k_tiles = k.div_ceil(bk);
    let grid = dispatch_grid_1d(total_tiles, max_workgroups_per_dimension);
    let a_tile = program.alloc_workgroup_tile(staging_element, bm, bk);
    let b_tile = program.alloc_workgroup_tile(staging_element, bk, bn);
    let b = b.clone();

    program.program_grid(LANES, grid, |program| {
        let tile_id = program.program_id(WorkgroupAxis::X)
            + program.program_id(WorkgroupAxis::Y) * grid[0]
            + program.program_id(WorkgroupAxis::Z) * grid[0] * grid[1];
        let tile_active = tile_id.clone().lt(total_tiles);
        let lane = program.lane();
        let (m_tile_base, n_tile) = if single_row {
            (Tile::u32(0), tile_id)
        } else {
            (tile_id.clone() / tiles_n * bm, tile_id % tiles_n)
        };
        let (lane_row, lane_col) = if single_row {
            (Tile::u32(0), lane)
        } else {
            (lane.clone() / (bn / tn), lane % (bn / tn))
        };
        let n_tile_base = n_tile * bn;
        let row_base = if single_row {
            Tile::u32(0)
        } else {
            m_tile_base.clone() + lane_row.clone() * tm
        };
        let col_base = n_tile_base.clone() + lane_col.clone() * tn;

        let sums = program.fold_vec(
            fusor_tile_ir::tile::range(k_tiles),
            vec![Tile::f32(0.0); (tm * tn) as usize * value_count],
            |program, k_tile, accs| {
                let k_base = k_tile * bk;
                stage_storage_tile_with_pre(
                    program,
                    &a_tile,
                    a,
                    &stor_cast,
                    &staging_cast,
                    &m_tile_base,
                    &k_base,
                    &tile_active,
                    m,
                    k,
                    epilogues,
                    bm,
                    bk,
                );
                let mut next = accs;
                for (value_idx, offset) in column_offsets.iter().enumerate() {
                    let b_col_base = match offset {
                        Some(offset) => n_tile_base.clone() + *offset,
                        None => n_tile_base.clone(),
                    };
                    program.fill_tile_quantized(&b_tile, &b, k_base.clone(), b_col_base);
                    program.workgroup_barrier();

                    let chunks = accumulate_register_tile_from_workgroup(
                        program,
                        &a_tile,
                        &b_tile,
                        &staging_cast,
                        &lane_row,
                        &lane_col,
                        shape,
                    );
                    for (idx, chunk) in chunks.into_iter().enumerate() {
                        let slot = idx * value_count + value_idx;
                        next[slot] = next[slot].clone() + program.bind(chunk);
                    }
                    program.workgroup_barrier();
                }
                next
            },
        );

        for (idx, values) in sums.chunks(value_count).enumerate() {
            let r = idx as u32 / tn;
            let c = idx as u32 % tn;
            let row = if single_row {
                Tile::u32(0)
            } else {
                row_base.clone() + r
            };
            let col = col_base.clone() + c;
            let extras = epilogues
                .post_extra_inputs
                .iter()
                .map(|extra| load_qmatmul_extra(program, extra, &row, &col, n))
                .collect::<Vec<_>>();
            let value = apply_qmatmul_post_epilogue_values(epilogues, values.to_vec(), extras);
            let value = stor_cast.from_accum(value);
            let mask = if single_row {
                tile_active.clone() & col.clone().lt(n)
            } else {
                tile_active.clone() & row.clone().lt(m) & col.clone().lt(n)
            };
            program.store(y.at((row, col)), value, mask);
        }
    });
}
