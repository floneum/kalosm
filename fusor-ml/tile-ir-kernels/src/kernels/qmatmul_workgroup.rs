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
//! Storage and staging element types travel as [`ScalarElement`] data through
//! [`AccumCast`]. The bodies share three building blocks:
//! - [`stage_storage_tile_with_pre`] — cooperative per-lane staging of a dense
//!   source into a workgroup tile, applying the optional pre-activation
//!   epilogue per element. Used for A in both kernels.
//! - [`TileBlock::fill_tile_quantized`] — per-lane dequantize-into-workgroup-
//!   tile for B. This path is not coop-forcing, preserving the lavapipe
//!   invariant that this kernel requests neither `SUBGROUP` nor
//!   `COOPERATIVE_MATRIX`.
//! - [`accumulate_register_tile_from_workgroup`] — per-lane register
//!   accumulation reading both staged tiles. Parameterized over the register
//!   tile shape (`tm`, `tn`), so the matmul body uses 4x4 and the gemv body
//!   uses 1x1.

use fusor_tile_ir::tile::{Mask, Program, Storage, Tile, TileBlock, WorkgroupTile};
use fusor_tile_ir::{QuantizedMatrix, ScalarElement, TileLiteral, WorkgroupAxis};

use crate::kernels::helpers::{dispatch_grid_1d, load_qmatmul_extra, scalar_of, AccumCast};
use crate::types::{
    apply_qmatmul_post_epilogue, apply_qmatmul_post_epilogue_values, apply_qmatmul_pre_epilogue,
    matrix_shape, QmatmulEpilogues,
};

const QMATMUL_LANES: u32 = 64;
const QGEMV_LANES: u32 = 64;
const QMATMUL_TM: u32 = 4;
const QMATMUL_TN: u32 = 4;
const QGEMV_TN: u32 = 1;

struct RegisterTileWorkgroups<'a> {
    a: &'a WorkgroupTile,
    b: &'a WorkgroupTile,
}

struct RegisterTileLane<'a> {
    row: &'a Tile,
    col: &'a Tile,
}

struct RegisterTileShape {
    bn: u32,
    bk: u32,
    tm: u32,
    tn: u32,
}

/// Stage `src` rows in `[row_base, row_base + rows)` and cols in
/// `[col_base, col_base + cols)` into the workgroup tile `dst`, applying
/// `pre` per element. Cooperative across all `lanes` workgroup lanes. Pads
/// out-of-bound source positions with zero, and guards the workgroup-tile
/// store so lanes with `flat >= rows * cols` don't write past the tile
/// (qgemv passes a 1xBK tile to a 64-lane workgroup; the unused lanes
/// would otherwise corrupt adjacent workgroup memory).
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
            Tile::literal(TileLiteral::f32(0.0)),
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
/// Layout: `A_tile` is row-major `BM x BK` (index = row*BK + k), `B_tile` is
/// row-major `BK x BN` (index = k*BN + col). `staging_cast` promotes the
/// staged-tile loads to the f32 accumulator.
fn accumulate_register_tile_from_workgroup(
    program: &mut TileBlock<'_>,
    tiles: RegisterTileWorkgroups<'_>,
    staging_cast: &AccumCast,
    lane: RegisterTileLane<'_>,
    shape: RegisterTileShape,
) -> Vec<Tile> {
    let RegisterTileShape { bn, bk, tm, tn } = shape;
    (0..tm * tn)
        .map(|idx| {
            let r = idx / tn;
            let c = idx % tn;
            let local_row = lane.row.clone() * tm + r;
            let local_col = lane.col.clone() * tn + c;
            let mut sum = Tile::literal(TileLiteral::f32(0.0));
            for kk in 0..bk {
                let a_value = staging_cast
                    .into_accum(program.load_workgroup(tiles.a, local_row.clone() * bk + kk));
                let b_value = staging_cast
                    .into_accum(program.load_workgroup(tiles.b, local_col.clone() + kk * bn));
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
    a: &Storage,
    b: &QuantizedMatrix,
    y: &Storage,
    epilogues: &QmatmulEpilogues<'_>,
    max_workgroups_per_dimension: u32,
) {
    qmatmul_workgroup_with_epilogues_impl(
        program,
        a,
        b,
        y,
        ScalarElement::F32,
        epilogues,
        max_workgroups_per_dimension,
    );
}

/// F16-staged variant of [`qmatmul_workgroup_with_epilogues`]. This requires
/// shader-f16 support but otherwise shares the f32 implementation and keeps
/// f32 accumulation/output.
pub fn qmatmul_workgroup_f16_with_epilogues(
    program: &mut Program,
    a: &Storage,
    b: &QuantizedMatrix,
    y: &Storage,
    epilogues: &QmatmulEpilogues<'_>,
    max_workgroups_per_dimension: u32,
) {
    qmatmul_workgroup_with_epilogues_impl(
        program,
        a,
        b,
        y,
        ScalarElement::F16,
        epilogues,
        max_workgroups_per_dimension,
    );
}

/// F16-storage and F16-staged variant of [`qmatmul_workgroup_with_epilogues`].
/// Accumulates in f32 and writes f16 outputs directly.
pub fn qmatmul_workgroup_storage_f16_with_epilogues(
    program: &mut Program,
    a: &Storage,
    b: &QuantizedMatrix,
    y: &Storage,
    epilogues: &QmatmulEpilogues<'_>,
    max_workgroups_per_dimension: u32,
) {
    qmatmul_workgroup_with_epilogues_impl(
        program,
        a,
        b,
        y,
        ScalarElement::F16,
        epilogues,
        max_workgroups_per_dimension,
    );
}

fn qmatmul_workgroup_with_epilogues_impl(
    program: &mut Program,
    a: &Storage,
    b: &QuantizedMatrix,
    y: &Storage,
    staging_element: ScalarElement,
    epilogues: &QmatmulEpilogues<'_>,
    max_workgroups_per_dimension: u32,
) {
    // BM/BN are pinned to the 4x4-register-tile geometry across QMATMUL_LANES.
    // BK is the K-axis staging chunk per pass.
    const BM: u32 = 32;
    const BN: u32 = 32;
    const BK: u32 = 8;
    let bk = BK;

    // F32 accumulation throughout; storage / staging elements are runtime data.
    let stor_scalar = scalar_of(a.element());
    let stor_cast = AccumCast::new(stor_scalar, ScalarElement::F32);
    let staging_cast = AccumCast::new(staging_element, ScalarElement::F32);

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
    let a_tile = program.alloc_workgroup_tile(staging_element, BM, bk);
    let b_tile = program.alloc_workgroup_tile(staging_element, bk, BN);
    let b_clone = b.clone();

    program.program_grid(QMATMUL_LANES, grid, |program| {
        let tile_id = program.program_id(WorkgroupAxis::X)
            + program.program_id(WorkgroupAxis::Y) * grid[0]
            + program.program_id(WorkgroupAxis::Z) * grid[0] * grid[1];
        let tile_active = tile_id.clone().lt(total_tiles);
        let m_tile = tile_id.clone() / tiles_n;
        let n_tile = tile_id % tiles_n;

        let lane = program.lane();
        let lane_row = lane.clone() / (BN / QMATMUL_TN);
        let lane_col = lane % (BN / QMATMUL_TN);
        let m_tile_base = m_tile * BM;
        let n_tile_base = n_tile * BN;
        let row_base = m_tile_base.clone() + lane_row.clone() * QMATMUL_TM;
        let col_base = n_tile_base.clone() + lane_col.clone() * QMATMUL_TN;

        let init: [Tile; (QMATMUL_TM * QMATMUL_TN) as usize] =
            std::array::from_fn(|_| Tile::literal(TileLiteral::f32(0.0)));
        let sums = program.fold(
            fusor_tile_ir::tile::range(k_tiles),
            init,
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
                    BM,
                    bk,
                    QMATMUL_LANES,
                );
                program.fill_tile_quantized(&b_tile, &b_clone, k_base, n_tile_base.clone());
                program.workgroup_barrier();

                let chunk_vec = accumulate_register_tile_from_workgroup(
                    program,
                    RegisterTileWorkgroups {
                        a: &a_tile,
                        b: &b_tile,
                    },
                    &staging_cast,
                    RegisterTileLane {
                        row: &lane_row,
                        col: &lane_col,
                    },
                    RegisterTileShape {
                        bn: BN,
                        bk,
                        tm: QMATMUL_TM,
                        tn: QMATMUL_TN,
                    },
                );
                let mut chunk_iter = chunk_vec.into_iter();
                let next: [Tile; (QMATMUL_TM * QMATMUL_TN) as usize] = std::array::from_fn(|idx| {
                    let chunk =
                        program.bind(chunk_iter.next().expect("register tile size matches"));
                    accs[idx].clone() + chunk
                });
                program.workgroup_barrier();
                next
            },
        );

        for (idx, sum) in sums.into_iter().enumerate() {
            let r = idx as u32 / QMATMUL_TN;
            let c = idx as u32 % QMATMUL_TN;
            let row = row_base.clone() + r;
            let col = col_base.clone() + c;
            let extras = epilogues
                .post_extra_inputs
                .iter()
                .map(|extra| load_qmatmul_extra(program, extra, &row, &col, n))
                .collect::<Vec<_>>();
            let value = apply_qmatmul_post_epilogue(epilogues, sum, extras);
            let value = stor_cast.from_accum(value);
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
/// [`accumulate_register_tile_from_workgroup`] with `tm = 1`, `tn = 1`.
pub fn qgemv_workgroup_with_epilogue(
    program: &mut Program,
    a: &Storage,
    b: &QuantizedMatrix,
    y: &Storage,
    epilogues: &QmatmulEpilogues<'_>,
    max_workgroups_per_dimension: u32,
) {
    qgemv_workgroup_with_epilogue_impl(
        program,
        a,
        b,
        y,
        ScalarElement::F32,
        epilogues,
        max_workgroups_per_dimension,
    );
}

/// F16-staged variant of [`qgemv_workgroup_with_epilogue`]. Requires
/// shader-f16 support; accumulation and output remain f32.
pub fn qgemv_workgroup_f16_with_epilogue(
    program: &mut Program,
    a: &Storage,
    b: &QuantizedMatrix,
    y: &Storage,
    epilogues: &QmatmulEpilogues<'_>,
    max_workgroups_per_dimension: u32,
) {
    qgemv_workgroup_with_epilogue_impl(
        program,
        a,
        b,
        y,
        ScalarElement::F16,
        epilogues,
        max_workgroups_per_dimension,
    );
}

/// F16-storage and F16-staged variant of [`qgemv_workgroup_with_epilogue`].
/// Accumulates in f32 and writes f16 outputs directly.
pub fn qgemv_workgroup_storage_f16_with_epilogue(
    program: &mut Program,
    a: &Storage,
    b: &QuantizedMatrix,
    y: &Storage,
    epilogues: &QmatmulEpilogues<'_>,
    max_workgroups_per_dimension: u32,
) {
    qgemv_workgroup_with_epilogue_impl(
        program,
        a,
        b,
        y,
        ScalarElement::F16,
        epilogues,
        max_workgroups_per_dimension,
    );
}

fn qgemv_workgroup_with_epilogue_impl(
    program: &mut Program,
    a: &Storage,
    b: &QuantizedMatrix,
    y: &Storage,
    staging_element: ScalarElement,
    epilogues: &QmatmulEpilogues<'_>,
    max_workgroups_per_dimension: u32,
) {
    // BN is pinned to QGEMV_LANES (one column per lane). BK is the K-axis
    // staging chunk per pass.
    const BN: u32 = QGEMV_LANES;
    const BK: u32 = 8;
    let bk = BK;

    let stor_scalar = scalar_of(a.element());
    let stor_cast = AccumCast::new(stor_scalar, ScalarElement::F32);
    let staging_cast = AccumCast::new(staging_element, ScalarElement::F32);

    let [m, k] = matrix_shape(&a.view().layout);
    let n = epilogues.post_output_cols(b.cols);
    assert_eq!(m, 1, "qgemv_workgroup expects a single input row");
    assert_eq!(k, b.rows, "qgemv K dimensions must match");
    let [y_m, y_n] = matrix_shape(&y.view().layout);
    assert_eq!(y_m, 1, "qgemv output must be single-row");
    assert_eq!(n, y_n, "qgemv output column count must match B");

    let tiles_n = n.div_ceil(BN);
    let k_tiles = k.div_ceil(bk);
    let grid = dispatch_grid_1d(tiles_n, max_workgroups_per_dimension);
    // BM=1 logical row tile. Reuse the stager with rows=1.
    let a_tile = program.alloc_workgroup_tile(staging_element, 1, bk);
    let b_tile = program.alloc_workgroup_tile(staging_element, bk, BN);
    let b_clone = b.clone();

    program.program_grid(QGEMV_LANES, grid, |program| {
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
        let col_base = n_tile_base.clone() + lane_col.clone() * QGEMV_TN;

        let post_accumulator_offsets = epilogues.post_accumulator_offsets().to_vec();
        let post_value_arity = post_accumulator_offsets.len();
        let sums = program.fold_vec(
            fusor_tile_ir::tile::range(k_tiles),
            vec![Tile::literal(TileLiteral::f32(0.0)); QGEMV_TN as usize * post_value_arity],
            |program, k_tile, accs| {
                let k_base = k_tile * bk;
                stage_storage_tile_with_pre(
                    program,
                    &a_tile,
                    a,
                    &stor_cast,
                    &staging_cast,
                    &row_base,
                    &k_base,
                    &tile_active,
                    1,
                    k,
                    epilogues,
                    1,
                    bk,
                    QGEMV_LANES,
                );
                let mut next = accs;
                for (value_idx, offset) in post_accumulator_offsets.iter().copied().enumerate() {
                    program.fill_tile_quantized(
                        &b_tile,
                        &b_clone,
                        k_base.clone(),
                        n_tile_base.clone() + offset,
                    );
                    program.workgroup_barrier();

                    let chunk_vec = accumulate_register_tile_from_workgroup(
                        program,
                        RegisterTileWorkgroups {
                            a: &a_tile,
                            b: &b_tile,
                        },
                        &staging_cast,
                        RegisterTileLane {
                            row: &lane_row,
                            col: &lane_col,
                        },
                        RegisterTileShape {
                            bn: BN,
                            bk,
                            tm: 1,
                            tn: QGEMV_TN,
                        },
                    );
                    for (idx, chunk) in chunk_vec.into_iter().enumerate() {
                        let accum_idx = idx * post_value_arity + value_idx;
                        next[accum_idx] = next[accum_idx].clone() + program.bind(chunk);
                    }
                    program.workgroup_barrier();
                }
                next
            },
        );

        for (idx, values) in sums.chunks(post_value_arity).enumerate() {
            let row = Tile::literal(TileLiteral::U32(0));
            let col = col_base.clone() + idx as u32;
            let extras = epilogues
                .post_extra_inputs
                .iter()
                .map(|extra| load_qmatmul_extra(program, extra, &row, &col, n))
                .collect::<Vec<_>>();
            let value = if values.len() == 1 {
                apply_qmatmul_post_epilogue(epilogues, values[0].clone(), extras)
            } else {
                apply_qmatmul_post_epilogue_values(epilogues, values.to_vec(), extras)
            };
            let value = stor_cast.from_accum(value);
            let mask = tile_active.clone().and(col.clone().lt(n));
            program.store(y.at((0u32, col)), value, mask);
        }
    });
}
