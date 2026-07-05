//! Dense matrix multiply program kernels.

use fusor_tile_ir::tile::{CoopAcc, Program, Storage, Tile, TileBlock};
use fusor_tile_ir::{CoopMatrixToken, ScalarElement, SubgroupToken, TileLiteral, WorkgroupAxis};

use crate::{
    dispatch::SubgroupConfig,
    kernels::helpers::zero_coop_acc_grid,
    kernels::helpers::{
        coop_load_a_fragments, coop_load_b_fragments, coop_mma_grid, coop_store_acc_grid,
        dispatch_grid_1d, scalar_of, zero_literal,
    },
    types::{cooperative_store_layout_supported, DenseMatmulEpilogues},
};

/// Logical shape for flattened batched dense matmul views.
#[derive(Clone, Copy, Debug)]
pub struct DenseMatmulShape {
    /// Number of independent matrices in the flattened batch prefix.
    pub batch: u32,
    /// Rows per lhs/output matrix.
    pub m: u32,
    /// Contracting dimension.
    pub k: u32,
    /// Columns per rhs/output matrix.
    pub n: u32,
}

/// Direct storage bindings for dense matrix multiplication kernels.
///
/// The storage element travels in each [`Storage`] view, so this bundle is not
/// generic over element type.
#[derive(Clone, Copy)]
pub struct DenseMatmulTensors<'a> {
    pub a: &'a Storage,
    pub b: &'a Storage,
    pub y: &'a Storage,
}

/// Cooperative-matrix tile geometry requested by the dense matmul dispatcher.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseCoopMatmulTile {
    pub bm: u32,
    pub bn: u32,
    pub bk: u32,
}

/// Capability and tile selection for a cooperative dense matmul attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseCoopMatmulConfig {
    pub coop: CoopMatrixToken,
    pub subgroups: SubgroupConfig,
    pub tile: DenseCoopMatmulTile,
}

#[derive(Clone, Copy)]
struct CoopTileEntry {
    tile: DenseCoopMatmulTile,
    row_groups: u32,
    col_groups: u32,
    n_passes: u32,
    single_buffered: bool,
}

impl CoopTileEntry {
    const fn block(self, subgroups: SubgroupConfig) -> u32 {
        subgroups.block_for_subgroups(self.row_groups * self.col_groups)
    }
}

/// Try to emit a fast cooperative-matrix batched matmul. Returns false
/// when shape/layout/epilogues require the generic path. The storage element
/// travels in the bound [`Storage`] views, so both F32 and F16 use the same
/// runtime dispatch table.
pub fn try_batched_coop_matmul(
    program: &mut Program,
    tensors: DenseMatmulTensors<'_>,
    shape: DenseMatmulShape,
    epilogues: &DenseMatmulEpilogues<'_>,
    max_workgroups_per_dimension: u32,
    config: DenseCoopMatmulConfig,
) -> bool {
    let DenseMatmulTensors { a, b, y } = tensors;
    let DenseCoopMatmulConfig {
        coop,
        subgroups,
        tile,
    } = config;
    let subgroup = subgroups.token();
    let DenseCoopMatmulTile { bm, bn, bk } = tile;
    // Shapes need not divide the tile geometry: edge tiles fill zero past
    // the logical extents, and the caller provides `y` with its rows padded
    // to `ceil(m / bm) * bm` per batch and its columns to `ceil(n / bn) * bn`
    // (the stores cover whole tiles; the pad region holds garbage the
    // logical view never reads).
    if !subgroups.is_fixed()
        || epilogues.pre_a.is_some()
        || epilogues.pre_b.is_some()
        || epilogues.post.is_some()
        || !cooperative_store_layout_supported(y.layout())
    {
        return false;
    }
    let total_tiles = shape.batch * shape.m.div_ceil(bm) * shape.n.div_ceil(bn);
    if total_tiles > max_workgroups_per_dimension {
        return false;
    }

    let Some(entry) = coop_tile_entry(tile) else {
        return false;
    };
    let block = entry.block(subgroups);
    if entry.single_buffered {
        batched_coop_matmul_perf_single(
            program,
            a,
            b,
            y,
            shape,
            max_workgroups_per_dimension,
            block,
            subgroup,
            coop,
            bm,
            bn,
            bk,
            entry.row_groups,
            entry.col_groups,
            entry.n_passes,
            subgroups,
        );
    } else {
        batched_coop_matmul_perf(
            program,
            a,
            b,
            y,
            shape,
            max_workgroups_per_dimension,
            block,
            subgroup,
            coop,
            bm,
            bn,
            bk,
            entry.row_groups,
            entry.col_groups,
            entry.n_passes,
            subgroups,
        );
    }
    true
}

/// Tile geometry per supported (bm, bn, bk). bk=16 across the board keeps
/// the double-buffered workgroup tile footprint inside Apple's 32 KB
/// threadgroup-memory limit; with bk=32 the per-WG shared memory for the
/// bigger BM/BN variants overflows (e.g. Tile128x64 bk=32 double-buffer
/// = ~50 KB). The (256, 256, 16) entry runs single-buffered because the
/// 256×K A tile would exceed the limit when doubled; its single-buffer
/// overhead is amortized by halving global A reads vs (128, 512, 16).
const COOP_TILE_TABLE: &[CoopTileEntry] = &[
    CoopTileEntry {
        tile: DenseCoopMatmulTile {
            bm: 256,
            bn: 256,
            bk: 16,
        },
        row_groups: 8,
        col_groups: 1,
        n_passes: 8,
        single_buffered: true,
    },
    CoopTileEntry {
        tile: DenseCoopMatmulTile {
            bm: 128,
            bn: 512,
            bk: 16,
        },
        row_groups: 4,
        col_groups: 2,
        n_passes: 8,
        single_buffered: false,
    },
    CoopTileEntry {
        tile: DenseCoopMatmulTile {
            bm: 128,
            bn: 256,
            bk: 16,
        },
        row_groups: 4,
        col_groups: 2,
        n_passes: 4,
        single_buffered: false,
    },
    CoopTileEntry {
        tile: DenseCoopMatmulTile {
            bm: 128,
            bn: 128,
            bk: 16,
        },
        row_groups: 4,
        col_groups: 4,
        n_passes: 2,
        single_buffered: false,
    },
    CoopTileEntry {
        tile: DenseCoopMatmulTile {
            bm: 128,
            bn: 64,
            bk: 16,
        },
        row_groups: 4,
        col_groups: 2,
        n_passes: 1,
        single_buffered: false,
    },
    CoopTileEntry {
        tile: DenseCoopMatmulTile {
            bm: 64,
            bn: 128,
            bk: 16,
        },
        row_groups: 2,
        col_groups: 4,
        n_passes: 2,
        single_buffered: false,
    },
    CoopTileEntry {
        tile: DenseCoopMatmulTile {
            bm: 64,
            bn: 64,
            bk: 16,
        },
        row_groups: 2,
        col_groups: 2,
        n_passes: 1,
        single_buffered: false,
    },
    // Small-side tiles for contractions with a 16-wide M or N side
    // (attention head_dim contractions, narrow-vocab heads). Fragment
    // sides stay multiples of COOP_DIM=8: 64×16 splits into 32×8
    // per-subgroup fragments, 16×64 into 8×32, and so on.
    CoopTileEntry {
        tile: DenseCoopMatmulTile {
            bm: 64,
            bn: 16,
            bk: 16,
        },
        row_groups: 2,
        col_groups: 2,
        n_passes: 1,
        single_buffered: false,
    },
    CoopTileEntry {
        tile: DenseCoopMatmulTile {
            bm: 16,
            bn: 64,
            bk: 16,
        },
        row_groups: 2,
        col_groups: 2,
        n_passes: 1,
        single_buffered: false,
    },
];

fn coop_tile_entry(tile: DenseCoopMatmulTile) -> Option<&'static CoopTileEntry> {
    COOP_TILE_TABLE.iter().find(|entry| entry.tile == tile)
}

/// Split-K partials for a starved cooperative-matrix tile grid: dispatch
/// `splits × total_tiles` workgroups, each running the coop K loop over one
/// contiguous span of `ceil(k_iterations / splits)` K-tiles and storing its
/// partial accumulator to `y` (the scratch buffer) at split-major rows —
/// row `(split · batch + b) · m_padded + m`. A combine kernel
/// ([`split_k_combine`]) then folds the `splits` partials into the real
/// output. Only the sum order changes versus the single-pass kernel.
///
/// Returns false when the tile geometry is unsupported (unknown or
/// single-buffered table entries) or the grid exceeds the dispatch limit.
pub fn try_batched_coop_matmul_split_k(
    program: &mut Program,
    tensors: DenseMatmulTensors<'_>,
    shape: DenseMatmulShape,
    splits: u32,
    max_workgroups_per_dimension: u32,
    config: DenseCoopMatmulConfig,
) -> bool {
    const COOP_DIM: u32 = 8;
    let DenseMatmulTensors { a, b, y } = tensors;
    let DenseCoopMatmulConfig {
        coop,
        subgroups,
        tile,
    } = config;
    let subgroup = subgroups.token();
    let DenseCoopMatmulTile { bm, bn, bk } = tile;
    if !subgroups.is_fixed() || splits < 2 || !cooperative_store_layout_supported(y.layout()) {
        return false;
    }
    let Some(entry) = coop_tile_entry(tile) else {
        return false;
    };
    // The split path targets tiny tile grids, which never select the
    // single-buffered (256, 256) geometry.
    if entry.single_buffered {
        return false;
    }
    let tiles_m = shape.m.div_ceil(bm);
    let tiles_n = shape.n.div_ceil(bn);
    let total_tiles = shape.batch * tiles_m * tiles_n;
    let Some(total_workgroups) = splits.checked_mul(total_tiles) else {
        return false;
    };
    if total_workgroups > max_workgroups_per_dimension {
        return false;
    }

    let block = entry.block(subgroups);
    let bn_pass: u32 = bn / entry.n_passes;
    let subgroup_rows: u32 = bm / entry.row_groups;
    let subgroup_cols_per_pass: u32 = bn_pass / entry.col_groups;
    let tile_rows_per_sg: u32 = subgroup_rows / COOP_DIM;
    let tile_cols_per_sg: u32 = subgroup_cols_per_pass / COOP_DIM;
    let scalar = scalar_of(a.element());

    let k_iterations = shape.k.div_ceil(bk);
    let span_iters = k_iterations.div_ceil(splits);
    let m_padded = tiles_m * bm;

    let a_tile = program.alloc_workgroup_tile_padded(scalar, bm, bk, 1);
    let b_tile = program.alloc_workgroup_tile_padded(scalar, bk, bn_pass, 1);

    let grid = dispatch_grid_1d(total_workgroups, max_workgroups_per_dimension);
    program.program_grid(block, grid, |program| {
        let wg_id = program.program_id(WorkgroupAxis::X)
            + program.program_id(WorkgroupAxis::Y) * grid[0]
            + program.program_id(WorkgroupAxis::Z) * grid[0] * grid[1];
        let split = program.bind(wg_id.clone() / total_tiles);
        let tile_id = wg_id % total_tiles;
        let batch = tile_id.clone() / (tiles_m * tiles_n);
        let local_tile = tile_id % (tiles_m * tiles_n);
        let m_tile = local_tile.clone() / tiles_n;
        let n_tile = local_tile % tiles_n;
        let row_base = m_tile * bm;
        let col_base = n_tile * bn;
        let a_batch_base = batch.clone() * shape.m;
        let b_batch_base = batch.clone() * shape.k;
        // Split-major scratch rows: slice `split` holds one full padded
        // [batch · m_padded, n_padded] partial.
        let y_batch_base = (split.clone() * shape.batch + batch) * m_padded;
        // The K bounds are always live: the last split's span may run past
        // the logical K extent, and those tiles must fill zero.
        let a_bounds: [Option<Tile>; 2] = [
            (!shape.m.is_multiple_of(bm)).then(|| a_batch_base.clone() + shape.m),
            Some(Tile::literal(TileLiteral::U32(shape.k))),
        ];
        let b_bounds: [Option<Tile>; 2] = [
            Some(b_batch_base.clone() + shape.k),
            (!shape.n.is_multiple_of(bn)).then(|| Tile::literal(TileLiteral::U32(shape.n))),
        ];

        let subgroup_id = subgroup.subgroup_id(program);
        let sg_row = subgroup_id.clone() / entry.col_groups;
        let sg_col = subgroup_id % entry.col_groups;
        let sg_row_base = sg_row * subgroup_rows;
        let sg_col_base_in_pass = sg_col * subgroup_cols_per_pass;

        let span_base = program.bind(split * span_iters);
        coop_perf_pass_loop(
            program,
            coop,
            scalar,
            entry.n_passes,
            bn_pass,
            tile_rows_per_sg,
            tile_cols_per_sg,
            y,
            &y_batch_base,
            &row_base,
            &col_base,
            &sg_row_base,
            &sg_col_base_in_pass,
            |program, pass_col_base, accs| {
                program.loop_range(span_iters, |program, iter_idx| {
                    let k_base = (span_base.clone() + iter_idx) * bk;
                    coop_stage_and_mma(
                        program,
                        coop,
                        a,
                        b,
                        &a_tile,
                        &b_tile,
                        &a_batch_base,
                        &b_batch_base,
                        &row_base,
                        pass_col_base,
                        &k_base,
                        &sg_row_base,
                        &sg_col_base_in_pass,
                        &a_bounds,
                        &b_bounds,
                        accs,
                        tile_rows_per_sg,
                        tile_cols_per_sg,
                        bk,
                        COOP_DIM,
                        scalar,
                    );
                    // Trailing barrier: the next iteration overwrites the
                    // tiles this one just read through the coop loads.
                    program.workgroup_barrier();
                });
            },
        );
    });
    true
}

/// Fold the split-K partials into the output. `y` is one read-write view of
/// the whole `(1 + splits)` -slice buffer — `[(1 + splits) · rows, cols]`
/// where rows `0..rows` are the real (padded) output and slice `s ∈
/// 1..=splits` holds one partial at rows `s · rows..`. One lane per output
/// element sums the partials and stores slice 0. A single binding keeps the
/// buffer from being bound with conflicting access modes.
pub fn split_k_combine(
    program: &mut Program,
    y: &Storage,
    rows: u32,
    cols: u32,
    splits: u32,
    max_workgroups_per_dimension: u32,
) {
    const BLOCK: u32 = 256;
    let total = rows * cols;
    let scalar = scalar_of(y.element());
    let zero = zero_literal(scalar);
    let grid = dispatch_grid_1d(total.div_ceil(BLOCK), max_workgroups_per_dimension);
    program.program_grid(BLOCK, grid, |program| {
        let wg_id = program.program_id(WorkgroupAxis::X)
            + program.program_id(WorkgroupAxis::Y) * grid[0]
            + program.program_id(WorkgroupAxis::Z) * grid[0] * grid[1];
        let index = program.bind(wg_id * BLOCK + program.lane());
        let active = index.clone().lt(total);
        let row = program.bind(index.clone() / cols);
        let col = program.bind(index % cols);
        let mut acc = program.load(
            y.at((row.clone() + rows, col.clone())),
            active.clone(),
            zero,
        );
        for split in 2..=splits {
            acc = acc
                + program.load(
                    y.at((row.clone() + split * rows, col.clone())),
                    active.clone(),
                    zero,
                );
        }
        program.store(y.at((row, col)), acc, active);
    });
}

/// Stage one `BK`-tile of A and B into `a_tile`/`b_tile`, barrier, then run the
/// `kk` MMA sweep into the accumulator grid. The caller decides the trailing
/// barrier; the K-pair shape elides it between halves.
#[allow(clippy::too_many_arguments)]
fn coop_stage_and_mma(
    program: &mut TileBlock<'_>,
    coop: CoopMatrixToken,
    a: &Storage,
    b: &Storage,
    a_tile: &fusor_tile_ir::tile::WorkgroupTile,
    b_tile: &fusor_tile_ir::tile::WorkgroupTile,
    a_batch_base: &Tile,
    b_batch_base: &Tile,
    row_base: &Tile,
    pass_col_base: &Tile,
    k_base: &Tile,
    sg_row_base: &Tile,
    sg_col_base_in_pass: &Tile,
    a_bounds: &[Option<Tile>; 2],
    b_bounds: &[Option<Tile>; 2],
    accs: &[Vec<CoopAcc>],
    tile_rows_per_sg: u32,
    tile_cols_per_sg: u32,
    bk: u32,
    coop_dim: u32,
    scalar: ScalarElement,
) {
    program.fill_tile_bounded(
        a_tile,
        a,
        a_batch_base.clone() + row_base.clone(),
        k_base,
        a_bounds.clone(),
    );
    program.fill_tile_bounded(
        b_tile,
        b,
        b_batch_base.clone() + k_base.clone(),
        pass_col_base,
        b_bounds.clone(),
    );
    program.workgroup_barrier();

    let kk_steps = bk / coop_dim;
    for kk in 0..kk_steps {
        let a_frags = coop_load_a_fragments(
            program,
            coop,
            a_tile,
            sg_row_base,
            kk,
            tile_rows_per_sg,
            scalar,
        );
        let b_frags = coop_load_b_fragments(
            program,
            coop,
            b_tile,
            sg_col_base_in_pass,
            kk,
            tile_cols_per_sg,
            scalar,
        );
        coop_mma_grid(program, coop, accs, &a_frags, &b_frags);
    }
}

/// Shared pass-loop scaffolding for the coop-perf matmul variants (single-
/// and double-buffered). For each of `N_PASSES` column sub-passes, allocates
/// a fresh accumulator grid, runs the caller-supplied K-loop body, then
/// cooperatively stores the result. Both variants only differ in the
/// per-pass K-buffering body, so they share this shell.
#[inline]
#[allow(clippy::too_many_arguments)]
fn coop_perf_pass_loop<F>(
    program: &mut TileBlock<'_>,
    coop: CoopMatrixToken,
    scalar: ScalarElement,
    n_passes: u32,
    bn_pass: u32,
    tile_rows_per_sg: u32,
    tile_cols_per_sg: u32,
    y: &Storage,
    y_batch_base: &Tile,
    row_base: &Tile,
    col_base: &Tile,
    sg_row_base: &Tile,
    sg_col_base_in_pass: &Tile,
    mut k_body: F,
) where
    F: FnMut(&mut TileBlock<'_>, &Tile, &[Vec<CoopAcc>]),
{
    for n_pass in 0..n_passes {
        let pass_col_base = col_base.clone() + n_pass * bn_pass;
        let accs = zero_coop_acc_grid(program, coop, scalar, tile_rows_per_sg, tile_cols_per_sg);

        k_body(program, &pass_col_base, &accs);

        coop_store_acc_grid(
            program,
            coop,
            &accs,
            y,
            Some(y_batch_base),
            row_base,
            &pass_col_base,
            sg_row_base,
            sg_col_base_in_pass,
        );
    }
}

/// Single-buffered cooperative-matrix batched matmul. Trades load/MMA
/// overlap for half the workgroup-memory footprint of
/// `batched_coop_matmul_perf` — useful when the doubled tile buffers would
/// pin the workgroup to 1-per-core occupancy on Apple Silicon (32 KB
/// threadgroup memory limit).
#[allow(clippy::too_many_arguments)]
fn batched_coop_matmul_perf_single(
    program: &mut Program,
    a: &Storage,
    b: &Storage,
    y: &Storage,
    shape: DenseMatmulShape,
    max_workgroups_per_dimension: u32,
    block: u32,
    subgroup: SubgroupToken,
    coop: CoopMatrixToken,
    bm: u32,
    bn: u32,
    bk: u32,
    row_groups: u32,
    col_groups: u32,
    n_passes: u32,
    subgroups: SubgroupConfig,
) {
    const COOP_DIM: u32 = 8;
    debug_assert!(n_passes >= 1);
    debug_assert_eq!(bn % n_passes, 0);
    let bn_pass: u32 = bn / n_passes;
    let subgroup_rows: u32 = bm / row_groups;
    let subgroup_cols_per_pass: u32 = bn_pass / col_groups;
    debug_assert_eq!(bm % row_groups, 0);
    debug_assert_eq!(bn_pass % col_groups, 0);
    debug_assert_eq!(subgroup_rows % COOP_DIM, 0);
    debug_assert_eq!(subgroup_cols_per_pass % COOP_DIM, 0);
    debug_assert_eq!(
        subgroups.block_for_subgroups(row_groups * col_groups),
        block
    );
    let tile_rows_per_sg: u32 = subgroup_rows / COOP_DIM;
    let tile_cols_per_sg: u32 = subgroup_cols_per_pass / COOP_DIM;

    let scalar = scalar_of(a.element());

    let tiles_m = shape.m.div_ceil(bm);
    let tiles_n = shape.n.div_ceil(bn);
    let total_tiles = shape.batch * tiles_m * tiles_n;
    let k_iterations = shape.k.div_ceil(bk);
    // The y rows are padded to whole tiles per batch (the caller allocates
    // the pad region); A/B are logical and edge tiles fill zero past the
    // extents.
    let m_padded = tiles_m * bm;

    let a_tile = program.alloc_workgroup_tile_padded(scalar, bm, bk, 1);
    let b_tile = program.alloc_workgroup_tile_padded(scalar, bk, bn_pass, 1);

    let grid = dispatch_grid_1d(total_tiles, max_workgroups_per_dimension);
    program.program_grid(block, grid, |program| {
        let tile_id = program.program_id(WorkgroupAxis::X)
            + program.program_id(WorkgroupAxis::Y) * grid[0]
            + program.program_id(WorkgroupAxis::Z) * grid[0] * grid[1];
        let batch = tile_id.clone() / (tiles_m * tiles_n);
        let local_tile = tile_id % (tiles_m * tiles_n);
        let m_tile = local_tile.clone() / tiles_n;
        let n_tile = local_tile % tiles_n;
        let row_base = m_tile * bm;
        let col_base = n_tile * bn;
        let a_batch_base = batch.clone() * shape.m;
        let b_batch_base = batch.clone() * shape.k;
        let y_batch_base = batch * m_padded;
        let a_bounds: [Option<Tile>; 2] = [
            (!shape.m.is_multiple_of(bm)).then(|| a_batch_base.clone() + shape.m),
            (!shape.k.is_multiple_of(bk)).then(|| Tile::literal(TileLiteral::U32(shape.k))),
        ];
        let b_bounds: [Option<Tile>; 2] = [
            (!shape.k.is_multiple_of(bk)).then(|| b_batch_base.clone() + shape.k),
            (!shape.n.is_multiple_of(bn)).then(|| Tile::literal(TileLiteral::U32(shape.n))),
        ];

        let subgroup_id = subgroup.subgroup_id(program);
        let sg_row = subgroup_id.clone() / col_groups;
        let sg_col = subgroup_id % col_groups;
        let sg_row_base = sg_row * subgroup_rows;
        let sg_col_base_in_pass = sg_col * subgroup_cols_per_pass;

        coop_perf_pass_loop(
            program,
            coop,
            scalar,
            n_passes,
            bn_pass,
            tile_rows_per_sg,
            tile_cols_per_sg,
            y,
            &y_batch_base,
            &row_base,
            &col_base,
            &sg_row_base,
            &sg_col_base_in_pass,
            |program, pass_col_base, accs| {
                program.loop_range(k_iterations, |program, iter_idx| {
                    let k_base = iter_idx * bk;
                    coop_stage_and_mma(
                        program,
                        coop,
                        a,
                        b,
                        &a_tile,
                        &b_tile,
                        &a_batch_base,
                        &b_batch_base,
                        &row_base,
                        pass_col_base,
                        &k_base,
                        &sg_row_base,
                        &sg_col_base_in_pass,
                        &a_bounds,
                        &b_bounds,
                        accs,
                        tile_rows_per_sg,
                        tile_cols_per_sg,
                        bk,
                        COOP_DIM,
                        scalar,
                    );
                    // Trailing barrier required: next iter overwrites the same
                    // tile that this iter just finished reading via coop loads.
                    program.workgroup_barrier();
                });
            },
        );
    });
}

/// Cooperative-matrix batched matmul.
///
/// Per-workgroup output tile is `BM × BN`. The N axis is split into
/// `N_PASSES` sub-passes of `BN/N_PASSES` columns each: a smaller B tile and
/// accumulator grid are reused across passes (matching the pattern in main's
/// `coop_gemm.rs`). Inside each pass the K loop is double-buffered with two
/// pairs of workgroup tiles, processing two `BK`-tiles per outer iteration
/// to amortize barriers; an odd `k_iterations` is closed out with a single
/// trailing tile. Workgroup tiles are allocated with one element of inner
/// padding to avoid Apple bank conflicts.
#[allow(clippy::too_many_arguments)]
fn batched_coop_matmul_perf(
    program: &mut Program,
    a: &Storage,
    b: &Storage,
    y: &Storage,
    shape: DenseMatmulShape,
    max_workgroups_per_dimension: u32,
    block: u32,
    subgroup: SubgroupToken,
    coop: CoopMatrixToken,
    bm: u32,
    bn: u32,
    bk: u32,
    row_groups: u32,
    col_groups: u32,
    n_passes: u32,
    subgroups: SubgroupConfig,
) {
    const COOP_DIM: u32 = 8;
    debug_assert!(n_passes >= 1, "n_passes must be at least 1");
    debug_assert_eq!(bn % n_passes, 0, "bn must be divisible by n_passes");
    let bn_pass: u32 = bn / n_passes;
    let subgroup_rows: u32 = bm / row_groups;
    let subgroup_cols_per_pass: u32 = bn_pass / col_groups;
    debug_assert_eq!(bm % row_groups, 0);
    debug_assert_eq!(bn_pass % col_groups, 0);
    debug_assert_eq!(subgroup_rows % COOP_DIM, 0);
    debug_assert_eq!(subgroup_cols_per_pass % COOP_DIM, 0);
    debug_assert_eq!(
        subgroups.block_for_subgroups(row_groups * col_groups),
        block
    );
    let tile_rows_per_sg: u32 = subgroup_rows / COOP_DIM;
    let tile_cols_per_sg: u32 = subgroup_cols_per_pass / COOP_DIM;

    let scalar = scalar_of(a.element());

    let tiles_m = shape.m.div_ceil(bm);
    let tiles_n = shape.n.div_ceil(bn);
    let total_tiles = shape.batch * tiles_m * tiles_n;
    let k_iterations = shape.k.div_ceil(bk);
    let k_pairs = k_iterations / 2;
    let k_remainder = k_iterations % 2;
    // The y rows are padded to whole tiles per batch (the caller allocates
    // the pad region); A/B are logical and edge tiles fill zero past the
    // extents.
    let m_padded = tiles_m * bm;

    // +1 inner padding on workgroup tiles avoids Apple shared-memory bank
    // conflicts on the inner stride (matches `stride_a = block_k + 1` in
    // `coop_gemm.rs`). Two A and two B tiles let the K loop issue both halves
    // of a K-pair before barriering.
    let a_tile_0 = program.alloc_workgroup_tile_padded(scalar, bm, bk, 1);
    let a_tile_1 = program.alloc_workgroup_tile_padded(scalar, bm, bk, 1);
    let b_tile_0 = program.alloc_workgroup_tile_padded(scalar, bk, bn_pass, 1);
    let b_tile_1 = program.alloc_workgroup_tile_padded(scalar, bk, bn_pass, 1);

    let grid = dispatch_grid_1d(total_tiles, max_workgroups_per_dimension);
    program.program_grid(block, grid, |program| {
        let tile_id = program.program_id(WorkgroupAxis::X)
            + program.program_id(WorkgroupAxis::Y) * grid[0]
            + program.program_id(WorkgroupAxis::Z) * grid[0] * grid[1];
        let batch = tile_id.clone() / (tiles_m * tiles_n);
        let local_tile = tile_id % (tiles_m * tiles_n);
        let m_tile = local_tile.clone() / tiles_n;
        let n_tile = local_tile % tiles_n;
        let row_base = m_tile * bm;
        let col_base = n_tile * bn;
        let a_batch_base = batch.clone() * shape.m;
        let b_batch_base = batch.clone() * shape.k;
        let y_batch_base = batch * m_padded;
        let a_bounds: [Option<Tile>; 2] = [
            (!shape.m.is_multiple_of(bm)).then(|| a_batch_base.clone() + shape.m),
            (!shape.k.is_multiple_of(bk)).then(|| Tile::literal(TileLiteral::U32(shape.k))),
        ];
        let b_bounds: [Option<Tile>; 2] = [
            (!shape.k.is_multiple_of(bk)).then(|| b_batch_base.clone() + shape.k),
            (!shape.n.is_multiple_of(bn)).then(|| Tile::literal(TileLiteral::U32(shape.n))),
        ];

        let subgroup_id = subgroup.subgroup_id(program);
        let sg_row = subgroup_id.clone() / col_groups;
        let sg_col = subgroup_id % col_groups;
        let sg_row_base = sg_row * subgroup_rows;
        let sg_col_base_in_pass = sg_col * subgroup_cols_per_pass;

        coop_perf_pass_loop(
            program,
            coop,
            scalar,
            n_passes,
            bn_pass,
            tile_rows_per_sg,
            tile_cols_per_sg,
            y,
            &y_batch_base,
            &row_base,
            &col_base,
            &sg_row_base,
            &sg_col_base_in_pass,
            |program, pass_col_base, accs| {
                if k_pairs > 0 {
                    program.loop_range(k_pairs, |program, pair_idx| {
                        let k_base_0 = pair_idx.clone() * (2 * bk);
                        let k_base_1 = pair_idx * (2 * bk) + bk;

                        // Two-barrier K-pair shape: the load into tile_1 happens
                        // *after* the MMA from tile_0 so the compiler can overlap
                        // the storage→workgroup copy with the running MMAs (they
                        // touch disjoint workgroup memory). The barrier-2 of the
                        // next iter gates this iter's MMA reads of tile_0/tile_1
                        // against the next iter's writes to the same tiles.
                        coop_stage_and_mma(
                            program,
                            coop,
                            a,
                            b,
                            &a_tile_0,
                            &b_tile_0,
                            &a_batch_base,
                            &b_batch_base,
                            &row_base,
                            pass_col_base,
                            &k_base_0,
                            &sg_row_base,
                            &sg_col_base_in_pass,
                            &a_bounds,
                            &b_bounds,
                            accs,
                            tile_rows_per_sg,
                            tile_cols_per_sg,
                            bk,
                            COOP_DIM,
                            scalar,
                        );

                        coop_stage_and_mma(
                            program,
                            coop,
                            a,
                            b,
                            &a_tile_1,
                            &b_tile_1,
                            &a_batch_base,
                            &b_batch_base,
                            &row_base,
                            pass_col_base,
                            &k_base_1,
                            &sg_row_base,
                            &sg_col_base_in_pass,
                            &a_bounds,
                            &b_bounds,
                            accs,
                            tile_rows_per_sg,
                            tile_cols_per_sg,
                            bk,
                            COOP_DIM,
                            scalar,
                        );
                        // No trailing barrier: next iter writes to tile_0 first
                        // (different from MMA-tile_1 reads above) — barrier-2 of
                        // the next iter (after its load_0) transitively gates
                        // any tile_1 races.
                    });
                }

                // Odd k_iterations: a single trailing tile after the pair loop.
                if k_remainder == 1 {
                    let k_base_epi = Tile::literal(TileLiteral::U32((k_iterations - 1) * bk));
                    coop_stage_and_mma(
                        program,
                        coop,
                        a,
                        b,
                        &a_tile_0,
                        &b_tile_0,
                        &a_batch_base,
                        &b_batch_base,
                        &row_base,
                        pass_col_base,
                        &k_base_epi,
                        &sg_row_base,
                        &sg_col_base_in_pass,
                        &a_bounds,
                        &b_bounds,
                        accs,
                        tile_rows_per_sg,
                        tile_cols_per_sg,
                        bk,
                        COOP_DIM,
                        scalar,
                    );
                    program.workgroup_barrier();
                }
            },
        );
    });
}
