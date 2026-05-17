//! Dense matrix multiply program kernels.

use fusor_tile_ir::tile::{Program, Storage, Tile};
use fusor_tile_ir::{CoopElement, TileLiteral, TileReduceOp, WorkgroupAxis, F32};

use crate::{
    grid::dot4_sum,
    kernels::helpers::{
        coop_load_a_fragments, coop_load_b_fragments, coop_mma_grid, coop_store_acc_grid,
        zero_coop_acc_grid, AccumCast,
    },
    types::{
        apply_optional_epilogue, cooperative_store_layout_supported, matrix_shape,
        DenseMatmulEpilogues,
    },
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

fn dispatch_grid_1d(total_workgroups: u32) -> [u32; 3] {
    assert!(total_workgroups > 0, "matmul dispatch must have workgroups");
    let x = total_workgroups.min(65_535);
    let y_needed = total_workgroups.div_ceil(x);
    let y = y_needed.min(65_535);
    let z = y_needed.div_ceil(y).max(1);
    [x, y, z]
}

/// Batched dense GEMV over flattened direct views:
/// A is `[batch * m, k]`, B is `[batch * k, 1]`, Y is `[batch * m, 1]`.
///
/// Generic over storage type `Stor` (F32 or F16); accumulates in F32 via the
/// `Stor: AccumCast<F32>` impl, which inserts the F16→F32 cast on load and
/// F32→F16 cast on store. The `<F32>` instantiation has identity casts and
/// matches the original F32-only body bit-for-bit; the `<F16>` instantiation
/// subsumes the former `batched_gemv_f16_accum_f32_with_epilogues`.
///
/// Each subgroup computes one output row. Lanes cooperatively walk K in
/// `VALUES_PER_LANE` chunks and then reduce the partial sums inside the
/// subgroup, avoiding the scalar-lane behavior of the generic edge matmul.
pub fn batched_gemv_with_epilogues<
    Stor: AccumCast<F32>,
    const ROWS_PER_WORKGROUP: usize,
    const VALUES_PER_LANE: usize,
    const BLOCK: usize,
>(
    program: &mut Program,
    a: &Storage<Stor, 2>,
    b: &Storage<Stor, 2>,
    y: &Storage<Stor, 2>,
    shape: DenseMatmulShape,
    epilogues: &DenseMatmulEpilogues<'_>,
) {
    const SUBGROUP_SIZE: u32 = 32;
    assert!(ROWS_PER_WORKGROUP > 0, "gemv rows must be non-zero");
    assert!(
        VALUES_PER_LANE > 0 && VALUES_PER_LANE.is_multiple_of(4),
        "gemv values per lane must be a non-zero multiple of dot4 width"
    );
    assert_eq!(
        ROWS_PER_WORKGROUP as u32 * SUBGROUP_SIZE,
        BLOCK as u32,
        "gemv maps one output row to each subgroup"
    );
    assert_eq!(shape.n, 1, "batched_gemv expects a single RHS column");

    let [a_rows, a_k] = matrix_shape(&a.view().layout);
    let [b_rows, b_n] = matrix_shape(&b.view().layout);
    let [y_rows, y_n] = matrix_shape(&y.view().layout);
    assert_eq!(shape.batch * shape.m, a_rows);
    assert_eq!(shape.k, a_k);
    assert_eq!(shape.batch * shape.k, b_rows);
    assert_eq!(1, b_n);
    assert_eq!(shape.batch * shape.m, y_rows);
    assert_eq!(1, y_n);

    let row_groups = shape.m.div_ceil(ROWS_PER_WORKGROUP as u32);
    let total_groups = shape.batch * row_groups;
    let grid = dispatch_grid_1d(total_groups);
    let k_per_iter = SUBGROUP_SIZE * VALUES_PER_LANE as u32;
    let k_iterations = shape.k.div_ceil(k_per_iter);

    program.program_grid::<BLOCK>(grid, |program| {
        let group_id = program.program_id(WorkgroupAxis::X)
            + program.program_id(WorkgroupAxis::Y) * grid[0]
            + program.program_id(WorkgroupAxis::Z) * grid[0] * grid[1];
        let group_active = group_id.clone().lt(total_groups);
        let batch_tile = group_id.clone() / row_groups;
        let row_group = group_id % row_groups;
        let row = row_group * ROWS_PER_WORKGROUP as u32 + program.subgroup_id();
        let lane = program.subgroup_lane();
        let row_in_bounds = group_active.clone().and(row.clone().lt(shape.m));
        let a_batch_base = batch_tile.clone() * shape.m;
        let b_batch_base = batch_tile.clone() * shape.k;
        let y_batch_base = batch_tile * shape.m;

        let [sum] = program.loop_fold_n::<1, _, _>(
            TileReduceOp::Sum,
            k_iterations,
            [TileLiteral::f32(0.0)],
            |program, loop_index| {
                let k_base = loop_index * k_per_iter + lane.clone() * VALUES_PER_LANE as u32;
                let a_values: [Tile; VALUES_PER_LANE] = std::array::from_fn(|i| {
                    let k_index = k_base.clone() + i as u32;
                    let mask = row_in_bounds.clone().and(k_index.clone().lt(shape.k));
                    let loaded = program.load(
                        a.at((a_batch_base.clone() + row.clone(), k_index)),
                        mask.clone(),
                        Stor::ZERO_STORAGE,
                    );
                    Tile::select(
                        Tile::from(mask),
                        apply_optional_epilogue(epilogues.pre_a, Stor::into_accum(loaded)),
                        Tile::literal(TileLiteral::f32(0.0)),
                    )
                });
                let b_values: [Tile; VALUES_PER_LANE] = std::array::from_fn(|i| {
                    let k_index = k_base.clone() + i as u32;
                    let mask = group_active.clone().and(k_index.clone().lt(shape.k));
                    let loaded = program.load(
                        b.at((b_batch_base.clone() + k_index, 0)),
                        mask.clone(),
                        Stor::ZERO_STORAGE,
                    );
                    Tile::select(
                        Tile::from(mask),
                        apply_optional_epilogue(epilogues.pre_b, Stor::into_accum(loaded)),
                        Tile::literal(TileLiteral::f32(0.0)),
                    )
                });
                [dot4_sum(program, &a_values, &b_values)]
            },
        );
        let reduced = program.subgroup_reduce_sum(sum);
        let value = Stor::from_accum(apply_optional_epilogue(epilogues.post, reduced));
        let mask = lane.eq(0).and(row_in_bounds);
        program.store(y.at((y_batch_base + row, 0)), value, mask);
    });
}

/// Batched dense matmul over flattened direct views, generic over storage
/// type `Stor`. Accumulates in F32 via `Stor: AccumCast<F32>`. The F32
/// instantiation matches the original F32-only body; the F16 instantiation
/// subsumes the former `batched_matmul_f16_accum_f32_with_epilogues`.
/// A is `[batch * m, k]`, B is `[batch * k, n]`, Y is `[batch * m, n]`.
pub fn batched_matmul_with_epilogues<
    Stor: AccumCast<F32>,
    const BM: usize,
    const BN: usize,
    const BK: usize,
>(
    program: &mut Program,
    a: &Storage<Stor, 2>,
    b: &Storage<Stor, 2>,
    y: &Storage<Stor, 2>,
    shape: DenseMatmulShape,
    epilogues: &DenseMatmulEpilogues<'_>,
) {
    const TM: usize = 4;
    const TN: usize = 4;
    const OUTS: usize = TM * TN;
    const LANES: usize = 64;
    assert!(
        BM > 0 && BN > 0 && BK > 0,
        "matmul tile shape must be non-zero"
    );
    assert_eq!(BM, 32, "register-tiled matmul currently expects BM=32");
    assert_eq!(BN, 32, "register-tiled matmul currently expects BN=32");

    let [a_rows, a_k] = matrix_shape(&a.view().layout);
    let [b_rows, b_n] = matrix_shape(&b.view().layout);
    let [y_rows, y_n] = matrix_shape(&y.view().layout);
    assert_eq!(shape.batch * shape.m, a_rows);
    assert_eq!(shape.k, a_k);
    assert_eq!(shape.batch * shape.k, b_rows);
    assert_eq!(shape.n, b_n);
    assert_eq!(shape.batch * shape.m, y_rows);
    assert_eq!(shape.n, y_n);

    let tiles_m = shape.m.div_ceil(BM as u32);
    let tiles_n = shape.n.div_ceil(BN as u32);
    let total_tiles = shape.batch * tiles_m * tiles_n;
    let k_tiles = shape.k.div_ceil(BK as u32);
    let grid = dispatch_grid_1d(total_tiles);
    let a_tile = program.alloc_workgroup_tile::<Stor>(BM as u32, BK as u32);
    let b_tile = program.alloc_workgroup_tile::<Stor>(BK as u32, BN as u32);

    program.program_grid::<LANES>(grid, |program| {
        let tile_id = program.program_id(WorkgroupAxis::X)
            + program.program_id(WorkgroupAxis::Y) * grid[0]
            + program.program_id(WorkgroupAxis::Z) * grid[0] * grid[1];
        let tile_active = tile_id.clone().lt(total_tiles);
        let batch_tile = tile_id.clone() / (tiles_m * tiles_n);
        let local_tile = tile_id % (tiles_m * tiles_n);
        let m_tile = local_tile.clone() / tiles_n;
        let n_tile = local_tile % tiles_n;

        let lane = program.lane();
        let lane_row = lane.clone() / (BN as u32 / TN as u32);
        let lane_col = lane % (BN as u32 / TN as u32);
        let row_base = m_tile * BM as u32 + lane_row.clone() * TM as u32;
        let col_base = n_tile * BN as u32 + lane_col.clone() * TN as u32;
        let a_batch_base = batch_tile.clone() * shape.m;
        let b_batch_base = batch_tile.clone() * shape.k;
        let y_batch_base = batch_tile * shape.m;

        let sums: [Tile; OUTS] = program.loop_fold_n::<OUTS, _, _>(
            TileReduceOp::Sum,
            k_tiles,
            [TileLiteral::f32(0.0); OUTS],
            |program, k_tile| {
                let k_base = k_tile * BK as u32;
                for pass in 0..(BM * BK).div_ceil(LANES) {
                    let flat = program.lane() + (pass * LANES) as u32;
                    let local_row = flat.clone() / BK as u32;
                    let local_k = flat.clone() % BK as u32;
                    let global_row = row_base.clone() + local_row.clone();
                    let global_k = k_base.clone() + local_k.clone();
                    let in_bounds = tile_active
                        .clone()
                        .and(flat.clone().lt((BM * BK) as u32))
                        .and(global_row.clone().lt(shape.m))
                        .and(global_k.clone().lt(shape.k));
                    let loaded = program.load(
                        a.at((a_batch_base.clone() + global_row, &global_k)),
                        in_bounds.clone(),
                        Stor::ZERO_STORAGE,
                    );
                    let value = Stor::from_accum(Tile::select(
                        Tile::from(in_bounds),
                        apply_optional_epilogue(epilogues.pre_a, Stor::into_accum(loaded)),
                        Tile::literal(TileLiteral::f32(0.0)),
                    ));
                    program.store_workgroup(a_tile, flat, value);
                }
                for pass in 0..(BK * BN).div_ceil(LANES) {
                    let flat = program.lane() + (pass * LANES) as u32;
                    let local_k = flat.clone() / BN as u32;
                    let local_col = flat.clone() % BN as u32;
                    let global_k = k_base.clone() + local_k.clone();
                    let global_col = col_base.clone() + local_col.clone();
                    let in_bounds = tile_active
                        .clone()
                        .and(flat.clone().lt((BK * BN) as u32))
                        .and(global_k.clone().lt(shape.k))
                        .and(global_col.clone().lt(shape.n));
                    let loaded = program.load(
                        b.at((b_batch_base.clone() + global_k, global_col)),
                        in_bounds.clone(),
                        Stor::ZERO_STORAGE,
                    );
                    let value = Stor::from_accum(Tile::select(
                        Tile::from(in_bounds),
                        apply_optional_epilogue(epilogues.pre_b, Stor::into_accum(loaded)),
                        Tile::literal(TileLiteral::f32(0.0)),
                    ));
                    program.store_workgroup(b_tile, flat, value);
                }
                program.workgroup_barrier();

                let chunk_sums: [Tile; OUTS] = std::array::from_fn(|idx| {
                    let r = idx / TN;
                    let c = idx % TN;
                    let local_row = lane_row.clone() * TM as u32 + r as u32;
                    let local_col = lane_col.clone() * TN as u32 + c as u32;
                    let mut sum = Tile::literal(TileLiteral::f32(0.0));
                    for kk in 0..BK {
                        let a_value = Stor::into_accum(
                            program.load_workgroup(
                                a_tile,
                                local_row.clone() * BK as u32 + kk as u32,
                            ),
                        );
                        let b_value = Stor::into_accum(
                            program.load_workgroup(
                                b_tile,
                                local_col.clone() + kk as u32 * BN as u32,
                            ),
                        );
                        sum = sum + a_value * b_value;
                    }
                    sum
                });
                let chunk_sums = chunk_sums.map(|sum| program.bind(sum));
                program.workgroup_barrier();
                chunk_sums
            },
        );

        for (idx, sum) in sums.into_iter().enumerate() {
            let r = idx / TN;
            let c = idx % TN;
            let row = row_base.clone() + r as u32;
            let col = col_base.clone() + c as u32;
            let value = Stor::from_accum(apply_optional_epilogue(epilogues.post, sum));
            let mask = tile_active
                .clone()
                .and(row.clone().lt(shape.m))
                .and(col.clone().lt(shape.n));
            program.store(y.at((y_batch_base.clone() + row, col)), value, mask);
        }
    });
}

/// Batched dense matmul fallback for partial tiles. This keeps the 4x4
/// register tile but reads directly from storage so skinny/edge shapes avoid
/// workgroup-tile corner cases. Generic over storage type `Stor` with F32
/// accumulation; subsumes the former `*_f16_accum_f32_register_*` variant.
pub fn batched_matmul_register_with_epilogues<
    Stor: AccumCast<F32>,
    const BM: usize,
    const BN: usize,
    const BK: usize,
>(
    program: &mut Program,
    a: &Storage<Stor, 2>,
    b: &Storage<Stor, 2>,
    y: &Storage<Stor, 2>,
    shape: DenseMatmulShape,
    epilogues: &DenseMatmulEpilogues<'_>,
) {
    const TM: usize = 4;
    const TN: usize = 4;
    const OUTS: usize = TM * TN;
    const LANES: usize = 64;
    assert_eq!(BM, 32, "register-tiled matmul currently expects BM=32");
    assert_eq!(BN, 32, "register-tiled matmul currently expects BN=32");

    let tiles_m = shape.m.div_ceil(BM as u32);
    let tiles_n = shape.n.div_ceil(BN as u32);
    let total_tiles = shape.batch * tiles_m * tiles_n;
    let grid = dispatch_grid_1d(total_tiles);

    program.program_grid::<LANES>(grid, |program| {
        let tile_id = program.program_id(WorkgroupAxis::X)
            + program.program_id(WorkgroupAxis::Y) * grid[0]
            + program.program_id(WorkgroupAxis::Z) * grid[0] * grid[1];
        let tile_active = tile_id.clone().lt(total_tiles);
        let batch_tile = tile_id.clone() / (tiles_m * tiles_n);
        let local_tile = tile_id % (tiles_m * tiles_n);
        let m_tile = local_tile.clone() / tiles_n;
        let n_tile = local_tile % tiles_n;

        let lane = program.lane();
        let lane_row = lane.clone() / (BN as u32 / TN as u32);
        let lane_col = lane % (BN as u32 / TN as u32);
        let row_base = m_tile * BM as u32 + lane_row * TM as u32;
        let col_base = n_tile * BN as u32 + lane_col * TN as u32;
        let a_batch_base = batch_tile.clone() * shape.m;
        let b_batch_base = batch_tile.clone() * shape.k;
        let y_batch_base = batch_tile * shape.m;

        let sums: [Tile; OUTS] = program.loop_fold_n::<OUTS, _, _>(
            TileReduceOp::Sum,
            shape.k,
            [TileLiteral::f32(0.0); OUTS],
            |program, k_index| {
                let a_values: [Tile; TM] = std::array::from_fn(|r| {
                    let row = row_base.clone() + r as u32;
                    let in_bounds = tile_active.clone().and(row.clone().lt(shape.m));
                    let loaded = program.load(
                        a.at((a_batch_base.clone() + row, &k_index)),
                        in_bounds.clone(),
                        Stor::ZERO_STORAGE,
                    );
                    Tile::select(
                        Tile::from(in_bounds),
                        apply_optional_epilogue(epilogues.pre_a, Stor::into_accum(loaded)),
                        Tile::literal(TileLiteral::f32(0.0)),
                    )
                });
                let b_values: [Tile; TN] = std::array::from_fn(|c| {
                    let col = col_base.clone() + c as u32;
                    let in_bounds = tile_active.clone().and(col.clone().lt(shape.n));
                    let loaded = program.load(
                        b.at((b_batch_base.clone() + k_index.clone(), col)),
                        in_bounds.clone(),
                        Stor::ZERO_STORAGE,
                    );
                    Tile::select(
                        Tile::from(in_bounds),
                        apply_optional_epilogue(epilogues.pre_b, Stor::into_accum(loaded)),
                        Tile::literal(TileLiteral::f32(0.0)),
                    )
                });
                std::array::from_fn(|idx| {
                    let r = idx / TN;
                    let c = idx % TN;
                    a_values[r].clone() * b_values[c].clone()
                })
            },
        );

        for (idx, sum) in sums.into_iter().enumerate() {
            let r = idx / TN;
            let c = idx % TN;
            let row = row_base.clone() + r as u32;
            let col = col_base.clone() + c as u32;
            let value = Stor::from_accum(apply_optional_epilogue(epilogues.post, sum));
            let mask = tile_active
                .clone()
                .and(row.clone().lt(shape.m))
                .and(col.clone().lt(shape.n));
            program.store(y.at((y_batch_base.clone() + row, col)), value, mask);
        }
    });
}


/// Try to emit a fast cooperative-matrix F32 batched matmul. Returns false
/// when shape/layout/epilogues require the generic path.
pub fn try_batched_coop_matmul_f32<const BM: usize, const BN: usize, const BK: usize>(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &Storage<F32, 2>,
    y: &Storage<F32, 2>,
    shape: DenseMatmulShape,
    epilogues: &DenseMatmulEpilogues<'_>,
) -> bool {
    if epilogues.pre_a.is_some()
        || epilogues.pre_b.is_some()
        || epilogues.post.is_some()
        || !shape.m.is_multiple_of(BM as u32)
        || !shape.n.is_multiple_of(BN as u32)
        || !shape.k.is_multiple_of(BK as u32)
        || !cooperative_store_layout_supported(&y.view().layout)
    {
        return false;
    }
    let total_tiles = shape.batch * (shape.m / BM as u32) * (shape.n / BN as u32);
    if total_tiles > 65_535 {
        return false;
    }

    // Variants are picked to match `coop_gemm.rs` on main: prefer the
    // (128x64) per-pass shape with N_PASSES=4 and BK=16 (effective 128x256
    // output coverage per workgroup) when N divides 256. Fall back to the
    // single-pass shapes for narrower N. ROW_GROUPS/COL_GROUPS are derived
    // from BLOCK/SUBGROUP_SIZE so each subgroup owns a 32x32 (or 32x
    // SUBGROUP_COLS_PER_PASS) MMA grid.
    // BK=16 across the board keeps the double-buffered workgroup tile
    // footprint inside Apple's 32 KB threadgroup-memory limit. With BK=32
    // the per-WG shared memory for the bigger BM/BN variants overflows
    // (e.g. Tile128x64 BK=32 double-buffer = ~50 KB), forcing slow paths.
    // The Tile256x256 variant uses single-buffered tiles because the 256×K
    // A tile would exceed the limit when doubled; its single-buffer overhead
    // is amortized by halving global A reads vs Tile128x512.
    match (BM, BN, BK) {
        (256, 256, 16) => batched_coop_matmul_perf_single::<F32, 256, 256, 16, 8, 1, 8, 256>(
            program, a, b, y, shape,
        ),
        (128, 512, 16) => batched_coop_matmul_perf::<F32, 128, 512, 16, 4, 2, 8, 256>(
            program, a, b, y, shape,
        ),
        (128, 256, 16) => batched_coop_matmul_perf::<F32, 128, 256, 16, 4, 2, 4, 256>(
            program, a, b, y, shape,
        ),
        (128, 128, 16) => {
            batched_coop_matmul_perf::<F32, 128, 128, 16, 4, 4, 2, 512>(program, a, b, y, shape)
        }
        (128, 64, 16) => {
            batched_coop_matmul_perf::<F32, 128, 64, 16, 4, 2, 1, 256>(program, a, b, y, shape)
        }
        (64, 128, 16) => {
            batched_coop_matmul_perf::<F32, 64, 128, 16, 2, 4, 2, 256>(program, a, b, y, shape)
        }
        (64, 64, 16) => {
            batched_coop_matmul_perf::<F32, 64, 64, 16, 2, 2, 1, 128>(program, a, b, y, shape)
        }
        _ => return false,
    }
    true
}

/// Single-buffered cooperative-matrix batched matmul. Trades load/MMA
/// overlap for half the workgroup-memory footprint of
/// `batched_coop_matmul_perf` — useful when the doubled tile buffers would
/// pin the workgroup to 1-per-core occupancy on Apple Silicon (32 KB
/// threadgroup memory limit).
fn batched_coop_matmul_perf_single<
    T: CoopElement,
    const BM: usize,
    const BN: usize,
    const BK: usize,
    const ROW_GROUPS: u32,
    const COL_GROUPS: u32,
    const N_PASSES: u32,
    const BLOCK: usize,
>(
    program: &mut Program,
    a: &Storage<T, 2>,
    b: &Storage<T, 2>,
    y: &Storage<T, 2>,
    shape: DenseMatmulShape,
) {
    const COOP_DIM: u32 = 8;
    const SUBGROUP_SIZE: u32 = 32;
    debug_assert!(N_PASSES >= 1);
    debug_assert_eq!(BN as u32 % N_PASSES, 0);
    let bn_pass: u32 = BN as u32 / N_PASSES;
    let subgroup_rows: u32 = BM as u32 / ROW_GROUPS;
    let subgroup_cols_per_pass: u32 = bn_pass / COL_GROUPS;
    debug_assert_eq!(BM as u32 % ROW_GROUPS, 0);
    debug_assert_eq!(bn_pass % COL_GROUPS, 0);
    debug_assert_eq!(subgroup_rows % COOP_DIM, 0);
    debug_assert_eq!(subgroup_cols_per_pass % COOP_DIM, 0);
    debug_assert_eq!(ROW_GROUPS * COL_GROUPS * SUBGROUP_SIZE, BLOCK as u32);
    let tile_rows_per_sg: u32 = subgroup_rows / COOP_DIM;
    let tile_cols_per_sg: u32 = subgroup_cols_per_pass / COOP_DIM;

    let tiles_m = shape.m / BM as u32;
    let tiles_n = shape.n / BN as u32;
    let total_tiles = shape.batch * tiles_m * tiles_n;
    let k_iterations = shape.k / BK as u32;

    let a_tile = program.alloc_workgroup_tile_padded::<T>(BM as u32, BK as u32, 1);
    let b_tile = program.alloc_workgroup_tile_padded::<T>(BK as u32, bn_pass, 1);

    let grid = dispatch_grid_1d(total_tiles);
    program.program_grid::<BLOCK>(grid, |program| {
        let tile_id = program.program_id(WorkgroupAxis::X)
            + program.program_id(WorkgroupAxis::Y) * grid[0]
            + program.program_id(WorkgroupAxis::Z) * grid[0] * grid[1];
        let batch = tile_id.clone() / (tiles_m * tiles_n);
        let local_tile = tile_id % (tiles_m * tiles_n);
        let m_tile = local_tile.clone() / tiles_n;
        let n_tile = local_tile % tiles_n;
        let row_base = m_tile * BM as u32;
        let col_base = n_tile * BN as u32;
        let a_batch_base = batch.clone() * shape.m;
        let b_batch_base = batch.clone() * shape.k;
        let y_batch_base = batch * shape.m;

        let subgroup_id = program.subgroup_id();
        let sg_row = subgroup_id.clone() / COL_GROUPS;
        let sg_col = subgroup_id % COL_GROUPS;
        let sg_row_base = sg_row * subgroup_rows;
        let sg_col_base_in_pass = sg_col * subgroup_cols_per_pass;

        for n_pass in 0..N_PASSES {
            let pass_col_base = col_base.clone() + n_pass * bn_pass;
            let accs = zero_coop_acc_grid(program, tile_rows_per_sg, tile_cols_per_sg);

            program.while_true(k_iterations, |program, iter_idx| {
                let k_base = iter_idx * BK as u32;
                program.copy_storage_to_tile(
                    a_tile,
                    a,
                    a_batch_base.clone() + row_base.clone(),
                    &k_base,
                );
                program.copy_storage_to_tile(
                    b_tile,
                    b,
                    b_batch_base.clone() + k_base.clone(),
                    &pass_col_base,
                );
                program.workgroup_barrier();

                let kk_steps = (BK as u32) / COOP_DIM;
                for kk in 0..kk_steps {
                    let a_frags = coop_load_a_fragments(
                        program,
                        a_tile,
                        &sg_row_base,
                        kk,
                        tile_rows_per_sg,
                    );
                    let b_frags = coop_load_b_fragments(
                        program,
                        b_tile,
                        &sg_col_base_in_pass,
                        kk,
                        tile_cols_per_sg,
                    );
                    coop_mma_grid(program, &accs, &a_frags, &b_frags);
                }
                // Trailing barrier required: next iter overwrites the same
                // tile that this iter just finished reading via coop loads.
                program.workgroup_barrier();
            });

            coop_store_acc_grid(
                program,
                &accs,
                y,
                Some(&y_batch_base),
                &row_base,
                &pass_col_base,
                &sg_row_base,
                &sg_col_base_in_pass,
            );
        }
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
fn batched_coop_matmul_perf<
    T: CoopElement,
    const BM: usize,
    const BN: usize,
    const BK: usize,
    const ROW_GROUPS: u32,
    const COL_GROUPS: u32,
    const N_PASSES: u32,
    const BLOCK: usize,
>(
    program: &mut Program,
    a: &Storage<T, 2>,
    b: &Storage<T, 2>,
    y: &Storage<T, 2>,
    shape: DenseMatmulShape,
) {
    const COOP_DIM: u32 = 8;
    const SUBGROUP_SIZE: u32 = 32;
    debug_assert!(N_PASSES >= 1, "N_PASSES must be at least 1");
    debug_assert_eq!(BN as u32 % N_PASSES, 0, "BN must be divisible by N_PASSES");
    let bn_pass: u32 = BN as u32 / N_PASSES;
    let subgroup_rows: u32 = BM as u32 / ROW_GROUPS;
    let subgroup_cols_per_pass: u32 = bn_pass / COL_GROUPS;
    debug_assert_eq!(BM as u32 % ROW_GROUPS, 0);
    debug_assert_eq!(bn_pass % COL_GROUPS, 0);
    debug_assert_eq!(subgroup_rows % COOP_DIM, 0);
    debug_assert_eq!(subgroup_cols_per_pass % COOP_DIM, 0);
    debug_assert_eq!(ROW_GROUPS * COL_GROUPS * SUBGROUP_SIZE, BLOCK as u32);
    let tile_rows_per_sg: u32 = subgroup_rows / COOP_DIM;
    let tile_cols_per_sg: u32 = subgroup_cols_per_pass / COOP_DIM;

    let tiles_m = shape.m / BM as u32;
    let tiles_n = shape.n / BN as u32;
    let total_tiles = shape.batch * tiles_m * tiles_n;
    let k_iterations = shape.k / BK as u32;
    let k_pairs = k_iterations / 2;
    let k_remainder = k_iterations % 2;

    // +1 inner padding on workgroup tiles avoids Apple shared-memory bank
    // conflicts on the inner stride (matches `stride_a = block_k + 1` in
    // `coop_gemm.rs`). Two A and two B tiles let the K loop issue both halves
    // of a K-pair before barriering.
    let a_tile_0 = program.alloc_workgroup_tile_padded::<T>(BM as u32, BK as u32, 1);
    let a_tile_1 = program.alloc_workgroup_tile_padded::<T>(BM as u32, BK as u32, 1);
    let b_tile_0 = program.alloc_workgroup_tile_padded::<T>(BK as u32, bn_pass, 1);
    let b_tile_1 = program.alloc_workgroup_tile_padded::<T>(BK as u32, bn_pass, 1);

    let grid = dispatch_grid_1d(total_tiles);
    program.program_grid::<BLOCK>(grid, |program| {
        let tile_id = program.program_id(WorkgroupAxis::X)
            + program.program_id(WorkgroupAxis::Y) * grid[0]
            + program.program_id(WorkgroupAxis::Z) * grid[0] * grid[1];
        let batch = tile_id.clone() / (tiles_m * tiles_n);
        let local_tile = tile_id % (tiles_m * tiles_n);
        let m_tile = local_tile.clone() / tiles_n;
        let n_tile = local_tile % tiles_n;
        let row_base = m_tile * BM as u32;
        let col_base = n_tile * BN as u32;
        let a_batch_base = batch.clone() * shape.m;
        let b_batch_base = batch.clone() * shape.k;
        let y_batch_base = batch * shape.m;

        let subgroup_id = program.subgroup_id();
        let sg_row = subgroup_id.clone() / COL_GROUPS;
        let sg_col = subgroup_id % COL_GROUPS;
        let sg_row_base = sg_row * subgroup_rows;
        let sg_col_base_in_pass = sg_col * subgroup_cols_per_pass;

        for n_pass in 0..N_PASSES {
            let pass_col_base = col_base.clone() + n_pass * bn_pass;
            let accs = zero_coop_acc_grid(program, tile_rows_per_sg, tile_cols_per_sg);

            if k_pairs > 0 {
                program.while_true(k_pairs, |program, pair_idx| {
                    let k_base_0 = pair_idx.clone() * (2 * BK as u32);
                    let k_base_1 = pair_idx * (2 * BK as u32) + BK as u32;
                    let kk_steps = (BK as u32) / COOP_DIM;

                    // Two-barrier K-pair shape: the load into tile_1 happens
                    // *after* the MMA from tile_0 so the compiler can overlap
                    // the storage→workgroup copy with the running MMAs (they
                    // touch disjoint workgroup memory). The barrier-2 of the
                    // next iter gates this iter's MMA reads of tile_0/tile_1
                    // against the next iter's writes to the same tiles.
                    program.copy_storage_to_tile(
                        a_tile_0,
                        a,
                        a_batch_base.clone() + row_base.clone(),
                        &k_base_0,
                    );
                    program.copy_storage_to_tile(
                        b_tile_0,
                        b,
                        b_batch_base.clone() + k_base_0.clone(),
                        &pass_col_base,
                    );
                    program.workgroup_barrier();

                    for kk in 0..kk_steps {
                        let a_frags = coop_load_a_fragments(
                            program,
                            a_tile_0,
                            &sg_row_base,
                            kk,
                            tile_rows_per_sg,
                        );
                        let b_frags = coop_load_b_fragments(
                            program,
                            b_tile_0,
                            &sg_col_base_in_pass,
                            kk,
                            tile_cols_per_sg,
                        );
                        coop_mma_grid(program, &accs, &a_frags, &b_frags);
                    }

                    program.copy_storage_to_tile(
                        a_tile_1,
                        a,
                        a_batch_base.clone() + row_base.clone(),
                        &k_base_1,
                    );
                    program.copy_storage_to_tile(
                        b_tile_1,
                        b,
                        b_batch_base.clone() + k_base_1.clone(),
                        &pass_col_base,
                    );
                    program.workgroup_barrier();

                    for kk in 0..kk_steps {
                        let a_frags = coop_load_a_fragments(
                            program,
                            a_tile_1,
                            &sg_row_base,
                            kk,
                            tile_rows_per_sg,
                        );
                        let b_frags = coop_load_b_fragments(
                            program,
                            b_tile_1,
                            &sg_col_base_in_pass,
                            kk,
                            tile_cols_per_sg,
                        );
                        coop_mma_grid(program, &accs, &a_frags, &b_frags);
                    }
                    // No trailing barrier: next iter writes to tile_0 first
                    // (different from MMA-tile_1 reads above) — barrier-2 of
                    // the next iter (after its load_0) transitively gates
                    // any tile_1 races.
                });
            }

            // Odd k_iterations: a single trailing tile after the pair loop.
            if k_remainder == 1 {
                let k_base_epi =
                    Tile::literal(TileLiteral::U32((k_iterations - 1) * BK as u32));
                program.copy_storage_to_tile(
                    a_tile_0,
                    a,
                    a_batch_base.clone() + row_base.clone(),
                    &k_base_epi,
                );
                program.copy_storage_to_tile(
                    b_tile_0,
                    b,
                    b_batch_base.clone() + k_base_epi.clone(),
                    &pass_col_base,
                );
                program.workgroup_barrier();

                let kk_steps = (BK as u32) / COOP_DIM;
                for kk in 0..kk_steps {
                    let a_frags = coop_load_a_fragments(
                        program,
                        a_tile_0,
                        &sg_row_base,
                        kk,
                        tile_rows_per_sg,
                    );
                    let b_frags = coop_load_b_fragments(
                        program,
                        b_tile_0,
                        &sg_col_base_in_pass,
                        kk,
                        tile_cols_per_sg,
                    );
                    coop_mma_grid(program, &accs, &a_frags, &b_frags);
                }
                program.workgroup_barrier();
            }

            coop_store_acc_grid(
                program,
                &accs,
                y,
                Some(&y_batch_base),
                &row_base,
                &pass_col_base,
                &sg_row_base,
                &sg_col_base_in_pass,
            );
        }
    });
}
