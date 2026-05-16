//! Dense matrix multiply program kernels.

use fusor_tile_ir::tile::{CoopAcc, CoopFragment, CoopRole, Program, Storage, Tile, TileBlock};
use fusor_tile_ir::{
    CoopElement, FloatElement, Numeric, TileLiteral, TileReduceOp, TileRef, WorkgroupAxis, F32,
};

use crate::types::{
    apply_optional_epilogue, cooperative_store_layout_supported, matrix_shape, DenseMatmulEpilogues,
};

/// Dense F32 matmul, scalar lane-mapped body.
///
/// ```
/// use fusor_tile_ir::{tile, Shape, F32};
/// use fusor_tile_ir_kernels::matmul;
///
/// let ir = tile::build(|program| {
///     let a = program.storage_read::<F32, 2>(Shape::new([8, 64]));
///     let b = program.storage_read::<F32, 2>(Shape::new([64, 16]));
///     let y = program.storage_write::<F32, 2>(Shape::new([8, 16]));
///     matmul::<64>(program, &a, &b, &y);
/// });
/// # let _ = ir;
/// ```
pub fn matmul<const BK: usize>(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &Storage<F32, 2>,
    y: &Storage<F32, 2>,
) {
    matmul_with_epilogues::<BK>(program, a, b, y, &DenseMatmulEpilogues::empty());
}

/// Dense F32 matmul with optional pre/post element-wise epilogues.
pub fn matmul_with_epilogues<const BK: usize>(
    program: &mut Program,
    a: &Storage<F32, 2>,
    b: &Storage<F32, 2>,
    y: &Storage<F32, 2>,
    epilogues: &DenseMatmulEpilogues<'_>,
) {
    assert!(BK > 0, "matmul K tile shape must be non-zero");
    let [m, k] = matrix_shape(&a.view().layout);
    let [b_k, n] = matrix_shape(&b.view().layout);
    let [y_m, y_n] = matrix_shape(&y.view().layout);
    assert_eq!(k, b_k, "matmul K dimensions must match");
    assert_eq!(m, y_m, "matmul output row count must match A");
    assert_eq!(n, y_n, "matmul output column count must match B");

    batched_matmul_with_epilogues::<F32, 32, 32, BK>(
        program,
        a,
        b,
        y,
        DenseMatmulShape { batch: 1, m, k, n },
        TileLiteral::f32(0.0),
        epilogues,
    );
}

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

/// Batched dense matmul over flattened direct views:
/// A is `[batch * m, k]`, B is `[batch * k, n]`, Y is `[batch * m, n]`.
pub fn batched_matmul_with_epilogues<T, const BM: usize, const BN: usize, const BK: usize>(
    program: &mut Program,
    a: &Storage<T, 2>,
    b: &Storage<T, 2>,
    y: &Storage<T, 2>,
    shape: DenseMatmulShape,
    zero: TileLiteral,
    epilogues: &DenseMatmulEpilogues<'_>,
) where
    T: FloatElement + Numeric,
{
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
    let a_tile = program.alloc_workgroup_tile::<T>(BM as u32, BK as u32);
    let b_tile = program.alloc_workgroup_tile::<T>(BK as u32, BN as u32);

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

        let sums: [Tile; OUTS] = program.loop_fold_n::<OUTS, _>(
            TileReduceOp::Sum,
            k_tiles,
            [zero; OUTS],
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
                        zero,
                    );
                    let value = Tile::select(
                        Tile::from(in_bounds),
                        apply_optional_epilogue(epilogues.pre_a, loaded),
                        Tile::literal(zero),
                    );
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
                        zero,
                    );
                    let value = Tile::select(
                        Tile::from(in_bounds),
                        apply_optional_epilogue(epilogues.pre_b, loaded),
                        Tile::literal(zero),
                    );
                    program.store_workgroup(b_tile, flat, value);
                }
                program.workgroup_barrier();

                let chunk_sums: [Tile; OUTS] = std::array::from_fn(|idx| {
                    let r = idx / TN;
                    let c = idx % TN;
                    let local_row = lane_row.clone() * TM as u32 + r as u32;
                    let local_col = lane_col.clone() * TN as u32 + c as u32;
                    let mut sum = Tile::literal(zero);
                    for kk in 0..BK {
                        let a_value = program
                            .load_workgroup(a_tile, local_row.clone() * BK as u32 + kk as u32);
                        let b_value = program
                            .load_workgroup(b_tile, local_col.clone() + kk as u32 * BN as u32);
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
            let value = apply_optional_epilogue(epilogues.post, sum).cast(T::ELEMENT);
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
/// workgroup-tile corner cases.
pub fn batched_matmul_register_with_epilogues<
    T,
    const BM: usize,
    const BN: usize,
    const BK: usize,
>(
    program: &mut Program,
    a: &Storage<T, 2>,
    b: &Storage<T, 2>,
    y: &Storage<T, 2>,
    shape: DenseMatmulShape,
    zero: TileLiteral,
    epilogues: &DenseMatmulEpilogues<'_>,
) where
    T: FloatElement + Numeric,
{
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

        let sums: [Tile; OUTS] = program.loop_fold_n::<OUTS, _>(
            TileReduceOp::Sum,
            shape.k,
            [zero; OUTS],
            |program, k_index| {
                let a_values: [Tile; TM] = std::array::from_fn(|r| {
                    let row = row_base.clone() + r as u32;
                    let in_bounds = tile_active.clone().and(row.clone().lt(shape.m));
                    let loaded = program.load(
                        a.at((a_batch_base.clone() + row, &k_index)),
                        in_bounds.clone(),
                        zero,
                    );
                    Tile::select(
                        Tile::from(in_bounds),
                        apply_optional_epilogue(epilogues.pre_a, loaded),
                        Tile::literal(zero),
                    )
                });
                let b_values: [Tile; TN] = std::array::from_fn(|c| {
                    let col = col_base.clone() + c as u32;
                    let in_bounds = tile_active.clone().and(col.clone().lt(shape.n));
                    let loaded = program.load(
                        b.at((b_batch_base.clone() + k_index.clone(), col)),
                        in_bounds.clone(),
                        zero,
                    );
                    Tile::select(
                        Tile::from(in_bounds),
                        apply_optional_epilogue(epilogues.pre_b, loaded),
                        Tile::literal(zero),
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
            let value = apply_optional_epilogue(epilogues.post, sum).cast(T::ELEMENT);
            let mask = tile_active
                .clone()
                .and(row.clone().lt(shape.m))
                .and(col.clone().lt(shape.n));
            program.store(y.at((y_batch_base.clone() + row, col)), value, mask);
        }
    });
}

/// Batched F16 matmul that accumulates in F32 and writes F16 output.
pub fn batched_matmul_f16_accum_f32_with_epilogues<
    const BM: usize,
    const BN: usize,
    const BK: usize,
>(
    program: &mut Program,
    a: &Storage<fusor_tile_ir::F16, 2>,
    b: &Storage<fusor_tile_ir::F16, 2>,
    y: &Storage<fusor_tile_ir::F16, 2>,
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
    let k_tiles = shape.k.div_ceil(BK as u32);
    let grid = dispatch_grid_1d(total_tiles);
    let a_tile = program.alloc_workgroup_tile::<fusor_tile_ir::F16>(BM as u32, BK as u32);
    let b_tile = program.alloc_workgroup_tile::<fusor_tile_ir::F16>(BK as u32, BN as u32);
    let zero_f16 = TileLiteral::F16(0);

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

        let sums: [Tile; OUTS] = program.loop_fold_n::<OUTS, _>(
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
                        zero_f16,
                    );
                    let value = Tile::select(
                        Tile::from(in_bounds),
                        apply_optional_epilogue(epilogues.pre_a, loaded),
                        Tile::literal(zero_f16),
                    )
                    .cast(fusor_tile_ir::ElementType::F16);
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
                        zero_f16,
                    );
                    let value = Tile::select(
                        Tile::from(in_bounds),
                        apply_optional_epilogue(epilogues.pre_b, loaded),
                        Tile::literal(zero_f16),
                    )
                    .cast(fusor_tile_ir::ElementType::F16);
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
                        let a_value = program
                            .load_workgroup(a_tile, local_row.clone() * BK as u32 + kk as u32)
                            .cast(fusor_tile_ir::ElementType::F32);
                        let b_value = program
                            .load_workgroup(b_tile, local_col.clone() + kk as u32 * BN as u32)
                            .cast(fusor_tile_ir::ElementType::F32);
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
            let value =
                apply_optional_epilogue(epilogues.post, sum.cast(fusor_tile_ir::ElementType::F16))
                    .cast(fusor_tile_ir::ElementType::F16);
            let mask = tile_active
                .clone()
                .and(row.clone().lt(shape.m))
                .and(col.clone().lt(shape.n));
            program.store(y.at((y_batch_base.clone() + row, col)), value, mask);
        }
    });
}

pub fn batched_matmul_f16_accum_f32_register_with_epilogues<
    const BM: usize,
    const BN: usize,
    const BK: usize,
>(
    program: &mut Program,
    a: &Storage<fusor_tile_ir::F16, 2>,
    b: &Storage<fusor_tile_ir::F16, 2>,
    y: &Storage<fusor_tile_ir::F16, 2>,
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
    let zero_f16 = TileLiteral::F16(0);

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

        let sums: [Tile; OUTS] = program.loop_fold_n::<OUTS, _>(
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
                        zero_f16,
                    );
                    Tile::select(
                        Tile::from(in_bounds),
                        apply_optional_epilogue(epilogues.pre_a, loaded),
                        Tile::literal(zero_f16),
                    )
                    .cast(fusor_tile_ir::ElementType::F32)
                });
                let b_values: [Tile; TN] = std::array::from_fn(|c| {
                    let col = col_base.clone() + c as u32;
                    let in_bounds = tile_active.clone().and(col.clone().lt(shape.n));
                    let loaded = program.load(
                        b.at((b_batch_base.clone() + k_index.clone(), col)),
                        in_bounds.clone(),
                        zero_f16,
                    );
                    Tile::select(
                        Tile::from(in_bounds),
                        apply_optional_epilogue(epilogues.pre_b, loaded),
                        Tile::literal(zero_f16),
                    )
                    .cast(fusor_tile_ir::ElementType::F32)
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
            let value =
                apply_optional_epilogue(epilogues.post, sum.cast(fusor_tile_ir::ElementType::F16))
                    .cast(fusor_tile_ir::ElementType::F16);
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
        || BK != 32
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

    match (BM, BN) {
        (64, 64) => batched_coop_matmul_perf::<F32, 64, 64, 32, 2, 2, 128>(program, a, b, y, shape),
        (64, 128) => {
            batched_coop_matmul_perf::<F32, 64, 128, 32, 2, 4, 256>(program, a, b, y, shape)
        }
        (128, 64) => {
            batched_coop_matmul_perf::<F32, 128, 64, 32, 4, 2, 256>(program, a, b, y, shape)
        }
        (128, 128) => {
            batched_coop_matmul_perf::<F32, 128, 128, 32, 4, 4, 512>(program, a, b, y, shape)
        }
        _ => return false,
    }
    true
}

fn zero_coop_acc_grid<T: CoopElement>(
    program: &mut TileBlock<'_>,
    rows: u32,
    cols: u32,
) -> Vec<Vec<CoopAcc<T, 8, 8>>> {
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| {
                    let acc = program.alloc_coop_acc::<T, 8, 8>();
                    program.zero_coop_acc(&acc);
                    acc
                })
                .collect()
        })
        .collect()
}

fn coop_load_a_fragments<T: CoopElement>(
    program: &mut TileBlock<'_>,
    tile: TileRef,
    sg_row_base: &Tile,
    kk: u32,
    rows: u32,
) -> Vec<CoopFragment<T, 8, 8>> {
    const COOP_DIM: u32 = 8;
    (0..rows)
        .map(|r| {
            program.coop_load::<T, 8, 8>(
                CoopRole::A,
                program.coop_tile_load(tile, sg_row_base.clone() + r * COOP_DIM, kk * COOP_DIM),
            )
        })
        .collect()
}

fn coop_load_b_fragments<T: CoopElement>(
    program: &mut TileBlock<'_>,
    tile: TileRef,
    sg_col_base: &Tile,
    kk: u32,
    cols: u32,
) -> Vec<CoopFragment<T, 8, 8>> {
    const COOP_DIM: u32 = 8;
    (0..cols)
        .map(|c| {
            program.coop_load::<T, 8, 8>(
                CoopRole::B,
                program.coop_tile_load(tile, kk * COOP_DIM, sg_col_base.clone() + c * COOP_DIM),
            )
        })
        .collect()
}

fn coop_mma_grid<T: CoopElement>(
    program: &mut TileBlock<'_>,
    accs: &[Vec<CoopAcc<T, 8, 8>>],
    a_frags: &[CoopFragment<T, 8, 8>],
    b_frags: &[CoopFragment<T, 8, 8>],
) {
    for (r, a) in a_frags.iter().enumerate() {
        for (c, b) in b_frags.iter().enumerate() {
            program.coop_mma(&accs[r][c], a, b);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn coop_store_acc_grid<T: CoopElement>(
    program: &mut TileBlock<'_>,
    accs: &[Vec<CoopAcc<T, 8, 8>>],
    y: &Storage<T, 2>,
    y_batch_base: &Tile,
    row_base: &Tile,
    col_base: &Tile,
    sg_row_base: &Tile,
    sg_col_base: &Tile,
) {
    const COOP_DIM: u32 = 8;
    for (r, row_accs) in accs.iter().enumerate() {
        for (c, acc) in row_accs.iter().enumerate() {
            let row =
                y_batch_base.clone() + row_base.clone() + sg_row_base.clone() + r as u32 * COOP_DIM;
            let col = col_base.clone() + sg_col_base.clone() + c as u32 * COOP_DIM;
            program.coop_store(acc, y, row, col);
        }
    }
}

fn batched_coop_matmul_perf<
    T: CoopElement,
    const BM: usize,
    const BN: usize,
    const BK: usize,
    const ROW_GROUPS: u32,
    const COL_GROUPS: u32,
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
    const SUBGROUP_ROWS: u32 = 32;
    const SUBGROUP_COLS: u32 = 32;
    debug_assert_eq!(ROW_GROUPS * SUBGROUP_ROWS, BM as u32);
    debug_assert_eq!(COL_GROUPS * SUBGROUP_COLS, BN as u32);
    debug_assert_eq!(ROW_GROUPS * COL_GROUPS * SUBGROUP_SIZE, BLOCK as u32);

    let tiles_m = shape.m / BM as u32;
    let tiles_n = shape.n / BN as u32;
    let total_tiles = shape.batch * tiles_m * tiles_n;
    let k_iterations = shape.k / BK as u32;

    let a_tile = program.alloc_workgroup_tile::<T>(BM as u32, BK as u32);
    let b_tile = program.alloc_workgroup_tile::<T>(BK as u32, BN as u32);
    let a_clone = a;
    let b_clone = b;
    let y_clone = y;

    const TILE_ROWS_PER_SG: u32 = SUBGROUP_ROWS / 8;
    const TILE_COLS_PER_SG: u32 = SUBGROUP_COLS / 8;

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
        let sg_row_base = sg_row * SUBGROUP_ROWS;
        let sg_col_base = sg_col * SUBGROUP_COLS;
        let accs = zero_coop_acc_grid(program, TILE_ROWS_PER_SG, TILE_COLS_PER_SG);

        program.while_true(k_iterations, |program, loop_index| {
            let k_base = loop_index * BK as u32;
            program.copy_storage_to_tile(
                a_tile,
                a_clone,
                a_batch_base.clone() + row_base.clone(),
                &k_base,
            );
            program.copy_storage_to_tile(
                b_tile,
                b_clone,
                b_batch_base.clone() + k_base.clone(),
                &col_base,
            );
            program.workgroup_barrier();

            let kk_steps = (BK as u32) / COOP_DIM;
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
            &y_batch_base,
            &row_base,
            &col_base,
            &sg_row_base,
            &sg_col_base,
        );
    });
}
