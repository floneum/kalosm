//! Dense matrix multiply program kernels.

use fusor_tile_ir::tile::{range, CoopAcc, Program, Storage, Tile, TileBlock};
use fusor_tile_ir::{ScalarElement, TileLiteral, WorkgroupAxis};

use crate::{
    grid::dot4_sum,
    kernels::helpers::{
        coop_load_a_fragments, coop_load_b_fragments, coop_mma_grid, coop_store_acc_grid,
        dispatch_grid_1d, scalar_of, zero_coop_acc_grid, AccumCast,
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

/// Workgroup tile geometry for the direct dense matmul kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DenseMatmulTile {
    pub bm: u32,
    pub bn: u32,
    pub bk: u32,
    pub tm: u32,
    pub tn: u32,
    pub lanes: u32,
}

impl DenseMatmulTile {
    pub const fn new(bm: u32, bn: u32, bk: u32, tm: u32, tn: u32, lanes: u32) -> Self {
        Self {
            bm,
            bn,
            bk,
            tm,
            tn,
            lanes,
        }
    }

    pub fn validate(self) {
        assert!(self.bm >= self.tm && self.bm.is_multiple_of(self.tm));
        assert!(self.bn >= self.tn && self.bn.is_multiple_of(self.tn));
        assert_eq!(
            self.lanes,
            (self.bm / self.tm) * (self.bn / self.tn),
            "dense matmul lanes must cover one thread tile per lane"
        );
    }
}

/// Direct storage bindings for dense matrix multiplication kernels.
///
/// Runtime-typed (ARBOR_DESIGN.md §2): the storage element travels in each
/// [`Storage`] view, so this bundle is no longer generic over a marker type.
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

#[derive(Clone, Copy)]
struct CoopTileEntry {
    tile: DenseCoopMatmulTile,
    row_groups: u32,
    col_groups: u32,
    n_passes: u32,
    block: u32,
    single_buffered: bool,
}

/// The accumulator element for every dense matmul kernel is F32. The storage
/// element (F32 or F16) travels in the [`Storage`] view; the runtime
/// [`AccumCast`] inserts the F16↔F32 cast pair on load/store and is the
/// identity for F32 storage — so the F32 path stays byte-identical to the
/// former F32-only body and the F16 path subsumes the former
/// `*_f16_accum_f32_*` variants.
fn accum_cast(storage: ScalarElement) -> AccumCast {
    AccumCast::new(storage, ScalarElement::F32)
}

/// Batched dense GEMV over flattened direct views:
/// A is `[batch * m, k]`, B is `[batch * k, 1]`, Y is `[batch * m, 1]`.
///
/// The storage element (F32 or F16) is recovered at runtime from the bound
/// [`Storage`] views; accumulation is in F32 via the [`AccumCast`], which
/// inserts the F16→F32 cast on load and F32→F16 cast on store. F32 storage has
/// identity casts and matches the original F32-only body bit-for-bit; F16
/// storage subsumes the former `batched_gemv_f16_accum_f32_with_epilogues`.
///
/// Each subgroup computes one output row. Lanes cooperatively walk K in
/// `VALUES_PER_LANE` chunks and then reduce the partial sums inside the
/// subgroup, avoiding the scalar-lane behavior of the generic edge matmul.
pub fn batched_gemv_with_epilogues(
    program: &mut Program,
    a: &Storage,
    b: &Storage,
    y: &Storage,
    shape: DenseMatmulShape,
    epilogues: &DenseMatmulEpilogues<'_>,
    max_workgroups_per_dimension: u32,
) {
    // Subgroup width × rows per workgroup = workgroup BLOCK (32 × 4 = 128).
    // Each lane folds VALUES_PER_LANE elements of K via dot4.
    const SUBGROUP_SIZE: u32 = 32;
    const ROWS_PER_WORKGROUP: u32 = 4;
    const VALUES_PER_LANE: u32 = 8;
    const BLOCK: u32 = ROWS_PER_WORKGROUP * SUBGROUP_SIZE;
    let rows_per_workgroup = ROWS_PER_WORKGROUP;
    let values_per_lane = VALUES_PER_LANE;
    assert_eq!(shape.n, 1, "batched_gemv expects a single RHS column");

    let cast = accum_cast(scalar_of(a.element()));

    let [a_rows, a_k] = matrix_shape(a.layout());
    let [b_rows, b_n] = matrix_shape(b.layout());
    let [y_rows, y_n] = matrix_shape(y.layout());
    assert_eq!(shape.batch * shape.m, a_rows);
    assert_eq!(shape.k, a_k);
    assert_eq!(shape.batch * shape.k, b_rows);
    assert_eq!(1, b_n);
    assert_eq!(shape.batch * shape.m, y_rows);
    assert_eq!(1, y_n);

    let row_groups = shape.m.div_ceil(rows_per_workgroup);
    let total_groups = shape.batch * row_groups;
    let grid = dispatch_grid_1d(total_groups, max_workgroups_per_dimension);
    let k_per_iter = SUBGROUP_SIZE * values_per_lane;
    let k_iterations = shape.k.div_ceil(k_per_iter);

    program.program_grid(BLOCK, grid, |program| {
        let group_id = program.program_id(WorkgroupAxis::X)
            + program.program_id(WorkgroupAxis::Y) * grid[0]
            + program.program_id(WorkgroupAxis::Z) * grid[0] * grid[1];
        let group_active = group_id.clone().lt(total_groups);
        let batch_tile = group_id.clone() / row_groups;
        let row_group = group_id % row_groups;
        let row = row_group * rows_per_workgroup + program.subgroup_id();
        let lane = program.subgroup_lane();
        let row_in_bounds = group_active.clone().and(row.clone().lt(shape.m));
        let a_batch_base = batch_tile.clone() * shape.m;
        let b_batch_base = batch_tile.clone() * shape.k;
        let y_batch_base = batch_tile * shape.m;

        let [sum] = program.fold(
            range(k_iterations),
            [Tile::literal(TileLiteral::f32(0.0))],
            |program, loop_index, [acc]| {
                let k_base = loop_index * k_per_iter + lane.clone() * values_per_lane;
                let a_values: Vec<Tile> = (0..values_per_lane)
                    .map(|i| {
                        let k_index = k_base.clone() + i;
                        let mask = row_in_bounds.clone().and(k_index.clone().lt(shape.k));
                        let loaded = program.load(
                            a.at((a_batch_base.clone() + row.clone(), k_index)),
                            mask.clone(),
                            cast.zero_storage(),
                        );
                        Tile::select(
                            mask,
                            apply_optional_epilogue(epilogues.pre_a, cast.into_accum(loaded)),
                            Tile::literal(TileLiteral::f32(0.0)),
                        )
                    })
                    .collect();
                let b_values: Vec<Tile> = (0..values_per_lane)
                    .map(|i| {
                        let k_index = k_base.clone() + i;
                        let mask = group_active.clone().and(k_index.clone().lt(shape.k));
                        let loaded = program.load(
                            b.at((b_batch_base.clone() + k_index, 0)),
                            mask.clone(),
                            cast.zero_storage(),
                        );
                        Tile::select(
                            mask,
                            apply_optional_epilogue(epilogues.pre_b, cast.into_accum(loaded)),
                            Tile::literal(TileLiteral::f32(0.0)),
                        )
                    })
                    .collect();
                [acc + dot4_sum(program, &a_values, &b_values)]
            },
        );
        let reduced = program.subgroup_reduce_sum(sum);
        let value = cast.from_accum(apply_optional_epilogue(epilogues.post, reduced));
        let mask = lane.eq(0).and(row_in_bounds);
        program.store(y.at((y_batch_base + row, 0)), value, mask);
    });
}

/// Batched dense matmul over flattened direct views. The storage element
/// (F32 or F16) is recovered at runtime from the bound [`Storage`] views;
/// accumulation is in F32 via the [`AccumCast`]. F32 storage matches the
/// original F32-only body; F16 storage subsumes the former
/// `batched_matmul_f16_accum_f32_with_epilogues`.
/// A is `[batch * m, k]`, B is `[batch * k, n]`, Y is `[batch * m, n]`.
pub fn batched_matmul_with_epilogues(
    program: &mut Program,
    a: &Storage,
    b: &Storage,
    y: &Storage,
    shape: DenseMatmulShape,
    epilogues: &DenseMatmulEpilogues<'_>,
    max_workgroups_per_dimension: u32,
    tile: DenseMatmulTile,
) {
    tile.validate();
    let DenseMatmulTile {
        bm,
        bn,
        bk,
        tm,
        tn,
        lanes,
    } = tile;
    let outs = (tm * tn) as usize;

    let scalar = scalar_of(a.element());
    let cast = accum_cast(scalar);

    let [a_rows, a_k] = matrix_shape(a.layout());
    let [b_rows, b_n] = matrix_shape(b.layout());
    let [y_rows, y_n] = matrix_shape(y.layout());
    assert_eq!(shape.batch * shape.m, a_rows);
    assert_eq!(shape.k, a_k);
    assert_eq!(shape.batch * shape.k, b_rows);
    assert_eq!(shape.n, b_n);
    assert_eq!(shape.batch * shape.m, y_rows);
    assert_eq!(shape.n, y_n);

    let tiles_m = shape.m.div_ceil(bm);
    let tiles_n = shape.n.div_ceil(bn);
    let total_tiles = shape.batch * tiles_m * tiles_n;
    let k_tiles = shape.k.div_ceil(bk);
    let grid = dispatch_grid_1d(total_tiles, max_workgroups_per_dimension);
    let a_tile = program.alloc_workgroup_tile(scalar, bm, bk);
    let b_tile = program.alloc_workgroup_tile(scalar, bk, bn);

    program.program_grid(lanes, grid, |program| {
        let tile_id = program.program_id(WorkgroupAxis::X)
            + program.program_id(WorkgroupAxis::Y) * grid[0]
            + program.program_id(WorkgroupAxis::Z) * grid[0] * grid[1];
        let tile_active = tile_id.clone().lt(total_tiles);
        let batch_tile = tile_id.clone() / (tiles_m * tiles_n);
        let local_tile = tile_id % (tiles_m * tiles_n);
        let m_tile = local_tile.clone() / tiles_n;
        let n_tile = local_tile % tiles_n;

        let lane = program.lane();
        let lane_row = lane.clone() / (bn / tn);
        let lane_col = lane % (bn / tn);
        let m_tile_base = m_tile * bm;
        let n_tile_base = n_tile * bn;
        let row_base = m_tile_base.clone() + lane_row.clone() * tm;
        let col_base = n_tile_base.clone() + lane_col.clone() * tn;
        let a_batch_base = batch_tile.clone() * shape.m;
        let b_batch_base = batch_tile.clone() * shape.k;
        let y_batch_base = batch_tile * shape.m;

        let sums = program.fold_vec(
            range(k_tiles),
            (0..outs)
                .map(|_| Tile::literal(TileLiteral::f32(0.0)))
                .collect(),
            |program, k_tile, accs| {
                let k_base = k_tile * bk;
                for pass in 0..(bm * bk).div_ceil(lanes) {
                    let flat = program.lane() + pass * lanes;
                    let local_row = flat.clone() / bk;
                    let local_k = flat.clone() % bk;
                    let global_row = m_tile_base.clone() + local_row.clone();
                    let global_k = k_base.clone() + local_k.clone();
                    let in_bounds = tile_active
                        .clone()
                        .and(flat.clone().lt(bm * bk))
                        .and(global_row.clone().lt(shape.m))
                        .and(global_k.clone().lt(shape.k));
                    let loaded = program.load(
                        a.at((a_batch_base.clone() + global_row, &global_k)),
                        in_bounds.clone(),
                        cast.zero_storage(),
                    );
                    let value = cast.from_accum(Tile::select(
                        in_bounds,
                        apply_optional_epilogue(epilogues.pre_a, cast.into_accum(loaded)),
                        Tile::literal(TileLiteral::f32(0.0)),
                    ));
                    program.store_workgroup(&a_tile, flat, value);
                }
                for pass in 0..(bk * bn).div_ceil(lanes) {
                    let flat = program.lane() + pass * lanes;
                    let local_k = flat.clone() / bn;
                    let local_col = flat.clone() % bn;
                    let global_k = k_base.clone() + local_k.clone();
                    let global_col = n_tile_base.clone() + local_col.clone();
                    let in_bounds = tile_active
                        .clone()
                        .and(flat.clone().lt(bk * bn))
                        .and(global_k.clone().lt(shape.k))
                        .and(global_col.clone().lt(shape.n));
                    let loaded = program.load(
                        b.at((b_batch_base.clone() + global_k, global_col)),
                        in_bounds.clone(),
                        cast.zero_storage(),
                    );
                    let value = cast.from_accum(Tile::select(
                        in_bounds,
                        apply_optional_epilogue(epilogues.pre_b, cast.into_accum(loaded)),
                        Tile::literal(TileLiteral::f32(0.0)),
                    ));
                    program.store_workgroup(&b_tile, flat, value);
                }
                program.workgroup_barrier();

                // Byte-identical to the original `loop_fold_n(Sum, …)` shape:
                // each chunk starts from a fresh `0.0` base (NOT the carried
                // accumulator), is bound to a local, and the carry-add wraps
                // the bound value as `acc + chunk` — exactly the `Add(LoadLocal
                // (acc), chunk)` the old fold framework emitted (ARBOR_DESIGN.md
                // §7: the new `fold` body returns the full update expression).
                let chunk_sums: Vec<_> = (0..outs as u32)
                    .map(|idx| {
                        let r = idx / tn;
                        let c = idx % tn;
                        let local_row = lane_row.clone() * tm + r;
                        let local_col = lane_col.clone() * tn + c;
                        let mut sum = Tile::literal(TileLiteral::f32(0.0));
                        for kk in 0..bk {
                            let a_value = cast.into_accum(
                                program.load_workgroup(&a_tile, local_row.clone() * bk + kk),
                            );
                            let b_value = cast.into_accum(
                                program.load_workgroup(&b_tile, local_col.clone() + kk * bn),
                            );
                            sum = sum + a_value * b_value;
                        }
                        sum
                    })
                    .collect();
                let chunk_sums: Vec<_> = chunk_sums
                    .into_iter()
                    .map(|sum| program.bind(sum))
                    .collect();
                program.workgroup_barrier();
                accs.into_iter()
                    .zip(chunk_sums)
                    .map(|(acc, chunk)| acc + chunk)
                    .collect()
            },
        );

        for (idx, sum) in sums.into_iter().enumerate() {
            let idx = idx as u32;
            let r = idx / tn;
            let c = idx % tn;
            let row = row_base.clone() + r;
            let col = col_base.clone() + c;
            let value = cast.from_accum(apply_optional_epilogue(epilogues.post, sum));
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
/// workgroup-tile corner cases. The storage element (F32 or F16) is recovered
/// at runtime from the bound [`Storage`] views with F32 accumulation; subsumes
/// the former `*_f16_accum_f32_register_*` variant.
pub fn batched_matmul_register_with_epilogues(
    program: &mut Program,
    a: &Storage,
    b: &Storage,
    y: &Storage,
    shape: DenseMatmulShape,
    epilogues: &DenseMatmulEpilogues<'_>,
    max_workgroups_per_dimension: u32,
) {
    // BM/BN are pinned to the register tile geometry (4x4 lanes × 8x8 = 32x32).
    const BM: u32 = 32;
    const BN: u32 = 32;
    const TM: u32 = 4;
    const TN: u32 = 4;
    const OUTS: usize = (TM * TN) as usize;
    const LANES: u32 = 64;

    let cast = accum_cast(scalar_of(a.element()));

    let tiles_m = shape.m.div_ceil(BM);
    let tiles_n = shape.n.div_ceil(BN);
    let total_tiles = shape.batch * tiles_m * tiles_n;
    let grid = dispatch_grid_1d(total_tiles, max_workgroups_per_dimension);

    program.program_grid(LANES, grid, |program| {
        let tile_id = program.program_id(WorkgroupAxis::X)
            + program.program_id(WorkgroupAxis::Y) * grid[0]
            + program.program_id(WorkgroupAxis::Z) * grid[0] * grid[1];
        let tile_active = tile_id.clone().lt(total_tiles);
        let batch_tile = tile_id.clone() / (tiles_m * tiles_n);
        let local_tile = tile_id % (tiles_m * tiles_n);
        let m_tile = local_tile.clone() / tiles_n;
        let n_tile = local_tile % tiles_n;

        let lane = program.lane();
        let lane_row = lane.clone() / (BN / TN);
        let lane_col = lane % (BN / TN);
        let row_base = m_tile * BM + lane_row * TM;
        let col_base = n_tile * BN + lane_col * TN;
        let a_batch_base = batch_tile.clone() * shape.m;
        let b_batch_base = batch_tile.clone() * shape.k;
        let y_batch_base = batch_tile * shape.m;

        let sums: [Tile; OUTS] = program.fold(
            range(shape.k),
            std::array::from_fn(|_| Tile::literal(TileLiteral::f32(0.0))),
            |program, k_index, accs| {
                let a_values: [Tile; TM as usize] = std::array::from_fn(|r| {
                    let row = row_base.clone() + r as u32;
                    let in_bounds = tile_active.clone().and(row.clone().lt(shape.m));
                    let loaded = program.load(
                        a.at((a_batch_base.clone() + row, &k_index)),
                        in_bounds.clone(),
                        cast.zero_storage(),
                    );
                    Tile::select(
                        in_bounds,
                        apply_optional_epilogue(epilogues.pre_a, cast.into_accum(loaded)),
                        Tile::literal(TileLiteral::f32(0.0)),
                    )
                });
                let b_values: [Tile; TN as usize] = std::array::from_fn(|c| {
                    let col = col_base.clone() + c as u32;
                    let in_bounds = tile_active.clone().and(col.clone().lt(shape.n));
                    let loaded = program.load(
                        b.at((b_batch_base.clone() + k_index.clone(), col)),
                        in_bounds.clone(),
                        cast.zero_storage(),
                    );
                    Tile::select(
                        in_bounds,
                        apply_optional_epilogue(epilogues.pre_b, cast.into_accum(loaded)),
                        Tile::literal(TileLiteral::f32(0.0)),
                    )
                });
                std::array::from_fn(|idx| {
                    let r = idx / TN as usize;
                    let c = idx % TN as usize;
                    accs[idx].clone() + a_values[r].clone() * b_values[c].clone()
                })
            },
        );

        for (idx, sum) in sums.into_iter().enumerate() {
            let r = idx / TN as usize;
            let c = idx % TN as usize;
            let row = row_base.clone() + r as u32;
            let col = col_base.clone() + c as u32;
            let value = cast.from_accum(apply_optional_epilogue(epilogues.post, sum));
            let mask = tile_active
                .clone()
                .and(row.clone().lt(shape.m))
                .and(col.clone().lt(shape.n));
            program.store(y.at((y_batch_base.clone() + row, col)), value, mask);
        }
    });
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
    tile: DenseCoopMatmulTile,
) -> bool {
    let DenseMatmulTensors { a, b, y } = tensors;
    let DenseCoopMatmulTile { bm, bn, bk } = tile;
    if epilogues.pre_a.is_some()
        || epilogues.pre_b.is_some()
        || epilogues.post.is_some()
        || !shape.m.is_multiple_of(bm)
        || !shape.n.is_multiple_of(bn)
        || !shape.k.is_multiple_of(bk)
        || !cooperative_store_layout_supported(y.layout())
    {
        return false;
    }
    let total_tiles = shape.batch * (shape.m / bm) * (shape.n / bn);
    if total_tiles > max_workgroups_per_dimension {
        return false;
    }

    // Tile geometry per supported (bm, bn, bk). bk=16 across the board keeps
    // the double-buffered workgroup tile footprint inside Apple's 32 KB
    // threadgroup-memory limit; with bk=32 the per-WG shared memory for the
    // bigger BM/BN variants overflows (e.g. Tile128x64 bk=32 double-buffer
    // = ~50 KB). The (256, 256, 16) entry runs single-buffered because the
    // 256×K A tile would exceed the limit when doubled; its single-buffer
    // overhead is amortized by halving global A reads vs (128, 512, 16).
    //
    // Schema: (bm, bn, bk, row_groups, col_groups, n_passes, block, single_buffered).
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
            block: 256,
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
            block: 256,
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
            block: 256,
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
            block: 512,
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
            block: 256,
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
            block: 256,
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
            block: 128,
            single_buffered: false,
        },
    ];
    let Some(entry) = COOP_TILE_TABLE.iter().find(|entry| entry.tile == tile) else {
        return false;
    };
    // Runtime block (ARBOR_DESIGN.md §5): the workgroup size is a value baked
    // by the lowerer, so the old `match block { 128 => ::<128>, … }` monomorph
    // dispatch collapses into a single runtime call.
    assert!(
        matches!(entry.block, 128 | 256 | 512),
        "unsupported coop matmul BLOCK {}",
        entry.block
    );
    if entry.single_buffered {
        batched_coop_matmul_perf_single(
            program,
            a,
            b,
            y,
            shape,
            max_workgroups_per_dimension,
            entry.block,
            bm,
            bn,
            bk,
            entry.row_groups,
            entry.col_groups,
            entry.n_passes,
        );
    } else {
        batched_coop_matmul_perf(
            program,
            a,
            b,
            y,
            shape,
            max_workgroups_per_dimension,
            entry.block,
            bm,
            bn,
            bk,
            entry.row_groups,
            entry.col_groups,
            entry.n_passes,
        );
    }
    true
}

/// Stage one `BK`-tile of A and B into `a_tile`/`b_tile`, barrier, then run the
/// `kk` MMA sweep into the accumulator grid. Folds the three structurally
/// identical staged-load→barrier→MMA bodies (single-buffer, K-pair half 0/1,
/// odd-K epilogue) into one (ARBOR_DESIGN.md §7). The caller decides the
/// trailing barrier — the K-pair shape elides it between halves.
#[allow(clippy::too_many_arguments)]
fn coop_stage_and_mma(
    program: &mut TileBlock<'_>,
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
    accs: &[Vec<CoopAcc>],
    tile_rows_per_sg: u32,
    tile_cols_per_sg: u32,
    bk: u32,
    coop_dim: u32,
    scalar: ScalarElement,
) {
    program.fill_tile(a_tile, a, a_batch_base.clone() + row_base.clone(), k_base);
    program.fill_tile(
        b_tile,
        b,
        b_batch_base.clone() + k_base.clone(),
        pass_col_base,
    );
    program.workgroup_barrier();

    let kk_steps = bk / coop_dim;
    for kk in 0..kk_steps {
        let a_frags =
            coop_load_a_fragments(program, a_tile, sg_row_base, kk, tile_rows_per_sg, scalar);
        let b_frags = coop_load_b_fragments(
            program,
            b_tile,
            sg_col_base_in_pass,
            kk,
            tile_cols_per_sg,
            scalar,
        );
        coop_mma_grid(program, accs, &a_frags, &b_frags);
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
        let accs = zero_coop_acc_grid(program, scalar, tile_rows_per_sg, tile_cols_per_sg);

        k_body(program, &pass_col_base, &accs);

        coop_store_acc_grid(
            program,
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
    bm: u32,
    bn: u32,
    bk: u32,
    row_groups: u32,
    col_groups: u32,
    n_passes: u32,
) {
    const COOP_DIM: u32 = 8;
    const SUBGROUP_SIZE: u32 = 32;
    debug_assert!(n_passes >= 1);
    debug_assert_eq!(bn % n_passes, 0);
    let bn_pass: u32 = bn / n_passes;
    let subgroup_rows: u32 = bm / row_groups;
    let subgroup_cols_per_pass: u32 = bn_pass / col_groups;
    debug_assert_eq!(bm % row_groups, 0);
    debug_assert_eq!(bn_pass % col_groups, 0);
    debug_assert_eq!(subgroup_rows % COOP_DIM, 0);
    debug_assert_eq!(subgroup_cols_per_pass % COOP_DIM, 0);
    debug_assert_eq!(row_groups * col_groups * SUBGROUP_SIZE, block);
    let tile_rows_per_sg: u32 = subgroup_rows / COOP_DIM;
    let tile_cols_per_sg: u32 = subgroup_cols_per_pass / COOP_DIM;

    let scalar = scalar_of(a.element());

    let tiles_m = shape.m / bm;
    let tiles_n = shape.n / bn;
    let total_tiles = shape.batch * tiles_m * tiles_n;
    let k_iterations = shape.k / bk;

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
        let y_batch_base = batch * shape.m;

        let subgroup_id = program.subgroup_id();
        let sg_row = subgroup_id.clone() / col_groups;
        let sg_col = subgroup_id % col_groups;
        let sg_row_base = sg_row * subgroup_rows;
        let sg_col_base_in_pass = sg_col * subgroup_cols_per_pass;

        coop_perf_pass_loop(
            program,
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
    bm: u32,
    bn: u32,
    bk: u32,
    row_groups: u32,
    col_groups: u32,
    n_passes: u32,
) {
    const COOP_DIM: u32 = 8;
    const SUBGROUP_SIZE: u32 = 32;
    debug_assert!(n_passes >= 1, "n_passes must be at least 1");
    debug_assert_eq!(bn % n_passes, 0, "bn must be divisible by n_passes");
    let bn_pass: u32 = bn / n_passes;
    let subgroup_rows: u32 = bm / row_groups;
    let subgroup_cols_per_pass: u32 = bn_pass / col_groups;
    debug_assert_eq!(bm % row_groups, 0);
    debug_assert_eq!(bn_pass % col_groups, 0);
    debug_assert_eq!(subgroup_rows % COOP_DIM, 0);
    debug_assert_eq!(subgroup_cols_per_pass % COOP_DIM, 0);
    debug_assert_eq!(row_groups * col_groups * SUBGROUP_SIZE, block);
    let tile_rows_per_sg: u32 = subgroup_rows / COOP_DIM;
    let tile_cols_per_sg: u32 = subgroup_cols_per_pass / COOP_DIM;

    let scalar = scalar_of(a.element());

    let tiles_m = shape.m / bm;
    let tiles_n = shape.n / bn;
    let total_tiles = shape.batch * tiles_m * tiles_n;
    let k_iterations = shape.k / bk;
    let k_pairs = k_iterations / 2;
    let k_remainder = k_iterations % 2;

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
        let y_batch_base = batch * shape.m;

        let subgroup_id = program.subgroup_id();
        let sg_row = subgroup_id.clone() / col_groups;
        let sg_col = subgroup_id % col_groups;
        let sg_row_base = sg_row * subgroup_rows;
        let sg_col_base_in_pass = sg_col * subgroup_cols_per_pass;

        coop_perf_pass_loop(
            program,
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
                            accs,
                            tile_rows_per_sg,
                            tile_cols_per_sg,
                            bk,
                            COOP_DIM,
                            scalar,
                        );

                        coop_stage_and_mma(
                            program,
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
