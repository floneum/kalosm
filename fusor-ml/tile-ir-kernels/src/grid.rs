//! qgemv-specific grid helpers — moved out of `fusor-tile-ir`'s `tile/grid.rs`
//! when the kernels split landed. `tile-ir`'s grid.rs still owns the generic
//! `tile::build` entry point.

use fusor_tile_ir::tile::{Mask, Q4KActivations, Storage, Tile, TileBlock};
use fusor_tile_ir::{TileLiteral, WorkgroupAxis, F32, U32};

#[derive(Clone, Copy)]
pub(crate) struct QgemvGrid {
    pub(crate) cols_per_workgroup: u32,
    pub(crate) workgroups_x: u32,
    pub(crate) dispatch_y: u32,
    pub(crate) n_cols: u32,
    pub(crate) full_cols: bool,
}

pub(crate) fn qgemv_grid<const SUBGROUPS: u32, const COLS_PER_SUBGROUP: usize>(
    n_cols: u32,
    requested_workgroups_x: u32,
) -> QgemvGrid {
    let cols_per_workgroup = SUBGROUPS * COLS_PER_SUBGROUP as u32;
    let total_workgroups = n_cols.div_ceil(cols_per_workgroup);
    let workgroups_x = requested_workgroups_x.min(total_workgroups.max(1));
    QgemvGrid {
        cols_per_workgroup,
        workgroups_x,
        dispatch_y: total_workgroups.div_ceil(workgroups_x),
        n_cols,
        full_cols: n_cols.is_multiple_of(cols_per_workgroup),
    }
}

impl QgemvGrid {
    pub(crate) fn mask(self, full_iterations: bool, in_bounds: Mask, col: &Tile<U32>) -> Mask {
        match (full_iterations, self.full_cols) {
            (true, true) => Mask::all(),
            (true, false) => col.lt(self.n_cols),
            (false, true) => in_bounds,
            (false, false) => in_bounds.and(col.lt(self.n_cols)),
        }
    }
}

#[derive(Clone)]
pub(crate) struct QgemvProgramScope {
    pub(crate) col0: Tile<U32>,
    pub(crate) lane: Tile<U32>,
}

pub(crate) struct QgemvStoreTarget<'a> {
    pub(crate) y: &'a Storage<F32, 2>,
    pub(crate) col0: Tile<U32>,
    pub(crate) lane: Tile<U32>,
    pub(crate) full_cols: bool,
    pub(crate) n_cols: u32,
    pub(crate) epilogues: &'a crate::types::QmatmulEpilogues<'a>,
}

pub(crate) fn qgemv_program_scope<const COLS_PER_SUBGROUP: usize>(
    program: &TileBlock<'_>,
    grid: QgemvGrid,
) -> QgemvProgramScope {
    let workgroup = program.program_id(WorkgroupAxis::X)
        + program.program_id(WorkgroupAxis::Y) * grid.workgroups_x;
    let col_group_base = workgroup * grid.cols_per_workgroup;
    let subgroup_col_base = program.subgroup_id() * COLS_PER_SUBGROUP as u32;
    QgemvProgramScope {
        col0: col_group_base + subgroup_col_base,
        lane: program.subgroup_lane(),
    }
}

/// Store subgroup-reduced qgemv sums, applying an optional post-reduce
/// epilogue between the subgroup reduce and the store. The `pre` slot is
/// ignored here because pre-epilogues are applied at load sites by the kernel
/// body.
pub(crate) fn store_qgemv_sums_with_epilogue<const COLS_PER_SUBGROUP: usize>(
    program: &mut TileBlock<'_>,
    sums: [Tile; COLS_PER_SUBGROUP],
    target: QgemvStoreTarget<'_>,
) {
    for (offset, sum) in sums.into_iter().enumerate() {
        let col = target.col0.clone() + offset as u32;
        let reduced = program.subgroup_reduce_sum(sum);
        let value = crate::types::apply_optional_epilogue(target.epilogues.post, reduced);
        let mut mask = target.lane.eq(0);
        if !target.full_cols {
            mask = mask.and(col.lt(target.n_cols));
        }
        program.store(target.y.at((0, col)), value, mask);
    }
}

pub(crate) fn q4k_ggml_activations(
    program: &mut TileBlock<'_>,
    a: &Storage<F32, 2>,
    row: impl Clone + Into<Tile<U32>>,
    vector_base: &Tile<U32>,
    in_bounds: Mask,
) -> Q4KActivations {
    let load_quad = |program: &mut TileBlock<'_>, base: u32| -> [Tile; 16] {
        std::array::from_fn(|j| {
            let offset = if j < 8 { j as u32 } else { (j - 8) as u32 + 32 } + base;
            let scalar = program.load(
                a.at((row.clone(), vector_base.clone() + offset)),
                in_bounds.clone(),
                0.0,
            );
            program.bind(scalar)
        })
    };
    let low = load_quad(program, 0);
    let high = load_quad(program, 128);

    let zero = TileLiteral::f32(0.0);
    let mut sums = [zero; 4].map(Tile::literal);
    for j in 0..8 {
        sums[0] = sums[0].clone() + low[j].clone();
        sums[1] = sums[1].clone() + low[j + 8].clone();
        sums[2] = sums[2].clone() + high[j].clone();
        sums[3] = sums[3].clone() + high[j + 8].clone();
    }

    Q4KActivations {
        low: low.clone(),
        high: high.clone(),
        sums,
    }
}

/// Q4K subgroup-lane decomposition shared by `qgemv_q4k_ggml` and
/// `qgemv_q4k_paired_ggml`. Splits a 32-wide subgroup into a 4x8 grid where
/// `ix = lane / 8` selects one of 4 K-blocks per workgroup pass and
/// `(iq, ir) = (it / 4, it % 4)` indexes into the 8-byte sub-block.
pub(crate) struct Q4KLane {
    pub(crate) ix: Tile<U32>,
    pub(crate) iq: Tile<U32>,
    pub(crate) ir: Tile<U32>,
}

pub(crate) fn q4k_lane_decomposition(lane: &Tile<U32>) -> Q4KLane {
    let ix = lane.clone() / 8;
    let it = lane.clone() % 8;
    let iq = it.clone() / 4;
    let ir = it % 4;
    Q4KLane { ix, iq, ir }
}

pub(crate) struct Q4KGgmlIteration {
    pub(crate) block: Tile<U32>,
    pub(crate) in_bounds: Mask,
    pub(crate) activations: Q4KActivations,
}

pub(crate) struct Q4KGgmlIterationRequest<'a, Row> {
    pub(crate) loop_index: Tile<U32>,
    pub(crate) a: &'a Storage<F32, 2>,
    pub(crate) row: Row,
    pub(crate) block_count: u32,
    pub(crate) full_block_iterations: bool,
    pub(crate) lane: &'a Q4KLane,
    pub(crate) base_mask: Mask,
}

pub(crate) fn q4k_ggml_iteration<Row>(
    program: &mut TileBlock<'_>,
    request: Q4KGgmlIterationRequest<'_, Row>,
) -> Q4KGgmlIteration
where
    Row: Clone + Into<Tile<U32>>,
{
    let Q4KGgmlIterationRequest {
        loop_index,
        a,
        row,
        block_count,
        full_block_iterations,
        lane,
        base_mask,
    } = request;
    let block = loop_index * 4 + lane.ix.clone();
    let in_bounds = if full_block_iterations {
        base_mask
    } else {
        base_mask.and(block.clone().lt(block_count))
    };
    let vector_base = block.clone() * 256 + lane.iq.clone() * 64 + lane.ir.clone() * 8;
    let activations = q4k_ggml_activations(program, a, row, &vector_base, in_bounds.clone());
    Q4KGgmlIteration {
        block,
        in_bounds,
        activations,
    }
}

pub(crate) struct Q6KLane {
    pub(crate) ix: Tile<U32>,
    pub(crate) ip: Tile<U32>,
    pub(crate) il: Tile<U32>,
    pub(crate) l0: Tile<U32>,
}

pub(crate) fn q6k_lane_decomposition(lane: &Tile<U32>) -> Q6KLane {
    let tid = lane.clone() / 2;
    let ix = lane.clone() % 2;
    let ip = tid.clone() / 8;
    let il = tid % 8;
    let l0 = il.clone() * 4;
    Q6KLane { ix, ip, il, l0 }
}

pub(crate) fn q6k_ggml_activations(
    program: &mut TileBlock<'_>,
    a: &Storage<F32, 2>,
    row: impl Clone + Into<Tile<U32>>,
    vector_base: &Tile<U32>,
    in_bounds: Mask,
) -> [Tile; 16] {
    let a_bound: [Tile; 16] = std::array::from_fn(|j| {
        let offset = (j / 4) as u32 + (j % 4) as u32 * 32;
        let scalar = program.load(
            a.at((row.clone(), vector_base.clone() + offset)),
            in_bounds.clone(),
            0.0,
        );
        program.bind(scalar)
    });
    a_bound
}

pub(crate) struct Q6KGgmlIteration {
    pub(crate) block: Tile<U32>,
    pub(crate) in_bounds: Mask,
    pub(crate) activations: [Tile; 16],
}

pub(crate) struct Q6KGgmlIterationRequest<'a, Row> {
    pub(crate) loop_index: Tile<U32>,
    pub(crate) a: &'a Storage<F32, 2>,
    pub(crate) row: Row,
    pub(crate) block_count: u32,
    pub(crate) full_block_iterations: bool,
    pub(crate) lane: &'a Q6KLane,
    pub(crate) base_mask: Mask,
}

pub(crate) fn q6k_ggml_iteration<Row>(
    program: &mut TileBlock<'_>,
    request: Q6KGgmlIterationRequest<'_, Row>,
) -> Q6KGgmlIteration
where
    Row: Clone + Into<Tile<U32>>,
{
    let Q6KGgmlIterationRequest {
        loop_index,
        a,
        row,
        block_count,
        full_block_iterations,
        lane,
        base_mask,
    } = request;
    let block = loop_index * 2 + lane.ix.clone();
    let in_bounds = if full_block_iterations {
        base_mask
    } else {
        base_mask.and(block.clone().lt(block_count))
    };
    let vector_base = block.clone() * 256 + lane.ip.clone() * 128 + lane.l0.clone();
    let activations = q6k_ggml_activations(program, a, row, &vector_base, in_bounds.clone());
    Q6KGgmlIteration {
        block,
        in_bounds,
        activations,
    }
}

pub(crate) fn dot4_sum<const VALUES: usize>(
    program: &TileBlock<'_>,
    a: &[Tile; VALUES],
    b: &[Tile; VALUES],
) -> Tile {
    debug_assert!(VALUES >= 4 && VALUES.is_multiple_of(4));
    let mut sum: Option<Tile> = None;
    for chunk in 0..VALUES / 4 {
        let a_vec = std::array::from_fn(|i| a[chunk * 4 + i].clone());
        let b_vec = std::array::from_fn(|i| b[chunk * 4 + i].clone());
        let a_vec = program.compose_vector::<F32, 4>(a_vec);
        let b_vec = program.compose_vector::<F32, 4>(b_vec);
        let term = program.vector_dot::<F32, 4>(a_vec, b_vec);
        sum = Some(match sum {
            Some(prev) => prev + term,
            None => term,
        });
    }
    sum.expect("VALUES >= 4")
}
