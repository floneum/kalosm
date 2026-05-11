

use crate::ir::{
    F32Bits, KernelIr,
    TileLiteral, F32,
};
use super::*;

#[derive(Clone, Copy)]
pub(super) struct QgemvGrid {
    pub(super) cols_per_workgroup: u32,
    pub(super) workgroups_x: u32,
    pub(super) dispatch_y: u32,
    pub(super) n_cols: u32,
    pub(super) full_cols: bool,
}

pub(super) fn qgemv_grid<const SUBGROUPS: u32, const COLS_PER_SUBGROUP: usize>(
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
    pub(super) fn mask<const BLOCK: usize>(
        self,
        full_iterations: bool,
        in_bounds: Mask<BLOCK>,
        col: &ScalarIndex,
    ) -> Mask<BLOCK> {
        match (full_iterations, self.full_cols) {
            (true, true) => Mask::all(),
            (true, false) => col.lt(self.n_cols),
            (false, true) => in_bounds,
            (false, false) => in_bounds.and(col.lt(self.n_cols)),
        }
    }
}

pub(super) fn store_qgemv_sums<const BLOCK: usize, const COLS_PER_SUBGROUP: usize>(
    program: &mut TileBlock<'_, BLOCK>,
    y: &Storage<F32, 2>,
    col0: ScalarIndex,
    lane: ScalarIndex,
    sums: [Tile<BLOCK>; COLS_PER_SUBGROUP],
    full_cols: bool,
    n_cols: u32,
) {
    for (offset, sum) in sums.into_iter().enumerate() {
        let col = col0.clone() + offset as u32;
        let reduced = program.subgroup_reduce_sum(sum);
        let mask = if full_cols {
            lane.eq(0)
        } else {
            lane.eq(0).and(col.lt(n_cols))
        };
        program.store(y.at(0, col), reduced, mask);
    }
}

#[derive(Clone)]
pub(super) struct Q4KGgmlActivations<const BLOCK: usize> {
    pub(super) low: [Tile<BLOCK>; 16],
    pub(super) high: [Tile<BLOCK>; 16],
    pub(super) sums: [Tile<BLOCK>; 4],
}

pub(super) fn q4k_ggml_activations<const BLOCK: usize>(
    program: &mut TileBlock<'_, BLOCK>,
    a: &Storage<F32, 2>,
    row: impl Clone + IntoIndex<BLOCK>,
    vector_base: &ScalarIndex,
    in_bounds: Mask<BLOCK>,
) -> Q4KGgmlActivations<BLOCK> {
    let load_quad = |program: &mut TileBlock<'_, BLOCK>, base: u32| -> [Bound<BLOCK>; 16] {
        std::array::from_fn(|j| {
            let offset = if j < 8 { j as u32 } else { (j - 8) as u32 + 32 } + base;
            let scalar = program.load(
                a.at(row.clone(), vector_base.clone() + offset),
                in_bounds.clone(),
                0.0,
            );
            program.bind(scalar)
        })
    };
    let low = load_quad(program, 0);
    let high = load_quad(program, 128);

    let zero = TileLiteral::F32(F32Bits::new(0.0));
    let mut sums = [zero; 4].map(Tile::literal);
    for j in 0..8 {
        sums[0] = sums[0].clone() + low[j].get();
        sums[1] = sums[1].clone() + low[j + 8].get();
        sums[2] = sums[2].clone() + high[j].get();
        sums[3] = sums[3].clone() + high[j + 8].get();
    }

    Q4KGgmlActivations {
        low: std::array::from_fn(|i| low[i].get()),
        high: std::array::from_fn(|i| high[i].get()),
        sums,
    }
}

/// Q4K subgroup-lane decomposition shared by `qgemv_q4k_ggml` and
/// `qgemv_q4k_paired_ggml`. Splits a 32-wide subgroup into a 4x8 grid where
/// `ix = lane / 8` selects one of 4 K-blocks per workgroup pass and
/// `(iq, ir) = (it / 4, it % 4)` indexes into the 8-byte sub-block.
pub(super) struct Q4KLane {
    pub ix: ScalarIndex,
    pub iq: ScalarIndex,
    pub ir: ScalarIndex,
}

pub(super) fn q4k_lane_decomposition(lane: &ScalarIndex) -> Q4KLane {
    let ix = lane.clone() / 8;
    let it = lane.clone() % 8;
    let iq = it.clone() / 4;
    let ir = it % 4;
    Q4KLane { ix, iq, ir }
}

pub(super) fn dot4_sum<const BLOCK: usize, const VALUES: usize>(
    program: &TileBlock<'_, BLOCK>,
    a: &[Tile<BLOCK>; VALUES],
    b: &[Tile<BLOCK>; VALUES],
) -> Tile<BLOCK> {
    debug_assert!(VALUES >= 4 && VALUES.is_multiple_of(4));
    let mut sum: Option<Tile<BLOCK>> = None;
    for chunk in 0..VALUES / 4 {
        let a_vec = std::array::from_fn(|i| a[chunk * 4 + i].clone());
        let b_vec = std::array::from_fn(|i| b[chunk * 4 + i].clone());
        let term = program.dot4(a_vec, b_vec);
        sum = Some(match sum {
            Some(prev) => prev + term,
            None => term,
        });
    }
    sum.expect("VALUES >= 4")
}


pub fn build(f: impl FnOnce(&mut Program)) -> KernelIr {
    let mut program = Program::new();
    f(&mut program);
    program.ir
}

