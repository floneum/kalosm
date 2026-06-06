//! qgemv-specific grid helpers — moved out of `fusor-tile-ir`'s `tile/grid.rs`
//! when the kernels split landed. `tile-ir`'s grid.rs still owns the generic
//! `tile::build` entry point.
//!
//! Tile and storage values carry their [`ScalarElement`] at runtime, so grid
//! helpers stay generic over element choices without Rust marker types.

use fusor_tile_ir::tile::{Mask, Storage, Tile, TileBlock};
use fusor_tile_ir::{ScalarElement, SubgroupToken, TileLiteral, WorkgroupAxis};

#[derive(Clone, Copy)]
pub(crate) struct QgemvGrid {
    pub(crate) workgroups_x: u32,
    pub(crate) dispatch_y: u32,
    pub(crate) n_cols: u32,
}

pub(crate) fn qgemv_grid(
    dispatch_subgroups: u32,
    cols_per_subgroup: u32,
    n_cols: u32,
    requested_workgroups_x: u32,
) -> QgemvGrid {
    let cols_per_workgroup = dispatch_subgroups * cols_per_subgroup;
    let total_workgroups = n_cols.div_ceil(cols_per_workgroup);
    let workgroups_x = requested_workgroups_x.min(total_workgroups.max(1));
    QgemvGrid {
        workgroups_x,
        dispatch_y: total_workgroups.div_ceil(workgroups_x),
        n_cols,
    }
}

impl QgemvGrid {
    pub(crate) fn mask(self, in_bounds: Mask, col: &Tile) -> Mask {
        in_bounds.and(col.lt(self.n_cols))
    }
}

#[derive(Clone)]
pub(crate) struct QgemvProgramScope {
    pub(crate) col0: Tile,
    pub(crate) lane: Tile,
}

pub(crate) struct QgemvStoreTarget<'a> {
    pub(crate) subgroup: SubgroupToken,
    pub(crate) y: &'a Storage,
    pub(crate) col0: Tile,
    pub(crate) lane: Tile,
    pub(crate) n_cols: u32,
    pub(crate) epilogues: &'a crate::types::QmatmulEpilogues<'a>,
}

pub(crate) fn qgemv_program_scope(
    program: &TileBlock<'_>,
    grid: QgemvGrid,
    cols_per_subgroup: u32,
    subgroup: SubgroupToken,
) -> QgemvProgramScope {
    let workgroup = program.program_id(WorkgroupAxis::X)
        + program.program_id(WorkgroupAxis::Y) * grid.workgroups_x;
    let col_group_base = workgroup * subgroup.num_subgroups(program) * cols_per_subgroup;
    let subgroup_col_base = subgroup.subgroup_id(program) * cols_per_subgroup;
    QgemvProgramScope {
        col0: col_group_base + subgroup_col_base,
        lane: subgroup.subgroup_lane(program),
    }
}

/// Store subgroup-reduced qgemv sums, applying an optional post-reduce
/// epilogue between the subgroup reduce and the store. The `pre` slot is
/// ignored here because pre-epilogues are applied at load sites by the kernel
/// body.
pub(crate) fn store_qgemv_sums_with_epilogue(
    program: &mut TileBlock<'_>,
    sums: Vec<Tile>,
    target: QgemvStoreTarget<'_>,
) {
    if target.epilogues.post_accumulator_offsets.is_empty() {
        for (offset, sum) in sums.into_iter().enumerate() {
            let col = target.col0.clone() + offset as u32;
            let reduced = target.subgroup.subgroup_reduce_sum(program, sum);
            let extras = target
                .epilogues
                .post_extra_inputs
                .iter()
                .map(|extra| match extra {
                    crate::types::QmatmulExtra::Column(vector) => {
                        program.load(vector.at(&col), col.lt(target.n_cols), 0.0)
                    }
                    crate::types::QmatmulExtra::Pointwise(tensor) => {
                        let row = Tile::literal(TileLiteral::U32(0));
                        program.load(tensor.at((row, &col)), col.lt(target.n_cols), 0.0)
                    }
                })
                .collect::<Vec<_>>();
            let value =
                crate::types::apply_qmatmul_post_epilogue(target.epilogues, reduced, extras);
            let mask = target.lane.eq(0u32).and(col.lt(target.n_cols));
            program.store(target.y.at((0u32, col)), value, mask);
        }
        return;
    }

    let value_arity = target.epilogues.post_value_arity();
    assert!(
        sums.len().is_multiple_of(value_arity),
        "qgemv sums must be grouped by output column"
    );
    for (offset, sums) in sums.chunks(value_arity).enumerate() {
        let col = target.col0.clone() + offset as u32;
        let reduced = sums
            .iter()
            .cloned()
            .map(|sum| target.subgroup.subgroup_reduce_sum(program, sum))
            .collect::<Vec<_>>();
        let extras = target
            .epilogues
            .post_extra_inputs
            .iter()
            .map(|extra| match extra {
                crate::types::QmatmulExtra::Column(vector) => {
                    program.load(vector.at(&col), col.lt(target.n_cols), 0.0)
                }
                crate::types::QmatmulExtra::Pointwise(tensor) => {
                    let row = Tile::literal(TileLiteral::U32(0));
                    program.load(tensor.at((row, &col)), col.lt(target.n_cols), 0.0)
                }
            })
            .collect::<Vec<_>>();
        let value =
            crate::types::apply_qmatmul_post_epilogue_values(target.epilogues, reduced, extras);
        let mask = target.lane.eq(0u32).and(col.lt(target.n_cols));
        program.store(target.y.at((0u32, col)), value, mask);
    }
}

pub(crate) fn dot4_sum(program: &TileBlock<'_>, a: &[Tile], b: &[Tile]) -> Tile {
    debug_assert_eq!(a.len(), b.len());
    let values = a.len();
    debug_assert!(values >= 4 && values.is_multiple_of(4));
    let mut sum: Option<Tile> = None;
    for chunk in 0..values / 4 {
        let a_vec: [Tile; 4] = std::array::from_fn(|i| a[chunk * 4 + i].clone());
        let b_vec: [Tile; 4] = std::array::from_fn(|i| b[chunk * 4 + i].clone());
        let a_vec = program.compose_vector(ScalarElement::F32, a_vec);
        let b_vec = program.compose_vector(ScalarElement::F32, b_vec);
        let term = program.vector_dot(a_vec, b_vec);
        sum = Some(match sum {
            Some(prev) => prev + term,
            None => term,
        });
    }
    sum.expect("values >= 4")
}
