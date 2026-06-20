use fusor_tile_ir::{
    TileLiteral,
    tile::{Mask, Program, Tile, TileBlock},
};

use crate::{nary_direct::linear_group, visit_tiled::distribute_workgroups};

#[derive(Clone, Copy, Debug)]
pub(crate) struct RowDispatchSpec {
    pub(crate) rows: u32,
    pub(crate) block: u32,
    pub(crate) dispatch_size: [u32; 3],
}

impl RowDispatchSpec {
    pub(crate) fn explicit(rows: u32, block: u32, dispatch_size: [u32; 3]) -> Self {
        Self {
            rows,
            block,
            dispatch_size,
        }
    }

    pub(crate) fn distributed(rows: u32, block: u32, max_workgroups_per_dimension: u32) -> Self {
        Self::explicit(
            rows,
            block,
            distribute_workgroups(rows, max_workgroups_per_dimension),
        )
    }

    pub(crate) fn single(block: u32) -> Self {
        Self::explicit(1, block, [1, 1, 1])
    }
}

pub(crate) struct RowDispatchContext {
    pub(crate) lane: Tile,
    pub(crate) row: Tile,
    pub(crate) active: Mask,
}

pub(crate) fn emit_row_grid(
    program: &mut Program,
    spec: RowDispatchSpec,
    body: impl FnOnce(&mut TileBlock<'_>, RowDispatchContext),
) {
    let rows = spec.rows;
    let dispatch_size = spec.dispatch_size;
    program.program_grid(spec.block, dispatch_size, |program| {
        let lane = program.lane();
        let row = linear_group(program, dispatch_size);
        let active = row.clone().lt(Tile::literal(TileLiteral::U32(rows)));
        body(program, RowDispatchContext { lane, row, active });
    });
}
