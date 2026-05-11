use fusor_tile_ir::{
    tile::{Mask, Tile, TileBlock},
    TileLiteral,
};

pub(super) const TOP_K_BLOCK: usize = 256;
pub(super) const MAX_F32: f32 = f32::MAX;
pub(super) const NEG_MAX_F32: f32 = -f32::MAX;

pub(super) fn all<const BLOCK: usize>() -> Mask<BLOCK> {
    Mask::all()
}

pub(super) fn f32_tile<const BLOCK: usize>(value: f32) -> Tile<BLOCK> {
    Tile::literal(TileLiteral::f32(value))
}

pub(super) fn u32_tile<const BLOCK: usize>(value: u32) -> Tile<BLOCK> {
    Tile::literal(TileLiteral::U32(value))
}

pub(super) fn add_scaled_index<const BLOCK: usize>(
    index: Tile<BLOCK>,
    component: Tile<BLOCK>,
    stride: u32,
) -> Tile<BLOCK> {
    match stride {
        0 => index,
        1 => index + component,
        _ => index + component * u32_tile(stride),
    }
}

/// `offset + sum(strides[i] * components[i])`. Strided index into a
/// rank-`N` row-major tensor, with a constant scalar offset folded in. The
/// `add_scaled_index` shortcut elides the multiply when the corresponding
/// stride is zero.
pub(super) fn index_n<const N: usize, const BLOCK: usize>(
    offset: u32,
    strides: [u32; N],
    components: [Tile<BLOCK>; N],
) -> Tile<BLOCK> {
    components
        .into_iter()
        .zip(strides)
        .fold(u32_tile(offset), |idx, (c, s)| add_scaled_index(idx, c, s))
}

pub(super) fn index2<const BLOCK: usize>(
    offset: u32,
    strides: [u32; 2],
    i0: Tile<BLOCK>,
    i1: Tile<BLOCK>,
) -> Tile<BLOCK> {
    index_n(offset, strides, [i0, i1])
}

pub(super) fn index4<const BLOCK: usize>(
    offset: u32,
    strides: [u32; 4],
    i0: Tile<BLOCK>,
    i1: Tile<BLOCK>,
    i2: Tile<BLOCK>,
    i3: Tile<BLOCK>,
) -> Tile<BLOCK> {
    index_n(offset, strides, [i0, i1, i2, i3])
}

pub(super) fn index4_const_last<const BLOCK: usize>(
    offset: u32,
    strides: [u32; 4],
    i0: Tile<BLOCK>,
    i1: Tile<BLOCK>,
    i2: Tile<BLOCK>,
    i3: u32,
) -> Tile<BLOCK> {
    let base = offset + i3 * strides[3];
    index_n(base, [strides[0], strides[1], strides[2]], [i0, i1, i2])
}

/// `lane == 0` mask for the common "only one lane writes" pattern in
/// kernels that broadcast a result via lane-zero stores.
pub(super) fn lane_zero<const BLOCK: usize>(
    program: &TileBlock<'_, BLOCK>,
    lane: &fusor_tile_ir::tile::Range<BLOCK>,
) -> Tile<BLOCK> {
    program.index(lane.clone()).eq(u32_tile(0))
}

/// Tree-reduce a workgroup-scratch array by halving stride, applying
/// `combine(lhs, rhs)` at each level. The combine closure is the only
/// difference between sum/max/bitwise-or reductions, which previously each
/// had their own near-identical loop.
pub(super) fn reduce_workgroup<const BLOCK: usize>(
    program: &mut TileBlock<'_, BLOCK>,
    scratch: fusor_tile_ir::TileRef,
    lane: fusor_tile_ir::tile::Range<BLOCK>,
    combine: impl Fn(Tile<BLOCK>, Tile<BLOCK>) -> Tile<BLOCK>,
) {
    let mut stride = BLOCK as u32 / 2;
    while stride > 0 {
        let participates = program.index(lane.clone()).lt(u32_tile(stride));
        program.if_then(participates, |program| {
            let lhs = program.load_workgroup(scratch, lane.clone());
            let rhs_index = lane.clone() + stride;
            let rhs = program.load_workgroup(scratch, rhs_index);
            program.store_workgroup(scratch, lane.clone(), combine(lhs, rhs));
        });
        program.workgroup_barrier();
        stride /= 2;
    }
}
