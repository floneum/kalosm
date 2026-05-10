use crate::{
    tile::{Mask, Tile, TileBlock},
    F32Bits, TileLiteral,
};

pub(super) const TOP_K_BLOCK: usize = 256;
pub(super) const MAX_F32: f32 = f32::MAX;
pub(super) const NEG_MAX_F32: f32 = -f32::MAX;

pub(super) fn all<const BLOCK: usize>() -> Mask<BLOCK> {
    Mask::all()
}

pub(super) fn f32_tile<const BLOCK: usize>(value: f32) -> Tile<BLOCK> {
    Tile::literal(TileLiteral::F32(F32Bits::new(value)))
}

pub(super) fn u32_tile<const BLOCK: usize>(value: u32) -> Tile<BLOCK> {
    Tile::literal(TileLiteral::U32(value))
}

pub(super) fn add_scaled_index<const BLOCK: usize>(
    index: Tile<BLOCK>,
    component: Tile<BLOCK>,
    stride: u32,
) -> Tile<BLOCK> {
    if stride == 0 {
        index
    } else {
        index + component * u32_tile(stride)
    }
}

pub(super) fn index2<const BLOCK: usize>(
    offset: u32,
    strides: [u32; 2],
    i0: Tile<BLOCK>,
    i1: Tile<BLOCK>,
) -> Tile<BLOCK> {
    let index = add_scaled_index(u32_tile(offset), i0, strides[0]);
    add_scaled_index(index, i1, strides[1])
}

pub(super) fn index3_with_base<const BLOCK: usize>(
    base: u32,
    strides: [u32; 3],
    i0: Tile<BLOCK>,
    i1: Tile<BLOCK>,
    i2: Tile<BLOCK>,
) -> Tile<BLOCK> {
    let index = add_scaled_index(u32_tile(base), i0, strides[0]);
    let index = add_scaled_index(index, i1, strides[1]);
    add_scaled_index(index, i2, strides[2])
}

pub(super) fn index4<const BLOCK: usize>(
    offset: u32,
    strides: [u32; 4],
    i0: Tile<BLOCK>,
    i1: Tile<BLOCK>,
    i2: Tile<BLOCK>,
    i3: Tile<BLOCK>,
) -> Tile<BLOCK> {
    let index = index3_with_base(offset, [strides[0], strides[1], strides[2]], i0, i1, i2);
    add_scaled_index(index, i3, strides[3])
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
    index3_with_base(base, [strides[0], strides[1], strides[2]], i0, i1, i2)
}

/// Tree-reduce a workgroup-scratch array by halving stride, applying
/// `combine(lhs, rhs)` at each level. The combine closure is the only
/// difference between sum/max/bitwise-or reductions, which previously each
/// had their own near-identical loop.
pub(super) fn reduce_workgroup<const BLOCK: usize>(
    program: &mut TileBlock<'_, BLOCK>,
    scratch: crate::TileRef,
    lane: crate::tile::Range<BLOCK>,
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
