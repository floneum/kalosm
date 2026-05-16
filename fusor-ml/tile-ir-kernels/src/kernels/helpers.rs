use fusor_tile_ir::{
    tile::{Tile, TileBlock},
    TileLiteral,
};

pub(super) const TOP_K_BLOCK: usize = 256;
pub(super) const MAX_F32: f32 = f32::MAX;
pub(super) const NEG_MAX_F32: f32 = -f32::MAX;

/// One component of a strided tensor index.
pub(super) enum IndexComponent {
    /// Compile-time scalar component that can be folded into the base offset.
    Static(u32),
    /// Per-lane component that must remain in the tile expression.
    Dynamic(Box<Tile>),
}

/// Convert one scalar or per-lane value into an index component.
pub(super) trait Index {
    /// Consume or clone into a component usable by [`index_n`].
    fn into_component(self) -> IndexComponent;
}

impl Index for u32 {
    fn into_component(self) -> IndexComponent {
        IndexComponent::Static(self)
    }
}

impl Index for Tile {
    fn into_component(self) -> IndexComponent {
        IndexComponent::Dynamic(Box::new(self))
    }
}

impl Index for &Tile {
    fn into_component(self) -> IndexComponent {
        IndexComponent::Dynamic(Box::new(self.clone()))
    }
}

/// Convert a rank-`R` list of components into index components.
pub(super) trait IntoIndexExpr<const R: usize> {
    /// Consume the list while preserving component order.
    fn into_indices(self) -> [IndexComponent; R];
}

impl<I, const R: usize> IntoIndexExpr<R> for [I; R]
where
    I: Index,
{
    fn into_indices(self) -> [IndexComponent; R] {
        self.map(Index::into_component)
    }
}

impl<I> IntoIndexExpr<1> for I
where
    I: Index,
{
    fn into_indices(self) -> [IndexComponent; 1] {
        [self.into_component()]
    }
}

impl<Prefix, Last> IntoIndexExpr<2> for (Prefix, Last)
where
    Prefix: IntoIndexExpr<1>,
    Last: Index,
{
    fn into_indices(self) -> [IndexComponent; 2] {
        let [i0] = self.0.into_indices();
        [i0, self.1.into_component()]
    }
}

impl<Prefix, Last> IntoIndexExpr<3> for (Prefix, Last)
where
    Prefix: IntoIndexExpr<2>,
    Last: Index,
{
    fn into_indices(self) -> [IndexComponent; 3] {
        let [i0, i1] = self.0.into_indices();
        [i0, i1, self.1.into_component()]
    }
}

impl<A, B, C> IntoIndexExpr<3> for (A, B, C)
where
    A: Index,
    B: Index,
    C: Index,
{
    fn into_indices(self) -> [IndexComponent; 3] {
        [
            self.0.into_component(),
            self.1.into_component(),
            self.2.into_component(),
        ]
    }
}

impl<Prefix, Last> IntoIndexExpr<4> for (Prefix, Last)
where
    Prefix: IntoIndexExpr<3>,
    Last: Index,
{
    fn into_indices(self) -> [IndexComponent; 4] {
        let [i0, i1, i2] = self.0.into_indices();
        [i0, i1, i2, self.1.into_component()]
    }
}

impl<A, B, C, D> IntoIndexExpr<4> for (A, B, C, D)
where
    A: Index,
    B: Index,
    C: Index,
    D: Index,
{
    fn into_indices(self) -> [IndexComponent; 4] {
        [
            self.0.into_component(),
            self.1.into_component(),
            self.2.into_component(),
            self.3.into_component(),
        ]
    }
}

/// `offset + sum(strides[i] * components[i])`. Strided index into a
/// rank-`N` row-major tensor, with a constant scalar offset folded in. The
/// fold elides the multiply when the corresponding stride is zero.
pub(super) fn index_n<const R: usize>(
    offset: u32,
    strides: [u32; R],
    components: impl IntoIndexExpr<R>,
) -> Tile {
    let mut folded_offset = offset;
    let mut dynamic_components = Vec::with_capacity(R);
    for (component, stride) in components.into_indices().into_iter().zip(strides) {
        match component {
            IndexComponent::Static(value) => {
                folded_offset = folded_offset.wrapping_add(value.wrapping_mul(stride));
            }
            IndexComponent::Dynamic(component) => {
                if stride != 0 {
                    dynamic_components.push((*component, stride));
                }
            }
        }
    }

    dynamic_components.into_iter().fold(
        Tile::literal(TileLiteral::U32(folded_offset)),
        |index, (component, stride)| match stride {
            0 => index,
            1 => index + component,
            _ => index + component * Tile::literal(TileLiteral::U32(stride)),
        },
    )
}

/// Tree-reduce a workgroup-scratch array by halving stride, applying
/// `combine(lhs, rhs)` at each level. The combine closure is the only
/// difference between sum/max/bitwise-or reductions, which previously each
/// had their own near-identical loop.
pub(super) fn reduce_workgroup(
    program: &mut TileBlock<'_>,
    scratch: fusor_tile_ir::TileRef,
    lane: fusor_tile_ir::tile::Tile,
    combine: impl Fn(Tile, Tile) -> Tile,
) {
    let mut stride = program.block_size() as u32 / 2;
    while stride > 0 {
        let participates = program
            .index(lane.clone())
            .lt(Tile::literal(TileLiteral::U32(stride)));
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
