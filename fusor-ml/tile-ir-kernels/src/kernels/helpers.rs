use fusor_tile_ir::{
    tile::{Range, ScalarIndex, Tile, TileBlock},
    TileLiteral,
};

pub(super) const TOP_K_BLOCK: usize = 256;
pub(super) const MAX_F32: f32 = f32::MAX;
pub(super) const NEG_MAX_F32: f32 = -f32::MAX;

/// One component of a strided tensor index.
pub(super) enum IndexComponent<const BLOCK: usize> {
    /// Compile-time scalar component that can be folded into the base offset.
    Static(u32),
    /// Per-lane component that must remain in the tile expression.
    Dynamic(Box<Tile<BLOCK>>),
}

/// Convert one scalar or per-lane value into an index component.
pub(super) trait Index<const BLOCK: usize> {
    /// Consume or clone into a component usable by [`index_n`].
    fn into_component(self) -> IndexComponent<BLOCK>;
}

impl<const BLOCK: usize> Index<BLOCK> for u32 {
    fn into_component(self) -> IndexComponent<BLOCK> {
        IndexComponent::Static(self)
    }
}

impl<const BLOCK: usize> Index<BLOCK> for Tile<BLOCK> {
    fn into_component(self) -> IndexComponent<BLOCK> {
        IndexComponent::Dynamic(Box::new(self))
    }
}

impl<const BLOCK: usize> Index<BLOCK> for &Tile<BLOCK> {
    fn into_component(self) -> IndexComponent<BLOCK> {
        IndexComponent::Dynamic(Box::new(self.clone()))
    }
}

impl<const BLOCK: usize> Index<BLOCK> for Range<BLOCK> {
    fn into_component(self) -> IndexComponent<BLOCK> {
        IndexComponent::Dynamic(Box::new(Tile::from_index(self)))
    }
}

impl<const BLOCK: usize> Index<BLOCK> for &Range<BLOCK> {
    fn into_component(self) -> IndexComponent<BLOCK> {
        IndexComponent::Dynamic(Box::new(Tile::from_index(self)))
    }
}

impl<const BLOCK: usize> Index<BLOCK> for ScalarIndex {
    fn into_component(self) -> IndexComponent<BLOCK> {
        IndexComponent::Dynamic(Box::new(Tile::from_index(self)))
    }
}

impl<const BLOCK: usize> Index<BLOCK> for &ScalarIndex {
    fn into_component(self) -> IndexComponent<BLOCK> {
        IndexComponent::Dynamic(Box::new(Tile::from_index(self)))
    }
}

/// Convert a rank-`R` list of components into index components.
pub(super) trait IntoIndex<const R: usize, const BLOCK: usize> {
    /// Consume the list while preserving component order.
    fn into_indices(self) -> [IndexComponent<BLOCK>; R];
}

impl<I, const R: usize, const BLOCK: usize> IntoIndex<R, BLOCK> for [I; R]
where
    I: Index<BLOCK>,
{
    fn into_indices(self) -> [IndexComponent<BLOCK>; R] {
        self.map(Index::into_component)
    }
}

impl<I, const BLOCK: usize> IntoIndex<1, BLOCK> for I
where
    I: Index<BLOCK>,
{
    fn into_indices(self) -> [IndexComponent<BLOCK>; 1] {
        [self.into_component()]
    }
}

impl<Prefix, Last, const BLOCK: usize> IntoIndex<2, BLOCK> for (Prefix, Last)
where
    Prefix: IntoIndex<1, BLOCK>,
    Last: Index<BLOCK>,
{
    fn into_indices(self) -> [IndexComponent<BLOCK>; 2] {
        let [i0] = self.0.into_indices();
        [i0, self.1.into_component()]
    }
}

impl<Prefix, Last, const BLOCK: usize> IntoIndex<3, BLOCK> for (Prefix, Last)
where
    Prefix: IntoIndex<2, BLOCK>,
    Last: Index<BLOCK>,
{
    fn into_indices(self) -> [IndexComponent<BLOCK>; 3] {
        let [i0, i1] = self.0.into_indices();
        [i0, i1, self.1.into_component()]
    }
}

impl<A, B, C, const BLOCK: usize> IntoIndex<3, BLOCK> for (A, B, C)
where
    A: Index<BLOCK>,
    B: Index<BLOCK>,
    C: Index<BLOCK>,
{
    fn into_indices(self) -> [IndexComponent<BLOCK>; 3] {
        [
            self.0.into_component(),
            self.1.into_component(),
            self.2.into_component(),
        ]
    }
}

impl<Prefix, Last, const BLOCK: usize> IntoIndex<4, BLOCK> for (Prefix, Last)
where
    Prefix: IntoIndex<3, BLOCK>,
    Last: Index<BLOCK>,
{
    fn into_indices(self) -> [IndexComponent<BLOCK>; 4] {
        let [i0, i1, i2] = self.0.into_indices();
        [i0, i1, i2, self.1.into_component()]
    }
}

impl<A, B, C, D, const BLOCK: usize> IntoIndex<4, BLOCK> for (A, B, C, D)
where
    A: Index<BLOCK>,
    B: Index<BLOCK>,
    C: Index<BLOCK>,
    D: Index<BLOCK>,
{
    fn into_indices(self) -> [IndexComponent<BLOCK>; 4] {
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
pub(super) fn index_n<const R: usize, const BLOCK: usize>(
    offset: u32,
    strides: [u32; R],
    components: impl IntoIndex<R, BLOCK>,
) -> Tile<BLOCK> {
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
pub(super) fn reduce_workgroup<const BLOCK: usize>(
    program: &mut TileBlock<'_, BLOCK>,
    scratch: fusor_tile_ir::TileRef,
    lane: fusor_tile_ir::tile::Range<BLOCK>,
    combine: impl Fn(Tile<BLOCK>, Tile<BLOCK>) -> Tile<BLOCK>,
) {
    let mut stride = BLOCK as u32 / 2;
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
