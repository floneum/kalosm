use fusor_tile_ir::{
    tile::{CoopAcc, CoopFragment, CoopRole, Storage, Tile, TileBlock, Workgroup},
    CoopElement, FloatElement, Numeric, TileLiteral, F16, F32, U32,
};

use crate::types::QmatmulExtra;

/// Storage-side conversion to/from an accumulator element type. The
/// `<F32, F32>` and `<F16, F16>` impls are identity; the `<F16, F32>` impl
/// inserts the cast pair that lets F16 storage be loaded into F32
/// accumulators and stored back. Used by the unified
/// `batched_matmul_with_epilogues<Stor, Accum, ...>` / `batched_gemv_*`
/// kernels so we don't have to duplicate every body per (storage, accum) pair.
pub trait AccumCast<Accum: FloatElement>: FloatElement {
    /// Storage-typed zero literal — for kernel load `fill` arguments.
    const ZERO_STORAGE: TileLiteral;
    /// Accumulator-typed zero literal — for fold init / select fallback.
    const ZERO_ACCUM: TileLiteral;
    /// Promote a freshly-loaded storage tile to the accumulator type.
    fn into_accum(tile: Tile<Self>) -> Tile<Accum>
    where
        Self: Sized;
    /// Demote a post-epilogue accumulator tile back to storage for the store.
    fn from_accum(tile: Tile<Accum>) -> Tile<Self>
    where
        Self: Sized;
}

impl AccumCast<F32> for F32 {
    const ZERO_STORAGE: TileLiteral = TileLiteral::F32(fusor_tile_ir::F32Bits(0));
    const ZERO_ACCUM: TileLiteral = TileLiteral::F32(fusor_tile_ir::F32Bits(0));
    fn into_accum(tile: Tile<F32>) -> Tile<F32> {
        tile
    }
    fn from_accum(tile: Tile<F32>) -> Tile<F32> {
        tile
    }
}

impl AccumCast<F32> for F16 {
    const ZERO_STORAGE: TileLiteral = TileLiteral::F16(0);
    const ZERO_ACCUM: TileLiteral = TileLiteral::F32(fusor_tile_ir::F32Bits(0));
    fn into_accum(tile: Tile<F16>) -> Tile<F32> {
        tile.cast::<F32>()
    }
    fn from_accum(tile: Tile<F32>) -> Tile<F16> {
        tile.cast::<F16>()
    }
}

pub(super) const TOP_K_BLOCK: usize = 256;
pub(super) const MAX_F32: f32 = f32::MAX;
pub(super) const NEG_MAX_F32: f32 = -f32::MAX;

/// One component of a strided tensor index.
pub(super) enum IndexComponent {
    /// Compile-time scalar component that can be folded into the base offset.
    Static(u32),
    /// Per-lane component that must remain in the tile expression.
    Dynamic(Box<Tile<U32>>),
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

impl Index for Tile<U32> {
    fn into_component(self) -> IndexComponent {
        IndexComponent::Dynamic(Box::new(self))
    }
}

impl Index for &Tile<U32> {
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
) -> Tile<U32> {
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
pub(super) fn reduce_workgroup<T>(
    program: &mut TileBlock<'_>,
    scratch: Workgroup<T>,
    lane: Tile<U32>,
    combine: impl Fn(Tile<T>, Tile<T>) -> Tile<T>,
) where
    T: Numeric,
{
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

/// Allocate a `rows x cols` grid of zero-initialized 8x8 cooperative
/// accumulators. Shared between dense and quantized cooperative matmul.
pub(super) fn zero_coop_acc_grid<T: CoopElement>(
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

/// Cooperatively load `rows` A-role 8x8 fragments from a workgroup tile.
pub(super) fn coop_load_a_fragments<T: CoopElement>(
    program: &mut TileBlock<'_>,
    tile: Workgroup<T>,
    sg_row_base: &Tile<U32>,
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

/// Cooperatively load `cols` B-role 8x8 fragments from a workgroup tile.
pub(super) fn coop_load_b_fragments<T: CoopElement>(
    program: &mut TileBlock<'_>,
    tile: Workgroup<T>,
    sg_col_base: &Tile<U32>,
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

/// Cooperatively load `cols` C-role fragments from a rank-1 column vector,
/// broadcasting each 8-column slice across the fragment rows.
pub(super) fn coop_load_c_broadcast_fragments<T: CoopElement>(
    program: &mut TileBlock<'_>,
    vector: &Storage<T, 1>,
    col_base: &Tile<U32>,
    cols: u32,
) -> Vec<CoopFragment<T, 8, 8>> {
    const COOP_DIM: u32 = 8;
    (0..cols)
        .map(|c| program.coop_load_broadcast_cols(vector, col_base.clone() + c * COOP_DIM))
        .collect()
}

/// MMA every `a_frag` × `b_frag` pair into the matching accumulator.
pub(super) fn coop_mma_grid<T: CoopElement>(
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

/// Initialize every accumulator row from a C-role column-broadcast fragment.
pub(super) fn coop_set_c_grid<T: CoopElement>(
    program: &mut TileBlock<'_>,
    accs: &[Vec<CoopAcc<T, 8, 8>>],
    c_frags: &[CoopFragment<T, 8, 8>],
) {
    for row_accs in accs {
        for (c, acc) in row_accs.iter().enumerate() {
            program.coop_set_acc(acc, &c_frags[c]);
        }
    }
}

/// 1D-logical workgroup count dispatched as a 3D grid clamped to
/// `max_per_dim` in each axis. Shared by dense and quantized matmul
/// dispatch paths.
pub(super) fn dispatch_grid_1d(total_workgroups: u32, max_per_dim: u32) -> [u32; 3] {
    assert!(total_workgroups > 0, "matmul dispatch must have workgroups");
    assert!(max_per_dim > 0, "max_per_dim must be non-zero");
    let x = total_workgroups.min(max_per_dim);
    let y_needed = total_workgroups.div_ceil(x);
    let y = y_needed.min(max_per_dim);
    let z = y_needed.div_ceil(y).max(1);
    [x, y, z]
}

/// Load the per-output-element extra activation/column for qmatmul: a
/// column vector indexed by `col` or a pointwise tensor indexed by
/// `(row, col)`, with an out-of-bound mask falling back to `0.0`.
pub(super) fn load_qmatmul_extra(
    program: &mut TileBlock<'_>,
    extra: &QmatmulExtra<'_>,
    row: &Tile<U32>,
    col: &Tile<U32>,
    n_cols: u32,
) -> Tile<F32> {
    match extra {
        QmatmulExtra::Column(vector) => program.load(vector.at(col), col.lt(n_cols), 0.0),
        QmatmulExtra::Pointwise(tensor) => {
            program.load(tensor.at((row, col)), col.lt(n_cols), 0.0)
        }
    }
}

/// Cooperatively store an accumulator grid into a rank-2 storage view.
/// `y_batch_base` is added to the row index when `Some` — batched matmul
/// kernels pass the batch row offset; single-batch quantized matmul passes
/// `None`.
#[allow(clippy::too_many_arguments)]
pub(super) fn coop_store_acc_grid<T: CoopElement>(
    program: &mut TileBlock<'_>,
    accs: &[Vec<CoopAcc<T, 8, 8>>],
    y: &Storage<T, 2>,
    y_batch_base: Option<&Tile<U32>>,
    row_base: &Tile<U32>,
    col_base: &Tile<U32>,
    sg_row_base: &Tile<U32>,
    sg_col_base: &Tile<U32>,
) {
    const COOP_DIM: u32 = 8;
    for (r, row_accs) in accs.iter().enumerate() {
        for (c, acc) in row_accs.iter().enumerate() {
            let local_row = row_base.clone() + sg_row_base.clone() + r as u32 * COOP_DIM;
            let row = match y_batch_base {
                Some(batch) => batch.clone() + local_row,
                None => local_row,
            };
            let col = col_base.clone() + sg_col_base.clone() + c as u32 * COOP_DIM;
            program.coop_store(acc, y, row, col);
        }
    }
}
