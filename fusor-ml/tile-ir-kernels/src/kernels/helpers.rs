use fusor_tile_ir::tile::{CoopAcc, Storage, Tile, TileBlock, WorkgroupTile};
use fusor_tile_ir::{CoopMatrixToken, ElementType, ScalarElement, TileLiteral};

use crate::types::QmatmulExtra;

/// Storage-side conversion to/from an accumulator element type. The storage and
/// accumulator elements are [`ScalarElement`] data. The `F32 -> F32` /
/// `F16 -> F16` cases are identity; the `F16 -> F32` case inserts the cast
/// pair that lets F16 storage be loaded into F32 accumulators and stored back.
/// Used by the unified `batched_matmul_with_epilogues` / `batched_gemv_*`
/// kernels so we don't have to duplicate every body per (storage, accum) pair.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct AccumCast {
    storage: ScalarElement,
    accum: ScalarElement,
}

impl AccumCast {
    /// Promote `storage` loads into `accum` accumulators. Both must be float
    /// scalars (`F32` or `F16`); the only non-identity pair is `F16 -> F32`.
    pub fn new(storage: ScalarElement, accum: ScalarElement) -> Self {
        assert!(
            matches!(storage, ScalarElement::F32 | ScalarElement::F16),
            "AccumCast storage element must be a float scalar"
        );
        assert!(
            matches!(accum, ScalarElement::F32 | ScalarElement::F16),
            "AccumCast accum element must be a float scalar"
        );
        Self { storage, accum }
    }

    /// The storage scalar element.
    pub fn storage(&self) -> ScalarElement {
        self.storage
    }

    /// The accumulator scalar element.
    pub fn accum(&self) -> ScalarElement {
        self.accum
    }

    /// Storage-typed zero literal — for kernel load `fill` arguments.
    pub fn zero_storage(&self) -> TileLiteral {
        zero_literal(self.storage)
    }

    /// Accumulator-typed zero literal — for fold init / select fallback.
    pub fn zero_accum(&self) -> TileLiteral {
        zero_literal(self.accum)
    }

    /// Promote a freshly-loaded storage tile to the accumulator type.
    pub fn into_accum(&self, tile: Tile) -> Tile {
        if self.storage == self.accum {
            tile
        } else {
            tile.cast(self.accum.element())
        }
    }

    /// Demote a post-epilogue accumulator tile back to storage for the store.
    pub fn from_accum(&self, tile: Tile) -> Tile {
        if self.storage == self.accum {
            tile
        } else {
            tile.cast(self.storage.element())
        }
    }
}

fn zero_literal(element: ScalarElement) -> TileLiteral {
    match element {
        ScalarElement::F32 => TileLiteral::f32(0.0),
        ScalarElement::F16 => TileLiteral::F16(0),
        ScalarElement::U32 => TileLiteral::U32(0),
        ScalarElement::Bool => TileLiteral::Bool(false),
    }
}

pub(super) const TOP_K_BLOCK: usize = 256;
// Naga's WGSL writer prints `f32::MAX` as a decimal literal that the WGSL
// parser rejects on WebGPU. Keep shader sentinels just below that edge.
pub(super) const MAX_F32: f32 = 3.40282e38;
pub(super) const NEG_MAX_F32: f32 = -MAX_F32;

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
    scratch: &WorkgroupTile,
    lane: Tile,
    combine: impl Fn(Tile, Tile) -> Tile,
) {
    let mut stride = program.block_size() / 2;
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

/// The 8x8 cooperative-matrix fragment shape shared by every coop helper.
const COOP_DIM: u32 = 8;

/// Allocate a `rows x cols` grid of cooperative accumulators, initializing each
/// cell from `init`. The init closure receives `(program, grid_row, grid_col)`
/// and returns the coop-`C` value to seed the accumulator: `coop_zero(..)` for
/// the zero-init case, or a `coop_load_c_broadcast(..)` fragment for the
/// preloaded-C case. Each accumulator is a mutable local seeded through
/// `store_local_coop`.
pub(super) fn coop_acc_grid<Init>(
    program: &mut TileBlock<'_>,
    coop: CoopMatrixToken,
    scalar: ScalarElement,
    rows: u32,
    cols: u32,
    mut init: Init,
) -> Vec<Vec<CoopAcc>>
where
    Init: FnMut(&mut TileBlock<'_>, CoopMatrixToken, u32, u32) -> Tile,
{
    (0..rows)
        .map(|r| {
            (0..cols)
                .map(|c| {
                    let acc = coop.alloc_coop_acc(program, scalar, COOP_DIM, COOP_DIM);
                    let seed = init(program, coop, r, c);
                    coop.store_local_coop(program, &acc, seed);
                    acc
                })
                .collect()
        })
        .collect()
}

/// Allocate a `rows x cols` grid of zero-initialized 8x8 cooperative
/// accumulators. Shared between dense and quantized cooperative matmul.
pub(super) fn zero_coop_acc_grid(
    program: &mut TileBlock<'_>,
    coop: CoopMatrixToken,
    scalar: ScalarElement,
    rows: u32,
    cols: u32,
) -> Vec<Vec<CoopAcc>> {
    coop_acc_grid(program, coop, scalar, rows, cols, |program, coop, _, _| {
        coop.coop_zero(program, scalar, COOP_DIM, COOP_DIM)
    })
}

/// Allocate a `rows x cols` grid of cooperative accumulators seeded from a
/// rank-1 column vector: every accumulator in grid-column `c` is initialized
/// from the C-role broadcast fragment at `col_base + c * 8`. This is the
/// qmatmul preloaded-C path.
pub(super) fn coop_acc_grid_set_c(
    program: &mut TileBlock<'_>,
    coop: CoopMatrixToken,
    vector: &Storage,
    col_base: &Tile,
    scalar: ScalarElement,
    rows: u32,
    cols: u32,
) -> Vec<Vec<CoopAcc>> {
    let c_frags = coop_load_c_broadcast_fragments(program, coop, vector, col_base, cols, scalar);
    coop_acc_grid(
        program,
        coop,
        scalar,
        rows,
        cols,
        |_program, _coop, _r, c| c_frags[c as usize].clone(),
    )
}

/// Cooperatively load `rows` A-role 8x8 fragments from a workgroup tile.
pub(super) fn coop_load_a_fragments(
    program: &TileBlock<'_>,
    coop: CoopMatrixToken,
    tile: &WorkgroupTile,
    sg_row_base: &Tile,
    kk: u32,
    rows: u32,
    scalar: ScalarElement,
) -> Vec<Tile> {
    (0..rows)
        .map(|r| {
            coop.coop_load_a(
                program,
                tile,
                sg_row_base.clone() + r * COOP_DIM,
                kk * COOP_DIM,
                scalar,
                COOP_DIM,
                COOP_DIM,
            )
        })
        .collect()
}

/// Cooperatively load `cols` B-role 8x8 fragments from a workgroup tile.
pub(super) fn coop_load_b_fragments(
    program: &TileBlock<'_>,
    coop: CoopMatrixToken,
    tile: &WorkgroupTile,
    sg_col_base: &Tile,
    kk: u32,
    cols: u32,
    scalar: ScalarElement,
) -> Vec<Tile> {
    (0..cols)
        .map(|c| {
            coop.coop_load_b(
                program,
                tile,
                kk * COOP_DIM,
                sg_col_base.clone() + c * COOP_DIM,
                scalar,
                COOP_DIM,
                COOP_DIM,
            )
        })
        .collect()
}

/// Cooperatively load `cols` C-role fragments from a rank-1 column vector,
/// broadcasting each 8-column slice across the fragment rows.
pub(super) fn coop_load_c_broadcast_fragments(
    program: &TileBlock<'_>,
    coop: CoopMatrixToken,
    vector: &Storage,
    col_base: &Tile,
    cols: u32,
    scalar: ScalarElement,
) -> Vec<Tile> {
    (0..cols)
        .map(|c| {
            coop.coop_load_c_broadcast(
                program,
                vector,
                col_base.clone() + c * COOP_DIM,
                scalar,
                COOP_DIM,
                COOP_DIM,
            )
        })
        .collect()
}

/// MMA every `a_frag` × `b_frag` pair into the matching accumulator. Each cell
/// emits `store_local_coop(acc, coop_mma(a, b, load_local_coop(acc)))`, which
/// the lowerer threads through the coop acc-value SSA memo.
pub(super) fn coop_mma_grid(
    program: &mut TileBlock<'_>,
    coop: CoopMatrixToken,
    accs: &[Vec<CoopAcc>],
    a_frags: &[Tile],
    b_frags: &[Tile],
) {
    for (r, a) in a_frags.iter().enumerate() {
        for (c, b) in b_frags.iter().enumerate() {
            let acc = &accs[r][c];
            let c_value = coop.load_local_coop(program, acc);
            let mma = coop.coop_mma(program, a.clone(), b.clone(), c_value);
            coop.store_local_coop(program, acc, mma);
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
    row: &Tile,
    col: &Tile,
    n_cols: u32,
) -> Tile {
    match extra {
        QmatmulExtra::Column(vector) => program.load(vector.at(col), col.lt(n_cols), 0.0),
        QmatmulExtra::Pointwise(tensor) => program.load(tensor.at((row, col)), col.lt(n_cols), 0.0),
    }
}

/// Cooperatively store an accumulator grid into a rank-2 storage view.
/// `y_batch_base` is added to the row index when `Some` — batched matmul
/// kernels pass the batch row offset; single-batch quantized matmul passes
/// `None`.
#[allow(clippy::too_many_arguments)]
pub(super) fn coop_store_acc_grid(
    program: &mut TileBlock<'_>,
    coop: CoopMatrixToken,
    accs: &[Vec<CoopAcc>],
    y: &Storage,
    y_batch_base: Option<&Tile>,
    row_base: &Tile,
    col_base: &Tile,
    sg_row_base: &Tile,
    sg_col_base: &Tile,
) {
    for (r, row_accs) in accs.iter().enumerate() {
        for (c, acc) in row_accs.iter().enumerate() {
            let local_row = row_base.clone() + sg_row_base.clone() + r as u32 * COOP_DIM;
            let row = match y_batch_base {
                Some(batch) => batch.clone() + local_row,
                None => local_row,
            };
            let col = col_base.clone() + sg_col_base.clone() + c as u32 * COOP_DIM;
            coop.coop_store(program, acc, y, row, col);
        }
    }
}

/// The scalar component of a (possibly vector / coop) element type. Used by the
/// ported coop helpers where the fragment scalar must be recovered from a
/// runtime [`ElementType`].
pub(super) fn scalar_of(element: ElementType) -> ScalarElement {
    match element {
        ElementType::F32 => ScalarElement::F32,
        ElementType::F16 => ScalarElement::F16,
        ElementType::U32 => ScalarElement::U32,
        ElementType::Bool => ScalarElement::Bool,
        ElementType::Vector { scalar, .. } => scalar,
        ElementType::CoopMatrix { scalar, .. } => scalar,
    }
}

/// Zero literal for a float `element` (F32 or F16). Panics on other types —
/// callers (flash / softmax) only ever stage F32 and F16.
pub(super) fn zero_fill(element: ElementType) -> TileLiteral {
    match scalar_of(element) {
        ScalarElement::F32 => TileLiteral::f32(0.0),
        ScalarElement::F16 => TileLiteral::F16(0),
        _ => panic!("only F32 and F16 element types are supported"),
    }
}

/// Whether `element`'s scalar is a float (F32 or F16).
pub(super) fn supports_float(element: ElementType) -> bool {
    matches!(scalar_of(element), ScalarElement::F32 | ScalarElement::F16)
}

/// A `u32` literal tile.
pub(super) fn u32_tile(value: u32) -> Tile {
    Tile::literal(TileLiteral::U32(value))
}
