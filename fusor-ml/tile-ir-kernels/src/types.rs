use std::sync::Arc;

use fusor_tile_ir::tile::{Storage, Tile};
use fusor_tile_ir::{Layout, TileLiteral};

type UnaryEpilogueBuilder = dyn Fn(Tile) -> Tile + Send + Sync;
type UnaryEpilogueWithExtrasBuilder = dyn Fn(&[Tile]) -> Tile + Send + Sync;

/// Single-input tile-IR epilogue, applied between a kernel's per-output
/// reduction and the final store. Used by post-element-wise fusion on
/// `q_mat_mul` / `rms_norm` / etc.
///
/// Pass `None` to the kernels when no epilogue is needed (zero overhead — the
/// kernels' store paths short-circuit on `None`). Construct one via
/// [`UnaryEpilogue::new`] when the resolver detects a post-op chain to fuse;
/// the closure runs at kernel-build time and produces a Tile-IR `Expr` tree
/// that is hashed into the pipeline cache key.
///
/// ```
/// use fusor_tile_ir_kernels::UnaryEpilogue;
///
/// let epilogue = UnaryEpilogue::new("relu", |tile| tile.relu());
/// assert_eq!(epilogue.label(), "relu");
/// ```
#[derive(Clone)]
pub struct UnaryEpilogue {
    label: &'static str,
    identity: u64,
    build: Arc<UnaryEpilogueBuilder>,
}

impl UnaryEpilogue {
    /// Build a unary epilogue from an arbitrary tile-IR closure.
    pub fn new<F>(label: &'static str, build: F) -> Self
    where
        F: Fn(Tile) -> Tile + Send + Sync + 'static,
    {
        let probe = Tile::literal(TileLiteral::f32(f32::from_bits(0x5EED_CA7E)));
        let identity = build(probe).signature_hash();
        Self {
            label,
            identity,
            build: Arc::new(build),
        }
    }

    /// Apply this epilogue to one tile expression.
    pub fn apply(&self, tile: Tile) -> Tile {
        (self.build)(tile)
    }

    /// Stable structural hash of the produced Tile-IR Expr tree.
    pub fn identity(&self) -> u64 {
        self.identity
    }

    /// Human-readable label for graph visualization and kernel names.
    pub fn label(&self) -> &'static str {
        self.label
    }
}

impl std::fmt::Debug for UnaryEpilogue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnaryEpilogue")
            .field("label", &self.label)
            .field("identity", &format_args!("{:#018x}", self.identity))
            .finish()
    }
}

impl PartialEq for UnaryEpilogue {
    fn eq(&self, other: &Self) -> bool {
        self.identity == other.identity
    }
}

impl Eq for UnaryEpilogue {}

impl std::hash::Hash for UnaryEpilogue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.identity.hash(state);
    }
}

#[derive(Clone)]
pub struct UnaryEpilogueWithExtras {
    label: &'static str,
    value_arity: usize,
    extras_arity: usize,
    identity: u64,
    build: Arc<UnaryEpilogueWithExtrasBuilder>,
}

impl UnaryEpilogueWithExtras {
    pub fn new<F>(label: &'static str, extras_arity: usize, build: F) -> Self
    where
        F: Fn(&[Tile]) -> Tile + Send + Sync + 'static,
    {
        Self::new_with_value_arity(label, 1, extras_arity, build)
    }

    pub fn new_with_value_arity<F>(
        label: &'static str,
        value_arity: usize,
        extras_arity: usize,
        build: F,
    ) -> Self
    where
        F: Fn(&[Tile]) -> Tile + Send + Sync + 'static,
    {
        assert!(value_arity > 0, "epilogue must consume at least one value");
        let mut values = Vec::with_capacity(value_arity + extras_arity);
        values.extend((0..value_arity).map(|idx| {
            Tile::literal(TileLiteral::f32(f32::from_bits(
                0x5EED_CA7Eu32.wrapping_add(idx as u32),
            )))
        }));
        values.extend((0..extras_arity).map(|idx| {
            Tile::literal(TileLiteral::f32(f32::from_bits(
                0x51A7_0000u32.wrapping_add(idx as u32),
            )))
        }));
        let identity = build(&values).signature_hash();
        Self {
            label,
            value_arity,
            extras_arity,
            identity,
            build: Arc::new(build),
        }
    }

    pub fn apply(&self, values: &[Tile]) -> Tile {
        assert_eq!(values.len(), self.value_arity + self.extras_arity);
        (self.build)(values)
    }

    pub fn identity(&self) -> u64 {
        self.identity
    }

    pub fn label(&self) -> &'static str {
        self.label
    }

    pub fn extras_arity(&self) -> usize {
        self.extras_arity
    }

    pub fn value_arity(&self) -> usize {
        self.value_arity
    }
}

/// Apply the optional epilogue to a tile. Identity (no allocation, no
/// dispatch) when `epilogue` is `None`. Kernels call this between their
/// per-output reduce and the store.
pub(crate) fn apply_optional_epilogue(epilogue: Option<&UnaryEpilogue>, tile: Tile) -> Tile {
    match epilogue {
        Some(ep) => ep.apply(tile),
        None => tile,
    }
}

pub(crate) fn apply_epilogue_with_extras(
    epilogue: Option<&UnaryEpilogueWithExtras>,
    tile: Tile,
    extras: Vec<Tile>,
) -> Tile {
    apply_epilogue_values_with_extras(epilogue, vec![tile], extras)
}

pub(crate) fn apply_epilogue_values_with_extras(
    epilogue: Option<&UnaryEpilogueWithExtras>,
    values: Vec<Tile>,
    extras: Vec<Tile>,
) -> Tile {
    if let Some(epilogue) = epilogue {
        let mut values = values;
        values.extend(extras);
        epilogue.apply(&values)
    } else {
        assert_eq!(values.len(), 1);
        values.into_iter().next().expect("single value")
    }
}

/// Bundle of pre- and post-reduce epilogues for dense F32 matmul kernels.
#[derive(Clone, Default)]
pub struct DenseMatmulEpilogues<'a> {
    /// Optional transform applied to each loaded lhs value before the product.
    pub pre_a: Option<&'a UnaryEpilogue>,
    /// Optional transform applied to each loaded rhs value before the product.
    pub pre_b: Option<&'a UnaryEpilogue>,
    /// Optional transform applied after the reduction and before the store.
    pub post: Option<&'a UnaryEpilogue>,
}

impl<'a> DenseMatmulEpilogues<'a> {
    /// No dense matmul epilogues.
    pub fn empty() -> Self {
        Self::default()
    }
}

/// Bundle of pre- and post-reduce epilogues for `qgemv` / `qmatmul` kernels.
/// `pre` is applied to each loaded activation tile before the dot product;
/// `post` is applied to each per-output reduced tile before the store. Either
/// may be `None`, in which case the kernel skips that injection point.
#[derive(Clone, Default)]
pub struct QmatmulEpilogues<'a> {
    /// Optional activation transform applied before each dot product.
    pub pre: Option<&'a UnaryEpilogue>,
    /// Optional activation transform that consumes the loaded activation plus
    /// ordered extra inputs.
    pub pre_with_extras: Option<&'a UnaryEpilogueWithExtras>,
    /// Ordered extra inputs passed after the activation tile to
    /// `pre_with_extras`.
    pub pre_extra_inputs: &'a [QmatmulExtra<'a>],
    /// Optional output transform applied after the reduction.
    pub post: Option<&'a UnaryEpilogue>,
    /// Optional output transform that consumes the reduced output plus
    /// per-column extra vectors.
    pub post_with_extras: Option<&'a UnaryEpilogueWithExtras>,
    /// Ordered extra inputs passed after the reduced output tile to
    /// `post_with_extras`.
    pub post_extra_inputs: &'a [QmatmulExtra<'a>],
    /// Matrix-column offsets for accumulator values passed to the post
    /// epilogue. Empty means the default single accumulator at output column
    /// `j`. Non-empty values make qgemv compute `acc(j + offset)` for each
    /// offset and pass those accumulators before `post_extra_inputs`.
    pub post_accumulator_offsets: &'a [u32],
    /// Optional rank-1 vector that is added to the accumulator before the
    /// cooperative store. This is a lowering choice for expressions whose
    /// post-op can be represented as `acc + column_vector`. Runtime-typed
    /// (ARBOR_DESIGN.md §2): the rank/element travel in the `Storage` view.
    pub post_acc_init_col_vector: Option<&'a Storage>,
}

#[derive(Clone, Copy)]
pub enum QmatmulExtra<'a> {
    /// Rank-1 f32 vector indexed by input/output column.
    Column(&'a Storage),
    /// Rank-2 f32 tensor indexed pointwise by the qmatmul dispatch row/column.
    Pointwise(&'a Storage),
}

impl<'a> QmatmulEpilogues<'a> {
    /// No qmatmul epilogues.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Only a post-reduce epilogue.
    pub fn post(post: &'a UnaryEpilogue) -> Self {
        Self {
            pre: None,
            pre_with_extras: None,
            pre_extra_inputs: &[],
            post: Some(post),
            post_with_extras: None,
            post_extra_inputs: &[],
            post_accumulator_offsets: &[],
            post_acc_init_col_vector: None,
        }
    }

    /// Only a pre-dot epilogue.
    pub fn pre(pre: &'a UnaryEpilogue) -> Self {
        Self {
            pre: Some(pre),
            pre_with_extras: None,
            pre_extra_inputs: &[],
            post: None,
            post_with_extras: None,
            post_extra_inputs: &[],
            post_accumulator_offsets: &[],
            post_acc_init_col_vector: None,
        }
    }

    pub fn post_accumulator_offsets(&self) -> &[u32] {
        const DEFAULT: &[u32] = &[0];
        if self.post_accumulator_offsets.is_empty() {
            DEFAULT
        } else {
            self.post_accumulator_offsets
        }
    }

    pub fn post_value_arity(&self) -> usize {
        self.post_accumulator_offsets().len()
    }

    pub fn post_output_cols(&self, matrix_cols: u32) -> u32 {
        let max_offset = self
            .post_accumulator_offsets()
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        matrix_cols.saturating_sub(max_offset)
    }
}

pub(crate) fn apply_qmatmul_pre_epilogue(
    epilogues: &QmatmulEpilogues<'_>,
    tile: Tile,
    extras: Vec<Tile>,
) -> Tile {
    if epilogues.pre_with_extras.is_some() {
        apply_epilogue_with_extras(epilogues.pre_with_extras, tile, extras)
    } else {
        apply_optional_epilogue(epilogues.pre, tile)
    }
}

pub(crate) fn apply_qmatmul_post_epilogue(
    epilogues: &QmatmulEpilogues<'_>,
    tile: Tile,
    extras: Vec<Tile>,
) -> Tile {
    apply_qmatmul_post_epilogue_values(epilogues, vec![tile], extras)
}

pub(crate) fn apply_qmatmul_post_epilogue_values(
    epilogues: &QmatmulEpilogues<'_>,
    values: Vec<Tile>,
    extras: Vec<Tile>,
) -> Tile {
    if epilogues.post_with_extras.is_some() {
        apply_epilogue_values_with_extras(epilogues.post_with_extras, values, extras)
    } else {
        assert_eq!(values.len(), 1);
        apply_optional_epilogue(
            epilogues.post,
            values.into_iter().next().expect("single value"),
        )
    }
}

pub(crate) fn matrix_shape(layout: &Layout) -> [u32; 2] {
    assert_eq!(layout.shape().rank(), 2, "matrix operands must be rank-2");
    [
        layout.shape().dims()[0].get(),
        layout.shape().dims()[1].get(),
    ]
}

pub(crate) fn cooperative_store_layout_supported(layout: &Layout) -> bool {
    if !layout.is_affine() || layout.shape().rank() != 2 {
        return false;
    }
    let strides = layout.affine_strides();
    strides[0] == 1 || strides[1] == 1
}
