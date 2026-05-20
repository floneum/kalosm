use std::sync::Arc;

use fusor_tile_ir::tile::{Storage, Tile};
use fusor_tile_ir::{Layout, TileLiteral, F32};

type PairedEpilogueBuilder = dyn Fn(&[Tile]) -> Tile + Send + Sync;
type UnaryEpilogueBuilder = dyn Fn(Tile) -> Tile + Send + Sync;
type UnaryEpilogueWithExtrasBuilder = dyn Fn(&[Tile]) -> Tile + Send + Sync;

/// Paired matmul epilogue. The matmul produces concatenated `[gate; up]`
/// columns; the kernel reduces each pair separately and applies this epilogue
/// before storing a single output column per pair.
///
/// Constructed exclusively by the resolver's paired-fusion rule when it
/// detects a `q_mat_mul → narrow → … → mul(narrow)` subgraph; the closure
/// re-emits the captured `NaryExpr` at the tile-IR level. Pipelines are
/// cached by the structural hash of the produced Expr tree.
///
/// ```
/// use fusor_tile_ir_kernels::PairedEpilogue;
///
/// let epilogue =
///     PairedEpilogue::with_extras("mul", 0, |tiles| tiles[0].clone() * tiles[1].clone());
/// assert_eq!(epilogue.arity(), 2);
/// ```
#[derive(Clone)]
pub struct PairedEpilogue {
    label: &'static str,
    identity: u64,
    /// Arity of the closure: `2 + extras`. Always `>= 2` — slot 0 is the gate
    /// tile, slot 1 is the up tile, slots 2.. are per-column broadcast tiles
    /// loaded by the kernel from the corresponding entries in
    /// `extra_inputs`.
    arity: usize,
    // The closure receives a slice of `arity` block-agnostic tile expressions.
    build: Arc<PairedEpilogueBuilder>,
}

impl PairedEpilogue {
    /// Build a paired epilogue with `extras_arity` additional per-column
    /// inputs beyond `(gate, up)`. The closure receives a slice of
    /// `2 + extras_arity` tiles; slot 0 is gate, slot 1 is up, slots
    /// `2..2+extras_arity` are the per-column extras in the order the
    /// resolver collected them.
    pub fn with_extras<F>(label: &'static str, extras_arity: usize, build: F) -> Self
    where
        F: Fn(&[Tile]) -> Tile + Send + Sync + 'static,
    {
        let arity = 2 + extras_arity;
        // Probe the closure with `arity` distinguishable placeholder tiles so
        // commutative differences (`gate * up` vs `up * gate`) and distinct
        // extras yield distinct structural hashes.
        let probes: Vec<Tile> = (0..arity)
            .map(|i| {
                let bits = 0xDEAD_0000u32 ^ (i as u32).wrapping_mul(0x9E37_79B9);
                Tile::literal(TileLiteral::f32(f32::from_bits(bits)))
            })
            .collect();
        let identity = build(&probes).signature_hash();
        Self {
            label,
            identity,
            arity,
            build: Arc::new(build),
        }
    }

    /// Number of input tiles this epilogue takes (always `>= 2`).
    pub fn arity(&self) -> usize {
        self.arity
    }

    /// Number of per-column extra inputs (arity - 2). The qgemv kernel must
    /// load exactly this many extras into the slice passed to the closure.
    pub fn extras_count(&self) -> usize {
        self.arity - 2
    }

    /// Build the per-output tile expression for this epilogue. The kernel
    /// must pass exactly `extras_count()` extra tiles; passing the wrong
    /// number is a programming error caught by `debug_assert`.
    pub fn apply(&self, gate: Tile, up: Tile, extras: &[Tile]) -> Tile {
        debug_assert_eq!(
            extras.len(),
            self.extras_count(),
            "paired epilogue extras count mismatch"
        );
        let mut tiles: Vec<Tile> = Vec::with_capacity(self.arity);
        tiles.push(gate);
        tiles.push(up);
        for extra in extras {
            tiles.push(extra.clone());
        }
        (self.build)(&tiles)
    }

    /// Stable structural hash of the produced Tile-IR Expr tree. Mix into
    /// pipeline cache keys so distinct epilogues do not alias.
    pub fn identity(&self) -> u64 {
        self.identity
    }

    /// Human-readable label for graph visualization and kernel names.
    pub fn label(&self) -> &'static str {
        self.label
    }
}

impl std::fmt::Debug for PairedEpilogue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PairedEpilogue")
            .field("label", &self.label)
            .field("identity", &format_args!("{:#018x}", self.identity))
            .finish()
    }
}

impl PartialEq for PairedEpilogue {
    fn eq(&self, other: &Self) -> bool {
        self.identity == other.identity
    }
}

impl Eq for PairedEpilogue {}

impl std::hash::Hash for PairedEpilogue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.identity.hash(state);
    }
}

/// Single-input tile-IR epilogue, applied between a kernel's per-output
/// reduction and the final store. Used by post-element-wise fusion on
/// `q_mat_mul` / `rms_norm` / etc. Mirrors [`PairedEpilogue`] but for the
/// one-output case (`act(value) -> out` rather than `act(gate, up) -> out`).
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
    extras_arity: usize,
    identity: u64,
    build: Arc<UnaryEpilogueWithExtrasBuilder>,
}

impl UnaryEpilogueWithExtras {
    pub fn new<F>(label: &'static str, extras_arity: usize, build: F) -> Self
    where
        F: Fn(&[Tile]) -> Tile + Send + Sync + 'static,
    {
        let mut values = Vec::with_capacity(1 + extras_arity);
        values.push(Tile::literal(TileLiteral::f32(f32::from_bits(0x5EED_CA7E))));
        values.extend((0..extras_arity).map(|idx| {
            Tile::literal(TileLiteral::f32(f32::from_bits(
                0x51A7_0000u32.wrapping_add(idx as u32),
            )))
        }));
        let identity = build(&values).signature_hash();
        Self {
            label,
            extras_arity,
            identity,
            build: Arc::new(build),
        }
    }

    pub fn apply(&self, values: &[Tile]) -> Tile {
        assert_eq!(values.len(), 1 + self.extras_arity);
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
    if let Some(epilogue) = epilogue {
        let mut values = Vec::with_capacity(1 + extras.len());
        values.push(tile);
        values.extend(extras);
        epilogue.apply(&values)
    } else {
        tile
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
    /// per-input-column extra vectors.
    pub pre_with_extras: Option<&'a UnaryEpilogueWithExtras>,
    /// Rank-1 vectors indexed by input column and passed after the activation
    /// tile to `pre_with_extras`.
    pub pre_extra_col_vectors: &'a [Storage<F32, 1>],
    /// Optional output transform applied after the reduction.
    pub post: Option<&'a UnaryEpilogue>,
    /// Optional output transform that consumes the reduced output plus
    /// per-column extra vectors.
    pub post_with_extras: Option<&'a UnaryEpilogueWithExtras>,
    /// Rank-1 vectors indexed by output column and passed after the reduced
    /// output tile to `post_with_extras`.
    pub post_extra_col_vectors: &'a [Storage<F32, 1>],
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
            pre_extra_col_vectors: &[],
            post: Some(post),
            post_with_extras: None,
            post_extra_col_vectors: &[],
        }
    }

    /// Only a pre-dot epilogue.
    pub fn pre(pre: &'a UnaryEpilogue) -> Self {
        Self {
            pre: Some(pre),
            pre_with_extras: None,
            pre_extra_col_vectors: &[],
            post: None,
            post_with_extras: None,
            post_extra_col_vectors: &[],
        }
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
    if epilogues.post_with_extras.is_some() {
        apply_epilogue_with_extras(epilogues.post_with_extras, tile, extras)
    } else {
        apply_optional_epilogue(epilogues.post, tile)
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
