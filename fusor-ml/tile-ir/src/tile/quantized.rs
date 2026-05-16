use super::block::{f32_fill, tiles_to_exprs};
use super::*;
use crate::ir::{DotK, Expr, PackedActivations};
use crate::quantized::QuantizedMatrix;

/// GGML-specific quantized dot helpers.
pub mod ggml {
    use super::*;

    /// Block-shaped coordinates used by GGML Q4K/Q6K dot kernels.
    ///
    /// These coordinates address a 256-element quantized block, the GGML inner
    /// sub-block indices, and the output column. Most callers get them from a
    /// kernel-specific lane decomposition rather than constructing them by hand.
    pub struct BlockCoords {
        pub(super) block: Box<Expr>,
        pub(super) c0: Box<Expr>,
        pub(super) c1: Box<Expr>,
        pub(super) col: Box<Expr>,
    }

    impl BlockCoords {
        /// Create block coordinates from tile indices.
        pub fn new(
            block: impl IntoIndex,
            c0: impl IntoIndex,
            c1: impl IntoIndex,
            col: impl IntoIndex,
        ) -> Self {
            Self {
                block: block.into_index(),
                c0: c0.into_index(),
                c1: c1.into_index(),
                col: col.into_index(),
            }
        }
    }

    /// Activations prepacked in the layout expected by the optimized GGML Q4K dot.
    ///
    /// `low` and `high` hold the two nibble halves for 16 lanes each. `sums` holds
    /// the activation sums used by the Q4K min-scale correction term.
    #[derive(Clone)]
    pub struct Q4KActivations {
        /// Low-half activation terms.
        pub low: [Tile; 16],
        /// High-half activation terms.
        pub high: [Tile; 16],
        /// Four activation sums for min-scale correction.
        pub sums: [Tile; 4],
    }

    impl Q4KActivations {
        /// Create a Q4K GGML activation pack.
        pub fn new(low: [Tile; 16], high: [Tile; 16], sums: [Tile; 4]) -> Self {
            Self { low, high, sums }
        }
    }
}

/// Format-specific quantized dot expression.
///
/// A `QuantizedDot` is a builder object; pass it to
/// [`TileBlock::quantized_dot`] to emit the tile expression. The constructors
/// encode the activation packing and K-coordinate shape required by the
/// corresponding lowerer.
///
/// ```no_run
/// use fusor_tile_ir::tile::{Mask, QuantizedDot};
/// use fusor_tile_ir::{tile, GgmlQuantFormat, Shape, F32};
///
/// let ir = tile::build(|program| {
///     let a = program.storage_read::<F32, 2>(Shape::new([1, 256]));
///     let b = fusor_tile_ir::QuantizedMatrix {
///         data: program
///             .storage_read::<fusor_tile_ir::U32, 1>(Shape::new([72]))
///             .view()
///             .clone(),
///         format: GgmlQuantFormat::Q8_0,
///         rows: 256,
///         cols: 1,
///     };
///     let y = program.storage_write::<F32, 2>(Shape::new([1, 1]));
///
///     program.program_grid::<8>([1, 1, 1], |block| {
///         let lane = block.lane();
///         let k_base = lane.clone() * 8u32;
///         let activations = std::array::from_fn(|i| {
///             block.load(a.at((0u32, k_base.clone() + i as u32)), Mask::all(), 0.0)
///         });
///         let dot = block.quantized_dot(QuantizedDot::f32_activations(
///             activations,
///             &b,
///             &k_base,
///             0u32,
///             Mask::all(),
///             0.0,
///         ));
///         let sum = block.group_reduce_sum::<8>(dot);
///         block.store(y.at((0u32, 0u32)), sum, lane.eq(0u32));
///     });
/// });
/// # let _ = ir;
/// ```
pub struct QuantizedDot {
    src: QuantizedMatrix,
    activations: PackedActivations,
    k: DotK,
    col: Box<Expr>,
    mask: Box<Expr>,
    fill: Box<Expr>,
    block_n: u32,
}

impl QuantizedDot {
    fn base_dot<const N: usize>(
        a: [Tile; N],
        activations: impl FnOnce(Vec<Expr>) -> PackedActivations,
        matrix: &QuantizedMatrix,
        k_base: impl IntoIndex,
        col: impl IntoIndex,
        mask: Mask,
        fill: f32,
    ) -> Self {
        Self {
            src: matrix.clone(),
            activations: activations(tiles_to_exprs(a)),
            k: DotK::Base(k_base.into_index()),
            col: col.into_index(),
            mask: mask.expr,
            fill: f32_fill(fill),
            block_n: N as u32,
        }
    }

    /// Build a dot over f32 activation values.
    ///
    /// `N` must be 8, 16, or 32.
    pub fn f32_activations<const N: usize>(
        a: [Tile; N],
        matrix: &QuantizedMatrix,
        k_base: impl IntoIndex,
        col: impl IntoIndex,
        mask: Mask,
        fill: f32,
    ) -> Self {
        assert!(
            N == 8 || N == 16 || N == 32,
            "f32 activation dot currently supports N == 8, N == 16, or N == 32"
        );
        Self::base_dot(a, PackedActivations::F32, matrix, k_base, col, mask, fill)
    }

    /// Build a Q4K/Q6K dot over Q8-packed activation values.
    ///
    /// `N` must be 8 or 16.
    pub fn q8_activations<const N: usize>(
        a: [Tile; N],
        matrix: &QuantizedMatrix,
        k_base: impl IntoIndex,
        col: impl IntoIndex,
        mask: Mask,
        fill: f32,
    ) -> Self {
        assert!(
            N == 8 || N == 16,
            "q8 activation dot currently supports N == 8 or N == 16"
        );
        Self::base_dot(a, PackedActivations::Q8, matrix, k_base, col, mask, fill)
    }

    /// Build an optimized GGML-layout Q4K dot.
    pub fn ggml_q4k(
        activations: ggml::Q4KActivations,
        matrix: &QuantizedMatrix,
        coords: ggml::BlockCoords,
        mask: Mask,
        fill: f32,
    ) -> Self {
        let ggml::Q4KActivations { low, high, sums } = activations;
        let ggml::BlockCoords { block, c0, c1, col } = coords;
        Self {
            src: matrix.clone(),
            activations: PackedActivations::Q4KGgml {
                low: tiles_to_exprs(low),
                high: tiles_to_exprs(high),
                sums: tiles_to_exprs(sums),
            },
            k: DotK::Block { block, c0, c1 },
            col,
            mask: mask.expr,
            fill: f32_fill(fill),
            block_n: 32,
        }
    }

    /// Build an optimized GGML-layout Q6K dot.
    pub fn ggml_q6k(
        a: [Tile; 16],
        matrix: &QuantizedMatrix,
        coords: ggml::BlockCoords,
        mask: Mask,
        fill: f32,
    ) -> Self {
        let ggml::BlockCoords { block, c0, c1, col } = coords;
        Self {
            src: matrix.clone(),
            activations: PackedActivations::F32(tiles_to_exprs(a)),
            k: DotK::Block { block, c0, c1 },
            col,
            mask: mask.expr,
            fill: f32_fill(fill),
            block_n: 16,
        }
    }
}

impl TileBlock<'_> {
    /// Emit a quantized dot tile expression.
    pub fn quantized_dot(&self, dot: QuantizedDot) -> Tile {
        Tile {
            expr: Expr::QuantizedDot {
                src: dot.src,
                activations: dot.activations,
                k: dot.k,
                col: dot.col,
                mask: dot.mask,
                fill: dot.fill,
                block_n: dot.block_n,
            },
        }
    }
}
