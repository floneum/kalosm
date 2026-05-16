use super::block::{f32_fill, tiles_to_exprs};
use super::*;
use crate::ir::{DotK, Expr, PackedActivations};
use crate::quantized::QuantizedMatrix;

/// Block-shaped coordinates used by GGML Q4K/Q6K dot kernels.
///
/// These coordinates address a 256-element quantized block, the GGML inner
/// sub-block indices, and the output column. Most callers get them from a
/// kernel-specific lane decomposition rather than constructing them by hand.
pub struct GgmlBlockCoords {
    block: Box<Expr>,
    c0: Box<Expr>,
    c1: Box<Expr>,
    col: Box<Expr>,
}

impl GgmlBlockCoords {
    /// Create block coordinates from tile indices.
    pub fn new<const BLOCK: usize>(
        block: impl IntoIndex<BLOCK>,
        c0: impl IntoIndex<BLOCK>,
        c1: impl IntoIndex<BLOCK>,
        col: impl IntoIndex<BLOCK>,
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
pub struct Q4KGgmlActivations<const BLOCK: usize> {
    /// Low-half activation terms.
    pub low: [Tile<BLOCK>; 16],
    /// High-half activation terms.
    pub high: [Tile<BLOCK>; 16],
    /// Four activation sums for min-scale correction.
    pub sums: [Tile<BLOCK>; 4],
}

impl<const BLOCK: usize> Q4KGgmlActivations<BLOCK> {
    /// Create a Q4K GGML activation pack.
    pub fn new(low: [Tile<BLOCK>; 16], high: [Tile<BLOCK>; 16], sums: [Tile<BLOCK>; 4]) -> Self {
        Self { low, high, sums }
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
///         let dot = block.quantized_dot(QuantizedDot::q8_0_dot8(
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
pub struct QuantizedDot<const BLOCK: usize> {
    src: QuantizedMatrix,
    activations: PackedActivations,
    k: DotK,
    col: Box<Expr>,
    mask: Box<Expr>,
    fill: Box<Expr>,
    block_n: u32,
}

impl<const BLOCK: usize> QuantizedDot<BLOCK> {
    fn base_dot<const N: usize>(
        a: [Tile<BLOCK>; N],
        activations: impl FnOnce(Vec<Expr>) -> PackedActivations,
        matrix: &QuantizedMatrix,
        k_base: impl IntoIndex<BLOCK>,
        col: impl IntoIndex<BLOCK>,
        mask: Mask<BLOCK>,
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

    /// Build a Q8_0 dot over eight f32 activation values.
    pub fn q8_0_dot8(
        a: [Tile<BLOCK>; 8],
        matrix: &QuantizedMatrix,
        k_base: impl IntoIndex<BLOCK>,
        col: impl IntoIndex<BLOCK>,
        mask: Mask<BLOCK>,
        fill: f32,
    ) -> Self {
        Self::base_dot(a, PackedActivations::F32, matrix, k_base, col, mask, fill)
    }

    /// Build a Q4K/Q6K dot over Q8-packed activation values.
    ///
    /// `N` must be 8 or 16.
    pub fn q8_activation<const N: usize>(
        a: [Tile<BLOCK>; N],
        matrix: &QuantizedMatrix,
        k_base: impl IntoIndex<BLOCK>,
        col: impl IntoIndex<BLOCK>,
        mask: Mask<BLOCK>,
        fill: f32,
    ) -> Self {
        assert!(
            N == 8 || N == 16,
            "q8 activation dot currently supports N == 8 or N == 16"
        );
        Self::base_dot(a, PackedActivations::Q8, matrix, k_base, col, mask, fill)
    }

    /// Build a Q4K dot over f32 activation values.
    ///
    /// `N` must be 8, 16, or 32.
    pub fn q4k_f32<const N: usize>(
        a: [Tile<BLOCK>; N],
        matrix: &QuantizedMatrix,
        k_base: impl IntoIndex<BLOCK>,
        col: impl IntoIndex<BLOCK>,
        mask: Mask<BLOCK>,
        fill: f32,
    ) -> Self {
        assert!(
            N == 8 || N == 16 || N == 32,
            "q4k f32 dot currently supports N == 8, N == 16, or N == 32"
        );
        Self::base_dot(a, PackedActivations::F32, matrix, k_base, col, mask, fill)
    }

    /// Build an optimized GGML-layout Q4K dot.
    pub fn q4k_ggml(
        activations: Q4KGgmlActivations<BLOCK>,
        matrix: &QuantizedMatrix,
        coords: GgmlBlockCoords,
        mask: Mask<BLOCK>,
        fill: f32,
    ) -> Self {
        let Q4KGgmlActivations { low, high, sums } = activations;
        let GgmlBlockCoords { block, c0, c1, col } = coords;
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
    pub fn q6k_ggml(
        a: [Tile<BLOCK>; 16],
        matrix: &QuantizedMatrix,
        coords: GgmlBlockCoords,
        mask: Mask<BLOCK>,
        fill: f32,
    ) -> Self {
        let GgmlBlockCoords { block, c0, c1, col } = coords;
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

impl<const BLOCK: usize> TileBlock<'_, BLOCK> {
    /// Emit a quantized dot tile expression.
    pub fn quantized_dot(&self, dot: QuantizedDot<BLOCK>) -> Tile<BLOCK> {
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
