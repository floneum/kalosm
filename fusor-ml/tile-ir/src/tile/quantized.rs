use super::block::{f32_fill, tiles_to_exprs};
use super::value::boxed_index;
use super::*;
use crate::ir::{DotK, Expr, PackedActivations};
use crate::quantized::QuantizedMatrix;

/// Format-neutral quantized dot expression.
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
///         let activations: [_; 8] = std::array::from_fn(|i| {
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
        k_base: impl Into<Tile>,
        col: impl Into<Tile>,
        mask: impl Into<Mask>,
        fill: f32,
    ) -> Self {
        Self {
            src: matrix.clone(),
            activations: activations(tiles_to_exprs(a)),
            k: DotK::Base(boxed_index(k_base)),
            col: boxed_index(col),
            mask: mask.into().expr,
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
        k_base: impl Into<Tile>,
        col: impl Into<Tile>,
        mask: impl Into<Mask>,
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
        k_base: impl Into<Tile>,
        col: impl Into<Tile>,
        mask: impl Into<Mask>,
        fill: f32,
    ) -> Self {
        assert!(
            N == 8 || N == 16,
            "q8 activation dot currently supports N == 8 or N == 16"
        );
        Self::base_dot(a, PackedActivations::Q8, matrix, k_base, col, mask, fill)
    }

    /// Build a block-coordinate Q4K dot from prepacked activation terms.
    pub fn q4k_block(
        low: [Tile; 16],
        high: [Tile; 16],
        sums: [Tile; 4],
        matrix: &QuantizedMatrix,
        block: impl Into<Tile>,
        c0: impl Into<Tile>,
        c1: impl Into<Tile>,
        col: impl Into<Tile>,
        mask: impl Into<Mask>,
        fill: f32,
    ) -> Self {
        Self {
            src: matrix.clone(),
            activations: PackedActivations::Q4KGgml {
                low: tiles_to_exprs(low),
                high: tiles_to_exprs(high),
                sums: tiles_to_exprs(sums),
            },
            k: DotK::Block {
                block: boxed_index(block),
                c0: boxed_index(c0),
                c1: boxed_index(c1),
            },
            col: boxed_index(col),
            mask: mask.into().expr,
            fill: f32_fill(fill),
            block_n: 32,
        }
    }

    /// Build a block-coordinate Q6K dot.
    pub fn q6k_block(
        a: [Tile; 16],
        matrix: &QuantizedMatrix,
        block: impl Into<Tile>,
        c0: impl Into<Tile>,
        c1: impl Into<Tile>,
        col: impl Into<Tile>,
        mask: impl Into<Mask>,
        fill: f32,
    ) -> Self {
        Self {
            src: matrix.clone(),
            activations: PackedActivations::F32(tiles_to_exprs(a)),
            k: DotK::Block {
                block: boxed_index(block),
                c0: boxed_index(c0),
                c1: boxed_index(c1),
            },
            col: boxed_index(col),
            mask: mask.into().expr,
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
