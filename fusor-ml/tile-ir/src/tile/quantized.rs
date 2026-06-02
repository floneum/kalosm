//! Quantized dequant-to-tiles composition.
//!
//! The over-fused multi-format `QuantizedDot`/`PackedActivations`/`DotK` op
//! surface is gone: the composable primitive is `Expr::Dequantize` (one node
//! producing all N lanes), projected per-lane with `Expr::LaneOf` over a single
//! `Expr::Shared`. The kernel then composes an ordinary `Dot`.
//!
//! The one quant dot that composition *cannot* express is the Q8 DP4a fast path
//! ([`quantized_dot_q8`](TileBlock::quantized_dot_q8)): it keeps the weights
//! quantized and dots them against int8-packed activations via `Dot4I8Packed`,
//! which would be lost if the weights were dequantized to f32 first. It stays an
//! irreducible `Expr::QuantizedDot` primitive.

use super::value::boxed_index;
use super::{Mask, Tile, TileBlock};
use crate::ir::{ElementType, Expr, ExprKind, QuantActivation, TileLiteral};
use crate::quantized::QuantizedMatrix;

impl TileBlock<'_> {
    /// Dequantize a block to `lanes` f32 tiles.
    ///
    /// Builds **one** `Expr::Shared(Dequantize { lanes, .. })` and clones that
    /// single shared node into `lanes` `LaneOf { block, lane }` projections, so
    /// the lowerer emits the dequant (and its scale lookup) exactly once and
    /// reuses the N lane handles. `lanes` carries the caller's `values_per_lane`
    /// tiling choice.
    pub fn load_quantized_block_vec(
        &mut self,
        lanes: u32,
        matrix: &QuantizedMatrix,
        k_base: impl Into<Tile>,
        col: impl Into<Tile>,
        mask: impl Into<Mask>,
        fill: f32,
    ) -> Vec<Tile> {
        assert!(
            lanes == 8 || lanes == 16,
            "block dequant supports 8 or 16 lanes"
        );
        let shared = Expr::new(
            ExprKind::Shared(Expr::new(
                ExprKind::Dequantize {
                    src: matrix.clone(),
                    k_base: boxed_index(k_base),
                    col: boxed_index(col),
                    mask: Box::new(mask.into().into_expr()),
                    fill: Box::new(Expr::new(
                        ExprKind::Literal(TileLiteral::f32(fill)),
                        ElementType::F32,
                    )),
                    lanes,
                },
                ElementType::F32,
            )),
            ElementType::F32,
        );
        (0..lanes)
            .map(|lane| {
                Tile::new(
                    ExprKind::LaneOf {
                        block: Box::new(shared.clone()),
                        lane,
                    },
                    ElementType::F32,
                )
            })
            .collect()
    }

    /// Fused per-column dot of f32 `activations` against a quantized `matrix`
    /// block (the weights are decoded to f32 once and accumulated directly).
    /// More compact than `load_quantized_block_vec` + `dot4_sum`, which
    /// re-decodes the block scale per lane. The lowerer supports Q8_0/Q6K dot8
    /// and Q4K dot8/16/32.
    pub fn quantized_dot_f32(
        &mut self,
        activations: &[Tile],
        matrix: &QuantizedMatrix,
        k_base: impl Into<Tile>,
        col: impl Into<Tile>,
        mask: impl Into<Mask>,
        fill: f32,
    ) -> Tile {
        self.quantized_dot(
            QuantActivation::F32,
            activations,
            matrix,
            k_base,
            col,
            mask,
            fill,
        )
    }

    /// Q8 DP4a dot of f32 `activations` against a quantized `matrix` block,
    /// keeping the weights quantized (the lowerer packs the activations to int8
    /// and emits `Dot4I8Packed`). This is the irreducible fast path that
    /// `load_quantized_block_vec` + `dot4_sum` cannot express — dequantizing the
    /// weights to f32 would drop the integer dot-product. Currently the lowerer
    /// supports the Q6K family.
    pub fn quantized_dot_q8(
        &mut self,
        activations: &[Tile],
        matrix: &QuantizedMatrix,
        k_base: impl Into<Tile>,
        col: impl Into<Tile>,
        mask: impl Into<Mask>,
        fill: f32,
    ) -> Tile {
        self.quantized_dot(
            QuantActivation::Q8,
            activations,
            matrix,
            k_base,
            col,
            mask,
            fill,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn quantized_dot(
        &mut self,
        packing: QuantActivation,
        activations: &[Tile],
        matrix: &QuantizedMatrix,
        k_base: impl Into<Tile>,
        col: impl Into<Tile>,
        mask: impl Into<Mask>,
        fill: f32,
    ) -> Tile {
        Tile::new(
            ExprKind::QuantizedDot {
                src: matrix.clone(),
                packing,
                activations: activations.iter().map(|a| a.clone().into_expr()).collect(),
                k_base: boxed_index(k_base),
                col: boxed_index(col),
                mask: Box::new(mask.into().into_expr()),
                fill: Box::new(Expr::new(
                    ExprKind::Literal(TileLiteral::f32(fill)),
                    ElementType::F32,
                )),
            },
            ElementType::F32,
        )
    }
}
