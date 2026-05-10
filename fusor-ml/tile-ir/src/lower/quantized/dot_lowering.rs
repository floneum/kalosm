//! Trait dispatch for the quantized dot-product `Expr` variants.
//!
//! Each variant in `Expr` that lowers to a fused quantized dot product
//! (`QuantizedQ8_0Dot8`, `QuantizedVecDot`, `QuantizedQ4KGgmlDot`,
//! `QuantizedQ6KGgmlDot`) implements [`QuantizedDotLowering`]. The match arm
//! in `lower_tile_expr_lane` only has to package the lowering context and
//! call [`QuantizedDotLowering::lower`]; the format-specific activation
//! shape lives on the impl struct, and each impl forwards into the
//! pre-existing per-format helper on `Lowerer`.
//!
//! The format-specific bit math (Q4K block-scale unpacking, Q6K trellis
//! dequant, Q8_0 lookup) stays in the helpers — only the dispatch shape
//! changes.

use naga::{Arena, Block, Expression, Handle};

use super::super::{Lowerer, LowerError, ScratchLocals};
use crate::ir::{F32Bits, QuantizedVecDotKind, Expr};
use crate::quantized::QuantizedMatrix;

/// Context the lowerer threads through when materialising a dot expression.
pub(in crate::lower) struct DotLoweringCtx<'b, 'e, 's> {
    pub(in crate::lower) expressions: &'e mut Arena<Expression>,
    pub(in crate::lower) scratch: ScratchLocals,
    pub(in crate::lower) body: &'b mut Block,
    pub(in crate::lower) spill_depth: usize,
    pub(in crate::lower) src: &'s QuantizedMatrix,
    pub(in crate::lower) col: &'s Expr,
    pub(in crate::lower) mask: &'s Expr,
    pub(in crate::lower) fill: F32Bits,
}

/// Common interface for the per-format quantized dot lowerings.
///
/// Each impl wraps an existing per-format helper on `Lowerer`. The trait
/// itself is purely about collapsing the match-arm dispatch in
/// `lower_tile_expr_lane`.
pub(in crate::lower) trait QuantizedDotLowering {
    fn lower(
        self,
        lowerer: &Lowerer<'_>,
        ctx: DotLoweringCtx<'_, '_, '_>,
    ) -> Result<Handle<Expression>, LowerError>;
}

/// Eight-wide Q8_0 (and Q6K-as-Q8) dot product. The format on the matrix
/// determines which inner helper runs.
pub(in crate::lower) struct Q8_0Dot8Lowering<'s> {
    pub(in crate::lower) a: &'s [Box<Expr>; 8],
    pub(in crate::lower) k_base: &'s Expr,
}

impl<'s> QuantizedDotLowering for Q8_0Dot8Lowering<'s> {
    fn lower(
        self,
        lowerer: &Lowerer<'_>,
        ctx: DotLoweringCtx<'_, '_, '_>,
    ) -> Result<Handle<Expression>, LowerError> {
        lowerer.lower_tile_quantized_q8_0_dot8_expr(
            ctx.expressions,
            ctx.scratch,
            ctx.body,
            self.a,
            ctx.src,
            self.k_base,
            ctx.col,
            ctx.mask,
            ctx.fill,
            ctx.src.format,
            ctx.spill_depth,
        )
    }
}

/// Variable-width quantized vector dot. `kind` selects the activation packing
/// strategy; `block_n` selects the inner kernel.
pub(in crate::lower) struct VecDotLowering<'s> {
    pub(in crate::lower) kind: QuantizedVecDotKind,
    pub(in crate::lower) a: &'s [Box<Expr>],
    pub(in crate::lower) k_base: &'s Expr,
    pub(in crate::lower) block_n: u32,
}

impl<'s> QuantizedDotLowering for VecDotLowering<'s> {
    fn lower(
        self,
        lowerer: &Lowerer<'_>,
        ctx: DotLoweringCtx<'_, '_, '_>,
    ) -> Result<Handle<Expression>, LowerError> {
        lowerer.lower_tile_quantized_vec_dot_expr(
            ctx.expressions,
            ctx.scratch,
            ctx.body,
            self.kind,
            self.a,
            ctx.src,
            self.k_base,
            ctx.col,
            ctx.mask,
            ctx.fill,
            self.block_n,
            ctx.spill_depth,
        )
    }
}

/// Q4K-specific dot. Carries the split low/high activation halves and the
/// per-quad sums alongside the Q4K (`block`, `iq`, `ir`) coordinate triple.
pub(in crate::lower) struct Q4KGgmlDotLowering<'s> {
    pub(in crate::lower) a_low: &'s [Box<Expr>],
    pub(in crate::lower) a_high: &'s [Box<Expr>],
    pub(in crate::lower) sums: &'s [Box<Expr>],
    pub(in crate::lower) block: &'s Expr,
    pub(in crate::lower) iq: &'s Expr,
    pub(in crate::lower) ir: &'s Expr,
}

impl<'s> QuantizedDotLowering for Q4KGgmlDotLowering<'s> {
    fn lower(
        self,
        lowerer: &Lowerer<'_>,
        ctx: DotLoweringCtx<'_, '_, '_>,
    ) -> Result<Handle<Expression>, LowerError> {
        lowerer.lower_tile_quantized_q4k_ggml_dot_expr(
            ctx.expressions,
            ctx.scratch,
            ctx.body,
            self.a_low,
            self.a_high,
            self.sums,
            ctx.src,
            self.block,
            self.iq,
            self.ir,
            ctx.col,
            ctx.mask,
            ctx.fill,
            ctx.spill_depth,
        )
    }
}

/// Q6K-specific dot. Carries the activation tile and the Q6K (`block`, `ip`,
/// `il`) coordinate triple.
pub(in crate::lower) struct Q6KGgmlDotLowering<'s> {
    pub(in crate::lower) a: &'s [Box<Expr>],
    pub(in crate::lower) block: &'s Expr,
    pub(in crate::lower) ip: &'s Expr,
    pub(in crate::lower) il: &'s Expr,
}

impl<'s> QuantizedDotLowering for Q6KGgmlDotLowering<'s> {
    fn lower(
        self,
        lowerer: &Lowerer<'_>,
        ctx: DotLoweringCtx<'_, '_, '_>,
    ) -> Result<Handle<Expression>, LowerError> {
        lowerer.lower_tile_quantized_q6k_ggml_dot_expr(
            ctx.expressions,
            ctx.scratch,
            ctx.body,
            self.a,
            ctx.src,
            self.block,
            self.ip,
            self.il,
            ctx.col,
            ctx.mask,
            ctx.fill,
            ctx.spill_depth,
        )
    }
}
