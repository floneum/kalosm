use super::*;

#[derive(Clone, Copy)]
pub(in crate::lower) struct MaskedF32Value<'a> {
    pub(in crate::lower) mask: &'a Expr,
    pub(in crate::lower) fill: &'a Expr,
    pub(in crate::lower) spill_depth: usize,
}

pub(in crate::lower) struct MaskedLocalValue<'a> {
    pub(in crate::lower) mask: &'a Expr,
    pub(in crate::lower) element: ElementType,
    pub(in crate::lower) fill: Handle<Expression>,
    pub(in crate::lower) spill_depth: usize,
}

pub(in crate::lower) struct TileLoopReduceSpec {
    pub(in crate::lower) iterations: u32,
    pub(in crate::lower) iter_var: LocalId,
    pub(in crate::lower) op: TileReduceOp,
    pub(in crate::lower) spill_depth: usize,
}

pub(in crate::lower) struct StorageLoadLowering<'a> {
    pub(in crate::lower) src: &'a StorageView,
    pub(in crate::lower) mask: &'a Expr,
    pub(in crate::lower) fill: &'a Expr,
    pub(in crate::lower) spill_depth: usize,
}

pub(in crate::lower) struct QuantizedDotLowering<'a> {
    pub(in crate::lower) src: &'a QuantizedMatrix,
    pub(in crate::lower) activations: &'a PackedActivations,
    pub(in crate::lower) k: &'a DotK,
    pub(in crate::lower) col: &'a Expr,
    pub(in crate::lower) masked: MaskedF32Value<'a>,
    pub(in crate::lower) block_n: u32,
}

pub(in crate::lower) struct GgmlBlockCoordExprs<'a> {
    pub(in crate::lower) block: &'a Expr,
    pub(in crate::lower) c0: &'a Expr,
    pub(in crate::lower) c1: &'a Expr,
    pub(in crate::lower) col: &'a Expr,
    pub(in crate::lower) spill_depth: usize,
}

pub(in crate::lower) struct MaskedQuantizedCol<'a> {
    pub(in crate::lower) k_base: &'a Expr,
    pub(in crate::lower) col: &'a Expr,
    pub(in crate::lower) masked: MaskedF32Value<'a>,
}

pub(in crate::lower) struct QuantizedBlockLaneLowering<'a> {
    pub(in crate::lower) id: BlockDequantId,
    pub(in crate::lower) src: &'a QuantizedMatrix,
    pub(in crate::lower) k_base: &'a Expr,
    pub(in crate::lower) col: &'a Expr,
    pub(in crate::lower) masked: MaskedF32Value<'a>,
    pub(in crate::lower) block_n: u32,
    pub(in crate::lower) lane: u32,
}

pub(in crate::lower) struct CoopFragmentLoad<'a> {
    pub(in crate::lower) id: CoopFragmentId,
    pub(in crate::lower) tile: TileRef,
    pub(in crate::lower) row: &'a Expr,
    pub(in crate::lower) col: &'a Expr,
    pub(in crate::lower) role: naga::CooperativeRole,
    pub(in crate::lower) scalar: ScalarElement,
    pub(in crate::lower) rows: u32,
    pub(in crate::lower) cols: u32,
}

pub(in crate::lower) struct CoopBroadcastLoad<'a> {
    pub(in crate::lower) id: CoopFragmentId,
    pub(in crate::lower) src: &'a StorageView,
    pub(in crate::lower) col: &'a Expr,
    pub(in crate::lower) role: naga::CooperativeRole,
    pub(in crate::lower) scalar: ScalarElement,
    pub(in crate::lower) rows: u32,
    pub(in crate::lower) cols: u32,
}

pub(in crate::lower) struct TileFoldLowering<'a> {
    pub(in crate::lower) count: &'a Expr,
    pub(in crate::lower) iter_var: LocalId,
    pub(in crate::lower) body: &'a [TileStmt],
    pub(in crate::lower) accumulators: &'a [crate::ir::FoldAccumulator],
}

mod expr;
mod load;
mod quantized;
mod scalar;
mod stmt;
mod types;
