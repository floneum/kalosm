#![allow(unused_imports)]
use std::marker::PhantomData;
use std::ops::{Add, BitAnd, BitXor, Div, Mul, Rem, Sub};

use crate::ir::{
    BlockDequantId, BufferAccess, BufferDecl, BufferRef, CoopAccDecl, CoopAccId, CoopFragmentId,
    CoopOperandRole, DynamicOffset, F32Bits, F32Vec4, Im2ColNhwcMap, KernelIr, Layout, LocalDecl,
    LocalRef, LoopFoldGroup, LoopFoldGroupId, MemoryLevel, Numeric, Op, PinId,
    QuantizedVecDotKind, Shape, StorageIndexMap, StorageView, TileBinaryOp, TileCompareOp,
    TileDecl, TileExpr, TileIndexExpr, TileIndexedStoreStmt, TileLevel, TileLinearLoadExpr,
    TileLiteral, TileLoadExpr, TileMaskExpr, TileOrigin, TileProgramOp, TileQuantizedLoadExpr,
    TileReduceOp, TileRef, TileScalarExpr, TileStmt, TileStoreStmt, TileUnaryOp, TileVec4LoadExpr,
    WorkgroupAxis, WorkgroupOffset, F32, U32,
};
use crate::quantized::{GgmlQuantFormat, QuantizedMatrix};
use super::*;

/// Activation pattern fused on top of a paired matmul reduction (gate, up).
///
/// The matmul produces concatenated `[gate; up]` columns; the kernel reduces
/// each pair separately and applies the chosen activation as `act(gate) * up`
/// before storing a single output column per pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PairedActivation {
    /// `silu(gate) * up` — used by Llama, Mistral, Qwen, Gemma SwiGLU FFNs.
    SwiGLU,
    /// `gelu(gate) * up` — used by GeGLU-style FFNs (some PaLM variants).
    GeGLU,
    /// `relu(gate) * up` — used by ReGLU-style FFNs.
    ReGLU,
}

impl PairedActivation {
    /// Build the per-output tile expression for this activation.
    pub fn apply<const BLOCK: usize>(self, gate: Tile<BLOCK>, up: Tile<BLOCK>) -> Tile<BLOCK> {
        match self {
            Self::SwiGLU => gate.silu() * up,
            Self::GeGLU => gate.gelu() * up,
            Self::ReGLU => gate.relu() * up,
        }
    }

    /// Lowercase identifier used in kernel names and cache keys.
    pub const fn label(self) -> &'static str {
        match self {
            Self::SwiGLU => "swiglu",
            Self::GeGLU => "geglu",
            Self::ReGLU => "reglu",
        }
    }
}

pub(super) fn matrix_shape(layout: &Layout) -> [u32; 2] {
    assert_eq!(layout.shape().rank(), 2, "matrix operands must be rank-2");
    [
        layout.shape().dims()[0].get(),
        layout.shape().dims()[1].get(),
    ]
}

pub(super) fn cooperative_store_layout_supported(layout: &Layout) -> bool {
    layout.shape().rank() == 2
        && layout.strides().rank() == 2
        && (layout.strides().values()[0] == 1 || layout.strides().values()[1] == 1)
}
