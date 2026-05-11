use crate::quantized::QuantizedMatrix;

use super::{
    BlockDequantId, ElementType, LocalRef, StorageView, TileBinaryOp, TileCompareOp, TileLiteral,
    TileReduceOp, TileRef, TileUnaryOp, WorkgroupAxis,
};

/// Built-in u32 quantities that show up as leaves in index/address arithmetic.
/// Promoted to `Expr::Builtin` so a single expression type can host both
/// per-lane data and indexing math.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Builtin {
    /// `@builtin(local_invocation_index)` — flat lane within the workgroup.
    Lane,
    /// Current iteration counter of the innermost structured `Fold` / `Loop`.
    LoopIndex,
    /// `@builtin(workgroup_id).{x|y|z}`.
    ProgramId(WorkgroupAxis),
    /// `@builtin(subgroup_id)`.
    SubgroupId,
    /// `@builtin(subgroup_invocation_id)` — lane within the subgroup.
    SubgroupLane,
    /// `@builtin(subgroup_size)` — runtime subgroup size.
    SubgroupSize,
    /// `@builtin(num_subgroups)` — number of subgroups per workgroup.
    NumSubgroups,
}

/// A rank-1 tile expression evaluated lane-wise.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Expr {
    Load(TileLoadExpr),
    LoadLinear(TileLinearLoadExpr),
    LoadWorkgroup {
        src: TileRef,
        index: Box<Expr>,
    },
    LoadLocal(LocalRef),
    QuantizedLoad(TileQuantizedLoadExpr),
    Literal(TileLiteral),
    /// A built-in u32 quantity (lane id, loop index, program id, subgroup
    /// builtins). Indexing arithmetic uses `Expr::Binary` over `Builtin`
    /// leaves and `Literal(U32(_))` constants.
    Builtin(Builtin),
    /// Reduce a tile expression via a shared-memory tree over `scratch`.
    /// `group_size` is the contiguous-lane group reduced together — typically
    /// the full workgroup, but smaller power-of-two groups are supported. When
    /// `iterations > 1`, `value` is first accumulated across that many loop
    /// iterations per lane before the cross-lane tree.
    Reduce {
        op: TileReduceOp,
        iterations: u32,
        value: Box<Expr>,
        scratch: TileRef,
        group_size: u32,
    },
    Unary {
        op: TileUnaryOp,
        value: Box<Expr>,
    },
    Binary {
        op: TileBinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Cast {
        value: Box<Expr>,
        to: ElementType,
    },
    Bitcast {
        value: Box<Expr>,
        to: ElementType,
    },
    Select {
        condition: Box<Expr>,
        accept: Box<Expr>,
        reject: Box<Expr>,
    },
    Compare {
        op: TileCompareOp,
        left: Box<Expr>,
        right: Box<Expr>,
        output: ElementType,
    },
    /// Reduction across the lanes of one subgroup. Lowers to
    /// `subgroupAdd`/`subgroupMax`/`subgroupMin` — no shared-memory tree, no
    /// workgroup-shape divisibility constraint.
    SubgroupReduce {
        op: TileReduceOp,
        value: Box<Expr>,
    },
    /// One lane of a fused N-wide quantized dequant. All lanes of the same
    /// `id` share the block scale lookup; the lowerer emits the helper once
    /// and reuses the result across lanes. `fill` is always-`f32`; see
    /// `TileLoadExpr` for the wider `fill` semantics.
    QuantizedBlockLane {
        id: BlockDequantId,
        src: QuantizedMatrix,
        k_base: Box<Expr>,
        col: Box<Expr>,
        mask: Box<Expr>,
        fill: Box<Expr>,
        block_n: u32,
        lane: u32,
    },
    /// Dot product between two `vec4<f32>` expressions.
    Vec4Dot {
        left: Box<Expr>,
        right: Box<Expr>,
    },
    /// `vec4<f32>(values[0], values[1], values[2], values[3])`. Combined with
    /// `Vec4Dot` this expresses the fused 4-way dot product the qgemv
    /// accelerator emits.
    Compose4 {
        values: [Box<Expr>; 4],
    },
    /// Per-column dot of activations against a dequantized quantized-matrix
    /// block. The activation packing (`activations`) and the K coordinate
    /// shape (`k`) together select the lowering helper. `fill` is the
    /// masked-out value; quantized dots always produce `f32`.
    QuantizedDot {
        src: QuantizedMatrix,
        activations: PackedActivations,
        k: DotK,
        col: Box<Expr>,
        mask: Box<Expr>,
        fill: Box<Expr>,
        /// Dot width hint. Meaningful for `F32`/`Q8` activation paths over a
        /// flat `Base` K coordinate; for `Block` K the lowerer dispatches on
        /// shape and ignores this value.
        block_n: u32,
    },
}

/// Activation packing for `Expr::QuantizedDot`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PackedActivations {
    /// Raw `f32` activations, fed directly to the format's dequant+dot helper.
    F32(Vec<Box<Expr>>),
    /// `f32` activations that get pre-packed to Q8 before the dot
    /// (Q4K-Q8, Q6K-Q8 paths).
    Q8(Vec<Box<Expr>>),
    /// Q4K-paired GGML activations: low/high halves and per-quad sums.
    Q4KGgml {
        low: Vec<Box<Expr>>,
        high: Vec<Box<Expr>>,
        sums: Vec<Box<Expr>>,
    },
}

/// Quantized-matrix K coordinate. The `Base` variant is a flat K offset; the
/// `Block` variant carries the per-format block + 2 inner sub-coords (Q4K
/// uses iq/ir, Q6K uses ip/il — same shape).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DotK {
    Base(Box<Expr>),
    Block {
        block: Box<Expr>,
        c0: Box<Expr>,
        c1: Box<Expr>,
    },
}

impl Expr {
    /// Recognize a Bool-typed expression that is statically `true`. Used by the
    /// lowerer to skip mask codegen entirely for unconditional masks.
    pub fn is_constant_true(&self) -> bool {
        match self {
            Expr::Literal(TileLiteral::Bool(true)) => true,
            Expr::Binary {
                op: TileBinaryOp::LogicalAnd,
                left,
                right,
            } => left.is_constant_true() && right.is_constant_true(),
            _ => false,
        }
    }

}

/// A masked rank-1 tile load. `fill` is the masked-out value — typically a
/// `Literal`, but any expression evaluable in the surrounding scope is
/// allowed (e.g. `Compose4` for a vec4 splat constant).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileLoadExpr {
    pub src: StorageView,
    pub row: Box<Expr>,
    pub col: Box<Expr>,
    pub mask: Box<Expr>,
    pub fill: Box<Expr>,
}


/// A masked rank-1 storage load. See `TileLoadExpr` for `fill` semantics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileLinearLoadExpr {
    pub src: StorageView,
    pub index: Box<Expr>,
    pub mask: Box<Expr>,
    pub fill: Box<Expr>,
}

/// A masked dequantizing rank-1 tile load from a packed quantized matrix.
/// `fill` is always-`f32`; see `TileLoadExpr` for the wider `fill` semantics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileQuantizedLoadExpr {
    pub src: QuantizedMatrix,
    pub row: Box<Expr>,
    pub col: Box<Expr>,
    pub mask: Box<Expr>,
    pub fill: Box<Expr>,
}
