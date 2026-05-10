use crate::quantized::QuantizedMatrix;

use super::{
    BlockDequantId, ElementType, F32Bits, LocalRef, StorageView, TileBinaryOp, TileCompareOp,
    TileLiteral, TileReduceOp, TileRef, TileUnaryOp, WorkgroupAxis,
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
    LoadVec4(TileVec4LoadExpr),
    LoadWorkgroup {
        src: TileRef,
        index: Box<Expr>,
    },
    LoadLocal(LocalRef),
    QuantizedLoad(TileQuantizedLoadExpr),
    Full(F32Bits),
    Literal(TileLiteral),
    /// A built-in u32 quantity (lane id, loop index, program id, subgroup
    /// builtins). Indexing arithmetic uses `Expr::Binary` over `Builtin`
    /// leaves and `Literal(U32(_))` constants.
    Builtin(Builtin),
    /// Reduce a tile expression to a single scalar via a workgroup-wide
    /// shared-memory reduction tree. `scratch` is the workgroup tile used as
    /// the reduction buffer.
    Reduce {
        op: TileReduceOp,
        value: Box<Expr>,
        scratch: TileRef,
    },
    /// Reduce a tile expression to a single scalar over `iterations` loop
    /// iterations, then across the workgroup. `scratch` is the workgroup
    /// tile used as the cross-lane reduction buffer.
    LoopReduce {
        op: TileReduceOp,
        iterations: u32,
        value: Box<Expr>,
        scratch: TileRef,
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
    /// Left-associated sum of a flat value list. This represents long
    /// unrolled accumulations without forcing the lowerer to recurse through
    /// a deep binary tree.
    Sum {
        values: Vec<Box<Expr>>,
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
    LoopFold {
        op: TileReduceOp,
        iterations: u32,
        value: Box<Expr>,
        initial: TileLiteral,
    },
    GroupReduce {
        op: TileReduceOp,
        value: Box<Expr>,
        scratch: TileRef,
        group_size: u32,
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
    /// and reuses the result across lanes.
    QuantizedBlockLane {
        id: BlockDequantId,
        src: QuantizedMatrix,
        k_base: Box<Expr>,
        col: Box<Expr>,
        mask: Box<Expr>,
        fill: F32Bits,
        block_n: u32,
        lane: u32,
    },
    /// Dot product between two `vec4<f32>` expressions.
    Vec4Dot {
        left: Box<Expr>,
        right: Box<Expr>,
    },
    /// `vec4<f32>(value, value, value, value)`.
    Vec4Splat {
        value: Box<Expr>,
    },
    /// `vec4<f32>(values[0], values[1], values[2], values[3])`. Combined with
    /// `Vec4Dot` this expresses the fused 4-way dot product the qgemv
    /// accelerator emits.
    Compose4 {
        values: [Box<Expr>; 4],
    },
    QuantizedQ8_0Dot8 {
        a: [Box<Expr>; 8],
        src: QuantizedMatrix,
        k_base: Box<Expr>,
        col: Box<Expr>,
        mask: Box<Expr>,
        fill: F32Bits,
    },
    QuantizedVecDot {
        kind: QuantizedVecDotKind,
        a: Vec<Box<Expr>>,
        src: QuantizedMatrix,
        k_base: Box<Expr>,
        col: Box<Expr>,
        mask: Box<Expr>,
        fill: F32Bits,
        block_n: u32,
    },
    QuantizedQ4KGgmlDot {
        a_low: Vec<Box<Expr>>,
        a_high: Vec<Box<Expr>>,
        sums: Vec<Box<Expr>>,
        src: QuantizedMatrix,
        block: Box<Expr>,
        iq: Box<Expr>,
        ir: Box<Expr>,
        col: Box<Expr>,
        mask: Box<Expr>,
        fill: F32Bits,
    },
    QuantizedQ6KGgmlDot {
        a: Vec<Box<Expr>>,
        src: QuantizedMatrix,
        block: Box<Expr>,
        ip: Box<Expr>,
        il: Box<Expr>,
        col: Box<Expr>,
        mask: Box<Expr>,
        fill: F32Bits,
    },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum QuantizedVecDotKind {
    Q8Activation,
    Q4KF32,
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

    /// Recognize a Bool-typed expression that is statically `false`. Mirror of
    /// `is_constant_true`.
    pub fn is_constant_false(&self) -> bool {
        match self {
            Expr::Literal(TileLiteral::Bool(false)) => true,
            Expr::Binary {
                op: TileBinaryOp::LogicalOr,
                left,
                right,
            } => left.is_constant_false() && right.is_constant_false(),
            _ => false,
        }
    }
}

/// A masked rank-1 tile load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileLoadExpr {
    pub src: StorageView,
    pub row: Box<Expr>,
    pub col: Box<Expr>,
    pub mask: Box<Expr>,
    pub fill: TileLiteral,
}

/// A masked rank-1 vec4 load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileVec4LoadExpr {
    pub src: StorageView,
    pub index: Box<Expr>,
    pub mask: Box<Expr>,
    pub fill: F32Bits,
}

/// A masked rank-1 storage load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileLinearLoadExpr {
    pub src: StorageView,
    pub index: Box<Expr>,
    pub mask: Box<Expr>,
    pub fill: TileLiteral,
}

/// A masked dequantizing rank-1 tile load from a packed quantized matrix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileQuantizedLoadExpr {
    pub src: QuantizedMatrix,
    pub row: Box<Expr>,
    pub col: Box<Expr>,
    pub mask: Box<Expr>,
    pub fill: F32Bits,
}
