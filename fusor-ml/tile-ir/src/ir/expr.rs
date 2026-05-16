use crate::quantized::QuantizedMatrix;

use super::{
    BlockDequantId, ElementType, LocalId, LocalRef, ScalarElement, StorageView, TileBinaryOp,
    TileCompareOp, TileLiteral, TileReduceOp, TileRef, TileUnaryOp, WorkgroupAxis,
};

/// Built-in u32 quantities that show up as leaves in index/address arithmetic.
/// Promoted to `Expr::Builtin` so a single expression type can host both
/// per-lane data and indexing math.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Builtin {
    /// `@builtin(local_invocation_index)` — flat lane within the workgroup.
    Lane,
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
    Literal(TileLiteral),
    /// A built-in u32 quantity (lane id, loop index, program id, subgroup
    /// builtins). Indexing arithmetic uses `Expr::Binary` over `Builtin`
    /// leaves and `Literal(U32(_))` constants.
    Builtin(Builtin),
    /// Reduce a tile expression via a shared-memory tree over `scratch`.
    /// `group_size` is the contiguous-lane group reduced together — typically
    /// the full workgroup, but smaller power-of-two groups are supported. When
    /// `iterations > 1`, `value` is first accumulated across that many loop
    /// iterations per lane before the cross-lane tree, and `iter_var` is the
    /// U32 local that the lowerer stores the current iteration into so the
    /// `value` expression can reference it via `LoadLocal(iter_var)`. For
    /// `iterations == 1`, `iter_var` is `None`.
    Reduce {
        op: TileReduceOp,
        iterations: u32,
        iter_var: Option<LocalId>,
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
    /// Returns Bool. To get a 0/1 numeric, follow with `Select(cond, 1, 0)`.
    Compare {
        op: TileCompareOp,
        left: Box<Expr>,
        right: Box<Expr>,
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
    /// Dot product between two vector expressions.
    VectorDot {
        scalar: ScalarElement,
        lanes: u32,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    /// Compose scalar values into a vector expression.
    ComposeVector {
        scalar: ScalarElement,
        lanes: u32,
        values: Vec<Expr>,
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
    F32(Vec<Expr>),
    /// `f32` activations that get pre-packed to Q8 before the dot
    /// (Q4K-Q8, Q6K-Q8 paths).
    Q8(Vec<Expr>),
    /// Q4K-paired GGML activations: low/high halves and per-quad sums.
    Q4KGgml {
        low: Vec<Expr>,
        high: Vec<Expr>,
        sums: Vec<Expr>,
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
    /// Element type of this expression. Walks the tree using the type carried
    /// in each leaf (`StorageView`, `LocalRef`, etc.) and the operator's known
    /// output type; no external context required.
    pub fn element(&self) -> ElementType {
        match self {
            Expr::Load(load) => match &load.src {
                LoadSource::Storage(view) => view.buffer.element,
                LoadSource::Quantized(_) => ElementType::F32,
            },
            Expr::LoadLinear(load) => load.src.buffer.element,
            Expr::LoadWorkgroup { src, .. } => src.element,
            Expr::LoadLocal(local) => local.element,
            Expr::Literal(value) => value.element(),
            Expr::Builtin(_) => ElementType::U32,
            Expr::Reduce { scratch, .. } => scratch.element,
            Expr::Unary { value, .. } | Expr::Binary { left: value, .. } => value.element(),
            Expr::Cast { to, .. } => *to,
            Expr::Bitcast { to, .. } => *to,
            Expr::Select { accept, .. } => accept.element(),
            Expr::Compare { .. } => ElementType::Bool,
            Expr::SubgroupReduce { value, .. } => value.element(),
            Expr::QuantizedBlockLane { .. } => ElementType::F32,
            Expr::VectorDot { scalar, .. } => scalar.element(),
            Expr::QuantizedDot { .. } => ElementType::F32,
            Expr::ComposeVector { scalar, lanes, .. } => ElementType::Vector {
                scalar: *scalar,
                lanes: *lanes,
            },
        }
    }

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

/// Source of an `Expr::Load`. The lowerer dispatches on the variant to choose
/// between a raw storage read and a dequantized read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoadSource {
    Storage(StorageView),
    Quantized(QuantizedMatrix),
}

/// A masked rank-2 tile load. `fill` is the masked-out value — typically a
/// `Literal`, but any expression evaluable in the surrounding scope is
/// allowed (e.g. `ComposeVector` for a vector splat constant). When `src` is
/// `Quantized`, the load dequantizes on the fly and the result is `f32`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileLoadExpr {
    pub src: LoadSource,
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
