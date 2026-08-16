//! The closed scalar vocabulary. One `Map` with a different [`ScalarExpr`] is
//! every elementwise unary, every comparison, `where_cond`, `clamp`, `relu`,
//! `sigmoid`, `silu`, `gelu` and `tanh_exact`.

use crate::dtype::{Dtype, RoundMode, Splat};
use crate::shape::SymId;
use rustc_hash::FxHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// The 21 unary math functions.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum UnOp {
    Exp,
    /// `exp` under a relaxed accuracy contract. A **distinct node**, not sugar
    /// for [`UnOp::Exp`]. The contract is a *permission* to substitute a
    /// cheaper sequence, and no backend currently takes it.
    ApproximateExp,
    /// Medium-accuracy `exp`. See [`UnOp::ApproximateExp`].
    LessApproximateExp,
    Exp2,
    Log,
    Log2,
    Sqrt,
    InverseSqrt,
    Sin,
    Cos,
    Tan,
    Tanh,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Asinh,
    Acosh,
    Atanh,
    Abs,
    Neg,
    /// Unpack a `u32` of two packed f16s into a 2-lane f32 vector — how
    /// native-layout GGUF f16 scales are read without `SHADER_F16`.
    Unpack2x16Float,
}

impl UnOp {
    /// True for the transcendentals priced at `DeviceFacts::trans_ps`.
    pub const fn is_transcendental(self) -> bool {
        !matches!(self, Self::Abs | Self::Neg | Self::Unpack2x16Float)
    }
}

/// The 15 binary ops.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Pow,
    Min,
    Max,
    BitAnd,
    BitOr,
    BitXor,
    Shr,
    Shl,
    LogicalAnd,
    LogicalOr,
}

impl BinOp {
    /// Commutative children are sorted by `Id` at construction, so
    /// commutativity is a canonical form rather than a rule family.
    pub const fn is_commutative(self) -> bool {
        matches!(
            self,
            Self::Add
                | Self::Mul
                | Self::Min
                | Self::Max
                | Self::BitAnd
                | Self::BitOr
                | Self::BitXor
                | Self::LogicalAnd
                | Self::LogicalOr
        )
    }

    /// Exactly associative, ignoring float rounding. Whether a *value* may
    /// be reassociated is `NumericContract::reassoc`.
    pub const fn is_associative(self) -> bool {
        matches!(
            self,
            Self::Add
                | Self::Mul
                | Self::Min
                | Self::Max
                | Self::BitAnd
                | Self::BitOr
                | Self::BitXor
                | Self::LogicalAnd
                | Self::LogicalOr
        )
    }
}

/// The 6 comparisons. Results are 1.0/0.0 in the operand dtype — there is
/// no boolean dtype at Logical.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CmpOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

/// A typed scalar literal.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Lit(pub Splat);

/// A hash-consed scalar expression tree. `Clone` is a refcount bump;
/// `PartialEq` compares the cached hash first. `Arc`, not `Rc`: kernel
/// building runs on worker threads.
#[derive(Clone, Debug)]
pub struct ScalarExpr(Arc<ScalarNode>);

/// A scalar node with its cached dtype and structural hash.
#[derive(Debug)]
pub struct ScalarNode {
    pub kind: ScalarKind,
    pub dtype: Dtype,
    pub hash: u64,
}

/// The closed scalar vocabulary. `Hash` is bottom-up: children contribute
/// their cached `structural_hash`, so hashing is O(1) per node.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ScalarKind {
    /// Operand `i` of the enclosing `Map`/`Map` body.
    Arg(u32),
    Lit(Lit),
    /// A runtime scalar read from the uniform block; never baked into a kernel.
    Uniform(SymId),
    /// The current coordinate along `axis` of the enclosing index space.
    IndexOf(u32),
    Un {
        op: UnOp,
        x: ScalarExpr,
    },
    Bin {
        op: BinOp,
        a: ScalarExpr,
        b: ScalarExpr,
    },
    Cmp {
        op: CmpOp,
        a: ScalarExpr,
        b: ScalarExpr,
    },
    /// `where_cond`: take `t` where `c != 0`, else `f`.
    Select {
        c: ScalarExpr,
        t: ScalarExpr,
        f: ScalarExpr,
    },
    /// Numeric conversion, differentiable both directions with no special
    /// case in `map_adjoint`.
    Cast {
        to: Dtype,
        x: ScalarExpr,
    },
    Bitcast {
        to: Dtype,
        x: ScalarExpr,
    },
    Round {
        mode: RoundMode,
        x: ScalarExpr,
    },
    Dot {
        a: ScalarExpr,
        b: ScalarExpr,
    },
    Splat {
        lanes: u32,
        x: ScalarExpr,
    },
}

impl ScalarExpr {
    pub fn new(kind: ScalarKind, dtype: Dtype) -> Self {
        let mut h = FxHasher::default();
        kind.hash(&mut h);
        dtype.hash(&mut h);
        let hash = h.finish();
        Self(Arc::new(ScalarNode { kind, dtype, hash }))
    }

    pub fn kind(&self) -> &ScalarKind {
        &self.0.kind
    }
    pub fn dtype(&self) -> Dtype {
        self.0.dtype
    }
    pub fn structural_hash(&self) -> u64 {
        self.0.hash
    }

    pub fn arg(i: u32, dtype: Dtype) -> Self {
        Self::new(ScalarKind::Arg(i), dtype)
    }
    pub fn lit(v: Splat) -> Self {
        Self::new(ScalarKind::Lit(Lit(v)), v.dtype())
    }
    pub fn uniform(sym: SymId, dtype: Dtype) -> Self {
        Self::new(ScalarKind::Uniform(sym), dtype)
    }
    pub fn index_of(axis: u32) -> Self {
        Self::new(ScalarKind::IndexOf(axis), Dtype::U32)
    }
    pub fn un(op: UnOp, x: Self) -> Self {
        let dtype = x.dtype();
        Self::new(ScalarKind::Un { op, x }, dtype)
    }
    pub fn bin(op: BinOp, a: Self, b: Self) -> Self {
        let dtype = a.dtype();
        Self::new(ScalarKind::Bin { op, a, b }, dtype)
    }
    pub fn cmp(op: CmpOp, a: Self, b: Self) -> Self {
        let dtype = a.dtype();
        Self::new(ScalarKind::Cmp { op, a, b }, dtype)
    }
    pub fn select(c: Self, t: Self, f: Self) -> Self {
        let dtype = t.dtype();
        Self::new(ScalarKind::Select { c, t, f }, dtype)
    }
    pub fn cast(to: Dtype, x: Self) -> Self {
        Self::new(ScalarKind::Cast { to, x }, to)
    }
    pub fn bitcast(to: Dtype, x: Self) -> Self {
        Self::new(ScalarKind::Bitcast { to, x }, to)
    }
    pub fn round(mode: RoundMode, x: Self) -> Self {
        let dtype = x.dtype();
        Self::new(ScalarKind::Round { mode, x }, dtype)
    }

    /// `IndexOf(i)` rewritten to `IndexOf(map(i))` throughout. What an
    /// absorbed producer's coordinates are called in its consumer's space —
    /// a permuted contraction operand walks producer axis `perm[j]` at its
    /// own axis `j`, so the body's axis names shift by `perm⁻¹`.
    pub fn remap_index_axes(&self, map: &impl Fn(u32) -> u32) -> Self {
        match &self.0.kind {
            ScalarKind::IndexOf(axis) => Self::index_of(map(*axis)),
            ScalarKind::Arg(_) | ScalarKind::Lit(_) | ScalarKind::Uniform(_) => self.clone(),
            ScalarKind::Un { op, x } => Self::un(*op, x.remap_index_axes(map)),
            ScalarKind::Bin { op, a, b } => {
                Self::bin(*op, a.remap_index_axes(map), b.remap_index_axes(map))
            }
            ScalarKind::Cmp { op, a, b } => {
                Self::cmp(*op, a.remap_index_axes(map), b.remap_index_axes(map))
            }
            ScalarKind::Select { c, t, f } => Self::select(
                c.remap_index_axes(map),
                t.remap_index_axes(map),
                f.remap_index_axes(map),
            ),
            ScalarKind::Cast { to, x } => Self::cast(*to, x.remap_index_axes(map)),
            ScalarKind::Bitcast { to, x } => Self::bitcast(*to, x.remap_index_axes(map)),
            ScalarKind::Round { mode, x } => Self::round(*mode, x.remap_index_axes(map)),
            ScalarKind::Dot { a, b } => Self::new(
                ScalarKind::Dot {
                    a: a.remap_index_axes(map),
                    b: b.remap_index_axes(map),
                },
                self.0.dtype,
            ),
            ScalarKind::Splat { lanes, x } => Self::new(
                ScalarKind::Splat {
                    lanes: *lanes,
                    x: x.remap_index_axes(map),
                },
                self.0.dtype,
            ),
        }
    }

    /// Whether this expression names a loop coordinate anywhere. A lowering
    /// that evaluates a body with no coordinate vector consults this to know
    /// whether it must build one.
    pub fn reads_index_of(&self) -> bool {
        match &self.0.kind {
            ScalarKind::IndexOf(_) => true,
            ScalarKind::Arg(_) | ScalarKind::Lit(_) | ScalarKind::Uniform(_) => false,
            ScalarKind::Un { x, .. }
            | ScalarKind::Cast { x, .. }
            | ScalarKind::Bitcast { x, .. }
            | ScalarKind::Round { x, .. }
            | ScalarKind::Splat { x, .. } => x.reads_index_of(),
            ScalarKind::Bin { a, b, .. }
            | ScalarKind::Cmp { a, b, .. }
            | ScalarKind::Dot { a, b } => a.reads_index_of() || b.reads_index_of(),
            ScalarKind::Select { c, t, f } => {
                c.reads_index_of() || t.reads_index_of() || f.reads_index_of()
            }
        }
    }

    /// Substitute `args` for `Arg(i)` throughout. This *is*
    /// elementwise-into-elementwise fusion: `pre.compose(body)` needs no
    /// rewrite rule at all, only a tree substitution.
    pub fn compose(&self, args: &[ScalarExpr]) -> Self {
        match &self.0.kind {
            ScalarKind::Arg(i) => args
                .get(*i as usize)
                .cloned()
                .unwrap_or_else(|| self.clone()),
            ScalarKind::Lit(_) | ScalarKind::Uniform(_) | ScalarKind::IndexOf(_) => self.clone(),
            ScalarKind::Un { op, x } => Self::un(*op, x.compose(args)),
            ScalarKind::Bin { op, a, b } => Self::bin(*op, a.compose(args), b.compose(args)),
            ScalarKind::Cmp { op, a, b } => Self::cmp(*op, a.compose(args), b.compose(args)),
            ScalarKind::Select { c, t, f } => {
                Self::select(c.compose(args), t.compose(args), f.compose(args))
            }
            ScalarKind::Cast { to, x } => Self::cast(*to, x.compose(args)),
            ScalarKind::Bitcast { to, x } => Self::bitcast(*to, x.compose(args)),
            ScalarKind::Round { mode, x } => Self::round(*mode, x.compose(args)),
            ScalarKind::Dot { a, b } => Self::new(
                ScalarKind::Dot {
                    a: a.compose(args),
                    b: b.compose(args),
                },
                self.0.dtype,
            ),
            ScalarKind::Splat { lanes, x } => Self::new(
                ScalarKind::Splat {
                    lanes: *lanes,
                    x: x.compose(args),
                },
                self.0.dtype,
            ),
        }
    }
}

impl PartialEq for ScalarExpr {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
            || (self.0.hash == other.0.hash && self.0.kind == other.0.kind)
    }
}
impl Eq for ScalarExpr {}
impl Hash for ScalarExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.0.hash);
    }
}
