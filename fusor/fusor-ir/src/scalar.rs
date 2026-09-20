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

    /// Replace an expression before descending into its children. Returning
    /// `None` preserves the node and recursively rewrites its operands.
    pub fn rewrite(&self, f: &mut impl FnMut(&Self) -> Option<Self>) -> Self {
        if let Some(replacement) = f(self) {
            return replacement;
        }
        self.map_children(&mut |child| child.rewrite(f))
    }

    /// Rebuild the immediate operands, retaining the node's operator.
    pub fn map_children(&self, f: &mut impl FnMut(&Self) -> Self) -> Self {
        match self.kind() {
            ScalarKind::Arg(_)
            | ScalarKind::Lit(_)
            | ScalarKind::Uniform(_)
            | ScalarKind::IndexOf(_) => self.clone(),
            ScalarKind::Un { op, x } => Self::un(*op, f(x)),
            ScalarKind::Bin { op, a, b } => Self::bin(*op, f(a), f(b)),
            ScalarKind::Cmp { op, a, b } => Self::cmp(*op, f(a), f(b)),
            ScalarKind::Select { c, t, f: other } => Self::select(f(c), f(t), f(other)),
            ScalarKind::Cast { to, x } => Self::cast(*to, f(x)),
            ScalarKind::Bitcast { to, x } => Self::bitcast(*to, f(x)),
            ScalarKind::Round { mode, x } => Self::round(*mode, f(x)),
            ScalarKind::Dot { a, b } => {
                Self::new(ScalarKind::Dot { a: f(a), b: f(b) }, self.dtype())
            }
            ScalarKind::Splat { lanes, x } => Self::new(
                ScalarKind::Splat {
                    lanes: *lanes,
                    x: f(x),
                },
                self.dtype(),
            ),
        }
    }

    /// Pre-order traversal; shared subexpressions are visited for each use.
    pub fn walk(&self, f: &mut impl FnMut(&Self)) {
        f(self);
        match self.kind() {
            ScalarKind::Un { x, .. }
            | ScalarKind::Cast { x, .. }
            | ScalarKind::Bitcast { x, .. }
            | ScalarKind::Round { x, .. }
            | ScalarKind::Splat { x, .. } => x.walk(f),
            ScalarKind::Bin { a, b, .. }
            | ScalarKind::Cmp { a, b, .. }
            | ScalarKind::Dot { a, b } => {
                a.walk(f);
                b.walk(f);
            }
            ScalarKind::Select { c, t, f: other } => {
                c.walk(f);
                t.walk(f);
                other.walk(f);
            }
            _ => {}
        }
    }

    /// Resolve an absorbed producer's axes in its consumer's iteration space.
    pub fn remap_index_axes(&self, map: &impl Fn(u32) -> u32) -> Self {
        self.rewrite(&mut |e| match e.kind() {
            ScalarKind::IndexOf(axis) => Some(Self::index_of(map(*axis))),
            _ => None,
        })
    }

    /// Whether the body needs iteration coordinates during lowering.
    pub fn reads_index_of(&self) -> bool {
        let mut found = false;
        self.walk(&mut |e| found |= matches!(e.kind(), ScalarKind::IndexOf(_)));
        found
    }

    /// Substitute operand expressions for `Arg(i)` throughout the body.
    pub fn compose(&self, args: &[ScalarExpr]) -> Self {
        self.rewrite(&mut |e| match e.kind() {
            ScalarKind::Arg(i) => args.get(*i as usize).cloned(),
            _ => None,
        })
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
