//! The closed scalar vocabulary. One `Map` with a different [`ScalarExpr`] is
//! every elementwise unary, every comparison, `where_cond`, `clamp`, `relu`,
//! `sigmoid`, `silu`, `gelu` and `tanh_exact`.

use crate::dtype::{Dtype, RoundMode, Splat};
use crate::shape::SymId;
use rustc_hash::{FxHashMap, FxHasher};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// A `Splat` read as f32 — the host-side view of a literal.
pub fn splat_f32(s: Splat) -> f32 {
    match s {
        Splat::F32(v) => v,
        Splat::F16(b) => half::f16::from_bits(b).to_f32(),
        Splat::BF16(b) => half::bf16::from_bits(b).to_f32(),
        Splat::U32(v) => v as f32,
        Splat::I32(v) => v as f32,
    }
}

/// The unary math functions.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum UnOp {
    Exp,
    /// `exp` under a relaxed accuracy contract, permitting a cheaper backend
    /// sequence. A distinct node from [`UnOp::Exp`], so hash-consing cannot
    /// merge the two.
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
    /// Unpack a `u32` of two packed f16s into a 2-lane f32 vector, which reads
    /// native-layout GGUF f16 scales without `SHADER_F16`.
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
    /// Commutative children are sorted by `Id` at construction, making
    /// commutativity a canonical form.
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
/// no boolean dtype at L0.
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
/// `PartialEq` compares the cached hash first. `Arc` because kernel building
/// runs on worker threads.
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
    /// Operand `i` of the enclosing `Map`/`KMap` body.
    Arg(u32),
    Lit(Lit),
    /// A runtime scalar read from the uniform block, never baked into a
    /// kernel.
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
    /// Numeric conversion, differentiable in both directions.
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

    /// Visit each direct child, in field order.
    pub fn for_each_child(&self, mut f: impl FnMut(&ScalarExpr)) {
        match &self.0.kind {
            ScalarKind::Arg(_)
            | ScalarKind::Lit(_)
            | ScalarKind::Uniform(_)
            | ScalarKind::IndexOf(_) => {}
            ScalarKind::Un { x, .. }
            | ScalarKind::Cast { x, .. }
            | ScalarKind::Bitcast { x, .. }
            | ScalarKind::Round { x, .. }
            | ScalarKind::Splat { x, .. } => f(x),
            ScalarKind::Bin { a, b, .. }
            | ScalarKind::Cmp { a, b, .. }
            | ScalarKind::Dot { a, b } => {
                f(a);
                f(b);
            }
            ScalarKind::Select { c, t, f: fe } => {
                f(c);
                f(t);
                f(fe);
            }
        }
    }

    /// Rebuild this node over `f` of each child, in field order. Constructor
    /// nodes re-derive their dtype from the rebuilt children; `Dot` and
    /// `Splat` keep this node's dtype.
    pub fn map_children(&self, mut f: impl FnMut(&ScalarExpr) -> ScalarExpr) -> Self {
        match &self.0.kind {
            ScalarKind::Arg(_)
            | ScalarKind::Lit(_)
            | ScalarKind::Uniform(_)
            | ScalarKind::IndexOf(_) => self.clone(),
            ScalarKind::Un { op, x } => Self::un(*op, f(x)),
            ScalarKind::Bin { op, a, b } => Self::bin(*op, f(a), f(b)),
            ScalarKind::Cmp { op, a, b } => Self::cmp(*op, f(a), f(b)),
            ScalarKind::Select { c, t, f: fe } => Self::select(f(c), f(t), f(fe)),
            ScalarKind::Cast { to, x } => Self::cast(*to, f(x)),
            ScalarKind::Bitcast { to, x } => Self::bitcast(*to, f(x)),
            ScalarKind::Round { mode, x } => Self::round(*mode, f(x)),
            ScalarKind::Dot { a, b } => {
                Self::new(ScalarKind::Dot { a: f(a), b: f(b) }, self.0.dtype)
            }
            ScalarKind::Splat { lanes, x } => Self::new(
                ScalarKind::Splat {
                    lanes: *lanes,
                    x: f(x),
                },
                self.0.dtype,
            ),
        }
    }

    /// `IndexOf(i)` rewritten to `IndexOf(map(i))` throughout, renaming an
    /// absorbed producer's axes into its consumer's index space.
    ///
    /// Memoized on `Arc` identity, so the rewrite is linear in node count.
    pub fn remap_index_axes(&self, map: &impl Fn(u32) -> u32) -> Self {
        let mut memo = FxHashMap::default();
        self.remap_memo(map, &mut memo)
    }

    fn remap_memo(
        &self,
        map: &impl Fn(u32) -> u32,
        memo: &mut FxHashMap<*const ScalarNode, ScalarExpr>,
    ) -> Self {
        if let Some(hit) = memo.get(&Arc::as_ptr(&self.0)) {
            return hit.clone();
        }
        let out = match &self.0.kind {
            ScalarKind::IndexOf(axis) => Self::index_of(map(*axis)),
            _ => self.map_children(|c| c.remap_memo(map, memo)),
        };
        memo.insert(Arc::as_ptr(&self.0), out.clone());
        out
    }

    /// Whether this expression names a loop coordinate anywhere. A lowering
    /// that evaluates a body with no coordinate vector consults this to know
    /// whether it must build one.
    pub fn reads_index_of(&self) -> bool {
        if matches!(self.0.kind, ScalarKind::IndexOf(_)) {
            return true;
        }
        let mut found = false;
        self.for_each_child(|c| found = found || c.reads_index_of());
        found
    }

    /// Whether this expression names the loop coordinate of `axis`.
    pub fn reads_index_of_axis(&self, axis: u32) -> bool {
        if let ScalarKind::IndexOf(a) = &self.0.kind {
            return *a == axis;
        }
        let mut found = false;
        self.for_each_child(|c| found = found || c.reads_index_of_axis(axis));
        found
    }

    /// Whether this expression rounds anywhere: the one syntactic marker of a
    /// value whose contract forbids reassociation.
    pub fn has_round(&self) -> bool {
        if matches!(self.0.kind, ScalarKind::Round { .. }) {
            return true;
        }
        let mut found = false;
        self.for_each_child(|c| found = found || c.has_round());
        found
    }

    /// Substitute `args` for `Arg(i)` throughout. `pre.compose(body)` is
    /// elementwise-into-elementwise fusion.
    ///
    /// Memoized on `Arc` identity, so substitution is linear in node count.
    pub fn compose(&self, args: &[ScalarExpr]) -> Self {
        let mut memo = FxHashMap::default();
        self.compose_memo(args, &mut memo)
    }

    fn compose_memo(
        &self,
        args: &[ScalarExpr],
        memo: &mut FxHashMap<*const ScalarNode, ScalarExpr>,
    ) -> Self {
        if let Some(hit) = memo.get(&Arc::as_ptr(&self.0)) {
            return hit.clone();
        }
        let out = match &self.0.kind {
            ScalarKind::Arg(i) => args
                .get(*i as usize)
                .cloned()
                .unwrap_or_else(|| self.clone()),
            _ => self.map_children(|c| c.compose_memo(args, memo)),
        };
        memo.insert(Arc::as_ptr(&self.0), out.clone());
        out
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
