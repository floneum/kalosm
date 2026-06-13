use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Rem, Shl, Shr, Sub};

use crate::ir::{
    Builtin, ElementType, Expr, ExprKind, Local, ScalarElement, StorageView, Tile as TileDeclRc,
    TileBinaryOp, TileCompareOp, TileLiteral, TileUnaryOp,
};

/// A rank-1-per-lane tile value. The element type travels in the IR
/// (`Expr::element()`). `Clone` is an `Rc` bump on the inner `Expr`.
#[derive(Clone)]
pub struct Tile {
    pub(super) expr: Expr,
}

/// A `Bool`-typed tile used as a per-lane mask. Just a [`Tile`] whose element
/// type is `Bool`.
pub type Mask = Tile;

impl Tile {
    pub(super) fn from_expr(expr: Expr) -> Self {
        Self { expr }
    }

    pub(super) fn new(kind: ExprKind, ty: ElementType) -> Self {
        Self::from_expr(Expr::new(kind, ty))
    }

    /// The runtime element type of this value.
    pub fn element(&self) -> ElementType {
        self.expr.element()
    }

    /// Consume and return the underlying IR expression.
    pub(crate) fn into_expr(self) -> Expr {
        self.expr
    }

    /// A typed scalar literal value.
    pub fn literal(value: impl Into<TileLiteral>) -> Self {
        let value = value.into();
        Self::new(ExprKind::Literal(value), value.element())
    }

    /// An f32 literal.
    pub fn f32(value: f32) -> Self {
        Self::literal(TileLiteral::f32(value))
    }

    /// An f16 literal from raw IEEE bits.
    pub fn f16_bits(value: u16) -> Self {
        Self::literal(TileLiteral::F16(value))
    }

    /// A u32 literal.
    pub fn u32(value: u32) -> Self {
        Self::literal(TileLiteral::U32(value))
    }

    /// A bool literal.
    pub fn bool(value: bool) -> Self {
        Self::literal(TileLiteral::Bool(value))
    }

    /// A built-in u32 quantity (lane id, program id, subgroup builtins).
    pub(super) fn builtin(builtin: Builtin) -> Self {
        Self::new(ExprKind::Builtin(builtin), ElementType::U32)
    }

    /// Structural hash of the underlying expression — powers the kernel cache
    /// key. O(1): reads the cached bottom-up hash on the node.
    pub fn signature_hash(&self) -> u64 {
        self.expr.structural_hash()
    }

    /// Apply a unary op, preserving the operand element type.
    pub fn unary(self, op: TileUnaryOp) -> Self {
        let ty = self.element();
        Self::new(
            ExprKind::Unary {
                op,
                value: Box::new(self.expr),
            },
            ty,
        )
    }

    /// Apply a binary op, preserving the left operand element type.
    pub fn binary(self, op: TileBinaryOp, rhs: Self) -> Self {
        let ty = self.element();
        Self::new(
            ExprKind::Binary {
                op,
                left: Box::new(self.expr),
                right: Box::new(rhs.expr),
            },
            ty,
        )
    }

    /// Numeric cast to a runtime element type.
    pub fn cast(self, to: ElementType) -> Self {
        Self::new(
            ExprKind::Cast {
                value: Box::new(self.expr),
                to,
            },
            to,
        )
    }

    /// Reinterpreting bitcast to a runtime element type.
    pub fn bitcast(self, to: ElementType) -> Self {
        Self::new(
            ExprKind::Bitcast {
                value: Box::new(self.expr),
                to,
            },
            to,
        )
    }

    /// Per-lane select; `accept`/`reject` share the result element type.
    pub fn select(condition: Mask, accept: Self, reject: Self) -> Self {
        let ty = accept.element();
        Self::new(
            ExprKind::Select {
                condition: Box::new(condition.expr),
                accept: Box::new(accept.expr),
                reject: Box::new(reject.expr),
            },
            ty,
        )
    }

    fn compare(op: TileCompareOp, left: Self, right: Self) -> Mask {
        Self::new(
            ExprKind::Compare {
                op,
                left: Box::new(left.expr),
                right: Box::new(right.expr),
            },
            ElementType::Bool,
        )
    }

    /// `self < rhs`.
    pub fn lt(&self, rhs: impl Into<Tile>) -> Mask {
        Self::compare(TileCompareOp::Lt, self.clone(), rhs.into())
    }
    /// `self <= rhs`.
    pub fn le(&self, rhs: impl Into<Tile>) -> Mask {
        Self::compare(TileCompareOp::Le, self.clone(), rhs.into())
    }
    /// `self > rhs`.
    pub fn gt(&self, rhs: impl Into<Tile>) -> Mask {
        Self::compare(TileCompareOp::Gt, self.clone(), rhs.into())
    }
    /// `self >= rhs`.
    pub fn ge(&self, rhs: impl Into<Tile>) -> Mask {
        Self::compare(TileCompareOp::Ge, self.clone(), rhs.into())
    }
    /// `self == rhs`.
    pub fn eq(&self, rhs: impl Into<Tile>) -> Mask {
        Self::compare(TileCompareOp::Eq, self.clone(), rhs.into())
    }
    /// `self != rhs`.
    pub fn ne(&self, rhs: impl Into<Tile>) -> Mask {
        Self::compare(TileCompareOp::Ne, self.clone(), rhs.into())
    }

    /// Elementwise maximum.
    pub fn max(self, rhs: impl Into<Tile>) -> Self {
        self.binary(TileBinaryOp::Max, rhs.into())
    }
    /// Elementwise minimum.
    pub fn min(self, rhs: impl Into<Tile>) -> Self {
        self.binary(TileBinaryOp::Min, rhs.into())
    }

    // ---- float math (callers ensure the element is a float at the frontend) ----
    /// Exponential.
    pub fn exp(self) -> Self {
        self.unary(TileUnaryOp::Exp)
    }
    /// Base-2 exponential.
    pub fn exp2(self) -> Self {
        self.unary(TileUnaryOp::Exp2)
    }
    /// Hyperbolic tangent.
    pub fn tanh(self) -> Self {
        self.unary(TileUnaryOp::Tanh)
    }
    /// Reciprocal square root.
    pub fn inverse_sqrt(self) -> Self {
        self.unary(TileUnaryOp::InverseSqrt)
    }
    /// Arithmetic negation.
    pub(crate) fn neg_unary(self) -> Self {
        self.unary(TileUnaryOp::Neg)
    }
    /// Sigmoid `1 / (1 + exp(-x))`.
    pub fn sigmoid(self) -> Self {
        let one = Self::f32(1.0);
        one.clone() / (one + self.neg_unary().exp())
    }
    /// SiLU `x * sigmoid(x)`.
    pub fn silu(self) -> Self {
        self.clone() * self.sigmoid()
    }
    /// Tanh-approximation GELU.
    pub fn gelu(self) -> Self {
        let half = Self::f32(0.5);
        let one = Self::f32(1.0);
        let coeff = Self::f32(0.044_715);
        let sqrt_2_over_pi = Self::f32(0.797_884_6);
        let x = self;
        let x_cubed = x.clone() * x.clone() * x.clone();
        let inner = sqrt_2_over_pi * (x.clone() + coeff * x_cubed);
        half * x * (one + inner.tanh())
    }
    /// `max(x, 0)`.
    pub fn relu(self) -> Self {
        let zero = Self::f32(0.0);
        let condition = self.gt(zero.clone());
        Self::select(condition, self, zero)
    }

    // u32 bit ops are the `&` `|` `^` `<<` `>>` operators (see impls below).
    /// Unpack a `u32` lane holding two packed f16 values into a `vec2<f32>`
    /// (lane 0 = low half, lane 1 = high half).
    pub fn unpack2x16float(self) -> Self {
        Self::new(
            ExprKind::Unary {
                op: TileUnaryOp::Unpack2x16Float,
                value: Box::new(self.expr),
            },
            ElementType::vector(ScalarElement::F32, 2),
        )
    }
    // ---- bool ops ----
    /// A statically-true mask (`Bool(true)`) — i.e. all lanes active.
    pub fn all() -> Mask {
        Self::bool(true)
    }
}

// ---- conversions into Tile ----

impl From<TileLiteral> for Tile {
    fn from(value: TileLiteral) -> Self {
        Self::literal(value)
    }
}

impl From<f32> for Tile {
    fn from(value: f32) -> Self {
        Self::f32(value)
    }
}

impl From<u32> for Tile {
    fn from(value: u32) -> Self {
        Self::u32(value)
    }
}

impl From<&u32> for Tile {
    fn from(value: &u32) -> Self {
        Self::u32(*value)
    }
}

impl From<bool> for Tile {
    fn from(value: bool) -> Self {
        Self::bool(value)
    }
}

impl From<&Tile> for Tile {
    fn from(value: &Tile) -> Self {
        value.clone()
    }
}

/// Box the IR expression behind an index-like tile.
pub(super) fn boxed_index(value: impl Into<Tile>) -> Box<Expr> {
    Box::new(value.into().expr)
}

macro_rules! impl_tile_binary {
    ($trait:ident, $method:ident, $op:expr) => {
        impl<Rhs> $trait<Rhs> for Tile
        where
            Rhs: Into<Tile>,
        {
            type Output = Tile;
            fn $method(self, rhs: Rhs) -> Self::Output {
                self.binary($op, rhs.into())
            }
        }
    };
}

macro_rules! impl_tile_mask_or_bitwise {
    ($trait:ident, $method:ident, $logical:expr, $bitwise:expr) => {
        impl<Rhs> $trait<Rhs> for Tile
        where
            Rhs: Into<Tile>,
        {
            type Output = Tile;
            fn $method(self, rhs: Rhs) -> Self::Output {
                let rhs = rhs.into();
                let lhs_element = self.element();
                let rhs_element = rhs.element();
                let op = if lhs_element == ElementType::Bool || rhs_element == ElementType::Bool {
                    assert!(
                        lhs_element == ElementType::Bool && rhs_element == ElementType::Bool,
                        "boolean tile operators require both operands to be Bool",
                    );
                    $logical
                } else {
                    $bitwise
                };
                self.binary(op, rhs)
            }
        }
    };
}

impl_tile_binary!(Add, add, TileBinaryOp::Add);
impl_tile_binary!(Sub, sub, TileBinaryOp::Sub);
impl_tile_binary!(Mul, mul, TileBinaryOp::Mul);
impl_tile_binary!(Div, div, TileBinaryOp::Div);
impl_tile_binary!(Rem, rem, TileBinaryOp::Rem);
impl_tile_mask_or_bitwise!(
    BitAnd,
    bitand,
    TileBinaryOp::LogicalAnd,
    TileBinaryOp::BitAnd
);
impl_tile_mask_or_bitwise!(BitOr, bitor, TileBinaryOp::LogicalOr, TileBinaryOp::BitOr);
impl_tile_binary!(BitXor, bitxor, TileBinaryOp::BitXor);
impl_tile_binary!(Shl, shl, TileBinaryOp::Shl);
impl_tile_binary!(Shr, shr, TileBinaryOp::Shr);

/// A private per-invocation local, runtime-typed. Holds the `Rc<LocalDecl>`;
/// `clone` is an `Rc` bump.
#[derive(Clone)]
pub struct PrivateLocal {
    pub(super) local: Local,
}

impl PrivateLocal {
    /// The runtime element type of this local.
    pub fn element(&self) -> ElementType {
        self.local.element
    }

    pub(super) fn decl(&self) -> &Local {
        &self.local
    }
}

/// A cooperative-matrix accumulator: a mutable coop-`C`-typed private local.
/// Coop ops are value-producing and composed through `store_local`/`load_local`.
#[derive(Clone)]
pub struct CoopAcc {
    pub(super) local: Local,
}

impl CoopAcc {
    /// Element type (a `CoopMatrix { role: C, .. }`).
    pub fn element(&self) -> ElementType {
        self.local.element
    }

    pub(super) fn decl(&self) -> &Local {
        &self.local
    }
}

/// A workgroup (or private) scratch tile handle, runtime-typed. Holds the
/// `Rc<TileDecl>`; `clone` is an `Rc` bump.
#[derive(Clone)]
pub struct WorkgroupTile {
    pub(super) tile: TileDeclRc,
}

impl WorkgroupTile {
    /// The runtime element type of this tile.
    pub fn element(&self) -> ElementType {
        self.tile.element
    }

    pub(super) fn decl(&self) -> &TileDeclRc {
        &self.tile
    }
}

/// Iteration bound for a counted loop / fold. Carries the (boxed) count
/// expression so `range(n)` reads naturally at call sites.
#[derive(Clone)]
pub struct FoldIter {
    pub(crate) count: Box<Expr>,
}

/// Counted-loop bound for `fold`. Accepts any index-like tile.
pub fn range(count: impl Into<Tile>) -> FoldIter {
    FoldIter {
        count: boxed_index(count),
    }
}

/// A workgroup scratch tile + per-element source address — the components of a
/// storage/quant access. Returned by [`Storage::at`](super::Storage::at).
pub struct Address {
    pub(super) view: StorageView,
    pub(super) addr: crate::ir::Addr,
}

impl Address {
    pub(super) fn load_expr(self, mask: Expr, fill: Expr) -> Expr {
        // A dense storage load produces the buffer's element type verbatim: a
        // load from a vector buffer is a vector value, a scalar buffer a scalar.
        let element = self.view.buffer.element;
        // For a vector buffer the masked-out fill must itself be a vector. The
        // lowerer's `cast_tile_value` does a scalar cast, not a splat, so the
        // fill has to arrive pre-composed.
        let fill = match element {
            ElementType::Vector { scalar, lanes } => {
                let scalar_element = scalar.element();
                let fill = if fill.element() == scalar_element {
                    fill
                } else {
                    Expr::new(
                        ExprKind::Cast {
                            value: Box::new(fill),
                            to: scalar_element,
                        },
                        scalar_element,
                    )
                };
                Expr::new(
                    ExprKind::Vec {
                        scalar,
                        lanes,
                        parts: (0..lanes).map(|_| fill.clone()).collect(),
                    },
                    element,
                )
            }
            _ => fill,
        };
        Expr::new(
            ExprKind::Load {
                src: crate::ir::Source::Storage(self.view),
                addr: self.addr,
                mask: Box::new(mask),
                fill: Box::new(fill),
            },
            element,
        )
    }

    pub(super) fn store_stmt(self, value: Expr, mask: Expr) -> crate::ir::Stmt {
        crate::ir::Stmt::Store {
            dst: self.view,
            addr: self.addr,
            value,
            mask: Box::new(mask),
        }
    }
}

/// The zero value of `element` as an IR expression: a typed zero literal, or a
/// vector of zero literals for a vector element.
pub(super) fn zero_expr(element: ElementType) -> Expr {
    let kind = match element {
        ElementType::F32 => ExprKind::Literal(TileLiteral::f32(0.0)),
        ElementType::F16 => ExprKind::Literal(TileLiteral::F16(0)),
        ElementType::U32 => ExprKind::Literal(TileLiteral::U32(0)),
        ElementType::Bool => ExprKind::Literal(TileLiteral::Bool(false)),
        ElementType::Vector { scalar, lanes } => {
            let part = zero_expr(scalar.element());
            let parts = (0..lanes).map(|_| part.clone()).collect();
            return Expr::new(
                ExprKind::Vec {
                    scalar,
                    lanes,
                    parts,
                },
                element,
            );
        }
        ElementType::CoopMatrix { .. } => panic!("cannot zero a cooperative-matrix value"),
    };
    Expr::new(kind, element)
}
