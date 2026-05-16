use std::marker::PhantomData;
use std::ops::{Add, BitAnd, BitXor, Div, Mul, Rem, Sub};

use crate::ir::{
    CoopFragmentId, CoopOperandRole, Expr, LocalRef, StorageView, TileBinaryOp, TileCompareOp,
    TileIndexedStoreStmt, TileLinearLoadExpr, TileLiteral, TileLoadExpr, TileStoreStmt,
    TileUnaryOp,
};
/// Handle to a cooperative-matrix accumulator local.
#[derive(Copy, Clone)]
pub struct CoopAcc<T, const ROWS: usize, const COLS: usize> {
    pub(super) local: LocalRef,
    pub(super) _ty: PhantomData<T>,
}

/// Handle to a cooperatively-loaded fragment SSA value. Reusable across any
/// number of compatible `coop_mma` calls in the same scope without re-loading.
#[derive(Copy, Clone)]
pub struct CoopFragment<T, const ROWS: usize, const COLS: usize> {
    pub(super) id: CoopFragmentId,
    pub(super) role: CoopOperandRole,
    pub(super) _ty: PhantomData<T>,
}

/// Handle to a bound subexpression. Each call to `get()` returns a fresh
/// `Tile` that lowers to a load from the private local that backs the binding.
/// Allocating the local and emitting the binding store happens at the call
/// site of `bind`.
#[derive(Clone, Copy)]
pub struct Bound {
    pub(super) local: LocalRef,
}

/// Iterator description passed to `TileBlock::fold`. Carries a counted
/// `0..count` range; future variants (chunks, strided, zip) would extend this
/// constructor.
#[derive(Clone)]
pub struct FoldIter {
    pub(crate) count: Box<Expr>,
}

/// Construct a counted `0..count` iterator for `TileBlock::fold`.
pub fn range(count: Tile) -> FoldIter {
    FoldIter {
        count: Box::new(count.expr),
    }
}

impl Bound {
    /// Load the bound value.
    pub fn get(&self) -> Tile {
        Tile {
            expr: Expr::LoadLocal(self.local),
        }
    }
}

/// Rank-2 storage address.
pub struct Address<T> {
    pub(super) view: StorageView,
    pub(super) row: Box<Expr>,
    pub(super) col: Box<Expr>,
    pub(super) _ty: PhantomData<T>,
}

/// Rank-1 storage address.
pub struct LinearAddress<T> {
    pub(super) view: StorageView,
    pub(super) index: Box<Expr>,
    pub(super) _ty: PhantomData<T>,
}

/// Convert values accepted by `TileBlock::load` fill parameters into IR
/// literals.
pub trait IntoTileLiteral {
    /// Convert into a tile literal.
    fn into_tile_literal(self) -> TileLiteral;
}

impl IntoTileLiteral for TileLiteral {
    fn into_tile_literal(self) -> TileLiteral {
        self
    }
}

impl IntoTileLiteral for f32 {
    fn into_tile_literal(self) -> TileLiteral {
        TileLiteral::f32(self)
    }
}

impl IntoTileLiteral for u32 {
    fn into_tile_literal(self) -> TileLiteral {
        TileLiteral::U32(self)
    }
}

impl IntoTileLiteral for bool {
    fn into_tile_literal(self) -> TileLiteral {
        TileLiteral::Bool(self)
    }
}

/// Public marker for rank-1 storage addresses accepted by tile storage I/O.
pub trait Rank1TileAddress {}

/// Public marker for rank-2 storage addresses accepted by tile storage I/O.
pub trait Rank2TileAddress {}

impl<T> Rank1TileAddress for LinearAddress<T> {}
impl<T> Rank2TileAddress for Address<T> {}

/// Address types accepted by `TileBlock::load`.
pub trait TileLoadAddress {
    #[doc(hidden)]
    fn load_expr(self, mask: Box<Expr>, fill: TileLiteral) -> Expr;
}

/// Address types accepted by `TileBlock::store`.
pub trait TileStoreAddress {
    #[doc(hidden)]
    fn store_stmt(self, value: Expr, mask: Box<Expr>) -> crate::ir::TileStmt;
}

impl<T> TileLoadAddress for Address<T> {
    fn load_expr(self, mask: Box<Expr>, fill: TileLiteral) -> Expr {
        Expr::Load(TileLoadExpr {
            src: crate::ir::LoadSource::Storage(self.view),
            row: self.row,
            col: self.col,
            mask,
            fill: Box::new(Expr::Literal(fill)),
        })
    }
}

impl<T: crate::ir::Numeric> TileLoadAddress for LinearAddress<T> {
    fn load_expr(self, mask: Box<Expr>, fill: TileLiteral) -> Expr {
        let fill = match T::ELEMENT {
            crate::ElementType::Vector { scalar, lanes } => {
                assert!(
                    (2..=4).contains(&lanes),
                    "vector load supports 2, 3, or 4 lanes"
                );
                assert_eq!(
                    fill.element(),
                    scalar.element(),
                    "vector load fill scalar type mismatch"
                );
                Expr::ComposeVector {
                    scalar,
                    lanes,
                    values: (0..lanes).map(|_| Expr::Literal(fill)).collect(),
                }
            }
            element => {
                assert_eq!(fill.element(), element, "linear load fill type mismatch");
                Expr::Literal(fill)
            }
        };
        Expr::LoadLinear(TileLinearLoadExpr {
            src: self.view,
            index: self.index,
            mask,
            fill: Box::new(fill),
        })
    }
}

impl<T> TileStoreAddress for Address<T> {
    fn store_stmt(self, value: Expr, mask: Box<Expr>) -> crate::ir::TileStmt {
        crate::ir::TileStmt::Store(TileStoreStmt {
            dst: self.view,
            row: self.row,
            col: self.col,
            value,
            mask,
        })
    }
}

impl<T> TileStoreAddress for LinearAddress<T> {
    fn store_stmt(self, value: Expr, mask: Box<Expr>) -> crate::ir::TileStmt {
        crate::ir::TileStmt::StoreIndexed(TileIndexedStoreStmt {
            dst: self.view,
            index: self.index,
            value,
            mask,
        })
    }
}

/// Private local handle.
pub struct Local<T> {
    pub(super) local: LocalRef,
    pub(super) _ty: PhantomData<T>,
}

/// Convert builder values into a tile index expression.
pub trait IntoIndex {
    /// Consume or clone into an index expression.
    fn into_index(self) -> Box<Expr>;
}

/// Scalar u32 index expression.
#[derive(Clone)]
pub struct ScalarIndex {
    pub(super) expr: Box<Expr>,
}

/// Per-lane u32 index expression.
#[derive(Clone)]
pub struct Range {
    pub(super) expr: Box<Expr>,
}

impl IntoIndex for ScalarIndex {
    fn into_index(self) -> Box<Expr> {
        self.expr
    }
}

impl IntoIndex for &ScalarIndex {
    fn into_index(self) -> Box<Expr> {
        self.expr.clone()
    }
}

impl IntoIndex for Range {
    fn into_index(self) -> Box<Expr> {
        self.expr
    }
}

impl IntoIndex for &Range {
    fn into_index(self) -> Box<Expr> {
        self.expr.clone()
    }
}

/// `Box<Expr::Literal(TileLiteral::U32(value)))` — the const-RHS shape every
/// `Index op u32` overload, `index_compare`, and `Fold`/`Loop` count field
/// builds.
pub(super) fn boxed_u32_literal(value: u32) -> Box<Expr> {
    Box::new(Expr::Literal(TileLiteral::U32(value)))
}

impl IntoIndex for u32 {
    fn into_index(self) -> Box<Expr> {
        boxed_u32_literal(self)
    }
}

impl IntoIndex for Tile {
    fn into_index(self) -> Box<Expr> {
        Box::new(self.expr)
    }
}

impl IntoIndex for &Tile {
    fn into_index(self) -> Box<Expr> {
        Box::new(self.expr.clone())
    }
}

pub(super) fn index_compare(left: Box<Expr>, op: TileCompareOp, value: u32) -> Mask {
    Mask {
        expr: Box::new(Expr::Compare {
            op,
            left,
            right: boxed_u32_literal(value),
        }),
    }
}

macro_rules! index_compare_methods {
    ($($name:ident => $op:ident),+ $(,)?) => {
        impl Range {
            $(
                #[doc = concat!("Compare this range with a u32 using `", stringify!($name), "`.")]
                pub fn $name(&self, value: u32) -> Mask {
                    index_compare(self.expr.clone(), TileCompareOp::$op, value)
                }
            )+
        }

        impl ScalarIndex {
            $(
                #[doc = concat!("Compare this scalar index with a u32 using `", stringify!($name), "`.")]
                pub fn $name(&self, value: u32) -> Mask {
                    index_compare(self.expr.clone(), TileCompareOp::$op, value)
                }
            )+
        }
    };
}

index_compare_methods!(lt => Lt, le => Le, gt => Gt, ge => Ge, eq => Eq);

/// Build `$ctor { expr: Expr::Binary(op, self.expr, U32(rhs)) }` — the body
/// every arm of `impl_index_u32_ops!` produces. The two assert arms (`Div`,
/// `Rem`) wrap this with a non-zero check on `rhs`.
impl ScalarIndex {
    fn binary_u32_lit(self, op: TileBinaryOp, rhs: u32) -> Self {
        Self {
            expr: Box::new(Expr::Binary {
                op,
                left: self.expr,
                right: boxed_u32_literal(rhs),
            }),
        }
    }
}

impl Range {
    fn binary_u32_lit(self, op: TileBinaryOp, rhs: u32) -> Self {
        Self {
            expr: Box::new(Expr::Binary {
                op,
                left: self.expr,
                right: boxed_u32_literal(rhs),
            }),
        }
    }
}

macro_rules! impl_index_u32_ops {
    (generic($($generics:tt)+), $ty:ty, $div_msg:literal, $mod_msg:literal) => {
        impl_index_u32_ops!(@impl [impl<$($generics)+>] $ty, $div_msg, $mod_msg);
    };
    ($ty:ty, $div_msg:literal, $mod_msg:literal) => {
        impl_index_u32_ops!(@impl [impl] $ty, $div_msg, $mod_msg);
    };
    (@impl [$($impl_head:tt)*] $ty:ty, $div_msg:literal, $mod_msg:literal) => {
        impl_index_u32_ops!(@arm [$($impl_head)*] $ty, Add, add, TileBinaryOp::Add);
        impl_index_u32_ops!(@arm [$($impl_head)*] $ty, Mul, mul, TileBinaryOp::Mul);
        impl_index_u32_ops!(@arm [$($impl_head)*] $ty, BitAnd, bitand, TileBinaryOp::BitAnd);
        impl_index_u32_ops!(@arm [$($impl_head)*] $ty, BitXor, bitxor, TileBinaryOp::BitXor);
        impl_index_u32_ops!(@assert_arm [$($impl_head)*] $ty, Div, div, TileBinaryOp::Div, $div_msg);
        impl_index_u32_ops!(@assert_arm [$($impl_head)*] $ty, Rem, rem, TileBinaryOp::Rem, $mod_msg);
    };
    (@arm [$($impl_head:tt)*] $ty:ty, $trait:ident, $method:ident, $op:expr) => {
        $($impl_head)* $trait<u32> for $ty {
            type Output = $ty;

            fn $method(self, rhs: u32) -> Self::Output {
                self.binary_u32_lit($op, rhs)
            }
        }
    };
    (@assert_arm [$($impl_head:tt)*] $ty:ty, $trait:ident, $method:ident, $op:expr, $msg:literal) => {
        $($impl_head)* $trait<u32> for $ty {
            type Output = $ty;

            fn $method(self, rhs: u32) -> Self::Output {
                assert!(rhs > 0, $msg);
                self.binary_u32_lit($op, rhs)
            }
        }
    };
}

impl_index_u32_ops!(
    ScalarIndex,
    "scalar index divisor must be non-zero",
    "scalar index modulus must be non-zero"
);
impl_index_u32_ops!(
    Range,
    "tile index divisor must be non-zero",
    "tile index modulus must be non-zero"
);

/// `lhs + rhs` as `Expr::Binary { Add, .. }`. Shared by the three `Add`
/// impls below — `ScalarIndex + ScalarIndex`, `ScalarIndex + Range`, and
/// `Range + ScalarIndex` all reduce to the same expression.
fn add_index_exprs(left: Box<Expr>, right: Box<Expr>) -> Box<Expr> {
    Box::new(Expr::Binary {
        op: TileBinaryOp::Add,
        left,
        right,
    })
}

impl Add<ScalarIndex> for ScalarIndex {
    type Output = ScalarIndex;

    fn add(self, rhs: ScalarIndex) -> Self::Output {
        ScalarIndex {
            expr: add_index_exprs(self.expr, rhs.expr),
        }
    }
}

impl Add<Range> for ScalarIndex {
    type Output = Range;

    fn add(self, rhs: Range) -> Self::Output {
        Range {
            expr: add_index_exprs(self.expr, rhs.expr),
        }
    }
}

impl Add<ScalarIndex> for Range {
    type Output = Range;

    fn add(self, rhs: ScalarIndex) -> Self::Output {
        Range {
            expr: add_index_exprs(self.expr, rhs.expr),
        }
    }
}

/// Per-lane boolean mask.
#[derive(Clone)]
pub struct Mask {
    pub(super) expr: Box<Expr>,
}

impl Mask {
    /// Mask that accepts every lane.
    pub fn all() -> Self {
        Self {
            expr: Box::new(Expr::Literal(TileLiteral::Bool(true))),
        }
    }

    /// Logical conjunction of two masks.
    pub fn and(self, rhs: Self) -> Self {
        Self {
            expr: Box::new(Expr::Binary {
                op: TileBinaryOp::LogicalAnd,
                left: self.expr,
                right: rhs.expr,
            }),
        }
    }
}

/// Scalar expression reduced to one workgroup value.
#[derive(Clone)]
pub struct Scalar {
    pub(super) expr: Expr,
}

impl Scalar {
    /// Build an f32 scalar literal.
    pub fn literal(value: f32) -> Self {
        Self {
            expr: Expr::Literal(TileLiteral::f32(value)),
        }
    }
}

/// Per-lane tile expression.
#[derive(Clone)]
pub struct Tile {
    pub(super) expr: Expr,
}

macro_rules! tile_unary_methods {
    ($($name:ident => $op:ident),+ $(,)?) => {
        $(
            #[doc = concat!("Apply unary `", stringify!($op), "`.")]
            pub fn $name(self) -> Self {
                self.unary(TileUnaryOp::$op)
            }
        )+
    };
}

macro_rules! tile_compare_methods {
    ($($name:ident => $op:ident),+ $(,)?) => {
        $(
            #[doc = concat!("Compare two tiles with `", stringify!($op), "`.")]
            pub fn $name(self, rhs: Self) -> Self {
                Self::compare_bool(TileCompareOp::$op, self, rhs)
            }
        )+
    };
}

macro_rules! tile_binary_methods {
    ($($name:ident => $op:ident),+ $(,)?) => {
        $(
            #[doc = concat!("Apply binary `", stringify!($op), "`.")]
            pub fn $name(self, rhs: Self) -> Self {
                self.binary(TileBinaryOp::$op, rhs)
            }
        )+
    };
}

impl Tile {
    /// Build a tile literal.
    pub fn literal(value: TileLiteral) -> Self {
        Self {
            expr: Expr::Literal(value),
        }
    }

    /// Build a tile from an index expression.
    pub fn from_index(index: impl IntoIndex) -> Self {
        Self {
            expr: *index.into_index(),
        }
    }

    /// Stable structural hash of this tile's expression tree. Used by host-time
    /// builders (e.g. matmul-epilogue cache keys) to key kernel pipelines on the
    /// resulting AST without depending on closure identity. Two tiles with
    /// identical Debug forms hash equal; two tiles with structurally distinct
    /// expressions hash distinct.
    pub fn signature_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        format!("{:?}", self.expr).hash(&mut hasher);
        hasher.finish()
    }

    /// Apply an arbitrary unary operation.
    pub fn unary(self, op: TileUnaryOp) -> Self {
        Self {
            expr: Expr::Unary {
                op,
                value: Box::new(self.expr),
            },
        }
    }

    tile_unary_methods!(exp => Exp, inverse_sqrt => InverseSqrt, exp2 => Exp2, tanh => Tanh, neg_unary => Neg);

    /// Sigmoid activation: `1 / (1 + exp(-x))`.
    pub fn sigmoid(self) -> Self {
        let one = Tile::literal(TileLiteral::f32(1.0));
        one.clone() / (one + self.neg_unary().exp())
    }

    /// SiLU (a.k.a. swish) activation: `x * sigmoid(x)`.
    pub fn silu(self) -> Self {
        self.clone() * self.sigmoid()
    }

    /// GELU activation, tanh approximation:
    /// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
    pub fn gelu(self) -> Self {
        let half = Tile::literal(TileLiteral::f32(0.5));
        let one = Tile::literal(TileLiteral::f32(1.0));
        let coeff = Tile::literal(TileLiteral::f32(0.044_715));
        let sqrt_2_over_pi = Tile::literal(TileLiteral::f32(0.797_884_6));
        let x = self;
        let x_cubed = x.clone() * x.clone() * x.clone();
        let inner = sqrt_2_over_pi * (x.clone() + coeff * x_cubed);
        half * x * (one + inner.tanh())
    }

    /// ReLU activation: `max(x, 0)`.
    pub fn relu(self) -> Self {
        let zero = Tile::literal(TileLiteral::f32(0.0));
        let condition = Tile::compare_bool(TileCompareOp::Gt, self.clone(), zero.clone());
        Tile::select(condition, self, zero)
    }

    /// Cast to another element type.
    pub fn cast(self, to: crate::ElementType) -> Self {
        Self {
            expr: Expr::Cast {
                value: Box::new(self.expr),
                to,
            },
        }
    }

    /// Bitcast to another element type.
    pub fn bitcast(self, to: crate::ElementType) -> Self {
        Self {
            expr: Expr::Bitcast {
                value: Box::new(self.expr),
                to,
            },
        }
    }

    /// Select between two tile values.
    pub fn select(condition: Self, accept: Self, reject: Self) -> Self {
        Self {
            expr: Expr::Select {
                condition: Box::new(condition.expr),
                accept: Box::new(accept.expr),
                reject: Box::new(reject.expr),
            },
        }
    }

    /// Compare two tiles producing a `Bool`-typed tile, then optionally
    /// broadcast `1`/`0` of `output`'s element type via `Select`. Pure builder
    /// convenience — `Expr::Compare` itself always produces `Bool`.
    pub fn compare(op: TileCompareOp, left: Self, right: Self, output: crate::ElementType) -> Self {
        let condition = Self::compare_bool(op, left, right);
        if output == crate::ElementType::Bool {
            condition
        } else {
            let one = TileLiteral::f32(1.0);
            let zero = TileLiteral::f32(0.0);
            let one = Tile::literal(one).cast(output);
            let zero = Tile::literal(zero).cast(output);
            Self::select(condition, one, zero)
        }
    }

    /// Compare two tiles and produce a bool tile.
    pub fn compare_bool(op: TileCompareOp, left: Self, right: Self) -> Self {
        Self {
            expr: Expr::Compare {
                op,
                left: Box::new(left.expr),
                right: Box::new(right.expr),
            },
        }
    }

    tile_compare_methods!(lt => Lt, le => Le, gt => Gt, ge => Ge, eq => Eq, ne => Ne);

    /// Apply an arbitrary binary operation.
    pub fn binary(self, op: TileBinaryOp, rhs: Self) -> Self {
        Tile {
            expr: Expr::Binary {
                op,
                left: Box::new(self.expr),
                right: Box::new(rhs.expr),
            },
        }
    }

    tile_binary_methods!(
        max => Max,
        min => Min,
        bit_and => BitAnd,
        bit_or => BitOr,
        bit_xor => BitXor,
        and => LogicalAnd,
        or => LogicalOr,
    );
}

impl From<Scalar> for Tile {
    fn from(value: Scalar) -> Self {
        Self { expr: value.expr }
    }
}

macro_rules! impl_tile_binary {
    ($trait:ident, $method:ident, $op:expr) => {
        impl $trait for Tile {
            type Output = Tile;

            fn $method(self, rhs: Self) -> Self::Output {
                self.binary($op, rhs)
            }
        }

        impl $trait<Scalar> for Tile {
            type Output = Tile;

            fn $method(self, rhs: Scalar) -> Self::Output {
                self.binary($op, rhs.into())
            }
        }
    };
}

impl_tile_binary!(Add, add, TileBinaryOp::Add);
impl_tile_binary!(Sub, sub, TileBinaryOp::Sub);
impl_tile_binary!(Mul, mul, TileBinaryOp::Mul);
impl_tile_binary!(Div, div, TileBinaryOp::Div);
impl_tile_binary!(Rem, rem, TileBinaryOp::Rem);
