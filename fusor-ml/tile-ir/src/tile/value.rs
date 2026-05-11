
use std::marker::PhantomData;
use std::ops::{Add, BitAnd, BitXor, Div, Mul, Rem, Sub};

use crate::ir::{
    CoopFragmentId,
    CoopOperandRole, F32Bits,
    LocalRef, StorageView, TileBinaryOp, TileCompareOp, Expr,
    TileLiteral, TileUnaryOp,
};
/// Handle to an 8x8 cooperative-matrix accumulator local.
#[derive(Copy, Clone)]
pub struct CoopAcc {
    pub(super) local: LocalRef,
}

/// Handle to a cooperatively-loaded 8x8 fragment SSA value. Reusable across
/// any number of `coop_mma` calls in the same scope without re-loading.
#[derive(Copy, Clone)]
pub struct CoopFragment {
    pub(super) id: CoopFragmentId,
    pub(super) role: CoopOperandRole,
}

/// Handle to a bound subexpression. Each call to `get()` returns a fresh
/// `Tile` that lowers to a load from the private local that backs the binding.
/// Allocating the local and emitting the binding store happens at the call
/// site of `bind`.
#[derive(Clone, Copy)]
pub struct Bound<const BLOCK: usize> {
    pub(super) local: LocalRef,
    pub(super) _block: PhantomData<[(); BLOCK]>,
}

/// Iterator description passed to `TileBlock::fold`. Carries a counted
/// `0..count` range; future variants (chunks, strided, zip) would extend this
/// constructor.
#[derive(Clone)]
pub struct FoldIter {
    pub(crate) count: Box<Expr>,
}

/// Construct a counted `0..count` iterator for `TileBlock::fold`.
pub fn range<const BLOCK: usize>(count: Tile<BLOCK>) -> FoldIter {
    FoldIter {
        count: Box::new(count.expr),
    }
}

impl<const BLOCK: usize> Bound<BLOCK> {
    pub fn get(&self) -> Tile<BLOCK> {
        Tile {
            expr: Expr::LoadLocal(self.local),
        }
    }
}

pub struct Address<T, const N: usize> {
    pub(super) view: StorageView,
    pub(super) row: Box<Expr>,
    pub(super) col: Box<Expr>,
    pub(super) _ty: PhantomData<T>,
}

pub struct LinearAddress<T, const N: usize> {
    pub(super) view: StorageView,
    pub(super) index: Box<Expr>,
    pub(super) _ty: PhantomData<T>,
}

pub struct Local<T, const N: usize> {
    pub(super) local: LocalRef,
    pub(super) _ty: PhantomData<(T, [(); N])>,
}

pub struct ErasedAddress<const N: usize> {
    pub(super) view: StorageView,
    pub(super) row: Box<Expr>,
    pub(super) col: Box<Expr>,
}

#[derive(Clone)]
pub struct LaneTile2d<const ROWS: usize, const COLS: usize, const N: usize> {
    pub(super) row: Range<N>,
    pub(super) col: Range<N>,
}

impl<const ROWS: usize, const COLS: usize, const N: usize> LaneTile2d<ROWS, COLS, N> {
    pub fn row(&self) -> Range<N> {
        self.row.clone()
    }

    pub fn col(&self) -> Range<N> {
        self.col.clone()
    }
}

pub trait IntoIndex<const N: usize> {
    fn into_index(self) -> Box<Expr>;
}

#[derive(Clone)]
pub struct ScalarIndex {
    pub(super) expr: Box<Expr>,
}

#[derive(Clone)]
pub struct Range<const N: usize> {
    pub(super) expr: Box<Expr>,
}

impl<const N: usize> IntoIndex<N> for ScalarIndex {
    fn into_index(self) -> Box<Expr> {
        self.expr
    }
}

impl<const N: usize> IntoIndex<N> for &ScalarIndex {
    fn into_index(self) -> Box<Expr> {
        self.expr.clone()
    }
}

impl<const N: usize> IntoIndex<N> for Range<N> {
    fn into_index(self) -> Box<Expr> {
        self.expr
    }
}

impl<const N: usize> IntoIndex<N> for &Range<N> {
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

impl<const N: usize> IntoIndex<N> for u32 {
    fn into_index(self) -> Box<Expr> {
        boxed_u32_literal(self)
    }
}

impl<const N: usize> IntoIndex<N> for Tile<N> {
    fn into_index(self) -> Box<Expr> {
        Box::new(self.expr)
    }
}

impl<const N: usize> IntoIndex<N> for &Tile<N> {
    fn into_index(self) -> Box<Expr> {
        Box::new(self.expr.clone())
    }
}

pub(super) fn index_compare<const N: usize>(left: Box<Expr>, op: TileCompareOp, value: u32) -> Mask<N> {
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
        impl<const N: usize> Range<N> {
            $(
                pub fn $name(&self, value: u32) -> Mask<N> {
                    index_compare(self.expr.clone(), TileCompareOp::$op, value)
                }
            )+
        }

        impl ScalarIndex {
            $(
                pub fn $name<const N: usize>(&self, value: u32) -> Mask<N> {
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

impl<const N: usize> Range<N> {
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
    generic(const N: usize),
    Range<N>,
    "tile index divisor must be non-zero",
    "tile index modulus must be non-zero"
);

/// `lhs + rhs` as `Expr::Binary { Add, .. }`. Shared by the three `Add`
/// impls below — `ScalarIndex + ScalarIndex`, `ScalarIndex + Range`, and
/// `Range + ScalarIndex` all reduce to the same expression.
fn add_index_exprs(left: Box<Expr>, right: Box<Expr>) -> Box<Expr> {
    Box::new(Expr::Binary { op: TileBinaryOp::Add, left, right })
}

impl Add<ScalarIndex> for ScalarIndex {
    type Output = ScalarIndex;

    fn add(self, rhs: ScalarIndex) -> Self::Output {
        ScalarIndex { expr: add_index_exprs(self.expr, rhs.expr) }
    }
}

impl<const N: usize> Add<Range<N>> for ScalarIndex {
    type Output = Range<N>;

    fn add(self, rhs: Range<N>) -> Self::Output {
        Range { expr: add_index_exprs(self.expr, rhs.expr) }
    }
}

impl<const N: usize> Add<ScalarIndex> for Range<N> {
    type Output = Range<N>;

    fn add(self, rhs: ScalarIndex) -> Self::Output {
        Range { expr: add_index_exprs(self.expr, rhs.expr) }
    }
}

#[derive(Clone)]
pub struct Mask<const N: usize> {
    pub(super) expr: Box<Expr>,
}

impl<const N: usize> Mask<N> {
    pub fn all() -> Self {
        Self {
            expr: Box::new(Expr::Literal(TileLiteral::Bool(true))),
        }
    }

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

#[derive(Clone)]
pub struct Scalar {
    pub(super) expr: Expr,
}

impl Scalar {
    pub fn literal(value: f32) -> Self {
        Self {
            expr: Expr::Literal(TileLiteral::F32(F32Bits::new(value))),
        }
    }
}

#[derive(Clone)]
pub struct Tile<const N: usize> {
    pub(super) expr: Expr,
}

macro_rules! tile_unary_methods {
    ($($name:ident => $op:ident),+ $(,)?) => {
        $(
            pub fn $name(self) -> Self {
                self.unary(TileUnaryOp::$op)
            }
        )+
    };
}

macro_rules! tile_compare_methods {
    ($($name:ident => $op:ident),+ $(,)?) => {
        $(
            pub fn $name(self, rhs: Self) -> Self {
                Self::compare_bool(TileCompareOp::$op, self, rhs)
            }
        )+
    };
}

macro_rules! tile_binary_methods {
    ($($name:ident => $op:ident),+ $(,)?) => {
        $(
            pub fn $name(self, rhs: Self) -> Self {
                self.binary(TileBinaryOp::$op, rhs)
            }
        )+
    };
}

impl<const N: usize> Tile<N> {
    pub fn literal(value: TileLiteral) -> Self {
        Self {
            expr: Expr::Literal(value),
        }
    }

    pub fn from_index(index: impl IntoIndex<N>) -> Self {
        Self {
            expr: *index.into_index(),
        }
    }

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
        let one = Tile::literal(TileLiteral::F32(F32Bits::new(1.0)));
        one.clone() / (one + self.neg_unary().exp())
    }

    /// SiLU (a.k.a. swish) activation: `x * sigmoid(x)`.
    pub fn silu(self) -> Self {
        self.clone() * self.sigmoid()
    }

    /// GELU activation, tanh approximation:
    /// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
    pub fn gelu(self) -> Self {
        let half = Tile::literal(TileLiteral::F32(F32Bits::new(0.5)));
        let one = Tile::literal(TileLiteral::F32(F32Bits::new(1.0)));
        let coeff = Tile::literal(TileLiteral::F32(F32Bits::new(0.044_715)));
        let sqrt_2_over_pi = Tile::literal(TileLiteral::F32(F32Bits::new(0.797_884_6)));
        let x = self;
        let x_cubed = x.clone() * x.clone() * x.clone();
        let inner = sqrt_2_over_pi * (x.clone() + coeff * x_cubed);
        half * x * (one + inner.tanh())
    }

    /// ReLU activation: `max(x, 0)`.
    pub fn relu(self) -> Self {
        let zero = Tile::literal(TileLiteral::F32(F32Bits::new(0.0)));
        let condition = Tile::compare_bool(TileCompareOp::Gt, self.clone(), zero.clone());
        Tile::select(condition, self, zero)
    }

    pub fn cast(self, to: crate::ElementType) -> Self {
        Self {
            expr: Expr::Cast {
                value: Box::new(self.expr),
                to,
            },
        }
    }

    pub fn bitcast(self, to: crate::ElementType) -> Self {
        Self {
            expr: Expr::Bitcast {
                value: Box::new(self.expr),
                to,
            },
        }
    }

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
            let one = TileLiteral::F32(F32Bits::new(1.0));
            let zero = TileLiteral::F32(F32Bits::new(0.0));
            let one = Tile::literal(one).cast(output);
            let zero = Tile::literal(zero).cast(output);
            Self::select(condition, one, zero)
        }
    }

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

impl<const N: usize> From<Scalar> for Tile<N> {
    fn from(value: Scalar) -> Self {
        Self {
            expr: value.expr,
        }
    }
}

macro_rules! impl_tile_binary {
    ($trait:ident, $method:ident, $op:expr) => {
        impl<const N: usize> $trait for Tile<N> {
            type Output = Tile<N>;

            fn $method(self, rhs: Self) -> Self::Output {
                self.binary($op, rhs)
            }
        }

        impl<const N: usize> $trait<Scalar> for Tile<N> {
            type Output = Tile<N>;

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
